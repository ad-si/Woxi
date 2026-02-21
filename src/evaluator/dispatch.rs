#[allow(unused_imports)]
use super::*;

pub fn evaluate_function_call_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  #[cfg(not(target_arch = "wasm32"))]
  {
    stacker::maybe_grow(64 * 1024, 2 * 1024 * 1024, || {
      evaluate_function_call_ast_inner(name, args)
    })
  }
  #[cfg(target_arch = "wasm32")]
  {
    evaluate_function_call_ast_inner(name, args)
  }
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
      (Expr::Identifier(word.to_string()), skipped + end)
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

  // Handle functions that would call interpret() if dispatched through evaluate_expression
  // These must be handled natively to avoid infinite recursion
  match name {
    "Function" => {
      match args.len() {
        // Function[body] — equivalent to body &
        1 => {
          return Ok(Expr::Function {
            body: Box::new(args[0].clone()),
          });
        }
        // Function[x, body] or Function[{x,y,...}, body]
        2 => {
          let params = match &args[0] {
            Expr::Identifier(name) => vec![name.clone()],
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
            _ => {
              return Ok(Expr::FunctionCall {
                name: "Function".to_string(),
                args: args.to_vec(),
              });
            }
          };
          return Ok(Expr::NamedFunction {
            params,
            body: Box::new(args[1].clone()),
          });
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Function".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    "Compile" if args.len() == 2 => {
      // Compile[{x, y, ...}, body] or Compile[{{x, _Real}, {y, _Integer}}, body]
      // Returns CompiledFunction[{x, y, ...}, body]
      let vars = match &args[0] {
        Expr::List(items) => {
          let mut var_names = Vec::new();
          for item in items {
            match item {
              Expr::Identifier(name) => var_names.push(name.clone()),
              // {x, _Real} or {x, _Integer} — extract just the variable name
              Expr::List(inner) if !inner.is_empty() => {
                if let Expr::Identifier(name) = &inner[0] {
                  var_names.push(name.clone());
                }
              }
              _ => {}
            }
          }
          var_names
        }
        Expr::Identifier(name) => vec![name.clone()],
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Compile".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let var_exprs: Vec<Expr> =
        vars.iter().map(|v| Expr::Identifier(v.clone())).collect();
      return Ok(Expr::FunctionCall {
        name: "CompiledFunction".to_string(),
        args: vec![Expr::List(var_exprs), args[1].clone()],
      });
    }
    "Rational" if args.len() == 2 => {
      // Rational[n, d] with integer arguments: simplify via make_rational_pub
      if let (Some(n), Some(d)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
      {
        if d == 0 {
          return Ok(Expr::Identifier("ComplexInfinity".to_string()));
        }
        return Ok(crate::functions::math_ast::make_rational_pub(n, d));
      }
      // Non-integer arguments: return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: args.to_vec(),
      });
    }
    "Module" => return module_ast(args),
    "Block" => return block_ast(args),
    "Assuming" if args.len() == 2 => return assuming_ast(args),
    "With" if args.len() == 2 => return with_ast(args),
    "Set" if args.len() == 2 => {
      return set_ast(&args[0], &args[1]);
    }
    "SetAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::Identifier(name) => vec![name.clone()],
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
        _ => vec![],
      };
      let attr: Vec<String> = match &args[1] {
        Expr::Identifier(a) => vec![a.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(a) = item {
              Some(a.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      if !func_names.is_empty() {
        let mut locked = false;
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              eprintln!("Attributes::locked: Symbol {} is locked.", func_name);
              locked = true;
              continue;
            }
            let entry = attrs.entry(func_name.clone()).or_insert_with(Vec::new);
            for a in &attr {
              if !entry.contains(a) {
                entry.push(a.clone());
              }
            }
          }
        });
        if locked {
          return Ok(Expr::Identifier("Null".to_string()));
        }
        return Ok(Expr::Identifier("Null".to_string()));
      }
    }
    "ClearAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::Identifier(name) => vec![name.clone()],
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
        _ => vec![],
      };
      let to_remove: Vec<String> = match &args[1] {
        Expr::Identifier(a) => vec![a.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(a) = item {
              Some(a.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      if !func_names.is_empty() {
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              eprintln!("Attributes::locked: Symbol {} is locked.", func_name);
              continue;
            }
            if let Some(entry) = attrs.get_mut(func_name) {
              entry.retain(|a| !to_remove.contains(a));
            }
          }
        });
        return Ok(Expr::Identifier("Null".to_string()));
      }
    }
    "Protect" => {
      let mut protected_syms = Vec::new();
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            let entry = attrs.entry(sym.clone()).or_insert_with(Vec::new);
            if !entry.contains(&"Protected".to_string()) {
              entry.push("Protected".to_string());
            }
          });
          protected_syms.push(Expr::Identifier(sym.clone()));
        }
      }
      return Ok(Expr::List(protected_syms));
    }
    "Unprotect" => {
      let mut unprotected_syms = Vec::new();
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          // Check if symbol is Locked (either builtin or user-defined)
          let is_locked = {
            let builtin = get_builtin_attributes(sym);
            if builtin.contains(&"Locked") {
              true
            } else {
              crate::FUNC_ATTRS.with(|m| {
                m.borrow()
                  .get(sym.as_str())
                  .is_some_and(|attrs| attrs.contains(&"Locked".to_string()))
              })
            }
          };
          if is_locked {
            eprintln!("Protect::locked: Symbol {} is locked.", sym);
            continue;
          }
          let was_protected = crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            if let Some(entry) = attrs.get_mut(sym) {
              let before_len = entry.len();
              entry.retain(|a| a != "Protected");
              before_len != entry.len()
            } else {
              false
            }
          });
          if was_protected {
            unprotected_syms.push(Expr::Identifier(sym.clone()));
          }
        }
      }
      return Ok(Expr::List(unprotected_syms));
    }
    "Clear" => {
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          ENV.with(|e| e.borrow_mut().remove(sym));
          crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
        }
      }
      return Ok(Expr::Identifier("Null".to_string()));
    }
    "ClearAll" => {
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          ENV.with(|e| e.borrow_mut().remove(sym));
          crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
          crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
          // Also remove upvalues owned by this symbol, and the FUNC_DEFS entries they created
          let up_defs = crate::UPVALUES.with(|m| m.borrow_mut().remove(sym));
          if let Some(up_defs) = up_defs {
            for (outer_func, params, _conds, _defaults, _heads, body) in
              &up_defs
            {
              let body_str = expr_to_string(body);
              crate::FUNC_DEFS.with(|m| {
                if let Some(entry) = m.borrow_mut().get_mut(outer_func) {
                  entry.retain(|(p, _, _, _, b)| {
                    !(p == params && expr_to_string(b) == body_str)
                  });
                }
              });
            }
          }
        }
      }
      return Ok(Expr::Identifier("Null".to_string()));
    }
    "HoldForm" if args.len() == 1 => {
      return Ok(Expr::FunctionCall {
        name: "HoldForm".to_string(),
        args: args.to_vec(),
      });
    }
    "Hold" if !args.is_empty() => {
      return Ok(Expr::FunctionCall {
        name: "Hold".to_string(),
        args: args.to_vec(),
      });
    }
    "HoldComplete" if !args.is_empty() => {
      return Ok(Expr::FunctionCall {
        name: "HoldComplete".to_string(),
        args: args.to_vec(),
      });
    }
    "ReleaseHold" if args.len() == 1 => {
      // ReleaseHold removes Hold, HoldForm, HoldPattern wrappers and evaluates the result
      match &args[0] {
        Expr::FunctionCall {
          name: hold_name,
          args: hold_args,
        } if (hold_name == "Hold"
          || hold_name == "HoldForm"
          || hold_name == "HoldPattern")
          && hold_args.len() == 1 =>
        {
          return evaluate_expr_to_expr(&hold_args[0]);
        }
        Expr::FunctionCall {
          name: hold_name,
          args: hold_args,
        } if (hold_name == "Hold"
          || hold_name == "HoldForm"
          || hold_name == "HoldPattern")
          && hold_args.len() > 1 =>
        {
          // ReleaseHold[Hold[a, b, c]] → Sequence[a, b, c] then evaluate
          let evaluated: Vec<Expr> = hold_args
            .iter()
            .map(evaluate_expr_to_expr)
            .collect::<Result<_, _>>()?;
          return Ok(Expr::FunctionCall {
            name: "Sequence".to_string(),
            args: evaluated,
          });
        }
        other => {
          // If not wrapped in Hold/HoldForm, evaluate the argument
          return evaluate_expr_to_expr(other);
        }
      }
    }
    // TimeRemaining[] → Infinity (no time limit)
    "TimeRemaining" if args.is_empty() => {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    // Evaluate[expr] - forces evaluation (identity outside Hold)
    "Evaluate" if args.len() == 1 => {
      return Ok(args[0].clone());
    }
    // Inert symbolic head — evaluates to itself (used as argument to StringSplit etc.)
    "RegularExpression" if args.len() == 1 => {
      return Ok(Expr::FunctionCall {
        name: "RegularExpression".to_string(),
        args: args.to_vec(),
      });
    }
    // Distribution heads — inert symbolic forms
    "UniformDistribution" if args.len() == 1 => {
      return Ok(Expr::FunctionCall {
        name: "UniformDistribution".to_string(),
        args: args.to_vec(),
      });
    }
    "NormalDistribution" => {
      // NormalDistribution[] defaults to NormalDistribution[0, 1]
      let norm_args = if args.is_empty() {
        vec![Expr::Integer(0), Expr::Integer(1)]
      } else {
        args.to_vec()
      };
      return Ok(Expr::FunctionCall {
        name: "NormalDistribution".to_string(),
        args: norm_args,
      });
    }
    "Names" if args.len() <= 1 => {
      let all_names = crate::get_defined_names();
      if args.is_empty() {
        // Names[] returns all defined symbol names
        let items: Vec<Expr> =
          all_names.into_iter().map(Expr::String).collect();
        return Ok(Expr::List(items));
      }
      // Names["pattern"] - pattern with * wildcards
      if let Expr::String(pattern) = &args[0] {
        let regex_pattern = format!(
          "^{}$",
          pattern
            .replace('.', "\\.")
            .replace('*', ".*")
            .replace('@', "[a-z0-9]*")
        );
        let re = regex::Regex::new(&regex_pattern);
        if let Ok(re) = re {
          let items: Vec<Expr> = all_names
            .into_iter()
            .filter(|n| re.is_match(n))
            .map(Expr::String)
            .collect();
          return Ok(Expr::List(items));
        }
      }
      return Ok(Expr::List(vec![]));
    }
    "ValueQ" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let has_value = ENV.with(|e| e.borrow().contains_key(sym));
        let has_func = crate::FUNC_DEFS.with(|m| m.borrow().contains_key(sym));
        return Ok(Expr::Identifier(
          if has_value || has_func {
            "True"
          } else {
            "False"
          }
          .to_string(),
        ));
      }
      return Ok(Expr::Identifier("False".to_string()));
    }
    "If" => {
      if args.len() >= 2 && args.len() <= 4 {
        let cond = evaluate_expr_to_expr(&args[0])?;
        if matches!(&cond, Expr::Identifier(s) if s == "True") {
          return evaluate_expr_to_expr(&args[1]);
        } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
          if args.len() >= 3 {
            return evaluate_expr_to_expr(&args[2]);
          } else {
            return Ok(Expr::Identifier("Null".to_string()));
          }
        } else if args.len() == 4 {
          // Non-boolean condition with default (4th arg)
          return evaluate_expr_to_expr(&args[3]);
        }
      } else if args.len() < 2 || args.len() > 4 {
        println!(
          "\nIf::argb: If called with {} arguments; between 2 and 4 arguments are expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
      }
    }
    // AST-native list functions - these avoid string round-trips
    "Map" if args.len() == 2 => {
      return list_helpers_ast::map_ast(&args[0], &args[1]);
    }
    "Map" if args.len() == 3 => {
      return list_helpers_ast::map_with_level_ast(
        &args[0], &args[1], &args[2],
      );
    }
    "MapAll" if args.len() == 2 => {
      return map_all_ast(&args[0], &args[1]);
    }
    "MapAt" if args.len() == 3 => {
      return list_helpers_ast::map_at_ast(&args[0], &args[1], &args[2]);
    }
    "Select" if args.len() == 2 => {
      return list_helpers_ast::select_ast(&args[0], &args[1], None);
    }
    "Select" if args.len() == 3 => {
      return list_helpers_ast::select_ast(&args[0], &args[1], Some(&args[2]));
    }
    "AllTrue" if args.len() == 2 => {
      return list_helpers_ast::all_true_ast(&args[0], &args[1]);
    }
    "AnyTrue" if args.len() == 2 => {
      return list_helpers_ast::any_true_ast(&args[0], &args[1]);
    }
    "NoneTrue" if args.len() == 2 => {
      return list_helpers_ast::none_true_ast(&args[0], &args[1]);
    }
    "Fold" if args.len() == 2 || args.len() == 3 => {
      if args.len() == 3 {
        return list_helpers_ast::fold_ast(&args[0], &args[1], &args[2]);
      }
      // Fold[f, {a, b, c, ...}] = Fold[f, a, {b, c, ...}]
      if let Expr::List(items) = &args[1] {
        if items.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let init = items[0].clone();
        let rest = Expr::List(items[1..].to_vec());
        return list_helpers_ast::fold_ast(&args[0], &init, &rest);
      }
      return Ok(Expr::FunctionCall {
        name: "Fold".to_string(),
        args: args.to_vec(),
      });
    }
    "CountBy" if args.len() == 2 => {
      return list_helpers_ast::count_by_ast(&args[0], &args[1]);
    }
    "GroupBy" if args.len() == 2 => {
      return list_helpers_ast::group_by_ast(&args[0], &args[1]);
    }
    "SortBy" if args.len() == 2 => {
      return list_helpers_ast::sort_by_ast(&args[0], &args[1]);
    }
    "Ordering" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::ordering_ast(args);
    }
    "Nest" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[2]) {
        return list_helpers_ast::nest_ast(&args[0], &args[1], n);
      }
    }
    "NestList" if args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[2]) {
        return list_helpers_ast::nest_list_ast(&args[0], &args[1], n);
      }
    }
    "FixedPoint" if args.len() >= 2 => {
      let max_iter = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return list_helpers_ast::fixed_point_ast(&args[0], &args[1], max_iter);
    }
    "Cases" if args.len() == 2 => {
      return list_helpers_ast::cases_ast(&args[0], &args[1]);
    }
    "Cases" if args.len() == 3 => {
      return list_helpers_ast::cases_with_level_ast(
        &args[0], &args[1], &args[2],
      );
    }
    "Position" if args.len() == 2 => {
      return list_helpers_ast::position_ast(&args[0], &args[1]);
    }
    "FirstPosition" if args.len() >= 2 => {
      return list_helpers_ast::first_position_ast(args);
    }
    "MapIndexed" if args.len() == 2 => {
      return list_helpers_ast::map_indexed_ast(&args[0], &args[1]);
    }
    "Tally" if args.len() == 1 => {
      return list_helpers_ast::tally_ast(&args[0]);
    }
    "Counts" if args.len() == 1 => {
      return list_helpers_ast::counts_ast(&args[0]);
    }
    "BinCounts" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::bin_counts_ast(args);
    }
    "HistogramList" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::histogram_list_ast(args);
    }
    "DeleteDuplicates" if args.len() == 1 => {
      return list_helpers_ast::delete_duplicates_ast(&args[0]);
    }
    "Union" => {
      return list_helpers_ast::union_ast(args);
    }
    "Intersection" => {
      return list_helpers_ast::intersection_ast(args);
    }
    "Complement" => {
      return list_helpers_ast::complement_ast(args);
    }
    "Dimensions" if args.len() == 1 => {
      return list_helpers_ast::dimensions_ast(args);
    }
    "Delete" if args.len() == 2 => {
      return list_helpers_ast::delete_ast(args);
    }
    "Order" if args.len() == 2 => {
      // Order[e1, e2]: 1 if e1 < e2, -1 if e1 > e2, 0 if equal (canonical ordering)
      let result =
        crate::functions::list_helpers_ast::compare_exprs(&args[0], &args[1]);
      return Ok(Expr::Integer(result as i128));
    }
    "OrderedQ" if args.len() == 1 => {
      return list_helpers_ast::ordered_q_ast(args);
    }
    "DeleteAdjacentDuplicates" if args.len() == 1 => {
      return list_helpers_ast::delete_adjacent_duplicates_ast(args);
    }
    "Commonest" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::commonest_ast(args);
    }
    "ComposeList" if args.len() == 2 => {
      return list_helpers_ast::compose_list_ast(args);
    }
    "ContainsOnly" if args.len() == 2 => {
      return list_helpers_ast::contains_only_ast(args);
    }
    "Pick" if args.len() == 2 || args.len() == 3 => {
      return list_helpers_ast::pick_ast(args);
    }
    "LengthWhile" if args.len() == 2 => {
      return list_helpers_ast::length_while_ast(args);
    }
    "TakeLargestBy" if args.len() == 3 => {
      return list_helpers_ast::take_largest_by_ast(args);
    }
    "TakeSmallestBy" if args.len() == 3 => {
      return list_helpers_ast::take_smallest_by_ast(args);
    }

    // Additional AST-native list functions
    "Table" if args.len() == 2 => {
      return list_helpers_ast::table_ast(&args[0], &args[1]);
    }
    "Table" if args.len() >= 3 => {
      // Multi-dimensional Table: Table[expr, iter1, iter2, ...]
      // Nest from innermost to outermost
      return list_helpers_ast::table_multi_ast(&args[0], &args[1..]);
    }
    "MapThread" if args.len() == 2 || args.len() == 3 => {
      let level = if args.len() == 3 {
        match &args[2] {
          Expr::Integer(n) if *n >= 1 => Some(*n as usize),
          _ => None,
        }
      } else {
        None
      };
      return list_helpers_ast::map_thread_ast(&args[0], &args[1], level);
    }
    "Partition" if args.len() == 2 || args.len() == 3 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let d = if args.len() == 3 {
          expr_to_i128(&args[2])
        } else {
          None
        };
        return list_helpers_ast::partition_ast(&args[0], n, d);
      }
    }
    "Permutations" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::permutations_ast(args);
    }
    "Signature" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        // Check for duplicates first
        let strs: Vec<String> =
          items.iter().map(crate::syntax::expr_to_string).collect();
        for i in 0..strs.len() {
          for j in (i + 1)..strs.len() {
            if strs[i] == strs[j] {
              return Ok(Expr::Integer(0));
            }
          }
        }
        // Count inversions to determine signature
        let mut inversions = 0;
        for i in 0..strs.len() {
          for j in (i + 1)..strs.len() {
            if strs[i] > strs[j] {
              inversions += 1;
            }
          }
        }
        return Ok(Expr::Integer(if inversions % 2 == 0 { 1 } else { -1 }));
      }
      return Ok(Expr::FunctionCall {
        name: "Signature".to_string(),
        args: args.to_vec(),
      });
    }
    "Subsets" if !args.is_empty() && args.len() <= 3 => {
      return list_helpers_ast::subsets_ast(args);
    }
    "Subsequences" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::subsequences_ast(args);
    }
    "SparseArray" if !args.is_empty() => {
      // Return SparseArray unevaluated (like Wolfram); use Normal[] to expand
      return Ok(Expr::FunctionCall {
        name: "SparseArray".to_string(),
        args: args.to_vec(),
      });
    }
    "Normal" if args.len() == 1 => {
      // Normal[SparseArray[...]] expands to a regular list
      if let Expr::FunctionCall {
        name,
        args: sa_args,
      } = &args[0]
        && name == "SparseArray"
      {
        return list_helpers_ast::sparse_array_ast(sa_args);
      }
      // Normal[Dataset[data, ...]] extracts the data
      if let Expr::FunctionCall {
        name,
        args: ds_args,
      } = &args[0]
        && name == "Dataset"
        && !ds_args.is_empty()
      {
        return Ok(ds_args[0].clone());
      }
      // Normal[SeriesData[x, x0, {c0, c1, ...}, nmin, nmax, den]]
      // => sum(c_i * (x - x0)^(nmin + i), i=0..len-1) when den=1
      if let Expr::FunctionCall {
        name,
        args: sd_args,
      } = &args[0]
        && name == "SeriesData"
        && sd_args.len() == 6
      {
        let var = &sd_args[0];
        let x0 = &sd_args[1];
        let coeffs = match &sd_args[2] {
          Expr::List(c) => c,
          _ => return Ok(args[0].clone()),
        };
        let nmin = match &sd_args[3] {
          Expr::Integer(n) => *n,
          _ => return Ok(args[0].clone()),
        };

        if coeffs.is_empty() {
          return Ok(Expr::Integer(0));
        }

        let is_zero_center = matches!(x0, Expr::Integer(0));

        // Build the base expression: x or (-x0 + x)
        let base = if is_zero_center {
          var.clone()
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(Expr::UnaryOp {
              op: crate::syntax::UnaryOperator::Minus,
              operand: Box::new(x0.clone()),
            }),
            right: Box::new(var.clone()),
          }
        };

        // Build terms: c_i * base^(nmin + i)
        let mut terms: Vec<Expr> = Vec::new();
        for (i, coeff) in coeffs.iter().enumerate() {
          if matches!(coeff, Expr::Integer(0)) {
            continue;
          }
          let power = nmin + i as i128;
          // base^power
          let base_pow = if power == 0 {
            None
          } else if power == 1 {
            Some(base.clone())
          } else {
            Some(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: Box::new(base.clone()),
              right: Box::new(Expr::Integer(power)),
            })
          };

          // Build c * x^n in Mathematica's canonical form:
          // Rational[-a,b]*x^n => -(a*x^n)/b  which prints as -(a*x^n)/b
          let term = match base_pow {
            None => coeff.clone(),
            Some(bp) => {
              // Evaluate the Times to get canonical form
              let t = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(coeff.clone()),
                right: Box::new(bp),
              };
              evaluate_expr_to_expr(&t)?
            }
          };
          terms.push(term);
        }

        if terms.is_empty() {
          return Ok(Expr::Integer(0));
        }

        // Combine terms with Plus, preserving order (low to high power)
        let result = terms
          .into_iter()
          .reduce(|acc, t| Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(acc),
            right: Box::new(t),
          })
          .unwrap();
        return Ok(result);
      }
      // For other expressions, Normal is identity
      return Ok(args[0].clone());
    }
    "First" if args.len() == 1 || args.len() == 2 => {
      let default = if args.len() == 2 {
        Some(&args[1])
      } else {
        None
      };
      return list_helpers_ast::first_ast(&args[0], default);
    }
    "Last" if args.len() == 1 || args.len() == 2 => {
      let default = if args.len() == 2 {
        Some(&args[1])
      } else {
        None
      };
      return list_helpers_ast::last_ast(&args[0], default);
    }
    "Rest" if args.len() == 1 => {
      return list_helpers_ast::rest_ast(&args[0]);
    }
    "Most" if args.len() == 1 => {
      return list_helpers_ast::most_ast(&args[0]);
    }
    "Take" if args.len() >= 2 => {
      return list_helpers_ast::take_multi_ast(&args[0], &args[1..]);
    }
    "Drop" if args.len() == 2 => {
      return list_helpers_ast::drop_ast(&args[0], &args[1]);
    }
    "Flatten" if args.len() == 1 => {
      return list_helpers_ast::flatten_ast(&args[0]);
    }
    "Flatten" if args.len() == 2 => {
      // Flatten[expr, n] or Flatten[expr, Infinity, head]
      if let Expr::Identifier(id) = &args[1] {
        if id == "Infinity" {
          // Flatten[expr, Infinity] same as Flatten[expr]
          return list_helpers_ast::flatten_ast(&args[0]);
        }
        // Flatten[expr, head] — treat identifier as head
        return list_helpers_ast::flatten_head_ast(&args[0], i128::MAX, id);
      }
      // Check for dimension spec: Flatten[list, {{2}, {1}}]
      if let Expr::List(outer) = &args[1]
        && !outer.is_empty()
        && matches!(&outer[0], Expr::List(_))
      {
        // Parse dimension spec: each element is a list of level numbers
        let mut dim_spec: Vec<Vec<usize>> = Vec::new();
        let mut valid = true;
        for item in outer {
          if let Expr::List(levels) = item {
            let mut group: Vec<usize> = Vec::new();
            for level in levels {
              if let Some(n) = expr_to_i128(level) {
                group.push(n as usize);
              } else {
                valid = false;
                break;
              }
            }
            dim_spec.push(group);
          } else {
            valid = false;
            break;
          }
          if !valid {
            break;
          }
        }
        if valid {
          return list_helpers_ast::flatten_dims_ast(&args[0], &dim_spec);
        }
      }
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::flatten_level_ast(&args[0], n);
      }
    }
    "Flatten" if args.len() == 3 => {
      // Flatten[expr, depth, head]
      let depth = match &args[1] {
        Expr::Identifier(id) if id == "Infinity" => i128::MAX,
        _ => expr_to_i128(&args[1]).unwrap_or(i128::MAX),
      };
      if let Expr::Identifier(head) = &args[2] {
        return list_helpers_ast::flatten_head_ast(&args[0], depth, head);
      }
    }
    "Level" if args.len() == 2 => {
      return list_helpers_ast::level_ast(&args[0], &args[1], false);
    }
    "Level" if args.len() == 3 => {
      // Extract Heads option
      let include_heads = match &args[2] {
        Expr::Rule {
          pattern,
          replacement,
        } => {
          if let Expr::Identifier(name) = pattern.as_ref() {
            if name == "Heads" {
              matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True")
            } else {
              false
            }
          } else {
            false
          }
        }
        _ => false,
      };
      return list_helpers_ast::level_ast(&args[0], &args[1], include_heads);
    }
    "Reverse" if args.len() == 1 => {
      return list_helpers_ast::reverse_ast(&args[0]);
    }
    "Reverse" if args.len() == 2 => {
      return list_helpers_ast::reverse_level_ast(&args[0], &args[1]);
    }
    "Sort" if args.len() == 1 => {
      return list_helpers_ast::sort_ast(&args[0]);
    }
    "Sort" if args.len() == 2 => {
      // Sort[list, p] - sort using comparator p
      // p[a, b] returns True if a should come before b
      if let Expr::List(items) = &args[0] {
        let cmp_name = crate::syntax::expr_to_string(&args[1]);
        let mut sorted = items.clone();
        match cmp_name.as_str() {
          "Greater" => {
            sorted
              .sort_by(|a, b| list_helpers_ast::canonical_cmp(a, b).reverse());
            return Ok(Expr::List(sorted));
          }
          "Less" => {
            sorted.sort_by(list_helpers_ast::canonical_cmp);
            return Ok(Expr::List(sorted));
          }
          _ => {
            // Custom comparator: evaluate p[a, b] for each comparison
            sorted.sort_by(|a, b| {
              let result = evaluate_expr_to_expr(&Expr::FunctionCall {
                name: cmp_name.clone(),
                args: vec![a.clone(), b.clone()],
              });
              match result {
                Ok(Expr::Identifier(s)) if s == "True" => {
                  std::cmp::Ordering::Less
                }
                _ => std::cmp::Ordering::Greater,
              }
            });
            return Ok(Expr::List(sorted));
          }
        }
      }
    }
    "ReverseSort" if args.len() == 1 || args.len() == 2 => {
      // ReverseSort[list] sorts then reverses
      // ReverseSort[list, p] sorts by p then reverses
      let sorted = evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Sort".to_string(),
        args: args.to_vec(),
      })?;
      if let Expr::List(mut items) = sorted {
        items.reverse();
        return Ok(Expr::List(items));
      }
      return Ok(sorted);
    }
    "List" => {
      // List[a, b, c] is equivalent to {a, b, c}
      return Ok(Expr::List(args.to_vec()));
    }
    "Range" => {
      return list_helpers_ast::range_ast(args);
    }
    "Accumulate" if args.len() == 1 => {
      return list_helpers_ast::accumulate_ast(&args[0]);
    }
    "AnglePath" => {
      return list_helpers_ast::angle_path_ast(args);
    }
    "Differences" if args.len() == 1 => {
      return list_helpers_ast::differences_ast(&args[0]);
    }
    "Differences" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::differences_n_ast(&args[0], n as usize);
      }
    }
    "Ratios" if args.len() == 1 => {
      if let Expr::List(items) = &args[0] {
        if items.len() < 2 {
          return Ok(Expr::List(vec![]));
        }
        let mut result = Vec::with_capacity(items.len() - 1);
        for i in 1..items.len() {
          let ratio = evaluate_expr_to_expr(&Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(items[i].clone()),
            right: Box::new(items[i - 1].clone()),
          })?;
          result.push(ratio);
        }
        return Ok(Expr::List(result));
      }
      return Ok(Expr::FunctionCall {
        name: "Ratios".to_string(),
        args: args.to_vec(),
      });
    }
    "Scan" if args.len() == 2 => {
      return list_helpers_ast::scan_ast(&args[0], &args[1]);
    }
    "FoldList" if args.len() == 2 || args.len() == 3 => {
      if args.len() == 3 {
        return list_helpers_ast::fold_list_ast(&args[0], &args[1], &args[2]);
      }
      // FoldList[f, {a, b, c, ...}] = FoldList[f, a, {b, c, ...}]
      if let Expr::List(items) = &args[1] {
        if items.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        let init = items[0].clone();
        let rest = Expr::List(items[1..].to_vec());
        return list_helpers_ast::fold_list_ast(&args[0], &init, &rest);
      }
      return Ok(Expr::FunctionCall {
        name: "FoldList".to_string(),
        args: args.to_vec(),
      });
    }
    "FixedPointList" if args.len() >= 2 => {
      let max_iter = if args.len() == 3 {
        expr_to_i128(&args[2])
      } else {
        None
      };
      return list_helpers_ast::fixed_point_list_ast(
        &args[0], &args[1], max_iter,
      );
    }
    "Transpose" if args.len() == 1 => {
      return list_helpers_ast::transpose_ast(&args[0]);
    }
    "Diagonal" if args.len() == 1 || args.len() == 2 => {
      let offset = if args.len() == 2 {
        match &args[1] {
          Expr::Integer(n) => *n as i64,
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Diagonal".to_string(),
              args: args.to_vec(),
            });
          }
        }
      } else {
        0
      };
      if let Expr::List(rows) = &args[0] {
        let mut result = Vec::new();
        let nrows = rows.len() as i64;
        for (i, row) in rows.iter().enumerate() {
          if let Expr::List(cols) = row {
            let j = i as i64 + offset;
            if j >= 0 && (j as usize) < cols.len() && (i as i64) < nrows {
              result.push(cols[j as usize].clone());
            }
          }
        }
        return Ok(Expr::List(result));
      }
    }
    "Riffle" if args.len() == 2 => {
      return list_helpers_ast::riffle_ast(&args[0], &args[1]);
    }
    "RotateLeft" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::rotate_left_ast(&args[0], n);
      }
      if let Expr::List(shifts) = &args[1] {
        return list_helpers_ast::rotate_multi_ast(&args[0], shifts, true);
      }
    }
    "RotateLeft" if args.len() == 1 => {
      return list_helpers_ast::rotate_left_ast(&args[0], 1);
    }
    "RotateRight" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::rotate_right_ast(&args[0], n);
      }
      if let Expr::List(shifts) = &args[1] {
        return list_helpers_ast::rotate_multi_ast(&args[0], shifts, false);
      }
    }
    "RotateRight" if args.len() == 1 => {
      return list_helpers_ast::rotate_right_ast(&args[0], 1);
    }
    "PadLeft" if args.len() == 1 => {
      // PadLeft[{{}, {1, 2}, {1, 2, 3}}] - auto-pad ragged array
      if let Expr::List(items) = &args[0] {
        let max_len = items
          .iter()
          .filter_map(|item| match item {
            Expr::List(sub) => Some(sub.len()),
            _ => None,
          })
          .max()
          .unwrap_or(0);
        let padded: Vec<Expr> = items
          .iter()
          .map(|item| {
            list_helpers_ast::pad_left_ast(
              item,
              max_len as i128,
              &Expr::Integer(0),
              None,
            )
            .unwrap_or_else(|_| item.clone())
          })
          .collect();
        return Ok(Expr::List(padded));
      }
    }
    "PadLeft" if args.len() >= 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() >= 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        let offset = if args.len() >= 4 {
          expr_to_i128(&args[3])
        } else {
          None
        };
        return list_helpers_ast::pad_left_ast(&args[0], n, &pad, offset);
      }
    }
    "PadRight" if args.len() == 1 => {
      // PadRight[{{}, {1, 2}, {1, 2, 3}}] - auto-pad ragged array
      if let Expr::List(items) = &args[0] {
        let max_len = items
          .iter()
          .filter_map(|item| match item {
            Expr::List(sub) => Some(sub.len()),
            _ => None,
          })
          .max()
          .unwrap_or(0);
        let padded: Vec<Expr> = items
          .iter()
          .map(|item| {
            list_helpers_ast::pad_right_ast(
              item,
              max_len as i128,
              &Expr::Integer(0),
              None,
            )
            .unwrap_or_else(|_| item.clone())
          })
          .collect();
        return Ok(Expr::List(padded));
      }
    }
    "PadRight" if args.len() >= 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() >= 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        let offset = if args.len() >= 4 {
          expr_to_i128(&args[3])
        } else {
          None
        };
        return list_helpers_ast::pad_right_ast(&args[0], n, &pad, offset);
      }
    }
    "Join" => {
      return list_helpers_ast::join_ast(args);
    }
    "Append" if args.len() == 2 => {
      return list_helpers_ast::append_ast(&args[0], &args[1]);
    }
    "Prepend" if args.len() == 2 => {
      return list_helpers_ast::prepend_ast(&args[0], &args[1]);
    }
    "DeleteDuplicatesBy" if args.len() == 2 => {
      return list_helpers_ast::delete_duplicates_by_ast(&args[0], &args[1]);
    }
    "Median" if args.len() == 1 => {
      return list_helpers_ast::median_ast(&args[0]);
    }
    "Count" if args.len() == 2 => {
      return list_helpers_ast::count_ast(&args[0], &args[1]);
    }
    "Count" if args.len() == 3 => {
      return list_helpers_ast::count_ast_level(
        &args[0],
        &args[1],
        Some(&args[2]),
      );
    }
    "ConstantArray" if args.len() == 2 => {
      return list_helpers_ast::constant_array_ast(&args[0], &args[1]);
    }
    "NestWhile" if args.len() >= 3 => {
      let max_iter = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return list_helpers_ast::nest_while_ast(
        &args[0], &args[1], &args[2], max_iter,
      );
    }
    "NestWhileList" if args.len() >= 3 => {
      let max_iter = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return list_helpers_ast::nest_while_list_ast(
        &args[0], &args[1], &args[2], max_iter,
      );
    }
    "Thread" if args.len() == 1 => {
      return list_helpers_ast::thread_ast(&args[0], None);
    }
    "Thread" if args.len() == 2 => {
      if let Expr::Identifier(head) = &args[1] {
        return list_helpers_ast::thread_ast(&args[0], Some(head));
      }
      return list_helpers_ast::thread_ast(&args[0], None);
    }
    "Through" if args.len() == 1 => {
      return list_helpers_ast::through_ast(&args[0], None);
    }
    "Through" if args.len() == 2 => {
      // Through[expr, h] - only apply if head of expr matches h
      let head_filter = crate::syntax::expr_to_string(&args[1]);
      return list_helpers_ast::through_ast(&args[0], Some(&head_filter));
    }
    "Operate" if args.len() == 2 || args.len() == 3 => {
      let p = &args[0];
      let expr = &args[1];
      let n = if args.len() == 3 {
        expr_to_i128(&args[2]).unwrap_or(1)
      } else {
        1
      };
      if n == 0 {
        return Ok(Expr::FunctionCall {
          name: "".to_string(),
          args: vec![expr.clone()],
        })
        .map(|_| Expr::FunctionCall {
          name: crate::syntax::expr_to_string(p),
          args: vec![expr.clone()],
        });
      }
      // For n >= 1, we need to wrap the head at depth n
      // For n == 1 (default): f[a, b] -> p[f][a, b]
      fn wrap_head_at_depth(expr: &Expr, p: &Expr, depth: i128) -> Expr {
        if depth == 0 {
          Expr::FunctionCall {
            name: crate::syntax::expr_to_string(p),
            args: vec![expr.clone()],
          }
        } else {
          match expr {
            Expr::FunctionCall { name, args } => {
              let wrapped_head = wrap_head_at_depth(
                &Expr::Identifier(name.clone()),
                p,
                depth - 1,
              );
              Expr::CurriedCall {
                func: Box::new(wrapped_head),
                args: args.clone(),
              }
            }
            Expr::CurriedCall { func, args } => {
              let wrapped_func = wrap_head_at_depth(func, p, depth - 1);
              Expr::CurriedCall {
                func: Box::new(wrapped_func),
                args: args.clone(),
              }
            }
            _ => Expr::FunctionCall {
              name: crate::syntax::expr_to_string(p),
              args: vec![expr.clone()],
            },
          }
        }
      }
      return Ok(wrap_head_at_depth(expr, p, n));
    }
    "TakeLargest" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::take_largest_ast(&args[0], n);
      }
    }
    "TakeSmallest" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::take_smallest_ast(&args[0], n);
      }
    }
    "MinimalBy" if args.len() == 2 => {
      return list_helpers_ast::minimal_by_ast(&args[0], &args[1]);
    }
    "MaximalBy" if args.len() == 2 => {
      return list_helpers_ast::maximal_by_ast(&args[0], &args[1]);
    }
    "ArrayDepth" if args.len() == 1 => {
      return list_helpers_ast::array_depth_ast(&args[0]);
    }
    "ArrayQ" if args.len() == 1 => {
      return list_helpers_ast::array_q_ast(&args[0]);
    }
    "VectorQ" if args.len() == 1 => {
      return list_helpers_ast::vector_q_ast(&args[0]);
    }
    "MatrixQ" if args.len() == 1 => {
      return list_helpers_ast::matrix_q_ast(&args[0]);
    }
    "TakeWhile" if args.len() == 2 => {
      return list_helpers_ast::take_while_ast(&args[0], &args[1]);
    }
    "Do" if args.len() == 2 => {
      return list_helpers_ast::do_ast(&args[0], &args[1]);
    }
    "Do" if args.len() > 2 => {
      // Multi-iterator Do: Do[body, {i, ...}, {j, ...}, ...]
      // Nest the iterators: outermost is first iterator, innermost is last
      // Build a nested Do: Do[Do[body, last_iter], ..., first_iter]
      let body = &args[0];
      let iters = &args[1..];
      let mut nested = body.clone();
      for iter in iters.iter().rev() {
        nested = Expr::FunctionCall {
          name: "Do".to_string(),
          args: vec![nested, iter.clone()],
        };
      }
      return evaluate_expr_to_expr(&nested);
    }
    "For" if args.len() == 3 || args.len() == 4 => {
      return for_ast(args);
    }
    "DeleteCases" if args.len() == 2 => {
      return list_helpers_ast::delete_cases_ast(&args[0], &args[1]);
    }
    "DeleteCases" if args.len() == 3 || args.len() == 4 => {
      // DeleteCases[list, pattern, levelspec] or DeleteCases[list, pattern, levelspec, n]
      // For now, levelspec is ignored (treated as level 1)
      let max_count = if args.len() == 4 {
        expr_to_i128(&args[3])
      } else {
        None
      };
      return list_helpers_ast::delete_cases_with_count_ast(
        &args[0], &args[1], max_count,
      );
    }
    "MinMax" if args.len() == 1 => {
      return list_helpers_ast::min_max_ast(&args[0]);
    }
    "Part" if args.len() == 2 => {
      return list_helpers_ast::part_ast(&args[0], &args[1]);
    }
    "Insert" if args.len() == 3 => {
      return list_helpers_ast::insert_ast(&args[0], &args[1], &args[2]);
    }
    "Array" if args.len() >= 2 && args.len() <= 4 => {
      if args.len() == 2
        && let Some(n) = expr_to_i128(&args[1])
      {
        return list_helpers_ast::array_ast(&args[0], n);
      }
      if matches!(&args[1], Expr::List(_)) || args.len() > 2 {
        return list_helpers_ast::array_multi_ast(args);
      }
    }
    "Gather" if args.len() == 1 => {
      return list_helpers_ast::gather_ast(&args[0]);
    }
    "GatherBy" if args.len() == 2 => {
      return list_helpers_ast::gather_by_ast(&args[1], &args[0]);
    }
    "Split" if args.len() == 1 || args.len() == 2 => {
      if args.len() == 1 {
        return list_helpers_ast::split_ast(&args[0]);
      }
      return list_helpers_ast::split_with_test_ast(&args[0], &args[1]);
    }
    "SplitBy" if args.len() == 2 => {
      return list_helpers_ast::split_by_ast(&args[1], &args[0]);
    }
    "Extract" if args.len() == 2 => {
      return list_helpers_ast::extract_ast(&args[0], &args[1]);
    }
    "Catenate" if args.len() == 1 => {
      return list_helpers_ast::catenate_ast(&args[0]);
    }
    "Apply" if args.len() == 2 => {
      return list_helpers_ast::apply_ast(&args[0], &args[1]);
    }
    "Apply" if args.len() == 3 => {
      return list_helpers_ast::apply_at_level_ast(
        &args[0], &args[1], &args[2],
      );
    }
    "Identity" if args.len() == 1 => {
      return list_helpers_ast::identity_ast(&args[0]);
    }
    // Composition[] → Identity
    "Composition" if args.is_empty() => {
      return Ok(Expr::Identifier("Identity".to_string()));
    }
    // Composition[f] → f
    "Composition" if args.len() == 1 => {
      return Ok(args[0].clone());
    }
    // Composition[f, Composition[g, h], k] → Composition[f, g, h, k]
    "Composition" if args.len() >= 2 => {
      let mut flat = Vec::new();
      for arg in args {
        if let Expr::FunctionCall { name: n, args: a } = arg
          && n == "Composition"
        {
          flat.extend(a.iter().cloned());
          continue;
        }
        flat.push(arg.clone());
      }
      return Ok(Expr::FunctionCall {
        name: "Composition".to_string(),
        args: flat,
      });
    }
    "Outer" if args.len() >= 3 => {
      return list_helpers_ast::outer_ast(&args[0], &args[1..]);
    }
    "Inner" if args.len() == 4 => {
      return list_helpers_ast::inner_ast(
        &args[0], &args[1], &args[2], &args[3],
      );
    }
    "ReplacePart" if args.len() == 2 => {
      return list_helpers_ast::replace_part_ast(&args[0], &args[1]);
    }

    // AST-native string functions
    "StringLength" if args.len() == 1 => {
      return crate::functions::string_ast::string_length_ast(args);
    }
    "StringTake" if args.len() == 2 => {
      return crate::functions::string_ast::string_take_ast(args);
    }
    "StringDrop" if args.len() == 2 => {
      return crate::functions::string_ast::string_drop_ast(args);
    }
    "Compress" if args.len() == 1 => {
      return crate::functions::string_ast::compress_ast(args);
    }
    "Uncompress" if args.len() == 1 => {
      return crate::functions::string_ast::uncompress_ast(args);
    }
    "StringJoin" => {
      return crate::functions::string_ast::string_join_ast(args);
    }
    "StringSplit" if !args.is_empty() => {
      return crate::functions::string_ast::string_split_ast(args);
    }
    "StringStartsQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_starts_q_ast(args);
    }
    "StringEndsQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_ends_q_ast(args);
    }
    "StringContainsQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_contains_q_ast(args);
    }
    "StringReplace" if args.len() == 2 || args.len() == 3 => {
      return crate::functions::string_ast::string_replace_ast(args);
    }
    "ToUpperCase" if args.len() == 1 => {
      return crate::functions::string_ast::to_upper_case_ast(args);
    }
    "ToLowerCase" if args.len() == 1 => {
      return crate::functions::string_ast::to_lower_case_ast(args);
    }
    "Characters" if args.len() == 1 => {
      return crate::functions::string_ast::characters_ast(args);
    }
    "StringRiffle" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::string_ast::string_riffle_ast(args);
    }
    "StringPosition" if args.len() == 2 || args.len() == 3 => {
      return crate::functions::string_ast::string_position_ast(args);
    }
    "StringMatchQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_match_q_ast(args);
    }
    "StringReverse" if args.len() == 1 => {
      return crate::functions::string_ast::string_reverse_ast(args);
    }
    "StringRepeat" if args.len() == 2 || args.len() == 3 => {
      return crate::functions::string_ast::string_repeat_ast(args);
    }
    "StringTrim" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::string_ast::string_trim_ast(args);
    }
    "StringCases" if args.len() == 2 => {
      return crate::functions::string_ast::string_cases_ast(args);
    }
    "ToString" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::string_ast::to_string_ast(args);
    }
    "ToExpression" if args.len() == 1 => {
      return crate::functions::string_ast::to_expression_ast(args);
    }
    "StringPadLeft" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::string_ast::string_pad_left_ast(args);
    }
    "StringPadRight" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::string_ast::string_pad_right_ast(args);
    }
    "EditDistance" if args.len() == 2 => {
      return crate::functions::string_ast::edit_distance_ast(args);
    }
    "LongestCommonSubsequence" if args.len() == 2 => {
      return crate::functions::string_ast::longest_common_subsequence_ast(
        args,
      );
    }
    "StringCount" if args.len() == 2 => {
      return crate::functions::string_ast::string_count_ast(args);
    }
    "StringFreeQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_free_q_ast(args);
    }
    "ToCharacterCode" if args.len() == 1 => {
      return crate::functions::string_ast::to_character_code_ast(args);
    }
    "FromCharacterCode" if args.len() == 1 => {
      return crate::functions::string_ast::from_character_code_ast(args);
    }
    "CharacterRange" if args.len() == 2 => {
      return crate::functions::string_ast::character_range_ast(args);
    }
    "IntegerString" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::string_ast::integer_string_ast(args);
    }
    "Alphabet" if args.is_empty() => {
      return crate::functions::string_ast::alphabet_ast(args);
    }
    "LetterQ" if args.len() == 1 => {
      return crate::functions::string_ast::letter_q_ast(args);
    }
    "UpperCaseQ" if args.len() == 1 => {
      return crate::functions::string_ast::upper_case_q_ast(args);
    }
    "LowerCaseQ" if args.len() == 1 => {
      return crate::functions::string_ast::lower_case_q_ast(args);
    }
    "DigitQ" if args.len() == 1 => {
      return crate::functions::string_ast::digit_q_ast(args);
    }
    "StringInsert" if args.len() == 3 => {
      return crate::functions::string_ast::string_insert_ast(args);
    }
    "StringDelete" if args.len() == 2 => {
      return crate::functions::string_ast::string_delete_ast(args);
    }
    "Capitalize" if args.len() == 1 => {
      return crate::functions::string_ast::capitalize_ast(args);
    }
    "Decapitalize" if args.len() == 1 => {
      return crate::functions::string_ast::decapitalize_ast(args);
    }
    "StringPart" if args.len() == 2 => {
      return crate::functions::string_ast::string_part_ast(args);
    }
    "StringTakeDrop" if args.len() == 2 => {
      return crate::functions::string_ast::string_take_drop_ast(args);
    }
    "HammingDistance" if args.len() == 2 => {
      return crate::functions::string_ast::hamming_distance_ast(args);
    }
    "CharacterCounts" if args.len() == 1 => {
      return crate::functions::string_ast::character_counts_ast(args);
    }
    "RemoveDiacritics" if args.len() == 1 => {
      return crate::functions::string_ast::remove_diacritics_ast(args);
    }
    "StringRotateLeft" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::string_ast::string_rotate_left_ast(args);
    }
    "StringRotateRight" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::string_ast::string_rotate_right_ast(args);
    }
    "AlphabeticSort" if args.len() == 1 => {
      return crate::functions::string_ast::alphabetic_sort_ast(args);
    }
    "Hash" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::string_ast::hash_ast(args);
    }

    // ── Image functions ──────────────────────────────────────────────
    "Image" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::image_constructor_ast(args);
    }
    "ImageQ" if args.len() == 1 => {
      return crate::functions::image_ast::image_q_ast(args);
    }
    "ImageDimensions" if args.len() == 1 => {
      return crate::functions::image_ast::image_dimensions_ast(args);
    }
    "ImageChannels" if args.len() == 1 => {
      return crate::functions::image_ast::image_channels_ast(args);
    }
    "ImageType" if args.len() == 1 => {
      return crate::functions::image_ast::image_type_ast(args);
    }
    "ImageData" if args.len() == 1 => {
      return crate::functions::image_ast::image_data_ast(args);
    }
    "ImageColorSpace" if args.len() == 1 => {
      return crate::functions::image_ast::image_color_space_ast(args);
    }
    "ColorNegate" if args.len() == 1 => {
      return crate::functions::image_ast::color_negate_ast(args);
    }
    "Binarize" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::binarize_ast(args);
    }
    "Blur" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::blur_ast(args);
    }
    "Sharpen" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::sharpen_ast(args);
    }
    "ImageAdjust" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::image_adjust_ast(args);
    }
    "ImageReflect" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::image_reflect_ast(args);
    }
    "ImageRotate" if args.len() == 2 => {
      return crate::functions::image_ast::image_rotate_ast(args);
    }
    "ImageResize" if args.len() == 2 => {
      return crate::functions::image_ast::image_resize_ast(args);
    }
    "ImageCrop" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::image_crop_ast(args);
    }
    "ImageTake" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::image_ast::image_take_ast(args);
    }
    "EdgeDetect" if args.len() == 1 => {
      return crate::functions::image_ast::edge_detect_ast(args);
    }
    "DominantColors" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::image_ast::dominant_colors_ast(args);
    }
    "ImageApply" if args.len() == 2 => {
      return crate::functions::image_ast::image_apply_ast(
        args,
        &evaluate_expr_to_expr,
      );
    }
    "ColorConvert" if args.len() == 2 => {
      if matches!(&args[0], Expr::Image { .. }) {
        return crate::functions::image_ast::color_convert_ast(args);
      }
    }
    "ImageCompose" if args.len() == 2 => {
      return crate::functions::image_ast::image_compose_ast(args);
    }
    "ImageAdd" if args.len() == 2 => {
      return crate::functions::image_ast::image_add_ast(args);
    }
    "ImageSubtract" if args.len() == 2 => {
      return crate::functions::image_ast::image_subtract_ast(args);
    }
    "ImageMultiply" if args.len() == 2 => {
      return crate::functions::image_ast::image_multiply_ast(args);
    }
    "RandomImage" if args.len() <= 2 => {
      return crate::functions::image_ast::random_image_ast(args);
    }

    // Import — works on both CLI and WASM (URLs), CLI-only for local files
    "Import" if args.len() == 1 => {
      let path = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Import".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let is_url = path.starts_with("http://") || path.starts_with("https://");

      if is_url {
        // URL import — available on both CLI and WASM
        #[cfg(not(target_arch = "wasm32"))]
        {
          return crate::functions::image_ast::import_image_from_url(&path);
        }
        #[cfg(target_arch = "wasm32")]
        {
          return crate::wasm::import_image_from_url_wasm(&path);
        }
      }

      // Local file import — CLI only
      #[cfg(not(target_arch = "wasm32"))]
      {
        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
        match ext.as_str() {
          "jpg" | "jpeg" | "png" | "gif" | "bmp" | "tiff" | "tif" => {
            return crate::functions::image_ast::import_image(&path);
          }
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Import: unsupported file format \"{}\"",
              ext
            )));
          }
        }
      }
      #[cfg(target_arch = "wasm32")]
      {
        return Err(InterpreterError::EvaluationError(
          "Import: local file access is not available in the browser".into(),
        ));
      }
    }
    // ReadList[source] or ReadList[source, type] or ReadList[source, type, n]
    #[cfg(not(target_arch = "wasm32"))]
    "ReadList" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::string_ast::read_list_ast(args);
    }
    // Get[file] — read and evaluate a file, returning the last result
    #[cfg(not(target_arch = "wasm32"))]
    "Get" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Get".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let content = match std::fs::read_to_string(&filename) {
        Ok(c) => c,
        Err(_) => {
          eprintln!("Get::noopen: Cannot open {}.", filename);
          return Ok(Expr::Identifier("$Failed".to_string()));
        }
      };
      // Use interpret to evaluate the file content (handles all node types
      // including FunctionDefinition, Expression, etc.)
      let result_str = crate::interpret(&content)?;
      let result = crate::syntax::string_to_expr(&result_str)
        .unwrap_or(Expr::Identifier(result_str));
      return Ok(result);
    }
    // Put[expr1, expr2, ..., "file"] — write expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "Put" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Put".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      let to_write = if exprs.is_empty() {
        String::new()
      } else {
        format!("{}\n", content)
      };
      match std::fs::write(&filename, to_write) {
        Ok(_) => return Ok(Expr::Identifier("Null".to_string())),
        Err(_e) => {
          eprintln!("Put::noopen: Cannot open {}.", filename);
          return Ok(Expr::Identifier("$Failed".to_string()));
        }
      }
    }
    // PutAppend[expr1, expr2, ..., "file"] — append expressions to a file
    #[cfg(not(target_arch = "wasm32"))]
    "PutAppend" if !args.is_empty() => {
      let filename = match args.last().unwrap() {
        Expr::String(s) => s.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "PutAppend".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let exprs = &args[..args.len() - 1];
      let content = exprs
        .iter()
        .map(crate::syntax::expr_to_string)
        .collect::<Vec<_>>()
        .join("\n");
      if !exprs.is_empty() {
        use std::io::Write;
        let to_write = format!("{}\n", content);
        match std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&filename)
        {
          Ok(mut file) => {
            if file.write_all(to_write.as_bytes()).is_err() {
              eprintln!("PutAppend::noopen: Cannot open {}.", filename);
              return Ok(Expr::Identifier("$Failed".to_string()));
            }
          }
          Err(_) => {
            eprintln!("PutAppend::noopen: Cannot open {}.", filename);
            return Ok(Expr::Identifier("$Failed".to_string()));
          }
        }
      }
      return Ok(Expr::Identifier("Null".to_string()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Export" if args.len() >= 2 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Err(InterpreterError::EvaluationError(format!(
            "Export: first argument must be a filename string, got {}",
            crate::syntax::expr_to_string(other)
          )));
        }
      };
      // Handle Image export
      if let Expr::Image {
        width,
        height,
        channels,
        data,
        ..
      } = &args[1]
      {
        crate::functions::image_ast::export_image(
          &filename, *width, *height, *channels, data,
        )?;
        return Ok(Expr::String(filename));
      }
      // The second argument has already been evaluated, which triggers
      // capture_graphics() for Plot expressions.  Grab the SVG.
      let content = match &args[1] {
        Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
          crate::get_captured_graphics().ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Export: no graphics to export".into(),
            )
          })?
        }
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_string(other),
      };
      std::fs::write(&filename, &content).map_err(|e| {
        InterpreterError::EvaluationError(format!("Export: {e}"))
      })?;
      return Ok(Expr::String(filename));
    }
    "ExportString" if args.len() == 2 => {
      // ExportString[expr, "SVG"] - return SVG string representation
      let format_str = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          // Return unevaluated for non-string format
          return Ok(Expr::FunctionCall {
            name: "ExportString".to_string(),
            args: args.to_vec(),
          });
        }
      };
      if format_str != "SVG" {
        // Only SVG supported for now; return unevaluated for other formats
        return Ok(Expr::FunctionCall {
          name: "ExportString".to_string(),
          args: args.to_vec(),
        });
      }
      let svg = match &args[0] {
        Expr::Identifier(s) if s == "-Graphics-" || s == "-Graphics3D-" => {
          crate::get_captured_graphics().unwrap_or_default()
        }
        Expr::FunctionCall {
          name: grid_name,
          args: grid_args,
        } if grid_name == "Grid" && !grid_args.is_empty() => {
          // Grid[...] → render as SVG table
          if crate::functions::graphics::grid_ast(grid_args).is_ok() {
            crate::get_captured_graphics().unwrap_or_default()
          } else {
            String::new()
          }
        }
        Expr::FunctionCall {
          name: ds_name,
          args: ds_args,
        } if ds_name == "Dataset" && !ds_args.is_empty() => {
          // Dataset[data, ...] → render as SVG table
          if let Some(svg) =
            crate::functions::graphics::dataset_to_svg(&ds_args[0])
          {
            svg
          } else {
            // Fallback: render as text SVG
            let markup =
              crate::functions::graphics::expr_to_svg_markup(&args[0]);
            let char_width = 8.4_f64;
            let font_size = 14_usize;
            let display_width =
              crate::functions::graphics::estimate_display_width(&args[0]);
            let width = (display_width * char_width).ceil() as usize;
            let (height, text_y) =
              if crate::functions::graphics::has_fraction(&args[0]) {
                (32_usize, 18_usize)
              } else {
                (font_size + 4, font_size)
              };
            format!(
              "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\
               <text x=\"0\" y=\"{text_y}\" font-family=\"monospace\" font-size=\"{font_size}\">{markup}</text>\
               </svg>",
              width, height
            )
          }
        }
        other => {
          // Non-graphics: render expression as SVG text with superscripts
          let markup = crate::functions::graphics::expr_to_svg_markup(other);
          let char_width = 8.4_f64;
          let font_size = 14_usize;
          let display_width =
            crate::functions::graphics::estimate_display_width(other);
          let width = (display_width * char_width).ceil() as usize;
          let (height, text_y) =
            if crate::functions::graphics::has_fraction(other) {
              (32_usize, 18_usize)
            } else {
              (font_size + 4, font_size)
            };
          format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\
             <text x=\"0\" y=\"{text_y}\" font-family=\"monospace\" font-size=\"{font_size}\">{markup}</text>\
             </svg>",
            width, height
          )
        }
      };
      return Ok(Expr::String(svg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Find" if args.len() == 2 => {
      // Find[stream_or_file, "text"] - find first line containing text
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Find".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let search = match &args[1] {
        Expr::String(s) => s.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Find: second argument must be a string".into(),
          ));
        }
      };
      let content = std::fs::read_to_string(&filename)
        .map_err(|e| InterpreterError::EvaluationError(format!("Find: {e}")))?;
      for line in content.lines() {
        if line.contains(&search) {
          return Ok(Expr::String(line.to_string()));
        }
      }
      return Ok(Expr::Identifier("EndOfFile".to_string()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "CreateFile" => {
      let filename_opt = if args.is_empty() {
        None
      } else if let Expr::String(s) = &args[0] {
        Some(s.clone())
      } else {
        let s = expr_to_raw_string(&args[0]);
        Some(s)
      };
      return match crate::utils::create_file(filename_opt) {
        Ok(path) => Ok(Expr::String(path.to_string_lossy().into_owned())),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      };
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Directory" if args.is_empty() => {
      return match std::env::current_dir() {
        Ok(path) => Ok(Expr::String(path.to_string_lossy().into_owned())),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      };
    }
    // OpenRead[file] — open a file for reading, return InputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenRead" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Ok(Expr::FunctionCall {
            name: "OpenRead".to_string(),
            args: vec![other.clone()],
          });
        }
      };
      if !std::path::Path::new(&filename).exists() {
        eprintln!("OpenRead::noopen: Cannot open {}.", filename);
        return Ok(Expr::Identifier("$Failed".to_string()));
      }
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      });
    }
    // OpenWrite[file] — open a file for writing, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenWrite" if args.len() <= 1 => {
      let filename = if args.is_empty() {
        let path = crate::utils::create_file(None)
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))?;
        path.to_string_lossy().into_owned()
      } else {
        match &args[0] {
          Expr::String(s) => s.clone(),
          other => {
            return Ok(Expr::FunctionCall {
              name: "OpenWrite".to_string(),
              args: vec![other.clone()],
            });
          }
        }
      };
      // Create or truncate the file
      std::fs::File::create(&filename).map_err(|e| {
        InterpreterError::EvaluationError(format!(
          "OpenWrite: cannot open {}: {}",
          filename, e
        ))
      })?;
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Ok(Expr::FunctionCall {
        name: "OutputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      });
    }
    // OpenAppend[file] — open a file for appending, return OutputStream[name, id]
    #[cfg(not(target_arch = "wasm32"))]
    "OpenAppend" if args.len() == 1 => {
      let filename = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Ok(Expr::FunctionCall {
            name: "OpenAppend".to_string(),
            args: vec![other.clone()],
          });
        }
      };
      // Open for appending (create if not exists)
      std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filename)
        .map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "OpenAppend: cannot open {}: {}",
            filename, e
          ))
        })?;
      let id = crate::register_stream(
        filename.clone(),
        crate::StreamKind::FileStream(filename.clone()),
      );
      return Ok(Expr::FunctionCall {
        name: "OutputStream".to_string(),
        args: vec![Expr::String(filename), Expr::Integer(id as i128)],
      });
    }
    // StringToStream["text"] — create an input stream from a string
    "StringToStream" if args.len() == 1 => {
      let text = match &args[0] {
        Expr::String(s) => s.clone(),
        other => {
          return Err(InterpreterError::EvaluationError(format!(
            "StringToStream: argument must be a string, got {}",
            crate::syntax::expr_to_string(other)
          )));
        }
      };
      let id = crate::register_stream(
        "String".to_string(),
        crate::StreamKind::StringStream(text),
      );
      return Ok(Expr::FunctionCall {
        name: "InputStream".to_string(),
        args: vec![
          Expr::String("String".to_string()),
          Expr::Integer(id as i128),
        ],
      });
    }
    // Close[stream] — close an open stream
    "Close" if args.len() == 1 => {
      // Extract stream ID from InputStream[name, id] or OutputStream[name, id]
      match &args[0] {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          let id = match &stream_args[1] {
            Expr::Integer(n) => *n as usize,
            _ => {
              return Ok(Expr::FunctionCall {
                name: "Close".to_string(),
                args: args.to_vec(),
              });
            }
          };
          match crate::close_stream(id) {
            Some(name) => return Ok(Expr::String(name)),
            None => {
              let stream_str = crate::syntax::expr_to_string(&args[0]);
              eprintln!("{} is not open.", stream_str);
              return Ok(Expr::FunctionCall {
                name: "Close".to_string(),
                args: args.to_vec(),
              });
            }
          }
        }
        Expr::String(s) => {
          eprintln!("{} is not open.", s);
          return Ok(Expr::FunctionCall {
            name: "Close".to_string(),
            args: args.to_vec(),
          });
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Close".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    // Read[stream] or Read[stream, type] — read from a stream
    "Read" if !args.is_empty() && args.len() <= 2 => {
      let stream = &args[0];
      let stream_id = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "InputStream"
          || stream_head == "OutputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            Some(*id as usize)
          } else {
            None
          }
        }
        _ => None,
      };

      if let Some(id) = stream_id
        && let Some((content, position)) = crate::get_stream_content(id)
      {
        let remaining = &content[position.min(content.len())..];

        // Determine the read type
        let read_type = if args.len() == 2 {
          &args[1]
        } else {
          &Expr::Identifier("Expression".to_string())
        };

        // Handle list of types: Read[stream, {type1, type2, ...}]
        if let Expr::List(types) = read_type {
          let mut results = Vec::new();
          let mut current_pos = position;
          for t in types {
            let rem = &content[current_pos.min(content.len())..];
            let (val, advance) = read_single_type(rem, t);
            current_pos += advance;
            results.push(val);
          }
          crate::advance_stream_position(id, current_pos);
          return Ok(Expr::List(results));
        }

        let (result, advance) = read_single_type(remaining, read_type);
        crate::advance_stream_position(id, position + advance);
        return Ok(result);
      }

      return Ok(Expr::FunctionCall {
        name: "Read".to_string(),
        args: args.to_vec(),
      });
    }
    // Write[stream, expr1, expr2, ...] — write expressions to a stream in OutputForm
    #[cfg(not(target_arch = "wasm32"))]
    "Write" if args.len() >= 2 => {
      let stream = &args[0];
      let file_path = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "OutputStream"
          || stream_head == "InputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let stream_id = *id as usize;
            crate::STREAM_REGISTRY.with(|reg| {
              let registry = reg.borrow();
              registry.get(&stream_id).and_then(|s| match &s.kind {
                crate::StreamKind::FileStream(path) => Some(path.clone()),
                _ => None,
              })
            })
          } else {
            None
          }
        }
        Expr::String(path) => Some(path.clone()),
        _ => None,
      };

      if let Some(path) = file_path {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&path)
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "Write: cannot open {}: {}",
              path, e
            ))
          })?;
        let mut content = String::new();
        for arg in &args[1..] {
          content.push_str(&crate::syntax::expr_to_string(arg));
        }
        content.push('\n');
        file.write_all(content.as_bytes()).map_err(|e| {
          InterpreterError::EvaluationError(format!(
            "Write: write error: {}",
            e
          ))
        })?;
        return Ok(Expr::Identifier("Null".to_string()));
      }

      return Ok(Expr::FunctionCall {
        name: "Write".to_string(),
        args: args.to_vec(),
      });
    }
    // WriteString[stream, "text1", "text2", ...] — write strings to a stream
    #[cfg(not(target_arch = "wasm32"))]
    "WriteString" if args.len() >= 2 => {
      let stream = &args[0];
      // Extract stream info
      let file_path = match stream {
        Expr::FunctionCall {
          name: stream_head,
          args: stream_args,
        } if (stream_head == "OutputStream"
          || stream_head == "InputStream")
          && stream_args.len() == 2 =>
        {
          if let Expr::Integer(id) = &stream_args[1] {
            let stream_id = *id as usize;
            crate::STREAM_REGISTRY.with(|reg| {
              let registry = reg.borrow();
              registry.get(&stream_id).and_then(|s| match &s.kind {
                crate::StreamKind::FileStream(path) => Some(path.clone()),
                _ => None,
              })
            })
          } else {
            None
          }
        }
        Expr::String(path) => Some(path.clone()),
        _ => None,
      };

      if let Some(path) = file_path {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
          .create(true)
          .append(true)
          .open(&path)
          .map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "WriteString: cannot open {}: {}",
              path, e
            ))
          })?;
        for arg in &args[1..] {
          let text = match arg {
            Expr::String(s) => s.clone(),
            other => crate::syntax::expr_to_string(other),
          };
          file.write_all(text.as_bytes()).map_err(|e| {
            InterpreterError::EvaluationError(format!(
              "WriteString: write error: {}",
              e
            ))
          })?;
        }
        return Ok(Expr::Identifier("Null".to_string()));
      }

      return Ok(Expr::FunctionCall {
        name: "WriteString".to_string(),
        args: args.to_vec(),
      });
    }
    "AbsoluteTime" => {
      return crate::functions::datetime_ast::absolute_time_ast(args);
    }
    "DateList" => {
      return crate::functions::datetime_ast::date_list_ast(args);
    }
    "DatePlus" if args.len() == 2 => {
      return crate::functions::datetime_ast::date_plus_ast(args);
    }
    "DateDifference" if args.len() >= 2 => {
      return crate::functions::datetime_ast::date_difference_ast(args);
    }
    "DateString" => {
      return crate::functions::datetime_ast::date_string_ast(args);
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Run" if args.len() == 1 => {
      if let Expr::String(cmd) = &args[0] {
        use std::process::Command;
        let status = Command::new("sh").arg("-c").arg(cmd).status();
        return match status {
          Ok(s) => {
            // Wolfram's Run returns the raw wait status (exit_code * 256)
            let code = s.code().unwrap_or(-1) as i128;
            Ok(Expr::Integer(code * 256))
          }
          Err(e) => Err(InterpreterError::EvaluationError(format!(
            "Run: failed to execute command: {}",
            e
          ))),
        };
      } else {
        return Err(InterpreterError::EvaluationError(
          "Run expects a string argument".into(),
        ));
      }
    }
    "Plot" if args.len() >= 2 => {
      return crate::functions::plot::plot_ast(args);
    }
    "Plot3D" if args.len() >= 3 => {
      return crate::functions::plot3d::plot3d_ast(args);
    }
    "Graphics" if !args.is_empty() => {
      return crate::functions::graphics::graphics_ast(args);
    }
    "Graphics3D" if !args.is_empty() => {
      return crate::functions::plot3d::graphics3d_ast(args);
    }
    // List-based plots
    "ListPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_plot_ast(args);
    }
    "ListLinePlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_line_plot_ast(args);
    }
    "ListLogPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_log_plot_ast(args);
    }
    "ListLogLogPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_log_log_plot_ast(args);
    }
    "ListLogLinearPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_log_linear_plot_ast(args);
    }
    "ListPolarPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_polar_plot_ast(args);
    }
    "ListStepPlot" if !args.is_empty() => {
      return crate::functions::list_plot::list_step_plot_ast(args);
    }
    // Parametric plots
    "ParametricPlot" if args.len() >= 2 => {
      return crate::functions::parametric_plot::parametric_plot_ast(args);
    }
    "PolarPlot" if args.len() >= 2 => {
      return crate::functions::parametric_plot::polar_plot_ast(args);
    }
    // Field/density plots
    "DensityPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::density_plot_ast(args);
    }
    "ContourPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::contour_plot_ast(args);
    }
    "RegionPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::region_plot_ast(args);
    }
    "VectorPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::vector_plot_ast(args);
    }
    "StreamPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::stream_plot_ast(args);
    }
    "StreamDensityPlot" if args.len() >= 3 => {
      return crate::functions::field_plot::stream_density_plot_ast(args);
    }
    "ListDensityPlot" if !args.is_empty() => {
      return crate::functions::field_plot::list_density_plot_ast(args);
    }
    "ListContourPlot" if !args.is_empty() => {
      return crate::functions::field_plot::list_contour_plot_ast(args);
    }
    "ArrayPlot" if !args.is_empty() => {
      return crate::functions::field_plot::array_plot_ast(args);
    }
    "MatrixPlot" if !args.is_empty() => {
      return crate::functions::field_plot::matrix_plot_ast(args);
    }
    // Charts
    "BarChart" if !args.is_empty() => {
      return crate::functions::chart::bar_chart_ast(args);
    }
    "PieChart" if !args.is_empty() => {
      return crate::functions::chart::pie_chart_ast(args);
    }
    "Histogram" if !args.is_empty() => {
      return crate::functions::chart::histogram_ast(args);
    }
    "BoxWhiskerChart" if !args.is_empty() => {
      return crate::functions::chart::box_whisker_chart_ast(args);
    }
    "BubbleChart" if !args.is_empty() => {
      return crate::functions::chart::bubble_chart_ast(args);
    }
    "SectorChart" if !args.is_empty() => {
      return crate::functions::chart::sector_chart_ast(args);
    }
    "DateListPlot" if !args.is_empty() => {
      return crate::functions::chart::date_list_plot_ast(args);
    }
    "WordCloud" if !args.is_empty() => {
      return crate::functions::chart::word_cloud_ast(args);
    }
    "Print" => {
      // 0 args → just output a newline and return Null
      if args.is_empty() {
        println!();
        crate::capture_stdout("");
        return Ok(Expr::Identifier("Null".to_string()));
      }
      // Format and print all arguments concatenated (like Wolfram Print)
      let display_str: String = args
        .iter()
        .map(crate::syntax::expr_to_output)
        .collect::<Vec<_>>()
        .join("");
      println!("{}", display_str);
      crate::capture_stdout(&display_str);
      return Ok(Expr::Identifier("Null".to_string()));
    }

    // AST-native predicate functions
    "NumberQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::number_q_ast(args);
    }
    "RealValuedNumberQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::real_valued_number_q_ast(args);
    }
    "Element" if args.len() == 2 => {
      return element_ast(&args[0], &args[1]);
    }
    "IntegerQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::integer_q_ast(args);
    }
    "BooleanQ" if args.len() == 1 => {
      return Ok(match &args[0] {
        Expr::Identifier(name) if name == "True" || name == "False" => {
          Expr::Identifier("True".to_string())
        }
        _ => Expr::Identifier("False".to_string()),
      });
    }
    "SymbolQ" if args.len() == 1 => {
      return Ok(match &args[0] {
        Expr::Identifier(_) => Expr::Identifier("True".to_string()),
        _ => Expr::Identifier("False".to_string()),
      });
    }
    "Boole" if args.len() == 1 => {
      return Ok(match &args[0] {
        Expr::Identifier(name) if name == "True" => Expr::Integer(1),
        Expr::Identifier(name) if name == "False" => Expr::Integer(0),
        _ => Expr::FunctionCall {
          name: "Boole".to_string(),
          args: args.to_vec(),
        },
      });
    }
    "DigitQ" if args.len() == 1 => {
      return Ok(match &args[0] {
        Expr::String(s) => {
          if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) {
            Expr::Identifier("True".to_string())
          } else {
            Expr::Identifier("False".to_string())
          }
        }
        _ => Expr::Identifier("False".to_string()),
      });
    }
    "LetterQ" if args.len() == 1 => {
      return Ok(match &args[0] {
        Expr::String(s) => {
          if !s.is_empty() && s.chars().all(|c| c.is_alphabetic()) {
            Expr::Identifier("True".to_string())
          } else {
            Expr::Identifier("False".to_string())
          }
        }
        _ => Expr::Identifier("False".to_string()),
      });
    }
    "Precision" if args.len() == 1 => {
      return crate::functions::math_ast::precision_ast(args);
    }
    "Accuracy" if args.len() == 1 => {
      return crate::functions::math_ast::accuracy_ast(args);
    }
    "O" if args.len() == 1 || args.len() == 2 => {
      // O[x] → SeriesData[x, 0, {}, 1, 1, 1]
      // O[x, x0] → SeriesData[x, x0, {}, 1, 1, 1]
      let var = args[0].clone();
      let center = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Integer(0)
      };
      return Ok(Expr::FunctionCall {
        name: "SeriesData".to_string(),
        args: vec![
          var,
          center,
          Expr::List(vec![]),
          Expr::Integer(1),
          Expr::Integer(1),
          Expr::Integer(1),
        ],
      });
    }
    "EvenQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::even_q_ast(args);
    }
    "LeapYearQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::leap_year_q_ast(args);
    }
    "OddQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::odd_q_ast(args);
    }
    "PalindromeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::palindrome_q_ast(args);
    }
    "SquareFreeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::square_free_q_ast(args);
    }
    "ListQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::list_q_ast(args);
    }
    "StringQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::string_q_ast(args);
    }
    // Symbol["name"] - Convert string to symbol identifier
    "Symbol" if args.len() == 1 => {
      if let Expr::String(name) = &args[0] {
        return Ok(Expr::Identifier(name.clone()));
      }
      return Ok(Expr::FunctionCall {
        name: "Symbol".to_string(),
        args: args.to_vec(),
      });
    }
    // SymbolName[sym] - Get the name of a symbol as a string
    "SymbolName" if args.len() == 1 => {
      if let Expr::Identifier(name) = &args[0] {
        return Ok(Expr::String(name.clone()));
      }
      return Ok(Expr::FunctionCall {
        name: "SymbolName".to_string(),
        args: args.to_vec(),
      });
    }
    // Unique[] - generate a unique symbol $nnn
    // Unique[x] - generate a unique symbol x$nnn
    // Unique["xxx"] - generate a unique symbol xxxnnn
    // Unique[{x, y, ...}] - generate list of unique symbols
    "Unique" if args.is_empty() => {
      let sym_name = crate::functions::scoping::unique_symbol("");
      // For Unique[], format is $nnn (just $counter)
      return Ok(Expr::Identifier(sym_name));
    }
    "Unique" if args.len() == 1 => {
      match &args[0] {
        Expr::Identifier(name) => {
          let sym_name = crate::functions::scoping::unique_symbol(name);
          return Ok(Expr::Identifier(sym_name));
        }
        Expr::String(name) => {
          // For strings, use sequential numbering without $
          let sym_name =
            crate::functions::scoping::unique_symbol_from_string(name);
          return Ok(Expr::Identifier(sym_name));
        }
        Expr::List(items) => {
          let mut result = Vec::new();
          for item in items {
            match item {
              Expr::Identifier(name) => {
                let sym_name = crate::functions::scoping::unique_symbol(name);
                result.push(Expr::Identifier(sym_name));
              }
              Expr::String(name) => {
                let sym_name =
                  crate::functions::scoping::unique_symbol_from_string(name);
                result.push(Expr::Identifier(sym_name));
              }
              _ => {
                return Ok(Expr::FunctionCall {
                  name: "Unique".to_string(),
                  args: args.to_vec(),
                });
              }
            }
          }
          return Ok(Expr::List(result));
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Unique".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    "AtomQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::atom_q_ast(args);
    }
    "NumericQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::numeric_q_ast(args);
    }
    "ExactNumberQ" if args.len() == 1 => {
      let is_exact = match &args[0] {
        Expr::Integer(_) | Expr::BigInteger(_) => true,
        Expr::FunctionCall {
          name: fname,
          args: rargs,
        } if fname == "Rational"
          && rargs.len() == 2
          && matches!(rargs[0], Expr::Integer(_))
          && matches!(rargs[1], Expr::Integer(_)) =>
        {
          true
        }
        _ => false,
      };
      return Ok(Expr::Identifier(
        if is_exact { "True" } else { "False" }.to_string(),
      ));
    }
    "InexactNumberQ" if args.len() == 1 => {
      let is_inexact = matches!(&args[0], Expr::Real(_) | Expr::BigFloat(_, _));
      return Ok(Expr::Identifier(
        if is_inexact { "True" } else { "False" }.to_string(),
      ));
    }
    "LevelQ" if args.len() == 1 => {
      let is_valid = match &args[0] {
        Expr::Integer(_) => true,
        Expr::Identifier(name) if name == "Infinity" => true,
        Expr::List(items) => items.iter().all(|item| {
          matches!(item, Expr::Integer(_))
            || matches!(item, Expr::Identifier(n) if n == "Infinity")
        }),
        _ => false,
      };
      return Ok(Expr::Identifier(
        if is_valid { "True" } else { "False" }.to_string(),
      ));
    }
    "Positive" | "PositiveQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::positive_q_ast(args);
    }
    "Negative" | "NegativeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::negative_q_ast(args);
    }
    "NonPositive" | "NonPositiveQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::non_positive_q_ast(args);
    }
    "NonNegative" | "NonNegativeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::non_negative_q_ast(args);
    }
    "PrimeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::prime_q_ast(args);
    }
    "CompositeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::composite_q_ast(args);
    }
    "PrimePowerQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::prime_power_q_ast(args);
    }
    "AssociationQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::association_q_ast(args);
    }
    "Between" if args.len() == 2 => {
      use crate::functions::math_ast::try_eval_to_f64;
      // Normalize arg order: Between[x, range] or Between[range, x] (from operator form)
      let (x, range_expr) = if matches!(&args[0], Expr::List(_))
        && !matches!(&args[1], Expr::List(_))
      {
        (&args[1], &args[0])
      } else {
        (&args[0], &args[1])
      };
      // Between[x, {min, max}] or Between[x, {{min1, max1}, {min2, max2}, ...}]
      if let Expr::List(range) = range_expr {
        // Check if it's a list of ranges (all elements are lists)
        let is_list_of_ranges =
          !range.is_empty() && range.iter().all(|r| matches!(r, Expr::List(_)));
        if range.len() == 2 && !is_list_of_ranges {
          // Single range: Between[x, {min, max}]
          if let (Some(xv), Some(lo), Some(hi)) = (
            try_eval_to_f64(x),
            try_eval_to_f64(&range[0]),
            try_eval_to_f64(&range[1]),
          ) {
            return Ok(Expr::Identifier(
              if lo <= xv && xv <= hi {
                "True"
              } else {
                "False"
              }
              .to_string(),
            ));
          }
        } else if is_list_of_ranges {
          // Multiple ranges: Between[x, {{min1, max1}, ...}]
          if let Some(xv) = try_eval_to_f64(x) {
            for r in range {
              if let Expr::List(pair) = r
                && pair.len() == 2
                && let (Some(lo), Some(hi)) =
                  (try_eval_to_f64(&pair[0]), try_eval_to_f64(&pair[1]))
                && lo <= xv
                && xv <= hi
              {
                return Ok(Expr::Identifier("True".to_string()));
              }
            }
            return Ok(Expr::Identifier("False".to_string()));
          }
        }
      }
    }
    "Between" if args.len() == 1 => {
      // Operator form: Between[{min, max}] returns itself (handled by curried call)
      if let Expr::List(range) = &args[0]
        && range.len() == 2
      {
        return Ok(Expr::FunctionCall {
          name: "Between".to_string(),
          args: args.to_vec(),
        });
      }
    }

    // AST-native association functions
    "Keys" if args.len() == 1 => {
      return crate::functions::association_ast::keys_ast(args);
    }
    "Values" if args.len() == 1 => {
      return crate::functions::association_ast::values_ast(args);
    }
    "KeyDropFrom" if args.len() == 2 => {
      return crate::functions::association_ast::key_drop_from_ast(args);
    }
    "KeyExistsQ" if args.len() == 2 => {
      return crate::functions::association_ast::key_exists_q_ast(args);
    }
    "Lookup" if args.len() >= 2 => {
      return crate::functions::association_ast::lookup_ast(args);
    }
    "KeySort" if args.len() == 1 => {
      return crate::functions::association_ast::key_sort_ast(args);
    }
    "KeyValueMap" if args.len() == 2 => {
      return crate::functions::association_ast::key_value_map_ast(args);
    }
    "FilterRules" if args.len() == 2 => {
      return filter_rules_ast(&args[0], &args[1]);
    }

    // Dataset
    "Dataset" if !args.is_empty() => {
      return Ok(crate::functions::dataset_ast::dataset_ast(args));
    }

    "MemberQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::member_q_ast(args);
    }
    "FreeQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::free_q_ast(args);
    }
    "MatchQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::match_q_ast(args);
    }
    "Divisible" if args.len() == 2 => {
      return crate::functions::predicate_ast::divisible_ast(args);
    }
    "SubsetQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::subset_q_ast(args);
    }
    "OptionQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::option_q_ast(args);
    }
    "Head" if args.len() == 1 => {
      return crate::functions::predicate_ast::head_ast(args);
    }
    "Length" if args.len() == 1 => {
      return crate::functions::predicate_ast::length_ast(args);
    }
    "Depth" if args.len() == 1 => {
      return crate::functions::predicate_ast::depth_ast(args);
    }

    "LeafCount" if args.len() == 1 => {
      return crate::functions::predicate_ast::leaf_count_ast(args);
    }

    "ByteCount" if args.len() == 1 => {
      return crate::functions::predicate_ast::byte_count_ast(args);
    }

    // MaxMemoryUsed[] — peak memory usage of the process
    "MaxMemoryUsed" if args.is_empty() => {
      let peak_bytes = crate::functions::memory::max_memory_used();
      return Ok(Expr::Integer(peak_bytes));
    }

    // MemoryInUse[] — current memory usage of the process
    "MemoryInUse" if args.is_empty() => {
      let rss_bytes = crate::functions::memory::memory_in_use();
      return Ok(Expr::Integer(rss_bytes));
    }

    // Introspection functions - return {} for symbols without stored definitions
    "Messages" | "DownValues" | "OwnValues" | "SubValues" | "NValues"
    | "FormatValues" | "DefaultValues"
      if args.len() == 1 =>
    {
      return Ok(Expr::List(vec![]));
    }
    "UpValues" if args.len() == 1 => {
      if let Expr::Identifier(sym) = &args[0] {
        let up_defs = crate::UPVALUES
          .with(|m| m.borrow().get(sym).cloned().unwrap_or_default());
        if up_defs.is_empty() {
          return Ok(Expr::List(vec![]));
        }
        // Return a list of RuleDelayed expressions
        let rules: Vec<Expr> = up_defs
          .iter()
          .map(|(outer_func, _params, _conds, _defaults, _heads, body)| {
            Expr::RuleDelayed {
              pattern: Box::new(Expr::FunctionCall {
                name: "HoldPattern".to_string(),
                args: vec![Expr::FunctionCall {
                  name: outer_func.clone(),
                  args: vec![Expr::Identifier("__".to_string())],
                }],
              }),
              replacement: Box::new(body.clone()),
            }
          })
          .collect();
        return Ok(Expr::List(rules));
      }
      return Ok(Expr::List(vec![]));
    }

    // FullForm - returns full form representation (unevaluated)
    "FullForm" if args.len() == 1 => {
      return crate::functions::predicate_ast::full_form_ast(&args[0]);
    }
    "CForm" if args.len() == 1 => {
      return Ok(Expr::FunctionCall {
        name: "CForm".to_string(),
        args: args.to_vec(),
      });
    }

    // Attributes[symbol] - returns the attributes of a built-in symbol
    "Attributes" if args.len() == 1 => {
      let sym_name = match &args[0] {
        Expr::Identifier(name) => name.as_str(),
        Expr::Constant(name) => name.as_str(),
        Expr::String(name) => name.as_str(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Attributes".to_string(),
            args: args.to_vec(),
          });
        }
      };
      // Check user-defined attributes first, then combine with built-in
      let user_attrs =
        crate::FUNC_ATTRS.with(|m| m.borrow().get(sym_name).cloned());
      let builtin = get_builtin_attributes(sym_name);
      let mut all_attr_strs: Vec<String> =
        builtin.iter().map(|a| a.to_string()).collect();
      if let Some(user) = user_attrs {
        for a in user {
          if !all_attr_strs.contains(&a) {
            all_attr_strs.push(a);
          }
        }
        all_attr_strs.sort();
      }
      return Ok(Expr::List(
        all_attr_strs
          .iter()
          .map(|a| Expr::Identifier(a.to_string()))
          .collect(),
      ));
    }

    // Context[] — return current context
    // Context[symbol] — return context of a symbol
    "Context" if args.is_empty() => {
      return Ok(Expr::String("Global`".to_string()));
    }
    "Context" if args.len() == 1 => {
      let sym_name = match &args[0] {
        Expr::Identifier(name) => name.clone(),
        Expr::String(name) => name.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Context".to_string(),
            args: args.to_vec(),
          });
        }
      };
      // Built-in symbols are in System` context
      let builtin = get_builtin_attributes(&sym_name);
      if !builtin.is_empty() {
        return Ok(Expr::String("System`".to_string()));
      }
      // User-defined symbols are in Global` context
      return Ok(Expr::String("Global`".to_string()));
    }

    // Options[f] — return stored options for function f
    // Options[f, opt] — return specific option for function f
    "Options" if args.len() == 1 || args.len() == 2 => {
      let func_arg = evaluate_expr_to_expr(&args[0])?;
      let func_name = match &func_arg {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Options".to_string(),
            args: vec![func_arg],
          });
        }
      };
      let stored =
        crate::FUNC_OPTIONS.with(|m| m.borrow().get(&func_name).cloned());
      let opts = stored.unwrap_or_default();
      if args.len() == 1 {
        return Ok(Expr::List(opts));
      } else {
        // Options[f, opt] — find the matching option
        let opt_arg = evaluate_expr_to_expr(&args[1])?;
        let opt_name = match &opt_arg {
          Expr::Identifier(name) => name.clone(),
          _ => {
            return Ok(Expr::List(vec![]));
          }
        };
        let matching: Vec<Expr> = opts
          .into_iter()
          .filter(|rule| match rule {
            Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
              matches!(pattern.as_ref(), Expr::Identifier(n) if *n == opt_name)
            }
            _ => false,
          })
          .collect();
        return Ok(Expr::List(matching));
      }
    }

    // Construct - creates function call f[a][b] etc.
    "Construct" if !args.is_empty() => {
      return crate::functions::predicate_ast::construct_ast(args);
    }

    // Quantity functions
    "Quantity" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::quantity_ast::quantity_ast(args);
    }
    "QuantityMagnitude" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::quantity_ast::quantity_magnitude_ast(args);
    }
    "QuantityUnit" if args.len() == 1 => {
      return crate::functions::quantity_ast::quantity_unit_ast(args);
    }
    "QuantityQ" if args.len() == 1 => {
      return crate::functions::quantity_ast::quantity_q_ast(args);
    }
    "CompatibleUnitQ" => {
      return crate::functions::quantity_ast::compatible_unit_q_ast(args);
    }
    "UnitConvert" => {
      return crate::functions::quantity_ast::unit_convert_ast(args);
    }

    // AST-native math functions
    "Plus" => {
      return crate::functions::math_ast::plus_ast(args);
    }
    "Times" => {
      return crate::functions::math_ast::times_ast(args);
    }
    "Minus" => {
      return crate::functions::math_ast::minus_ast(args);
    }
    "Subtract" if args.len() == 2 => {
      return crate::functions::math_ast::subtract_ast(args);
    }
    "Divide" => {
      if args.len() == 2 {
        return crate::functions::math_ast::divide_ast(args);
      } else {
        println!(
          "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
        return Ok(Expr::FunctionCall {
          name: "Divide".to_string(),
          args: args.to_vec(),
        });
      }
    }
    "Power" if args.len() == 2 => {
      return crate::functions::math_ast::power_ast(args);
    }
    "Max" => {
      return crate::functions::math_ast::max_ast(args);
    }
    "Min" => {
      return crate::functions::math_ast::min_ast(args);
    }
    "RankedMax" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let idx = (k - 1) as usize;
          if idx < sorted.len() {
            return Ok(sorted[idx].clone());
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "RankedMax".to_string(),
        args: args.to_vec(),
      });
    }
    "RankedMin" if args.len() == 2 => {
      if let Expr::List(items) = &args[0] {
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(k) = expr_to_i128(&args[1]) {
          let idx = (k - 1) as usize;
          if idx < sorted.len() {
            return Ok(sorted[idx].clone());
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "RankedMin".to_string(),
        args: args.to_vec(),
      });
    }
    "Quantile" if args.len() == 2 => {
      return crate::functions::math_ast::quantile_ast(args);
    }
    "Quartiles" if args.len() == 1 => {
      // Quartiles uses Quantile with parameters {{1/2, 0}, {0, 1}}
      // Formula: pos = 1/2 + n*q, then linear interpolation
      if let Expr::List(items) = &args[0] {
        if items.is_empty() {
          return Ok(Expr::FunctionCall {
            name: "Quartiles".to_string(),
            args: args.to_vec(),
          });
        }
        // Sort numerically
        let mut sorted = items.clone();
        sorted.sort_by(|a, b| {
          let fa = crate::functions::math_ast::try_eval_to_f64(a);
          let fb = crate::functions::math_ast::try_eval_to_f64(b);
          fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = sorted.len() as i128;
        let mut results = Vec::new();
        for (qn, qd) in [(1i128, 4i128), (1, 2), (3, 4)] {
          // pos = 1/2 + n * q = 1/2 + n*qn/qd
          // pos = (qd + 2*n*qn) / (2*qd)
          let pos_num = qd + 2 * n * qn;
          let pos_den = 2 * qd;
          let j = pos_num / pos_den; // Floor
          let frac_num = pos_num - j * pos_den; // remainder
          // frac = frac_num / pos_den
          if frac_num == 0 {
            // Exact position
            results.push(sorted[(j - 1) as usize].clone());
          } else {
            // Interpolate: (1 - frac)*sorted[j-1] + frac*sorted[j]
            // = ((pos_den - frac_num)*sorted[j-1] + frac_num*sorted[j]) / pos_den
            let lo = &sorted[(j - 1) as usize];
            let hi = &sorted[j as usize];
            let w_lo = pos_den - frac_num;
            let w_hi = frac_num;
            // Try integer arithmetic
            if let (Some(lo_v), Some(hi_v)) = (
              crate::functions::math_ast::try_eval_to_f64(lo),
              crate::functions::math_ast::try_eval_to_f64(hi),
            ) {
              let lo_i = lo_v as i128;
              let hi_i = hi_v as i128;
              if lo_i as f64 == lo_v && hi_i as f64 == hi_v {
                // Exact rational: (w_lo*lo_i + w_hi*hi_i) / pos_den
                let num = w_lo * lo_i + w_hi * hi_i;
                results.push(crate::functions::math_ast::make_rational(
                  num, pos_den,
                ));
              } else {
                results.push(crate::functions::math_ast::num_to_expr(
                  (w_lo as f64 * lo_v + w_hi as f64 * hi_v) / pos_den as f64,
                ));
              }
            } else {
              // Symbolic fallback
              results.push(Expr::FunctionCall {
                name: "Quartiles".to_string(),
                args: args.to_vec(),
              });
              break;
            }
          }
        }
        if results.len() == 3 {
          return Ok(Expr::List(results));
        }
      }
    }
    "Abs" if args.len() == 1 => {
      return crate::functions::math_ast::abs_ast(args);
    }
    "RealAbs" if args.len() == 1 => {
      // RealAbs is same as Abs for real-valued arguments
      match &args[0] {
        Expr::Real(f) => return Ok(Expr::Real(f.abs())),
        Expr::Integer(n) => return Ok(Expr::Integer(n.abs())),
        _ => {
          if let Some(n) = crate::functions::math_ast::try_eval_to_f64(&args[0])
          {
            return Ok(crate::functions::math_ast::num_to_expr(n.abs()));
          }
        }
      }
      return Ok(Expr::FunctionCall {
        name: "RealAbs".to_string(),
        args: args.to_vec(),
      });
    }
    "RealSign" if args.len() == 1 => {
      match &args[0] {
        Expr::Real(f) => {
          return Ok(Expr::Integer(if *f > 0.0 {
            1
          } else if *f < 0.0 {
            -1
          } else {
            0
          }));
        }
        Expr::Integer(n) => {
          return Ok(Expr::Integer(if *n > 0 {
            1
          } else if *n < 0 {
            -1
          } else {
            0
          }));
        }
        Expr::FunctionCall {
          name: rname,
          args: rargs,
        } if rname == "Rational" && rargs.len() == 2 => {
          if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
            let sign = if (*n > 0 && *d > 0) || (*n < 0 && *d < 0) {
              1
            } else if *n == 0 {
              0
            } else {
              -1
            };
            return Ok(Expr::Integer(sign));
          }
        }
        _ => {
          // Stay symbolic for complex or symbolic args
          return Ok(Expr::FunctionCall {
            name: "RealSign".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    "Sign" if args.len() == 1 => {
      return crate::functions::math_ast::sign_ast(args);
    }
    "Sqrt" if args.len() == 1 => {
      return crate::functions::math_ast::sqrt_ast(args);
    }
    "Surd" if args.len() == 2 => {
      return crate::functions::math_ast::surd_ast(args);
    }
    "Floor" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::floor_ast(args);
    }
    "Ceiling" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::ceiling_ast(args);
    }
    "Round" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::round_ast(args);
    }
    "Mod" if args.len() == 2 || args.len() == 3 => {
      return crate::functions::math_ast::mod_ast(args);
    }
    "Quotient" if args.len() == 2 => {
      return crate::functions::math_ast::quotient_ast(args);
    }
    "QuotientRemainder" if args.len() == 2 => {
      let q = crate::functions::math_ast::quotient_ast(args)?;
      let r = crate::functions::math_ast::mod_ast(args)?;
      return Ok(Expr::List(vec![q, r]));
    }
    "GCD" => {
      return crate::functions::math_ast::gcd_ast(args);
    }
    "ExtendedGCD" if args.len() >= 2 => {
      return crate::functions::math_ast::extended_gcd_ast(args);
    }
    "LCM" => {
      return crate::functions::math_ast::lcm_ast(args);
    }
    "Total" => {
      return crate::functions::math_ast::total_ast(args);
    }
    "Fourier" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::fourier_ast(args);
    }
    "InverseFourier" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::inverse_fourier_ast(args);
    }
    "Mean" if args.len() == 1 => {
      return crate::functions::math_ast::mean_ast(args);
    }
    "Variance" if args.len() == 1 => {
      return crate::functions::math_ast::variance_ast(args);
    }
    "StandardDeviation" if args.len() == 1 => {
      return crate::functions::math_ast::standard_deviation_ast(args);
    }
    "GeometricMean" if args.len() == 1 => {
      return crate::functions::math_ast::geometric_mean_ast(args);
    }
    "HarmonicMean" if args.len() == 1 => {
      return crate::functions::math_ast::harmonic_mean_ast(args);
    }
    "RootMeanSquare" if args.len() == 1 => {
      return crate::functions::math_ast::root_mean_square_ast(args);
    }
    "Covariance" if args.len() == 2 => {
      return crate::functions::math_ast::covariance_ast(args);
    }
    "Correlation" if args.len() == 2 => {
      return crate::functions::math_ast::correlation_ast(args);
    }
    "CentralMoment" if args.len() == 2 => {
      return crate::functions::math_ast::central_moment_ast(args);
    }
    "Kurtosis" if args.len() == 1 => {
      return crate::functions::math_ast::kurtosis_ast(args);
    }
    "Skewness" if args.len() == 1 => {
      return crate::functions::math_ast::skewness_ast(args);
    }
    "IntegerLength" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::integer_length_ast(args);
    }
    "IntegerReverse" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::integer_reverse_ast(args);
    }
    "Rescale" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::rescale_ast(args);
    }
    "Normalize" if args.len() == 1 => {
      return crate::functions::math_ast::normalize_ast(args);
    }
    "Norm" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::norm_ast(args);
    }
    "EuclideanDistance" if args.len() == 2 => {
      return crate::functions::math_ast::euclidean_distance_ast(args);
    }
    "ManhattanDistance" if args.len() == 2 => {
      return crate::functions::math_ast::manhattan_distance_ast(args);
    }
    "SquaredEuclideanDistance" if args.len() == 2 => {
      return crate::functions::math_ast::squared_euclidean_distance_ast(args);
    }
    "Factorial" if args.len() == 1 => {
      return crate::functions::math_ast::factorial_ast(args);
    }
    "Factorial2" if args.len() == 1 => {
      return crate::functions::math_ast::factorial2_ast(args);
    }
    "Subfactorial" if args.len() == 1 => {
      return crate::functions::math_ast::subfactorial_ast(args);
    }
    "Pochhammer" if args.len() == 2 => {
      return crate::functions::math_ast::pochhammer_ast(args);
    }
    "Gamma" if args.len() == 1 => {
      return crate::functions::math_ast::gamma_ast(args);
    }
    "BesselJ" if args.len() == 2 => {
      return crate::functions::math_ast::bessel_j_ast(args);
    }
    "BesselY" if args.len() == 2 => {
      return crate::functions::math_ast::bessel_y_ast(args);
    }
    "AiryAi" if args.len() == 1 => {
      return crate::functions::math_ast::airy_ai_ast(args);
    }
    "Hypergeometric0F1" if args.len() == 2 => {
      return crate::functions::math_ast::hypergeometric_0f1_ast(args);
    }
    "Hypergeometric1F1" if args.len() == 3 => {
      return crate::functions::math_ast::hypergeometric1f1_ast(args);
    }
    "Hypergeometric2F1" if args.len() == 4 => {
      return crate::functions::math_ast::hypergeometric2f1_ast(args);
    }
    "HypergeometricU" if args.len() == 3 => {
      return crate::functions::math_ast::hypergeometric_u_ast(args);
    }
    "EllipticK" if args.len() == 1 => {
      return crate::functions::math_ast::elliptic_k_ast(args);
    }
    "EllipticE" if args.len() == 1 => {
      return crate::functions::math_ast::elliptic_e_ast(args);
    }
    "EllipticF" if args.len() == 2 => {
      return crate::functions::math_ast::elliptic_f_ast(args);
    }
    "EllipticPi" if args.len() == 2 || args.len() == 3 => {
      return crate::functions::math_ast::elliptic_pi_ast(args);
    }
    "Zeta" if args.len() == 1 => {
      return crate::functions::math_ast::zeta_ast(args);
    }
    "PolyGamma" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::polygamma_ast(args);
    }
    "LegendreP" if args.len() == 2 => {
      return crate::functions::math_ast::legendre_p_ast(args);
    }
    "JacobiP" if args.len() == 4 => {
      return crate::functions::math_ast::jacobi_p_ast(args);
    }
    "SphericalHarmonicY" if args.len() == 4 => {
      return crate::functions::math_ast::spherical_harmonic_y_ast(args);
    }
    "LegendreQ" if args.len() == 2 => {
      return crate::functions::math_ast::legendre_q_ast(args);
    }
    "PolyLog" if args.len() == 2 => {
      return crate::functions::math_ast::polylog_ast(args);
    }
    "LerchPhi" if args.len() == 3 => {
      return crate::functions::math_ast::lerch_phi_ast(args);
    }
    "ExpIntegralEi" if args.len() == 1 => {
      return crate::functions::math_ast::exp_integral_ei_ast(args);
    }
    "ExpIntegralE" if args.len() == 2 => {
      return crate::functions::math_ast::exp_integral_e_ast(args);
    }
    "BesselI" if args.len() == 2 => {
      return crate::functions::math_ast::bessel_i_ast(args);
    }
    "BesselK" if args.len() == 2 => {
      return crate::functions::math_ast::bessel_k_ast(args);
    }
    "EllipticTheta" if args.len() == 3 => {
      return crate::functions::math_ast::elliptic_theta_ast(args);
    }
    "WeierstrassP" if args.len() == 2 => {
      return crate::functions::math_ast::weierstrass_p_ast(args);
    }
    "JacobiAmplitude" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_amplitude_ast(args);
    }
    "JacobiDN" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_dn_ast(args);
    }
    "JacobiSN" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_sn_ast(args);
    }
    "JacobiCN" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_cn_ast(args);
    }
    "JacobiSC" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_sc_ast(args);
    }
    "JacobiDC" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_dc_ast(args);
    }
    "JacobiCD" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_cd_ast(args);
    }
    "JacobiSD" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_sd_ast(args);
    }
    "JacobiCS" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_cs_ast(args);
    }
    "JacobiDS" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_ds_ast(args);
    }
    "JacobiNS" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_ns_ast(args);
    }
    "JacobiND" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_nd_ast(args);
    }
    "JacobiNC" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_nc_ast(args);
    }
    "ChebyshevT" if args.len() == 2 => {
      return crate::functions::math_ast::chebyshev_t_ast(args);
    }
    "ChebyshevU" if args.len() == 2 => {
      return crate::functions::math_ast::chebyshev_u_ast(args);
    }
    "GegenbauerC" if args.len() == 3 => {
      return crate::functions::math_ast::gegenbauer_c_ast(args);
    }
    "LaguerreL" if args.len() == 2 => {
      return crate::functions::math_ast::laguerre_l_ast(args);
    }
    "Beta" if args.len() == 2 => {
      return crate::functions::math_ast::beta_ast(args);
    }
    "LogIntegral" if args.len() == 1 => {
      return crate::functions::math_ast::log_integral_ast(args);
    }
    "HermiteH" if args.len() == 2 => {
      return crate::functions::math_ast::hermite_h_ast(args);
    }
    "N" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::n_ast(args);
    }
    "RandomInteger" => {
      return crate::functions::math_ast::random_integer_ast(args);
    }
    "RandomReal" => {
      return crate::functions::math_ast::random_real_ast(args);
    }
    "RandomChoice" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::random_choice_ast(args);
    }
    "RandomSample" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::random_sample_ast(args);
    }
    "RandomVariate" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::random_variate_ast(args);
    }
    "SeedRandom" if args.len() <= 1 => {
      return crate::functions::math_ast::seed_random_ast(args);
    }
    "Clip" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::clip_ast(args);
    }
    "Sin" if args.len() == 1 => {
      return crate::functions::math_ast::sin_ast(args);
    }
    "Cos" if args.len() == 1 => {
      return crate::functions::math_ast::cos_ast(args);
    }
    "Tan" if args.len() == 1 => {
      return crate::functions::math_ast::tan_ast(args);
    }
    "Sec" if args.len() == 1 => {
      return crate::functions::math_ast::sec_ast(args);
    }
    "Csc" if args.len() == 1 => {
      return crate::functions::math_ast::csc_ast(args);
    }
    "Cot" if args.len() == 1 => {
      return crate::functions::math_ast::cot_ast(args);
    }
    "Exp" if args.len() == 1 => {
      return crate::functions::math_ast::exp_ast(args);
    }
    "Erf" if args.len() == 1 => {
      return crate::functions::math_ast::erf_ast(args);
    }
    "Erfc" if args.len() == 1 => {
      return crate::functions::math_ast::erfc_ast(args);
    }
    "Log" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::log_ast(args);
    }
    "Log10" if args.len() == 1 => {
      return crate::functions::math_ast::log10_ast(args);
    }
    "Log2" if args.len() == 1 => {
      return crate::functions::math_ast::log2_ast(args);
    }
    "ArcSin" if args.len() == 1 => {
      return crate::functions::math_ast::arcsin_ast(args);
    }
    "ArcCos" if args.len() == 1 => {
      return crate::functions::math_ast::arccos_ast(args);
    }
    "ArcTan" if args.len() == 1 => {
      return crate::functions::math_ast::arctan_ast(args);
    }
    "Sinh" if args.len() == 1 => {
      return crate::functions::math_ast::sinh_ast(args);
    }
    "Cosh" if args.len() == 1 => {
      return crate::functions::math_ast::cosh_ast(args);
    }
    "Tanh" if args.len() == 1 => {
      return crate::functions::math_ast::tanh_ast(args);
    }
    "Coth" if args.len() == 1 => {
      return crate::functions::math_ast::coth_ast(args);
    }
    "Sech" if args.len() == 1 => {
      return crate::functions::math_ast::sech_ast(args);
    }
    "Csch" if args.len() == 1 => {
      return crate::functions::math_ast::csch_ast(args);
    }
    "ArcSinh" if args.len() == 1 => {
      return crate::functions::math_ast::arcsinh_ast(args);
    }
    "ArcCosh" if args.len() == 1 => {
      return crate::functions::math_ast::arccosh_ast(args);
    }
    "ArcTanh" if args.len() == 1 => {
      return crate::functions::math_ast::arctanh_ast(args);
    }
    "ArcCoth" if args.len() == 1 => {
      return crate::functions::math_ast::arccoth_ast(args);
    }
    "ArcSech" if args.len() == 1 => {
      return crate::functions::math_ast::arcsech_ast(args);
    }
    "ArcCot" if args.len() == 1 => {
      return crate::functions::math_ast::arccot_ast(args);
    }
    "ArcCsc" if args.len() == 1 => {
      return crate::functions::math_ast::arccsc_ast(args);
    }
    "ArcSec" if args.len() == 1 => {
      return crate::functions::math_ast::arcsec_ast(args);
    }
    "ArcCsch" if args.len() == 1 => {
      return crate::functions::math_ast::arccsch_ast(args);
    }
    "Gudermannian" if args.len() == 1 => {
      return crate::functions::math_ast::gudermannian_ast(args);
    }
    "InverseGudermannian" if args.len() == 1 => {
      return crate::functions::math_ast::inverse_gudermannian_ast(args);
    }
    "LogisticSigmoid" if args.len() == 1 => {
      return crate::functions::math_ast::logistic_sigmoid_ast(args);
    }
    "ProductLog" if args.len() == 1 => {
      return crate::functions::math_ast::product_log_ast(args);
    }
    "Prime" if args.len() == 1 => {
      return crate::functions::math_ast::prime_ast(args);
    }
    "Fibonacci" if args.len() == 1 => {
      return crate::functions::math_ast::fibonacci_ast(args);
    }
    "LinearRecurrence" if args.len() == 3 => {
      return crate::functions::math_ast::linear_recurrence_ast(args);
    }
    "IntegerDigits" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::integer_digits_ast(args);
    }
    "RealDigits" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::real_digits_ast(args);
    }
    "FromDigits" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::from_digits_ast(args);
    }
    "IntegerName" if args.len() == 1 => {
      return crate::functions::math_ast::integer_name_ast(args);
    }
    "RomanNumeral" if args.len() == 1 => {
      return crate::functions::math_ast::roman_numeral_ast(args);
    }
    "FactorInteger" if args.len() == 1 => {
      return crate::functions::math_ast::factor_integer_ast(args);
    }
    "IntegerPartitions" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::integer_partitions_ast(args);
    }
    "Divisors" if args.len() == 1 => {
      return crate::functions::math_ast::divisors_ast(args);
    }
    "DivisorSigma" if args.len() == 2 => {
      return crate::functions::math_ast::divisor_sigma_ast(args);
    }
    "MoebiusMu" if args.len() == 1 => {
      return crate::functions::math_ast::moebius_mu_ast(args);
    }
    "EulerPhi" if args.len() == 1 => {
      return crate::functions::math_ast::euler_phi_ast(args);
    }
    "JacobiSymbol" if args.len() == 2 => {
      return crate::functions::math_ast::jacobi_symbol_ast(args);
    }
    "CoprimeQ" if args.len() >= 2 => {
      return crate::functions::math_ast::coprime_q_ast(args);
    }
    "Re" if args.len() == 1 => {
      return crate::functions::math_ast::re_ast(args);
    }
    "Im" if args.len() == 1 => {
      return crate::functions::math_ast::im_ast(args);
    }
    "Conjugate" if args.len() == 1 => {
      return crate::functions::math_ast::conjugate_ast(args);
    }
    "Arg" if args.len() == 1 => {
      return crate::functions::math_ast::arg_ast(args);
    }
    "Rationalize" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::rationalize_ast(args);
    }
    "Numerator" if args.len() == 1 => {
      return crate::functions::math_ast::numerator_ast(args);
    }
    "Denominator" if args.len() == 1 => {
      return crate::functions::math_ast::denominator_ast(args);
    }
    "Binomial" if args.len() == 2 => {
      return crate::functions::math_ast::binomial_ast(args);
    }
    "Multinomial" => {
      return crate::functions::math_ast::multinomial_ast(args);
    }
    "PowerMod" if args.len() == 3 => {
      return crate::functions::math_ast::power_mod_ast(args);
    }
    "PrimePi" if args.len() == 1 => {
      return crate::functions::math_ast::prime_pi_ast(args);
    }
    "PartitionsP" if args.len() == 1 => {
      return crate::functions::math_ast::partitions_p_ast(args);
    }
    "NextPrime" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::next_prime_ast(args);
    }
    "ModularInverse" if args.len() == 2 => {
      return crate::functions::math_ast::modular_inverse_ast(args);
    }
    "BitLength" if args.len() == 1 => {
      return crate::functions::math_ast::bit_length_ast(args);
    }
    "BitAnd" if !args.is_empty() => {
      return crate::functions::math_ast::bit_and_ast(args);
    }
    "BitOr" if !args.is_empty() => {
      return crate::functions::math_ast::bit_or_ast(args);
    }
    "BitXor" if !args.is_empty() => {
      return crate::functions::math_ast::bit_xor_ast(args);
    }
    "BitNot" if args.len() == 1 => {
      return crate::functions::math_ast::bit_not_ast(args);
    }
    "IntegerExponent" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::integer_exponent_ast(args);
    }
    "IntegerPart" if args.len() == 1 => {
      return crate::functions::math_ast::integer_part_ast(args);
    }
    "FractionalPart" if args.len() == 1 => {
      return crate::functions::math_ast::fractional_part_ast(args);
    }
    "MixedFractionParts" if args.len() == 1 => {
      return crate::functions::math_ast::mixed_fraction_parts_ast(args);
    }
    "Chop" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::chop_ast(args);
    }
    "PowerExpand" if args.len() == 1 => {
      return crate::functions::math_ast::power_expand_ast(args);
    }
    "Variables" if args.len() == 1 => {
      return crate::functions::math_ast::variables_ast(args);
    }
    "CubeRoot" if args.len() == 1 => {
      return crate::functions::math_ast::cube_root_ast(args);
    }
    "Subdivide" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::subdivide_ast(args);
    }
    "DigitCount" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::digit_count_ast(args);
    }
    "DigitSum" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::digit_sum_ast(args);
    }
    "ContinuedFraction" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::continued_fraction_ast(args);
    }
    "FromContinuedFraction" if args.len() == 1 => {
      return crate::functions::math_ast::from_continued_fraction_ast(args);
    }
    "LucasL" if args.len() == 1 => {
      return crate::functions::math_ast::lucas_l_ast(args);
    }
    "ChineseRemainder" if args.len() == 2 => {
      return crate::functions::math_ast::chinese_remainder_ast(args);
    }
    "DivisorSum" if args.len() == 2 => {
      return crate::functions::math_ast::divisor_sum_ast(args);
    }
    "BernoulliB" if args.len() == 1 => {
      return crate::functions::math_ast::bernoulli_b_ast(args);
    }
    "BellB" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::math_ast::bell_b_ast(args);
    }
    "PauliMatrix" if args.len() == 1 => {
      return crate::functions::math_ast::pauli_matrix_ast(args);
    }
    "CatalanNumber" if args.len() == 1 => {
      return crate::functions::math_ast::catalan_number_ast(args);
    }
    "StirlingS1" if args.len() == 2 => {
      return crate::functions::math_ast::stirling_s1_ast(args);
    }
    "StirlingS2" if args.len() == 2 => {
      return crate::functions::math_ast::stirling_s2_ast(args);
    }
    "FrobeniusNumber" if args.len() == 1 => {
      return crate::functions::math_ast::frobenius_number_ast(args);
    }
    "HarmonicNumber" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::harmonic_number_ast(args);
    }
    "CoefficientList" if args.len() == 2 => {
      return crate::functions::polynomial_ast::coefficient_list_ast(args);
    }

    // AST-native boolean functions
    "And" if args.len() >= 2 => {
      return crate::functions::boolean_ast::and_ast(args);
    }
    "Or" if args.len() >= 2 => {
      return crate::functions::boolean_ast::or_ast(args);
    }
    "Not" => {
      if args.len() == 1 {
        return crate::functions::boolean_ast::not_ast(args);
      } else {
        println!(
          "\nNot::argx: Not called with {} arguments; 1 argument is expected.",
          args.len()
        );
        use std::io::{self, Write};
        io::stdout().flush().ok();
      }
    }
    "Xor" if !args.is_empty() => {
      return crate::functions::boolean_ast::xor_ast(args);
    }
    "Equivalent" if args.len() >= 2 => {
      return crate::functions::boolean_ast::equivalent_ast(args);
    }
    "Return" => {
      let val = if args.is_empty() {
        Expr::Identifier("Null".to_string())
      } else {
        args[0].clone()
      };
      return Err(InterpreterError::ReturnValue(Box::new(val)));
    }
    "SameQ" => {
      return crate::functions::boolean_ast::same_q_ast(args);
    }
    "UnsameQ" => {
      return crate::functions::boolean_ast::unsame_q_ast(args);
    }
    "Which" if args.len() >= 2 && args.len().is_multiple_of(2) => {
      return crate::functions::boolean_ast::which_ast(args);
    }
    "While" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::boolean_ast::while_ast(args);
    }
    "Equal" => {
      return crate::functions::boolean_ast::equal_ast(args);
    }
    "Unequal" if args.len() >= 2 => {
      return crate::functions::boolean_ast::unequal_ast(args);
    }
    "Less" if args.len() >= 2 => {
      return crate::functions::boolean_ast::less_ast(args);
    }
    "Greater" if args.len() >= 2 => {
      return crate::functions::boolean_ast::greater_ast(args);
    }
    "LessEqual" if args.len() >= 2 => {
      return crate::functions::boolean_ast::less_equal_ast(args);
    }
    "GreaterEqual" if args.len() >= 2 => {
      return crate::functions::boolean_ast::greater_equal_ast(args);
    }
    "Boole" if args.len() == 1 => {
      return crate::functions::boolean_ast::boole_ast(args);
    }
    "TrueQ" if args.len() == 1 => {
      return crate::functions::boolean_ast::true_q_ast(args);
    }
    "Implies" if args.len() == 2 => {
      return crate::functions::boolean_ast::implies_ast(args);
    }
    "Nand" if args.len() >= 2 => {
      return crate::functions::boolean_ast::nand_ast(args);
    }
    "Nor" if args.len() >= 2 => {
      return crate::functions::boolean_ast::nor_ast(args);
    }
    "LogicalExpand" if args.len() == 1 => {
      return crate::functions::boolean_ast::logical_expand_ast(args);
    }

    // AST-native polynomial functions
    // Distribute[f[x1, x2, ...]] - distribute f over Plus
    // Distribute[expr, g] - distribute over g instead of Plus
    // Distribute[expr, g, f] - only distribute if outer head is f
    "Distribute" if !args.is_empty() && args.len() <= 3 => {
      return distribute_ast(args);
    }
    "PolynomialRemainder" if args.len() == 3 => {
      return crate::functions::polynomial_ast::polynomial_remainder_ast(args);
    }
    "PolynomialQuotient" if args.len() == 3 => {
      return crate::functions::polynomial_ast::polynomial_quotient_ast(args);
    }
    "Expand" if args.len() == 1 => {
      return crate::functions::polynomial_ast::expand_ast(args);
    }
    "Factor" if args.len() == 1 => {
      return crate::functions::polynomial_ast::factor_ast(args);
    }
    "FactorList" if args.len() == 1 => {
      return crate::functions::polynomial_ast::factor_list_ast(args);
    }
    "Simplify" if args.len() == 1 => {
      return crate::functions::polynomial_ast::simplify_ast(args);
    }
    "Coefficient" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::polynomial_ast::coefficient_ast(args);
    }
    "Exponent" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::polynomial_ast::exponent_ast(args);
    }
    "PolynomialQ" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::polynomial_ast::polynomial_q_ast(args);
    }
    "Solve" if args.len() == 2 => {
      return crate::functions::polynomial_ast::solve_ast(args);
    }
    "SolveAlways" if args.len() == 2 => {
      return crate::functions::polynomial_ast::solve_always_ast(args);
    }
    "Roots" if args.len() == 2 => {
      return crate::functions::polynomial_ast::roots_ast(args);
    }
    "ToRules" if args.len() == 1 => {
      return crate::functions::polynomial_ast::to_rules_ast(args);
    }
    "Eliminate" if args.len() == 2 => {
      return crate::functions::polynomial_ast::eliminate_ast(args);
    }
    "Reduce" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::polynomial_ast::reduce_ast(args);
    }
    "FindRoot" if args.len() == 2 => {
      return crate::functions::polynomial_ast::find_root_ast(args);
    }
    "FindMinimum" if args.len() == 2 => {
      return crate::functions::polynomial_ast::find_minimum_ast(args, false);
    }
    "FindMaximum" if args.len() == 2 => {
      return crate::functions::polynomial_ast::find_minimum_ast(args, true);
    }

    // AST-native list generation
    "Tuples" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::list_helpers_ast::tuples_ast(args);
    }

    // Use AST-native calculus functions
    // Derivative[n, f, x] - nth derivative of user-defined function f evaluated at x
    // Comes from f'[x], f''[x], etc. syntax (parsed as curried Derivative[n][f][x],
    // then flattened by apply_curried_call)
    "Derivative" if args.len() == 3 => {
      if let (Expr::Integer(n), Expr::Identifier(func_name)) =
        (&args[0], &args[1])
      {
        let n = *n as usize;
        // Look up the user-defined function
        let overloads = crate::FUNC_DEFS.with(|m| {
          let defs = m.borrow();
          defs.get(func_name).cloned()
        });
        if let Some(overloads) = overloads {
          // Find the first single-parameter overload
          for (params, _conditions, _defaults, _heads, body_expr) in &overloads
          {
            if params.len() == 1 {
              let param = &params[0];
              // Differentiate body n times with respect to param
              let mut deriv = body_expr.clone();
              for _ in 0..n {
                deriv = crate::functions::calculus_ast::differentiate_expr(
                  &deriv, param,
                )?;
              }
              // Substitute the actual argument for the param
              let substituted =
                crate::syntax::substitute_variable(&deriv, param, &args[2]);
              return evaluate_expr_to_expr(&substituted);
            }
          }
        }
      }
      // Return symbolic as flat FunctionCall (formatted as curried in expr_to_string)
      return Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      });
    }
    // Derivative[n, f] or Derivative[n] - return symbolic
    "Derivative" if args.len() <= 2 => {
      return Ok(Expr::FunctionCall {
        name: "Derivative".to_string(),
        args: args.to_vec(),
      });
    }
    "D" if args.len() == 2 => {
      return crate::functions::calculus_ast::d_ast(args);
    }
    "Dt" if args.len() == 2 => {
      return crate::functions::calculus_ast::dt_ast(args);
    }
    "Curl" if args.len() == 2 => {
      return crate::functions::calculus_ast::curl_ast(args);
    }
    "Integrate" if args.len() == 2 => {
      return crate::functions::calculus_ast::integrate_ast(args);
    }
    "NIntegrate" if args.len() == 2 => {
      return crate::functions::calculus_ast::nintegrate_ast(args);
    }
    "Limit" if (2..=3).contains(&args.len()) => {
      return crate::functions::calculus_ast::limit_ast(args);
    }
    "Series" if args.len() == 2 => {
      return crate::functions::calculus_ast::series_ast(args);
    }

    // AST-native linear algebra functions
    "Dot" if args.len() == 2 => {
      return crate::functions::linear_algebra_ast::dot_ast(args);
    }
    "Det" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::det_ast(args);
    }
    "Minors" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::linear_algebra_ast::minors_ast(args);
    }
    "Inverse" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::inverse_ast(args);
    }
    "PseudoInverse" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::pseudo_inverse_ast(args);
    }
    "LinearSolve" if args.len() == 2 => {
      return crate::functions::linear_algebra_ast::linear_solve_ast(args);
    }
    "Tr" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::tr_ast(args);
    }
    "IdentityMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::identity_matrix_ast(args);
    }
    "BoxMatrix" if args.len() == 1 => {
      if let Some(n) = expr_to_i128(&args[0])
        && n >= 0
      {
        let size = (2 * n + 1) as usize;
        let row = Expr::List(vec![Expr::Integer(1); size]);
        return Ok(Expr::List(vec![row; size]));
      }
    }
    "DiagonalMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::diagonal_matrix_ast(args);
    }
    "DiamondMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::diamond_matrix_ast(args);
    }
    "DiskMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::disk_matrix_ast(args);
    }
    "LeviCivitaTensor" if args.len() == 2 => {
      if matches!(&args[1], Expr::Identifier(h) if h == "List") {
        return crate::functions::linear_algebra_ast::levi_civita_tensor_ast(
          &args[..1],
        );
      }
    }
    "Eigenvalues" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::eigenvalues_ast(args);
    }
    "Eigenvectors" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::eigenvectors_ast(args);
    }
    "Eigensystem" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::eigensystem_ast(args);
    }
    "RowReduce" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::row_reduce_ast(args);
    }
    "MatrixRank" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::matrix_rank_ast(args);
    }
    "NullSpace" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::null_space_ast(args);
    }
    "ConjugateTranspose" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::conjugate_transpose_ast(
        args,
      );
    }
    "Fit" if args.len() == 3 => {
      return crate::functions::linear_algebra_ast::fit_ast(args);
    }
    "Cross" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::linear_algebra_ast::cross_ast(args);
    }
    "Projection" if args.len() == 2 => {
      return crate::functions::linear_algebra_ast::projection_ast(args);
    }

    // CellularAutomaton
    "CellularAutomaton" if args.len() == 3 => {
      return crate::functions::cellular_automaton_ast::cellular_automaton_ast(
        args,
      );
    }

    // AST-native additional association functions
    "AssociationMap" if args.len() == 2 => {
      return crate::functions::association_ast::association_map_ast(args);
    }
    "AssociationThread" if args.len() == 2 => {
      return crate::functions::association_ast::association_thread_ast(args);
    }
    "Merge" if args.len() == 2 => {
      return crate::functions::association_ast::merge_ast(args);
    }
    "KeyMap" if args.len() == 2 => {
      return crate::functions::association_ast::key_map_ast(args);
    }
    "KeySelect" if args.len() == 2 => {
      return crate::functions::association_ast::key_select_ast(args);
    }
    "KeyTake" if args.len() == 2 => {
      return crate::functions::association_ast::key_take_ast(args);
    }
    "KeyDrop" if args.len() == 2 => {
      return crate::functions::association_ast::key_drop_ast(args);
    }

    // AST-native additional polynomial/CAS functions
    "ExpandAll" if args.len() == 1 => {
      return crate::functions::polynomial_ast::expand_all_ast(args);
    }
    "Cancel" if args.len() == 1 => {
      return crate::functions::polynomial_ast::cancel_ast(args);
    }
    "Collect" if args.len() == 2 => {
      return crate::functions::polynomial_ast::collect_ast(args);
    }
    "Together" if args.len() == 1 => {
      return crate::functions::polynomial_ast::together_ast(args);
    }
    "Apart" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::polynomial_ast::apart_ast(args);
    }

    // AST-native utility math functions
    "Unitize" if args.len() == 1 => {
      return crate::functions::math_ast::unitize_ast(args);
    }
    "Ramp" if args.len() == 1 => {
      return crate::functions::math_ast::ramp_ast(args);
    }
    "KroneckerDelta" => {
      return crate::functions::math_ast::kronecker_delta_ast(args);
    }
    "UnitStep" if !args.is_empty() => {
      return crate::functions::math_ast::unit_step_ast(args);
    }
    "Complex" if args.len() == 2 => {
      // Complex[a, b] → a + b*I, evaluated to simplify iterated Complex
      let real = &args[0];
      let imag = &args[1];
      // If imaginary part is 0 (integer), return just the real part
      if matches!(imag, Expr::Integer(0)) {
        return Ok(real.clone());
      }
      // If real part is 0 and imaginary is 1, return I
      if matches!(real, Expr::Integer(0)) && matches!(imag, Expr::Integer(1)) {
        return Ok(Expr::Identifier("I".to_string()));
      }
      // Check if both parts are purely real numbers (no I involved) for non-evaluated path
      let imag_has_i = contains_i(imag);
      if !imag_has_i {
        // If real part is 0, return b*I
        if matches!(real, Expr::Integer(0)) {
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(imag.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          });
        }
        // If imaginary is 1, return a + I
        if matches!(imag, Expr::Integer(1)) {
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Plus,
            left: Box::new(real.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          });
        }
        // General case without I in imag: a + b*I (or a - |b|*I if b < 0)
        // Check if imag is negative to format as subtraction
        let (is_neg, abs_imag) = match imag {
          Expr::Real(f) if *f < 0.0 => (true, Expr::Real(-f)),
          Expr::Integer(n) if *n < 0 => (true, Expr::Integer(-n)),
          _ => (false, imag.clone()),
        };
        if is_neg {
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Minus,
            left: Box::new(real.clone()),
            right: Box::new(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Times,
              left: Box::new(abs_imag),
              right: Box::new(Expr::Identifier("I".to_string())),
            }),
          });
        }
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(real.clone()),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(imag.clone()),
            right: Box::new(Expr::Identifier("I".to_string())),
          }),
        });
      }
      // Imaginary part contains I (iterated Complex), evaluate algebraically
      // Complex[a, b] where b has form (re_b + im_b*I):
      // a + (re_b + im_b*I)*I = a + re_b*I + im_b*I^2 = (a - im_b) + re_b*I
      // Try to extract complex components from imag
      if let Some((re_b, im_b)) =
        crate::functions::math_ast::try_extract_complex_float(imag)
      {
        // Both a and components are numeric
        if let Some(a) = crate::functions::math_ast::try_eval_to_f64(real) {
          let new_re = a - im_b;
          let new_im = re_b;
          // Reconstruct as Complex[new_re, new_im]
          let re_expr = if new_re == (new_re as i128 as f64) {
            Expr::Integer(new_re as i128)
          } else {
            Expr::Real(new_re)
          };
          let im_expr = if new_im == (new_im as i128 as f64) {
            Expr::Integer(new_im as i128)
          } else {
            Expr::Real(new_im)
          };
          return evaluate_function_call_ast("Complex", &[re_expr, im_expr]);
        }
      }
      // Fallback: build a + b*I expression and evaluate
      let bi = evaluate_function_call_ast(
        "Times",
        &[imag.clone(), Expr::Identifier("I".to_string())],
      )?;
      return evaluate_function_call_ast("Plus", &[real.clone(), bi]);
    }
    "ConditionalExpression" if args.len() == 2 => match &args[1] {
      Expr::Identifier(name) if name == "True" => {
        return Ok(args[0].clone());
      }
      Expr::Identifier(name) if name == "False" => {
        return Ok(Expr::Identifier("Undefined".to_string()));
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ConditionalExpression".to_string(),
          args: args.to_vec(),
        });
      }
    },
    "DirectedInfinity" if args.len() <= 1 => {
      if args.is_empty() {
        // DirectedInfinity[] = ComplexInfinity
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      match &args[0] {
        Expr::Integer(1) => {
          return Ok(Expr::Identifier("Infinity".to_string()));
        }
        Expr::Integer(-1) => {
          return Ok(Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand: Box::new(Expr::Identifier("Infinity".to_string())),
          });
        }
        Expr::Integer(0) => {
          return Ok(Expr::Identifier("ComplexInfinity".to_string()));
        }
        _ => {
          // Try to normalize: DirectedInfinity[z] → DirectedInfinity[z/Abs[z]]
          if let Some(((re_n, re_d), (im_n, im_d))) =
            crate::functions::math_ast::try_extract_complex_exact(&args[0])
          {
            if im_n == 0 {
              // Pure real: just check sign
              if re_n > 0 {
                return Ok(Expr::Identifier("Infinity".to_string()));
              } else if re_n < 0 {
                return Ok(Expr::UnaryOp {
                  op: crate::syntax::UnaryOperator::Minus,
                  operand: Box::new(Expr::Identifier("Infinity".to_string())),
                });
              } else {
                return Ok(Expr::Identifier("ComplexInfinity".to_string()));
              }
            }
            // Compute magnitude squared: (re_n/re_d)^2 + (im_n/im_d)^2
            let mag_sq_num = re_n
              .checked_mul(re_n)
              .and_then(|a| {
                im_d.checked_mul(im_d).and_then(|b| a.checked_mul(b))
              })
              .and_then(|a| {
                im_n
                  .checked_mul(im_n)
                  .and_then(|c| {
                    re_d.checked_mul(re_d).and_then(|d| c.checked_mul(d))
                  })
                  .and_then(|b| a.checked_add(b))
              });
            let mag_sq_den = re_d.checked_mul(re_d).and_then(|a| {
              im_d.checked_mul(im_d).and_then(|b| a.checked_mul(b))
            });

            if let (Some(msn), Some(msd)) = (mag_sq_num, mag_sq_den) {
              // Build z/Abs[z] = z / Sqrt[msn/msd]
              let sqrt_arg = if msd == 1 {
                Expr::Integer(msn)
              } else {
                Expr::FunctionCall {
                  name: "Rational".to_string(),
                  args: vec![Expr::Integer(msn), Expr::Integer(msd)],
                }
              };
              let normalized = Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(args[0].clone()),
                right: Box::new(Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![sqrt_arg],
                }),
              };
              let normalized = evaluate_expr_to_expr(&normalized)?;
              // Check if normalized reduced to 1 or -1
              if matches!(&normalized, Expr::Integer(1)) {
                return Ok(Expr::Identifier("Infinity".to_string()));
              }
              if matches!(&normalized, Expr::Integer(-1)) {
                return Ok(Expr::UnaryOp {
                  op: crate::syntax::UnaryOperator::Minus,
                  operand: Box::new(Expr::Identifier("Infinity".to_string())),
                });
              }
              return Ok(Expr::FunctionCall {
                name: "DirectedInfinity".to_string(),
                args: vec![normalized],
              });
            }
          }
          return Ok(Expr::FunctionCall {
            name: "DirectedInfinity".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }

    // Echo[expr] - prints ">> expr" and returns expr
    // Echo[expr, label] - prints ">> label expr" and returns expr
    // Echo[expr, label, f] - prints ">> label f[expr]" and returns expr
    "Echo" if !args.is_empty() && args.len() <= 3 => {
      let label = if args.len() >= 2 {
        crate::syntax::expr_to_output(&args[1])
      } else {
        ">> ".to_string()
      };
      let display_expr = if args.len() == 3 {
        let f_applied = match &args[2] {
          Expr::Identifier(f_name) => Expr::FunctionCall {
            name: f_name.clone(),
            args: vec![args[0].clone()],
          },
          other => Expr::FunctionCall {
            name: "Apply".to_string(),
            args: vec![other.clone(), args[0].clone()],
          },
        };
        let result = evaluate_expr_to_expr(&f_applied)?;
        crate::syntax::expr_to_output(&result)
      } else {
        crate::syntax::expr_to_output(&args[0])
      };
      let line = if args.len() >= 2 {
        format!(">> {} {}", label, display_expr)
      } else {
        format!(">> {}", display_expr)
      };
      println!("{}", line);
      crate::capture_stdout(&line);
      return Ok(args[0].clone());
    }

    // Sow[expr] or Sow[expr, tag] - adds expr to the current Reap collection
    "Sow" if args.len() == 1 || args.len() == 2 => {
      let tag = if args.len() == 2 {
        args[1].clone()
      } else {
        Expr::Identifier("None".to_string())
      };
      crate::SOW_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(last) = stack.last_mut() {
          last.push((args[0].clone(), tag));
        }
      });
      return Ok(args[0].clone());
    }

    // Reap[expr] or Reap[expr, pattern] - evaluates expr, collecting all Sow'd values
    "Reap" if args.len() == 1 || args.len() == 2 => {
      // Push a new collection
      crate::SOW_STACK.with(|stack| {
        stack.borrow_mut().push(Vec::new());
      });
      // Evaluate the expression
      let result = evaluate_expr_to_expr(&args[0])?;
      // Pop the collection
      let sowed = crate::SOW_STACK
        .with(|stack| stack.borrow_mut().pop().unwrap_or_default());

      if args.len() == 1 {
        // Reap[expr] - group by unique tags, preserving order of first appearance
        if sowed.is_empty() {
          return Ok(Expr::List(vec![result, Expr::List(vec![])]));
        }
        let mut tag_order: Vec<Expr> = Vec::new();
        let mut tag_groups: Vec<Vec<Expr>> = Vec::new();
        for (val, tag) in &sowed {
          if let Some(idx) = tag_order
            .iter()
            .position(|t| expr_to_string(t) == expr_to_string(tag))
          {
            tag_groups[idx].push(val.clone());
          } else {
            tag_order.push(tag.clone());
            tag_groups.push(vec![val.clone()]);
          }
        }
        let groups: Vec<Expr> =
          tag_groups.into_iter().map(Expr::List).collect();
        return Ok(Expr::List(vec![result, Expr::List(groups)]));
      } else {
        // Reap[expr, patt] or Reap[expr, {patt1, patt2, ...}]
        let patt_arg = evaluate_expr_to_expr(&args[1])?;
        let patterns = match &patt_arg {
          Expr::List(pats) => pats.clone(),
          _ => vec![patt_arg.clone()],
        };
        let is_list_form = matches!(&patt_arg, Expr::List(_));

        let mut result_groups: Vec<Expr> = Vec::new();
        for patt in &patterns {
          // Collect all sowed values whose tag matches the pattern
          let mut matched: Vec<Expr> = Vec::new();
          let is_blank = matches!(patt, Expr::Pattern { .. });
          for (val, tag) in &sowed {
            if is_blank || expr_to_string(tag) == expr_to_string(patt) {
              matched.push(val.clone());
            }
          }
          if is_list_form {
            // {patt1, patt2, ...} form: each pattern gets a list wrapping
            if matched.is_empty() {
              result_groups.push(Expr::List(vec![]));
            } else {
              result_groups.push(Expr::List(vec![Expr::List(matched)]));
            }
          } else {
            // single pattern form: just the matched list
            if !matched.is_empty() {
              result_groups.push(Expr::List(matched));
            }
          }
        }
        return Ok(Expr::List(vec![result, Expr::List(result_groups)]));
      }
    }

    // ReplaceAll and ReplaceRepeated function call forms
    "ReplaceAll" if args.len() == 2 => {
      let expr = &args[0];
      let rules = &args[1];
      return apply_replace_all_ast(expr, rules);
    }
    "ReplaceRepeated" if args.len() == 2 => {
      let expr = &args[0];
      let rules = &args[1];
      return apply_replace_repeated_ast(expr, rules);
    }
    "Replace" if args.len() == 2 => {
      return apply_replace_ast(&args[0], &args[1]);
    }

    // Form wrappers — transparent, just return the inner expression
    "MathMLForm" | "StandardForm" | "InputForm" | "OutputForm"
      if !args.is_empty() =>
    {
      return Ok(args[0].clone());
    }

    // Symbolic operators with no built-in meaning — just return as-is with evaluated args
    "Therefore" | "Because" | "TableForm" | "Row" | "In" | "Grid" => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }

    _ => {}
  }

  // Check for user-defined functions
  // Clone overloads to avoid holding the borrow across evaluate calls
  let overloads = crate::FUNC_DEFS.with(|m| {
    let defs = m.borrow();
    defs.get(name).cloned()
  });

  if let Some(overloads) = overloads {
    for (params, conditions, param_defaults, param_heads, body_expr) in
      &overloads
    {
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
      for cond_opt in conditions.iter() {
        if let Some(cond_expr) = cond_opt {
          // Substitute all parameters with their argument values in the condition
          let mut substituted_cond = cond_expr.clone();
          for (param, arg) in params.iter().zip(effective_args.iter()) {
            substituted_cond =
              crate::syntax::substitute_variable(&substituted_cond, param, arg);
          }
          // Evaluate the condition - it must return True
          match evaluate_expr_to_expr(&substituted_cond) {
            Ok(Expr::Identifier(s)) if s == "True" => {} // condition met
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
      // Catch Return[] at the function call boundary
      return match evaluate_expr_to_expr(&substituted) {
        Err(InterpreterError::ReturnValue(val)) => Ok(*val),
        other => other,
      };
    }
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
      // Catch Return[] at the function call boundary
      return match evaluate_expr_to_expr(&substituted) {
        Err(InterpreterError::ReturnValue(val)) => Ok(*val),
        other => other,
      };
    }
  }

  // Package/message/system functions - no-op in Woxi, returns Null
  if name == "Needs"
    || name == "Message"
    || name == "Begin"
    || name == "End"
    || name == "BeginPackage"
    || name == "EndPackage"
    || name == "Off"
    || name == "On"
    || name == "Remove"
    || name == "SetOptions"
    || name == "ClearAttributes"
  {
    return Ok(Expr::Identifier("Null".to_string()));
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
    | "Darker"
    | "Lighter"
    | "Directive"
    | "Point"
    | "Line"
    | "Circle"
    | "Disk"
    | "Rectangle"
    | "Polygon"
    | "Arrow"
    | "BezierCurve"
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
    | "MatrixForm"
    | "Out"
    | "Condition"
    | "Show"
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
    | "Information"
    | "ListPlot3D"
    | "TreeForm" => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
    _ => {}
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
