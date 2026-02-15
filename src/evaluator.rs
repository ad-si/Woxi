use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
use crate::{
  ENV, InterpreterError, PART_DEPTH, StoredValue, format_real_result,
  format_result, interpret,
};

use std::collections::HashSet;
use std::sync::LazyLock;

/// Set of known Wolfram Language function names (from functions.csv)
/// that are NOT yet implemented in Woxi.
static KNOWN_WOLFRAM_FUNCTIONS: LazyLock<HashSet<&'static str>> =
  LazyLock::new(|| {
    let csv = include_str!("../functions.csv");
    csv
      .lines()
      .skip(1) // skip header
      .filter_map(|line| {
        let mut parts = line.splitn(3, ',');
        let name = parts.next()?.trim();
        let _desc = parts.next()?;
        let status = parts.next().unwrap_or("").trim();
        // Only include functions that are NOT implemented (not ✅)
        if status != "✅" && !name.is_empty() && name != "-----" {
          Some(name)
        } else {
          None
        }
      })
      .collect()
  });

/// Check if a function name is a known Wolfram Language function
/// that hasn't been implemented yet.
fn is_known_wolfram_function(name: &str) -> bool {
  KNOWN_WOLFRAM_FUNCTIONS.contains(name)
}

/// Evaluate an Expr AST directly without re-parsing.
/// This is the core optimization that avoids re-parsing function bodies.
pub fn evaluate_expr(expr: &Expr) -> Result<String, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(n.to_string()),
    Expr::BigInteger(n) => Ok(n.to_string()),
    Expr::Real(f) => Ok(crate::syntax::format_real(*f)),
    Expr::BigFloat(digits, prec) => Ok(format!("{}`{}.", digits, prec)),
    Expr::String(s) => Ok(format!("\"{}\"", s)),
    Expr::Identifier(name) => {
      // Look up in environment
      if let Some(stored) = ENV.with(|e| e.borrow().get(name).cloned()) {
        match stored {
          StoredValue::ExprVal(e) => Ok(expr_to_string(&e)),
          StoredValue::Raw(val) => Ok(val),
          StoredValue::Association(items) => {
            let parts: Vec<String> = items
              .iter()
              .map(|(k, v)| format!("{} -> {}", k, v))
              .collect();
            Ok(format!("<|{}|>", parts.join(", ")))
          }
        }
      } else {
        // Return as symbolic
        Ok(name.clone())
      }
    }
    Expr::Slot(n) => {
      // Slots should be replaced before evaluation
      // If we get here, return as-is
      if *n == 1 {
        Ok("#".to_string())
      } else {
        Ok(format!("#{}", n))
      }
    }
    Expr::Constant(name) => Ok(name.clone()),
    Expr::List(items) => {
      let evaluated: Vec<String> = items
        .iter()
        .map(evaluate_expr)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|s| s != "Nothing")
        .collect();
      Ok(format!("{{{}}}", evaluated.join(", ")))
    }
    Expr::FunctionCall { name, args } => {
      // Special handling for If - lazy evaluation of branches
      if name == "If" && (args.len() == 2 || args.len() == 3) {
        let cond = evaluate_expr(&args[0])?;
        if cond == "True" {
          return evaluate_expr(&args[1]);
        } else if cond == "False" {
          if args.len() == 3 {
            return evaluate_expr(&args[2]);
          } else {
            return Ok("Null".to_string());
          }
        } else {
          // Condition didn't evaluate to True/False - return unevaluated
          let args_str: Vec<String> = args.iter().map(expr_to_string).collect();
          return Ok(format!("If[{}]", args_str.join(", ")));
        }
      }
      // Functions with HoldAll: pass args unevaluated
      if name == "Function" {
        let result = evaluate_function_call_ast(name, args)?;
        return Ok(expr_to_string(&result));
      }
      // Evaluate using AST path to avoid interpret() recursion
      let evaluated_args: Vec<Expr> = args
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;
      let result = evaluate_function_call_ast(name, &evaluated_args)?;
      Ok(expr_to_string(&result))
    }
    Expr::BinaryOp { op, left, right } => {
      let left_val = evaluate_expr(left)?;
      let right_val = evaluate_expr(right)?;
      let any_real = left_val.contains('.') || right_val.contains('.');
      let fmt = |v: f64| {
        if any_real {
          format_real_result(v)
        } else {
          format_result(v)
        }
      };

      match op {
        BinaryOperator::Plus => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(fmt(l + r))
          } else {
            // Symbolic
            Ok(format!("{} + {}", left_val, right_val))
          }
        }
        BinaryOperator::Minus => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(fmt(l - r))
          } else {
            Ok(format!("{} - {}", left_val, right_val))
          }
        }
        BinaryOperator::Times => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(fmt(l * r))
          } else {
            Ok(format!("{} * {}", left_val, right_val))
          }
        }
        BinaryOperator::Divide => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            if r == 0.0 {
              Err(InterpreterError::EvaluationError("Division by zero".into()))
            } else {
              Ok(fmt(l / r))
            }
          } else {
            Ok(format!("{} / {}", left_val, right_val))
          }
        }
        BinaryOperator::Power => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(fmt(l.powf(r)))
          } else {
            Ok(format!("{}^{}", left_val, right_val))
          }
        }
        BinaryOperator::And => {
          let l = left_val == "True";
          let r = right_val == "True";
          Ok(if l && r { "True" } else { "False" }.to_string())
        }
        BinaryOperator::Or => {
          let l = left_val == "True";
          let r = right_val == "True";
          Ok(if l || r { "True" } else { "False" }.to_string())
        }
        BinaryOperator::StringJoin => {
          // Remove quotes if present
          let l = left_val.trim_matches('"');
          let r = right_val.trim_matches('"');
          Ok(format!("\"{}{}\"", l, r))
        }
        BinaryOperator::Alternatives => {
          // Alternatives stays symbolic (used in pattern matching)
          Ok(format!("{} | {}", left_val, right_val))
        }
      }
    }
    Expr::UnaryOp { op, operand } => {
      let val = evaluate_expr(operand)?;
      match op {
        UnaryOperator::Minus => {
          if let Ok(n) = val.parse::<f64>() {
            Ok(format_result(-n))
          } else {
            Ok(format!("-{}", val))
          }
        }
        UnaryOperator::Not => {
          if val == "True" {
            Ok("False".to_string())
          } else if val == "False" {
            Ok("True".to_string())
          } else {
            Ok(format!("!{}", val))
          }
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      if operands.len() < 2 || operators.is_empty() {
        return Ok("True".to_string());
      }

      let values: Vec<String> = operands
        .iter()
        .map(evaluate_expr)
        .collect::<Result<_, _>>()?;

      // Evaluate comparison chain
      for i in 0..operators.len() {
        let left = &values[i];
        let right = &values[i + 1];
        let op = &operators[i];

        let result = match op {
          ComparisonOp::Equal => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l == r
            } else {
              left == right
            }
          }
          ComparisonOp::NotEqual => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l != r
            } else {
              left != right
            }
          }
          ComparisonOp::Less => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l < r
            } else {
              return Ok(format!("{} < {}", left, right));
            }
          }
          ComparisonOp::LessEqual => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l <= r
            } else {
              return Ok(format!("{} <= {}", left, right));
            }
          }
          ComparisonOp::Greater => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l > r
            } else {
              return Ok(format!("{} > {}", left, right));
            }
          }
          ComparisonOp::GreaterEqual => {
            if let (Ok(l), Ok(r)) = (left.parse::<f64>(), right.parse::<f64>())
            {
              l >= r
            } else {
              return Ok(format!("{} >= {}", left, right));
            }
          }
          ComparisonOp::SameQ => left == right,
          ComparisonOp::UnsameQ => left != right,
        };

        if !result {
          return Ok("False".to_string());
        }
      }
      Ok("True".to_string())
    }
    Expr::CompoundExpr(exprs) => {
      let mut result = String::new();
      for e in exprs {
        result = evaluate_expr(e)?;
      }
      Ok(result)
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| {
          let key = evaluate_expr(k)?;
          let val = evaluate_expr(v)?;
          Ok(format!("{} -> {}", key, val))
        })
        .collect::<Result<_, InterpreterError>>()?;
      Ok(format!("<|{}|>", parts.join(", ")))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let p = evaluate_expr(pattern)?;
      let r = evaluate_expr(replacement)?;
      Ok(format!("{} -> {}", p, r))
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let p = expr_to_string(pattern);
      let r = expr_to_string(replacement);
      Ok(format!("{} :> {}", p, r))
    }
    Expr::ReplaceAll { expr: e, rules } => {
      // Use AST-based evaluation to avoid interpret() recursion
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_rules = evaluate_expr_to_expr(rules)?;
      let result = apply_replace_all_ast(&evaluated_expr, &evaluated_rules)?;
      Ok(expr_to_string(&result))
    }
    Expr::ReplaceRepeated { expr: e, rules } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_rules = evaluate_expr_to_expr(rules)?;
      let result =
        apply_replace_repeated_ast(&evaluated_expr, &evaluated_rules)?;
      Ok(expr_to_string(&result))
    }
    Expr::Map { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      let result = apply_map_ast(func, &evaluated_list)?;
      Ok(expr_to_string(&result))
    }
    Expr::Apply { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      let result = apply_apply_ast(func, &evaluated_list)?;
      Ok(expr_to_string(&result))
    }
    Expr::MapApply { func, list } => {
      let evaluated_list = evaluate_expr_to_expr(list)?;
      let result = apply_map_apply_ast(func, &evaluated_list)?;
      Ok(expr_to_string(&result))
    }
    Expr::PrefixApply { func, arg } => {
      let evaluated_arg = evaluate_expr_to_expr(arg)?;
      let result = apply_function_to_arg(func, &evaluated_arg)?;
      Ok(expr_to_string(&result))
    }
    Expr::Postfix { expr: e, func } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let result = apply_postfix_ast(&evaluated_expr, func)?;
      Ok(expr_to_string(&result))
    }
    Expr::Part { expr: e, index } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_idx = evaluate_expr_to_expr(index)?;
      let result = extract_part_ast(&evaluated_expr, &evaluated_idx)?;
      Ok(expr_to_string(&result))
    }
    Expr::Function { body } => {
      // Return anonymous function as-is (with & appended)
      Ok(format!("{}&", expr_to_string(body)))
    }
    Expr::NamedFunction { params, body } => {
      Ok(expr_to_string(&Expr::NamedFunction {
        params: params.clone(),
        body: body.clone(),
      }))
    }
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        Ok(format!("{}_{}", name, h))
      } else {
        Ok(format!("{}_", name))
      }
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Ok(expr_to_string(&Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default.clone(),
    })),
    Expr::Raw(s) => {
      // Fallback: parse to AST and evaluate to avoid interpret() recursion
      let parsed = string_to_expr(s)?;
      let result = evaluate_expr_to_expr(&parsed)?;
      Ok(expr_to_string(&result))
    }
    Expr::CurriedCall { func, args } => {
      // Evaluate the curried call: f[a][b] -> apply f[a] to b
      let evaluated_func = evaluate_expr_to_expr(func)?;
      let evaluated_args: Vec<Expr> = args
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;
      // Apply curried function
      let result = apply_curried_call(&evaluated_func, &evaluated_args)?;
      Ok(expr_to_string(&result))
    }
  }
}

/// Evaluate an Expr AST and return a new Expr (not a string).
/// This is the core function for AST-based evaluation without string round-trips.
pub fn evaluate_expr_to_expr(expr: &Expr) -> Result<Expr, InterpreterError> {
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
        // Return as symbolic identifier
        Ok(Expr::Identifier(name.clone()))
      }
    }
    Expr::Slot(n) => {
      // Slots should be replaced before evaluation
      Ok(Expr::Slot(*n))
    }
    Expr::Constant(name) => Ok(Expr::Constant(name.clone())),
    Expr::List(items) => {
      let evaluated: Vec<Expr> = items
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|e| !matches!(e, Expr::Identifier(s) if s == "Nothing"))
        .collect();
      Ok(Expr::List(evaluated))
    }
    Expr::FunctionCall { name, args } => {
      // Special handling for If - lazy evaluation of branches
      if name == "If" && (args.len() == 2 || args.len() == 3) {
        let cond = evaluate_expr_to_expr(&args[0])?;
        if matches!(&cond, Expr::Identifier(s) if s == "True") {
          return evaluate_expr_to_expr(&args[1]);
        } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
          if args.len() == 3 {
            return evaluate_expr_to_expr(&args[2]);
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
      // Special handling for Module/Block - don't evaluate args (body needs local bindings first)
      if name == "Module" {
        return module_ast(args);
      }
      if name == "Block" {
        return block_ast(args);
      }
      // Special handling for Set - first arg must be identifier or Part, second gets evaluated
      if name == "Set" && args.len() == 2 {
        return set_ast(&args[0], &args[1]);
      }
      // Special handling for SetDelayed - stores function definitions
      if name == "SetDelayed" && args.len() == 2 {
        return set_delayed_ast(&args[0], &args[1]);
      }
      // Special handling for Increment/Decrement - x++ / x--
      if (name == "Increment" || name == "Decrement")
        && args.len() == 1
        && let Expr::Identifier(var_name) = &args[0]
      {
        let current = ENV.with(|e| e.borrow().get(var_name).cloned());
        let current_val = match current {
          Some(StoredValue::ExprVal(e)) => e,
          Some(StoredValue::Raw(s)) => {
            crate::syntax::string_to_expr(&s).unwrap_or(Expr::Integer(0))
          }
          _ => Expr::Integer(0),
        };
        let delta = if name == "Increment" {
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
        return Ok(current_val);
      }
      // Special handling for AddTo, SubtractFrom, TimesBy, DivideBy - x += y, x -= y, etc.
      if (name == "AddTo"
        || name == "SubtractFrom"
        || name == "TimesBy"
        || name == "DivideBy")
        && args.len() == 2
        && let Expr::Identifier(var_name) = &args[0]
      {
        let rhs = evaluate_expr_to_expr(&args[1])?;
        let current = ENV.with(|e| e.borrow().get(var_name).cloned());
        let current_val = match current {
          Some(StoredValue::ExprVal(e)) => e,
          Some(StoredValue::Raw(s)) => {
            crate::syntax::string_to_expr(&s).unwrap_or(Expr::Integer(0))
          }
          _ => Expr::Integer(0),
        };
        let op = match name.as_str() {
          "AddTo" => crate::syntax::BinaryOperator::Plus,
          "SubtractFrom" => crate::syntax::BinaryOperator::Minus,
          "TimesBy" => crate::syntax::BinaryOperator::Times,
          "DivideBy" => crate::syntax::BinaryOperator::Divide,
          _ => unreachable!(),
        };
        let new_val = evaluate_expr_to_expr(&Expr::BinaryOp {
          op,
          left: Box::new(current_val),
          right: Box::new(rhs),
        })?;
        ENV.with(|e| {
          e.borrow_mut().insert(
            var_name.clone(),
            StoredValue::Raw(crate::syntax::expr_to_string(&new_val)),
          );
        });
        return Ok(new_val);
      }
      // Special handling for AppendTo, PrependTo - x = Append[x, elem]
      if (name == "AppendTo" || name == "PrependTo")
        && args.len() == 2
        && let Expr::Identifier(var_name) = &args[0]
      {
        let elem = evaluate_expr_to_expr(&args[1])?;
        let current = ENV.with(|e| e.borrow().get(var_name).cloned());
        let current_val = match current {
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
        let new_val = match current_val {
          Expr::List(mut items) => {
            if name == "AppendTo" {
              items.push(elem);
            } else {
              items.insert(0, elem);
            }
            Expr::List(items)
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
      // Special handling for Switch - lazy evaluation of branches
      if name == "Switch" && args.len() >= 3 {
        return crate::functions::control_flow_ast::switch_ast(args);
      }
      // Special handling for Piecewise - lazy evaluation of branches
      if name == "Piecewise" && !args.is_empty() && args.len() <= 2 {
        return crate::functions::control_flow_ast::piecewise_ast(args);
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
        || name == "HoldForm"
        || name == "ValueQ"
        || name == "Reap"
        || name == "Plot"
      {
        // Pass unevaluated args to the function dispatcher
        return evaluate_function_call_ast(name, args);
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
      // Check if name is a variable holding an association (for nested access: assoc["a", "b"])
      if let Some(StoredValue::Association(_)) = var_val {
        // Evaluate arguments and perform nested access
        let evaluated_args: Vec<Expr> = args
          .iter()
          .map(evaluate_expr_to_expr)
          .collect::<Result<_, _>>()?;
        return association_nested_access(name, &evaluated_args);
      }

      // Evaluate arguments
      let evaluated_args: Vec<Expr> = args
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;
      // Dispatch to function implementation
      evaluate_function_call_ast(name, &evaluated_args)
    }
    Expr::BinaryOp { op, left, right } => {
      let left_val = evaluate_expr_to_expr(left)?;
      let right_val = evaluate_expr_to_expr(right)?;

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
            // 0 - x => -x (Times[-1, x])
            Ok(Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: Box::new(right_val),
            })
          } else if crate::functions::quantity_ast::is_quantity(&left_val)
            .is_some()
            || crate::functions::quantity_ast::is_quantity(&right_val).is_some()
          {
            // Quantity subtraction: delegate to subtract_ast → Plus[a, Times[-1, b]]
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
          } else {
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
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
        BinaryOperator::And => {
          let l = matches!(&left_val, Expr::Identifier(s) if s == "True");
          let r = matches!(&right_val, Expr::Identifier(s) if s == "True");
          Ok(Expr::Identifier(
            if l && r { "True" } else { "False" }.to_string(),
          ))
        }
        BinaryOperator::Or => {
          let l = matches!(&left_val, Expr::Identifier(s) if s == "True");
          let r = matches!(&right_val, Expr::Identifier(s) if s == "True");
          Ok(Expr::Identifier(
            if l || r { "True" } else { "False" }.to_string(),
          ))
        }
        BinaryOperator::StringJoin => {
          let l = expr_to_raw_string(&left_val);
          let r = expr_to_raw_string(&right_val);
          Ok(Expr::String(format!("{}{}", l, r)))
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
          if let Some(n) = expr_to_number(&val) {
            Ok(num_to_expr(-n))
          } else {
            Ok(Expr::UnaryOp {
              op: *op,
              operand: Box::new(val),
            })
          }
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

      // Evaluate comparison chain
      // Use try_eval_to_f64 for numeric comparisons (handles symbolic Pi, E, Degree, Sin[...], etc.)
      use crate::functions::math_ast::try_eval_to_f64;
      for i in 0..operators.len() {
        let left = &values[i];
        let right = &values[i + 1];
        let op = &operators[i];

        let result = match op {
          ComparisonOp::SameQ => {
            if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l == r
            } else {
              expr_to_string(left) == expr_to_string(right)
            }
          }
          ComparisonOp::Equal => {
            if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l == r
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
            if let (Some(l), Some(r)) =
              (try_eval_to_f64(left), try_eval_to_f64(right))
            {
              l != r
            } else {
              expr_to_string(left) != expr_to_string(right)
            }
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
      for e in exprs {
        result = evaluate_expr_to_expr(e)?;
      }
      Ok(result)
    }
    Expr::Association(items) => {
      let evaluated: Result<Vec<(Expr, Expr)>, InterpreterError> = items
        .iter()
        .map(|(k, v)| {
          let key = evaluate_expr_to_expr(k)?;
          let val = evaluate_expr_to_expr(v)?;
          Ok((key, val))
        })
        .collect();
      Ok(Expr::Association(evaluated?))
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
      apply_replace_all_ast(&evaluated_expr, &evaluated_rules)
    }
    Expr::ReplaceRepeated { expr: e, rules } => {
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_rules = evaluate_expr_to_expr(rules)?;
      apply_replace_repeated_ast(&evaluated_expr, &evaluated_rules)
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
        indices.push(evaluate_expr_to_expr(idx)?);
      }

      // Check if any index is All — use apply_part_indices path only when needed
      let has_all = indices
        .iter()
        .any(|idx| matches!(idx, Expr::Identifier(s) if s == "All"));

      let result = if has_all {
        // All requires collecting indices and mapping — must clone base
        let base_val = eval_part_base(base_expr)?;
        apply_part_indices(&base_val, &indices)?
      } else {
        // Fast path: no All, use original optimized approach
        // Apply indices one at a time with ENV optimization for identifiers
        let mut result = if let Expr::Identifier(var_name) = base_expr {
          let env_result = ENV.with(|env| {
            let env = env.borrow();
            if let Some(StoredValue::ExprVal(stored)) = env.get(var_name) {
              Some(extract_part_ast(stored, &indices[0]))
            } else {
              None
            }
          });
          if let Some(r) = env_result {
            r?
          } else {
            let evaluated_expr = evaluate_expr_to_expr(base_expr)?;
            extract_part_ast(&evaluated_expr, &indices[0])?
          }
        } else {
          let evaluated_expr = evaluate_expr_to_expr(base_expr)?;
          extract_part_ast(&evaluated_expr, &indices[0])?
        };
        // Apply remaining indices
        for idx in &indices[1..] {
          result = extract_part_ast(&result, idx)?;
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
          eprintln!();
          eprintln!(
            "Part::partd: Part specification {} is longer than depth of object.",
            part_str
          );
        }
      }
      Ok(result)
    }
    Expr::Function { body } => {
      // Return anonymous function as-is
      Ok(Expr::Function { body: body.clone() })
    }
    Expr::NamedFunction { params, body } => {
      // Return named function as-is
      Ok(Expr::NamedFunction {
        params: params.clone(),
        body: body.clone(),
      })
    }
    Expr::Pattern { name, head } => Ok(Expr::Pattern {
      name: name.clone(),
      head: head.clone(),
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
    Expr::Raw(s) => {
      // Fallback: parse and evaluate the raw string
      let parsed = string_to_expr(s)?;
      evaluate_expr_to_expr(&parsed)
    }
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
    | Expr::Slot(_) => false,
    Expr::List(items) => items.iter().any(has_free_symbols),
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbols(left) || has_free_symbols(right)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbols(operand),
    Expr::FunctionCall { args, .. } => args.iter().any(has_free_symbols),
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

/// Convert an Expr to a number if possible
fn expr_to_number(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
    Expr::Real(f) => Some(*f),
    Expr::Constant(name) => constant_to_f64(name),
    _ => None,
  }
}

/// Extract an i128 from Integer or BigInteger (if it fits)
fn expr_to_i128(expr: &Expr) -> Option<i128> {
  use num_traits::ToPrimitive;
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => n.to_i128(),
    _ => None,
  }
}

/// Resolve a named constant to its numeric f64 value.
/// Constants (Pi, E, Degree) are kept symbolic — use try_eval_to_f64 for numeric evaluation.
fn constant_to_f64(_name: &str) -> Option<f64> {
  None
}

/// Convert a number to an appropriate Expr (Integer if whole, Real otherwise)
fn num_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// Convert an Expr to BigInt if it's an integer type
fn expr_to_bigint(expr: &Expr) -> Option<num_bigint::BigInt> {
  match expr {
    Expr::Integer(n) => Some(num_bigint::BigInt::from(*n)),
    Expr::BigInteger(n) => Some(n.clone()),
    _ => None,
  }
}

/// Check if an expression requires BigInt arithmetic (exceeds f64 precision).
/// f64 can only represent integers exactly up to 2^53.
fn needs_bigint(expr: &Expr) -> bool {
  match expr {
    Expr::BigInteger(_) => true,
    Expr::Integer(n) => n.unsigned_abs() > (1u128 << 53),
    _ => false,
  }
}

/// Apply a binary operation when at least one operand is a BigInteger or large Integer
fn bigint_binary_op<F>(left: &Expr, right: &Expr, op: F) -> Option<Expr>
where
  F: FnOnce(num_bigint::BigInt, num_bigint::BigInt) -> num_bigint::BigInt,
{
  if !needs_bigint(left) && !needs_bigint(right) {
    return None;
  }
  let l = expr_to_bigint(left)?;
  let r = expr_to_bigint(right)?;
  Some(crate::functions::math_ast::bigint_to_expr(op(l, r)))
}

/// Thread a binary operation over lists (e.g., {1,2,3} + 2 -> {3,4,5})
fn thread_binary_op(
  left: &Expr,
  right: &Expr,
  op: BinaryOperator,
) -> Result<Expr, InterpreterError> {
  // Helper to apply the binary operation to two evaluated expressions
  fn apply_op(
    l: &Expr,
    r: &Expr,
    op: BinaryOperator,
  ) -> Result<Expr, InterpreterError> {
    // Check if BigInt arithmetic is needed
    if (needs_bigint(l) || needs_bigint(r))
      && let (Some(lb), Some(rb)) = (expr_to_bigint(l), expr_to_bigint(r))
    {
      let result = match op {
        BinaryOperator::Plus => lb + rb,
        BinaryOperator::Minus => lb - rb,
        BinaryOperator::Times => lb * rb,
        _ => {
          // For Divide/Power, fall through to f64 path
          let ln = expr_to_number(l);
          let rn = expr_to_number(r);
          if let (Some(a), Some(b)) = (ln, rn) {
            return Ok(num_to_expr(a / b));
          }
          return Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          });
        }
      };
      return Ok(crate::functions::math_ast::bigint_to_expr(result));
    }
    let ln = expr_to_number(l);
    let rn = expr_to_number(r);
    let any_real = matches!(l, Expr::Real(_)) || matches!(r, Expr::Real(_));
    match op {
      BinaryOperator::Plus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a + b))
          } else {
            Ok(num_to_expr(a + b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Minus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a - b))
          } else {
            Ok(num_to_expr(a - b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Times => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a * b))
          } else {
            Ok(num_to_expr(a * b))
          }
        } else if matches!(l, Expr::Integer(0)) || matches!(r, Expr::Integer(0))
        {
          Ok(Expr::Integer(0))
        } else if matches!(l, Expr::Integer(1)) {
          Ok(r.clone())
        } else if matches!(r, Expr::Integer(1)) {
          Ok(l.clone())
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Divide => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if b == 0.0 {
            Err(InterpreterError::EvaluationError("Division by zero".into()))
          } else if any_real {
            Ok(Expr::Real(a / b))
          } else {
            Ok(num_to_expr(a / b))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      BinaryOperator::Power => {
        if let (Some(a), Some(b)) = (ln, rn) {
          if any_real {
            Ok(Expr::Real(a.powf(b)))
          } else {
            Ok(num_to_expr(a.powf(b)))
          }
        } else {
          Ok(Expr::BinaryOp {
            op,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
          })
        }
      }
      _ => Ok(Expr::BinaryOp {
        op,
        left: Box::new(l.clone()),
        right: Box::new(r.clone()),
      }),
    }
  }

  match (left, right) {
    (Expr::List(left_items), Expr::List(right_items)) => {
      // Both lists - element-wise operation
      if left_items.len() != right_items.len() {
        return Err(InterpreterError::EvaluationError(
          "Lists must have the same length for element-wise operations".into(),
        ));
      }
      let results: Result<Vec<Expr>, _> = left_items
        .iter()
        .zip(right_items.iter())
        .map(|(l, r)| apply_op(l, r, op))
        .collect();
      Ok(Expr::List(results?))
    }
    (Expr::List(items), scalar) => {
      // List op scalar - broadcast scalar
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_op(item, scalar, op))
        .collect();
      Ok(Expr::List(results?))
    }
    (scalar, Expr::List(items)) => {
      // Scalar op list - broadcast scalar
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_op(scalar, item, op))
        .collect();
      Ok(Expr::List(results?))
    }
    _ => apply_op(left, right, op),
  }
}

/// Extract raw string content from an Expr (without quotes for strings)
fn expr_to_raw_string(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    _ => expr_to_string(expr),
  }
}

/// Parse a string to Expr (wrapper for syntax::string_to_expr)
fn string_to_expr(s: &str) -> Result<Expr, InterpreterError> {
  crate::syntax::string_to_expr(s)
}

/// Dispatch function call to built-in implementations (AST version).
/// This is the AST equivalent of the string-based function dispatch.
/// IMPORTANT: This function must NOT call interpret() to avoid infinite recursion.
/// Built-in Listable functions (thread automatically over list arguments)
fn is_builtin_listable(name: &str) -> bool {
  matches!(
    name,
    "Fibonacci"
      | "LucasL"
      | "Sin"
      | "Cos"
      | "Tan"
      | "Sec"
      | "Csc"
      | "Cot"
      | "Sinh"
      | "Cosh"
      | "Tanh"
      | "Coth"
      | "Sech"
      | "Csch"
      | "ArcSin"
      | "ArcCos"
      | "ArcTan"
      | "ArcSinh"
      | "ArcCosh"
      | "ArcTanh"
      | "Exp"
      | "Log"
      | "Log2"
      | "Log10"
      | "Abs"
      | "Sign"
      | "Floor"
      | "Ceiling"
      | "Round"
      | "Sqrt"
      | "Surd"
      | "Factorial"
      | "Gamma"
      | "Erf"
      | "Erfc"
      | "Prime"
      | "Power"
      | "Plus"
      | "Times"
      | "Mod"
      | "Quotient"
      | "GCD"
      | "LCM"
      | "Binomial"
      | "Multinomial"
      | "IntegerDigits"
      | "FactorInteger"
      | "IntegerLength"
      | "RealDigits"
      | "RomanNumeral"
      | "EulerPhi"
      | "MoebiusMu"
      | "DivisorSigma"
      | "BernoulliB"
      | "CatalanNumber"
      | "StirlingS1"
      | "StirlingS2"
      | "ContinuedFraction"
      | "Boole"
      | "BitLength"
      | "EvenQ"
      | "OddQ"
      | "PrimeQ"
      | "Positive"
      | "Negative"
      | "NonPositive"
      | "NonNegative"
      | "StringLength"
  )
}

/// Thread a Listable function over list arguments.
/// Returns Some(result) if threading was applied, None otherwise.
fn thread_listable(
  name: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  // Check if any argument is a list
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if !has_list {
    return Ok(None);
  }

  // Find the list length (all lists must have the same length)
  let mut list_len = None;
  for arg in args {
    if let Expr::List(items) = arg {
      match list_len {
        None => list_len = Some(items.len()),
        Some(n) if n != items.len() => {
          // Mismatched list lengths — don't thread, let the function handle it
          return Ok(None);
        }
        _ => {}
      }
    }
  }

  let len = match list_len {
    Some(n) => n,
    None => return Ok(None),
  };

  // Thread element-wise
  let mut results = Vec::with_capacity(len);
  for i in 0..len {
    let threaded_args: Vec<Expr> = args
      .iter()
      .map(|arg| {
        if let Expr::List(items) = arg {
          items[i].clone()
        } else {
          arg.clone()
        }
      })
      .collect();
    results.push(evaluate_function_call_ast(name, &threaded_args)?);
  }
  Ok(Some(Expr::List(results)))
}

pub fn evaluate_function_call_ast(
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
    "Module" => return module_ast(args),
    "Block" => return block_ast(args),
    "With" if args.len() == 2 => return with_ast(args),
    "Set" if args.len() == 2 => {
      return set_ast(&args[0], &args[1]);
    }
    "SetAttributes" if args.len() == 2 => {
      if let Expr::Identifier(func_name) = &args[0] {
        let attr = match &args[1] {
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
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          let entry = attrs.entry(func_name.clone()).or_insert_with(Vec::new);
          for a in attr {
            if !entry.contains(&a) {
              entry.push(a);
            }
          }
        });
        return Ok(Expr::Identifier("Null".to_string()));
      }
    }
    "ClearAll" => {
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          ENV.with(|e| e.borrow_mut().remove(sym));
          crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
          crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
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
    "Fold" if args.len() == 3 => {
      return list_helpers_ast::fold_ast(&args[0], &args[1], &args[2]);
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

    // Additional AST-native list functions
    "Table" if args.len() == 2 => {
      return list_helpers_ast::table_ast(&args[0], &args[1]);
    }
    "Table" if args.len() >= 3 => {
      // Multi-dimensional Table: Table[expr, iter1, iter2, ...]
      // Nest from innermost to outermost
      return list_helpers_ast::table_multi_ast(&args[0], &args[1..]);
    }
    "MapThread" if args.len() == 2 => {
      return list_helpers_ast::map_thread_ast(&args[0], &args[1]);
    }
    "Partition" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::partition_ast(&args[0], n);
      }
    }
    "Permutations" if !args.is_empty() && args.len() <= 2 => {
      return list_helpers_ast::permutations_ast(args);
    }
    "Subsets" if !args.is_empty() && args.len() <= 2 => {
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
      // For other expressions, Normal is identity
      return Ok(args[0].clone());
    }
    "First" if args.len() == 1 => {
      return list_helpers_ast::first_ast(&args[0]);
    }
    "Last" if args.len() == 1 => {
      return list_helpers_ast::last_ast(&args[0]);
    }
    "Rest" if args.len() == 1 => {
      return list_helpers_ast::rest_ast(&args[0]);
    }
    "Most" if args.len() == 1 => {
      return list_helpers_ast::most_ast(&args[0]);
    }
    "Take" if args.len() == 2 => {
      return list_helpers_ast::take_ast(&args[0], &args[1]);
    }
    "Drop" if args.len() == 2 => {
      return list_helpers_ast::drop_ast(&args[0], &args[1]);
    }
    "Flatten" if args.len() == 1 => {
      return list_helpers_ast::flatten_ast(&args[0]);
    }
    "Flatten" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::flatten_level_ast(&args[0], n);
      }
    }
    "Reverse" if args.len() == 1 => {
      return list_helpers_ast::reverse_ast(&args[0]);
    }
    "Sort" if args.len() == 1 => {
      return list_helpers_ast::sort_ast(&args[0]);
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
    "Differences" if args.len() == 1 => {
      return list_helpers_ast::differences_ast(&args[0]);
    }
    "Scan" if args.len() == 2 => {
      return list_helpers_ast::scan_ast(&args[0], &args[1]);
    }
    "FoldList" if args.len() == 3 => {
      return list_helpers_ast::fold_list_ast(&args[0], &args[1], &args[2]);
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
    "Riffle" if args.len() == 2 => {
      return list_helpers_ast::riffle_ast(&args[0], &args[1]);
    }
    "RotateLeft" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::rotate_left_ast(&args[0], n);
      }
    }
    "RotateLeft" if args.len() == 1 => {
      return list_helpers_ast::rotate_left_ast(&args[0], 1);
    }
    "RotateRight" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::rotate_right_ast(&args[0], n);
      }
    }
    "RotateRight" if args.len() == 1 => {
      return list_helpers_ast::rotate_right_ast(&args[0], 1);
    }
    "PadLeft" if args.len() >= 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() == 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        return list_helpers_ast::pad_left_ast(&args[0], n, &pad);
      }
    }
    "PadRight" if args.len() >= 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        let pad = if args.len() == 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        return list_helpers_ast::pad_right_ast(&args[0], n, &pad);
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
    "Product" => {
      return list_helpers_ast::product_ast(args);
    }
    "Sum" => {
      return list_helpers_ast::sum_ast(args);
    }
    "Thread" if args.len() == 1 => {
      return list_helpers_ast::thread_ast(&args[0]);
    }
    "Through" if args.len() == 1 => {
      return list_helpers_ast::through_ast(&args[0]);
    }
    "Through" if args.len() == 2 => {
      // Through[expr, h] - only apply if head of expr matches h
      // For a list like {f, g}, the head is List, so Through[{f, g}, 0] returns {f, g}
      return Ok(args[0].clone());
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
    "TakeWhile" if args.len() == 2 => {
      return list_helpers_ast::take_while_ast(&args[0], &args[1]);
    }
    "Do" if args.len() == 2 => {
      return list_helpers_ast::do_ast(&args[0], &args[1]);
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
    "Array" if args.len() == 2 => {
      if let Some(n) = expr_to_i128(&args[1]) {
        return list_helpers_ast::array_ast(&args[0], n);
      }
    }
    "Gather" if args.len() == 1 => {
      return list_helpers_ast::gather_ast(&args[0]);
    }
    "GatherBy" if args.len() == 2 => {
      return list_helpers_ast::gather_by_ast(&args[1], &args[0]);
    }
    "Split" if args.len() == 1 => {
      return list_helpers_ast::split_ast(&args[0]);
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
    "Identity" if args.len() == 1 => {
      return list_helpers_ast::identity_ast(&args[0]);
    }
    "Outer" if args.len() == 3 => {
      return list_helpers_ast::outer_ast(&args[0], &args[1], &args[2]);
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
    "StringReplace" if args.len() == 2 => {
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
    "StringPosition" if args.len() == 2 => {
      return crate::functions::string_ast::string_position_ast(args);
    }
    "StringMatchQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_match_q_ast(args);
    }
    "StringReverse" if args.len() == 1 => {
      return crate::functions::string_ast::string_reverse_ast(args);
    }
    "StringRepeat" if args.len() == 2 => {
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

    // AST-native file and date functions (not available in WASM)
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
      // The second argument has already been evaluated, which triggers
      // capture_graphics() for Plot expressions.  Grab the SVG.
      let content = match &args[1] {
        Expr::Identifier(s) if s == "-Graphics-" => {
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
    #[cfg(not(target_arch = "wasm32"))]
    "DateString" => {
      use chrono::Local;
      let current_time = Local::now();
      let default_format = "%a, %d %b %Y %H:%M:%S";

      return match args.len() {
        0 => Ok(Expr::String(current_time.format(default_format).to_string())),
        1 => {
          // DateString[Now] or DateString["format"]
          if matches!(&args[0], Expr::Identifier(s) if s == "Now") {
            Ok(Expr::String(current_time.format(default_format).to_string()))
          } else if let Expr::String(format_str) = &args[0] {
            let fmt = match format_str.as_str() {
              "ISODateTime" => "%Y-%m-%dT%H:%M:%S",
              _ => format_str.as_str(),
            };
            Ok(Expr::String(current_time.format(fmt).to_string()))
          } else {
            Ok(Expr::String(current_time.format(default_format).to_string()))
          }
        }
        2 => {
          // DateString[Now, "format"]
          if !matches!(&args[0], Expr::Identifier(s) if s == "Now") {
            return Err(InterpreterError::EvaluationError(
              "DateString: First argument currently must be Now.".into(),
            ));
          }
          if let Expr::String(format_str) = &args[1] {
            let fmt = match format_str.as_str() {
              "ISODateTime" => "%Y-%m-%dT%H:%M:%S",
              _ => format_str.as_str(),
            };
            Ok(Expr::String(current_time.format(fmt).to_string()))
          } else {
            Err(InterpreterError::EvaluationError(
              "DateString: Second argument must be a format string.".into(),
            ))
          }
        }
        _ => Err(InterpreterError::EvaluationError(
          "DateString: Called with invalid number of arguments. Expected 0, 1, or 2.".into(),
        )),
      };
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
    "IntegerQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::integer_q_ast(args);
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
    "ListQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::list_q_ast(args);
    }
    "StringQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::string_q_ast(args);
    }
    "AtomQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::atom_q_ast(args);
    }
    "NumericQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::numeric_q_ast(args);
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
    "AssociationQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::association_q_ast(args);
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
    "Head" if args.len() == 1 => {
      return crate::functions::predicate_ast::head_ast(args);
    }
    "Length" if args.len() == 1 => {
      return crate::functions::predicate_ast::length_ast(args);
    }
    "Depth" if args.len() == 1 => {
      return crate::functions::predicate_ast::depth_ast(args);
    }

    // FullForm - returns full form representation (unevaluated)
    "FullForm" if args.len() == 1 => {
      return crate::functions::predicate_ast::full_form_ast(&args[0]);
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
    "Abs" if args.len() == 1 => {
      return crate::functions::math_ast::abs_ast(args);
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
    "GCD" => {
      return crate::functions::math_ast::gcd_ast(args);
    }
    "LCM" => {
      return crate::functions::math_ast::lcm_ast(args);
    }
    "Total" => {
      return crate::functions::math_ast::total_ast(args);
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
    "IntegerLength" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::integer_length_ast(args);
    }
    "Rescale" if !args.is_empty() && args.len() <= 3 => {
      return crate::functions::math_ast::rescale_ast(args);
    }
    "Normalize" if args.len() == 1 => {
      return crate::functions::math_ast::normalize_ast(args);
    }
    "Factorial" if args.len() == 1 => {
      return crate::functions::math_ast::factorial_ast(args);
    }
    "Gamma" if args.len() == 1 => {
      return crate::functions::math_ast::gamma_ast(args);
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
    "Clip" if !args.is_empty() && args.len() <= 2 => {
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
    "Prime" if args.len() == 1 => {
      return crate::functions::math_ast::prime_ast(args);
    }
    "Fibonacci" if args.len() == 1 => {
      return crate::functions::math_ast::fibonacci_ast(args);
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
    "CoprimeQ" if args.len() == 2 => {
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
    "NextPrime" if args.len() == 1 => {
      return crate::functions::math_ast::next_prime_ast(args);
    }
    "BitLength" if args.len() == 1 => {
      return crate::functions::math_ast::bit_length_ast(args);
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
    "Chop" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::chop_ast(args);
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
    "Xor" if args.len() >= 2 => {
      return crate::functions::boolean_ast::xor_ast(args);
    }
    "Return" => {
      let val = if args.is_empty() {
        Expr::Identifier("Null".to_string())
      } else {
        args[0].clone()
      };
      return Err(InterpreterError::ReturnValue(Box::new(val)));
    }
    "SameQ" if args.len() >= 2 => {
      return crate::functions::boolean_ast::same_q_ast(args);
    }
    "UnsameQ" if args.len() >= 2 => {
      return crate::functions::boolean_ast::unsame_q_ast(args);
    }
    "Which" if args.len() >= 2 && args.len().is_multiple_of(2) => {
      return crate::functions::boolean_ast::which_ast(args);
    }
    "While" if args.len() == 1 || args.len() == 2 => {
      return crate::functions::boolean_ast::while_ast(args);
    }
    "Equal" if args.len() >= 2 => {
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

    // AST-native polynomial functions
    "Expand" if args.len() == 1 => {
      return crate::functions::polynomial_ast::expand_ast(args);
    }
    "Factor" if args.len() == 1 => {
      return crate::functions::polynomial_ast::factor_ast(args);
    }
    "Simplify" if args.len() == 1 => {
      return crate::functions::polynomial_ast::simplify_ast(args);
    }
    "Coefficient" if args.len() >= 2 && args.len() <= 3 => {
      return crate::functions::polynomial_ast::coefficient_ast(args);
    }
    "Exponent" if args.len() == 2 => {
      return crate::functions::polynomial_ast::exponent_ast(args);
    }
    "PolynomialQ" if args.len() == 2 => {
      return crate::functions::polynomial_ast::polynomial_q_ast(args);
    }
    "Solve" if args.len() == 2 => {
      return crate::functions::polynomial_ast::solve_ast(args);
    }

    // AST-native list generation
    "Tuples" if args.len() == 2 => {
      return crate::functions::list_helpers_ast::tuples_ast(args);
    }

    // Use AST-native calculus functions
    "D" if args.len() == 2 => {
      return crate::functions::calculus_ast::d_ast(args);
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
    "Inverse" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::inverse_ast(args);
    }
    "Tr" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::tr_ast(args);
    }
    "IdentityMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::identity_matrix_ast(args);
    }
    "DiagonalMatrix" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::diagonal_matrix_ast(args);
    }
    "Eigenvalues" if args.len() == 1 => {
      return crate::functions::linear_algebra_ast::eigenvalues_ast(args);
    }
    "Cross" if args.len() == 2 => {
      return crate::functions::linear_algebra_ast::cross_ast(args);
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

    // Symbolic operators with no built-in meaning — just return as-is with evaluated args
    "Therefore" | "Because" => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }

    // Display wrapper — keeps evaluated args, formatting handled by expr_to_output
    "TableForm" => {
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

/// AST-based Module implementation to avoid interpret() recursion
fn module_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Module expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let local_vars = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          // x = value (assignment via Rule syntax internally or via Set)
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              vars.push((var_name.clone(), Some(set_args[1].clone())));
            }
          }
          // x = value (parsed as Rule or identifier = expr)
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              vars.push((var_name.clone(), Some(replacement.as_ref().clone())));
            }
          }
          // Just a variable name without initialization
          Expr::Identifier(var_name) => {
            vars.push((var_name.clone(), None));
          }
          // Try to extract from raw text (for cases like "x = 5")
          Expr::Raw(s) => {
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            } else {
              vars.push((s.trim().to_string(), None));
            }
          }
          // BinaryOp with some assignment-like pattern (less common)
          _ => {
            // Try to convert to string and parse
            let s = expr_to_string(item);
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() && !name.contains(' ') {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            }
          }
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Module expects a list of variable declarations as first argument"
          .into(),
      ));
    }
  };

  // Save previous bindings and set up new ones
  let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();

  for (var_name, init_expr) in &local_vars {
    let val = if let Some(expr) = init_expr {
      // Evaluate the initialization expression
      let evaluated = evaluate_expr_to_expr(expr)?;
      expr_to_string(&evaluated)
    } else {
      // Uninitialized variable - generate a unique symbol
      crate::functions::scoping::unique_symbol(var_name)
    };

    // Save current binding and set new one
    let pv = ENV.with(|e| {
      e.borrow_mut()
        .insert(var_name.clone(), StoredValue::Raw(val))
    });
    prev.push((var_name.clone(), pv));
  }

  // Evaluate the body expression (Return propagates through Module, not caught here)
  let result = evaluate_expr_to_expr(body_expr);

  // Restore previous bindings (even on error/Return)
  for (var_name, old) in prev {
    ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(v) = old {
        env.insert(var_name, v);
      } else {
        env.remove(&var_name);
      }
    });
  }

  result
}

/// AST-based Block implementation - dynamic scoping (like Module but without unique symbols).
/// Block[{x = 1, y}, body] - variables are localized but use their original names.
fn block_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Block expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations (same as Module)
  let local_vars = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              vars.push((var_name.clone(), Some(set_args[1].clone())));
            }
          }
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              vars.push((var_name.clone(), Some(replacement.as_ref().clone())));
            }
          }
          Expr::Identifier(var_name) => {
            vars.push((var_name.clone(), None));
          }
          _ => {
            let s = expr_to_string(item);
            if let Some((name, init)) = s.split_once('=') {
              let name = name.trim();
              let init = init.trim();
              if !name.is_empty() && !name.contains(' ') {
                let init_expr = string_to_expr(init)?;
                vars.push((name.to_string(), Some(init_expr)));
              }
            }
          }
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Block expects a list of variable declarations as first argument"
          .into(),
      ));
    }
  };

  // Save previous bindings and set up new ones
  let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();

  for (var_name, init_expr) in &local_vars {
    let pv = if let Some(expr) = init_expr {
      let evaluated = evaluate_expr_to_expr(expr)?;
      let val = expr_to_string(&evaluated);
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Raw(val))
      })
    } else {
      // Block with no initializer removes the variable binding so it evaluates as a symbol
      ENV.with(|e| e.borrow_mut().remove(var_name))
    };
    prev.push((var_name.clone(), pv));
  }

  // Evaluate the body expression (Return propagates through Block, not caught here)
  let result = evaluate_expr_to_expr(body_expr);

  // Restore previous bindings (even if body returned an error)
  for (var_name, old) in prev {
    ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(v) = old {
        env.insert(var_name, v);
      } else {
        env.remove(&var_name);
      }
    });
  }

  result
}

/// AST-based For loop: For[init, test, incr, body]
fn for_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(format!(
      "For expects 3 or 4 arguments; {} given",
      args.len()
    )));
  }

  let init = &args[0];
  let test = &args[1];
  let incr = &args[2];
  let body = args.get(3);

  const MAX_ITERATIONS: usize = 100000;

  // Evaluate the initialization
  evaluate_expr_to_expr(init)?;

  let mut iterations = 0;
  loop {
    // Evaluate the test condition
    let test_result = evaluate_expr_to_expr(test)?;
    match test_result {
      Expr::Identifier(s) if s == "True" => {}
      Expr::Identifier(s) if s == "False" => break,
      _ => break,
    }

    // Evaluate the body (if provided)
    if let Some(body) = body {
      match evaluate_expr_to_expr(body) {
        Ok(_) => {}
        Err(InterpreterError::BreakSignal) => break,
        Err(InterpreterError::ContinueSignal) => {}
        Err(e) => return Err(e),
      }
    }

    // Evaluate the increment
    evaluate_expr_to_expr(incr)?;

    iterations += 1;
    if iterations >= MAX_ITERATIONS {
      return Err(InterpreterError::EvaluationError(
        "For: maximum iterations exceeded".into(),
      ));
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based With implementation - substitutes bindings into body before evaluation.
/// With[{x = val, y = val2}, body] replaces x and y in body with evaluated values.
fn with_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let bindings: Vec<(String, Expr)> = match vars_expr {
    Expr::List(items) => {
      let mut vars = Vec::new();
      for item in items {
        match item {
          Expr::FunctionCall {
            name,
            args: set_args,
          } if name == "Set" && set_args.len() == 2 => {
            if let Expr::Identifier(var_name) = &set_args[0] {
              let val = evaluate_expr_to_expr(&set_args[1])?;
              vars.push((var_name.clone(), val));
            }
          }
          Expr::Rule {
            pattern,
            replacement,
          } => {
            if let Expr::Identifier(var_name) = pattern.as_ref() {
              let val = evaluate_expr_to_expr(replacement)?;
              vars.push((var_name.clone(), val));
            }
          }
          _ => {}
        }
      }
      vars
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "With expects a list of variable declarations as first argument".into(),
      ));
    }
  };

  // Substitute all bindings into the body
  let mut substituted = body_expr.clone();
  for (var_name, val) in &bindings {
    substituted =
      crate::syntax::substitute_variable(&substituted, var_name, val);
  }

  // Evaluate the substituted body
  evaluate_expr_to_expr(&substituted)
}

/// Recursively set a value at a path of indices within an Expr.
/// Supports lists and FunctionCall arguments (e.g., Grid[[1, row, col]]).
fn set_part_deep(
  expr: &mut Expr,
  indices: &[Expr],
  value: &Expr,
) -> Result<(), InterpreterError> {
  if indices.is_empty() {
    *expr = value.clone();
    return Ok(());
  }

  let idx = match &indices[0] {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64().ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Part assignment: index too large".into(),
        )
      })?
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Part assignment: index must be an integer".into(),
      ));
    }
  };

  match expr {
    Expr::List(items) => {
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of list does not exist.",
          idx
        )));
      }
      set_part_deep(&mut items[actual_idx as usize], &indices[1..], value)
    }
    Expr::FunctionCall { args, .. } => {
      // Part 0 is the head, Part 1.. are arguments (1-indexed)
      if idx == 0 {
        return Err(InterpreterError::EvaluationError(
          "Cannot set Part 0 (head) of a function call".into(),
        ));
      }
      let actual_idx = (idx - 1) as usize;
      if actual_idx >= args.len() {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {} of expression does not exist.",
          idx
        )));
      }
      set_part_deep(&mut args[actual_idx], &indices[1..], value)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Part assignment: cannot index into this expression".into(),
    )),
  }
}

/// AST-based Set implementation to handle Part assignment on associations and lists
fn set_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  // Handle Part assignment: var[[indices]] = value
  if let Expr::Part { .. } = lhs {
    // Flatten nested Part to get base variable and list of indices
    let mut indices = Vec::new();
    let mut current = lhs;
    while let Expr::Part { expr, index } = current {
      indices.push(index.as_ref().clone());
      current = expr.as_ref();
    }
    indices.reverse();

    let var_name = match current {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment requires a variable name".into(),
        ));
      }
    };

    // Evaluate indices
    let mut eval_indices = Vec::new();
    for idx in &indices {
      eval_indices.push(evaluate_expr_to_expr(idx)?);
    }

    // Evaluate the RHS
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Single-index association assignment: myHash[["key"]] = value
    if eval_indices.len() == 1 {
      let is_assoc = crate::ENV.with(|e| {
        let env = e.borrow();
        matches!(env.get(&var_name), Some(StoredValue::Association(_)))
      });
      if is_assoc {
        // Use expr_to_string to match the storage format (preserves string quotes)
        let key = expr_to_string(&eval_indices[0]);
        crate::ENV.with(|e| {
          let mut env = e.borrow_mut();
          if let Some(StoredValue::Association(pairs)) = env.get_mut(&var_name)
          {
            if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == &key) {
              pair.1 = expr_to_string(&rhs_value);
            } else {
              pairs.push((key, expr_to_string(&rhs_value)));
            }
          }
        });
        return Ok(rhs_value);
      }
    }

    // General Part assignment: modify in-place if ExprVal, otherwise parse/modify/store
    let modified_in_place = crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(StoredValue::ExprVal(expr)) = env.get_mut(&var_name) {
        // Modify directly in place — no clone needed
        set_part_deep(expr, &eval_indices, &rhs_value)
      } else {
        Err(InterpreterError::EvaluationError("not ExprVal".into()))
      }
    });
    if modified_in_place.is_ok() {
      return Ok(rhs_value);
    }

    // Fallback: parse stored string, modify, store back as ExprVal
    let stored_str = crate::ENV.with(|e| {
      let env = e.borrow();
      match env.get(&var_name) {
        Some(StoredValue::Raw(s)) => Some(s.clone()),
        _ => None,
      }
    });
    if let Some(stored_str) = stored_str {
      let mut stored_expr =
        string_to_expr(&stored_str).unwrap_or(Expr::Raw(stored_str));
      set_part_deep(&mut stored_expr, &eval_indices, &rhs_value)?;
      crate::ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name, StoredValue::ExprVal(stored_expr))
      });
      return Ok(rhs_value);
    }

    return Err(InterpreterError::EvaluationError(format!(
      "Variable {} not found",
      var_name
    )));
  }

  // Handle simple identifier assignment: x = value
  if let Expr::Identifier(var_name) = lhs {
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check if RHS is an association
    if let Expr::Association(items) = &rhs_value {
      let pairs: Vec<(String, String)> = items
        .iter()
        .map(|(k, v)| {
          // Use expr_to_string for keys to preserve type info
          // (e.g. Expr::String("x") → "\"x\"", Expr::Identifier("x") → "x")
          (expr_to_string(k), expr_to_string(v))
        })
        .collect();
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Association(pairs))
      });
    } else if matches!(
      &rhs_value,
      Expr::List(_)
        | Expr::FunctionCall { .. }
        | Expr::String(_)
        | Expr::Function { .. }
        | Expr::NamedFunction { .. }
    ) {
      // Store lists, function calls, functions, and strings as ExprVal for faithful roundtrip
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::ExprVal(rhs_value.clone()))
      });
    } else {
      ENV.with(|e| {
        e.borrow_mut().insert(
          var_name.clone(),
          StoredValue::Raw(expr_to_string(&rhs_value)),
        )
      });
    }

    return Ok(rhs_value);
  }

  // Handle DownValues: f[val1, val2, ...] = rhs
  // Store as a function definition with literal-match conditions
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Build param names and conditions for each argument
    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let defaults = vec![None; lhs_args.len()];
    let heads = vec![None; lhs_args.len()];

    for (i, arg) in lhs_args.iter().enumerate() {
      let param_name = format!("_dv{}", i);
      // Evaluate the literal argument value
      let eval_arg = evaluate_expr_to_expr(arg)?;
      // Condition: _dvN === eval_arg (using SameQ for exact matching)
      conditions.push(Some(Expr::Comparison {
        operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
        operators: vec![crate::syntax::ComparisonOp::SameQ],
      }));
      params.push(param_name);
    }

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      // Insert at the beginning so specific values take priority over general patterns
      entry.insert(0, (params, conditions, defaults, heads, rhs_value.clone()));
    });

    return Ok(rhs_value);
  }

  Err(InterpreterError::EvaluationError(
    "First argument of Set must be an identifier, part extract, or function call".into(),
  ))
}

/// Handle SetDelayed[f[patterns...], body] — stores a function definition.
/// This handles cases that the PEG FunctionDefinition rule doesn't parse,
/// such as list-pattern arguments: f[{x_Integer, y_Integer}] := body.
fn set_delayed_ast(lhs: &Expr, body: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults: Vec<Option<Expr>> = Vec::new();
    let mut heads: Vec<Option<String>> = Vec::new();
    // We also need to track substitutions for list-pattern destructuring
    let mut body_substitutions: Vec<(String, Vec<(String, Option<String>)>)> =
      Vec::new();

    for (i, arg) in lhs_args.iter().enumerate() {
      match arg {
        // List pattern: {x_Integer, y_Integer} — destructure a list argument
        Expr::List(patterns) => {
          let param_name = format!("_lp{}", i);
          // Condition: argument must be a list with the right length
          conditions.push(Some(Expr::Comparison {
            operands: vec![
              Expr::FunctionCall {
                name: "Length".to_string(),
                args: vec![Expr::Identifier(param_name.clone())],
              },
              Expr::Integer(patterns.len() as i128),
            ],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
          // Extract pattern names and head constraints from list elements
          let mut element_bindings = Vec::new();
          for pat in patterns {
            let (pat_name, head) = extract_pattern_info(pat);
            element_bindings.push((pat_name, head));
          }
          body_substitutions.push((param_name.clone(), element_bindings));
          params.push(param_name);
          defaults.push(None);
          heads.push(Some("List".to_string()));
        }
        // Simple pattern: x_ or x_Head
        _ => {
          let (pat_name, head) = extract_pattern_info(arg);
          params.push(pat_name);
          conditions.push(None);
          defaults.push(None);
          heads.push(head);
        }
      }
    }

    // Build the body with list-destructuring substitutions.
    // For each list-pattern param, replace references to element names
    // with Part[param, index] expressions.
    let mut final_body = body.clone();
    for (param_name, element_bindings) in &body_substitutions {
      for (idx, (elem_name, _head)) in element_bindings.iter().enumerate() {
        if !elem_name.is_empty() {
          // Replace elem_name with Part[param_name, idx+1]
          let part_expr = Expr::FunctionCall {
            name: "Part".to_string(),
            args: vec![
              Expr::Identifier(param_name.clone()),
              Expr::Integer((idx + 1) as i128),
            ],
          };
          final_body = crate::syntax::substitute_variable(
            &final_body,
            elem_name,
            &part_expr,
          );
        }
      }
    }

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      entry.push((params, conditions, defaults, heads, final_body));
    });

    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Fallback: return symbolic form
  Ok(Expr::FunctionCall {
    name: "SetDelayed".to_string(),
    args: vec![lhs.clone(), body.clone()],
  })
}

/// Extract a pattern name and optional head constraint from a pattern expression.
/// e.g., x_Integer -> ("x", Some("Integer")), x_ -> ("x", None)
fn extract_pattern_info(expr: &Expr) -> (String, Option<String>) {
  match expr {
    Expr::Identifier(name) => {
      // Could be a pattern like "x_Integer" or "x_" in text form
      if let Some(pos) = name.find('_') {
        let pat_name = name[..pos].to_string();
        let head = &name[pos + 1..];
        if head.is_empty() {
          (pat_name, None)
        } else {
          (pat_name, Some(head.to_string()))
        }
      } else {
        (name.clone(), None)
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      // Pattern[name, Blank[head]] or Pattern[name, Blank[]]
      if let Expr::Identifier(pat_name) = &args[0]
        && let Expr::FunctionCall {
          name: blank_name,
          args: blank_args,
        } = &args[1]
        && blank_name == "Blank"
      {
        let head = blank_args.first().and_then(|a| {
          if let Expr::Identifier(h) = a {
            Some(h.clone())
          } else {
            None
          }
        });
        return (pat_name.clone(), head);
      }
      (String::new(), None)
    }
    _ => {
      // Try to extract from string representation
      let s = expr_to_string(expr);
      if let Some(pos) = s.find('_') {
        let pat_name = s[..pos].to_string();
        let head = &s[pos + 1..];
        if head.is_empty() {
          (pat_name, None)
        } else {
          (pat_name, Some(head.to_string()))
        }
      } else {
        (String::new(), None)
      }
    }
  }
}

/// Perform nested access on an association: assoc["a", "b"] -> assoc["a"]["b"]
fn association_nested_access(
  var_name: &str,
  keys: &[Expr],
) -> Result<Expr, InterpreterError> {
  if keys.is_empty() {
    // Return the association itself
    return ENV.with(|e| {
      if let Some(StoredValue::Association(pairs)) = e.borrow().get(var_name) {
        let items: Vec<(Expr, Expr)> = pairs
          .iter()
          .map(|(k, v)| {
            (
              string_to_expr(k).unwrap_or(Expr::Identifier(k.clone())),
              string_to_expr(v).unwrap_or(Expr::Raw(v.clone())),
            )
          })
          .collect();
        Ok(Expr::Association(items))
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "{} is not an association",
          var_name
        )))
      }
    });
  }

  // Get the association
  let assoc = ENV.with(|e| e.borrow().get(var_name).cloned());
  match assoc {
    Some(StoredValue::Association(pairs)) => {
      // Perform nested access
      let mut current_val: Option<String> = None;
      let mut current_pairs = pairs;

      for key in keys {
        // Use expr_to_string to match storage format (preserves string quotes)
        let key_str = expr_to_string(key);

        // Look up key in current association
        if let Some((_, val)) =
          current_pairs.iter().find(|(k, _)| k == &key_str)
        {
          // Check if val is a nested association
          if val.starts_with("<|") && val.ends_with("|>") {
            // Parse the nested association
            match crate::interpret(&format!("Keys[{}]", val)) {
              Ok(_) => {
                // It's an association - we need to continue drilling down
                // Parse the association into pairs
                match parse_association_string(val) {
                  Ok(nested_pairs) => {
                    current_pairs = nested_pairs;
                    current_val = None;
                  }
                  Err(_) => {
                    current_val = Some(val.clone());
                  }
                }
              }
              Err(_) => {
                current_val = Some(val.clone());
              }
            }
          } else {
            current_val = Some(val.clone());
          }
        } else {
          // Key not found
          return Ok(Expr::FunctionCall {
            name: var_name.to_string(),
            args: keys.to_vec(),
          });
        }
      }

      // Return the final value
      if let Some(val) = current_val {
        string_to_expr(&val).or(Ok(Expr::Raw(val)))
      } else {
        // Return remaining association
        let items: Vec<(Expr, Expr)> = current_pairs
          .iter()
          .map(|(k, v)| {
            (
              Expr::Identifier(k.clone()),
              string_to_expr(v).unwrap_or(Expr::Raw(v.clone())),
            )
          })
          .collect();
        Ok(Expr::Association(items))
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{} is not an association",
      var_name
    ))),
  }
}

/// Parse an association string like "<|a -> 1, b -> 2|>" into pairs
fn parse_association_string(
  s: &str,
) -> Result<Vec<(String, String)>, InterpreterError> {
  if !s.starts_with("<|") || !s.ends_with("|>") {
    return Err(InterpreterError::EvaluationError(
      "Not an association".into(),
    ));
  }
  let inner = &s[2..s.len() - 2]; // Strip <| and |>
  let mut pairs = Vec::new();

  // Simple parsing - split by ", " and then by " -> "
  for item in split_association_items(inner) {
    if let Some(arrow_pos) = item.find(" -> ") {
      let key = item[..arrow_pos].trim().to_string();
      let val = item[arrow_pos + 4..].trim().to_string();
      pairs.push((key, val));
    }
  }

  Ok(pairs)
}

/// Split association items handling nested associations
fn split_association_items(s: &str) -> Vec<String> {
  let mut items = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in s.chars() {
    match c {
      '<' => {
        depth += 1;
        current.push(c);
      }
      '>' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        items.push(current.trim().to_string());
        current = String::new();
      }
      _ => current.push(c),
    }
  }
  if !current.trim().is_empty() {
    items.push(current.trim().to_string());
  }
  items
}

/// Check if a pattern Expr contains any Expr::Pattern nodes (named blanks like n_).
fn contains_pattern(expr: &Expr) -> bool {
  match expr {
    Expr::Pattern { .. } => true,
    Expr::BinaryOp { left, right, .. } => {
      contains_pattern(left) || contains_pattern(right)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_pattern),
    Expr::List(items) => items.iter().any(contains_pattern),
    Expr::UnaryOp { operand, .. } => contains_pattern(operand),
    _ => false,
  }
}

/// Try AST-based structural pattern matching for a single rule on an expression.
/// Returns Some(result) if the pattern matched and was replaced.
fn try_ast_pattern_replace(
  expr: &Expr,
  pattern: &Expr,
  replacement: &Expr,
  condition: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  // Check if pattern contains any Expr::Pattern nodes
  if !contains_pattern(pattern) {
    return Ok(None);
  }

  match expr {
    Expr::List(items) => {
      // Apply structural pattern matching to each list element
      let mut results = Vec::new();
      let mut any_matched = false;
      for item in items {
        if let Some(result) =
          try_ast_pattern_replace_single(item, pattern, replacement, condition)?
        {
          results.push(result);
          any_matched = true;
        } else {
          results.push(item.clone());
        }
      }
      if any_matched {
        Ok(Some(Expr::List(results)))
      } else {
        Ok(None)
      }
    }
    _ => try_ast_pattern_replace_single(expr, pattern, replacement, condition),
  }
}

/// Try to match a single expression against a structural pattern.
fn try_ast_pattern_replace_single(
  value: &Expr,
  pattern: &Expr,
  replacement: &Expr,
  condition: Option<&str>,
) -> Result<Option<Expr>, InterpreterError> {
  if let Some(bindings) = match_pattern(value, pattern) {
    // Check condition if present
    if let Some(cond_str) = condition {
      // Substitute bindings into condition and evaluate
      let mut substituted_cond = cond_str.to_string();
      for (var, val) in &bindings {
        substituted_cond =
          replace_var_with_value(&substituted_cond, var, &expr_to_string(val));
      }
      match interpret(&substituted_cond) {
        Ok(result) if result == "True" => {}
        _ => return Ok(None), // Condition not satisfied
      }
    }
    // Substitute bindings into replacement using apply_bindings
    return Ok(Some(apply_bindings(replacement, &bindings)?));
  }
  Ok(None)
}

/// Extract the pattern Expr and optional /; condition string from a rule's pattern field.
/// Handles Expr::Raw("pattern_str /; condition_str") by parsing the pattern part
/// and returning the condition string separately.
fn extract_pattern_and_condition(pattern: &Expr) -> (Expr, Option<String>) {
  match pattern {
    Expr::Raw(s) if s.contains(" /; ") => {
      // Split on " /; " to get pattern and condition
      if let Some(idx) = s.find(" /; ") {
        let pattern_str = &s[..idx];
        let condition_str = &s[idx + 4..];
        if let Ok(pattern_expr) = crate::syntax::string_to_expr(pattern_str) {
          return (pattern_expr, Some(condition_str.to_string()));
        }
      }
      (pattern.clone(), None)
    }
    _ => (pattern.clone(), None),
  }
}

/// Apply ReplaceAll operation on AST (expr /. rules)
/// Uses AST-based structural pattern matching for patterns containing blanks (n_, x_Head, etc.),
/// falls back to string-based matching for simple patterns.
fn apply_replace_all_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  // Try AST-based structural pattern matching first for single rules
  match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let (pat_expr, condition) = extract_pattern_and_condition(pattern);
      if contains_pattern(&pat_expr)
        && let Some(result) = try_ast_pattern_replace(
          expr,
          &pat_expr,
          replacement,
          condition.as_deref(),
        )?
      {
        return Ok(result);
      }
    }
    _ => {}
  }

  // Extract pattern and replacement strings for string-based matching
  let (pattern_str, replacement_str) = match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (expr_to_string(pattern), expr_to_string(replacement)),
    Expr::List(items) if !items.is_empty() => {
      // Multiple rules - try each rule in order on each subexpression,
      // using the first matching rule (Wolfram semantics: simultaneous application)
      let rule_pairs: Vec<(String, String)> = items
        .iter()
        .filter_map(|rule| match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => Some((expr_to_string(pattern), expr_to_string(replacement))),
          _ => None,
        })
        .collect();
      let expr_str = expr_to_string(expr);
      let result =
        apply_replace_all_direct_multi_rules(&expr_str, &rule_pairs)?;
      return string_to_expr(&result);
    }
    _ => return Ok(expr.clone()),
  };

  // Fall back to string-based function for simple patterns
  let expr_str = expr_to_string(expr);
  let result =
    apply_replace_all_direct(&expr_str, &pattern_str, &replacement_str)?;
  string_to_expr(&result)
}

/// Apply ReplaceRepeated operation on AST (expr //. rules)
/// Routes to string-based implementation for correct pattern matching support
fn apply_replace_repeated_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      let pattern_str = expr_to_string(pattern);
      let replacement_str = expr_to_string(replacement);
      let expr_str = expr_to_string(expr);
      let result = apply_replace_repeated_direct(
        &expr_str,
        &pattern_str,
        &replacement_str,
      )?;
      string_to_expr(&result)
    }
    Expr::List(items) if !items.is_empty() => {
      // Multiple rules - repeatedly try each rule in order, using first match
      let rule_pairs: Vec<(String, String)> = items
        .iter()
        .filter_map(|rule| match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => Some((expr_to_string(pattern), expr_to_string(replacement))),
          _ => None,
        })
        .collect();
      let mut current = expr_to_string(expr);
      let max_iterations = 1000;
      for _ in 0..max_iterations {
        let next = apply_replace_all_direct_multi_rules(&current, &rule_pairs)?;
        if next == current {
          break;
        }
        current = next;
      }
      string_to_expr(&current)
    }
    _ => Ok(expr.clone()),
  }
}

/// Check if two Expr values are structurally equal
#[allow(dead_code)]
fn expr_equal(a: &Expr, b: &Expr) -> bool {
  match (a, b) {
    (Expr::Integer(x), Expr::Integer(y)) => x == y,
    (Expr::Real(x), Expr::Real(y)) => x == y,
    (Expr::String(x), Expr::String(y)) => x == y,
    (Expr::Identifier(x), Expr::Identifier(y)) => x == y,
    (Expr::Slot(x), Expr::Slot(y)) => x == y,
    (Expr::Constant(x), Expr::Constant(y)) => x == y,
    (Expr::List(xs), Expr::List(ys)) => {
      xs.len() == ys.len()
        && xs.iter().zip(ys.iter()).all(|(x, y)| expr_equal(x, y))
    }
    (
      Expr::FunctionCall { name: n1, args: a1 },
      Expr::FunctionCall { name: n2, args: a2 },
    ) => {
      n1 == n2
        && a1.len() == a2.len()
        && a1.iter().zip(a2.iter()).all(|(x, y)| expr_equal(x, y))
    }
    _ => expr_to_string(a) == expr_to_string(b),
  }
}

/// Apply a list of rules once to an expression
#[allow(dead_code)]
fn apply_rules_once(
  expr: &Expr,
  rules: &[(&Expr, &Expr)],
) -> Result<Expr, InterpreterError> {
  // Try to match each rule against the expression
  for (pattern, replacement) in rules {
    if let Some(bindings) = match_pattern(expr, pattern) {
      return apply_bindings(replacement, &bindings);
    }
  }

  // No rule matched at the top level, try to apply rules to subexpressions
  match expr {
    Expr::List(items) => {
      let new_items: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_rules_once(item, rules))
        .collect();
      Ok(Expr::List(new_items?))
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .map(|arg| apply_rules_once(arg, rules))
        .collect();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args?,
      })
    }
    Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
      op: *op,
      left: Box::new(apply_rules_once(left, rules)?),
      right: Box::new(apply_rules_once(right, rules)?),
    }),
    _ => Ok(expr.clone()),
  }
}

/// Match a pattern against an expression, returning bindings if successful
fn match_pattern(expr: &Expr, pattern: &Expr) -> Option<Vec<(String, Expr)>> {
  match pattern {
    Expr::Pattern { name, head } => {
      // Check head constraint if present
      if let Some(h) = head {
        let expr_head = get_expr_head(expr);
        if expr_head != *h {
          return None;
        }
      }
      Some(vec![(name.clone(), expr.clone())])
    }
    Expr::Identifier(name) if name.ends_with('_') => {
      // Blank pattern like x_
      let var_name = name.trim_end_matches('_');
      Some(vec![(var_name.to_string(), expr.clone())])
    }
    Expr::Integer(n) => {
      if matches!(expr, Expr::Integer(m) if m == n) {
        Some(vec![])
      } else if let Expr::BigInteger(m) = expr {
        use num_traits::ToPrimitive;
        if m.to_i128() == Some(*n) {
          Some(vec![])
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::BigInteger(n) => {
      if let Expr::BigInteger(m) = expr {
        if m == n { Some(vec![]) } else { None }
      } else if let Expr::Integer(m) = expr {
        if num_bigint::BigInt::from(*m) == *n {
          Some(vec![])
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::Real(f) => {
      if matches!(expr, Expr::Real(g) if (f - g).abs() < f64::EPSILON) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::Identifier(name) => {
      if matches!(expr, Expr::Identifier(n) if n == name) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::String(s) => {
      if matches!(expr, Expr::String(t) if t == s) {
        Some(vec![])
      } else {
        None
      }
    }
    Expr::List(pat_items) => {
      if let Expr::List(expr_items) = expr {
        if pat_items.len() != expr_items.len() {
          return None;
        }
        let mut bindings = Vec::new();
        for (p, e) in pat_items.iter().zip(expr_items.iter()) {
          if let Some(b) = match_pattern(e, p) {
            bindings.extend(b);
          } else {
            return None;
          }
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::FunctionCall {
      name: pat_name,
      args: pat_args,
    } => {
      if let Expr::FunctionCall {
        name: expr_name,
        args: expr_args,
      } = expr
      {
        if pat_name != expr_name || pat_args.len() != expr_args.len() {
          return None;
        }
        let mut bindings = Vec::new();
        for (p, e) in pat_args.iter().zip(expr_args.iter()) {
          if let Some(b) = match_pattern(e, p) {
            bindings.extend(b);
          } else {
            return None;
          }
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: pat_op,
      left: pat_left,
      right: pat_right,
    } => {
      if let Expr::BinaryOp {
        op: expr_op,
        left: expr_left,
        right: expr_right,
      } = expr
      {
        if pat_op != expr_op {
          return None;
        }
        let mut bindings = Vec::new();
        if let Some(b) = match_pattern(expr_left, pat_left) {
          bindings.extend(b);
        } else {
          return None;
        }
        if let Some(b) = match_pattern(expr_right, pat_right) {
          bindings.extend(b);
        } else {
          return None;
        }
        Some(bindings)
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: pat_op,
      operand: pat_operand,
    } => {
      if let Expr::UnaryOp {
        op: expr_op,
        operand: expr_operand,
      } = expr
      {
        if pat_op != expr_op {
          return None;
        }
        match_pattern(expr_operand, pat_operand)
      } else {
        None
      }
    }
    _ => {
      // For other patterns, check structural equality
      if expr_equal(expr, pattern) {
        Some(vec![])
      } else {
        None
      }
    }
  }
}

/// Get the head of an expression (for pattern matching with head constraints)
fn get_expr_head(expr: &Expr) -> String {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer".to_string(),
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real".to_string(),
    Expr::String(_) => "String".to_string(),
    Expr::List(_) => "List".to_string(),
    Expr::FunctionCall { name, .. } => name.clone(),
    Expr::Association(_) => "Association".to_string(),
    _ => "Symbol".to_string(),
  }
}

/// Get the head of an expression from its string representation (for string-based pattern matching)
fn get_string_expr_head(expr: &str) -> String {
  let expr = expr.trim();
  if expr.starts_with('"') && expr.ends_with('"') {
    "String".to_string()
  } else if expr.starts_with('{') && expr.ends_with('}') {
    "List".to_string()
  } else if expr.starts_with("<|") && expr.ends_with("|>") {
    "Association".to_string()
  } else if expr.contains('[') && expr.ends_with(']') {
    // FunctionCall: extract the function name
    let bracket_pos = expr.find('[').unwrap();
    expr[..bracket_pos].to_string()
  } else if expr.contains('.') && expr.parse::<f64>().is_ok() {
    "Real".to_string()
  } else if expr.parse::<i64>().is_ok() {
    "Integer".to_string()
  } else {
    "Symbol".to_string()
  }
}

/// Apply bindings to a replacement expression
fn apply_bindings(
  replacement: &Expr,
  bindings: &[(String, Expr)],
) -> Result<Expr, InterpreterError> {
  let mut result = replacement.clone();
  for (name, value) in bindings {
    result = crate::syntax::substitute_variable(&result, name, value);
  }
  // Evaluate the result after substitution
  evaluate_expr_to_expr(&result)
}

/// Apply Map operation on AST (func /@ list)
fn apply_map_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items
        .iter()
        .map(|item| apply_function_to_arg(func, item))
        .collect();
      Ok(Expr::List(results?))
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
fn apply_apply_ast(func: &Expr, list: &Expr) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) => items.clone(),
    _ => {
      // Not a list, return unevaluated
      return Ok(Expr::Apply {
        func: Box::new(func.clone()),
        list: Box::new(list.clone()),
      });
    }
  };

  // Apply converts List[a, b, c] to func[a, b, c]
  match func {
    Expr::Identifier(func_name) => {
      evaluate_function_call_ast(func_name, &items)
    }
    Expr::FunctionCall {
      name: func_name, ..
    } => evaluate_function_call_ast(func_name, &items),
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
    Expr::NamedFunction { params, body } => {
      // Named-parameter function applied to list items
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(items.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    _ => Ok(Expr::Apply {
      func: Box::new(func.clone()),
      list: Box::new(list.clone()),
    }),
  }
}

/// Apply MapApply operation on AST (f @@@ {{a, b}, {c, d}} -> {f[a, b], f[c, d]})
fn apply_map_apply_ast(
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

  Ok(Expr::List(results?))
}

/// Apply Postfix operation on AST (expr // func)
fn apply_postfix_ast(
  expr: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  apply_function_to_arg(func, expr)
}

/// Get the innermost base of nested Part expressions
fn get_part_base(expr: &Expr) -> &Expr {
  match expr {
    Expr::Part { expr: inner, .. } => get_part_base(inner),
    _ => expr,
  }
}

/// Apply a chain of Part indices to an expression, handling All by mapping over elements
fn apply_part_indices(
  expr: &Expr,
  indices: &[Expr],
) -> Result<Expr, InterpreterError> {
  if indices.is_empty() {
    return Ok(expr.clone());
  }
  let idx = &indices[0];
  let rest = &indices[1..];
  if matches!(idx, Expr::Identifier(s) if s == "All") {
    // All: map remaining indices over each element
    if rest.is_empty() {
      // Part[expr, All] with no more indices — return as-is
      return Ok(expr.clone());
    }
    match expr {
      Expr::List(items) => {
        let mapped: Result<Vec<Expr>, InterpreterError> = items
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::List(mapped?))
      }
      Expr::FunctionCall { name, args } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: name.clone(),
          args: mapped?,
        })
      }
      _ => apply_part_indices(expr, rest),
    }
  } else {
    // Normal index: extract, then continue with remaining indices
    let extracted = extract_part_ast(expr, idx)?;
    apply_part_indices(&extracted, rest)
  }
}

/// Evaluate the base expression of a Part, with optimization for identifiers in ENV
fn eval_part_base(e: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::Identifier(var_name) = e {
    let env_result = ENV.with(|env| {
      let env = env.borrow();
      if let Some(StoredValue::ExprVal(stored)) = env.get(var_name) {
        Some(Ok(stored.clone()))
      } else {
        None
      }
    });
    if let Some(r) = env_result {
      return r;
    }
  }
  evaluate_expr_to_expr(e)
}

/// Extract part from expression on AST (expr[[index]])
fn extract_part_ast(
  expr: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  // For associations, handle key-based lookup (can be any Expr)
  if let Expr::Association(items) = expr {
    // Try to find a matching key
    let index_str = crate::syntax::expr_to_string(index);
    for (key, val) in items {
      let key_str = crate::syntax::expr_to_string(key);
      // Compare keys (strip quotes from strings for comparison)
      let key_cmp = key_str.trim_matches('"');
      let index_cmp = index_str.trim_matches('"');
      if key_cmp == index_cmp {
        return Ok(val.clone());
      }
    }
    // Key not found - return unevaluated Part
    return Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    });
  }

  let idx = match index {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i64() {
        Some(v) => v,
        None => {
          return Ok(Expr::Part {
            expr: Box::new(expr.clone()),
            index: Box::new(index.clone()),
          });
        }
      }
    }
    Expr::Real(f) => *f as i64,
    _ => {
      return Ok(Expr::Part {
        expr: Box::new(expr.clone()),
        index: Box::new(index.clone()),
      });
    }
  };

  match expr {
    Expr::List(items) => {
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(items[actual_idx as usize].clone())
      } else {
        // Print warning to stderr and return unevaluated Part expression
        let expr_str = crate::syntax::expr_to_string(expr);
        eprintln!();
        eprintln!("Part::partw: Part {} of {} does not exist.", idx, expr_str);
        Ok(Expr::Part {
          expr: Box::new(expr.clone()),
          index: Box::new(index.clone()),
        })
      }
    }
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      let len = chars.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(Expr::String(chars[actual_idx as usize].to_string()))
      } else {
        // Print warning to stderr and return unevaluated Part expression
        eprintln!();
        eprintln!("Part::partw: Part {} of \"{}\" does not exist.", idx, s);
        Ok(Expr::Part {
          expr: Box::new(expr.clone()),
          index: Box::new(index.clone()),
        })
      }
    }
    _ => Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    }),
  }
}

/// Apply a function to an argument (helper for Map, Postfix, etc.)
pub fn apply_function_to_arg(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      // Simple function name: f applied to arg
      evaluate_function_call_ast(name, &[arg.clone()])
    }
    Expr::Function { body } => {
      // Anonymous function: substitute # with arg and evaluate
      let substituted = crate::syntax::substitute_slots(body, &[arg.clone()]);
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      // Named-parameter function: substitute params with arg
      let mut substituted = (**body).clone();
      if let Some(param) = params.first() {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
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
          | "StringMatchQ"
          | "MemberQ"
          | "Select"
      ) && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![arg.clone(), args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else {
        let mut new_args = args.clone();
        new_args.push(arg.clone());
        evaluate_function_call_ast(name, &new_args)
      }
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

/// Apply a curried call: f[a][b, c] applies function result f[a] to args [b, c]
fn apply_curried_call(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      // Simple function name applied to args
      evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      // Anonymous function: substitute # with args and evaluate
      let substituted = crate::syntax::substitute_slots(body, args);
      evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body } => {
      // Named-parameter function: substitute each param with corresponding arg
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
      evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall {
      name,
      args: func_args,
    } => {
      // Curried function: f[a][b] becomes f[a, b]
      // Special case: operator forms where f[x][y] becomes f[y, x]
      if matches!(
        name.as_str(),
        "ReplaceAll"
          | "ReplaceRepeated"
          | "StringStartsQ"
          | "StringEndsQ"
          | "StringContainsQ"
          | "StringMatchQ"
          | "MemberQ"
          | "Select"
      ) && func_args.len() == 1
        && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![args[0].clone(), func_args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
      } else if name == "Composition" && !func_args.is_empty() {
        // Composition[f, g, h][x] applies functions right-to-left: f[g[h[x]]]
        let mut result = args.to_vec();
        for f in func_args.iter().rev() {
          let intermediate = apply_curried_call(f, &result)?;
          result = vec![intermediate];
        }
        Ok(result.into_iter().next().unwrap())
      } else {
        // Standard curried call: append args
        let mut new_args = func_args.clone();
        new_args.extend(args.iter().cloned());
        evaluate_function_call_ast(name, &new_args)
      }
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

/// Replace pattern with replacement in an expression string
/// This handles symbolic replacement (e.g., replacing 'a' with 'x' in '{a, b}')
fn replace_in_expr(expr: &str, pattern: &str, replacement: &str) -> String {
  // For function call patterns like f[2], we need more sophisticated matching
  if pattern.contains('[') && pattern.contains(']') {
    // Function call pattern - do literal replacement
    expr.replace(pattern, replacement)
  } else {
    // Symbol replacement - need to be careful not to replace partial matches
    // Use word boundary-aware replacement
    let mut result = String::new();
    let mut chars = expr.chars().peekable();
    let pattern_chars: Vec<char> = pattern.chars().collect();

    while let Some(c) = chars.next() {
      // Check if we're at the start of a potential match
      let mut potential_match = vec![c];
      let mut iter_clone = chars.clone();

      // Build up potential match
      while potential_match.len() < pattern_chars.len() {
        if let Some(nc) = iter_clone.next() {
          potential_match.push(nc);
        } else {
          break;
        }
      }

      if potential_match == pattern_chars {
        // Check if this is a whole word match (not part of identifier)
        let prev_char = result.chars().last();
        let next_char = iter_clone.next();

        let prev_ok =
          prev_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let next_ok =
          next_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        if prev_ok && next_ok {
          // It's a whole word match, do the replacement
          result.push_str(replacement);
          // Consume the matched characters
          for _ in 1..pattern_chars.len() {
            chars.next();
          }
          continue;
        }
      }

      result.push(c);
    }

    result
  }
}

/// Check if a pattern is a Wolfram blank pattern (x_, x_?test, x_ /; cond)
fn parse_wolfram_pattern(pattern: &str) -> Option<WolframPattern> {
  let pattern = pattern.trim();

  // Check for conditional pattern: x_ /; condition
  if let Some(cond_idx) = pattern.find(" /; ") {
    let before_cond = &pattern[..cond_idx];
    let condition = &pattern[cond_idx + 4..];

    // The part before /; should be a simple blank pattern (x_)
    if let Some(underscore_idx) = before_cond.find('_') {
      let var_name = before_cond[..underscore_idx].trim().to_string();
      if !var_name.is_empty()
        && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
      {
        return Some(WolframPattern::Conditional {
          var_name,
          condition: condition.trim().to_string(),
        });
      }
    }
  }

  // Check for pattern test: x_?test
  if let Some(underscore_idx) = pattern.find('_') {
    let after_underscore = &pattern[underscore_idx + 1..];
    if after_underscore.starts_with('?') {
      let var_name = pattern[..underscore_idx].trim().to_string();
      let test_func = after_underscore[1..].trim().to_string();
      if !var_name.is_empty()
        && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
        && !test_func.is_empty()
      {
        return Some(WolframPattern::Test {
          var_name,
          test_func,
        });
      }
    }
  }

  // Check for head blank pattern: x_Head (e.g. x_Integer, x_String)
  // or simple blank pattern: x_
  if let Some(underscore_idx) = pattern.find('_') {
    let var_name = pattern[..underscore_idx].trim().to_string();
    let after_underscore = &pattern[underscore_idx + 1..];
    if !var_name.is_empty()
      && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
      && !pattern.contains(' ')
    {
      if after_underscore.is_empty() {
        return Some(WolframPattern::Blank { var_name });
      } else if after_underscore
        .chars()
        .all(|c| c.is_alphanumeric() || c == '$')
      {
        return Some(WolframPattern::HeadBlank {
          var_name,
          head: after_underscore.to_string(),
        });
      }
    }
  }

  None
}

/// Wolfram pattern types
enum WolframPattern {
  /// x_ - matches any single expression
  Blank { var_name: String },
  /// x_Head - matches if Head[x] == Head (e.g. x_Integer, x_String)
  HeadBlank { var_name: String, head: String },
  /// x_?test - matches if test[x] is True
  Test { var_name: String, test_func: String },
  /// x_ /; condition - matches if condition (with x substituted) is True
  Conditional { var_name: String, condition: String },
}

/// Replace a variable name with a value, respecting word boundaries
/// This avoids replacing 'i' inside strings like "Fizz"
fn replace_var_with_value(text: &str, var_name: &str, value: &str) -> String {
  let mut result = String::new();
  let var_chars: Vec<char> = var_name.chars().collect();
  let chars: Vec<char> = text.chars().collect();
  let mut i = 0;

  while i < chars.len() {
    // Check if we're inside a string literal
    if chars[i] == '"' {
      result.push(chars[i]);
      i += 1;
      // Copy everything until the closing quote
      while i < chars.len() && chars[i] != '"' {
        result.push(chars[i]);
        i += 1;
      }
      if i < chars.len() {
        result.push(chars[i]); // closing quote
        i += 1;
      }
      continue;
    }

    // Check if we're at the start of the variable name
    if i + var_chars.len() <= chars.len() {
      let potential_match: Vec<char> = chars[i..i + var_chars.len()].to_vec();
      if potential_match == var_chars {
        // Check word boundaries
        let prev_ok = i == 0 || {
          let prev = chars[i - 1];
          !prev.is_alphanumeric() && prev != '_' && prev != '$'
        };
        let next_ok = i + var_chars.len() >= chars.len() || {
          let next = chars[i + var_chars.len()];
          !next.is_alphanumeric() && next != '_' && next != '$'
        };

        if prev_ok && next_ok {
          result.push_str(value);
          i += var_chars.len();
          continue;
        }
      }
    }

    result.push(chars[i]);
    i += 1;
  }

  result
}

/// Apply a Wolfram pattern replacement to an expression
fn apply_wolfram_pattern(
  expr: &str,
  pattern: &WolframPattern,
  replacement: &str,
) -> Result<Option<String>, InterpreterError> {
  match pattern {
    WolframPattern::Blank { var_name } => {
      // x_ matches any expression - substitute var_name with expr in replacement
      let result = replace_var_with_value(replacement, var_name, expr);
      Ok(Some(result))
    }
    WolframPattern::HeadBlank { var_name, head } => {
      // x_Head matches if Head[expr] == head
      let expr_head = get_string_expr_head(expr);
      if expr_head == *head {
        let result = replace_var_with_value(replacement, var_name, expr);
        Ok(Some(result))
      } else {
        Ok(None)
      }
    }
    WolframPattern::Test {
      var_name,
      test_func,
    } => {
      // x_?test - check if test[expr] is True
      let test_expr = format!("{}[{}]", test_func, expr);
      match interpret(&test_expr) {
        Ok(result) if result == "True" => {
          let replaced = replace_var_with_value(replacement, var_name, expr);
          Ok(Some(replaced))
        }
        _ => Ok(None), // Test failed, no match
      }
    }
    WolframPattern::Conditional {
      var_name,
      condition,
    } => {
      // x_ /; condition - check if condition (with var substituted) is True
      let substituted_condition =
        replace_var_with_value(condition, var_name, expr);
      match interpret(&substituted_condition) {
        Ok(result) if result == "True" => {
          let replaced = replace_var_with_value(replacement, var_name, expr);
          Ok(Some(replaced))
        }
        _ => Ok(None), // Condition failed, no match
      }
    }
  }
}

/// Evaluate a FullForm string expression and return the result in FullForm.
/// Unlike `interpret()`, this preserves string quotes so that round-tripping
/// through string_to_expr works correctly.
fn evaluate_fullform(s: &str) -> Result<String, InterpreterError> {
  let expr = crate::syntax::string_to_expr(s)?;
  let result = evaluate_expr_to_expr(&expr)?;
  Ok(expr_to_string(&result))
}

/// Apply ReplaceAll with direct pattern and replacement strings
fn apply_replace_all_direct(
  expr: &str,
  pattern: &str,
  replacement: &str,
) -> Result<String, InterpreterError> {
  // Check if the pattern is a Wolfram pattern (x_, x_?test, x_ /; cond)
  if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern) {
    // For list expressions, apply pattern to each element
    if expr.starts_with('{') && expr.ends_with('}') {
      let inner = &expr[1..expr.len() - 1];
      // Split by comma, being careful about nested structures
      let elements = split_list_elements(inner);
      let mut results = Vec::new();

      for elem in elements {
        let elem = elem.trim();
        if let Some(replaced) =
          apply_wolfram_pattern(elem, &wolfram_pattern, replacement)?
        {
          // Try to evaluate the replacement (preserve FullForm for round-tripping)
          let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
          results.push(evaluated);
        } else {
          // No match, keep original
          results.push(elem.to_string());
        }
      }

      return Ok(format!("{{{}}}", results.join(", ")));
    } else {
      // Single expression - apply pattern directly
      if let Some(replaced) =
        apply_wolfram_pattern(expr, &wolfram_pattern, replacement)?
      {
        let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
        return Ok(evaluated);
      }
      return Ok(expr.to_string());
    }
  }

  // Fall back to literal string replacement for non-pattern cases
  let result = replace_in_expr(expr, pattern, replacement);

  // Re-evaluate the result to simplify if possible
  if result != *expr {
    // Try to evaluate the result, but if it fails, return as-is
    evaluate_fullform(&result).or(Ok(result))
  } else {
    Ok(result)
  }
}

/// Apply ReplaceAll with multiple rules simultaneously.
/// For each subexpression, try each rule in order and use the first match.
fn apply_replace_all_direct_multi_rules(
  expr: &str,
  rules: &[(String, String)],
) -> Result<String, InterpreterError> {
  // For list expressions, apply rules to each element
  if expr.starts_with('{') && expr.ends_with('}') {
    let inner = &expr[1..expr.len() - 1];
    let elements = split_list_elements(inner);
    let mut results = Vec::new();

    for elem in elements {
      let elem = elem.trim();
      let replaced = apply_first_matching_rule(elem, rules)?;
      results.push(replaced);
    }

    return Ok(format!("{{{}}}", results.join(", ")));
  }

  // For non-list expressions, first try matching the whole expression
  // against each rule (needed for Wolfram patterns like i_ /; cond)
  for (pattern, replacement) in rules {
    if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern)
      && let Some(replaced) =
        apply_wolfram_pattern(expr, &wolfram_pattern, replacement)?
    {
      let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
      return Ok(evaluated);
    }
  }

  // Then apply literal symbol rules simultaneously in one pass
  let result = replace_in_expr_multi_rules(expr, rules);

  if result != *expr {
    evaluate_fullform(&result).or(Ok(result))
  } else {
    Ok(result)
  }
}

/// Try each rule in order on a single expression, return the result of
/// the first matching rule. If no rule matches, return the original.
fn apply_first_matching_rule(
  elem: &str,
  rules: &[(String, String)],
) -> Result<String, InterpreterError> {
  for (pattern, replacement) in rules {
    // Check if this is a Wolfram pattern
    if let Some(wolfram_pattern) = parse_wolfram_pattern(pattern) {
      if let Some(replaced) =
        apply_wolfram_pattern(elem, &wolfram_pattern, replacement)?
      {
        let evaluated = evaluate_fullform(&replaced).unwrap_or(replaced);
        return Ok(evaluated);
      }
    } else {
      // Literal replacement - check if it matches
      let result = replace_in_expr(elem, pattern, replacement);
      if result != elem {
        let evaluated = evaluate_fullform(&result).unwrap_or(result);
        return Ok(evaluated);
      }
    }
  }
  // No rule matched
  Ok(elem.to_string())
}

/// Replace multiple patterns simultaneously in an expression, respecting word boundaries.
/// At each position, try each rule in order and use the first match.
fn replace_in_expr_multi_rules(
  expr: &str,
  rules: &[(String, String)],
) -> String {
  // Separate function-call patterns from symbol patterns
  let mut func_rules: Vec<(&str, &str)> = Vec::new();
  let mut symbol_rules: Vec<(&str, &str)> = Vec::new();

  for (pattern, replacement) in rules {
    if pattern.contains('[') && pattern.contains(']') {
      func_rules.push((pattern, replacement));
    } else {
      symbol_rules.push((pattern, replacement));
    }
  }

  // First apply function-call pattern rules (literal string replacement, first match wins)
  let mut current = expr.to_string();
  for (pattern, replacement) in &func_rules {
    let next = current.replace(pattern, replacement);
    if next != current {
      current = next;
      break; // First matching rule wins
    }
  }

  // Then apply symbol rules simultaneously in one pass
  if symbol_rules.is_empty() {
    return current;
  }

  let mut result = String::new();
  let chars: Vec<char> = current.chars().collect();
  let mut i = 0;

  while i < chars.len() {
    let mut matched = false;

    // Try each symbol rule at this position (first match wins)
    for (pattern, replacement) in &symbol_rules {
      let pat_chars: Vec<char> = pattern.chars().collect();
      if i + pat_chars.len() <= chars.len()
        && chars[i..i + pat_chars.len()] == pat_chars[..]
      {
        // Check word boundaries
        let prev_char = if i > 0 { Some(chars[i - 1]) } else { None };
        let next_char = if i + pat_chars.len() < chars.len() {
          Some(chars[i + pat_chars.len()])
        } else {
          None
        };

        let prev_ok =
          prev_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');
        let next_ok =
          next_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        if prev_ok && next_ok {
          result.push_str(replacement);
          i += pat_chars.len();
          matched = true;
          break;
        }
      }
    }

    if !matched {
      result.push(chars[i]);
      i += 1;
    }
  }

  result
}

/// Split a list's inner content by commas, respecting nested structures
fn split_list_elements(inner: &str) -> Vec<String> {
  let mut elements = Vec::new();
  let mut current = String::new();
  let mut depth = 0;

  for c in inner.chars() {
    match c {
      '{' | '[' | '(' | '<' => {
        depth += 1;
        current.push(c);
      }
      '}' | ']' | ')' | '>' => {
        depth -= 1;
        current.push(c);
      }
      ',' if depth == 0 => {
        elements.push(current.trim().to_string());
        current.clear();
      }
      _ => {
        current.push(c);
      }
    }
  }

  if !current.is_empty() {
    elements.push(current.trim().to_string());
  }

  elements
}

/// Apply ReplaceRepeated with direct pattern and replacement strings
fn apply_replace_repeated_direct(
  expr: &str,
  pattern: &str,
  replacement: &str,
) -> Result<String, InterpreterError> {
  let mut current = expr.to_string();
  let max_iterations = 1000; // Prevent infinite loops

  for _ in 0..max_iterations {
    let next = replace_in_expr(&current, pattern, replacement);

    if next == current {
      // No more changes, we're done
      break;
    }

    // Re-evaluate to simplify (preserve FullForm for round-tripping)
    current = evaluate_fullform(&next).unwrap_or(next);
  }

  Ok(current)
}
