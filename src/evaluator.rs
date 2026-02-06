use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, expr_to_string,
};
use crate::{ENV, InterpreterError, StoredValue, format_result, interpret};

/// Evaluate an Expr AST directly without re-parsing.
/// This is the core optimization that avoids re-parsing function bodies.
pub fn evaluate_expr(expr: &Expr) -> Result<String, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(n.to_string()),
    Expr::Real(f) => Ok(format_result(*f)),
    Expr::String(s) => Ok(format!("\"{}\"", s)),
    Expr::Identifier(name) => {
      // Look up in environment
      if let Some(StoredValue::Raw(val)) =
        ENV.with(|e| e.borrow().get(name).cloned())
      {
        Ok(val)
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
    Expr::Constant(name) => match name.as_str() {
      "Pi" | "-Pi" => {
        if name.starts_with('-') {
          Ok(format_result(-std::f64::consts::PI))
        } else {
          Ok(format_result(std::f64::consts::PI))
        }
      }
      _ => Ok(name.clone()),
    },
    Expr::List(items) => {
      let evaluated: Result<Vec<String>, _> =
        items.iter().map(evaluate_expr).collect();
      Ok(format!("{{{}}}", evaluated?.join(", ")))
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

      match op {
        BinaryOperator::Plus => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(format_result(l + r))
          } else {
            // Symbolic
            Ok(format!("{} + {}", left_val, right_val))
          }
        }
        BinaryOperator::Minus => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(format_result(l - r))
          } else {
            Ok(format!("{} - {}", left_val, right_val))
          }
        }
        BinaryOperator::Times => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(format_result(l * r))
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
              Ok(format_result(l / r))
            }
          } else {
            Ok(format!("{} / {}", left_val, right_val))
          }
        }
        BinaryOperator::Power => {
          if let (Ok(l), Ok(r)) =
            (left_val.parse::<f64>(), right_val.parse::<f64>())
          {
            Ok(format_result(l.powf(r)))
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
      let p = expr_to_string(pattern);
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
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        Ok(format!("{}_{}", name, h))
      } else {
        Ok(format!("{}_", name))
      }
    }
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
    Expr::Real(f) => Ok(Expr::Real(*f)),
    Expr::String(s) => Ok(Expr::String(s.clone())),
    Expr::Identifier(name) => {
      // Look up in environment
      if let Some(stored) = ENV.with(|e| e.borrow().get(name).cloned()) {
        match stored {
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
    Expr::Constant(name) => match name.as_str() {
      "Pi" => Ok(Expr::Real(std::f64::consts::PI)),
      "-Pi" => Ok(Expr::Real(-std::f64::consts::PI)),
      "E" => Ok(Expr::Real(std::f64::consts::E)),
      _ => Ok(Expr::Constant(name.clone())),
    },
    Expr::List(items) => {
      let evaluated: Result<Vec<Expr>, _> =
        items.iter().map(evaluate_expr_to_expr).collect();
      Ok(Expr::List(evaluated?))
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
      // Special handling for Module - don't evaluate args (body needs local bindings first)
      if name == "Module" {
        return module_ast(args);
      }
      // Special handling for Set - first arg must be identifier or Part, second gets evaluated
      if name == "Set" && args.len() == 2 {
        return set_ast(&args[0], &args[1]);
      }
      // Special handling for Table, Do, With - don't evaluate args (body needs iteration/bindings)
      // These functions take unevaluated expressions as first argument
      if name == "Table" || name == "Do" || name == "With" {
        // Pass unevaluated args to the function dispatcher
        return evaluate_function_call_ast(name, args);
      }
      // Check if name is a variable holding an association (for nested access: assoc["a", "b"])
      let assoc_val = ENV.with(|e| e.borrow().get(name).cloned());
      if let Some(StoredValue::Association(_)) = assoc_val {
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
          if let (Some(l), Some(r)) = (left_num, right_num) {
            Ok(num_to_expr(l - r))
          } else {
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
          }
        }
        BinaryOperator::Times => {
          if let (Some(l), Some(r)) = (left_num, right_num) {
            Ok(num_to_expr(l * r))
          } else {
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
          }
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
      for i in 0..operators.len() {
        let left = &values[i];
        let right = &values[i + 1];
        let op = &operators[i];

        let result = match op {
          ComparisonOp::Equal | ComparisonOp::SameQ => {
            if let (Some(l), Some(r)) =
              (expr_to_number(left), expr_to_number(right))
            {
              l == r
            } else {
              expr_to_string(left) == expr_to_string(right)
            }
          }
          ComparisonOp::NotEqual | ComparisonOp::UnsameQ => {
            if let (Some(l), Some(r)) =
              (expr_to_number(left), expr_to_number(right))
            {
              l != r
            } else {
              expr_to_string(left) != expr_to_string(right)
            }
          }
          ComparisonOp::Less => {
            if let (Some(l), Some(r)) =
              (expr_to_number(left), expr_to_number(right))
            {
              l < r
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
              (expr_to_number(left), expr_to_number(right))
            {
              l <= r
            } else {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
          ComparisonOp::Greater => {
            if let (Some(l), Some(r)) =
              (expr_to_number(left), expr_to_number(right))
            {
              l > r
            } else {
              return Ok(Expr::Comparison {
                operands: values,
                operators: operators.clone(),
              });
            }
          }
          ComparisonOp::GreaterEqual => {
            if let (Some(l), Some(r)) =
              (expr_to_number(left), expr_to_number(right))
            {
              l >= r
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
      let r = evaluate_expr_to_expr(replacement)?;
      Ok(Expr::Rule {
        pattern: pattern.clone(),
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
      let evaluated_expr = evaluate_expr_to_expr(e)?;
      let evaluated_idx = evaluate_expr_to_expr(index)?;
      extract_part_ast(&evaluated_expr, &evaluated_idx)
    }
    Expr::Function { body } => {
      // Return anonymous function as-is
      Ok(Expr::Function { body: body.clone() })
    }
    Expr::Pattern { name, head } => Ok(Expr::Pattern {
      name: name.clone(),
      head: head.clone(),
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

/// Convert an Expr to a number if possible
fn expr_to_number(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// Convert a number to an appropriate Expr (Integer if whole, Real otherwise)
fn num_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
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
    let ln = expr_to_number(l);
    let rn = expr_to_number(r);
    match op {
      BinaryOperator::Plus => {
        if let (Some(a), Some(b)) = (ln, rn) {
          Ok(num_to_expr(a + b))
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
          Ok(num_to_expr(a - b))
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
          Ok(num_to_expr(a * b))
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
          Ok(num_to_expr(a.powf(b)))
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
pub fn evaluate_function_call_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  use crate::functions::list_helpers_ast;

  // Handle functions that would call interpret() if dispatched through evaluate_expression
  // These must be handled natively to avoid infinite recursion
  match name {
    "Module" => return module_ast(args),
    "Set" if args.len() == 2 => {
      return set_ast(&args[0], &args[1]);
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
    "Select" if args.len() == 2 => {
      return list_helpers_ast::select_ast(&args[0], &args[1]);
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
    "Nest" if args.len() == 3 => {
      if let Expr::Integer(n) = &args[2] {
        return list_helpers_ast::nest_ast(&args[0], &args[1], *n);
      }
    }
    "NestList" if args.len() == 3 => {
      if let Expr::Integer(n) = &args[2] {
        return list_helpers_ast::nest_list_ast(&args[0], &args[1], *n);
      }
    }
    "FixedPoint" if args.len() >= 2 => {
      let max_iter = if args.len() == 3 {
        if let Expr::Integer(n) = &args[2] {
          Some(*n)
        } else {
          None
        }
      } else {
        None
      };
      return list_helpers_ast::fixed_point_ast(&args[0], &args[1], max_iter);
    }
    "Cases" if args.len() == 2 => {
      return list_helpers_ast::cases_ast(&args[0], &args[1]);
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

    // Additional AST-native list functions
    "Table" if args.len() == 2 => {
      return list_helpers_ast::table_ast(&args[0], &args[1]);
    }
    "MapThread" if args.len() == 2 => {
      return list_helpers_ast::map_thread_ast(&args[0], &args[1]);
    }
    "Partition" if args.len() == 2 => {
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::partition_ast(&args[0], *n);
      }
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
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::flatten_level_ast(&args[0], *n);
      }
    }
    "Reverse" if args.len() == 1 => {
      return list_helpers_ast::reverse_ast(&args[0]);
    }
    "Sort" if args.len() == 1 => {
      return list_helpers_ast::sort_ast(&args[0]);
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
        if let Expr::Integer(n) = &args[2] {
          Some(*n)
        } else {
          None
        }
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
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::rotate_left_ast(&args[0], *n);
      }
    }
    "RotateLeft" if args.len() == 1 => {
      return list_helpers_ast::rotate_left_ast(&args[0], 1);
    }
    "RotateRight" if args.len() == 2 => {
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::rotate_right_ast(&args[0], *n);
      }
    }
    "RotateRight" if args.len() == 1 => {
      return list_helpers_ast::rotate_right_ast(&args[0], 1);
    }
    "PadLeft" if args.len() >= 2 => {
      if let Expr::Integer(n) = &args[1] {
        let pad = if args.len() == 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        return list_helpers_ast::pad_left_ast(&args[0], *n, &pad);
      }
    }
    "PadRight" if args.len() >= 2 => {
      if let Expr::Integer(n) = &args[1] {
        let pad = if args.len() == 3 {
          args[2].clone()
        } else {
          Expr::Integer(0)
        };
        return list_helpers_ast::pad_right_ast(&args[0], *n, &pad);
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
        if let Expr::Integer(n) = &args[3] {
          Some(*n)
        } else {
          None
        }
      } else {
        None
      };
      return list_helpers_ast::nest_while_ast(
        &args[0], &args[1], &args[2], max_iter,
      );
    }
    "NestWhileList" if args.len() >= 3 => {
      let max_iter = if args.len() == 4 {
        if let Expr::Integer(n) = &args[3] {
          Some(*n)
        } else {
          None
        }
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
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::take_largest_ast(&args[0], *n);
      }
    }
    "TakeSmallest" if args.len() == 2 => {
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::take_smallest_ast(&args[0], *n);
      }
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
    "DeleteCases" if args.len() == 2 => {
      return list_helpers_ast::delete_cases_ast(&args[0], &args[1]);
    }
    "DeleteCases" if args.len() == 3 || args.len() == 4 => {
      // DeleteCases[list, pattern, levelspec] or DeleteCases[list, pattern, levelspec, n]
      // For now, levelspec is ignored (treated as level 1)
      let max_count = if args.len() == 4 {
        if let Expr::Integer(n) = &args[3] {
          Some(*n)
        } else {
          None
        }
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
      if let Expr::Integer(n) = &args[1] {
        return list_helpers_ast::array_ast(&args[0], *n);
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
    "StringSplit" if args.len() == 2 => {
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
    "ToString" if args.len() == 1 => {
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
    "StringCount" if args.len() == 2 => {
      return crate::functions::string_ast::string_count_ast(args);
    }
    "StringFreeQ" if args.len() == 2 => {
      return crate::functions::string_ast::string_free_q_ast(args);
    }

    // AST-native file and date functions
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
    "Print" => {
      // 0 args → just output a newline and return Null
      if args.is_empty() {
        println!();
        crate::capture_stdout("");
        return Ok(Expr::Identifier("Null".to_string()));
      }
      // 1 arg accepted
      if args.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Print expects at most 1 argument".into(),
        ));
      }
      // Format and print the argument
      let display_str = match &args[0] {
        Expr::String(s) => s.clone(),
        other => expr_to_string(other),
      };
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

    "MemberQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::member_q_ast(args);
    }
    "FreeQ" if args.len() == 2 => {
      return crate::functions::predicate_ast::free_q_ast(args);
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
    "Floor" if args.len() == 1 => {
      return crate::functions::math_ast::floor_ast(args);
    }
    "Ceiling" if args.len() == 1 => {
      return crate::functions::math_ast::ceiling_ast(args);
    }
    "Round" if args.len() == 1 => {
      return crate::functions::math_ast::round_ast(args);
    }
    "Mod" if args.len() == 2 => {
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
    "Total" if args.len() == 1 => {
      return crate::functions::math_ast::total_ast(args);
    }
    "Mean" if args.len() == 1 => {
      return crate::functions::math_ast::mean_ast(args);
    }
    "Factorial" if args.len() == 1 => {
      return crate::functions::math_ast::factorial_ast(args);
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
    "Sin" if args.len() == 1 => {
      return crate::functions::math_ast::sin_ast(args);
    }
    "Cos" if args.len() == 1 => {
      return crate::functions::math_ast::cos_ast(args);
    }
    "Tan" if args.len() == 1 => {
      return crate::functions::math_ast::tan_ast(args);
    }
    "Exp" if args.len() == 1 => {
      return crate::functions::math_ast::exp_ast(args);
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
    "Prime" if args.len() == 1 => {
      return crate::functions::math_ast::prime_ast(args);
    }
    "IntegerDigits" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::integer_digits_ast(args);
    }
    "FromDigits" if !args.is_empty() && args.len() <= 2 => {
      return crate::functions::math_ast::from_digits_ast(args);
    }
    "FactorInteger" if args.len() == 1 => {
      return crate::functions::math_ast::factor_integer_ast(args);
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
    "SameQ" if args.len() >= 2 => {
      return crate::functions::boolean_ast::same_q_ast(args);
    }
    "UnsameQ" if args.len() >= 2 => {
      return crate::functions::boolean_ast::unsame_q_ast(args);
    }
    "Which" if args.len() >= 2 && args.len().is_multiple_of(2) => {
      return crate::functions::boolean_ast::which_ast(args);
    }
    "While" if args.len() == 2 => {
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

    // Use AST-native calculus functions
    "D" if args.len() == 2 => {
      return crate::functions::calculus_ast::d_ast(args);
    }
    "Integrate" if args.len() == 2 => {
      return crate::functions::calculus_ast::integrate_ast(args);
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

    _ => {}
  }

  // Check for user-defined functions
  let user_func_result = crate::FUNC_DEFS.with(|m| {
    let defs = m.borrow();
    if let Some(overloads) = defs.get(name) {
      // Find matching arity
      for (params, body_expr) in overloads {
        if params.len() == args.len() {
          // Found a match - substitute parameters with arguments
          let mut substituted = body_expr.clone();
          for (param, arg) in params.iter().zip(args.iter()) {
            substituted =
              crate::syntax::substitute_variable(&substituted, param, arg);
          }
          return Some(substituted);
        }
      }
    }
    None
  });

  if let Some(body) = user_func_result {
    return evaluate_expr_to_expr(&body);
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

  // Evaluate the body expression using AST evaluation
  let result = evaluate_expr_to_expr(body_expr)?;

  // Restore previous bindings
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

  Ok(result)
}

/// AST-based Set implementation to handle Part assignment on associations
fn set_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  // Handle Part assignment: myHash[["key"]] = value
  if let Expr::Part { expr, index } = lhs
    && let Expr::Identifier(var_name) = expr.as_ref()
  {
    // Evaluate the RHS
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Get the key - extract string from String or use identifier name
    let key = match index.as_ref() {
      Expr::String(s) => s.clone(),
      Expr::Identifier(s) => s.clone(),
      other => expr_to_string(other),
    };

    // Update or add the key in the association
    ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(StoredValue::Association(pairs)) = env.get_mut(var_name) {
        // Update existing key or add new key
        if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == &key) {
          pair.1 = expr_to_string(&rhs_value);
        } else {
          pairs.push((key, expr_to_string(&rhs_value)));
        }
        Ok(())
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "{} is not an association",
          var_name
        )))
      }
    })?;

    // Return the updated association as Expr
    return ENV.with(|e| {
      let env = e.borrow();
      if let Some(StoredValue::Association(pairs)) = env.get(var_name) {
        let items: Vec<(Expr, Expr)> = pairs
          .iter()
          .map(|(k, v)| {
            let key_expr = Expr::Identifier(k.clone());
            let val_expr = string_to_expr(v).unwrap_or(Expr::Raw(v.clone()));
            (key_expr, val_expr)
          })
          .collect();
        Ok(Expr::Association(items))
      } else {
        Err(InterpreterError::EvaluationError(
          "Variable not found".into(),
        ))
      }
    });
  }

  // Handle simple identifier assignment: x = value
  if let Expr::Identifier(var_name) = lhs {
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check if RHS is an association
    if let Expr::Association(items) = &rhs_value {
      let pairs: Vec<(String, String)> = items
        .iter()
        .map(|(k, v)| {
          // Extract key without quotes for consistent storage
          let key_str = match k {
            Expr::String(s) => s.clone(),
            Expr::Identifier(s) => s.clone(),
            other => expr_to_string(other),
          };
          (key_str, expr_to_string(v))
        })
        .collect();
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Association(pairs))
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

  Err(InterpreterError::EvaluationError(
    "First argument of Set must be an identifier or part extract".into(),
  ))
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
              Expr::Identifier(k.clone()),
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
        let key_str = match key {
          Expr::String(s) => s.clone(),
          Expr::Identifier(s) => s.clone(),
          other => expr_to_string(other),
        };

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
      let key_raw = item[..arrow_pos].trim();
      // Strip quotes from key if present
      let key = key_raw.trim_matches('"').to_string();
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

/// Apply ReplaceAll operation on AST (expr /. rules)
/// Routes to string-based implementation for correct pattern matching support
fn apply_replace_all_ast(
  expr: &Expr,
  rules: &Expr,
) -> Result<Expr, InterpreterError> {
  // Extract pattern and replacement from rules
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
      // Multiple rules - apply each rule in order to the expression
      // Each rule is applied to the result of the previous rule
      let mut current = expr_to_string(expr);
      for rule in items {
        let (pat, repl) = match rule {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => (expr_to_string(pattern), expr_to_string(replacement)),
          _ => continue,
        };
        // Apply this rule to the current expression
        current = apply_replace_all_direct(&current, &pat, &repl)?;
      }
      return string_to_expr(&current);
    }
    _ => return Ok(expr.clone()),
  };

  // Use the string-based function which handles all pattern types correctly
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
  // Extract pattern and replacement from rules
  let (pattern_str, replacement_str) = match rules {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => (expr_to_string(pattern), expr_to_string(replacement)),
    _ => return Ok(expr.clone()),
  };

  // Use the string-based function which handles all pattern types correctly
  let expr_str = expr_to_string(expr);
  let result =
    apply_replace_repeated_direct(&expr_str, &pattern_str, &replacement_str)?;
  string_to_expr(&result)
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn get_expr_head(expr: &Expr) -> String {
  match expr {
    Expr::Integer(_) => "Integer".to_string(),
    Expr::Real(_) => "Real".to_string(),
    Expr::String(_) => "String".to_string(),
    Expr::List(_) => "List".to_string(),
    Expr::FunctionCall { name, .. } => name.clone(),
    Expr::Association(_) => "Association".to_string(),
    _ => "Symbol".to_string(),
  }
}

/// Apply bindings to a replacement expression
#[allow(dead_code)]
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
        // Print warning and return unevaluated Part expression
        use std::io::{self, Write};
        let expr_str = crate::syntax::expr_to_string(expr);
        println!(
          "\nPart::partw: Part {} of {} does not exist.",
          idx, expr_str
        );
        io::stdout().flush().ok();
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
        // Print warning and return unevaluated Part expression
        use std::io::{self, Write};
        println!("\nPart::partw: Part {} of \"{}\" does not exist.", idx, s);
        io::stdout().flush().ok();
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
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to b becomes f[a, b]
      // Special case: ReplaceAll and ReplaceRepeated operator forms
      // ReplaceAll[rules][expr] becomes ReplaceAll[expr, rules]
      // ReplaceRepeated[rules][expr] becomes ReplaceRepeated[expr, rules]
      if (name == "ReplaceAll" || name == "ReplaceRepeated") && args.len() == 1
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
    Expr::FunctionCall {
      name,
      args: func_args,
    } => {
      // Curried function: f[a][b] becomes f[a, b]
      // Special case: ReplaceAll and ReplaceRepeated operator forms
      // ReplaceAll[rules][expr] becomes ReplaceAll[expr, rules]
      // ReplaceRepeated[rules][expr] becomes ReplaceRepeated[expr, rules]
      if (name == "ReplaceAll" || name == "ReplaceRepeated")
        && func_args.len() == 1
        && args.len() == 1
      {
        // Operator form: prepend the argument instead of appending
        let new_args = vec![args[0].clone(), func_args[0].clone()];
        evaluate_function_call_ast(name, &new_args)
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

  // Check for simple blank pattern: x_
  if pattern.ends_with('_') && !pattern.contains(' ') {
    let var_name = pattern[..pattern.len() - 1].trim().to_string();
    if !var_name.is_empty()
      && var_name.chars().all(|c| c.is_alphanumeric() || c == '$')
    {
      return Some(WolframPattern::Blank { var_name });
    }
  }

  None
}

/// Wolfram pattern types
enum WolframPattern {
  /// x_ - matches any single expression
  Blank { var_name: String },
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
          // Try to evaluate the replacement
          let evaluated = interpret(&replaced).unwrap_or(replaced);
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
        let evaluated = interpret(&replaced).unwrap_or(replaced);
        return Ok(evaluated);
      }
      return Ok(expr.to_string());
    }
  }

  // Fall back to literal string replacement for non-pattern cases
  let result = replace_in_expr(expr, pattern, replacement);

  // Re-evaluate the result to simplify if possible
  if result != *expr {
    // Try to interpret the result, but if it fails, return as-is
    interpret(&result).or(Ok(result))
  } else {
    Ok(result)
  }
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

    // Re-evaluate to simplify
    current = interpret(&next).unwrap_or(next);
  }

  Ok(current)
}
