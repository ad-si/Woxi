use crate::syntax::{
  AST, BinaryOperator, ComparisonOp, Expr, UnaryOperator, WoNum,
  expr_to_string, str_to_wonum, wonum_to_number_str,
};
use crate::utils::create_file;
use crate::{
  ENV, FUNC_DEFS, InterpreterError, Rule, StoredValue, apply_map_operator,
  eval_association, extract_string, format_result, functions, interpret,
  parse_list_string, store_function_definition,
};

/// Check if a result value needs to be quoted when substituted into an expression.
/// Returns true for string values that would not parse correctly without quotes.
fn needs_string_quotes(s: &str) -> bool {
  // Already quoted - no need to add more quotes
  if s.starts_with('"') && s.ends_with('"') {
    return false;
  }
  // Lists don't need quoting
  if s.starts_with('{') && s.ends_with('}') {
    return false;
  }
  // Associations don't need quoting
  if s.starts_with("<|") && s.ends_with("|>") {
    return false;
  }
  // Numbers don't need quoting
  if s.parse::<f64>().is_ok() {
    return false;
  }
  // Valid identifiers don't need quoting (unless they're meant to be strings)
  // But we can't distinguish easily, so we use a conservative approach:
  // If it looks like a valid identifier (letters/digits/_), don't quote it
  // since it might be a symbolic result
  if !s.is_empty()
    && s
      .chars()
      .all(|c| c.is_alphanumeric() || c == '_' || c == '$')
  {
    return false;
  }
  // Everything else (empty strings, strings with spaces/special chars) needs quoting
  true
}

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
      // Special handling for Set - first arg must be identifier, second gets evaluated
      if name == "Set" && args.len() == 2 {
        if let Expr::Identifier(var_name) = &args[0] {
          let value = evaluate_expr_to_expr(&args[1])?;
          let value_str = expr_to_string(&value);
          crate::ENV.with(|e| {
            e.borrow_mut().insert(
              var_name.clone(),
              crate::StoredValue::Raw(value_str.clone()),
            )
          });
          return Ok(value);
        }
        return Err(InterpreterError::EvaluationError(
          "First argument of Set must be an identifier".into(),
        ));
      }
      // Special handling for Table, Do, With - don't evaluate args (body needs iteration/bindings)
      // These functions take unevaluated expressions as first argument
      if name == "Table" || name == "Do" || name == "With" {
        // Pass unevaluated args to the function dispatcher
        return evaluate_function_call_ast(name, args);
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
          if let (Some(l), Some(r)) = (left_num, right_num) {
            Ok(num_to_expr(l + r))
          } else {
            // Symbolic
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
          }
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
          if let (Some(l), Some(r)) = (left_num, right_num) {
            if r == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "Division by zero".into(),
              ));
            }
            Ok(Expr::Real(l / r))
          } else {
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
          }
        }
        BinaryOperator::Power => {
          if let (Some(l), Some(r)) = (left_num, right_num) {
            Ok(Expr::Real(l.powf(r)))
          } else {
            Ok(Expr::BinaryOp {
              op: *op,
              left: Box::new(left_val),
              right: Box::new(right_val),
            })
          }
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
    "If" => {
      if args.len() >= 2 && args.len() <= 3 {
        let cond = evaluate_expr_to_expr(&args[0])?;
        if matches!(&cond, Expr::Identifier(s) if s == "True") {
          return evaluate_expr_to_expr(&args[1]);
        } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
          if args.len() == 3 {
            return evaluate_expr_to_expr(&args[2]);
          } else {
            return Ok(Expr::Identifier("Null".to_string()));
          }
        }
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
    "MinMax" if args.len() == 1 => {
      return list_helpers_ast::min_max_ast(&args[0]);
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
    "PositiveQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::positive_q_ast(args);
    }
    "NegativeQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::negative_q_ast(args);
    }
    "NonPositiveQ" if args.len() == 1 => {
      return crate::functions::predicate_ast::non_positive_q_ast(args);
    }
    "NonNegativeQ" if args.len() == 1 => {
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
    "Minus" if args.len() == 1 => {
      return crate::functions::math_ast::minus_ast(args);
    }
    "Subtract" if args.len() == 2 => {
      return crate::functions::math_ast::subtract_ast(args);
    }
    "Divide" if args.len() == 2 => {
      return crate::functions::math_ast::divide_ast(args);
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

    _ => {}
  }

  // Fallback: Convert args to strings and dispatch through Pair-based evaluation
  let args_str: Vec<String> = args.iter().map(expr_to_string).collect();
  let call_str = format!("{}[{}]", name, args_str.join(", "));

  // Parse the call string to get a Pair, then call evaluate_expression directly
  // This bypasses the interpret() -> evaluate_expr_to_expr path to avoid infinite recursion
  let pairs = crate::parse(&call_str)?;
  let program = pairs
    .into_iter()
    .next()
    .ok_or(InterpreterError::EmptyInput)?;

  for node in program.into_inner() {
    if node.as_rule() == crate::Rule::Expression {
      match evaluate_expression(node) {
        Ok(result_str) => {
          // Check if result is a quoted string - return as String directly
          // to avoid re-parsing issues (e.g., "hello world" being parsed as ImplicitTimes)
          if result_str.starts_with('"')
            && result_str.ends_with('"')
            && result_str.len() >= 2
          {
            let inner = &result_str[1..result_str.len() - 1];
            return Ok(Expr::String(inner.to_string()));
          }
          return string_to_expr(&result_str);
        }
        Err(InterpreterError::EvaluationError(e))
          if e.starts_with("Unknown function:") =>
        {
          // Unknown function - return as symbolic function call
          return Ok(Expr::FunctionCall {
            name: name.to_string(),
            args: args.to_vec(),
          });
        }
        Err(e) => return Err(e),
      }
    }
  }

  // Fallback: return as symbolic function call
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
        Err(InterpreterError::EvaluationError(format!(
          "Part index {} out of range for list of length {}",
          idx, len
        )))
      }
    }
    Expr::String(s) => {
      let chars: Vec<char> = s.chars().collect();
      let len = chars.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(Expr::String(chars[actual_idx as usize].to_string()))
      } else {
        Err(InterpreterError::EvaluationError(format!(
          "Part index {} out of range for string of length {}",
          idx, len
        )))
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
      let mut new_args = args.clone();
      new_args.push(arg.clone());
      evaluate_function_call_ast(name, &new_args)
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

/// Represents either a numeric value or a symbolic term in an expression
#[derive(Debug, Clone)]
enum SymbolicTerm {
  Numeric(f64),
  Symbol(String),
  List(Vec<String>), // For threading arithmetic over lists
}

/// Try to evaluate a term, returning either a numeric value or a symbolic term
fn try_evaluate_term(
  term: pest::iterators::Pair<Rule>,
) -> Result<SymbolicTerm, InterpreterError> {
  match term.as_rule() {
    Rule::Term => {
      let inner = term.into_inner().next().unwrap();
      try_evaluate_term(inner)
    }
    Rule::NumericValue => {
      let inner = term.into_inner().next().unwrap();
      try_evaluate_term(inner)
    }
    Rule::Constant => match term.as_str() {
      "Pi" => Ok(SymbolicTerm::Numeric(std::f64::consts::PI)),
      _ => Err(InterpreterError::EvaluationError(format!(
        "Unknown constant: {}",
        term.as_str()
      ))),
    },
    Rule::Integer => {
      let val = term.as_str().parse::<i64>().map_err(|_| {
        InterpreterError::EvaluationError("invalid integer literal".to_string())
      })?;
      Ok(SymbolicTerm::Numeric(val as f64))
    }
    Rule::Real => {
      let val = term.as_str().parse::<f64>().map_err(|_| {
        InterpreterError::EvaluationError("invalid float literal".to_string())
      })?;
      Ok(SymbolicTerm::Numeric(val))
    }
    Rule::String => {
      // Keep strings as symbols (with quotes preserved for SameQ comparison)
      Ok(SymbolicTerm::Symbol(term.as_str().to_string()))
    }
    Rule::Expression | Rule::ExpressionNoImplicit => {
      let result = evaluate_expression(term)?;
      // Try to parse as numeric, otherwise keep as symbolic
      if let Ok(val) = result.parse::<f64>() {
        Ok(SymbolicTerm::Numeric(val))
      } else {
        Ok(SymbolicTerm::Symbol(result))
      }
    }
    Rule::FunctionCall => {
      let result = evaluate_function_call(term)?;
      if result == "True" || result == "False" {
        Ok(SymbolicTerm::Symbol(result))
      } else if let Ok(val) = result.parse::<f64>() {
        Ok(SymbolicTerm::Numeric(val))
      } else {
        Ok(SymbolicTerm::Symbol(result))
      }
    }
    Rule::Identifier => {
      let id = term.as_str();
      match id {
        "True" | "False" => Ok(SymbolicTerm::Symbol(id.to_string())),
        "Now" => Err(InterpreterError::EvaluationError(
          "Identifier 'Now' cannot be directly used as a numeric value."
            .to_string(),
        )),
        _ => {
          // Check if the identifier has a stored value
          if let Some(StoredValue::Raw(val)) =
            ENV.with(|e| e.borrow().get(id).cloned())
          {
            if let Ok(num) = val.parse::<f64>() {
              return Ok(SymbolicTerm::Numeric(num));
            }
            return Ok(SymbolicTerm::Symbol(val));
          }
          // Unknown identifier - keep as symbolic
          Ok(SymbolicTerm::Symbol(id.to_string()))
        }
      }
    }
    Rule::List => {
      // Parse list and return as SymbolicTerm::List for threading
      let items: Vec<String> = term
        .into_inner()
        .filter(|item| item.as_str() != ",")
        .map(|item| evaluate_expression(item))
        .collect::<Result<_, _>>()?;
      Ok(SymbolicTerm::List(items))
    }
    _ => {
      // Fall back to numeric evaluation for other cases
      let val = evaluate_term(term)?;
      Ok(SymbolicTerm::Numeric(val))
    }
  }
}

pub fn evaluate_expression(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  let mut inner = expr.clone().into_inner();

  if let Some(fun_call) = inner.next()
    && fun_call.as_rule() == Rule::FunctionCall
  {
    let mut idents = fun_call.clone().into_inner();
    if let Some(ident) = idents.next()
      && ident.as_span().as_str() == "CreateFile"
    {
      let filename_opt = idents
        .next()
        .map(|pair| pair.as_span().as_str().replace("\"", ""));
      return match create_file(filename_opt) {
        Ok(path) => Ok(path.to_string_lossy().into_owned()),
        Err(err) => Err(InterpreterError::EvaluationError(err.to_string())),
      };
    }
  };

  if inner.len() == 3
    && let (Some(a), Some(b), Some(c)) =
      (inner.next(), inner.next(), inner.next())
    && a.as_rule() == Rule::NumericValue
    && b.as_rule() == Rule::Operator
    && c.as_rule() == Rule::NumericValue
  {
    let op = b.as_span().as_str();
    match op {
      "+" => {
        return evaluate_ast(AST::Plus(vec![
          str_to_wonum(a.as_span().as_str()),
          str_to_wonum(c.as_span().as_str()),
        ]));
      }
      "*" => {
        return evaluate_ast(AST::Times(vec![
          str_to_wonum(a.as_span().as_str()),
          str_to_wonum(c.as_span().as_str()),
        ]));
      }
      "-" => {
        return evaluate_ast(AST::Plus(vec![
          str_to_wonum(a.as_span().as_str()),
          -str_to_wonum(c.as_span().as_str()),
        ]));
      }
      "/" => {
        let divisor = str_to_wonum(c.as_span().as_str());
        let is_zero = match divisor {
          WoNum::Int(i) => i == 0,
          WoNum::Float(f) => f == 0.0,
        };
        if is_zero {
          return Err(InterpreterError::EvaluationError(
            "Division by zero".into(),
          ));
        }
        return evaluate_ast(AST::Divide(vec![
          str_to_wonum(a.as_span().as_str()),
          divisor,
        ]));
      }
      _ => {}
    }
  }
  evaluate_pairs(expr)
}

pub fn evaluate_ast(ast: AST) -> Result<String, InterpreterError> {
  Ok(match ast {
    AST::Plus(nums) => wonum_to_number_str(nums.into_iter().sum()),

    AST::Times(wo_nums) => wonum_to_number_str(
      wo_nums
        .into_iter()
        .fold(WoNum::Float(1.0), |acc, wo_num| acc * wo_num),
    ),

    AST::Minus(wo_nums) => {
      if wo_nums.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Minus expects at least 1 argument".into(),
        ));
      }
      if wo_nums.len() == 1 {
        // Unary minus
        wonum_to_number_str(-wo_nums.into_iter().next().unwrap())
      } else {
        // Multiple arguments - wrong arity, follow old behavior
        use std::io::{self, Write};
        println!(
          "\nMinus::argx: Minus called with {} arguments; 1 argument is expected.",
          wo_nums.len()
        );
        io::stdout().flush().ok();
        // Return the expression with minus signs
        let parts: Vec<String> =
          wo_nums.iter().map(|w| wonum_to_number_str(*w)).collect();
        parts.join(" − ")
      }
    }

    AST::Divide(wo_nums) => {
      if wo_nums.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Divide expects exactly 2 arguments".into(),
        ));
      }
      let mut iter = wo_nums.into_iter();
      let a = iter.next().unwrap();
      let b = iter.next().unwrap();
      wonum_to_number_str(a / b)
    }

    AST::Abs(wo_num) => wonum_to_number_str(wo_num.abs()),

    AST::Sign(wo_num) => wo_num.sign().to_string(),

    AST::Sqrt(wo_num) => match wo_num.sqrt() {
      Ok(result) => wonum_to_number_str(result),
      Err(msg) => return Err(InterpreterError::EvaluationError(msg)),
    },

    AST::Floor(wo_num) => wonum_to_number_str(wo_num.floor()),

    AST::Ceiling(wo_num) => wonum_to_number_str(wo_num.ceiling()),

    AST::Round(wo_num) => wonum_to_number_str(wo_num.round()),

    AST::CreateFile(filename_opt) => match create_file(filename_opt) {
      Ok(filename) => filename.to_string_lossy().into_owned(),
      Err(err_msg) => err_msg.to_string(),
    },
  })
}

pub fn evaluate_pairs(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  match expr.as_rule() {
    Rule::String => Ok(expr.as_str().trim_matches('"').to_string()),
    Rule::Association => {
      let (_pairs, disp) = eval_association(expr)?;
      Ok(disp)
    }
    Rule::PostfixApplication => {
      // Chained postfix: x // f // g -> g[f[x]]
      let inner: Vec<_> = expr.into_inner().collect();
      // First item is PostfixBase, rest are PostfixFunction (Identifier or BaseFunctionCall)
      let base_expr = inner[0].clone();
      let mut result = evaluate_pairs(base_expr)?;

      // Apply functions left-to-right
      for postfix_func in inner.iter().skip(1) {
        // PostfixFunction contains either Identifier or BaseFunctionCall
        let func = postfix_func.clone().into_inner().next().unwrap();
        match func.as_rule() {
          Rule::Identifier => {
            let func_name = func.as_str();
            let expr_str = format!("{}[{}]", func_name, result);
            result = interpret(&expr_str)?;
          }
          Rule::BaseFunctionCall => {
            // For BaseFunctionCall like Map[Print], we need to apply it to the result
            // e.g., x // Map[Print] becomes Map[Print][x]
            let func_str = func.as_str();
            let expr_str = format!("{}[{}]", func_str, result);
            result = interpret(&expr_str)?;
          }
          _ => {
            return Err(InterpreterError::EvaluationError(
              "Right operand of // must be a function".into(),
            ));
          }
        }
      }
      Ok(result)
    }
    Rule::PostfixBase => {
      // PostfixBase wraps ReplaceRepeatedExpr, ReplaceAllExpr, or Term
      let inner = expr.into_inner().next().unwrap();
      evaluate_expression(inner)
    }
    Rule::NumericValue => {
      // numeric literal directly inside an expression (e.g. x == 2)
      evaluate_term(expr).map(format_result)
    }
    Rule::ReplaceAllExpr => {
      // expr /. pattern -> replacement OR expr /. {rule1, rule2, ...}
      let mut inner = expr.into_inner();
      let expr_term = inner.next().unwrap();
      let rule_pair = inner.next().unwrap(); // ReplacementRule or List

      // Get the expression to transform (try to evaluate, but use as-is if it fails)
      let expr_str = evaluate_expression(expr_term.clone())
        .unwrap_or_else(|_| expr_term.as_str().to_string());

      // Check if we have a list of rules or a single rule
      if rule_pair.as_rule() == Rule::List {
        // Multiple replacement rules - apply them in order
        let mut result = expr_str;
        for item in rule_pair.into_inner() {
          if item.as_rule() == Rule::ReplacementRule {
            let mut rule_inner = item.into_inner();
            let pattern =
              rule_inner.next().unwrap().as_str().trim().to_string();
            let replacement =
              rule_inner.next().unwrap().as_str().trim().to_string();
            result = apply_replace_all_direct(&result, &pattern, &replacement)?;
          }
        }
        Ok(result)
      } else {
        // Single replacement rule
        let mut rule_inner = rule_pair.into_inner();
        let pattern = rule_inner.next().unwrap().as_str().trim().to_string();
        let replacement =
          rule_inner.next().unwrap().as_str().trim().to_string();
        apply_replace_all_direct(&expr_str, &pattern, &replacement)
      }
    }
    Rule::ReplaceRepeatedExpr => {
      // expr //. pattern -> replacement
      let mut inner = expr.into_inner();
      let expr_term = inner.next().unwrap();
      let rule_pair = inner.next().unwrap(); // ReplacementRule

      // Get the expression to transform (try to evaluate, but use as-is if it fails)
      let expr_str = evaluate_expression(expr_term.clone())
        .unwrap_or_else(|_| expr_term.as_str().to_string());

      // Extract pattern and replacement from the rule
      let mut rule_inner = rule_pair.into_inner();
      let pattern = rule_inner.next().unwrap().as_str().trim().to_string();
      let replacement = rule_inner.next().unwrap().as_str().trim().to_string();

      apply_replace_repeated_direct(&expr_str, &pattern, &replacement)
    }
    Rule::Term => {
      let mut inner = expr.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        } else if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        } else if first.as_rule() == Rule::Association {
          let (_pairs, disp) = eval_association(first)?;
          return Ok(disp);
        } else if first.as_rule() == Rule::String {
          return Ok(first.as_str().trim_matches('"').to_string());
        } else if first.as_rule() == Rule::Integer
          || first.as_rule() == Rule::Real
          || first.as_rule() == Rule::Constant
          || first.as_rule() == Rule::NumericValue
          || first.as_rule() == Rule::Identifier
          || first.as_rule() == Rule::Slot
        {
          return evaluate_term(first).map(format_result);
        }
        // --- handle PartExtract at Term level ---
        else if first.as_rule() == Rule::PartExtract {
          return evaluate_expression(first);
        }
        // --- handle ImplicitTimes (implicit multiplication) ---
        else if first.as_rule() == Rule::ImplicitTimes {
          return evaluate_expression(first);
        }
        // --- handle anonymous functions (return as-is when not applied) ---
        else if first.as_rule() == Rule::SimpleAnonymousFunction
          || first.as_rule() == Rule::FunctionAnonymousFunction
          || first.as_rule() == Rule::ListAnonymousFunction
        {
          return Ok(first.as_str().to_string());
        }
      }
      evaluate_term(expr).map(format_result)
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication: x y z -> Times[x, y, z]
      let factors: Vec<_> = expr.into_inner().collect();
      let mut product = 1.0;
      let mut symbolic_factors: Vec<String> = Vec::new();
      let mut all_numeric = true;

      for factor in factors {
        match try_evaluate_term(factor.clone())? {
          SymbolicTerm::Numeric(v) => product *= v,
          SymbolicTerm::Symbol(s) => {
            all_numeric = false;
            symbolic_factors.push(s);
          }
          SymbolicTerm::List(items) => {
            // Format list as a symbol for symbolic multiplication
            all_numeric = false;
            symbolic_factors.push(format!("{{{}}}", items.join(", ")));
          }
        }
      }

      if all_numeric {
        Ok(format_result(product))
      } else {
        // Return as symbolic Times expression
        let mut parts: Vec<String> = Vec::new();
        if product != 1.0 {
          parts.push(format_result(product));
        }
        parts.extend(symbolic_factors);
        Ok(parts.join(" "))
      }
    }
    Rule::PartExtract => {
      let original_expr = expr.as_str().to_string();
      let mut inner = expr.into_inner();
      let first_pair = inner.next().unwrap();
      let key_pair = inner.next().unwrap();
      let raw = key_pair.as_str().trim();
      let key = if raw.starts_with('"') && raw.ends_with('"') {
        raw.trim_matches('"').to_string()
      } else {
        raw.to_string()
      };

      // Helper to print Part warning in Mathematica style
      // Returns "\0" as a sentinel value to signal that main should not print anything
      let print_part_warning = |idx: &str,
                                list_repr: &str,
                                expr_repr: &str|
       -> String {
        println!();
        println!("Part::partw: Part {} of {} does not exist.", idx, list_repr);
        println!("{}", expr_repr);
        "\0".to_string() // sentinel value to suppress main's output
      };

      // If first_pair is a List, use get_list_items for proper nested handling
      if first_pair.as_rule() == Rule::List {
        let list_repr = first_pair.as_str().to_string();
        if let Ok(idx) = key.parse::<usize>() {
          let items = functions::list::get_list_items(&first_pair)?;
          if idx >= 1 && idx <= items.len() {
            return evaluate_expression(items[idx - 1].clone());
          }
          // Index out of bounds - print warning and return sentinel to suppress main's output
          return Ok(print_part_warning(&key, &list_repr, &original_expr));
        }
        return Err(InterpreterError::EvaluationError(
          "Index must be a positive integer".into(),
        ));
      }

      // For identifiers, check if it's an association first
      let name = first_pair.as_str();
      if let Some(StoredValue::Association(vec)) =
        ENV.with(|e| e.borrow().get(name).cloned())
      {
        for (k, v) in vec {
          if k == key {
            return Ok(v);
          }
        }
        return Err(InterpreterError::EvaluationError("Key not found".into()));
      }
      // If not found as association, try to evaluate as a list and use numeric index
      if let Ok(list_str) = evaluate_expression(first_pair.clone()) {
        // Try to parse as a list: {a, b, c}
        if let Some(items) = parse_list_string(&list_str) {
          if let Ok(idx) = key.parse::<usize>() {
            if idx >= 1 && idx <= items.len() {
              return Ok(items[idx - 1].clone());
            }
            // Index out of bounds - print warning and return sentinel to suppress main's output
            // For variable access like x[[10]], show the evaluated list in the expression
            let expr_repr = format!("{}[[{}]]", list_str, key);
            return Ok(print_part_warning(&key, &list_str, &expr_repr));
          }
          return Err(InterpreterError::EvaluationError(
            "Index must be a positive integer".into(),
          ));
        }
      }
      Err(InterpreterError::EvaluationError(
        "Argument must be an association or list".into(),
      ))
    }
    Rule::Expression | Rule::ExpressionNoImplicit => {
      {
        let mut expr_inner = expr.clone().into_inner();
        if let Some(first) = expr_inner.next()
          && expr_inner.next().is_none()
        {
          // Single child expression - delegate to appropriate handler
          // IMPORTANT: FunctionCall is checked here for performance, to avoid
          // unnecessary operator pattern matching for nested function calls
          match first.as_rule() {
            Rule::FunctionCall => return evaluate_function_call(first),
            Rule::PartExtract => return evaluate_expression(first),
            Rule::ReplaceAllExpr => return evaluate_pairs(first),
            Rule::ReplaceRepeatedExpr => return evaluate_pairs(first),
            Rule::PostfixApplication => return evaluate_pairs(first),
            Rule::PostfixBase => return evaluate_pairs(first),
            Rule::List => return evaluate_expression(first),
            Rule::Identifier => {
              let id = first.as_str();
              if id == "Now" {
                return Ok("CURRENT_TIME_MARKER".to_string());
              }
              if let Some(stored) = ENV.with(|e| e.borrow().get(id).cloned()) {
                return Ok(match stored {
                  StoredValue::Association(pairs) => format!(
                    "<|{}|>",
                    pairs
                      .iter()
                      .map(|(k, v)| format!("{} -> {}", k, v))
                      .collect::<Vec<_>>()
                      .join(", ")
                  ),
                  StoredValue::Raw(val) => val,
                });
              }
              return Ok(id.to_string());
            }
            Rule::NumericValue | Rule::Integer | Rule::Real => {
              return evaluate_term(first).map(format_result);
            }
            Rule::String => {
              return Ok(first.as_str().to_string());
            }
            Rule::Term => {
              // Unwrap term and evaluate its content
              if let Some(inner) = first.into_inner().next() {
                return evaluate_expression(inner);
              }
            }
            _ => {}
          }
        }
      }
      // --- special case: Map operator ----------------------------------
      {
        let items: Vec<_> = expr.clone().into_inner().collect();
        if items.len() == 1 && items[0].as_rule() == Rule::Association {
          let (_pairs, disp) = eval_association(items[0].clone())?;
          return Ok(disp);
        }
        // Handle chained @ operators (right-associative): f @ g @ x -> f[g[x]]
        // Check if all odd-indexed items are @ operators
        let all_at = items.len() >= 3
          && items.len() % 2 == 1
          && items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "@");

        if all_at {
          // Process right-to-left: f @ g @ h @ x -> f[g[h[x]]]
          // Start with the rightmost term and work backwards
          let terms: Vec<_> = items.iter().step_by(2).cloned().collect();
          // Evaluate the rightmost term first
          // For strings, preserve the Wolfram representation (with quotes)
          let rightmost = terms[terms.len() - 1].clone();
          let mut result = if rightmost.as_rule() == Rule::String {
            rightmost.as_str().to_string() // Keep quotes
          } else {
            evaluate_expression(rightmost)?
          };
          // Apply functions from right to left
          for i in (0..terms.len() - 1).rev() {
            let func = &terms[i];
            if func.as_rule() == Rule::Identifier {
              let func_name = func.as_str();
              let expr = format!("{}[{}]", func_name, result);
              result = interpret(&expr)?;
            } else if matches!(
              func.as_rule(),
              Rule::SimpleAnonymousFunction
                | Rule::FunctionAnonymousFunction
                | Rule::ParenAnonymousFunction
                | Rule::ListAnonymousFunction
            ) {
              // Anonymous function: substitute # with the argument
              let mut body =
                func.as_str().trim().trim_end_matches('&').to_string();
              // For ParenAnonymousFunction, remove the outer parens
              if func.as_rule() == Rule::ParenAnonymousFunction {
                body = body
                  .trim_start_matches('(')
                  .trim_end_matches(')')
                  .to_string();
              }
              // Quote the result if it's a string value that needs quoting for expression parsing
              let substitution = if needs_string_quotes(&result) {
                format!("\"{}\"", result)
              } else {
                result.clone()
              };
              let expr = body.replace('#', &substitution);
              result = interpret(&expr)?;
            } else {
              return Err(InterpreterError::EvaluationError(
                "Left operand of @ must be a function".into(),
              ));
            }
          }
          return Ok(result);
        }

        // Handle chained // operators (left-associative): x // f // g -> g[f[x]]
        let all_postfix = items.len() >= 3
          && items.len() % 2 == 1
          && items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "//");

        if all_postfix {
          // Process left-to-right: x // f // g -> g[f[x]]
          let terms: Vec<_> = items.iter().step_by(2).cloned().collect();
          // Evaluate the leftmost term first (the argument)
          let mut result = evaluate_expression(terms[0].clone())?;
          // Apply functions from left to right
          for i in 1..terms.len() {
            let func = &terms[i];
            if func.as_rule() == Rule::Identifier {
              let func_name = func.as_str();
              let expr = format!("{}[{}]", func_name, result);
              result = interpret(&expr)?;
            } else {
              return Err(InterpreterError::EvaluationError(
                "Right operand of // must be a function".into(),
              ));
            }
          }
          return Ok(result);
        }

        // Handle <> (StringJoin) operator: s1 <> s2 <> s3 -> StringJoin[s1, s2, s3]
        let all_string_join = items.len() >= 3
          && items.len() % 2 == 1
          && items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "<>");

        if all_string_join {
          // Collect all terms (skip the operators)
          let terms: Vec<_> = items.iter().step_by(2).cloned().collect();
          // Evaluate each term and build a StringJoin call
          let mut args: Vec<String> = Vec::new();
          for term in terms {
            let evaluated = evaluate_expression(term)?;
            // Preserve quotes for string arguments
            if evaluated.starts_with('"') && evaluated.ends_with('"') {
              args.push(evaluated);
            } else {
              args.push(format!("\"{}\"", evaluated));
            }
          }
          let expr = format!("StringJoin[{}]", args.join(", "));
          return interpret(&expr);
        }

        if items.len() == 3
          && items[1].as_rule() == Rule::Operator
          && items[1].as_str() == "/@"
        {
          return apply_map_operator(items[0].clone(), items[2].clone());
        }

        // Handle @@@ (MapApply) operator: f @@@ {{a, b}, {c, d}} -> {f[a, b], f[c, d]}
        if items.len() == 3
          && items[1].as_rule() == Rule::Operator
          && items[1].as_str() == "@@@"
        {
          let func = items[0].clone();
          let list = items[2].clone();

          let func_name = if func.as_rule() == Rule::Identifier {
            func.as_str().to_string()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Left operand of @@@ must be a function".into(),
            ));
          };

          // Evaluate the list and extract its sublists
          let list_str = evaluate_expression(list)?;
          if list_str.starts_with('{') && list_str.ends_with('}') {
            let inner = &list_str[1..list_str.len() - 1];
            let elements = split_list_elements(inner);
            let mut results = Vec::new();
            for elem in elements {
              // Each element should be a list that we apply the function to
              if elem.starts_with('{') && elem.ends_with('}') {
                let inner = &elem[1..elem.len() - 1];
                let expr = format!("{}[{}]", func_name, inner);
                match interpret(&expr) {
                  Ok(result) => results.push(result),
                  Err(InterpreterError::EvaluationError(e))
                    if e.starts_with("Unknown function:") =>
                  {
                    results.push(expr);
                  }
                  Err(e) => return Err(e),
                }
              } else {
                // If element is not a list, return it unapplied (mimics Wolfram behavior)
                results.push(format!("{}[{}]", func_name, elem));
              }
            }
            return Ok(format!("{{{}}}", results.join(", ")));
          } else {
            return Err(InterpreterError::EvaluationError(
              "Right operand of @@@ must be a list".into(),
            ));
          }
        }

        // Handle @@ (Apply) operator: f @@ {a, b, c} -> f[a, b, c]
        if items.len() == 3
          && items[1].as_rule() == Rule::Operator
          && items[1].as_str() == "@@"
        {
          let func = items[0].clone();
          let list = items[2].clone();

          let func_name = if func.as_rule() == Rule::Identifier {
            func.as_str().to_string()
          } else {
            return Err(InterpreterError::EvaluationError(
              "Left operand of @@ must be a function".into(),
            ));
          };

          // Evaluate the list and extract its items
          let list_str = evaluate_expression(list)?;
          // Parse the evaluated list to extract elements
          if list_str.starts_with('{') && list_str.ends_with('}') {
            let inner = &list_str[1..list_str.len() - 1];
            let expr = format!("{}[{}]", func_name, inner);
            // Try to evaluate; if function is unknown, return the symbolic form
            match interpret(&expr) {
              Ok(result) => return Ok(result),
              Err(InterpreterError::EvaluationError(e))
                if e.starts_with("Unknown function:") =>
              {
                return Ok(expr);
              }
              Err(e) => return Err(e),
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "Right operand of @@ must be a list".into(),
            ));
          }
        }

        // --- handle = and := assignment operators ---
        if items.len() >= 3 && items[1].as_rule() == Rule::Operator {
          match items[1].as_str() {
            "=" => {
              let lhs = items[0].clone();

              // Handle association update: myHash[["key"]] = value
              if lhs.as_rule() == Rule::PartExtract {
                let mut lhs_inner = lhs.into_inner();
                let ident = lhs_inner.next().unwrap();
                let key_expr = lhs_inner.next().unwrap();
                let var_name = ident.as_str().to_string();
                let key = extract_string(key_expr)?;

                // Evaluate RHS
                let rhs_value = evaluate_pairs(items[2].clone())?;

                // Update or add the key in the association
                ENV.with(|e| {
                  let mut env = e.borrow_mut();
                  if let Some(StoredValue::Association(pairs)) =
                    env.get_mut(&var_name)
                  {
                    // Update existing key or add new key
                    if let Some(pair) =
                      pairs.iter_mut().find(|(k, _)| k == &key)
                    {
                      pair.1 = rhs_value.clone();
                    } else {
                      pairs.push((key.clone(), rhs_value.clone()));
                    }
                  } else {
                    return Err(InterpreterError::EvaluationError(format!(
                      "{} is not an association",
                      var_name
                    )));
                  }
                  Ok(())
                })?;

                // Return the updated association
                return ENV.with(|e| {
                  let env = e.borrow();
                  if let Some(StoredValue::Association(pairs)) =
                    env.get(&var_name)
                  {
                    let disp_parts: Vec<String> = pairs
                      .iter()
                      .map(|(k, v)| format!("{} -> {}", k, v))
                      .collect();
                    Ok(format!("<|{}|>", disp_parts.join(", ")))
                  } else {
                    Err(InterpreterError::EvaluationError(
                      "Variable not found".into(),
                    ))
                  }
                });
              }

              // LHS must be an identifier for regular assignment
              if lhs.as_rule() != Rule::Identifier {
                return Err(InterpreterError::EvaluationError(
                  "Left-hand side of assignment must be an identifier or part extract".into(),
                ));
              }
              let name = lhs.as_str().to_string();

              // --- association assignment  (x = <| … |>) -------------------------
              if items.len() == 3 && items[2].as_rule() == Rule::Association {
                let (pairs, disp) = eval_association(items[2].clone())?;
                ENV.with(|e| {
                  e.borrow_mut().insert(name, StoredValue::Association(pairs))
                });
                return Ok(disp);
              }

              // --- generic RHS: may be any (possibly complex) expression ----------
              // evaluate everything that comes after the first '='
              let full_txt = expr.as_str();
              // Find the first '=' that is not part of an operator like '==' or '!='
              // This is a simple approach: split on the first '=' that is not preceded or followed by '='
              // (Assumes no whitespace between '='s in '==', '!=' etc.)
              let mut eq_index = None;
              let chars: Vec<char> = full_txt.chars().collect();
              for i in 0..chars.len() {
                if chars[i] == '=' {
                  let prev = if i > 0 { chars[i - 1] } else { '\0' };
                  let next = if i + 1 < chars.len() {
                    chars[i + 1]
                  } else {
                    '\0'
                  };
                  if prev != '=' && next != '=' {
                    eq_index = Some(i);
                    break;
                  }
                }
              }
              let rhs_txt = if let Some(idx) = eq_index {
                &full_txt[idx + 1..]
              } else {
                ""
              };
              let rhs_txt = rhs_txt.trim();

              let val = interpret(rhs_txt)?; // recursive evaluation
              ENV.with(|e| {
                e.borrow_mut().insert(name, StoredValue::Raw(val.clone()))
              });
              return Ok(val);
            }
            ":=" => {
              // Detect a definition like  f[x_] := body
              let lhs_pair = items[0].clone();

              // unwrap a possible Term wrapper so we can directly
              // look at the FunctionCall node
              let func_call_pair = if lhs_pair.as_rule() == Rule::Term {
                let mut inner = lhs_pair.clone().into_inner();
                let first = inner.next().unwrap();
                if first.as_rule() == Rule::FunctionCall {
                  first
                } else {
                  lhs_pair.clone()
                }
              } else {
                lhs_pair.clone()
              };

              // Only treat it as a user-function definition if the left side
              // really is a `FunctionCall` whose arguments are *patterns*
              if func_call_pair.as_rule() == Rule::FunctionCall {
                let mut fc_inner = func_call_pair.clone().into_inner();
                let func_name = fc_inner.next().unwrap().as_str().to_owned();

                // collect all pattern arguments (e.g. x_, y_, …  →  ["x","y"])
                let mut params = Vec::new();
                for arg in fc_inner.filter(|p| p.as_rule() == Rule::Pattern) {
                  params.push(arg.as_str().trim_end_matches('_').to_owned());
                }

                // Obtain the complete *text* to the right of the ":=" so we
                // don't lose operators like "* 2"
                let full_txt = expr.as_str();

                // find the first occurrence of ":="
                let rhs_txt = full_txt
                  .split_once(":=")
                  .map(|x| x.1)
                  .unwrap_or("")
                  .trim()
                  .to_owned();

                // Parse the body text into an Expr AST
                // We need to parse it to get a pest Pair, then convert to Expr
                let body_expr = match crate::parse(&rhs_txt) {
                  Ok(mut pairs) => {
                    if let Some(program) = pairs.next() {
                      // Get the first expression from the program
                      if let Some(expr_pair) = program.into_inner().next() {
                        crate::syntax::pair_to_expr(expr_pair)
                      } else {
                        Expr::Raw(rhs_txt)
                      }
                    } else {
                      Expr::Raw(rhs_txt)
                    }
                  }
                  Err(_) => Expr::Raw(rhs_txt),
                };

                FUNC_DEFS.with(|m| {
                  let mut defs = m.borrow_mut();
                  let entry = defs.entry(func_name).or_insert_with(Vec::new);
                  // Remove any existing definition with the same arity
                  let arity = params.len();
                  entry.retain(|(p, _)| p.len() != arity);
                  // Add the new definition with parsed AST
                  entry.push((params, body_expr));
                });

                return Ok("Null".to_string());
              }

              // not a function definition → keep previous behaviour
              return Ok("Null".to_string());
            }
            _ => { /* fall-through to the maths/other logic below */ }
          }
        }

        // --- relational operators '==' and '!=' ------------------------------
        if items.len() >= 3 && items.len() % 2 == 1 {
          let all_eq = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "==");
          let all_neq = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "!=");

          if all_eq {
            // Evaluate all sub-expressions and compare as strings
            let ref_val = evaluate_expression(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cmp_val = evaluate_expression(items[idx].clone())?;
              if cmp_val != ref_val {
                return Ok("False".to_string());
              }
            }
            return Ok("True".to_string());
          }
          if all_neq {
            use std::collections::HashSet;
            let mut seen = HashSet::new();
            for idx in (0..items.len()).step_by(2) {
              let v = evaluate_expression(items[idx].clone())?;
              if !seen.insert(v) {
                return Ok("False".to_string());
              }
            }
            return Ok("True".to_string());
          }

          let all_gt = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == ">");
          let all_lt = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "<");
          let all_ge = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == ">=");
          let all_le = items
            .iter()
            .skip(1)
            .step_by(2)
            .all(|p| p.as_rule() == Rule::Operator && p.as_str() == "<=");

          if all_gt {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev.partial_cmp(&cur) != Some(std::cmp::Ordering::Greater) {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_lt {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev.partial_cmp(&cur) != Some(std::cmp::Ordering::Less) {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_ge {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev < cur {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          if all_le {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (2..items.len()).step_by(2) {
              let cur = evaluate_term(items[idx].clone())?;
              if prev > cur {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
          // --- mixed <  >  <=  >= comparisons -----------------------------
          let all_ineq = items.iter().skip(1).step_by(2).all(|p| {
            p.as_rule() == Rule::Operator
              && matches!(p.as_str(), ">" | "<" | ">=" | "<=")
          });

          if all_ineq {
            let mut prev = evaluate_term(items[0].clone())?;
            for idx in (1..items.len()).step_by(2) {
              let op = items[idx].as_str();
              let cur = evaluate_term(items[idx + 1].clone())?;

              let ok = match op {
                ">" => prev > cur,
                "<" => prev < cur,
                ">=" => prev >= cur,
                "<=" => prev <= cur,
                _ => unreachable!(),
              };
              if !ok {
                return Ok("False".to_string());
              }
              prev = cur;
            }
            return Ok("True".to_string());
          }
        }
      }
      let expr_str = expr.as_str();
      let mut inner = expr.into_inner();
      let first = inner.next().unwrap();
      if inner.clone().next().is_none() {
        if first.as_rule() == Rule::List {
          return evaluate_expression(first);
        } else if first.as_rule() == Rule::Identifier {
          // Evaluate identifier as in the main Rule::Identifier arm
          let id = first.as_str();
          if id == "Now" {
            return Ok("CURRENT_TIME_MARKER".to_string());
          }
          if let Some(stored) = ENV.with(|e| e.borrow().get(id).cloned()) {
            return Ok(match stored {
              StoredValue::Association(pairs) => format!(
                "<|{}|>",
                pairs
                  .iter()
                  .map(|(k, v)| format!("{} -> {}", k, v))
                  .collect::<Vec<_>>()
                  .join(", ")
              ),
              StoredValue::Raw(val) => val,
            });
          }
          return Ok(id.to_string());
        } else if first.as_rule() == Rule::FunctionCall {
          return evaluate_function_call(first);
        } else if first.as_rule() == Rule::Term {
          return evaluate_expression(first.into_inner().next().unwrap());
        } else if first.as_rule() == Rule::NumericValue {
          // Evaluate the numeric value as a number and format as string
          return evaluate_term(first).map(format_result);
        } else if first.as_rule() == Rule::String {
          // Return string with quotes preserved
          return Ok(first.as_str().to_string());
        }
      }

      // Check for special operators: //, /., //.
      // These are handled inline in the new grammar structure
      // The grammar produces PostfixFunction for //, and List/ReplacementRule for /. and //.
      let items: Vec<_> = inner.clone().collect();
      if !items.is_empty() {
        let first_rule = items[0].as_rule();

        // Handle postfix application: term // func // func2 ...
        // The grammar produces PostfixFunction items directly (// is consumed)
        if first_rule == Rule::PostfixFunction
          || first_rule == Rule::BaseFunctionCall
          || (first_rule == Rule::Identifier
            && items.iter().all(|i| {
              matches!(
                i.as_rule(),
                Rule::PostfixFunction
                  | Rule::BaseFunctionCall
                  | Rule::Identifier
              )
            }))
        {
          // Check if all items are PostfixFunction/BaseFunctionCall/Identifier
          let all_postfix = items.iter().all(|i| {
            matches!(
              i.as_rule(),
              Rule::PostfixFunction | Rule::BaseFunctionCall | Rule::Identifier
            )
          });
          if all_postfix {
            let mut result = evaluate_expression(first.clone())?;
            for func in &items {
              // Apply each function to the result
              let func_name = func.as_str();
              let expr = format!("{}[{}]", func_name, result);
              result = interpret(&expr)?;
            }
            return Ok(result);
          }
        }

        // Handle ReplaceRepeated: term //. rule
        // and ReplaceAll: term /. rule
        // The grammar produces List or ReplacementRule items directly
        if first_rule == Rule::List || first_rule == Rule::ReplacementRule {
          // Get the base expression value
          let base_result = evaluate_expression(first.clone())
            .unwrap_or_else(|_| first.as_str().to_string());

          // Determine if this is /. or //. based on the original expression
          let is_replace_repeated =
            expr_str.contains("//.") && !expr_str.contains("///.");

          let rule_pair = &items[0];

          let mut result = if is_replace_repeated {
            // ReplaceRepeated: apply rule repeatedly until no change
            let mut rule_inner = rule_pair.clone().into_inner();
            let pattern =
              rule_inner.next().unwrap().as_str().trim().to_string();
            let replacement =
              rule_inner.next().unwrap().as_str().trim().to_string();
            apply_replace_repeated_direct(&base_result, &pattern, &replacement)?
          } else {
            // ReplaceAll: apply rule once
            if rule_pair.as_rule() == Rule::List {
              // Multiple replacement rules - apply them in order
              let mut r = base_result;
              for item in rule_pair.clone().into_inner() {
                if item.as_rule() == Rule::ReplacementRule {
                  let mut rule_inner = item.into_inner();
                  let pattern =
                    rule_inner.next().unwrap().as_str().trim().to_string();
                  let replacement =
                    rule_inner.next().unwrap().as_str().trim().to_string();
                  r = apply_replace_all_direct(&r, &pattern, &replacement)?;
                }
              }
              r
            } else {
              // Single replacement rule
              let mut rule_inner = rule_pair.clone().into_inner();
              let pattern =
                rule_inner.next().unwrap().as_str().trim().to_string();
              let replacement =
                rule_inner.next().unwrap().as_str().trim().to_string();
              apply_replace_all_direct(&base_result, &pattern, &replacement)?
            }
          };

          // Check for postfix functions after the replacement (items[1:] are PostfixFunction)
          for func in items.iter().skip(1) {
            if matches!(
              func.as_rule(),
              Rule::PostfixFunction | Rule::BaseFunctionCall | Rule::Identifier
            ) {
              let func_name = func.as_str();
              let expr = format!("{}[{}]", func_name, result);
              result = interpret(&expr)?;
            }
          }

          return Ok(result);
        }
      }

      // Collect all terms and operators using symbolic evaluation
      let mut terms: Vec<SymbolicTerm> = vec![try_evaluate_term(first)?];
      let mut ops: Vec<&str> = vec![];
      while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        // For special operators, the next item might not be a Term
        if op == "//" || op == "/." || op == "//." {
          // These should have been handled above
          return Err(InterpreterError::EvaluationError(format!(
            "Unexpected operator {} in expression",
            op
          )));
        }
        let term = inner.next().unwrap();
        ops.push(op);
        terms.push(try_evaluate_term(term)?);
      }

      // Check for logical/comparison operators that need special handling
      let has_logical_ops = ops.iter().any(|op| {
        matches!(
          *op,
          "&&" | "||" | "===" | "=!=" | "==" | "!=" | "<" | ">" | "<=" | ">="
        )
      });

      if has_logical_ops {
        // Convert to function call form and evaluate
        // For expressions like "a && b && c", convert to "And[a, b, c]"
        // For "a == b", convert to "Equal[a, b]"

        // Get string representations of terms
        let term_strs: Vec<String> = terms
          .iter()
          .map(|t| match t {
            SymbolicTerm::Numeric(v) => format_result(*v),
            SymbolicTerm::Symbol(s) => s.clone(),
            SymbolicTerm::List(items) => format!("{{{}}}", items.join(", ")),
          })
          .collect();

        // Check if all operators are the same
        if ops.iter().all(|&o| o == ops[0]) {
          // Use string-based helpers directly to avoid re-parsing
          let result = match ops[0] {
            "&&" => functions::boolean::and_strs(&term_strs),
            "||" => functions::boolean::or_strs(&term_strs),
            "===" => functions::boolean::same_q_strs(&term_strs),
            "=!=" => functions::boolean::unsame_q_strs(&term_strs),
            "==" => functions::math::equal_strs(&term_strs),
            "!=" => functions::math::unequal_strs(&term_strs),
            "<" => functions::math::less_strs(&term_strs),
            ">" => functions::math::greater_strs(&term_strs),
            "<=" => functions::math::less_equal_strs(&term_strs),
            ">=" => functions::math::greater_equal_strs(&term_strs),
            _ => unreachable!(),
          };
          return Ok(result);
        }

        // Mixed operators - evaluate left to right
        let mut result = term_strs[0].clone();
        for (i, &op) in ops.iter().enumerate() {
          // Use string-based helpers directly to avoid re-parsing
          let pair = vec![result.clone(), term_strs[i + 1].clone()];
          result = match op {
            "&&" => functions::boolean::and_strs(&pair),
            "||" => functions::boolean::or_strs(&pair),
            "===" => functions::boolean::same_q_strs(&pair),
            "=!=" => functions::boolean::unsame_q_strs(&pair),
            "==" => functions::math::equal_strs(&pair),
            "!=" => functions::math::unequal_strs(&pair),
            "<" => functions::math::less_strs(&pair),
            ">" => functions::math::greater_strs(&pair),
            "<=" => functions::math::less_equal_strs(&pair),
            ">=" => functions::math::greater_equal_strs(&pair),
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "Unexpected operator: {}",
                op
              )));
            }
          };
        }
        return Ok(result);
      }

      // Check if any term is a list - if so, thread the operation
      let has_list = terms.iter().any(|t| matches!(t, SymbolicTerm::List(_)));

      if has_list {
        // Thread arithmetic operations over lists
        // Find the list length (all lists must have the same length)
        let mut list_len: Option<usize> = None;
        for term in &terms {
          if let SymbolicTerm::List(items) = term {
            match list_len {
              None => list_len = Some(items.len()),
              Some(len) if len != items.len() => {
                return Err(InterpreterError::EvaluationError(
                  "Lists have incompatible shapes".into(),
                ));
              }
              _ => {}
            }
          }
        }

        let len = list_len.unwrap();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
          // Build expression for this index
          let mut expr_parts: Vec<String> = Vec::new();

          // First term
          let first_val = match &terms[0] {
            SymbolicTerm::Numeric(v) => format_result(*v),
            SymbolicTerm::Symbol(s) => s.clone(),
            SymbolicTerm::List(items) => items[i].clone(),
          };
          expr_parts.push(first_val);

          // Remaining terms with operators
          for (j, op) in ops.iter().enumerate() {
            let val = match &terms[j + 1] {
              SymbolicTerm::Numeric(v) => format_result(*v),
              SymbolicTerm::Symbol(s) => s.clone(),
              SymbolicTerm::List(items) => items[i].clone(),
            };
            expr_parts.push((*op).to_string());
            expr_parts.push(val);
          }

          // Evaluate the expression for this index
          let sub_expr = expr_parts.join(" ");
          let result = interpret(&sub_expr)?;
          results.push(result);
        }

        return Ok(format!("{{{}}}", results.join(", ")));
      }

      // Check if all terms are numeric - if so, use pure numeric evaluation
      let all_numeric =
        terms.iter().all(|t| matches!(t, SymbolicTerm::Numeric(_)));

      if all_numeric {
        // Pure numeric path - extract f64 values
        let mut values: Vec<f64> = terms
          .iter()
          .map(|t| match t {
            SymbolicTerm::Numeric(v) => *v,
            _ => unreachable!(),
          })
          .collect();

        // First pass: handle exponentiation (highest precedence)
        let mut i = 0;
        while i < ops.len() {
          if ops[i] == "^" {
            values[i] = values[i].powf(values[i + 1]);
            values.remove(i + 1);
            ops.remove(i);
          } else {
            i += 1;
          }
        }
        // Second pass: handle multiplication and division
        let mut i = 0;
        while i < ops.len() {
          if ops[i] == "*" {
            values[i] *= values[i + 1];
            values.remove(i + 1);
            ops.remove(i);
          } else if ops[i] == "/" {
            if values[i + 1] == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "Division by zero".to_string(),
              ));
            }
            values[i] /= values[i + 1];
            values.remove(i + 1);
            ops.remove(i);
          } else {
            i += 1;
          }
        }
        // Third pass: handle addition and subtraction
        let mut result = values[0];
        for (op, &val) in ops.iter().zip(values.iter().skip(1)) {
          if *op == "+" {
            result += val;
          } else if *op == "-" {
            result -= val;
          } else {
            return Err(InterpreterError::EvaluationError(format!(
              "Unexpected operator: {}",
              op
            )));
          }
        }
        return Ok(format_result(result));
      }

      // Symbolic path - handle mixed numeric and symbolic terms
      // For now, only handle addition and subtraction with symbols
      // Check if we only have + and - operators (symbolic-safe)
      let only_add_sub = ops.iter().all(|op| *op == "+" || *op == "-");

      if only_add_sub {
        // Collect numeric sum and symbolic terms with their signs
        let mut numeric_sum: f64 = 0.0;
        let mut symbolic_terms: Vec<(i8, String)> = vec![]; // (sign, symbol)

        // Process first term
        match &terms[0] {
          SymbolicTerm::Numeric(v) => numeric_sum += v,
          SymbolicTerm::Symbol(s) => symbolic_terms.push((1, s.clone())),
          SymbolicTerm::List(_) => unreachable!(), // Lists are handled above
        }

        // Process remaining terms with their operators
        for (i, op) in ops.iter().enumerate() {
          let sign: i8 = if *op == "+" { 1 } else { -1 };
          match &terms[i + 1] {
            SymbolicTerm::Numeric(v) => {
              if sign == 1 {
                numeric_sum += v;
              } else {
                numeric_sum -= v;
              }
            }
            SymbolicTerm::Symbol(s) => {
              symbolic_terms.push((sign, s.clone()));
            }
            SymbolicTerm::List(_) => unreachable!(), // Lists are handled above
          }
        }

        // Format the result: numeric part + symbolic terms
        let mut parts: Vec<String> = vec![];

        // Add numeric part if non-zero or if there are no symbolic terms
        if numeric_sum != 0.0 || symbolic_terms.is_empty() {
          parts.push(format_result(numeric_sum));
        }

        // Add symbolic terms
        for (sign, sym) in symbolic_terms {
          if parts.is_empty() {
            // First term
            if sign == -1 {
              parts.push(format!("-{}", sym));
            } else {
              parts.push(sym);
            }
          } else {
            // Subsequent terms
            if sign == -1 {
              parts.push(format!("- {}", sym));
            } else {
              parts.push(format!("+ {}", sym));
            }
          }
        }

        return Ok(parts.join(" "));
      }

      // Fall back to numeric evaluation for complex expressions with *, /, ^
      let mut values: Vec<f64> = terms
        .iter()
        .map(|t| match t {
          SymbolicTerm::Numeric(v) => *v,
          SymbolicTerm::Symbol(_) => 0.0, // Treat unknown symbols as 0 for complex ops
          SymbolicTerm::List(_) => unreachable!(), // Lists are handled above
        })
        .collect();

      // First pass: handle exponentiation (highest precedence)
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "^" {
          values[i] = values[i].powf(values[i + 1]);
          values.remove(i + 1);
          ops.remove(i);
        } else {
          i += 1;
        }
      }
      // Second pass: handle multiplication and division
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "*" {
          values[i] *= values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        } else if ops[i] == "/" {
          if values[i + 1] == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          values[i] /= values[i + 1];
          values.remove(i + 1);
          ops.remove(i);
        } else {
          i += 1;
        }
      }
      // Third pass: handle addition and subtraction
      let mut result = values[0];
      for (op, &val) in ops.iter().zip(values.iter().skip(1)) {
        if *op == "+" {
          result += val;
        } else if *op == "-" {
          result -= val;
        } else {
          return Err(InterpreterError::EvaluationError(format!(
            "Unexpected operator: {}",
            op
          )));
        }
      }
      Ok(format_result(result))
    }
    Rule::Program => {
      let mut last = None;
      for node in expr.into_inner() {
        match node.as_rule() {
          Rule::Expression | Rule::ExpressionNoImplicit => {
            last = Some(evaluate_expression(node)?);
          }
          Rule::FunctionDefinition => {
            store_function_definition(node)?;
            last = Some("Null".to_string());
          }
          _ => {}
        }
      }
      last.ok_or(InterpreterError::EmptyInput)
    }
    Rule::List => {
      let items: Vec<String> = expr
        .into_inner()
        .filter(|item| item.as_str() != ",")
        .map(|item| evaluate_expression(item))
        .collect::<Result<_, _>>()?;
      Ok(format!("{{{}}}", items.join(", ")))
    }
    Rule::FunctionCall => evaluate_function_call(expr),
    Rule::CompoundExpression => {
      // Evaluate each expression in sequence, return the last one
      let mut last_result = "Null".to_string();
      for inner in expr.into_inner() {
        last_result = evaluate_expression(inner)?;
      }
      Ok(last_result)
    }
    Rule::Identifier => {
      let id = expr.as_str();
      if id == "Now" {
        return Ok("CURRENT_TIME_MARKER".to_string());
      }
      if let Some(stored) = ENV.with(|e| e.borrow().get(id).cloned()) {
        return Ok(match stored {
          StoredValue::Association(pairs) => format!(
            "<|{}|>",
            pairs
              .iter()
              .map(|(k, v)| format!("{} -> {}", k, v))
              .collect::<Vec<_>>()
              .join(", ")
          ),
          StoredValue::Raw(val) => val,
        });
      }
      Ok(id.to_string())
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unexpected rule: {:?}",
      expr.as_rule()
    ))),
  }
}

/// Convert function arguments to WoNum for AST construction
/// Fast evaluation of a pair to f64 - avoids expression overhead for simple cases
fn fast_eval_to_f64(
  pair: pest::iterators::Pair<Rule>,
) -> Result<f64, InterpreterError> {
  match pair.as_rule() {
    Rule::Integer => {
      pair.as_str().parse::<i64>().map(|n| n as f64).map_err(|_| {
        InterpreterError::EvaluationError("invalid integer".into())
      })
    }
    Rule::Real => pair
      .as_str()
      .parse::<f64>()
      .map_err(|_| InterpreterError::EvaluationError("invalid real".into())),
    Rule::NumericValue => {
      let inner = pair.into_inner().next().unwrap();
      fast_eval_to_f64(inner)
    }
    Rule::FunctionCall => {
      let result = evaluate_function_call(pair)?;
      result
        .parse::<f64>()
        .map_err(|_| InterpreterError::EvaluationError("not a number".into()))
    }
    Rule::Identifier => {
      let id = pair.as_str();
      if let Some(StoredValue::Raw(val)) =
        ENV.with(|e| e.borrow().get(id).cloned())
      {
        return val.parse::<f64>().map_err(|_| {
          InterpreterError::EvaluationError("not a number".into())
        });
      }
      Err(InterpreterError::EvaluationError(format!(
        "undefined variable: {}",
        id
      )))
    }
    Rule::Expression | Rule::ExpressionNoImplicit => {
      let mut inner = pair.clone().into_inner();
      if let Some(first) = inner.next()
        && inner.next().is_none()
      {
        // Single child - try fast path
        return fast_eval_to_f64(first);
      }
      // Multi-child expression - fall back to full evaluation
      let value = evaluate_expression(pair)?;
      value
        .parse::<f64>()
        .map_err(|_| InterpreterError::EvaluationError("not a number".into()))
    }
    _ => {
      let value = evaluate_expression(pair)?;
      value
        .parse::<f64>()
        .map_err(|_| InterpreterError::EvaluationError("not a number".into()))
    }
  }
}

pub fn args_to_wonums(
  args_pairs: &[pest::iterators::Pair<Rule>],
) -> Result<Vec<WoNum>, InterpreterError> {
  args_pairs
    .iter()
    .map(|pair| {
      let value = fast_eval_to_f64(pair.clone())?;
      // Check if it's an integer
      if value.fract() == 0.0 && value.abs() < (i64::MAX as f64) {
        Ok(WoNum::Int(value as i128))
      } else {
        Ok(WoNum::Float(value))
      }
    })
    .collect()
}

pub fn evaluate_term(
  term: pest::iterators::Pair<Rule>,
) -> Result<f64, InterpreterError> {
  match term.as_rule() {
    Rule::Term => {
      let inner = term.into_inner().next().unwrap();
      evaluate_term(inner)
    }
    Rule::NumericValue => {
      let inner = term.into_inner().next().unwrap();
      evaluate_term(inner)
    }
    Rule::Constant => match term.as_str() {
      "Pi" => Ok(std::f64::consts::PI),
      _ => Err(InterpreterError::EvaluationError(format!(
        "Unknown constant: {}",
        term.as_str()
      ))),
    },
    Rule::Integer => {
      term.as_str().parse::<i64>().map(|n| n as f64).map_err(|_| {
        InterpreterError::EvaluationError("invalid integer literal".to_string())
      })
    }
    Rule::Real => term.as_str().parse::<f64>().map_err(|_| {
      InterpreterError::EvaluationError("invalid float literal".to_string())
    }),
    Rule::String => {
      // For string arguments in string functions, just return 0.0 (never used as a number)
      Ok(0.0)
    }
    Rule::Expression | Rule::ExpressionNoImplicit => evaluate_expression(term)
      .and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      }),
    Rule::FunctionCall => evaluate_function_call(term).and_then(|s| {
      if s == "True" {
        Ok(1.0)
      } else if s == "False" {
        Ok(0.0)
      } else {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      }
    }),
    Rule::Identifier => {
      match term.as_str() {
        "True" => Ok(1.0),
        "False" => Ok(0.0),
        "Now" => Err(InterpreterError::EvaluationError(
          "Identifier 'Now' cannot be directly used as a numeric value."
            .to_string(),
        )),
        id => {
          if let Some(StoredValue::Raw(val)) =
            ENV.with(|e| e.borrow().get(id).cloned())
          {
            return val
              .parse::<f64>()
              .map_err(|e| InterpreterError::EvaluationError(e.to_string()));
          }
          Ok(0.0) // unknown / non-numeric identifier
        }
      }
    }
    Rule::Slot => {
      // For slot (#), we'll return 1.0 as a default value when evaluated as a term
      // It will be replaced with the actual value in the anonymous function evaluation
      Ok(1.0)
    }
    Rule::List => Err(InterpreterError::EvaluationError(
      "Cannot evaluate a list as a numeric value".to_string(),
    )),
    Rule::PartExtract => {
      // Instead of error, evaluate as expression (so PartExtract can be handled at expression level)
      evaluate_expression(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication - evaluate and convert to number
      evaluate_pairs(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
    Rule::PostfixApplication | Rule::PostfixBase => {
      // Evaluate as expression and convert to number
      evaluate_expression(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
    Rule::SimpleAnonymousFunction
    | Rule::FunctionAnonymousFunction
    | Rule::ParenAnonymousFunction
    | Rule::ListAnonymousFunction => {
      // Anonymous functions when evaluated as a term (not applied) should error
      // or return 0.0 as a placeholder
      Ok(0.0)
    }
    Rule::CompoundExpression => {
      // Evaluate compound expression and convert to number
      evaluate_expression(term).and_then(|s| {
        s.parse::<f64>()
          .map_err(|e| InterpreterError::EvaluationError(e.to_string()))
      })
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Unexpected rule in Term: {:?}",
      term.as_rule()
    ))),
  }
}

pub fn evaluate_function_call(
  func_call: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  // 2. Store the full textual form of the call for later error messages
  let call_text = func_call.as_str().to_string(); // keep original text

  let func_call_str = func_call.as_str();

  // Count bracket groups in the source string
  let mut depth = 0;
  let mut bracket_count = 0;
  for c in func_call_str.chars() {
    match c {
      '[' => {
        if depth == 0 {
          bracket_count += 1;
        }
        depth += 1;
      }
      ']' => {
        depth -= 1;
      }
      _ => {}
    }
  }

  let is_chained = bracket_count > 1;

  let mut inner = func_call.into_inner();
  let func_name_pair = inner.next().unwrap();

  // Handle chained function calls like ReplaceRepeated[rule][expr] or Map[f][list]
  if is_chained && func_name_pair.as_rule() == Rule::Identifier {
    let func_name = func_name_pair.as_str();

    // For Map, handle the curried form: Map[f][list] -> Map[f, list]
    if func_name == "Map" {
      // Parse the argument groups from the source text
      let mut depth = 0;
      let mut groups: Vec<String> = Vec::new();
      let mut current = String::new();
      let mut in_args = false;

      // Skip the function name
      let after_name = &func_call_str[func_name.len()..];

      for c in after_name.chars() {
        match c {
          '[' => {
            if depth == 0 {
              in_args = true;
              current.clear();
            } else {
              current.push(c);
            }
            depth += 1;
          }
          ']' => {
            depth -= 1;
            if depth == 0 && in_args {
              groups.push(current.clone());
              in_args = false;
            } else {
              current.push(c);
            }
          }
          _ => {
            if in_args {
              current.push(c);
            }
          }
        }
      }

      if groups.len() >= 2 {
        let func_arg = groups[0].trim();
        let list_arg = groups[1].trim();

        // Build the standard Map[f, list] call
        let map_call = format!("Map[{}, {}]", func_arg, list_arg);
        return interpret(&map_call);
      }
    }

    // For ReplaceRepeated, we need to handle the curried form specially
    if func_name == "ReplaceRepeated" {
      // Parse the argument groups from the source text
      // ReplaceRepeated[rule][expr] -> extract rule and expr
      let mut depth = 0;
      let mut groups: Vec<String> = Vec::new();
      let mut current = String::new();
      let mut in_args = false;

      // Skip the function name
      let after_name = &func_call_str[func_name.len()..];

      for c in after_name.chars() {
        match c {
          '[' => {
            if depth == 0 {
              in_args = true;
              current.clear();
            } else {
              current.push(c);
            }
            depth += 1;
          }
          ']' => {
            depth -= 1;
            if depth == 0 && in_args {
              groups.push(current.clone());
              in_args = false;
            } else {
              current.push(c);
            }
          }
          _ => {
            if in_args {
              current.push(c);
            }
          }
        }
      }

      if groups.len() >= 2 {
        let rule_str = groups[0].trim();
        let expr_str = groups[1].trim();

        // Parse the rule (pattern -> replacement)
        if let Some(arrow_idx) = rule_str.find("->") {
          let pattern = rule_str[..arrow_idx].trim().to_string();
          let replacement = rule_str[arrow_idx + 2..].trim().to_string();

          // The expr_str may contain an undefined function, use as-is
          return apply_replace_repeated_direct(
            expr_str,
            &pattern,
            &replacement,
          );
        }
      }
    }
  }

  // treat identifier that refers to an association variable
  if func_name_pair.as_rule() == Rule::Identifier
    && let Some(StoredValue::Association(pairs)) =
      ENV.with(|e| e.borrow().get(func_name_pair.as_str()).cloned())
  {
    let args: Vec<_> = inner.clone().filter(|p| p.as_str() != ",").collect();

    // ---- single key lookup ---
    if args.len() == 1 {
      let arg_pair = args.into_iter().next().unwrap();
      let key_str = extract_string(arg_pair)?; // must be string
      for (k, v) in &pairs {
        if *k == key_str {
          return Ok(v.clone());
        }
      }
      return Err(InterpreterError::EvaluationError("Key not found".into()));
    }

    // ---- nested key lookup (multiple keys) ---
    if args.len() > 1 {
      // Get the first key
      let first_key = extract_string(args[0].clone())?;
      let mut current_value: Option<String> = None;

      // Find the value for the first key
      for (k, v) in &pairs {
        if *k == first_key {
          current_value = Some(v.clone());
          break;
        }
      }

      let mut current_value = current_value.ok_or_else(|| {
        InterpreterError::EvaluationError("Key not found".into())
      })?;

      // Iterate through remaining keys
      for arg in args.iter().skip(1) {
        let key_str = extract_string(arg.clone())?;

        // Parse the current value as an association
        // Association format: <|key1 -> val1, key2 -> val2|>
        if !current_value.starts_with("<|") || !current_value.ends_with("|>") {
          return Err(InterpreterError::EvaluationError(
            "Nested access requires association value".into(),
          ));
        }

        // Parse the inner association - clone to avoid borrow issues
        let inner_content =
          current_value[2..current_value.len() - 2].to_string();
        let mut found = false;
        let mut next_value = String::new();

        // Split by ", " but be careful about nested associations
        let mut depth = 0;
        let mut start = 0;
        let chars: Vec<char> = inner_content.chars().collect();
        let mut i = 0;

        while i < chars.len() {
          match chars[i] {
            '<' if i + 1 < chars.len() && chars[i + 1] == '|' => {
              depth += 1;
              i += 2;
              continue;
            }
            '|' if i + 1 < chars.len() && chars[i + 1] == '>' => {
              depth -= 1;
              i += 2;
              continue;
            }
            ',' if depth == 0 => {
              let item = inner_content[start..i].trim();
              if let Some((k, v)) = item.split_once(" -> ")
                && k.trim() == key_str
              {
                next_value = v.trim().to_string();
                found = true;
                break;
              }
              start = i + 1;
            }
            _ => {}
          }
          i += 1;
        }

        // Check the last item if not found yet
        if !found {
          let item = inner_content[start..].trim();
          if let Some((k, v)) = item.split_once(" -> ")
            && k.trim() == key_str
          {
            next_value = v.trim().to_string();
            found = true;
          }
        }

        if !found {
          return Err(InterpreterError::EvaluationError(
            "Key not found".into(),
          ));
        }

        current_value = next_value;
      }

      return Ok(current_value);
    }
  }

  // ----- anonymous function -------------------------------------------------
  if func_name_pair.as_rule() == Rule::SimpleAnonymousFunction {
    // inspect the parts that form the anonymous function
    let parts: Vec<_> = func_name_pair.clone().into_inner().collect();

    // fetch the argument list / single argument of the call  →  #&[arg]
    let arg = inner.next().ok_or_else(|| {
      InterpreterError::EvaluationError(
        "Expected arguments for anonymous function".to_string(),
      )
    })?;

    // ----------------------------------------------------------------------
    // simple identity function  (#&)
    // ----------------------------------------------------------------------
    if parts.len() == 1 {
      // just return the evaluated argument unchanged
      return evaluate_expression(arg);
    }

    // ----------------------------------------------------------------------
    // operator form  (# op term &)
    // (old behaviour, preserved)
    // ----------------------------------------------------------------------
    let operator = parts[1].as_str();
    let operand = parts[2].clone();

    // the existing code that:
    //  • expects `arg` to be a list,
    //  • iterates over its elements,
    //  • applies the operator to each element,
    //  • collects the results and returns them,
    // stays exactly as it was – just use the
    // `operator`, `operand`, and `arg` variables defined above.

    // Extract list from the argument
    let list = match arg.as_rule() {
      Rule::List => arg,
      Rule::Expression | Rule::ExpressionNoImplicit => {
        let mut inner_expr = arg.into_inner();
        if let Some(first) = inner_expr.next() {
          if first.as_rule() == Rule::List {
            first
          } else {
            return Err(InterpreterError::EvaluationError(
              "Anonymous function must be applied to a list".to_string(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "Empty expression in anonymous function arguments".to_string(),
          ));
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(format!(
          "Anonymous function must be applied to a list, got {:?}",
          arg.as_rule()
        )));
      }
    };

    let items: Vec<_> = list
      .into_inner()
      .filter(|item| item.as_str() != ",")
      .collect();
    let mut results = Vec::new();

    for item in items {
      let item_value = evaluate_term(item.clone())?;

      let result = match operator {
        "+" => item_value + evaluate_term(operand.clone())?,
        "-" => item_value - evaluate_term(operand.clone())?,
        "*" => item_value * evaluate_term(operand.clone())?,
        "/" => {
          let denominator = evaluate_term(operand.clone())?;
          if denominator == 0.0 {
            return Err(InterpreterError::EvaluationError(
              "Division by zero".to_string(),
            ));
          }
          item_value / denominator
        }
        "^" => item_value.powf(evaluate_term(operand.clone())?),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "Unsupported operator in anonymous function: {}",
            operator
          )));
        }
      };

      results.push(format_result(result));
    }

    return Ok(format!("{{{}}}", results.join(", ")));
  }

  // Handle regular function case
  let func_name = func_name_pair.as_str();
  // collect all arguments (ignore literal commas generated by the grammar)
  let args_pairs: Vec<pest::iterators::Pair<Rule>> =
    inner.filter(|p| p.as_str() != ",").collect();

  // ----- user-defined functions ------------------------------------------
  // Look up function definitions and find one matching the arity
  let func_defs = FUNC_DEFS.with(|m| m.borrow().get(func_name).cloned());
  if let Some(definitions) = func_defs {
    let call_arity = args_pairs.len();

    // Find a definition matching the call arity
    let matching_def = definitions
      .iter()
      .find(|(params, _)| params.len() == call_arity)
      .or_else(|| {
        // Fallback: if no exact match, try definition with 0 params and 1 arg
        // (for cases where parser didn't capture the param)
        if call_arity == 1 {
          definitions.iter().find(|(params, _)| params.is_empty())
        } else {
          None
        }
      });

    if let Some((params, body)) = matching_def {
      // Special case: 0 params but 1 argument (parser workaround)
      if params.is_empty() && call_arity == 1 {
        let param = "x".to_string();
        let val = evaluate_expression(args_pairs[0].clone())?;
        let prev = ENV.with(|e| {
          e.borrow_mut()
            .insert(param.clone(), StoredValue::Raw(val.clone()))
        });
        // Evaluate the AST directly using evaluate_expr_to_expr to avoid recursion
        let result_expr = evaluate_expr_to_expr(body)?;
        let result = expr_to_string(&result_expr);
        ENV.with(|e| {
          let mut env = e.borrow_mut();
          if let Some(v) = prev {
            env.insert(param, v);
          } else {
            env.remove(&param);
          }
        });
        return Ok(result);
      }

      // evaluate actual arguments
      let mut arg_vals = Vec::new();
      for p in &args_pairs {
        arg_vals.push(evaluate_expression(p.clone())?);
      }

      // save previous bindings, bind new ones
      let mut prev: Vec<(String, Option<StoredValue>)> = Vec::new();
      for (param, val) in params.iter().zip(arg_vals.iter()) {
        let pv = ENV.with(|e| {
          e.borrow_mut()
            .insert(param.clone(), StoredValue::Raw(val.clone()))
        });
        prev.push((param.clone(), pv));
      }

      // Evaluate the AST directly using evaluate_expr_to_expr to avoid recursion
      let result_expr = evaluate_expr_to_expr(body)?;
      let result = expr_to_string(&result_expr);

      // restore previous bindings
      for (param, old) in prev {
        ENV.with(|e| {
          let mut env = e.borrow_mut();
          if let Some(v) = old {
            env.insert(param, v);
          } else {
            env.remove(&param);
          }
        });
      }
      return Ok(result);
    } else {
      // No matching arity found
      let available_arities: Vec<_> =
        definitions.iter().map(|(p, _)| p.len()).collect();
      return Err(InterpreterError::EvaluationError(format!(
        "{} called with {} arguments; available arities: {:?}",
        func_name, call_arity, available_arities
      )));
    }
  }

  match func_name {
    "Module" => functions::scoping::module(&args_pairs),
    "DateString" => functions::date::date_string(&args_pairs),
    "Keys" => functions::association::keys(&args_pairs),
    "Values" => functions::association::values(&args_pairs),
    "KeyDropFrom" => functions::association::key_drop_from(&args_pairs),
    "KeyExistsQ" => functions::association::key_exists_q(&args_pairs),
    "Set" => {
      // --- arity -----------------------------------------------------------
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Set expects exactly 2 arguments".into(),
        ));
      }

      // --- extract variable name (first arg must be an identifier) ---------
      let var_pair = &args_pairs[0];
      let var_name = match var_pair.as_rule() {
        Rule::Identifier => var_pair.as_str().to_string(),
        Rule::Expression | Rule::ExpressionNoImplicit => {
          let mut inner = var_pair.clone().into_inner();
          if let Some(first) = inner.next() {
            if first.as_rule() == Rule::Identifier && inner.next().is_none() {
              first.as_str().to_string()
            } else {
              return Err(InterpreterError::EvaluationError(
                "First argument of Set must be an identifier".into(),
              ));
            }
          } else {
            return Err(InterpreterError::EvaluationError(
              "First argument of Set must be an identifier".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "First argument of Set must be an identifier".into(),
          ));
        }
      };

      // --- evaluate & store RHS -------------------------------------------
      let rhs_pair = &args_pairs[1];
      if rhs_pair.as_rule() == Rule::Association {
        let (pairs, disp) = eval_association(rhs_pair.clone())?;
        ENV.with(|e| {
          e.borrow_mut()
            .insert(var_name, StoredValue::Association(pairs))
        });
        Ok(disp)
      } else {
        let val = evaluate_expression(rhs_pair.clone())?;
        ENV.with(|e| {
          e.borrow_mut()
            .insert(var_name, StoredValue::Raw(val.clone()))
        });
        Ok(val)
      }
    }

    // Boolean Functions
    "And" => functions::boolean::and(&args_pairs),
    "Or" => functions::boolean::or(&args_pairs),
    "Xor" => functions::boolean::xor(&args_pairs),
    "Not" => functions::boolean::not(&args_pairs, &call_text),
    "SameQ" => functions::boolean::same_q(&args_pairs),
    "UnsameQ" => functions::boolean::unsame_q(&args_pairs),
    "If" => functions::boolean::if_condition(&args_pairs, &call_text),
    "Which" => functions::boolean::which(&args_pairs),
    "Do" => functions::boolean::do_loop(&args_pairs),
    "While" => functions::boolean::while_loop(&args_pairs),

    // Numeric Functions
    "Sin" => functions::numeric::sin(&args_pairs),
    "Prime" => functions::numeric::prime(&args_pairs),
    "Plus" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Plus expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Plus(wonums))
    }
    "Times" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Times expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Times(wonums))
    }
    "Minus" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Minus expects at least 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Minus(wonums))
    }
    "Subtract" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "Subtract expects exactly 2 arguments".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      let mut iter = wonums.into_iter();
      let a = iter.next().unwrap();
      let b = iter.next().unwrap();
      // Subtract[a, b] = a - b = a + (-b)
      evaluate_ast(AST::Plus(vec![a, -b]))
    }
    "Abs" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Abs expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Abs(wonums.into_iter().next().unwrap()))
    }
    "Sign" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sign expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Sign(wonums.into_iter().next().unwrap()))
    }
    "Sqrt" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Sqrt expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Sqrt(wonums.into_iter().next().unwrap()))
    }
    "Floor" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Floor expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Floor(wonums.into_iter().next().unwrap()))
    }
    "Ceiling" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Ceiling expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Ceiling(wonums.into_iter().next().unwrap()))
    }
    "Round" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Round expects exactly 1 argument".into(),
        ));
      }
      let wonums = args_to_wonums(&args_pairs)?;
      evaluate_ast(AST::Round(wonums.into_iter().next().unwrap()))
    }
    "Max" => functions::numeric::max(&args_pairs),
    "Min" => functions::numeric::min(&args_pairs),
    "Mod" => functions::numeric::modulo(&args_pairs),
    "Power" => functions::numeric::power(&args_pairs),
    "Factorial" => functions::numeric::factorial(&args_pairs),
    "GCD" => functions::numeric::gcd(&args_pairs),
    "LCM" => functions::numeric::lcm(&args_pairs),
    "Exp" => functions::numeric::exp(&args_pairs),
    "Log" => functions::numeric::log(&args_pairs),
    "Log10" => functions::numeric::log10(&args_pairs),
    "Log2" => functions::numeric::log2(&args_pairs),
    "Cos" => functions::numeric::cos(&args_pairs),
    "Tan" => functions::numeric::tan(&args_pairs),
    "ArcSin" => functions::numeric::arcsin(&args_pairs),
    "ArcCos" => functions::numeric::arccos(&args_pairs),
    "ArcTan" => functions::numeric::arctan(&args_pairs),
    "Quotient" => functions::numeric::quotient(&args_pairs),
    "N" => functions::numeric::numeric_eval(&args_pairs),
    "IntegerDigits" => functions::numeric::integer_digits(&args_pairs),
    "FromDigits" => functions::numeric::from_digits(&args_pairs),
    "FactorInteger" => functions::numeric::factor_integer(&args_pairs),
    "Re" => functions::numeric::re(&args_pairs),
    "Im" => functions::numeric::im(&args_pairs),
    "Conjugate" => functions::numeric::conjugate(&args_pairs),
    "Rationalize" => functions::numeric::rationalize(&args_pairs),
    "Arg" => functions::numeric::arg(&args_pairs),
    "Divisors" => functions::numeric::divisors(&args_pairs),
    "DivisorSigma" => functions::numeric::divisor_sigma(&args_pairs),
    "MoebiusMu" => functions::numeric::moebius_mu(&args_pairs),
    "EulerPhi" => functions::numeric::euler_phi(&args_pairs),
    "CoprimeQ" => functions::numeric::coprime_q(&args_pairs),

    // Calculus Functions
    "D" => functions::calculus::derivative(&args_pairs),
    "Integrate" => functions::calculus::integral(&args_pairs),

    "Equal" => functions::math::equal(&args_pairs),
    "Unequal" => functions::math::unequal(&args_pairs),
    "Greater" => functions::math::greater(&args_pairs),
    "GreaterEqual" => functions::math::greater_equal(&args_pairs),
    "Less" => functions::math::less(&args_pairs),
    "LessEqual" => functions::math::less_equal(&args_pairs),

    "NumberQ" => functions::predicate::number_q(&args_pairs),
    "IntegerQ" => functions::predicate::integer_q(&args_pairs),
    "EvenQ" => functions::predicate::even_odd_q(&args_pairs, "EvenQ"),
    "OddQ" => functions::predicate::even_odd_q(&args_pairs, "OddQ"),
    "ListQ" => functions::predicate::list_q(&args_pairs),
    "StringQ" => functions::predicate::string_q(&args_pairs),
    "AtomQ" => functions::predicate::atom_q(&args_pairs),
    "PrimeQ" => functions::predicate::prime_q(&args_pairs),
    "NumericQ" => functions::predicate::numeric_q(&args_pairs),
    "Positive" => functions::predicate::positive(&args_pairs),
    "Negative" => functions::predicate::negative(&args_pairs),
    "NonPositive" => functions::predicate::non_positive(&args_pairs),
    "NonNegative" => functions::predicate::non_negative(&args_pairs),
    "Divisible" => functions::predicate::divisible(&args_pairs),

    "RandomInteger" => functions::math::random_integer(&args_pairs),

    // String Functions
    "StringLength" => functions::string::string_length(&args_pairs),
    "StringTake" => functions::string::string_take(&args_pairs),
    "StringDrop" => functions::string::string_drop(&args_pairs),
    "StringJoin" => functions::string::string_join(&args_pairs),
    "StringSplit" => functions::string::string_split(&args_pairs),
    "StringStartsQ" => functions::string::string_starts_q(&args_pairs),
    "StringEndsQ" => functions::string::string_ends_q(&args_pairs),
    "StringReplace" => functions::string::string_replace(&args_pairs),
    "ToUpperCase" => functions::string::to_upper_case(&args_pairs),
    "ToLowerCase" => functions::string::to_lower_case(&args_pairs),
    "StringContainsQ" => functions::string::string_contains_q(&args_pairs),
    "Characters" => functions::string::characters(&args_pairs),
    "StringRiffle" => functions::string::string_riffle(&args_pairs),
    "StringPosition" => functions::string::string_position(&args_pairs),
    "StringMatchQ" => functions::string::string_match_q(&args_pairs),
    "StringReverse" => functions::string::string_reverse(&args_pairs),
    "StringRepeat" => functions::string::string_repeat(&args_pairs),
    "StringTrim" => functions::string::string_trim(&args_pairs),
    "StringCases" => functions::string::string_cases(&args_pairs),
    "ToString" => functions::string::to_string(&args_pairs),
    "ToExpression" => functions::string::to_expression(&args_pairs),
    "StringPadLeft" => functions::string::string_pad_left(&args_pairs),
    "StringPadRight" => functions::string::string_pad_right(&args_pairs),

    // List Functions
    "Map" => functions::list_helpers::map_list(&args_pairs),
    "First" | "Last" => {
      functions::list_helpers::first_or_last(func_name, &args_pairs)
    }
    "Rest" | "Most" => {
      functions::list_helpers::rest_or_most(func_name, &args_pairs)
    }
    "MemberQ" => functions::list::member_q(&args_pairs),
    "Take" => functions::list::take(&args_pairs),
    "Drop" => functions::list::drop(&args_pairs),
    "Append" => functions::list::append(&args_pairs),
    "Prepend" => functions::list::prepend(&args_pairs),
    "Part" => functions::list::part(&args_pairs),
    "Length" => functions::list::length(&args_pairs),
    "Reverse" => functions::list::reverse(&args_pairs),
    "Range" => functions::list::range(&args_pairs),
    "Join" => functions::list::join(&args_pairs),
    "Sort" => functions::list::sort(&args_pairs),
    "Insert" => functions::list::insert(&args_pairs),

    // Aggregation Functions
    "Total" => functions::list_helpers::total(&args_pairs),
    "Mean" => functions::list_helpers::mean(&args_pairs),
    "Median" => functions::list_helpers::median(&args_pairs),
    "Product" => functions::list_helpers::product(&args_pairs),
    "Accumulate" => functions::list_helpers::accumulate(&args_pairs),
    "Differences" => functions::list_helpers::differences(&args_pairs),
    "Divide" => {
      if args_pairs.len() != 2 {
        use std::io::{self, Write};
        println!(
          "\nDivide::argrx: Divide called with {} arguments; 2 arguments are expected.",
          args_pairs.len()
        );
        io::stdout().flush().ok();
        return Ok(call_text.to_string()); // return unevaluated expression
      }
      let wonums = args_to_wonums(&args_pairs)?;
      // Check for division by zero
      let divisor = &wonums[1];
      let is_zero = match divisor {
        WoNum::Int(i) => *i == 0,
        WoNum::Float(f) => *f == 0.0,
      };
      if is_zero {
        return Err(InterpreterError::EvaluationError(
          "Division by zero".into(),
        ));
      }
      evaluate_ast(AST::Divide(wonums))
    }

    // Basic Functions
    "Select" => functions::list_helpers::select(&args_pairs),
    "AllTrue" => functions::list_helpers::all_true(&args_pairs),
    "Flatten" => functions::list_helpers::flatten(&args_pairs),
    "Cases" => functions::list_helpers::cases(&args_pairs),
    "DeleteCases" => functions::list_helpers::delete_cases(&args_pairs),
    "MapThread" => functions::list_helpers::map_thread(&args_pairs),
    "Partition" => functions::list_helpers::partition(&args_pairs),
    "SortBy" => functions::list_helpers::sort_by(&args_pairs),
    "GroupBy" => functions::list_helpers::group_by(&args_pairs),
    "Array" => functions::list_helpers::array(&args_pairs),
    "Fold" => functions::list_helpers::fold(&args_pairs),
    "FoldList" => functions::list_helpers::fold_list(&args_pairs),
    "Nest" => functions::list_helpers::nest(&args_pairs),
    "NestList" => functions::list_helpers::nest_list(&args_pairs),
    "NestWhile" => functions::list_helpers::nest_while(&args_pairs),
    "NestWhileList" => functions::list_helpers::nest_while_list(&args_pairs),
    "Through" => functions::list_helpers::through(&args_pairs),
    "TakeLargest" => functions::list_helpers::take_largest(&args_pairs),
    "TakeSmallest" => functions::list_helpers::take_smallest(&args_pairs),
    "ArrayDepth" => functions::list_helpers::array_depth(&args_pairs),
    "Table" => functions::list_helpers::table(&args_pairs),
    "Position" => functions::list_helpers::position(&args_pairs),
    "Count" => functions::list_helpers::count(&args_pairs),
    "DeleteDuplicates" => {
      functions::list_helpers::delete_duplicates(&args_pairs)
    }
    "Union" => functions::list_helpers::union(&args_pairs),
    "Intersection" => functions::list_helpers::intersection(&args_pairs),
    "Complement" => functions::list_helpers::complement(&args_pairs),
    "ConstantArray" => functions::list_helpers::constant_array(&args_pairs),
    "Tally" => functions::list_helpers::tally(&args_pairs),
    "ReplacePart" => functions::list_helpers::replace_part(&args_pairs),
    "MinMax" => functions::list_helpers::min_max(&args_pairs),
    "PadLeft" => functions::list_helpers::pad_left(&args_pairs),
    "PadRight" => functions::list_helpers::pad_right(&args_pairs),
    "RotateLeft" => functions::list_helpers::rotate_left(&args_pairs),
    "RotateRight" => functions::list_helpers::rotate_right(&args_pairs),
    "Riffle" => functions::list_helpers::riffle(&args_pairs),
    "AnyTrue" => functions::list_helpers::any_true(&args_pairs),
    "NoneTrue" => functions::list_helpers::none_true(&args_pairs),
    "Transpose" => functions::list_helpers::transpose(&args_pairs),
    "Thread" => functions::list_helpers::thread(&args_pairs),
    "MapIndexed" => functions::list_helpers::map_indexed(&args_pairs),
    "FixedPoint" => functions::list_helpers::fixed_point(&args_pairs),
    "FixedPointList" => functions::list_helpers::fixed_point_list(&args_pairs),
    "Scan" => functions::list_helpers::scan(&args_pairs),
    "Gather" => functions::list_helpers::gather(&args_pairs),
    "GatherBy" => functions::list_helpers::gather_by(&args_pairs),
    "Split" => functions::list_helpers::split(&args_pairs),
    "SplitBy" => functions::list_helpers::split_by(&args_pairs),
    "FreeQ" => functions::list_helpers::free_q(&args_pairs),
    "Extract" => functions::list_helpers::extract(&args_pairs),
    "Catenate" => functions::list_helpers::catenate(&args_pairs),
    "TakeWhile" => functions::list_helpers::take_while(&args_pairs),
    "Apply" => functions::list_helpers::apply(&args_pairs),
    "Composition" => functions::list_helpers::composition(&args_pairs),
    "Identity" => functions::list_helpers::identity(&args_pairs),
    "Outer" => functions::list_helpers::outer(&args_pairs),
    "Inner" => functions::list_helpers::inner(&args_pairs),
    "Print" => functions::io::print(&args_pairs),

    // Replacement functions
    "ReplaceAll" => {
      if args_pairs.len() != 2 {
        return Err(InterpreterError::EvaluationError(
          "ReplaceAll expects exactly 2 arguments".into(),
        ));
      }
      let expr_str = evaluate_expression(args_pairs[0].clone())
        .unwrap_or_else(|_| args_pairs[0].as_str().to_string());
      let rule = args_pairs[1].clone();
      // Extract pattern and replacement from the rule (ReplacementRule)
      let mut rule_inner = rule.into_inner();
      let pattern = rule_inner
        .next()
        .map(|p| p.as_str().trim().to_string())
        .ok_or_else(|| {
          InterpreterError::EvaluationError("Invalid rule format".into())
        })?;
      let replacement = rule_inner
        .next()
        .map(|p| p.as_str().trim().to_string())
        .ok_or_else(|| {
          InterpreterError::EvaluationError("Invalid rule format".into())
        })?;
      apply_replace_all_direct(&expr_str, &pattern, &replacement)
    }

    "ReplaceRepeated" => {
      // ReplaceRepeated can be called as ReplaceRepeated[rule][expr]
      // or ReplaceRepeated[expr, rule]
      if args_pairs.len() == 1 {
        // Curried form: ReplaceRepeated[rule] returns a function
        // that will be applied via the next function call
        let rule_str = args_pairs[0].as_str().to_string();
        Ok(format!("ReplaceRepeated[{}]", rule_str))
      } else if args_pairs.len() == 2 {
        let expr_str = evaluate_expression(args_pairs[0].clone())
          .unwrap_or_else(|_| args_pairs[0].as_str().to_string());
        let rule = args_pairs[1].clone();
        // Extract pattern and replacement from the rule (ReplacementRule)
        let mut rule_inner = rule.into_inner();
        let pattern = rule_inner
          .next()
          .map(|p| p.as_str().trim().to_string())
          .ok_or_else(|| {
            InterpreterError::EvaluationError("Invalid rule format".into())
          })?;
        let replacement = rule_inner
          .next()
          .map(|p| p.as_str().trim().to_string())
          .ok_or_else(|| {
            InterpreterError::EvaluationError("Invalid rule format".into())
          })?;
        apply_replace_repeated_direct(&expr_str, &pattern, &replacement)
      } else {
        Err(InterpreterError::EvaluationError(
          "ReplaceRepeated expects 1 or 2 arguments".into(),
        ))
      }
    }

    // FullForm - returns the full form representation of an expression
    "FullForm" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "FullForm expects exactly 1 argument".into(),
        ));
      }
      // Get the raw expression and convert to full form
      let arg = &args_pairs[0];
      expr_to_full_form(arg.clone())
    }

    // Head - returns the head of an expression
    "Head" => {
      if args_pairs.len() != 1 {
        return Err(InterpreterError::EvaluationError(
          "Head expects exactly 1 argument".into(),
        ));
      }
      let arg = &args_pairs[0];
      get_head(arg.clone())
    }

    // Construct - constructs an expression from head and arguments
    "Construct" => {
      if args_pairs.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Construct expects at least 1 argument".into(),
        ));
      }
      // First argument is the head - try to evaluate, but use raw string if evaluation fails
      let head = match evaluate_expression(args_pairs[0].clone()) {
        Ok(result) => result,
        Err(InterpreterError::EvaluationError(e))
          if e.starts_with("Unknown function:") =>
        {
          // For unknown functions, use the raw string representation
          args_pairs[0].as_str().to_string()
        }
        Err(e) => return Err(e),
      };
      // Rest of the arguments
      let args: Vec<String> = args_pairs[1..]
        .iter()
        .map(|p| {
          match evaluate_expression(p.clone()) {
            Ok(result) => Ok(result),
            Err(InterpreterError::EvaluationError(e))
              if e.starts_with("Unknown function:") =>
            {
              // For unknown functions, use the raw string representation
              Ok(p.as_str().to_string())
            }
            Err(e) => Err(e),
          }
        })
        .collect::<Result<_, _>>()?;
      Ok(format!("{}[{}]", head, args.join(", ")))
    }

    // ClearAll - clear all definitions for symbols
    "ClearAll" => {
      for arg in &args_pairs {
        let name = arg.as_str().trim();
        // Remove from environment
        ENV.with(|e| {
          e.borrow_mut().remove(name);
        });
        // Remove from function definitions
        FUNC_DEFS.with(|f| {
          f.borrow_mut().remove(name);
        });
      }
      Ok("Null".to_string())
    }

    _ => Err(InterpreterError::EvaluationError(format!(
      "Unknown function: {}",
      func_name
    ))),
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

/// Convert an expression to its full form representation
fn expr_to_full_form(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  match expr.as_rule() {
    Rule::Expression | Rule::ExpressionNoImplicit => {
      let items: Vec<_> = expr.clone().into_inner().collect();

      // Single element expression - recurse
      if items.len() == 1 {
        return expr_to_full_form(items[0].clone());
      }

      // Check for operators
      if items.len() >= 3 {
        // Collect terms and operators
        let terms: Vec<_> = items.iter().step_by(2).cloned().collect();
        let ops: Vec<&str> = items
          .iter()
          .skip(1)
          .step_by(2)
          .map(|p| p.as_str())
          .collect();

        // Check if all operators are the same
        if !ops.is_empty() && ops.iter().all(|&o| o == ops[0]) {
          let op = ops[0];
          let head = match op {
            "+" => "Plus",
            "*" | "" => "Times", // empty string for implicit multiplication
            "-" => "Plus",       // Subtraction is Plus with negation
            "/" => "Times",      // Division is Times with Power[-1]
            "^" => "Power",
            "->" => "Rule",
            "==" => "Equal",
            "===" => "SameQ",
            "=!=" => "UnsameQ",
            "!=" => "Unequal",
            "<" => "Less",
            ">" => "Greater",
            "<=" => "LessEqual",
            ">=" => "GreaterEqual",
            "&&" => "And",
            "||" => "Or",
            "=" => {
              // Assignment - evaluate and return the result
              // This matches Wolfram's behavior where FullForm[a=b] returns b
              return evaluate_expression(expr);
            }
            ":=" => "SetDelayed",
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "Unknown operator: {}",
                op
              )));
            }
          };

          // Handle subtraction specially (convert to Plus with negation)
          if op == "-" {
            let mut args = Vec::new();
            for (i, term) in terms.iter().enumerate() {
              let term_str = expr_to_full_form(term.clone())?;
              if i == 0 {
                args.push(term_str);
              } else {
                // Negate subsequent terms
                args.push(format!("Times[-1, {}]", term_str));
              }
            }
            return Ok(format!("{}[{}]", head, args.join(", ")));
          }

          // Handle binary operators like Power
          if op == "^" && terms.len() == 2 {
            let base = expr_to_full_form(terms[0].clone())?;
            let exp = expr_to_full_form(terms[1].clone())?;
            return Ok(format!("Power[{}, {}]", base, exp));
          }

          // Handle Rule operator
          if op == "->" && terms.len() == 2 {
            let lhs = expr_to_full_form(terms[0].clone())?;
            let rhs = expr_to_full_form(terms[1].clone())?;
            return Ok(format!("Rule[{}, {}]", lhs, rhs));
          }

          // For n-ary operators like Plus and Times
          let args: Vec<String> = terms
            .into_iter()
            .map(|t| expr_to_full_form(t))
            .collect::<Result<_, _>>()?;
          return Ok(format!("{}[{}]", head, args.join(", ")));
        }
      }

      // Default: evaluate the expression
      evaluate_expression(expr)
    }
    Rule::Term => {
      let inner = expr.into_inner().next().unwrap();
      expr_to_full_form(inner)
    }
    Rule::FunctionCall => {
      let mut inner = expr.into_inner();
      let func_name = inner.next().unwrap().as_str();
      let args: Vec<String> = inner
        .filter(|p| p.as_str() != ",")
        .map(|p| expr_to_full_form(p))
        .collect::<Result<_, _>>()?;
      Ok(format!("{}[{}]", func_name, args.join(", ")))
    }
    Rule::List => {
      let items: Vec<String> = expr
        .into_inner()
        .filter(|p| p.as_str() != ",")
        .map(|p| expr_to_full_form(p))
        .collect::<Result<_, _>>()?;
      Ok(format!("List[{}]", items.join(", ")))
    }
    Rule::Identifier => Ok(expr.as_str().to_string()),
    Rule::Integer => Ok(expr.as_str().to_string()),
    Rule::Real => Ok(expr.as_str().to_string()),
    Rule::NumericValue => {
      let inner = expr.into_inner().next().unwrap();
      expr_to_full_form(inner)
    }
    Rule::ReplacementRule => {
      let mut inner = expr.into_inner();
      let lhs = inner.next().unwrap();
      let rhs = inner.next().unwrap();
      let lhs_str = expr_to_full_form(lhs)?;
      let rhs_str = expr_to_full_form(rhs)?;
      Ok(format!("Rule[{}, {}]", lhs_str, rhs_str))
    }
    Rule::ImplicitTimes => {
      // Implicit multiplication: x y z -> Times[x, y, z]
      let factors: Vec<String> = expr
        .into_inner()
        .map(|p| expr_to_full_form(p))
        .collect::<Result<_, _>>()?;
      Ok(format!("Times[{}]", factors.join(", ")))
    }
    _ => {
      // Default: just return the string representation
      Ok(expr.as_str().to_string())
    }
  }
}

/// Get the head of an expression
fn get_head(
  expr: pest::iterators::Pair<Rule>,
) -> Result<String, InterpreterError> {
  match expr.as_rule() {
    Rule::Expression | Rule::ExpressionNoImplicit => {
      let items: Vec<_> = expr.clone().into_inner().collect();

      // Single element expression - recurse
      if items.len() == 1 {
        return get_head(items[0].clone());
      }

      // Check for operators
      if items.len() >= 3 {
        // Look at the first operator
        let op = items[1].as_str();
        return Ok(
          match op {
            "+" => "Plus",
            "*" => "Times",
            "-" => "Plus", // Subtraction is treated as Plus with negation
            "/" => "Times",
            "^" => "Power",
            "->" => "Rule",
            "==" => "Equal",
            "===" => "SameQ",
            "=!=" => "UnsameQ",
            "!=" => "Unequal",
            "<" => "Less",
            ">" => "Greater",
            "<=" => "LessEqual",
            ">=" => "GreaterEqual",
            "&&" => "And",
            "||" => "Or",
            _ => {
              return Err(InterpreterError::EvaluationError(format!(
                "Unknown operator: {}",
                op
              )));
            }
          }
          .to_string(),
        );
      }

      // Default: evaluate and return Symbol head
      Ok("Symbol".to_string())
    }
    Rule::Term => {
      let inner = expr.into_inner().next().unwrap();
      get_head(inner)
    }
    Rule::FunctionCall => {
      let mut inner = expr.into_inner();
      let func_name = inner.next().unwrap().as_str();
      Ok(func_name.to_string())
    }
    Rule::List => Ok("List".to_string()),
    Rule::Identifier => Ok("Symbol".to_string()),
    Rule::Integer => Ok("Integer".to_string()),
    Rule::Real => Ok("Real".to_string()),
    Rule::NumericValue => {
      let inner = expr.into_inner().next().unwrap();
      get_head(inner)
    }
    Rule::String => Ok("String".to_string()),
    Rule::Association => Ok("Association".to_string()),
    Rule::ReplacementRule => Ok("Rule".to_string()),
    Rule::ImplicitTimes => Ok("Times".to_string()),
    _ => {
      // Default: return the rule name as head
      Ok(format!("{:?}", expr.as_rule()))
    }
  }
}
