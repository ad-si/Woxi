#[allow(unused_imports)]
use super::*;

/// Helper to construct an RGBColor Expr from numeric values.
/// Uses Integer for whole numbers (0, 1) and Real for fractional values.
fn rgb_color_expr(r: f64, g: f64, b: f64) -> Expr {
  fn num(v: f64) -> Expr {
    if v == 0.0 {
      Expr::Integer(0)
    } else if v == 1.0 {
      Expr::Integer(1)
    } else {
      Expr::Real(v)
    }
  }
  Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![num(r), num(g), num(b)],
  }
}

/// Helper to construct a GrayLevel Expr.
fn gray_level_expr(g: f64) -> Expr {
  fn num(v: f64) -> Expr {
    if v == 0.0 {
      Expr::Integer(0)
    } else if v == 1.0 {
      Expr::Integer(1)
    } else {
      Expr::Real(v)
    }
  }
  Expr::FunctionCall {
    name: "GrayLevel".to_string(),
    args: vec![num(g)],
  }
}

/// Map named color identifiers to their Wolfram Language evaluation result.
/// Basic colors → RGBColor[r, g, b] or GrayLevel[g]
/// Light* colors → RGBColor[r, g, b] or GrayLevel[g]
pub fn named_color_expr_pub(name: &str) -> Option<Expr> {
  named_color_expr(name)
}

fn named_color_expr(name: &str) -> Option<Expr> {
  Some(match name {
    // Basic colors
    "Red" => rgb_color_expr(1.0, 0.0, 0.0),
    "Green" => rgb_color_expr(0.0, 1.0, 0.0),
    "Blue" => rgb_color_expr(0.0, 0.0, 1.0),
    "Black" => gray_level_expr(0.0),
    "White" => gray_level_expr(1.0),
    "Gray" => gray_level_expr(0.5),
    "Cyan" => rgb_color_expr(0.0, 1.0, 1.0),
    "Magenta" => rgb_color_expr(1.0, 0.0, 1.0),
    "Yellow" => rgb_color_expr(1.0, 1.0, 0.0),
    "Brown" => rgb_color_expr(0.6, 0.4, 0.2),
    "Orange" => rgb_color_expr(1.0, 0.5, 0.0),
    "Pink" => rgb_color_expr(1.0, 0.5, 0.5),
    "Purple" => rgb_color_expr(0.5, 0.0, 0.5),
    // Light colors
    "LightRed" => rgb_color_expr(1.0, 0.85, 0.85),
    "LightBlue" => rgb_color_expr(0.87, 0.94, 1.0),
    "LightGreen" => rgb_color_expr(0.88, 1.0, 0.88),
    "LightGray" => gray_level_expr(0.85),
    "LightOrange" => rgb_color_expr(1.0, 0.9, 0.8),
    "LightYellow" => rgb_color_expr(1.0, 1.0, 0.85),
    "LightPurple" => rgb_color_expr(0.94, 0.88, 0.94),
    "LightCyan" => rgb_color_expr(0.9, 1.0, 1.0),
    "LightMagenta" => rgb_color_expr(1.0, 0.9, 1.0),
    "LightBrown" => rgb_color_expr(0.94, 0.91, 0.88),
    "LightPink" => rgb_color_expr(1.0, 0.85, 0.85),
    _ => return None,
  })
}

/// Find the index of a Label[tag] in a list of expressions where `tag` matches `goto_tag`.
/// Both the goto tag and each label's tag are compared after evaluation,
/// since neither Goto nor Label has HoldAll.
pub fn find_label_index(exprs: &[Expr], goto_tag: &Expr) -> Option<usize> {
  let tag_str = expr_to_string(goto_tag);
  for (i, expr) in exprs.iter().enumerate() {
    if let Expr::FunctionCall { name, args } = expr
      && name == "Label"
      && args.len() == 1
    {
      // Evaluate the label's tag to match Goto's evaluated tag
      let label_tag =
        evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
      if expr_to_string(&label_tag) == tag_str {
        return Some(i);
      }
    }
  }
  None
}

/// Prepare arguments for iterating functions (Sum, Product, NSum).
/// The body (args[0]) is kept unevaluated to preserve the iteration variable.
/// Iterator specs (args[1..]) have their bounds evaluated but variable names preserved.
pub fn prepare_iterating_function_args(
  args: &[Expr],
) -> Result<Vec<Expr>, InterpreterError> {
  let mut result = Vec::new();

  // Body stays unevaluated
  result.push(args[0].clone());

  // Process iterator specs
  for arg in &args[1..] {
    if let Expr::List(items) = arg {
      if items.is_empty() {
        result.push(arg.clone());
        continue;
      }
      let mut new_items = Vec::new();
      // First element is the variable name — keep unevaluated
      new_items.push(items[0].clone());
      // Remaining elements are bounds — evaluate them
      for item in &items[1..] {
        new_items.push(evaluate_expr_to_expr(item)?);
      }
      result.push(Expr::List(new_items));
    } else {
      result.push(evaluate_expr_to_expr(arg)?);
    }
  }

  Ok(result)
}

/// Early dispatch for FunctionCall in evaluate_expr — handles held functions
/// before argument evaluation. Returns Some(result) if handled, None otherwise.
#[inline(never)]
pub fn evaluate_expr_early_dispatch(
  name: &str,
  args: &[Expr],
) -> Result<Option<String>, InterpreterError> {
  match name {
    "If" if args.len() == 2 || args.len() == 3 => {
      let cond = evaluate_expr(&args[0])?;
      if cond == "True" {
        return Ok(Some(evaluate_expr(&args[1])?));
      } else if cond == "False" {
        if args.len() == 3 {
          return Ok(Some(evaluate_expr(&args[2])?));
        } else {
          return Ok(Some("Null".to_string()));
        }
      } else {
        let args_str: Vec<String> = args.iter().map(expr_to_string).collect();
        return Ok(Some(format!("If[{}]", args_str.join(", "))));
      }
    }
    "Function" | "Protect" | "Unprotect" | "Condition" | "MessageName"
    | "Information" | "Definition" | "FullDefinition" => {
      let result = evaluate_function_call_ast(name, args)?;
      return Ok(Some(expr_to_string(&result)));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      // HoldRest: evaluate first arg (filename), hold second (symbols)
      let filename_expr = evaluate_expr_to_expr(&args[0])?;
      let result =
        evaluate_function_call_ast(name, &[filename_expr, args[1].clone()])?;
      return Ok(Some(expr_to_string(&result)));
    }
    "CompoundExpression" => {
      if args.is_empty() {
        return Ok(Some("Null".to_string()));
      }
      let mut result = String::new();
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..args.len() {
          match evaluate_expr(&args[i]) {
            Ok(val) => result = val,
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(args, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
      }
      return Ok(Some(result));
    }
    "Pattern" if args.len() == 2 => {
      let result = evaluate_pattern_function_ast(args)?;
      return Ok(Some(expr_to_string(&result)));
    }
    "RuleDelayed" if args.len() == 2 => {
      return Ok(Some(expr_to_string(&evaluate_rule_delayed_ast(args)?)));
    }
    "AbsoluteTiming" | "Timing" if args.len() == 1 => {
      let start = std::time::Instant::now();
      let result = evaluate_expr(&args[0])?;
      let elapsed = start.elapsed().as_secs_f64();
      return Ok(Some(format!(
        "{{{}, {}}}",
        format_real_result(elapsed),
        result
      )));
    }
    "Sum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      let result = crate::functions::list_helpers_ast::sum_ast(&prepared)?;
      return Ok(Some(expr_to_string(&result)));
    }
    "Product" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      let result = crate::functions::list_helpers_ast::product_ast(&prepared)?;
      return Ok(Some(expr_to_string(&result)));
    }
    "NSum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      let result = crate::functions::math_ast::nsum_ast(&prepared)?;
      return Ok(Some(expr_to_string(&result)));
    }
    "NProduct" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      let result = crate::functions::math_ast::nproduct_ast(&prepared)?;
      return Ok(Some(expr_to_string(&result)));
    }
    _ => {}
  }
  Ok(None)
}

/// Early dispatch for FunctionCall in evaluate_expr_to_expr — handles held functions
/// before argument evaluation. Returns Some(result) if handled, None otherwise.
#[inline(never)]
pub fn evaluate_expr_to_expr_early_dispatch(
  name: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  match name {
    "Protect" | "Unprotect" | "Condition" | "MessageName" => {
      return Ok(Some(evaluate_function_call_ast(name, args)?));
    }
    #[cfg(not(target_arch = "wasm32"))]
    "Save" if args.len() == 2 => {
      // HoldRest: evaluate first arg (filename), hold second (symbols)
      let filename_expr = evaluate_expr_to_expr(&args[0])?;
      return Ok(Some(evaluate_function_call_ast(
        name,
        &[filename_expr, args[1].clone()],
      )?));
    }
    "CompoundExpression" => {
      if args.is_empty() {
        return Ok(Some(Expr::Identifier("Null".to_string())));
      }
      let mut result = Expr::Identifier("Null".to_string());
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..args.len() {
          match evaluate_expr_to_expr(&args[i]) {
            Ok(val) => result = val,
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(args, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
      }
      return Ok(Some(result));
    }
    "Pattern" if args.len() == 2 => {
      return Ok(Some(evaluate_pattern_function_ast(args)?));
    }
    "RuleDelayed" if args.len() == 2 => {
      return Ok(Some(evaluate_rule_delayed_ast(args)?));
    }
    "AbsoluteTiming" | "Timing" if args.len() == 1 => {
      let start = std::time::Instant::now();
      let result = evaluate_expr_to_expr(&args[0])?;
      let elapsed = start.elapsed().as_secs_f64();
      return Ok(Some(Expr::List(vec![Expr::Real(elapsed), result])));
    }
    "Sum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::list_helpers_ast::sum_ast(
        &prepared,
      )?));
    }
    "Product" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::list_helpers_ast::product_ast(
        &prepared,
      )?));
    }
    "NSum" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::math_ast::nsum_ast(&prepared)?));
    }
    "NProduct" if args.len() >= 2 => {
      let prepared = prepare_iterating_function_args(args)?;
      return Ok(Some(crate::functions::math_ast::nproduct_ast(&prepared)?));
    }
    _ => {}
  }
  Ok(None)
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
        // Handle named colors (Red → RGBColor[1, 0, 0], etc.)
        if let Some(color_expr) = named_color_expr(name) {
          return Ok(expr_to_string(&color_expr));
        }
        // Return as symbolic
        Ok(name.clone())
      }
    }
    Expr::SlotSequence(n) => Ok(format!("##{}", n)),
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
      // Try early dispatch for held functions
      if let Some(result) = evaluate_expr_early_dispatch(name, args)? {
        return Ok(result);
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
          if left_val == "False" || right_val == "False" {
            Ok("False".to_string())
          } else if left_val == "True" && right_val == "True" {
            Ok("True".to_string())
          } else if left_val == "True" {
            Ok(right_val)
          } else if right_val == "True" {
            Ok(left_val)
          } else {
            Ok(format!("{} && {}", left_val, right_val))
          }
        }
        BinaryOperator::Or => {
          if left_val == "True" || right_val == "True" {
            Ok("True".to_string())
          } else if left_val == "False" && right_val == "False" {
            Ok("False".to_string())
          } else if left_val == "False" {
            Ok(right_val)
          } else if right_val == "False" {
            Ok(left_val)
          } else {
            Ok(format!("{} || {}", left_val, right_val))
          }
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
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..exprs.len() {
          match evaluate_expr(&exprs[i]) {
            Ok(val) => result = val,
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(exprs, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
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
    Expr::PatternTest { name, test } => {
      Ok(expr_to_string(&Expr::PatternTest {
        name: name.clone(),
        test: test.clone(),
      }))
    }
    Expr::Raw(s) => {
      // Fallback: parse to AST and evaluate to avoid interpret() recursion
      let parsed = string_to_expr(s)?;
      let result = evaluate_expr_to_expr(&parsed)?;
      Ok(expr_to_string(&result))
    }
    Expr::Image { .. } => Ok("-Image-".to_string()),
    Expr::Graphics { is_3d, .. } => {
      Ok(if *is_3d { "-Graphics3D-" } else { "-Graphics-" }.to_string())
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
  // Trampoline loop: tail-recursive calls return TailCall instead of
  // recursing, so the call stack stays flat regardless of recursion depth.
  let mut current = expr.clone();
  loop {
    match evaluate_expr_to_expr_inner(&current) {
      Err(InterpreterError::TailCall(next)) => {
        current = *next;
      }
      result => return result,
    }
  }
}

pub fn evaluate_expr_to_expr_inner(
  expr: &Expr,
) -> Result<Expr, InterpreterError> {
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
        #[cfg(not(target_arch = "wasm32"))]
        if name == "Now" {
          use chrono::Local;
          let now = Local::now();
          let seconds = now
            .format("%S%.f")
            .to_string()
            .parse::<f64>()
            .unwrap_or(0.0);
          let tz_offset_hours = now.offset().local_minus_utc() as f64 / 3600.0;
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(vec![
                Expr::Integer(
                  now.format("%Y").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  now.format("%m").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  now.format("%d").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  now.format("%H").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  now.format("%M").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Real(seconds),
              ]),
              Expr::String("Instant".to_string()),
              Expr::String("Gregorian".to_string()),
              Expr::Real(tz_offset_hours),
            ],
          });
        }
        // Handle Today/Tomorrow/Yesterday → DateObject[{y, m, d}, Day]
        #[cfg(not(target_arch = "wasm32"))]
        if name == "Today" || name == "Tomorrow" || name == "Yesterday" {
          use chrono::{Duration, Local};
          let now = Local::now();
          let date = match name.as_str() {
            "Tomorrow" => now + Duration::days(1),
            "Yesterday" => now - Duration::days(1),
            _ => now,
          };
          return Ok(Expr::FunctionCall {
            name: "DateObject".to_string(),
            args: vec![
              Expr::List(vec![
                Expr::Integer(
                  date.format("%Y").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  date.format("%m").to_string().parse::<i128>().unwrap(),
                ),
                Expr::Integer(
                  date.format("%d").to_string().parse::<i128>().unwrap(),
                ),
              ]),
              Expr::String("Day".to_string()),
            ],
          });
        }
        // Handle system $ variables
        if name.starts_with('$')
          && let Some(val) = get_system_variable(name)
        {
          return Ok(val);
        }
        // Handle named colors (Red → RGBColor[1, 0, 0], etc.)
        if let Some(color_expr) = named_color_expr(name) {
          return Ok(color_expr);
        }
        // Return as symbolic identifier
        Ok(Expr::Identifier(name.clone()))
      }
    }
    Expr::Slot(n) => {
      // Slots should be replaced before evaluation
      Ok(Expr::Slot(*n))
    }
    Expr::SlotSequence(n) => Ok(Expr::SlotSequence(*n)),
    Expr::Constant(name) => Ok(Expr::Constant(name.clone())),
    Expr::List(items) => {
      let mut evaluated: Vec<Expr> = Vec::with_capacity(items.len());
      for item in items {
        let val = evaluate_expr_to_expr(item)?;
        if matches!(&val, Expr::Identifier(s) if s == "Nothing") {
          continue;
        }
        // Flatten Sequence in lists
        if let Expr::FunctionCall { name, args } = &val
          && name == "Sequence"
        {
          evaluated.extend(args.iter().cloned());
          continue;
        }
        evaluated.push(val);
      }
      Ok(Expr::List(evaluated))
    }
    Expr::FunctionCall { name, args } => {
      // Special handling for If - lazy evaluation of branches
      if name == "If" && (args.len() == 2 || args.len() == 3) {
        let cond = evaluate_expr_to_expr(&args[0])?;
        if matches!(&cond, Expr::Identifier(s) if s == "True") {
          return Err(InterpreterError::TailCall(Box::new(args[1].clone())));
        } else if matches!(&cond, Expr::Identifier(s) if s == "False") {
          if args.len() == 3 {
            return Err(InterpreterError::TailCall(Box::new(args[2].clone())));
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
      // Special handling for Module/Block/Assuming - don't evaluate args (body needs local bindings first)
      if name == "Module" {
        return module_ast(args);
      }
      if name == "Block" {
        return block_ast(args);
      }
      if name == "Assuming" && args.len() == 2 {
        return assuming_ast(args);
      }
      // Special handling for Set - first arg must be identifier or Part, second gets evaluated
      if name == "Set" && args.len() == 2 {
        return set_ast(&args[0], &args[1]);
      }
      // Special handling for SetDelayed - stores function definitions
      if name == "SetDelayed" && args.len() == 2 {
        return set_delayed_ast(&args[0], &args[1]);
      }
      // Special handling for TagSetDelayed - stores upvalue definitions
      if name == "TagSetDelayed" && args.len() == 3 {
        return tag_set_delayed_ast(&args[0], &args[1], &args[2], false);
      }
      // Special handling for TagSet - stores evaluated upvalue definitions
      if name == "TagSet" && args.len() == 3 {
        return tag_set_delayed_ast(&args[0], &args[1], &args[2], true);
      }
      // Special handling for UpSet - automatically determines tags from LHS
      if name == "UpSet" && args.len() == 2 {
        return upset_ast(&args[0], &args[1]);
      }
      // Special handling for UpSetDelayed - like UpSet but delayed (no RHS evaluation)
      if name == "UpSetDelayed" && args.len() == 2 {
        return upset_delayed_ast(&args[0], &args[1]);
      }
      // Special handling for Increment/Decrement - x++ / x--
      // and PreIncrement/PreDecrement - ++x / --x
      if (name == "Increment"
        || name == "Decrement"
        || name == "PreIncrement"
        || name == "PreDecrement")
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
        let delta = if name == "Increment" || name == "PreIncrement" {
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
        // Post-increment/decrement returns old value; pre returns new value
        if name == "PreIncrement" || name == "PreDecrement" {
          return Ok(new_val);
        }
        return Ok(current_val);
      }
      // Special handling for Unset - x =. (removes definition)
      if name == "Unset" && args.len() == 1 {
        if let Expr::Identifier(var_name) = &args[0] {
          ENV.with(|e| {
            e.borrow_mut().remove(var_name);
          });
        }
        return Ok(Expr::Identifier("Null".to_string()));
      }
      // Special handling for Information/Definition/FullDefinition - hold argument unevaluated
      if (name == "Information"
        || name == "Definition"
        || name == "FullDefinition")
        && args.len() == 1
      {
        return dispatch::evaluate_function_call_ast(name, args);
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
          e.borrow_mut()
            .insert(var_name.clone(), StoredValue::ExprVal(new_val.clone()));
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
        let mut current_val = match current {
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
        let new_val = match &mut current_val {
          Expr::List(items) => {
            let mut items = std::mem::take(items);
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
      // Label[tag] is not evaluated — it stays symbolic.
      // CompoundExpression handles it via find_label_index on the AST.
      // Special handling for Goto[tag] — evaluates the tag then raises GotoSignal
      if name == "Goto" && args.len() == 1 {
        let tag = evaluate_expr_to_expr(&args[0])?;
        return Err(InterpreterError::GotoSignal(Box::new(tag)));
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
      // Special handling for Abort[] - abort computation
      if name == "Abort" && args.is_empty() {
        return Err(InterpreterError::Abort);
      }
      // Quit[] / Exit[] - terminate the process
      if (name == "Quit" || name == "Exit") && args.len() <= 1 {
        #[cfg(not(target_arch = "wasm32"))]
        {
          let code = if args.len() == 1 {
            if let Ok(val) = evaluate_expr_to_expr(&args[0]) {
              crate::functions::math_ast::try_eval_to_f64(&val)
                .map(|f| f as i32)
                .unwrap_or(0)
            } else {
              0
            }
          } else {
            0
          };
          std::process::exit(code);
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Err(InterpreterError::Abort);
        }
      }
      // Interrupt[] behaves like Abort[] in batch mode
      if name == "Interrupt" && args.is_empty() {
        return Err(InterpreterError::Abort);
      }
      // Pause[n] - sleep for n seconds and return Null
      if name == "Pause" && args.len() == 1 {
        if let Ok(val) = evaluate_expr_to_expr(&args[0])
          && let Some(secs) = crate::functions::math_ast::try_eval_to_f64(&val)
          && secs > 0.0
        {
          std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        }
        return Ok(Expr::Identifier("Null".to_string()));
      }
      // Special handling for CheckAbort[expr, failexpr]
      if name == "CheckAbort" && args.len() == 2 {
        match evaluate_expr_to_expr(&args[0]) {
          Ok(result) => return Ok(result),
          Err(InterpreterError::Abort) => {
            return Err(InterpreterError::TailCall(Box::new(args[1].clone())));
          }
          Err(e) => return Err(e),
        }
      }
      // Special handling for Check[expr, failexpr]
      if name == "Check" && args.len() == 2 {
        let warnings_before = crate::get_captured_warnings().len();
        match evaluate_expr_to_expr(&args[0]) {
          Ok(result) => {
            let warnings_after = crate::get_captured_warnings().len();
            if warnings_after > warnings_before {
              return Err(InterpreterError::TailCall(Box::new(
                args[1].clone(),
              )));
            }
            return Ok(result);
          }
          Err(_) => {
            return Err(InterpreterError::TailCall(Box::new(args[1].clone())));
          }
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
        || name == "Clear"
        || name == "Hold"
        || name == "HoldForm"
        || name == "HoldComplete"
        || name == "Unevaluated"
        || name == "ValueQ"
        || name == "Reap"
        || name == "Plot"
        || name == "Plot3D"
        || name == "Graphics"
        || name == "ParametricPlot"
        || name == "PolarPlot"
        || name == "DensityPlot"
        || name == "ContourPlot"
        || name == "RegionPlot"
        || name == "StreamPlot"
        || name == "VectorPlot"
        || name == "StreamDensityPlot"
        || name == "Show"
        || name == "GraphicsRow"
        || name == "GraphicsColumn"
        || name == "GraphicsGrid"
      {
        // Flatten Sequence even in held args (unless SequenceHold)
        let args = flatten_sequences(name, args);
        // Pass unevaluated args to the function dispatcher
        return evaluate_function_call_ast(name, &args);
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
      // Variable holds a symbol name (e.g. t = Flatten) — re-dispatch as that function
      if let Some(resolved_name) = resolve_identifier_to_func_name(name) {
        return Err(InterpreterError::TailCall(Box::new(Expr::FunctionCall {
          name: resolved_name,
          args: args.to_vec(),
        })));
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

      // Try early dispatch for held functions
      if let Some(result) = evaluate_expr_to_expr_early_dispatch(name, args)? {
        return Ok(result);
      }

      // Evaluate arguments
      let evaluated_args: Vec<Expr> = args
        .iter()
        .map(evaluate_expr_to_expr)
        .collect::<Result<_, _>>()?;
      // Flatten Sequence arguments (unless function has SequenceHold attribute)
      let evaluated_args = flatten_sequences(name, &evaluated_args);
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
            // 0 - x => -x via times_ast for proper distribution
            crate::functions::math_ast::times_ast(&[
              Expr::Integer(-1),
              right_val,
            ])
          } else if crate::functions::quantity_ast::is_quantity(&left_val)
            .is_some()
            || crate::functions::quantity_ast::is_quantity(&right_val).is_some()
          {
            // Quantity subtraction: delegate to subtract_ast → Plus[a, Times[-1, b]]
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
          } else {
            // Use subtract_ast for proper distribution of -1 over Plus
            crate::functions::math_ast::subtract_ast(&[left_val, right_val])
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
          crate::functions::boolean_ast::and_ast(&[left_val, right_val])
        }
        BinaryOperator::Or => {
          crate::functions::boolean_ast::or_ast(&[left_val, right_val])
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
          // Use times_ast for proper distribution: -(a+b) → -a - b
          crate::functions::math_ast::times_ast(&[Expr::Integer(-1), val])
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
            // SameQ tests structural identity, not numeric equality.
            // Integer[1] !== Real[1.] even though they are numerically equal.
            expr_to_string(left) == expr_to_string(right)
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
            // UnsameQ tests structural non-identity, not numeric inequality.
            expr_to_string(left) != expr_to_string(right)
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
      let mut start_index = 0;
      'goto_loop: loop {
        for i in start_index..exprs.len() {
          match evaluate_expr_to_expr(&exprs[i]) {
            Ok(val) => result = val,
            Err(InterpreterError::GotoSignal(tag)) => {
              if let Some(label_idx) = find_label_index(exprs, &tag) {
                start_index = label_idx + 1;
                continue 'goto_loop;
              } else {
                return Err(InterpreterError::GotoSignal(tag));
              }
            }
            Err(e) => return Err(e),
          }
        }
        break;
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
      let result = apply_replace_all_ast(&evaluated_expr, &evaluated_rules)?;
      Err(InterpreterError::TailCall(Box::new(result)))
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
        if let Expr::FunctionCall { name, args } = idx
          && name == "Span"
          && (args.len() == 2 || args.len() == 3)
        {
          let mut evaluated_args = Vec::with_capacity(args.len());
          for arg in args {
            if matches!(arg, Expr::Identifier(s) if s == "All") {
              evaluated_args.push(arg.clone());
            } else {
              evaluated_args.push(evaluate_expr_to_expr(arg)?);
            }
          }
          indices.push(Expr::FunctionCall {
            name: "Span".to_string(),
            args: evaluated_args,
          });
        } else {
          indices.push(evaluate_expr_to_expr(idx)?);
        }
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
    Expr::PatternTest { name, test } => Ok(Expr::PatternTest {
      name: name.clone(),
      test: test.clone(),
    }),
    Expr::Raw(s) => {
      // Fallback: parse and evaluate the raw string
      let parsed = string_to_expr(s)?;
      Err(InterpreterError::TailCall(Box::new(parsed)))
    }
    Expr::Image { .. } => Ok(expr.clone()),
    Expr::Graphics { .. } => Ok(expr.clone()),
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
    | Expr::Slot(_)
    | Expr::SlotSequence(_) => false,
    Expr::List(items) => items.iter().any(has_free_symbols),
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbols(left) || has_free_symbols(right)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbols(operand),
    Expr::FunctionCall { name, args } => {
      has_free_symbols(&Expr::Identifier(name.clone()))
        || args.iter().any(has_free_symbols)
    }
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
