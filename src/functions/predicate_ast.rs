//! AST-native predicate functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper to create boolean result
fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// NumberQ[expr] - Tests if the expression is a number
pub fn number_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumberQ expects exactly 1 argument".into(),
    ));
  }
  let is_number = matches!(&args[0], Expr::Integer(_) | Expr::Real(_));
  Ok(bool_expr(is_number))
}

/// IntegerQ[expr] - Tests if the expression is an integer (not a real like 3.0)
pub fn integer_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IntegerQ expects exactly 1 argument".into(),
    ));
  }
  // Only Expr::Integer is considered an integer, not Expr::Real even if it's 3.0
  let is_integer = matches!(&args[0], Expr::Integer(_));
  Ok(bool_expr(is_integer))
}

/// EvenQ[n] - Tests if a number is even
pub fn even_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EvenQ expects exactly 1 argument".into(),
    ));
  }
  let is_even = match &args[0] {
    Expr::Integer(n) => n % 2 == 0,
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        (*f as i64) % 2 == 0
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };
  Ok(bool_expr(is_even))
}

/// OddQ[n] - Tests if a number is odd
pub fn odd_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OddQ expects exactly 1 argument".into(),
    ));
  }
  let is_odd = match &args[0] {
    Expr::Integer(n) => n % 2 != 0,
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        (*f as i64) % 2 != 0
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };
  Ok(bool_expr(is_odd))
}

/// ListQ[expr] - Tests if the expression is a list
pub fn list_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ListQ expects exactly 1 argument".into(),
    ));
  }
  let is_list = matches!(&args[0], Expr::List(_));
  Ok(bool_expr(is_list))
}

/// StringQ[expr] - Tests if the expression is a string
pub fn string_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StringQ expects exactly 1 argument".into(),
    ));
  }
  let is_string = matches!(&args[0], Expr::String(_));
  Ok(bool_expr(is_string))
}

/// AtomQ[expr] - Tests if the expression is atomic (not compound)
pub fn atom_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AtomQ expects exactly 1 argument".into(),
    ));
  }
  let is_atom = matches!(
    &args[0],
    Expr::Integer(_)
      | Expr::Real(_)
      | Expr::String(_)
      | Expr::Identifier(_)
      | Expr::Constant(_)
  );
  Ok(bool_expr(is_atom))
}

/// NumericQ[expr] - Tests if the expression is numeric (evaluates to a number)
pub fn numeric_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NumericQ expects exactly 1 argument".into(),
    ));
  }
  // Same as NumberQ for evaluated expressions
  let is_numeric = matches!(&args[0], Expr::Integer(_) | Expr::Real(_));
  Ok(bool_expr(is_numeric))
}

/// PositiveQ[x] - Tests if x is a positive number
pub fn positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PositiveQ expects exactly 1 argument".into(),
    ));
  }
  let is_positive = match &args[0] {
    Expr::Integer(n) => *n > 0,
    Expr::Real(f) => *f > 0.0,
    _ => false,
  };
  Ok(bool_expr(is_positive))
}

/// NegativeQ[x] - Tests if x is a negative number
pub fn negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NegativeQ expects exactly 1 argument".into(),
    ));
  }
  let is_negative = match &args[0] {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    _ => false,
  };
  Ok(bool_expr(is_negative))
}

/// NonPositiveQ[x] - Tests if x is <= 0
pub fn non_positive_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonPositiveQ expects exactly 1 argument".into(),
    ));
  }
  let is_non_positive = match &args[0] {
    Expr::Integer(n) => *n <= 0,
    Expr::Real(f) => *f <= 0.0,
    _ => false,
  };
  Ok(bool_expr(is_non_positive))
}

/// NonNegativeQ[x] - Tests if x is >= 0
pub fn non_negative_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NonNegativeQ expects exactly 1 argument".into(),
    ));
  }
  let is_non_negative = match &args[0] {
    Expr::Integer(n) => *n >= 0,
    Expr::Real(f) => *f >= 0.0,
    _ => false,
  };
  Ok(bool_expr(is_non_negative))
}

/// PrimeQ[n] - Tests if n is a prime number
pub fn prime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimeQ expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        *f as i128
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };

  let is_prime = if n <= 1 {
    false
  } else if n <= 3 {
    true
  } else if n % 2 == 0 || n % 3 == 0 {
    false
  } else {
    let mut i = 5i128;
    let mut result = true;
    while i * i <= n {
      if n % i == 0 || n % (i + 2) == 0 {
        result = false;
        break;
      }
      i += 6;
    }
    result
  };
  Ok(bool_expr(is_prime))
}

/// CompositeQ[n] - Tests if n is a composite (non-prime > 1) number
pub fn composite_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CompositeQ expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) => {
      if f.fract() == 0.0 {
        *f as i128
      } else {
        return Ok(bool_expr(false));
      }
    }
    _ => return Ok(bool_expr(false)),
  };

  // Composite numbers are > 1 and not prime
  if n <= 1 {
    return Ok(bool_expr(false));
  }

  // Check if it's prime (if so, it's not composite)
  let is_prime = if n <= 3 {
    true
  } else if n % 2 == 0 || n % 3 == 0 {
    false
  } else {
    let mut i = 5i128;
    let mut result = true;
    while i * i <= n {
      if n % i == 0 || n % (i + 2) == 0 {
        result = false;
        break;
      }
      i += 6;
    }
    result
  };
  Ok(bool_expr(!is_prime))
}

/// AssociationQ[expr] - Tests if the expression is an association
pub fn association_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AssociationQ expects exactly 1 argument".into(),
    ));
  }
  let is_assoc = matches!(&args[0], Expr::Association(_));
  Ok(bool_expr(is_assoc))
}

/// MemberQ[list, elem] - Tests if elem is a member of list
pub fn member_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "MemberQ expects exactly 2 arguments".into(),
    ));
  }
  let list = match &args[0] {
    Expr::List(items) => items,
    _ => return Ok(bool_expr(false)),
  };

  let target_str = crate::syntax::expr_to_string(&args[1]);
  for item in list {
    if crate::syntax::expr_to_string(item) == target_str {
      return Ok(bool_expr(true));
    }
  }
  Ok(bool_expr(false))
}

/// FreeQ[expr, form] - Tests if expr is free of form
pub fn free_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "FreeQ expects exactly 2 arguments".into(),
    ));
  }

  let form_str = crate::syntax::expr_to_string(&args[1]);

  fn contains_form(expr: &Expr, form: &str) -> bool {
    if crate::syntax::expr_to_string(expr) == form {
      return true;
    }
    match expr {
      Expr::List(items) => items.iter().any(|e| contains_form(e, form)),
      Expr::FunctionCall { args, .. } => {
        args.iter().any(|e| contains_form(e, form))
      }
      Expr::BinaryOp { left, right, .. } => {
        contains_form(left, form) || contains_form(right, form)
      }
      Expr::UnaryOp { operand, .. } => contains_form(operand, form),
      _ => false,
    }
  }

  Ok(bool_expr(!contains_form(&args[0], &form_str)))
}

/// Divisible[n, m] - Tests if n is divisible by m
/// Returns unevaluated if arguments are not exact numbers (non-integer Reals)
pub fn divisible_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Divisible expects exactly 2 arguments".into(),
    ));
  }

  // Check if first argument is a non-exact number (Real with fractional part)
  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) if f.fract() == 0.0 => *f as i128,
    Expr::Real(_) => {
      // Non-exact number - return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Check if second argument is a non-exact number
  let m = match &args[1] {
    Expr::Integer(m) => *m,
    Expr::Real(f) if f.fract() == 0.0 => *f as i128,
    Expr::Real(_) => {
      // Non-exact number - return unevaluated
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisible".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if m == 0 {
    return Err(InterpreterError::EvaluationError(
      "Divisible: divisor cannot be zero".into(),
    ));
  }

  Ok(bool_expr(n % m == 0))
}

/// Head[expr] - Returns the head of an expression
pub fn head_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Head expects exactly 1 argument".into(),
    ));
  }
  let head = match &args[0] {
    Expr::Integer(_) => "Integer",
    Expr::Real(_) => "Real",
    Expr::String(_) => "String",
    Expr::Identifier(_) => "Symbol",
    Expr::List(_) => "List",
    Expr::Association(_) => "Association",
    Expr::FunctionCall { name, .. } => {
      return Ok(Expr::Identifier(name.clone()));
    }
    Expr::Rule { .. } => "Rule",
    Expr::RuleDelayed { .. } => "RuleDelayed",
    Expr::BinaryOp { op, .. } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Plus => "Plus",
        BinaryOperator::Minus => "Plus", // Minus is represented as Plus internally
        BinaryOperator::Times => "Times",
        BinaryOperator::Divide => "Times", // Divide is represented as Times internally
        BinaryOperator::Power => "Power",
        BinaryOperator::And => "And",
        BinaryOperator::Or => "Or",
        BinaryOperator::StringJoin => "StringJoin",
      }
    }
    Expr::UnaryOp { op, .. } => {
      use crate::syntax::UnaryOperator;
      match op {
        UnaryOperator::Minus => "Times",
        UnaryOperator::Not => "Not",
      }
    }
    Expr::Comparison { .. } => "Comparison",
    Expr::Map { .. } => "Map",
    Expr::Apply { .. } => "Apply",
    Expr::Part { .. } => "Part",
    Expr::Function { .. } => "Function",
    Expr::Pattern { .. } => "Pattern",
    _ => "Symbol",
  };
  Ok(Expr::Identifier(head.to_string()))
}

/// Length[list] - Returns the length of a list
pub fn length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Length expects exactly 1 argument".into(),
    ));
  }
  let len = match &args[0] {
    Expr::List(items) => items.len() as i128,
    Expr::String(s) => s.chars().count() as i128,
    Expr::Association(items) => items.len() as i128,
    _ => 0,
  };
  Ok(Expr::Integer(len))
}

/// Depth[expr] - Returns the depth of an expression
pub fn depth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Depth expects exactly 1 argument".into(),
    ));
  }

  fn calc_depth(expr: &Expr) -> i128 {
    match expr {
      Expr::List(items) => 1 + items.iter().map(calc_depth).max().unwrap_or(0),
      Expr::FunctionCall { args, .. } => {
        1 + args.iter().map(calc_depth).max().unwrap_or(0)
      }
      Expr::Association(items) => {
        1 + items
          .iter()
          .flat_map(|(k, v)| [calc_depth(k), calc_depth(v)])
          .max()
          .unwrap_or(0)
      }
      _ => 1,
    }
  }

  Ok(Expr::Integer(calc_depth(&args[0])))
}

/// Helper to format a real number
fn format_real_helper(f: f64) -> String {
  if f.fract() == 0.0 && f.abs() < 1e15 {
    format!("{:.1}", f)
  } else {
    format!("{}", f)
  }
}

/// Helper to convert Expr to its FullForm string representation
fn expr_to_full_form(expr: &Expr) -> String {
  match expr {
    Expr::Integer(n) => n.to_string(),
    Expr::Real(f) => format_real_helper(*f),
    Expr::String(s) => format!("\"{}\"", s),
    Expr::Identifier(s) => s.clone(),
    Expr::Slot(n) => {
      if *n == 1 {
        "Slot[1]".to_string()
      } else {
        format!("Slot[{}]", n)
      }
    }
    Expr::Constant(c) => c.clone(),
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_full_form).collect();
      format!("List[{}]", parts.join(", "))
    }
    Expr::FunctionCall { name, args } => {
      let parts: Vec<String> = args.iter().map(expr_to_full_form).collect();
      format!("{}[{}]", name, parts.join(", "))
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;

      // Helper to collect all operands for associative operators (Plus, Times)
      fn collect_operands(
        expr: &Expr,
        target_op: &BinaryOperator,
      ) -> Vec<String> {
        match expr {
          Expr::BinaryOp { op, left, right } if op == target_op => {
            let mut parts = collect_operands(left, target_op);
            parts.extend(collect_operands(right, target_op));
            parts
          }
          _ => vec![expr_to_full_form(expr)],
        }
      }

      match op {
        BinaryOperator::Plus => {
          let parts = collect_operands(expr, &BinaryOperator::Plus);
          format!("Plus[{}]", parts.join(", "))
        }
        BinaryOperator::Times => {
          let parts = collect_operands(expr, &BinaryOperator::Times);
          format!("Times[{}]", parts.join(", "))
        }
        BinaryOperator::Minus => {
          format!(
            "Plus[{}, Times[-1, {}]]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Divide => {
          format!(
            "Times[{}, Power[{}, -1]]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Power => {
          format!(
            "Power[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::And => {
          format!(
            "And[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::Or => {
          format!(
            "Or[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
        BinaryOperator::StringJoin => {
          format!(
            "StringJoin[{}, {}]",
            expr_to_full_form(left),
            expr_to_full_form(right)
          )
        }
      }
    }
    Expr::UnaryOp { op, operand } => {
      use crate::syntax::UnaryOperator;
      match op {
        UnaryOperator::Minus => {
          format!("Times[-1, {}]", expr_to_full_form(operand))
        }
        UnaryOperator::Not => {
          format!("Not[{}]", expr_to_full_form(operand))
        }
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      use crate::syntax::ComparisonOp;
      // For single comparison like a < b, return Less[a, b]
      if operators.len() == 1 {
        let func_name = match &operators[0] {
          ComparisonOp::Equal => "Equal",
          ComparisonOp::NotEqual => "Unequal",
          ComparisonOp::Less => "Less",
          ComparisonOp::LessEqual => "LessEqual",
          ComparisonOp::Greater => "Greater",
          ComparisonOp::GreaterEqual => "GreaterEqual",
          ComparisonOp::SameQ => "SameQ",
          ComparisonOp::UnsameQ => "UnsameQ",
        };
        let parts: Vec<String> =
          operands.iter().map(expr_to_full_form).collect();
        format!("{}[{}]", func_name, parts.join(", "))
      } else {
        // Mixed comparison chain
        let parts: Vec<String> =
          operands.iter().map(expr_to_full_form).collect();
        format!("Inequality[{}]", parts.join(", "))
      }
    }
    Expr::CompoundExpr(exprs) => {
      let parts: Vec<String> = exprs.iter().map(expr_to_full_form).collect();
      format!("CompoundExpression[{}]", parts.join(", "))
    }
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| {
          format!("Rule[{}, {}]", expr_to_full_form(k), expr_to_full_form(v))
        })
        .collect();
      format!("Association[{}]", parts.join(", "))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "Rule[{}, {}]",
        expr_to_full_form(pattern),
        expr_to_full_form(replacement)
      )
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      format!(
        "RuleDelayed[{}, {}]",
        expr_to_full_form(pattern),
        expr_to_full_form(replacement)
      )
    }
    Expr::ReplaceAll { expr, rules } => {
      format!(
        "ReplaceAll[{}, {}]",
        expr_to_full_form(expr),
        expr_to_full_form(rules)
      )
    }
    Expr::ReplaceRepeated { expr, rules } => {
      format!(
        "ReplaceRepeated[{}, {}]",
        expr_to_full_form(expr),
        expr_to_full_form(rules)
      )
    }
    Expr::Map { func, list } => {
      format!(
        "Map[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::Apply { func, list } => {
      format!(
        "Apply[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::MapApply { func, list } => {
      format!(
        "MapApply[{}, {}]",
        expr_to_full_form(func),
        expr_to_full_form(list)
      )
    }
    Expr::PrefixApply { func, arg } => {
      format!("{}[{}]", expr_to_full_form(func), expr_to_full_form(arg))
    }
    Expr::Postfix { expr, func } => {
      format!("{}[{}]", expr_to_full_form(func), expr_to_full_form(expr))
    }
    Expr::Part { expr, index } => {
      format!(
        "Part[{}, {}]",
        expr_to_full_form(expr),
        expr_to_full_form(index)
      )
    }
    Expr::Function { body } => {
      format!("Function[{}]", expr_to_full_form(body))
    }
    Expr::Pattern { name, head } => {
      if let Some(h) = head {
        format!("Pattern[{}, Blank[{}]]", name, h)
      } else {
        format!("Pattern[{}, Blank[]]", name)
      }
    }
    Expr::Raw(s) => s.clone(),
  }
}

/// FullForm[expr] - Returns the full form representation of an expression
pub fn full_form_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(Expr::String(expr_to_full_form(arg)))
}

/// Construct[f, a, b, c, ...] - Creates function call f[a, b, c, ...]
/// In Wolfram, Construct[f, a, b, c] returns f[a, b, c] (all args together)
/// To get curried calls like f[a][b][c], use Fold[Construct, f, {a, b, c}]
pub fn construct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Construct expects at least 1 argument".into(),
    ));
  }

  if args.len() == 1 {
    // Construct[f] returns f
    return Ok(args[0].clone());
  }

  // Get the head (function name)
  let head = &args[0];
  let func_args = &args[1..];

  // Build the function name from the head expression
  let head_name = match head {
    Expr::Identifier(name) => name.clone(),
    Expr::FunctionCall {
      name,
      args: inner_args,
    } if inner_args.is_empty() => name.clone(),
    // If head is already a function call like f[a], we need to use it as string for nested application
    _ => crate::syntax::expr_to_string(head),
  };

  // Construct[f, a, b, c] => f[a, b, c]
  Ok(Expr::FunctionCall {
    name: head_name,
    args: func_args.to_vec(),
  })
}
