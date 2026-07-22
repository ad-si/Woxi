#[allow(unused_imports)]
use super::*;
pub(crate) use crate::functions::math_ast::gcd_i128;
use crate::syntax::{BinaryOperator, ComparisonOp, UnaryOperator};

/// Convert an Expr to a boolean value.
/// Returns Some(true) for Identifier("True"), Some(false) for Identifier("False").
pub fn expr_to_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// Apply a function/predicate to an argument and return the resulting Expr.
/// Uses the existing apply_function_to_arg from evaluator.
pub fn apply_func_ast(
  func: &Expr,
  arg: &Expr,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::apply_function_to_arg(func, arg)
}

/// Apply a binary function to two arguments.
pub fn apply_func_to_two_args(
  func: &Expr,
  arg1: &Expr,
  arg2: &Expr,
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => crate::evaluator::evaluate_function_call_ast(
      name,
      &[arg1.clone(), arg2.clone()],
    ),
    Expr::Function { body } => {
      // Anonymous function with two slots
      let substituted =
        crate::syntax::substitute_slots(body, &[arg1.clone(), arg2.clone()]);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let args_vec = [arg1, arg2];
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(args_vec.iter())
        .map(|(p, a)| (p.as_str(), *a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to (b, c) becomes f[a, b, c]
      let mut new_args = args.clone();
      new_args.push(arg1.clone());
      new_args.push(arg2.clone());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(
        &func_str,
        &[arg1.clone(), arg2.clone()],
      )
    }
  }
}

/// Apply a function to an arbitrary number of arguments (generalizes
/// `apply_func_to_two_args` to n-ary application).
pub fn apply_func_to_args(
  func: &Expr,
  call_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, call_args)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, call_args);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(call_args.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args } => {
      // Curried function: f[a] applied to (b, c, …) becomes f[a, b, c, …]
      let mut new_args = args.clone();
      for a in call_args {
        new_args.push(a.clone());
      }
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(&func_str, call_args)
    }
  }
}

/// Get the head of an expression as a string. Operator nodes
/// (`Expr::BinaryOp`/`UnaryOp`/`Comparison`) resolve to their Wolfram
/// canonical head — `x^2` is `Power`, `a*b`/`a/b`/`-x` are `Times`,
/// `a-b` is `Plus`, `a==b` is `Equal` — so head-constrained patterns
/// (`_Power`, `_Plus`, …) match them correctly. Mirrors `Head[]`.
pub fn get_expr_head_str(expr: &Expr) -> &str {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer",
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real",
    Expr::String(_) => "String",
    Expr::List(_) => "List",
    Expr::FunctionCall { name, .. } => name,
    Expr::Association(_) => "Association",
    Expr::Rule { .. } => "Rule",
    Expr::RuleDelayed { .. } => "RuleDelayed",
    Expr::BinaryOp { op, left, .. } => match op {
      // a - b is stored as a Minus node but is Plus[a, Times[-1, b]] in WL.
      BinaryOperator::Plus | BinaryOperator::Minus => "Plus",
      BinaryOperator::Times => "Times",
      // a/b is Times[a, Power[b, -1]]; 1/b is just Power[b, -1].
      BinaryOperator::Divide => {
        if matches!(left.as_ref(), Expr::Integer(1)) {
          "Power"
        } else {
          "Times"
        }
      }
      BinaryOperator::Power => "Power",
      BinaryOperator::And => "And",
      BinaryOperator::Or => "Or",
      BinaryOperator::StringJoin => "StringJoin",
      BinaryOperator::Alternatives => "Alternatives",
    },
    Expr::UnaryOp { op, .. } => match op {
      UnaryOperator::Minus => "Times",
      UnaryOperator::Not => "Not",
    },
    Expr::Comparison { operators, .. } => {
      let first = operators.first().copied();
      if operators.iter().all(|o| Some(*o) == first) {
        match first {
          Some(ComparisonOp::Equal) | None => "Equal",
          Some(ComparisonOp::NotEqual) => "Unequal",
          Some(ComparisonOp::Less) => "Less",
          Some(ComparisonOp::LessEqual) => "LessEqual",
          Some(ComparisonOp::Greater) => "Greater",
          Some(ComparisonOp::GreaterEqual) => "GreaterEqual",
          Some(ComparisonOp::SameQ) => "SameQ",
          Some(ComparisonOp::UnsameQ) => "UnsameQ",
        }
      } else {
        "Inequality"
      }
    }
    _ => "Symbol",
  }
}

/// Helper to extract i128 from Expr
pub(crate) fn expr_to_i128(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i128()
    }
    Expr::Real(f) if f.fract() == 0.0 => Some(*f as i128),
    _ => None,
  }
}

/// Like `expr_to_i128`, but also floors fractional Reals and Rationals so
/// iterator bounds like `Do[..., {i, 1, 7/2}]` or `Do[..., {i, 1, 3.5}]`
/// behave the same as wolframscript (which iterates up to `Floor[bound]`).
pub(crate) fn expr_to_i128_floor(expr: &Expr) -> Option<i128> {
  if let Some(n) = expr_to_i128(expr) {
    return Some(n);
  }
  let f = crate::functions::math_ast::try_eval_to_f64_with_infinity(expr)?;
  if f.is_infinite() || f.is_nan() {
    return None;
  }
  Some(f.floor() as i128)
}

/// Apply a function to n arguments.
pub fn apply_func_to_n_args(
  func: &Expr,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  match func {
    Expr::Identifier(name) => {
      crate::evaluator::evaluate_function_call_ast(name, args)
    }
    Expr::Function { body } => {
      let substituted = crate::syntax::substitute_slots(body, args);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::NamedFunction { params, body, .. } => {
      let bindings: Vec<(&str, &Expr)> = params
        .iter()
        .zip(args.iter())
        .map(|(p, a)| (p.as_str(), a))
        .collect();
      let substituted = crate::syntax::substitute_variables(body, &bindings);
      crate::evaluator::evaluate_expr_to_expr(&substituted)
    }
    Expr::FunctionCall { name, args: fa } => {
      let mut new_args = fa.clone();
      new_args.extend(args.iter().cloned());
      crate::evaluator::evaluate_function_call_ast(name, &new_args)
    }
    _ => {
      let func_str = crate::syntax::expr_to_string(func);
      crate::evaluator::evaluate_function_call_ast(&func_str, args)
    }
  }
}

/// Helper to extract f64 from Expr
pub(crate) fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// `expr_to_f64` extended with exact rationals (`Rational[n, d]`), for the
/// histogram/bin-count family whose data wolframscript machine-numericizes.
pub(crate) fn numeric_expr_to_f64(expr: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Rational"
    && args.len() == 2
  {
    return match (expr_to_f64(&args[0]), expr_to_f64(&args[1])) {
      (Some(n), Some(d)) if d != 0.0 => Some(n / d),
      _ => None,
    };
  }
  expr_to_f64(expr)
}

/// Helper to convert f64 to appropriate Expr
pub fn f64_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// Identity[x] - returns x unchanged
pub fn identity_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(arg.clone())
}

/// Check if an expression is the symbol Nothing
pub fn is_nothing(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "Nothing")
}

/// Check if an expression is an atomic (non-compound) expression
pub fn is_atom_expr(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(_) | Expr::Constant(_) | Expr::String(_))
}
