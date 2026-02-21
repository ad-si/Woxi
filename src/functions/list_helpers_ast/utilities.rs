#[allow(unused_imports)]
use super::*;

/// Convert an Expr to a boolean value.
/// Returns Some(true) for Identifier("True"), Some(false) for Identifier("False").
pub fn expr_to_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// Convert a boolean to an Expr.
pub fn bool_to_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
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
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      let args_vec = [arg1, arg2];
      for (param, arg) in params.iter().zip(args_vec.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
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

/// Get the head of an expression as a string
pub fn get_expr_head_str(expr: &Expr) -> &str {
  match expr {
    Expr::Integer(_) | Expr::BigInteger(_) => "Integer",
    Expr::Real(_) | Expr::BigFloat(_, _) => "Real",
    Expr::String(_) => "String",
    Expr::List(_) => "List",
    Expr::FunctionCall { name, .. } => name,
    Expr::Association(_) => "Association",
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
    Expr::NamedFunction { params, body } => {
      let mut substituted = (**body).clone();
      for (param, arg) in params.iter().zip(args.iter()) {
        substituted =
          crate::syntax::substitute_variable(&substituted, param, arg);
      }
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

/// Helper to convert f64 to appropriate Expr
pub fn f64_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

pub(crate) fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Identity[x] - returns x unchanged
pub fn identity_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  Ok(arg.clone())
}

/// Check if an expression is an atomic (non-compound) expression
pub fn is_atom_expr(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(_) | Expr::Constant(_) | Expr::String(_))
}
