//! AST-native boolean functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::syntax::Expr;

/// Helper to check if an Expr is True or False
fn as_bool(expr: &Expr) -> Option<bool> {
  match expr {
    Expr::Identifier(s) if s == "True" => Some(true),
    Expr::Identifier(s) if s == "False" => Some(false),
    _ => None,
  }
}

/// And[expr1, expr2, ...] - Logical AND
pub fn and_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "And expects at least 2 arguments".into(),
    ));
  }

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    if !as_bool(&evaluated).unwrap_or(false) {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Or[expr1, expr2, ...] - Logical OR
pub fn or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Or expects at least 2 arguments".into(),
    ));
  }

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    if as_bool(&evaluated).unwrap_or(false) {
      return Ok(Expr::Identifier("True".to_string()));
    }
  }
  Ok(Expr::Identifier("False".to_string()))
}

/// Not[expr] - Logical NOT
pub fn not_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    // Return unevaluated for wrong number of arguments
    return Ok(Expr::FunctionCall {
      name: "Not".to_string(),
      args: args.to_vec(),
    });
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  let val = as_bool(&evaluated).unwrap_or(false);
  Ok(Expr::Identifier(
    if val { "False" } else { "True" }.to_string(),
  ))
}

/// Xor[expr1, expr2, ...] - Logical XOR
pub fn xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Xor expects at least 2 arguments".into(),
    ));
  }

  let mut true_count = 0;
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    if as_bool(&evaluated).unwrap_or(false) {
      true_count += 1;
    }
  }
  Ok(Expr::Identifier(
    if true_count % 2 == 1 { "True" } else { "False" }.to_string(),
  ))
}

/// SameQ[expr1, expr2] - Tests whether expressions are identical
pub fn same_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "SameQ expects at least 2 arguments".into(),
    ));
  }

  let first = evaluate_expr_to_expr(&args[0])?;
  let first_str = crate::syntax::expr_to_string(&first);

  for arg in args.iter().skip(1) {
    let val = evaluate_expr_to_expr(arg)?;
    let val_str = crate::syntax::expr_to_string(&val);
    if val_str != first_str {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// UnsameQ[expr1, expr2] - Tests whether expressions are not identical
pub fn unsame_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "UnsameQ expects at least 2 arguments".into(),
    ));
  }

  let first = evaluate_expr_to_expr(&args[0])?;
  let first_str = crate::syntax::expr_to_string(&first);

  for arg in args.iter().skip(1) {
    let val = evaluate_expr_to_expr(arg)?;
    let val_str = crate::syntax::expr_to_string(&val);
    if val_str != first_str {
      return Ok(Expr::Identifier("True".to_string()));
    }
  }
  Ok(Expr::Identifier("False".to_string()))
}

/// Which[test1, value1, test2, value2, ...] - Multi-way conditional
pub fn which_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || !args.len().is_multiple_of(2) {
    return Err(InterpreterError::EvaluationError(
      "Which expects an even number of arguments (test-value pairs)".into(),
    ));
  }

  for i in (0..args.len()).step_by(2) {
    let test = evaluate_expr_to_expr(&args[i])?;
    if let Some(true) = as_bool(&test) {
      return evaluate_expr_to_expr(&args[i + 1]);
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// While[test, body] - While loop
pub fn while_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "While expects exactly 2 arguments".into(),
    ));
  }

  const MAX_ITERATIONS: usize = 100000;
  let mut iterations = 0;

  loop {
    let test = evaluate_expr_to_expr(&args[0])?;
    match as_bool(&test) {
      Some(true) => {
        match evaluate_expr_to_expr(&args[1]) {
          Ok(_) => {}
          Err(InterpreterError::BreakSignal) => break,
          Err(InterpreterError::ContinueSignal) => {}
          Err(e) => return Err(e),
        }
        iterations += 1;
        if iterations >= MAX_ITERATIONS {
          return Err(InterpreterError::EvaluationError(
            "While: maximum iterations exceeded".into(),
          ));
        }
      }
      Some(false) => break,
      None => {
        return Err(InterpreterError::EvaluationError(
          "While: test must evaluate to True or False".into(),
        ));
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// Equal[a, b] or a == b - Tests for equality
pub fn equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Equal expects at least 2 arguments".into(),
    ));
  }

  let first_str = crate::syntax::expr_to_string(&args[0]);

  for arg in args.iter().skip(1) {
    let val_str = crate::syntax::expr_to_string(arg);
    if val_str != first_str {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Unequal[a, b] or a != b - Tests for inequality
pub fn unequal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Unequal expects at least 2 arguments".into(),
    ));
  }

  // All arguments must be pairwise different
  let strs: Vec<String> =
    args.iter().map(crate::syntax::expr_to_string).collect();

  for i in 0..strs.len() {
    for j in i + 1..strs.len() {
      if strs[i] == strs[j] {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Helper to extract numeric value from Expr — delegates to try_eval_to_f64 for full recursive evaluation
fn expr_to_num(expr: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64(expr)
}

/// Less[a, b] or a < b - Tests if a is less than b
pub fn less_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Less expects at least 2 arguments".into(),
    ));
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Less".to_string(),
        args: args.to_vec(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Less".to_string(),
          args: args.to_vec(),
        });
      }
    };
    if prev >= curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Greater[a, b] or a > b - Tests if a is greater than b
pub fn greater_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Greater expects at least 2 arguments".into(),
    ));
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Greater".to_string(),
        args: args.to_vec(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Greater".to_string(),
          args: args.to_vec(),
        });
      }
    };
    if prev <= curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// LessEqual[a, b] or a <= b - Tests if a is less than or equal to b
pub fn less_equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "LessEqual expects at least 2 arguments".into(),
    ));
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LessEqual".to_string(),
        args: args.to_vec(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "LessEqual".to_string(),
          args: args.to_vec(),
        });
      }
    };
    if prev > curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// GreaterEqual[a, b] or a >= b - Tests if a is greater than or equal to b
pub fn greater_equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "GreaterEqual expects at least 2 arguments".into(),
    ));
  }

  let mut prev = match expr_to_num(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "GreaterEqual".to_string(),
        args: args.to_vec(),
      });
    }
  };

  for arg in args.iter().skip(1) {
    let curr = match expr_to_num(arg) {
      Some(n) => n,
      None => {
        return Ok(Expr::FunctionCall {
          name: "GreaterEqual".to_string(),
          args: args.to_vec(),
        });
      }
    };
    if prev < curr {
      return Ok(Expr::Identifier("False".to_string()));
    }
    prev = curr;
  }
  Ok(Expr::Identifier("True".to_string()))
}

/// Boole[expr] - Converts True to 1 and False to 0
pub fn boole_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Boole expects exactly 1 argument".into(),
    ));
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  match as_bool(&evaluated) {
    Some(true) => Ok(Expr::Integer(1)),
    Some(false) => Ok(Expr::Integer(0)),
    None => Ok(Expr::FunctionCall {
      name: "Boole".to_string(),
      args: vec![evaluated],
    }),
  }
}

/// TrueQ[expr] - Returns True if expr is explicitly True
pub fn true_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "TrueQ expects exactly 1 argument".into(),
    ));
  }

  let evaluated = evaluate_expr_to_expr(&args[0])?;
  Ok(Expr::Identifier(
    if matches!(&evaluated, Expr::Identifier(s) if s == "True") {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

/// Implies[a, b] - Logical implication (a implies b, i.e., !a || b)
pub fn implies_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Implies expects exactly 2 arguments".into(),
    ));
  }

  let a = evaluate_expr_to_expr(&args[0])?;
  match as_bool(&a) {
    Some(false) => Ok(Expr::Identifier("True".to_string())), // False implies anything
    Some(true) => {
      let b = evaluate_expr_to_expr(&args[1])?;
      Ok(Expr::Identifier(
        if as_bool(&b).unwrap_or(false) {
          "True"
        } else {
          "False"
        }
        .to_string(),
      ))
    }
    None => Ok(Expr::FunctionCall {
      name: "Implies".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Nand[expr1, expr2, ...] - Logical NAND (Not And)
pub fn nand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Nand expects at least 2 arguments".into(),
    ));
  }

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    if !as_bool(&evaluated).unwrap_or(false) {
      return Ok(Expr::Identifier("True".to_string()));
    }
  }
  Ok(Expr::Identifier("False".to_string()))
}

/// Nor[expr1, expr2, ...] - Logical NOR (Not Or)
pub fn nor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Nor expects at least 2 arguments".into(),
    ));
  }

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    if as_bool(&evaluated).unwrap_or(false) {
      return Ok(Expr::Identifier("False".to_string()));
    }
  }
  Ok(Expr::Identifier("True".to_string()))
}
