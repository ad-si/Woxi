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

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(false) => return Ok(Expr::Identifier("False".to_string())),
      Some(true) => {} // Skip True values
      None => remaining.push(evaluated),
    }
  }
  match remaining.len() {
    0 => Ok(Expr::Identifier("True".to_string())),
    1 => Ok(remaining.into_iter().next().unwrap()),
    _ => Ok(Expr::FunctionCall {
      name: "And".to_string(),
      args: remaining,
    }),
  }
}

/// Or[expr1, expr2, ...] - Logical OR
pub fn or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Or expects at least 2 arguments".into(),
    ));
  }

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => return Ok(Expr::Identifier("True".to_string())),
      Some(false) => {} // Skip False values
      None => remaining.push(evaluated),
    }
  }
  match remaining.len() {
    0 => Ok(Expr::Identifier("False".to_string())),
    1 => Ok(remaining.into_iter().next().unwrap()),
    _ => Ok(Expr::FunctionCall {
      name: "Or".to_string(),
      args: remaining,
    }),
  }
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
  match as_bool(&evaluated) {
    Some(true) => Ok(Expr::Identifier("False".to_string())),
    Some(false) => Ok(Expr::Identifier("True".to_string())),
    None => Ok(Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Not,
      operand: Box::new(evaluated),
    }),
  }
}

/// Xor[expr1, expr2, ...] - Logical XOR
pub fn xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Xor expects at least 1 argument".into(),
    ));
  }
  // Single argument: Xor[x] => x
  if args.len() == 1 {
    return evaluate_expr_to_expr(&args[0]);
  }

  let mut true_count = 0;
  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => true_count += 1,
      Some(false) => {} // Skip False
      None => remaining.push(evaluated),
    }
  }
  // If there are symbolic args, combine: known true values flip parity
  if !remaining.is_empty() {
    // If odd number of True values, add True to remaining
    if true_count % 2 == 1 {
      remaining.insert(0, Expr::Identifier("True".to_string()));
    }
    return match remaining.len() {
      1 => Ok(remaining.into_iter().next().unwrap()),
      _ => Ok(Expr::FunctionCall {
        name: "Xor".to_string(),
        args: remaining,
      }),
    };
  }
  Ok(Expr::Identifier(
    if true_count % 2 == 1 { "True" } else { "False" }.to_string(),
  ))
}

/// SameQ[expr1, expr2] - Tests whether expressions are identical
pub fn same_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // SameQ[] and SameQ[x] return True (vacuously true)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
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
  // UnsameQ[] and UnsameQ[x] return True (vacuously true)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Evaluate all arguments and get string representations
  let mut strs = Vec::with_capacity(args.len());
  for arg in args {
    let val = evaluate_expr_to_expr(arg)?;
    strs.push(crate::syntax::expr_to_string(&val));
  }

  // UnsameQ is True only if ALL pairs are different
  for i in 0..strs.len() {
    for j in (i + 1)..strs.len() {
      if strs[i] == strs[j] {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }
  Ok(Expr::Identifier("True".to_string()))
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
    match as_bool(&test) {
      Some(true) => return evaluate_expr_to_expr(&args[i + 1]),
      Some(false) => {} // Skip this pair
      None => {
        // Non-boolean condition: return Which with remaining pairs
        let mut remaining = vec![test, args[i + 1].clone()];
        for j in ((i + 2)..args.len()).step_by(1) {
          remaining.push(args[j].clone());
        }
        return Ok(Expr::FunctionCall {
          name: "Which".to_string(),
          args: remaining,
        });
      }
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// While[test, body] or While[test] - While loop
/// Single-arg form: evaluates test repeatedly (do-while pattern with side effects)
pub fn while_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "While expects 1 or 2 arguments".into(),
    ));
  }

  const MAX_ITERATIONS: usize = 100000;
  let mut iterations = 0;

  loop {
    let test = evaluate_expr_to_expr(&args[0])?;
    match as_bool(&test) {
      Some(true) => {
        if args.len() == 2 {
          match evaluate_expr_to_expr(&args[1]) {
            Ok(_) => {}
            Err(InterpreterError::BreakSignal) => break,
            Err(InterpreterError::ContinueSignal) => {}
            Err(InterpreterError::ReturnValue(val)) => return Ok(*val),
            Err(e) => return Err(e),
          }
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
/// Returns True if all args are identical, False if all are numeric and differ,
/// or stays symbolic (unevaluated) if args contain symbols and aren't identical.
pub fn equal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Equal[] and Equal[x] return True (like wolframscript)
  if args.len() < 2 {
    return Ok(Expr::Identifier("True".to_string()));
  }

  use crate::functions::math_ast::try_eval_to_f64;

  let first_str = crate::syntax::expr_to_string(&args[0]);
  let mut all_identical = true;

  for arg in args.iter().skip(1) {
    let val_str = crate::syntax::expr_to_string(arg);
    if val_str != first_str {
      all_identical = false;
      break;
    }
  }

  if all_identical {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Check if all args are numeric
  let nums: Vec<Option<f64>> = args.iter().map(try_eval_to_f64).collect();
  if nums.iter().all(|n| n.is_some()) {
    let first = nums[0].unwrap();
    for n in nums.iter().skip(1) {
      if n.unwrap() != first {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Only stay symbolic if at least one arg has free symbols
  if args.iter().any(crate::evaluator::has_free_symbols) {
    Ok(Expr::FunctionCall {
      name: "Equal".to_string(),
      args: args.to_vec(),
    })
  } else {
    // No free symbols, not identical → False
    Ok(Expr::Identifier("False".to_string()))
  }
}

/// Unequal[a, b] or a != b - Tests for inequality
/// Returns False if any args are identical, True if all are numeric and pairwise different,
/// or stays symbolic (unevaluated) if args contain symbols and aren't identical.
pub fn unequal_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Unequal expects at least 2 arguments".into(),
    ));
  }

  use crate::functions::math_ast::try_eval_to_f64;

  // Check if any pair is structurally identical → False
  let strs: Vec<String> =
    args.iter().map(crate::syntax::expr_to_string).collect();

  for i in 0..strs.len() {
    for j in i + 1..strs.len() {
      if strs[i] == strs[j] {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }

  // Check if all args are numeric
  let nums: Vec<Option<f64>> = args.iter().map(try_eval_to_f64).collect();
  if nums.iter().all(|n| n.is_some()) {
    // All numeric and pairwise different (checked above via strings)
    return Ok(Expr::Identifier("True".to_string()));
  }

  // Only stay symbolic if at least one arg has free symbols
  if args.iter().any(crate::evaluator::has_free_symbols) {
    Ok(Expr::FunctionCall {
      name: "Unequal".to_string(),
      args: args.to_vec(),
    })
  } else {
    // No free symbols, pairwise different → True
    Ok(Expr::Identifier("True".to_string()))
  }
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
      match as_bool(&b) {
        Some(val) => Ok(Expr::Identifier(
          if val { "True" } else { "False" }.to_string(),
        )),
        None => Ok(b), // True implies symbolic expr → return the expr
      }
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

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(false) => return Ok(Expr::Identifier("True".to_string())),
      Some(true) => {} // Skip True values
      None => remaining.push(evaluated),
    }
  }
  if remaining.is_empty() {
    // All were True → Nand is False
    Ok(Expr::Identifier("False".to_string()))
  } else {
    // Some symbolic: Nand[remaining...]
    Ok(Expr::FunctionCall {
      name: "Nand".to_string(),
      args: remaining,
    })
  }
}

/// Nor[expr1, expr2, ...] - Logical NOR (Not Or)
pub fn nor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Nor expects at least 2 arguments".into(),
    ));
  }

  let mut remaining = Vec::new();
  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => return Ok(Expr::Identifier("False".to_string())),
      Some(false) => {} // Skip False values
      None => remaining.push(evaluated),
    }
  }
  if remaining.is_empty() {
    // All were False → Nor is True
    Ok(Expr::Identifier("True".to_string()))
  } else {
    // Some symbolic: Nor[remaining...]
    Ok(Expr::FunctionCall {
      name: "Nor".to_string(),
      args: remaining,
    })
  }
}

/// Equivalent[expr1, expr2, ...] - True if all args have the same truth value
pub fn equivalent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Equivalent expects at least 2 arguments".into(),
    ));
  }

  let mut has_true = false;
  let mut has_false = false;
  let mut remaining = Vec::new();

  for arg in args {
    let evaluated = evaluate_expr_to_expr(arg)?;
    match as_bool(&evaluated) {
      Some(true) => has_true = true,
      Some(false) => has_false = true,
      None => remaining.push(evaluated),
    }
  }

  // If we have both True and False, it's False
  if has_true && has_false {
    return Ok(Expr::Identifier("False".to_string()));
  }

  // If all are known and the same value, it's True
  if remaining.is_empty() {
    return Ok(Expr::Identifier("True".to_string()));
  }

  // If some symbolic: keep them with any known value
  let mut result = remaining;
  if has_true {
    result.insert(0, Expr::Identifier("True".to_string()));
  }
  if has_false {
    result.insert(0, Expr::Identifier("False".to_string()));
  }
  if result.len() == 1 {
    return Ok(Expr::Identifier("True".to_string()));
  }
  Ok(Expr::FunctionCall {
    name: "Equivalent".to_string(),
    args: result,
  })
}
