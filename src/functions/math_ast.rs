//! AST-native math functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper to extract numeric value from Expr
fn expr_to_num(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// Helper to create numeric result (Integer if whole, Real otherwise)
fn num_to_expr(n: f64) -> Expr {
  if n.fract() == 0.0 && n.abs() < i128::MAX as f64 {
    Expr::Integer(n as i128)
  } else {
    Expr::Real(n)
  }
}

/// Plus[args...] - Sum of arguments, with list threading
pub fn plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => Ok(num_to_expr(x + y)),
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  // Simple numeric sum
  let mut sum = 0.0;
  let mut all_numeric = true;
  for arg in args {
    if let Some(n) = expr_to_num(arg) {
      sum += n;
    } else {
      all_numeric = false;
      break;
    }
  }

  if all_numeric {
    Ok(num_to_expr(sum))
  } else {
    // Return symbolic Plus
    Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Times[args...] - Product of arguments, with list threading
pub fn times_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => Ok(num_to_expr(x * y)),
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  // Simple numeric product
  let mut product = 1.0;
  let mut all_numeric = true;
  for arg in args {
    if let Some(n) = expr_to_num(arg) {
      product *= n;
    } else {
      all_numeric = false;
      break;
    }
  }

  if all_numeric {
    Ok(num_to_expr(product))
  } else {
    Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Minus[a] - Unary negation only
/// Note: Minus with 2 arguments is not valid in Wolfram Language
/// (use Subtract for that)
pub fn minus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    // Return unevaluated for wrong number of arguments - fallback to Pair-based
    // which handles the error message properly
    return Ok(Expr::FunctionCall {
      name: "Minus".to_string(),
      args: args.to_vec(),
    });
  }

  // Unary minus
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(num_to_expr(-n))
  } else {
    Ok(Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(args[0].clone()),
    })
  }
}

/// Divide[a, b] - Division with list threading
pub fn divide_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Divide expects exactly 2 arguments".into(),
    ));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => {
          if y == 0.0 {
            Err(InterpreterError::EvaluationError("Division by zero".into()))
          } else {
            Ok(num_to_expr(x / y))
          }
        }
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  match (expr_to_num(&args[0]), expr_to_num(&args[1])) {
    (Some(a), Some(b)) => {
      if b == 0.0 {
        Err(InterpreterError::EvaluationError("Division by zero".into()))
      } else {
        Ok(num_to_expr(a / b))
      }
    }
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(args[0].clone()),
      right: Box::new(args[1].clone()),
    }),
  }
}

/// Power[a, b] - Exponentiation with list threading
pub fn power_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Power expects exactly 2 arguments".into(),
    ));
  }

  // Check for list threading
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(args, |a, b| {
      match (expr_to_num(a), expr_to_num(b)) {
        (Some(x), Some(y)) => Ok(num_to_expr(x.powf(y))),
        _ => Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(a.clone()),
          right: Box::new(b.clone()),
        }),
      }
    });
  }

  match (expr_to_num(&args[0]), expr_to_num(&args[1])) {
    (Some(a), Some(b)) => Ok(num_to_expr(a.powf(b))),
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(args[0].clone()),
      right: Box::new(args[1].clone()),
    }),
  }
}

/// Thread a binary operation over lists
fn thread_binary_over_lists<F>(
  args: &[Expr],
  op: F,
) -> Result<Expr, InterpreterError>
where
  F: Fn(&Expr, &Expr) -> Result<Expr, InterpreterError>,
{
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Binary operation expects 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(left), Expr::List(right)) => {
      // List + List: element-wise operation
      if left.len() != right.len() {
        return Err(InterpreterError::EvaluationError(
          "Lists must have the same length".into(),
        ));
      }
      let results: Result<Vec<Expr>, _> = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| op(l, r))
        .collect();
      Ok(Expr::List(results?))
    }
    (Expr::List(items), scalar) => {
      // List + scalar: broadcast scalar
      let results: Result<Vec<Expr>, _> =
        items.iter().map(|item| op(item, scalar)).collect();
      Ok(Expr::List(results?))
    }
    (scalar, Expr::List(items)) => {
      // scalar + List: broadcast scalar
      let results: Result<Vec<Expr>, _> =
        items.iter().map(|item| op(scalar, item)).collect();
      Ok(Expr::List(results?))
    }
    _ => op(&args[0], &args[1]),
  }
}

/// Subtract[a, b] - Alias for Minus
pub fn subtract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  minus_ast(args)
}

/// Max[args...] or Max[list] - Maximum value
pub fn max_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("-Infinity".to_string()));
  }

  // Flatten if single list argument
  let items: Vec<&Expr> = if args.len() == 1 {
    match &args[0] {
      Expr::List(items) => {
        if items.is_empty() {
          return Ok(Expr::Identifier("-Infinity".to_string()));
        }
        items.iter().collect()
      }
      _ => args.iter().collect(),
    }
  } else {
    args.iter().collect()
  };

  let mut max: Option<f64> = None;
  for item in items {
    if let Some(n) = expr_to_num(item) {
      max = Some(match max {
        Some(m) => m.max(n),
        None => n,
      });
    } else {
      // Non-numeric, return symbolic
      return Ok(Expr::FunctionCall {
        name: "Max".to_string(),
        args: args.to_vec(),
      });
    }
  }

  Ok(num_to_expr(max.unwrap_or(f64::NEG_INFINITY)))
}

/// Min[args...] or Min[list] - Minimum value
pub fn min_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Flatten if single list argument
  let items: Vec<&Expr> = if args.len() == 1 {
    match &args[0] {
      Expr::List(items) => {
        if items.is_empty() {
          return Ok(Expr::Identifier("Infinity".to_string()));
        }
        items.iter().collect()
      }
      _ => args.iter().collect(),
    }
  } else {
    args.iter().collect()
  };

  let mut min: Option<f64> = None;
  for item in items {
    if let Some(n) = expr_to_num(item) {
      min = Some(match min {
        Some(m) => m.min(n),
        None => n,
      });
    } else {
      return Ok(Expr::FunctionCall {
        name: "Min".to_string(),
        args: args.to_vec(),
      });
    }
  }

  Ok(num_to_expr(min.unwrap_or(f64::INFINITY)))
}

/// Abs[x] - Absolute value
pub fn abs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Abs expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(num_to_expr(n.abs()))
  } else {
    Ok(Expr::FunctionCall {
      name: "Abs".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Sign[x] - Sign of a number (-1, 0, or 1)
pub fn sign_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sign expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Integer(if n > 0.0 {
      1
    } else if n < 0.0 {
      -1
    } else {
      0
    }))
  } else {
    Ok(Expr::FunctionCall {
      name: "Sign".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Sqrt[x] - Square root
pub fn sqrt_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sqrt expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    if n < 0.0 {
      // Complex result - return symbolic
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    } else {
      Ok(num_to_expr(n.sqrt()))
    }
  } else {
    Ok(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Floor[x] - Floor function
pub fn floor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Floor expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Integer(n.floor() as i128))
  } else {
    Ok(Expr::FunctionCall {
      name: "Floor".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Ceiling[x] - Ceiling function
pub fn ceiling_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Ceiling expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Integer(n.ceil() as i128))
  } else {
    Ok(Expr::FunctionCall {
      name: "Ceiling".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Round[x] - Round to nearest integer using banker's rounding (round half to even)
pub fn round_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Round expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    // Banker's rounding: round half to even
    let rounded = if n.fract().abs() == 0.5 {
      let floor = n.floor();
      if floor as i128 % 2 == 0 {
        floor
      } else {
        n.ceil()
      }
    } else {
      n.round()
    };
    Ok(Expr::Integer(rounded as i128))
  } else {
    Ok(Expr::FunctionCall {
      name: "Round".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Mod[a, b] - Modulus
pub fn mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Mod expects exactly 2 arguments".into(),
    ));
  }
  match (&args[0], &args[1]) {
    (Expr::Integer(a), Expr::Integer(b)) => {
      if *b == 0 {
        Err(InterpreterError::EvaluationError(
          "Mod: division by zero".into(),
        ))
      } else {
        // Wolfram's Mod always returns non-negative result
        let result = ((*a % *b) + *b) % *b;
        Ok(Expr::Integer(result))
      }
    }
    _ => {
      if let (Some(a), Some(b)) = (expr_to_num(&args[0]), expr_to_num(&args[1]))
      {
        if b == 0.0 {
          Err(InterpreterError::EvaluationError(
            "Mod: division by zero".into(),
          ))
        } else {
          let result = ((a % b) + b) % b;
          Ok(num_to_expr(result))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "Mod".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// Quotient[a, b] - Integer quotient
pub fn quotient_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Quotient expects exactly 2 arguments".into(),
    ));
  }
  match (&args[0], &args[1]) {
    (Expr::Integer(a), Expr::Integer(b)) => {
      if *b == 0 {
        Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ))
      } else {
        Ok(Expr::Integer(a / b))
      }
    }
    _ => {
      if let (Some(a), Some(b)) = (expr_to_num(&args[0]), expr_to_num(&args[1]))
      {
        if b == 0.0 {
          Err(InterpreterError::EvaluationError(
            "Quotient: division by zero".into(),
          ))
        } else {
          Ok(Expr::Integer((a / b).floor() as i128))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "Quotient".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// GCD[a, b, ...] - Greatest common divisor
pub fn gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
  }

  let mut result: Option<i128> = None;
  for arg in args {
    if let Expr::Integer(n) = arg {
      result = Some(match result {
        Some(r) => gcd(r, *n),
        None => n.abs(),
      });
    } else {
      return Ok(Expr::FunctionCall {
        name: "GCD".to_string(),
        args: args.to_vec(),
      });
    }
  }

  Ok(Expr::Integer(result.unwrap_or(0)))
}

/// LCM[a, b, ...] - Least common multiple
pub fn lcm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
  }

  fn lcm(a: i128, b: i128) -> i128 {
    if a == 0 || b == 0 {
      0
    } else {
      (a.abs() / gcd(a, b)) * b.abs()
    }
  }

  let mut result: Option<i128> = None;
  for arg in args {
    if let Expr::Integer(n) = arg {
      result = Some(match result {
        Some(r) => lcm(r, *n),
        None => n.abs(),
      });
    } else {
      return Ok(Expr::FunctionCall {
        name: "LCM".to_string(),
        args: args.to_vec(),
      });
    }
  }

  Ok(Expr::Integer(result.unwrap_or(1)))
}

/// Total[list] - Sum of all elements in a list
pub fn total_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Total expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      let mut sum = 0.0;
      for item in items {
        if let Some(n) = expr_to_num(item) {
          sum += n;
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec(),
          });
        }
      }
      Ok(num_to_expr(sum))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Total".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Mean[list] - Arithmetic mean
pub fn mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Mean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "Mean: empty list".into(),
        ));
      }
      let mut sum = 0.0;
      for item in items {
        if let Some(n) = expr_to_num(item) {
          sum += n;
        } else {
          return Ok(Expr::FunctionCall {
            name: "Mean".to_string(),
            args: args.to_vec(),
          });
        }
      }
      Ok(num_to_expr(sum / items.len() as f64))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Factorial[n] or n!
pub fn factorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial expects exactly 1 argument".into(),
    ));
  }
  if let Expr::Integer(n) = &args[0] {
    if *n < 0 {
      return Err(InterpreterError::EvaluationError(
        "Factorial: argument must be non-negative".into(),
      ));
    }
    let mut result: i128 = 1;
    for i in 2..=*n {
      result = result.saturating_mul(i);
    }
    Ok(Expr::Integer(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Factorial".to_string(),
      args: args.to_vec(),
    })
  }
}

/// N[expr] or N[expr, n] - Numeric evaluation
pub fn n_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "N expects 1 or 2 arguments".into(),
    ));
  }
  // For now, just return the numeric value if it's already numeric
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::Real(f) => Ok(Expr::Real(*f)),
    Expr::Constant(c) => match c.as_str() {
      "Pi" => Ok(Expr::Real(std::f64::consts::PI)),
      "E" => Ok(Expr::Real(std::f64::consts::E)),
      _ => Ok(args[0].clone()),
    },
    _ => Ok(args[0].clone()),
  }
}

/// RandomInteger[max] or RandomInteger[{min, max}] - Random integer
pub fn random_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  let mut rng = rand::thread_rng();

  match args.len() {
    0 => Ok(Expr::Integer(rng.gen_range(0..=1))),
    1 => match &args[0] {
      Expr::Integer(max) => {
        if *max < 0 {
          Err(InterpreterError::EvaluationError(
            "RandomInteger: max must be non-negative".into(),
          ))
        } else {
          Ok(Expr::Integer(rng.gen_range(0..=*max)))
        }
      }
      Expr::List(items) if items.len() == 2 => {
        if let (Expr::Integer(min), Expr::Integer(max)) = (&items[0], &items[1])
        {
          if min > max {
            Err(InterpreterError::EvaluationError(
              "RandomInteger: min must be <= max".into(),
            ))
          } else {
            Ok(Expr::Integer(rng.gen_range(*min..=*max)))
          }
        } else {
          Err(InterpreterError::EvaluationError(
            "RandomInteger: range must be integers".into(),
          ))
        }
      }
      _ => Err(InterpreterError::EvaluationError(
        "RandomInteger: invalid argument".into(),
      )),
    },
    2 => {
      // RandomInteger[range, n] - generate n random integers
      let n = match &args[1] {
        Expr::Integer(n) if *n > 0 => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: second argument must be a positive integer".into(),
          ));
        }
      };

      let (min, max) = match &args[0] {
        Expr::Integer(m) => (0i128, *m),
        Expr::List(items) if items.len() == 2 => {
          if let (Expr::Integer(min), Expr::Integer(max)) =
            (&items[0], &items[1])
          {
            (*min, *max)
          } else {
            return Err(InterpreterError::EvaluationError(
              "RandomInteger: range must be integers".into(),
            ));
          }
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomInteger: invalid range".into(),
          ));
        }
      };

      if min > max {
        return Err(InterpreterError::EvaluationError(
          "RandomInteger: min must be <= max".into(),
        ));
      }

      let results: Vec<Expr> = (0..n)
        .map(|_| Expr::Integer(rng.gen_range(min..=max)))
        .collect();
      Ok(Expr::List(results))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomInteger expects 0, 1, or 2 arguments".into(),
    )),
  }
}

/// RandomReal[] or RandomReal[max] or RandomReal[{min, max}]
pub fn random_real_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  let mut rng = rand::thread_rng();

  match args.len() {
    0 => Ok(Expr::Real(rng.gen_range(0.0..1.0))),
    1 => match &args[0] {
      Expr::Integer(max) => {
        Ok(Expr::Real(rng.gen_range(0.0..1.0) * *max as f64))
      }
      Expr::Real(max) => Ok(Expr::Real(rng.gen_range(0.0..1.0) * *max)),
      Expr::List(items) if items.len() == 2 => {
        let min = expr_to_num(&items[0]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomReal: invalid min".into())
        })?;
        let max = expr_to_num(&items[1]).ok_or_else(|| {
          InterpreterError::EvaluationError("RandomReal: invalid max".into())
        })?;
        Ok(Expr::Real(min + rng.gen_range(0.0..1.0) * (max - min)))
      }
      _ => Err(InterpreterError::EvaluationError(
        "RandomReal: invalid argument".into(),
      )),
    },
    _ => Err(InterpreterError::EvaluationError(
      "RandomReal expects 0 or 1 argument".into(),
    )),
  }
}

/// Sin, Cos, Tan, etc. - Trigonometric functions
pub fn sin_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sin expects 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Real(n.sin()))
  } else {
    Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: args.to_vec(),
    })
  }
}

pub fn cos_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cos expects 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Real(n.cos()))
  } else {
    Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: args.to_vec(),
    })
  }
}

pub fn tan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tan expects 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Real(n.tan()))
  } else {
    Ok(Expr::FunctionCall {
      name: "Tan".to_string(),
      args: args.to_vec(),
    })
  }
}

pub fn exp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Exp expects 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_num(&args[0]) {
    Ok(Expr::Real(n.exp()))
  } else {
    Ok(Expr::FunctionCall {
      name: "Exp".to_string(),
      args: args.to_vec(),
    })
  }
}

pub fn log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      if let Some(n) = expr_to_num(&args[0]) {
        if n > 0.0 {
          Ok(Expr::Real(n.ln()))
        } else {
          Err(InterpreterError::EvaluationError(
            "Log: argument must be positive".into(),
          ))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "Log".to_string(),
          args: args.to_vec(),
        })
      }
    }
    2 => {
      // Log[base, x]
      if let (Some(base), Some(x)) =
        (expr_to_num(&args[0]), expr_to_num(&args[1]))
      {
        if base > 0.0 && base != 1.0 && x > 0.0 {
          Ok(Expr::Real(x.ln() / base.ln()))
        } else {
          Err(InterpreterError::EvaluationError(
            "Log: invalid arguments".into(),
          ))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "Log".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Err(InterpreterError::EvaluationError(
      "Log expects 1 or 2 arguments".into(),
    )),
  }
}
