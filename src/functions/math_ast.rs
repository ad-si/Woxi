//! AST-native math functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Helper - constants are kept symbolic, no direct f64 conversion in expr_to_num.
fn constant_to_f64(_name: &str) -> Option<f64> {
  None
}

fn expr_to_num(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::Constant(name) => constant_to_f64(name),
    // Handle Rational[numer, denom]
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Some(numer), Some(denom)) =
        (expr_to_num(&args[0]), expr_to_num(&args[1]))
        && denom != 0.0
      {
        return Some(numer / denom);
      }
      None
    }
    _ => None,
  }
}

/// Recursively try to evaluate any expression to f64.
/// This handles constants (Pi, E, Degree), arithmetic operations, and known functions.
/// Used by N[], comparisons, and anywhere a numeric value is needed from a symbolic expression.
pub fn try_eval_to_f64(expr: &Expr) -> Option<f64> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::Constant(name) => match name.as_str() {
      "Pi" => Some(std::f64::consts::PI),
      "-Pi" => Some(-std::f64::consts::PI),
      "E" => Some(std::f64::consts::E),
      "Degree" => Some(std::f64::consts::PI / 180.0),
      "-Degree" => Some(-std::f64::consts::PI / 180.0),
      _ => None,
    },
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => try_eval_to_f64(operand).map(|v| -v),
    Expr::BinaryOp { op, left, right } => {
      let l = try_eval_to_f64(left)?;
      let r = try_eval_to_f64(right)?;
      match op {
        BinaryOperator::Plus => Some(l + r),
        BinaryOperator::Minus => Some(l - r),
        BinaryOperator::Times => Some(l * r),
        BinaryOperator::Divide => {
          if r != 0.0 {
            Some(l / r)
          } else {
            None
          }
        }
        BinaryOperator::Power => Some(l.powf(r)),
        _ => None,
      }
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Rational" if args.len() == 2 => {
        let n = try_eval_to_f64(&args[0])?;
        let d = try_eval_to_f64(&args[1])?;
        if d != 0.0 { Some(n / d) } else { None }
      }
      "Sin" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.sin()),
      "Cos" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.cos()),
      "Tan" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.tan()),
      "ArcSin" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.asin())
      }
      "ArcCos" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.acos())
      }
      "ArcTan" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.atan())
      }
      "Sqrt" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.sqrt()),
      "Abs" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.abs()),
      "Exp" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.exp()),
      "Log" if args.len() == 1 => try_eval_to_f64(&args[0])
        .and_then(|v| if v > 0.0 { Some(v.ln()) } else { None }),
      "Log" if args.len() == 2 => {
        let base = try_eval_to_f64(&args[0])?;
        let val = try_eval_to_f64(&args[1])?;
        if base > 0.0 && base != 1.0 && val > 0.0 {
          Some(val.ln() / base.ln())
        } else {
          None
        }
      }
      "Log10" if args.len() == 1 => try_eval_to_f64(&args[0])
        .and_then(|v| if v > 0.0 { Some(v.log10()) } else { None }),
      "Log2" if args.len() == 1 => try_eval_to_f64(&args[0])
        .and_then(|v| if v > 0.0 { Some(v.log2()) } else { None }),
      "Power" if args.len() == 2 => {
        let b = try_eval_to_f64(&args[0])?;
        let e = try_eval_to_f64(&args[1])?;
        Some(b.powf(e))
      }
      "Floor" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.floor())
      }
      "Ceiling" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.ceil())
      }
      "Round" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.round())
      }
      "Times" => {
        let mut product = 1.0;
        for arg in args {
          product *= try_eval_to_f64(arg)?;
        }
        Some(product)
      }
      "Plus" => {
        let mut sum = 0.0;
        for arg in args {
          sum += try_eval_to_f64(arg)?;
        }
        Some(sum)
      }
      "Factorial" if args.len() == 1 => {
        if let Expr::Integer(n) = &args[0]
          && *n >= 0
          && *n <= 170
        {
          let mut result = 1.0f64;
          for i in 2..=(*n as u64) {
            result *= i as f64;
          }
          return Some(result);
        }
        None
      }
      _ => None,
    },
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

/// Compute GCD of two integers using Euclidean algorithm
fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let temp = b;
    b = a % b;
    a = temp;
  }
  a
}

/// Create a rational or integer result from numerator/denominator
/// Simplifies the fraction and returns Integer if denominator is 1
fn make_rational(numer: i128, denom: i128) -> Expr {
  if denom == 0 {
    // Division by zero - shouldn't reach here but be safe
    return Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(numer), Expr::Integer(denom)],
    };
  }

  // Handle sign: put sign in numerator
  let (numer, denom) = if denom < 0 {
    (-numer, -denom)
  } else {
    (numer, denom)
  };

  // Simplify by GCD
  let g = gcd(numer, denom);
  let (numer, denom) = (numer / g, denom / g);

  if denom == 1 {
    Expr::Integer(numer)
  } else {
    Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(numer), Expr::Integer(denom)],
    }
  }
}

/// Multiply a numeric scalar (Integer, Rational, or Real) by an expression.
/// Handles identity (1 * x = x) and zero (0 * x = 0).
fn multiply_scalar_by_expr(
  scalar: &Expr,
  expr: &Expr,
) -> Result<Expr, InterpreterError> {
  match scalar {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(expr.clone()),
    Expr::Integer(-1) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), expr.clone()],
    }),
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(scalar.clone()),
      right: Box::new(expr.clone()),
    }),
  }
}

/// Plus[args...] - Sum of arguments, with list threading
pub fn plus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  // Flatten nested Plus arguments
  let mut flat_args: Vec<Expr> = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Plus" => {
        flat_args.extend(inner_args.clone());
      }
      _ => flat_args.push(arg.clone()),
    }
  }

  // Check for list threading
  let has_list = flat_args.iter().any(|a| matches!(a, Expr::List(_)));
  if has_list {
    return thread_binary_over_lists(&flat_args, |a, b| {
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
  let mut any_real = false;
  for arg in &flat_args {
    if let Some(n) = expr_to_num(arg) {
      sum += n;
      if matches!(arg, Expr::Real(_)) {
        any_real = true;
      }
    } else {
      all_numeric = false;
      break;
    }
  }

  if all_numeric {
    if any_real {
      Ok(Expr::Real(sum))
    } else {
      Ok(num_to_expr(sum))
    }
  } else {
    // Separate numeric and symbolic terms
    let mut numeric_sum = 0.0;
    let mut has_numeric = false;
    let mut has_real = false;
    let mut symbolic_args: Vec<Expr> = Vec::new();

    for arg in &flat_args {
      if let Some(n) = expr_to_num(arg) {
        numeric_sum += n;
        has_numeric = true;
        if matches!(arg, Expr::Real(_)) {
          has_real = true;
        }
      } else {
        symbolic_args.push(arg.clone());
      }
    }

    // Build final args: numeric sum first (if non-zero), then symbolic terms sorted
    let mut final_args: Vec<Expr> = Vec::new();
    if has_numeric && numeric_sum != 0.0 {
      if has_real {
        final_args.push(Expr::Real(numeric_sum));
      } else {
        final_args.push(num_to_expr(numeric_sum));
      }
    }

    // Sort symbolic terms: polynomial-like terms first, then transcendental functions
    // This gives Mathematica-like ordering where x^2 comes before Sin[x]
    symbolic_args.sort_by(|a, b| {
      // Priority: lower number = appears earlier
      // 0 = polynomial-like (variables, products, powers, divisions)
      // 1 = transcendental functions (Sin, Cos, Exp, Log, etc.)
      fn term_priority(e: &Expr) -> i32 {
        match e {
          // Pure identifiers (variables) are polynomial-like
          Expr::Identifier(_) => 0,
          // Powers are polynomial-like
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            ..
          } => 0,
          // Division is polynomial-like
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            ..
          } => 0,
          // Times is polynomial-like
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            ..
          } => 0,
          // FunctionCall: Times is polynomial-like, transcendental functions come later
          Expr::FunctionCall { name, .. } => {
            match name.as_str() {
              "Times" | "Power" | "Plus" | "Rational" => 0,
              // Transcendental functions come after polynomial terms
              "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh"
              | "Cosh" | "Tanh" | "Coth" | "Sech" | "Csch" | "ArcSin"
              | "ArcCos" | "ArcTan" | "ArcCot" | "ArcSec" | "ArcCsc"
              | "Exp" | "Log" | "Factorial" => 1,
              // Other functions are polynomial-like (could be user-defined)
              _ => 0,
            }
          }
          // Unary minus: use inner priority
          Expr::UnaryOp { operand, .. } => term_priority(operand),
          // Everything else is polynomial-like
          _ => 0,
        }
      }
      let pa = term_priority(a);
      let pb = term_priority(b);
      if pa != pb {
        pa.cmp(&pb)
      } else {
        // Same priority - sort alphabetically
        crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
      }
    });
    final_args.extend(symbolic_args);

    if final_args.is_empty() {
      Ok(Expr::Integer(0))
    } else if final_args.len() == 1 {
      Ok(final_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: final_args,
      })
    }
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
  let mut any_real = false;
  for arg in args {
    if let Some(n) = expr_to_num(arg) {
      product *= n;
      if matches!(arg, Expr::Real(_)) {
        any_real = true;
      }
    } else {
      all_numeric = false;
      break;
    }
  }

  if all_numeric {
    if any_real {
      Ok(Expr::Real(product))
    } else {
      Ok(num_to_expr(product))
    }
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
  if args.len() == 1 {
    // Unary minus
    if matches!(&args[0], Expr::Real(_))
      && let Some(n) = expr_to_num(&args[0])
    {
      return Ok(Expr::Real(-n));
    }
    if let Some(n) = expr_to_num(&args[0]) {
      Ok(num_to_expr(-n))
    } else {
      Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(args[0].clone()),
      })
    }
  } else {
    // Wrong arity - print error to stderr and return unevaluated expression
    eprintln!();
    eprintln!(
      "Minus::argx: Minus called with {} arguments; 1 argument is expected.",
      args.len()
    );
    // Return as a BinaryOp chain with Minus operator
    if args.is_empty() {
      return Err(InterpreterError::EvaluationError(
        "Minus expects at least 1 argument".into(),
      ));
    }
    // Build chain of Minus operators: a - b - c - ...
    let mut result = args[0].clone();
    for arg in &args[1..] {
      result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(result),
        right: Box::new(arg.clone()),
      };
    }
    Ok(result)
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
    return thread_binary_over_lists(args, divide_two);
  }

  divide_two(&args[0], &args[1])
}

/// Helper for division of two arguments
fn divide_two(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  // For two integers, keep as Rational (fraction)
  if let (Expr::Integer(numer), Expr::Integer(denom)) = (a, b) {
    if *denom == 0 {
      return Err(InterpreterError::EvaluationError("Division by zero".into()));
    }
    return Ok(make_rational(*numer, *denom));
  }

  // Simplify (n * expr) / d where n, d are integers → simplify coefficient
  if let Expr::Integer(d) = b
    && *d != 0
    && let Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } = a
  {
    // (Integer * expr) / Integer → (n/d) * expr
    if let Expr::Integer(n) = left.as_ref() {
      let coeff = make_rational(*n, *d);
      return multiply_scalar_by_expr(&coeff, right);
    }
    // (expr * Integer) / Integer → expr * (n/d)
    if let Expr::Integer(n) = right.as_ref() {
      let coeff = make_rational(*n, *d);
      return multiply_scalar_by_expr(&coeff, left);
    }
  }

  // For reals, perform floating-point division
  match (expr_to_num(a), expr_to_num(b)) {
    (Some(x), Some(y)) => {
      if y == 0.0 {
        Err(InterpreterError::EvaluationError("Division by zero".into()))
      } else {
        Ok(Expr::Real(x / y))
      }
    }
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(a.clone()),
      right: Box::new(b.clone()),
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
    return thread_binary_over_lists(args, power_two);
  }

  power_two(&args[0], &args[1])
}

/// Helper for Power of two arguments
fn power_two(base: &Expr, exp: &Expr) -> Result<Expr, InterpreterError> {
  // Special case: 0^0 is Indeterminate (matches Wolfram)
  let base_is_zero = matches!(base, Expr::Integer(0))
    || matches!(base, Expr::Real(f) if *f == 0.0);
  let exp_is_zero = matches!(exp, Expr::Integer(0))
    || matches!(exp, Expr::Real(f) if *f == 0.0);
  if base_is_zero && exp_is_zero {
    let base_str = crate::syntax::expr_to_string(base);
    let exp_str = crate::syntax::expr_to_string(exp);
    // Align exponent above the base in the warning message
    // "Power::indet: Indeterminate expression " is 39 chars
    // Exponent starts at column 39 + len(base), right-align needs + len(exp)
    let padding = 39 + base_str.len() + exp_str.len();
    eprintln!();
    eprintln!("{:>width$}", exp_str, width = padding);
    eprintln!(
      "Power::indet: Indeterminate expression {}  encountered.",
      base_str
    );
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }

  // Special case: integer base with negative integer exponent -> Rational
  if let (Expr::Integer(b), Expr::Integer(e)) = (base, exp)
    && *e < 0
  {
    // b^(-n) = 1 / b^n = Rational[1, b^n]
    let pos_exp = (-*e) as u32;
    if let Some(denom) = b.checked_pow(pos_exp) {
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(denom)],
      });
    }
  }

  // If either operand is Real, result is Real (even if whole number)
  let has_real = matches!(base, Expr::Real(_)) || matches!(exp, Expr::Real(_));

  match (expr_to_num(base), expr_to_num(exp)) {
    (Some(a), Some(b)) => {
      let result = a.powf(b);
      if has_real {
        Ok(Expr::Real(result))
      } else {
        // Both were integers - use num_to_expr to get Integer when result is whole
        Ok(num_to_expr(result))
      }
    }
    _ => Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(exp.clone()),
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

/// Subtract[a, b] - Returns a - b
pub fn subtract_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Subtract".to_string(),
      args: args.to_vec(),
    });
  }
  // Subtract[a, b] = a + (-1 * b)
  let negated_b = times_ast(&[Expr::Integer(-1), args[1].clone()])?;
  plus_ast(&[args[0].clone(), negated_b])
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

  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  for item in &items {
    if let Some(n) = try_eval_to_f64(item) {
      match best_val {
        Some(m) if n > m => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        None => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        _ => {}
      }
    } else {
      // Non-numeric, return symbolic
      return Ok(Expr::FunctionCall {
        name: "Max".to_string(),
        args: args.to_vec(),
      });
    }
  }

  match best_expr {
    Some(expr) => Ok((*expr).clone()),
    None => Ok(num_to_expr(f64::NEG_INFINITY)),
  }
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

  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  for item in &items {
    if let Some(n) = try_eval_to_f64(item) {
      match best_val {
        Some(m) if n < m => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        None => {
          best_val = Some(n);
          best_expr = Some(item);
        }
        _ => {}
      }
    } else {
      return Ok(Expr::FunctionCall {
        name: "Min".to_string(),
        args: args.to_vec(),
      });
    }
  }

  match best_expr {
    Some(expr) => Ok((*expr).clone()),
    None => Ok(num_to_expr(f64::INFINITY)),
  }
}

/// Abs[x] - Absolute value
pub fn abs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Abs expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
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
  if let Some(n) = try_eval_to_f64(&args[0]) {
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
  match &args[0] {
    // Perfect squares: Sqrt[0]=0, Sqrt[1]=1, Sqrt[4]=2, etc.
    Expr::Integer(n) if *n >= 0 => {
      let root = (*n as f64).sqrt();
      if root.fract() == 0.0 && root.abs() < i128::MAX as f64 {
        return Ok(Expr::Integer(root as i128));
      }
      // Simplify: extract largest perfect square factor
      // e.g., Sqrt[12] = 2*Sqrt[3]
      let n_val = *n as u64;
      let mut outside = 1u64;
      let mut inside = n_val;
      let mut factor = 2u64;
      while factor * factor <= inside {
        while inside.is_multiple_of(factor * factor) {
          outside *= factor;
          inside /= factor * factor;
        }
        factor += 1;
      }
      if outside > 1 && inside > 1 {
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(outside as i128),
            Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::Integer(inside as i128)],
            },
          ],
        });
      }
      // Not a perfect square, return symbolic
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    }
    // Sqrt[Rational[a, b]] — evaluate for perfect squares
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *n >= 0
        && *d > 0
      {
        let nr = (*n as f64).sqrt();
        let dr = (*d as f64).sqrt();
        if nr.fract() == 0.0 && dr.fract() == 0.0 {
          return Ok(make_rational(nr as i128, dr as i128));
        }
      }
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    }
    Expr::Real(f) if *f >= 0.0 => Ok(Expr::Real(f.sqrt())),
    _ => Ok(Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Surd[x, n] - Real-valued nth root
pub fn surd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Surd expects exactly 2 arguments".into(),
    ));
  }
  match (expr_to_num(&args[0]), expr_to_num(&args[1])) {
    (Some(x), Some(n)) => {
      if n == 0.0 {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Real-valued nth root: sign(x) * |x|^(1/n)
      let result = if x < 0.0 && n.fract() == 0.0 && (n as i128) % 2 != 0 {
        // Odd integer root of negative number
        -((-x).powf(1.0 / n))
      } else if x < 0.0 {
        // Even root of negative number - return symbolic
        return Ok(Expr::FunctionCall {
          name: "Surd".to_string(),
          args: args.to_vec(),
        });
      } else {
        x.powf(1.0 / n)
      };
      Ok(num_to_expr(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Surd".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Floor[x] - Floor function
pub fn floor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Floor expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
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
  if let Some(n) = try_eval_to_f64(&args[0]) {
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
      if let (Some(a), Some(b)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
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
      if let (Some(a), Some(b)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
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
      // Try to compute exact integer sum first
      let mut int_sum: Option<i128> = Some(0);
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => {
            if let Some(s) = int_sum {
              int_sum = s.checked_add(*n);
            }
          }
          Expr::Real(_) => {
            has_real = true;
            int_sum = None;
          }
          _ => {
            int_sum = None;
          }
        }
      }

      if let Some(sum) = int_sum {
        // All integers - return exact rational or integer
        let count = items.len() as i128;
        if sum % count == 0 {
          Ok(Expr::Integer(sum / count))
        } else {
          // Return as Rational
          let g = gcd_helper(sum.abs(), count);
          let num = sum / g;
          let denom = count / g;
          Ok(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(denom)],
          })
        }
      } else if has_real {
        // Has real numbers - compute float
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
      } else {
        // Non-numeric elements
        Ok(Expr::FunctionCall {
          name: "Mean".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Helper function to compute GCD
fn gcd_helper(a: i128, b: i128) -> i128 {
  if b == 0 { a } else { gcd_helper(b, a % b) }
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

/// Gamma[n] - Gamma function: Gamma[n] = (n-1)! for positive integers
pub fn gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Gamma expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      if *n <= 0 {
        // Gamma has poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Gamma[n] = (n-1)! for positive integers
      let mut result: i128 = 1;
      for i in 2..*n {
        result = result.saturating_mul(i);
      }
      Ok(Expr::Integer(result))
    }
    Expr::Real(f) => {
      if *f <= 0.0 && f.fract() == 0.0 {
        // Poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Use Stirling's approximation via the standard library's tgamma equivalent
      // Rust doesn't have tgamma in std, but we can compute via the Lanczos approximation
      let result = gamma_fn(*f);
      if result.is_infinite() {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else {
        Ok(Expr::Real(result))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Lanczos approximation for the Gamma function
fn gamma_fn(x: f64) -> f64 {
  if x < 0.5 {
    // Reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
    std::f64::consts::PI
      / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
  } else {
    let x = x - 1.0;
    let g = 7.0;
    let c = [
      0.999_999_999_999_809_9,
      676.5203681218851,
      -1259.1392167224028,
      771.323_428_777_653_1,
      -176.615_029_162_140_6,
      12.507343278686905,
      -0.13857109526572012,
      9.984_369_578_019_572e-6,
      1.5056327351493116e-7,
    ];
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
      sum += ci / (x + i as f64);
    }
    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
  }
}

/// N[expr] or N[expr, n] - Numeric evaluation
pub fn n_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "N expects 1 or 2 arguments".into(),
    ));
  }
  n_eval(&args[0])
}

/// Recursively convert an expression to numeric (Real) form
fn n_eval(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::Real(_) => Ok(expr.clone()),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items.iter().map(n_eval).collect();
      Ok(Expr::List(results?))
    }
    _ => {
      if let Some(v) = try_eval_to_f64(expr) {
        Ok(Expr::Real(v))
      } else {
        Ok(expr.clone())
      }
    }
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

/// Clip[x
pub fn clip_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Clip expects 1 or 2 arguments".into(),
    ));
  }

  let x = match expr_to_num(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Clip".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let (min_val, max_val) = if args.len() == 2 {
    match &args[1] {
      Expr::List(bounds) if bounds.len() == 2 => {
        let min = expr_to_num(&bounds[0]).ok_or_else(|| {
          InterpreterError::EvaluationError("Clip: min must be numeric".into())
        })?;
        let max = expr_to_num(&bounds[1]).ok_or_else(|| {
          InterpreterError::EvaluationError("Clip: max must be numeric".into())
        })?;
        (min, max)
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Clip: second argument must be {min, max}".into(),
        ));
      }
    }
  } else {
    (-1.0, 1.0)
  };

  let clipped = if x < min_val {
    min_val
  } else if x > max_val {
    max_val
  } else {
    x
  };

  Ok(num_to_expr(clipped))
}

/// RandomChoice[list
pub fn random_choice_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  let mut rng = rand::thread_rng();

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomChoice expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => {
      return Err(InterpreterError::EvaluationError(
        "RandomChoice: list cannot be empty".into(),
      ));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomChoice".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    let idx = rng.gen_range(0..items.len());
    Ok(items[idx].clone())
  } else {
    let n = match &args[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomChoice: second argument must be a non-negative integer".into(),
        ));
      }
    };
    let result: Vec<Expr> = (0..n)
      .map(|_| {
        let idx = rng.gen_range(0..items.len());
        items[idx].clone()
      })
      .collect();
    Ok(Expr::List(result))
  }
}

/// RandomSample[list
pub fn random_sample_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::seq::SliceRandom;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "RandomSample expects 1 or 2 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomSample".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut rng = rand::thread_rng();

  if args.len() == 1 {
    let mut shuffled = items.clone();
    shuffled.shuffle(&mut rng);
    Ok(Expr::List(shuffled))
  } else {
    let n = match &args[1] {
      Expr::Integer(n) if *n >= 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomSample: second argument must be a non-negative integer".into(),
        ));
      }
    };
    if n > items.len() {
      return Err(InterpreterError::EvaluationError(format!(
        "RandomSample: cannot sample {} elements from list of length {}",
        n,
        items.len()
      )));
    }
    let sampled: Vec<Expr> =
      items.choose_multiple(&mut rng, n).cloned().collect();
    Ok(Expr::List(sampled))
  }
}

/// Try to express a symbolic expression as a rational multiple of Pi: k*Pi/n.
/// Returns Some((k, n)) in lowest terms, None if not recognized.
/// Handles patterns: Pi, n*Pi, Pi/d, n*Pi/d, n*Degree, Degree, and FunctionCall variants.
fn try_symbolic_pi_fraction(expr: &Expr) -> Option<(i64, i64)> {
  use crate::syntax::BinaryOperator;

  // Helper to extract an integer value
  fn as_int(e: &Expr) -> Option<i64> {
    match e {
      Expr::Integer(n) => Some(*n as i64),
      // Handle negative constants like -Pi parsed as Constant("-Pi")
      _ => None,
    }
  }

  // Helper to check if expr is Pi
  fn is_pi(e: &Expr) -> bool {
    matches!(e, Expr::Constant(name) if name == "Pi")
  }

  // Helper to check if expr is Degree
  fn is_degree(e: &Expr) -> bool {
    matches!(e, Expr::Constant(name) if name == "Degree")
  }

  // Helper to reduce fraction
  fn reduce(k: i64, n: i64) -> (i64, i64) {
    if k == 0 {
      return (0, 1);
    }
    let g = gcd(k.unsigned_abs() as i128, n.unsigned_abs() as i128) as i64;
    let (k, n) = (k / g, n / g);
    if n < 0 { (-k, -n) } else { (k, n) }
  }

  match expr {
    // Pi => (1, 1)
    _ if is_pi(expr) => Some((1, 1)),
    // -Pi => (-1, 1)
    Expr::Constant(name) if name == "-Pi" => Some((-1, 1)),
    // Degree => (1, 180)
    _ if is_degree(expr) => Some((1, 180)),
    // -Degree => (-1, 180)
    Expr::Constant(name) if name == "-Degree" => Some((-1, 180)),
    // Integer(0) => (0, 1) — sin(0) etc.
    Expr::Integer(0) => Some((0, 1)),

    // n * Pi or Pi * n
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if is_pi(right)
        && let Some(n) = as_int(left)
      {
        return Some(reduce(n, 1));
      }
      if is_pi(left)
        && let Some(n) = as_int(right)
      {
        return Some(reduce(n, 1));
      }
      // n * Degree or Degree * n
      if is_degree(right)
        && let Some(n) = as_int(left)
      {
        return Some(reduce(n, 180));
      }
      if is_degree(left)
        && let Some(n) = as_int(right)
      {
        return Some(reduce(n, 180));
      }
      // n * (Pi / d) or (Pi / d) * n
      if let Some(n) = as_int(left)
        && let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: num,
          right: den,
        } = right.as_ref()
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
      }
      if let Some(n) = as_int(right)
        && let Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: num,
          right: den,
        } = left.as_ref()
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
      }
      None
    }

    // Pi / d
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      if is_pi(left)
        && let Some(d) = as_int(right)
      {
        return Some(reduce(1, d));
      }
      // (n * Pi) / d
      if let Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: tl,
        right: tr,
      } = left.as_ref()
      {
        if is_pi(tr)
          && let (Some(n), Some(d)) = (as_int(tl), as_int(right))
        {
          return Some(reduce(n, d));
        }
        if is_pi(tl)
          && let (Some(n), Some(d)) = (as_int(tr), as_int(right))
        {
          return Some(reduce(n, d));
        }
      }
      None
    }

    // FunctionCall("Times", [n, Pi]) or FunctionCall("Times", [n, Degree])
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if is_pi(&args[1])
        && let Some(n) = as_int(&args[0])
      {
        return Some(reduce(n, 1));
      }
      if is_pi(&args[0])
        && let Some(n) = as_int(&args[1])
      {
        return Some(reduce(n, 1));
      }
      if is_degree(&args[1])
        && let Some(n) = as_int(&args[0])
      {
        return Some(reduce(n, 180));
      }
      if is_degree(&args[0])
        && let Some(n) = as_int(&args[1])
      {
        return Some(reduce(n, 180));
      }
      // Times[Rational[k, n], Pi]
      if let Expr::FunctionCall {
        name: rn,
        args: rargs,
      } = &args[0]
        && rn == "Rational"
        && rargs.len() == 2
        && is_pi(&args[1])
        && let (Some(k), Some(n)) = (as_int(&rargs[0]), as_int(&rargs[1]))
      {
        return Some(reduce(k, n));
      }
      None
    }

    // Rational[k, n] * Pi handled via BinaryOp(Times, Rational[k,n], Pi)
    // FunctionCall("Rational", [k, n]) as a standalone — not a Pi fraction
    _ => None,
  }
}

/// Exact Sin value for k*Pi/n. Returns None if no simple exact form.
fn exact_sin(k: i64, n: i64) -> Option<Expr> {
  // Normalize to [0, 2*Pi) i.e., k mod 2n, with k in [0, 2n)
  let period = 2 * n;
  let k = ((k % period) + period) % period;
  // Use symmetry: sin is periodic with period 2*Pi
  // Map to first quadrant and track sign
  let (k_ref, sign) = if k <= n / 2 {
    // First quadrant: [0, Pi/2]
    // k/n is in [0, 1/2], angle is in [0, Pi/2]
    // But we need to check: k*Pi/n in [0, Pi/2] means k/n <= 1/2
    (k, 1i64)
  } else if k <= n {
    // Second quadrant: (Pi/2, Pi]
    // sin(Pi - x) = sin(x)
    (n - k, 1)
  } else if k < 2 * n {
    // Third and fourth quadrants: (Pi, 2*Pi)
    // sin(Pi + x) = -sin(x)
    if k <= 3 * n / 2 {
      (k - n, -1)
    } else {
      (2 * n - k, -1)
    }
  } else {
    (0, 1) // k == 2n, sin(2*Pi) = 0
  };

  // Reduce k_ref/n to lowest terms for table lookup
  let g = gcd(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);
  // Now compute sin(kr * Pi / nr) for first quadrant reference angle
  let val = match (kr, nr) {
    (0, _) => Expr::Integer(0),
    // sin(Pi/6) = 1/2
    (1, 6) => make_rational(1, 2),
    // sin(Pi/4) = 1/Sqrt[2]
    (1, 4) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(2)],
      }),
    },
    // sin(Pi/3) = Sqrt[3]/2
    (1, 3) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
      right: Box::new(Expr::Integer(2)),
    },
    // sin(Pi/2) = 1
    (1, 2) => Expr::Integer(1),
    _ => return None,
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Exact Cos value for k*Pi/n. Uses cos(x) = sin(Pi/2 - x).
fn exact_cos(k: i64, n: i64) -> Option<Expr> {
  // cos(k*Pi/n) = sin(Pi/2 - k*Pi/n) = sin((n - 2k)*Pi/(2n))
  // Simplify: cos(k*Pi/n) = sin((n/2 - k)*Pi/n) -- only works if n is even
  // Better: use direct table
  let period = 2 * n;
  let k = ((k % period) + period) % period;
  // Map to first quadrant
  let (k_ref, sign) = if k <= n / 2 {
    (k, 1i64)
  } else if k <= n {
    (n - k, -1)
  } else if k <= 3 * n / 2 {
    (k - n, -1)
  } else {
    (2 * n - k, 1)
  };

  // Reduce k_ref/n to lowest terms for table lookup
  let g = gcd(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);
  let val = match (kr, nr) {
    (0, _) => Expr::Integer(1),
    // cos(Pi/6) = Sqrt[3]/2
    (1, 6) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
      right: Box::new(Expr::Integer(2)),
    },
    // cos(Pi/4) = 1/Sqrt[2]
    (1, 4) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(2)],
      }),
    },
    // cos(Pi/3) = 1/2
    (1, 3) => make_rational(1, 2),
    // cos(Pi/2) = 0
    (1, 2) => Expr::Integer(0),
    _ => return None,
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Exact Tan value for k*Pi/n.
/// Tan has period Pi, so normalize k mod n. Undefined when cos = 0.
fn exact_tan(k: i64, n: i64) -> Option<Expr> {
  // Tan has period Pi, so reduce k*Pi/n mod Pi => (k mod n)*Pi/n
  let k_mod = ((k % n) + n) % n; // in [0, n)
  // Use symmetry: tan(-x) = -tan(x), tan(Pi - x) = -tan(x)
  // Normalize to [0, Pi/2) i.e., k_mod/n in [0, 1/2)
  // Check for Pi/2: k_mod*2 == n means angle is Pi/2 (undefined)
  if k_mod * 2 == n {
    return None; // tan(Pi/2) is undefined
  }
  let (k_ref, n_ref, sign) = if k_mod * 2 < n {
    // First half [0, Pi/2): positive
    (k_mod, n, 1i64)
  } else {
    // Second half (Pi/2, Pi): tan(Pi - x) = -tan(x)
    (n - k_mod, n, -1)
  };
  // Reduce fraction k_ref/n_ref
  let g = gcd(k_ref as i128, n_ref as i128) as i64;
  let (kr, nr) = (k_ref / g, n_ref / g);

  let val = match (kr, nr) {
    (0, _) => Expr::Integer(0),
    // tan(Pi/6) = 1/Sqrt[3]
    (1, 6) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
    },
    // tan(Pi/4) = 1
    (1, 4) => Expr::Integer(1),
    // tan(Pi/3) = Sqrt[3]
    (1, 3) => Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Integer(3)],
    },
    _ => return None,
  };

  if sign == -1 {
    if matches!(val, Expr::Integer(0)) {
      Some(Expr::Integer(0))
    } else {
      Some(negate_expr(val))
    }
  } else {
    Some(val)
  }
}

/// Negate an Expr, simplifying integer, rational, and division cases
fn negate_expr(expr: Expr) -> Expr {
  match expr {
    Expr::Integer(n) => Expr::Integer(-n),
    Expr::Real(f) => Expr::Real(-f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0] {
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()],
        }
      } else {
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), Expr::FunctionCall { name, args }],
        }
      }
    }
    // -a/b => (-a)/b
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(negate_expr(*left)),
      right,
    },
    other => Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), other],
    },
  }
}

/// Sin, Cos, Tan - Trigonometric functions (fully symbolic)
/// Only evaluate to float for Real arguments. For integer/symbolic args,
/// try exact Pi-fraction lookup, otherwise return unevaluated.
pub fn sin_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sin expects 1 argument".into(),
    ));
  }
  // Real args: evaluate numerically
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.sin()));
  }
  // Try symbolic Pi-fraction: Sin[k*Pi/n]
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_sin(k, n)
  {
    return Ok(exact);
  }
  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Sin".to_string(),
    args: args.to_vec(),
  })
}

pub fn cos_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cos expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.cos()));
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_cos(k, n)
  {
    return Ok(exact);
  }
  Ok(Expr::FunctionCall {
    name: "Cos".to_string(),
    args: args.to_vec(),
  })
}

pub fn tan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tan expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.tan()));
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_tan(k, n)
  {
    return Ok(exact);
  }
  Ok(Expr::FunctionCall {
    name: "Tan".to_string(),
    args: args.to_vec(),
  })
}

pub fn exp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Exp expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(1)),
    Expr::Real(f) => Ok(Expr::Real(f.exp())),
    _ => Ok(Expr::FunctionCall {
      name: "Exp".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Log[1] = 0
      if matches!(&args[0], Expr::Integer(1)) {
        return Ok(Expr::Integer(0));
      }
      if let Expr::Real(f) = &args[0] {
        if *f > 0.0 {
          return Ok(Expr::Real(f.ln()));
        } else {
          return Err(InterpreterError::EvaluationError(
            "Log: argument must be positive".into(),
          ));
        }
      }
      Ok(Expr::FunctionCall {
        name: "Log".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // Log[base, x] — evaluate only for Real args
      if let (Expr::Real(base), Expr::Real(x)) = (&args[0], &args[1])
        && *base > 0.0
        && *base != 1.0
        && *x > 0.0
      {
        return Ok(Expr::Real(x.ln() / base.ln()));
      }
      Ok(Expr::FunctionCall {
        name: "Log".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "Log expects 1 or 2 arguments".into(),
    )),
  }
}

/// Log10[x] - Base-10 logarithm
pub fn log10_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log10 expects exactly 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0]
    && *f > 0.0
  {
    return Ok(Expr::Real(f.log10()));
  }
  Ok(Expr::FunctionCall {
    name: "Log10".to_string(),
    args: args.to_vec(),
  })
}

/// Log2[x] - Base-2 logarithm
pub fn log2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Log2 expects exactly 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0]
    && *f > 0.0
  {
    return Ok(Expr::Real(f.log2()));
  }
  Ok(Expr::FunctionCall {
    name: "Log2".to_string(),
    args: args.to_vec(),
  })
}

/// ArcSin[x] - Inverse sine (symbolic)
pub fn arcsin_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSin expects exactly 1 argument".into(),
    ));
  }
  // Exact values: ArcSin[0] = 0, ArcSin[1] = Pi/2, ArcSin[-1] = -Pi/2
  // ArcSin[1/2] = Pi/6, ArcSin[-1/2] = -Pi/6
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Integer(-1) => {
      return Ok(negate_expr(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      }));
    }
    Expr::Real(f) => {
      if (-1.0..=1.0).contains(f) {
        return Ok(Expr::Real(f.asin()));
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcSin".to_string(),
    args: args.to_vec(),
  })
}

/// ArcCos[x] - Inverse cosine (symbolic)
pub fn arccos_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCos expects exactly 1 argument".into(),
    ));
  }
  // Exact values: ArcCos[0] = Pi/2, ArcCos[1] = 0, ArcCos[-1] = Pi
  match &args[0] {
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Integer(0) => {
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Integer(-1) => return Ok(Expr::Constant("Pi".to_string())),
    Expr::Real(f) => {
      if (-1.0..=1.0).contains(f) {
        return Ok(Expr::Real(f.acos()));
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcCos".to_string(),
    args: args.to_vec(),
  })
}

/// ArcTan[x] - Inverse tangent (symbolic)
pub fn arctan_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcTan expects exactly 1 argument".into(),
    ));
  }
  // Exact values: ArcTan[0] = 0, ArcTan[1] = Pi/4, ArcTan[-1] = -Pi/4
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      });
    }
    Expr::Integer(-1) => {
      return Ok(negate_expr(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      }));
    }
    Expr::Real(f) => return Ok(Expr::Real(f.atan())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcTan".to_string(),
    args: args.to_vec(),
  })
}

/// Prime[n] - Returns the nth prime number
pub fn prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Prime expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      if *n < 1 {
        return Err(InterpreterError::EvaluationError(
          "Prime function argument must be a positive integer greater than 0"
            .into(),
        ));
      }
      Ok(Expr::Integer(crate::nth_prime(*n as usize) as i128))
    }
    Expr::Real(f) => {
      if f.fract() != 0.0 || *f < 1.0 {
        return Err(InterpreterError::EvaluationError(
          "Prime function argument must be a positive integer greater than 0"
            .into(),
        ));
      }
      Ok(Expr::Integer(crate::nth_prime(*f as usize) as i128))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Prime".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Fibonacci[n] - Returns the nth Fibonacci number
pub fn fibonacci_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match &args[0] {
    Expr::Integer(n) => {
      if *n < 0 {
        // F(-n) = (-1)^(n+1) * F(n)
        let pos_n = (-*n) as u128;
        let fib = fibonacci_number(pos_n);
        let sign = if pos_n.is_multiple_of(2) {
          -1i128
        } else {
          1i128
        };
        Ok(Expr::Integer(sign * fib))
      } else {
        Ok(Expr::Integer(fibonacci_number(*n as u128)))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn fibonacci_number(n: u128) -> i128 {
  if n == 0 {
    return 0;
  }
  let mut a: i128 = 0;
  let mut b: i128 = 1;
  for _ in 1..n {
    let tmp = a + b;
    a = b;
    b = tmp;
  }
  b
}

/// IntegerDigits[n
pub fn integer_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits expects 1 or 2 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) => n.unsigned_abs(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerDigits".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let base: u128 = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) if *b >= 2 => *b as u128,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "IntegerDigits: base must be an integer >= 2".into(),
        ));
      }
    }
  } else {
    10
  };

  if n == 0 {
    return Ok(Expr::List(vec![Expr::Integer(0)]));
  }

  let mut digits = Vec::new();
  let mut num = n;
  while num > 0 {
    digits.push(Expr::Integer((num % base) as i128));
    num /= base;
  }
  digits.reverse();
  Ok(Expr::List(digits))
}

/// FromDigits[list
pub fn from_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  let base: i128 = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) if *b >= 2 => *b,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "FromDigits: base must be an integer >= 2".into(),
        ));
      }
    }
  } else {
    10
  };

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FromDigits".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut result: i128 = 0;
  for item in items {
    if let Expr::Integer(d) = item {
      if *d < 0 || *d >= base {
        return Err(InterpreterError::EvaluationError(format!(
          "FromDigits: invalid digit {} for base {}",
          d, base
        )));
      }
      result = result * base + *d;
    } else {
      return Ok(Expr::FunctionCall {
        name: "FromDigits".to_string(),
        args: args.to_vec(),
      });
    }
  }
  Ok(Expr::Integer(result))
}

/// FactorInteger[n] - Returns the prime factorization of n
pub fn factor_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger expects exactly 1 argument".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FactorInteger".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
    ));
  }

  let mut factors: Vec<Expr> = Vec::new();
  let mut num = n.unsigned_abs();

  if n < 0 {
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)]));
  }

  if num == 1 {
    return Ok(Expr::List(factors));
  }

  // Handle factor of 2
  let mut count = 0i128;
  while num % 2 == 0 {
    count += 1;
    num /= 2;
  }
  if count > 0 {
    factors.push(Expr::List(vec![Expr::Integer(2), Expr::Integer(count)]));
  }

  // Handle odd factors
  let mut i: u128 = 3;
  while i * i <= num {
    let mut count = 0i128;
    while num % i == 0 {
      count += 1;
      num /= i;
    }
    if count > 0 {
      factors.push(Expr::List(vec![
        Expr::Integer(i as i128),
        Expr::Integer(count),
      ]));
    }
    i += 2;
  }

  if num > 1 {
    factors.push(Expr::List(vec![
      Expr::Integer(num as i128),
      Expr::Integer(1),
    ]));
  }

  Ok(Expr::List(factors))
}

/// Divisors[n] - Returns a sorted list of all divisors of n
pub fn divisors_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Divisors expects exactly 1 argument".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n != 0 => n.unsigned_abs(),
    Expr::Integer(_) => {
      return Err(InterpreterError::EvaluationError(
        "Divisors: argument cannot be zero".into(),
      ));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Divisors".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut divs = Vec::new();
  let sqrt_n = (n as f64).sqrt() as u128;

  for i in 1..=sqrt_n {
    if n % i == 0 {
      divs.push(i);
      if i != n / i {
        divs.push(n / i);
      }
    }
  }

  divs.sort();
  Ok(Expr::List(
    divs.into_iter().map(|d| Expr::Integer(d as i128)).collect(),
  ))
}

/// DivisorSigma[k, n] - Returns the sum of the k-th powers of divisors of n
pub fn divisor_sigma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSigma expects exactly 2 arguments".into(),
    ));
  }

  let k = match &args[0] {
    Expr::Integer(k) if *k >= 0 => *k as u32,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: first argument must be a non-negative integer".into(),
      ));
    }
  };

  let n = match &args[1] {
    Expr::Integer(n) if *n != 0 => n.unsigned_abs(),
    Expr::Integer(_) => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: second argument cannot be zero".into(),
      ));
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DivisorSigma".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let sqrt_n = (n as f64).sqrt() as u128;
  let mut sum: u128 = 0;

  for i in 1..=sqrt_n {
    if n % i == 0 {
      sum += i.pow(k);
      if i != n / i {
        sum += (n / i).pow(k);
      }
    }
  }

  Ok(Expr::Integer(sum as i128))
}

/// MoebiusMu[n] - Returns the Möbius function value
pub fn moebius_mu_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MoebiusMu expects exactly 1 argument".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 1 => *n as u128,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MoebiusMu: argument must be a positive integer".into(),
      ));
    }
  };

  if n == 1 {
    return Ok(Expr::Integer(1));
  }

  let mut num = n;
  let mut prime_count = 0;

  // Check for factor 2
  if num % 2 == 0 {
    prime_count += 1;
    num /= 2;
    if num % 2 == 0 {
      return Ok(Expr::Integer(0)); // Has squared factor
    }
  }

  // Check odd factors
  let mut i: u128 = 3;
  while i * i <= num {
    if num % i == 0 {
      prime_count += 1;
      num /= i;
      if num % i == 0 {
        return Ok(Expr::Integer(0)); // Has squared factor
      }
    }
    i += 2;
  }

  if num > 1 {
    prime_count += 1;
  }

  Ok(Expr::Integer(if prime_count % 2 == 0 { 1 } else { -1 }))
}

/// EulerPhi[n] - Returns Euler's totient function
pub fn euler_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EulerPhi expects exactly 1 argument".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 1 => *n as u128,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "EulerPhi: argument must be a positive integer".into(),
      ));
    }
  };

  let mut num = n;
  let mut result = n;

  let mut p: u128 = 2;
  while p * p <= num {
    if num % p == 0 {
      while num % p == 0 {
        num /= p;
      }
      result -= result / p;
    }
    p += 1;
  }

  if num > 1 {
    result -= result / num;
  }

  Ok(Expr::Integer(result as i128))
}

/// CoprimeQ[a, b] - Tests if two integers are coprime
pub fn coprime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ expects exactly 2 arguments".into(),
    ));
  }

  let (a, b) = match (&args[0], &args[1]) {
    (Expr::Integer(a), Expr::Integer(b)) => {
      (a.unsigned_abs(), b.unsigned_abs())
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CoprimeQ".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // GCD using Euclidean algorithm
  let (mut a, mut b) = (a.max(1), b.max(1));
  while b != 0 {
    let temp = b;
    b = a % b;
    a = temp;
  }

  Ok(Expr::Identifier(
    if a == 1 { "True" } else { "False" }.to_string(),
  ))
}

/// Re[z] - Real part of a complex number (for real numbers, returns the number itself)
pub fn re_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Re expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_) | Expr::Real(_) => Ok(args[0].clone()),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      Ok(inner[0].clone())
    }
    _ => Ok(Expr::FunctionCall {
      name: "Re".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Im[z] - Imaginary part of a complex number (for real numbers, returns 0)
pub fn im_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Im expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_) | Expr::Real(_) => Ok(Expr::Integer(0)),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      Ok(inner[1].clone())
    }
    _ => Ok(Expr::FunctionCall {
      name: "Im".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Conjugate[z] - Complex conjugate (for real numbers, returns the number itself)
pub fn conjugate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Conjugate expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_) | Expr::Real(_) => Ok(args[0].clone()),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      // Negate the imaginary part
      let neg_imag = match &inner[1] {
        Expr::Integer(n) => Expr::Integer(-*n),
        Expr::Real(f) => Expr::Real(-*f),
        other => Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(other.clone()),
        },
      };
      Ok(Expr::FunctionCall {
        name: "Complex".to_string(),
        args: vec![inner[0].clone(), neg_imag],
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Conjugate".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Arg[z] - Argument (phase angle) of a complex number
pub fn arg_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Arg expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      if *n > 0 {
        Ok(Expr::Integer(0))
      } else if *n < 0 {
        Ok(Expr::Identifier("Pi".to_string()))
      } else {
        Ok(Expr::Integer(0))
      }
    }
    Expr::Real(f) => {
      if *f > 0.0 {
        Ok(Expr::Integer(0))
      } else if *f < 0.0 {
        Ok(Expr::Identifier("Pi".to_string()))
      } else {
        Ok(Expr::Integer(0))
      }
    }
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      if let (Some(re), Some(im)) =
        (expr_to_num(&inner[0]), expr_to_num(&inner[1]))
      {
        Ok(Expr::Real(im.atan2(re)))
      } else {
        Ok(Expr::FunctionCall {
          name: "Arg".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Arg".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Rationalize[x
pub fn rationalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Rationalize expects 1 or 2 arguments".into(),
    ));
  }

  let x = match expr_to_num(&args[0]) {
    Some(x) => x,
    None => {
      return Ok(Expr::FunctionCall {
        name: "Rationalize".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // For integers, just return the integer
  if x.fract() == 0.0 {
    return Ok(Expr::Integer(x as i128));
  }

  let tolerance = if args.len() == 2 {
    expr_to_num(&args[1]).unwrap_or(f64::EPSILON)
  } else {
    f64::EPSILON
  };

  let max_denom: i64 = if args.len() == 1 { 100000 } else { i64::MAX };

  let (num, denom) = find_rational(x, tolerance, max_denom);

  // Verify the approximation is within tolerance
  let approx = num as f64 / denom as f64;
  if (approx - x).abs() >= tolerance {
    return Ok(num_to_expr(x));
  }

  if denom == 1 {
    Ok(Expr::Integer(num as i128))
  } else {
    // Return as a fraction using Rational or as BinaryOp Divide
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num as i128), Expr::Integer(denom as i128)],
    })
  }
}

/// Find best rational approximation using continued fractions
fn find_rational(x: f64, tolerance: f64, max_denom: i64) -> (i64, i64) {
  if x == 0.0 {
    return (0, 1);
  }

  let sign = if x < 0.0 { -1 } else { 1 };
  let x = x.abs();

  let mut p0: i64 = 0;
  let mut q0: i64 = 1;
  let mut p1: i64 = 1;
  let mut q1: i64 = 0;

  let mut xi = x;

  for _ in 0..50 {
    let ai = xi.floor() as i64;
    let p2 = ai * p1 + p0;
    let q2 = ai * q1 + q0;

    if q2 == 0 || q2 > max_denom {
      break;
    }

    let approx = p2 as f64 / q2 as f64;
    if (approx - x).abs() < tolerance {
      return (sign * p2, q2);
    }

    let frac = xi - ai as f64;
    if frac.abs() < 1e-15 {
      break;
    }
    xi = 1.0 / frac;

    p0 = p1;
    q0 = q1;
    p1 = p2;
    q1 = q2;
  }

  (sign * p1, q1)
}
