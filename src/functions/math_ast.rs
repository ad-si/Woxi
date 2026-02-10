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
      "Sec" if args.len() == 1 => try_eval_to_f64(&args[0]).and_then(|v| {
        let c = v.cos();
        if c != 0.0 { Some(1.0 / c) } else { None }
      }),
      "Csc" if args.len() == 1 => try_eval_to_f64(&args[0]).and_then(|v| {
        let s = v.sin();
        if s != 0.0 { Some(1.0 / s) } else { None }
      }),
      "Cot" if args.len() == 1 => try_eval_to_f64(&args[0]).and_then(|v| {
        let s = v.sin();
        if s != 0.0 { Some(v.cos() / s) } else { None }
      }),
      "ArcSin" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.asin())
      }
      "ArcCos" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.acos())
      }
      "ArcTan" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.atan())
      }
      "Sinh" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.sinh()),
      "Cosh" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.cosh()),
      "Tanh" if args.len() == 1 => try_eval_to_f64(&args[0]).map(|v| v.tanh()),
      "Coth" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| 1.0 / v.tanh())
      }
      "Sech" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| 1.0 / v.cosh())
      }
      "Csch" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| 1.0 / v.sinh())
      }
      "ArcSinh" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.asinh())
      }
      "ArcCosh" if args.len() == 1 => try_eval_to_f64(&args[0])
        .and_then(|v| if v >= 1.0 { Some(v.acosh()) } else { None }),
      "ArcTanh" if args.len() == 1 => try_eval_to_f64(&args[0])
        .and_then(|v| if v.abs() < 1.0 { Some(v.atanh()) } else { None }),
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

/// Convert a BigInt to Expr::Integer if it fits in i128, otherwise Expr::BigInteger
pub fn bigint_to_expr(n: num_bigint::BigInt) -> Expr {
  use num_traits::ToPrimitive;
  match n.to_i128() {
    Some(i) => Expr::Integer(i),
    None => Expr::BigInteger(n),
  }
}

/// Check if an expression requires BigInt arithmetic (exceeds f64 precision).
/// f64 can only represent integers exactly up to 2^53.
fn needs_bigint_arithmetic(expr: &Expr) -> bool {
  match expr {
    Expr::BigInteger(_) => true,
    Expr::Integer(n) => n.unsigned_abs() > (1u128 << 53),
    _ => false,
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
/// Public wrapper for creating rational expressions
pub fn make_rational_pub(numer: i128, denom: i128) -> Expr {
  make_rational(numer, denom)
}

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

  // Check if any argument needs BigInt arithmetic (BigInteger or large Integer exceeding f64 precision)
  let has_bigint = flat_args.iter().any(needs_bigint_arithmetic);

  if has_bigint {
    use num_bigint::BigInt;
    let mut big_sum = BigInt::from(0);
    let mut all_int = true;
    let mut symbolic_args: Vec<Expr> = Vec::new();

    for arg in &flat_args {
      match arg {
        Expr::Integer(n) => big_sum += BigInt::from(*n),
        Expr::BigInteger(n) => big_sum += n,
        _ => {
          all_int = false;
          symbolic_args.push(arg.clone());
        }
      }
    }

    if all_int {
      return Ok(bigint_to_expr(big_sum));
    }

    let mut final_args: Vec<Expr> = Vec::new();
    if big_sum != BigInt::from(0) {
      final_args.push(bigint_to_expr(big_sum));
    }
    final_args.extend(symbolic_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Plus".to_string(),
      args: final_args,
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

  // Check if any argument needs BigInt arithmetic (BigInteger or large Integer exceeding f64 precision)
  let has_bigint = args.iter().any(needs_bigint_arithmetic);

  if has_bigint {
    use num_bigint::BigInt;
    let mut big_product = BigInt::from(1);
    let mut all_int = true;
    let mut symbolic_args: Vec<Expr> = Vec::new();

    for arg in args {
      match arg {
        Expr::Integer(n) => big_product *= BigInt::from(*n),
        Expr::BigInteger(n) => big_product *= n,
        _ => {
          all_int = false;
          symbolic_args.push(arg.clone());
        }
      }
    }

    if all_int {
      return Ok(bigint_to_expr(big_product));
    }

    // 0 * anything = 0
    if big_product == BigInt::from(0) {
      return Ok(Expr::Integer(0));
    }

    let mut final_args: Vec<Expr> = Vec::new();
    if big_product != BigInt::from(1) {
      final_args.push(bigint_to_expr(big_product));
    }
    final_args.extend(symbolic_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    });
  }

  // Separate numeric and symbolic arguments
  let mut numeric_product = 1.0;
  let mut any_real = false;
  let mut symbolic_args: Vec<Expr> = Vec::new();

  for arg in args {
    if let Some(n) = expr_to_num(arg) {
      numeric_product *= n;
      if matches!(arg, Expr::Real(_)) {
        any_real = true;
      }
    } else {
      symbolic_args.push(arg.clone());
    }
  }

  // If all arguments are numeric, return the product
  if symbolic_args.is_empty() {
    if any_real {
      return Ok(Expr::Real(numeric_product));
    } else {
      return Ok(num_to_expr(numeric_product));
    }
  }

  // 0 * anything = 0
  if numeric_product == 0.0 {
    return Ok(Expr::Integer(0));
  }

  // Build final args: numeric coefficient (if not 1) + symbolic terms
  let mut final_args: Vec<Expr> = Vec::new();
  if numeric_product != 1.0 || any_real {
    if any_real {
      final_args.push(Expr::Real(numeric_product));
    } else {
      final_args.push(num_to_expr(numeric_product));
    }
  }
  final_args.extend(symbolic_args);

  if final_args.len() == 1 {
    Ok(final_args.remove(0))
  } else {
    Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    })
  }
}

/// Minus[a] - Unary negation only
/// Note: Minus with 2 arguments is not valid in Wolfram Language
/// (use Subtract for that)
pub fn minus_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() == 1 {
    // Unary minus - handle BigInteger and large Integer directly
    match &args[0] {
      Expr::Integer(n) => return Ok(Expr::Integer(-n)),
      Expr::BigInteger(n) => return Ok(bigint_to_expr(-n)),
      Expr::Real(f) => return Ok(Expr::Real(-f)),
      _ => {}
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

  // Special case: Integer^Rational — keep symbolic unless result is exact integer
  if let Expr::Integer(b) = base
    && let Expr::FunctionCall { name, args: rargs } = exp
    && name == "Rational"
    && rargs.len() == 2
    && let (Expr::Integer(numer), Expr::Integer(denom)) = (&rargs[0], &rargs[1])
  {
    // Try to compute exact integer root
    let result = (*b as f64).powf(*numer as f64 / *denom as f64);
    if result.fract() == 0.0 && result.is_finite() {
      return Ok(Expr::Integer(result as i128));
    }
    // Not exact — keep symbolic
    return Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: Box::new(base.clone()),
      right: Box::new(exp.clone()),
    });
  }

  // Integer^Integer with non-negative exponent: use exact BigInt arithmetic
  if let (Expr::Integer(b), Expr::Integer(e)) = (base, exp)
    && *e >= 0
  {
    use num_bigint::BigInt;
    let base_big = BigInt::from(*b);
    let result = num_traits::pow::pow(base_big, *e as usize);
    return Ok(bigint_to_expr(result));
  }

  // BigInteger base with integer exponent
  if let (Expr::BigInteger(b), Expr::Integer(e)) = (base, exp)
    && *e >= 0
  {
    let result = num_traits::pow::pow(b.clone(), *e as usize);
    return Ok(bigint_to_expr(result));
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
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Round expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    // Round[x, a] - round x to nearest multiple of a
    if let (Some(x), Some(a)) =
      (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
    {
      if a == 0.0 {
        return Ok(args[0].clone());
      }
      let rounded = (x / a).round() * a;
      if rounded.fract() == 0.0 && rounded.abs() < i128::MAX as f64 {
        return Ok(Expr::Integer(rounded as i128));
      }
      return Ok(Expr::Real(rounded));
    }
    return Ok(Expr::FunctionCall {
      name: "Round".to_string(),
      args: args.to_vec(),
    });
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
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
      _ => None,
    }
  }

  // Helper to extract a rational value (k, n) from Rational[k, n]
  fn as_rational(e: &Expr) -> Option<(i64, i64)> {
    if let Expr::FunctionCall { name, args } = e
      && name == "Rational"
      && args.len() == 2
      && let (Some(k), Some(n)) = (as_int(&args[0]), as_int(&args[1]))
    {
      Some((k, n))
    } else {
      None
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
      // Rational[k, d] * Pi or Pi * Rational[k, d]
      if is_pi(right)
        && let Some((k, d)) = as_rational(left)
      {
        return Some(reduce(k, d));
      }
      if is_pi(left)
        && let Some((k, d)) = as_rational(right)
      {
        return Some(reduce(k, d));
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
  // Check for Pi/2: k_mod*2 == n means angle is Pi/2 → ComplexInfinity
  if k_mod * 2 == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
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

fn exact_sec(k: i64, n: i64) -> Option<Expr> {
  // Sec has period 2*Pi, and Sec(-x) = Sec(x), Sec(Pi-x) = -Sec(x)
  // Reduce to [0, 2*Pi)
  let k_mod = ((k % (2 * n)) + 2 * n) % (2 * n);
  // Use Sec(x) = Sec(2*Pi - x) symmetry to reduce to [0, Pi]
  let k2 = if k_mod > n { 2 * n - k_mod } else { k_mod };
  // In [0, Pi]: Sec(Pi - x) = -Sec(x), so reduce to [0, Pi/2]
  let (k_ref, sign) = if k2 * 2 > n {
    (n - k2, -1i64) // Pi - angle
  } else {
    (k2, 1)
  };
  // Check for Pi/2: k_ref*2 == n means Sec(Pi/2) = ComplexInfinity
  if k_ref * 2 == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  let g = gcd(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    (0, _) => Expr::Integer(1),
    // Sec(Pi/6) = 2/Sqrt[3]
    (1, 6) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
    },
    // Sec(Pi/4) = Sqrt[2]
    (1, 4) => Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Integer(2)],
    },
    // Sec(Pi/3) = 2
    (1, 3) => Expr::Integer(2),
    _ => return None,
  };

  if sign == -1 {
    Some(negate_expr(val))
  } else {
    Some(val)
  }
}

fn exact_csc(k: i64, n: i64) -> Option<Expr> {
  // Csc has period 2*Pi, Csc(-x) = -Csc(x), Csc(Pi-x) = Csc(x)
  // Reduce to [0, 2*Pi)
  let k_mod = ((k % (2 * n)) + 2 * n) % (2 * n);
  // Csc(0) and Csc(Pi) are ComplexInfinity
  if k_mod == 0 || k_mod == n {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Use Csc(2*Pi - x) = -Csc(x) to reduce to [0, Pi]
  let (k2, sign1) = if k_mod > n {
    (2 * n - k_mod, -1i64)
  } else {
    (k_mod, 1)
  };
  // In (0, Pi): Csc(Pi - x) = Csc(x), so reduce to (0, Pi/2]
  let k_ref = if k2 * 2 > n { n - k2 } else { k2 };

  let g = gcd(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    // Csc(Pi/2) = 1
    (1, 2) => Expr::Integer(1),
    // Csc(Pi/6) = 2
    (1, 6) => Expr::Integer(2),
    // Csc(Pi/4) = Sqrt[2]
    (1, 4) => Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Integer(2)],
    },
    // Csc(Pi/3) = 2/Sqrt[3]
    (1, 3) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(2)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
    },
    _ => return None,
  };

  if sign1 == -1 {
    Some(negate_expr(val))
  } else {
    Some(val)
  }
}

fn exact_cot(k: i64, n: i64) -> Option<Expr> {
  // Cot has period Pi, Cot(-x) = -Cot(x)
  // Reduce k*Pi/n mod Pi => (k mod n)*Pi/n
  let k_mod = ((k % n) + n) % n;
  // Cot(0) = ComplexInfinity
  if k_mod == 0 {
    return Some(Expr::Identifier("ComplexInfinity".to_string()));
  }
  // Cot(Pi/2) = 0
  if k_mod * 2 == n {
    return Some(Expr::Integer(0));
  }
  // Use Cot(Pi - x) = -Cot(x) to reduce to (0, Pi/2)
  let (k_ref, sign) = if k_mod * 2 > n {
    (n - k_mod, -1i64)
  } else {
    (k_mod, 1)
  };

  let g = gcd(k_ref as i128, n as i128) as i64;
  let (kr, nr) = (k_ref / g, n / g);

  let val = match (kr, nr) {
    // Cot(Pi/6) = Sqrt[3]
    (1, 6) => Expr::FunctionCall {
      name: "Sqrt".to_string(),
      args: vec![Expr::Integer(3)],
    },
    // Cot(Pi/4) = 1
    (1, 4) => Expr::Integer(1),
    // Cot(Pi/3) = 1/Sqrt[3]
    (1, 3) => Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(1)),
      right: Box::new(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(3)],
      }),
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
    // -(a/b) => Times[-1, a/b] to match Wolfram output style
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      // If numerator is an integer, negate it directly: -(n/b) => (-n)/b
      if let Expr::Integer(n) = *left
        && n > 1
      {
        return Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Integer(-n)),
          right,
        };
      }
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left,
          right,
        }),
      }
    }
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

pub fn sec_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sec expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    let c = f.cos();
    if c == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(num_to_expr(1.0 / c));
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_sec(k, n)
  {
    return Ok(exact);
  }
  Ok(Expr::FunctionCall {
    name: "Sec".to_string(),
    args: args.to_vec(),
  })
}

pub fn csc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Csc expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sin();
    if s == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(num_to_expr(1.0 / s));
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_csc(k, n)
  {
    return Ok(exact);
  }
  Ok(Expr::FunctionCall {
    name: "Csc".to_string(),
    args: args.to_vec(),
  })
}

pub fn cot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cot expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sin();
    if s == 0.0 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    return Ok(num_to_expr(f.cos() / s));
  }
  if let Some((k, n)) = try_symbolic_pi_fraction(&args[0])
    && let Some(exact) = exact_cot(k, n)
  {
    return Ok(exact);
  }
  Ok(Expr::FunctionCall {
    name: "Cot".to_string(),
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

// ─── Hyperbolic Trig Functions ────────────────────────────────────

/// Sinh[x] - Hyperbolic sine
pub fn sinh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sinh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.sinh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "Sinh".to_string(),
    args: args.to_vec(),
  })
}

/// Cosh[x] - Hyperbolic cosine
pub fn cosh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Cosh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(f.cosh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "Cosh".to_string(),
    args: args.to_vec(),
  })
}

/// Tanh[x] - Hyperbolic tangent
pub fn tanh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Tanh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.tanh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "Tanh".to_string(),
    args: args.to_vec(),
  })
}

/// Coth[x] - Hyperbolic cotangent
pub fn coth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Coth expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    let t = f.tanh();
    if t == 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Coth: division by zero".into(),
      ));
    }
    return Ok(Expr::Real(1.0 / t));
  }
  Ok(Expr::FunctionCall {
    name: "Coth".to_string(),
    args: args.to_vec(),
  })
}

/// Sech[x] - Hyperbolic secant
pub fn sech_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sech expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(1.0 / f.cosh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "Sech".to_string(),
    args: args.to_vec(),
  })
}

/// Csch[x] - Hyperbolic cosecant
pub fn csch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Csch expects 1 argument".into(),
    ));
  }
  if let Expr::Real(f) = &args[0] {
    let s = f.sinh();
    if s == 0.0 {
      return Err(InterpreterError::EvaluationError(
        "Csch: division by zero".into(),
      ));
    }
    return Ok(Expr::Real(1.0 / s));
  }
  Ok(Expr::FunctionCall {
    name: "Csch".to_string(),
    args: args.to_vec(),
  })
}

/// ArcSinh[x] - Inverse hyperbolic sine
pub fn arcsinh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSinh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.asinh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcSinh".to_string(),
    args: args.to_vec(),
  })
}

/// ArcCosh[x] - Inverse hyperbolic cosine
pub fn arccosh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCosh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if *f >= 1.0 => return Ok(Expr::Real(f.acosh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcCosh".to_string(),
    args: args.to_vec(),
  })
}

/// ArcTanh[x] - Inverse hyperbolic tangent
pub fn arctanh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcTanh expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if f.abs() < 1.0 => return Ok(Expr::Real(f.atanh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcTanh".to_string(),
    args: args.to_vec(),
  })
}

// ─── Number Theory Functions ─────────────────────────────────────

/// DigitCount[n] - counts of each digit 1-9,0 in base 10
/// DigitCount[n, b] - counts of each digit in base b
/// DigitCount[n, b, d] - count of specific digit d in base b
pub fn digit_count_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "DigitCount expects 1 to 3 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => n.abs(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DigitCount".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() >= 2 {
    match &args[1] {
      Expr::Integer(b) if *b >= 2 => *b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitCount".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    10
  };

  // Get digit list in the given base
  let mut digits = Vec::new();
  let mut val = n;
  if val == 0 {
    digits.push(0);
  } else {
    while val > 0 {
      digits.push((val % base) as usize);
      val /= base;
    }
  }

  if args.len() == 3 {
    // DigitCount[n, b, d] - count of specific digit d
    let d = match &args[2] {
      Expr::Integer(d) => *d as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitCount".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let count = digits.iter().filter(|&&x| x == d).count();
    Ok(Expr::Integer(count as i128))
  } else {
    // DigitCount[n] or DigitCount[n, b] - list of counts for digits 1..base-1, 0
    // Wolfram returns counts in order: digit 1, digit 2, ..., digit (base-1), digit 0
    let mut counts = vec![0i128; base as usize];
    for &d in &digits {
      counts[d] += 1;
    }
    // Reorder: digits 1, 2, ..., base-1, 0
    let mut result = Vec::with_capacity(base as usize);
    for d in 1..base as usize {
      result.push(Expr::Integer(counts[d]));
    }
    result.push(Expr::Integer(counts[0]));
    Ok(Expr::List(result))
  }
}

/// DigitSum[n] - sum of digits in base 10
/// DigitSum[n, b] - sum of digits in base b
pub fn digit_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "DigitSum expects 1 or 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => n.abs(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DigitSum".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) if *b >= 2 => *b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "DigitSum".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    10
  };

  let mut sum: i128 = 0;
  let mut val = n;
  if val == 0 {
    return Ok(Expr::Integer(0));
  }
  while val > 0 {
    sum += val % base;
    val /= base;
  }
  Ok(Expr::Integer(sum))
}

// --- Minimal arbitrary-precision unsigned integer for high-precision ContinuedFraction ---

/// Little-endian base-2^64 unsigned integer.
#[derive(Clone, Debug)]
struct BigUint {
  digits: Vec<u64>,
}

impl BigUint {
  fn zero() -> Self {
    BigUint { digits: vec![0] }
  }

  fn from_u64(n: u64) -> Self {
    BigUint { digits: vec![n] }
  }

  fn from_u128(n: u128) -> Self {
    let lo = n as u64;
    let hi = (n >> 64) as u64;
    let mut b = BigUint {
      digits: vec![lo, hi],
    };
    b.trim();
    b
  }

  fn is_zero(&self) -> bool {
    self.digits.iter().all(|&d| d == 0)
  }

  fn trim(&mut self) {
    while self.digits.len() > 1 && *self.digits.last().unwrap() == 0 {
      self.digits.pop();
    }
  }

  fn cmp(&self, other: &BigUint) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let a_len = self.digits.len();
    let b_len = other.digits.len();
    if a_len != b_len {
      return a_len.cmp(&b_len);
    }
    for i in (0..a_len).rev() {
      match self.digits[i].cmp(&other.digits[i]) {
        Ordering::Equal => continue,
        ord => return ord,
      }
    }
    Ordering::Equal
  }

  /// self + other
  fn add(&self, other: &BigUint) -> BigUint {
    let max_len = self.digits.len().max(other.digits.len());
    let mut result = Vec::with_capacity(max_len + 1);
    let mut carry: u64 = 0;
    for i in 0..max_len {
      let a = if i < self.digits.len() {
        self.digits[i]
      } else {
        0
      };
      let b = if i < other.digits.len() {
        other.digits[i]
      } else {
        0
      };
      let (s1, c1) = a.overflowing_add(b);
      let (s2, c2) = s1.overflowing_add(carry);
      result.push(s2);
      carry = (c1 as u64) + (c2 as u64);
    }
    if carry > 0 {
      result.push(carry);
    }
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self - other (assumes self >= other)
  fn sub(&self, other: &BigUint) -> BigUint {
    let mut result = Vec::with_capacity(self.digits.len());
    let mut borrow: u64 = 0;
    for i in 0..self.digits.len() {
      let a = self.digits[i];
      let b = if i < other.digits.len() {
        other.digits[i]
      } else {
        0
      };
      let (s1, c1) = a.overflowing_sub(b);
      let (s2, c2) = s1.overflowing_sub(borrow);
      result.push(s2);
      borrow = (c1 as u64) + (c2 as u64);
    }
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self * other
  fn mul(&self, other: &BigUint) -> BigUint {
    let n = self.digits.len() + other.digits.len();
    let mut result = vec![0u64; n];
    for i in 0..self.digits.len() {
      let mut carry: u128 = 0;
      for j in 0..other.digits.len() {
        let prod = (self.digits[i] as u128) * (other.digits[j] as u128)
          + (result[i + j] as u128)
          + carry;
        result[i + j] = prod as u64;
        carry = prod >> 64;
      }
      if carry > 0 {
        result[i + other.digits.len()] += carry as u64;
      }
    }
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// self * scalar
  fn mul_u64(&self, scalar: u64) -> BigUint {
    let mut result = Vec::with_capacity(self.digits.len() + 1);
    let mut carry: u128 = 0;
    for &d in &self.digits {
      let prod = (d as u128) * (scalar as u128) + carry;
      result.push(prod as u64);
      carry = prod >> 64;
    }
    if carry > 0 {
      result.push(carry as u64);
    }
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// Division: returns (quotient, remainder)
  fn divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
    use std::cmp::Ordering;
    if other.is_zero() {
      panic!("BigUint division by zero");
    }
    match self.cmp(other) {
      Ordering::Less => return (BigUint::zero(), self.clone()),
      Ordering::Equal => return (BigUint::from_u64(1), BigUint::zero()),
      _ => {}
    }
    if other.digits.len() == 1 {
      let d = other.digits[0];
      let (q, r) = self.divmod_u64(d);
      return (q, BigUint::from_u64(r));
    }
    // Long division
    self.long_divmod(other)
  }

  /// Divide by a single u64, returns (quotient, remainder)
  fn divmod_u64(&self, d: u64) -> (BigUint, u64) {
    let mut result = vec![0u64; self.digits.len()];
    let mut rem: u128 = 0;
    for i in (0..self.digits.len()).rev() {
      rem = (rem << 64) | (self.digits[i] as u128);
      result[i] = (rem / d as u128) as u64;
      rem %= d as u128;
    }
    let mut q = BigUint { digits: result };
    q.trim();
    (q, rem as u64)
  }

  /// Long division for multi-digit divisors
  fn long_divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
    // Shift-and-subtract algorithm operating on bits
    let mut remainder = BigUint::zero();
    let self_bits = self.bit_len();
    let mut quotient_digits = vec![0u64; self_bits.div_ceil(64)];
    for i in (0..self_bits).rev() {
      // remainder = remainder << 1 | bit_i(self)
      remainder = remainder.shl1();
      if self.bit(i) {
        remainder.digits[0] |= 1;
      }
      if remainder.cmp(other) != std::cmp::Ordering::Less {
        remainder = remainder.sub(other);
        quotient_digits[i / 64] |= 1u64 << (i % 64);
      }
    }
    let mut q = BigUint {
      digits: quotient_digits,
    };
    q.trim();
    (q, remainder)
  }

  fn bit_len(&self) -> usize {
    if self.is_zero() {
      return 0;
    }
    let top = self.digits.len() - 1;
    top * 64 + (64 - self.digits[top].leading_zeros() as usize)
  }

  fn bit(&self, i: usize) -> bool {
    let word = i / 64;
    let bit = i % 64;
    if word >= self.digits.len() {
      false
    } else {
      (self.digits[word] >> bit) & 1 == 1
    }
  }

  fn shl1(&self) -> BigUint {
    let mut result = Vec::with_capacity(self.digits.len() + 1);
    let mut carry = 0u64;
    for &d in &self.digits {
      result.push((d << 1) | carry);
      carry = d >> 63;
    }
    if carry > 0 {
      result.push(carry);
    }
    let mut r = BigUint { digits: result };
    r.trim();
    r
  }

  /// Convert to i128 if it fits
  fn to_i128(&self) -> Option<i128> {
    match self.digits.len() {
      1 => Some(self.digits[0] as i128),
      2 => {
        let val = (self.digits[1] as u128) << 64 | (self.digits[0] as u128);
        if val <= i128::MAX as u128 {
          Some(val as i128)
        } else {
          None
        }
      }
      _ => None,
    }
  }

  fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
      let (_, r) = a.divmod(&b);
      a = b;
      b = r;
    }
    a
  }
}

/// Signed big rational number: numerator/denominator with sign.
#[derive(Clone)]
struct BigRational {
  num: BigUint,
  den: BigUint,
  negative: bool,
}

impl BigRational {
  fn zero() -> Self {
    BigRational {
      num: BigUint::zero(),
      den: BigUint::from_u64(1),
      negative: false,
    }
  }

  fn from_i64(n: i64) -> Self {
    BigRational {
      num: BigUint::from_u64(n.unsigned_abs()),
      den: BigUint::from_u64(1),
      negative: n < 0,
    }
  }

  fn reduce(&mut self) {
    if self.num.is_zero() {
      self.negative = false;
      self.den = BigUint::from_u64(1);
      return;
    }
    let g = BigUint::gcd(&self.num, &self.den);
    if !g.is_zero() && g.digits != vec![1] {
      let (qn, _) = self.num.divmod(&g);
      let (qd, _) = self.den.divmod(&g);
      self.num = qn;
      self.den = qd;
    }
  }

  /// self + other
  fn add(&self, other: &BigRational) -> BigRational {
    // a/b + c/d = (a*d + c*b) / (b*d) respecting signs
    let ad = self.num.mul(&other.den);
    let cb = other.num.mul(&self.den);
    let bd = self.den.mul(&other.den);
    let (num, negative) = if self.negative == other.negative {
      (ad.add(&cb), self.negative)
    } else {
      match ad.cmp(&cb) {
        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => {
          (ad.sub(&cb), self.negative)
        }
        std::cmp::Ordering::Less => (cb.sub(&ad), other.negative),
      }
    };
    let mut r = BigRational {
      num,
      den: bd,
      negative,
    };
    r.reduce();
    r
  }

  /// self - other
  fn sub(&self, other: &BigRational) -> BigRational {
    let neg_other = BigRational {
      num: other.num.clone(),
      den: other.den.clone(),
      negative: !other.negative,
    };
    self.add(&neg_other)
  }

  /// self * scalar (positive integer)
  fn mul_u64(&self, s: u64) -> BigRational {
    let mut r = BigRational {
      num: self.num.mul_u64(s),
      den: self.den.clone(),
      negative: self.negative,
    };
    r.reduce();
    r
  }

  /// Floor division: returns (floor, remainder) such that self = floor + remainder/1
  /// where 0 <= remainder < 1 (for positive self)
  fn floor_and_remainder(&self) -> (i128, BigRational) {
    let (q, r) = self.num.divmod(&self.den);
    let q_i128 = q.to_i128().unwrap_or(0);
    let floor_val = if self.negative && !r.is_zero() {
      -(q_i128 + 1)
    } else if self.negative {
      -q_i128
    } else {
      q_i128
    };
    // remainder = self - floor_val
    let floor_rat = BigRational::from_i64(floor_val as i64);
    let rem = self.sub(&floor_rat);
    (floor_val, rem)
  }

  /// 1 / self
  fn reciprocal(&self) -> BigRational {
    BigRational {
      num: self.den.clone(),
      den: self.num.clone(),
      negative: self.negative,
    }
  }
}

/// Compute atan(1/k) as a BigRational using the Taylor series:
/// atan(x) = x - x^3/3 + x^5/5 - ...
/// For x = 1/k: atan(1/k) = 1/k - 1/(3*k^3) + 1/(5*k^5) - ...
fn big_atan_recip(k: u64, terms: usize) -> BigRational {
  let mut result = BigRational::zero();
  let k2 = k as u128 * k as u128; // k^2 as u128
  // power_denom tracks k^(2n+1) as BigUint
  let mut power_denom = BigUint::from_u64(k);
  let k2_big = BigUint::from_u128(k2);
  for n in 0..terms {
    let divisor = (2 * n + 1) as u64;
    // term = 1 / (divisor * power_denom)
    let term = BigRational {
      num: BigUint::from_u64(1),
      den: power_denom.mul_u64(divisor),
      negative: n % 2 != 0,
    };
    result = result.add(&term);
    power_denom = power_denom.mul(&k2_big);
  }
  result
}

/// Compute Pi as a BigRational using Machin's formula:
/// Pi/4 = 4*atan(1/5) - atan(1/239)
fn pi_as_big_rational(terms: usize) -> BigRational {
  let atan5 = big_atan_recip(5, terms);
  let atan239 = big_atan_recip(239, terms);
  // Pi = 4 * (4*atan(1/5) - atan(1/239))
  let four_atan5 = atan5.mul_u64(4);
  let diff = four_atan5.sub(&atan239);
  diff.mul_u64(4)
}

/// Compute E as a BigRational using the series: e = sum(1/k!, k=0..terms)
fn e_as_big_rational(terms: usize) -> BigRational {
  let mut result = BigRational::zero();
  let mut factorial = BigUint::from_u64(1);
  for k in 0..terms {
    if k > 0 {
      factorial = factorial.mul_u64(k as u64);
    }
    let term = BigRational {
      num: BigUint::from_u64(1),
      den: factorial.clone(),
      negative: false,
    };
    result = result.add(&term);
  }
  result
}

/// Compute the continued fraction of a BigRational, returning up to n terms.
fn continued_fraction_from_big_rational(
  val: &BigRational,
  n: usize,
) -> Vec<i128> {
  let mut result = Vec::new();
  let mut current = val.clone();
  for _ in 0..n {
    let (floor_val, rem) = current.floor_and_remainder();
    result.push(floor_val);
    if rem.num.is_zero() {
      break;
    }
    current = rem.reciprocal();
  }
  result
}

/// Try to compute a constant expression as a high-precision BigRational.
/// Returns None if the expression is not a recognized constant.
fn try_constant_as_big_rational(
  expr: &Expr,
  n_terms: usize,
) -> Option<BigRational> {
  // Use n_terms + 10 series terms for safety margin
  let series_terms = n_terms + 10;
  match expr {
    Expr::Constant(name) if name == "Pi" => {
      Some(pi_as_big_rational(series_terms))
    }
    Expr::Constant(name) if name == "E" => {
      Some(e_as_big_rational(series_terms))
    }
    Expr::Constant(name) if name == "-Pi" => {
      let mut pi = pi_as_big_rational(series_terms);
      pi.negative = true;
      Some(pi)
    }
    Expr::Constant(name) if name == "-E" => {
      let mut e = e_as_big_rational(series_terms);
      e.negative = true;
      Some(e)
    }
    _ => None,
  }
}

/// ContinuedFraction[x] - exact continued fraction for rational numbers
/// ContinuedFraction[x, n] - first n terms for real numbers
pub fn continued_fraction_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ContinuedFraction expects 1 or 2 arguments".into(),
    ));
  }

  // Handle Rational[p, q] or Integer
  match &args[0] {
    Expr::Integer(n) => {
      return Ok(Expr::List(vec![Expr::Integer(*n)]));
    }
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rargs[0], &rargs[1]) {
        let mut result = Vec::new();
        let mut a = *p;
        let mut b = *q;
        while b != 0 {
          let quotient = if (a < 0) != (b < 0) && a % b != 0 {
            a / b - 1
          } else {
            a / b
          };
          result.push(Expr::Integer(quotient));
          let rem = a - quotient * b;
          a = b;
          b = rem;
        }
        return Ok(Expr::List(result));
      }
    }
    _ => {}
  }

  // For expressions with n terms
  if args.len() == 2 {
    let n = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ContinuedFraction".to_string(),
          args: args.to_vec(),
        });
      }
    };

    // Try high-precision computation for known constants
    if let Some(big_rat) = try_constant_as_big_rational(&args[0], n) {
      let cf = continued_fraction_from_big_rational(&big_rat, n);
      return Ok(Expr::List(cf.into_iter().map(Expr::Integer).collect()));
    }

    // Fall back to f64 for generic real expressions
    if let Some(x) = try_eval_to_f64(&args[0]) {
      let mut result = Vec::new();
      let mut val = x;
      for _ in 0..n {
        let a = val.floor() as i128;
        result.push(Expr::Integer(a));
        let frac = val - a as f64;
        if frac.abs() < 1e-10 {
          break;
        }
        val = 1.0 / frac;
      }
      return Ok(Expr::List(result));
    }
  }

  Ok(Expr::FunctionCall {
    name: "ContinuedFraction".to_string(),
    args: args.to_vec(),
  })
}

/// FromContinuedFraction[{a0, a1, a2, ...}] - reconstruct a number from its continued fraction
pub fn from_continued_fraction_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FromContinuedFraction expects 1 argument".into(),
    ));
  }

  let elements = match &args[0] {
    Expr::List(elems) => elems,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FromContinuedFraction".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if elements.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Collect all integers
  let mut ints: Vec<i128> = Vec::new();
  for elem in elements {
    match elem {
      Expr::Integer(n) => ints.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FromContinuedFraction".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  // Build fraction from right to left: start with last element, then a_i + 1/acc
  // Use numerator/denominator representation to stay exact
  let mut num = *ints.last().unwrap();
  let mut den: i128 = 1;

  for i in (0..ints.len() - 1).rev() {
    // acc = num/den, we want ints[i] + 1/acc = ints[i] + den/num = (ints[i]*num + den)/num
    let new_num = ints[i] * num + den;
    let new_den = num;
    num = new_num;
    den = new_den;
  }

  // Simplify by GCD
  fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a
  }

  let g = gcd(num, den);
  num /= g;
  den /= g;

  // Normalize sign: keep denominator positive
  if den < 0 {
    num = -num;
    den = -den;
  }

  if den == 1 {
    Ok(Expr::Integer(num))
  } else {
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(num), Expr::Integer(den)],
    })
  }
}

/// LucasL[n] - Lucas number L_n
pub fn lucas_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LucasL expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LucasL".to_string(),
        args: args.to_vec(),
      });
    }
  };
  // L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2)
  if n == 0 {
    return Ok(Expr::Integer(2));
  }
  if n == 1 {
    return Ok(Expr::Integer(1));
  }
  let mut a: i128 = 2;
  let mut b: i128 = 1;
  for _ in 2..=n {
    let c = a + b;
    a = b;
    b = c;
  }
  Ok(Expr::Integer(b))
}

/// ChineseRemainder[{r1,r2,...}, {m1,m2,...}] - Chinese Remainder Theorem
pub fn chinese_remainder_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ChineseRemainder expects exactly 2 arguments".into(),
    ));
  }
  let remainders = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChineseRemainder".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let moduli = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ChineseRemainder".to_string(),
        args: args.to_vec(),
      });
    }
  };
  if remainders.len() != moduli.len() {
    return Err(InterpreterError::EvaluationError(
      "ChineseRemainder: lists must have the same length".into(),
    ));
  }

  let mut r_vals = Vec::new();
  let mut m_vals = Vec::new();
  for (r, m) in remainders.iter().zip(moduli.iter()) {
    match (r, m) {
      (Expr::Integer(rv), Expr::Integer(mv)) if *mv > 0 => {
        r_vals.push(*rv);
        m_vals.push(*mv);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "ChineseRemainder".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  // Extended GCD helper
  fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
      (a, 1, 0)
    } else {
      let (g, x1, y1) = extended_gcd(b, a % b);
      (g, y1, x1 - (a / b) * y1)
    }
  }

  // Solve using CRT iteratively
  let mut result = r_vals[0].rem_euclid(m_vals[0]);
  let mut modulus = m_vals[0];

  for i in 1..r_vals.len() {
    let ri = r_vals[i].rem_euclid(m_vals[i]);
    let mi = m_vals[i];
    let (g, p, _) = extended_gcd(modulus, mi);
    if (ri - result) % g != 0 {
      return Err(InterpreterError::EvaluationError(
        "ChineseRemainder: no solution exists".into(),
      ));
    }
    let lcm = modulus / g * mi;
    result =
      (result + modulus * ((ri - result) / g % (mi / g)) * p).rem_euclid(lcm);
    modulus = lcm;
  }

  Ok(Expr::Integer(result))
}

/// DivisorSum[n, form] - applies form to each divisor and sums
pub fn divisor_sum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "DivisorSum expects exactly 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n > 0 => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "DivisorSum".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let func = &args[1];

  // Get divisors
  let mut divs = Vec::new();
  for i in 1..=n {
    if n % i == 0 {
      divs.push(i);
    }
  }

  // Apply function to each divisor and sum
  let mut sum = Expr::Integer(0);
  for d in divs {
    let val = crate::evaluator::apply_function_to_arg(func, &Expr::Integer(d))?;
    sum = crate::functions::math_ast::plus_ast(&[sum, val])?;
  }
  Ok(sum)
}

// ─── Combinatorics Functions ─────────────────────────────────────

/// BernoulliB[n] - nth Bernoulli number
pub fn bernoulli_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BernoulliB expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BernoulliB".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Use the formula: B(n) computed via the explicit sum formula
  // Store as rational numbers (numer, denom)
  // B(0) = 1, B(1) = -1/2, B(odd>1) = 0
  if n == 0 {
    return Ok(Expr::Integer(1));
  }
  if n == 1 {
    return Ok(make_rational(-1, 2));
  }
  if n % 2 != 0 {
    return Ok(Expr::Integer(0));
  }

  // Compute using the recurrence: sum_{k=0}^{n-1} C(n,k) * B(k) / (n - k + 1) = 0 ... wait
  // Better: B(n) = -1/(n+1) * sum_{k=0}^{n-1} C(n+1, k) * B(k)
  // We'll compute all Bernoulli numbers up to n

  // Represent as (numerator, denominator)
  let mut b: Vec<(i128, i128)> = Vec::with_capacity(n + 1);
  b.push((1, 1)); // B(0) = 1
  if n >= 1 {
    b.push((-1, 2)); // B(1) = -1/2
  }

  fn rat_gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
      let t = b;
      b = a % b;
      a = t;
    }
    a
  }

  fn rat_add(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.1 + b.0 * a.1;
    let den = a.1 * b.1;
    let g = rat_gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  fn rat_mul(a: (i128, i128), b: (i128, i128)) -> (i128, i128) {
    let num = a.0 * b.0;
    let den = a.1 * b.1;
    let g = rat_gcd(num, den);
    if den < 0 {
      (-num / g, -den / g)
    } else {
      (num / g, den / g)
    }
  }

  for m in 2..=n {
    if m % 2 != 0 && m > 1 {
      b.push((0, 1));
      continue;
    }
    // B(m) = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B(k)
    let mut sum: (i128, i128) = (0, 1);
    let mut binom: i128 = 1; // C(m+1, k) starting at k=0
    for k in 0..m {
      sum = rat_add(sum, rat_mul((binom, 1), b[k]));
      binom = binom * (m as i128 + 1 - k as i128) / (k as i128 + 1);
    }
    let result = rat_mul((-1, m as i128 + 1), sum);
    b.push(result);
  }

  let (num, den) = b[n];
  Ok(make_rational(num, den))
}

/// CatalanNumber[n] - nth Catalan number = C(2n,n)/(n+1)
pub fn catalan_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CatalanNumber expects exactly 1 argument".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CatalanNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // C(2n, n) / (n + 1)
  let mut result: i128 = 1;
  for i in 0..n {
    result = result * (2 * n - i) / (i + 1);
  }
  result /= n + 1;
  Ok(Expr::Integer(result))
}

/// StirlingS1[n, k] - Stirling number of the first kind (signed)
pub fn stirling_s1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS1 expects exactly 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS1".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 => *k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS1".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if k > n {
    return Ok(Expr::Integer(0));
  }
  if n == 0 && k == 0 {
    return Ok(Expr::Integer(1));
  }
  if k == 0 {
    return Ok(Expr::Integer(0));
  }

  // s(n,k) = s(n-1,k-1) - (n-1)*s(n-1,k) (signed Stirling S1)
  // Use DP table
  let mut table = vec![vec![0i128; k + 1]; n + 1];
  table[0][0] = 1;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] = table[i - 1][j - 1] - (i as i128 - 1) * table[i - 1][j];
    }
  }
  Ok(Expr::Integer(table[n][k]))
}

/// StirlingS2[n, k] - Stirling number of the second kind
pub fn stirling_s2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS2 expects exactly 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS2".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match &args[1] {
    Expr::Integer(k) if *k >= 0 => *k as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS2".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if k > n {
    return Ok(Expr::Integer(0));
  }
  if n == 0 && k == 0 {
    return Ok(Expr::Integer(1));
  }
  if k == 0 {
    return Ok(Expr::Integer(0));
  }

  // S(n,k) = k*S(n-1,k) + S(n-1,k-1)
  let mut table = vec![vec![0i128; k + 1]; n + 1];
  table[0][0] = 1;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] = j as i128 * table[i - 1][j] + table[i - 1][j - 1];
    }
  }
  Ok(Expr::Integer(table[n][k]))
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

  // Handle string argument: FromDigits["1234"] or FromDigits["1a", 16]
  if let Expr::String(s) = &args[0] {
    let mut result: i128 = 0;
    for ch in s.chars() {
      let d = if ch.is_ascii_digit() {
        (ch as i128) - ('0' as i128)
      } else if ch.is_ascii_lowercase() {
        (ch as i128) - ('a' as i128) + 10
      } else if ch.is_ascii_uppercase() {
        (ch as i128) - ('A' as i128) + 10
      } else {
        return Ok(Expr::FunctionCall {
          name: "FromDigits".to_string(),
          args: args.to_vec(),
        });
      };
      if d >= base {
        return Err(InterpreterError::EvaluationError(format!(
          "FromDigits: invalid digit {} for base {}",
          ch, base
        )));
      }
      result = result * base + d;
    }
    return Ok(Expr::Integer(result));
  }

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
// ─── IntegerPartitions ─────────────────────────────────────────────
// IntegerPartitions[n] — all partitions of n
// IntegerPartitions[n, k] — partitions with at most k parts
// IntegerPartitions[n, {k}] — partitions with exactly k parts
// IntegerPartitions[n, {kmin, kmax}] — partitions with kmin..kmax parts
// IntegerPartitions[n, kspec, {d1, d2, ...}] — using only given elements
// kspec can be All (equivalent to no constraint)
pub fn integer_partitions_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Parse n
  let n = match &args[0] {
    Expr::Integer(v) => *v,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerPartitions".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n < 0 {
    return Ok(Expr::List(vec![]));
  }
  let n = n as u64;

  // Parse length constraints from second arg
  let (min_len, max_len) = if args.len() >= 2 {
    match &args[1] {
      // IntegerPartitions[n, k] — at most k parts
      Expr::Integer(k) if *k >= 0 => (1, *k as u64),
      // IntegerPartitions[n, All] — no constraint
      Expr::Identifier(s) if s == "All" => (1, n.max(1)),
      // IntegerPartitions[n, {k}] — exactly k parts
      Expr::List(lst) if lst.len() == 1 => match &lst[0] {
        Expr::Integer(k) if *k >= 0 => (*k as u64, *k as u64),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "IntegerPartitions".to_string(),
            args: args.to_vec(),
          });
        }
      },
      // IntegerPartitions[n, {kmin, kmax}] — range of parts
      Expr::List(lst) if lst.len() == 2 => match (&lst[0], &lst[1]) {
        (Expr::Integer(lo), Expr::Integer(hi)) if *lo >= 0 && *hi >= 0 => {
          (*lo as u64, *hi as u64)
        }
        _ => {
          return Ok(Expr::FunctionCall {
            name: "IntegerPartitions".to_string(),
            args: args.to_vec(),
          });
        }
      },
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerPartitions".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (1, n.max(1))
  };

  // Parse allowed elements from third arg
  let allowed: Option<Vec<u64>> = if args.len() == 3 {
    match &args[2] {
      Expr::List(elems) => {
        let mut vals = Vec::new();
        for e in elems {
          match e {
            Expr::Integer(v) if *v > 0 => vals.push(*v as u64),
            _ => {
              return Ok(Expr::FunctionCall {
                name: "IntegerPartitions".to_string(),
                args: args.to_vec(),
              });
            }
          }
        }
        vals.sort_unstable();
        vals.dedup();
        vals.reverse(); // descending for generation
        Some(vals)
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerPartitions".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    None
  };

  // Special case: n == 0
  // The only partition of 0 is the empty partition {}, which has 0 parts
  if n == 0 {
    if min_len == 0 || (args.len() < 2) {
      return Ok(Expr::List(vec![Expr::List(vec![])]));
    } else {
      return Ok(Expr::List(vec![]));
    }
  }

  let mut result = Vec::new();
  let mut current = Vec::new();

  match &allowed {
    Some(elems) => {
      generate_partitions_restricted(
        n,
        max_len,
        min_len,
        elems,
        0,
        &mut current,
        &mut result,
      );
    }
    None => {
      generate_partitions(n, n, max_len, min_len, &mut current, &mut result);
    }
  }

  Ok(Expr::List(
    result
      .into_iter()
      .map(|p| {
        Expr::List(p.into_iter().map(|v| Expr::Integer(v as i128)).collect())
      })
      .collect(),
  ))
}

/// Generate all partitions of `remaining` where each part <= `max_part`,
/// with total number of parts between `min_len` and `max_len`.
fn generate_partitions(
  remaining: u64,
  max_part: u64,
  max_len: u64,
  min_len: u64,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
) {
  if remaining == 0 {
    if current.len() as u64 >= min_len {
      result.push(current.clone());
    }
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  let upper = remaining.min(max_part);
  for part in (1..=upper).rev() {
    current.push(part);
    generate_partitions(
      remaining - part,
      part,
      max_len,
      min_len,
      current,
      result,
    );
    current.pop();
  }
}

/// Generate partitions using only elements from `elems` (sorted descending).
fn generate_partitions_restricted(
  remaining: u64,
  max_len: u64,
  min_len: u64,
  elems: &[u64],
  start_idx: usize,
  current: &mut Vec<u64>,
  result: &mut Vec<Vec<u64>>,
) {
  if remaining == 0 {
    if current.len() as u64 >= min_len {
      result.push(current.clone());
    }
    return;
  }
  if current.len() as u64 >= max_len {
    return;
  }
  for i in start_idx..elems.len() {
    let part = elems[i];
    if part > remaining {
      continue;
    }
    current.push(part);
    generate_partitions_restricted(
      remaining - part,
      max_len,
      min_len,
      elems,
      i,
      current,
      result,
    );
    current.pop();
  }
}

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

// ─── Numerator / Denominator ───────────────────────────────────────

/// Numerator[x] - Returns the numerator of a rational expression
pub fn numerator_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Numerator expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    // Numerator[Rational[a, b]] → a
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      Ok(rargs[0].clone())
    }
    // Numerator[integer] → integer
    Expr::Integer(_) => Ok(args[0].clone()),
    // Numerator[real] → real
    Expr::Real(_) => Ok(args[0].clone()),
    // Numerator[a / b] (BinaryOp Divide) → a
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      ..
    } => Ok(left.as_ref().clone()),
    // Numerator[Times[-1, x]] where x is Power[y, -1] → -1 (i.e. -1/y)
    // Numerator[Times[a, Power[b, -1]]] → a
    Expr::FunctionCall { name, args: targs }
      if name == "Times" && targs.len() == 2 =>
    {
      // Check if second factor is Power[_, -1] (denominator form)
      if is_reciprocal(&targs[1]) {
        return Ok(targs[0].clone());
      }
      if is_reciprocal(&targs[0]) {
        return Ok(targs[1].clone());
      }
      // Not a fraction form
      Ok(args[0].clone())
    }
    _ => Ok(args[0].clone()),
  }
}

/// Denominator[x] - Returns the denominator of a rational expression
pub fn denominator_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Denominator expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    // Denominator[Rational[a, b]] → b
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      Ok(rargs[1].clone())
    }
    // Denominator[integer] → 1
    Expr::Integer(_) => Ok(Expr::Integer(1)),
    // Denominator[real] → 1
    Expr::Real(_) => Ok(Expr::Integer(1)),
    // Denominator[a / b] → b
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      right,
      ..
    } => Ok(right.as_ref().clone()),
    // Denominator[Times[a, Power[b, -1]]] → b
    Expr::FunctionCall { name, args: targs }
      if name == "Times" && targs.len() == 2 =>
    {
      if let Some(base) = get_reciprocal_base(&targs[1]) {
        return Ok(base);
      }
      if let Some(base) = get_reciprocal_base(&targs[0]) {
        return Ok(base);
      }
      Ok(Expr::Integer(1))
    }
    _ => Ok(Expr::Integer(1)),
  }
}

/// Check if an expression is Power[x, -1]
fn is_reciprocal(expr: &Expr) -> bool {
  get_reciprocal_base(expr).is_some()
}

/// If expr is Power[base, -1], return Some(base), else None
fn get_reciprocal_base(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      if let Expr::Integer(-1) = right.as_ref() {
        Some(left.as_ref().clone())
      } else {
        match right.as_ref() {
          Expr::UnaryOp {
            op: crate::syntax::UnaryOperator::Minus,
            operand,
          } if matches!(operand.as_ref(), Expr::Integer(1)) => {
            Some(left.as_ref().clone())
          }
          _ => None,
        }
      }
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      if let Expr::Integer(-1) = &args[1] {
        Some(args[0].clone())
      } else {
        None
      }
    }
    _ => None,
  }
}

// ─── Binomial ──────────────────────────────────────────────────────

/// Binomial[n, k] - Binomial coefficient
pub fn binomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Binomial expects exactly 2 arguments".into(),
    ));
  }
  match (&args[0], &args[1]) {
    (Expr::Integer(n), Expr::Integer(k)) => {
      Ok(Expr::Integer(binomial_coeff(*n, *k)))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Binomial".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute binomial coefficient for arbitrary integers (generalized)
fn binomial_coeff(n: i128, k: i128) -> i128 {
  if k < 0 {
    return 0;
  }
  if k == 0 {
    return 1;
  }
  if n >= 0 {
    if k > n {
      return 0;
    }
    // Use the smaller of k and n-k for efficiency
    let k = k.min(n - k);
    let mut result: i128 = 1;
    for i in 0..k {
      result = result * (n - i) / (i + 1);
    }
    result
  } else {
    // Generalized: Binomial[-n, k] = (-1)^k * Binomial[n+k-1, k]
    let sign = if k % 2 == 0 { 1 } else { -1 };
    sign * binomial_coeff(-n + k - 1, k)
  }
}

// ─── Multinomial ───────────────────────────────────────────────────

/// Multinomial[n1, n2, ...] - Multinomial coefficient (n1+n2+...)! / (n1! * n2! * ...)
pub fn multinomial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }
  let mut ints = Vec::new();
  for arg in args {
    match arg {
      Expr::Integer(n) => {
        if *n < 0 {
          return Err(InterpreterError::EvaluationError(
            "Multinomial: arguments must be non-negative integers".into(),
          ));
        }
        ints.push(*n);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Multinomial".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  // Compute using iterated binomial coefficients: Multinomial[a,b,c] = C(a+b+c, a) * C(b+c, b)
  let mut total: i128 = 0;
  let mut result: i128 = 1;
  for &ni in &ints {
    total += ni;
    result *= binomial_coeff(total, ni);
  }
  Ok(Expr::Integer(result))
}

// ─── PowerMod ──────────────────────────────────────────────────────

/// PowerMod[a, b, m] - Modular exponentiation: a^b mod m
pub fn power_mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PowerMod expects exactly 3 arguments".into(),
    ));
  }
  match (&args[0], &args[1], &args[2]) {
    (Expr::Integer(base), Expr::Integer(exp), Expr::Integer(modulus)) => {
      if *modulus == 0 {
        return Err(InterpreterError::EvaluationError(
          "PowerMod: modulus cannot be zero".into(),
        ));
      }
      let m = modulus.unsigned_abs();
      if *exp < 0 {
        // Negative exponent: compute modular inverse first
        // a^(-e) mod m = (a^(-1))^e mod m
        if let Some(inv) = mod_inverse(*base, *modulus) {
          let result = mod_pow_unsigned(inv as u128, (-*exp) as u128, m);
          Ok(Expr::Integer(result as i128))
        } else {
          Err(InterpreterError::EvaluationError(
            "PowerMod: modular inverse does not exist".into(),
          ))
        }
      } else {
        // Normalize base to be non-negative mod m
        let b = base.rem_euclid(*modulus);
        let result = mod_pow_unsigned(b as u128, *exp as u128, m);
        Ok(Expr::Integer(result as i128))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "PowerMod".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Binary exponentiation: base^exp mod modulus (all unsigned)
/// Uses BigUint for intermediate multiplication to avoid u128 overflow.
fn mod_pow_unsigned(base: u128, mut exp: u128, modulus: u128) -> u128 {
  use num_bigint::BigUint;
  use num_traits::ToPrimitive;

  if modulus == 1 {
    return 0;
  }
  let m = BigUint::from(modulus);
  let mut result = BigUint::from(1u32);
  let mut b = BigUint::from(base % modulus);
  while exp > 0 {
    if exp % 2 == 1 {
      result = result * &b % &m;
    }
    exp >>= 1;
    b = &b * &b % &m;
  }
  result.to_u128().unwrap_or(0)
}

/// Extended Euclidean algorithm for modular inverse
fn mod_inverse(a: i128, m: i128) -> Option<i128> {
  let m_abs = m.abs();
  let a = ((a % m_abs) + m_abs) % m_abs;
  let (mut old_r, mut r) = (a, m_abs);
  let (mut old_s, mut s) = (1i128, 0i128);
  while r != 0 {
    let q = old_r / r;
    let temp_r = r;
    r = old_r - q * r;
    old_r = temp_r;
    let temp_s = s;
    s = old_s - q * s;
    old_s = temp_s;
  }
  if old_r != 1 {
    None // No inverse exists
  } else {
    Some(((old_s % m_abs) + m_abs) % m_abs)
  }
}

// ─── PrimePi ───────────────────────────────────────────────────────

/// PrimePi[n] - Counts the number of primes <= n
pub fn prime_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PrimePi expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      if *n < 2 {
        return Ok(Expr::Integer(0));
      }
      let n_usize = *n as usize;
      let mut count: i128 = 0;
      for i in 2..=n_usize {
        if crate::is_prime(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    Expr::Real(f) => {
      if *f < 2.0 {
        return Ok(Expr::Integer(0));
      }
      let n = f.floor() as usize;
      let mut count: i128 = 0;
      for i in 2..=n {
        if crate::is_prime(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    _ => Ok(Expr::FunctionCall {
      name: "PrimePi".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── BigInt Primality (Miller-Rabin) ──────────────────────────────

/// Miller-Rabin primality test for BigInt values.
/// Uses deterministic witnesses for small numbers and a set of strong
/// witnesses that provides correct results for all numbers < 3.317e24,
/// plus additional witnesses for larger numbers.
pub fn is_prime_bigint(n: &num_bigint::BigInt) -> bool {
  use num_bigint::BigInt;
  use num_traits::{One, Zero};

  let one = BigInt::one();
  let two = &one + &one;
  let three = &two + &one;

  if *n <= one {
    return false;
  }
  if *n == two || *n == three {
    return true;
  }
  if (n % &two).is_zero() || (n % &three).is_zero() {
    return false;
  }

  // Small primes trial division
  let small_primes: &[u64] = &[
    5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
    79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
    241, 251,
  ];
  for &p in small_primes {
    let bp = BigInt::from(p);
    if *n == bp {
      return true;
    }
    if (n % &bp).is_zero() {
      return false;
    }
  }

  // Write n-1 = d * 2^r
  let n_minus_1 = n - &one;
  let mut d = n_minus_1.clone();
  let mut r: u64 = 0;
  while (&d % &two).is_zero() {
    d /= &two;
    r += 1;
  }

  // Witness bases — deterministic for numbers < 3.317e24,
  // and probabilistically correct (error < 2^-128) for larger numbers
  let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

  'witness: for &a in witnesses {
    let a_big = BigInt::from(a);
    if a_big >= *n {
      continue;
    }
    let mut x = a_big.modpow(&d, n);
    if x == one || x == n_minus_1 {
      continue;
    }
    for _ in 0..r - 1 {
      x = x.modpow(&two, n);
      if x == n_minus_1 {
        continue 'witness;
      }
    }
    return false;
  }
  true
}

// ─── NextPrime ─────────────────────────────────────────────────────

/// NextPrime[n] - Returns the smallest prime greater than n
/// Negative primes (-2, -3, -5, ...) are included in the search space.
pub fn next_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "NextPrime expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer(next_prime_after(*n))),
    Expr::Real(f) => Ok(Expr::Integer(next_prime_after(f.floor() as i128))),
    Expr::BigInteger(n) => Ok(bigint_to_expr(next_prime_after_bigint(n))),
    _ => Ok(Expr::FunctionCall {
      name: "NextPrime".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Find the smallest prime > n (including negative primes like -2, -3, -5, ...).
fn next_prime_after(n: i128) -> i128 {
  // For n >= 1: search upward from n+1
  if n >= 1 {
    let mut candidate = n + 1;
    while !crate::is_prime(candidate as usize) {
      candidate += 1;
    }
    return candidate;
  }
  // For n < -2: check negative primes between n and -2 (exclusive of n, inclusive of -2)
  // A negative prime is -p where p is a positive prime.
  // Search from n+1 upward: for each candidate c, check if |c| is prime.
  if n < -2 {
    for c in (n + 1)..=-2 {
      if c < 0 && crate::is_prime((-c) as usize) {
        return c;
      }
    }
  }
  // No negative prime found > n, or n is -2, -1, or 0: smallest positive prime is 2
  2
}

/// Find the smallest prime > n for BigInt values.
fn next_prime_after_bigint(n: &num_bigint::BigInt) -> num_bigint::BigInt {
  use num_bigint::BigInt;
  use num_traits::{One, Zero};

  let one = BigInt::one();
  let two = &one + &one;

  // For positive n, search upward
  if *n >= one {
    let mut candidate = n + &one;
    // Ensure candidate is odd
    if (&candidate % &two).is_zero() {
      candidate += &one;
    }
    // If candidate is 2, check it
    if candidate == two {
      return two;
    }
    loop {
      if is_prime_bigint(&candidate) {
        return candidate;
      }
      candidate += &two;
    }
  }

  // For negative or zero n, delegate to i128 path for small values
  // since BigInt negative values that reach here would be small enough
  let zero = BigInt::zero();
  if *n <= zero {
    return BigInt::from(2);
  }

  unreachable!()
}

// ─── BitLength ─────────────────────────────────────────────────────

/// BitLength[n] - Number of bits in the binary representation
/// For n >= 0: Floor[Log2[n]] + 1, with BitLength[0] = 0
/// For n < 0: BitLength[-n - 1] (2's complement convention)
pub fn bit_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BitLength expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      let val = if *n < 0 { (-*n) - 1 } else { *n };
      if val == 0 {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::Integer(128 - val.leading_zeros() as i128))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "BitLength".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── IntegerExponent ──────────────────────────────────────────────

/// IntegerExponent[n, b] - largest power of b that divides n
/// IntegerExponent[n] - largest power of 2 that divides n
pub fn integer_exponent_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerExponent expects 1 or 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerExponent".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) => *b,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerExponent".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    2 // default base
  };

  if n == 0 {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  if base <= 1 {
    return Ok(Expr::FunctionCall {
      name: "IntegerExponent".to_string(),
      args: args.to_vec(),
    });
  }

  let mut count = 0i128;
  let mut val = n.abs();
  while val > 0 && val % base == 0 {
    count += 1;
    val /= base;
  }
  Ok(Expr::Integer(count))
}

// ─── IntegerPart / FractionalPart ──────────────────────────────────

/// IntegerPart[x] - Integer part (truncation towards zero)
pub fn integer_part_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "IntegerPart expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer(*n)),
    Expr::Real(f) => Ok(Expr::Integer(f.trunc() as i128)),
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *d != 0
      {
        // Truncate towards zero
        return Ok(Expr::Integer(n / d));
      }
      Ok(Expr::FunctionCall {
        name: "IntegerPart".to_string(),
        args: args.to_vec(),
      })
    }
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        Ok(Expr::Integer(f.trunc() as i128))
      } else {
        Ok(Expr::FunctionCall {
          name: "IntegerPart".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

/// FractionalPart[x] - Fractional part: x - IntegerPart[x]
pub fn fractional_part_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FractionalPart expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) => Ok(Expr::Integer(0)),
    Expr::Real(f) => {
      let frac = *f - f.trunc();
      if frac == 0.0 {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::Real(frac))
      }
    }
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *d != 0
      {
        let int_part = n / d;
        let frac_n = n - int_part * d;
        if frac_n == 0 {
          return Ok(Expr::Integer(0));
        }
        return Ok(make_rational(frac_n, *d));
      }
      Ok(Expr::FunctionCall {
        name: "FractionalPart".to_string(),
        args: args.to_vec(),
      })
    }
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        let frac = f - f.trunc();
        if frac == 0.0 {
          Ok(Expr::Integer(0))
        } else {
          Ok(Expr::Real(frac))
        }
      } else {
        Ok(Expr::FunctionCall {
          name: "FractionalPart".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

// ─── Chop ──────────────────────────────────────────────────────────

/// Chop[x] or Chop[x, delta] - Replaces approximate real numbers close to zero by exact 0
pub fn chop_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Chop expects 1 or 2 arguments".into(),
    ));
  }
  let tolerance = if args.len() == 2 {
    match try_eval_to_f64(&args[1]) {
      Some(t) => t,
      None => {
        return Ok(Expr::FunctionCall {
          name: "Chop".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1e-10 // Default tolerance
  };

  chop_expr(&args[0], tolerance)
}

fn chop_expr(expr: &Expr, tolerance: f64) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Real(f) => {
      if f.abs() < tolerance {
        Ok(Expr::Integer(0))
      } else {
        Ok(expr.clone())
      }
    }
    Expr::List(items) => {
      let chopped: Result<Vec<Expr>, _> =
        items.iter().map(|e| chop_expr(e, tolerance)).collect();
      Ok(Expr::List(chopped?))
    }
    _ => Ok(expr.clone()),
  }
}

// ─── CubeRoot ──────────────────────────────────────────────────────

/// CubeRoot[x] - Real-valued cube root
pub fn cube_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CubeRoot expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      // Check for perfect cubes
      let sign = n.signum();
      let abs_n = n.unsigned_abs();
      let root = (abs_n as f64).cbrt().round() as u128;
      if root * root * root == abs_n {
        return Ok(Expr::Integer(sign * root as i128));
      }
      // Not a perfect cube — return n^(1/3)
      Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(args[0].clone()),
        right: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(3)],
        }),
      })
    }
    Expr::Real(f) => Ok(Expr::Real(f.signum() * f.abs().cbrt())),
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        Ok(Expr::Real(f.signum() * f.abs().cbrt()))
      } else {
        Ok(Expr::FunctionCall {
          name: "CubeRoot".to_string(),
          args: args.to_vec(),
        })
      }
    }
  }
}

// ─── Subdivide ─────────────────────────────────────────────────────

/// Subdivide[n] - subdivide [0,1] into n equal parts
/// Subdivide[xmax, n] - subdivide [0, xmax] into n equal parts
/// Subdivide[xmin, xmax, n] - subdivide [xmin, xmax] into n equal parts
pub fn subdivide_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Subdivide expects 1 to 3 arguments".into(),
    ));
  }
  let (xmin, xmax, n_val) = match args.len() {
    1 => {
      // Subdivide[n]
      match &args[0] {
        Expr::Integer(n) => (0i128, 1i128, *n),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Subdivide".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    2 => {
      // Subdivide[xmax, n]
      match (&args[0], &args[1]) {
        (Expr::Integer(xmax), Expr::Integer(n)) => (0, *xmax, *n),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "Subdivide".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    3 => {
      // Subdivide[xmin, xmax, n]
      match (&args[0], &args[1], &args[2]) {
        (Expr::Integer(xmin), Expr::Integer(xmax), Expr::Integer(n)) => {
          (*xmin, *xmax, *n)
        }
        _ => {
          // Try float version
          if let (Some(xmin_f), Some(xmax_f), Some(n_f)) = (
            try_eval_to_f64(&args[0]),
            try_eval_to_f64(&args[1]),
            try_eval_to_f64(&args[2]),
          ) {
            let n = n_f as i128;
            if n < 0 {
              return Err(InterpreterError::EvaluationError(
                "Subdivide: n must be non-negative".into(),
              ));
            }
            let mut items = Vec::with_capacity(n as usize + 1);
            for i in 0..=n {
              let t = i as f64 / n as f64;
              let val = xmin_f + t * (xmax_f - xmin_f);
              items.push(num_to_expr(val));
            }
            return Ok(Expr::List(items));
          }
          return Ok(Expr::FunctionCall {
            name: "Subdivide".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
    _ => unreachable!(),
  };
  if n_val < 0 {
    return Err(InterpreterError::EvaluationError(
      "Subdivide: n must be non-negative".into(),
    ));
  }
  if n_val == 0 {
    return Ok(Expr::List(vec![Expr::Integer(xmin)]));
  }
  let mut items = Vec::with_capacity(n_val as usize + 1);
  let range = xmax - xmin;
  for i in 0..=n_val {
    // Compute xmin + i * range / n as exact rational
    let numer = xmin * n_val + i * range;
    items.push(make_rational(numer, n_val));
  }
  Ok(Expr::List(items))
}

/// Variance[list] - Sample variance (unbiased, divides by n-1)
/// Variance[{1, 2, 3}] => 1
/// Variance[{1.0, 2.0, 3.0}] => 1.0
pub fn variance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Variance expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.len() < 2 {
        return Err(InterpreterError::EvaluationError(
          "Variance: need at least 2 elements".into(),
        ));
      }
      // Try all-integer exact path
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => int_vals.push(*n),
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        // Exact: Variance = Sum[(xi - mean)^2] / (n-1)
        // = (n * Sum[xi^2] - (Sum[xi])^2) / (n * (n-1))
        let n = int_vals.len() as i128;
        let sum: i128 = int_vals.iter().sum();
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        let numer = n * sum_sq - sum * sum;
        let denom = n * (n - 1);
        return Ok(make_rational(numer, denom));
      }
      if has_real || !all_int {
        // Float path
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "Variance".to_string(),
              args: args.to_vec(),
            });
          }
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let var =
          vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        return Ok(num_to_expr(var));
      }
      Ok(Expr::FunctionCall {
        name: "Variance".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Variance".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// StandardDeviation[list] - Sample standard deviation (Sqrt of Variance)
/// StandardDeviation[{1, 2, 3}] => 1
/// StandardDeviation[{1.0, 2.0, 3.0}] => 1.0
pub fn standard_deviation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "StandardDeviation expects exactly 1 argument".into(),
    ));
  }
  let var = variance_ast(args)?;
  match &var {
    Expr::Integer(n) => {
      // StandardDeviation is Sqrt[Variance]
      let v = *n;
      if v == 0 {
        return Ok(Expr::Integer(0));
      }
      if v == 1 {
        return Ok(Expr::Integer(1));
      }
      // Check if it's a perfect square
      let root = (v as f64).sqrt() as i128;
      for candidate in [root - 1, root, root + 1] {
        if candidate >= 0 && candidate * candidate == v {
          return Ok(Expr::Integer(candidate));
        }
      }
      // Return Sqrt[n] symbolically
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![Expr::Integer(v)],
      })
    }
    Expr::Real(f) => Ok(num_to_expr(f.sqrt())),
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      // Sqrt[p/q] - try to simplify
      if let (Expr::Integer(p), Expr::Integer(q)) = (&fargs[0], &fargs[1]) {
        let fval = (*p as f64) / (*q as f64);
        if fval >= 0.0 {
          // Check if both are perfect squares
          let p_root = (*p as f64).sqrt() as i128;
          let q_root = (*q as f64).sqrt() as i128;
          if p_root * p_root == *p && q_root * q_root == *q {
            return Ok(make_rational(p_root, q_root));
          }
        }
        // Return Sqrt[Rational[p, q]] symbolically
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![var.clone()],
        });
      }
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![var],
      })
    }
    // If variance returned symbolic, wrap in Sqrt
    _ => Ok(Expr::FunctionCall {
      name: "StandardDeviation".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// GeometricMean[list] - Geometric mean: (product of elements)^(1/n)
/// GeometricMean[{2, 8}] => 4
/// GeometricMean[{1.0, 2.0, 3.0}] => 1.8171205928321397
pub fn geometric_mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "GeometricMean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "GeometricMean: empty list".into(),
        ));
      }
      // Float path
      let mut vals = Vec::new();
      for item in items {
        if let Some(v) = try_eval_to_f64(item) {
          vals.push(v);
        } else {
          return Ok(Expr::FunctionCall {
            name: "GeometricMean".to_string(),
            args: args.to_vec(),
          });
        }
      }
      let n = vals.len() as f64;
      let product: f64 = vals.iter().product();
      let result = product.powf(1.0 / n);
      // Check if result is an integer
      Ok(num_to_expr(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "GeometricMean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// HarmonicMean[list] - Harmonic mean: n / Sum[1/xi]
/// HarmonicMean[{1, 2, 3}] => 18/11
/// HarmonicMean[{1.0, 2.0, 3.0}] => 1.6363636363636365
pub fn harmonic_mean_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "HarmonicMean expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "HarmonicMean: empty list".into(),
        ));
      }
      // Try all-integer exact path using rational arithmetic
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => {
            if *n == 0 {
              return Err(InterpreterError::EvaluationError(
                "HarmonicMean: division by zero".into(),
              ));
            }
            int_vals.push(*n);
          }
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        // HarmonicMean = n / Sum[1/xi]
        // = n / (Sum[product/xi] / product)
        // = n * product / Sum[product/xi]
        // Use rational sum: Sum[1/xi] = numer/denom
        let n = int_vals.len() as i128;
        let mut sum_numer: i128 = 0;
        let mut sum_denom: i128 = 1;
        for &x in &int_vals {
          // Add 1/x to sum_numer/sum_denom
          // a/b + 1/x = (a*x + b) / (b*x)
          sum_numer = sum_numer * x + sum_denom;
          sum_denom *= x;
          // Simplify to avoid overflow
          let g = gcd(sum_numer.abs(), sum_denom.abs());
          sum_numer /= g;
          sum_denom /= g;
        }
        // HarmonicMean = n / (sum_numer/sum_denom) = n * sum_denom / sum_numer
        let result_numer = n * sum_denom;
        let result_denom = sum_numer;
        return Ok(make_rational(result_numer, result_denom));
      }
      if has_real || !all_int {
        // Float path
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            if v == 0.0 {
              return Err(InterpreterError::EvaluationError(
                "HarmonicMean: division by zero".into(),
              ));
            }
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "HarmonicMean".to_string(),
              args: args.to_vec(),
            });
          }
        }
        let n = vals.len() as f64;
        let sum_recip: f64 = vals.iter().map(|x| 1.0 / x).sum();
        return Ok(num_to_expr(n / sum_recip));
      }
      Ok(Expr::FunctionCall {
        name: "HarmonicMean".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HarmonicMean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// RootMeanSquare[list] - Sqrt[Mean[list^2]]
/// RootMeanSquare[{1, 2, 3}] => Sqrt[14/3]
/// RootMeanSquare[{1.0, 2.0, 3.0}] => 2.160246899469287
pub fn root_mean_square_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RootMeanSquare expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Err(InterpreterError::EvaluationError(
          "RootMeanSquare: empty list".into(),
        ));
      }
      // Try all-integer exact path
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      let mut has_real = false;
      for item in items {
        match item {
          Expr::Integer(n) => int_vals.push(*n),
          Expr::Real(_) => {
            all_int = false;
            has_real = true;
            break;
          }
          _ => {
            all_int = false;
            break;
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        let n = int_vals.len() as i128;
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        // RMS = Sqrt[sum_sq / n]
        let g = gcd(sum_sq.abs(), n);
        let numer = sum_sq / g;
        let denom = n / g;
        // Check if numer/denom is a perfect square
        if denom == 1 {
          let root = (numer as f64).sqrt() as i128;
          if root * root == numer {
            return Ok(Expr::Integer(root));
          }
          return Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(numer)],
          });
        }
        // Return Sqrt[Rational[numer, denom]]
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![make_rational(numer, denom)],
        });
      }
      if has_real || !all_int {
        let mut vals = Vec::new();
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            return Ok(Expr::FunctionCall {
              name: "RootMeanSquare".to_string(),
              args: args.to_vec(),
            });
          }
        }
        let n = vals.len() as f64;
        let mean_sq = vals.iter().map(|x| x * x).sum::<f64>() / n;
        return Ok(num_to_expr(mean_sq.sqrt()));
      }
      Ok(Expr::FunctionCall {
        name: "RootMeanSquare".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "RootMeanSquare".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// IntegerLength[n] - Number of digits of n in base 10
/// IntegerLength[n, b] - Number of digits in base b
/// IntegerLength[12345] => 5
/// IntegerLength[255, 16] => 2
/// IntegerLength[0] => 0
pub fn integer_length_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerLength expects 1 or 2 arguments".into(),
    ));
  }
  let n = match &args[0] {
    Expr::Integer(n) => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerLength".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) => {
        if *b < 2 {
          return Err(InterpreterError::EvaluationError(
            "IntegerLength: base must be at least 2".into(),
          ));
        }
        *b
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerLength".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    10
  };

  if n == 0 {
    return Ok(Expr::Integer(0));
  }
  let mut abs_n = n.abs();
  let mut count = 0i128;
  while abs_n > 0 {
    abs_n /= base;
    count += 1;
  }
  Ok(Expr::Integer(count))
}

/// Rescale[x, {xmin, xmax}] - rescales x to [0,1]
/// Rescale[x, {xmin, xmax}, {ymin, ymax}] - rescales x to [ymin,ymax]
/// Rescale[list] - rescales list elements to [0,1] based on min/max
pub fn rescale_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Rescale expects 1 to 3 arguments".into(),
    ));
  }

  // Rescale[list] - auto-detect min/max
  if args.len() == 1 {
    if let Expr::List(items) = &args[0] {
      if items.is_empty() {
        return Ok(Expr::List(vec![]));
      }
      // Find min and max
      let mut vals = Vec::new();
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      for item in items {
        match item {
          Expr::Integer(n) => {
            vals.push(*n as f64);
            int_vals.push(*n);
          }
          Expr::Real(f) => {
            vals.push(*f);
            all_int = false;
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Rescale".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      if all_int && !int_vals.is_empty() {
        let min_val = *int_vals.iter().min().unwrap();
        let max_val = *int_vals.iter().max().unwrap();
        if min_val == max_val {
          return Ok(Expr::List(vec![Expr::Integer(0); items.len()]));
        }
        let range = max_val - min_val;
        let result: Vec<Expr> = int_vals
          .iter()
          .map(|x| make_rational(x - min_val, range))
          .collect();
        return Ok(Expr::List(result));
      }
      let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
      let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      if (max_val - min_val).abs() < f64::EPSILON {
        return Ok(Expr::List(vec![Expr::Integer(0); items.len()]));
      }
      let result: Vec<Expr> = vals
        .iter()
        .map(|x| num_to_expr((x - min_val) / (max_val - min_val)))
        .collect();
      return Ok(Expr::List(result));
    }
    // Single non-list value needs {xmin, xmax}
    return Ok(Expr::FunctionCall {
      name: "Rescale".to_string(),
      args: args.to_vec(),
    });
  }

  // Rescale[x, {xmin, xmax}] or Rescale[x, {xmin, xmax}, {ymin, ymax}]
  let range = match &args[1] {
    Expr::List(r) if r.len() == 2 => r,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Rescale".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let (ymin, ymax) = if args.len() == 3 {
    match &args[2] {
      Expr::List(r) if r.len() == 2 => (&r[0], &r[1]),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Rescale".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    (&Expr::Integer(0) as &Expr, &Expr::Integer(1) as &Expr)
  };

  // Try integer exact path
  if let (
    Expr::Integer(x),
    Expr::Integer(xmin),
    Expr::Integer(xmax),
    Expr::Integer(yn),
    Expr::Integer(yx),
  ) = (&args[0], &range[0], &range[1], ymin, ymax)
  {
    if xmax == xmin {
      return Err(InterpreterError::EvaluationError(
        "Rescale: xmin and xmax must be different".into(),
      ));
    }
    // result = ymin + (x - xmin) * (ymax - ymin) / (xmax - xmin)
    let numer = yn * (xmax - xmin) + (x - xmin) * (yx - yn);
    let denom = xmax - xmin;
    return Ok(make_rational(numer, denom));
  }

  // Float path
  if let (Some(x), Some(xmin), Some(xmax)) = (
    try_eval_to_f64(&args[0]),
    try_eval_to_f64(&range[0]),
    try_eval_to_f64(&range[1]),
  ) {
    if (xmax - xmin).abs() < f64::EPSILON {
      return Err(InterpreterError::EvaluationError(
        "Rescale: xmin and xmax must be different".into(),
      ));
    }
    let t = (x - xmin) / (xmax - xmin);
    if args.len() == 3
      && let (Some(yn), Some(yx)) =
        (try_eval_to_f64(ymin), try_eval_to_f64(ymax))
    {
      return Ok(num_to_expr(yn + t * (yx - yn)));
    }
    return Ok(num_to_expr(t));
  }

  Ok(Expr::FunctionCall {
    name: "Rescale".to_string(),
    args: args.to_vec(),
  })
}

/// Normalize[v] - normalizes a vector to unit length
/// Normalize[{3, 4}] => {3/5, 4/5}
/// Normalize[{0, 0, 0}] => {0, 0, 0}
pub fn normalize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Normalize expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::List(items) => {
      if items.is_empty() {
        return Ok(Expr::List(vec![]));
      }
      // Compute the Euclidean norm
      let mut vals = Vec::new();
      let mut all_int = true;
      let mut int_vals: Vec<i128> = Vec::new();
      for item in items {
        match item {
          Expr::Integer(n) => {
            vals.push(*n as f64);
            int_vals.push(*n);
          }
          Expr::Real(f) => {
            vals.push(*f);
            all_int = false;
          }
          _ => {
            return Ok(Expr::FunctionCall {
              name: "Normalize".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
      let norm_sq: f64 = vals.iter().map(|x| x * x).sum();
      if norm_sq == 0.0 {
        return Ok(args[0].clone());
      }
      let norm = norm_sq.sqrt();

      if all_int {
        // Try to keep exact: each element / Sqrt[sum_sq]
        let sum_sq: i128 = int_vals.iter().map(|x| x * x).sum();
        // Check if sum_sq is a perfect square
        let root = (sum_sq as f64).sqrt() as i128;
        if root * root == sum_sq && root > 0 {
          // Exact: each element / root
          let result: Vec<Expr> =
            int_vals.iter().map(|x| make_rational(*x, root)).collect();
          return Ok(Expr::List(result));
        }
        // Return as xi / Sqrt[sum_sq]
        let result: Vec<Expr> = int_vals
          .iter()
          .map(|x| {
            if *x == 0 {
              Expr::Integer(0)
            } else {
              // x / Sqrt[sum_sq] = x * Power[sum_sq, -1/2]
              Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(Expr::Integer(*x)),
                right: Box::new(Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::Integer(sum_sq)],
                }),
              }
            }
          })
          .collect();
        return Ok(Expr::List(result));
      }

      // Float path
      let result: Vec<Expr> =
        vals.iter().map(|x| num_to_expr(x / norm)).collect();
      Ok(Expr::List(result))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Normalize".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Unitize[x] - returns 0 for 0, 1 for anything else
/// Unitize[list] - maps over lists
pub fn unitize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Unitize expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(_) => Ok(Expr::Integer(1)),
    Expr::Real(f) if *f == 0.0 => Ok(Expr::Integer(0)),
    Expr::Real(_) => Ok(Expr::Integer(1)),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| unitize_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Unitize".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Ramp[x] - returns max(0, x)
/// Ramp[list] - maps over lists
pub fn ramp_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Ramp expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer((*n).max(0))),
    Expr::Real(f) => Ok(if *f > 0.0 {
      Expr::Real(*f)
    } else {
      Expr::Real(0.0)
    }),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| ramp_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Ramp".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// KroneckerDelta[args...] - returns 1 if all arguments are equal, 0 otherwise
/// KroneckerDelta[n] - returns 1 if n==0, 0 if n!=0 (equivalent to KroneckerDelta[n, 0])
pub fn kronecker_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Single argument: KroneckerDelta[n] is 1 if n==0, 0 otherwise
  if args.len() == 1 {
    return match &args[0] {
      Expr::Integer(n) => Ok(Expr::Integer(if *n == 0 { 1 } else { 0 })),
      Expr::Real(f) => Ok(Expr::Integer(if *f == 0.0 { 1 } else { 0 })),
      _ => Ok(Expr::FunctionCall {
        name: "KroneckerDelta".to_string(),
        args: args.to_vec(),
      }),
    };
  }

  // Multi argument: check if all are equal
  // First check if any are symbolic (non-numeric)
  let mut has_symbolic = false;
  let mut all_equal = true;
  let first_str = crate::syntax::expr_to_string(&args[0]);
  for arg in &args[1..] {
    let s = crate::syntax::expr_to_string(arg);
    if s != first_str {
      all_equal = false;
      // Check if both are numeric and compare numerically
      if let (Some(a), Some(b)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(arg))
      {
        if a != b {
          return Ok(Expr::Integer(0));
        } else {
          all_equal = true;
        }
      } else {
        has_symbolic = true;
      }
    }
  }
  if has_symbolic {
    Ok(Expr::FunctionCall {
      name: "KroneckerDelta".to_string(),
      args: args.to_vec(),
    })
  } else if all_equal {
    Ok(Expr::Integer(1))
  } else {
    Ok(Expr::Integer(0))
  }
}

/// UnitStep[x] - returns 0 for x < 0, 1 for x >= 0
/// UnitStep[x1, x2, ...] - returns 1 if all xi >= 0, 0 if any xi < 0
/// UnitStep[list] - maps over lists (single arg only)
pub fn unit_step_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "UnitStep expects at least 1 argument".into(),
    ));
  }

  // Multi-arg: UnitStep[x1, x2, ...] = product of UnitStep[xi]
  if args.len() > 1 {
    let mut has_symbolic = false;
    for arg in args {
      match arg {
        Expr::Integer(n) => {
          if *n < 0 {
            return Ok(Expr::Integer(0));
          }
        }
        Expr::Real(f) => {
          if *f < 0.0 {
            return Ok(Expr::Integer(0));
          }
        }
        _ => {
          has_symbolic = true;
        }
      }
    }
    if has_symbolic {
      return Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec(),
      });
    }
    return Ok(Expr::Integer(1));
  }

  // Single arg
  match &args[0] {
    Expr::Integer(n) => Ok(Expr::Integer(if *n >= 0 { 1 } else { 0 })),
    Expr::Real(f) => Ok(Expr::Integer(if *f >= 0.0 { 1 } else { 0 })),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, InterpreterError> =
        items.iter().map(|x| unit_step_ast(&[x.clone()])).collect();
      Ok(Expr::List(results?))
    }
    _ => Ok(Expr::FunctionCall {
      name: "UnitStep".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// FrobeniusNumber[{a1, a2, ...}] - Largest integer that cannot be represented
/// as a non-negative integer linear combination of the given positive integers.
/// Returns Infinity if the GCD is not 1, -1 if 1 is in the set.
pub fn frobenius_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "FrobeniusNumber".to_string(),
      args: args.to_vec(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "FrobeniusNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "FrobeniusNumber".to_string(),
      args: args.to_vec(),
    });
  }

  // Extract positive integers
  let mut nums: Vec<i128> = Vec::new();
  for item in items {
    match item {
      Expr::Integer(n) if *n > 0 => nums.push(*n),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "FrobeniusNumber".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  // If 1 is in the set, every non-negative integer is representable
  if nums.contains(&1) {
    return Ok(Expr::Integer(-1));
  }

  // Compute GCD of all elements
  fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
  }

  let mut g = nums[0];
  for &n in &nums[1..] {
    g = gcd(g, n);
  }

  // If GCD > 1, infinitely many integers can't be represented
  if g > 1 {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // For two coprime numbers, use the closed formula: a*b - a - b
  if nums.len() == 2 {
    let (a, b) = (nums[0], nums[1]);
    return Ok(Expr::Integer(a * b - a - b));
  }

  // General case: dynamic programming
  // Upper bound for Frobenius number: a1*a2 - a1 - a2 (using two smallest)
  nums.sort();
  let a0 = nums[0] as usize;

  // Use the round-robin algorithm (Wilf) based on shortest paths
  // Build shortest-path array: n[i] = smallest number representable that is ≡ i (mod a0)
  let mut n_arr = vec![i128::MAX; a0];
  n_arr[0] = 0;

  // BFS/relaxation
  let mut changed = true;
  while changed {
    changed = false;
    for residue in 0..a0 {
      if n_arr[residue] == i128::MAX {
        continue;
      }
      for &aj in &nums[1..] {
        let new_val = n_arr[residue] + aj;
        let new_residue = (new_val as usize) % a0;
        if new_val < n_arr[new_residue] {
          n_arr[new_residue] = new_val;
          changed = true;
        }
      }
    }
  }

  // Frobenius number is max(n_arr) - a0
  let max_n = *n_arr.iter().max().unwrap();
  Ok(Expr::Integer(max_n - a0 as i128))
}
