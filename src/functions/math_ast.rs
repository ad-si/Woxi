//! AST-native math functions.
//!
//! These functions work directly with `Expr` AST nodes.

use crate::InterpreterError;
use crate::syntax::Expr;
use num_bigint::BigInt;
use num_traits::Signed;

/// Helper - constants are kept symbolic, no direct f64 conversion in expr_to_num.
fn constant_to_f64(_name: &str) -> Option<f64> {
  None
}

fn expr_to_num(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::Constant(name) => constant_to_f64(name),
    Expr::Identifier(name) => constant_to_f64(name),
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

/// Compute the error function erf(x) using the Taylor series.
/// erf(x) = (2/sqrt(pi)) * sum_{n=0}^{inf} (-1)^n * x^(2n+1) / (n! * (2n+1))
pub fn erf_f64(x: f64) -> f64 {
  // erf is an odd function
  let sign = if x < 0.0 { -1.0 } else { 1.0 };
  let x = x.abs();

  if x > 4.0 {
    // For large |x|, erf(x) ≈ 1 - erfc(x) using continued fraction
    return sign * (1.0 - erfc_cf(x));
  }

  // Taylor series: erf(x) = (2/sqrt(pi)) * sum_{n=0}^inf (-1)^n * x^(2n+1) / (n! * (2n+1))
  let mut sum = 0.0;
  let mut term = x; // first term: x
  sum += term;
  for n in 1..100 {
    term *= -x * x / (n as f64);
    let contribution = term / (2 * n + 1) as f64;
    sum += contribution;
    if contribution.abs() < 1e-16 * sum.abs() {
      break;
    }
  }
  sign * sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// Compute erfc(x) for large x using the continued fraction representation.
/// erfc(x) = exp(-x^2) / (x * sqrt(pi)) * CF
fn erfc_cf(x: f64) -> f64 {
  // Modified Lentz's method for the continued fraction
  // erfc(x) = (exp(-x^2)/sqrt(pi)) * 1/(x + 1/(2x + 2/(x + 3/(2x + ...))))
  // Using the form: a_n / (b_n + ...) where a_1=1, a_n = (n-1)/2, b_n = x
  let mut f = x;
  let mut c = x;
  let mut d = 0.0;
  let tiny = 1e-30;

  for n in 1..200 {
    let a_n = n as f64 * 0.5;
    let b_n = x;
    d = b_n + a_n * d;
    if d.abs() < tiny {
      d = tiny;
    }
    c = b_n + a_n / c;
    if c.abs() < tiny {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = c * d;
    f *= delta;
    if (delta - 1.0).abs() < 1e-15 {
      break;
    }
  }

  (-x * x).exp() / (f * std::f64::consts::PI.sqrt())
}

/// Recursively try to evaluate any expression to f64.
/// This handles constants (Pi, E, Degree), arithmetic operations, and known functions.
/// Used by N[], comparisons, and anywhere a numeric value is needed from a symbolic expression.
pub fn try_eval_to_f64(expr: &Expr) -> Option<f64> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    Expr::BigFloat(digits, _) => digits.parse::<f64>().ok(),
    Expr::Constant(name) => match name.as_str() {
      "Pi" => Some(std::f64::consts::PI),
      "-Pi" => Some(-std::f64::consts::PI),
      "E" => Some(std::f64::consts::E),
      "Degree" => Some(std::f64::consts::PI / 180.0),
      "-Degree" => Some(-std::f64::consts::PI / 180.0),
      _ => None,
    },
    Expr::Identifier(name) => match name.as_str() {
      "Pi" => Some(std::f64::consts::PI),
      "E" => Some(std::f64::consts::E),
      "Degree" => Some(std::f64::consts::PI / 180.0),
      "EulerGamma" => Some(0.5772156649015329),
      "Catalan" => Some(0.915_965_594_177_219),
      "GoldenRatio" => Some(1.618_033_988_749_895),
      "Glaisher" => Some(1.2824271291006226),
      "Khinchin" => Some(2.6854520010653064),
      "MachinePrecision" => Some(15.954589770191003),
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
      "Erf" if args.len() == 1 => try_eval_to_f64(&args[0]).map(erf_f64),
      "Erfc" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| 1.0 - erf_f64(v))
      }
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
      "Floor" if args.len() == 2 => {
        let x = try_eval_to_f64(&args[0])?;
        let a = try_eval_to_f64(&args[1])?;
        if a == 0.0 {
          None
        } else {
          Some((x / a).floor() * a)
        }
      }
      "Ceiling" if args.len() == 1 => {
        try_eval_to_f64(&args[0]).map(|v| v.ceil())
      }
      "Ceiling" if args.len() == 2 => {
        let x = try_eval_to_f64(&args[0])?;
        let a = try_eval_to_f64(&args[1])?;
        if a == 0.0 {
          None
        } else {
          Some((x / a).ceil() * a)
        }
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
pub fn num_to_expr(n: f64) -> Expr {
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

/// Try to extract an i128 from Integer or BigInteger (if it fits).
fn expr_to_i128(e: &Expr) -> Option<i128> {
  use num_traits::ToPrimitive;
  match e {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => n.to_i128(),
    _ => None,
  }
}

/// Extract a BigInt from Integer or BigInteger.
fn expr_to_bigint(e: &Expr) -> Option<BigInt> {
  match e {
    Expr::Integer(n) => Some(BigInt::from(*n)),
    Expr::BigInteger(n) => Some(n.clone()),
    _ => None,
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

/// Extract an exact rational (numer, denom) from an Expr.
/// Integer n → (n, 1), Rational[n, d] → (n, d), otherwise None.
fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
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

pub fn make_rational(numer: i128, denom: i128) -> Expr {
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

/// Convert complex rational components to an Expr.
/// Given (re_num/re_den) + (im_num/im_den)*I, produce the canonical expression.
fn complex_rational_to_expr(
  re_n: i128,
  re_d: i128,
  im_n: i128,
  im_d: i128,
) -> Expr {
  let real_part = make_rational(re_n, re_d);
  let imag_part = make_rational(im_n, im_d);

  // Pure real
  if im_n == 0 {
    return real_part;
  }

  // Pure imaginary
  let i_expr = Expr::Identifier("I".to_string());
  if re_n == 0 {
    if matches!(&imag_part, Expr::Integer(1)) {
      return i_expr;
    }
    if matches!(&imag_part, Expr::Integer(-1)) {
      return Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(i_expr),
      };
    }
    return Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(imag_part),
      right: Box::new(i_expr),
    };
  }

  // General complex: real + imag*I
  let imag_term = if matches!(&imag_part, Expr::Integer(1)) {
    i_expr
  } else if matches!(&imag_part, Expr::Integer(-1)) {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(i_expr),
    }
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(imag_part),
      right: Box::new(i_expr),
    }
  };

  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(real_part),
    right: Box::new(imag_term),
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

  // Handle Quantity arithmetic before anything else
  if let Some(result) = crate::functions::quantity_ast::try_quantity_plus(args)
  {
    return result;
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

  // Check for Infinity + (-Infinity) → Indeterminate
  {
    let mut has_pos_inf = false;
    let mut has_neg_inf = false;
    for arg in &flat_args {
      match arg {
        Expr::Identifier(name) if name == "Infinity" => has_pos_inf = true,
        Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand,
        } if matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity") => {
          has_neg_inf = true
        }
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left,
          right,
        } if matches!(left.as_ref(), Expr::Integer(-1))
          && matches!(right.as_ref(), Expr::Identifier(n) if n == "Infinity") =>
        {
          has_neg_inf = true
        }
        _ => {}
      }
    }
    if has_pos_inf && has_neg_inf {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // Infinity + finite terms → Infinity, -Infinity + finite terms → -Infinity
    if has_pos_inf {
      return Ok(Expr::Identifier("Infinity".to_string()));
    }
    if has_neg_inf {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
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

  // Classify arguments: exact (Integer/Rational), real (Real), or symbolic
  let mut has_real = false;
  let mut all_numeric = true;
  for arg in &flat_args {
    match arg {
      Expr::Real(_) => has_real = true,
      Expr::Integer(_) => {}
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational"
          && rargs.len() == 2
          && matches!(rargs[0], Expr::Integer(_))
          && matches!(rargs[1], Expr::Integer(_)) => {}
      _ => {
        all_numeric = false;
      }
    }
  }

  // If all numeric and no Reals, use exact rational arithmetic
  if all_numeric && !has_real {
    // Sum as exact rational: (numer, denom)
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    for arg in &flat_args {
      if let Some((n, d)) = expr_to_rational(arg) {
        // sum_n/sum_d + n/d = (sum_n*d + n*sum_d) / (sum_d*d)
        sum_n = sum_n * d + n * sum_d;
        sum_d *= d;
        let g = gcd(sum_n, sum_d);
        sum_n /= g;
        sum_d /= g;
        // Keep denom positive
        if sum_d < 0 {
          sum_n = -sum_n;
          sum_d = -sum_d;
        }
      }
    }
    return Ok(make_rational(sum_n, sum_d));
  }

  // If all numeric but has Reals, use f64
  if all_numeric {
    let mut sum = 0.0;
    for arg in &flat_args {
      if let Some(n) = expr_to_num(arg) {
        sum += n;
      }
    }
    return Ok(Expr::Real(sum));
  }

  {
    // Separate numeric and symbolic terms
    let mut symbolic_args: Vec<Expr> = Vec::new();
    let mut has_exact = false;
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    let mut real_sum: f64 = 0.0;
    let mut has_real_term = false;

    for arg in &flat_args {
      if let Some((n, d)) = expr_to_rational(arg) {
        sum_n = sum_n * d + n * sum_d;
        sum_d *= d;
        let g = gcd(sum_n, sum_d);
        sum_n /= g;
        sum_d /= g;
        if sum_d < 0 {
          sum_n = -sum_n;
          sum_d = -sum_d;
        }
        has_exact = true;
      } else if let Expr::Real(f) = arg {
        real_sum += f;
        has_real_term = true;
      } else {
        symbolic_args.push(arg.clone());
      }
    }

    // Build final args: numeric sum first (if non-zero), then symbolic terms sorted
    let mut final_args: Vec<Expr> = Vec::new();

    // If we have both exact and real, convert exact to f64 and combine
    if has_exact && has_real_term {
      let total = (sum_n as f64) / (sum_d as f64) + real_sum;
      if total != 0.0 {
        final_args.push(Expr::Real(total));
      }
    } else if has_real_term && real_sum != 0.0 {
      final_args.push(Expr::Real(real_sum));
    } else if has_exact && sum_n != 0 {
      final_args.push(make_rational(sum_n, sum_d));
    }

    // Collect like terms: group symbolic terms by their base expression
    // e.g. E + E → 2*E, 3*x + 2*x → 5*x
    let collected = collect_like_terms(&symbolic_args);

    // Sort symbolic terms: polynomial-like terms first, then transcendental functions
    // This gives Mathematica-like ordering where x^2 comes before Sin[x].
    // For alphabetical comparison, strip the leading "-" from negated terms
    // so that -x sorts next to x rather than before everything.
    let mut sorted_symbolic = collected;
    sorted_symbolic.sort_by(|a, b| {
      let pa = term_priority(a);
      let pb = term_priority(b);
      if pa != pb {
        pa.cmp(&pb)
      } else {
        let sa = crate::syntax::expr_to_string(a);
        let sb = crate::syntax::expr_to_string(b);
        let sa_stripped = sa.strip_prefix('-').unwrap_or(&sa);
        let sb_stripped = sb.strip_prefix('-').unwrap_or(&sb);
        sa_stripped.cmp(sb_stripped)
      }
    });
    final_args.extend(sorted_symbolic);

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

/// Coefficient: either exact rational or approximate real
#[derive(Clone)]
enum Coeff {
  Exact(i128, i128), // (numer, denom)
  Real(f64),
}

impl Coeff {
  fn is_zero(&self) -> bool {
    match self {
      Coeff::Exact(n, _) => *n == 0,
      Coeff::Real(f) => *f == 0.0,
    }
  }
  fn is_one(&self) -> bool {
    match self {
      Coeff::Exact(n, d) => *n == 1 && *d == 1,
      Coeff::Real(f) => *f == 1.0,
    }
  }
  fn to_f64(&self) -> f64 {
    match self {
      Coeff::Exact(n, d) => *n as f64 / *d as f64,
      Coeff::Real(f) => *f,
    }
  }
  fn add(&self, other: &Coeff) -> Coeff {
    match (self, other) {
      (Coeff::Exact(n1, d1), Coeff::Exact(n2, d2)) => {
        let mut sn = n1 * d2 + n2 * d1;
        let mut sd = d1 * d2;
        let g = gcd(sn, sd);
        sn /= g;
        sd /= g;
        if sd < 0 {
          sn = -sn;
          sd = -sd;
        }
        Coeff::Exact(sn, sd)
      }
      _ => Coeff::Real(self.to_f64() + other.to_f64()),
    }
  }
  fn negate(&self) -> Coeff {
    match self {
      Coeff::Exact(n, d) => Coeff::Exact(-n, *d),
      Coeff::Real(f) => Coeff::Real(-f),
    }
  }
  fn to_expr(&self) -> Expr {
    match self {
      Coeff::Exact(n, d) => make_rational(*n, *d),
      Coeff::Real(f) => Expr::Real(*f),
    }
  }
}

/// Decompose a term into (coefficient, base_expression).
/// E.g. `3*x` → (Exact(3,1), x), `x` → (Exact(1,1), x), `-x` → (Exact(-1,1), x),
/// `1.5*x` → (Real(1.5), x), `Rational[3,4]*x` → (Exact(3,4), x).
fn decompose_term(e: &Expr) -> (Coeff, Expr) {
  match e {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      // Check if first arg is a numeric coefficient (integer/rational)
      if let Some((n, d)) = expr_to_rational(&args[0]) {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        };
        return (Coeff::Exact(n, d), base);
      }
      // Check if first arg is a Real coefficient
      if let Expr::Real(f) = &args[0] {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        };
        return (Coeff::Real(*f), base);
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some((n, d)) = expr_to_rational(left) {
        return (Coeff::Exact(n, d), *right.clone());
      }
      if let Expr::Real(f) = left.as_ref() {
        return (Coeff::Real(*f), *right.clone());
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (c, base) = decompose_term(operand);
      return (c.negate(), base);
    }
    _ => {}
  }
  (Coeff::Exact(1, 1), e.clone())
}

/// Collect like terms: group symbolic terms by their base expression
/// and sum their coefficients. E.g. [E, E] → [2*E], [3*x, 2*x] → [5*x].
fn collect_like_terms(terms: &[Expr]) -> Vec<Expr> {
  use std::collections::BTreeMap;

  // Group by string representation of base → sum of coefficients
  let mut groups: Vec<(String, Expr, Coeff)> = Vec::new();
  let mut index: BTreeMap<String, usize> = BTreeMap::new();

  for term in terms {
    let (c, base) = decompose_term(term);
    let key = crate::syntax::expr_to_string(&base);
    if let Some(&idx) = index.get(&key) {
      let (_, _, ref mut sum_c) = groups[idx];
      *sum_c = sum_c.add(&c);
    } else {
      index.insert(key.clone(), groups.len());
      groups.push((key, base, c));
    }
  }

  let mut result = Vec::new();
  for (_, base, c) in groups {
    if c.is_zero() {
      continue; // terms cancelled
    }
    if c.is_one() {
      result.push(base);
    } else {
      // Reconstruct as flat Times[coefficient, base_args...] to preserve formatting
      let coeff = c.to_expr();
      let mut times_args = vec![coeff];
      // Flatten base if it's already a Times
      match &base {
        Expr::FunctionCall { name, args: bargs } if name == "Times" => {
          times_args.extend(bargs.clone());
        }
        _ => {
          times_args.push(base);
        }
      }
      result.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: times_args,
      });
    }
  }
  result
}

/// Sort symbolic factors in Times using the same ordering as Wolfram:
/// polynomial-like terms first (variables, powers), then transcendental functions,
/// with alphabetical ordering within each group.
/// Compute term priority for sorting: 0 = polynomial-like, 1 = transcendental.
fn term_priority(e: &Expr) -> i32 {
  match e {
    Expr::Identifier(_) => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      ..
    } => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      ..
    } => term_priority(left),
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => term_priority(left).max(term_priority(right)),
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Times" | "Power" | "Plus" | "Rational" => 0,
      "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh"
      | "Tanh" | "Coth" | "Sech" | "Csch" | "ArcSin" | "ArcCos" | "ArcTan"
      | "ArcCot" | "ArcSec" | "ArcCsc" | "Exp" | "Log" | "Factorial"
      | "Erf" | "Erfc" => 1,
      _ => 0,
    },
    Expr::UnaryOp { operand, .. } => term_priority(operand),
    _ => 0,
  }
}

/// Sub-priority for Times factor ordering: identifiers before compound expressions.
/// This ensures simple symbols sort before sums/products, matching Wolfram behavior.
fn times_factor_subpriority(e: &Expr) -> i32 {
  match e {
    Expr::Identifier(_) | Expr::Constant(_) => 0,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      ..
    } => times_factor_subpriority(left),
    Expr::BinaryOp {
      op:
        crate::syntax::BinaryOperator::Plus | crate::syntax::BinaryOperator::Minus,
      ..
    } => 1,
    Expr::FunctionCall { name, .. } => match name.as_str() {
      "Times" | "Power" | "Rational" => 0,
      "Plus" => 1,
      _ => 2,
    },
    _ => 0,
  }
}

fn sort_symbolic_factors(symbolic_args: &mut [Expr]) {
  symbolic_args.sort_by(|a, b| {
    let pa = term_priority(a);
    let pb = term_priority(b);
    if pa != pb {
      return pa.cmp(&pb);
    }
    // Within same priority, identifiers before function calls
    let sa = times_factor_subpriority(a);
    let sb = times_factor_subpriority(b);
    if sa != sb {
      return sa.cmp(&sb);
    }
    crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
  });
}

/// Extract (base, exponent) from an expression for power combining in Times.
/// x → (x, 1), x^n → (x, n), Sqrt[x] → (x, 1/2)
fn extract_base_exponent(expr: &Expr) -> (Expr, Expr) {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => (*left.clone(), *right.clone()),
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      (args[0].clone(), make_rational(1, 2))
    }
    _ => (expr.clone(), Expr::Integer(1)),
  }
}

/// Combine like bases in a list of symbolic factors: x^a * x^b → x^(a+b)
fn combine_like_bases(args: Vec<Expr>) -> Result<Vec<Expr>, InterpreterError> {
  if args.len() <= 1 {
    return Ok(args);
  }

  // Use string representation of base as grouping key
  let mut groups: Vec<(String, Expr, Vec<Expr>)> = Vec::new(); // (base_key, base, exponents)
  let mut non_combinable: Vec<Expr> = Vec::new();

  for arg in &args {
    // Don't combine Plus, Times, or complex expressions - only identifiers, constants, and powers thereof
    let (base, exp) = extract_base_exponent(arg);
    let combinable = match &base {
      Expr::Identifier(_) | Expr::Constant(_) => true,
      Expr::FunctionCall { name, .. } => {
        matches!(name.as_str(), "Sqrt" | "Log" | "Sin" | "Cos")
      }
      _ => false,
    };
    if !combinable {
      non_combinable.push(arg.clone());
      continue;
    }
    let base_key = crate::syntax::expr_to_string(&base);
    if let Some(group) = groups.iter_mut().find(|(k, _, _)| *k == base_key) {
      group.2.push(exp);
    } else {
      groups.push((base_key, base, vec![exp]));
    }
  }

  let mut result: Vec<Expr> = Vec::new();
  for (_key, base, exponents) in groups {
    if exponents.len() == 1 {
      // Single occurrence — no combining needed, reconstruct original form
      if matches!(&exponents[0], Expr::Integer(1)) {
        result.push(base);
      } else {
        result.push(power_two(&base, &exponents[0])?);
      }
    } else {
      // Multiple occurrences — add exponents
      let combined_exp = plus_ast(&exponents)?;
      if matches!(&combined_exp, Expr::Integer(0)) {
        // x^0 = 1, skip (will be absorbed into coefficient)
        continue;
      }
      result.push(power_two(&base, &combined_exp)?);
    }
  }
  result.extend(non_combinable);

  // Second pass: combine numeric bases with the same fractional exponent
  // e.g. Sqrt[2] * Sqrt[3] = 2^(1/2) * 3^(1/2) → 6^(1/2) = Sqrt[6]
  let mut combined: Vec<Expr> = Vec::new();
  let mut used = vec![false; result.len()];
  for i in 0..result.len() {
    if used[i] {
      continue;
    }
    let (base_i, exp_i) = extract_base_exponent(&result[i]);
    // Only combine integer bases with rational exponents
    let is_numeric_base =
      matches!(&base_i, Expr::Integer(_) | Expr::BigInteger(_));
    let is_rational_exp = matches!(
      &exp_i,
      Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2
    );
    if !is_numeric_base || !is_rational_exp {
      combined.push(result[i].clone());
      continue;
    }
    let exp_key = crate::syntax::expr_to_string(&exp_i);
    let mut bases_to_multiply = vec![base_i];
    for j in (i + 1)..result.len() {
      if used[j] {
        continue;
      }
      let (base_j, exp_j) = extract_base_exponent(&result[j]);
      if crate::syntax::expr_to_string(&exp_j) == exp_key
        && matches!(&base_j, Expr::Integer(_) | Expr::BigInteger(_))
      {
        bases_to_multiply.push(base_j);
        used[j] = true;
      }
    }
    if bases_to_multiply.len() == 1 {
      combined.push(result[i].clone());
    } else {
      let product = times_ast(&bases_to_multiply)?;
      combined.push(power_two(&product, &exp_i)?);
    }
  }

  Ok(combined)
}

/// Times[args...] - Product of arguments, with list threading
pub fn times_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  // Handle Quantity arithmetic before anything else
  if let Some(result) = crate::functions::quantity_ast::try_quantity_times(args)
  {
    return result;
  }

  // Flatten nested Times arguments
  let mut flat_args: Vec<Expr> = Vec::new();
  for arg in args {
    match arg {
      Expr::FunctionCall {
        name,
        args: inner_args,
      } if name == "Times" => {
        flat_args.extend(inner_args.clone());
      }
      _ => flat_args.push(arg.clone()),
    }
  }
  let args = &flat_args;

  // Try complex multiplication: if all args can be extracted as exact complex
  // numbers (and at least one has nonzero imaginary part), multiply them.
  if args.len() >= 2 {
    let complex_parts: Vec<_> =
      args.iter().map(try_extract_complex_exact).collect();
    if complex_parts.iter().all(|c| c.is_some()) {
      let has_imaginary = complex_parts.iter().any(|c| {
        if let Some((_, (im, _))) = c {
          *im != 0
        } else {
          false
        }
      });
      if has_imaginary {
        // Multiply all complex parts
        let mut re_n: i128 = complex_parts[0].unwrap().0.0;
        let mut re_d: i128 = complex_parts[0].unwrap().0.1;
        let mut im_n: i128 = complex_parts[0].unwrap().1.0;
        let mut im_d: i128 = complex_parts[0].unwrap().1.1;
        let mut ok = true;
        for cp in &complex_parts[1..] {
          let ((cn, cd), (dn, dd)) = cp.unwrap();
          // (re + im*i) * (cn/cd + dn/dd*i)
          let new_re = (|| {
            let a = re_n.checked_mul(cn)?.checked_mul(im_d)?.checked_mul(dd)?;
            let b = im_n.checked_mul(dn)?.checked_mul(re_d)?.checked_mul(cd)?;
            let num = a.checked_sub(b)?;
            let den =
              re_d.checked_mul(cd)?.checked_mul(im_d)?.checked_mul(dd)?;
            Some((num, den))
          })();
          let new_im = (|| {
            let a = re_n.checked_mul(dn)?.checked_mul(im_d)?.checked_mul(cd)?;
            let b = im_n.checked_mul(cn)?.checked_mul(re_d)?.checked_mul(dd)?;
            let num = a.checked_add(b)?;
            let den =
              re_d.checked_mul(dd)?.checked_mul(im_d)?.checked_mul(cd)?;
            Some((num, den))
          })();
          if let (Some((rn, rd)), Some((in_, id))) = (new_re, new_im) {
            re_n = rn;
            re_d = rd;
            im_n = in_;
            im_d = id;
          } else {
            ok = false;
            break;
          }
        }
        if ok {
          return Ok(complex_rational_to_expr(re_n, re_d, im_n, im_d));
        }
      }
    }
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

    symbolic_args = combine_like_bases(symbolic_args)?;
    sort_symbolic_factors(&mut symbolic_args);
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

  // Separate into: integers, rationals, reals, and symbolic arguments
  let mut int_product: i128 = 1;
  let mut int_overflow = false;
  let mut has_int = false;
  let mut rat_numer: i128 = 1;
  let mut rat_denom: i128 = 1;
  let mut has_rational = false;
  let mut real_product: f64 = 1.0;
  let mut any_real = false;
  let mut symbolic_args: Vec<Expr> = Vec::new();

  for arg in args {
    match arg {
      Expr::Integer(n) => {
        if let Some(result) = int_product.checked_mul(*n) {
          int_product = result;
        } else {
          int_overflow = true;
        }
        has_int = true;
      }
      Expr::Real(f) => {
        real_product *= f;
        any_real = true;
      }
      Expr::FunctionCall { name, args: rargs }
        if name == "Rational" && rargs.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1]) {
          if let (Some(rn), Some(rd)) =
            (rat_numer.checked_mul(*n), rat_denom.checked_mul(*d))
          {
            rat_numer = rn;
            rat_denom = rd;
          } else {
            int_overflow = true;
          }
          has_rational = true;
        } else {
          symbolic_args.push(arg.clone());
        }
      }
      _ => {
        if let Some(n) = expr_to_num(arg) {
          real_product *= n;
          any_real = true;
        } else {
          symbolic_args.push(arg.clone());
        }
      }
    }
  }

  // If overflow detected, fall back to BigInt arithmetic
  if int_overflow {
    use num_bigint::BigInt;
    let mut big_product = BigInt::from(1);
    let mut sym_args: Vec<Expr> = Vec::new();
    for arg in args {
      match arg {
        Expr::Integer(n) => big_product *= BigInt::from(*n),
        Expr::BigInteger(n) => big_product *= n,
        _ => sym_args.push(arg.clone()),
      }
    }
    if sym_args.is_empty() {
      return Ok(bigint_to_expr(big_product));
    }
    if big_product == BigInt::from(0) {
      return Ok(Expr::Integer(0));
    }
    sym_args = combine_like_bases(sym_args)?;
    sort_symbolic_factors(&mut sym_args);
    let mut final_args: Vec<Expr> = Vec::new();
    if big_product != BigInt::from(1) {
      final_args.push(bigint_to_expr(big_product));
    }
    final_args.extend(sym_args);
    if final_args.len() == 1 {
      return Ok(final_args.remove(0));
    }
    return Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: final_args,
    });
  }

  // If any Real, try to convert symbolic constants (Pi, E, etc.) to floats
  if any_real {
    let mut remaining_symbolic: Vec<Expr> = Vec::new();
    for arg in symbolic_args.drain(..) {
      if let Some(f) = try_eval_to_f64(&arg) {
        real_product *= f;
      } else {
        remaining_symbolic.push(arg);
      }
    }
    symbolic_args = remaining_symbolic;

    let total = (int_product as f64)
      * (rat_numer as f64 / rat_denom as f64)
      * real_product;
    if symbolic_args.is_empty() {
      return Ok(Expr::Real(total));
    }
    if total == 0.0 {
      // Check if any remaining symbolic arg involves I (imaginary unit)
      let has_imag = symbolic_args
        .iter()
        .any(|a| matches!(a, Expr::Identifier(s) if s == "I"));
      if has_imag {
        // 0.0 * I → 0. + 0.*I (Complex form)
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Plus,
          left: Box::new(Expr::Real(0.0)),
          right: Box::new(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Real(0.0)),
            right: Box::new(Expr::Identifier("I".to_string())),
          }),
        });
      }
      // 0.0 * x → 0. (approximate zero, not exact)
      return Ok(Expr::Real(0.0));
    }
    symbolic_args = combine_like_bases(symbolic_args)?;
    sort_symbolic_factors(&mut symbolic_args);
    let mut final_args: Vec<Expr> = Vec::new();
    if total != 1.0 {
      final_args.push(Expr::Real(total));
    }
    final_args.extend(symbolic_args);
    return if final_args.len() == 1 {
      Ok(final_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: final_args,
      })
    };
  }

  // Exact arithmetic: combine integer * rational
  // Result is (int_product * rat_numer) / rat_denom
  let combined_numer = int_product * rat_numer;
  let combined_denom = rat_denom;
  let coeff = if has_rational || (has_int && combined_denom != 1) {
    make_rational(combined_numer, combined_denom)
  } else {
    Expr::Integer(int_product)
  };

  // If all arguments are numeric, return the product
  if symbolic_args.is_empty() {
    return Ok(coeff);
  }

  // 0 * anything = 0
  if combined_numer == 0 {
    return Ok(Expr::Integer(0));
  }

  // Handle Infinity in symbolic args: n * Infinity = ±Infinity
  if symbolic_args.len() == 1 {
    let is_pos_inf =
      matches!(&symbolic_args[0], Expr::Identifier(s) if s == "Infinity");
    let is_neg_inf = match &symbolic_args[0] {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity"),
      _ => false,
    };
    if is_pos_inf || is_neg_inf {
      let coeff_positive = combined_numer > 0;
      // Positive * Infinity or Negative * (-Infinity) → Infinity
      // Negative * Infinity or Positive * (-Infinity) → -Infinity
      let result_positive = coeff_positive == is_pos_inf;
      if result_positive {
        return Ok(Expr::Identifier("Infinity".to_string()));
      } else {
        return Ok(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Identifier("Infinity".to_string())),
        });
      }
    }
  }

  // Times[-1, Plus[args...]] distributes: -1*(a+b+c) → (-a)+(-b)+(-c)
  // This only applies when coefficient is exactly -1 and the sole symbolic arg is Plus
  if matches!(&coeff, Expr::Integer(-1))
    && symbolic_args.len() == 1
    && matches!(&symbolic_args[0], Expr::FunctionCall { name, .. } if name == "Plus")
    && let Expr::FunctionCall {
      name: _,
      args: plus_args,
    } = &symbolic_args[0]
  {
    let negated: Result<Vec<Expr>, InterpreterError> = plus_args
      .iter()
      .map(|a| times_ast(&[Expr::Integer(-1), a.clone()]))
      .collect();
    return plus_ast(&negated?);
  }

  // Combine like bases: x^a * x^b → x^(a+b)
  symbolic_args = combine_like_bases(symbolic_args)?;

  // If all symbolic args canceled (e.g. x^2 * x^(-2)), return coefficient
  if symbolic_args.is_empty() {
    return Ok(coeff);
  }

  // Build final args: coefficient (if not 1) + sorted symbolic terms
  sort_symbolic_factors(&mut symbolic_args);
  let mut final_args: Vec<Expr> = Vec::new();
  let is_unit = matches!(&coeff, Expr::Integer(1));
  if !is_unit {
    final_args.push(coeff);
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
    // Use times_ast for proper distribution: -(a+b) → -a - b
    times_ast(&[Expr::Integer(-1), args[0].clone()])
  } else {
    // Wrong arity - print error to stderr and return unevaluated expression
    eprintln!();
    eprintln!(
      "Minus::argx: Minus called with {} arguments; 1 argument is expected.",
      args.len()
    );
    // Return unevaluated (like Wolfram) — expr_to_string handles display
    Ok(Expr::FunctionCall {
      name: "Minus".to_string(),
      args: args.to_vec(),
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
    return thread_binary_over_lists(args, divide_two);
  }

  divide_two(&args[0], &args[1])
}

/// Check if an expression represents an infinite quantity
fn is_infinity_like(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) => name == "Infinity" || name == "ComplexInfinity",
    Expr::FunctionCall { name, .. } => name == "DirectedInfinity",
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity"),
    _ => false,
  }
}

/// Helper for division of two arguments
fn divide_two(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  // finite / Infinity or finite / DirectedInfinity[z] → 0
  if is_infinity_like(b) && !is_infinity_like(a) {
    return Ok(Expr::Integer(0));
  }

  // Handle Quantity division
  if let Some(result) =
    crate::functions::quantity_ast::try_quantity_divide(a, b)
  {
    return result;
  }

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
  {
    // BinaryOp form
    if let Expr::BinaryOp {
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
    // FunctionCall("Times", ...) form
    if let Expr::FunctionCall { name, args: targs } = a
      && name == "Times"
    {
      for (i, arg) in targs.iter().enumerate() {
        if let Expr::Integer(n) = arg {
          let coeff = make_rational(*n, *d);
          let mut rest: Vec<Expr> = targs
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, e)| e.clone())
            .collect();
          if rest.len() == 1 {
            return multiply_scalar_by_expr(&coeff, &rest.remove(0));
          } else {
            let rest_expr = Expr::FunctionCall {
              name: "Times".to_string(),
              args: rest,
            };
            return multiply_scalar_by_expr(&coeff, &rest_expr);
          }
        }
      }
    }
  }

  // Complex number division: (a + b*I) / integer → simplify
  if let Expr::Integer(d) = b
    && *d != 0
    && let Some(((re_n, re_d), (im_n, im_d))) = try_extract_complex_exact(a)
    && im_n != 0
  {
    // (re + im*I) / d = re/d + (im/d)*I
    let new_re_n = re_n;
    let new_re_d = re_d * *d;
    let new_im_n = im_n;
    let new_im_d = im_d * *d;
    return Ok(complex_rational_to_expr(
      new_re_n, new_re_d, new_im_n, new_im_d,
    ));
  }

  // For reals, perform floating-point division
  // Use try_eval_to_f64 when at least one operand is Real to handle constants like Pi/4.0
  let has_real = matches!(a, Expr::Real(_)) || matches!(b, Expr::Real(_));
  let eval_fn = if has_real {
    |e: &Expr| try_eval_to_f64(e)
  } else {
    |e: &Expr| expr_to_num(e)
  };
  match (eval_fn(a), eval_fn(b)) {
    (Some(x), Some(y)) => {
      if y == 0.0 {
        Err(InterpreterError::EvaluationError("Division by zero".into()))
      } else {
        Ok(Expr::Real(x / y))
      }
    }
    _ => {
      // x / x → 1 for identical symbolic expressions
      if crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b) {
        return Ok(Expr::Integer(1));
      }
      // Flatten nested divisions: (a/b)/c → a/(b*c), a/(b/c) → (a*c)/b
      let (num, den) = flatten_division(a, b);
      Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(num),
        right: Box::new(den),
      })
    }
  }
}

/// Flatten nested divisions into a single numerator and denominator.
/// (a/b)/c → (a, b*c), a/(b/c) → (a*c, b), (a/b)/(c/d) → (a*d, b*c)
fn flatten_division(a: &Expr, b: &Expr) -> (Expr, Expr) {
  // Extract (numerator, denominator) from each side
  let (a_num, a_den) = extract_num_den(a);
  let (b_num, b_den) = extract_num_den(b);

  // a/b = (a_num/a_den) / (b_num/b_den) = (a_num * b_den) / (a_den * b_num)
  let num = if a_den.is_none() && b_den.is_none() {
    a_num.clone()
  } else if let Some(bd) = &b_den {
    // a_num * b_den
    build_times_simple(&a_num, bd)
  } else {
    a_num.clone()
  };

  let den = if a_den.is_none() && b_den.is_none() {
    b_num.clone()
  } else if let Some(ad) = &a_den {
    if b_den.is_some() {
      // a_den * b_num
      build_times_simple(ad, &b_num)
    } else {
      // a_den * b_num (b has no denominator)
      build_times_simple(ad, &b_num)
    }
  } else {
    // a has no denominator, b has denominator
    b_num.clone()
  };

  (num, den)
}

/// Extract numerator and optional denominator from an expression.
fn extract_num_den(e: &Expr) -> (Expr, Option<Expr>) {
  match e {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => (*left.clone(), Some(*right.clone())),
    _ => (e.clone(), None),
  }
}

/// Build Times[a, b] without full evaluation (just structural)
fn build_times_simple(a: &Expr, b: &Expr) -> Expr {
  // For simple integer multiplication, compute directly
  if let (Expr::Integer(x), Expr::Integer(y)) = (a, b) {
    return Expr::Integer(x * y);
  }
  // For 1 * x, just return x
  if matches!(a, Expr::Integer(1)) {
    return b.clone();
  }
  if matches!(b, Expr::Integer(1)) {
    return a.clone();
  }
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
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
  // Handle Quantity^n
  if let Some(result) =
    crate::functions::quantity_ast::try_quantity_power(base, exp)
  {
    return result;
  }

  // x^1 -> x
  if matches!(exp, Expr::Integer(1)) {
    return Ok(base.clone());
  }

  // x^0 -> 1 (for non-zero x; 0^0 is Indeterminate, handled below)
  if matches!(exp, Expr::Integer(0))
    && !matches!(base, Expr::Integer(0))
    && !matches!(base, Expr::Real(f) if *f == 0.0)
  {
    return Ok(Expr::Integer(1));
  }

  // Sqrt[x]^n → x^(n/2)
  if let Expr::FunctionCall { name, args: fargs } = base
    && name == "Sqrt"
    && fargs.len() == 1
    && let Expr::Integer(n) = exp
  {
    if *n == 2 {
      return Ok(fargs[0].clone());
    }
    // Sqrt[x]^n = x^(n/2)
    return power_two(&fargs[0], &make_rational(*n, 2));
  }

  // (base^exp1)^exp2 -> base^(exp1*exp2) when both exponents are integers
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left: inner_base,
    right: inner_exp,
  } = base
    && let (Expr::Integer(e1), Expr::Integer(e2)) = (inner_exp.as_ref(), exp)
  {
    let combined = *e1 * *e2;
    return power_two(inner_base, &Expr::Integer(combined));
  }

  // I^n cycles with period 4: I^0=1, I^1=I, I^2=-1, I^3=-I
  if let Expr::Identifier(name) = base
    && name == "I"
    && let Expr::Integer(n) = exp
  {
    let r = ((*n % 4) + 4) % 4; // always non-negative mod
    return Ok(match r {
      0 => Expr::Integer(1),
      1 => Expr::Identifier("I".to_string()),
      2 => Expr::Integer(-1),
      3 => negate_expr(Expr::Identifier("I".to_string())),
      _ => unreachable!(),
    });
  }

  // E^(complex) → Euler's formula: E^(a + b*I) = E^a * (Cos[b] + I*Sin[b])
  if matches!(base, Expr::Constant(c) if c == "E") {
    // Try float complex extraction for the exponent
    if let Some((re, im)) = try_extract_complex_float(exp)
      && im != 0.0
    {
      let mag = re.exp();
      let cos_val = im.cos();
      let sin_val = im.sin();
      let real_part = mag * cos_val;
      let imag_part = mag * sin_val;
      if imag_part == 0.0 {
        return Ok(Expr::Real(real_part));
      }
      if real_part == 0.0 {
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Real(imag_part)),
          right: Box::new(Expr::Identifier("I".to_string())),
        });
      }
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(Expr::Real(real_part)),
        right: Box::new(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Real(imag_part)),
          right: Box::new(Expr::Identifier("I".to_string())),
        }),
      });
    }
  }

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

  // Special case: Rational^Integer -> exact rational result
  if let Expr::FunctionCall {
    name: rname,
    args: rargs,
  } = base
    && rname == "Rational"
    && rargs.len() == 2
    && let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1])
    && let Expr::Integer(e) = exp
  {
    if *e > 0 {
      let pe = *e as u32;
      if let (Some(new_num), Some(new_den)) =
        (num.checked_pow(pe), den.checked_pow(pe))
      {
        return Ok(make_rational(new_num, new_den));
      }
    } else if *e < 0 {
      // (a/b)^(-n) = (b/a)^n = b^n / a^n
      let pe = (-*e) as u32;
      if let (Some(new_num), Some(new_den)) =
        (den.checked_pow(pe), num.checked_pow(pe))
      {
        return Ok(make_rational(new_num, new_den));
      }
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
    // Simplify n^(p/q) by prime factorization
    // n = p1^k1 * p2^k2 * ...
    // n^(p/q) = product of p_i^(k_i*p/q)
    // Each p_i^(k_i*p/q) = p_i^floor_i * p_i^(rem_i/q)
    if *numer > 0 && *denom > 0 && *b > 0 {
      let d = *denom as u64;
      let n = *numer as u64;
      let mut outside: i128 = 1;
      // Collect (prime, remainder_exponent) pairs for the radical part
      let mut radical_factors: Vec<(i128, u64)> = Vec::new();
      let mut remaining = *b as u64;
      let mut factor = 2u64;
      while factor * factor <= remaining {
        let mut count = 0u64;
        while remaining.is_multiple_of(factor) {
          remaining /= factor;
          count += 1;
        }
        if count > 0 {
          let total = count * n;
          let extracted = total / d;
          let leftover = total % d;
          if extracted > 0 {
            outside *= (factor as i128).pow(extracted as u32);
          }
          if leftover > 0 {
            radical_factors.push((factor as i128, leftover));
          }
        }
        factor += 1;
      }
      if remaining > 1 {
        let total = n; // count=1
        let extracted = total / d;
        let leftover = total % d;
        if extracted > 0 {
          outside *= (remaining as i128).pow(extracted as u32);
        }
        if leftover > 0 {
          radical_factors.push((remaining as i128, leftover));
        }
      }

      let has_radical = !radical_factors.is_empty();
      let has_outside = outside > 1;

      if !has_radical {
        // Fully simplified
        return Ok(Expr::Integer(outside));
      }

      // Only simplify if we actually extracted something outside,
      // or if the radical has fewer prime factors than the original
      // (prevents infinite recursion: 6^(1/3) → 2^(1/3)*3^(1/3) → 6^(1/3) ...)
      if !has_outside && radical_factors.len() > 1 {
        // No simplification possible, keep original form
        if *numer == 1 && *denom == 2 {
          return Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![base.clone()],
          });
        }
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(base.clone()),
          right: Box::new(exp.clone()),
        });
      }

      // Build radical part: product of p_i^(rem_i/q)
      let mut rad_parts: Vec<Expr> = Vec::new();
      for (prime, rem_exp) in &radical_factors {
        let g = gcd_i128(*rem_exp as i128, d as i128);
        let reduced_num = *rem_exp as i128 / g;
        let reduced_den = d as i128 / g;
        if reduced_den == 1 {
          // Integer power
          rad_parts.push(Expr::Integer(prime.pow(reduced_num as u32)));
        } else if reduced_num == 1 && reduced_den == 2 {
          rad_parts.push(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(*prime)],
          });
        } else {
          rad_parts.push(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(Expr::Integer(*prime)),
            right: Box::new(make_rational(reduced_num, reduced_den)),
          });
        }
      }

      // Combine: outside * rad_parts
      let mut all_factors: Vec<Expr> = Vec::new();
      if has_outside {
        all_factors.push(Expr::Integer(outside));
      }
      all_factors.extend(rad_parts);

      if all_factors.len() == 1 {
        return Ok(all_factors.remove(0));
      }
      return times_ast(&all_factors);
    }

    // If exponent is 1/2, display as Sqrt[base]
    if *numer == 1 && *denom == 2 && *b > 0 {
      return Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![base.clone()],
      });
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
    _ => {
      // x^(1/2) → Sqrt[x]
      if let Expr::FunctionCall { name, args: rargs } = exp
        && name == "Rational"
        && rargs.len() == 2
        && matches!(&rargs[0], Expr::Integer(1))
        && matches!(&rargs[1], Expr::Integer(2))
      {
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![base.clone()],
        });
      }
      Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base.clone()),
        right: Box::new(exp.clone()),
      })
    }
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

/// Recursively flatten all List arguments for Max/Min
fn flatten_lists(args: &[Expr]) -> Vec<&Expr> {
  let mut result = Vec::new();
  for arg in args {
    match arg {
      Expr::List(items) => result.extend(flatten_lists(items)),
      _ => result.push(arg),
    }
  }
  result
}

/// Like try_eval_to_f64 but also handles Infinity/-Infinity (for Max/Min)
fn try_eval_to_f64_with_infinity(expr: &Expr) -> Option<f64> {
  // Check by string representation for Infinity forms
  let s = crate::syntax::expr_to_string(expr);
  if s == "Infinity" {
    return Some(f64::INFINITY);
  }
  if s == "-Infinity" {
    return Some(f64::NEG_INFINITY);
  }
  try_eval_to_f64(expr)
}

/// Max[args...] or Max[list] - Maximum value
pub fn max_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("-Infinity".to_string()));
  }

  // Flatten all nested lists
  let items = flatten_lists(args);
  if items.is_empty() {
    return Ok(Expr::Identifier("-Infinity".to_string()));
  }

  // Separate numeric and symbolic arguments
  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  let mut symbolic: Vec<Expr> = Vec::new();
  for item in &items {
    if let Some(n) = try_eval_to_f64_with_infinity(item) {
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
      symbolic.push((*item).clone());
    }
  }

  if symbolic.is_empty() {
    // All numeric
    match best_expr {
      Some(expr) => Ok((*expr).clone()),
      None => Ok(num_to_expr(f64::NEG_INFINITY)),
    }
  } else {
    // Mixed: keep max numeric and all symbolic args
    let mut result_args: Vec<Expr> = Vec::new();
    if let Some(expr) = best_expr {
      result_args.push((*expr).clone());
    }
    result_args.extend(symbolic);
    if result_args.len() == 1 {
      Ok(result_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Max".to_string(),
        args: result_args,
      })
    }
  }
}

/// Min[args...] or Min[list] - Minimum value
pub fn min_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Flatten all nested lists
  let items = flatten_lists(args);
  if items.is_empty() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }

  // Separate numeric and symbolic arguments
  let mut best_val: Option<f64> = None;
  let mut best_expr: Option<&Expr> = None;
  let mut symbolic: Vec<Expr> = Vec::new();
  for item in &items {
    if let Some(n) = try_eval_to_f64_with_infinity(item) {
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
      symbolic.push((*item).clone());
    }
  }

  if symbolic.is_empty() {
    // All numeric
    match best_expr {
      Some(expr) => Ok((*expr).clone()),
      None => Ok(num_to_expr(f64::INFINITY)),
    }
  } else {
    // Mixed: keep min numeric and all symbolic args
    let mut result_args: Vec<Expr> = Vec::new();
    if let Some(expr) = best_expr {
      result_args.push((*expr).clone());
    }
    result_args.extend(symbolic);
    if result_args.len() == 1 {
      Ok(result_args.remove(0))
    } else {
      Ok(Expr::FunctionCall {
        name: "Min".to_string(),
        args: result_args,
      })
    }
  }
}

/// Try to extract exact complex parts (re, im) from an expression as rational tuples (num, den).
/// Returns Some(((re_num, re_den), (im_num, im_den))) if the expression is a numeric complex number.
/// Handles patterns like: a + b*I, a - b*I, b*I, I, and plain reals.
pub fn try_extract_complex_exact(
  expr: &Expr,
) -> Option<((i128, i128), (i128, i128))> {
  use crate::syntax::BinaryOperator;
  match expr {
    // Pure imaginary: I
    Expr::Identifier(name) if name == "I" => Some(((0, 1), (1, 1))),
    // Real number (integer or rational)
    _ if expr_to_rational(expr).is_some() => {
      let (n, d) = expr_to_rational(expr).unwrap();
      Some(((n, d), (0, 1)))
    }
    // -I or -expr
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let ((rn, rd), (in_, id)) = try_extract_complex_exact(operand)?;
      Some(((-rn, rd), (-in_, id)))
    }
    // Times: n * I, or n * (m * I), etc.
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let lc = try_extract_complex_exact(left)?;
      let rc = try_extract_complex_exact(right)?;
      // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
      let ((an, ad_), (bn, bd_)) = lc;
      let ((cn, cd), (dn, dd)) = rc;
      // Real part: (an/ad_)*(cn/cd) - (bn/bd_)*(dn/dd)
      // = (an*cn)/(ad_*cd) - (bn*dn)/(bd_*dd)
      let re_n1 = an.checked_mul(cn)?;
      let re_d1 = ad_.checked_mul(cd)?;
      let re_n2 = bn.checked_mul(dn)?;
      let re_d2 = bd_.checked_mul(dd)?;
      // re = re_n1/re_d1 - re_n2/re_d2 = (re_n1*re_d2 - re_n2*re_d1) / (re_d1*re_d2)
      let re_num = re_n1.checked_mul(re_d2)? - re_n2.checked_mul(re_d1)?;
      let re_den = re_d1.checked_mul(re_d2)?;
      // Imag part: (an/ad_)*(dn/dd) + (bn/bd_)*(cn/cd)
      let im_n1 = an.checked_mul(dn)?;
      let im_d1 = ad_.checked_mul(dd)?;
      let im_n2 = bn.checked_mul(cn)?;
      let im_d2 = bd_.checked_mul(cd)?;
      let im_num = im_n1.checked_mul(im_d2)? + im_n2.checked_mul(im_d1)?;
      let im_den = im_d1.checked_mul(im_d2)?;
      Some(((re_num, re_den), (im_num, im_den)))
    }
    // Plus: a + b*I
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let ((an, ad_), (bn, bd_)) = try_extract_complex_exact(left)?;
      let ((cn, cd), (dn, dd)) = try_extract_complex_exact(right)?;
      // (a + bi) + (c + di) = (a+c) + (b+d)i
      let re_num = an.checked_mul(cd)? + cn.checked_mul(ad_)?;
      let re_den = ad_.checked_mul(cd)?;
      let im_num = bn.checked_mul(dd)? + dn.checked_mul(bd_)?;
      let im_den = bd_.checked_mul(dd)?;
      Some(((re_num, re_den), (im_num, im_den)))
    }
    // Minus: a - b*I
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let ((an, ad_), (bn, bd_)) = try_extract_complex_exact(left)?;
      let ((cn, cd), (dn, dd)) = try_extract_complex_exact(right)?;
      let re_num = an.checked_mul(cd)? - cn.checked_mul(ad_)?;
      let re_den = ad_.checked_mul(cd)?;
      let im_num = bn.checked_mul(dd)? - dn.checked_mul(bd_)?;
      let im_den = bd_.checked_mul(dd)?;
      Some(((re_num, re_den), (im_num, im_den)))
    }
    // FunctionCall Times[...] - flattened form
    Expr::FunctionCall { name, args } if name == "Times" => {
      if args.is_empty() {
        return None;
      }
      let mut result = try_extract_complex_exact(&args[0])?;
      for arg in &args[1..] {
        let rhs = try_extract_complex_exact(arg)?;
        let ((an, ad_), (bn, bd_)) = result;
        let ((cn, cd), (dn, dd)) = rhs;
        let re_n1 = an.checked_mul(cn)?;
        let re_d1 = ad_.checked_mul(cd)?;
        let re_n2 = bn.checked_mul(dn)?;
        let re_d2 = bd_.checked_mul(dd)?;
        let re_num = re_n1.checked_mul(re_d2)? - re_n2.checked_mul(re_d1)?;
        let re_den = re_d1.checked_mul(re_d2)?;
        let im_n1 = an.checked_mul(dn)?;
        let im_d1 = ad_.checked_mul(dd)?;
        let im_n2 = bn.checked_mul(cn)?;
        let im_d2 = bd_.checked_mul(cd)?;
        let im_num = im_n1.checked_mul(im_d2)? + im_n2.checked_mul(im_d1)?;
        let im_den = im_d1.checked_mul(im_d2)?;
        result = ((re_num, re_den), (im_num, im_den));
      }
      Some(result)
    }
    // FunctionCall Plus[...] - flattened form
    Expr::FunctionCall { name, args } if name == "Plus" => {
      if args.is_empty() {
        return None;
      }
      let mut result = try_extract_complex_exact(&args[0])?;
      for arg in &args[1..] {
        let rhs = try_extract_complex_exact(arg)?;
        let ((an, ad_), (bn, bd_)) = result;
        let ((cn, cd), (dn, dd)) = rhs;
        let re_num = an.checked_mul(cd)? + cn.checked_mul(ad_)?;
        let re_den = ad_.checked_mul(cd)?;
        let im_num = bn.checked_mul(dd)? + dn.checked_mul(bd_)?;
        let im_den = bd_.checked_mul(dd)?;
        result = ((re_num, re_den), (im_num, im_den));
      }
      Some(result)
    }
    _ => None,
  }
}

/// Try to extract float complex parts (re, im) from an expression.
/// Returns Some((re, im)) if the expression contains float components with I.
pub fn try_extract_complex_float(expr: &Expr) -> Option<(f64, f64)> {
  use crate::syntax::{BinaryOperator, UnaryOperator};
  match expr {
    Expr::Identifier(name) if name == "I" => Some((0.0, 1.0)),
    Expr::Integer(n) => Some((*n as f64, 0.0)),
    Expr::Real(f) => Some((*f, 0.0)),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (re, im) = try_extract_complex_float(operand)?;
      Some((-re, -im))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (a, b) = try_extract_complex_float(left)?;
      let (c, d) = try_extract_complex_float(right)?;
      Some((a * c - b * d, a * d + b * c))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let (a, b) = try_extract_complex_float(left)?;
      let (c, d) = try_extract_complex_float(right)?;
      Some((a + c, b + d))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (a, b) = try_extract_complex_float(left)?;
      let (c, d) = try_extract_complex_float(right)?;
      Some((a - c, b - d))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      if args.is_empty() {
        return None;
      }
      let mut result = try_extract_complex_float(&args[0])?;
      for arg in &args[1..] {
        let (c, d) = try_extract_complex_float(arg)?;
        let (a, b) = result;
        result = (a * c - b * d, a * d + b * c);
      }
      Some(result)
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      if args.is_empty() {
        return None;
      }
      let mut result = try_extract_complex_float(&args[0])?;
      for arg in &args[1..] {
        let (c, d) = try_extract_complex_float(arg)?;
        result = (result.0 + c, result.1 + d);
      }
      Some(result)
    }
    _ => None,
  }
}

/// Build a complex number expression from float parts.
fn build_complex_float_expr(re: f64, im: f64) -> Expr {
  let i_expr = Expr::Identifier("I".to_string());
  let im_abs = im.abs();

  if im == 0.0 {
    return num_to_expr(re);
  }

  let im_term = if im_abs == 1.0 {
    i_expr
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(Expr::Real(im_abs)),
      right: Box::new(i_expr),
    }
  };

  if re == 0.0 {
    if im > 0.0 {
      im_term
    } else {
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(im_term),
      }
    }
  } else {
    let re_expr = num_to_expr(re);
    if im > 0.0 {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(re_expr),
        right: Box::new(im_term),
      }
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(re_expr),
        right: Box::new(im_term),
      }
    }
  }
}

/// Build a complex number expression from exact rational parts.
/// Handles special cases: 0 + bi = bi, a + 0i = a, coefficient ±1 elision, etc.
fn build_complex_expr(
  re_num: i128,
  re_den: i128,
  im_num: i128,
  im_den: i128,
) -> Expr {
  let re = make_rational(re_num, re_den);
  let g_i = gcd(im_num.abs(), im_den.abs());
  let (in_s, id_s) = if im_den < 0 {
    (-im_num / g_i, -im_den / g_i)
  } else {
    (im_num / g_i, im_den / g_i)
  };

  let i_expr = Expr::Identifier("I".to_string());

  // Build the imaginary term with correct sign handling
  let im_abs_num = in_s.abs();
  let im_positive = in_s > 0;

  // Build |im| * I
  let im_term = if im_abs_num == 0 {
    return re; // Pure real
  } else if im_abs_num == 1 && id_s == 1 {
    i_expr
  } else {
    let coeff = make_rational(im_abs_num, id_s);
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(coeff),
      right: Box::new(i_expr),
    }
  };

  // Check if real part is zero
  let g_r = gcd(re_num.abs(), re_den.abs());
  let re_simplified = if re_den == 0 { re_num } else { re_num / g_r };
  if re_simplified == 0 {
    // Pure imaginary
    return if im_positive {
      im_term
    } else if im_abs_num == 1 && id_s == 1 {
      // -I
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(im_term),
      }
    } else {
      // -n*I: build Times[-n, I] directly
      let coeff = make_rational(-im_abs_num, id_s);
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(coeff),
        right: Box::new(Expr::Identifier("I".to_string())),
      }
    };
  }

  // Build re ± im*I
  if im_positive {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(re),
      right: Box::new(im_term),
    }
  } else {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(re),
      right: Box::new(im_term),
    }
  }
}

/// Try to extract complex parts as (re: f64, im: f64) from an expression.
/// Handles Real, Integer, I, Plus, Times, Minus, Complex, and FunctionCall variants.
pub fn try_extract_complex_f64(expr: &Expr) -> Option<(f64, f64)> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::Identifier(name) if name == "I" => Some((0.0, 1.0)),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (r, i) = try_extract_complex_f64(operand)?;
      Some((-r, -i))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (ar, ai) = try_extract_complex_f64(left)?;
      let (br, bi) = try_extract_complex_f64(right)?;
      Some((ar * br - ai * bi, ar * bi + ai * br))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let (ar, ai) = try_extract_complex_f64(left)?;
      let (br, bi) = try_extract_complex_f64(right)?;
      Some((ar + br, ai + bi))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let (ar, ai) = try_extract_complex_f64(left)?;
      let (br, bi) = try_extract_complex_f64(right)?;
      Some((ar - br, ai - bi))
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      let mut re = 0.0;
      let mut im = 0.0;
      for arg in args {
        let (r, i) = try_extract_complex_f64(arg)?;
        re += r;
        im += i;
      }
      Some((re, im))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut re = 1.0;
      let mut im = 0.0;
      for arg in args {
        let (r, i) = try_extract_complex_f64(arg)?;
        let new_re = re * r - im * i;
        let new_im = re * i + im * r;
        re = new_re;
        im = new_im;
      }
      Some((re, im))
    }
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      let r = try_eval_to_f64(&args[0])?;
      let i = try_eval_to_f64(&args[1])?;
      Some((r, i))
    }
    _ => {
      // Try as pure real
      let v = try_eval_to_f64(expr)?;
      Some((v, 0.0))
    }
  }
}

/// Check if an expression contains Infinity (used for Abs)
fn contains_infinity(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name)
      if name == "Infinity" || name == "ComplexInfinity" =>
    {
      true
    }
    Expr::UnaryOp { operand, .. } => contains_infinity(operand),
    Expr::BinaryOp { left, right, .. } => {
      contains_infinity(left) || contains_infinity(right)
    }
    Expr::FunctionCall { name, args } if name == "Times" || name == "Plus" => {
      args.iter().any(contains_infinity)
    }
    Expr::FunctionCall { name, args }
      if name == "DirectedInfinity" && !args.is_empty() =>
    {
      true
    }
    _ => false,
  }
}

/// Abs[x] - Absolute value
pub fn abs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Abs expects exactly 1 argument".into(),
    ));
  }
  // Handle any expression containing Infinity → Infinity
  if contains_infinity(&args[0]) {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    return Ok(num_to_expr(n.abs()));
  }
  // Handle exact complex numbers: Abs[a + b*I] = Sqrt[a^2 + b^2]
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(&args[0]) {
    let g_r = gcd(rn, rd);
    let (rn, rd) = if rd < 0 {
      (-rn / g_r, -rd / g_r)
    } else {
      (rn / g_r, rd / g_r)
    };
    let g_i = gcd(in_, id);
    let (in_, id) = if id < 0 {
      (-in_ / g_i, -id / g_i)
    } else {
      (in_ / g_i, id / g_i)
    };
    if in_ == 0 {
      // Pure real
      return Ok(make_rational(rn.abs(), rd));
    }
    // Abs = Sqrt[(rn/rd)^2 + (in_/id)^2]
    // = Sqrt[(rn^2 * id^2 + in_^2 * rd^2) / (rd^2 * id^2)]
    if let (Some(num2), Some(den2)) = (
      rn.checked_mul(rn)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b)))
        .and_then(|a| {
          in_
            .checked_mul(in_)
            .and_then(|c| rd.checked_mul(rd).and_then(|d| c.checked_mul(d)))
            .and_then(|b| a.checked_add(b))
        }),
      rd.checked_mul(rd)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b))),
    ) {
      // Check if num2 is a perfect square
      let num_sqrt = (num2 as f64).sqrt().round() as i128;
      let den_sqrt = (den2 as f64).sqrt().round() as i128;
      if num_sqrt * num_sqrt == num2 && den_sqrt * den_sqrt == den2 {
        return Ok(make_rational(num_sqrt, den_sqrt));
      }
      // Return Sqrt[num2/den2]
      return Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: vec![make_rational(num2, den2)],
      });
    }
  }
  // Handle floating-point complex numbers: Abs[3.0 + I] = sqrt(10)
  if let Some((re, im)) = try_extract_complex_f64(&args[0])
    && im != 0.0
  {
    return Ok(num_to_expr((re * re + im * im).sqrt()));
  }
  Ok(Expr::FunctionCall {
    name: "Abs".to_string(),
    args: args.to_vec(),
  })
}

/// Sign[x] - Sign of a number (-1, 0, or 1)
/// For complex z: Sign[z] = z / Abs[z]
pub fn sign_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sign expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    return Ok(Expr::Integer(if n > 0.0 {
      1
    } else if n < 0.0 {
      -1
    } else {
      0
    }));
  }
  // Handle complex numbers: Sign[a + b*I] = (a + b*I) / Abs[a + b*I]
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(&args[0]) {
    let g_r = gcd(rn, rd);
    let (rn, rd) = if rd < 0 {
      (-rn / g_r, -rd / g_r)
    } else {
      (rn / g_r, rd / g_r)
    };
    let g_i = gcd(in_, id);
    let (in_, id) = if id < 0 {
      (-in_ / g_i, -id / g_i)
    } else {
      (in_ / g_i, id / g_i)
    };
    if in_ == 0 {
      // Pure real - already handled above by try_eval_to_f64
      return Ok(Expr::Integer(if rn > 0 {
        1
      } else if rn < 0 {
        -1
      } else {
        0
      }));
    }
    // Compute |z|^2 = (rn/rd)^2 + (in_/id)^2 = (rn^2*id^2 + in_^2*rd^2) / (rd^2*id^2)
    if let (Some(abs2_num), Some(abs2_den)) = (
      rn.checked_mul(rn)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b)))
        .and_then(|a| {
          in_
            .checked_mul(in_)
            .and_then(|c| rd.checked_mul(rd).and_then(|d| c.checked_mul(d)))
            .and_then(|b| a.checked_add(b))
        }),
      rd.checked_mul(rd)
        .and_then(|a| id.checked_mul(id).and_then(|b| a.checked_mul(b))),
    ) {
      // Simplify |z|^2 fraction
      let g_abs = gcd(abs2_num.abs(), abs2_den.abs());
      let (abs2_n, abs2_d) = (abs2_num / g_abs, abs2_den / g_abs);

      // Check if |z|^2 is a perfect square (so |z| is rational)
      let abs2_sqrt = (abs2_n as f64).sqrt().round() as i128;
      let den_sqrt = (abs2_d as f64).sqrt().round() as i128;
      if abs2_sqrt * abs2_sqrt == abs2_n
        && den_sqrt * den_sqrt == abs2_d
        && abs2_sqrt != 0
      {
        // |z| = abs2_sqrt / den_sqrt
        // Sign = z / |z| = (rn/rd + (in_/id)*I) * (den_sqrt/abs2_sqrt)
        // Real part: (rn * den_sqrt) / (rd * abs2_sqrt)
        // Imag part: (in_ * den_sqrt) / (id * abs2_sqrt)
        let sign_re_num = rn.checked_mul(den_sqrt);
        let sign_re_den = rd.checked_mul(abs2_sqrt);
        let sign_im_num = in_.checked_mul(den_sqrt);
        let sign_im_den = id.checked_mul(abs2_sqrt);
        if let (Some(srn), Some(srd), Some(sin), Some(sid)) =
          (sign_re_num, sign_re_den, sign_im_num, sign_im_den)
        {
          return Ok(build_complex_expr(srn, srd, sin, sid));
        }
      } else {
        // |z| is irrational: Sign[z] = z / Sqrt[|z|^2]
        // = (a + b*I) / Sqrt[|z|^2]
        // Build (a + b*I) * (1/Sqrt[|z|^2])
        let z_expr = build_complex_expr(rn, rd, in_, id);
        let abs_expr = Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![make_rational(abs2_n, abs2_d)],
        };
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(z_expr),
          right: Box::new(abs_expr),
        });
      }
    }
  }
  Ok(Expr::FunctionCall {
    name: "Sign".to_string(),
    args: args.to_vec(),
  })
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
    // Sqrt[Rational[a, b]] — simplify by extracting perfect square factors
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&rargs[0], &rargs[1])
        && *n >= 0
        && *d > 0
      {
        // Extract perfect square factors from numerator and denominator
        let extract_square_factor = |val: u64| -> (u64, u64) {
          let mut outside = 1u64;
          let mut inside = val;
          let mut factor = 2u64;
          while factor * factor <= inside {
            while inside.is_multiple_of(factor * factor) {
              outside *= factor;
              inside /= factor * factor;
            }
            factor += 1;
          }
          (outside, inside)
        };
        let (n_out, n_in) = extract_square_factor(*n as u64);
        let (d_out, d_in) = extract_square_factor(*d as u64);
        // Result: (n_out / d_out) * Sqrt[n_in / d_in]
        if n_in == 1 && d_in == 1 {
          // Perfect square: just a rational
          return Ok(make_rational(n_out as i128, d_out as i128));
        }
        if d_in == 1 {
          // Result: (n_out / d_out) * Sqrt[n_in]
          let sqrt_part = Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(n_in as i128)],
          };
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        } else if n_in == 1 {
          // Result: n_out / (d_out * Sqrt[d_in])
          let denom = if d_out == 1 {
            Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![Expr::Integer(d_in as i128)],
            }
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(d_out as i128),
                Expr::FunctionCall {
                  name: "Sqrt".to_string(),
                  args: vec![Expr::Integer(d_in as i128)],
                },
              ],
            }
          };
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Divide,
            left: Box::new(Expr::Integer(n_out as i128)),
            right: Box::new(denom),
          });
        } else {
          // General case: (n_out / d_out) * Sqrt[n_in / d_in]
          let sqrt_part = Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![make_rational(n_in as i128, d_in as i128)],
          };
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        }
      }
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    }
    // Sqrt[-n] for negative integer → I * Sqrt[n]
    Expr::Integer(n) if *n < 0 => {
      let pos = -*n;
      let sqrt_pos = sqrt_ast(&[Expr::Integer(pos)])?;
      times_ast(&[Expr::Identifier("I".to_string()), sqrt_pos])
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
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Floor expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return floor_ceil_two_arg(&args[0], &args[1], true);
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

/// Ceiling[x] or Ceiling[x, a] - Ceiling function
pub fn ceiling_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Ceiling expects 1 or 2 arguments".into(),
    ));
  }
  if args.len() == 2 {
    return floor_ceil_two_arg(&args[0], &args[1], false);
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

/// Helper for Floor[x, a] and Ceiling[x, a]
/// Floor[x, a] = a * Floor[x/a], Ceiling[x, a] = a * Ceiling[x/a]
fn floor_ceil_two_arg(
  x: &Expr,
  a: &Expr,
  is_floor: bool,
) -> Result<Expr, InterpreterError> {
  // Try rational arithmetic first for exact results
  if let (Some(xn), Some(xd), Some(an), Some(ad)) = (
    expr_numerator(x),
    expr_denominator(x),
    expr_numerator(a),
    expr_denominator(a),
  ) {
    if an == 0 {
      // Floor[x, 0] or Ceiling[x, 0] → Indeterminate
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // x/a = (xn * ad) / (xd * an)
    let num = xn * ad;
    let den = xd * an;
    // Floor of rational num/den
    let floored = if is_floor {
      rational_floor(num, den)
    } else {
      rational_ceil(num, den)
    };
    // Result = a * floored = (an/ad) * floored = (an * floored) / ad
    let res_num = an * floored;
    if ad == 1 {
      return Ok(Expr::Integer(res_num));
    }
    let g = gcd_i128(res_num.abs(), ad.abs());
    let rn = res_num / g;
    let rd = ad / g;
    if rd == 1 {
      return Ok(Expr::Integer(rn));
    }
    return Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(rn), Expr::Integer(rd)],
    });
  }
  // Fall back to floating point
  if let (Some(xf), Some(af)) = (try_eval_to_f64(x), try_eval_to_f64(a)) {
    if af == 0.0 {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = if is_floor {
      (xf / af).floor() * af
    } else {
      (xf / af).ceil() * af
    };
    // Result type follows `a`: if `a` is Real, result is Real; otherwise Integer
    let a_is_float = matches!(a, Expr::Real(_));
    if a_is_float {
      return Ok(Expr::Real(result));
    }
    if result.fract() == 0.0 && result.abs() < i128::MAX as f64 {
      return Ok(Expr::Integer(result as i128));
    }
    return Ok(Expr::Real(result));
  }
  let name = if is_floor { "Floor" } else { "Ceiling" };
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: vec![x.clone(), a.clone()],
  })
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
    let eval_a = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
    let eval_x = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

    // Check if a is a Rational (n/d)
    if let Expr::FunctionCall { name, args: rargs } = &eval_a
      && name == "Rational"
      && rargs.len() == 2
      && let (Some(x_val), Some(a_val)) =
        (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_a))
      && a_val != 0.0
    {
      let n = (x_val / a_val).round() as i128;
      // Return n * (num/den) as a rational
      if let (Some(num), Some(den)) =
        (expr_to_i128(&rargs[0]), expr_to_i128(&rargs[1]))
      {
        return Ok(make_rational_pub(n * num, den));
      }
    }

    // Check if a is symbolic (like Pi) — return n * a
    if let (Some(x_val), Some(a_val)) =
      (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_a))
    {
      if a_val == 0.0 {
        return Ok(eval_x);
      }
      let n = (x_val / a_val).round() as i128;
      // If a is not a plain number, return n * a symbolically
      let a_is_real = matches!(&eval_a, Expr::Real(_));
      let a_is_int = matches!(&eval_a, Expr::Integer(_));
      if !a_is_real && !a_is_int {
        // Symbolic: return n * a
        return crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(n)),
          right: Box::new(eval_a),
        });
      }
      let rounded = n as f64 * a_val;
      // When the step a is Real, result should be Real
      let x_is_real = matches!(&eval_x, Expr::Real(_));
      if (a_is_real || x_is_real)
        && rounded.fract() == 0.0
        && rounded.abs() < i128::MAX as f64
      {
        return Ok(Expr::Real(rounded));
      }
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

/// Mod[m, n] - Modulus, or Mod[m, n, d] with offset
pub fn mod_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "Mod expects 2 or 3 arguments; {} given",
      args.len()
    )));
  }

  // 3-argument form: Mod[m, n, d] = m - n * Floor[(m - d) / n]
  if args.len() == 3 {
    return mod3_ast(&args[0], &args[1], &args[2]);
  }

  // 2-argument form: Mod[m, n]
  mod2_ast(&args[0], &args[1])
}

/// Helper to extract (numerator, denominator) from Integer or Rational
fn try_as_rational(expr: &Expr) -> Option<(i128, i128)> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some((*n, *d))
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Mod[m, n] - 2-argument form
fn mod2_ast(m: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
  // Try exact rational arithmetic first
  if let (Some((mn, md)), Some((nn, nd))) =
    (try_as_rational(m), try_as_rational(n))
  {
    if nn == 0 && nd != 0 {
      // Mod[m, 0] => Indeterminate
      eprintln!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      );
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // Convert to common denominator: m = mn/md, n = nn/nd
    // Mod[mn/md, nn/nd] = Mod[mn*nd, nn*md] / (md*nd)
    let num_m = mn * nd;
    let num_n = nn * md;
    let common_d = md * nd;
    // Wolfram Mod: result = ((num_m % num_n) + num_n) % num_n, then divide by common_d
    let rem = ((num_m % num_n) + num_n) % num_n;
    return Ok(make_rational(rem, common_d));
  }

  // Float fallback
  if let (Some(a), Some(b)) = (try_eval_to_f64(m), try_eval_to_f64(n)) {
    if b == 0.0 {
      eprintln!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      );
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = ((a % b) + b) % b;
    return Ok(num_to_expr(result));
  }

  // Symbolic
  Ok(Expr::FunctionCall {
    name: "Mod".to_string(),
    args: vec![m.clone(), n.clone()],
  })
}

/// Mod[m, n, d] - 3-argument form: m - n * Floor[(m - d) / n]
fn mod3_ast(m: &Expr, n: &Expr, d: &Expr) -> Result<Expr, InterpreterError> {
  // Try exact rational arithmetic
  if let (Some((mn, md)), Some((nn, nd)), Some((dn, dd))) =
    (try_as_rational(m), try_as_rational(n), try_as_rational(d))
  {
    if nn == 0 && nd != 0 {
      eprintln!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      );
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // m - d = (mn*dd - dn*md) / (md*dd)
    let diff_n = mn * dd - dn * md;
    let diff_d = md * dd;
    // (m - d) / n = (diff_n * nd) / (diff_d * nn)
    let quot_n = diff_n * nd;
    let quot_d = diff_d * nn;
    // Floor of quot_n / quot_d
    let fl = floor_div(quot_n, quot_d);
    // result = m - n * fl = mn/md - nn/nd * fl = (mn*nd - nn*md*fl) / (md*nd)
    let res_n = mn * nd - nn * md * fl;
    let res_d = md * nd;
    return Ok(make_rational(res_n, res_d));
  }

  // Float fallback
  if let (Some(a), Some(b), Some(c)) =
    (try_eval_to_f64(m), try_eval_to_f64(n), try_eval_to_f64(d))
  {
    if b == 0.0 {
      eprintln!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      );
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    let result = a - b * ((a - c) / b).floor();
    return Ok(num_to_expr(result));
  }

  // Symbolic
  Ok(Expr::FunctionCall {
    name: "Mod".to_string(),
    args: vec![m.clone(), n.clone(), d.clone()],
  })
}

/// Integer floor division: floor(a / b)
fn floor_div(a: i128, b: i128) -> i128 {
  if b == 0 {
    return 0;
  }
  let d = a / b;
  let r = a % b;
  // Adjust if remainder has opposite sign to divisor
  if r != 0 && (r ^ b) < 0 { d - 1 } else { d }
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
        Ok(Expr::Integer(floor_div(*a, *b)))
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

fn bigint_gcd(a: BigInt, b: BigInt) -> BigInt {
  use num_traits::Zero;
  let (mut a, mut b) = (a.abs(), b.abs());
  while !b.is_zero() {
    let t = b.clone();
    b = &a % &b;
    a = t;
  }
  a
}

/// GCD[a, b, ...] - Greatest common divisor
pub fn gcd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let mut result: Option<BigInt> = None;
  for arg in args {
    let val = match arg {
      Expr::Integer(n) => BigInt::from(*n),
      Expr::BigInteger(n) => n.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "GCD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    result = Some(match result {
      Some(r) => bigint_gcd(r, val),
      None => val.abs(),
    });
  }

  Ok(bigint_to_expr(result.unwrap_or_else(|| BigInt::from(0))))
}

/// LCM[a, b, ...] - Least common multiple
pub fn lcm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }

  let mut result: Option<BigInt> = None;
  for arg in args {
    let val = match arg {
      Expr::Integer(n) => BigInt::from(*n),
      Expr::BigInteger(n) => n.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "LCM".to_string(),
          args: args.to_vec(),
        });
      }
    };
    result = Some(match result {
      Some(r) => {
        use num_traits::Zero;
        if r.is_zero() || val.is_zero() {
          BigInt::from(0)
        } else {
          let g = bigint_gcd(r.clone(), val.clone());
          (r.abs() / g) * val.abs()
        }
      }
      None => val.abs(),
    });
  }

  Ok(bigint_to_expr(result.unwrap_or_else(|| BigInt::from(1))))
}

/// Total[list] - Sum of all elements in a list
/// Total[list, n] - Sum across levels 1 through n
/// Total[list, {n}] - Sum at exactly level n
pub fn total_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Total expects 1 or 2 arguments".into(),
    ));
  }

  // Parse level spec from second argument
  let level_spec = if args.len() == 2 {
    match &args[1] {
      // Total[list, {n}] - exact level n
      Expr::List(items) if items.len() == 1 => {
        if let Some(n) = expr_to_num(&items[0]) {
          TotalLevelSpec::Exact(n as usize)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec(),
          });
        }
      }
      // Total[list, Infinity]
      Expr::Identifier(s) if s == "Infinity" => {
        TotalLevelSpec::Through(usize::MAX)
      }
      // Total[list, n] - through level n
      _ => {
        if let Some(n) = expr_to_num(&args[1]) {
          TotalLevelSpec::Through(n as usize)
        } else {
          return Ok(Expr::FunctionCall {
            name: "Total".to_string(),
            args: args.to_vec(),
          });
        }
      }
    }
  } else {
    TotalLevelSpec::Through(1)
  };

  match &args[0] {
    Expr::List(_) => total_with_level(&args[0], &level_spec),
    // Total[x] for non-list returns x
    other => Ok(other.clone()),
  }
}

enum TotalLevelSpec {
  Through(usize), // sum levels 1..=n
  Exact(usize),   // sum at exactly level n
}

/// Sum a list at level 1 using Plus (Apply[Plus, list])
/// Handles nested lists by recursively adding element-wise.
fn total_sum_level1(items: &[Expr]) -> Result<Expr, InterpreterError> {
  if items.is_empty() {
    return Ok(Expr::Integer(0));
  }
  let mut acc = items[0].clone();
  for item in &items[1..] {
    acc = add_exprs_recursive(&acc, item)?;
  }
  Ok(acc)
}

/// Recursively add two expressions, threading over lists element-wise
fn add_exprs_recursive(a: &Expr, b: &Expr) -> Result<Expr, InterpreterError> {
  match (a, b) {
    (Expr::List(la), Expr::List(lb)) if la.len() == lb.len() => {
      let results: Result<Vec<Expr>, _> = la
        .iter()
        .zip(lb.iter())
        .map(|(x, y)| add_exprs_recursive(x, y))
        .collect();
      Ok(Expr::List(results?))
    }
    _ => plus_ast(&[a.clone(), b.clone()]),
  }
}

/// Recursively apply Total with level spec
fn total_with_level(
  expr: &Expr,
  level_spec: &TotalLevelSpec,
) -> Result<Expr, InterpreterError> {
  match level_spec {
    TotalLevelSpec::Through(n) => total_through_level(expr, *n),
    TotalLevelSpec::Exact(n) => total_at_exact_level(expr, *n),
  }
}

/// Total[list, n] - sum across levels 1 through n
/// Level 0 means no summing, level 1 means Apply[Plus, list], etc.
fn total_through_level(
  expr: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  if n == 0 {
    return Ok(expr.clone());
  }
  match expr {
    Expr::List(items) => {
      // First, recursively process sublists for levels 2..n
      if n > 1 {
        let processed: Vec<Expr> = items
          .iter()
          .map(|item| total_through_level(item, n - 1))
          .collect::<Result<Vec<_>, _>>()?;
        total_sum_level1(&processed)
      } else {
        total_sum_level1(items)
      }
    }
    _ => Ok(expr.clone()),
  }
}

/// Total[list, {n}] - sum at exactly level n
/// Level 1 = sum the outermost list, level 2 = sum each sublist, etc.
fn total_at_exact_level(
  expr: &Expr,
  n: usize,
) -> Result<Expr, InterpreterError> {
  if n <= 1 {
    // Sum at this level
    match expr {
      Expr::List(items) => total_sum_level1(items),
      _ => Ok(expr.clone()),
    }
  } else {
    // Recurse into sublists, summing at deeper level
    match expr {
      Expr::List(items) => {
        let processed: Vec<Expr> = items
          .iter()
          .map(|item| total_at_exact_level(item, n - 1))
          .collect::<Result<Vec<_>, _>>()?;
        Ok(Expr::List(processed))
      }
      _ => Ok(expr.clone()),
    }
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
        // Check for list-of-lists (matrix) → compute column-wise mean
        if items.iter().all(|item| matches!(item, Expr::List(_))) {
          return mean_columnwise(items);
        }
        // Non-numeric elements - compute symbolically: Total[list] / Length[list]
        // Evaluate the sum first, then wrap in division (don't distribute)
        let sum_expr = Expr::FunctionCall {
          name: "Plus".to_string(),
          args: items.clone(),
        };
        let evaluated_sum = crate::evaluator::evaluate_expr_to_expr(&sum_expr)?;
        let n = items.len() as i128;
        // Use BinaryOp::Divide to represent (sum) / n without distributing
        Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(evaluated_sum),
          right: Box::new(Expr::Integer(n)),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Mean of columns in a list-of-lists (matrix)
fn mean_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&Vec<Expr>> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Mean".to_string(),
      args: vec![Expr::List(rows.to_vec())],
    });
  }
  let ncols = row_vecs[0].len();
  let nrows = row_vecs.len();
  let mut col_means = Vec::new();
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    let mean_result = mean_ast(&[Expr::List(col_items)])?;
    col_means.push(mean_result);
  }
  let _ = nrows; // used indirectly through mean_ast
  Ok(Expr::List(col_means))
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
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < 0 {
      return Err(InterpreterError::EvaluationError(
        "Factorial: argument must be non-negative".into(),
      ));
    }
    let mut result = BigInt::from(1);
    for i in 2..=n {
      result *= i;
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Factorial".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Factorial2[n] - Double factorial: n!! = n * (n-2) * (n-4) * ...
pub fn factorial2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Factorial2 expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < -1 {
      // Negative odd integers: n!! = (n+2)!! / (n+2)
      // (-1)!! = 1, (-3)!! = -1, (-5)!! = 1/3, (-7)!! = -1/15, ...
      if n % 2 != 0 {
        // Compute by working from -1 down
        let mut numer: i128 = 1;
        let mut denom: i128 = 1;
        let mut k = -1i128;
        while k > n {
          k -= 2;
          // (k)!! = (k+2)!! / (k+2)
          denom *= k + 2;
          // Simplify
          let g = gcd_i128(numer.abs(), denom.abs());
          numer /= g;
          denom /= g;
        }
        if denom < 0 {
          numer = -numer;
          denom = -denom;
        }
        if denom == 1 {
          return Ok(Expr::Integer(numer));
        }
        return Ok(make_rational(numer, denom));
      }
      return Ok(Expr::FunctionCall {
        name: "Factorial2".to_string(),
        args: args.to_vec(),
      });
    }
    if n == -1 || n == 0 {
      return Ok(Expr::Integer(1));
    }
    let mut result = BigInt::from(1);
    let mut i = n;
    while i >= 2 {
      result *= i;
      i -= 2;
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Factorial2".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Subfactorial[n] - Count of derangements: !n = n! * Sum[(-1)^k/k!, {k, 0, n}]
pub fn subfactorial_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Subfactorial expects exactly 1 argument".into(),
    ));
  }
  if let Some(n) = expr_to_i128(&args[0]) {
    if n < 0 {
      return Ok(Expr::FunctionCall {
        name: "Subfactorial".to_string(),
        args: args.to_vec(),
      });
    }
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Use recurrence: !n = (n-1) * (!(n-1) + !(n-2))
    let mut prev2 = BigInt::from(1); // !0 = 1
    let mut prev1 = BigInt::from(0); // !1 = 0
    if n == 1 {
      return Ok(Expr::Integer(0));
    }
    for i in 2..=n {
      let current = BigInt::from(i - 1) * (&prev1 + &prev2);
      prev2 = prev1;
      prev1 = current;
    }
    Ok(bigint_to_expr(prev1))
  } else {
    Ok(Expr::FunctionCall {
      name: "Subfactorial".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Pochhammer[a, n] - Rising factorial (Pochhammer symbol): a * (a+1) * ... * (a+n-1)
pub fn pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Pochhammer expects exactly 2 arguments".into(),
    ));
  }
  if let (Some(a), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      return Ok(Expr::FunctionCall {
        name: "Pochhammer".to_string(),
        args: args.to_vec(),
      });
    }
    let mut result = BigInt::from(1);
    for i in 0..n {
      result *= BigInt::from(a + i);
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Pochhammer".to_string(),
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
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n <= 0 {
        // Gamma has poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Gamma[n] = (n-1)! for positive integers
      let mut result = BigInt::from(1);
      for i in 2..n {
        result *= i;
      }
      Ok(bigint_to_expr(result))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f <= 0.0 && f.fract() == 0.0 {
        // Poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Use Stirling's approximation via the standard library's tgamma equivalent
      // Rust doesn't have tgamma in std, but we can compute via the Lanczos approximation
      let result = gamma_fn(f);
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
  if args.len() == 2 {
    // N[expr, precision] — arbitrary-precision evaluation
    let precision = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "N: precision must be a positive integer".into(),
        ));
      }
    };
    return n_eval_arbitrary(&args[0], precision);
  }
  n_eval(&args[0])
}

/// Recursively convert an expression to numeric (Real) form
fn n_eval(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok(Expr::Real(*n as f64)),
    Expr::Real(_) => Ok(expr.clone()),
    Expr::BigFloat(_, _) => Ok(expr.clone()),
    Expr::List(items) => {
      let results: Result<Vec<Expr>, _> = items.iter().map(n_eval).collect();
      Ok(Expr::List(results?))
    }
    Expr::BigInteger(n) => Ok(Expr::Real(
      n.to_string().parse::<f64>().unwrap_or(f64::INFINITY),
    )),
    Expr::FunctionCall { name, args } => {
      // First try to evaluate the whole expression to a number
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      // Otherwise, recursively apply N to arguments
      let new_args: Result<Vec<Expr>, _> = args.iter().map(n_eval).collect();
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: new_args?,
      })
    }
    Expr::BinaryOp { op, left, right } => {
      if let Some(v) = try_eval_to_f64(expr) {
        return Ok(Expr::Real(v));
      }
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(n_eval(left)?),
        right: Box::new(n_eval(right)?),
      })
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

/// Convert decimal digit precision to the nominal bit-precision.
/// `astro-float` internally rounds up to 64-bit word boundaries.
/// Minimum 128 bits (2 words) to avoid precision issues with small values.
fn nominal_bits(precision: usize) -> usize {
  // Compute the minimum bits for the requested decimal precision, then
  // round up to the next 64-bit word boundary. Add one extra word of
  // guard bits to match Wolfram's output behavior (which displays all
  // digits from the full word-aligned precision, giving slightly more
  // digits than requested).
  let base_bits =
    (precision as f64 * std::f64::consts::LOG2_10).ceil() as usize;
  // Round up to next word boundary, then add one extra word for guard bits
  let bits = ((base_bits + 63) & !63) + 64;
  bits.max(128)
}

/// Convert a BigFloat to a decimal string using num-bigint for the base
/// conversion, avoiding astro-float's `format()` which panics on wasm32.
/// If `max_digits` is Some(n), truncate the output to at most n significant digits.
fn bigfloat_to_string(
  bf: &astro_float::BigFloat,
  max_digits: Option<usize>,
) -> Result<String, InterpreterError> {
  use num_bigint::BigUint;
  use num_traits::Zero;

  // Extract raw parts: mantissa words, significant bits, sign, exponent
  let (words, sig_bits, sign, exponent, _inexact) =
    bf.as_raw_parts().ok_or_else(|| {
      InterpreterError::EvaluationError("N: cannot format NaN or Inf".into())
    })?;

  if sig_bits == 0 || words.iter().all(|&w| w == 0) {
    return Ok("0.".to_string());
  }

  // Build a BigUint from the mantissa words (little-endian u64 words).
  let mantissa = BigUint::from_bytes_le(
    &words
      .iter()
      .flat_map(|w| w.to_le_bytes())
      .collect::<Vec<u8>>(),
  );

  // The value is: sign * mantissa * 2^(exponent - mantissa_bit_length)
  // where mantissa_bit_length = words.len() * 64.
  let mantissa_bits = words.len() * 64;
  let shift = exponent as i64 - mantissa_bits as i64;

  // Compute target digits: use max_digits if given (from requested precision),
  // otherwise derive from the mantissa bit count.
  let target_digits = if let Some(d) = max_digits {
    // Add a few extra digits for rounding, we'll truncate the output later
    d + 5
  } else {
    (mantissa_bits as f64 / std::f64::consts::LOG2_10).ceil() as usize + 2
  };

  // Strategy: compute mantissa * 10^target_digits * 2^shift, then divide
  // by 10^target_digits later to place the decimal point.
  //
  // If shift >= 0: integer_part = mantissa << shift
  // If shift < 0: we need to compute mantissa * 10^target_digits >> (-shift)
  //   to get target_digits of fractional precision.

  let (int_digits, decimal_exp) = if shift >= 0 {
    // Value = mantissa * 2^shift (an integer, possibly very large)
    let int_val = &mantissa << (shift as u64);
    let s = int_val.to_string();
    let len = s.len();
    // decimal_exp = number of digits in integer part
    (s, len as i64)
  } else {
    // Value = mantissa / 2^(-shift)
    // Multiply mantissa by 10^target_digits first to preserve fractional digits
    let neg_shift = (-shift) as u64;
    let scale = BigUint::from(10u32).pow(target_digits as u32);
    let scaled = &mantissa * &scale;

    // Now divide by 2^(-shift) with rounding
    let divisor = BigUint::from(1u32) << neg_shift;
    let result = (&scaled + (&divisor >> 1u32)) / &divisor;

    if result.is_zero() {
      return Ok("0.".to_string());
    }

    let s = result.to_string();
    // The decimal point should be placed target_digits from the right
    let decimal_exp = s.len() as i64 - target_digits as i64;
    (s, decimal_exp)
  };

  // Build the decimal string with the decimal point
  let is_negative = sign.is_negative();
  let prefix = if is_negative { "-" } else { "" };

  // Truncate significant digits if max_digits is specified.
  // This removes noise from guard bits.
  let int_digits = if let Some(max_d) = max_digits {
    // Count significant digits (excluding leading zeros for numbers < 1)
    let sig_start = if decimal_exp <= 0 {
      // For 0.00xxx, all digits are significant
      0
    } else {
      0
    };
    let sig_count = int_digits.len() - sig_start;
    if sig_count > max_d {
      int_digits[..sig_start + max_d].to_string()
    } else {
      int_digits
    }
  } else {
    int_digits
  };
  let digits = int_digits.as_bytes();

  if decimal_exp <= 0 {
    // Number like 0.000xxxx
    let zeros = (-decimal_exp) as usize;
    let trimmed = int_digits.trim_end_matches('0');
    if trimmed.is_empty() {
      Ok(format!("{}0.", prefix))
    } else {
      let frac: String = format!("{}{}", "0".repeat(zeros), trimmed);
      let frac = frac.trim_end_matches('0');
      if frac.is_empty() {
        Ok(format!("{}0.", prefix))
      } else {
        Ok(format!("{}0.{}", prefix, frac))
      }
    }
  } else {
    let dp = decimal_exp as usize;
    if dp >= digits.len() {
      // All digits are in the integer part
      let padded = format!("{}{}", int_digits, "0".repeat(dp - digits.len()));
      Ok(format!("{}{}.", prefix, padded))
    } else {
      // Some digits before decimal, some after
      let int_part = &int_digits[..dp];
      let frac_part = int_digits[dp..].trim_end_matches('0');
      if frac_part.is_empty() {
        Ok(format!("{}{}.", prefix, int_part))
      } else {
        Ok(format!("{}{}.{}", prefix, int_part, frac_part))
      }
    }
  }
}

/// N[expr, precision] — arbitrary-precision numeric evaluation using BigFloat
fn n_eval_arbitrary(
  expr: &Expr,
  precision: usize,
) -> Result<Expr, InterpreterError> {
  // Handle List recursively at the Expr level
  if let Expr::List(items) = expr {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|e| n_eval_arbitrary(e, precision))
      .collect();
    return Ok(Expr::List(results?));
  }

  use astro_float::{Consts, RoundingMode};

  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;

  // Compute at the nominal bit-precision. astro-float internally
  // rounds up to 64-bit word boundaries, giving us the correct
  // number of output digits.
  let bits = nominal_bits(precision);
  let result = expr_to_bigfloat(expr, bits, rm, &mut cc)?;

  let decimal = bigfloat_to_string(&result, None)?;

  Ok(Expr::BigFloat(decimal, precision))
}

/// Recursively convert an Expr to a BigFloat with the given precision.
fn expr_to_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<astro_float::BigFloat, InterpreterError> {
  use crate::syntax::BinaryOperator;
  use astro_float::BigFloat;

  match expr {
    Expr::Integer(n) => Ok(BigFloat::from_i128(*n, bits)),
    Expr::BigInteger(n) => {
      // Convert BigInt to BigFloat by parsing its decimal string
      let s = n.to_string();
      Ok(BigFloat::parse(&s, astro_float::Radix::Dec, bits, rm, cc))
    }
    Expr::Real(f) => Ok(BigFloat::from_f64(*f, bits)),
    Expr::BigFloat(digits, _) => Ok(BigFloat::parse(
      digits,
      astro_float::Radix::Dec,
      bits,
      rm,
      cc,
    )),
    Expr::Constant(name) => match name.as_str() {
      "Pi" | "-Pi" => {
        let pi = cc.pi(bits, rm);
        if name == "-Pi" { Ok(pi.neg()) } else { Ok(pi) }
      }
      "E" => Ok(cc.e(bits, rm)),
      "Degree" => {
        let pi = cc.pi(bits, rm);
        let d180 = BigFloat::from_i32(180, bits);
        Ok(pi.div(&d180, bits, rm))
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "N: cannot evaluate constant {} to arbitrary precision",
        name
      ))),
    },
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let val = expr_to_bigfloat(operand, bits, rm, cc)?;
      Ok(val.neg())
    }
    Expr::BinaryOp { op, left, right } => {
      let l = expr_to_bigfloat(left, bits, rm, cc)?;
      let r = expr_to_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => Ok(l.add(&r, bits, rm)),
        BinaryOperator::Minus => Ok(l.sub(&r, bits, rm)),
        BinaryOperator::Times => Ok(l.mul(&r, bits, rm)),
        BinaryOperator::Divide => Ok(l.div(&r, bits, rm)),
        BinaryOperator::Power => Ok(l.pow(&r, bits, rm, cc)),
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for arbitrary precision".into(),
        )),
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Rational" if args.len() == 2 => {
          let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(n.div(&d, bits, rm))
        }
        "Sqrt" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sqrt(bits, rm))
        }
        "Sin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sin(bits, rm, cc))
        }
        "Cos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cos(bits, rm, cc))
        }
        "Tan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tan(bits, rm, cc))
        }
        "Exp" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.exp(bits, rm, cc))
        }
        "Log" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.ln(bits, rm, cc))
        }
        "Log" if args.len() == 2 => {
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let val = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(val.log(&base, bits, rm, cc))
        }
        "Abs" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.abs())
        }
        "Sinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.sinh(bits, rm, cc))
        }
        "Cosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.cosh(bits, rm, cc))
        }
        "Tanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.tanh(bits, rm, cc))
        }
        "ArcSin" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asin(bits, rm, cc))
        }
        "ArcCos" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acos(bits, rm, cc))
        }
        "ArcTan" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atan(bits, rm, cc))
        }
        "ArcSinh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.asinh(bits, rm, cc))
        }
        "ArcCosh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.acosh(bits, rm, cc))
        }
        "ArcTanh" if args.len() == 1 => {
          let v = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(v.atanh(bits, rm, cc))
        }
        "Plus" => {
          // Evaluated Plus[a, b, c, ...] as a function call
          let mut result = BigFloat::from_i32(0, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.add(&v, bits, rm);
          }
          Ok(result)
        }
        "Times" => {
          let mut result = BigFloat::from_i32(1, bits);
          for arg in args {
            let v = expr_to_bigfloat(arg, bits, rm, cc)?;
            result = result.mul(&v, bits, rm);
          }
          Ok(result)
        }
        "Power" if args.len() == 2 => {
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let exp = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(base.pow(&exp, bits, rm, cc))
        }
        _ => Err(InterpreterError::EvaluationError(format!(
          "N: cannot evaluate {}[...] to arbitrary precision",
          name
        ))),
      }
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to arbitrary precision: {}",
      crate::syntax::expr_to_string(expr)
    ))),
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
    2 => {
      // RandomReal[range, n] - generate n random reals
      let n = match &args[1] {
        Expr::Integer(n) if *n > 0 => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomReal: second argument must be a positive integer".into(),
          ));
        }
      };

      let (min, max) = match &args[0] {
        Expr::Integer(m) => (0.0, *m as f64),
        Expr::Real(m) => (0.0, *m),
        Expr::List(items) if items.len() == 2 => {
          let lo = expr_to_num(&items[0]).ok_or_else(|| {
            InterpreterError::EvaluationError("RandomReal: invalid min".into())
          })?;
          let hi = expr_to_num(&items[1]).ok_or_else(|| {
            InterpreterError::EvaluationError("RandomReal: invalid max".into())
          })?;
          (lo, hi)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "RandomReal: invalid range".into(),
          ));
        }
      };

      let results: Vec<Expr> = (0..n)
        .map(|_| Expr::Real(min + rng.gen_range(0.0..1.0) * (max - min)))
        .collect();
      Ok(Expr::List(results))
    }
    _ => Err(InterpreterError::EvaluationError(
      "RandomReal expects 0, 1, or 2 arguments".into(),
    )),
  }
}

/// Clip[x
pub fn clip_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Clip expects 1 to 3 arguments".into(),
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

  let (min_val, max_val) = if args.len() >= 2 {
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

  // 3rd arg: replacement values {vMin, vMax} for out-of-range values
  if args.len() == 3
    && let Expr::List(replacements) = &args[2]
    && replacements.len() == 2
  {
    if x < min_val {
      return Ok(replacements[0].clone());
    } else if x > max_val {
      return Ok(replacements[1].clone());
    } else {
      return Ok(args[0].clone());
    }
  }

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

/// RandomVariate[dist] or RandomVariate[dist, n]
/// Supports UniformDistribution[{min, max}] and NormalDistribution[mu, sigma]
pub fn random_variate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use rand::Rng;
  use rand_distr::{Distribution, Normal};

  let mut rng = rand::thread_rng();

  let dist = &args[0];
  let n = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(n) if *n > 0 => Some(*n as usize),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RandomVariate: second argument must be a positive integer".into(),
        ));
      }
    }
  } else {
    None
  };

  // Extract distribution parameters
  let sample_fn: Box<dyn FnMut() -> f64> = match dist {
    Expr::FunctionCall { name, args: dargs }
      if name == "UniformDistribution" =>
    {
      if dargs.len() == 1 {
        if let Expr::List(bounds) = &dargs[0] {
          if bounds.len() == 2 {
            let lo = expr_to_num(&bounds[0]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid min bound".into(),
              )
            })?;
            let hi = expr_to_num(&bounds[1]).ok_or_else(|| {
              InterpreterError::EvaluationError(
                "UniformDistribution: invalid max bound".into(),
              )
            })?;
            Box::new(move || rng.gen_range(lo..hi))
          } else {
            return Err(InterpreterError::EvaluationError(
              "UniformDistribution: expected {min, max}".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "UniformDistribution: expected a list {min, max}".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "UniformDistribution: expected 1 argument".into(),
        ));
      }
    }
    Expr::FunctionCall { name, args: dargs }
      if name == "NormalDistribution" =>
    {
      let (mu, sigma) = match dargs.len() {
        0 => (0.0, 1.0),
        2 => {
          let mu = expr_to_num(&dargs[0]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "NormalDistribution: invalid mean".into(),
            )
          })?;
          let sigma = expr_to_num(&dargs[1]).ok_or_else(|| {
            InterpreterError::EvaluationError(
              "NormalDistribution: invalid standard deviation".into(),
            )
          })?;
          (mu, sigma)
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "NormalDistribution: expected 0 or 2 arguments".into(),
          ));
        }
      };
      let normal = Normal::new(mu, sigma).map_err(|e| {
        InterpreterError::EvaluationError(format!("NormalDistribution: {}", e))
      })?;
      Box::new(move || normal.sample(&mut rng))
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "RandomVariate".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Generate samples
  let mut sample_fn = sample_fn;
  match n {
    None => Ok(Expr::Real(sample_fn())),
    Some(count) => {
      let results: Vec<Expr> =
        (0..count).map(|_| Expr::Real(sample_fn())).collect();
      Ok(Expr::List(results))
    }
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
      // Times[n, BinaryOp::Divide(Pi, d)] or Times[n, BinaryOp::Divide(n2*Pi, d)]
      if let Some(n) = as_int(&args[0])
        && let Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: num,
          right: den,
        } = &args[1]
        && is_pi(num)
        && let Some(d) = as_int(den)
      {
        return Some(reduce(n, d));
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
    Expr::Integer(1) => Ok(Expr::Constant("E".to_string())),
    Expr::Real(f) => Ok(Expr::Real(f.exp())),
    _ => power_two(&Expr::Constant("E".to_string()), &args[0]),
  }
}

pub fn erf_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Erf expects 1 argument".into(),
    ));
  }
  // Helper: negate the Erf of the positive part
  let negate_erf = |inner: Expr| -> Expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::FunctionCall {
        name: "Erf".to_string(),
        args: vec![inner],
      }),
    }
  };
  match &args[0] {
    // Erf[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // Erf[-x] = -Erf[x] (UnaryOp form)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => Ok(negate_erf(*operand.clone())),
    // Erf[Times[-1, x]] = -Erf[x] (evaluated form of -x)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return Ok(negate_erf(fargs[1].clone()));
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return Ok(negate_erf(fargs[0].clone()));
      }
      // Negative integer coefficient: Times[-n, x] -> -Erf[Times[n, x]]
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()],
        };
        return Ok(negate_erf(pos_arg));
      }
      Ok(Expr::FunctionCall {
        name: "Erf".to_string(),
        args: args.to_vec(),
      })
    }
    // BinaryOp::Times form: -1 * x
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        return Ok(negate_erf(*right.clone()));
      }
      if matches!(right.as_ref(), Expr::Integer(-1)) {
        return Ok(negate_erf(*left.clone()));
      }
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
      {
        let pos_arg = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(-*n)),
          right: right.clone(),
        };
        return Ok(negate_erf(pos_arg));
      }
      Ok(Expr::FunctionCall {
        name: "Erf".to_string(),
        args: args.to_vec(),
      })
    }
    // Erf[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_erf(Expr::Integer(-*n))),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(erf_f64(*f))),
    // Otherwise symbolic
    _ => Ok(Expr::FunctionCall {
      name: "Erf".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn erfc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Erfc expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // Erfc[0] = 1
    Expr::Integer(0) => Ok(Expr::Integer(1)),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(1.0 - erf_f64(*f))),
    // Otherwise symbolic
    _ => Ok(Expr::FunctionCall {
      name: "Erfc".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Log[0] = -Infinity
      if matches!(&args[0], Expr::Integer(0)) {
        return Ok(Expr::UnaryOp {
          op: crate::syntax::UnaryOperator::Minus,
          operand: Box::new(Expr::Constant("Infinity".to_string())),
        });
      }
      // Log[1] = 0
      if matches!(&args[0], Expr::Integer(1)) {
        return Ok(Expr::Integer(0));
      }
      // Log[E] = 1
      if matches!(&args[0], Expr::Constant(c) if c == "E") {
        return Ok(Expr::Integer(1));
      }
      // Log[E^n] = n only when n is a concrete integer
      if let Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } = &args[0]
        && matches!(left.as_ref(), Expr::Constant(c) if c == "E")
        && matches!(right.as_ref(), Expr::Integer(_) | Expr::BigInteger(_))
      {
        return Ok(*right.clone());
      }
      if let Expr::FunctionCall {
        name,
        args: pow_args,
      } = &args[0]
        && name == "Power"
        && pow_args.len() == 2
        && matches!(&pow_args[0], Expr::Constant(c) if c == "E")
        && matches!(&pow_args[1], Expr::Integer(_) | Expr::BigInteger(_))
      {
        return Ok(pow_args[1].clone());
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
      // Log[base, x] — integer base and argument
      if let (Some(base), Some(x)) =
        (expr_to_i128(&args[0]), expr_to_i128(&args[1]))
        && base > 1
        && x > 0
      {
        // Check if x is an exact power of base
        let mut val = x;
        let mut exp = 0i128;
        while val > 1 && val % base == 0 {
          val /= base;
          exp += 1;
        }
        if val == 1 {
          return Ok(Expr::Integer(exp));
        }
        // Return Log[x]/Log[base] symbolically
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![args[1].clone()],
          }),
          right: Box::new(Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![args[0].clone()],
          }),
        });
      }
      // Log[base, x] — evaluate for Real args
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
  match &args[0] {
    Expr::Integer(n) if *n > 0 => {
      // Check if n is an exact power of 10
      let mut val = *n;
      let mut exp = 0i128;
      while val > 1 && val % 10 == 0 {
        val /= 10;
        exp += 1;
      }
      if val == 1 {
        return Ok(Expr::Integer(exp));
      }
    }
    Expr::Real(f) if *f > 0.0 => {
      return Ok(Expr::Real(f.log10()));
    }
    _ => {}
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
  match &args[0] {
    Expr::Integer(n) if *n > 0 => {
      // Check if n is an exact power of 2
      let val = *n as u128;
      if val.is_power_of_two() {
        return Ok(Expr::Integer(val.trailing_zeros() as i128));
      }
    }
    Expr::Real(f) if *f > 0.0 => {
      return Ok(Expr::Real(f.log2()));
    }
    _ => {}
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
    Expr::Integer(1) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(-1) => {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    Expr::Real(f) if f.abs() < 1.0 => return Ok(Expr::Real(f.atanh())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcTanh".to_string(),
    args: args.to_vec(),
  })
}

pub fn arccot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCot expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(pi_over_n(2)), // Pi/2
    Expr::Integer(1) => return Ok(pi_over_n(4)), // Pi/4
    Expr::Integer(-1) => {
      // -Pi/4
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(pi_over_n(4)),
      });
    }
    Expr::Real(f) => return Ok(Expr::Real((1.0 / f).atan())),
    _ => {}
  }
  if let Some(f) = try_eval_to_f64(&args[0]) {
    return Ok(Expr::Real((1.0 / f).atan()));
  }
  Ok(Expr::FunctionCall {
    name: "ArcCot".to_string(),
    args: args.to_vec(),
  })
}

pub fn arccsc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCsc expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(1) => return Ok(pi_over_n(2)), // Pi/2
    Expr::Integer(-1) => return Ok(negative_pi_over_2()), // -Pi/2
    _ => {}
  }
  if let Some(f) = try_eval_to_f64(&args[0])
    && f.abs() >= 1.0
  {
    return Ok(Expr::Real((1.0 / f).asin()));
  }
  Ok(Expr::FunctionCall {
    name: "ArcCsc".to_string(),
    args: args.to_vec(),
  })
}

pub fn arcsec_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSec expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Integer(-1) => return Ok(Expr::Identifier("Pi".to_string())),
    _ => {}
  }
  if let Some(f) = try_eval_to_f64(&args[0])
    && f.abs() >= 1.0
  {
    return Ok(Expr::Real((1.0 / f).acos()));
  }
  Ok(Expr::FunctionCall {
    name: "ArcSec".to_string(),
    args: args.to_vec(),
  })
}

pub fn arccsch_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCsch expects 1 argument".into(),
    ));
  }
  if let Expr::Integer(0) = &args[0] {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if let Some(f) = try_eval_to_f64(&args[0])
    && f != 0.0
  {
    return Ok(Expr::Real((1.0 / f).asinh()));
  }
  Ok(Expr::FunctionCall {
    name: "ArcCsch".to_string(),
    args: args.to_vec(),
  })
}

/// Helper to construct -Pi/2 matching wolframscript output format
/// Construct Pi/n as an AST expression
fn pi_over_n(n: i128) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::Constant("Pi".to_string())),
    right: Box::new(Expr::Integer(n)),
  }
}

fn negative_pi_over_2() -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::Integer(2)),
    }),
    right: Box::new(Expr::Constant("Pi".to_string())),
  }
}

/// Gudermannian[x] - the Gudermannian function: 2 ArcTan[Tanh[x/2]]
/// Gudermannian[0] = 0, Gudermannian[Infinity] = Pi/2, Gudermannian[-Infinity] = -Pi/2
pub fn gudermannian_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Gudermannian expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if *f == 0.0 => return Ok(Expr::Real(0.0)),
    Expr::Real(f) => {
      // Gudermannian[x] = 2 * atan(tanh(x/2))
      let result = 2.0 * (f / 2.0).tanh().atan();
      return Ok(Expr::Real(result));
    }
    Expr::Identifier(name) if name == "Infinity" => {
      // Gudermannian[Infinity] = Pi/2
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      });
    }
    Expr::Identifier(name) if name == "ComplexInfinity" => {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    Expr::Identifier(name) if name == "Undefined" => {
      return Ok(Expr::Identifier("Undefined".to_string()));
    }
    Expr::Identifier(name) if name == "Indeterminate" => {
      return Ok(Expr::Identifier("Indeterminate".to_string()));
    }
    // -Infinity (as UnaryOp)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } if matches!(operand.as_ref(), Expr::Identifier(n) if n == "Infinity") => {
      return Ok(negative_pi_over_2());
    }
    // -Infinity (as BinaryOp)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1))
      && matches!(right.as_ref(), Expr::Identifier(n) if n == "Infinity") =>
    {
      return Ok(negative_pi_over_2());
    }
    // -Infinity (as FunctionCall)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times"
        && fargs.len() == 2
        && matches!(fargs[0], Expr::Integer(-1))
        && matches!(&fargs[1], Expr::Identifier(n) if n == "Infinity") =>
    {
      return Ok(negative_pi_over_2());
    }
    _ => {}
  }
  // Symbolic: return unevaluated (matching wolframscript behavior)
  Ok(Expr::FunctionCall {
    name: "Gudermannian".to_string(),
    args: args.to_vec(),
  })
}

/// InverseGudermannian[x] - the inverse Gudermannian function
pub fn inverse_gudermannian_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "InverseGudermannian expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) if *f == 0.0 => return Ok(Expr::Real(0.0)),
    Expr::Real(f) => {
      // InverseGudermannian[x] = 2 * atanh(tan(x/2))
      let result = 2.0 * (f / 2.0).tan().atanh();
      return Ok(Expr::Real(result));
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "InverseGudermannian".to_string(),
    args: args.to_vec(),
  })
}

/// LogisticSigmoid[x] = 1 / (1 + Exp[-x])
pub fn logistic_sigmoid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogisticSigmoid expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => {
      // LogisticSigmoid[0] = 1/2
      return Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)],
      });
    }
    Expr::Real(f) => {
      let result = 1.0 / (1.0 + (-f).exp());
      return Ok(Expr::Real(result));
    }
    Expr::Integer(n) => {
      let f = *n as f64;
      let result = 1.0 / (1.0 + (-f).exp());
      return Ok(Expr::Real(result));
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "LogisticSigmoid".to_string(),
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
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "DigitCount".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
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
  use num_traits::Zero;
  let big_base = BigInt::from(base);
  let mut digits = Vec::new();
  let mut val = n;
  if val.is_zero() {
    digits.push(0usize);
  } else {
    while !val.is_zero() {
      use num_traits::ToPrimitive;
      let rem = (&val % &big_base).to_usize().unwrap_or(0);
      digits.push(rem);
      val /= &big_base;
    }
  }

  if args.len() == 3 {
    // DigitCount[n, b, d] - count of specific digit d
    let d = match expr_to_i128(&args[2]) {
      Some(d) => d as usize,
      None => {
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
  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "DigitSum".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
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

  use num_traits::Zero;
  let big_base = BigInt::from(base);
  let mut sum = BigInt::from(0);
  let mut val = n;
  if val.is_zero() {
    return Ok(Expr::Integer(0));
  }
  while !val.is_zero() {
    sum += &val % &big_base;
    val /= &big_base;
  }
  Ok(bigint_to_expr(sum))
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
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
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
  let mut a = BigInt::from(2);
  let mut b = BigInt::from(1);
  for _ in 2..=n {
    let c = &a + &b;
    a = b;
    b = c;
  }
  Ok(bigint_to_expr(b))
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
      (r, m) if expr_to_i128(r).is_some() && expr_to_i128(m).is_some() => {
        let mv = expr_to_i128(m).unwrap();
        if mv > 0 {
          r_vals.push(expr_to_i128(r).unwrap());
          m_vals.push(mv);
        } else {
          return Ok(Expr::FunctionCall {
            name: "ChineseRemainder".to_string(),
            args: args.to_vec(),
          });
        }
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
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n > 0 => n,
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
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
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

/// BellB[n] - nth Bell number
/// BellB[n, x] - nth Bell polynomial
pub fn bell_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "BellB expects 1 or 2 arguments".into(),
    ));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BellB".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 1 {
    // Bell number B_n via the Bell triangle
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute using the Bell triangle (first column of each row)
    let mut row = vec![1i128];
    for _ in 1..=n {
      let mut new_row = vec![*row.last().unwrap()];
      for j in 1..=row.len() {
        let val = new_row[j - 1] + row[j - 1];
        new_row.push(val);
      }
      row = new_row;
    }
    Ok(Expr::Integer(row[0]))
  } else {
    // Bell polynomial B_n(x) = sum_{k=0}^{n} S(n,k) * x^k
    // where S(n,k) is the Stirling number of the second kind
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    // Compute Stirling numbers of the second kind for all k
    let mut stirling = vec![vec![0i128; n + 1]; n + 1];
    stirling[0][0] = 1;
    for i in 1..=n {
      for k in 1..=i {
        stirling[i][k] =
          k as i128 * stirling[i - 1][k] + stirling[i - 1][k - 1];
      }
    }
    // Build polynomial: sum_{k=0}^{n} S(n,k) * x^k
    let x = &args[1];
    let mut terms = Vec::new();
    for k in 0..=n {
      let s = stirling[n][k];
      if s == 0 {
        continue;
      }
      let coeff = Expr::Integer(s);
      let term = if k == 0 {
        coeff
      } else if k == 1 {
        if s == 1 {
          x.clone()
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(coeff),
            right: Box::new(x.clone()),
          }
        }
      } else {
        let power = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(x.clone()),
          right: Box::new(Expr::Integer(k as i128)),
        };
        if s == 1 {
          power
        } else {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(coeff),
            right: Box::new(power),
          }
        }
      };
      terms.push(term);
    }
    if terms.is_empty() {
      return Ok(Expr::Integer(0));
    }
    // Build sum of terms
    let mut result = terms[0].clone();
    for term in &terms[1..] {
      result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(result),
        right: Box::new(term.clone()),
      };
    }
    crate::evaluator::evaluate_expr_to_expr(&result)
  }
}

/// PauliMatrix[k] - kth Pauli matrix (k=0,1,2,3)
pub fn pauli_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PauliMatrix expects exactly 1 argument".into(),
    ));
  }
  let k = match expr_to_i128(&args[0]) {
    Some(k) => k,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PauliMatrix".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let i_expr = Expr::Identifier("I".to_string());
  let neg_i = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(i_expr.clone()),
  };
  match k {
    0 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
    ])),
    1 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(0), Expr::Integer(1)]),
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
    ])),
    2 => {
      let neg_i_eval = crate::evaluator::evaluate_expr_to_expr(&neg_i)?;
      Ok(Expr::List(vec![
        Expr::List(vec![Expr::Integer(0), neg_i_eval]),
        Expr::List(vec![i_expr, Expr::Integer(0)]),
      ]))
    }
    3 => Ok(Expr::List(vec![
      Expr::List(vec![Expr::Integer(1), Expr::Integer(0)]),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(-1)]),
    ])),
    _ => Ok(Expr::FunctionCall {
      name: "PauliMatrix".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// CatalanNumber[n] - nth Catalan number = C(2n,n)/(n+1)
pub fn catalan_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CatalanNumber expects exactly 1 argument".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
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
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS1".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
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
  // Use DP table with BigInt to avoid overflow
  let zero = BigInt::from(0);
  let one = BigInt::from(1);
  let mut table = vec![vec![zero.clone(); k + 1]; n + 1];
  table[0][0] = one;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] =
        &table[i - 1][j - 1] - BigInt::from(i - 1) * &table[i - 1][j];
    }
  }
  Ok(bigint_to_expr(table[n][k].clone()))
}

/// StirlingS2[n, k] - Stirling number of the second kind
pub fn stirling_s2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StirlingS2 expects exactly 2 arguments".into(),
    ));
  }
  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "StirlingS2".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let k = match expr_to_i128(&args[1]) {
    Some(k) if k >= 0 => k as usize,
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
  let zero = BigInt::from(0);
  let one = BigInt::from(1);
  let mut table = vec![vec![zero.clone(); k + 1]; n + 1];
  table[0][0] = one;
  for i in 1..=n {
    for j in 1..=k.min(i) {
      table[i][j] = BigInt::from(j) * &table[i - 1][j] + &table[i - 1][j - 1];
    }
  }
  Ok(bigint_to_expr(table[n][k].clone()))
}

/// Digamma function approximation using the asymptotic series.
/// ψ(x) for x > 0 using recurrence and Stirling-like expansion.
fn digamma(mut x: f64) -> f64 {
  let mut result = 0.0;
  // Use recurrence ψ(x+1) = ψ(x) + 1/x to shift x to large values
  while x < 20.0 {
    result -= 1.0 / x;
    x += 1.0;
  }
  // Asymptotic expansion: ψ(x) ~ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k})
  result += x.ln() - 0.5 / x;
  let x2 = x * x;
  let mut xpow = x2;
  // B_2/(2·x^2), B_4/(4·x^4), B_6/(6·x^6), ...
  let coeffs = [
    1.0 / 12.0,      // B_2/2 = 1/12
    1.0 / 120.0,     // B_4/4 = -1/30 / 4 => but sign alternates
    1.0 / 252.0,     // B_6/6
    1.0 / 240.0,     // B_8/8
    5.0 / 660.0,     // B_10/10
    691.0 / 32760.0, // B_12/12
    7.0 / 12.0,      // B_14/14
  ];
  // Signs: -, +, -, +, -, +, -
  let signs = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
  for i in 0..coeffs.len() {
    result += signs[i] * coeffs[i] / xpow;
    xpow *= x2;
  }
  result
}

/// HarmonicNumber[n] - Returns the nth harmonic number H_n = 1 + 1/2 + ... + 1/n.
/// HarmonicNumber[n, r] - Returns the generalized harmonic number H_{n,r} = Sum[1/k^r, {k,1,n}].
pub fn harmonic_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "HarmonicNumber expects 1 or 2 arguments".into(),
    ));
  }

  // Handle real/float argument: H(x) = digamma(x+1) + EulerGamma
  if args.len() == 1
    && let Some(x) = expr_to_num(&args[0])
    && expr_to_i128(&args[0]).is_none()
  {
    // Real input - use digamma approximation
    // Euler-Mascheroni constant
    const EULER_GAMMA: f64 = 0.5772156649015329;
    let result = digamma(x + 1.0) + EULER_GAMMA;
    return Ok(Expr::Real(result));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 0 => n,
    Some(_) => {
      return Ok(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: args.to_vec(),
      });
    }
    None => {
      return Ok(Expr::FunctionCall {
        name: "HarmonicNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let r = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(r) => r,
      None => {
        return Ok(Expr::FunctionCall {
          name: "HarmonicNumber".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  if n == 0 {
    return Ok(Expr::Integer(0));
  }

  // Compute as exact rational: sum of 1/k^r for k = 1 to n
  // Use BigInt numerator and denominator
  fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
    use num_traits::Zero;
    let mut a = if *a < BigInt::zero() {
      -a.clone()
    } else {
      a.clone()
    };
    let mut b = if *b < BigInt::zero() {
      -b.clone()
    } else {
      b.clone()
    };
    while !b.is_zero() {
      let t = b.clone();
      b = &a % &b;
      a = t;
    }
    a
  }

  let mut num = BigInt::from(0);
  let mut den = BigInt::from(1);
  for k in 1..=n {
    let k_big = BigInt::from(k);
    let k_pow = num_traits::pow::pow(k_big, r as usize);
    // Add 1/k_pow to num/den: num/den + 1/k_pow = (num*k_pow + den) / (den*k_pow)
    num = &num * &k_pow + &den;
    den = &den * &k_pow;
    // Reduce
    let g = bigint_gcd(&num, &den);
    use num_traits::One;
    if g > BigInt::one() {
      num /= &g;
      den /= &g;
    }
  }

  // Convert to our representation
  if den == BigInt::from(1) {
    Ok(bigint_to_expr(num))
  } else {
    Ok(Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![bigint_to_expr(num), bigint_to_expr(den)],
    })
  }
}

/// Prime[n] - Returns the nth prime number
pub fn prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Prime expects exactly 1 argument".into(),
    ));
  }
  match expr_to_i128(&args[0]).or_else(|| {
    if let Expr::Real(f) = &args[0] {
      if f.fract() == 0.0 {
        Some(*f as i128)
      } else {
        None
      }
    } else {
      None
    }
  }) {
    Some(n) if n >= 1 => {
      Ok(Expr::Integer(crate::nth_prime(n as usize) as i128))
    }
    _ => {
      // Wolfram returns unevaluated for non-positive or non-integer arguments
      Ok(Expr::FunctionCall {
        name: "Prime".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Fibonacci[n] - Returns the nth Fibonacci number
pub fn fibonacci_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 0 {
        let pos_n = (-n) as u128;
        let fib = fibonacci_number_bigint(pos_n);
        let sign = if pos_n.is_multiple_of(2) {
          BigInt::from(-1)
        } else {
          BigInt::from(1)
        };
        Ok(bigint_to_expr(sign * fib))
      } else {
        Ok(bigint_to_expr(fibonacci_number_bigint(n as u128)))
      }
    }
    None => Ok(Expr::FunctionCall {
      name: "Fibonacci".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn fibonacci_number_bigint(n: u128) -> BigInt {
  if n == 0 {
    return BigInt::from(0);
  }
  let mut a = BigInt::from(0);
  let mut b = BigInt::from(1);
  for _ in 1..n {
    let tmp = &a + &b;
    a = b;
    b = tmp;
  }
  b
}

/// IntegerDigits[n], IntegerDigits[n, b], IntegerDigits[n, b, len]
pub fn integer_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "IntegerDigits expects 1 to 3 arguments".into(),
    ));
  }

  let n = match expr_to_bigint(&args[0]) {
    Some(n) => n.abs(),
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntegerDigits".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let base = if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => BigInt::from(b),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "IntegerDigits: base must be an integer >= 2".into(),
        ));
      }
    }
  } else {
    BigInt::from(10)
  };

  use num_traits::Zero;

  let mut digits = Vec::new();
  if n.is_zero() {
    digits.push(Expr::Integer(0));
  } else {
    let mut num = n;
    while !num.is_zero() {
      digits.push(bigint_to_expr(&num % &base));
      num /= &base;
    }
    digits.reverse();
  }

  // Handle optional length parameter
  if args.len() == 3 {
    match expr_to_i128(&args[2]) {
      Some(len) if len >= 0 => {
        let len = len as usize;
        if digits.len() < len {
          // Pad with zeros on the left
          let mut padded = vec![Expr::Integer(0); len - digits.len()];
          padded.append(&mut digits);
          digits = padded;
        } else if digits.len() > len {
          // Truncate from the left (keep least significant digits)
          digits = digits[digits.len() - len..].to_vec();
        }
      }
      _ => {
        return Err(InterpreterError::EvaluationError(
          "IntegerDigits: length must be a non-negative integer".into(),
        ));
      }
    }
  }

  Ok(Expr::List(digits))
}

/// Extract decimal digits and exponent from a BigFloat.
/// Returns (digit_chars, decimal_exponent) where digit_chars are ASCII digit bytes
/// and decimal_exponent is the number of integer digits (position of decimal point).
fn bigfloat_to_digits(
  bf: &astro_float::BigFloat,
) -> Result<(Vec<u8>, i64), InterpreterError> {
  use num_bigint::BigUint;
  use num_traits::Zero;

  let (words, sig_bits, _sign, exponent, _inexact) =
    bf.as_raw_parts().ok_or_else(|| {
      InterpreterError::EvaluationError(
        "RealDigits: cannot extract NaN or Inf".into(),
      )
    })?;

  if sig_bits == 0 || words.iter().all(|&w| w == 0) {
    return Ok((vec![b'0'], 0));
  }

  let mantissa = BigUint::from_bytes_le(
    &words
      .iter()
      .flat_map(|w| w.to_le_bytes())
      .collect::<Vec<u8>>(),
  );

  let mantissa_bits = words.len() * 64;
  let shift = exponent as i64 - mantissa_bits as i64;

  let target_digits =
    (mantissa_bits as f64 / std::f64::consts::LOG2_10).ceil() as usize + 2;

  let (int_digits, decimal_exp) = if shift >= 0 {
    let int_val = &mantissa << (shift as u64);
    let s = int_val.to_string();
    let len = s.len();
    (s, len as i64)
  } else {
    let neg_shift = (-shift) as u64;
    let scale = BigUint::from(10u32).pow(target_digits as u32);
    let scaled = &mantissa * &scale;
    let result = &scaled >> neg_shift;

    if result.is_zero() {
      return Ok((vec![b'0'], 0));
    }

    let s = result.to_string();
    let decimal_exp = s.len() as i64 - target_digits as i64;
    (s, decimal_exp)
  };

  Ok((int_digits.into_bytes(), decimal_exp))
}

/// RealDigits[x, base, num_digits] — extract decimal digits of a real number.
/// Returns {digit_list, exponent}.
pub fn real_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "RealDigits expects 1 to 3 arguments".into(),
    ));
  }

  // Only base 10 is supported for now
  if args.len() >= 2 {
    match expr_to_i128(&args[1]) {
      Some(10) => {}
      Some(_) => {
        return Ok(Expr::FunctionCall {
          name: "RealDigits".to_string(),
          args: args.to_vec(),
        });
      }
      None => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: base must be an integer".into(),
        ));
      }
    }
  }

  let num_digits: usize = if args.len() >= 3 {
    match expr_to_i128(&args[2]) {
      Some(n) if n > 0 => n as usize,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "RealDigits: number of digits must be a positive integer".into(),
        ));
      }
    }
  } else {
    // Default: use machine-precision (~16 digits)
    16
  };

  let expr = &args[0];

  // Determine sign and work with absolute value
  let is_negative = match expr {
    Expr::Integer(n) => *n < 0,
    Expr::Real(f) => *f < 0.0,
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      ..
    } => true,
    _ => false,
  };

  let abs_expr = if is_negative {
    Expr::FunctionCall {
      name: "Abs".to_string(),
      args: vec![expr.clone()],
    }
  } else {
    expr.clone()
  };

  // Check for exact zero
  let is_zero = matches!(&abs_expr, Expr::Integer(0))
    || matches!(&abs_expr, Expr::Real(f) if *f == 0.0);

  if is_zero {
    let digits = vec![Expr::Integer(0); num_digits];
    return Ok(Expr::List(vec![Expr::List(digits), Expr::Integer(0)]));
  }

  // Compute with extra precision to avoid rounding errors in the last digits
  let extra = 10;
  let precision = num_digits + extra;

  use astro_float::{Consts, RoundingMode};
  let mut cc = Consts::new().map_err(|e| {
    InterpreterError::EvaluationError(format!("BigFloat init error: {}", e))
  })?;
  let rm = RoundingMode::ToEven;
  let bits = nominal_bits(precision);

  let bf = expr_to_bigfloat(&abs_expr, bits, rm, &mut cc)?;

  let (raw_digits, decimal_exp) = bigfloat_to_digits(&bf)?;

  // raw_digits are the significant digits, decimal_exp is the exponent
  // (number of digits before the decimal point).
  // We need exactly num_digits digits.
  let mut digits: Vec<i128> = raw_digits
    .iter()
    .filter(|b| b.is_ascii_digit())
    .map(|b| (*b - b'0') as i128)
    .collect();

  // Pad with zeros if we don't have enough digits
  while digits.len() < num_digits {
    digits.push(0);
  }

  // Truncate to requested number of digits
  digits.truncate(num_digits);

  let digit_exprs: Vec<Expr> =
    digits.iter().map(|&d| Expr::Integer(d)).collect();

  Ok(Expr::List(vec![
    Expr::List(digit_exprs),
    Expr::Integer(decimal_exp as i128),
  ]))
}

/// FromDigits[list
pub fn from_digits_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "FromDigits expects 1 or 2 arguments".into(),
    ));
  }

  let base: i128 = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "FromDigits: base must be an integer >= 2".into(),
        ));
      }
    }
  } else {
    10
  };

  let big_base = BigInt::from(base);

  // Handle string argument: FromDigits["1234"] or FromDigits["1a", 16]
  if let Expr::String(s) = &args[0] {
    let mut result = BigInt::from(0);
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
      result = result * &big_base + BigInt::from(d);
    }
    return Ok(bigint_to_expr(result));
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

  // Check if all items are numeric
  let all_numeric = items.iter().all(|item| expr_to_i128(item).is_some());

  if all_numeric {
    let mut result = BigInt::from(0);
    for item in items {
      let d = expr_to_i128(item).unwrap();
      result = result * &big_base + BigInt::from(d);
    }
    Ok(bigint_to_expr(result))
  } else {
    // Symbolic: build expression base*(base*(...) + d1) + d2
    // i.e., Horner form: ((d0 * base + d1) * base + d2) * base + ...
    let base_expr = Expr::Integer(base);
    let mut result = items[0].clone();
    for item in &items[1..] {
      // result = result * base + item
      result = Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![base_expr.clone(), result],
          },
          item.clone(),
        ],
      };
      result = crate::evaluator::evaluate_expr_to_expr(&result)?;
    }
    Ok(result)
  }
}

/// FactorInteger[n] - Returns the prime factorization of n
pub fn factor_integer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => factor_integer_i128(*n),
    Expr::BigInteger(n) => factor_integer_bigint(n),
    _ => Ok(Expr::FunctionCall {
      name: "FactorInteger".to_string(),
      args: args.to_vec(),
    }),
  }
}

fn factor_integer_i128(n: i128) -> Result<Expr, InterpreterError> {
  if n == 0 {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
    ));
  }

  // For large integers where trial division would be too slow,
  // delegate to the BigInt path which uses Pollard's rho
  if n.unsigned_abs() > (1u128 << 53) {
    return factor_integer_bigint(&BigInt::from(n));
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
  while num.is_multiple_of(2) {
    count += 1;
    num /= 2;
  }
  if count > 0 {
    factors.push(Expr::List(vec![Expr::Integer(2), Expr::Integer(count)]));
  }

  // Handle odd factors (safe for small n where trial division is fast)
  let mut i: u128 = 3;
  while i * i <= num {
    let mut count = 0i128;
    while num.is_multiple_of(i) {
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

fn factor_integer_bigint(n: &BigInt) -> Result<Expr, InterpreterError> {
  use num_traits::{One, Signed, Zero};

  if n.is_zero() {
    return Err(InterpreterError::EvaluationError(
      "FactorInteger: argument cannot be zero".into(),
    ));
  }

  let mut factors: Vec<Expr> = Vec::new();

  if n.is_negative() {
    factors.push(Expr::List(vec![Expr::Integer(-1), Expr::Integer(1)]));
  }

  let mut remaining = n.abs();
  let one = BigInt::one();

  if remaining == one {
    return Ok(Expr::List(factors));
  }

  // Trial division for small primes
  let two = BigInt::from(2);
  let mut count = 0i128;
  while (&remaining % &two).is_zero() {
    count += 1;
    remaining /= &two;
  }
  if count > 0 {
    factors.push(Expr::List(vec![Expr::Integer(2), Expr::Integer(count)]));
  }

  let trial_limit = 1_000_000u64;
  let mut i = 3u64;
  while i <= trial_limit {
    let bi = BigInt::from(i);
    if &bi * &bi > remaining {
      break;
    }
    let mut count = 0i128;
    while (&remaining % &bi).is_zero() {
      count += 1;
      remaining /= &bi;
    }
    if count > 0 {
      factors.push(Expr::List(vec![
        Expr::Integer(i as i128),
        Expr::Integer(count),
      ]));
    }
    i += 2;
  }

  // Factor remaining cofactor using num_prime (Pollard's rho + SQUFOF)
  if remaining > one {
    let remaining_uint = remaining.to_biguint().unwrap();
    let prime_factors = num_prime::nt_funcs::factorize(remaining_uint);
    for (factor, exponent) in prime_factors {
      let factor_bigint = BigInt::from(factor);
      // Merge with existing factor or add new entry
      let mut merged = false;
      for f in factors.iter_mut() {
        if let Expr::List(pair) = f {
          let matches = match (&pair[0], &factor_bigint) {
            (Expr::Integer(a), b) => BigInt::from(*a) == *b,
            (Expr::BigInteger(a), b) => a == b,
            _ => false,
          };
          if matches {
            if let Expr::Integer(ref mut exp) = pair[1] {
              *exp += exponent as i128;
            }
            merged = true;
            break;
          }
        }
      }
      if !merged {
        factors.push(Expr::List(vec![
          bigint_to_expr(factor_bigint),
          Expr::Integer(exponent as i128),
        ]));
      }
    }
  }

  // Sort factors by prime value
  factors.sort_by(|a, b| {
    let a_val = match a {
      Expr::List(pair) => match &pair[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => BigInt::zero(),
      },
      _ => BigInt::zero(),
    };
    let b_val = match b {
      Expr::List(pair) => match &pair[0] {
        Expr::Integer(n) => BigInt::from(*n),
        Expr::BigInteger(n) => n.clone(),
        _ => BigInt::zero(),
      },
      _ => BigInt::zero(),
    };
    a_val.cmp(&b_val)
  });

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
  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
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
      e if expr_to_i128(e).is_some_and(|k| k >= 0) => {
        (1, expr_to_i128(e).unwrap() as u64)
      }
      // IntegerPartitions[n, All] — no constraint
      Expr::Identifier(s) if s == "All" => (1, n.max(1)),
      // IntegerPartitions[n, {k}] — exactly k parts
      Expr::List(lst) if lst.len() == 1 => match expr_to_i128(&lst[0]) {
        Some(k) if k >= 0 => (k as u64, k as u64),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "IntegerPartitions".to_string(),
            args: args.to_vec(),
          });
        }
      },
      // IntegerPartitions[n, {kmin, kmax}] — range of parts
      Expr::List(lst) if lst.len() == 2 => {
        match (expr_to_i128(&lst[0]), expr_to_i128(&lst[1])) {
          (Some(lo), Some(hi)) if lo >= 0 && hi >= 0 => (lo as u64, hi as u64),
          _ => {
            return Ok(Expr::FunctionCall {
              name: "IntegerPartitions".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }
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
            e if expr_to_i128(e).is_some_and(|v| v > 0) => {
              vals.push(expr_to_i128(e).unwrap() as u64)
            }
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

  let n = match expr_to_i128(&args[0]) {
    Some(0) => {
      return Err(InterpreterError::EvaluationError(
        "Divisors: argument cannot be zero".into(),
      ));
    }
    Some(n) => n.unsigned_abs(),
    None => {
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

  let k = match expr_to_i128(&args[0]) {
    Some(k) if k >= 0 => k as u32,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: first argument must be a non-negative integer".into(),
      ));
    }
  };

  let n = match expr_to_i128(&args[1]) {
    Some(0) => {
      return Err(InterpreterError::EvaluationError(
        "DivisorSigma: second argument cannot be zero".into(),
      ));
    }
    Some(n) => n.unsigned_abs(),
    None => {
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

  let n = match expr_to_i128(&args[0]) {
    Some(n) if n >= 1 => n as u128,
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

  let n = match expr_to_i128(&args[0]) {
    Some(0) => return Ok(Expr::Integer(0)),
    Some(n) if n >= 1 => n as u128,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "EulerPhi: argument must be a non-negative integer".into(),
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

/// CoprimeQ[a, b, ...] - Tests if integers are pairwise coprime
pub fn coprime_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "CoprimeQ expects at least 2 arguments".into(),
    ));
  }

  // Extract all integer values
  let nums: Vec<u128> = args
    .iter()
    .filter_map(|a| expr_to_i128(a).map(|n| n.unsigned_abs()))
    .collect();

  if nums.len() != args.len() {
    return Ok(Expr::FunctionCall {
      name: "CoprimeQ".to_string(),
      args: args.to_vec(),
    });
  }

  // Check all pairs are coprime
  for i in 0..nums.len() {
    for j in (i + 1)..nums.len() {
      let (mut a, mut b) = (nums[i].max(1), nums[j].max(1));
      while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
      }
      if a != 1 {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
  }

  Ok(Expr::Identifier("True".to_string()))
}

/// Re[z] - Real part of a complex number (for real numbers, returns the number itself)
pub fn re_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Re expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_) | Expr::Real(_) => return Ok(args[0].clone()),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      return Ok(inner[0].clone());
    }
    _ => {}
  }

  // Try exact integer/rational complex extraction: handles a + b*I patterns
  if let Some(((rn, rd), _)) = try_extract_complex_exact(&args[0]) {
    return Ok(make_rational(rn, rd));
  }

  // Try float complex extraction: handles 1.5 + 2.5*I patterns
  if let Some((re, _)) = try_extract_complex_float(&args[0]) {
    return Ok(Expr::Real(re));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "Re".to_string(),
    args: args.to_vec(),
  })
}

/// Im[z] - Imaginary part of a complex number (for real numbers, returns 0)
pub fn im_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Im expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_) | Expr::Real(_) => return Ok(Expr::Integer(0)),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      return Ok(inner[1].clone());
    }
    _ => {}
  }

  // Try exact integer/rational complex extraction: handles a + b*I patterns
  if let Some((_, (in_, id))) = try_extract_complex_exact(&args[0]) {
    return Ok(make_rational(in_, id));
  }

  // Try float complex extraction: handles 1.5 + 2.5*I patterns
  if let Some((_, im)) = try_extract_complex_float(&args[0]) {
    return Ok(Expr::Real(im));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "Im".to_string(),
    args: args.to_vec(),
  })
}

/// Helper: check if an expression is a known real value (Integer, Real, or Rational)
fn is_known_real(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) => true,
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      matches!((&args[0], &args[1]), (Expr::Integer(_), Expr::Integer(_)))
    }
    _ => false,
  }
}

/// Conjugate a single expression, returning the conjugated form.
/// For known reals, returns the value; for I, returns -I;
/// for Plus/Times/List, distributes; otherwise wraps in Conjugate[...].
fn conjugate_one(expr: &Expr) -> Result<Expr, InterpreterError> {
  // Known real numbers are their own conjugate
  if is_known_real(expr) {
    return Ok(expr.clone());
  }

  // Complex[a, b] -> Complex[a, -b]
  if let Expr::FunctionCall { name, args } = expr
    && name == "Complex"
    && args.len() == 2
  {
    let neg_imag = match &args[1] {
      Expr::Integer(n) => Expr::Integer(-*n),
      Expr::Real(f) => Expr::Real(-*f),
      other => Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(other.clone()),
      },
    };
    return Ok(Expr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![args[0].clone(), neg_imag],
    });
  }

  // Try exact integer/rational complex extraction: handles numeric a + b*I patterns
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(expr) {
    return Ok(build_complex_expr(rn, rd, -in_, id));
  }

  // Try float complex extraction: handles 1.5 + 2.5*I patterns
  if let Some((re, im)) = try_extract_complex_float(expr) {
    return Ok(build_complex_float_expr(re, -im));
  }

  // I -> -I
  if let Expr::Identifier(name) = expr
    && name == "I"
  {
    return times_ast(&[Expr::Integer(-1), Expr::Identifier("I".to_string())]);
  }

  // Distribute over Plus (both FunctionCall and BinaryOp forms)
  if let Expr::FunctionCall { name, args } = expr
    && name == "Plus"
    && !args.is_empty()
  {
    let conj_args: Vec<Expr> =
      args.iter().map(conjugate_one).collect::<Result<_, _>>()?;
    return plus_ast(&conj_args);
  }
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left,
    right,
  } = expr
  {
    return plus_ast(&[conjugate_one(left)?, conjugate_one(right)?]);
  }
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Minus,
    left,
    right,
  } = expr
  {
    let conj_left = conjugate_one(left)?;
    let neg_conj_right =
      times_ast(&[Expr::Integer(-1), conjugate_one(right)?])?;
    return plus_ast(&[conj_left, neg_conj_right]);
  }

  // Distribute over Times: factor out known-real and I factors
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && !args.is_empty()
  {
    // Separate into known-real/I factors and symbolic factors
    let mut real_factors: Vec<Expr> = Vec::new();
    let mut symbolic_factors: Vec<Expr> = Vec::new();
    let mut i_count: usize = 0;

    for arg in args {
      if is_known_real(arg) {
        real_factors.push(arg.clone());
      } else if matches!(arg, Expr::Identifier(n) if n == "I") {
        i_count += 1;
      } else {
        symbolic_factors.push(arg.clone());
      }
    }

    // If we have any known factors or I to pull out
    if !real_factors.is_empty() || i_count > 0 {
      // Handle I^n: I^0=1, I^1=-I (conjugated), I^2=-1, I^3=I (conjugated)
      let i_mod = i_count % 4;
      // Build the I contribution as a single expression
      let i_factor: Option<Expr> = match i_mod {
        1 => {
          // Conjugate[I] = -I
          Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), Expr::Identifier("I".to_string())],
          })
        }
        2 => {
          // I*I = -1, Conjugate[-1] = -1
          Some(Expr::Integer(-1))
        }
        3 => {
          // I^3 = -I, Conjugate[-I] = I
          Some(Expr::Identifier("I".to_string()))
        }
        _ => None, // i_mod == 0: no I factor
      };

      // Build the conjugated symbolic part
      let conj_symbolic = if symbolic_factors.is_empty() {
        None
      } else if symbolic_factors.len() == 1 {
        Some(conjugate_one(&symbolic_factors[0])?)
      } else {
        Some(conjugate_one(&Expr::FunctionCall {
          name: "Times".to_string(),
          args: symbolic_factors,
        })?)
      };

      // Combine: real_factors * i_factor * conj_symbolic
      let mut all_factors: Vec<Expr> = real_factors;
      if let Some(ifact) = i_factor {
        all_factors.push(ifact);
      }
      if let Some(cs) = conj_symbolic {
        all_factors.push(cs);
      }

      return times_ast(&all_factors);
    }
  }
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left,
    right,
  } = expr
  {
    // Convert to flattened form and handle there
    let flat = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![*left.clone(), *right.clone()],
    };
    return conjugate_one(&flat);
  }

  // Distribute over List (both Expr::List and FunctionCall "List")
  if let Expr::List(items) = expr {
    let conj_items: Vec<Expr> =
      items.iter().map(conjugate_one).collect::<Result<_, _>>()?;
    return Ok(Expr::List(conj_items));
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "List"
  {
    let conj_args: Vec<Expr> =
      args.iter().map(conjugate_one).collect::<Result<_, _>>()?;
    return Ok(Expr::FunctionCall {
      name: "List".to_string(),
      args: conj_args,
    });
  }

  // UnaryOp Minus: Conjugate[-x] = -Conjugate[x]
  if let Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand,
  } = expr
  {
    return times_ast(&[Expr::Integer(-1), conjugate_one(operand)?]);
  }

  // Default: return unevaluated Conjugate[expr]
  Ok(Expr::FunctionCall {
    name: "Conjugate".to_string(),
    args: vec![expr.clone()],
  })
}

/// Conjugate[z] - Complex conjugate (for real numbers, returns the number itself)
pub fn conjugate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Conjugate expects exactly 1 argument".into(),
    ));
  }

  conjugate_one(&args[0])
}

/// Arg[z] - Argument (phase angle) of a complex number
pub fn arg_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Arg expects exactly 1 argument".into(),
    ));
  }

  // Try exact rational (pure real: integer or rational)
  if let Some((n, d)) = expr_to_rational(&args[0]) {
    let sign = if d > 0 { n } else { -n };
    return if sign > 0 {
      Ok(Expr::Integer(0))
    } else if sign < 0 {
      Ok(Expr::Identifier("Pi".to_string()))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Pure real float
  if let Expr::Real(f) = &args[0] {
    return if *f > 0.0 {
      Ok(Expr::Integer(0))
    } else if *f < 0.0 {
      Ok(Expr::Identifier("Pi".to_string()))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Try exact complex extraction (handles Plus, Times, I, etc.)
  if let Some(((rn, rd), (in_, id))) = try_extract_complex_exact(&args[0]) {
    // Normalize signs to have positive denominators
    let g_r = gcd(rn.abs(), rd.abs());
    let (rn, rd) = if rd < 0 {
      (-rn / g_r, -rd / g_r)
    } else {
      (rn / g_r, rd / g_r)
    };
    let g_i = gcd(in_.abs(), id.abs());
    let (in_, id) = if id < 0 {
      (-in_ / g_i, -id / g_i)
    } else {
      (in_ / g_i, id / g_i)
    };

    // Pure real
    if in_ == 0 {
      return if rn > 0 {
        Ok(Expr::Integer(0))
      } else if rn < 0 {
        Ok(Expr::Identifier("Pi".to_string()))
      } else {
        Ok(Expr::Integer(0))
      };
    }

    // Pure imaginary
    if rn == 0 {
      return if in_ > 0 {
        Ok(make_rational_times_pi(1, 2))
      } else {
        Ok(make_rational_times_pi(-1, 2))
      };
    }

    // General complex: compute ratio = |im/re| = (|in_| * rd) / (id * |rn|)
    let ratio_n = in_.abs().checked_mul(rd);
    let ratio_d = id.checked_mul(rn.abs());
    if let (Some(mut ratio_n), Some(mut ratio_d)) = (ratio_n, ratio_d) {
      let g = gcd(ratio_n.abs(), ratio_d.abs());
      ratio_n /= g;
      ratio_d /= g;

      // Try to find exact ArcTan value as a fraction of Pi
      // ArcTan[0] = 0, ArcTan[1] = Pi/4
      let arctan_pi_frac: Option<(i128, i128)> = if ratio_n == 0 {
        Some((0, 1))
      } else if ratio_n == ratio_d {
        // ArcTan[1] = Pi/4
        Some((1, 4))
      } else {
        None
      };

      let re_positive = (rn > 0 && rd > 0) || (rn < 0 && rd < 0);

      if let Some((pi_n, pi_d)) = arctan_pi_frac {
        // We have ArcTan[|ratio|] = (pi_n/pi_d) * Pi
        // Now apply sign and quadrant
        let atan_sign = if in_ > 0 { 1i128 } else { -1i128 };

        if re_positive {
          // Arg = atan_sign * (pi_n/pi_d) * Pi
          return Ok(make_rational_times_pi(atan_sign * pi_n, pi_d));
        } else {
          // re < 0: Arg = atan_sign * (Pi - (pi_n/pi_d) * Pi)
          //       = atan_sign * ((pi_d - pi_n)/pi_d) * Pi
          let result_n = pi_d - pi_n;
          return Ok(make_rational_times_pi(atan_sign * result_n, pi_d));
        }
      } else {
        // ArcTan doesn't simplify to exact Pi fraction
        // Build ArcTan[ratio] expression
        let ratio_expr = make_rational(ratio_n, ratio_d);
        let arctan_expr = Expr::FunctionCall {
          name: "ArcTan".to_string(),
          args: vec![ratio_expr],
        };

        let re_positive = (rn > 0 && rd > 0) || (rn < 0 && rd < 0);

        if re_positive {
          // Arg = sign * ArcTan[|ratio|]
          if in_ > 0 {
            return Ok(arctan_expr);
          } else {
            return Ok(negate_expr(arctan_expr));
          }
        } else {
          // re < 0, im >= 0: Pi - ArcTan[|ratio|]
          // re < 0, im < 0: -Pi + ArcTan[|ratio|]
          let pi = Expr::Identifier("Pi".to_string());
          if in_ > 0 {
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Minus,
              left: Box::new(pi),
              right: Box::new(arctan_expr),
            });
          } else {
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(negate_expr(pi)),
              right: Box::new(arctan_expr),
            });
          }
        }
      }
    }
  }

  // Try float extraction
  if let Some((re, im)) = try_extract_complex_float(&args[0])
    && (im != 0.0 || re != 0.0)
  {
    return Ok(Expr::Real(im.atan2(re)));
  }

  Ok(Expr::FunctionCall {
    name: "Arg".to_string(),
    args: args.to_vec(),
  })
}

/// Helper: create a rational multiple of Pi as an expression.
/// make_rational_times_pi(n, d) = (n/d) * Pi
fn make_rational_times_pi(n: i128, d: i128) -> Expr {
  if n == 0 {
    return Expr::Integer(0);
  }
  let g = gcd(n.abs(), d.abs());
  let (n, d) = if d < 0 {
    (-n / g, -d / g)
  } else {
    (n / g, d / g)
  };
  if d == 1 {
    if n == 1 {
      Expr::Identifier("Pi".to_string())
    } else if n == -1 {
      negate_expr(Expr::Identifier("Pi".to_string()))
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(n)),
        right: Box::new(Expr::Identifier("Pi".to_string())),
      }
    }
  } else {
    let coeff = make_rational(n, d);
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left: Box::new(coeff),
      right: Box::new(Expr::Identifier("Pi".to_string())),
    }
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
  // When tolerance is 0, we want the best possible rational approximation
  if tolerance > 0.0 {
    let approx = num as f64 / denom as f64;
    if (approx - x).abs() >= tolerance {
      return Ok(num_to_expr(x));
    }
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
    let p2 = match ai.checked_mul(p1).and_then(|v| v.checked_add(p0)) {
      Some(v) => v,
      None => break, // overflow, stop iterating
    };
    let q2 = match ai.checked_mul(q1).and_then(|v| v.checked_add(q0)) {
      Some(v) => v,
      None => break, // overflow, stop iterating
    };

    if q2 == 0 || q2 > max_denom {
      break;
    }

    let approx = p2 as f64 / q2 as f64;
    if tolerance > 0.0 {
      if (approx - x).abs() < tolerance {
        return (sign * p2, q2);
      }
    } else {
      // tolerance == 0: stop when the approximation exactly equals x as a float
      if approx == x {
        return (sign * p2, q2);
      }
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
  match (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    (Some(n), Some(k)) => Ok(Expr::Integer(binomial_coeff(n, k))),
    _ => {
      // Try Real evaluation using Gamma function
      let n_f64 = match &args[0] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      };
      let k_f64 = match &args[1] {
        Expr::Real(f) => Some(*f),
        Expr::Integer(n) => Some(*n as f64),
        _ => None,
      };
      if let (Some(n), Some(k)) = (n_f64, k_f64) {
        // Binomial[n, k] = Gamma[n+1] / (Gamma[k+1] * Gamma[n-k+1])
        // Use log-gamma for better precision
        fn lgamma(x: f64) -> f64 {
          // Lanczos approximation for log-gamma
          if x < 0.5 {
            let reflection =
              (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln();
            reflection - lgamma(1.0 - x)
          } else {
            let x = x - 1.0;
            let g = 7.0_f64;
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
            0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t
              + sum.ln()
          }
        }
        let log_result =
          lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
        let result = log_result.exp();
        Ok(Expr::Real(result))
      } else {
        Ok(Expr::FunctionCall {
          name: "Binomial".to_string(),
          args: args.to_vec(),
        })
      }
    }
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
    match expr_to_i128(arg) {
      Some(n) if n >= 0 => ints.push(n),
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "Multinomial: arguments must be non-negative integers".into(),
        ));
      }
      None => {
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
  match (
    expr_to_bigint(&args[0]),
    expr_to_bigint(&args[1]),
    expr_to_bigint(&args[2]),
  ) {
    (Some(base), Some(exp), Some(modulus)) => {
      use num_traits::Zero;
      if modulus.is_zero() {
        return Err(InterpreterError::EvaluationError(
          "PowerMod: modulus cannot be zero".into(),
        ));
      }
      if exp < BigInt::from(0) {
        // Negative exponent: need modular inverse
        // Try i128 path for mod_inverse
        use num_traits::ToPrimitive;
        match (base.to_i128(), modulus.to_i128()) {
          (Some(b), Some(m)) => {
            if let Some(inv) = mod_inverse(b, m) {
              let pos_exp = (-exp).to_u128().unwrap_or(0);
              let result =
                mod_pow_unsigned(inv as u128, pos_exp, m.unsigned_abs());
              Ok(Expr::Integer(result as i128))
            } else {
              Err(InterpreterError::EvaluationError(
                "PowerMod: modular inverse does not exist".into(),
              ))
            }
          }
          _ => Ok(Expr::FunctionCall {
            name: "PowerMod".to_string(),
            args: args.to_vec(),
          }),
        }
      } else {
        let result = base.modpow(&exp, &modulus);
        // Ensure non-negative result
        let result = ((result % &modulus) + &modulus) % &modulus;
        Ok(bigint_to_expr(result))
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
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n < 2 {
        return Ok(Expr::Integer(0));
      }
      let n_usize = n as usize;
      let mut count: i128 = 0;
      for i in 2..=n_usize {
        if crate::is_prime(i) {
          count += 1;
        }
      }
      Ok(Expr::Integer(count))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f < 2.0 {
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
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "NextPrime expects 1 or 2 arguments".into(),
    ));
  }

  let k: i128 = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(k) => *k,
      _ => {
        return Ok(Expr::FunctionCall {
          name: "NextPrime".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    1
  };

  // Handle BigInteger separately (only supports k=1 / forward)
  if let Expr::BigInteger(n) = &args[0] {
    if k == 1 {
      return Ok(bigint_to_expr(next_prime_after_bigint(n)));
    } else if k > 1 {
      let mut current = next_prime_after_bigint(n);
      for _ in 1..k {
        current = next_prime_after_bigint(&current);
      }
      return Ok(bigint_to_expr(current));
    }
    return Ok(Expr::FunctionCall {
      name: "NextPrime".to_string(),
      args: args.to_vec(),
    });
  }

  let n = match &args[0] {
    Expr::Integer(n) => *n,
    Expr::Real(f) => f.floor() as i128,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NextPrime".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if k > 0 {
    let mut current = n;
    for _ in 0..k {
      current = next_prime_after(current);
    }
    Ok(Expr::Integer(current))
  } else if k < 0 {
    let mut current = n;
    for _ in 0..(-k) {
      current = prev_prime_before(current);
    }
    Ok(Expr::Integer(current))
  } else {
    // k == 0: return unevaluated
    Ok(Expr::FunctionCall {
      name: "NextPrime".to_string(),
      args: args.to_vec(),
    })
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

/// Find the largest prime < n (including negative primes).
/// The prime sequence is: ..., -7, -5, -3, -2, 2, 3, 5, 7, 11, ...
fn prev_prime_before(n: i128) -> i128 {
  // For n > 2: search downward among positive primes
  if n > 2 {
    let mut candidate = n - 1;
    while candidate >= 2 {
      if crate::is_prime(candidate as usize) {
        return candidate;
      }
      candidate -= 1;
    }
    // No positive prime found < n, so previous is -2
    return -2;
  }
  // For n == 2 or n == 1 or n == 0 or n == -1: previous prime is -2
  if n > -2 {
    return -2;
  }
  // For n == -2: previous prime is -3
  // For n < -2: search downward among negative primes
  let mut candidate = n - 1;
  loop {
    if crate::is_prime((-candidate) as usize) {
      return candidate;
    }
    candidate -= 1;
  }
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

// ─── ModularInverse ────────────────────────────────────────────────

/// ModularInverse[a, m] - the modular inverse of a modulo m.
/// Returns k such that a*k ≡ 1 (mod m), or returns unevaluated if no inverse exists.
pub fn modular_inverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ModularInverse expects exactly 2 arguments".into(),
    ));
  }
  let a = match expr_to_bigint(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "ModularInverse".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let m = match expr_to_bigint(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "ModularInverse".to_string(),
        args: args.to_vec(),
      });
    }
  };

  use num_traits::{One, Zero};

  if m.is_zero() {
    return Ok(Expr::FunctionCall {
      name: "ModularInverse".to_string(),
      args: args.to_vec(),
    });
  }

  // Extended Euclidean algorithm
  let m_abs = if m < BigInt::zero() {
    -m.clone()
  } else {
    m.clone()
  };
  let (gcd, x, _) = extended_gcd(&a, &m_abs);
  if !gcd.is_one() && gcd != -BigInt::one() {
    // Not coprime, no inverse exists - return unevaluated
    return Ok(Expr::FunctionCall {
      name: "ModularInverse".to_string(),
      args: args.to_vec(),
    });
  }

  // Normalize result to be in range [0, |m|-1]
  let result = ((x % &m_abs) + &m_abs) % &m_abs;
  Ok(bigint_to_expr(result))
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
  use num_traits::Zero;
  if a.is_zero() {
    return (b.clone(), BigInt::zero(), BigInt::from(1));
  }
  let (g, x, y) = extended_gcd(&(b % a), a);
  (g, y - (b / a) * &x, x)
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
  match expr_to_bigint(&args[0]) {
    Some(n) => {
      use num_traits::Zero;
      let val = if n < BigInt::from(0) {
        -n - BigInt::from(1)
      } else {
        n
      };
      if val.is_zero() {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::Integer(val.bits() as i128))
      }
    }
    None => Ok(Expr::FunctionCall {
      name: "BitLength".to_string(),
      args: args.to_vec(),
    }),
  }
}

// ─── Bitwise operations ──────────────────────────────────────────

/// BitAnd[n1, n2, ...] - bitwise AND
pub fn bit_and_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r & &n,
          None => n,
        });
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "BitAnd".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitAnd".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// BitOr[n1, n2, ...] - bitwise OR
pub fn bit_or_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r | &n,
          None => n,
        });
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "BitOr".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitOr".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// BitXor[n1, n2, ...] - bitwise XOR
pub fn bit_xor_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut result: Option<BigInt> = None;
  for arg in args {
    match expr_to_bigint(arg) {
      Some(n) => {
        result = Some(match result {
          Some(r) => r ^ &n,
          None => n,
        });
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "BitXor".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  match result {
    Some(n) => Ok(bigint_to_expr(n)),
    None => Ok(Expr::FunctionCall {
      name: "BitXor".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// BitNot[n] - bitwise NOT (complement)
pub fn bit_not_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "BitNot expects exactly 1 argument".into(),
    ));
  }
  match expr_to_bigint(&args[0]) {
    Some(n) => Ok(bigint_to_expr(!n)),
    None => Ok(Expr::FunctionCall {
      name: "BitNot".to_string(),
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
  let n = match expr_to_i128(&args[0]) {
    Some(n) => n,
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntegerExponent".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) => b,
      None => {
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

// ─── MixedFractionParts ────────────────────────────────────────────

pub fn mixed_fraction_parts_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "MixedFractionParts expects exactly 1 argument".into(),
    ));
  }
  let int_part = integer_part_ast(args)?;
  let frac_part = fractional_part_ast(args)?;
  Ok(Expr::List(vec![int_part, frac_part]))
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
        // Try float path first
        let mut vals = Vec::new();
        let mut all_numeric = true;
        for item in items {
          if let Some(v) = expr_to_num(item) {
            vals.push(v);
          } else {
            all_numeric = false;
            break;
          }
        }
        if all_numeric && !vals.is_empty() {
          let n = vals.len() as f64;
          let mean = vals.iter().sum::<f64>() / n;
          let var =
            vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
          return Ok(num_to_expr(var));
        }
        // Check for list-of-lists → compute column-wise
        if items.iter().all(|item| matches!(item, Expr::List(_))) {
          return variance_columnwise(items);
        }
        // Symbolic/complex path: compute Variance = Sum[Abs[xi - mean]^2] / (n-1)
        return variance_symbolic(items);
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

/// Compute variance symbolically
fn variance_symbolic(items: &[Expr]) -> Result<Expr, InterpreterError> {
  let n = items.len();
  if n < 2 {
    return Err(InterpreterError::EvaluationError(
      "Variance: need at least 2 elements".into(),
    ));
  }
  // Compute mean symbolically
  let mean = mean_ast(&[Expr::List(items.to_vec())])?;
  // Compute Sum[Abs[xi - mean]^2] / (n-1)
  let mut sum_sq_terms = Vec::new();
  for item in items {
    // (xi - mean)
    let diff = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        item.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), mean.clone()],
        },
      ],
    };
    // Abs[xi - mean]^2
    let abs_sq = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Abs".to_string(),
          args: vec![diff],
        },
        Expr::Integer(2),
      ],
    };
    sum_sq_terms.push(abs_sq);
  }
  let sum_sq = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: sum_sq_terms,
  };
  let result = Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      sum_sq,
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![Expr::Integer((n - 1) as i128), Expr::Integer(-1)],
      },
    ],
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Variance of columns in a list-of-lists (matrix)
fn variance_columnwise(rows: &[Expr]) -> Result<Expr, InterpreterError> {
  let row_vecs: Vec<&Vec<Expr>> = rows
    .iter()
    .filter_map(|r| {
      if let Expr::List(items) = r {
        Some(items)
      } else {
        None
      }
    })
    .collect();
  if row_vecs.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Variance".to_string(),
      args: vec![Expr::List(rows.to_vec())],
    });
  }
  let ncols = row_vecs[0].len();
  let mut col_vars = Vec::new();
  for col in 0..ncols {
    let col_items: Vec<Expr> = row_vecs
      .iter()
      .map(|r| {
        if col < r.len() {
          r[col].clone()
        } else {
          Expr::Integer(0)
        }
      })
      .collect();
    let var_result = variance_ast(&[Expr::List(col_items)])?;
    col_vars.push(var_result);
  }
  Ok(Expr::List(col_vars))
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
  // For list-of-lists, the variance returns a list of column variances
  let var = variance_ast(args)?;
  match &var {
    Expr::List(items) => {
      // Apply Sqrt to each element
      let mut results = Vec::new();
      for item in items {
        results.push(sqrt_ast(&[item.clone()])?);
      }
      Ok(Expr::List(results))
    }
    Expr::Integer(_) | Expr::Real(_) | Expr::FunctionCall { .. } => {
      sqrt_ast(&[var.clone()])
    }
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

/// Covariance[list1, list2] - Sample covariance of two numeric lists
pub fn covariance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Covariance".to_string(),
      args: args.to_vec(),
    });
  }
  let (xs, ys) = match (&args[0], &args[1]) {
    (Expr::List(xs), Expr::List(ys))
      if xs.len() == ys.len() && xs.len() >= 2 =>
    {
      (xs, ys)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Covariance".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut x_vals = Vec::new();
  let mut y_vals = Vec::new();
  for (x, y) in xs.iter().zip(ys.iter()) {
    match (expr_to_num(x), expr_to_num(y)) {
      (Some(xv), Some(yv)) => {
        x_vals.push(xv);
        y_vals.push(yv);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Covariance".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  let n = x_vals.len() as f64;
  let mean_x = x_vals.iter().sum::<f64>() / n;
  let mean_y = y_vals.iter().sum::<f64>() / n;
  let cov: f64 = x_vals
    .iter()
    .zip(y_vals.iter())
    .map(|(x, y)| (x - mean_x) * (y - mean_y))
    .sum::<f64>()
    / (n - 1.0);
  Ok(num_to_expr(cov))
}

/// Correlation[list1, list2] - Pearson correlation coefficient
pub fn correlation_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "Correlation".to_string(),
      args: args.to_vec(),
    });
  }
  let (xs, ys) = match (&args[0], &args[1]) {
    (Expr::List(xs), Expr::List(ys))
      if xs.len() == ys.len() && xs.len() >= 2 =>
    {
      (xs, ys)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Correlation".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut x_vals = Vec::new();
  let mut y_vals = Vec::new();
  for (x, y) in xs.iter().zip(ys.iter()) {
    match (expr_to_num(x), expr_to_num(y)) {
      (Some(xv), Some(yv)) => {
        x_vals.push(xv);
        y_vals.push(yv);
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "Correlation".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }
  let n = x_vals.len() as f64;
  let mean_x = x_vals.iter().sum::<f64>() / n;
  let mean_y = y_vals.iter().sum::<f64>() / n;
  let cov: f64 = x_vals
    .iter()
    .zip(y_vals.iter())
    .map(|(x, y)| (x - mean_x) * (y - mean_y))
    .sum::<f64>();
  let var_x: f64 = x_vals.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
  let var_y: f64 = y_vals.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
  let denom = (var_x * var_y).sqrt();
  if denom == 0.0 {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  Ok(num_to_expr(cov / denom))
}

/// CentralMoment[list, r] - r-th central moment of a numeric list
pub fn central_moment_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "CentralMoment".to_string(),
      args: args.to_vec(),
    });
  }
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let r = match expr_to_num(&args[1]) {
    Some(r) => r as i32,
    None => {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let mut vals = Vec::new();
  for item in items {
    if let Some(v) = expr_to_num(item) {
      vals.push(v);
    } else {
      return Ok(Expr::FunctionCall {
        name: "CentralMoment".to_string(),
        args: args.to_vec(),
      });
    }
  }
  let n = vals.len() as f64;
  let mean = vals.iter().sum::<f64>() / n;
  let moment: f64 = vals.iter().map(|x| (x - mean).powi(r)).sum::<f64>() / n;
  Ok(num_to_expr(moment))
}

/// Kurtosis[list] - CentralMoment[list, 4] / CentralMoment[list, 2]^2
pub fn kurtosis_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec(),
    });
  }
  let m4 = central_moment_ast(&[args[0].clone(), Expr::Integer(4)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  match (expr_to_num(&m4), expr_to_num(&m2)) {
    (Some(m4v), Some(m2v)) if m2v != 0.0 => Ok(num_to_expr(m4v / (m2v * m2v))),
    _ => Ok(Expr::FunctionCall {
      name: "Kurtosis".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Skewness[list] - CentralMoment[list, 3] / CentralMoment[list, 2]^(3/2)
pub fn skewness_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
      args: args.to_vec(),
    });
  }
  let m3 = central_moment_ast(&[args[0].clone(), Expr::Integer(3)])?;
  let m2 = central_moment_ast(&[args[0].clone(), Expr::Integer(2)])?;
  match (expr_to_num(&m3), expr_to_num(&m2)) {
    (Some(m3v), Some(m2v)) if m2v != 0.0 => {
      Ok(num_to_expr(m3v / m2v.powf(1.5)))
    }
    _ => Ok(Expr::FunctionCall {
      name: "Skewness".to_string(),
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

  let base_i128 = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "IntegerLength: base must be at least 2".into(),
        ));
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntegerLength".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    10
  };

  // Try BigInt path first (handles both Integer and BigInteger)
  if let Some(n) = expr_to_bigint(&args[0]) {
    use num_traits::Zero;
    if n.is_zero() {
      return Ok(Expr::Integer(0));
    }
    let base_big = BigInt::from(base_i128);
    let mut abs_n = if n < BigInt::zero() { -n } else { n };
    let mut count = 0i128;
    while abs_n > BigInt::zero() {
      abs_n /= &base_big;
      count += 1;
    }
    return Ok(Expr::Integer(count));
  }

  Ok(Expr::FunctionCall {
    name: "IntegerLength".to_string(),
    args: args.to_vec(),
  })
}

/// IntegerReverse[n] - reverse the digits of an integer in base 10.
/// IntegerReverse[n, b] - reverse the digits of n in base b.
pub fn integer_reverse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerReverse expects 1 or 2 arguments".into(),
    ));
  }

  let base = if args.len() == 2 {
    match expr_to_i128(&args[1]) {
      Some(b) if b >= 2 => b,
      Some(_) => {
        return Err(InterpreterError::EvaluationError(
          "IntegerReverse: base must be at least 2".into(),
        ));
      }
      None => {
        return Ok(Expr::FunctionCall {
          name: "IntegerReverse".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    10
  };

  // Handle BigInteger
  if let Some(n) = expr_to_bigint(&args[0]) {
    use num_traits::Zero;
    let mut abs_n = if n < BigInt::zero() { -n } else { n };
    let base_big = BigInt::from(base);
    let mut result = BigInt::zero();
    while abs_n > BigInt::zero() {
      result = result * &base_big + (&abs_n % &base_big);
      abs_n /= &base_big;
    }
    return Ok(bigint_to_expr(result));
  }

  Ok(Expr::FunctionCall {
    name: "IntegerReverse".to_string(),
    args: args.to_vec(),
  })
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
/// Norm[v] - Euclidean norm (L2) of a vector
/// Norm[v, p] - Lp norm
pub fn norm_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Norm expects 1 or 2 arguments".into(),
    ));
  }
  // Determine the norm parameter p
  let p_expr = if args.len() == 2 {
    Some(args[1].clone())
  } else {
    None
  };
  let p_val = match &p_expr {
    Some(e) => try_eval_to_f64(e),
    None => Some(2.0),
  };
  // Check for Infinity norm
  let is_infinity = match &p_expr {
    Some(Expr::Identifier(s)) if s == "Infinity" => true,
    Some(Expr::FunctionCall { name, args })
      if name == "DirectedInfinity" && args.len() == 1 =>
    {
      true
    }
    _ => p_val == Some(f64::INFINITY),
  };

  match &args[0] {
    Expr::List(items) => {
      let mut vals = Vec::new();
      let mut all_numeric = true;
      for item in items {
        match try_eval_to_f64(item) {
          Some(v) => vals.push(v),
          None => {
            all_numeric = false;
            break;
          }
        }
      }

      let p = p_val.unwrap_or(2.0);

      if all_numeric {
        if p == 1.0 {
          let result: f64 = vals.iter().map(|x| x.abs()).sum();
          return Ok(num_to_expr(result));
        }
        if is_infinity {
          let result = vals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
          return Ok(num_to_expr(result));
        }
        let sum: f64 = vals.iter().map(|x| x.abs().powf(p)).sum();
        let result = sum.powf(1.0 / p);
        if p == 2.0 && items.iter().all(|i| matches!(i, Expr::Integer(_))) {
          let sum_sq: i128 = items
            .iter()
            .filter_map(|i| {
              if let Expr::Integer(n) = i {
                Some(n * n)
              } else {
                None
              }
            })
            .sum();
          let root = (sum_sq as f64).sqrt() as i128;
          if root * root == sum_sq {
            return Ok(Expr::Integer(root));
          }
          return Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Expr::Integer(sum_sq)],
          });
        }
        Ok(num_to_expr(result))
      } else {
        // Symbolic vector: build symbolic norm expression
        if is_infinity {
          // Max[Abs[x], Abs[y], ...]
          let abs_items: Vec<Expr> = items
            .iter()
            .map(|item| Expr::FunctionCall {
              name: "Abs".to_string(),
              args: vec![item.clone()],
            })
            .collect();
          Ok(Expr::FunctionCall {
            name: "Max".to_string(),
            args: abs_items,
          })
        } else if p == 2.0 {
          // Sqrt[Abs[x]^2 + Abs[y]^2 + ...]
          let abs_sq_items: Vec<Expr> = items
            .iter()
            .map(|item| Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![
                Expr::FunctionCall {
                  name: "Abs".to_string(),
                  args: vec![item.clone()],
                },
                Expr::Integer(2),
              ],
            })
            .collect();
          let sum = if abs_sq_items.len() == 1 {
            abs_sq_items.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: abs_sq_items,
            }
          };
          Ok(Expr::FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![sum],
          })
        } else {
          Ok(Expr::FunctionCall {
            name: "Norm".to_string(),
            args: args.to_vec(),
          })
        }
      }
    }
    // Norm of a scalar: Abs[x]
    _ => {
      if let Some(v) = try_eval_to_f64(&args[0]) {
        Ok(num_to_expr(v.abs()))
      } else {
        // Evaluate Abs[x] (handles complex numbers etc.)
        crate::evaluator::evaluate_function_call_ast("Abs", &[args[0].clone()])
      }
    }
  }
}

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
            // Symbolic case: return {elem/Sqrt[sum_of_squares], ...}
            // like Mathematica does for Normalize[{a, b}] → {a/Sqrt[a^2+b^2], b/Sqrt[a^2+b^2]}
            let squared_terms: Vec<Expr> = items
              .iter()
              .map(|e| Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(e.clone()),
                right: Box::new(Expr::Integer(2)),
              })
              .collect();
            let sum_of_squares = if squared_terms.len() == 1 {
              squared_terms.into_iter().next().unwrap()
            } else {
              Expr::FunctionCall {
                name: "Plus".to_string(),
                args: squared_terms,
              }
            };
            let norm_expr = Expr::FunctionCall {
              name: "Sqrt".to_string(),
              args: vec![sum_of_squares],
            };
            let result: Vec<Expr> = items
              .iter()
              .map(|e| Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Divide,
                left: Box::new(e.clone()),
                right: Box::new(norm_expr.clone()),
              })
              .collect();
            return Ok(Expr::List(result));
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
    Expr::Constant(c) => match c.as_str() {
      "Pi" | "E" | "Degree" => Ok(Expr::Integer(1)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec(),
      }),
    },
    Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(1)),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
        Ok(Expr::Integer(0))
      }
      Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(0)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec(),
      }),
    },
    // Times[-1, x] pattern (e.g. -Pi parses as Times[-1, Pi])
    Expr::FunctionCall { name, args: fargs }
      if name == "Times"
        && fargs.len() == 2
        && matches!(fargs[0], Expr::Integer(-1)) =>
    {
      match &fargs[1] {
        Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
          Ok(Expr::Integer(0))
        }
        Expr::Identifier(n) if n == "Infinity" => Ok(Expr::Integer(0)),
        _ => Ok(Expr::FunctionCall {
          name: "UnitStep".to_string(),
          args: args.to_vec(),
        }),
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(-1)) => match right.as_ref() {
      Expr::Constant(c) if matches!(c.as_str(), "Pi" | "E" | "Degree") => {
        Ok(Expr::Integer(0))
      }
      Expr::Identifier(n) if n == "Infinity" => Ok(Expr::Integer(0)),
      _ => Ok(Expr::FunctionCall {
        name: "UnitStep".to_string(),
        args: args.to_vec(),
      }),
    },
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

/// Precision[x] - the number of significant decimal digits
/// Returns MachinePrecision for machine reals, Infinity for exact numbers
pub fn precision_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Precision expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::Real(_) => Ok(Expr::Identifier("MachinePrecision".to_string())),
    Expr::BigFloat(_, prec) => Ok(Expr::Real(*prec as f64)),
    Expr::Identifier(name)
      if name == "Infinity" || name == "ComplexInfinity" =>
    {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      ..
    } => {
      // Exact rationals like 1/2 have infinite precision
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::List(items) => {
      // Precision of a list is the minimum precision of its elements
      let mut min_prec: Option<f64> = None;
      for item in items {
        let p = precision_ast(&[item.clone()])?;
        match p {
          Expr::Identifier(name) if name == "Infinity" => {}
          Expr::Identifier(name) if name == "MachinePrecision" => {
            let mp = 15.954589770191003; // $MachinePrecision
            min_prec = Some(min_prec.map_or(mp, |v: f64| v.min(mp)));
          }
          Expr::Real(f) => {
            min_prec = Some(min_prec.map_or(f, |v: f64| v.min(f)));
          }
          _ => {}
        }
      }
      match min_prec {
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(Expr::Identifier("Infinity".to_string())),
      }
    }
    // For symbolic expressions, check if any subexpression has finite precision
    Expr::FunctionCall { args: fargs, .. } => {
      let mut min_prec: Option<f64> = None;
      for arg in fargs {
        let p = precision_ast(&[arg.clone()])?;
        match p {
          Expr::Identifier(name) if name == "Infinity" => {}
          Expr::Identifier(name) if name == "MachinePrecision" => {
            let mp = 15.954589770191003;
            min_prec = Some(min_prec.map_or(mp, |v: f64| v.min(mp)));
          }
          Expr::Real(f) => {
            min_prec = Some(min_prec.map_or(f, |v: f64| v.min(f)));
          }
          _ => {}
        }
      }
      match min_prec {
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(Expr::Identifier("Infinity".to_string())),
      }
    }
    _ => Ok(Expr::Identifier("Infinity".to_string())),
  }
}

/// Accuracy[x] - the number of significant decimal digits to the right of the decimal point
/// Returns Infinity for exact numbers, computes from precision for approximate numbers
pub fn accuracy_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Accuracy expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Constant(_) => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::Real(f) => {
      // Accuracy = MachinePrecision - Log10[Abs[x]]
      // MachinePrecision ≈ 15.9546
      let machine_precision = 15.954589770191003_f64;
      if *f == 0.0 {
        // Accuracy[0.] is very large (represents machine epsilon)
        return Ok(Expr::Real(machine_precision + (2.0_f64).powi(52).log10()));
      }
      let accuracy = machine_precision - f.abs().log10();
      Ok(Expr::Real(accuracy))
    }
    Expr::Identifier(name)
      if name == "Infinity"
        || name == "ComplexInfinity"
        || name == "Indeterminate" =>
    {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      ..
    } => Ok(Expr::Identifier("Infinity".to_string())),
    // Symbolic identifiers (variables) have infinite accuracy
    Expr::Identifier(_) => Ok(Expr::Identifier("Infinity".to_string())),
    // For symbolic expressions, check subexpressions
    Expr::FunctionCall { args: fargs, .. } => {
      let mut has_finite = false;
      for arg in fargs {
        let a = accuracy_ast(&[arg.clone()])?;
        if !matches!(&a, Expr::Identifier(n) if n == "Infinity") {
          has_finite = true;
          break;
        }
      }
      if has_finite {
        Ok(Expr::FunctionCall {
          name: "Accuracy".to_string(),
          args: args.to_vec(),
        })
      } else {
        Ok(Expr::Identifier("Infinity".to_string()))
      }
    }
    _ => Ok(Expr::Identifier("Infinity".to_string())),
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
      e if expr_to_i128(e).is_some_and(|n| n > 0) => {
        nums.push(expr_to_i128(e).unwrap())
      }
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

// ── IntegerName ──────────────────────────────────────────────────────────

const ONES: [&str; 20] = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
];

const TENS: [&str; 10] = [
  "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
  "ninety",
];

const SCALES: [&str; 7] = [
  "",
  "thousand",
  "million",
  "billion",
  "trillion",
  "quadrillion",
  "quintillion",
];

/// Spell out a number 0..=999 in English words.
/// Uses U+2010 HYPHEN for compound numbers like "twenty‐one".
fn spell_below_1000(n: u64) -> String {
  if n == 0 {
    return String::new();
  }
  let mut parts = Vec::new();
  let hundreds = n / 100;
  let remainder = n % 100;
  if hundreds > 0 {
    parts.push(format!("{} hundred", ONES[hundreds as usize]));
  }
  if remainder > 0 {
    if remainder < 20 {
      parts.push(ONES[remainder as usize].to_string());
    } else {
      let tens = remainder / 10;
      let ones = remainder % 10;
      if ones == 0 {
        parts.push(TENS[tens as usize].to_string());
      } else {
        // U+2010 HYPHEN between tens and ones
        parts.push(format!(
          "{}\u{2010}{}",
          TENS[tens as usize], ONES[ones as usize]
        ));
      }
    }
  }
  parts.join(" ")
}

pub fn integer_name_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // IntegerName[n] - convert integer to English name
  // IntegerName also works on lists
  if args.len() == 1
    && let Expr::List(items) = &args[0]
  {
    let results: Result<Vec<Expr>, InterpreterError> = items
      .iter()
      .map(|item| integer_name_ast(&[item.clone()]))
      .collect();
    return Ok(Expr::List(results?));
  }

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "IntegerName".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let negative = n < 0;
  let abs_n = n.unsigned_abs();

  if abs_n == 0 {
    return Ok(Expr::String("zero".to_string()));
  }

  // For numbers 1..=999, spell out entirely in words
  if abs_n <= 999 {
    let word = spell_below_1000(abs_n as u64);
    let result = if negative {
      format!("negative {}", word)
    } else {
      word
    };
    return Ok(Expr::String(result));
  }

  // For numbers >= 1000, break into groups of 3 digits.
  // Higher groups use digit representation, the lowest group (< 1000) uses words.
  let mut groups: Vec<(u64, usize)> = Vec::new(); // (group_value, scale_index)
  let mut remaining = abs_n as u64;
  let mut scale_idx = 0;
  while remaining > 0 {
    let group = remaining % 1000;
    if group > 0 {
      groups.push((group, scale_idx));
    }
    remaining /= 1000;
    scale_idx += 1;
  }
  groups.reverse();

  let mut parts = Vec::new();
  for &(group, sidx) in &groups {
    if sidx == 0 {
      // Lowest group: use digits (for numbers >= 1000)
      parts.push(format!("{}", group));
    } else {
      // Higher groups: use digits + scale word
      let scale = if sidx < SCALES.len() {
        SCALES[sidx]
      } else {
        return Ok(Expr::FunctionCall {
          name: "IntegerName".to_string(),
          args: args.to_vec(),
        });
      };
      parts.push(format!("{} {}", group, scale));
    }
  }

  let result = parts.join(" ");
  let result = if negative {
    format!("negative {}", result)
  } else {
    result
  };
  Ok(Expr::String(result))
}

pub fn roman_numeral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // RomanNumeral[n] - convert integer to Roman numeral string
  // Returns a Symbol (not a String) — e.g. RomanNumeral[2025] => MMXXV
  // Negative integers: convert the absolute value
  // Zero: returns N
  // Non-integer: return unevaluated

  let n = match expr_to_i128(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "RomanNumeral".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if n == 0 {
    return Ok(Expr::Identifier("N".to_string()));
  }

  let abs_n = n.unsigned_abs() as u64;

  // For values >= 5000, Wolfram uses display forms with overscript bars.
  // We support up to 4999 with plain Roman numerals.
  if abs_n >= 5000 {
    return Ok(Expr::FunctionCall {
      name: "RomanNumeral".to_string(),
      args: args.to_vec(),
    });
  }

  const VALUES: [(u64, &str); 13] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
  ];

  let mut result = String::new();
  let mut remaining = abs_n;
  for &(value, numeral) in &VALUES {
    while remaining >= value {
      result.push_str(numeral);
      remaining -= value;
    }
  }

  Ok(Expr::Identifier(result))
}

/// GCD for i128 values
fn gcd_i128(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Extract numerator from Integer or Rational expr
fn expr_numerator(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(n) => Some(*n),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0] {
        Some(*n)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Extract denominator from Integer or Rational expr
fn expr_denominator(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(_) => Some(1),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(d) = &args[1] {
        Some(*d)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Floor of a rational number num/den
fn rational_floor(num: i128, den: i128) -> i128 {
  if den == 0 {
    return 0; // shouldn't happen, caller checks
  }
  // Normalize sign so den > 0
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if num >= 0 {
    num / den
  } else {
    // For negative: floor division
    (num - den + 1) / den
  }
}

/// Ceiling of a rational number num/den
fn rational_ceil(num: i128, den: i128) -> i128 {
  if den == 0 {
    return 0;
  }
  let (num, den) = if den < 0 { (-num, -den) } else { (num, den) };
  if num >= 0 {
    (num + den - 1) / den
  } else {
    num / den
  }
}

/// Quantile[list, q] - the q-th quantile of the list
/// Quantile[list, {q1, q2, ...}] - multiple quantiles
pub fn quantile_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Quantile expects exactly 2 arguments".into(),
    ));
  }
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Sort the items numerically
  let mut sorted: Vec<&Expr> = items.iter().collect();
  sorted.sort_by(|a, b| {
    let fa = try_eval_to_f64(a);
    let fb = try_eval_to_f64(b);
    fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
  });

  // Handle list of quantiles
  if let Expr::List(qs) = &args[1] {
    let results: Result<Vec<Expr>, _> =
      qs.iter().map(|q| quantile_single(&sorted, q)).collect();
    return Ok(Expr::List(results?));
  }

  quantile_single(&sorted, &args[1])
}

fn quantile_single(
  sorted: &[&Expr],
  q: &Expr,
) -> Result<Expr, InterpreterError> {
  let n = sorted.len();
  // Default Quantile uses Type 1 (inverse of CDF)
  // Index = Ceiling[q * n]
  let q_val = match q {
    Expr::Integer(n) => *n as f64,
    Expr::Real(f) => *f,
    Expr::FunctionCall { name, args: rargs }
      if name == "Rational" && rargs.len() == 2 =>
    {
      if let (Expr::Integer(num), Expr::Integer(den)) = (&rargs[0], &rargs[1]) {
        *num as f64 / *den as f64
      } else {
        return Ok(Expr::FunctionCall {
          name: "Quantile".to_string(),
          args: vec![
            Expr::List(sorted.iter().cloned().cloned().collect()),
            q.clone(),
          ],
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Quantile".to_string(),
        args: vec![
          Expr::List(sorted.iter().cloned().cloned().collect()),
          q.clone(),
        ],
      });
    }
  };

  let idx = (q_val * n as f64).ceil() as usize;
  let idx = idx.max(1).min(n);
  Ok(sorted[idx - 1].clone())
}

// ─── PowerExpand ──────────────────────────────────────────────────────

/// PowerExpand[expr] - expand powers of products and powers
/// Rules: (a^b)^c -> a^(b*c), (a*b)^c -> a^c * b^c
pub fn power_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PowerExpand expects exactly 1 argument".into(),
    ));
  }
  Ok(power_expand_recursive(&args[0]))
}

fn power_expand_recursive(expr: &Expr) -> Expr {
  // Helper to extract (base, exponent) from any Power representation
  let extract_power = |e: &Expr| -> Option<(Expr, Expr)> {
    match e {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left,
        right,
      } => Some((*left.clone(), *right.clone())),
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        Some((args[0].clone(), args[1].clone()))
      }
      // Sqrt[x] = Power[x, 1/2]
      Expr::FunctionCall { name, args }
        if name == "Sqrt" && args.len() == 1 =>
      {
        Some((
          args[0].clone(),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
        ))
      }
      _ => None,
    }
  };

  // Helper to extract Times factors
  let extract_times = |e: &Expr| -> Option<Vec<Expr>> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        Some(args.clone())
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => Some(vec![*left.clone(), *right.clone()]),
      _ => None,
    }
  };

  match expr {
    _ if extract_power(expr).is_some() => {
      let (raw_base, raw_exp) = extract_power(expr).unwrap();
      let base = power_expand_recursive(&raw_base);
      let exp = power_expand_recursive(&raw_exp);

      // (a^b)^c -> a^(b*c)
      if let Some((inner_base, inner_exp)) = extract_power(&base) {
        let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
          Ok(r) => r,
          Err(_) => Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(inner_exp),
            right: Box::new(exp),
          },
        };
        return match crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(inner_base),
          right: Box::new(new_exp),
        }) {
          Ok(r) => r,
          Err(_) => expr.clone(),
        };
      }

      // (a*b*...)^c -> a^c * b^c * ...
      if let Some(factors) = extract_times(&base) {
        let expanded: Vec<Expr> = factors
          .iter()
          .map(|factor| Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: Box::new(factor.clone()),
            right: Box::new(exp.clone()),
          })
          .collect();
        return match times_ast(&expanded) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Times".to_string(),
            args: expanded,
          },
        };
      }

      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      }
    }
    Expr::FunctionCall { name, args } => {
      let new_args: Vec<Expr> =
        args.iter().map(power_expand_recursive).collect();
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(power_expand_recursive(left)),
      right: Box::new(power_expand_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(power_expand_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(power_expand_recursive).collect())
    }
    _ => expr.clone(),
  }
}

// ─── Variables ──────────────────────────────────────────────────────

/// Variables[expr] - list of variables in a polynomial expression
pub fn variables_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Variables expects exactly 1 argument".into(),
    ));
  }
  let mut vars = Vec::new();
  collect_variables(&args[0], &mut vars);
  // Sort canonically and deduplicate
  vars.sort_by(|a, b| {
    let ord = crate::functions::list_helpers_ast::compare_exprs(a, b);
    if ord > 0 {
      std::cmp::Ordering::Less
    } else if ord < 0 {
      std::cmp::Ordering::Greater
    } else {
      std::cmp::Ordering::Equal
    }
  });
  vars.dedup_by(|a, b| {
    crate::syntax::expr_to_string(a) == crate::syntax::expr_to_string(b)
  });
  Ok(Expr::List(vars))
}

fn collect_variables(expr: &Expr, vars: &mut Vec<Expr>) {
  match expr {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::String(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => {}
    Expr::Identifier(s) => {
      // Skip built-in constants
      if !matches!(
        s.as_str(),
        "True"
          | "False"
          | "Null"
          | "Pi"
          | "E"
          | "I"
          | "Infinity"
          | "ComplexInfinity"
          | "Indeterminate"
      ) {
        vars.push(expr.clone());
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_variables(item, vars);
      }
    }
    Expr::BinaryOp { op, left, right } => {
      match op {
        crate::syntax::BinaryOperator::Plus
        | crate::syntax::BinaryOperator::Minus
        | crate::syntax::BinaryOperator::Times
        | crate::syntax::BinaryOperator::Power
        | crate::syntax::BinaryOperator::Divide => {
          collect_variables(left, vars);
          collect_variables(right, vars);
        }
        _ => {
          // Treat as atomic variable-like term
          vars.push(expr.clone());
        }
      }
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Plus" | "Times" | "Power" | "Rational" => {
          for arg in args {
            collect_variables(arg, vars);
          }
        }
        _ => {
          // Non-polynomial function (like Sin, Cos) — treat as variable
          vars.push(expr.clone());
        }
      }
    }
    Expr::UnaryOp { op: _, operand } => {
      collect_variables(operand, vars);
    }
    _ => {
      vars.push(expr.clone());
    }
  }
}

/// LinearRecurrence[ker, init, n] - generates a linear recurrence sequence of length n.
/// LinearRecurrence[ker, init, {nmin, nmax}] - returns elements nmin through nmax.
/// LinearRecurrence[ker, init, {n}] - returns the list containing only the nth element.
pub fn linear_recurrence_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "LinearRecurrence".to_string(),
      args: args.to_vec(),
    });
  }

  let kernel = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let init = match &args[1] {
    Expr::List(items) => items.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse the third argument: n, {n}, or {nmin, nmax}
  let (total_n, range) = match &args[2] {
    Expr::Integer(n) => (*n as usize, None),
    Expr::List(items) if items.len() == 1 => {
      if let Some(n) = expr_to_i128(&items[0]) {
        (n as usize, Some((n as usize, n as usize)))
      } else {
        return Ok(Expr::FunctionCall {
          name: "LinearRecurrence".to_string(),
          args: args.to_vec(),
        });
      }
    }
    Expr::List(items) if items.len() == 2 => {
      if let (Some(nmin), Some(nmax)) =
        (expr_to_i128(&items[0]), expr_to_i128(&items[1]))
      {
        (nmax as usize, Some((nmin as usize, nmax as usize)))
      } else {
        return Ok(Expr::FunctionCall {
          name: "LinearRecurrence".to_string(),
          args: args.to_vec(),
        });
      }
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LinearRecurrence".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut seq = init.clone();

  // Extend sequence to total_n elements
  while seq.len() < total_n {
    let mut next = Expr::Integer(0);
    for (i, coeff) in kernel.iter().enumerate() {
      let idx = seq.len() - 1 - i;
      let term = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(seq[idx].clone()),
      };
      next = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(next),
        right: Box::new(term),
      };
    }
    // Evaluate the expression to simplify
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&next)?;
    seq.push(evaluated);
  }

  match range {
    None => Ok(Expr::List(seq[..total_n].to_vec())),
    Some((nmin, nmax)) => Ok(Expr::List(seq[nmin - 1..nmax].to_vec())),
  }
}

/// EuclideanDistance[u, v] - Euclidean distance between two points
pub fn euclidean_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "EuclideanDistance: vectors must have the same length".into(),
        ));
      }
      // Build Sqrt[Sum[Abs[u_i - v_i]^2]]
      let mut sum_args = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs_diff =
          crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        let sq = crate::evaluator::evaluate_function_call_ast(
          "Power",
          &[abs_diff, Expr::Integer(2)],
        )?;
        sum_args.push(sq);
      }
      let sum =
        crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)?;
      crate::evaluator::evaluate_function_call_ast("Sqrt", &[sum])
    }
    _ => {
      // Scalar distance: Abs[u - v]
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
    }
  }
}

/// SquaredEuclideanDistance[u, v] - squared Euclidean distance
pub fn squared_euclidean_distance_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "SquaredEuclideanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "SquaredEuclideanDistance: vectors must have the same length".into(),
        ));
      }
      let mut sum_args = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs_diff =
          crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        let sq = crate::evaluator::evaluate_function_call_ast(
          "Power",
          &[abs_diff, Expr::Integer(2)],
        )?;
        sum_args.push(sq);
      }
      crate::evaluator::evaluate_function_call_ast("Plus", &sum_args)
    }
    _ => {
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      let abs_diff =
        crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
      crate::evaluator::evaluate_function_call_ast(
        "Power",
        &[abs_diff, Expr::Integer(2)],
      )
    }
  }
}

/// ManhattanDistance[u, v] - Manhattan (L1) distance between two points
pub fn manhattan_distance_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ManhattanDistance expects exactly 2 arguments".into(),
    ));
  }

  match (&args[0], &args[1]) {
    (Expr::List(u), Expr::List(v)) => {
      if u.len() != v.len() {
        return Err(InterpreterError::EvaluationError(
          "ManhattanDistance: vectors must have the same length".into(),
        ));
      }
      let mut abs_diffs = Vec::new();
      for (ui, vi) in u.iter().zip(v.iter()) {
        let diff = crate::evaluator::evaluate_function_call_ast(
          "Subtract",
          &[ui.clone(), vi.clone()],
        )?;
        let abs = crate::evaluator::evaluate_function_call_ast("Abs", &[diff])?;
        abs_diffs.push(abs);
      }
      crate::evaluator::evaluate_function_call_ast("Plus", &abs_diffs)
    }
    _ => {
      // Scalar distance: Abs[u - v]
      let diff = crate::evaluator::evaluate_function_call_ast(
        "Subtract",
        &[args[0].clone(), args[1].clone()],
      )?;
      crate::evaluator::evaluate_function_call_ast("Abs", &[diff])
    }
  }
}
