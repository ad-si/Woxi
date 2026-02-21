#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;
use num_bigint::BigInt;

/// Helper - constants are kept symbolic, no direct f64 conversion in expr_to_num.
pub fn constant_to_f64(_name: &str) -> Option<f64> {
  None
}

pub fn expr_to_num(expr: &Expr) -> Option<f64> {
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
pub fn erfc_cf(x: f64) -> f64 {
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
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_f64()
    }
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
pub fn expr_to_i128(e: &Expr) -> Option<i128> {
  use num_traits::ToPrimitive;
  match e {
    Expr::Integer(n) => Some(*n),
    Expr::BigInteger(n) => n.to_i128(),
    _ => None,
  }
}

/// Extract a BigInt from Integer or BigInteger.
pub fn expr_to_bigint(e: &Expr) -> Option<BigInt> {
  match e {
    Expr::Integer(n) => Some(BigInt::from(*n)),
    Expr::BigInteger(n) => Some(n.clone()),
    _ => None,
  }
}

/// Check if an expression requires BigInt arithmetic (exceeds f64 precision).
/// f64 can only represent integers exactly up to 2^53.
pub fn needs_bigint_arithmetic(expr: &Expr) -> bool {
  match expr {
    Expr::BigInteger(_) => true,
    Expr::Integer(n) => n.unsigned_abs() > (1u128 << 53),
    _ => false,
  }
}

/// Extract an exact rational (numer, denom) from an Expr.
/// Integer n → (n, 1), Rational[n, d] → (n, d), otherwise None.
pub fn expr_to_rational(expr: &Expr) -> Option<(i128, i128)> {
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
pub fn gcd(a: i128, b: i128) -> i128 {
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
pub fn complex_rational_to_expr(
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
pub fn multiply_scalar_by_expr(
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
pub fn build_complex_float_expr(re: f64, im: f64) -> Expr {
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
pub fn build_complex_expr(
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
pub fn contains_infinity(expr: &Expr) -> bool {
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

/// Extract f64 from various numeric expression types
pub fn expr_to_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Real(f) => Some(*f),
    Expr::Integer(n) => Some(*n as f64),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&args[0], &args[1]) {
        Some(*p as f64 / *q as f64)
      } else {
        None
      }
    }
    _ => None,
  }
}
