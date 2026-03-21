#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Try to express a symbolic expression as a rational multiple of Pi: k*Pi/n.
/// Returns Some((k, n)) in lowest terms, None if not recognized.
/// Handles patterns: Pi, n*Pi, Pi/d, n*Pi/d, n*Degree, Degree, and FunctionCall variants.
pub fn try_symbolic_pi_fraction(expr: &Expr) -> Option<(i64, i64)> {
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
    matches!(e, Expr::Constant(name) | Expr::Identifier(name) if name == "Pi")
  }

  // Helper to check if expr is Degree
  fn is_degree(e: &Expr) -> bool {
    matches!(e, Expr::Constant(name) | Expr::Identifier(name) if name == "Degree")
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
pub fn exact_sin(k: i64, n: i64) -> Option<Expr> {
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
pub fn exact_cos(k: i64, n: i64) -> Option<Expr> {
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
pub fn exact_tan(k: i64, n: i64) -> Option<Expr> {
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

pub fn exact_sec(k: i64, n: i64) -> Option<Expr> {
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

pub fn exact_csc(k: i64, n: i64) -> Option<Expr> {
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

pub fn exact_cot(k: i64, n: i64) -> Option<Expr> {
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
pub fn negate_expr(mut expr: Expr) -> Expr {
  match &mut expr {
    Expr::Integer(n) => return Expr::Integer(-*n),
    Expr::Real(f) => return Expr::Real(-*f),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let Expr::Integer(n) = &args[0] {
        return Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-n), args[1].clone()],
        };
      }
      let name = std::mem::take(name);
      let args = std::mem::take(args);
      return Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), Expr::FunctionCall { name, args }],
      };
    }
    // -(a/b) => Times[-1, a/b] to match Wolfram output style
    Expr::BinaryOp { op, left, right }
      if *op == crate::syntax::BinaryOperator::Divide =>
    {
      // If numerator is an integer, negate it directly: -(n/b) => (-n)/b
      if let Expr::Integer(n) = &**left
        && *n > 1
      {
        let neg_left = Box::new(Expr::Integer(-*n));
        let right = std::mem::replace(right, Box::new(Expr::Integer(0)));
        return Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: neg_left,
          right,
        };
      }
      // fall through to default
    }
    _ => {}
  }
  // Default: wrap in Times[-1, expr]
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![Expr::Integer(-1), expr],
  }
}

/// Build a Complex float result expression from real and imaginary parts.
pub fn build_complex_float_result(
  re: f64,
  im: f64,
) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_function_call_ast(
    "Complex",
    &[Expr::Real(re), Expr::Real(im)],
  )
}

/// Check if an argument is Indeterminate or ComplexInfinity.
/// For periodic/trig functions, both should return Indeterminate.
pub fn is_indeterminate_or_complex_infinity(expr: &Expr) -> bool {
  matches!(expr, Expr::Identifier(s) if s == "Indeterminate" || s == "ComplexInfinity")
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Real args: evaluate numerically
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.sin()));
  }
  // Exact complex: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
  // For purely imaginary (a=0): sin(bi) = i*sinh(b)
  if let Some(((re_num, re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Sin[b*I] → I*Sinh[b]
      let im_expr = make_rational(im_num, im_den);
      let sinh_val =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // I * sinh_val
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), sinh_val],
      );
    }
    // General complex with exact Pi-fraction real part:
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    let re_expr = make_rational(re_num, re_den);
    if let Some((k, n)) = try_symbolic_pi_fraction(&re_expr)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // sin(a)*cosh(b)
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[sin_a, cosh_b],
      )?;
      // cos(a)*sinh(b)
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, sinh_b],
      )?;
      // I * im_coeff
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      // real_term + im_term
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated (matches Wolfram)
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: args.to_vec(),
    });
  }
  // General complex with non-float components: try splitting into
  // real_part + im*I where real_part can be any expression (e.g., Pi, Pi/6)
  if !contains_float(&args[0])
    && let Some((real_part, (im_num, im_den))) = try_split_real_imag(&args[0])
    && im_num != 0
  {
    if matches!(&real_part, Expr::Integer(0)) {
      // Pure imaginary (shouldn't reach here normally, but handle it)
      let im_expr = make_rational(im_num, im_den);
      let sinh_val =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), sinh_val],
      );
    }
    // Check if the real part is a Pi-fraction
    if let Some((k, n)) = try_symbolic_pi_fraction(&real_part)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[sin_a, cosh_b],
      )?;
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, sinh_b],
      )?;
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: args.to_vec(),
    });
  }
  // Complex float: sin(a+bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
  // Only use float path when the argument actually contains a float value
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let sin_re = re.sin() * im.cosh();
    let cos_re = re.cos() * im.sinh();
    return build_complex_float_result(sin_re, cos_re);
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.cos()));
  }
  // Exact complex: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
  // For purely imaginary (a=0): cos(bi) = cosh(b)
  if let Some(((re_num, re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Cos[b*I] → Cosh[b]
      let im_expr = make_rational(im_num, im_den);
      return crate::evaluator::evaluate_function_call_ast("Cosh", &[im_expr]);
    }
    // General complex with exact Pi-fraction real part:
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    let re_expr = make_rational(re_num, re_den);
    if let Some((k, n)) = try_symbolic_pi_fraction(&re_expr)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      // cos(a)*cosh(b)
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, cosh_b],
      )?;
      // -sin(a)*sinh(b) (note the minus sign)
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[negate_expr(sin_a), sinh_b],
      )?;
      // I * im_coeff
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      // real_term + im_term
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    // Non-Pi-fraction real part: leave unevaluated
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: args.to_vec(),
    });
  }
  // General complex with non-float components: try splitting into
  // real_part + im*I
  if !contains_float(&args[0])
    && let Some((real_part, (im_num, im_den))) = try_split_real_imag(&args[0])
    && im_num != 0
  {
    if matches!(&real_part, Expr::Integer(0)) {
      let im_expr = make_rational(im_num, im_den);
      return crate::evaluator::evaluate_function_call_ast("Cosh", &[im_expr]);
    }
    if let Some((k, n)) = try_symbolic_pi_fraction(&real_part)
      && let (Some(sin_a), Some(cos_a)) = (exact_sin(k, n), exact_cos(k, n))
    {
      let im_expr = make_rational(im_num, im_den);
      let cosh_b = crate::evaluator::evaluate_function_call_ast(
        "Cosh",
        &[im_expr.clone()],
      )?;
      let sinh_b =
        crate::evaluator::evaluate_function_call_ast("Sinh", &[im_expr])?;
      let real_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[cos_a, cosh_b],
      )?;
      let im_coeff = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[negate_expr(sin_a), sinh_b],
      )?;
      let im_term = crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), im_coeff],
      )?;
      return crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[real_term, im_term],
      );
    }
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: args.to_vec(),
    });
  }
  // Complex float: cos(a+bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    let cos_re = re.cos() * im.cosh();
    let sin_re = -(re.sin() * im.sinh());
    return build_complex_float_result(cos_re, sin_re);
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if let Expr::Real(f) = &args[0] {
    return Ok(num_to_expr(f.tan()));
  }
  // Exact complex: for purely imaginary: tan(bi) = i*tanh(b)
  if let Some(((re_num, _re_den), (im_num, im_den))) =
    try_extract_complex_exact(&args[0])
    && im_num != 0
  {
    if re_num == 0 {
      // Pure imaginary: Tan[b*I] → I*Tanh[b]
      let im_expr = make_rational(im_num, im_den);
      let tanh_val =
        crate::evaluator::evaluate_function_call_ast("Tanh", &[im_expr])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[Expr::Identifier("I".to_string()), tanh_val],
      );
    }
    // Non-zero real part: leave unevaluated (matches Wolfram)
    return Ok(Expr::FunctionCall {
      name: "Tan".to_string(),
      args: args.to_vec(),
    });
  }
  // Complex float: tan(z) = sin(z)/cos(z)
  if contains_float(&args[0])
    && let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
  {
    // sin(a+bi)
    let sin_re = re.sin() * im.cosh();
    let sin_im = re.cos() * im.sinh();
    // cos(a+bi)
    let cos_re = re.cos() * im.cosh();
    let cos_im = -(re.sin() * im.sinh());
    // (sin_re + sin_im*i) / (cos_re + cos_im*i)
    let denom = cos_re * cos_re + cos_im * cos_im;
    let res_re = (sin_re * cos_re + sin_im * cos_im) / denom;
    let res_im = (sin_im * cos_re - sin_re * cos_im) / denom;
    return build_complex_float_result(res_re, res_im);
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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
  if is_indeterminate_or_complex_infinity(&args[0]) {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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

pub fn erfi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Erfi expects 1 argument".into(),
    ));
  }
  // Helper: compute -Erfi[inner] by evaluating Erfi[inner] first, then negating
  let negate_erfi = |inner: Expr| -> Result<Expr, InterpreterError> {
    let inner_result = erfi_ast(&[inner])?;
    Ok(Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(inner_result),
    })
  };
  match &args[0] {
    // Erfi[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // Erfi[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Erfi[-x] = -Erfi[x] (UnaryOp form)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => negate_erfi(*operand.clone()),
    // Erfi[Times[-1, x]] = -Erfi[x] (evaluated form of -x)
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return negate_erfi(fargs[1].clone());
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return negate_erfi(fargs[0].clone());
      }
      // Negative integer coefficient: Times[-n, x] -> -Erfi[Times[n, x]]
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()],
        };
        return negate_erfi(pos_arg);
      }
      Ok(Expr::FunctionCall {
        name: "Erfi".to_string(),
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
        return negate_erfi(*right.clone());
      }
      if matches!(right.as_ref(), Expr::Integer(-1)) {
        return negate_erfi(*left.clone());
      }
      if let Expr::Integer(n) = left.as_ref()
        && *n < 0
      {
        let pos_arg = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(-*n)),
          right: right.clone(),
        };
        return negate_erfi(pos_arg);
      }
      Ok(Expr::FunctionCall {
        name: "Erfi".to_string(),
        args: args.to_vec(),
      })
    }
    // Erfi[-n] for negative integer
    Expr::Integer(n) if *n < 0 => negate_erfi(Expr::Integer(-*n)),
    // Numeric evaluation for Real arguments
    Expr::Real(f) => Ok(Expr::Real(erfi_f64(*f))),
    // Otherwise symbolic
    _ => Ok(Expr::FunctionCall {
      name: "Erfi".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if !args.is_empty()
    && matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate")
  {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
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
      // Log[-n] for negative integers: Log[-1] = I*Pi, Log[-n] = I*Pi + Log[n]
      if let Expr::Integer(n) = &args[0]
        && *n < 0
      {
        let abs_n = -*n;
        // I*Pi
        let i_pi = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Identifier("I".to_string())),
          right: Box::new(Expr::Constant("Pi".to_string())),
        };
        if abs_n == 1 {
          return Ok(i_pi);
        }
        // I*Pi + Log[abs_n]
        let log_n = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![Expr::Integer(abs_n)],
        };
        return crate::evaluator::evaluate_function_call_ast(
          "Plus",
          &[i_pi, log_n],
        );
      }
      // Log[-expr] for negated expressions: check for Times[-1, ...]
      {
        let inner = match &args[0] {
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1)) => {
            Some(*right.clone())
          }
          Expr::FunctionCall { name, args: targs }
            if name == "Times"
              && targs.len() == 2
              && matches!(&targs[0], Expr::Integer(-1)) =>
          {
            Some(targs[1].clone())
          }
          _ => None,
        };
        if let Some(inner_expr) = inner {
          let i_pi = Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Identifier("I".to_string())),
            right: Box::new(Expr::Constant("Pi".to_string())),
          };
          let log_x =
            crate::evaluator::evaluate_function_call_ast("Log", &[inner_expr])?;
          return crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[i_pi, log_x],
          );
        }
      }
      // Log[p/q] where 0 < p < q: return -Log[q/p]
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && matches!((&rat_args[0], &rat_args[1]), (Expr::Integer(p), Expr::Integer(q)) if *p > 0 && *q > 0 && *p < *q)
        && let (Expr::Integer(p), Expr::Integer(q)) =
          (&rat_args[0], &rat_args[1])
      {
        let inverted = make_rational(*q, *p);
        let log_inv = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![inverted],
        };
        return Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(-1)),
          right: Box::new(log_inv),
        });
      }
      if let Expr::Real(f) = &args[0] {
        if *f > 0.0 {
          return Ok(Expr::Real(f.ln()));
        } else if *f == 0.0 {
          return Ok(Expr::Identifier("Indeterminate".to_string()));
        } else {
          // Log of negative real: return complex result
          let re = f.abs().ln();
          let im = std::f64::consts::PI;
          return crate::evaluator::evaluate_function_call_ast(
            "Complex",
            &[Expr::Real(re), Expr::Real(im)],
          );
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
  // Symbolic fallback: Log10[x] = Log[x] / Log[10]
  Ok(Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: args.to_vec(),
    }),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(10)],
    }),
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
  // Symbolic fallback: Log2[x] = Log[x] / Log[2]
  Ok(Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: args.to_vec(),
    }),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![Expr::Integer(2)],
    }),
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
      // -1/2*Pi = Times[Rational[-1, 2], Pi]
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(2)],
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      });
    }
    Expr::Real(f) => {
      if (-1.0..=1.0).contains(f) {
        return Ok(Expr::Real(f.asin()));
      }
    }
    _ => {}
  }
  // Check for special rational/irrational values via numeric comparison
  if let Some(v) = try_eval_to_f64(&args[0])
    && let Some(result) = arcsin_special_value(v)
  {
    return Ok(result);
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
  // Check for special rational/irrational values via numeric comparison
  if let Some(v) = try_eval_to_f64(&args[0])
    && let Some(result) = arccos_special_value(v)
  {
    return Ok(result);
  }
  Ok(Expr::FunctionCall {
    name: "ArcCos".to_string(),
    args: args.to_vec(),
  })
}

/// Check if a float value matches a known ArcCos special angle
pub fn arccos_special_value(v: f64) -> Option<Expr> {
  let eps = 1e-12;

  // Helper to build n*Pi/d
  let pi_frac = |num: i128, den: i128| -> Expr {
    if den == 1 {
      if num == 1 {
        Expr::Constant("Pi".to_string())
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(num)),
          right: Box::new(Expr::Constant("Pi".to_string())),
        }
      }
    } else {
      let numerator = if num == 1 {
        Expr::Constant("Pi".to_string())
      } else {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(num)),
          right: Box::new(Expr::Constant("Pi".to_string())),
        }
      };
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(numerator),
        right: Box::new(Expr::Integer(den)),
      }
    }
  };

  // ArcCos special values (value -> result as n*Pi/d)
  let table: &[(f64, i128, i128)] = &[
    (0.5, 1, 3),                              // ArcCos[1/2] = Pi/3
    (-0.5, 2, 3),                             // ArcCos[-1/2] = 2*Pi/3
    (std::f64::consts::FRAC_1_SQRT_2, 1, 4),  // ArcCos[Sqrt[2]/2] = Pi/4
    (-std::f64::consts::FRAC_1_SQRT_2, 3, 4), // ArcCos[-Sqrt[2]/2] = 3*Pi/4
    (0.8660254037844386, 1, 6),               // ArcCos[Sqrt[3]/2] = Pi/6
    (-0.8660254037844386, 5, 6),              // ArcCos[-Sqrt[3]/2] = 5*Pi/6
    (0.9659258262890682, 1, 12), // ArcCos[(1+Sqrt[3])/(2*Sqrt[2])] = Pi/12
    (-0.9659258262890682, 11, 12), // ArcCos[-(1+Sqrt[3])/(2*Sqrt[2])] = 11*Pi/12
  ];

  for &(val, num, den) in table {
    if (v - val).abs() < eps {
      return Some(pi_frac(num, den));
    }
  }
  None
}

/// Check if a float value matches a known ArcSin special angle
pub fn arcsin_special_value(v: f64) -> Option<Expr> {
  let eps = 1e-12;

  // ArcSin special values (|value| -> Pi/d)
  let table: &[(f64, i128)] = &[
    (0.5, 6),                             // ArcSin[1/2] = Pi/6
    (std::f64::consts::FRAC_1_SQRT_2, 4), // ArcSin[Sqrt[2]/2] = Pi/4
    (0.8660254037844386, 3),              // ArcSin[Sqrt[3]/2] = Pi/3
  ];

  for &(val, den) in table {
    if (v.abs() - val).abs() < eps {
      if v < 0.0 {
        // Negative: build Times[Rational[-1, den], Pi] to display as -1/den*Pi
        return Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(-1), Expr::Integer(den)],
          }),
          right: Box::new(Expr::Constant("Pi".to_string())),
        });
      } else if den == 1 {
        return Some(Expr::Constant("Pi".to_string()));
      } else {
        return Some(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left: Box::new(Expr::Constant("Pi".to_string())),
          right: Box::new(Expr::Integer(den)),
        });
      }
    }
  }
  None
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
      // -1/4*Pi = Times[Rational[-1, 4], Pi]
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(4)],
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      });
    }
    Expr::Real(f) => return Ok(Expr::Real(f.atan())),
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcTan".to_string(),
    args: args.to_vec(),
  })
}

/// ArcTan[x, y] - Two-argument arctangent (atan2)
/// Returns the angle in radians of the point (x, y).
/// ArcTan[x, y] = ArcTan[y/x] with quadrant adjustment.
pub fn arctan2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ArcTan expects exactly 2 arguments".into(),
    ));
  }
  let x = &args[0];
  let y = &args[1];

  // Numeric case: both Real
  if let (Expr::Real(fx), Expr::Real(fy)) = (x, y) {
    return Ok(Expr::Real(fy.atan2(*fx)));
  }

  // Exact special values for common angles
  if let (Some(xn), Some(yn)) = (expr_to_i128(x), expr_to_i128(y)) {
    return Ok(match (xn, yn) {
      (_, 0) if xn > 0 => Expr::Integer(0), // ArcTan[+x, 0] = 0
      (_, 0) if xn < 0 => Expr::Constant("Pi".to_string()), // ArcTan[-x, 0] = Pi
      (0, _) if yn > 0 => Expr::BinaryOp {
        // ArcTan[0, +y] = Pi/2
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      },
      (0, _) if yn < 0 => Expr::BinaryOp {
        // ArcTan[0, -y] = -Pi/2
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(2)],
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      },
      _ if xn > 0 && yn == xn => Expr::BinaryOp {
        // ArcTan[a, a] = Pi/4
        op: crate::syntax::BinaryOperator::Divide,
        left: Box::new(Expr::Constant("Pi".to_string())),
        right: Box::new(Expr::Integer(4)),
      },
      _ if xn > 0 && yn == -xn => Expr::BinaryOp {
        // ArcTan[a, -a] = -Pi/4
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(-1), Expr::Integer(4)],
        }),
        right: Box::new(Expr::Constant("Pi".to_string())),
      },
      _ => Expr::FunctionCall {
        name: "ArcTan".to_string(),
        args: args.to_vec(),
      },
    });
  }

  // Return unevaluated for symbolic args
  Ok(Expr::FunctionCall {
    name: "ArcTan".to_string(),
    args: args.to_vec(),
  })
}

/// Try to split an expression into real part and exact imaginary coefficient.
/// Returns Some((real_part_expr, (im_num, im_den))) if the expression
/// has the form `real_part + (im_num/im_den)*I` where the imaginary
/// coefficient is exact (integer or rational).
/// This is more general than try_extract_complex_exact since real_part
/// can be any expression (e.g., Pi, Pi/6, symbolic).
fn try_split_real_imag(expr: &Expr) -> Option<(Expr, (i128, i128))> {
  use crate::syntax::BinaryOperator;

  // Helper: check if expr is purely I*rational (no real part)
  fn as_imag_only(e: &Expr) -> Option<(i128, i128)> {
    match e {
      Expr::Identifier(name) if name == "I" => Some((1, 1)),
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => {
        let (n, d) = as_imag_only(operand)?;
        Some((-n, d))
      }
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left,
        right,
      } => {
        if matches!(&**right, Expr::Identifier(s) if s == "I")
          && let Some((n, d)) = expr_to_rational(left)
        {
          return Some((n, d));
        }
        if matches!(&**left, Expr::Identifier(s) if s == "I")
          && let Some((n, d)) = expr_to_rational(right)
        {
          return Some((n, d));
        }
        None
      }
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        // Check if I is one of the args
        let mut i_idx = None;
        for (idx, arg) in args.iter().enumerate() {
          if matches!(arg, Expr::Identifier(s) if s == "I") {
            i_idx = Some(idx);
            break;
          }
        }
        let i_idx = i_idx?;
        // Remaining args should form a rational
        let remaining: Vec<&Expr> = args
          .iter()
          .enumerate()
          .filter(|(i, _)| *i != i_idx)
          .map(|(_, a)| a)
          .collect();
        if remaining.len() == 1
          && let Some((n, d)) = expr_to_rational(remaining[0])
        {
          return Some((n, d));
        }
        None
      }
      _ => None,
    }
  }

  // Pure imaginary
  if let Some(im) = as_imag_only(expr) {
    return Some((Expr::Integer(0), im));
  }

  // a + b*I or a - b*I
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      if let Some(im) = as_imag_only(right) {
        return Some((*left.clone(), im));
      }
      if let Some(im) = as_imag_only(left) {
        return Some((*right.clone(), im));
      }
      None
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      if let Some((n, d)) = as_imag_only(right) {
        return Some((*left.clone(), (-n, d)));
      }
      None
    }
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      // Check if one of the args is purely imaginary
      for (idx, arg) in args.iter().enumerate() {
        if let Some(im) = as_imag_only(arg) {
          let remaining: Vec<Expr> = args
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .map(|(_, a)| a.clone())
            .collect();
          let real_part = if remaining.len() == 1 {
            remaining.into_iter().next().unwrap()
          } else {
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: remaining,
            }
          };
          return Some((real_part, im));
        }
      }
      None
    }
    _ => None,
  }
}

// ─── Hyperbolic Trig Functions ────────────────────────────────────

/// Try to extract a "negated" form of an expression.
/// Returns Some(positive_expr) if the expression looks negative
/// (negative integer, negative rational, unary minus, Times[-1, x]).
fn try_extract_negated(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-n)),
    Expr::Real(f) if *f < 0.0 => Some(Expr::Real(-f)),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => Some((**operand).clone()),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        if *n < 0 {
          Some(make_rational(n.abs(), *d))
        } else {
          None
        }
      } else {
        None
      }
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some(negated) = try_extract_negated(left) {
        if matches!(&negated, Expr::Integer(1)) {
          Some((**right).clone())
        } else {
          Some(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(negated),
            right: right.clone(),
          })
        }
      } else {
        None
      }
    }
    // Handle evaluated Times[-1, x, ...] function call form
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      if let Some(negated_first) = try_extract_negated(&args[0]) {
        if args.len() == 1 {
          return Some(negated_first);
        }
        if matches!(&negated_first, Expr::Integer(1)) {
          if args.len() == 2 {
            return Some(args[1].clone());
          }
          return Some(Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          });
        }
        let mut new_args = vec![negated_first];
        new_args.extend_from_slice(&args[1..]);
        Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: new_args,
        })
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Sinh[x] - Hyperbolic sine
pub fn sinh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Sinh expects 1 argument".into(),
    ));
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.sinh())),
    _ => {}
  }
  // Odd function: Sinh[-x] → -Sinh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Sinh", &[pos])?;
    return Ok(negate_expr(inner));
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(f.cosh())),
    _ => {}
  }
  // Even function: Cosh[-x] → Cosh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Cosh", &[pos]);
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => return Ok(Expr::Real(f.tanh())),
    _ => {}
  }
  // Odd function: Tanh[-x] → -Tanh[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Tanh", &[pos])?;
    return Ok(negate_expr(inner));
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Real(f) => {
      let t = f.tanh();
      if t == 0.0 {
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      return Ok(Expr::Real(1.0 / t));
    }
    _ => {}
  }
  // Odd function: Coth[-x] → -Coth[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Coth", &[pos])?;
    return Ok(negate_expr(inner));
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Integer(1)),
    Expr::Real(f) => return Ok(Expr::Real(1.0 / f.cosh())),
    _ => {}
  }
  // Even function: Sech[-x] → Sech[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    return crate::evaluator::evaluate_function_call_ast("Sech", &[pos]);
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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
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
  // Odd function: Csch[-x] → -Csch[x]
  if let Some(pos) = try_extract_negated(&args[0]) {
    let inner = crate::evaluator::evaluate_function_call_ast("Csch", &[pos])?;
    return Ok(negate_expr(inner));
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
    Expr::Integer(0) => {
      // ArcCosh[0] = (I/2)*Pi
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
          Expr::Identifier("I".to_string()),
          Expr::Constant("Pi".to_string()),
        ],
      );
    }
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => {
      if *f >= 1.0 {
        return Ok(Expr::Real(f.acosh()));
      }
      // For 0 <= f < 1 or f < 0, return complex result
      // ArcCosh[x] = I*ArcCos[x] for real x in [-1,1]
      let x = *f;
      let acos_x = x.acos();
      return crate::evaluator::evaluate_function_call_ast(
        "Complex",
        &[Expr::Real(0.0), Expr::Real(acos_x)],
      );
    }
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

/// ArcCoth[x] - Inverse hyperbolic cotangent
pub fn arccoth_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcCoth expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => {
      // ArcCoth[0] = (I/2)*Pi
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
          Expr::Identifier("I".to_string()),
          Expr::Constant("Pi".to_string()),
        ],
      );
    }
    Expr::Integer(1) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(-1) => {
      return Ok(Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand: Box::new(Expr::Identifier("Infinity".to_string())),
      });
    }
    Expr::Real(f) => {
      // ArcCoth[x] = ArcTanh[1/x]
      // For |x| > 1: real result
      // For |x| < 1: complex result
      // For x == 0: I*Pi/2
      let x = *f;
      if x.abs() > 1.0 {
        return Ok(Expr::Real((1.0 / x).atanh()));
      } else if x == 0.0 {
        return crate::evaluator::evaluate_function_call_ast(
          "Complex",
          &[Expr::Real(0.0), Expr::Real(std::f64::consts::FRAC_PI_2)],
        );
      } else {
        // |x| <= 1 and x != 0: complex result
        // ArcCoth[x] = (1/2) * ln((x+1)/(x-1))
        // For |x| < 1: real part = (1/2)*ln(|(x+1)/(x-1)|), imag part = ±Pi/2
        let re = 0.5 * ((x + 1.0) / (x - 1.0)).abs().ln();
        let im = if x > 0.0 {
          -std::f64::consts::FRAC_PI_2
        } else {
          std::f64::consts::FRAC_PI_2
        };
        return crate::evaluator::evaluate_function_call_ast(
          "Complex",
          &[Expr::Real(re), Expr::Real(im)],
        );
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcCoth".to_string(),
    args: args.to_vec(),
  })
}

/// ArcSech[x] - Inverse hyperbolic secant
pub fn arcsech_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ArcSech expects 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(0) => return Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(1) => return Ok(Expr::Integer(0)),
    Expr::Real(f) => {
      // ArcSech[x] = ArcCosh[1/x]
      let x = *f;
      if x > 0.0 && x <= 1.0 {
        return Ok(Expr::Real((1.0 / x).acosh()));
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ArcSech".to_string(),
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
pub fn pi_over_n(n: i128) -> Expr {
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(Expr::Constant("Pi".to_string())),
    right: Box::new(Expr::Integer(n)),
  }
}

pub fn negative_pi_over_2() -> Expr {
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
      // Gudermannian[ComplexInfinity] is unevaluated in Wolfram
      return Ok(Expr::FunctionCall {
        name: "Gudermannian".to_string(),
        args: args.to_vec(),
      });
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

// ─── TrigExpand ─────────────────────────────────────────────────────────

/// TrigExpand[expr] — expand trig functions of sums and integer multiples.
pub fn trig_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "TrigExpand".to_string(),
      args: args.to_vec(),
    });
  }
  let result = trig_expand_recursive(&args[0]);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Recursively apply TrigExpand to an expression.
fn trig_expand_recursive(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args } if args.len() == 1 => {
      let trig_name = name.as_str();
      match trig_name {
        "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc" | "Sinh" | "Cosh"
        | "Tanh" | "Coth" | "Sech" | "Csch" => {
          // First recursively expand the argument
          let arg = trig_expand_recursive(&args[0]);
          expand_trig_function(trig_name, &arg)
        }
        _ => {
          // Recurse into other function calls
          Expr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(trig_expand_recursive).collect(),
          }
        }
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(trig_expand_recursive).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(trig_expand_recursive(left)),
      right: Box::new(trig_expand_recursive(right)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(trig_expand_recursive).collect())
    }
    _ => expr.clone(),
  }
}

/// Expand a single trig function application.
fn expand_trig_function(name: &str, arg: &Expr) -> Expr {
  // Try to decompose the argument as a sum of terms
  let terms = collect_additive_terms(arg);

  if terms.len() >= 2 {
    // It's a sum — apply addition formulas
    match name {
      "Sin" => expand_sin_sum(&terms),
      "Cos" => expand_cos_sum(&terms),
      "Tan" => {
        let sin_exp = expand_sin_sum(&terms);
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&sin_exp, &cos_exp)
      }
      "Cot" => {
        let sin_exp = expand_sin_sum(&terms);
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&cos_exp, &sin_exp)
      }
      "Sec" => {
        let cos_exp = expand_cos_sum(&terms);
        make_divide(&Expr::Integer(1), &cos_exp)
      }
      "Csc" => {
        let sin_exp = expand_sin_sum(&terms);
        make_divide(&Expr::Integer(1), &sin_exp)
      }
      "Sinh" => expand_sinh_sum(&terms),
      "Cosh" => expand_cosh_sum(&terms),
      "Tanh" => {
        let s = expand_sinh_sum(&terms);
        let c = expand_cosh_sum(&terms);
        make_divide(&s, &c)
      }
      "Coth" => {
        let s = expand_sinh_sum(&terms);
        let c = expand_cosh_sum(&terms);
        make_divide(&c, &s)
      }
      "Sech" => {
        let c = expand_cosh_sum(&terms);
        make_divide(&Expr::Integer(1), &c)
      }
      "Csch" => {
        let s = expand_sinh_sum(&terms);
        make_divide(&Expr::Integer(1), &s)
      }
      _ => make_fn(name, &[arg.clone()]),
    }
  } else {
    // Single term — check for integer multiple
    let (coeff, base) = extract_integer_factor(arg);
    if coeff >= 2 {
      match name {
        "Sin" => expand_sin_multiple(coeff as usize, &base),
        "Cos" => expand_cos_multiple(coeff as usize, &base),
        "Tan" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&s, &c)
        }
        "Cot" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&c, &s)
        }
        "Sec" => {
          let c = expand_cos_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &c)
        }
        "Csc" => {
          let s = expand_sin_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &s)
        }
        "Sinh" => expand_sinh_multiple(coeff as usize, &base),
        "Cosh" => expand_cosh_multiple(coeff as usize, &base),
        "Tanh" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&s, &c)
        }
        "Coth" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&c, &s)
        }
        "Sech" => {
          let c = expand_cosh_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &c)
        }
        "Csch" => {
          let s = expand_sinh_multiple(coeff as usize, &base);
          make_divide(&Expr::Integer(1), &s)
        }
        _ => make_fn(name, &[arg.clone()]),
      }
    } else {
      // Nothing to expand
      make_fn(name, &[arg.clone()])
    }
  }
}

/// Collect additive terms from a Plus/Minus expression.
fn collect_additive_terms(expr: &Expr) -> Vec<Expr> {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      terms.extend(collect_additive_terms(right));
      terms
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left,
      right,
    } => {
      let mut terms = collect_additive_terms(left);
      // Negate the right side
      let neg = make_times(&Expr::Integer(-1), right);
      terms.push(neg);
      terms
    }
    Expr::FunctionCall { name, args } if name == "Plus" => {
      args.iter().flat_map(collect_additive_terms).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// Extract integer factor from n*x. Returns (n, x).
fn extract_integer_factor(expr: &Expr) -> (i128, Expr) {
  use crate::syntax::BinaryOperator;
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Expr::Integer(n) = left.as_ref() {
        return (*n, *right.clone());
      }
      if let Expr::Integer(n) = right.as_ref() {
        return (*n, *left.clone());
      }
      (1, expr.clone())
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if let Expr::Integer(n) = &args[0] {
        return (*n, args[1].clone());
      }
      if let Expr::Integer(n) = &args[1] {
        return (*n, args[0].clone());
      }
      (1, expr.clone())
    }
    _ => (1, expr.clone()),
  }
}

// ─── Sin/Cos expansion helpers ──────────────────────────────────────

/// Expand Sin[a + b + ...] using recursive addition formula.
fn expand_sin_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Sin", &[terms[0].clone()]);
  }
  // Sin[a + rest] = Sin[a]*Cos[rest] + Cos[a]*Sin[rest]
  let a = &terms[0];
  let rest = &terms[1..];
  let sin_a = make_fn("Sin", &[a.clone()]);
  let cos_a = make_fn("Cos", &[a.clone()]);
  let sin_rest = expand_sin_sum(rest);
  let cos_rest = expand_cos_sum(rest);
  make_plus(
    &make_times(&sin_a, &cos_rest),
    &make_times(&cos_a, &sin_rest),
  )
}

/// Expand Cos[a + b + ...] using recursive addition formula.
fn expand_cos_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Cos", &[terms[0].clone()]);
  }
  // Cos[a + rest] = Cos[a]*Cos[rest] - Sin[a]*Sin[rest]
  let a = &terms[0];
  let rest = &terms[1..];
  let sin_a = make_fn("Sin", &[a.clone()]);
  let cos_a = make_fn("Cos", &[a.clone()]);
  let sin_rest = expand_sin_sum(rest);
  let cos_rest = expand_cos_sum(rest);
  make_minus(
    &make_times(&cos_a, &cos_rest),
    &make_times(&sin_a, &sin_rest),
  )
}

/// Expand Sin[n*x] using Chebyshev-like formula:
/// Sin[n*x] = Sum_{k} (-1)^k * C(n, 2k+1) * cos(x)^(n-2k-1) * sin(x)^(2k+1)
fn expand_sin_multiple(n: usize, x: &Expr) -> Expr {
  let sin_x = make_fn("Sin", &[x.clone()]);
  let cos_x = make_fn("Cos", &[x.clone()]);
  let mut terms = Vec::new();

  for k in 0..=(n - 1) / 2 {
    let sign = if k % 2 == 0 { 1i128 } else { -1 };
    let binom = binomial(n as i128, (2 * k + 1) as i128);
    let coeff = sign * binom;
    let cos_pow = n - 2 * k - 1;
    let sin_pow = 2 * k + 1;

    let mut term = Expr::Integer(coeff);
    if cos_pow > 0 {
      term = make_times(&term, &make_power(&cos_x, cos_pow as i128));
    }
    if sin_pow > 0 {
      term = make_times(&term, &make_power(&sin_x, sin_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

/// Expand Cos[n*x] using Chebyshev-like formula:
/// Cos[n*x] = Sum_{k} (-1)^k * C(n, 2k) * cos(x)^(n-2k) * sin(x)^(2k)
fn expand_cos_multiple(n: usize, x: &Expr) -> Expr {
  let sin_x = make_fn("Sin", &[x.clone()]);
  let cos_x = make_fn("Cos", &[x.clone()]);
  let mut terms = Vec::new();

  for k in 0..=n / 2 {
    let sign = if k % 2 == 0 { 1i128 } else { -1 };
    let binom = binomial(n as i128, (2 * k) as i128);
    let coeff = sign * binom;
    let cos_pow = n - 2 * k;
    let sin_pow = 2 * k;

    let mut term = Expr::Integer(coeff);
    if cos_pow > 0 {
      term = make_times(&term, &make_power(&cos_x, cos_pow as i128));
    }
    if sin_pow > 0 {
      term = make_times(&term, &make_power(&sin_x, sin_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

// ─── Hyperbolic expansion helpers ───────────────────────────────────

/// Expand Sinh[a + b + ...] recursively.
fn expand_sinh_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Sinh", &[terms[0].clone()]);
  }
  let a = &terms[0];
  let rest = &terms[1..];
  // Sinh[a + rest] = Sinh[a]*Cosh[rest] + Cosh[a]*Sinh[rest]
  make_plus(
    &make_times(&make_fn("Sinh", &[a.clone()]), &expand_cosh_sum(rest)),
    &make_times(&make_fn("Cosh", &[a.clone()]), &expand_sinh_sum(rest)),
  )
}

/// Expand Cosh[a + b + ...] recursively.
fn expand_cosh_sum(terms: &[Expr]) -> Expr {
  if terms.len() == 1 {
    return make_fn("Cosh", &[terms[0].clone()]);
  }
  let a = &terms[0];
  let rest = &terms[1..];
  // Cosh[a + rest] = Cosh[a]*Cosh[rest] + Sinh[a]*Sinh[rest]
  make_plus(
    &make_times(&make_fn("Cosh", &[a.clone()]), &expand_cosh_sum(rest)),
    &make_times(&make_fn("Sinh", &[a.clone()]), &expand_sinh_sum(rest)),
  )
}

/// Expand Sinh[n*x] using the multiple-angle formula.
fn expand_sinh_multiple(n: usize, x: &Expr) -> Expr {
  let sinh_x = make_fn("Sinh", &[x.clone()]);
  let cosh_x = make_fn("Cosh", &[x.clone()]);
  let mut terms = Vec::new();

  // Sinh[n*x] = Sum_{k} C(n, 2k+1) * cosh(x)^(n-2k-1) * sinh(x)^(2k+1)
  // (same as sin but without (-1)^k)
  for k in 0..=(n - 1) / 2 {
    let binom = binomial(n as i128, (2 * k + 1) as i128);
    let cosh_pow = n - 2 * k - 1;
    let sinh_pow = 2 * k + 1;

    let mut term = Expr::Integer(binom);
    if cosh_pow > 0 {
      term = make_times(&term, &make_power(&cosh_x, cosh_pow as i128));
    }
    if sinh_pow > 0 {
      term = make_times(&term, &make_power(&sinh_x, sinh_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

/// Expand Cosh[n*x] using the multiple-angle formula.
fn expand_cosh_multiple(n: usize, x: &Expr) -> Expr {
  let sinh_x = make_fn("Sinh", &[x.clone()]);
  let cosh_x = make_fn("Cosh", &[x.clone()]);
  let mut terms = Vec::new();

  // Cosh[n*x] = Sum_{k} C(n, 2k) * cosh(x)^(n-2k) * sinh(x)^(2k)
  // (same as cos but without (-1)^k)
  for k in 0..=n / 2 {
    let binom = binomial(n as i128, (2 * k) as i128);
    let cosh_pow = n - 2 * k;
    let sinh_pow = 2 * k;

    let mut term = Expr::Integer(binom);
    if cosh_pow > 0 {
      term = make_times(&term, &make_power(&cosh_x, cosh_pow as i128));
    }
    if sinh_pow > 0 {
      term = make_times(&term, &make_power(&sinh_x, sinh_pow as i128));
    }
    terms.push(term);
  }

  build_sum(&terms)
}

// ─── AST building helpers ───────────────────────────────────────────

fn make_fn(name: &str, args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec(),
  }
}

fn make_times(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a.clone(), b.clone()],
  }
}

fn make_plus(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![a.clone(), b.clone()],
  }
}

fn make_minus(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![
      a.clone(),
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), b.clone()],
      },
    ],
  }
}

fn make_divide(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![
      a.clone(),
      Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![b.clone(), Expr::Integer(-1)],
      },
    ],
  }
}

fn make_power(base: &Expr, exp: i128) -> Expr {
  if exp == 1 {
    base.clone()
  } else {
    Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base.clone(), Expr::Integer(exp)],
    }
  }
}

fn build_sum(terms: &[Expr]) -> Expr {
  if terms.is_empty() {
    return Expr::Integer(0);
  }
  if terms.len() == 1 {
    return terms[0].clone();
  }
  Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms.to_vec(),
  }
}

/// Compute binomial coefficient C(n, k).
fn binomial(n: i128, k: i128) -> i128 {
  if k < 0 || k > n {
    return 0;
  }
  let k = k.min(n - k);
  let mut result = 1i128;
  for i in 0..k {
    result = result * (n - i) / (i + 1);
  }
  result
}

// ─── TrigReduce ─────────────────────────────────────────────────────────

/// TrigReduce[expr] — rewrite products and powers of trig functions as
/// sums of trig functions with linear arguments.
pub fn trig_reduce_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "TrigReduce".to_string(),
      args: args.to_vec(),
    });
  }
  let result = trig_reduce_recursive(&args[0]);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Recursively apply TrigReduce.
fn trig_reduce_recursive(expr: &Expr) -> Expr {
  match expr {
    // Handle Times[...] — look for products of trig functions
    Expr::FunctionCall { name, args } if name == "Times" => {
      trig_reduce_product(args)
    }
    // Handle BinaryOp Times — two-factor product
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      let l = trig_reduce_recursive(left);
      let r = trig_reduce_recursive(right);
      reduce_two_factor_product(&l, &r)
    }
    // Handle Power[Sin[x], n] or Power[Cos[x], n]
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => {
      let base = trig_reduce_recursive(left);
      if let Expr::Integer(n) = right.as_ref()
        && *n >= 2
        && let Some(result) = reduce_trig_power(&base, *n)
      {
        return result;
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(trig_reduce_recursive(right)),
      }
    }
    // Recurse into other structures
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(trig_reduce_recursive).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(trig_reduce_recursive(left)),
      right: Box::new(trig_reduce_recursive(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(trig_reduce_recursive(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(trig_reduce_recursive).collect())
    }
    _ => expr.clone(),
  }
}

/// Handle Times[a1, a2, ...] — find pairs of trig functions to reduce.
fn trig_reduce_product(factors: &[Expr]) -> Expr {
  let mut reduced: Vec<Expr> =
    factors.iter().map(trig_reduce_recursive).collect();

  // Repeatedly try to find and reduce pairs of trig functions
  let mut changed = true;
  while changed {
    changed = false;
    for i in 0..reduced.len() {
      for j in (i + 1)..reduced.len() {
        if let Some(result) = try_reduce_trig_pair(&reduced[i], &reduced[j]) {
          // Replace i with the result, remove j
          let mut new_factors = Vec::new();
          for (k, f) in reduced.iter().enumerate() {
            if k == i {
              new_factors.push(result.clone());
            } else if k != j {
              new_factors.push(f.clone());
            }
          }
          reduced = new_factors;
          changed = true;
          break;
        }
      }
      if changed {
        break;
      }
    }
  }

  if reduced.len() == 1 {
    reduced.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: reduced,
    }
  }
}

/// Try to reduce product of two terms using product-to-sum identities.
fn reduce_two_factor_product(a: &Expr, b: &Expr) -> Expr {
  if let Some(result) = try_reduce_trig_pair(a, b) {
    return result;
  }
  // Check if either factor is a trig power
  if let Some(result) = try_reduce_with_power(a, b) {
    return result;
  }
  if let Some(result) = try_reduce_with_power(b, a) {
    return result;
  }
  Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(a.clone()),
    right: Box::new(b.clone()),
  }
}

/// Try product-to-sum for Sin[a]*Cos[b], Sin[a]*Sin[b], Cos[a]*Cos[b].
fn try_reduce_trig_pair(a: &Expr, b: &Expr) -> Option<Expr> {
  let (a_name, a_arg) = extract_trig(a)?;
  let (b_name, b_arg) = extract_trig(b)?;

  let sum = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Plus,
    left: Box::new(a_arg.clone()),
    right: Box::new(b_arg.clone()),
  };
  let diff = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Minus,
    left: Box::new(a_arg.clone()),
    right: Box::new(b_arg.clone()),
  };

  let half = |e: Expr| -> Expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(e),
      right: Box::new(Expr::Integer(2)),
    }
  };

  let sin = |e: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![e],
    }
  };
  let cos = |e: Expr| -> Expr {
    Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![e],
    }
  };

  match (a_name, b_name) {
    // Sin[a]*Cos[b] = (Sin[a+b] + Sin[a-b])/2
    ("Sin", "Cos") | ("Cos", "Sin") => {
      let (sin_arg, cos_arg) = if a_name == "Sin" {
        (a_arg, b_arg)
      } else {
        (b_arg, a_arg)
      };
      let s = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(sin_arg.clone()),
        right: Box::new(cos_arg.clone()),
      };
      let d = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(sin_arg),
        right: Box::new(cos_arg),
      };
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(sin(s)),
        right: Box::new(sin(d)),
      };
      Some(half(result))
    }
    // Cos[a]*Cos[b] = (Cos[a-b] + Cos[a+b])/2
    ("Cos", "Cos") => {
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Plus,
        left: Box::new(cos(diff)),
        right: Box::new(cos(sum)),
      };
      Some(half(result))
    }
    // Sin[a]*Sin[b] = (Cos[a-b] - Cos[a+b])/2
    ("Sin", "Sin") => {
      let result = Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Minus,
        left: Box::new(cos(diff)),
        right: Box::new(cos(sum)),
      };
      Some(half(result))
    }
    _ => None,
  }
}

/// Reduce trig power: Sin[x]^n or Cos[x]^n using direct formulas.
fn reduce_trig_power(base: &Expr, n: i128) -> Option<Expr> {
  let (trig_name, arg) = extract_trig(base)?;
  if !matches!(trig_name, "Sin" | "Cos") {
    return None;
  }

  // Use the power-reduction formulas directly:
  // For even n:
  //   cos^n(x) = 1/2^n * C(n,n/2) + 2/2^n * sum_{k=0}^{n/2-1} C(n,k) cos((n-2k)x)
  //   sin^n(x) = 1/2^n * C(n,n/2) + 2/2^n * sum_{k=0}^{n/2-1} (-1)^(n/2-k) C(n,k) cos((n-2k)x)
  // For odd n:
  //   cos^n(x) = 2/2^n * sum_{k=0}^{(n-1)/2} C(n,k) cos((n-2k)x)
  //   sin^n(x) = 2/2^n * sum_{k=0}^{(n-1)/2} (-1)^((n-1)/2-k) C(n,k) sin((n-2k)x)

  let is_cos = trig_name == "Cos";
  let nu = n as usize;
  let denom = 1i128 << n; // 2^n

  let make_int_times_arg = |coeff: i128, m: i128| -> Expr {
    if m == 1 {
      arg.clone()
    } else {
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(Expr::Integer(coeff)),
        right: Box::new(arg.clone()),
      }
    }
  };

  // Collect (numerator_coeff, trig_expr_or_None_for_constant) pairs
  let mut num_terms: Vec<(i128, Option<Expr>)> = Vec::new();

  if nu.is_multiple_of(2) {
    // Even power
    let half_n = nu / 2;
    // Constant term: C(n, n/2)
    let const_binom = binomial(n, half_n as i128);
    num_terms.push((const_binom, None));

    // Sum terms
    for k in 0..half_n {
      let m = n - 2 * k as i128;
      let binom_val = binomial(n, k as i128);
      let coeff = 2 * binom_val;
      let sign = if is_cos {
        1
      } else if (half_n - k).is_multiple_of(2) {
        1
      } else {
        -1
      };
      let trig_arg = make_int_times_arg(m, m);
      let trig_call = Expr::FunctionCall {
        name: "Cos".to_string(),
        args: vec![trig_arg],
      };
      num_terms.push((sign * coeff, Some(trig_call)));
    }
  } else {
    // Odd power
    let half_n = (nu - 1) / 2;
    let trig_fn = if is_cos { "Cos" } else { "Sin" };
    for k in 0..=half_n {
      let m = n - 2 * k as i128;
      let binom_val = binomial(n, k as i128);
      let coeff = 2 * binom_val;
      let sign = if is_cos {
        1
      } else if (half_n - k).is_multiple_of(2) {
        1
      } else {
        -1
      };
      let trig_arg = make_int_times_arg(m, m);
      let trig_call = Expr::FunctionCall {
        name: trig_fn.to_string(),
        args: vec![trig_arg],
      };
      num_terms.push((sign * coeff, Some(trig_call)));
    }
  }

  // Compute GCD of all numerator coefficients and denom to simplify the fraction
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

  let mut g = denom;
  for (c, _) in &num_terms {
    g = gcd(g, *c);
  }
  let reduced_denom = denom / g;

  // Build numerator terms with reduced coefficients
  let mut terms: Vec<Expr> = Vec::new();
  for (c, trig_opt) in num_terms {
    let reduced_c = c / g;
    match trig_opt {
      None => {
        // Constant term
        terms.push(Expr::Integer(reduced_c));
      }
      Some(trig_call) => {
        if reduced_c == 1 {
          terms.push(trig_call);
        } else if reduced_c == -1 {
          terms.push(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(trig_call),
          });
        } else {
          terms.push(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Times,
            left: Box::new(Expr::Integer(reduced_c)),
            right: Box::new(trig_call),
          });
        }
      }
    }
  }

  let numerator = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  };

  if reduced_denom == 1 {
    Some(numerator)
  } else {
    Some(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(numerator),
      right: Box::new(Expr::Integer(reduced_denom)),
    })
  }
}

fn try_reduce_with_power(a: &Expr, b: &Expr) -> Option<Expr> {
  // Check if a is Sin[x]^n or Cos[x]^n
  if let Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Power,
    left,
    right,
  } = a
    && let Expr::Integer(n) = right.as_ref()
    && *n >= 2
    && let Some(reduced_power) = reduce_trig_power(left, *n)
  {
    return Some(reduce_two_factor_product(&reduced_power, b));
  }
  None
}

/// Extract trig function name and argument.
fn extract_trig(expr: &Expr) -> Option<(&str, Expr)> {
  match expr {
    Expr::FunctionCall { name, args }
      if args.len() == 1
        && matches!(
          name.as_str(),
          "Sin" | "Cos" | "Tan" | "Cot" | "Sec" | "Csc"
        ) =>
    {
      Some((name.as_str(), args[0].clone()))
    }
    _ => None,
  }
}
