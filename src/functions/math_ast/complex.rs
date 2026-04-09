#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Re[z] - Real part of a complex number (for real numbers, returns the number itself)
pub fn re_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Re expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => return Ok(args[0].clone()),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      return Ok(inner[0].clone());
    }
    _ => {}
  }

  // Known real constants (Pi, E, etc.) are their own real part
  // Check before complex extraction which would convert them to floats
  if is_known_real(&args[0]) {
    return Ok(args[0].clone());
  }

  // Try exact integer/rational complex extraction: handles a + b*I patterns
  if let Some(((rn, rd), _)) = try_extract_complex_exact(&args[0]) {
    return Ok(make_rational(rn, rd));
  }

  // Try float complex extraction: handles 1.5 + 2.5*I patterns
  if let Some((re, _)) = try_extract_complex_float(&args[0]) {
    return Ok(Expr::Real(re));
  }

  // Handle I * real_expr: Re[I * real] = 0
  if let Some(real_part) = extract_i_times_real(&args[0]) {
    let _ = real_part; // only need to know it's purely imaginary
    return Ok(Expr::Integer(0));
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
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(_, _) => return Ok(Expr::Integer(0)),
    Expr::FunctionCall { name, args: inner }
      if name == "Complex" && inner.len() == 2 =>
    {
      return Ok(inner[1].clone());
    }
    _ => {}
  }

  // Known real constants (Pi, E, etc.) have imaginary part 0
  // Check before complex extraction which would convert them to floats
  if is_known_real(&args[0]) {
    return Ok(Expr::Integer(0));
  }

  // Try exact integer/rational complex extraction: handles a + b*I patterns
  if let Some((_, (in_, id))) = try_extract_complex_exact(&args[0]) {
    return Ok(make_rational(in_, id));
  }

  // Try float complex extraction: handles 1.5 + 2.5*I patterns
  if let Some((_, im)) = try_extract_complex_float(&args[0]) {
    return Ok(Expr::Real(im));
  }

  // Handle I * real_expr: Im[I * real] = real
  if let Some(real_part) = extract_i_times_real(&args[0]) {
    return Ok(real_part);
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "Im".to_string(),
    args: args.to_vec(),
  })
}

/// Extract the real factor from an expression of the form `I * real_expr`.
/// Returns Some(real_expr) if the expression is I times a NumericQ real expression,
/// or None otherwise.
fn extract_i_times_real(expr: &Expr) -> Option<Expr> {
  // Check for Times[I, ...] or Times[-1, I, ...] etc.
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
  {
    // Find if I is among the factors
    let mut i_count = 0;
    let mut other_factors: Vec<Expr> = Vec::new();
    for arg in args {
      if matches!(arg, Expr::Identifier(s) if s == "I") {
        i_count += 1;
      } else {
        other_factors.push(arg.clone());
      }
    }
    // Exactly one I, and all other factors are real-valued
    if i_count == 1
      && !other_factors.is_empty()
      && other_factors.iter().all(is_real_valued)
    {
      return if other_factors.len() == 1 {
        Some(other_factors.into_iter().next().unwrap())
      } else {
        Some(Expr::FunctionCall {
          name: "Times".to_string(),
          args: other_factors,
        })
      };
    }
  }
  None
}

/// Check if an expression is known to be real-valued (not just known real constants,
/// but also NumericQ expressions that evaluate to real numbers).
fn is_real_valued(expr: &Expr) -> bool {
  if is_known_real(expr) {
    return true;
  }
  // NumericQ expressions: check if they evaluate to a real number
  if crate::functions::predicate_ast::is_numeric_q_pub(expr) {
    // Try to evaluate numerically - if it gives a real, it's real-valued
    if let Some(val) = crate::functions::math_ast::try_eval_to_f64(expr) {
      return val.is_finite() || val.is_nan();
    }
  }
  false
}

/// Helper: check if an expression is a known real value (Integer, Real, or Rational)
pub fn is_known_real(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) => true,
    Expr::Constant(c) if c == "Pi" || c == "E" => true,
    Expr::Identifier(s)
      if s == "Pi"
        || s == "E"
        || s == "Infinity"
        || s == "EulerGamma"
        || s == "GoldenRatio"
        || s == "Catalan"
        || s == "Degree"
        || s == "Glaisher"
        || s == "Khinchin" =>
    {
      true
    }
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
pub fn conjugate_one(expr: &Expr) -> Result<Expr, InterpreterError> {
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
pub fn make_rational_times_pi(n: i128, d: i128) -> Expr {
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

/// Rationalize[x] or Rationalize[x, tol]
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

  if args.len() == 2 {
    // Rationalize[x, tol]: find rational with smallest denominator within tolerance
    let tolerance = expr_to_num(&args[1]).unwrap_or(f64::EPSILON);
    let (num, denom) = if tolerance == 0.0 {
      // Tolerance 0: find best rational approximation (exact float match)
      find_rational(x, 0.0, i64::MAX)
    } else {
      find_rational_smallest_denom(x, tolerance)
    };
    let approx = num as f64 / denom as f64;
    if tolerance > 0.0 && (approx - x).abs() >= tolerance {
      return Ok(num_to_expr(x));
    }
    if denom == 1 {
      Ok(Expr::Integer(num as i128))
    } else {
      Ok(Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(num as i128), Expr::Integer(denom as i128)],
      })
    }
  } else {
    // Rationalize[x] with no tolerance: for machine-precision numbers, use a default
    // tolerance to find a simple nearby rational, matching Wolfram Language behavior.
    // Wolfram uses approximately 10^(-Ceiling[MachinePrecision/2]) ≈ 10^-8,
    // but in practice a tolerance around 10^-6 gives the best match to Wolfram output.
    let default_tol = 5e-6;
    let (num, denom) = find_rational_smallest_denom(x, default_tol);
    let approx = num as f64 / denom as f64;
    if (approx - x).abs() <= default_tol {
      if denom == 1 {
        Ok(Expr::Integer(num as i128))
      } else {
        Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(num as i128), Expr::Integer(denom as i128)],
        })
      }
    } else {
      // No close-enough small-denominator match found; return as Real
      Ok(num_to_expr(x))
    }
  }
}

/// Find best rational approximation using continued fractions
pub fn find_rational(x: f64, tolerance: f64, max_denom: i64) -> (i64, i64) {
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

/// Find rational with smallest denominator within tolerance.
/// Uses continued fractions with semi-convergent checking.
pub fn find_rational_smallest_denom(x: f64, tolerance: f64) -> (i64, i64) {
  if x == 0.0 {
    return (0, 1);
  }

  let sign: i64 = if x < 0.0 { -1 } else { 1 };
  let x = x.abs();

  // p0, q0 = p_{n-2}, q_{n-2}; p1, q1 = p_{n-1}, q_{n-1}
  let mut p0: i64 = 0;
  let mut q0: i64 = 1;
  let mut p1: i64 = 1;
  let mut q1: i64 = 0;

  let mut xi = x;

  for _ in 0..50 {
    let ai = xi.floor() as i64;

    let p2 = match ai.checked_mul(p1).and_then(|v| v.checked_add(p0)) {
      Some(v) => v,
      None => break,
    };
    let q2 = match ai.checked_mul(q1).and_then(|v| v.checked_add(q0)) {
      Some(v) => v,
      None => break,
    };

    if q2 == 0 {
      break;
    }

    // Before accepting this full convergent, check semi-convergents
    // Semi-convergents: (p0 + j * p1) / (q0 + j * q1) for j = 1, ..., ai - 1
    if ai > 1 {
      for j in 1..ai {
        let sp = p0 + j * p1;
        let sq = q0 + j * q1;
        if sq > 0 {
          let approx = sp as f64 / sq as f64;
          if (approx - x).abs() < tolerance {
            return (sign * sp, sq);
          }
        }
      }
    }

    // Check the full convergent
    let approx = p2 as f64 / q2 as f64;
    if (approx - x).abs() < tolerance {
      return (sign * p2, q2);
    }

    let frac = xi - ai as f64;
    if frac.abs() < 1e-15 {
      return (sign * p2, q2);
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
pub fn is_reciprocal(expr: &Expr) -> bool {
  get_reciprocal_base(expr).is_some()
}

/// If expr is Power[base, -1], return Some(base), else None
pub fn get_reciprocal_base(expr: &Expr) -> Option<Expr> {
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
