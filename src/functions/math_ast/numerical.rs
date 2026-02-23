#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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
pub fn n_eval(expr: &Expr) -> Result<Expr, InterpreterError> {
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
      // Otherwise, recursively apply N to arguments and re-evaluate
      let new_args: Result<Vec<Expr>, _> = args.iter().map(n_eval).collect();
      let new_args = new_args?;
      // Re-evaluate the function with numeric arguments
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args,
      };
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
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
pub fn nominal_bits(precision: usize) -> usize {
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
pub fn bigfloat_to_string(
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
pub fn n_eval_arbitrary(
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
pub fn expr_to_bigfloat(
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
    _ => {
      // Scalar: Normalize[x] = x / Norm[x]
      let norm_val = norm_ast(args)?;
      // If norm is 0, return the original
      let is_zero = match &norm_val {
        Expr::Integer(0) => true,
        Expr::Real(f) if *f == 0.0 => true,
        _ => false,
      };
      if is_zero {
        return Ok(args[0].clone());
      }
      crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[args[0].clone(), norm_val],
      )
    }
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
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
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
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
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

pub fn power_expand_recursive(expr: &Expr) -> Expr {
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

pub fn collect_variables(expr: &Expr, vars: &mut Vec<Expr>) {
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

/// Core DFT computation shared by Fourier and InverseFourier.
/// `sign` is +1 for Fourier, -1 for InverseFourier (before applying `b`).
/// FourierParameters {a, b}: F_s = n^((a-1)/2) * sum_{r=0}^{n-1} u_r * exp(2*pi*i*b*(r*s)/n)
pub fn dft_core(
  data: &[(f64, f64)],
  param_a: f64,
  param_b: f64,
  inverse: bool,
) -> Vec<(f64, f64)> {
  let n = data.len();
  if n == 0 {
    return vec![];
  }
  let nf = n as f64;
  // For Fourier: scaling = n^((a-1)/2), exponent sign from b
  // For InverseFourier: scaling = n^((-1-a)/2), exponent sign from -b
  let (scaling, exp_sign) = if inverse {
    (nf.powf((-1.0 - param_a) / 2.0), -param_b)
  } else {
    (nf.powf((param_a - 1.0) / 2.0), param_b)
  };

  let two_pi_over_n = 2.0 * std::f64::consts::PI / nf;
  let mut result = Vec::with_capacity(n);

  for s in 0..n {
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for r in 0..n {
      let angle = two_pi_over_n * exp_sign * (r as f64) * (s as f64);
      let (sin_a, cos_a) = angle.sin_cos();
      let (ur, ui) = data[r];
      // (ur + ui*i) * (cos_a + sin_a*i) = (ur*cos_a - ui*sin_a) + (ur*sin_a + ui*cos_a)*i
      sum_re += ur * cos_a - ui * sin_a;
      sum_im += ur * sin_a + ui * cos_a;
    }
    result.push((scaling * sum_re, scaling * sum_im));
  }
  result
}

/// Round a floating-point number to clean up near-integer/near-half values.
/// This accounts for floating-point errors in DFT trig computations.
pub fn fourier_round(x: f64) -> f64 {
  if x.abs() < 1e-14 {
    return 0.0;
  }
  // Check if x is very close to an integer or half-integer
  let rounded = x.round();
  if (x - rounded).abs() < 1e-14 {
    return rounded;
  }
  // Check half-integer
  let doubled = (x * 2.0).round();
  if (x * 2.0 - doubled).abs() < 1e-13 {
    return doubled / 2.0;
  }
  x
}

/// Build an Expr for a Fourier/InverseFourier result element.
/// If `force_complex` is true, always output as Complex (even if im == 0).
pub fn fourier_result_to_expr(re: f64, im: f64, force_complex: bool) -> Expr {
  let re = fourier_round(re);
  let im = fourier_round(im);

  if force_complex || im != 0.0 {
    // Build via Complex[re, im] which the evaluator will format as a + b*I
    crate::evaluator::evaluate_function_call_ast(
      "Complex",
      &[Expr::Real(re), Expr::Real(im)],
    )
    .unwrap_or_else(|_| build_complex_float_expr(re, im))
  } else {
    Expr::Real(re)
  }
}

/// Parse FourierParameters option from args, returning (a, b).
/// Default is {0, 1}.
pub fn parse_fourier_parameters(
  args: &[Expr],
) -> Result<(f64, f64), InterpreterError> {
  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "FourierParameters")
    {
      if let Expr::List(params) = replacement.as_ref()
        && params.len() == 2
      {
        let a = try_eval_to_f64(&params[0]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: first parameter must be numeric".into(),
          )
        })?;
        let b = try_eval_to_f64(&params[1]).ok_or_else(|| {
          InterpreterError::EvaluationError(
            "FourierParameters: second parameter must be numeric".into(),
          )
        })?;
        return Ok((a, b));
      }
      return Err(InterpreterError::EvaluationError(
        "FourierParameters must be a list of two numbers".into(),
      ));
    }
  }
  Ok((0.0, 1.0)) // default
}

/// Shared implementation for Fourier and InverseFourier
pub fn fourier_impl(
  func_name: &str,
  args: &[Expr],
  inverse: bool,
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects 1 or 2 arguments",
      func_name
    )));
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      crate::capture_warning(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract numeric values as complex pairs
  let mut data: Vec<(f64, f64)> = Vec::with_capacity(items.len());
  for item in items {
    if let Some((re, im)) = try_extract_complex_float(item) {
      data.push((re, im));
    } else {
      crate::capture_warning(&format!(
        "{}::fftl: Argument {} is not a nonempty list or rectangular array of numeric quantities.",
        func_name,
        crate::syntax::expr_to_string(&args[0])
      ));
      return Ok(Expr::FunctionCall {
        name: func_name.to_string(),
        args: args.to_vec(),
      });
    }
  }

  let (param_a, param_b) = parse_fourier_parameters(&args[1..])?;
  let result = dft_core(&data, param_a, param_b, inverse);

  // Determine if any element has nonzero imaginary part
  let any_complex = result.iter().any(|(_, im)| {
    let im_r = fourier_round(*im);
    im_r != 0.0
  });

  let exprs: Vec<Expr> = result
    .iter()
    .map(|&(re, im)| fourier_result_to_expr(re, im, any_complex))
    .collect();

  Ok(Expr::List(exprs))
}

/// Fourier[list] or Fourier[list, opts] - Discrete Fourier transform
pub fn fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("Fourier", args, false)
}

/// InverseFourier[list] or InverseFourier[list, opts] - Inverse discrete Fourier transform
pub fn inverse_fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fourier_impl("InverseFourier", args, true)
}

/// Wynn epsilon algorithm for series acceleration.
/// Takes a sequence of partial sums and returns an accelerated estimate.
/// Polynomial extrapolation using Neville's algorithm.
/// Given points (x_i, y_i) representing (1/n, S_n), extrapolate to x=0.
pub fn neville_extrapolation(xs: &[f64], ys: &[f64]) -> f64 {
  let n = xs.len();
  if n == 0 {
    return 0.0;
  }
  if n == 1 {
    return ys[0];
  }

  let mut c = ys.to_vec();
  for j in 1..n {
    for i in (j..n).rev() {
      // Neville's algorithm for interpolation at x=0:
      // c[i] = (x_i * c[i-1] - x_{i-j} * c[i]) / (x_i - x_{i-j})
      // Since we evaluate at x=0, this simplifies to:
      let denom = xs[i] - xs[i - j];
      if denom.abs() < 1e-300 {
        continue;
      }
      c[i] = (xs[i] * c[i - 1] - xs[i - j] * c[i]) / denom;
    }
  }
  c[n - 1]
}

/// NSum[expr, {i, min, max}] - Numerical summation
pub fn nsum_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: "NSum".to_string(),
      args: args.to_vec(),
    });
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "NSum".to_string(),
      args: args.to_vec(),
    });
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Check for infinite sum
  let is_infinite = matches!(&items[2], Expr::Identifier(s) if s == "Infinity");

  if is_infinite {
    // Numerical infinite sum using polynomial extrapolation in 1/n
    // 1. Compute partial sums at several checkpoint values of n
    // 2. Extrapolate to n → ∞ (i.e., 1/n → 0) using Neville's algorithm
    let checkpoints: Vec<i64> = vec![50, 100, 150, 200, 300, 400, 500];
    let mut running_sum = 0.0_f64;
    let mut checkpoint_idx = 0;
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    let max_n = *checkpoints.last().unwrap();
    for i in min_val..(min_val + max_n) {
      let sub_val = Expr::Integer(i as i128);
      let substituted =
        crate::syntax::substitute_variable(body, &var_name, &sub_val);
      let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
      let term = match try_eval_to_f64(&val) {
        Some(f) => f,
        None => {
          return Ok(Expr::FunctionCall {
            name: "NSum".to_string(),
            args: args.to_vec(),
          });
        }
      };

      if !term.is_finite() {
        break;
      }
      running_sum += term;

      let current_n = i - min_val + 1;
      if checkpoint_idx < checkpoints.len()
        && current_n == checkpoints[checkpoint_idx]
      {
        xs.push(1.0 / current_n as f64);
        ys.push(running_sum);
        checkpoint_idx += 1;
      }
    }

    if xs.is_empty() {
      return Ok(Expr::Real(running_sum));
    }

    let result = neville_extrapolation(&xs, &ys);
    return Ok(Expr::Real(result));
  }

  // Finite sum
  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NSum".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut sum = 0.0_f64;
  for i in min_val..=max_val {
    let sub_val = Expr::Integer(i as i128);
    let substituted =
      crate::syntax::substitute_variable(body, &var_name, &sub_val);
    let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
    let term = match try_eval_to_f64(&val) {
      Some(f) => f,
      None => {
        return Ok(Expr::FunctionCall {
          name: "NSum".to_string(),
          args: args.to_vec(),
        });
      }
    };
    sum += term;
  }

  Ok(Expr::Real(sum))
}

/// NProduct[f, {i, imin, imax}] - Numerical product
pub fn nproduct_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: "NProduct".to_string(),
      args: args.to_vec(),
    });
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let items = match iter_spec {
    Expr::List(items) if items.len() >= 2 => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let var_name = match &items[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if items.len() < 3 {
    return Ok(Expr::FunctionCall {
      name: "NProduct".to_string(),
      args: args.to_vec(),
    });
  }

  let min_val = match try_eval_to_f64(&items[1]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let max_val = match try_eval_to_f64(&items[2]) {
    Some(v) => v as i64,
    None => {
      return Ok(Expr::FunctionCall {
        name: "NProduct".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut product = 1.0_f64;
  for i in min_val..=max_val {
    let sub_val = Expr::Integer(i as i128);
    let substituted =
      crate::syntax::substitute_variable(body, &var_name, &sub_val);
    let val = crate::evaluator::evaluate_expr_to_expr(&substituted)?;
    let term = match try_eval_to_f64(&val) {
      Some(f) => f,
      None => {
        return Ok(Expr::FunctionCall {
          name: "NProduct".to_string(),
          args: args.to_vec(),
        });
      }
    };
    product *= term;
  }

  Ok(Expr::Real(product))
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
