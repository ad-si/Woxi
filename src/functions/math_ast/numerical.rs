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
    // Precision can be a numeric expression (e.g. N[Pi, Pi])
    let precision = match &args[1] {
      Expr::Integer(n) if *n > 0 => *n as usize,
      other => {
        // Try evaluating to a float
        if let Some(v) = try_eval_to_f64(other) {
          let p = v.floor() as i128;
          if p > 0 {
            p as usize
          } else {
            return Err(InterpreterError::EvaluationError(
              "N: precision must be a positive number".into(),
            ));
          }
        } else {
          return Err(InterpreterError::EvaluationError(
            "N: precision must be a positive number".into(),
          ));
        }
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
      // Special case for functions that stay symbolic when called
      // directly with a Real but have a numeric value triggered by N[].
      if args.len() == 1 && let Some(n) = expr_to_i128(&args[0]) {
        if name == "AiryAiZero"
          && let Some(r) = crate::functions::math_ast::airy_ai_zero_n_eval(n)
        {
          return Ok(r);
        }
        if name == "AiryBiZero"
          && let Some(r) = crate::functions::math_ast::airy_bi_zero_n_eval(n)
        {
          return Ok(r);
        }
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
      let new_expr = Expr::BinaryOp {
        op: *op,
        left: Box::new(n_eval(left)?),
        right: Box::new(n_eval(right)?),
      };
      // Re-evaluate to allow numeric simplification (e.g. complex powers)
      let result = match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => result,
        Err(_) => new_expr,
      };
      // If the result is still a Power with complex operands, force
      // numeric evaluation via z^w = exp(w * log(z))
      if let Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: ref base,
        right: ref exp,
      } = result
        && let (Some((a, b)), Some((c, d))) = (
          try_extract_complex_float(base),
          try_extract_complex_float(exp),
        )
        && (b != 0.0 || d != 0.0)
      {
        let abs_z = (a * a + b * b).sqrt();
        if abs_z > 0.0 {
          let ln_abs = abs_z.ln();
          let arg_z = b.atan2(a);
          let re_exp = c * ln_abs - d * arg_z;
          let im_exp = d * ln_abs + c * arg_z;
          let mag = re_exp.exp();
          let re = mag * im_exp.cos();
          let im = mag * im_exp.sin();
          let re = if re.abs() < 1e-15 { 0.0 } else { re };
          let im = if im.abs() < 1e-15 { 0.0 } else { im };
          if im == 0.0 {
            // Preserve complex form: re + 0.*I (matching Wolfram's convention
            // for complex power results where imaginary part is numerically zero)
            return Ok(Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Plus,
              left: Box::new(Expr::Real(re)),
              right: Box::new(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Times,
                left: Box::new(Expr::Real(0.0)),
                right: Box::new(Expr::Identifier("I".to_string())),
              }),
            });
          }
          return Ok(build_complex_float_expr(re, im));
        }
      }
      Ok(result)
    }
    // Rule: keep the pattern, apply N to the replacement
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval(replacement)?),
    }),
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
  // round up to the next 64-bit word boundary to match Wolfram's output
  // behavior (which displays all digits from the full word-aligned
  // precision, giving slightly more digits than requested).
  let base_bits =
    (precision as f64 * std::f64::consts::LOG2_10).ceil() as usize;
  // Round up to next word boundary
  let bits = (base_bits + 63) & !63;
  bits.max(128)
}

/// Extract an integer value from an Expr, if it is one.
fn try_as_integer(expr: &Expr) -> Option<i128> {
  match expr {
    Expr::Integer(n) => Some(*n),
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      if let Expr::Integer(n) = operand.as_ref() {
        Some(-n)
      } else {
        None
      }
    }
    _ => None,
  }
}

/// Compute base^n for integer n, handling negative exponents.
/// Uses BigFloat::powi for the absolute value, then inverts if needed.
fn bigfloat_powi(
  base: &astro_float::BigFloat,
  n: i128,
  bits: usize,
  rm: astro_float::RoundingMode,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if n == 0 {
    return BigFloat::from_i32(1, bits);
  }

  let abs_n = n.unsigned_abs() as usize;
  let result = base.powi(abs_n, bits, rm);

  if n < 0 {
    BigFloat::from_i32(1, bits).div(&result, bits, rm)
  } else {
    result
  }
}

/// Convert a BigFloat to a decimal string.
/// If `max_digits` is Some(n), truncate the output to at most n significant digits.
pub fn bigfloat_to_string(
  bf: &astro_float::BigFloat,
  max_digits: Option<usize>,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  if bf.is_zero() {
    return Ok("0.".to_string());
  }

  let (sign, digits, exponent) = bf
    .convert_to_radix(astro_float::Radix::Dec, rm, cc)
    .map_err(|e| {
      InterpreterError::EvaluationError(format!("N: format error: {}", e))
    })?;

  if digits.is_empty() || digits.iter().all(|&d| d == 0) {
    return Ok("0.".to_string());
  }

  let is_negative = sign == astro_float::Sign::Neg;
  let prefix = if is_negative { "-" } else { "" };

  // Convert digit values to ASCII string
  let digit_str: String = digits.iter().map(|&d| (b'0' + d) as char).collect();

  // Truncate to max_digits if specified
  let digit_str = if let Some(max_d) = max_digits {
    if digit_str.len() > max_d {
      digit_str[..max_d].to_string()
    } else {
      digit_str
    }
  } else {
    digit_str
  };

  // exponent from convert_to_radix: value = 0.d1d2d3... * 10^exponent
  let decimal_exp = exponent as i64;

  if decimal_exp <= 0 {
    // Number like 0.000xxxx
    let zeros = (-decimal_exp) as usize;
    let trimmed = digit_str.trim_end_matches('0');
    if trimmed.is_empty() {
      Ok(format!("{}0.", prefix))
    } else {
      let frac = format!("{}{}", "0".repeat(zeros), trimmed);
      let frac = frac.trim_end_matches('0');
      if frac.is_empty() {
        Ok(format!("{}0.", prefix))
      } else {
        Ok(format!("{}0.{}", prefix, frac))
      }
    }
  } else {
    let dp = decimal_exp as usize;
    if dp >= digit_str.len() {
      // All digits are in the integer part
      let padded =
        format!("{}{}", &digit_str, "0".repeat(dp - digit_str.len()));
      Ok(format!("{}{}.", prefix, padded))
    } else {
      // Some digits before decimal, some after
      let int_part = &digit_str[..dp];
      let frac_part = digit_str[dp..].trim_end_matches('0');
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

  // Machine-precision Reals already live at MachinePrecision and cannot be
  // promoted by N — wolframscript returns the Real unchanged.
  if let Expr::Real(_) = expr {
    return Ok(expr.clone());
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

  // Try full conversion to BigFloat first (fast path for purely numeric expressions)
  match expr_to_bigfloat(expr, bits, rm, &mut cc) {
    Ok(result) => {
      let decimal = bigfloat_to_string(&result, None, rm, &mut cc)?;
      Ok(Expr::BigFloat(decimal, precision as f64))
    }
    Err(_) => {
      // Try complex BigFloat evaluation (handles expressions with I)
      if let Ok((re, im)) = expr_to_complex_bigfloat(expr, bits, rm, &mut cc) {
        // For complex function results, compute per-component precision markers
        if let Expr::FunctionCall { name: _, args } = expr
          && !im.is_zero()
          && args.len() == 1
          && let Ok(input_complex) =
            expr_to_complex_bigfloat(&args[0], bits, rm, &mut cc)
          && let Ok((prec_re_str, prec_im_str)) =
            compute_complex_precision_markers(
              &input_complex.0,
              &input_complex.1,
              &re,
              &im,
              precision,
              rm,
              &mut cc,
            )
        {
          return build_complex_result_with_string_precision(
            re,
            im,
            &prec_re_str,
            &prec_im_str,
            precision,
            rm,
            &mut cc,
          );
        }
        return build_complex_bigfloat_result(re, im, precision, rm, &mut cc);
      }
      // Fall back to partial evaluation: convert numeric sub-expressions
      // to arbitrary precision while leaving symbolic parts as-is
      n_eval_arbitrary_partial(expr, precision, bits, rm, &mut cc)
    }
  }
}

/// Recursively apply arbitrary-precision evaluation to sub-expressions.
/// Numeric parts are converted to BigFloat; symbolic parts are left as-is.
fn n_eval_arbitrary_partial(
  expr: &Expr,
  precision: usize,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  // If the whole expression can be converted to BigFloat, do it
  if let Ok(result) = expr_to_bigfloat(expr, bits, rm, cc) {
    let decimal = bigfloat_to_string(&result, None, rm, cc)?;
    return Ok(Expr::BigFloat(decimal, precision as f64));
  }

  match expr {
    Expr::FunctionCall { name, args } => {
      let new_args: Result<Vec<Expr>, _> = args
        .iter()
        .map(|a| n_eval_arbitrary_partial(a, precision, bits, rm, cc))
        .collect();
      let new_expr = Expr::FunctionCall {
        name: name.clone(),
        args: new_args?,
      };
      // Try to re-evaluate after converting numeric args
      match crate::evaluator::evaluate_expr_to_expr(&new_expr) {
        Ok(result) => Ok(result),
        Err(_) => Ok(new_expr),
      }
    }
    Expr::BinaryOp { op, left, right } => {
      let l = n_eval_arbitrary_partial(left, precision, bits, rm, cc)?;
      let r = n_eval_arbitrary_partial(right, precision, bits, rm, cc)?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let inner = n_eval_arbitrary_partial(operand, precision, bits, rm, cc)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(inner),
      })
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Ok(Expr::Rule {
      pattern: pattern.clone(),
      replacement: Box::new(n_eval_arbitrary_partial(
        replacement,
        precision,
        bits,
        rm,
        cc,
      )?),
    }),
    // Identifiers, symbols, and other non-numeric expressions: leave as-is
    _ => Ok(expr.clone()),
  }
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
    Expr::Identifier(name) if name == "GoldenRatio" => {
      // GoldenRatio = (1 + Sqrt[5]) / 2
      let five = BigFloat::from_i32(5, bits);
      let sqrt5 = five.sqrt(bits, rm);
      let one = BigFloat::from_i32(1, bits);
      let numer = one.add(&sqrt5, bits, rm);
      let two = BigFloat::from_i32(2, bits);
      Ok(numer.div(&two, bits, rm))
    }
    Expr::Identifier(name) if name == "EulerGamma" => {
      Ok(compute_euler_gamma(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Catalan" => {
      Ok(compute_catalan(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Glaisher" => {
      Ok(compute_glaisher(bits, rm, cc))
    }
    Expr::Identifier(name) if name == "Khinchin" => {
      Ok(compute_khinchin(bits, rm, cc))
    }
    // MachinePrecision = Log10[2^53] = 53 * Log10[2].
    // `N[MachinePrecision, 30]` should yield the arbitrary-precision
    // decimal expansion `15.9545897701910033463281614204`.
    Expr::Identifier(name) if name == "MachinePrecision" => {
      let two = BigFloat::from_i32(2, bits);
      let ten = BigFloat::from_i32(10, bits);
      let log2 = two.ln(bits, rm, cc);
      let log10 = ten.ln(bits, rm, cc);
      let log10_2 = log2.div(&log10, bits, rm);
      let fifty_three = BigFloat::from_i32(53, bits);
      Ok(fifty_three.mul(&log10_2, bits, rm))
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let val = expr_to_bigfloat(operand, bits, rm, cc)?;
      Ok(val.neg())
    }
    Expr::BinaryOp { op, left, right } => {
      // For integer exponents, use powi (repeated squaring) for efficiency
      if matches!(op, BinaryOperator::Power)
        && let Some(n) = try_as_integer(right)
      {
        let base = expr_to_bigfloat(left, bits, rm, cc)?;
        return Ok(bigfloat_powi(&base, n, bits, rm));
      }
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
          if let Some(n) = try_as_integer(&args[1]) {
            let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
            return Ok(bigfloat_powi(&base, n, bits, rm));
          }
          let base = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          let exp = expr_to_bigfloat(&args[1], bits, rm, cc)?;
          Ok(base.pow(&exp, bits, rm, cc))
        }
        "Erf" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erf(&x, bits, rm, cc))
        }
        "Erfc" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfc(&x, bits, rm, cc))
        }
        "Erfi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_erfi(&x, bits, rm, cc))
        }
        "ExpIntegralEi" if args.len() == 1 => {
          let x = expr_to_bigfloat(&args[0], bits, rm, cc)?;
          Ok(bigfloat_exp_integral_ei(&x, bits, rm, cc))
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

/// Convert an expression to a complex (BigFloat, BigFloat) pair with given precision.
/// Returns (real_part, imaginary_part) as BigFloats.
/// Handles expressions involving the imaginary unit I.
fn expr_to_complex_bigfloat(
  expr: &Expr,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use crate::syntax::BinaryOperator;
  use astro_float::BigFloat;

  // Fast path: if purely real, delegate
  if let Ok(val) = expr_to_bigfloat(expr, bits, rm, cc) {
    return Ok((val, BigFloat::from_i32(0, bits)));
  }

  match expr {
    // I → (0, 1)
    Expr::Identifier(name) if name == "I" => {
      Ok((BigFloat::from_i32(0, bits), BigFloat::from_i32(1, bits)))
    }
    // Unary minus
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (re, im) = expr_to_complex_bigfloat(operand, bits, rm, cc)?;
      Ok((re.neg(), im.neg()))
    }
    // Binary operations
    Expr::BinaryOp { op, left, right } => {
      let (lr, li) = expr_to_complex_bigfloat(left, bits, rm, cc)?;
      let (rr, ri) = expr_to_complex_bigfloat(right, bits, rm, cc)?;
      match op {
        BinaryOperator::Plus => {
          Ok((lr.add(&rr, bits, rm), li.add(&ri, bits, rm)))
        }
        BinaryOperator::Minus => {
          Ok((lr.sub(&rr, bits, rm), li.sub(&ri, bits, rm)))
        }
        BinaryOperator::Times => {
          // (lr + li*i) * (rr + ri*i) = (lr*rr - li*ri) + (lr*ri + li*rr)*i
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          Ok((re, im))
        }
        BinaryOperator::Divide => {
          // (lr + li*i) / (rr + ri*i)
          let denom =
            rr.mul(&rr, bits, rm).add(&ri.mul(&ri, bits, rm), bits, rm);
          let re = lr.mul(&rr, bits, rm).add(&li.mul(&ri, bits, rm), bits, rm);
          let im = li.mul(&rr, bits, rm).sub(&lr.mul(&ri, bits, rm), bits, rm);
          Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
        }
        BinaryOperator::Power => {
          // Only handle real^integer for now
          if li.is_zero()
            && ri.is_zero()
            && let Some(n) = try_as_integer(right)
          {
            return Ok((
              bigfloat_powi(&lr, n, bits, rm),
              BigFloat::from_i32(0, bits),
            ));
          }
          Err(InterpreterError::EvaluationError(
            "N: complex Power not supported yet".into(),
          ))
        }
        _ => Err(InterpreterError::EvaluationError(
          "N: unsupported binary operator for complex arbitrary precision"
            .into(),
        )),
      }
    }
    // Function calls
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Times" => {
        let mut result =
          (BigFloat::from_i32(1, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          let (lr, li) = result;
          let re = lr.mul(&rr, bits, rm).sub(&li.mul(&ri, bits, rm), bits, rm);
          let im = lr.mul(&ri, bits, rm).add(&li.mul(&rr, bits, rm), bits, rm);
          result = (re, im);
        }
        Ok(result)
      }
      "Plus" => {
        let mut result =
          (BigFloat::from_i32(0, bits), BigFloat::from_i32(0, bits));
        for arg in args {
          let (rr, ri) = expr_to_complex_bigfloat(arg, bits, rm, cc)?;
          result = (result.0.add(&rr, bits, rm), result.1.add(&ri, bits, rm));
        }
        Ok(result)
      }
      "Complex" if args.len() == 2 => {
        let re = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let im = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((re, im))
      }
      "Sin" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((sin_a.mul(&cosh_b, bits, rm), cos_a.mul(&sinh_b, bits, rm)))
      }
      "Cos" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        Ok((
          cos_a.mul(&cosh_b, bits, rm),
          sin_a.mul(&sinh_b, bits, rm).neg(),
        ))
      }
      "Tan" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tan(z) = sin(z)/cos(z)
        let sin_a = a.sin(bits, rm, cc);
        let cos_a = a.cos(bits, rm, cc);
        let cosh_b = b.cosh(bits, rm, cc);
        let sinh_b = b.sinh(bits, rm, cc);
        // sin(z) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        let sr = sin_a.mul(&cosh_b, bits, rm);
        let si = cos_a.mul(&sinh_b, bits, rm);
        // cos(z) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        let cr = cos_a.mul(&cosh_b, bits, rm);
        let ci = sin_a.mul(&sinh_b, bits, rm).neg();
        // (sr + si*i) / (cr + ci*i)
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Exp" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
        let exp_a = a.exp(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((exp_a.mul(&cos_b, bits, rm), exp_a.mul(&sin_b, bits, rm)))
      }
      "Sinh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((sinh_a.mul(&cos_b, bits, rm), cosh_a.mul(&sin_b, bits, rm)))
      }
      "Cosh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        Ok((cosh_a.mul(&cos_b, bits, rm), sinh_a.mul(&sin_b, bits, rm)))
      }
      "Tanh" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // tanh(z) = sinh(z)/cosh(z)
        let sinh_a = a.sinh(bits, rm, cc);
        let cosh_a = a.cosh(bits, rm, cc);
        let cos_b = b.cos(bits, rm, cc);
        let sin_b = b.sin(bits, rm, cc);
        let sr = sinh_a.mul(&cos_b, bits, rm);
        let si = cosh_a.mul(&sin_b, bits, rm);
        let cr = cosh_a.mul(&cos_b, bits, rm);
        let ci = sinh_a.mul(&sin_b, bits, rm);
        let denom = cr.mul(&cr, bits, rm).add(&ci.mul(&ci, bits, rm), bits, rm);
        let re = sr.mul(&cr, bits, rm).add(&si.mul(&ci, bits, rm), bits, rm);
        let im = si.mul(&cr, bits, rm).sub(&sr.mul(&ci, bits, rm), bits, rm);
        Ok((re.div(&denom, bits, rm), im.div(&denom, bits, rm)))
      }
      "Log" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        // log(a+bi) = ln(|z|) + i*arg(z)
        // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let ln_abs = abs_val.ln(bits, rm, cc);
        // atan2(b, a) implemented using atan
        let arg = if a.is_zero() {
          let half_pi =
            cc.pi(bits, rm).div(&BigFloat::from_i32(2, bits), bits, rm);
          if b.is_negative() {
            half_pi.neg()
          } else {
            half_pi
          }
        } else if a.is_negative() {
          let atan_val = b.div(&a, bits, rm).atan(bits, rm, cc);
          if b.is_negative() {
            atan_val.sub(&cc.pi(bits, rm), bits, rm)
          } else {
            atan_val.add(&cc.pi(bits, rm), bits, rm)
          }
        } else {
          b.div(&a, bits, rm).atan(bits, rm, cc)
        };
        Ok((ln_abs, arg))
      }
      "Sqrt" if args.len() == 1 => {
        // sqrt(a+bi) = sqrt((|z|+a)/2) + i*sign(b)*sqrt((|z|-a)/2)
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        let abs_val = abs_sq.sqrt(bits, rm);
        let two = BigFloat::from_i32(2, bits);
        let re = abs_val.add(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        let mut im =
          abs_val.sub(&a, bits, rm).div(&two, bits, rm).sqrt(bits, rm);
        if b.is_negative() {
          im = im.neg();
        }
        Ok((re, im))
      }
      "Rational" if args.len() == 2 => {
        let n = expr_to_bigfloat(&args[0], bits, rm, cc)?;
        let d = expr_to_bigfloat(&args[1], bits, rm, cc)?;
        Ok((n.div(&d, bits, rm), BigFloat::from_i32(0, bits)))
      }
      "Power" if args.len() == 2 => {
        if let Some(n) = try_as_integer(&args[1]) {
          let (re, im) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
          if im.is_zero() {
            return Ok((
              bigfloat_powi(&re, n, bits, rm),
              BigFloat::from_i32(0, bits),
            ));
          }
        }
        Err(InterpreterError::EvaluationError(
          "N: complex Power not fully supported".into(),
        ))
      }
      "Abs" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        let abs_sq = a.mul(&a, bits, rm).add(&b.mul(&b, bits, rm), bits, rm);
        Ok((abs_sq.sqrt(bits, rm), BigFloat::from_i32(0, bits)))
      }
      "ExpIntegralEi" if args.len() == 1 => {
        let (a, b) = expr_to_complex_bigfloat(&args[0], bits, rm, cc)?;
        complex_exp_integral_ei(a, b, bits, rm, cc)
      }
      _ => Err(InterpreterError::EvaluationError(format!(
        "N: cannot evaluate {}[...] to complex arbitrary precision",
        name
      ))),
    },
    _ => Err(InterpreterError::EvaluationError(format!(
      "N: cannot evaluate expression to complex arbitrary precision: {}",
      crate::syntax::expr_to_string(expr)
    ))),
  }
}

/// Build a properly formatted complex result from BigFloat real and imaginary parts.
fn build_complex_bigfloat_result(
  re: astro_float::BigFloat,
  im: astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let i_expr = Expr::Identifier("I".to_string());
  let max_digits: Option<usize> = None;

  if im.is_zero() {
    let re_str = bigfloat_to_string(&re, None, rm, cc)?;
    return Ok(Expr::BigFloat(re_str, precision as f64));
  }

  let im_negative = im.is_negative();
  let im_abs = if im_negative { im.neg() } else { im.clone() };
  let im_str = bigfloat_to_string(&im_abs, max_digits, rm, cc)?;

  let im_bf = Expr::BigFloat(im_str, precision as f64);

  // Build |im| * I term (always positive coefficient)
  let abs_im_term = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(im_bf),
    right: Box::new(i_expr),
  };

  if re.is_zero() {
    if im_negative {
      // Pure negative imaginary: -|im|*I
      let neg_im_str = bigfloat_to_string(&im, max_digits, rm, cc)?;
      let neg_im_bf = Expr::BigFloat(neg_im_str, precision as f64);
      return Ok(Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left: Box::new(neg_im_bf),
        right: Box::new(Expr::Identifier("I".to_string())),
      });
    }
    return Ok(abs_im_term);
  }

  let re_str = bigfloat_to_string(&re, max_digits, rm, cc)?;
  let re_bf = Expr::BigFloat(re_str, precision as f64);

  if im_negative {
    // re - |im|*I
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(re_bf),
      right: Box::new(abs_im_term),
    })
  } else {
    // re + |im|*I
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(re_bf),
      right: Box::new(abs_im_term),
    })
  }
}

/// Build a complex result with per-component string precision markers.
/// Uses Expr::Raw to embed the precision marker directly in the formatted string.
fn build_complex_result_with_string_precision(
  re: astro_float::BigFloat,
  im: astro_float::BigFloat,
  prec_re_str: &str,
  prec_im_str: &str,
  _precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<Expr, InterpreterError> {
  let re_str = bigfloat_to_string(&re, None, rm, cc)?;

  let re_raw = Expr::Raw(format!("{}`{}", re_str, prec_re_str));

  let im_negative = im.is_negative();
  let im_abs_str = if im_negative {
    bigfloat_to_string(&im.neg(), None, rm, cc)?
  } else {
    bigfloat_to_string(&im, None, rm, cc)?
  };
  let im_raw = Expr::Raw(format!("{}`{}", im_abs_str, prec_im_str));

  let i_expr = Expr::Identifier("I".to_string());
  let abs_im_term = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Times,
    left: Box::new(im_raw),
    right: Box::new(i_expr),
  };

  if im_negative {
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left: Box::new(re_raw),
      right: Box::new(abs_im_term),
    })
  } else {
    Ok(Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left: Box::new(re_raw),
      right: Box::new(abs_im_term),
    })
  }
}

/// Compute per-component precision markers for a complex function evaluation.
///
/// Uses the formula:
///   accuracy = p + log10(|input|) - log10(|output|)
///   precision_component = accuracy + log10(|component|)
///
/// Computes using BigFloat arithmetic for accuracy, then converts to f64.
fn compute_complex_precision_markers(
  in_re: &astro_float::BigFloat,
  in_im: &astro_float::BigFloat,
  out_re: &astro_float::BigFloat,
  out_im: &astro_float::BigFloat,
  precision: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(String, String), InterpreterError> {
  use astro_float::BigFloat;
  let pbits = 256; // high precision for the log computations

  // |input| = sqrt(in_re^2 + in_im^2)
  let in_abs_sq =
    in_re
      .mul(in_re, pbits, rm)
      .add(&in_im.mul(in_im, pbits, rm), pbits, rm);
  let in_abs = in_abs_sq.sqrt(pbits, rm);

  // |output| = sqrt(out_re^2 + out_im^2)
  let out_abs_sq = out_re.mul(out_re, pbits, rm).add(
    &out_im.mul(out_im, pbits, rm),
    pbits,
    rm,
  );
  let out_abs = out_abs_sq.sqrt(pbits, rm);

  // log10(x) = ln(x) / ln(10)
  let ln10 = BigFloat::from_i32(10, pbits).ln(pbits, rm, cc);
  let log10_in = in_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_out = out_abs.ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_re = out_re.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);
  let log10_im = out_im.abs().ln(pbits, rm, cc).div(&ln10, pbits, rm);

  let p_bf = BigFloat::from_i32(precision as i32, pbits);
  // accuracy = p + log10(|input|) - log10(|output|)
  let accuracy = p_bf.add(&log10_in, pbits, rm).sub(&log10_out, pbits, rm);

  // precision_re = accuracy + log10(|re|)
  let prec_re_bf = accuracy.add(&log10_re, pbits, rm);
  // precision_im = accuracy + log10(|im|)
  let prec_im_bf = accuracy.add(&log10_im, pbits, rm);

  // Format precision markers directly from BigFloat with f64-like precision
  // This avoids f64 rounding issues by going from BigFloat → string directly
  let prec_re_s = format_bigfloat_as_precision_marker(&prec_re_bf, rm, cc)?;
  let prec_im_s = format_bigfloat_as_precision_marker(&prec_im_bf, rm, cc)?;

  Ok((prec_re_s, prec_im_s))
}

/// Format a BigFloat as a precision marker string, matching Wolfram's formatting.
/// Wolfram displays precision markers with ~15-17 significant digits (f64-like precision),
/// stripping trailing zeros.
fn format_bigfloat_as_precision_marker(
  bf: &astro_float::BigFloat,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<String, InterpreterError> {
  // Format with ~17 significant digits (matching f64 precision)
  let full_str = bigfloat_to_string(bf, Some(17), rm, cc)?;

  // Parse to f64 and back to get Wolfram-style formatting
  // Wolfram uses a formatting that shows all distinguishing digits
  let val: f64 = full_str.trim_end_matches('.').parse().unwrap_or(0.0);

  if val.fract() == 0.0 {
    Ok(format!("{}.", val as i64))
  } else {
    // Use Rust's Debug format which shows the exact f64 representation
    // with minimum digits to uniquely represent the value
    Ok(format!("{:?}", val))
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

  // Handle list first argument: Rescale[{x1, x2, ...}, range] maps over elements
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        let mut new_args = args.to_vec();
        new_args[0] = item.clone();
        rescale_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?));
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
/// Recursively check whether an expression tree contains any
/// inexact-Real or BigFloat leaf. Used by Norm (and similar) to
/// decide between exact/symbolic and machine-precision numerical
/// evaluation.
fn contains_inexact_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::List(items) => items.iter().any(contains_inexact_real),
    Expr::FunctionCall { args, .. } => args.iter().any(contains_inexact_real),
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact_real(left) || contains_inexact_real(right)
    }
    Expr::UnaryOp { operand, .. } => contains_inexact_real(operand),
    _ => false,
  }
}

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
      let p = p_val.unwrap_or(2.0);

      // "Inexact" mode: any item contains a Real/BigFloat leaf — collapse
      // to a machine-precision numeric result, mirroring Wolfram's
      // behavior (Norm[{1.0, 2, 3}] → 3.741…).
      // Otherwise stay in "exact" mode and build a symbolic expression.
      let inexact = items.iter().any(contains_inexact_real);

      if inexact {
        let mut vals = Vec::with_capacity(items.len());
        for item in items {
          match try_eval_to_f64(item) {
            Some(v) => vals.push(v),
            None => {
              return Ok(Expr::FunctionCall {
                name: "Norm".to_string(),
                args: args.to_vec(),
              });
            }
          }
        }
        if p == 1.0 {
          let result: f64 = vals.iter().map(|x| x.abs()).sum();
          return Ok(num_to_expr(result));
        }
        if is_infinity {
          let result = vals.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
          return Ok(num_to_expr(result));
        }
        let sum: f64 = vals.iter().map(|x| x.abs().powf(p)).sum();
        return Ok(num_to_expr(sum.powf(1.0 / p)));
      }

      // Exact/symbolic mode.
      use crate::evaluator::evaluate_expr_to_expr;

      // For each item decide whether to wrap in Abs: if the item is a
      // numerically-evaluable (hence known-real) expression — integers,
      // rationals, Pi, Sin[1], 2 Sin[2], … — drop the Abs and build
      // item^p directly so that known scalars combine. For unknown
      // symbols (x, f[x]) keep Abs to preserve correctness over ℂ.
      let is_real_valued =
        |item: &Expr| -> bool { try_eval_to_f64(item).is_some() };

      if is_infinity {
        // Max[Abs[x], Abs[y], ...]
        let abs_items: Vec<Expr> = items
          .iter()
          .map(|item| Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![item.clone()],
          })
          .collect();
        return evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Max".to_string(),
          args: abs_items,
        });
      }

      if p == 1.0 {
        // Sum of Abs[item]
        let terms: Vec<Expr> = items
          .iter()
          .map(|item| Expr::FunctionCall {
            name: "Abs".to_string(),
            args: vec![item.clone()],
          })
          .collect();
        let sum = if terms.len() == 1 {
          terms.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: terms,
          }
        };
        return evaluate_expr_to_expr(&sum);
      }

      if p == 2.0 {
        // Sqrt[Plus[item^2, ...]] (or Abs[item]^2 for unknown items)
        let sq_items: Vec<Expr> = items
          .iter()
          .map(|item| {
            let base = if is_real_valued(item) {
              item.clone()
            } else {
              Expr::FunctionCall {
                name: "Abs".to_string(),
                args: vec![item.clone()],
              }
            };
            Expr::FunctionCall {
              name: "Power".to_string(),
              args: vec![base, Expr::Integer(2)],
            }
          })
          .collect();
        let sum = if sq_items.len() == 1 {
          sq_items.into_iter().next().unwrap()
        } else {
          Expr::FunctionCall {
            name: "Plus".to_string(),
            args: sq_items,
          }
        };
        let sum_eval = evaluate_expr_to_expr(&sum)?;
        return evaluate_expr_to_expr(&make_sqrt(sum_eval));
      }

      // Fallback: leave unevaluated for other p values
      Ok(Expr::FunctionCall {
        name: "Norm".to_string(),
        args: args.to_vec(),
      })
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
  if args.len() == 2 {
    // Normalize[v, f] — divides v by f[v]
    let norm_val =
      crate::functions::list_helpers_ast::apply_func_ast(&args[1], &args[0])?;
    // v / norm_val
    let result = crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(args[0].clone()),
      right: Box::new(norm_val),
    })?;
    return Ok(result);
  }
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Normalize expects 1 or 2 arguments".into(),
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
                left: Box::new(Expr::FunctionCall {
                  name: "Abs".to_string(),
                  args: vec![e.clone()],
                }),
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
            let norm_expr = make_sqrt(sum_of_squares);
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
                right: Box::new(make_sqrt(Expr::Integer(sum_sq))),
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
/// Unitize[x, dx] - returns 0 when |x| < dx, 1 otherwise
/// Unitize[list] - maps over lists
pub fn unitize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "Unitize expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: Unitize[x, dx]
  if args.len() == 2 {
    // Thread over lists in the first argument.
    if let Expr::List(items) = &args[0] {
      let results: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|x| unitize_ast(&[x.clone(), args[1].clone()]))
        .collect();
      return Ok(Expr::List(results?));
    }

    let x_val =
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(&args[0]);
    let tol_val =
      crate::functions::math_ast::numeric_utils::try_eval_to_f64(&args[1]);
    if let (Some(x), Some(tol)) = (x_val, tol_val) {
      if x.abs() < tol {
        return Ok(Expr::Integer(0));
      }
      return Ok(Expr::Integer(1));
    }

    // Non-numeric arguments remain unevaluated.
    return Ok(Expr::FunctionCall {
      name: "Unitize".to_string(),
      args: args.to_vec(),
    });
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
    Expr::BigFloat(digits, prec) => {
      // A literal zero BigFloat (e.g. `0.`20`, `0.00`2`) reports
      // MachinePrecision in Wolfram — there's no precision to speak of when
      // all significant digits are zero.
      if digits.parse::<f64>().is_ok_and(|f| f == 0.0) {
        Ok(Expr::Identifier("MachinePrecision".to_string()))
      } else {
        Ok(Expr::Real(*prec))
      }
    }
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
      // Precision of a list is the minimum precision of its elements.
      // If the minimum comes from a machine-real element, return the
      // symbol MachinePrecision (matches wolframscript).
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      for item in items {
        let p = precision_ast(&[item.clone()])?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => match min_prec {
            None => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            Some(v) if f < v => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            _ => {}
          },
          _ => {}
        }
      }
      match min_prec {
        Some(_) if min_is_machine => {
          Ok(Expr::Identifier("MachinePrecision".to_string()))
        }
        Some(p) => Ok(Expr::Real(p)),
        None => Ok(Expr::Identifier("Infinity".to_string())),
      }
    }
    // For symbolic expressions, check if any subexpression has finite precision.
    // If the minimum comes from a machine-real element, return the symbol
    // MachinePrecision (matches wolframscript).
    Expr::FunctionCall { args: fargs, .. } => {
      let mp: f64 = 15.954589770191003;
      let mut min_prec: Option<f64> = None;
      let mut min_is_machine = false;
      for arg in fargs {
        let p = precision_ast(&[arg.clone()])?;
        match p {
          Expr::Identifier(ref name) if name == "Infinity" => {}
          Expr::Identifier(ref name) if name == "MachinePrecision" => {
            match min_prec {
              None => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              Some(v) if mp < v => {
                min_prec = Some(mp);
                min_is_machine = true;
              }
              _ => {}
            }
          }
          Expr::Real(f) => match min_prec {
            None => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            Some(v) if f < v => {
              min_prec = Some(f);
              min_is_machine = false;
            }
            _ => {}
          },
          _ => {}
        }
      }
      match min_prec {
        Some(_) if min_is_machine => {
          Ok(Expr::Identifier("MachinePrecision".to_string()))
        }
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
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "PowerExpand expects 1 or 2 arguments".into(),
    ));
  }
  // Second argument (Assumptions) is accepted but not used
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
      expr if is_sqrt(expr).is_some() => {
        let sqrt_arg = is_sqrt(expr).unwrap();
        Some((
          sqrt_arg.clone(),
          Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(2)],
          },
        ))
      }
      _ => None,
    }
  };

  // Helper to recursively collect all Times factors (flattening nested Times
  // and converting Divide to Times with Power[..., -1])
  fn collect_times_factors(e: &Expr) -> Vec<Expr> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Times" && args.len() >= 2 =>
      {
        args.iter().flat_map(collect_times_factors).collect()
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Times,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.extend(collect_times_factors(right));
        factors
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left,
        right,
      } => {
        let mut factors = collect_times_factors(left);
        factors.push(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: right.clone(),
          right: Box::new(Expr::Integer(-1)),
        });
        factors
      }
      _ => vec![e.clone()],
    }
  }

  let extract_times = |e: &Expr| -> Option<Vec<Expr>> {
    let factors = collect_times_factors(e);
    if factors.len() >= 2 {
      Some(factors)
    } else {
      None
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
      // Also applies (a^r)^c -> a^(r*c) for each factor
      if let Some(factors) = extract_times(&base) {
        let expanded: Vec<Expr> = factors
          .iter()
          .map(|factor| {
            // If factor is itself a power, apply (a^r)^c -> a^(r*c)
            if let Some((inner_base, inner_exp)) = extract_power(factor) {
              let new_exp = match times_ast(&[inner_exp.clone(), exp.clone()]) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Times,
                  left: Box::new(inner_exp),
                  right: Box::new(exp.clone()),
                },
              };
              match power_two(&inner_base, &new_exp) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(inner_base),
                  right: Box::new(new_exp),
                },
              }
            } else {
              match power_two(factor, &exp) {
                Ok(r) => r,
                Err(_) => Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(factor.clone()),
                  right: Box::new(exp.clone()),
                },
              }
            }
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

      // E^(a*Log[b]) -> b^a  and  E^Log[b] -> b
      if matches!(&base, Expr::Constant(c) if c == "E") {
        // Check if exponent is Log[b]
        if let Expr::FunctionCall { name: ln, args: la } = &exp
          && ln == "Log"
          && la.len() == 1
        {
          return la[0].clone();
        }
        // Check if exponent is a product containing Log[b]
        let exp_factors = collect_times_factors(&exp);
        if exp_factors.len() >= 2 {
          // Find the Log factor
          let log_idx = exp_factors.iter().position(|f| {
            matches!(f, Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1)
          });
          if let Some(idx) = log_idx
            && let Expr::FunctionCall { args: la, .. } = &exp_factors[idx]
          {
            let log_arg = la[0].clone();
            // Remaining factors form the new exponent
            let mut remaining: Vec<Expr> = exp_factors.clone();
            remaining.remove(idx);
            let new_exp = if remaining.len() == 1 {
              remaining.into_iter().next().unwrap()
            } else {
              match times_ast(&remaining) {
                Ok(r) => r,
                Err(_) => {
                  return Expr::BinaryOp {
                    op: crate::syntax::BinaryOperator::Power,
                    left: Box::new(base),
                    right: Box::new(exp),
                  };
                }
              }
            };
            return match power_two(&log_arg, &new_exp) {
              Ok(r) => r,
              Err(_) => Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: Box::new(log_arg),
                right: Box::new(new_exp),
              },
            };
          }
        }
      }

      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Power,
        left: Box::new(base),
        right: Box::new(exp),
      }
    }
    // Log expansion rules (assuming positive reals):
    // Log[a*b*...] -> Log[a] + Log[b] + ...
    // Log[a^b] -> b*Log[a]
    Expr::FunctionCall { name, args } if name == "Log" && args.len() == 1 => {
      // First, recursively expand the argument
      let expanded_arg = power_expand_recursive(&args[0]);

      // Log of a quotient: Log[a/b] -> Log[a] - Log[b]
      // Convert a/b to Times[a, Power[b, -1]] and fall through to product rule
      let expanded_arg = match &expanded_arg {
        Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Divide,
          left,
          right,
        } => Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            *left.clone(),
            Expr::BinaryOp {
              op: crate::syntax::BinaryOperator::Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          ],
        },
        _ => expanded_arg,
      };

      // Log of a product: Log[a*b*...] -> Log[a] + Log[b] + ...
      if let Some(factors) = extract_times(&expanded_arg) {
        let log_terms: Vec<Expr> = factors
          .iter()
          .map(|f| {
            power_expand_recursive(&Expr::FunctionCall {
              name: "Log".to_string(),
              args: vec![f.clone()],
            })
          })
          .collect();
        return match plus_ast(&log_terms) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Plus".to_string(),
            args: log_terms,
          },
        };
      }

      // Log of a power: Log[a^b] -> b*Log[a]
      if let Some((base, exp)) = extract_power(&expanded_arg) {
        let log_expr = Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![base],
        };
        // Evaluate Log[base] to simplify cases like Log[E] -> 1
        let log_base = crate::evaluator::evaluate_expr_to_expr(&log_expr)
          .unwrap_or(log_expr);
        let log_base = power_expand_recursive(&log_base);
        return match times_ast(&[exp, log_base]) {
          Ok(r) => r,
          Err(_) => Expr::FunctionCall {
            name: "Log".to_string(),
            args: vec![expanded_arg],
          },
        };
      }

      // Evaluate Log to simplify cases like Log[E] -> 1, Log[1] -> 0
      let log_expr = Expr::FunctionCall {
        name: "Log".to_string(),
        args: vec![expanded_arg.clone()],
      };
      crate::evaluator::evaluate_expr_to_expr(&log_expr).unwrap_or(log_expr)
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
  // Deduplicate while preserving order
  let mut seen = std::collections::HashSet::new();
  vars.retain(|v| seen.insert(crate::syntax::expr_to_string(v)));
  // For List input, sort in canonical order (alphabetical);
  // for non-List input, preserve first-appearance order (matching Wolfram).
  if matches!(&args[0], Expr::List(_)) {
    vars.sort_by(|a, b| {
      crate::syntax::expr_to_string(a).cmp(&crate::syntax::expr_to_string(b))
    });
  }
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
      crate::emit_message(&format!(
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
      crate::emit_message(&format!(
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

/// Compute the error function erf(x) using BigFloat arithmetic.
/// Uses the Taylor series: erf(x) = (2/sqrt(π)) * Σ_{n=0}^{∞} (-1)^n * x^(2n+1) / (n! * (2n+1))
/// Compute erfc(x) for x > 0 using the continued fraction representation.
/// erfc(x) = exp(-x²) / (f * sqrt(π)) where f is computed via modified Lentz's method.
fn bigfloat_erfc_cf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Use extra guard bits for intermediate computation
  let work_bits = bits + 64;

  let half = BigFloat::from_i32(1, work_bits).div(
    &BigFloat::from_i32(2, work_bits),
    work_bits,
    rm,
  );

  // Modified Lentz's method for the continued fraction
  // erfc(x) = (exp(-x²)/sqrt(π)) * 1/(x + 1/(2x + 2/(x + 3/(2x + ...))))
  // Using: a_n = n * 0.5, b_n = x
  let mut f = x.clone();
  let mut c = x.clone();
  let mut d = BigFloat::from_i32(0, work_bits);

  let max_iterations = work_bits * 2 + 200;
  for n in 1..max_iterations {
    // a_n = n * 0.5
    let a_n = BigFloat::from_i32(n as i32, work_bits).mul(&half, work_bits, rm);

    // d = x + a_n * d
    d = x.add(&a_n.mul(&d, work_bits, rm), work_bits, rm);
    // Guard against zero
    if d.is_zero() {
      d = BigFloat::min_positive_normal(work_bits);
    }

    // c = x + a_n / c
    c = x.add(&a_n.div(&c, work_bits, rm), work_bits, rm);
    if c.is_zero() {
      c = BigFloat::min_positive_normal(work_bits);
    }

    // d = 1/d
    d = BigFloat::from_i32(1, work_bits).div(&d, work_bits, rm);
    let delta = c.mul(&d, work_bits, rm);
    f = f.mul(&delta, work_bits, rm);

    // Check convergence: |delta - 1| is negligible
    let one = BigFloat::from_i32(1, work_bits);
    let diff = delta.sub(&one, work_bits, rm).abs();
    if diff.is_zero() {
      break;
    }
    if let Some(diff_exp) = diff.exponent()
      && diff_exp < -(work_bits as i32)
    {
      break;
    }
  }

  // erfc(x) = exp(-x²) / (f * sqrt(π))
  let x2 = x.mul(x, work_bits, rm);
  let neg_x2 = x2.neg();
  let exp_neg_x2 = neg_x2.exp(work_bits, rm, cc);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let denom = f.mul(&sqrt_pi, work_bits, rm);
  exp_neg_x2.div(&denom, bits, rm)
}

fn bigfloat_erf(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erf is odd: erf(-x) = -erf(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  // For large |x|, use the continued fraction for erfc(x) and compute erf = 1 - erfc.
  // The Taylor series suffers from catastrophic cancellation for large arguments.
  let four = BigFloat::from_i32(4, bits);
  if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction
    let erfc_val = bigfloat_erfc_cf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    let result = one.sub(&erfc_val, bits, rm);
    return if is_negative { result.neg() } else { result };
  }

  // For small |x| (≤ 4), use the Taylor series with extra guard bits to handle cancellation.
  // With |x| ≤ 4, the peak term is ~exp(x²/2) ≈ exp(8) ≈ 2981, needing ~12 extra bits.
  // We use 64 guard bits for safety.
  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), alternating sign
  // sum = Σ (-1)^n * contribution_n
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    if n % 2 == 1 {
      sum = sum.sub(&contribution, work_bits, rm);
    } else {
      sum = sum.add(&contribution, work_bits, rm);
    }

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// Compute erfc(x) with arbitrary precision.
/// For large x, uses continued fraction directly. For small x, uses 1 - erf(x).
fn bigfloat_erfc(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(1, bits);
  }

  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let four = BigFloat::from_i32(4, bits);
  let result = if x_abs.cmp(&four) == Some(1) {
    // |x| > 4: use continued fraction directly for best precision
    bigfloat_erfc_cf(&x_abs, bits, rm, cc)
  } else {
    // |x| <= 4: compute via 1 - erf(x)
    let erf_val = bigfloat_erf(&x_abs, bits, rm, cc);
    let one = BigFloat::from_i32(1, bits);
    one.sub(&erf_val, bits, rm)
  };

  // erfc(-x) = 2 - erfc(x)
  if is_negative {
    let two = BigFloat::from_i32(2, bits);
    two.sub(&result, bits, rm)
  } else {
    result
  }
}

/// Compute erfi(x) with arbitrary precision.
/// erfi(x) = (2/sqrt(pi)) * sum_{n=0}^{inf} x^(2n+1) / (n! * (2n+1))
/// Unlike erf, the terms do NOT alternate in sign.
fn bigfloat_erfi(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  if x.is_zero() {
    return BigFloat::from_i32(0, bits);
  }

  // erfi is odd: erfi(-x) = -erfi(x)
  let is_negative = x.is_negative();
  let x_abs = x.abs();

  let work_bits = bits + 64;

  // Taylor series: term_0 = x, term_n = term_{n-1} * x^2 / n
  // contribution_n = term_n / (2n+1), all terms positive (no alternating sign)
  let x2 = x_abs.mul(&x_abs, work_bits, rm);
  let mut term = x_abs.clone();
  let mut sum = x_abs.clone();

  let max_iterations = work_bits * 2 + 100;
  for n in 1..max_iterations {
    term = term.mul(&x2, work_bits, rm);
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    term = term.div(&n_bf, work_bits, rm);

    let denom = BigFloat::from_i32((2 * n + 1) as i32, work_bits);
    let contribution = term.div(&denom, work_bits, rm);

    sum = sum.add(&contribution, work_bits, rm);

    if contribution.is_zero() {
      break;
    }
    if let (Some(c_exp), Some(s_exp)) =
      (contribution.exponent(), sum.exponent())
      && s_exp - c_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Multiply by 2/sqrt(π), round to final precision
  let two = BigFloat::from_i32(2, work_bits);
  let pi = cc.pi(work_bits, rm);
  let sqrt_pi = pi.sqrt(work_bits, rm);
  let factor = two.div(&sqrt_pi, work_bits, rm);
  let result = sum.mul(&factor, bits, rm);

  if is_negative { result.neg() } else { result }
}

/// Compute the exponential integral Ei(x) using BigFloat arithmetic.
/// For real x: Ei(x) = γ + ln|x| + Σ_{n=1}^{∞} x^n / (n * n!)
/// where γ is the Euler-Mascheroni constant.
fn bigfloat_exp_integral_ei(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  let work_bits = bits + 64;

  // γ (Euler-Mascheroni constant)
  let euler_gamma = compute_euler_gamma(work_bits, rm, cc);

  // ln(|x|)
  let ln_x = x.abs().ln(work_bits, rm, cc);

  // Power series: Σ_{n=1}^{∞} x^n / (n * n!)
  let mut sum = BigFloat::from_i32(0, work_bits);
  let mut x_pow = x.clone(); // x^1
  let mut factorial = BigFloat::from_i32(1, work_bits); // 1!

  let max_iterations = bits * 2 + 100;
  for n in 1..max_iterations {
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    if n > 1 {
      x_pow = x_pow.mul(x, work_bits, rm);
      factorial = factorial.mul(&n_bf, work_bits, rm);
    }
    // term = x^n / (n * n!)
    let denom = n_bf.mul(&factorial, work_bits, rm);
    let term = x_pow.div(&denom, work_bits, rm);
    sum = sum.add(&term, work_bits, rm);

    if term.abs().is_zero() {
      break;
    }
    if let (Some(t_exp), Some(s_exp)) =
      (term.abs().exponent(), sum.abs().exponent())
      && s_exp - t_exp > (work_bits as i32)
    {
      break;
    }
  }

  // Ei(x) = γ + ln(|x|) + sum (final result rounded to requested bits)
  euler_gamma.add(&ln_x, work_bits, rm).add(&sum, bits, rm)
}

/// Compute the complex exponential integral Ei(z) using BigFloat arithmetic.
/// For complex z = a + bi: Ei(z) = γ + Log(z) + Σ_{n=1}^{∞} z^n / (n * n!)
/// where γ is the Euler-Mascheroni constant and Log is the complex logarithm.
fn complex_exp_integral_ei(
  a: astro_float::BigFloat,
  b: astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Result<(astro_float::BigFloat, astro_float::BigFloat), InterpreterError> {
  use astro_float::BigFloat;

  // Use many extra guard bits to ensure the rounding to `bits` is correct
  let work_bits = bits + 256;

  // γ (Euler-Mascheroni constant) - purely real
  let euler_gamma = compute_euler_gamma(work_bits, rm, cc);

  // Complex Log(z) = ln|z| + i*arg(z)
  // |z| = sqrt(a^2 + b^2), arg(z) = atan2(b, a)
  let abs_sq =
    a.mul(&a, work_bits, rm)
      .add(&b.mul(&b, work_bits, rm), work_bits, rm);
  let abs_z = abs_sq.sqrt(work_bits, rm);
  let ln_abs_z = abs_z.ln(work_bits, rm, cc);
  // atan2(b, a) via atan and quadrant adjustment
  let arg_z = {
    let zero = BigFloat::from_i32(0, work_bits);
    let pi = cc.pi(work_bits, rm);
    if !a.is_zero() {
      let ratio = b.div(&a, work_bits, rm);
      let atan_val = ratio.atan(work_bits, rm, cc);
      if a.is_positive() {
        atan_val
      } else if b.is_negative() {
        atan_val.sub(&pi, work_bits, rm)
      } else {
        atan_val.add(&pi, work_bits, rm)
      }
    } else if b.is_positive() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
    } else if b.is_negative() {
      pi.div(&BigFloat::from_i32(2, work_bits), work_bits, rm)
        .neg()
    } else {
      zero
    }
  };

  // Start sum: γ + ln|z| for real part, arg(z) for imaginary part
  let mut sum_re = euler_gamma.add(&ln_abs_z, work_bits, rm);
  let mut sum_im = arg_z;

  // Power series: Σ_{n=1}^{∞} z^n / (n * n!)
  // Track z^n as (pow_re, pow_im), start with z^1 = (a, b)
  let mut pow_re = a.clone();
  let mut pow_im = b.clone();
  let mut factorial = BigFloat::from_i32(1, work_bits); // n!

  let max_iterations = bits * 2 + 100;
  for n in 1..max_iterations {
    let n_bf = BigFloat::from_i32(n as i32, work_bits);
    if n > 1 {
      // z^n = z^(n-1) * z
      let new_re = pow_re.mul(&a, work_bits, rm).sub(
        &pow_im.mul(&b, work_bits, rm),
        work_bits,
        rm,
      );
      let new_im = pow_re.mul(&b, work_bits, rm).add(
        &pow_im.mul(&a, work_bits, rm),
        work_bits,
        rm,
      );
      pow_re = new_re;
      pow_im = new_im;
      factorial = factorial.mul(&n_bf, work_bits, rm);
    }
    // term = z^n / (n * n!)
    let denom = n_bf.mul(&factorial, work_bits, rm);
    let term_re = pow_re.div(&denom, work_bits, rm);
    let term_im = pow_im.div(&denom, work_bits, rm);

    sum_re = sum_re.add(&term_re, work_bits, rm);
    sum_im = sum_im.add(&term_im, work_bits, rm);

    // Check convergence
    let term_abs_sq = term_re.mul(&term_re, work_bits, rm).add(
      &term_im.mul(&term_im, work_bits, rm),
      work_bits,
      rm,
    );
    if term_abs_sq.is_zero() {
      break;
    }
    let sum_abs_sq = sum_re.mul(&sum_re, work_bits, rm).add(
      &sum_im.mul(&sum_im, work_bits, rm),
      work_bits,
      rm,
    );
    if let (Some(t_exp), Some(s_exp)) =
      (term_abs_sq.exponent(), sum_abs_sq.exponent())
    {
      // Compare squared magnitudes, so convergence is 2x the bit threshold
      if s_exp - t_exp > (2 * work_bits as i32) {
        break;
      }
    }
  }

  // Perform final addition at target `bits` precision to truncate mantissa
  let zero = BigFloat::from_i32(0, bits);
  let result_re = sum_re.add(&zero, bits, rm);
  let result_im = sum_im.add(&zero, bits, rm);

  Ok((result_re, result_im))
}

/// Compute the Euler-Mascheroni constant γ to the given precision.
fn compute_euler_gamma(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // High-precision string for Euler-Mascheroni constant (105 digits)
  let gamma_str = "0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495";
  BigFloat::parse(gamma_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_catalan(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // High-precision string for Catalan's constant G (OEIS A006752, ~105 digits)
  let catalan_str = "0.91596559417721901505460351493238411077414937428167213426649811962176301977625476947935651292611510624857";
  BigFloat::parse(catalan_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_glaisher(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Glaisher–Kinkelin constant A (OEIS A074962, ~105 digits).
  let glaisher_str = "1.28242712910062263687534256886979172776768892732500119206374002174040630883966455201507550549353290381";
  BigFloat::parse(glaisher_str, astro_float::Radix::Dec, bits, rm, cc)
}

fn compute_khinchin(
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> astro_float::BigFloat {
  use astro_float::BigFloat;

  // Khinchin's constant K_0 (OEIS A002210, ~105 digits).
  let khinchin_str = "2.68545200106530644530971483548179569382038229399446295305115234555721885953715200280114117493184769799";
  BigFloat::parse(khinchin_str, astro_float::Radix::Dec, bits, rm, cc)
}

/// ListFourierSequenceTransform[{a0, a1, ..., an}, omega] — discrete-time Fourier transform.
///
/// Computes Sum[a_k * E^(-I * omega * k), {k, 0, n-1}].
pub fn list_fourier_sequence_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "ListFourierSequenceTransform".to_string(),
      args: args.to_vec(),
    });
  }

  let list = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "ListFourierSequenceTransform".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if list.is_empty() {
    return Ok(Expr::List(vec![]));
  }

  let omega = &args[1];

  // Build the sum: Sum[a_k * E^(-I * omega * k), {k, 0, n-1}]
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::syntax::BinaryOperator;

  let mut terms: Vec<Expr> = Vec::new();
  for (k, coeff) in list.iter().enumerate() {
    if matches!(coeff, Expr::Integer(0)) {
      continue;
    }

    if k == 0 {
      // E^0 = 1, so just add the coefficient
      terms.push(coeff.clone());
    } else {
      // a_k * E^(-I * omega * k)
      let k_expr = Expr::Integer(k as i128);
      // -I * omega * k
      let exponent = Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![
          Expr::Integer(-1),
          Expr::Identifier("I".to_string()),
          omega.clone(),
          k_expr,
        ],
      };
      let exp_term = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier("E".to_string())),
        right: Box::new(exponent),
      };
      let term = Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(coeff.clone()),
        right: Box::new(exp_term),
      };
      terms.push(term);
    }
  }

  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }

  let sum = if terms.len() == 1 {
    terms.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  };

  evaluate_expr_to_expr(&sum)
}

/// Generic window function evaluator.
/// All window functions are defined on [-1/2, 1/2] and return 0 outside.
pub fn window_function_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec(),
    });
  }

  // For exact rational arguments, try exact evaluation
  if let Expr::FunctionCall {
    name: fname,
    args: fargs,
  } = &args[0]
    && fname == "Rational"
    && fargs.len() == 2
    && let (Some(n), Some(d)) =
      (try_eval_to_f64(&fargs[0]), try_eval_to_f64(&fargs[1]))
  {
    let x = n / d;
    if x.abs() > 0.5 {
      return Ok(Expr::Integer(0));
    }
  }

  let x = match try_eval_to_f64(&args[0]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
  };

  if x.abs() > 0.5 {
    return Ok(if matches!(&args[0], Expr::Real(_)) {
      Expr::Real(0.0)
    } else {
      Expr::Integer(0)
    });
  }

  let pi = std::f64::consts::PI;
  let val = match name {
    "HammingWindow" => 25.0 / 46.0 + 21.0 / 46.0 * (2.0 * pi * x).cos(),
    "HannWindow" => (1.0 + (2.0 * pi * x).cos()) / 2.0,
    "BlackmanWindow" => {
      0.42 + 0.5 * (2.0 * pi * x).cos() + 0.08 * (4.0 * pi * x).cos()
    }
    "DirichletWindow" => 1.0,
    "BartlettWindow" => 1.0 - 2.0 * x.abs(),
    "WelchWindow" => 1.0 - 4.0 * x * x,
    "CosineWindow" => (pi * x).cos(),
    "ConnesWindow" => {
      let t = 1.0 - 4.0 * x * x;
      t * t
    }
    "LanczosWindow" => {
      let arg = 2.0 * x;
      if arg.abs() < 1e-15 {
        1.0
      } else {
        (pi * arg).sin() / (pi * arg)
      }
    }
    "ExactBlackmanWindow" => {
      3946.0 / 18608.0
        + 9274.0 / 18608.0 * (2.0 * pi * x).cos()
        + 5388.0 / 18608.0 * (4.0 * pi * x).cos()
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: name.to_string(),
        args: args.to_vec(),
      });
    }
  };

  // For exact arguments, try to return exact results
  if matches!(&args[0], Expr::Integer(_))
    || matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Rational")
  {
    // Check for common exact values
    let rounded = fourier_round(val);
    if (rounded - rounded.round()).abs() < 1e-14 && rounded.abs() < 1e18 {
      return Ok(Expr::Integer(rounded.round() as i128));
    }
    // Try to express as a simple fraction
    if let Some((n, d)) = approximate_rational(rounded) {
      return Ok(make_rational(n, d));
    }
  }

  Ok(Expr::Real(val))
}

/// Try to express a float as a simple rational p/q with small denominator.
fn approximate_rational(val: f64) -> Option<(i128, i128)> {
  if val == 0.0 {
    return Some((0, 1));
  }
  // Try denominators up to 10000
  for d in 1..=10000i128 {
    let n = (val * d as f64).round() as i128;
    let approx = n as f64 / d as f64;
    if (approx - val).abs() < 1e-14 {
      let g = gcd(n.abs(), d);
      return Some((n / g, d / g));
    }
  }
  None
}

/// BandpassFilter[data, {omega1, omega2}] or BandpassFilter[data, {omega1, omega2}, n]
/// Applies a bandpass FIR filter using a windowed-sinc kernel with exact Hamming window.
/// Default order n = length of data. Default SampleRate = 1.
pub fn bandpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "BandpassFilter".to_string(),
      args: args.to_vec(),
    });
  }

  // Extract data list
  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Extract {omega1, omega2}
  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let o1 = match try_eval_to_f64(&freqs[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandpassFilter".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let o2 = match try_eval_to_f64(&freqs[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandpassFilter".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (o1, o2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse optional order (3rd argument, if numeric) and options (Rule arguments)
  let mut order = items.len(); // default order = signal length
  let mut sample_rate = 1.0_f64;

  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        order = *n as usize;
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  // Scale frequencies by sample rate
  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  // Compute FIR kernel using windowed-sinc with exact Hamming window (alpha = 25/46)
  let n = order;
  let kernel = bandpass_kernel(n, w1, w2);

  // Try numeric path first
  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() == items.len() {
    // All numeric — fast path
    let result = convolve_edge_padded(&data, &kernel);
    return Ok(Expr::List(result.into_iter().map(Expr::Real).collect()));
  }

  // Symbolic path: convolve symbolically
  let result = convolve_edge_padded_symbolic(items, &kernel);
  Ok(Expr::List(result))
}

/// Compute the bandpass FIR kernel of length n using windowed-sinc with exact Hamming window.
fn bandpass_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;

    // Exact Hamming window: alpha = 25/46
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };

    // Bandpass sinc kernel
    let sinc_bp = if t.abs() < 1e-15 {
      (omega2 - omega1) / std::f64::consts::PI
    } else {
      ((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };

    kernel.push(w * sinc_bp);
  }
  kernel
}

/// Convolve data with kernel using edge-padding (repeat boundary values).
/// Returns output of same length as data.
fn convolve_edge_padded(data: &[f64], kernel: &[f64]) -> Vec<f64> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    let mut sum = 0.0;
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let val = if idx < 0 {
        data[0]
      } else if idx >= len as isize {
        data[len - 1]
      } else {
        data[idx as usize]
      };
      sum += kernel[j] * val;
    }
    result.push(fourier_round(sum));
  }
  result
}

/// Symbolic convolution: for each output position, compute a symbolic sum of kernel[j] * data[idx].
/// Uses edge-padding (repeat boundary values) like the numeric version.
fn convolve_edge_padded_symbolic(data: &[Expr], kernel: &[f64]) -> Vec<Expr> {
  let n = kernel.len();
  let len = data.len();
  let left_pad = n / 2;

  let mut result = Vec::with_capacity(len);
  for m in 0..len {
    // Group kernel coefficients by data index
    let mut coeff_by_idx: Vec<f64> = vec![0.0; len];
    for j in 0..n {
      let idx = m as isize + j as isize - left_pad as isize;
      let data_idx = if idx < 0 {
        0
      } else if idx >= len as isize {
        len - 1
      } else {
        idx as usize
      };
      coeff_by_idx[data_idx] += kernel[j];
    }
    // Build symbolic sum
    let mut terms: Vec<Expr> = Vec::new();
    for (i, &coeff) in coeff_by_idx.iter().enumerate() {
      if coeff == 0.0 {
        continue;
      }
      terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Real(coeff), data[i].clone()],
      });
    }
    let expr = if terms.is_empty() {
      Expr::Integer(0)
    } else if terms.len() == 1 {
      terms.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "Plus".to_string(),
        args: terms,
      }
    };
    // Evaluate the symbolic expression to simplify
    match crate::evaluator::evaluate_expr_to_expr(&expr) {
      Ok(e) => result.push(e),
      Err(_) => result.push(expr),
    }
  }
  result
}

/// LowpassFilter[data, omega_c] or LowpassFilter[data, omega_c, n]
/// Applies a lowpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn lowpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "LowpassFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "LowpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let omega_c = match try_eval_to_f64(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "LowpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut order = items.len();
  let mut sample_rate = 1.0_f64;

  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        order = *n as usize;
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "LowpassFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let kernel = lowpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the lowpass FIR kernel of length n, normalized to sum to 1.
fn lowpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_lp = if t.abs() < 1e-15 {
      omega_c / std::f64::consts::PI
    } else {
      (omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_lp);
  }
  // Normalize to sum to 1 (unity DC gain)
  let sum: f64 = kernel.iter().sum();
  if sum.abs() > 1e-15 {
    for v in &mut kernel {
      *v /= sum;
    }
  }
  kernel
}

/// HighpassFilter[data, omega_c] or HighpassFilter[data, omega_c, n]
/// Applies a highpass FIR filter using a windowed-sinc kernel with exact Hamming window.
pub fn highpass_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "HighpassFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "HighpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let omega_c = match try_eval_to_f64(&args[1]) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "HighpassFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut order = items.len();
  let mut sample_rate = 1.0_f64;

  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        order = *n as usize;
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let wc = omega_c / sample_rate;

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "HighpassFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let kernel = highpass_kernel(order, wc);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the highpass FIR kernel of length n.
/// Highpass = delta - lowpass (spectral inversion).
fn highpass_kernel(n: usize, omega_c: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    // Ideal highpass: delta(t) - sinc_lp(t) for the windowed version
    let sinc_hp = if t.abs() < 1e-15 {
      1.0 - omega_c / std::f64::consts::PI
    } else {
      -(omega_c * t).sin() / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_hp);
  }
  kernel
}

/// BandstopFilter[data, {omega1, omega2}] or BandstopFilter[data, {omega1, omega2}, n]
/// Applies a bandstop (notch) FIR filter.
pub fn bandstop_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 4 {
    return Ok(Expr::FunctionCall {
      name: "BandstopFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let items = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandstopFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let (omega1, omega2) = match &args[1] {
    Expr::List(freqs) if freqs.len() == 2 => {
      let o1 = match try_eval_to_f64(&freqs[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandstopFilter".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let o2 = match try_eval_to_f64(&freqs[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "BandstopFilter".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (o1, o2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "BandstopFilter".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut order = items.len();
  let mut sample_rate = 1.0_f64;

  for i in 2..args.len() {
    match &args[i] {
      Expr::Integer(n) if *n > 0 => {
        order = *n as usize;
      }
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(name) = pattern.as_ref()
          && name == "SampleRate"
          && let Some(v) = try_eval_to_f64(replacement)
        {
          sample_rate = v;
        }
      }
      _ => {}
    }
  }

  let w1 = omega1 / sample_rate;
  let w2 = omega2 / sample_rate;

  let data: Vec<f64> = items.iter().filter_map(try_eval_to_f64).collect();
  if data.len() != items.len() {
    return Ok(Expr::FunctionCall {
      name: "BandstopFilter".to_string(),
      args: args.to_vec(),
    });
  }

  let kernel = bandstop_kernel(order, w1, w2);
  let result = convolve_edge_padded(&data, &kernel);

  Ok(Expr::List(result.into_iter().map(Expr::Real).collect()))
}

/// Compute the bandstop FIR kernel of length n.
/// Bandstop = delta - bandpass (spectral inversion).
fn bandstop_kernel(n: usize, omega1: f64, omega2: f64) -> Vec<f64> {
  let mut kernel = Vec::with_capacity(n);
  for j in 0..n {
    let t = j as f64 - (n as f64 - 1.0) / 2.0;
    let w = if n <= 1 {
      1.0
    } else {
      25.0 / 46.0
        - 21.0 / 46.0
          * (2.0 * std::f64::consts::PI * j as f64 / (n as f64 - 1.0)).cos()
    };
    let sinc_bs = if t.abs() < 1e-15 {
      1.0 - (omega2 - omega1) / std::f64::consts::PI
    } else {
      -((omega2 * t).sin() - (omega1 * t).sin()) / (std::f64::consts::PI * t)
    };
    kernel.push(w * sinc_bs);
  }
  kernel
}
