#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Handle Sqrt[Quantity[mag, unit]] by delegating to Power[quantity, 1/2]
  if crate::functions::quantity_ast::is_quantity(&args[0]).is_some() {
    let half = make_rational(1, 2);
    if let Some(result) =
      crate::functions::quantity_ast::try_quantity_power(&args[0], &half)
    {
      return result;
    }
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
    // Sqrt[expr^2] → expr, Sqrt[expr^(2n)] → expr^n
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: base,
      right: exp,
    } if matches!(exp.as_ref(), Expr::Integer(n) if *n > 0 && n % 2 == 0) => {
      if let Expr::Integer(n) = exp.as_ref() {
        let half = n / 2;
        if half == 1 {
          return Ok(*base.clone());
        } else {
          return Ok(Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base.clone(),
            right: Box::new(Expr::Integer(half)),
          });
        }
      }
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    }
    // Sqrt of a product: Sqrt[n * expr^2 * ...] → simplify factors
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      ..
    }
    | Expr::FunctionCall { name: _, args: _ }
      if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Times") =>
    {
      let factors =
        crate::functions::polynomial_ast::collect_multiplicative_factors(
          &args[0],
        );
      // Separate into: integer part, squared symbolic factors, remainder
      let mut int_product: i128 = 1;
      let mut outside: Vec<Expr> = Vec::new(); // factors to move outside sqrt
      let mut inside: Vec<Expr> = Vec::new(); // factors to keep inside sqrt

      for f in &factors {
        match f {
          Expr::Integer(n) => {
            int_product *= n;
          }
          // expr^2 → move expr outside
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(2)) => {
            outside.push(*base.clone());
          }
          // expr^(2n) → move expr^n outside
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(n) if *n > 0 && n % 2 == 0) => {
            if let Expr::Integer(n) = exp.as_ref() {
              outside.push(Expr::BinaryOp {
                op: crate::syntax::BinaryOperator::Power,
                left: base.clone(),
                right: Box::new(Expr::Integer(n / 2)),
              });
            }
          }
          _ => {
            inside.push(f.clone());
          }
        }
      }

      // Handle the integer part: extract perfect square factors
      let sign = if int_product < 0 { -1i128 } else { 1i128 };
      let abs_int = (int_product * sign) as u64;
      let mut int_outside = 1u64;
      let mut int_inside = abs_int;
      let mut factor = 2u64;
      while factor * factor <= int_inside {
        while int_inside.is_multiple_of(factor * factor) {
          int_outside *= factor;
          int_inside /= factor * factor;
        }
        factor += 1;
      }

      if int_outside <= 1 && outside.is_empty() {
        // No simplification possible
        if let Some(result) = try_sqrt_gaussian(&args[0]) {
          return Ok(result);
        }
        return Ok(Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: args.to_vec(),
        });
      }

      // Build outside factors
      if int_outside > 1 {
        outside.insert(0, Expr::Integer(int_outside as i128));
      }

      // Build inside factors
      if int_inside > 1 {
        inside.insert(0, Expr::Integer(int_inside as i128));
      }
      if sign < 0 {
        inside.insert(0, Expr::Integer(-1));
      }

      let outside_expr = if outside.len() == 1 {
        outside.remove(0)
      } else {
        crate::functions::polynomial_ast::build_product(outside)
      };

      if inside.is_empty() {
        Ok(outside_expr)
      } else {
        let inside_expr = if inside.len() == 1 {
          inside.remove(0)
        } else {
          crate::functions::polynomial_ast::build_product(inside)
        };
        let sqrt_part = Expr::FunctionCall {
          name: "Sqrt".to_string(),
          args: vec![inside_expr],
        };
        times_ast(&[outside_expr, sqrt_part])
      }
    }
    // Sqrt[a + b*I] for Gaussian integers — try to find exact Gaussian integer sqrt
    _ => {
      if let Some(result) = try_sqrt_gaussian(&args[0]) {
        return Ok(result);
      }
      Ok(Expr::FunctionCall {
        name: "Sqrt".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Try to compute an exact Gaussian integer square root of a complex number.
///
/// For `z = a + b*I` (Gaussian integer), finds `p + q*I` such that `(p+qI)^2 = z`,
/// choosing the principal root with `p >= 0` (or `p == 0, q > 0`).
/// Returns `None` if no such integer solution exists.
fn try_sqrt_gaussian(expr: &Expr) -> Option<Expr> {
  use crate::functions::math_ast::numeric_utils::try_extract_complex_exact;
  let ((rn, rd), (in_, id)) = try_extract_complex_exact(expr)?;
  // Normalize to integers a and b where z = a + b*I
  // Require both parts to be integers (denominator 1 after reducing)
  let g_r = crate::functions::math_ast::numeric_utils::gcd(rn, rd).abs();
  let (rn, rd) = (rn / g_r, rd / g_r);
  let g_i = crate::functions::math_ast::numeric_utils::gcd(in_, id).abs();
  let (in_, id) = (in_ / g_i, id / g_i);
  // Pure real cases are handled by existing code; skip to avoid duplication
  if in_ == 0 {
    return None;
  }
  if rd != 1 || id != 1 {
    return None; // Only handle Gaussian integers, not Gaussian rationals
  }
  let a = rn; // real part
  let b = in_; // imaginary part
  // |z|^2 = a^2 + b^2; needs to be a perfect square n^2
  let m = a.checked_mul(a)?.checked_add(b.checked_mul(b)?)?;
  let n = (m as f64).sqrt() as i128;
  if n * n != m {
    return None; // |z| is not an integer
  }
  // p^2 = (n + a) / 2; needs n + a to be even and non-negative
  let n_plus_a = n.checked_add(a)?;
  if n_plus_a < 0 || n_plus_a % 2 != 0 {
    return None;
  }
  let p_sq = n_plus_a / 2;
  let p = (p_sq as f64).sqrt() as i128;
  if p * p != p_sq {
    return None; // p is not an integer
  }
  // Determine q: q = b / (2*p)
  if p == 0 {
    // Handle p == 0: then a == -n <= 0 and b != 0 → q^2 = n, pick q > 0
    let q_sq = n;
    let q = (q_sq as f64).sqrt() as i128;
    if q * q != q_sq {
      return None;
    }
    // Choose q with same sign as b (principal root convention)
    let q = if b < 0 { -q } else { q };
    return Some(build_complex_expr(0, 1, q, 1));
  }
  let two_p = p.checked_mul(2)?;
  if b % two_p != 0 {
    return None;
  }
  let q = b / two_p;
  // Verify: (p + q*I)^2 = p^2 - q^2 + 2pq*I should equal a + b*I
  let check_re = p.checked_mul(p)?.checked_sub(q.checked_mul(q)?)?;
  let check_im = p.checked_mul(q)?.checked_mul(2)?;
  if check_re != a || check_im != b {
    return None;
  }
  Some(build_complex_expr(p, 1, q, 1))
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
  } else if let Some((re, im)) = try_extract_complex_float(&args[0]) {
    if im == 0.0 {
      Ok(Expr::Integer(re.floor() as i128))
    } else {
      // Floor[a + b*I] = Floor[a] + Floor[b]*I
      let floor_re = re.floor() as i128;
      let floor_im = im.floor() as i128;
      crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[
          Expr::Integer(floor_re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(floor_im),
              Expr::Identifier("I".to_string()),
            ],
          },
        ],
      )
    }
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
  } else if let Some((re, im)) = try_extract_complex_float(&args[0]) {
    if im == 0.0 {
      Ok(Expr::Integer(re.ceil() as i128))
    } else {
      // Ceiling[a + b*I] = Ceiling[a] + Ceiling[b]*I
      let ceil_re = re.ceil() as i128;
      let ceil_im = im.ceil() as i128;
      crate::evaluator::evaluate_function_call_ast(
        "Plus",
        &[
          Expr::Integer(ceil_re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![
              Expr::Integer(ceil_im),
              Expr::Identifier("I".to_string()),
            ],
          },
        ],
      )
    }
  } else {
    Ok(Expr::FunctionCall {
      name: "Ceiling".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Helper for Floor[x, a] and Ceiling[x, a]
/// Floor[x, a] = a * Floor[x/a], Ceiling[x, a] = a * Ceiling[x/a]
pub fn floor_ceil_two_arg(
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
pub fn try_as_rational(expr: &Expr) -> Option<(i128, i128)> {
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
pub fn mod2_ast(m: &Expr, n: &Expr) -> Result<Expr, InterpreterError> {
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
pub fn mod3_ast(
  m: &Expr,
  n: &Expr,
  d: &Expr,
) -> Result<Expr, InterpreterError> {
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
pub fn floor_div(a: i128, b: i128) -> i128 {
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

pub fn chop_expr(
  expr: &Expr,
  tolerance: f64,
) -> Result<Expr, InterpreterError> {
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
