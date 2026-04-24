#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// Check if an expression is known to be non-negative without assumptions.
/// Used for simplifications like Sqrt[x^2] → x (only valid when x >= 0).
fn is_known_non_negative(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(n) => *n >= 0,
    Expr::Real(f) => *f >= 0.0,
    // Known positive constants
    Expr::Constant(name) => matches!(
      name.as_str(),
      "Pi"
        | "E"
        | "EulerGamma"
        | "GoldenRatio"
        | "Degree"
        | "Catalan"
        | "Glaisher"
        | "Khinchin"
    ),
    // Abs[anything] is always non-negative
    Expr::FunctionCall { name, args } if name == "Abs" && args.len() == 1 => {
      true
    }
    // Sqrt[anything] is always non-negative (for real results)
    expr if is_sqrt(expr).is_some() => true,
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
  // Handle integers and reals directly
  match &args[0] {
    Expr::Integer(n) => return Ok(Expr::Integer(n.abs())),
    Expr::Real(f) => return Ok(Expr::Real(f.abs())),
    _ => {}
  }
  // Handle exact complex numbers and rationals: Abs[a + b*I] = Sqrt[a^2 + b^2]
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
      return Ok(make_sqrt(make_rational(num2, den2)));
    }
  }
  // Handle floating-point complex numbers: Abs[3.0 + I] = sqrt(10)
  if let Some((re, im)) = try_extract_complex_f64(&args[0])
    && im != 0.0
  {
    return Ok(num_to_expr((re * re + im * im).sqrt()));
  }
  // Fallback: try to evaluate to f64 for numeric expressions (e.g. Abs[Sqrt[2] + 1])
  if let Some(n) = try_eval_to_f64(&args[0]) {
    return Ok(num_to_expr(n.abs()));
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
  // Handle Infinity, -Infinity, ComplexInfinity, Indeterminate
  if matches!(&args[0], Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Integer(1));
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  if matches!(&args[0], Expr::Identifier(s) if s == "Indeterminate") {
    return Ok(Expr::Identifier("Indeterminate".to_string()));
  }
  // Check for -Infinity (UnaryOp::Minus applied to Infinity)
  if let Expr::UnaryOp {
    op: crate::syntax::UnaryOperator::Minus,
    operand,
  } = &args[0]
    && matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity")
  {
    return Ok(Expr::Integer(-1));
  }
  // Check for negative infinity: Times[n, Infinity] where n < 0
  if let Expr::FunctionCall { name, args: fargs } = &args[0]
    && name == "Times"
  {
    let has_infinity = fargs
      .iter()
      .any(|a| matches!(a, Expr::Identifier(s) if s == "Infinity"));
    if has_infinity {
      let coeff = fargs.iter().find_map(|a| {
        if let Expr::Integer(n) = a {
          Some(*n as f64)
        } else {
          try_eval_to_f64(a)
        }
      });
      if let Some(c) = coeff {
        if c < 0.0 {
          return Ok(Expr::Integer(-1));
        } else if c > 0.0 {
          return Ok(Expr::Integer(1));
        }
      }
    }
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
        let abs_expr = make_sqrt(make_rational(abs2_n, abs2_d));
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
  // Sqrt[Infinity] = Infinity
  if matches!(&args[0], Expr::Identifier(s) if s == "Infinity") {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  // Sqrt[ComplexInfinity] = ComplexInfinity
  if matches!(&args[0], Expr::Identifier(s) if s == "ComplexInfinity") {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
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
            make_sqrt(Expr::Integer(inside as i128)),
          ],
        });
      }
      // Not a perfect square, return symbolic
      Ok(make_sqrt(args[0].clone()))
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
          let sqrt_part = make_sqrt(Expr::Integer(n_in as i128));
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        } else if n_in == 1 {
          // Result: n_out / (d_out * Sqrt[d_in])
          let denom = if d_out == 1 {
            make_sqrt(Expr::Integer(d_in as i128))
          } else {
            Expr::FunctionCall {
              name: "Times".to_string(),
              args: vec![
                Expr::Integer(d_out as i128),
                make_sqrt(Expr::Integer(d_in as i128)),
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
          let sqrt_part = make_sqrt(make_rational(n_in as i128, d_in as i128));
          if n_out == 1 && d_out == 1 {
            return Ok(sqrt_part);
          }
          let coeff = make_rational(n_out as i128, d_out as i128);
          return times_ast(&[coeff, sqrt_part]);
        }
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt[-n] for negative integer → I * Sqrt[n]
    Expr::Integer(n) if *n < 0 => {
      let pos = -*n;
      let sqrt_pos = sqrt_ast(&[Expr::Integer(pos)])?;
      times_ast(&[Expr::Identifier("I".to_string()), sqrt_pos])
    }
    Expr::Real(f) if *f >= 0.0 => Ok(Expr::Real(f.sqrt())),
    // Sqrt[base^(2n)] → base^n only when base is known non-negative
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left: base,
      right: exp,
    } if matches!(exp.as_ref(), Expr::Integer(n) if *n > 0 && n % 2 == 0)
      && is_known_non_negative(base) =>
    {
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
      Ok(make_sqrt(args[0].clone()))
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
          // expr^2 → move expr outside only if expr is known non-negative
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(2))
            && is_known_non_negative(base) =>
          {
            outside.push(*base.clone());
          }
          // expr^(2n) → move expr^n outside (for any even n, including negative)
          // only if expr is known non-negative
          Expr::BinaryOp {
            op: crate::syntax::BinaryOperator::Power,
            left: base,
            right: exp,
          } if matches!(exp.as_ref(), Expr::Integer(n) if n % 2 == 0 && *n != 0)
            && is_known_non_negative(base) =>
          {
            if let Expr::Integer(n) = exp.as_ref() {
              let half = n / 2;
              if half == 1 {
                outside.push(*base.clone());
              } else {
                outside.push(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: base.clone(),
                  right: Box::new(Expr::Integer(half)),
                });
              }
            }
          }
          // Power[base, even_n] (FunctionCall form) → move base^(n/2) outside
          // only if base is known non-negative
          Expr::FunctionCall {
            name: pname,
            args: pargs,
          } if pname == "Power"
            && pargs.len() == 2
            && matches!(&pargs[1], Expr::Integer(n) if n % 2 == 0 && *n != 0)
            && is_known_non_negative(&pargs[0]) =>
          {
            if let Expr::Integer(n) = &pargs[1] {
              let half = n / 2;
              if half == 1 {
                outside.push(pargs[0].clone());
              } else {
                outside.push(Expr::BinaryOp {
                  op: crate::syntax::BinaryOperator::Power,
                  left: Box::new(pargs[0].clone()),
                  right: Box::new(Expr::Integer(half)),
                });
              }
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
        return Ok(make_sqrt(args[0].clone()));
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
        let sqrt_part = make_sqrt(inside_expr);
        times_ast(&[outside_expr, sqrt_part])
      }
    }
    // Sqrt[Plus[...]] — extract perfect square GCD from integer coefficients.
    // E.g. Sqrt[4 + 36*t^2 + 36*t^4] → 2*Sqrt[1 + 9*t^2 + 9*t^4]
    _ if matches!(&args[0], Expr::FunctionCall { name, .. } if name == "Plus") =>
    {
      if let Some(result) = try_sqrt_plus_gcd(&args[0]) {
        return Ok(result);
      }
      if let Some(result) = try_sqrt_gaussian(&args[0]) {
        return Ok(result);
      }
      Ok(make_sqrt(args[0].clone()))
    }
    // Sqrt[a + b*I] for Gaussian integers — try to find exact Gaussian integer sqrt
    _ => {
      if let Some(result) = try_sqrt_gaussian(&args[0]) {
        return Ok(result);
      }
      Ok(make_sqrt(args[0].clone()))
    }
  }
}

/// Extract the integer coefficient from a Plus term.
/// Returns (coefficient, base) where term = coefficient * base.
/// For Integer(n), returns (n, Integer(1)).
/// For Times[n, rest], returns (n, rest).
fn extract_int_coeff(term: &Expr) -> Option<(i128, Expr)> {
  match term {
    Expr::Integer(n) => Some((*n, Expr::Integer(1))),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() >= 2
        && matches!(&args[0], Expr::Integer(_)) =>
    {
      if let Expr::Integer(n) = &args[0] {
        let base = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec(),
          }
        };
        Some((*n, base))
      } else {
        None
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      let (c, base) = extract_int_coeff(operand)?;
      Some((-c, base))
    }
    _ => None,
  }
}

/// Try to simplify Sqrt[Plus[...]] by extracting the GCD of integer coefficients.
/// If the GCD is a perfect square, factor it out.
/// E.g. Sqrt[4 + 36*t^2 + 36*t^4] → 2*Sqrt[1 + 9*t^2 + 9*t^4]
fn try_sqrt_plus_gcd(expr: &Expr) -> Option<Expr> {
  use crate::functions::math_ast::numeric_utils::gcd;

  let terms = match expr {
    Expr::FunctionCall { name, args } if name == "Plus" => args,
    _ => return None,
  };

  if terms.is_empty() {
    return None;
  }

  // Extract integer coefficients from each Plus term
  let mut pairs: Vec<(i128, Expr)> = Vec::new();
  for term in terms {
    match extract_int_coeff(term) {
      Some(pair) => pairs.push(pair),
      None => return None,
    }
  }

  // Compute GCD of absolute values of all coefficients
  let mut g = pairs[0].0.abs();
  for &(c, _) in &pairs[1..] {
    g = gcd(g, c.abs()).unsigned_abs() as i128;
    if g <= 1 {
      return None;
    }
  }

  if g <= 1 {
    return None;
  }

  // Find the largest perfect square factor of the GCD
  let mut sqrt_factor = 1i128;
  let mut remaining_g = g;
  let mut f = 2i128;
  while f * f <= remaining_g {
    while remaining_g % (f * f) == 0 {
      sqrt_factor *= f;
      remaining_g /= f * f;
    }
    f += 1;
  }

  if sqrt_factor <= 1 {
    return None;
  }

  // Factor to divide out from under the sqrt: sqrt_factor^2
  let factor_out = sqrt_factor * sqrt_factor;

  // Build new terms with coefficients divided by factor_out
  let mut new_terms: Vec<Expr> = Vec::new();
  for (coeff, base) in &pairs {
    let new_coeff = coeff / factor_out;
    if matches!(base, Expr::Integer(1)) {
      // Bare integer term
      new_terms.push(Expr::Integer(new_coeff));
    } else if new_coeff == 1 {
      new_terms.push(base.clone());
    } else if new_coeff == -1 {
      new_terms.push(super::trigonometric::negate_expr(base.clone()));
    } else {
      new_terms.push(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(new_coeff), base.clone()],
      });
    }
  }

  let new_sum = if new_terms.len() == 1 {
    new_terms.remove(0)
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: new_terms,
    }
  };

  let sqrt_part = make_sqrt(new_sum);

  if sqrt_factor == 1 {
    Some(sqrt_part)
  } else {
    times_ast(&[Expr::Integer(sqrt_factor), sqrt_part]).ok()
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

/// Compute Floor or Ceiling via arbitrary-precision BigFloat when the value
/// exceeds what i128 can represent. `approx` is the f64 approximation used
/// only to pick a target precision. Returns an `Expr::BigInteger` on
/// success, or `None` if BigFloat evaluation of the input fails.
fn floor_via_bigfloat(
  expr: &Expr,
  approx: f64,
  is_floor: bool,
) -> Option<Expr> {
  use astro_float::{Consts, RoundingMode};
  if !approx.is_finite() {
    return None;
  }
  // Target precision: at least as many decimal digits as the integer part,
  // plus a safety margin.
  let mag_digits = approx.abs().log10().ceil() as i64;
  let precision = (mag_digits.max(20) as usize) + 10;
  let bits = crate::functions::math_ast::numerical::nominal_bits(precision);
  let mut cc = Consts::new().ok()?;
  let rm = RoundingMode::ToEven;
  let bf = crate::functions::math_ast::numerical::expr_to_bigfloat(
    expr, bits, rm, &mut cc,
  )
  .ok()?;
  let decimal = crate::functions::math_ast::numerical::bigfloat_to_string(
    &bf, None, rm, &mut cc,
  )
  .ok()?;
  // decimal looks like "[-]DIGITS.FRAC" (or just "[-]DIGITS.").
  let (sign, rest) = if let Some(s) = decimal.strip_prefix('-') {
    ("-", s)
  } else {
    ("", decimal.as_str())
  };
  let dot_pos = rest.find('.')?;
  let int_part = &rest[..dot_pos];
  let frac_part = &rest[dot_pos + 1..];
  // Parse the integer part as a BigInt.
  let int_str = format!("{}{}", sign, int_part);
  let int_bi = int_str.parse::<num_bigint::BigInt>().ok()?;
  // For Floor on a negative number with a non-zero fractional part, subtract 1.
  // For Ceiling on a positive number with a non-zero fractional part, add 1.
  let has_frac = !frac_part.chars().all(|c| c == '0');
  use num_bigint::BigInt;
  let adjusted = if is_floor {
    if sign == "-" && has_frac {
      int_bi - BigInt::from(1)
    } else {
      int_bi
    }
  } else if sign != "-" && has_frac {
    int_bi + BigInt::from(1)
  } else {
    int_bi
  };
  // Downshift to Integer if it fits i128.
  if let Ok(small) = adjusted.to_string().parse::<i128>() {
    Some(Expr::Integer(small))
  } else {
    Some(Expr::BigInteger(adjusted))
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
    // i128 fits ~38 decimal digits; beyond that, `.floor() as i128` saturates
    // at i128::MAX, mis-computing Floor for large symbolic products like
    // `Pi * 10^100`. Fall through to the arbitrary-precision path.
    // f64 has ~15-16 digits of integer precision, so any magnitude beyond
    // that needs BigFloat to preserve the integer part exactly.
    if n.is_finite() && n.abs() < 1e15 {
      return Ok(Expr::Integer(n.floor() as i128));
    }
    // BigFloat fallback: compute the expression to enough precision to
    // resolve the integer part exactly, then parse the leading digits as
    // a BigInt.
    if let Some(result) = floor_via_bigfloat(&args[0], n, true) {
      return Ok(result);
    }
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
    // f64 has ~15-16 digits of integer precision, so any magnitude beyond
    // that needs BigFloat to preserve the integer part exactly.
    if n.is_finite() && n.abs() < 1e15 {
      return Ok(Expr::Integer(n.ceil() as i128));
    }
    if let Some(result) = floor_via_bigfloat(&args[0], n, false) {
      return Ok(result);
    }
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

/// Banker's rounding: round half to even
fn bankers_round(n: f64) -> f64 {
  if n.fract().abs() == 0.5 {
    let floor = n.floor();
    if floor as i128 % 2 == 0 {
      floor
    } else {
      n.ceil()
    }
  } else {
    n.round()
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
    let eval_a = crate::evaluator::evaluate_expr_to_expr(&args[1])?;
    let eval_x = crate::evaluator::evaluate_expr_to_expr(&args[0])?;

    // Complex step: Round[x, a] = a * Round[x/a]
    if is_complex_number(&eval_a) || is_complex_number(&eval_x) {
      let quotient = crate::evaluator::evaluate_function_call_ast(
        "Divide",
        &[eval_x.clone(), eval_a.clone()],
      )?;
      let rounded = round_ast(&[quotient])?;
      return crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[eval_a, rounded],
      );
    }

    // Check if a is a Rational (n/d)
    if let Expr::FunctionCall { name, args: rargs } = &eval_a
      && name == "Rational"
      && rargs.len() == 2
      && let (Some(x_val), Some(a_val)) =
        (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_a))
      && a_val != 0.0
    {
      let n = bankers_round(x_val / a_val) as i128;
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
      let n = bankers_round(x_val / a_val) as i128;
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
      // When the step a is Real, result should be Real;
      // when a is Integer, result should be Integer (if whole)
      if a_is_int && rounded.fract() == 0.0 && rounded.abs() < i128::MAX as f64
      {
        return Ok(Expr::Integer(rounded as i128));
      }
      if a_is_real || matches!(&eval_x, Expr::Real(_)) || rounded.fract() != 0.0
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
  // Complex[re, im]: round real and imaginary parts separately
  if let Expr::FunctionCall { name, args: cargs } = &args[0]
    && name == "Complex"
    && cargs.len() == 2
  {
    let re = round_ast(&[cargs[0].clone()])?;
    let im = round_ast(&[cargs[1].clone()])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[
        re,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![im, Expr::Identifier("I".to_string())],
        },
      ],
    );
  }
  // Exact complex in Plus/Times form: extract and round parts separately
  if let Some(((rn, rd), (in_, id))) =
    crate::functions::math_ast::try_extract_complex_exact(&args[0])
    && in_ != 0
  {
    let re_rat = make_rational_pub(rn, rd);
    let im_rat = make_rational_pub(in_, id);
    let re_rounded = round_ast(&[re_rat])?;
    let im_rounded = round_ast(&[im_rat])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Plus",
      &[
        re_rounded,
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![im_rounded, Expr::Identifier("I".to_string())],
        },
      ],
    );
  }
  if let Some(n) = try_eval_to_f64(&args[0]) {
    let rounded = bankers_round(n);
    Ok(Expr::Integer(rounded as i128))
  } else {
    Ok(Expr::FunctionCall {
      name: "Round".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Returns true if `expr` is a complex number with a non-zero imaginary part.
/// Handles Complex[re, im], Plus[re, Times[im, I]], I, etc.
fn is_complex_number(expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Complex"
    && args.len() == 2
  {
    return !matches!(&args[1], Expr::Integer(0));
  }
  if let Some(((_, _), (in_, _))) =
    crate::functions::math_ast::try_extract_complex_exact(expr)
  {
    return in_ != 0;
  }
  false
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
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
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
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0] encountered.",
        crate::syntax::expr_to_string(m)
      ));
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
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      ));
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
      crate::emit_message(&format!(
        "Mod::indet: Indeterminate expression Mod[{}, 0, {}] encountered.",
        crate::syntax::expr_to_string(m),
        crate::syntax::expr_to_string(d)
      ));
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
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(format!(
      "Quotient expects 2 or 3 arguments; {} given",
      args.len()
    )));
  }

  // 3-argument form: Quotient[n, m, d] = Floor[(n - d) / m]
  if args.len() == 3 {
    if let (Some((nn, nd)), Some((mn, md)), Some((dn, dd))) = (
      try_as_rational(&args[0]),
      try_as_rational(&args[1]),
      try_as_rational(&args[2]),
    ) {
      if mn == 0 {
        return Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ));
      }
      // (n - d) = (nn*dd - dn*nd) / (nd*dd)
      let diff_n = nn * dd - dn * nd;
      let diff_d = nd * dd;
      // (n - d) / m = (diff_n * md) / (diff_d * mn)
      let q_n = diff_n * md;
      let q_d = diff_d * mn;
      return Ok(Expr::Integer(floor_div(q_n, q_d)));
    }
    if let (Some(a), Some(b), Some(c)) = (
      try_eval_to_f64(&args[0]),
      try_eval_to_f64(&args[1]),
      try_eval_to_f64(&args[2]),
    ) {
      if b == 0.0 {
        return Err(InterpreterError::EvaluationError(
          "Quotient: division by zero".into(),
        ));
      }
      return Ok(Expr::Integer(((a - c) / b).floor() as i128));
    }
    return Ok(Expr::FunctionCall {
      name: "Quotient".to_string(),
      args: args.to_vec(),
    });
  }

  // 2-argument form: Quotient[n, m] = Floor[n / m]
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

  // Handle list first argument: thread Clip over each element
  if let Expr::List(items) = &args[0] {
    let results: Result<Vec<Expr>, _> = items
      .iter()
      .map(|item| {
        let mut new_args = args.to_vec();
        new_args[0] = item.clone();
        clip_ast(&new_args)
      })
      .collect();
    return Ok(Expr::List(results?));
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
  use num_bigint::BigInt;
  use num_traits::Zero;

  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "IntegerExponent expects 1 or 2 arguments".into(),
    ));
  }
  // Accept both Integer and BigInteger for the first argument so that
  // IntegerExponent[100!, 10] works even when the factorial overflows i128.
  let n: BigInt = match &args[0] {
    Expr::Integer(k) => BigInt::from(*k),
    Expr::BigInteger(k) => k.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "IntegerExponent".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let base: BigInt = if args.len() == 2 {
    match &args[1] {
      Expr::Integer(b) => BigInt::from(*b),
      Expr::BigInteger(b) => b.clone(),
      _ => {
        return Ok(Expr::FunctionCall {
          name: "IntegerExponent".to_string(),
          args: args.to_vec(),
        });
      }
    }
  } else {
    BigInt::from(10) // default base is 10 in Wolfram
  };

  if n.is_zero() {
    return Ok(Expr::Identifier("Infinity".to_string()));
  }
  if base <= BigInt::from(1) {
    return Ok(Expr::FunctionCall {
      name: "IntegerExponent".to_string(),
      args: args.to_vec(),
    });
  }

  let mut count: i128 = 0;
  let mut val = n.clone();
  if val.sign() == num_bigint::Sign::Minus {
    val = -val;
  }
  while !val.is_zero() && (&val % &base).is_zero() {
    count += 1;
    val /= &base;
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
    // Preserve Real type: FractionalPart[3.0] = 0. (not 0)
    Expr::Real(f) => Ok(Expr::Real(*f - f.trunc())),
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
      // For symbolic constants (Pi, E, GoldenRatio, etc.), return x - Floor[x]
      // symbolically to avoid losing precision by converting to float.
      let is_known_constant = matches!(&args[0], Expr::Constant(_))
        || matches!(&args[0], Expr::Identifier(s) if matches!(s.as_str(),
          "GoldenRatio" | "EulerGamma" | "Catalan" | "Khinchin" | "Glaisher"));
      if is_known_constant {
        let floor_val =
          crate::functions::math_ast::floor_ast(&[args[0].clone()])?;
        if let Expr::Integer(n) = &floor_val {
          if *n == 0 {
            return Ok(args[0].clone());
          }
          return crate::evaluator::evaluate_function_call_ast(
            "Plus",
            &[args[0].clone(), Expr::Integer(-*n)],
          );
        }
      }
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
    // Recursively chop into function calls (Plus, Times, Complex, …) and
    // re-evaluate so that chopped subterms like `Complex[0, 0]` collapse
    // to 0 and additive identities are simplified.
    Expr::FunctionCall { name, args } => {
      let chopped_args: Vec<Expr> = args
        .iter()
        .map(|a| chop_expr(a, tolerance))
        .collect::<Result<_, _>>()?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: name.clone(),
        args: chopped_args,
      })
    }
    Expr::BinaryOp { op, left, right } => {
      let new_left = chop_expr(left, tolerance)?;
      let new_right = chop_expr(right, tolerance)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::BinaryOp {
        op: *op,
        left: Box::new(new_left),
        right: Box::new(new_right),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let new_operand = chop_expr(operand, tolerance)?;
      crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
        op: *op,
        operand: Box::new(new_operand),
      })
    }
    _ => Ok(expr.clone()),
  }
}

// ─── CubeRoot ──────────────────────────────────────────────────────

/// CubeRoot[x] - Real-valued cube root
/// Extract the largest perfect cube factor from n.
/// Returns (cube_root_of_cube_part, remainder) such that n = cube_part^3 * remainder.
fn extract_cube_factor(mut n: u128) -> (u128, u128) {
  let mut cube_root = 1u128;
  // Trial division by small primes
  let mut p = 2u128;
  while p * p * p <= n {
    let mut count = 0u32;
    while n.is_multiple_of(p) {
      n /= p;
      count += 1;
    }
    let cube_groups = count / 3;
    let leftover = count % 3;
    for _ in 0..cube_groups {
      cube_root *= p;
    }
    for _ in 0..leftover {
      n *= p; // put non-cube parts back into remainder
    }
    p += if p == 2 { 1 } else { 2 };
  }
  (cube_root, n)
}

pub fn cube_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CubeRoot expects exactly 1 argument".into(),
    ));
  }
  match &args[0] {
    Expr::Integer(n) => {
      if *n == 0 {
        return Ok(Expr::Integer(0));
      }
      let sign = n.signum();
      let abs_n = n.unsigned_abs();
      // Check for perfect cube
      let root = (abs_n as f64).cbrt().round() as u128;
      if root * root * root == abs_n {
        return Ok(Expr::Integer(sign * root as i128));
      }
      // Factor out the largest perfect cube
      // Find prime factorization and extract cube parts
      let (cube_part, remainder) = extract_cube_factor(abs_n);
      if cube_part > 1 {
        // CubeRoot[n] = cube_part * CubeRoot[remainder]
        let cube_root_remainder = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Integer(remainder as i128)),
          right: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(3)],
          }),
        };
        let result = Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Times,
          left: Box::new(Expr::Integer(sign * cube_part as i128)),
          right: Box::new(cube_root_remainder),
        };
        crate::evaluator::evaluate_expr_to_expr(&result)
      } else {
        // No cube factor — return n^(1/3)
        Ok(Expr::BinaryOp {
          op: crate::syntax::BinaryOperator::Power,
          left: Box::new(Expr::Integer(sign * abs_n as i128)),
          right: Box::new(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(1), Expr::Integer(3)],
          }),
        })
      }
    }
    Expr::Real(f) => Ok(Expr::Real(f.signum() * f.abs().cbrt())),
    _ => {
      if let Some(f) = try_eval_to_f64(&args[0]) {
        Ok(Expr::Real(f.signum() * f.abs().cbrt()))
      } else {
        // Canonicalize CubeRoot[x] → Surd[x, 3]
        Ok(Expr::FunctionCall {
          name: "Surd".to_string(),
          args: vec![args[0].clone(), Expr::Integer(3)],
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

  // Extract (xmin, xmax, n) based on arity
  let (xmin, xmax, n_expr) = match args.len() {
    1 => (Expr::Integer(0), Expr::Integer(1), args[0].clone()),
    2 => (Expr::Integer(0), args[0].clone(), args[1].clone()),
    3 => (args[0].clone(), args[1].clone(), args[2].clone()),
    _ => unreachable!(),
  };

  // n must be a positive integer
  let n_val = match &n_expr {
    Expr::Integer(n) if *n > 0 => *n,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Subdivide".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Fast path: both endpoints are integers
  if let (Expr::Integer(xmin_i), Expr::Integer(xmax_i)) = (&xmin, &xmax) {
    let mut items = Vec::with_capacity(n_val as usize + 1);
    let range = xmax_i - xmin_i;
    for i in 0..=n_val {
      let numer = xmin_i * n_val + i * range;
      items.push(make_rational(numer, n_val));
    }
    return Ok(Expr::List(items));
  }

  // General path: build xmin + i*(xmax - xmin)/n symbolically and evaluate.
  // For vector endpoints, thread element-wise.
  let is_vector =
    matches!(&xmin, Expr::List(_)) || matches!(&xmax, Expr::List(_));
  if is_vector {
    // Both must be lists of the same length
    if let (Expr::List(xmin_items), Expr::List(xmax_items)) = (&xmin, &xmax) {
      if xmin_items.len() != xmax_items.len() {
        return Ok(Expr::FunctionCall {
          name: "Subdivide".to_string(),
          args: args.to_vec(),
        });
      }
      let dim = xmin_items.len();
      let mut result_items = Vec::with_capacity(n_val as usize + 1);
      for i in 0..=n_val {
        let mut point = Vec::with_capacity(dim);
        for d in 0..dim {
          let val =
            subdivide_scalar_at(&xmin_items[d], &xmax_items[d], i, n_val)?;
          point.push(val);
        }
        result_items.push(Expr::List(point));
      }
      return Ok(Expr::List(result_items));
    }
    return Ok(Expr::FunctionCall {
      name: "Subdivide".to_string(),
      args: args.to_vec(),
    });
  }

  // Scalar, general (symbolic or float) endpoints
  let mut items = Vec::with_capacity(n_val as usize + 1);
  for i in 0..=n_val {
    items.push(subdivide_scalar_at(&xmin, &xmax, i, n_val)?);
  }
  Ok(Expr::List(items))
}

/// Compute xmin*(n-i)/n + xmax*i/n for a single scalar pair of endpoints.
fn subdivide_scalar_at(
  xmin: &Expr,
  xmax: &Expr,
  i: i128,
  n: i128,
) -> Result<Expr, InterpreterError> {
  use crate::evaluator::evaluate_expr_to_expr;
  use crate::syntax::BinaryOperator;

  if i == 0 {
    return Ok(xmin.clone());
  }
  if i == n {
    return Ok(xmax.clone());
  }

  // Build: xmin*(n-i)/n + xmax*i/n  and evaluate
  let coeff_min = make_rational(n - i, n);
  let coeff_max = make_rational(i, n);

  let term_min = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(coeff_min),
    right: Box::new(xmin.clone()),
  };
  let term_max = Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(coeff_max),
    right: Box::new(xmax.clone()),
  };
  let result = Expr::BinaryOp {
    op: BinaryOperator::Plus,
    left: Box::new(term_min),
    right: Box::new(term_max),
  };
  evaluate_expr_to_expr(&result)
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

/// DiscreteDelta[n1, n2, ...] - returns 1 if all ni are 0, 0 otherwise
/// DiscreteDelta[] - returns 1
pub fn discrete_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(Expr::Integer(1));
  }
  let mut has_symbolic = false;
  for arg in args {
    match arg {
      Expr::Integer(n) => {
        if *n != 0 {
          return Ok(Expr::Integer(0));
        }
      }
      Expr::Real(f) => {
        if *f != 0.0 {
          return Ok(Expr::Integer(0));
        }
      }
      _ => {
        has_symbolic = true;
      }
    }
  }
  if has_symbolic {
    Ok(Expr::FunctionCall {
      name: "DiscreteDelta".to_string(),
      args: args.to_vec(),
    })
  } else {
    Ok(Expr::Integer(1))
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

pub fn heaviside_theta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "HeavisideTheta expects at least 1 argument".into(),
    ));
  }

  // Multi-arg: HeavisideTheta[x1, x2, ...] = product, 0 if any xi < 0
  if args.len() > 1 {
    let mut has_zero = false;
    let mut remaining = Vec::new();
    for arg in args {
      match arg {
        Expr::Integer(n) => {
          if *n < 0 {
            return Ok(Expr::Integer(0));
          }
          if *n == 0 {
            has_zero = true;
            remaining.push(arg.clone());
          }
          // n > 0: contributes 1, skip
        }
        Expr::Real(f) => {
          if *f < 0.0 {
            return Ok(Expr::Integer(0));
          }
          if *f == 0.0 {
            has_zero = true;
            remaining.push(arg.clone());
          }
        }
        _ => {
          remaining.push(arg.clone());
        }
      }
    }
    // If any arg is zero, HeavisideTheta[0] is undefined so the whole
    // expression stays unevaluated with ALL original arguments sorted.
    if has_zero {
      let mut sorted_args = args.to_vec();
      sorted_args.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
      return Ok(Expr::FunctionCall {
        name: "HeavisideTheta".to_string(),
        args: sorted_args,
      });
    }
    if remaining.is_empty() {
      return Ok(Expr::Integer(1));
    }
    // Sort remaining args for canonical form
    remaining.sort_by(crate::functions::list_helpers_ast::canonical_cmp);
    return Ok(Expr::FunctionCall {
      name: "HeavisideTheta".to_string(),
      args: remaining,
    });
  }

  // Single arg
  match &args[0] {
    Expr::Integer(n) => {
      if *n > 0 {
        Ok(Expr::Integer(1))
      } else if *n < 0 {
        Ok(Expr::Integer(0))
      } else {
        // HeavisideTheta[0] stays symbolic
        Ok(Expr::FunctionCall {
          name: "HeavisideTheta".to_string(),
          args: vec![Expr::Integer(0)],
        })
      }
    }
    Expr::Real(f) => {
      if *f > 0.0 {
        Ok(Expr::Integer(1))
      } else if *f < 0.0 {
        Ok(Expr::Integer(0))
      } else {
        Ok(Expr::FunctionCall {
          name: "HeavisideTheta".to_string(),
          args: vec![Expr::Real(0.0)],
        })
      }
    }
    Expr::Constant(c) => match c.as_str() {
      "Pi" | "E" | "Degree" => Ok(Expr::Integer(1)),
      _ => Ok(Expr::FunctionCall {
        name: "HeavisideTheta".to_string(),
        args: args.to_vec(),
      }),
    },
    Expr::Identifier(name) if name == "Infinity" => Ok(Expr::Integer(1)),
    _ => Ok(Expr::FunctionCall {
      name: "HeavisideTheta".to_string(),
      args: args.to_vec(),
    }),
  }
}

pub fn dirac_delta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "DiracDelta expects at least 1 argument".into(),
    ));
  }

  // For numeric arguments: 0 for any non-zero value, symbolic at 0
  for arg in args {
    match arg {
      Expr::Integer(n) if *n != 0 => return Ok(Expr::Integer(0)),
      Expr::Real(f) if *f != 0.0 => return Ok(Expr::Integer(0)),
      Expr::Constant(_) => return Ok(Expr::Integer(0)), // Pi, E, etc. are non-zero
      Expr::Identifier(name) if name == "Infinity" => {
        return Ok(Expr::Integer(0));
      }
      _ => {}
    }
  }

  // If all args are zero, or mixed with symbolic, stay symbolic
  Ok(Expr::FunctionCall {
    name: "DiracDelta".to_string(),
    args: args.to_vec(),
  })
}

/// UnitBox[x] = 1 for |x| <= 1/2, 0 otherwise
pub fn unit_box_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      // |n| <= 1/2 only if n == 0
      Ok(Expr::Integer(if v == 0 { 1 } else { 0 }))
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(Expr::Integer(if v <= 0.5 { 1 } else { 0 }))
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_val = (*n as f64 / *d as f64).abs();
        return Ok(Expr::Integer(if abs_val <= 0.5 { 1 } else { 0 }));
      }
      Ok(Expr::FunctionCall {
        name: "UnitBox".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "UnitBox".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// HeavisidePi[x] = 1 for |x| < 1/2, 0 for |x| > 1/2, unevaluated at |x| = 1/2
pub fn heaviside_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      // |n| < 1/2 only if n == 0
      Ok(Expr::Integer(if *n == 0 { 1 } else { 0 }))
    }
    Expr::Real(f) => {
      let v = f.abs();
      if (v - 0.5).abs() < f64::EPSILON {
        // At boundary, return unevaluated
        Ok(Expr::FunctionCall {
          name: "HeavisidePi".to_string(),
          args: args.to_vec(),
        })
      } else if v < 0.5 {
        Ok(Expr::Integer(1))
      } else {
        Ok(Expr::Integer(0))
      }
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        // Compare |n/d| with 1/2: |2*n| vs |d|
        let two_n_abs = (2 * *n).unsigned_abs();
        let d_abs = d.unsigned_abs();
        return if two_n_abs == d_abs {
          // At boundary, return unevaluated
          Ok(Expr::FunctionCall {
            name: "HeavisidePi".to_string(),
            args: args.to_vec(),
          })
        } else if two_n_abs < d_abs {
          Ok(Expr::Integer(1))
        } else {
          Ok(Expr::Integer(0))
        };
      }
      Ok(Expr::FunctionCall {
        name: "HeavisidePi".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HeavisidePi".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// UnitTriangle[x] = 1 - |x| for |x| <= 1, 0 otherwise
pub fn unit_triangle_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      Ok(if v <= 1 {
        Expr::Integer(1 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(if v <= 1.0 {
        Expr::Real(1.0 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_val = (*n as f64 / *d as f64).abs();
        if abs_val <= 1.0 {
          // 1 - |n/d| = (|d| - |n|) / |d|
          let abs_n = n.abs();
          let abs_d = d.abs();
          let num = abs_d - abs_n;
          if num == 0 {
            return Ok(Expr::Integer(0));
          }
          return Ok(Expr::FunctionCall {
            name: "Rational".to_string(),
            args: vec![Expr::Integer(num), Expr::Integer(abs_d)],
          });
        } else {
          return Ok(Expr::Integer(0));
        }
      }
      Ok(Expr::FunctionCall {
        name: "UnitTriangle".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "UnitTriangle".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// HeavisideLambda[x] = 1 - |x| for |x| < 1, 0 for |x| >= 1
pub fn heaviside_lambda_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let x = &args[0];
  match x {
    Expr::Integer(n) => {
      let v = (*n).abs();
      Ok(if v < 1 {
        Expr::Integer(1 - v)
      } else {
        Expr::Integer(0)
      })
    }
    Expr::Real(f) => {
      let v = f.abs();
      Ok(if v < 1.0 {
        Expr::Real(1.0 - v)
      } else {
        Expr::Real(0.0)
      })
    }
    Expr::FunctionCall { name, args: fargs }
      if name == "Rational" && fargs.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&fargs[0], &fargs[1])
        && *d != 0
      {
        let abs_n = n.abs();
        let abs_d = d.abs();
        // Compare |n/d| with 1: |n| vs |d|
        if abs_n >= abs_d {
          return Ok(Expr::Integer(0));
        }
        // 1 - |n/d| = (|d| - |n|) / |d|
        let num = abs_d - abs_n;
        return Ok(crate::functions::math_ast::make_rational_pub(num, abs_d));
      }
      Ok(Expr::FunctionCall {
        name: "HeavisideLambda".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "HeavisideLambda".to_string(),
      args: args.to_vec(),
    }),
  }
}
