#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// ExpIntegralEi[x] - Exponential integral Ei(x)
pub fn exp_integral_ei_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExpIntegralEi expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // ExpIntegralEi[0] = -Infinity
    Expr::Integer(0) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // ExpIntegralEi[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(exp_integral_ei_numeric(*x))),
    // Check for -Infinity or other cases
    other => {
      if is_neg_infinity(other) {
        return Ok(Expr::Integer(0));
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "ExpIntegralEi".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Ei(x) numerically
/// For all x: Ei(x) = γ + ln|x| + Σ_{n=1}^∞ x^n / (n * n!)
pub fn exp_integral_ei_numeric(x: f64) -> f64 {
  if x.abs() < 40.0 {
    // Power series: Ei(x) = γ + ln|x| + Σ x^n / (n * n!)
    let euler_gamma = 0.5772156649015329;
    let mut sum = euler_gamma + x.abs().ln();
    let mut term = 1.0;
    for n in 1..200 {
      let nf = n as f64;
      term *= x / nf;
      sum += term / nf;
      if (term / nf).abs() < 1e-16 * sum.abs() {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |x|: Ei(x) ~ e^x/x * Σ n!/x^n
    // Use continued fraction for better convergence
    let mut result = 0.0;
    for n in (1..=100).rev() {
      let nf = n as f64;
      result = nf / (1.0 + nf / (x + result));
    }
    result = x.exp() / (x + result);
    if x > 0.0 {
      result
    } else {
      // For large negative x, Ei(x) ≈ e^x/x * (1 + 1!/x + 2!/x^2 + ...)
      let mut sum = 1.0;
      let mut term = 1.0;
      for n in 1..100 {
        term *= n as f64 / x;
        sum += term;
        if term.abs() < 1e-16 * sum.abs() {
          break;
        }
      }
      sum * x.exp() / x
    }
  }
}

/// CosIntegral[z] - Cosine integral Ci(z)
/// Ci(z) = γ + ln(z) + ∫₀ᶻ (cos(t)-1)/t dt
pub fn cos_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CosIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // CosIntegral[0] = -Infinity
    Expr::Integer(0) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // CosIntegral[Infinity] = 0
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(0)),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(cos_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // CosIntegral[-Infinity] = I*Pi
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Identifier("I".to_string()),
            Expr::Constant("Pi".to_string()),
          ],
        });
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "CosIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Ci(z) numerically for real z
/// Ci(z) = γ + ln|z| + ∫₀ᶻ (cos(t)-1)/t dt
/// = γ + ln|z| + Σ_{n=1}^∞ (-1)^n z^(2n) / (2n · (2n)!)
fn cos_integral_numeric(z: f64) -> f64 {
  let euler_gamma = 0.5772156649015329;

  if z.abs() < 40.0 {
    // Power series: Ci(z) = γ + ln|z| + Σ (-1)^n z^(2n) / (2n · (2n)!)
    let mut sum = euler_gamma + z.abs().ln();
    let z2 = z * z;
    let mut term = 1.0; // Will build up z^(2n) / (2n)!
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      // term *= z^2 / ((2n-1) * 2n)
      term *= z2 / ((n2 - 1.0) * n2);
      let contrib = if n % 2 == 1 { -term } else { term };
      sum += contrib / n2;
      if (contrib / n2).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |z|
    // Ci(z) ≈ sin(z)/z * f(z) - cos(z)/z * g(z)
    // where f(z) = Σ (-1)^n (2n)! / z^(2n)
    //       g(z) = Σ (-1)^n (2n+1)! / z^(2n+1)
    let mut f = 0.0;
    let mut g = 0.0;
    let mut f_term = 1.0;
    let mut g_term = 1.0 / z;
    for n in 0..100 {
      f += if n % 2 == 0 { f_term } else { -f_term };
      g += if n % 2 == 0 { g_term } else { -g_term };
      let n2 = (2 * n + 1) as f64;
      let n2p = (2 * n + 2) as f64;
      f_term *= n2 * n2p / (z * z);
      g_term *= n2p * (n2p + 1.0) / (z * z);
      if f_term.abs() < 1e-16 && g_term.abs() < 1e-16 {
        break;
      }
      // Divergent series: stop when terms start growing
      if n > 0
        && (f_term.abs() > (n2 * n2p / (z * z) * f_term).abs()
          || g_term.abs() > 1e10)
      {
        break;
      }
    }
    f / z * z.sin() - g * z.cos()
  }
}

/// FresnelS[z] - Fresnel sine integral S(z) = ∫₀ᶻ sin(π t²/2) dt
pub fn fresnel_s_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FresnelS expects exactly 1 argument".into(),
    ));
  }

  // Helper: negate FresnelS (odd function)
  let negate_fresnel_s = |inner: Expr| -> Expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::FunctionCall {
        name: "FresnelS".to_string(),
        args: vec![inner],
      }),
    }
  };

  match &args[0] {
    // FresnelS[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // FresnelS[Infinity] = 1/2
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(crate::functions::math_ast::make_rational(1, 2))
    }
    // FresnelS[ComplexInfinity] = Indeterminate
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      Ok(Expr::Identifier("Indeterminate".to_string()))
    }
    // FresnelS[I] = -I*FresnelS[1]
    Expr::Identifier(s) if s == "I" => {
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::Integer(-1),
          Expr::Identifier("I".to_string()),
          Expr::FunctionCall {
            name: "FresnelS".to_string(),
            args: vec![Expr::Integer(1)],
          },
        ],
      )
    }
    // FresnelS[-x] = -FresnelS[x] (UnaryOp form)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // Check for -Infinity first
      if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") {
        return Ok(crate::functions::math_ast::make_rational(-1, 2));
      }
      Ok(negate_fresnel_s(*operand.clone()))
    }
    // FresnelS[Times[-1, x]] = -FresnelS[x]
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return Ok(negate_fresnel_s(fargs[1].clone()));
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return Ok(negate_fresnel_s(fargs[0].clone()));
      }
      // Negative integer coefficient: Times[-n, x] -> -FresnelS[Times[n, x]]
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()],
        };
        return Ok(negate_fresnel_s(pos_arg));
      }
      // FresnelS[I*z] = -I*FresnelS[z]
      if matches!(&fargs[0], Expr::Identifier(s) if s == "I") {
        return crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[
            Expr::Integer(-1),
            Expr::Identifier("I".to_string()),
            Expr::FunctionCall {
              name: "FresnelS".to_string(),
              args: vec![fargs[1].clone()],
            },
          ],
        );
      }
      Ok(Expr::FunctionCall {
        name: "FresnelS".to_string(),
        args: args.to_vec(),
      })
    }
    // FresnelS[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_fresnel_s(Expr::Integer(-*n))),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(fresnel_s_numeric(*x))),
    // Check for -Infinity (Times[-1, Infinity] form)
    other => {
      if is_neg_infinity(other) {
        return Ok(crate::functions::math_ast::make_rational(-1, 2));
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "FresnelS".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute S(x) = ∫₀ˣ sin(π t²/2) dt numerically
/// Uses the identity: S(x) = Im[(1+i)/2 · erf(√π/2 · (1+i) · x)]
fn fresnel_s_numeric(x: f64) -> f64 {
  if x == 0.0 {
    return 0.0;
  }
  let (_, s) = fresnel_cs_via_erf(x);
  s
}

/// FresnelC[z] - Fresnel cosine integral C(z) = ∫₀ᶻ cos(π t²/2) dt
pub fn fresnel_c_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "FresnelC expects exactly 1 argument".into(),
    ));
  }

  // Helper: negate FresnelC (odd function)
  let negate_fresnel_c = |inner: Expr| -> Expr {
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::FunctionCall {
        name: "FresnelC".to_string(),
        args: vec![inner],
      }),
    }
  };

  match &args[0] {
    // FresnelC[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // FresnelC[Infinity] = 1/2
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(crate::functions::math_ast::make_rational(1, 2))
    }
    // FresnelC[ComplexInfinity] = Indeterminate
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      Ok(Expr::Identifier("Indeterminate".to_string()))
    }
    // FresnelC[I] = I*FresnelC[1]
    Expr::Identifier(s) if s == "I" => {
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::Identifier("I".to_string()),
          Expr::FunctionCall {
            name: "FresnelC".to_string(),
            args: vec![Expr::Integer(1)],
          },
        ],
      )
    }
    // FresnelC[-x] = -FresnelC[x] (UnaryOp form)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      // Check for -Infinity first
      if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") {
        return Ok(crate::functions::math_ast::make_rational(-1, 2));
      }
      Ok(negate_fresnel_c(*operand.clone()))
    }
    // FresnelC[Times[-1, x]] = -FresnelC[x]
    Expr::FunctionCall { name, args: fargs }
      if name == "Times" && fargs.len() == 2 =>
    {
      if matches!(&fargs[0], Expr::Integer(-1)) {
        return Ok(negate_fresnel_c(fargs[1].clone()));
      }
      if matches!(&fargs[1], Expr::Integer(-1)) {
        return Ok(negate_fresnel_c(fargs[0].clone()));
      }
      // Negative integer coefficient
      if let Expr::Integer(n) = &fargs[0]
        && *n < 0
      {
        let pos_arg = Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-*n), fargs[1].clone()],
        };
        return Ok(negate_fresnel_c(pos_arg));
      }
      // FresnelC[I*z] = I*FresnelC[z]
      if matches!(&fargs[0], Expr::Identifier(s) if s == "I") {
        return crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[
            Expr::Identifier("I".to_string()),
            Expr::FunctionCall {
              name: "FresnelC".to_string(),
              args: vec![fargs[1].clone()],
            },
          ],
        );
      }
      Ok(Expr::FunctionCall {
        name: "FresnelC".to_string(),
        args: args.to_vec(),
      })
    }
    // FresnelC[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_fresnel_c(Expr::Integer(-*n))),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(fresnel_c_numeric(*x))),
    // Check for -Infinity (Times[-1, Infinity] form)
    other => {
      if is_neg_infinity(other) {
        return Ok(crate::functions::math_ast::make_rational(-1, 2));
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "FresnelC".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute C(x) = ∫₀ˣ cos(π t²/2) dt numerically
/// Uses the identity: C(x) = Re[(1+i)/2 · erf(√π/2 · (1+i) · x)]
fn fresnel_c_numeric(x: f64) -> f64 {
  if x == 0.0 {
    return 0.0;
  }
  let (c, _) = fresnel_cs_via_erf(x);
  c
}

/// Compute both Fresnel integrals C(x) and S(x) via the error function identity:
///   C(x) + i·S(x) = (1+i)/2 · erf(√π/2 · (1+i) · x)
/// Returns (C(x), S(x)).
fn fresnel_cs_via_erf(x: f64) -> (f64, f64) {
  // z = √π/2 · (1+i) · x = (√π/2 · x) + i·(√π/2 · x)
  let sqrt_pi_half = std::f64::consts::PI.sqrt() / 2.0;
  let t = sqrt_pi_half * x;
  let (er, ei) = erf_complex(t, t);
  // Using: C = (E_r + E_i)/2, S = (E_r - E_i)/2
  // where erf((1+i)·t) = E_r + i·E_i
  let c = (er + ei) / 2.0;
  let s = (er - ei) / 2.0;
  (c, s)
}

/// Compute the complex error function erf(a + bi) = (u, v) where result = u + iv.
/// Uses the identity erf(z) = 1 - exp(-z²)·w(iz) where w is the Faddeeva function.
/// The Faddeeva function is computed via the Laplace continued fraction for Im(z) > 0.
fn erf_complex(a: f64, b: f64) -> (f64, f64) {
  // Compute w(iz) where iz = -b + ai
  let (wz_re, wz_im) = faddeeva(-b, a);
  // exp(-z²) where z = a + bi, z² = a²-b² + 2abi
  let z2_re = a * a - b * b;
  let z2_im = 2.0 * a * b;
  let exp_re = (-z2_re).exp() * (-z2_im).cos();
  let exp_im = (-z2_re).exp() * (-z2_im).sin();
  // erf(z) = 1 - exp(-z²)·w(iz)
  let prod_re = exp_re * wz_re - exp_im * wz_im;
  let prod_im = exp_re * wz_im + exp_im * wz_re;
  (1.0 - prod_re, -prod_im)
}

/// Compute the Faddeeva function w(z) = exp(-z²)·erfc(-iz) for complex z.
/// Uses the algorithm from Poppe & Wijers (1990) / Weideman (1994).
/// For Im(z) >= 0, uses the Laplace continued fraction.
/// Returns (Re(w), Im(w)).
fn faddeeva(a: f64, b: f64) -> (f64, f64) {
  // For Im(z) < 0, use the reflection: w(z) = 2·exp(-z²) - w(-z)
  if b < 0.0 {
    let (wr, wi) = faddeeva(-a, -b);
    let z2_re = a * a - b * b;
    let z2_im = 2.0 * a * b;
    let exp_re = (-z2_re).exp() * (-z2_im).cos();
    let exp_im = (-z2_re).exp() * (-z2_im).sin();
    return (2.0 * exp_re - wr, 2.0 * exp_im - wi);
  }

  let r2 = a * a + b * b;

  if r2 < 3.0 {
    // For small |z|, use Taylor series for erf then convert:
    // w(z) = exp(-z²) · (1 - erf(-iz))
    // But this has the same cancellation issue. Instead, use the direct series:
    // w(z) = Σ_{n=0}^∞ (iz)^n / Γ(n/2 + 1)
    // which is equivalent to the power series for the Faddeeva function.
    //
    // For small |z|, compute w(z) = exp(-z²)·(1 - erf(-iz)) using Taylor series.
    // erf(-iz) = erf(b - ai)
    let (erf_niz_re, erf_niz_im) = erf_complex_taylor(b, -a);
    // w(z) = exp(-z²) · (1 - erf(-iz))
    let wz2_re = a * a - b * b;
    let wz2_im = 2.0 * a * b;
    let wexp_re = (-wz2_re).exp() * (-wz2_im).cos();
    let wexp_im = (-wz2_re).exp() * (-wz2_im).sin();
    let erfc_re = 1.0 - erf_niz_re;
    let erfc_im = -erf_niz_im;
    (
      wexp_re * erfc_re - wexp_im * erfc_im,
      wexp_re * erfc_im + wexp_im * erfc_re,
    )
  } else {
    // For large |z| with Im(z) >= 0, use the Laplace continued fraction:
    // w(z) = i/(√π) · 1/(z - 1/2/(z - 1/(z - 3/2/(z - 2/(z - ...)))))
    // Evaluated using modified Lentz's algorithm.
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();

    // Modified Lentz's method for complex continued fraction
    // w(z) = i/√π · CF where CF = 1/(z - a₁/(z - a₂/(z - ...)))
    // with aₙ = n/2
    let tiny = 1e-30;
    let mut f_re = a;
    let mut f_im = b;
    if f_re.abs() < tiny && f_im.abs() < tiny {
      f_re = tiny;
    }
    let mut c_re = f_re;
    let mut c_im = f_im;
    let mut d_re = 0.0;
    let mut d_im = 0.0;

    for n in 1..300 {
      let a_n = n as f64 * 0.5;
      // d = z + a_n * d → d = (a + a_n*d_re, b + a_n*d_im)
      // Actually: continued fraction b_n + a_n / ... where b_n = z, a_n = -n/2
      // The CF is: z - (1/2)/(z - 1/(z - (3/2)/(z - ...)))
      // In standard form: b₀ = z, a₁ = -1/2, b₁ = z, a₂ = -1, b₂ = z, ...
      // aₙ = -n/2, bₙ = z for n >= 1

      // d = b_n + a_n · d = z - (n/2) · d
      let nd_re = a + (-a_n) * d_re;
      let nd_im = b + (-a_n) * d_im;
      d_re = nd_re;
      d_im = nd_im;

      // d = 1/d
      let mag2 = d_re * d_re + d_im * d_im;
      if mag2 < tiny * tiny {
        d_re = tiny;
        d_im = 0.0;
      } else {
        let inv_re = d_re / mag2;
        let inv_im = -d_im / mag2;
        d_re = inv_re;
        d_im = inv_im;
      }

      // c = b_n + a_n / c = z - (n/2) / c
      let c_mag2 = c_re * c_re + c_im * c_im;
      let inv_c_re = c_re / c_mag2;
      let inv_c_im = -c_im / c_mag2;
      c_re = a + (-a_n) * inv_c_re;
      c_im = b + (-a_n) * inv_c_im;

      // delta = c * d
      let delta_re = c_re * d_re - c_im * d_im;
      let delta_im = c_re * d_im + c_im * d_re;

      // f *= delta
      let nf_re = f_re * delta_re - f_im * delta_im;
      let nf_im = f_re * delta_im + f_im * delta_re;
      f_re = nf_re;
      f_im = nf_im;

      if (delta_re - 1.0).abs() + delta_im.abs() < 1e-15 {
        break;
      }
    }

    // w(z) = i/(√π · f)  →  i/f = i/(f_re + i·f_im) = (f_im + i·f_re) / |f|²
    // Wait: 1/f first, then multiply by i.
    let f_mag2 = f_re * f_re + f_im * f_im;
    let inv_f_re = f_re / f_mag2;
    let inv_f_im = -f_im / f_mag2;
    // i · (inv_f_re + i·inv_f_im) = -inv_f_im + i·inv_f_re
    (-inv_f_im * inv_sqrt_pi, inv_f_re * inv_sqrt_pi)
  }
}

/// Compute erf(a+bi) using Taylor series with reverse summation for precision.
fn erf_complex_taylor(a: f64, b: f64) -> (f64, f64) {
  let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
  let z2_re = a * a - b * b;
  let z2_im = 2.0 * a * b;
  let mut terms_re = Vec::new();
  let mut terms_im = Vec::new();
  let mut term_re = a;
  let mut term_im = b;
  terms_re.push(term_re);
  terms_im.push(term_im);
  for n in 1..300 {
    let new_re = -(term_re * z2_re - term_im * z2_im) / n as f64;
    let new_im = -(term_re * z2_im + term_im * z2_re) / n as f64;
    term_re = new_re;
    term_im = new_im;
    let denom = (2 * n + 1) as f64;
    terms_re.push(term_re / denom);
    terms_im.push(term_im / denom);
    if (term_re / denom).abs() + (term_im / denom).abs() < 1e-20 {
      break;
    }
  }
  // Sum from smallest to largest for better precision
  let mut sum_re = 0.0;
  let mut sum_im = 0.0;
  for i in (0..terms_re.len()).rev() {
    sum_re += terms_re[i];
    sum_im += terms_im[i];
  }
  (two_over_sqrt_pi * sum_re, two_over_sqrt_pi * sum_im)
}

/// Public wrapper for FresnelS numeric computation (used by try_eval_to_f64)
pub fn fresnel_s_numeric_pub(x: f64) -> f64 {
  fresnel_s_numeric(x)
}

/// Public wrapper for FresnelC numeric computation (used by try_eval_to_f64)
pub fn fresnel_c_numeric_pub(x: f64) -> f64 {
  fresnel_c_numeric(x)
}

/// SinIntegral[z] - Sine integral Si(z)
pub fn sin_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SinIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // SinIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // SinIntegral[Infinity] = Pi/2
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        crate::functions::math_ast::make_rational(1, 2),
        Expr::Constant("Pi".to_string()),
      ],
    }),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(sin_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // SinIntegral[-Infinity] = -Pi/2
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            crate::functions::math_ast::make_rational(-1, 2),
            Expr::Constant("Pi".to_string()),
          ],
        });
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "SinIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Si(z) numerically for real z
/// Si(z) = Σ_{n=0}^∞ (-1)^n z^(2n+1) / ((2n+1) · (2n+1)!)
fn sin_integral_numeric(z: f64) -> f64 {
  if z.abs() < 40.0 {
    let z2 = z * z;
    let mut sum = z;
    let mut term = z;
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      let n2p1 = n2 + 1.0;
      term *= -z2 / (n2 * n2p1);
      sum += term / n2p1;
      if (term / n2p1).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |z|
    // Si(z) ≈ π/2 - cos(z)/z * f(z) - sin(z)/z * g(z)
    let sign = if z > 0.0 { 1.0 } else { -1.0 };
    let az = z.abs();
    let mut f = 0.0;
    let mut g = 0.0;
    let mut f_term = 1.0;
    let mut g_term = 1.0 / az;
    for n in 0..100 {
      f += if n % 2 == 0 { f_term } else { -f_term };
      g += if n % 2 == 0 { g_term } else { -g_term };
      let n2 = (2 * n + 1) as f64;
      let n2p = (2 * n + 2) as f64;
      let old_f = f_term;
      f_term *= n2 * n2p / (az * az);
      g_term *= n2p * (n2p + 1.0) / (az * az);
      if f_term.abs() < 1e-16 && g_term.abs() < 1e-16 {
        break;
      }
      if f_term.abs() > old_f.abs() {
        break;
      }
    }
    sign * (std::f64::consts::FRAC_PI_2 - f / az * az.cos() - g * az.sin())
  }
}

/// ExpIntegralE[n, z] - Generalized exponential integral E_n(z)
pub fn exp_integral_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ExpIntegralE expects exactly 2 arguments".into(),
    ));
  }

  let n_expr = &args[0];
  let z_expr = &args[1];

  // E_n(0): E_1(0) = ComplexInfinity, E_n(0) = 1/(n-1) for n > 1
  if is_expr_zero(z_expr)
    && let Expr::Integer(n) = n_expr
  {
    if *n == 1 {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    } else if *n > 1 {
      return Ok(make_rational(1, *n - 1));
    }
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = exp_integral_en(n as i64, z);
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "ExpIntegralE".to_string(),
    args: args.to_vec(),
  })
}

/// Compute E_n(z) = ∫_1^∞ e^{-zt}/t^n dt
pub fn exp_integral_en(n: i64, z: f64) -> f64 {
  if n == 0 {
    // E_0(z) = e^{-z}/z
    return (-z).exp() / z;
  }

  // Compute E_1(z) first, then use recurrence for higher n
  let e1 = exp_integral_e1(z);

  if n == 1 {
    return e1;
  }

  // Recurrence: E_{n+1}(z) = (e^{-z} - z*E_n(z))/n
  let mut e_prev = e1;
  let exp_neg_z = (-z).exp();
  for k in 1..n {
    let e_next = (exp_neg_z - z * e_prev) / k as f64;
    e_prev = e_next;
  }
  e_prev
}

/// Compute E_1(z) via series for small z, continued fraction for large z
pub fn exp_integral_e1(z: f64) -> f64 {
  let euler_gamma = 0.5772156649015329;

  if z <= 0.0 {
    return f64::INFINITY;
  }

  if z < 1.5 {
    // Series: E_1(z) = -γ - ln(z) + Σ_{n=1}^∞ (-1)^{n+1} z^n / (n * n!)
    let mut sum = -euler_gamma - z.ln();
    let mut term = 1.0;
    for n in 1..200 {
      let nf = n as f64;
      term *= -z / nf;
      sum -= term / nf;
      if (term / nf).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Continued fraction: E_1(z) = e^{-z} * 1/(z + 1/(1 + 1/(z + 2/(1 + 2/(z + ...)))))
    // Using Lentz-Thompson algorithm with a_i, b_i coefficients
    // E_1(z) = e^{-z} * CF where CF = 1/(z+) 1/(1+) 1/(z+) 2/(1+) 2/(z+) 3/(1+) ...
    // Equivalently using the modified Lentz: a_0=0, b_0=z, then alternating
    // numerators [1, 1, 2, 2, 3, 3, ...] and denominators [1, z, 1, z, 1, z, ...]
    let mut f = z; // b_0
    let mut c = z;
    let mut d = 0.0;

    for i in 1..200 {
      let a = ((i + 1) / 2) as f64; // 1, 1, 2, 2, 3, 3, ...
      let b = if i % 2 == 1 { 1.0 } else { z };

      d = b + a * d;
      if d.abs() < 1e-30 {
        d = 1e-30;
      }
      c = b + a / c;
      if c.abs() < 1e-30 {
        c = 1e-30;
      }
      d = 1.0 / d;
      let delta = c * d;
      f *= delta;
      if (delta - 1.0).abs() < 1e-16 {
        break;
      }
    }

    (-z).exp() / f
  }
}

/// LogIntegral[x] - Logarithmic integral Li(x) = Ei(ln(x))
pub fn log_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "LogIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // LogIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // LogIntegral[1] = -Infinity (pole at x=1 since ln(1)=0)
    Expr::Integer(1) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // Numeric evaluation: Li(x) = Ei(ln(x))
    Expr::Real(x) => {
      let result = exp_integral_ei_numeric(x.ln());
      Ok(Expr::Real(result))
    }
    // Unevaluated
    _ => Ok(Expr::FunctionCall {
      name: "LogIntegral".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// RiemannR[x] - Riemann's prime counting function estimate
/// Uses the Gram series: R(x) = 1 + Σ_{n=1}^∞ (ln x)^n / (n * n! * ζ(n+1))
pub fn riemann_r_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RiemannR expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // RiemannR[1] = 1 (exact special value)
    Expr::Integer(1) => Ok(Expr::Integer(1)),
    // Numeric evaluation for real values
    Expr::Real(x) if *x > 0.0 => Ok(Expr::Real(riemann_r_numeric(*x))),
    // Unevaluated for symbolic and other args
    _ => Ok(Expr::FunctionCall {
      name: "RiemannR".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute Riemann R function numerically using the Gram series.
/// Computes (ln x)^n / n! incrementally to avoid precision loss from
/// large intermediate values.  Uses Kahan compensated summation.
fn riemann_r_numeric(x: f64) -> f64 {
  let ln_x = x.ln();
  let mut sum = 1.0_f64;
  let mut comp = 0.0_f64; // Kahan compensation
  let mut ratio = 1.0_f64; // (ln x)^n / n!  (updated incrementally)

  for n in 1..=200 {
    ratio *= ln_x / n as f64; // ratio = (ln x)^n / n!
    let zeta_val = zeta_numeric((n + 1) as f64);
    let term = ratio / (n as f64 * zeta_val);
    // Kahan compensated addition
    let y = term - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }
  sum
}

/// SinhIntegral[z] - Hyperbolic sine integral Shi(z)
/// Shi(z) = ∫₀ᶻ sinh(t)/t dt
pub fn sinh_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SinhIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // SinhIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // SinhIntegral[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(sinh_integral_numeric(*x))),
    other => {
      if is_neg_infinity(other) {
        // SinhIntegral[-Infinity] = -Infinity
        return Ok(Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(-1),
            Expr::Identifier("Infinity".to_string()),
          ],
        });
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "SinhIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Shi(z) numerically for real z
/// Shi(z) = Σ_{n=0}^∞ z^(2n+1) / ((2n+1) · (2n+1)!)
fn sinh_integral_numeric(z: f64) -> f64 {
  if z.abs() < 40.0 {
    // Power series: Shi(z) = Σ z^(2n+1) / ((2n+1) · (2n+1)!)
    let z2 = z * z;
    let mut sum = z;
    let mut term = z;
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      let n2p1 = n2 + 1.0;
      // term *= z^2 / (2n * (2n+1))
      term *= z2 / (n2 * n2p1);
      sum += term / n2p1;
      if (term / n2p1).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // For large |z|, use relationship: Shi(z) = (E1(-z) - E1(z)) / 2 + ln|z| - ln|z|
    // Actually, use: Shi(z) = (exp(z)/(2z)) * Σ ... asymptotic
    // Simpler: Shi(z) = (Ei(z) - E1(z)) / 2  where Ei is exponential integral
    // Or directly: Shi(z) = cosh(z)/z * f(z) + sinh(z)/z * g(z)
    // where the asymptotic expansion mirrors Si but with hyperbolic functions
    // For practical purposes, use: Shi(z) = (e^z/(2z)) * Σ_{k=0} k! / z^k
    //                                      - (e^{-z}/(2z)) * Σ_{k=0} (-1)^k k! / z^k
    // This is divergent but gives good results when truncated
    let sign = z.signum();
    let az = z.abs();
    // Shi(z) is odd, so Shi(z) = sign * Shi(|z|)
    // Use: Shi(z) = (e^z - e^{-z})/(2z) + integral correction
    // Better: use direct series for moderate z and Ei for large z
    // Shi(z) = -i * Si(i*z) ... but let's just extend the series range or use Ei
    // Shi(z) = (Ei(z) + Ei(-z))/2 + ... no, Shi(z) = (Ei(z) - Ei(-z))/2 - (ln(z) - ln(-z))/2
    // For real z > 0: Shi(z) = (Ei(z) - E1(z)) / 2
    // where Ei(z) = -E1(-z) + i*pi for z > 0... this gets complex.
    // Let's just use the continued fraction / asymptotic for e^z and e^{-z} terms.
    // Shi(z) = e^z/(2z) * Σ n!/(z^n) (even terms) - e^{-z}/(2z) * Σ n!/(z^n) (alt signs)
    // Actually simpler: just use the power series with higher precision cutoff
    // For z up to ~700 (before exp overflow), the asymptotic expansion works
    let mut sum_p = 1.0_f64; // positive exponential coefficient
    let mut sum_n = 1.0_f64; // negative exponential coefficient
    let mut term = 1.0_f64;
    for k in 1..100 {
      term *= (k as f64) / az;
      sum_p += term;
      sum_n += if k % 2 == 0 { term } else { -term };
      if term.abs() < 1e-15 {
        break;
      }
      // Divergent asymptotic series: stop when terms start growing
      if term > (k as f64) / az * term {
        break;
      }
    }
    if az < 500.0 {
      sign * (az.exp() * sum_p - (-az).exp() * sum_n) / (2.0 * az)
    } else {
      // For very large z, e^{-z} term is negligible
      sign * az.exp() * sum_p / (2.0 * az)
    }
  }
}

/// CoshIntegral[z] - Hyperbolic cosine integral Chi(z)
/// Chi(z) = γ + ln(z) + ∫₀ᶻ (cosh(t)-1)/t dt
pub fn cosh_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "CoshIntegral expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // CoshIntegral[0] = -Infinity
    Expr::Integer(0) => Ok(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
    }),
    // CoshIntegral[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(Expr::Identifier("Infinity".to_string()))
    }
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(cosh_integral_numeric(*x))),
    other => {
      if is_neg_infinity(other) {
        // CoshIntegral[-Infinity] = Infinity (real part)
        return Ok(Expr::Identifier("Infinity".to_string()));
      }
      // Unevaluated
      Ok(Expr::FunctionCall {
        name: "CoshIntegral".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Chi(z) numerically for real z
/// Chi(z) = γ + ln|z| + ∫₀ᶻ (cosh(t)-1)/t dt
/// = γ + ln|z| + Σ_{n=1}^∞ z^(2n) / (2n · (2n)!)
fn cosh_integral_numeric(z: f64) -> f64 {
  let euler_gamma = 0.5772156649015329;

  if z.abs() < 40.0 {
    // Power series: Chi(z) = γ + ln|z| + Σ z^(2n) / (2n · (2n)!)
    let mut sum = euler_gamma + z.abs().ln();
    let z2 = z * z;
    let mut term = 1.0; // Will build up z^(2n) / (2n)!
    for n in 1..200 {
      let n2 = (2 * n) as f64;
      // term *= z^2 / ((2n-1) * 2n)
      term *= z2 / ((n2 - 1.0) * n2);
      sum += term / n2;
      if (term / n2).abs() < 1e-16 * sum.abs().max(1e-300) {
        break;
      }
    }
    sum
  } else {
    // Asymptotic expansion for large |z|
    // Chi(z) ≈ e^z/(2z) * Σ + e^{-z}/(2z) * Σ (alternating)
    let az = z.abs();
    let mut sum_p = 1.0_f64;
    let mut sum_n = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1..100 {
      term *= (k as f64) / az;
      sum_p += term;
      sum_n += if k % 2 == 0 { term } else { -term };
      if term.abs() < 1e-15 {
        break;
      }
    }
    if az < 500.0 {
      (az.exp() * sum_p + (-az).exp() * sum_n) / (2.0 * az)
    } else {
      az.exp() * sum_p / (2.0 * az)
    }
  }
}
