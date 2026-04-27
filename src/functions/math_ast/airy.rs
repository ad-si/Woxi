#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

/// AiryAi[x] - Airy function of the first kind
pub fn airy_ai_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryAi expects exactly 1 argument".into(),
    ));
  }

  // AiryAi[0] = 3^(-2/3) / Gamma[2/3]
  if matches!(&args[0], Expr::Integer(0)) {
    return airy_build_value((-2, 3), (2, 3), false);
  }

  // AiryAi[AiryAiZero[k]] = 0 by definition
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "AiryAiZero"
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_ai(x_f)));
  }

  // Complex floating-point argument: power series via complex arithmetic.
  if let Some((re, im)) =
    crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
    && contains_inexact_real(&args[0])
  {
    let (ar, ai) = airy_ai_complex(re, im);
    return Ok(crate::functions::math_ast::build_complex_float_expr(ar, ai));
  }

  Ok(Expr::FunctionCall {
    name: "AiryAi".to_string(),
    args: args.to_vec(),
  })
}

/// True if the expression tree contains a Real or BigFloat node.
fn contains_inexact_real(expr: &Expr) -> bool {
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => true,
    Expr::UnaryOp { operand, .. } => contains_inexact_real(operand),
    Expr::BinaryOp { left, right, .. } => {
      contains_inexact_real(left) || contains_inexact_real(right)
    }
    Expr::FunctionCall { args, .. } => args.iter().any(contains_inexact_real),
    _ => false,
  }
}

/// Complex Airy Ai via the power series
///   Ai(z) = c1·f(z) − c2·g(z)
///   f(z) = Σ z^(3k) / [(2·3)(5·6)…((3k−1)·3k)]
///   g(z) = Σ z^(3k+1) / [(3·4)(6·7)…(3k·(3k+1))]
/// Converges quickly for |z| up to ~10; sufficient for the
/// `0.5 + I`-style arguments in the test suite.
fn airy_ai_complex(re: f64, im: f64) -> (f64, f64) {
  let c1 = 0.3550280538878172;
  let c2 = 0.2588194037928068;
  let z = (re, im);
  let z3 = cmul(cmul(z, z), z);
  let mut f = (1.0, 0.0);
  let mut g = z;
  let mut f_term = (1.0, 0.0);
  let mut g_term = z;
  for k in 1..1000 {
    let k3 = 3 * k;
    let f_div = ((k3 - 1) as f64) * (k3 as f64);
    let g_div = (k3 as f64) * ((k3 + 1) as f64);
    f_term = (cmul(f_term, z3).0 / f_div, cmul(f_term, z3).1 / f_div);
    g_term = (cmul(g_term, z3).0 / g_div, cmul(g_term, z3).1 / g_div);
    f = (f.0 + f_term.0, f.1 + f_term.1);
    g = (g.0 + g_term.0, g.1 + g_term.1);
    let mag_f = f_term.0 * f_term.0 + f_term.1 * f_term.1;
    let mag_g = g_term.0 * g_term.0 + g_term.1 * g_term.1;
    if mag_f < 1e-32 && mag_g < 1e-32 {
      break;
    }
  }
  (c1 * f.0 - c2 * g.0, c1 * f.1 - c2 * g.1)
}

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Compute Ai(x) using the two power series
/// Ai(x) = c1 * f(x) - c2 * g(x)
/// f(x) = 1 + x^3/(2·3) + x^6/(2·3·5·6) + ...
/// g(x) = x + x^4/(3·4) + x^7/(3·4·6·7) + ...
/// c1 = Ai(0), c2 = -Ai'(0)
pub fn airy_ai(x: f64) -> f64 {
  let c1 = 0.3550280538878172; // Ai(0)
  let c2 = 0.2588194037928068; // -Ai'(0)

  // For Ai: use power series for moderate/negative x, asymptotic for large positive x
  // The threshold is lower than Bi because Ai decays exponentially, causing cancellation
  if x < 6.0 {
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..1000 {
      let k3 = 3 * k;
      f_term *= x * x * x / ((k3 as f64 - 1.0) * k3 as f64);
      g_term *= x * x * x / (k3 as f64 * (k3 as f64 + 1.0));
      f += f_term;
      g += g_term;
      if f_term.abs() < 1e-16 * f.abs().max(1e-300)
        && g_term.abs() < 1e-16 * g.abs().max(1e-300)
      {
        break;
      }
    }

    c1 * f - c2 * g
  } else {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor =
      (-zeta).exp() / (2.0 * std::f64::consts::PI.sqrt() * x.powf(0.25));
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
      term *= -num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// AiryBi[x] - Airy function of the second kind
pub fn airy_bi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryBi expects exactly 1 argument".into(),
    ));
  }

  // AiryBi[0] = 3^(5/6) / (3 * Gamma[2/3])
  if matches!(&args[0], Expr::Integer(0)) {
    return airy_build_value((5, 6), (2, 3), true);
  }

  // AiryBi[AiryBiZero[k]] = 0 by definition
  if let Expr::FunctionCall { name, .. } = &args[0]
    && name == "AiryBiZero"
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi(x_f)));
  }

  // Complex floating-point argument: same power series, complex arithmetic.
  if let Some((re, im)) =
    crate::functions::math_ast::try_extract_complex_float(&args[0])
    && im != 0.0
    && contains_inexact_real(&args[0])
  {
    let (br, bi) = airy_bi_complex(re, im);
    return Ok(crate::functions::math_ast::build_complex_float_expr(br, bi));
  }

  Ok(Expr::FunctionCall {
    name: "AiryBi".to_string(),
    args: args.to_vec(),
  })
}

/// Complex Airy Bi via the same power series as `airy_ai_complex`,
/// recombined per Bi(z) = √3·(c1·f(z) + c2·g(z)).
fn airy_bi_complex(re: f64, im: f64) -> (f64, f64) {
  let c1 = 0.3550280538878172;
  let c2 = 0.2588194037928068;
  let sqrt3 = 3.0_f64.sqrt();
  let z = (re, im);
  let z3 = cmul(cmul(z, z), z);
  let mut f = (1.0, 0.0);
  let mut g = z;
  let mut f_term = (1.0, 0.0);
  let mut g_term = z;
  for k in 1..1000 {
    let k3 = 3 * k;
    let f_div = ((k3 - 1) as f64) * (k3 as f64);
    let g_div = (k3 as f64) * ((k3 + 1) as f64);
    let nf = cmul(f_term, z3);
    f_term = (nf.0 / f_div, nf.1 / f_div);
    let ng = cmul(g_term, z3);
    g_term = (ng.0 / g_div, ng.1 / g_div);
    f = (f.0 + f_term.0, f.1 + f_term.1);
    g = (g.0 + g_term.0, g.1 + g_term.1);
    let mag_f = f_term.0 * f_term.0 + f_term.1 * f_term.1;
    let mag_g = g_term.0 * g_term.0 + g_term.1 * g_term.1;
    if mag_f < 1e-32 && mag_g < 1e-32 {
      break;
    }
  }
  let re = sqrt3 * (c1 * f.0 + c2 * g.0);
  let im = sqrt3 * (c1 * f.1 + c2 * g.1);
  (re, im)
}

/// Compute Bi(x) using the two power series
/// Bi(x) = sqrt(3) * (c1 * f(x) + c2 * g(x))
/// where f and g are the same series as for Ai(x)
/// c1 = Ai(0), c2 = -Ai'(0)
pub fn airy_bi(x: f64) -> f64 {
  let c1 = 0.3550280538878172; // Ai(0)
  let c2 = 0.2588194037928068; // -Ai'(0)
  let sqrt3 = 3.0_f64.sqrt();

  if x < 15.0 {
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..1000 {
      let k3 = 3 * k;
      f_term *= x * x * x / ((k3 as f64 - 1.0) * k3 as f64);
      g_term *= x * x * x / (k3 as f64 * (k3 as f64 + 1.0));
      f += f_term;
      g += g_term;
      if f_term.abs() < 1e-16 * f.abs().max(1e-300)
        && g_term.abs() < 1e-16 * g.abs().max(1e-300)
      {
        break;
      }
    }

    sqrt3 * (c1 * f + c2 * g)
  } else {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor = zeta.exp() / (std::f64::consts::PI.sqrt() * x.powf(0.25));
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
      term *= num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// AiryAiPrime[x] - Derivative of the Airy function Ai(x)
pub fn airy_ai_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryAiPrime expects exactly 1 argument".into(),
    ));
  }

  // AiryAiPrime[0] = -3^(2/3) / (3 * Gamma[1/3])
  if matches!(&args[0], Expr::Integer(0)) {
    let result = airy_build_value((2, 3), (1, 3), true)?;
    return crate::evaluator::evaluate_expr_to_expr(&Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand: Box::new(result),
    });
  }

  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_ai_prime(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryAiPrime".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Ai'(x) using power series and asymptotic expansion
/// Ai'(x) = c1 * f'(x) - c2 * g'(x)
pub fn airy_ai_prime(x: f64) -> f64 {
  let c1 = 0.3550280538878172; // Ai(0)
  let c2 = 0.2588194037928068; // -Ai'(0)

  if x < 6.0 {
    let x3 = x * x * x;

    // f'(x): f'_term_1 = x^2/2, ratio f'_{k}/f'_{k-1} = x^3 / ((3k-3)(3k-1))
    let mut fp_term = x * x / 2.0;
    let mut fp = fp_term;
    for k in 2..1000 {
      let k3 = (3 * k) as f64;
      fp_term *= x3 / ((k3 - 3.0) * (k3 - 1.0));
      fp += fp_term;
      if fp_term.abs() < 1e-16 * fp.abs().max(1e-300) {
        break;
      }
    }

    // g'(x): g'_0 = 1, g'_term_1 = x^3/3, ratio g'_{k}/g'_{k-1} = x^3 / ((3k-2)(3k))
    let mut gp_term = x3 / 3.0;
    let mut gp = 1.0 + gp_term;
    for k in 2..1000 {
      let k3 = (3 * k) as f64;
      gp_term *= x3 / ((k3 - 2.0) * k3);
      gp += gp_term;
      if gp_term.abs() < 1e-16 * gp.abs().max(1e-300) {
        break;
      }
    }

    c1 * fp - c2 * gp
  } else {
    // Asymptotic for large positive x
    // Ai'(x) ~ -x^(1/4)/(2*sqrt(pi)) * exp(-2/3 * x^(3/2)) * (1 + ...)
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor =
      -(-zeta).exp() * x.powf(0.25) / (2.0 * std::f64::consts::PI.sqrt());
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      // Asymptotic coefficients for Ai'
      let num = (6.0 * kf - 7.0) * (6.0 * kf - 5.0) * (6.0 * kf - 1.0);
      term *= num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// AiryBiPrime[x] - Derivative of the Airy function Bi(x)
pub fn airy_bi_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryBiPrime expects exactly 1 argument".into(),
    ));
  }

  // AiryBiPrime[0] = 3^(1/6) / Gamma[1/3]
  if matches!(&args[0], Expr::Integer(0)) {
    // 3^(1/6) / Gamma[1/3] — no factor of 3 in denominator
    let power_3 = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Integer(3),
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(6)],
        },
      ],
    };
    let gamma = Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: vec![Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(3)],
      }],
    };
    let result = Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left: Box::new(power_3),
      right: Box::new(gamma),
    };
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi_prime(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryBiPrime".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Bi'(x) using the same series as Ai' but with different sign
/// Bi'(x) = sqrt(3) * (c1 * f'(x) + c2 * g'(x))
pub fn airy_bi_prime(x: f64) -> f64 {
  let c1 = 0.3550280538878172;
  let c2 = 0.2588194037928068;
  let sqrt3 = 3.0_f64.sqrt();

  if x < 15.0 {
    let x3 = x * x * x;

    let mut fp = x * x / 2.0;
    let mut fp_term = fp;
    for k in 2..1000 {
      let k3 = (3 * k) as f64;
      fp_term *= x3 / ((k3 - 3.0) * (k3 - 1.0));
      fp += fp_term;
      if fp_term.abs() < 1e-16 * fp.abs().max(1e-300) {
        break;
      }
    }

    let mut gp = 1.0;
    let mut gp_term = x3 / 3.0;
    gp += gp_term;
    for k in 2..1000 {
      let k3 = (3 * k) as f64;
      gp_term *= x3 / ((k3 - 2.0) * k3);
      gp += gp_term;
      if gp_term.abs() < 1e-16 * gp.abs().max(1e-300) {
        break;
      }
    }

    sqrt3 * (c1 * fp + c2 * gp)
  } else {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor = zeta.exp() * x.powf(0.25) / std::f64::consts::PI.sqrt();
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut prev_term = f64::MAX;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 7.0) * (6.0 * kf - 5.0) * (6.0 * kf - 1.0);
      term *= -num / (216.0 * kf * zeta);
      if term.abs() > prev_term {
        break;
      }
      prev_term = term.abs();
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  }
}

/// Build 3^(p/q) / Gamma[r/s] or 3^(p/q) / (3 * Gamma[r/s])
fn airy_build_value(
  power_frac: (i128, i128),
  gamma_frac: (i128, i128),
  with_extra_3: bool,
) -> Result<Expr, InterpreterError> {
  let power_3 = Expr::FunctionCall {
    name: "Power".to_string(),
    args: vec![
      Expr::Integer(3),
      Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(power_frac.0), Expr::Integer(power_frac.1)],
      },
    ],
  };
  let gamma = Expr::FunctionCall {
    name: "Gamma".to_string(),
    args: vec![Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(gamma_frac.0), Expr::Integer(gamma_frac.1)],
    }],
  };
  let denom = if with_extra_3 {
    Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(3), gamma],
    }
  } else {
    gamma
  };
  let result = Expr::BinaryOp {
    op: crate::syntax::BinaryOperator::Divide,
    left: Box::new(power_3),
    right: Box::new(denom),
  };
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// Compute the n-th real zero of AiryAi using Newton's method on the
/// asymptotic starting point a_n ≈ -t^(2/3)(1 + 5/(48 t^2) - ...),
/// t = 3π(4n-1)/8. Returns an exact-symbolic `AiryAiZero[n]` wrapper
/// when `n` is a plain integer; N[] (or a Real argument) triggers the
/// numeric evaluation.
fn airy_ai_zero_f64(n: i128) -> f64 {
  let t = 3.0 * std::f64::consts::PI * (4.0 * n as f64 - 1.0) / 8.0;
  let t2 = t * t;
  let correction = 1.0 + 5.0 / (48.0 * t2) - 5.0 / (36.0 * t2 * t2);
  let mut x = -t.powf(2.0 / 3.0) * correction;
  // Polish with Newton's method on Ai: x_{k+1} = x - Ai(x)/Ai'(x).
  for _ in 0..40 {
    let f = airy_ai(x);
    let fp = airy_ai_prime(x);
    if fp == 0.0 {
      break;
    }
    let dx = f / fp;
    x -= dx;
    if dx.abs() < 1e-15 * x.abs().max(1.0) {
      break;
    }
  }
  x
}

/// Same as airy_ai_zero_f64 but for AiryBi. Starting point:
/// a_n ≈ -t^(2/3)(1 + ...), t = 3π(4n-3)/8.
fn airy_bi_zero_f64(n: i128) -> f64 {
  let t = 3.0 * std::f64::consts::PI * (4.0 * n as f64 - 3.0) / 8.0;
  let t2 = t * t;
  let correction = 1.0 - 7.0 / (48.0 * t2) + 35.0 / (288.0 * t2 * t2);
  let mut x = -t.powf(2.0 / 3.0) * correction;
  for _ in 0..40 {
    let f = airy_bi(x);
    let fp = airy_bi_prime(x);
    if fp == 0.0 {
      break;
    }
    let dx = f / fp;
    x -= dx;
    if dx.abs() < 1e-15 * x.abs().max(1.0) {
      break;
    }
  }
  x
}

/// AiryAiZero[n] - n-th real zero of AiryAi. Stays symbolic; numeric
/// evaluation is only triggered through N[] (see `airy_ai_zero_n_eval`).
pub fn airy_ai_zero_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryAiZero expects exactly 1 argument".into(),
    ));
  }
  Ok(Expr::FunctionCall {
    name: "AiryAiZero".to_string(),
    args: args.to_vec(),
  })
}

/// AiryBiZero[n] - n-th real zero of AiryBi. Stays symbolic until N[].
pub fn airy_bi_zero_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryBiZero expects exactly 1 argument".into(),
    ));
  }
  Ok(Expr::FunctionCall {
    name: "AiryBiZero".to_string(),
    args: args.to_vec(),
  })
}

/// Numeric evaluation of AiryAiZero[n] for use by N[]. Accepts a
/// positive integer `n` and returns the zero as an `Expr::Real`.
pub fn airy_ai_zero_n_eval(n: i128) -> Option<Expr> {
  if n >= 1 {
    Some(Expr::Real(airy_ai_zero_f64(n)))
  } else {
    None
  }
}

/// Numeric evaluation of AiryBiZero[n] for use by N[].
pub fn airy_bi_zero_n_eval(n: i128) -> Option<Expr> {
  if n >= 1 {
    Some(Expr::Real(airy_bi_zero_f64(n)))
  } else {
    None
  }
}
