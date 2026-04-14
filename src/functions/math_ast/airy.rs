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

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_ai(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryAi".to_string(),
    args: args.to_vec(),
  })
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

  // Numeric evaluation
  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi(x_f)));
  }

  Ok(Expr::FunctionCall {
    name: "AiryBi".to_string(),
    args: args.to_vec(),
  })
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
fn airy_ai_prime(x: f64) -> f64 {
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
fn airy_bi_prime(x: f64) -> f64 {
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
