#[allow(unused_imports)]
use super::*;

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
  if let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
    && contains_inexact_real(&args[0])
  {
    let (ar, ai) = airy_ai_complex(re, im);
    return Ok(build_complex_float_expr(ar, ai));
  }

  Ok(unevaluated("AiryAi", args))
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

/// Machine Ai(x) (see [`airy_f64`]).
fn airy_ai(x: f64) -> f64 {
  if x == 0.0 {
    return 0.3550280538878173;
  }
  airy_f64(AiryKind::Ai, x).unwrap_or(f64::NAN)
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
  if let Some((re, im)) = try_extract_complex_float(&args[0])
    && im != 0.0
    && contains_inexact_real(&args[0])
  {
    let (br, bi) = airy_bi_complex(re, im);
    return Ok(build_complex_float_expr(br, bi));
  }

  Ok(unevaluated("AiryBi", args))
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

/// Machine Bi(x) (see [`airy_f64`]).
fn airy_bi(x: f64) -> f64 {
  if x == 0.0 {
    return 0.6149266274460007;
  }
  airy_f64(AiryKind::Bi, x).unwrap_or(f64::NAN)
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
    return crate::evaluator::evaluate_expr_to_expr(&neg1(result));
  }

  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_ai_prime(x_f)));
  }

  Ok(unevaluated("AiryAiPrime", args))
}

/// Machine Ai'(x) (see [`airy_f64`]).
fn airy_ai_prime(x: f64) -> f64 {
  if x == 0.0 {
    return -0.2588194037928068;
  }
  airy_f64(AiryKind::AiPrime, x).unwrap_or(f64::NAN)
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
    let power_3 = pow(
      Expr::Integer(3),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(6)]),
    );
    let gamma = call1(
      "Gamma",
      call("Rational", vec![Expr::Integer(1), Expr::Integer(3)]),
    );
    let result = div2(power_3, gamma);
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  if let Some(x_f) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi_prime(x_f)));
  }

  Ok(unevaluated("AiryBiPrime", args))
}

/// Machine Bi'(x) (see [`airy_f64`]).
fn airy_bi_prime(x: f64) -> f64 {
  if x == 0.0 {
    return 0.44828835735382633;
  }
  airy_f64(AiryKind::BiPrime, x).unwrap_or(f64::NAN)
}

/// Build 3^(p/q) / Gamma[r/s] or 3^(p/q) / (3 * Gamma[r/s])
fn airy_build_value(
  power_frac: (i128, i128),
  gamma_frac: (i128, i128),
  with_extra_3: bool,
) -> Result<Expr, InterpreterError> {
  let power_3 = pow(
    Expr::Integer(3),
    call(
      "Rational",
      vec![Expr::Integer(power_frac.0), Expr::Integer(power_frac.1)],
    ),
  );
  let gamma = call1(
    "Gamma",
    call(
      "Rational",
      vec![Expr::Integer(gamma_frac.0), Expr::Integer(gamma_frac.1)],
    ),
  );
  let denom = if with_extra_3 {
    times(vec![Expr::Integer(3), gamma])
  } else {
    gamma
  };
  let result = div2(power_3, denom);
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
  Ok(unevaluated("AiryAiZero", args))
}

/// AiryBiZero[n] - n-th real zero of AiryBi. Stays symbolic until N[].
pub fn airy_bi_zero_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "AiryBiZero expects exactly 1 argument".into(),
    ));
  }
  Ok(unevaluated("AiryBiZero", args))
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

/// Scorer function Hi(x) = (1/Pi) integral_0^inf exp(x t - t^3/3) dt, computed
/// by Simpson's rule on a truncated interval (the t^3 term makes the integrand
/// decay super-exponentially, so a finite upper limit captures the full value).
fn scorer_hi(x: f64) -> f64 {
  let f = |t: f64| (x * t - t * t * t / 3.0).exp();
  // The integrand peaks near t = sqrt(x) (for x > 0) with log-height peak_exp;
  // extend the upper limit until the exponent has dropped ~50 below the peak.
  let peak_exp = if x > 0.0 {
    2.0 / 3.0 * x.powf(1.5)
  } else {
    0.0
  };
  let target = peak_exp - 50.0;
  let mut t_hi = 4.0_f64.max((3.0 * (peak_exp + 50.0)).cbrt());
  while x * t_hi - t_hi * t_hi * t_hi / 3.0 > target {
    t_hi += 1.0;
  }
  let n = 8000usize; // even number of Simpson panels
  let h = t_hi / n as f64;
  let mut sum = f(0.0) + f(t_hi);
  for i in 1..n {
    let t = i as f64 * h;
    sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(t);
  }
  (sum * h / 3.0) / std::f64::consts::PI
}

/// Build the exact value at 0: `numer / (3 * 3^(1/6) * Gamma[2/3])`, the shared
/// closed form of ScorerGi[0] (numer = 1) and ScorerHi[0] (numer = 2).
fn scorer_value_at_zero(numer: i128) -> Result<Expr, InterpreterError> {
  let rational = |a: i128, b: i128| {
    call("Rational", vec![Expr::Integer(a), Expr::Integer(b)])
  };
  let three_sixth = pow(Expr::Integer(3), rational(1, 6));
  let gamma = call1("Gamma", rational(2, 3));
  let denom = times(vec![Expr::Integer(3), three_sixth, gamma]);
  let result = div2(Expr::Integer(numer), denom);
  crate::evaluator::evaluate_expr_to_expr(&result)
}

/// ScorerHi[x] — inhomogeneous Airy function solving y'' - x y = 1/Pi.
pub fn scorer_hi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ScorerHi expects exactly 1 argument".into(),
    ));
  }
  if matches!(&args[0], Expr::Integer(0)) {
    return scorer_value_at_zero(2);
  }
  if let Some(x) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(scorer_hi(x)));
  }
  Ok(unevaluated("ScorerHi", args))
}

/// ScorerGi[x] — inhomogeneous Airy function solving y'' - x y = -1/Pi.
/// Gi(x) = Bi(x) - Hi(x) (DLMF 9.12.3).
pub fn scorer_gi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ScorerGi expects exactly 1 argument".into(),
    ));
  }
  if matches!(&args[0], Expr::Integer(0)) {
    return scorer_value_at_zero(1);
  }
  if let Some(x) = expr_to_f64(&args[0])
    && matches!(&args[0], Expr::Real(_))
  {
    return Ok(Expr::Real(airy_bi(x) - scorer_hi(x)));
  }
  Ok(unevaluated("ScorerGi", args))
}
