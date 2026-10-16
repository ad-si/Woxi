#[allow(unused_imports)]
use super::*;

/// ExpIntegralEi[x] - Exponential integral Ei(x)
pub fn exp_integral_ei_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ExpIntegralEi expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    // ExpIntegralEi[0] = -Infinity
    Expr::Integer(0) => Ok(neg(id_expr("Infinity"))),
    // ExpIntegralEi[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => Ok(id_expr("Infinity")),
    // Numeric evaluation
    Expr::Real(x) if *x == 0.0 => Ok(neg(id_expr("Infinity"))),
    Expr::Real(x) => Ok(
      exp_integral_ei_f64(*x)
        .map_or_else(|| unevaluated("ExpIntegralEi", args), Expr::Real),
    ),
    // Check for -Infinity or other cases
    other => {
      if is_neg_infinity(other) {
        return Ok(Expr::Integer(0));
      }
      // Machine-precision complex argument (only when it carries an
      // inexact part; exact complex arguments stay symbolic).
      if crate::syntax::contains_inexact(other)
        && let Some((re, im)) = try_extract_complex_float(other)
        && im != 0.0
      {
        let (r, i) = complex_exp_integral_ei_f64(re, im);
        return Ok(build_complex_float_expr(r, i));
      }
      // Unevaluated
      Ok(unevaluated("ExpIntegralEi", args))
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
    Expr::Integer(0) => Ok(neg(id_expr("Infinity"))),
    // CosIntegral[Infinity] = 0
    Expr::Identifier(s) if s == "Infinity" => Ok(Expr::Integer(0)),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(cos_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // CosIntegral[-Infinity] = I*Pi
        return Ok(times(vec![id_expr("I"), const_expr("Pi")]));
      }
      // Unevaluated
      Ok(unevaluated("CosIntegral", args))
    }
  }
}

/// Compute Ci(z) numerically for real z
/// Ci(z) = γ + ln|z| + ∫₀ᶻ (cos(t)-1)/t dt
/// = γ + ln|z| + Σ_{n=1}^∞ (-1)^n z^(2n) / (2n · (2n)!)
fn cos_integral_numeric(z: f64) -> f64 {
  let euler_gamma = std::f64::consts::EULER_GAMMA;

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

  // FresnelS is odd: FresnelS[-x] = -FresnelS[x]. Fold every negated argument
  // form (including negative rational coefficients, e.g. FresnelS[-x/2]) via the
  // shared helper; recursion evaluates special values such as
  // FresnelS[-Infinity] = -1/2.
  if let Some(pos) = strip_negation(&args[0]) {
    let inner = fresnel_s_ast(&[pos])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }

  // Helper: negate FresnelS (odd function)
  let negate_fresnel_s =
    |inner: Expr| -> Expr { neg1(call1("FresnelS", inner)) };

  match &args[0] {
    // FresnelS[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // FresnelS[Infinity] = 1/2
    Expr::Identifier(s) if s == "Infinity" => Ok(make_rational(1, 2)),
    // FresnelS[ComplexInfinity] = Indeterminate
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      Ok(id_expr("Indeterminate"))
    }
    // FresnelS[I] = -I*FresnelS[1]
    Expr::Identifier(s) if s == "I" => {
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[
          Expr::Integer(-1),
          id_expr("I"),
          call1("FresnelS", Expr::Integer(1)),
        ],
      )
    }
    // FresnelS[-x] = -FresnelS[x] (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      // Check for -Infinity first
      if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") {
        return Ok(make_rational(-1, 2));
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
        let pos_arg = times(vec![Expr::Integer(-*n), fargs[1].clone()]);
        return Ok(negate_fresnel_s(pos_arg));
      }
      // FresnelS[I*z] = -I*FresnelS[z]
      if matches!(&fargs[0], Expr::Identifier(s) if s == "I") {
        return crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[
            Expr::Integer(-1),
            id_expr("I"),
            call1("FresnelS", fargs[1].clone()),
          ],
        );
      }
      Ok(unevaluated("FresnelS", args))
    }
    // FresnelS[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_fresnel_s(Expr::Integer(-*n))),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(fresnel_s_numeric(*x))),
    // Check for -Infinity (Times[-1, Infinity] form)
    other => {
      if is_neg_infinity(other) {
        return Ok(make_rational(-1, 2));
      }
      // Unevaluated
      Ok(unevaluated("FresnelS", args))
    }
  }
}

/// Compute S(x) = ∫₀ˣ sin(π t²/2) dt numerically
/// Uses the identity: S(x) = Im[(1+i)/2 · erf(√π/2 · (1+i) · x)]
pub fn fresnel_s_numeric(x: f64) -> f64 {
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

  // FresnelC is odd: FresnelC[-x] = -FresnelC[x] (see fresnel_s_ast).
  if let Some(pos) = strip_negation(&args[0]) {
    let inner = fresnel_c_ast(&[pos])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }

  // Helper: negate FresnelC (odd function)
  let negate_fresnel_c =
    |inner: Expr| -> Expr { neg1(call1("FresnelC", inner)) };

  match &args[0] {
    // FresnelC[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // FresnelC[Infinity] = 1/2
    Expr::Identifier(s) if s == "Infinity" => Ok(make_rational(1, 2)),
    // FresnelC[ComplexInfinity] = Indeterminate
    Expr::Identifier(s) if s == "ComplexInfinity" => {
      Ok(id_expr("Indeterminate"))
    }
    // FresnelC[I] = I*FresnelC[1]
    Expr::Identifier(s) if s == "I" => {
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[id_expr("I"), call1("FresnelC", Expr::Integer(1))],
      )
    }
    // FresnelC[-x] = -FresnelC[x] (UnaryOp form)
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      // Check for -Infinity first
      if matches!(operand.as_ref(), Expr::Identifier(s) if s == "Infinity") {
        return Ok(make_rational(-1, 2));
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
        let pos_arg = times(vec![Expr::Integer(-*n), fargs[1].clone()]);
        return Ok(negate_fresnel_c(pos_arg));
      }
      // FresnelC[I*z] = I*FresnelC[z]
      if matches!(&fargs[0], Expr::Identifier(s) if s == "I") {
        return crate::evaluator::evaluate_function_call_ast(
          "Times",
          &[id_expr("I"), call1("FresnelC", fargs[1].clone())],
        );
      }
      Ok(unevaluated("FresnelC", args))
    }
    // FresnelC[-n] for negative integer
    Expr::Integer(n) if *n < 0 => Ok(negate_fresnel_c(Expr::Integer(-*n))),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(fresnel_c_numeric(*x))),
    // Check for -Infinity (Times[-1, Infinity] form)
    other => {
      if is_neg_infinity(other) {
        return Ok(make_rational(-1, 2));
      }
      // Unevaluated
      Ok(unevaluated("FresnelC", args))
    }
  }
}

/// Compute C(x) = ∫₀ˣ cos(π t²/2) dt numerically
/// Uses the identity: C(x) = Re[(1+i)/2 · erf(√π/2 · (1+i) · x)]
pub fn fresnel_c_numeric(x: f64) -> f64 {
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
  let c = f64::midpoint(er, ei);
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

/// SinIntegral[z] - Sine integral Si(z)
pub fn sin_integral_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "SinIntegral expects exactly 1 argument".into(),
    ));
  }

  // Si is odd: SinIntegral[-x] = -SinIntegral[x]. Fold every negated
  // argument form via the shared helper, as Erf, Erfi and the Fresnel
  // integrals do; recursion then evaluates special values such as
  // SinIntegral[-Infinity] = -Pi/2. `strip_negation` deliberately leaves
  // machine reals alone — the numeric kernel below is already sign-correct.
  if let Some(pos) = strip_negation(&args[0]) {
    let inner = sin_integral_ast(&[pos])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }

  match &args[0] {
    // SinIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // SinIntegral[Infinity] = Pi/2
    Expr::Identifier(s) if s == "Infinity" => {
      Ok(times(vec![make_rational(1, 2), const_expr("Pi")]))
    }
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(sin_integral_numeric(*x))),
    // Check for -Infinity
    other => {
      if is_neg_infinity(other) {
        // SinIntegral[-Infinity] = -Pi/2
        return Ok(times(vec![make_rational(-1, 2), const_expr("Pi")]));
      }
      // Unevaluated
      Ok(unevaluated("SinIntegral", args))
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
  if is_expr_zero(z_expr) {
    match n_expr {
      Expr::Integer(1) => return Ok(id_expr("ComplexInfinity")),
      Expr::Integer(n) if *n > 1 => {
        // An inexact zero gives an inexact result: E_2(0.) = 1., not 1.
        if matches!(z_expr, Expr::Real(_)) {
          return Ok(Expr::Real(1.0 / (*n - 1) as f64));
        }
        return Ok(make_rational(1, *n - 1));
      }
      // E_ν(0) = 1/(ν − 1) converges only for ν > 1.
      Expr::Real(nu) if *nu > 1.0 => return Ok(Expr::Real(1.0 / (nu - 1.0))),
      Expr::Real(_) => return Ok(id_expr("ComplexInfinity")),
      _ => {}
    }
  }

  // E_0(z) = E^(-z)/z — an exact elementary closed form, matching
  // wolframscript (e.g. ExpIntegralE[0, 1/2] = 2/Sqrt[E]). Symbolic and exact
  // arguments are handled here; a machine-Real argument folds to the same value
  // the numeric path below would give.
  if matches!(n_expr, Expr::Integer(0)) {
    let exp_neg_z =
      call("Exp", vec![times2(Expr::Integer(-1), z_expr.clone())]);
    let result = div2(exp_neg_z, z_expr.clone());
    return crate::evaluator::evaluate_expr_to_expr(&result);
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval
    && let Some((re, im)) = exp_integral_e_f64(n_val.unwrap(), z_val.unwrap())
  {
    return Ok(if im == 0.0 {
      Expr::Real(re)
    } else {
      build_complex_float_expr(re, im)
    });
  }

  Ok(unevaluated("ExpIntegralE", args))
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
    Expr::Integer(1) => Ok(neg(id_expr("Infinity"))),
    // Numeric evaluation: Li(x) = Ei(ln(x))
    Expr::Real(x) if *x == 0.0 => Ok(Expr::Real(0.0)),
    Expr::Real(x) if *x == 1.0 => Ok(neg(id_expr("Infinity"))),
    Expr::Real(x) => Ok(match log_integral_f64(*x) {
      Some((re, 0.0)) => Expr::Real(re),
      Some((re, im)) => build_complex_float_expr(re, im),
      None => unevaluated("LogIntegral", args),
    }),
    // Unevaluated
    _ => Ok(unevaluated("LogIntegral", args)),
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
    _ => Ok(unevaluated("RiemannR", args)),
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

  // Shi is odd, exactly as Si is above.
  if let Some(pos) = strip_negation(&args[0]) {
    let inner = sinh_integral_ast(&[pos])?;
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[Expr::Integer(-1), inner],
    );
  }

  match &args[0] {
    // SinhIntegral[0] = 0
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    // SinhIntegral[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => Ok(id_expr("Infinity")),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(sinh_integral_numeric(*x))),
    other => {
      if is_neg_infinity(other) {
        // SinhIntegral[-Infinity] = -Infinity
        return Ok(neg(id_expr("Infinity")));
      }
      // Unevaluated
      Ok(unevaluated("SinhIntegral", args))
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
    Expr::Integer(0) => Ok(neg(id_expr("Infinity"))),
    // CoshIntegral[Infinity] = Infinity
    Expr::Identifier(s) if s == "Infinity" => Ok(id_expr("Infinity")),
    // Numeric evaluation
    Expr::Real(x) => Ok(Expr::Real(cosh_integral_numeric(*x))),
    other => {
      if is_neg_infinity(other) {
        // CoshIntegral[-Infinity] = Infinity (real part)
        return Ok(id_expr("Infinity"));
      }
      // Unevaluated
      Ok(unevaluated("CoshIntegral", args))
    }
  }
}

/// Compute Chi(z) numerically for real z
/// Chi(z) = γ + ln|z| + ∫₀ᶻ (cosh(t)-1)/t dt
/// = γ + ln|z| + Σ_{n=1}^∞ z^(2n) / (2n · (2n)!)
fn cosh_integral_numeric(z: f64) -> f64 {
  let euler_gamma = std::f64::consts::EULER_GAMMA;

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

/// FresnelF[z] / FresnelG[z] — the auxiliary Fresnel modulus functions
/// (DLMF 7.5 with the Wolfram π/2 normalization):
///   f(z) = (C(z) - 1/2) Sin[π z²/2] - (S(z) - 1/2) Cos[π z²/2]
///   g(z) = -(C(z) - 1/2) Cos[π z²/2] - (S(z) - 1/2) Sin[π z²/2]
/// Exact points: f(0) = g(0) = 1/2 and f(∞) = g(∞) = 0; other exact or
/// symbolic arguments stay unevaluated like wolframscript. Lists thread.
///
/// Machine values compose from the FresnelS/C machine values, so the last
/// digits can differ from wolframscript (which uses dedicated rational
/// approximations), and for large |x| the subtraction cancellation limits
/// relative accuracy to roughly 1e-13 — the known real-path divergence
/// class.
pub fn fresnel_fg_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated(name, args));
  if args.len() != 1 {
    return unevaluated();
  }
  match &args[0] {
    Expr::Integer(0) => {
      return Ok(call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]));
    }
    Expr::Constant(c) | Expr::Identifier(c) if c == "Infinity" => {
      return Ok(Expr::Integer(0));
    }
    Expr::List(items) => {
      let threaded: Result<Vec<Expr>, InterpreterError> = items
        .iter()
        .map(|item| fresnel_fg_ast(name, std::slice::from_ref(item)))
        .collect();
      return Ok(Expr::List(threaded?.into()));
    }
    _ => {}
  }
  if !matches!(&args[0], Expr::Real(_) | Expr::BigFloat(..)) {
    return unevaluated();
  }
  let Some(x) = try_eval_to_f64(&args[0]) else {
    return unevaluated();
  };
  Ok(Expr::Real(fresnel_fg_numeric(name == "FresnelF", x)))
}

/// The machine value of f(x) or g(x) via the FresnelS/C machine values.
pub(crate) fn fresnel_fg_numeric(f_variant: bool, x: f64) -> f64 {
  let s = fresnel_s_numeric(x);
  let c = fresnel_c_numeric(x);
  let phase = std::f64::consts::PI * x * x / 2.0;
  let (sin_p, cos_p) = (phase.sin(), phase.cos());
  if f_variant {
    (c - 0.5) * sin_p - (s - 0.5) * cos_p
  } else {
    -(c - 0.5) * cos_p - (s - 0.5) * sin_p
  }
}

type C64 = (f64, f64);

fn c_mul(a: C64, b: C64) -> C64 {
  (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn c_div(a: C64, b: C64) -> C64 {
  let d = b.0 * b.0 + b.1 * b.1;
  ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

fn c_ln(a: C64) -> C64 {
  (a.0.hypot(a.1).ln(), a.1.atan2(a.0))
}

fn c_exp(a: C64) -> C64 {
  let m = a.0.exp();
  (m * a.1.cos(), m * a.1.sin())
}

/// E1(w) for complex w (not on the negative real axis): power series for
/// small |w|, modified Lentz continued fraction otherwise.
fn complex_e1_f64(w: C64) -> C64 {
  if w.0.hypot(w.1) <= 3.0 {
    let lw = c_ln(w);
    let mut sum = (0.0, 0.0);
    let mut term = (1.0, 0.0);
    let neg_w = (-w.0, -w.1);
    for n in 1..200 {
      let nf = n as f64;
      term = c_mul(term, neg_w);
      term = (term.0 / nf, term.1 / nf);
      let t = (term.0 / nf, term.1 / nf);
      sum = (sum.0 + t.0, sum.1 + t.1);
      if t.0.hypot(t.1) < 1e-17 * sum.0.hypot(sum.1) {
        break;
      }
    }
    (-std::f64::consts::EULER_GAMMA - lw.0 - sum.0, -lw.1 - sum.1)
  } else {
    let tiny = 1e-300;
    let mut b = (w.0 + 1.0, w.1);
    let mut c = (1.0 / tiny, 0.0);
    let mut d = c_div((1.0, 0.0), b);
    let mut h = d;
    for i in 1..1000 {
      let a = -((i * i) as f64);
      b = (b.0 + 2.0, b.1);
      d = c_div((1.0, 0.0), (a * d.0 + b.0, a * d.1 + b.1));
      let ac = c_div((a, 0.0), c);
      c = (b.0 + ac.0, b.1 + ac.1);
      let del = c_mul(c, d);
      h = c_mul(h, del);
      if (del.0 - 1.0).abs() + del.1.abs() < 1e-16 {
        break;
      }
    }
    c_mul(h, c_exp((-w.0, -w.1)))
  }
}

/// Ei(z) = -E1(-z) + (ln z - ln(1/z))/2 - ln(-z) for complex z.
fn complex_exp_integral_ei_f64(re: f64, im: f64) -> C64 {
  let z = (re, im);
  let e1 = complex_e1_f64((-re, -im));
  let lz = c_ln(z);
  let linv = c_ln(c_div((1.0, 0.0), z));
  let lnz = c_ln((-re, -im));
  (
    -e1.0 + 0.5 * (lz.0 - linv.0) - lnz.0,
    -e1.1 + 0.5 * (lz.1 - linv.1) - lnz.1,
  )
}

// ---------------------------------------------------------------------------
// Real exponential integrals: the single numeric core behind ExpIntegralE,
// ExpIntegralEi, LogIntegral and Gamma[a, z], at any precision. Machine
// values are rounded from a 64-bit result, so the series' cancellation (a
// power series for E_ν(z) at z = 20 sums terms near e^20 to a result near
// e^-20) never reaches the f64 answer.
// ---------------------------------------------------------------------------

/// Binary exponent of `b`, or a huge negative number for zero.
fn bf_exp(b: &astro_float::BigFloat) -> i32 {
  if b.is_zero() {
    i32::MIN / 4
  } else {
    b.exponent().unwrap_or(i32::MIN / 4)
  }
}

/// Runs `eval` at `bits` plus a guard, growing the guard until the bits the
/// evaluation reports losing to cancellation leave `bits` + 16 correct ones.
fn with_cancellation_guard<T>(
  bits: usize,
  mut eval: impl FnMut(usize) -> (T, i32),
) -> T {
  let cap = 16 * bits + 4096;
  let mut guard = 64;
  loop {
    let (value, lost) = eval(bits + guard);
    let lost = lost.max(0) as usize;
    if lost + 16 <= guard || guard > cap {
      return value;
    }
    guard = (lost + 64).min(cap + 1);
  }
}

/// Integer value of an integral BigFloat of moderate size.
fn bf_small_integer(b: &astro_float::BigFloat) -> Option<i64> {
  if !b.is_int() {
    return None;
  }
  let f = crate::functions::math_ast::bigfloat_to_f64_approx(b)?;
  (f.abs() < 1.0e9).then_some(f as i64)
}

/// The generalized exponential integral E_ν(z) for real ν and real z ≠ 0,
/// as (real part, imaginary part). A negative z lies on the branch cut,
/// where the value is the limit from above (principal `Log` and powers),
/// matching wolframscript. `None` for z = 0.
pub(crate) fn exp_integral_e_bigfloat(
  nu: &astro_float::BigFloat,
  z: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<(astro_float::BigFloat, astro_float::BigFloat)> {
  use astro_float::BigFloat;
  if z.is_zero() {
    return None;
  }
  let zero = || BigFloat::from_i32(0, bits);
  let n_int = bf_small_integer(nu);

  // E_{-m}(z) = m! e^{-z} z^{-m-1} Σ_{k=0}^{m} z^k/k! is elementary.
  if let Some(n) = n_int
    && n <= 0
  {
    let m = (-n) as usize;
    let re = with_cancellation_guard(bits, |w| {
      let mut term = BigFloat::from_i32(1, w);
      let mut sum = term.clone();
      let mut max_exp = 0;
      for k in 1..=m {
        term = term
          .mul(z, w, rm)
          .div(&BigFloat::from_u64(k as u64, w), w, rm);
        max_exp = max_exp.max(bf_exp(&term));
        sum = sum.add(&term, w, rm);
      }
      let lost = max_exp - bf_exp(&sum);
      let mut fact = BigFloat::from_i32(1, w);
      for k in 2..=m {
        fact = fact.mul(&BigFloat::from_u64(k as u64, w), w, rm);
      }
      let value = fact
        .mul(&z.neg().exp(w, rm, cc), w, rm)
        .mul(&sum, w, rm)
        .div(&bigfloat_powi(z, m as i128 + 1, w, rm), w, rm);
      (value, lost)
    });
    return Some((re.add(&zero(), bits, rm), zero()));
  }

  // Legendre's continued fraction converges fast once z (and z + ν, which
  // keeps the partial denominators positive) is past a precision-dependent
  // threshold; below it the power series is cheaper.
  let threshold = 2.0 + bits as f64 / 32.0;
  let zf = crate::functions::math_ast::bigfloat_to_f64_approx(z)?;
  let nuf = crate::functions::math_ast::bigfloat_to_f64_approx(nu)?;
  if zf > threshold
    && zf + nuf > threshold
    && let Some(v) = exp_integral_e_cf(nu, z, bits + 64, rm, cc)
  {
    return Some((v.add(&zero(), bits, rm), zero()));
  }

  let negative = z.is_negative();
  let (re, im) = with_cancellation_guard(bits, |w| {
    exp_integral_e_series(nu, n_int, z, negative, w, rm, cc)
  });
  Some((re.add(&zero(), bits, rm), im.add(&zero(), bits, rm)))
}

/// E_ν(z), z > 0, by the continued fraction of Numerical Recipes §6.3,
/// e^{-z} (1/(z+ν−) 1·ν/(z+ν+2−) 2(ν+1)/(z+ν+4−) …), modified Lentz.
/// `None` if it fails to converge.
fn exp_integral_e_cf(
  nu: &astro_float::BigFloat,
  z: &astro_float::BigFloat,
  w: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  use astro_float::BigFloat;
  let one = BigFloat::from_i32(1, w);
  let two = BigFloat::from_i32(2, w);
  let nu_m1 = nu.sub(&one, w, rm);
  let mut b = z.add(nu, w, rm);
  let mut c = BigFloat::from_f64(1.0e300, w);
  let mut d = one.div(&b, w, rm);
  let mut h = d.clone();
  for i in 1..(50 * w + 10_000) {
    let i_bf = BigFloat::from_u64(i as u64, w);
    let an = i_bf.mul(&nu_m1.add(&i_bf, w, rm), w, rm).neg();
    b = b.add(&two, w, rm);
    d = one.div(&an.mul(&d, w, rm).add(&b, w, rm), w, rm);
    c = b.add(&an.div(&c, w, rm), w, rm);
    let del = c.mul(&d, w, rm);
    h = h.mul(&del, w, rm);
    if bf_exp(&del.sub(&one, w, rm)) < -(w as i32) + 2 {
      return Some(h.mul(&z.neg().exp(w, rm, cc), w, rm));
    }
  }
  None
}

/// E_ν(z) for ν not a non-positive integer, by its power series
/// (DLMF 8.19.7 for integer ν = n ≥ 1, 8.19.10 otherwise), at working
/// precision `w`. Returns ((re, im), bits lost to cancellation).
fn exp_integral_e_series(
  nu: &astro_float::BigFloat,
  n_int: Option<i64>,
  z: &astro_float::BigFloat,
  negative: bool,
  w: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> ((astro_float::BigFloat, astro_float::BigFloat), i32) {
  use astro_float::BigFloat;
  let one = BigFloat::from_i32(1, w);
  let x = z.abs();
  let ln_x = crate::functions::math_ast::ln_above_integer(&x, w, rm, cc);
  let pi = cc.pi(w, rm);
  let minus_z = z.neg();

  // The non-series part, and the index the series skips (n − 1).
  let (lead_re, lead_im, skip, a) = if let Some(n) = n_int {
    // (−z)^{n−1}/(n−1)! · (ψ(n) − ln z), with ψ(n) = −γ + H_{n−1}.
    let mut coef = one.clone();
    let mut psi =
      crate::functions::math_ast::compute_euler_gamma(w, rm, cc).neg();
    for j in 1..n {
      let j_bf = BigFloat::from_u64(j as u64, w);
      coef = coef.mul(&minus_z, w, rm).div(&j_bf, w, rm);
      psi = psi.add(&one.div(&j_bf, w, rm), w, rm);
    }
    let lead_re = coef.mul(&psi.sub(&ln_x, w, rm), w, rm);
    // ln z = ln|z| + iπ on the cut.
    let lead_im = if negative {
      coef.mul(&pi, w, rm).neg()
    } else {
      BigFloat::from_i32(0, w)
    };
    (lead_re, lead_im, Some(n as usize - 1), None)
  } else {
    // Γ(1 − ν) z^{ν−1}, with z^{ν−1} = |z|^{ν−1} e^{iπ(ν−1)} on the cut.
    let a = one.sub(nu, w, rm);
    let gamma_a = crate::functions::math_ast::gamma_bigfloat(&a, w, rm, cc);
    let lead = gamma_a.mul(&ln_x.mul(&a, w, rm).neg().exp(w, rm, cc), w, rm);
    if negative {
      let phase = pi.mul(&a, w, rm).neg();
      (
        lead.mul(&phase.cos(w, rm, cc), w, rm),
        lead.mul(&phase.sin(w, rm, cc), w, rm),
        None,
        Some(a),
      )
    } else {
      (lead, BigFloat::from_i32(0, w), None, Some(a))
    }
  };

  // − Σ_{k≥0, k≠n−1} (−z)^k / (k! (k + 1 − ν)).
  let one_minus_nu = a.unwrap_or_else(|| one.sub(nu, w, rm));
  let xf =
    crate::functions::math_ast::bigfloat_to_f64_approx(&x).unwrap_or(0.0);
  let mut max_exp = bf_exp(&lead_re);
  let mut re = lead_re;
  let mut term = one.clone(); // (−z)^k / k!
  let mut k: usize = 0;
  loop {
    if skip != Some(k) {
      let denom = BigFloat::from_u64(k as u64, w).add(&one_minus_nu, w, rm);
      let contribution = term.div(&denom, w, rm);
      max_exp = max_exp.max(bf_exp(&contribution));
      re = re.sub(&contribution, w, rm);
      if k as f64 > xf && bf_exp(&contribution) < max_exp - w as i32 - 8 {
        break;
      }
    }
    k += 1;
    term =
      term
        .mul(&minus_z, w, rm)
        .div(&BigFloat::from_u64(k as u64, w), w, rm);
  }
  let lost = max_exp - bf_exp(&re);
  ((re, lead_im), lost)
}

/// The exponential integral Ei(x) for real x ≠ 0: −E₁(−x) for x < 0, the
/// power series γ + ln x + Σ x^k/(k·k!) for moderate x > 0 and the
/// asymptotic series e^x/x Σ k!/x^k once its smallest term is negligible.
pub(crate) fn exp_integral_ei_bigfloat(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  use astro_float::BigFloat;
  if x.is_zero() {
    return None;
  }
  let zero = BigFloat::from_i32(0, bits);
  if x.is_negative() {
    let one = BigFloat::from_i32(1, bits);
    let (re, _) = exp_integral_e_bigfloat(&one, &x.neg(), bits, rm, cc)?;
    return Some(re.neg());
  }
  let xf = crate::functions::math_ast::bigfloat_to_f64_approx(x)?;
  let value = if xf > (bits as f64 + 16.0) * std::f64::consts::LN_2 + 2.0 {
    // The terms shrink until k ≈ x, by then below 2^-(bits+16).
    let w = bits + 64;
    let mut sum = BigFloat::from_i32(1, w);
    let mut term = BigFloat::from_i32(1, w);
    let mut k = 1u64;
    while (k as f64) < xf {
      term = term.mul(&BigFloat::from_u64(k, w), w, rm).div(x, w, rm);
      if bf_exp(&term) < -(w as i32) {
        break;
      }
      sum = sum.add(&term, w, rm);
      k += 1;
    }
    x.exp(w, rm, cc).div(x, w, rm).mul(&sum, w, rm)
  } else {
    with_cancellation_guard(bits, |w| {
      let gamma = crate::functions::math_ast::compute_euler_gamma(w, rm, cc);
      let ln_x = crate::functions::math_ast::ln_above_integer(x, w, rm, cc);
      let mut sum = BigFloat::from_i32(0, w);
      let mut power = BigFloat::from_i32(1, w); // x^k / k!
      for k in 1u64.. {
        let k_bf = BigFloat::from_u64(k, w);
        power = power.mul(x, w, rm).div(&k_bf, w, rm);
        let term = power.div(&k_bf, w, rm);
        sum = sum.add(&term, w, rm);
        if bf_exp(&term) < bf_exp(&sum) - w as i32 - 8 {
          break;
        }
      }
      let max_exp = bf_exp(&gamma).max(bf_exp(&ln_x)).max(bf_exp(&sum));
      let value = gamma.add(&ln_x, w, rm).add(&sum, w, rm);
      let lost = max_exp - bf_exp(&value);
      (value, lost)
    })
  };
  Some(value.add(&zero, bits, rm))
}

/// The logarithmic integral li(x) = Ei(ln x) for real x > 0, x ≠ 1. The
/// logarithm is taken with 128 guard bits, which absorbs Ei's amplification
/// of its error near li's root.
pub(crate) fn log_integral_bigfloat(
  x: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<astro_float::BigFloat> {
  if !x.is_positive() {
    return None;
  }
  let t = x.ln(bits + 128, rm, cc);
  exp_integral_ei_bigfloat(&t, bits, rm, cc)
}

/// Gamma[a, z] = z^a E_{1−a}(z) for real a and real z ≠ 0, as (re, im);
/// a negative z takes the principal power, as wolframscript does.
pub(crate) fn upper_incomplete_gamma_bigfloat(
  a: &astro_float::BigFloat,
  z: &astro_float::BigFloat,
  bits: usize,
  rm: astro_float::RoundingMode,
  cc: &mut astro_float::Consts,
) -> Option<(astro_float::BigFloat, astro_float::BigFloat)> {
  use astro_float::BigFloat;
  let w = bits + 32;
  let one = BigFloat::from_i32(1, w);
  let (e_re, e_im) = exp_integral_e_bigfloat(&one.sub(a, w, rm), z, w, rm, cc)?;
  // z^a = |z|^a (cos πa + i sin πa) for z < 0.
  let modulus =
    crate::functions::math_ast::ln_above_integer(&z.abs(), w, rm, cc)
      .mul(a, w, rm)
      .exp(w, rm, cc);
  let (p_re, p_im) = if z.is_negative() {
    let phase = cc.pi(w, rm).mul(a, w, rm);
    (
      modulus.mul(&phase.cos(w, rm, cc), w, rm),
      modulus.mul(&phase.sin(w, rm, cc), w, rm),
    )
  } else {
    (modulus, BigFloat::from_i32(0, w))
  };
  let re = p_re
    .mul(&e_re, w, rm)
    .sub(&p_im.mul(&e_im, w, rm), bits, rm);
  let im = p_re
    .mul(&e_im, w, rm)
    .add(&p_im.mul(&e_re, w, rm), bits, rm);
  Some((re, im))
}

/// Runs `f` with this thread's cached astro-float constants at 64 bits and
/// rounds the (re, im) result to f64 — the machine-precision entry point.
fn machine_complex(
  f: impl FnOnce(
    astro_float::RoundingMode,
    &mut astro_float::Consts,
  ) -> Option<(astro_float::BigFloat, astro_float::BigFloat)>,
) -> Option<(f64, f64)> {
  crate::functions::math_ast::with_thread_consts(|cc| {
    let (re, im) = f(astro_float::RoundingMode::ToEven, cc)?;
    Some((
      crate::functions::math_ast::bigfloat_to_f64_approx(&re)?,
      crate::functions::math_ast::bigfloat_to_f64_approx(&im)?,
    ))
  })
}

/// Machine E_ν(z) for real ν, z ≠ 0, as (re, im).
pub fn exp_integral_e_f64(nu: f64, z: f64) -> Option<(f64, f64)> {
  use astro_float::BigFloat;
  machine_complex(|rm, cc| {
    let (nu, z) = (BigFloat::from_f64(nu, 64), BigFloat::from_f64(z, 64));
    exp_integral_e_bigfloat(&nu, &z, 64, rm, cc)
  })
}

/// Machine Ei(x) for real x ≠ 0.
pub fn exp_integral_ei_f64(x: f64) -> Option<f64> {
  use astro_float::BigFloat;
  machine_complex(|rm, cc| {
    let v = exp_integral_ei_bigfloat(&BigFloat::from_f64(x, 64), 64, rm, cc)?;
    Some((v, BigFloat::from_i32(0, 64)))
  })
  .map(|(re, _)| re)
}

/// Machine li(x) for real x ≠ 0, 1, as (re, im): a negative x gives
/// Ei(ln|x| + iπ).
pub fn log_integral_f64(x: f64) -> Option<(f64, f64)> {
  use astro_float::BigFloat;
  machine_complex(|rm, cc| {
    let xb = BigFloat::from_f64(x, 64);
    if x > 0.0 {
      let v = log_integral_bigfloat(&xb, 64, rm, cc)?;
      return Some((v, BigFloat::from_i32(0, 64)));
    }
    let t = xb.abs().ln(64 + 128, rm, cc);
    let pi = cc.pi(64 + 128, rm);
    Some(crate::functions::math_ast::complex_exp_integral_ei(
      &t, &pi, 64, rm, cc,
    ))
  })
}

/// Machine Gamma[a, z] for real a and real z ≠ 0, as (re, im).
pub fn upper_incomplete_gamma_f64(a: f64, z: f64) -> Option<(f64, f64)> {
  use astro_float::BigFloat;
  machine_complex(|rm, cc| {
    let (a, z) = (BigFloat::from_f64(a, 64), BigFloat::from_f64(z, 64));
    upper_incomplete_gamma_bigfloat(&a, &z, 64, rm, cc)
  })
}
