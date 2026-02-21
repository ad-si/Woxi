use crate::InterpreterError;
use crate::syntax::{Expr, BinaryOperator, UnaryOperator};
use num_bigint::BigInt;
#[allow(unused_imports)]
use super::*;

/// Pochhammer[a, n] - Rising factorial (Pochhammer symbol): a * (a+1) * ... * (a+n-1)
pub fn pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Pochhammer expects exactly 2 arguments".into(),
    ));
  }
  if let (Some(a), Some(n)) = (expr_to_i128(&args[0]), expr_to_i128(&args[1])) {
    if n == 0 {
      return Ok(Expr::Integer(1));
    }
    if n < 0 {
      // Pochhammer[a, -n] = 1/((a-1)(a-2)...(a-n))
      return Ok(Expr::FunctionCall {
        name: "Pochhammer".to_string(),
        args: args.to_vec(),
      });
    }
    let mut result = BigInt::from(1);
    for i in 0..n {
      result *= BigInt::from(a + i);
    }
    Ok(bigint_to_expr(result))
  } else {
    Ok(Expr::FunctionCall {
      name: "Pochhammer".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Gamma[n] - Gamma function: Gamma[n] = (n-1)! for positive integers
pub fn gamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Gamma expects exactly 1 argument".into(),
    ));
  }
  match expr_to_i128(&args[0]) {
    Some(n) => {
      if n <= 0 {
        // Gamma has poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Gamma[n] = (n-1)! for positive integers
      let mut result = BigInt::from(1);
      for i in 2..n {
        result *= i;
      }
      Ok(bigint_to_expr(result))
    }
    None if matches!(&args[0], Expr::Real(_)) => {
      let f = if let Expr::Real(f) = &args[0] {
        *f
      } else {
        unreachable!()
      };
      if f <= 0.0 && f.fract() == 0.0 {
        // Poles at non-positive integers
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      // Use Stirling's approximation via the standard library's tgamma equivalent
      // Rust doesn't have tgamma in std, but we can compute via the Lanczos approximation
      let result = gamma_fn(f);
      if result.is_infinite() {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else {
        Ok(Expr::Real(result))
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "Gamma".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Lanczos approximation for the Gamma function
pub fn gamma_fn(x: f64) -> f64 {
  if x < 0.5 {
    // Reflection formula: Gamma(1-z) * Gamma(z) = pi / sin(pi*z)
    std::f64::consts::PI
      / ((std::f64::consts::PI * x).sin() * gamma_fn(1.0 - x))
  } else {
    let x = x - 1.0;
    let g = 7.0;
    let c = [
      0.999_999_999_999_809_9,
      676.5203681218851,
      -1259.1392167224028,
      771.323_428_777_653_1,
      -176.615_029_162_140_6,
      12.507343278686905,
      -0.13857109526572012,
      9.984_369_578_019_572e-6,
      1.5056327351493116e-7,
    ];
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
      sum += ci / (x + i as f64);
    }
    let t = x + g + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
  }
}

/// BesselJ[n, z] - Bessel function of the first kind
pub fn bessel_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselJ expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let n_val = match n_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: BesselJ[n, 0] for integer n
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = bessel_j(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "BesselJ".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Bessel J_n(z) using series expansion
pub fn bessel_j(n: f64, z: f64) -> f64 {
  // Handle negative integer orders: J_{-n}(z) = (-1)^n * J_n(z)
  if n < 0.0 && n == n.floor() {
    let n_abs = -n;
    let sign = if n_abs as i64 % 2 == 0 { 1.0 } else { -1.0 };
    return sign * bessel_j(n_abs, z);
  }

  // Special case: z = 0
  if z == 0.0 {
    return if n == 0.0 { 1.0 } else { 0.0 };
  }

  // Series: J_n(z) = sum_{m=0}^{inf} (-1)^m / (m! * Gamma(n+m+1)) * (z/2)^(2m+n)
  let half_z = z / 2.0;
  let mut sum = 0.0;
  let mut term = half_z.powf(n) / gamma_fn(n + 1.0);
  sum += term;

  for m in 1..300 {
    term *= -half_z * half_z / (m as f64 * (n + m as f64));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// BesselI[n, z] - Modified Bessel function of the first kind
pub fn bessel_i_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselI expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselI[n, 0] for integer n: I_0(0) = 1, I_n(0) = 0 for n != 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_i(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselI".to_string(),
    args: args.to_vec(),
  })
}

/// Compute I_n(z) using series: I_n(z) = Σ (z/2)^{2m+n} / (m! * Γ(n+m+1))
pub fn bessel_i(n: f64, z: f64) -> f64 {
  if n < 0.0 && n == n.floor() {
    // I_{-n}(z) = I_n(z) for integer n
    return bessel_i(-n, z);
  }

  if z == 0.0 {
    return if n == 0.0 { 1.0 } else { 0.0 };
  }

  let half_z = z / 2.0;
  let mut sum = 0.0;
  let mut term = half_z.powf(n) / gamma_fn(n + 1.0);
  sum += term;

  for m in 1..300 {
    term *= half_z * half_z / (m as f64 * (n + m as f64));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// BesselK[n, z] - Modified Bessel function of the second kind
pub fn bessel_k_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselK expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselK[n, 0]: K_0(0) = Infinity, K_n(0) = ComplexInfinity for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::Identifier("Infinity".to_string()))
    } else {
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_k(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselK".to_string(),
    args: args.to_vec(),
  })
}

/// Compute K_n(z) for integer n using the Temme method for small z
/// and the recurrence relation for integer orders.
pub fn bessel_k(n: f64, z: f64) -> f64 {
  let n_int = n.round() as i64;
  let n_abs = n_int.unsigned_abs();

  // K_{-n}(z) = K_n(z)
  let n_abs_f = n_abs as f64;

  // Compute K_0 and K_1 directly, then use recurrence for higher orders
  let (k0, k1) = bessel_k01(z);
  if n_abs == 0 {
    return k0;
  }
  if n_abs == 1 {
    return k1;
  }

  // Recurrence: K_{n+1}(z) = (2n/z) * K_n(z) + K_{n-1}(z)
  let mut km1 = k0;
  let mut k = k1;
  for i in 1..n_abs {
    let kp1 = (2.0 * i as f64 / z) * k + km1;
    km1 = k;
    k = kp1;
  }

  // If original n was non-integer, we'd need a different approach
  // but for now we handle integer n via recurrence
  if (n - n_abs_f).abs() > 1e-10 && n >= 0.0 {
    // Non-integer order: use K_v(z) = (π/2) * (I_{-v}(z) - I_v(z)) / sin(vπ)
    let pi = std::f64::consts::PI;
    let i_neg = bessel_i(-n, z);
    let i_pos = bessel_i(n, z);
    return (pi / 2.0) * (i_neg - i_pos) / (n * pi).sin();
  }

  k
}

/// Compute K_0(z) and K_1(z) simultaneously using the Temme series for small z
/// and asymptotic expansion for large z.
pub fn bessel_k01(z: f64) -> (f64, f64) {
  if z <= 2.0 {
    // Series for K_0 and K_1
    let euler_gamma = 0.5772156649015329;
    let lnz2 = (z / 2.0).ln();
    let t = z * z / 4.0;

    // K_0(z) = -(ln(z/2) + γ) * I_0(z) + Σ_{m=1}^∞ (z/2)^{2m} * H_m / (m!)^2
    let i0 = bessel_i(0.0, z);
    let mut k0 = -(lnz2 + euler_gamma) * i0;
    let mut term0 = 1.0;
    let mut hm = 0.0;
    for m in 1..100 {
      let mf = m as f64;
      term0 *= t / (mf * mf);
      hm += 1.0 / mf;
      k0 += term0 * hm;
      if (term0 * hm).abs() < 1e-17 * k0.abs().max(1e-300) {
        break;
      }
    }

    // K_1(z) = (1/z) + (ln(z/2) + γ - 1) * I_1(z)
    //   - (z/2) * Σ_{m=0}^∞ (z²/4)^m * (H_m + H_{m+1}) / (2 * m! * (m+1)!)
    // But the correct form is:
    // K_1(z) = (1/z) - (ln(z/2) + γ) * I_1(z) + (z/4) * Σ_{m=0}^∞ ...
    // Let's use recurrence from K_0 instead: K_1 = -K_0' (for the derivative)
    // Or more reliably, compute via the Wronskian: I_n * K_{n+1} + I_{n+1} * K_n = 1/z
    // So K_1 = (1/z - I_1 * K_0) / I_0
    let i1 = bessel_i(1.0, z);
    let k1 = (1.0 / z - i1 * k0) / i0;

    (k0, k1)
  } else {
    // Asymptotic: K_n(z) ~ sqrt(π/(2z)) * e^{-z} * P_n(z)
    // where P_n(z) = 1 + (4n²-1)/(8z) + (4n²-1)(4n²-9)/(2!(8z)²) + ...
    let pi = std::f64::consts::PI;
    let prefactor = (pi / (2.0 * z)).sqrt() * (-z).exp();

    // K_0 asymptotic (μ = 0)
    let mut sum0 = 1.0;
    let mut term0 = 1.0;
    for m in 1..30 {
      let mf = m as f64;
      let factor = -(0.0 - (2.0 * mf - 1.0).powi(2)) / (8.0 * z * mf);
      term0 *= factor;
      if term0.abs() < 1e-17 {
        break;
      }
      sum0 += term0;
    }

    // K_1 asymptotic (μ = 4)
    let mut sum1 = 1.0;
    let mut term1 = 1.0;
    for m in 1..30 {
      let mf = m as f64;
      let factor = (4.0 - (2.0 * mf - 1.0).powi(2)) / (8.0 * z * mf);
      term1 *= factor;
      if term1.abs() < 1e-17 {
        break;
      }
      sum1 += term1;
    }

    (prefactor * sum0, prefactor * sum1)
  }
}

/// Compute Jacobi elliptic functions sn(u, m), cn(u, m), dn(u, m) numerically
/// using the descending Landen (AGM) transformation.
/// Parameter m is the parameter (not the modulus k; m = k^2).
pub fn jacobi_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
  // Edge cases
  if m.abs() < 1e-16 {
    // m = 0: sn = sin(u), cn = cos(u), dn = 1
    return (u.sin(), u.cos(), 1.0);
  }
  if (m - 1.0).abs() < 1e-16 {
    // m = 1: sn = tanh(u), cn = sech(u), dn = sech(u)
    let s = u.tanh();
    let c = 1.0 / u.cosh();
    return (s, c, c);
  }

  // AGM iteration to compute the sequence of a_n, b_n, c_n
  let mut a = vec![1.0];
  let mut b = vec![(1.0 - m).sqrt()];
  let mut c = vec![m.sqrt()];

  let n_max = 50;
  for _ in 0..n_max {
    let a_prev = *a.last().unwrap();
    let b_prev = *b.last().unwrap();
    a.push((a_prev + b_prev) / 2.0);
    b.push((a_prev * b_prev).sqrt());
    c.push((a_prev - b_prev) / 2.0);

    if c.last().unwrap().abs() < 1e-16 {
      break;
    }
  }

  let n = a.len() - 1;
  // phi_N = 2^N * a_N * u
  let mut phi = (1u64 << n as u64) as f64 * a[n] * u;

  // Descend: phi_{n-1} from phi_n
  for i in (1..=n).rev() {
    phi = (phi + (c[i] / a[i] * phi.sin()).asin()) / 2.0;
  }

  let sn = phi.sin();
  let cn = phi.cos();
  let dn = (1.0 - m * sn * sn).sqrt();

  (sn, cn, dn)
}

/// Hypergeometric0F1[a, z] - confluent hypergeometric limit function
/// 0F1(a; z) = Σ z^k / (k! * Pochhammer(a,k)) for k = 0, 1, 2, ...
pub fn hypergeometric_0f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric0F1 expects exactly 2 arguments".into(),
    ));
  }

  let a_expr = &args[0];
  let z_expr = &args[1];

  // Hypergeometric0F1[a, 0] = 1
  if is_expr_zero(z_expr) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  let a_val = try_eval_to_f64(a_expr);
  let z_val = try_eval_to_f64(z_expr);

  if let (Some(a), Some(z)) = (a_val, z_val)
    && (matches!(a_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_)))
  {
    return Ok(Expr::Real(hypergeometric_0f1_f64(a, z)));
  }

  Ok(Expr::FunctionCall {
    name: "Hypergeometric0F1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute 0F1(a; z) numerically via series expansion
pub fn hypergeometric_0f1_f64(a: f64, z: f64) -> f64 {
  let mut sum = 1.0;
  let mut term = 1.0;
  for k in 0..200 {
    let kf = k as f64;
    term *= z / ((kf + 1.0) * (a + kf));
    sum += term;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }
  sum
}

/// JacobiAmplitude[u, m] - amplitude for Jacobi elliptic functions
/// am(u, m) = arcsin(sn(u, m)), inverse of EllipticF
pub fn jacobi_amplitude_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiAmplitude expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiAmplitude[0, m] = 0
  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (try_eval_to_f64(u), try_eval_to_f64(m))
    && (matches!(u, Expr::Real(_)) || matches!(m, Expr::Real(_)))
  {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn.atan2(cn)));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiAmplitude".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDN[u, m] - Jacobi elliptic function dn
pub fn jacobi_dn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiDN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 0] = 1
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }

  // JacobiDN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiDN[-u, m] = JacobiDN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiDN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiDN".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiSN[u, m] - Jacobi elliptic function sn
pub fn jacobi_sn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiSN[0, m] = 0
  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }

  // JacobiSN[u, 0] = Sin[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[u, 1] = Tanh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Tanh".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiSN[-u, m] = -JacobiSN[u, m] (odd function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(Expr::FunctionCall {
        name: "JacobiSN".to_string(),
        args: vec![inner, m.clone()],
      }),
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiSN".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCN[u, m] - Jacobi elliptic function cn
pub fn jacobi_cn_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCN expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let m = &args[1];

  // JacobiCN[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }

  // JacobiCN[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[u, 1] = Sech[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sech".to_string(),
      args: vec![u.clone()],
    });
  }

  // JacobiCN[-u, m] = JacobiCN[u, m] (even function)
  if let Some(inner) = extract_negated_expr(u) {
    return Ok(Expr::FunctionCall {
      name: "JacobiCN".to_string(),
      args: vec![inner, m.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "JacobiCN".to_string(),
    args: args.to_vec(),
  })
}

/// Extract the inner expression from a negated expression like Times[-1, x] or -x
pub fn extract_negated_expr(expr: &Expr) -> Option<Expr> {
  match expr {
    // Times[-1, x] form (FunctionCall)
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(args[0], Expr::Integer(-1)) =>
    {
      Some(args[1].clone())
    }
    // BinaryOp Times[-1, x] form
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(left.as_ref(), Expr::Integer(-1)) {
        Some(right.as_ref().clone())
      } else if matches!(right.as_ref(), Expr::Integer(-1)) {
        Some(left.as_ref().clone())
      } else {
        None
      }
    }
    // UnaryOp Minus form
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some(operand.as_ref().clone()),
    _ => None,
  }
}

/// JacobiSC[u, m] - Jacobi SC elliptic function (SN/CN)
pub fn jacobi_sc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiSC[0, m] = 0
  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }
  // JacobiSC[u, 0] = Tan[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Tan".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiSC[u, 1] = Sinh[u]
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSC".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDC[u, m] - Jacobi DC elliptic function (DN/CN)
pub fn jacobi_dc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiDC[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }
  // JacobiDC[u, 0] = Sec[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiDC[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDC".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCD[u, m] - Jacobi CD elliptic function (CN/DN)
pub fn jacobi_cd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  // JacobiCD[0, m] = 1
  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }
  // JacobiCD[u, 0] = Cos[u]
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cos".to_string(),
      args: vec![u.clone()],
    });
  }
  // JacobiCD[u, 1] = 1
  if is_expr_one(m) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCD".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiSD[u, m] - Jacobi SD elliptic function (SN/DN)
pub fn jacobi_sd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiSD expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Integer(0));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sin".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Sinh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(sn / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiSD".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiCS[u, m] - Jacobi CS elliptic function (CN/SN)
pub fn jacobi_cs_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiCS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Cot".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(cn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiCS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiDS[u, m] - Jacobi DS elliptic function (DN/SN)
pub fn jacobi_ds_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiDS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Csch".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(dn / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiDS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiNS[u, m] - Jacobi NS elliptic function (1/SN)
pub fn jacobi_ns_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNS expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Csc".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Coth".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (sn, _, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / sn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNS".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiND[u, m] - Jacobi ND elliptic function (1/DN)
pub fn jacobi_nd_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiND expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(Expr::Integer(1));
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, _, dn) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / dn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiND".to_string(),
    args: args.to_vec(),
  })
}

/// JacobiNC[u, m] - Jacobi NC elliptic function (1/CN)
pub fn jacobi_nc_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "JacobiNC expects exactly 2 arguments".into(),
    ));
  }
  let u = &args[0];
  let m = &args[1];

  if is_expr_zero(u) {
    return Ok(Expr::Integer(1));
  }
  if is_expr_zero(m) {
    return Ok(Expr::FunctionCall {
      name: "Sec".to_string(),
      args: vec![u.clone()],
    });
  }
  if is_expr_one(m) {
    return Ok(Expr::FunctionCall {
      name: "Cosh".to_string(),
      args: vec![u.clone()],
    });
  }

  if let (Some(u_f), Some(m_f)) = (expr_to_f64(u), expr_to_f64(m)) {
    let (_, cn, _) = jacobi_elliptic(u_f, m_f);
    return Ok(Expr::Real(1.0 / cn));
  }

  Ok(Expr::FunctionCall {
    name: "JacobiNC".to_string(),
    args: args.to_vec(),
  })
}

/// Check if an expression is numerically zero
pub fn is_expr_zero(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(0) => true,
    Expr::Real(f) => *f == 0.0,
    _ => false,
  }
}

/// Check if an expression is numerically one
pub fn is_expr_one(expr: &Expr) -> bool {
  match expr {
    Expr::Integer(1) => true,
    Expr::Real(f) => *f == 1.0,
    _ => false,
  }
}

/// Check if an expression represents -Infinity
pub fn is_neg_infinity(expr: &Expr) -> bool {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && let (Expr::Integer(-1), Expr::Identifier(s)) = (&args[0], &args[1])
  {
    return s == "Infinity";
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Times,
    left,
    right,
  } = expr
    && let (Expr::Integer(-1), Expr::Identifier(s)) =
      (left.as_ref(), right.as_ref())
  {
    return s == "Infinity";
  }
  if let Expr::UnaryOp {
    op: UnaryOperator::Minus,
    operand,
  } = expr
    && let Expr::Identifier(s) = operand.as_ref()
  {
    return s == "Infinity";
  }
  false
}

/// EllipticK[m] - Complete elliptic integral of the first kind
pub fn elliptic_k_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EllipticK expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(0) => {
      // EllipticK[0] = Pi/2
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      })
    }
    Expr::Integer(1) => {
      // EllipticK[1] = ComplexInfinity (pole)
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    }
    Expr::Real(f) => {
      if *f == 1.0 {
        Ok(Expr::Identifier("ComplexInfinity".to_string()))
      } else if *f < 1.0 {
        // Compute via arithmetic-geometric mean: K(m) = pi / (2 * AGM(1, sqrt(1 - m)))
        Ok(Expr::Real(elliptic_k(*f)))
      } else {
        // m > 1 requires complex numbers, return unevaluated
        Ok(Expr::FunctionCall {
          name: "EllipticK".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticK".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute complete elliptic integral K(m) using the arithmetic-geometric mean
pub fn elliptic_k(m: f64) -> f64 {
  let mut a = 1.0;
  let mut b = (1.0 - m).sqrt();
  for _ in 0..100 {
    let a_new = (a + b) / 2.0;
    let b_new = (a * b).sqrt();
    if (a_new - b_new).abs() < 1e-16 {
      return std::f64::consts::PI / (2.0 * a_new);
    }
    a = a_new;
    b = b_new;
  }
  std::f64::consts::PI / (2.0 * a)
}

/// EllipticE[m] - Complete elliptic integral of the second kind
pub fn elliptic_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "EllipticE expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(0) => {
      // EllipticE[0] = Pi/2
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(2)),
      })
    }
    Expr::Integer(1) => {
      // EllipticE[1] = 1
      Ok(Expr::Integer(1))
    }
    Expr::Real(f) => {
      if *f == 0.0 {
        Ok(Expr::Real(std::f64::consts::FRAC_PI_2))
      } else if *f == 1.0 {
        Ok(Expr::Real(1.0))
      } else if *f < 1.0 {
        Ok(Expr::Real(elliptic_e(*f)))
      } else {
        Ok(Expr::FunctionCall {
          name: "EllipticE".to_string(),
          args: args.to_vec(),
        })
      }
    }
    _ => Ok(Expr::FunctionCall {
      name: "EllipticE".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute complete elliptic integral E(m) via series expansion
/// E(m) = (pi/2) * [1 - sum_{n=1}^inf ((2n-1)!!/(2n)!!)^2 * m^n / (2n-1)]
pub fn elliptic_e(m: f64) -> f64 {
  // E(m) = (pi/2) * Σ_{n=0}^∞ ((2n)!/(2^n n!))^2 * (-m^n)/((2n-1)*4^n)
  // Simpler: E(m) = (pi/2) * [1 - Σ_{n=1}^∞ (1/2 choose n)^2 * m^n / (2n-1)]
  // Using the series: E(m) = pi/2 * Σ t_n where
  // t_0 = 1, t_n = t_{n-1} * ((2n-1)/(2n))^2 * m * (2n-1)/(2n+1-2) ... hmm

  // Better: use the standard power series
  // E(m) = pi/2 * [1 - (1/2)^2 * m/1 - (1*3)^2/(2*4)^2 * m^2/3 - (1*3*5)^2/(2*4*6)^2 * m^3/5 - ...]
  // Term_n = ((2n-1)!!/(2n)!!)^2 * m^n / (2n-1) for n >= 1
  let mut sum = 1.0;
  let mut coeff = 1.0; // ((2n-1)!!/(2n)!!)^2

  for n in 1..500 {
    let nf = n as f64;
    // coeff_n = coeff_{n-1} * ((2n-1)/(2n))^2
    let ratio = (2.0 * nf - 1.0) / (2.0 * nf);
    coeff *= ratio * ratio;
    let term = coeff * m.powi(n) / (2.0 * nf - 1.0);
    sum -= term;
    if term.abs() < 1e-16 * sum.abs() {
      break;
    }
  }

  std::f64::consts::FRAC_PI_2 * sum
}

/// EllipticF[phi, m] - Incomplete elliptic integral of the first kind
/// F(phi, m) = integral from 0 to phi of 1/sqrt(1 - m*sin^2(theta)) dtheta
pub fn elliptic_f_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "EllipticF expects exactly 2 arguments".into(),
    ));
  }

  let phi_expr = &args[0];
  let m_expr = &args[1];

  // EllipticF[0, m] = 0
  if is_expr_zero(phi_expr) {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation
  let phi_val = expr_to_f64(phi_expr);
  let m_val = expr_to_f64(m_expr);
  let is_numeric_eval = phi_val.is_some()
    && m_val.is_some()
    && (matches!(phi_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

  if is_numeric_eval {
    let phi = phi_val.unwrap();
    let m = m_val.unwrap();
    return Ok(Expr::Real(elliptic_f(phi, m)));
  }

  Ok(Expr::FunctionCall {
    name: "EllipticF".to_string(),
    args: args.to_vec(),
  })
}

/// Compute incomplete elliptic integral F(phi, m) via Gauss-Legendre quadrature
pub fn elliptic_f(phi: f64, m: f64) -> f64 {
  if phi == 0.0 {
    return 0.0;
  }
  // For phi = pi/2, this is the complete integral K(m)
  // Use numerical integration with adaptive Simpson's rule
  let n = 1000;
  let h = phi / n as f64;
  // Simpson's 1/3 rule
  let f = |theta: f64| 1.0 / (1.0 - m * theta.sin().powi(2)).sqrt();
  let mut sum = f(0.0) + f(phi);
  for i in 1..n {
    let theta = i as f64 * h;
    if i % 2 == 0 {
      sum += 2.0 * f(theta);
    } else {
      sum += 4.0 * f(theta);
    }
  }
  sum * h / 3.0
}

/// EllipticPi[n, m] - Complete elliptic integral of the third kind
/// EllipticPi[n, phi, m] - Incomplete elliptic integral of the third kind
/// Pi(n|m) = integral from 0 to pi/2 of 1/((1-n*sin^2(theta))*sqrt(1-m*sin^2(theta))) dtheta
pub fn elliptic_pi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 && args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "EllipticPi expects 2 or 3 arguments".into(),
    ));
  }

  if args.len() == 2 {
    // Complete: EllipticPi[n, m]
    let n_expr = &args[0];
    let m_expr = &args[1];

    // EllipticPi[0, m] = EllipticK[m]
    if is_expr_zero(n_expr) {
      return elliptic_k_ast(&[m_expr.clone()]);
    }

    let n_val = try_eval_to_f64(n_expr);
    let m_val = try_eval_to_f64(m_expr);
    let is_numeric = n_val.is_some()
      && m_val.is_some()
      && (matches!(n_expr, Expr::Real(_)) || matches!(m_expr, Expr::Real(_)));

    if is_numeric {
      let n = n_val.unwrap();
      let m = m_val.unwrap();
      return Ok(Expr::Real(elliptic_pi_f64(
        n,
        std::f64::consts::FRAC_PI_2,
        m,
      )));
    }
  } else {
    // Incomplete: EllipticPi[n, phi, m]
    let n_expr = &args[0];
    let phi_expr = &args[1];
    let m_expr = &args[2];

    // EllipticPi[n, 0, m] = 0
    if is_expr_zero(phi_expr) {
      return Ok(Expr::Integer(0));
    }

    let n_val = try_eval_to_f64(n_expr);
    let phi_val = try_eval_to_f64(phi_expr);
    let m_val = try_eval_to_f64(m_expr);
    let is_numeric = n_val.is_some()
      && phi_val.is_some()
      && m_val.is_some()
      && (matches!(n_expr, Expr::Real(_))
        || matches!(phi_expr, Expr::Real(_))
        || matches!(m_expr, Expr::Real(_)));

    if is_numeric {
      let n = n_val.unwrap();
      let phi = phi_val.unwrap();
      let m = m_val.unwrap();
      return Ok(Expr::Real(elliptic_pi_f64(n, phi, m)));
    }
  }

  Ok(Expr::FunctionCall {
    name: "EllipticPi".to_string(),
    args: args.to_vec(),
  })
}

/// Compute incomplete elliptic integral of the third kind via Simpson's rule
pub fn elliptic_pi_f64(n: f64, phi: f64, m: f64) -> f64 {
  if phi == 0.0 {
    return 0.0;
  }
  let num_steps = 1000;
  let h = phi / num_steps as f64;
  let f = |theta: f64| {
    let sin2 = theta.sin().powi(2);
    1.0 / ((1.0 - n * sin2) * (1.0 - m * sin2).sqrt())
  };
  let mut sum = f(0.0) + f(phi);
  for i in 1..num_steps {
    let theta = i as f64 * h;
    if i % 2 == 0 {
      sum += 2.0 * f(theta);
    } else {
      sum += 4.0 * f(theta);
    }
  }
  sum * h / 3.0
}

/// BesselY[n, z] - Bessel function of the second kind
pub fn bessel_y_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselY expects exactly 2 arguments".into(),
    ));
  }
  let n_expr = &args[0];
  let z_expr = &args[1];

  // BesselY[n, 0]: Y_0(0) = -Infinity, Y_n(0) = ComplexInfinity for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
  {
    return if *n == 0 {
      Ok(Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![Expr::Integer(-1), Expr::Identifier("Infinity".to_string())],
      })
    } else {
      Ok(Expr::Identifier("ComplexInfinity".to_string()))
    };
  }

  // Numeric evaluation
  let n_val = expr_to_f64(n_expr);
  let z_val = expr_to_f64(z_expr);
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let result = bessel_y(n_val.unwrap(), z_val.unwrap());
    return Ok(Expr::Real(result));
  }

  Ok(Expr::FunctionCall {
    name: "BesselY".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Bessel Y_n(z) (second kind)
pub fn bessel_y(n: f64, z: f64) -> f64 {
  // Handle negative integer orders: Y_{-n}(z) = (-1)^n * Y_n(z)
  if n < 0.0 && n == n.floor() {
    let n_abs = -n;
    let sign = if n_abs as i64 % 2 == 0 { 1.0 } else { -1.0 };
    return sign * bessel_y(n_abs, z);
  }

  if n == n.floor() && n >= 0.0 {
    // Integer order: use Y_0, Y_1 from series, then recurrence
    let n_int = n as i64;
    let y0 = bessel_y0(z);
    if n_int == 0 {
      return y0;
    }
    let y1 = bessel_y1(z);
    if n_int == 1 {
      return y1;
    }
    // Recurrence: Y_{n+1}(z) = (2n/z) * Y_n(z) - Y_{n-1}(z)
    let mut y_prev = y0;
    let mut y_curr = y1;
    for k in 1..n_int {
      let y_next = (2.0 * k as f64 / z) * y_curr - y_prev;
      y_prev = y_curr;
      y_curr = y_next;
    }
    y_curr
  } else {
    // Non-integer order: Y_n(z) = (J_n(z)*cos(nπ) - J_{-n}(z)) / sin(nπ)
    let j_n = bessel_j(n, z);
    let j_neg_n = bessel_j(-n, z);
    let n_pi = n * std::f64::consts::PI;
    (j_n * n_pi.cos() - j_neg_n) / n_pi.sin()
  }
}

/// Y_0(z) = (2/π) * (J_0(z) * (ln(z/2) + γ) + Σ (-1)^{m+1} H_m * (z/2)^{2m} / (m!)^2)
pub fn bessel_y0(z: f64) -> f64 {
  let two_over_pi = 2.0 / std::f64::consts::PI;
  let euler_gamma = 0.5772156649015329;
  let half_z = z / 2.0;

  let j0 = bessel_j(0.0, z);

  // Series part: Σ_{m=1}^{∞} (-1)^{m+1} * H_m * (z/2)^{2m} / (m!)^2
  let mut series_sum = 0.0;
  let mut term = 1.0; // (z/2)^{2m} / (m!)^2 starting value
  let mut h_m = 0.0; // harmonic number H_m

  for m in 1..300 {
    let mf = m as f64;
    h_m += 1.0 / mf;
    term *= half_z * half_z / (mf * mf);
    let sign = if m % 2 == 0 { -1.0 } else { 1.0 };
    series_sum += sign * h_m * term;
    if (h_m * term).abs() < 1e-17 * series_sum.abs().max(1e-300) {
      break;
    }
  }

  two_over_pi * (j0 * (half_z.ln() + euler_gamma) + series_sum)
}

/// Y_1(z) = (2/π) * (J_1(z) * ln(z/2) - 1/z + Σ ...)
pub fn bessel_y1(z: f64) -> f64 {
  let two_over_pi = 2.0 / std::f64::consts::PI;
  let euler_gamma = 0.5772156649015329;
  let half_z = z / 2.0;

  let j1 = bessel_j(1.0, z);

  // Y_1(z) = (2/π) * (J_1(z)*(ln(z/2) + γ) - 1/z - (1/2) Σ_{m=0}^∞ (-1)^m (H_m + H_{m+1}) (z/2)^{2m+1} / (m!(m+1)!))
  let mut series_sum = 0.0;
  let mut term = half_z; // (z/2)^{2m+1} / (m! * (m+1)!) starting with m=0: (z/2)^1 / (0! * 1!) = z/2
  let mut h_m = 0.0; // H_0 = 0
  let mut h_m1 = 1.0; // H_1 = 1

  series_sum += (h_m + h_m1) * term; // m=0 term (sign = +1 for (-1)^0)

  for m in 1..300 {
    let mf = m as f64;
    h_m += 1.0 / mf;
    h_m1 += 1.0 / (mf + 1.0);
    term *= -half_z * half_z / (mf * (mf + 1.0));
    series_sum += (h_m + h_m1) * term;
    if ((h_m + h_m1) * term).abs() < 1e-17 * series_sum.abs().max(1e-300) {
      break;
    }
  }

  two_over_pi
    * (j1 * ((half_z).ln() + euler_gamma) - 1.0 / z - 0.5 * series_sum)
}

/// EllipticTheta[a, z, q] - Jacobi theta function
/// a = 1,2,3,4 selects which theta function
pub fn elliptic_theta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "EllipticTheta expects exactly 3 arguments".into(),
    ));
  }

  let a_val = match &args[0] {
    Expr::Integer(n) if *n >= 1 && *n <= 4 => *n as u32,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "EllipticTheta".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let q = &args[2];

  // EllipticTheta[a, z, 0]: theta1=0, theta2=0, theta3=1, theta4=1
  if is_expr_zero(q) {
    return match a_val {
      1 | 2 => Ok(Expr::Integer(0)),
      3 | 4 => Ok(Expr::Integer(1)),
      _ => unreachable!(),
    };
  }

  // Numeric evaluation when both z and q are numeric
  let z = &args[1];
  if let (Some(z_f), Some(q_f)) = (expr_to_f64(z), expr_to_f64(q)) {
    return Ok(Expr::Real(elliptic_theta_numeric(a_val, z_f, q_f)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "EllipticTheta".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Jacobi theta function numerically via series expansion
pub fn elliptic_theta_numeric(a: u32, z: f64, q: f64) -> f64 {
  match a {
    1 => {
      // θ₁(z, q) = 2 Σ_{n=0}^∞ (-1)^n q^{(n+1/2)²} sin((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let term = q.powf(exp) * ((2.0 * nf + 1.0) * z).sin();
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      2.0 * sum
    }
    2 => {
      // θ₂(z, q) = 2 Σ_{n=0}^∞ q^{(n+1/2)²} cos((2n+1)z)
      let mut sum = 0.0;
      for n in 0..100 {
        let nf = n as f64;
        let exp = (nf + 0.5) * (nf + 0.5);
        let term = q.powf(exp) * ((2.0 * nf + 1.0) * z).cos();
        sum += term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      2.0 * sum
    }
    3 => {
      // θ₃(z, q) = 1 + 2 Σ_{n=1}^∞ q^{n²} cos(2nz)
      let mut sum = 1.0;
      for n in 1..100 {
        let nf = n as f64;
        let term = q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += 2.0 * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      sum
    }
    4 => {
      // θ₄(z, q) = 1 + 2 Σ_{n=1}^∞ (-1)^n q^{n²} cos(2nz)
      let mut sum = 1.0;
      for n in 1..100 {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += 2.0 * sign * term;
        if term.abs() < 1e-16 {
          break;
        }
      }
      sum
    }
    _ => unreachable!(),
  }
}

/// WeierstrassP[u, {g2, g3}] - Weierstrass elliptic function ℘(u; g₂, g₃)
pub fn weierstrass_p_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassP expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole at origin)
  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(g2_f), Some(g3_f)) =
    (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeierstrassP".to_string(),
    args: args.to_vec(),
  })
}

/// Compute WeierstrassP numerically using cubic roots + Jacobi elliptic functions
pub fn weierstrass_p_numeric(u: f64, g2: f64, g3: f64) -> f64 {
  // Solve depressed cubic: t³ - (g2/4)t - (g3/4) = 0
  let p = -g2 / 4.0;
  let q = -g3 / 4.0;

  // Cubic discriminant: Δ = -4p³ - 27q²
  let delta = -4.0 * p * p * p - 27.0 * q * q;

  if delta >= 0.0 && p < -1e-16 {
    // Three real roots — use trigonometric method
    let mp3 = -p / 3.0; // > 0
    let r = mp3.sqrt();
    let cos_arg = (-q / (2.0 * mp3 * r)).clamp(-1.0, 1.0);
    let alpha = cos_arg.acos();

    let pi = std::f64::consts::PI;
    let mut roots = [
      2.0 * r * (alpha / 3.0).cos(),
      2.0 * r * ((alpha + 2.0 * pi) / 3.0).cos(),
      2.0 * r * ((alpha + 4.0 * pi) / 3.0).cos(),
    ];
    roots.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let e1 = roots[0];
    let e2 = roots[1];
    let e3 = roots[2];

    let denom = e1 - e3;
    if denom.abs() < 1e-300 {
      return weierstrass_p_laurent(u, g2, g3);
    }

    let m = (e2 - e3) / denom;
    let z = denom.sqrt() * u;
    let (sn, _, _) = jacobi_elliptic(z, m);

    if sn.abs() < 1e-300 {
      return f64::INFINITY;
    }
    e3 + denom / (sn * sn)
  } else {
    // One real root or degenerate — use Laurent series
    weierstrass_p_laurent(u, g2, g3)
  }
}

/// Laurent series fallback: ℘(u) = 1/u² + Σ cₖ u^{2k}
pub fn weierstrass_p_laurent(u: f64, g2: f64, g3: f64) -> f64 {
  let max_terms = 30;
  let mut c = vec![0.0; max_terms];
  if max_terms > 1 {
    c[1] = g2 / 20.0;
  }
  if max_terms > 2 {
    c[2] = g3 / 28.0;
  }
  for k in 3..max_terms {
    let mut sum = 0.0;
    for j in 1..=(k - 2) {
      sum += c[j] * c[k - 1 - j];
    }
    c[k] = 3.0 / ((2 * k + 3) as f64 * (k - 1) as f64) * sum;
  }

  let u2 = u * u;
  let mut result = 1.0 / u2;
  let mut u_power = 1.0;
  for k in 1..max_terms {
    u_power *= u2;
    result += c[k] * u_power;
  }
  result
}

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

/// Zeta[s] - Riemann zeta function
pub fn zeta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "Zeta expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n == 1 {
        // Zeta[1] = ComplexInfinity (pole)
        return Ok(Expr::Identifier("ComplexInfinity".to_string()));
      }
      if n == 0 {
        // Zeta[0] = -1/2
        return Ok(make_rational(-1, 2));
      }
      if n > 0 && n % 2 == 0 {
        // Positive even integer: Zeta[2n] = |B_{2n}| * 2^{2n-1} * Pi^{2n} / (2n)!
        if let Some(expr) = zeta_positive_even(n as usize) {
          return Ok(expr);
        }
      }
      if n < 0 {
        let abs_n = (-n) as usize;
        if abs_n.is_multiple_of(2) {
          // Negative even integer: Zeta[-2k] = 0 (trivial zeros)
          return Ok(Expr::Integer(0));
        }
        // Negative odd integer: Zeta[-n] = (-1)^n * B_{n+1} / (n+1)
        if let Some(expr) = zeta_negative_odd(abs_n) {
          return Ok(expr);
        }
      }
      // Positive odd integer >= 3 or overflow: return unevaluated
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec(),
      })
    }
    Expr::Real(f) => {
      // Numeric evaluation
      let result = zeta_numeric(*f);
      Ok(Expr::Real(result))
    }
    _ => {
      // Symbolic argument: return unevaluated
      Ok(Expr::FunctionCall {
        name: "Zeta".to_string(),
        args: args.to_vec(),
      })
    }
  }
}

/// Compute Bernoulli number B_n as (numerator, denominator).
/// Returns None if overflow occurs during computation.
pub fn bernoulli_number(n: usize) -> Option<(i128, i128)> {
  if n == 0 {
    return Some((1, 1));
  }
  if n == 1 {
    return Some((-1, 2));
  }
  if n % 2 == 1 {
    return Some((0, 1));
  }

  // Compute all even Bernoulli numbers up to B_n using the recurrence:
  // B_m = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B_k
  let mut b: Vec<(i128, i128)> = vec![(0, 1); n + 1];
  b[0] = (1, 1);
  b[1] = (-1, 2);

  for m in (2..=n).step_by(2) {
    let mut sum_n: i128 = 0;
    let mut sum_d: i128 = 1;
    let mut binom: i128 = 1; // C(m+1, 0)

    for k in 0..m {
      if k > 0 {
        // C(m+1, k) = C(m+1, k-1) * (m+2-k) / k
        binom = binom.checked_mul((m + 2 - k) as i128)? / (k as i128);
      }
      let (bk_n, bk_d) = b[k];
      if bk_n == 0 {
        continue;
      }

      // Add binom * B_k to sum: sum_n/sum_d + binom*bk_n/bk_d
      let term_n = binom.checked_mul(bk_n)?;
      let term_d = bk_d;

      let new_n = sum_n
        .checked_mul(term_d)?
        .checked_add(term_n.checked_mul(sum_d)?)?;
      let new_d = sum_d.checked_mul(term_d)?;
      let g = gcd(new_n.abs(), new_d.abs());
      sum_n = new_n / g;
      sum_d = new_d / g;
    }

    // B_m = -sum / (m+1)
    let bm_n = -sum_n;
    let bm_d = sum_d.checked_mul((m + 1) as i128)?;
    let g = gcd(bm_n.abs(), bm_d.abs());
    b[m] = (bm_n / g, bm_d / g);
  }

  Some(b[n])
}

/// Compute Zeta[2n] for positive even integer 2n.
/// Returns the exact expression |B_{2n}| * 2^(2n-1) * Pi^(2n) / (2n)!
pub fn zeta_positive_even(two_n: usize) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(two_n)?;
  if b_num == 0 {
    return None;
  }

  // Compute coefficient: |B_{2n}| * 2^(2n-1) / (2n)!
  let mut num = b_num.abs();
  let mut den = b_den.abs();

  // Multiply by 2^(2n-1)
  for _ in 0..(two_n - 1) {
    num = num.checked_mul(2)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Divide by (2n)!
  for k in 1..=two_n {
    den = den.checked_mul(k as i128)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Build: num * Pi^(2n) / den
  let pi_power = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Identifier("Pi".to_string())),
    right: Box::new(Expr::Integer(two_n as i128)),
  };

  if num == 1 && den == 1 {
    Some(pi_power)
  } else if num == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(den)),
    })
  } else if den == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(pi_power),
    })
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(den)),
    })
  }
}

/// Compute Zeta[-n] for negative odd integer -n.
/// Returns (-1)^n * B_{n+1} / (n+1)
pub fn zeta_negative_odd(abs_n: usize) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(abs_n + 1)?;
  // (-1)^n * B_{n+1} / (n+1)
  let sign: i128 = if abs_n.is_multiple_of(2) { 1 } else { -1 };
  let result_num = sign.checked_mul(b_num)?;
  let result_den = b_den.checked_mul((abs_n + 1) as i128)?;
  Some(make_rational(result_num, result_den))
}

/// Compute Zeta(s) numerically for real s using Euler-Maclaurin formula.
pub fn zeta_numeric(s: f64) -> f64 {
  use std::f64::consts::PI;

  if (s - 1.0).abs() < 1e-15 {
    return f64::INFINITY;
  }

  // For s < 0.5, use the reflection formula:
  // zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
  if s < 0.5 {
    return 2.0_f64.powf(s)
      * PI.powf(s - 1.0)
      * (PI * s / 2.0).sin()
      * gamma_fn(1.0 - s)
      * zeta_numeric(1.0 - s);
  }

  // For s >= 0.5, use Euler-Maclaurin summation
  let n: usize = 20;
  let nf = n as f64;

  // Direct sum: sum_{k=1}^{N-1} k^{-s}
  let mut sum = 0.0;
  for k in 1..n {
    sum += (k as f64).powf(-s);
  }

  // Integral correction: N^{1-s} / (s-1)
  sum += nf.powf(1.0 - s) / (s - 1.0);
  // Endpoint correction: N^{-s} / 2
  sum += 0.5 * nf.powf(-s);

  // Bernoulli corrections: B_{2p}/(2p)! * prod_{j=0}^{2p-2}(s+j) * N^{-(s+2p-1)}
  let bof: [f64; 10] = [
    1.0 / 12.0,                          // B_2/2!
    -1.0 / 720.0,                        // B_4/4!
    1.0 / 30240.0,                       // B_6/6!
    -1.0 / 1209600.0,                    // B_8/8!
    1.0 / 47900160.0,                    // B_10/10!
    -691.0 / 1307674368000.0,            // B_12/12!
    7.0 / 523069747200.0,                // B_14/14!
    -3617.0 / 10670622842880000.0,       // B_16/16!
    43867.0 / 5109094217170944000.0,     // B_18/18!
    -174611.0 / 802857662698291200000.0, // B_20/20!
  ];

  for (p_idx, &coeff) in bof.iter().enumerate() {
    let two_p = 2 * (p_idx + 1);
    // Rising factorial: prod_{j=0}^{2p-2} (s+j)
    let mut rising = 1.0;
    for j in 0..(two_p - 1) {
      rising *= s + j as f64;
    }
    sum += coeff * rising * nf.powf(-(s + (two_p - 1) as f64));
  }

  sum
}

/// PolyGamma[z] - digamma function (equivalent to PolyGamma[0, z])
/// PolyGamma[n, z] - n-th derivative of the digamma function
pub fn polygamma_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (n_val, z_expr) = match args.len() {
    1 => (0_i128, &args[0]),
    2 => match &args[0] {
      Expr::Integer(n) => (*n, &args[1]),
      Expr::Real(f) => {
        // Real n: evaluate numerically if z is also numeric
        if let Some(z) = extract_f64(z_expr_from_args(&args[1])) {
          return Ok(Expr::Real(polygamma_numeric(*f as usize, z)));
        }
        return Ok(Expr::FunctionCall {
          name: "PolyGamma".to_string(),
          args: args.to_vec(),
        });
      }
      _ => {
        return Ok(Expr::FunctionCall {
          name: "PolyGamma".to_string(),
          args: args.to_vec(),
        });
      }
    },
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PolyGamma expects 1 or 2 arguments".into(),
      ));
    }
  };

  if n_val < 0 {
    return Ok(Expr::FunctionCall {
      name: "PolyGamma".to_string(),
      args: args.to_vec(),
    });
  }
  let n = n_val as usize;

  // Check for poles: z = 0 or negative integer
  if let Expr::Integer(z) = z_expr
    && *z <= 0
  {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  match z_expr {
    Expr::Integer(z) if *z > 0 => {
      let z = *z as usize;
      if n == 0 {
        // Digamma at positive integer: psi(z) = H_{z-1} - EulerGamma
        return Ok(polygamma_digamma_integer(z));
      }
      if n % 2 == 1 {
        // Odd n: exact result via Zeta (n+1 is even)
        if let Some(expr) = polygamma_odd_integer(n, z) {
          return Ok(expr);
        }
      }
      // Even n >= 2: return unevaluated (involves odd Zeta values)
      Ok(Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: vec![Expr::Integer(n as i128), Expr::Integer(z as i128)],
      })
    }
    Expr::Real(f) => Ok(Expr::Real(polygamma_numeric(n, *f))),
    _ => {
      // Symbolic: return unevaluated in 2-arg form
      Ok(Expr::FunctionCall {
        name: "PolyGamma".to_string(),
        args: if args.len() == 1 {
          vec![Expr::Integer(0), args[0].clone()]
        } else {
          args.to_vec()
        },
      })
    }
  }
}

pub fn extract_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

pub fn z_expr_from_args(expr: &Expr) -> &Expr {
  expr
}

/// Build digamma at positive integer: H_{z-1} - EulerGamma
pub fn polygamma_digamma_integer(z: usize) -> Expr {
  let euler = Expr::Identifier("EulerGamma".to_string());
  if z == 1 {
    // H_0 = 0, so result is -EulerGamma
    return Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(-1)),
      right: Box::new(euler),
    };
  }
  // Compute H_{z-1} = Σ_{k=1}^{z-1} 1/k as rational
  let (h_num, h_den) = harmonic_rational(z - 1);
  let h_expr = make_rational(h_num, h_den);
  Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(h_expr),
    right: Box::new(euler),
  }
}

/// Compute H_n = 1 + 1/2 + ... + 1/n as (numerator, denominator)
pub fn harmonic_rational(n: usize) -> (i128, i128) {
  let mut num: i128 = 0;
  let mut den: i128 = 1;
  for k in 1..=n {
    // num/den + 1/k = (num*k + den) / (den*k)
    num = num * (k as i128) + den;
    den *= k as i128;
    let g = gcd(num.abs(), den.abs());
    num /= g;
    den /= g;
  }
  (num, den)
}

/// Build exact PolyGamma[n, z] for odd n >= 1 and positive integer z.
/// Returns n! * (zeta(n+1) - partial_sum)
pub fn polygamma_odd_integer(n: usize, z: usize) -> Option<Expr> {
  // Get zeta(n+1) as a symbolic expression (raw, not multiplied by n!)
  let zeta_expr = zeta_positive_even(n + 1)?;

  // Compute n!
  let mut nfact: i128 = 1;
  for i in 2..=n {
    nfact = nfact.checked_mul(i as i128)?;
  }

  if z == 1 {
    // No partial sum. Result = n! * zeta(n+1)
    // Need to multiply the coefficient of zeta by n!
    return polygamma_multiply_zeta_by_nfact(n + 1, nfact);
  }

  // Compute partial sum = Σ_{k=1}^{z-1} 1/k^{n+1}
  let (ps_num, ps_den) = partial_sum_powers(z - 1, n + 1)?;

  // Inner expression: Plus[-partial_sum, zeta(n+1)]
  let neg_ps = make_rational(-ps_num, ps_den);
  let inner = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: vec![neg_ps, zeta_expr],
  };

  if nfact == 1 {
    // n = 1: just the inner expression
    Some(inner)
  } else {
    // n >= 3: Times[n!, inner]
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(nfact)),
      right: Box::new(inner),
    })
  }
}

/// Multiply zeta(2n) coefficient by n! and build the expression
pub fn polygamma_multiply_zeta_by_nfact(two_n: usize, nfact: i128) -> Option<Expr> {
  let (b_num, b_den) = bernoulli_number(two_n)?;
  if b_num == 0 {
    return None;
  }

  // Same as zeta_positive_even but multiply by nfact
  let mut num = b_num.abs().checked_mul(nfact)?;
  let mut den = b_den.abs();

  // Multiply by 2^(2n-1)
  for _ in 0..(two_n - 1) {
    num = num.checked_mul(2)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  // Divide by (2n)!
  for k in 1..=two_n {
    den = den.checked_mul(k as i128)?;
    let g = gcd(num, den);
    num /= g;
    den /= g;
  }

  let pi_power = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(Expr::Identifier("Pi".to_string())),
    right: Box::new(Expr::Integer(two_n as i128)),
  };

  if num == 1 && den == 1 {
    Some(pi_power)
  } else if num == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(pi_power),
      right: Box::new(Expr::Integer(den)),
    })
  } else if den == 1 {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(num)),
      right: Box::new(pi_power),
    })
  } else {
    Some(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(num)),
        right: Box::new(pi_power),
      }),
      right: Box::new(Expr::Integer(den)),
    })
  }
}

/// Compute Σ_{k=1}^{n} 1/k^power as (numerator, denominator)
pub fn partial_sum_powers(n: usize, power: usize) -> Option<(i128, i128)> {
  let mut sum_n: i128 = 0;
  let mut sum_d: i128 = 1;
  for k in 1..=n {
    let k_pow = (k as i128).checked_pow(power as u32)?;
    let new_n = sum_n.checked_mul(k_pow)?.checked_add(sum_d)?;
    let new_d = sum_d.checked_mul(k_pow)?;
    let g = gcd(new_n.abs(), new_d.abs());
    sum_n = new_n / g;
    sum_d = new_d / g;
  }
  Some((sum_n, sum_d))
}

/// Compute polygamma function numerically
pub fn polygamma_numeric(n: usize, mut z: f64) -> f64 {
  if n == 0 {
    return digamma(z);
  }

  let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 }; // (-1)^{n+1}
  let nfact = {
    let mut f = 1.0_f64;
    for i in 2..=n {
      f *= i as f64;
    }
    f
  };

  // Use recurrence to shift z to a large value
  let mut shift_sum = 0.0;
  while z < 20.0 {
    shift_sum += 1.0 / z.powi((n + 1) as i32);
    z += 1.0;
  }

  // Asymptotic expansion for ψ^(n)(z) at large z
  // ψ^(n)(z) = (-1)^{n-1} * [(n-1)!/z^n + n!/(2z^{n+1})
  //             + Σ_k B_{2k}/(2k) * prod_{j=0}^{n-1}(2k+j) / z^{n+2k}]
  let sign_asymp = if n.is_multiple_of(2) { -1.0 } else { 1.0 }; // (-1)^{n-1}
  let nm1_fact = nfact / n as f64;

  let mut asymp = nm1_fact / z.powi(n as i32);
  asymp += nfact / (2.0 * z.powi((n + 1) as i32));

  let bernoulli = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
  ];
  for (ki, &b2k) in bernoulli.iter().enumerate() {
    let k = ki + 1;
    let two_k = 2 * k;
    let mut prod = 1.0;
    for j in 0..n {
      prod *= (two_k + j) as f64;
    }
    asymp += b2k / (two_k as f64) * prod / z.powi((n + two_k) as i32);
  }

  asymp *= sign_asymp;
  asymp + sign * nfact * shift_sum
}

/// Beta[a, b] - Euler beta function
/// Beta[a, b] = Gamma[a] * Gamma[b] / Gamma[a + b]
pub fn beta_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "Beta expects exactly 2 arguments".into(),
    ));
  }

  // Try to evaluate for positive integer arguments
  // Beta[m, n] = (m-1)! * (n-1)! / (m+n-1)!
  if let (Expr::Integer(a), Expr::Integer(b)) = (&args[0], &args[1])
    && *a > 0
    && *b > 0
  {
    let a_u = (*a - 1) as usize;
    let b_u = (*b - 1) as usize;
    let ab_u = (*a + *b - 1) as usize;
    if let (Some(a_fact), Some(b_fact), Some(ab_fact)) = (
      factorial_i128(a_u),
      factorial_i128(b_u),
      factorial_i128(ab_u),
    ) {
      return Ok(make_rational(a_fact * b_fact, ab_fact));
    }
  }

  // Try rational args for half-integer cases
  // Beta[p/q, r/s] for half-integers involves Gamma at half-integers
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
  {
    // Check if both are positive half-integers (n + 1/2)
    let a_half = a_f * 2.0;
    let b_half = b_f * 2.0;
    if a_half == a_half.round()
      && b_half == b_half.round()
      && a_f > 0.0
      && b_f > 0.0
      && a_half.fract() == 0.0
      && b_half.fract() == 0.0
    {
      let a2 = a_half as i128;
      let b2 = b_half as i128;

      // At least one must be odd (half-integer) for Pi to appear
      if a2 % 2 != 0 || b2 % 2 != 0 {
        // Check if both args are Real (then numeric)
        if matches!(&args[0], Expr::Real(_))
          || matches!(&args[1], Expr::Real(_))
        {
          let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
          return Ok(Expr::Real(result));
        }
        // For exact half-integer arguments, compute via Gamma
        // If sum is integer, result is rational * sqrt(pi) or rational * pi
        let sum2 = a2 + b2;
        if sum2 % 2 == 0 {
          // Both half-integers or both integers, sum is integer
          // Result involves sqrt(pi) terms that may cancel
          // Use numeric for now unless both are half-integers with integer sum
          // Beta[a, b] = Γ(a)Γ(b)/Γ(a+b)
          // When a, b are half-integers, Γ(n+1/2) = (2n)! sqrt(π) / (4^n n!)
          // So Gamma product has π, and if sum is integer, Γ(sum) is (sum-1)!
          // Beta = Γ(a)Γ(b) / (sum-1)!
          let sum_int = (sum2 / 2) as usize;
          if let Some(sum_fact) = factorial_i128(sum_int - 1) {
            // Compute Γ(a) * Γ(b) / (sum-1)! where a, b are half-integers
            // Γ(k/2) for odd k: Γ((2m+1)/2) = (2m)! π^{1/2} / (4^m m!)
            // For even k: Γ(k/2) = ((k/2)-1)!
            let gamma_a = gamma_half_integer_parts(a2);
            let gamma_b = gamma_half_integer_parts(b2);
            if let (
              Some((a_num, a_den, a_pi_pow)),
              Some((b_num, b_den, b_pi_pow)),
            ) = (gamma_a, gamma_b)
            {
              let total_pi_pow = a_pi_pow + b_pi_pow; // each half-integer contributes 1/2
              let num =
                a_num.checked_mul(b_num).unwrap_or_else(|| a_num * b_num);
              let den = a_den
                .checked_mul(b_den)
                .and_then(|v| v.checked_mul(sum_fact))
                .unwrap_or(1);
              let g = gcd(num.abs(), den.abs());
              let (num, den) = if g > 0 {
                (num / g, den / g)
              } else {
                (num, den)
              };

              if total_pi_pow == 0 {
                return Ok(make_rational(num, den));
              } else if total_pi_pow == 2 {
                // Two sqrt(Pi) factors = Pi
                // Result is (num/den) * Pi
                if den == 1 {
                  if num == 1 {
                    return Ok(Expr::Identifier("Pi".to_string()));
                  }
                  return Ok(Expr::BinaryOp {
                    op: BinaryOperator::Times,
                    left: Box::new(Expr::Integer(num)),
                    right: Box::new(Expr::Identifier("Pi".to_string())),
                  });
                }
                return Ok(Expr::BinaryOp {
                  op: BinaryOperator::Divide,
                  left: Box::new(if num == 1 {
                    Expr::Identifier("Pi".to_string())
                  } else {
                    Expr::BinaryOp {
                      op: BinaryOperator::Times,
                      left: Box::new(Expr::Integer(num)),
                      right: Box::new(Expr::Identifier("Pi".to_string())),
                    }
                  }),
                  right: Box::new(Expr::Integer(den)),
                });
              }
            }
          }
        }
      }
    }
  }

  // Numeric evaluation
  if let (Some(a_f), Some(b_f)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && (matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_)))
  {
    let result = gamma_fn(a_f) * gamma_fn(b_f) / gamma_fn(a_f + b_f);
    return Ok(Expr::Real(result));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "Beta".to_string(),
    args: args.to_vec(),
  })
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

/// Compute parts of Gamma at half-integer: Gamma(k/2) for integer k > 0
/// Returns (numerator, denominator, pi_power) where result = (num/den) * Pi^(pi_power/2)
/// pi_power is 0 or 1 (representing sqrt(Pi)^pi_power)
pub fn gamma_half_integer_parts(k2: i128) -> Option<(i128, i128, i128)> {
  if k2 <= 0 {
    return None;
  }
  if k2 % 2 == 0 {
    // k2 = 2m, so Gamma(m) = (m-1)!
    let m = (k2 / 2) as usize;
    let fact = factorial_i128(m - 1)?;
    Some((fact, 1, 0))
  } else {
    // k2 = 2m+1, so Gamma(m + 1/2) = (2m)! * sqrt(pi) / (4^m * m!)
    let m = ((k2 - 1) / 2) as usize;
    let two_m_fact = factorial_i128(2 * m)?;
    let m_fact = factorial_i128(m)?;
    let four_m = 4i128.checked_pow(m as u32)?;
    Some((two_m_fact, four_m * m_fact, 1))
  }
}

/// PolyLog[s, z] - Polylogarithm function Li_s(z)
pub fn polylog_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "PolyLog expects exactly 2 arguments".into(),
    ));
  }

  let s_expr = &args[0];
  let z_expr = &args[1];

  match s_expr {
    Expr::Integer(s) => {
      return polylog_integer_s(*s, z_expr, args);
    }
    Expr::Real(sf) => {
      if let Some(zf) = extract_f64(z_expr) {
        return Ok(Expr::Real(polylog_numeric(*sf, zf)));
      }
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: args.to_vec(),
  })
}

pub fn polylog_integer_s(
  s: i128,
  z_expr: &Expr,
  orig_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // s = 1: PolyLog[1, z] = -Log[1-z]
  if s == 1 {
    return polylog_s1(z_expr);
  }

  // s = 0: PolyLog[0, z] = z/(1-z)
  if s == 0 {
    return polylog_s0(z_expr);
  }

  // s < 0: rational function via Eulerian numbers
  if s < 0 {
    return polylog_negative_s((-s) as usize, z_expr);
  }

  // s >= 2: special values at z = 0, 1, -1
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      // PolyLog[s, 1] = Zeta[s]
      return zeta_ast(&[Expr::Integer(s)]);
    }
    Expr::Integer(-1) => {
      // PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s]
      return polylog_at_neg1(s);
    }
    Expr::Real(f) => {
      return Ok(Expr::Real(polylog_numeric(s as f64, *f)));
    }
    _ => {}
  }

  Ok(Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: orig_args.to_vec(),
  })
}

/// PolyLog[1, z] = -Log[1-z]
pub fn polylog_s1(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Identifier("Infinity".to_string())),
    Expr::Integer(z) => {
      // 1-z is an integer, construct -Log[1-z]
      let one_minus_z = Expr::Integer(1 - z);
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(-1)),
        right: Box::new(Expr::FunctionCall {
          name: "Log".to_string(),
          args: vec![one_minus_z],
        }),
      })
    }
    Expr::FunctionCall {
      name,
      args: rat_args,
    } if name == "Rational" && rat_args.len() == 2 => {
      if let (Expr::Integer(p), Expr::Integer(q)) = (&rat_args[0], &rat_args[1])
      {
        // 1 - p/q = (q-p)/q
        let one_minus_z = make_rational(q - p, *q);
        // Evaluate Log to get simplification (e.g., Log[1/2] → -Log[2])
        let log_val = log_ast(&[one_minus_z])?;
        // Negate: -Log[1 - p/q], simplifying double negation
        match &log_val {
          Expr::BinaryOp {
            op: BinaryOperator::Times,
            left,
            right,
          } if matches!(left.as_ref(), Expr::Integer(-1)) => {
            // -(-expr) = expr
            Ok(*right.clone())
          }
          _ => Ok(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(log_val),
          }),
        }
      } else {
        polylog_s1_symbolic(z_expr)
      }
    }
    Expr::Real(f) => {
      let result = -(1.0 - f).ln();
      Ok(Expr::Real(result))
    }
    _ => polylog_s1_symbolic(z_expr),
  }
}

pub fn polylog_s1_symbolic(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  let one_minus_z = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(z_expr.clone()),
  };
  Ok(Expr::BinaryOp {
    op: BinaryOperator::Times,
    left: Box::new(Expr::Integer(-1)),
    right: Box::new(Expr::FunctionCall {
      name: "Log".to_string(),
      args: vec![one_minus_z],
    }),
  })
}

/// PolyLog[0, z] = z/(1-z)
pub fn polylog_s0(z_expr: &Expr) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => Ok(Expr::Integer(0)),
    Expr::Integer(1) => Ok(Expr::Identifier("ComplexInfinity".to_string())),
    Expr::Integer(z) => {
      // z/(1-z) as rational
      Ok(make_rational(*z, 1 - z))
    }
    Expr::Real(f) => Ok(Expr::Real(f / (1.0 - f))),
    _ => {
      // z/(1-z)
      Ok(Expr::BinaryOp {
        op: BinaryOperator::Divide,
        left: Box::new(z_expr.clone()),
        right: Box::new(Expr::BinaryOp {
          op: BinaryOperator::Minus,
          left: Box::new(Expr::Integer(1)),
          right: Box::new(z_expr.clone()),
        }),
      })
    }
  }
}

/// PolyLog[-n, z] for n >= 1 using Eulerian numbers
pub fn polylog_negative_s(
  n: usize,
  z_expr: &Expr,
) -> Result<Expr, InterpreterError> {
  match z_expr {
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    Expr::Integer(1) => {
      return Ok(Expr::Identifier("ComplexInfinity".to_string()));
    }
    Expr::Integer(z) => {
      // Evaluate numerically for integer z != 0, 1
      let zf = *z as f64;
      return Ok(Expr::Real(polylog_numeric(-(n as f64), zf)));
    }
    Expr::Real(f) => return Ok(Expr::Real(polylog_numeric(-(n as f64), *f))),
    _ => {}
  }

  // Compute Eulerian numbers A(n, k) for k = 0..n-1
  let eulerian = eulerian_numbers(n);

  // Build numerator: Σ A(n, k) * x^{k+1}
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &a) in eulerian.iter().enumerate() {
    if a == 0 {
      continue;
    }
    let x_power = if k + 1 == 1 {
      z_expr.clone()
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(z_expr.clone()),
        right: Box::new(Expr::Integer((k + 1) as i128)),
      }
    };
    let term = if a == 1 {
      x_power
    } else {
      Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(a)),
        right: Box::new(x_power),
      }
    };
    terms.push(term);
  }

  let numerator = if terms.len() == 1 {
    terms.into_iter().next().unwrap()
  } else {
    Expr::FunctionCall {
      name: "Plus".to_string(),
      args: terms,
    }
  };

  // Denominator: (1 - x)^{n+1}
  let one_minus_x = Expr::BinaryOp {
    op: BinaryOperator::Minus,
    left: Box::new(Expr::Integer(1)),
    right: Box::new(z_expr.clone()),
  };
  let denominator = Expr::BinaryOp {
    op: BinaryOperator::Power,
    left: Box::new(one_minus_x),
    right: Box::new(Expr::Integer((n + 1) as i128)),
  };

  Ok(Expr::BinaryOp {
    op: BinaryOperator::Divide,
    left: Box::new(numerator),
    right: Box::new(denominator),
  })
}

/// Compute Eulerian numbers A(n, k) for k = 0, ..., n-1
pub fn eulerian_numbers(n: usize) -> Vec<i128> {
  if n == 0 {
    return vec![];
  }
  let mut a = vec![0_i128; n];
  a[0] = 1;
  for m in 2..=n {
    let mut new_a = vec![0_i128; n];
    new_a[0] = 1;
    for k in 1..m {
      new_a[k] = (k as i128 + 1) * a[k] + (m as i128 - k as i128) * a[k - 1];
    }
    a = new_a;
  }
  a[..n].to_vec()
}

/// PolyLog[s, -1] = -(1 - 2^{1-s}) * Zeta[s] for s >= 2
pub fn polylog_at_neg1(s: i128) -> Result<Expr, InterpreterError> {
  let s_usize = s as usize;

  if s % 2 == 0 {
    // Even s: Zeta[s] is exact
    if let Some((b_num, b_den)) = bernoulli_number(s_usize) {
      if b_num == 0 {
        return Ok(Expr::Integer(0));
      }

      // Compute Zeta coefficient: |B_{2n}| * 2^{2n-1} / (2n)!
      let mut znum = b_num.abs();
      let mut zden = b_den.abs();
      for _ in 0..(s_usize - 1) {
        znum = match znum.checked_mul(2) {
          Some(v) => v,
          None => return Ok(unevaluated_polylog(s, -1)),
        };
        let g = gcd(znum, zden);
        znum /= g;
        zden /= g;
      }
      for k in 1..=s_usize {
        zden = match zden.checked_mul(k as i128) {
          Some(v) => v,
          None => return Ok(unevaluated_polylog(s, -1)),
        };
        let g = gcd(znum, zden);
        znum /= g;
        zden /= g;
      }

      // Multiply by -(1 - 2^{1-s}) = (1 - 2^{s-1}) / 2^{s-1}
      let pow2 = 1_i128 << (s_usize - 1);
      let coeff_num = 1 - pow2; // negative
      let coeff_den = pow2;

      let mut final_num = match coeff_num.checked_mul(znum) {
        Some(v) => v,
        None => return Ok(unevaluated_polylog(s, -1)),
      };
      let mut final_den = match coeff_den.checked_mul(zden) {
        Some(v) => v,
        None => return Ok(unevaluated_polylog(s, -1)),
      };
      if final_den < 0 {
        final_num = -final_num;
        final_den = -final_den;
      }
      let g = gcd(final_num.abs(), final_den);
      final_num /= g;
      final_den /= g;

      // Build coefficient * Pi^s
      let pi_power = Expr::BinaryOp {
        op: BinaryOperator::Power,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(Expr::Integer(s)),
      };

      if final_num.abs() == 1 && final_den == 1 {
        if final_num == 1 {
          return Ok(pi_power);
        } else {
          return Ok(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(-1)),
            right: Box::new(pi_power),
          });
        }
      } else if final_num.abs() == 1 {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(make_rational(final_num, final_den)),
          right: Box::new(pi_power),
        });
      } else if final_den == 1 {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Times,
          left: Box::new(Expr::Integer(final_num)),
          right: Box::new(pi_power),
        });
      } else {
        return Ok(Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(Expr::BinaryOp {
            op: BinaryOperator::Times,
            left: Box::new(Expr::Integer(final_num)),
            right: Box::new(pi_power),
          }),
          right: Box::new(Expr::Integer(final_den)),
        });
      }
    }
  } else {
    // Odd s >= 3: Zeta stays symbolic
    // PolyLog[s, -1] = cn/cd * Zeta[s] where cn < 0
    let pow2 = 1_i128 << (s_usize - 1);
    let coeff_num = 1 - pow2;
    let coeff_den = pow2;
    let g = gcd(coeff_num.abs(), coeff_den);
    let cn = coeff_num / g;
    let cd = coeff_den / g;

    let zeta_expr = Expr::FunctionCall {
      name: "Zeta".to_string(),
      args: vec![Expr::Integer(s)],
    };

    if cd == 1 {
      return Ok(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(cn)),
        right: Box::new(zeta_expr),
      });
    }
    // Use (cn*Zeta[s])/cd format
    return Ok(Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Integer(cn)),
        right: Box::new(zeta_expr),
      }),
      right: Box::new(Expr::Integer(cd)),
    });
  }

  Ok(unevaluated_polylog(s, -1))
}

pub fn unevaluated_polylog(s: i128, z: i128) -> Expr {
  Expr::FunctionCall {
    name: "PolyLog".to_string(),
    args: vec![Expr::Integer(s), Expr::Integer(z)],
  }
}

/// Compute polylogarithm numerically using series summation
pub fn polylog_numeric(s: f64, z: f64) -> f64 {
  if z == 0.0 {
    return 0.0;
  }
  if z.abs() <= 1.0 {
    let mut sum = 0.0;
    let mut z_power = z;
    for k in 1..=10000 {
      let term = z_power / (k as f64).powf(s);
      sum += term;
      if term.abs() < 1e-15 * sum.abs().max(1e-300) {
        break;
      }
      z_power *= z;
    }
    sum
  } else {
    f64::NAN
  }
}

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

  if x.abs() < 6.0 {
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..200 {
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
  } else if x > 0.0 {
    // Asymptotic for large positive x
    let zeta = 2.0 / 3.0 * x.powf(1.5);
    let prefactor =
      (-zeta).exp() / (2.0 * std::f64::consts::PI.sqrt() * x.powf(0.25));
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..30 {
      let kf = k as f64;
      let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
      term *= -num / (216.0 * kf * zeta);
      sum += term;
      if term.abs() < 1e-16 {
        break;
      }
    }
    prefactor * sum
  } else {
    // Large negative x: use series (extend range)
    let mut f = 1.0;
    let mut g = x;
    let mut f_term = 1.0;
    let mut g_term = x;

    for k in 1..500 {
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
  }
}

/// Hypergeometric1F1[a, b, z] - Kummer's confluent hypergeometric function
pub fn hypergeometric1f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric1F1 expects exactly 3 arguments".into(),
    ));
  }

  // 1F1[a, b, 0] = 1
  if is_expr_zero(&args[2]) {
    return Ok(Expr::Integer(1));
  }

  // Numeric evaluation
  let a_val = expr_to_f64(&args[0]);
  let b_val = expr_to_f64(&args[1]);
  let z_val = expr_to_f64(&args[2]);

  if let (Some(a), Some(b), Some(z)) = (a_val, b_val, z_val) {
    let has_real = matches!(&args[0], Expr::Real(_))
      || matches!(&args[1], Expr::Real(_))
      || matches!(&args[2], Expr::Real(_));
    if has_real {
      return Ok(Expr::Real(hypergeometric_1f1(a, b, z)));
    }
  }

  Ok(Expr::FunctionCall {
    name: "Hypergeometric1F1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute 1F1(a, b; z) = Σ (a)_n z^n / ((b)_n n!)
pub fn hypergeometric_1f1(a: f64, b: f64, z: f64) -> f64 {
  let mut sum = 1.0;
  let mut term = 1.0;

  for n in 0..1000 {
    let nf = n as f64;
    term *= (a + nf) * z / ((b + nf) * (nf + 1.0));
    sum += term;
    if term.abs() < 1e-16 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// HypergeometricU[a, b, z] - confluent hypergeometric function of the second kind
pub fn hypergeometric_u_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "HypergeometricU expects exactly 3 arguments".into(),
    ));
  }

  // Numeric evaluation when at least one argument is Real and all are numeric
  let a_val = expr_to_f64(&args[0]);
  let b_val = expr_to_f64(&args[1]);
  let z_val = expr_to_f64(&args[2]);

  if let (Some(a), Some(b), Some(z)) = (a_val, b_val, z_val) {
    let has_real = matches!(&args[0], Expr::Real(_))
      || matches!(&args[1], Expr::Real(_))
      || matches!(&args[2], Expr::Real(_));
    if has_real {
      return Ok(Expr::Real(hypergeometric_u_f64(a, b, z)));
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "HypergeometricU".to_string(),
    args: args.to_vec(),
  })
}

/// Compute U(a, b, z) numerically using the relation:
/// U(a,b,z) = Γ(1-b)/Γ(a+1-b) * M(a,b,z) + Γ(b-1)/Γ(a) * z^(1-b) * M(a+1-b,2-b,z)
/// where M = 1F1. For integer b, use Richardson extrapolation on the b parameter.
pub fn hypergeometric_u_f64(a: f64, b: f64, z: f64) -> f64 {
  let b_int = b.round();
  let is_b_integer = (b - b_int).abs() < 1e-10;

  if is_b_integer {
    // Use Richardson extrapolation: evaluate at several offsets and extrapolate
    // to the limit b -> integer. This cancels the leading error terms.
    let h = 0.001;
    let u1 = hypergeometric_u_nonint(a, b + h, z);
    let u2 = hypergeometric_u_nonint(a, b - h, z);
    let u3 = hypergeometric_u_nonint(a, b + 2.0 * h, z);
    let u4 = hypergeometric_u_nonint(a, b - 2.0 * h, z);
    // Richardson extrapolation: (4 * f(h) - f(2h)) / 3
    let avg_h = (u1 + u2) / 2.0;
    let avg_2h = (u3 + u4) / 2.0;
    (4.0 * avg_h - avg_2h) / 3.0
  } else {
    hypergeometric_u_nonint(a, b, z)
  }
}

pub fn hypergeometric_u_nonint(a: f64, b: f64, z: f64) -> f64 {
  let g1b = gamma_fn(1.0 - b);
  let ga1b = gamma_fn(a + 1.0 - b);
  let gb1 = gamma_fn(b - 1.0);
  let ga = gamma_fn(a);

  let term1 = if ga1b.is_infinite() || ga1b == 0.0 {
    0.0
  } else {
    g1b / ga1b * hypergeometric_1f1(a, b, z)
  };

  let term2 = if ga.is_infinite() || ga == 0.0 {
    0.0
  } else {
    gb1 / ga * z.powf(1.0 - b) * hypergeometric_1f1(a + 1.0 - b, 2.0 - b, z)
  };

  term1 + term2
}

pub fn hypergeometric2f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 4 {
    return Err(InterpreterError::EvaluationError(
      "Hypergeometric2F1 expects exactly 4 arguments".into(),
    ));
  }

  // Special case: z = 0 => result is 1
  if matches!(&args[3], Expr::Integer(0)) {
    return Ok(Expr::Integer(1));
  }

  // Try numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args
    .iter()
    .map(|a| match a {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(f) => Some(*f),
      _ => None,
    })
    .collect();

  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a = vals[0].unwrap();
    let b = vals[1].unwrap();
    let c = vals[2].unwrap();
    let z = vals[3].unwrap();
    let result = hypergeometric2f1(a, b, c, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "Hypergeometric2F1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute 2F1(a, b; c; z) using series expansion
pub fn hypergeometric2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
  // Series: sum_{n=0}^{inf} (a)_n (b)_n / (c)_n / n! * z^n
  let mut sum = 1.0;
  let mut term = 1.0;

  for n in 0..1000 {
    let nf = n as f64;
    term *= (a + nf) * (b + nf) / (c + nf) * z / (nf + 1.0);
    sum += term;
    if term.abs() < 1e-16 * sum.abs().max(1e-300) {
      break;
    }
  }
  sum
}

/// ProductLog[z] - Lambert W function (principal branch)
pub fn product_log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "ProductLog expects 1 argument".into(),
    ));
  }
  match &args[0] {
    // ProductLog[0] = 0
    Expr::Integer(0) => return Ok(Expr::Integer(0)),
    // ProductLog[E] = 1
    Expr::Identifier(s) if s == "E" => return Ok(Expr::Integer(1)),
    Expr::Constant(s) if s == "E" => return Ok(Expr::Integer(1)),
    // ProductLog[-1/E] = -1
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Divide,
      left,
      right,
    } => {
      if let Expr::Integer(-1) = left.as_ref()
        && (matches!(right.as_ref(), Expr::Identifier(s) if s == "E")
          || matches!(right.as_ref(), Expr::Constant(s) if s == "E"))
      {
        return Ok(Expr::Integer(-1));
      }
    }
    // ProductLog[x.] for float
    Expr::Real(f) => {
      if *f >= -1.0 / std::f64::consts::E {
        // Use iterative approximation (Halley's method)
        let x = *f;
        let mut w = if x < 1.0 { x } else { x.ln() };
        for _ in 0..50 {
          let ew = w.exp();
          let wew = w * ew;
          let delta = wew - x;
          if delta.abs() < 1e-15 {
            break;
          }
          w -= delta / (ew * (w + 1.0) - (w + 2.0) * delta / (2.0 * (w + 1.0)));
        }
        return Ok(Expr::Real(w));
      }
    }
    _ => {}
  }
  Ok(Expr::FunctionCall {
    name: "ProductLog".to_string(),
    args: args.to_vec(),
  })
}

// ─── Number Theory Functions ─────────────────────────────────────

/// LerchPhi[z, s, a] - Lerch transcendent Φ(z, s, a) = Σ_{k=0}^∞ z^k / (k+a)^s
pub fn lerch_phi_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "LerchPhi expects exactly 3 arguments".into(),
    ));
  }

  let z = &args[0];
  let s = &args[1];
  let a = &args[2];

  // Special case: z = 0 → a^(-s)
  if is_expr_zero(z) {
    if let (Some(af), Some(sf)) = (try_eval_to_f64(a), try_eval_to_f64(s)) {
      return Ok(Expr::Real(af.powf(-sf)));
    }
    return crate::evaluator::evaluate_function_call_ast(
      "Power",
      &[
        a.clone(),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), s.clone()],
        },
      ],
    );
  }

  // Numeric evaluation
  if let (Some(zf), Some(sf), Some(af)) =
    (try_eval_to_f64(z), try_eval_to_f64(s), try_eval_to_f64(a))
    && (zf.abs() < 1.0 || ((zf - 1.0).abs() < 1e-16 && sf > 1.0))
  {
    let result = lerch_phi_numeric(zf, sf, af);
    if result.is_finite() {
      return Ok(Expr::Real(result));
    }
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "LerchPhi".to_string(),
    args: args.to_vec(),
  })
}

/// Compute LerchPhi numerically via series: Σ z^k / (k+a)^s
pub fn lerch_phi_numeric(z: f64, s: f64, a: f64) -> f64 {
  // For z = 1, this is the Hurwitz zeta: Σ 1/(k+a)^s
  // Use Euler-Maclaurin to add tail correction for better convergence
  let n_terms = if (z - 1.0).abs() < 1e-14 { 200 } else { 1000 };
  let mut sum = 0.0;
  let mut z_pow = 1.0; // z^k
  for k in 0..n_terms {
    let denom = (k as f64 + a).powf(s);
    if denom.abs() > 1e-300 {
      let term = z_pow / denom;
      sum += term;
      if term.abs() < 1e-15 * sum.abs() && k > 5 {
        return sum;
      }
    }
    z_pow *= z;
    if z_pow.abs() < 1e-300 {
      return sum;
    }
  }

  // For z=1 (Hurwitz zeta), add integral tail: ∫_{N}^∞ 1/(t+a)^s dt = (N+a)^(1-s)/(s-1)
  if (z - 1.0).abs() < 1e-14 && s > 1.0 {
    let n = n_terms as f64;
    let tail = (n + a).powf(1.0 - s) / (s - 1.0);
    // Plus first-order Euler-Maclaurin correction: f(N)/2
    let f_n = 1.0 / (n + a).powf(s);
    sum += tail + f_n / 2.0;
  }

  sum
}

