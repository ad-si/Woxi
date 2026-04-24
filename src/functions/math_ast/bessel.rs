#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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

  // Closed-form rules at ±1/2: BesselJ[±1/2, z] = Sqrt[2/(Pi z)] * {Sin,Cos}[z].
  if let Some((n_num, n_den)) =
    crate::functions::math_ast::expr_to_rational(n_expr)
    && n_den == 2
  {
    use crate::syntax::Expr::*;
    let trig = match n_num {
      1 => Some("Sin"),
      -1 => Some("Cos"),
      _ => None,
    };
    if let Some(t) = trig {
      // Build Sqrt[2] {Trig}[z] / (Sqrt[z] Sqrt[Pi])
      let expr = FunctionCall {
        name: "Times".to_string(),
        args: vec![
          FunctionCall {
            name: "Sqrt".to_string(),
            args: vec![Integer(2)],
          },
          FunctionCall {
            name: t.to_string(),
            args: vec![z_expr.clone()],
          },
          FunctionCall {
            name: "Power".to_string(),
            args: vec![
              FunctionCall {
                name: "Times".to_string(),
                args: vec![
                  FunctionCall {
                    name: "Sqrt".to_string(),
                    args: vec![z_expr.clone()],
                  },
                  FunctionCall {
                    name: "Sqrt".to_string(),
                    args: vec![Identifier("Pi".to_string())],
                  },
                ],
              },
              Integer(-1),
            ],
          },
        ],
      };
      return crate::evaluator::evaluate_expr_to_expr(&expr);
    }
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "BesselJ".to_string(),
    args: args.to_vec(),
  })
}

/// BesselJZero[n, k] — k-th positive zero of the Bessel function J_n
pub fn bessel_j_zero_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "BesselJZero expects exactly 2 arguments".into(),
    ));
  }

  // Extract numeric values
  let n_val = match &args[0] {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let k_val = match &args[1] {
    Expr::Integer(k) => Some(*k),
    Expr::Real(f) if *f == f.floor() && *f > 0.0 => Some(*f as i128),
    _ => None,
  };

  // Numeric evaluation when n is Real or k is Real
  let has_real =
    matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_));
  if has_real
    && let (Some(n), Some(k)) = (n_val, k_val)
    && k >= 1
  {
    let result = bessel_j_zero(n, k as usize);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated (symbolic)
  Ok(Expr::FunctionCall {
    name: "BesselJZero".to_string(),
    args: args.to_vec(),
  })
}

/// Find the k-th positive zero of J_n(x) using bisection + Newton's method
fn bessel_j_zero(n: f64, k: usize) -> f64 {
  // Find zeros by scanning for sign changes, then refining with bisection
  let step = 0.5;
  let start = if n > 0.0 { n * 0.5 } else { step };
  let mut x = start;
  let mut prev_val = bessel_j(n, x);
  let mut zeros_found = 0;

  loop {
    let next_x = x + step;
    let next_val = bessel_j(n, next_x);

    if prev_val * next_val < 0.0 {
      // Sign change: there's a zero in [x, next_x]
      zeros_found += 1;
      if zeros_found == k {
        // Refine using bisection then Newton
        let mut lo = x;
        let mut hi = next_x;

        // Bisection to get close
        for _ in 0..60 {
          let mid = (lo + hi) / 2.0;
          let mid_val = bessel_j(n, mid);
          if mid_val == 0.0 {
            return mid;
          }
          if bessel_j(n, lo) * mid_val < 0.0 {
            hi = mid;
          } else {
            lo = mid;
          }
        }

        // Newton's method to polish
        let mut root = (lo + hi) / 2.0;
        for _ in 0..20 {
          let jn = bessel_j(n, root);
          // J'_n(x) = n/x * J_n(x) - J_{n+1}(x)
          let jn1 = bessel_j(n + 1.0, root);
          let deriv = (n / root) * jn - jn1;
          if deriv.abs() < 1e-300 {
            break;
          }
          let delta = jn / deriv;
          root -= delta;
          if delta.abs() < 1e-15 * root.abs() {
            break;
          }
        }
        return root;
      }
    }

    prev_val = next_val;
    x = next_x;

    // Safety: if we've searched very far, give up
    if x > n + (k as f64) * std::f64::consts::PI + 100.0 {
      break;
    }
  }

  f64::NAN
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

/// SphericalBesselJ[n, z] — spherical Bessel function of the first kind.
/// SphericalBesselJ[n, z] = Sqrt[Pi/(2z)] * BesselJ[n + 1/2, z]
pub fn spherical_bessel_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "SphericalBesselJ".to_string(),
      args: args.to_vec(),
    });
  }

  let n_expr = &args[0];
  let z_expr = &args[1];

  // Handle z = 0 case: SphericalBesselJ[0, 0] = 1, SphericalBesselJ[n, 0] = 0 for n > 0
  if matches!(z_expr, Expr::Integer(0))
    && let Some(n) = expr_to_i128(n_expr)
  {
    if n == 0 {
      return Ok(Expr::Integer(1));
    } else if n > 0 {
      return Ok(Expr::Integer(0));
    }
  }

  // For numeric evaluation, compute Sqrt[Pi/(2z)] * BesselJ[n + 1/2, z]
  let has_real =
    matches!(n_expr, Expr::Real(_)) || matches!(z_expr, Expr::Real(_));
  let n_f64 = try_eval_to_f64(n_expr);
  let z_f64 = try_eval_to_f64(z_expr);

  if has_real && let (Some(n_val), Some(z_val)) = (n_f64, z_f64) {
    let bessel_result =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "BesselJ".to_string(),
        args: vec![Expr::Real(n_val + 0.5), Expr::Real(z_val)],
      })?;
    if let Some(bj) = try_eval_to_f64(&bessel_result) {
      let result = (std::f64::consts::PI / (2.0 * z_val)).sqrt() * bj;
      return Ok(Expr::Real(result));
    }
  }

  // Return unevaluated for symbolic case
  Ok(Expr::FunctionCall {
    name: "SphericalBesselJ".to_string(),
    args: args.to_vec(),
  })
}

/// Helper to build a Hankel-type function from two Bessel-type results
fn build_hankel(
  name: &str,
  j_name: &str,
  y_name: &str,
  sign: i128,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "{} expects exactly 2 arguments",
      name
    )));
  }
  // Try numeric evaluation: compute BesselJ + sign*I*BesselY
  let j_result =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: j_name.to_string(),
      args: args.to_vec(),
    })?;
  let y_result =
    crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: y_name.to_string(),
      args: args.to_vec(),
    })?;
  // Only proceed if both evaluated to numbers
  let j_val = try_eval_to_f64(&j_result);
  let y_val = try_eval_to_f64(&y_result);
  if let (Some(j), Some(y)) = (j_val, y_val)
    && j.is_finite()
    && y.is_finite()
  {
    let expr = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Real(j),
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Complex".to_string(),
              args: vec![Expr::Integer(0), Expr::Integer(sign)],
            },
            Expr::Real(y),
          ],
        },
      ],
    };
    return crate::evaluator::evaluate_expr_to_expr(&expr);
  }
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec(),
  })
}

/// HankelH1[n, z] = BesselJ[n, z] + I * BesselY[n, z]
pub fn hankel_h1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  build_hankel("HankelH1", "BesselJ", "BesselY", 1, args)
}

/// HankelH2[n, z] = BesselJ[n, z] - I * BesselY[n, z]
pub fn hankel_h2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  build_hankel("HankelH2", "BesselJ", "BesselY", -1, args)
}

/// SphericalHankelH1[n, z] = SphericalBesselJ[n, z] + I * SphericalBesselY[n, z]
pub fn spherical_hankel_h1_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  build_hankel(
    "SphericalHankelH1",
    "SphericalBesselJ",
    "SphericalBesselY",
    1,
    args,
  )
}

/// SphericalHankelH2[n, z] = SphericalBesselJ[n, z] - I * SphericalBesselY[n, z]
pub fn spherical_hankel_h2_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  build_hankel(
    "SphericalHankelH2",
    "SphericalBesselJ",
    "SphericalBesselY",
    -1,
    args,
  )
}

// ─── Kelvin functions ─────────────────────────────────────────────────
// Defined via BesselJ on a rotated argument:
//   ber(x) + I·bei(x) = BesselJ[0, x·e^(3 Pi I/4)]
//   ker(x) + I·kei(x) = e^(-Pi·I/2)·BesselK[0, x·e^(Pi I/4)]
// The real-valued series for ber and bei are used directly.

/// Real-valued `ber(x)` via its power series.
fn ber_series(x: f64) -> f64 {
  let u = (x * 0.5) * (x * 0.5);
  let mut term = 1.0; // k=0 term
  let mut sum = term;
  for k in 1..200usize {
    let k2 = (2 * k) as f64;
    term *= -u * u / ((k2 - 1.0) * k2 * (k2 - 1.0) * k2);
    sum += term;
    if term.abs() < 1e-18 * sum.abs().max(1.0) {
      break;
    }
  }
  sum
}

/// Real-valued `bei(x)` via its power series.
fn bei_series(x: f64) -> f64 {
  let u = (x * 0.5) * (x * 0.5);
  let mut term = u; // k=0 term = (x/2)^2
  let mut sum = term;
  for k in 1..200usize {
    let k2 = (2 * k) as f64;
    term *= -u * u / (k2 * (k2 + 1.0) * k2 * (k2 + 1.0));
    sum += term;
    if term.abs() < 1e-18 * sum.abs().max(1.0) {
      break;
    }
  }
  sum
}

/// KelvinBer[x] - real part of BesselJ[0, x·e^(3πI/4)].
pub fn kelvin_ber_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KelvinBer expects 1 argument".into(),
    ));
  }
  if let Some(x) = match &args[0] {
    Expr::Real(f) => Some(*f),
    _ => None,
  } {
    return Ok(Expr::Real(ber_series(x)));
  }
  Ok(Expr::FunctionCall {
    name: "KelvinBer".to_string(),
    args: args.to_vec(),
  })
}

/// KelvinBei[x] - imaginary part of BesselJ[0, x·e^(3πI/4)].
pub fn kelvin_bei_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "KelvinBei expects 1 argument".into(),
    ));
  }
  if let Some(x) = match &args[0] {
    Expr::Real(f) => Some(*f),
    _ => None,
  } {
    return Ok(Expr::Real(bei_series(x)));
  }
  Ok(Expr::FunctionCall {
    name: "KelvinBei".to_string(),
    args: args.to_vec(),
  })
}
