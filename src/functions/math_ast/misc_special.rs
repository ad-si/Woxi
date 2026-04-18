#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;
use num_bigint::BigInt;

/// QPochhammer[a, q, n] — q-Pochhammer symbol.
/// Computes Product[(1 - a*q^k), {k, 0, n-1}] for non-negative integer n.
pub fn q_pochhammer_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "QPochhammer".to_string(),
      args: args.to_vec(),
    });
  }

  let a = &args[0];
  let q = &args[1];
  let n_expr = &args[2];

  // n must be a non-negative integer
  let n = match expr_to_i128(n_expr) {
    Some(n) if n >= 0 => n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "QPochhammer".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // QPochhammer[a, q, 0] = 1
  if n == 0 {
    return Ok(Expr::Integer(1));
  }

  // Compute the product symbolically: Product[(1 - a*q^k), {k, 0, n-1}]
  // Build each factor and multiply using the evaluator
  let mut result = Expr::Integer(1);
  for k in 0..n {
    // Compute q^k
    let qk = if k == 0 {
      Expr::Integer(1)
    } else {
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Power".to_string(),
        args: vec![q.clone(), Expr::Integer(k as i128)],
      })?
    };
    // Compute a * q^k
    let aqk = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![a.clone(), qk],
    })?;
    // Compute 1 - a*q^k
    let factor =
      crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Integer(1),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Integer(-1), aqk],
          },
        ],
      })?;
    // Multiply into result
    result = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![result, factor],
    })?;
  }

  Ok(result)
}

/// ProductLog[z] - Lambert W function (principal branch)
pub fn product_log_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "ProductLog expects 1 or 2 arguments".into(),
    ));
  }

  // Two-argument form: ProductLog[k, z] — branch k of Lambert W
  if args.len() == 2 {
    // ProductLog[0, z] == ProductLog[z] (principal branch)
    if matches!(&args[0], Expr::Integer(0)) {
      return product_log_ast(&args[1..]);
    }
    // Otherwise return unevaluated
    return Ok(Expr::FunctionCall {
      name: "ProductLog".to_string(),
      args: args.to_vec(),
    });
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

/// Helper: convert a list of Expr to Vec<f64>, returning None if any element is not numeric
fn expr_list_to_f64_vec(list: &[Expr]) -> Option<Vec<f64>> {
  list
    .iter()
    .map(|e| match e {
      Expr::Real(x) => Some(*x),
      Expr::Integer(n) => Some(*n as f64),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some(*n as f64 / *d as f64)
        } else {
          None
        }
      }
      _ => None,
    })
    .collect()
}

/// MeijerG[{{a1,...,an}, {an+1,...,ap}}, {{b1,...,bm}, {bm+1,...,bq}}, z]
/// Meijer G-function
pub fn meijer_g_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "MeijerG".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse upper parameters: {{a1,...,an}, {an+1,...,ap}}
  let (upper_n, upper_rest) = match &args[0] {
    Expr::List(v) if v.len() == 2 => {
      let list_n = match &v[0] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let list_rest = match &v[1] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (list_n, list_rest)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MeijerG".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse lower parameters: {{b1,...,bm}, {bm+1,...,bq}}
  let (lower_m, lower_rest) = match &args[1] {
    Expr::List(v) if v.len() == 2 => {
      let list_m = match &v[0] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let list_rest = match &v[1] {
        Expr::List(l) => l.clone(),
        _ => {
          return Ok(Expr::FunctionCall {
            name: "MeijerG".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (list_m, list_rest)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MeijerG".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let z = &args[2];

  let n = upper_n.len(); // number of a parameters in first list
  let _p = n + upper_rest.len(); // total number of upper parameters
  let m = lower_m.len(); // number of b parameters in first list
  let _q = m + lower_rest.len(); // total number of lower parameters

  // Consistency check: need m > 0 or n > 0, and p+q < 2(m+n) or other conditions
  if m == 0 && n == 0 {
    // No poles to sum over - function doesn't exist
    return Ok(Expr::FunctionCall {
      name: "MeijerG".to_string(),
      args: args.to_vec(),
    });
  }

  // MeijerG[{{}, {}}, {{0}, {}}, 0] = 1
  if let Expr::Integer(0) = z
    && n == 0
    && m == 1
    && lower_rest.is_empty()
    && upper_n.is_empty()
    && upper_rest.is_empty()
    && let Some(b0) = expr_to_i128(&lower_m[0])
    && b0 == 0
  {
    return Ok(Expr::Integer(1));
  }

  // Try numeric evaluation
  let z_val = match z {
    Expr::Real(x) => Some(*x),
    Expr::Integer(n) => Some(*n as f64),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
        Some(*n as f64 / *d as f64)
      } else {
        None
      }
    }
    _ => None,
  };

  let has_real = matches!(z, Expr::Real(_))
    || upper_n.iter().any(|e| matches!(e, Expr::Real(_)))
    || upper_rest.iter().any(|e| matches!(e, Expr::Real(_)))
    || lower_m.iter().any(|e| matches!(e, Expr::Real(_)))
    || lower_rest.iter().any(|e| matches!(e, Expr::Real(_)));

  if let Some(z_val) = z_val {
    let a_n_vals = expr_list_to_f64_vec(&upper_n);
    let a_rest_vals = expr_list_to_f64_vec(&upper_rest);
    let b_m_vals = expr_list_to_f64_vec(&lower_m);
    let b_rest_vals = expr_list_to_f64_vec(&lower_rest);

    if let (Some(a_n), Some(a_rest), Some(b_m), Some(b_rest)) =
      (a_n_vals, a_rest_vals, b_m_vals, b_rest_vals)
    {
      // Check hdiv condition: a_i - b_j must not be a positive integer
      // for i=1,...,n and j=1,...,m
      for &ai in &a_n {
        for &bj in &b_m {
          let diff = ai - bj;
          if diff > 0.0
            && (diff - diff.round()).abs() < 1e-14
            && diff.round() >= 1.0
          {
            // hdiv: function does not exist
            return Ok(Expr::FunctionCall {
              name: "MeijerG".to_string(),
              args: args.to_vec(),
            });
          }
        }
      }

      // All parameters are numeric - compute
      let mut all_a = a_n.clone();
      all_a.extend_from_slice(&a_rest);
      let mut all_b = b_m.clone();
      all_b.extend_from_slice(&b_rest);

      if has_real || matches!(z, Expr::Real(_)) {
        let result = meijer_g_numeric(n, m, &all_a, &all_b, z_val);
        if result.is_finite() {
          return Ok(Expr::Real(result));
        }
      } else {
        // Integer/rational args - only evaluate via N[]
        // Return unevaluated for pure integer/rational input
        return Ok(Expr::FunctionCall {
          name: "MeijerG".to_string(),
          args: args.to_vec(),
        });
      }
    }
  }

  Ok(Expr::FunctionCall {
    name: "MeijerG".to_string(),
    args: args.to_vec(),
  })
}

/// Numeric evaluation of MeijerG using residue series.
///
/// G^{m,n}_{p,q}(z | a_1,...,a_p; b_1,...,b_q) =
///   -Σ Res[integrand, s = b_h + k] for h=0..m-1, k=0,1,2,...
fn meijer_g_numeric(n: usize, m: usize, a: &[f64], b: &[f64], z: f64) -> f64 {
  let p = a.len();
  let q = b.len();

  // z = 0 special case
  if z == 0.0 {
    if m == 1 && b[0] == 0.0 && n == 0 && p == 0 {
      return 1.0;
    }
    return f64::NAN;
  }

  // Choose convergent series:
  // Left series (poles of Γ(b_j-s)) converges when q > p, or q = p and |z| < 1
  // Right series (poles of Γ(1-a_j+s)) converges when p > q, or p = q and |z| > 1
  // For right series, use the inversion formula: G^{m,n}_{p,q}(z) = G^{n,m}_{q,p}(1/z | ...)
  let use_left = if q > p {
    true
  } else if p > q {
    false
  } else {
    // p == q
    z.abs() <= 1.0
  };

  if use_left {
    meijer_g_direct_series(n, m, p, q, a, b, z)
  } else {
    // Apply inversion: G^{m,n}_{p,q}(z | a; b) = G^{n,m}_{q,p}(1/z | 1-b; 1-a)
    // The parameter order is preserved: new upper = {1-b_1,...,1-b_q}
    //                                   new lower = {1-a_1,...,1-a_p}
    // with new_n = m (first m upper params), new_m = n (first n lower params)
    let new_a: Vec<f64> = b.iter().map(|&bi| 1.0 - bi).collect();
    let new_b: Vec<f64> = a.iter().map(|&ai| 1.0 - ai).collect();
    let new_n = m;
    let new_m = n;
    let new_p = q;
    let new_q = p;
    meijer_g_direct_series(new_n, new_m, new_p, new_q, &new_a, &new_b, 1.0 / z)
  }
}

/// Direct residue series computation for MeijerG.
/// Computes residues numerically at each pole location, handling
/// coinciding poles and zero-pole cancellations automatically.
fn meijer_g_direct_series(
  n: usize,
  m: usize,
  p: usize,
  q: usize,
  a: &[f64],
  b: &[f64],
  z: f64,
) -> f64 {
  let max_terms = 500;

  // Evaluate the full MeijerG integrand at a point s (away from poles):
  // I(s) = ∏_{j<m} Γ(b_j-s) * ∏_{j<n} Γ(1-a_j+s) / ∏_{j≥n,j<p} Γ(a_j-s) / ∏_{j≥m,j<q} Γ(1-b_j+s) * z^s
  let eval_integrand = |s: f64| -> f64 {
    let mut val = z.powf(s);
    for j in 0..m {
      val *= gamma_fn(b[j] - s);
    }
    for j in 0..n {
      val *= gamma_fn(1.0 - a[j] + s);
    }
    for j in n..p {
      let g = gamma_fn(a[j] - s);
      if g.abs() < 1e-300 {
        return 0.0;
      }
      val /= g;
    }
    for j in m..q {
      let g = gamma_fn(1.0 - b[j] + s);
      if g.abs() < 1e-300 {
        return 0.0;
      }
      val /= g;
    }
    val
  };

  let mut total = 0.0;

  for h in 0..m {
    for k in 0..max_terms {
      let s0 = b[h] + k as f64;

      // Check if this pole location is already "owned" by a smaller h
      let mut already_counted = false;
      for j in 0..h {
        let diff = s0 - b[j];
        if diff >= -1e-14
          && (diff - diff.round()).abs() < 1e-10
          && diff.round() >= 0.0
        {
          already_counted = true;
          break;
        }
      }
      if already_counted {
        continue;
      }

      // Count apparent pole order from numerator Gamma functions
      let mut pole_order: usize = 0;
      for j in 0..m {
        let diff = s0 - b[j];
        if diff >= -1e-14 && (diff - diff.round()).abs() < 1e-10 {
          pole_order += 1;
        }
      }
      for j in 0..n {
        let arg = 1.0 - a[j] + s0;
        if arg <= 1e-14
          && (arg - arg.round()).abs() < 1e-10
          && arg.round() <= 0.0
        {
          pole_order += 1;
        }
      }

      // Count zeros from denominator 1/Γ functions
      let mut zero_order = 0;
      for j in n..p {
        let arg = a[j] - s0;
        if (arg - arg.round()).abs() < 1e-10 && arg.round() <= 0.0 {
          zero_order += 1;
        }
      }
      for j in m..q {
        let arg = 1.0 - b[j] + s0;
        if (arg - arg.round()).abs() < 1e-10 && arg.round() <= 0.0 {
          zero_order += 1;
        }
      }

      let effective_order = pole_order.saturating_sub(zero_order);

      if effective_order == 0 {
        continue; // no pole here
      }

      // Compute residue numerically using:
      // g(s) = (s-s₀)^{pole_order} * I(s)  [regularized integrand]
      // The effective pole is of order effective_order in g.
      // Residue of I at s₀ = g^{(pole_order-1)}(s₀) / (pole_order-1)!
      let res = meijer_g_numerical_residue(&eval_integrand, s0, pole_order);

      if !res.is_finite() || res.is_nan() {
        continue;
      }

      let prev_total = total;
      total -= res; // G = -Σ Res

      // Convergence check
      if k > 5 && res.abs() < 1e-14 * total.abs().max(1e-100) {
        break;
      }
      if k > 5
        && prev_total != 0.0
        && (total - prev_total).abs() < 1e-14 * total.abs().max(1e-100)
      {
        break;
      }
    }
  }

  total
}

/// Compute residue of f(s) at a pole of given apparent order using numerical differentiation.
/// Uses the regularized function g(s) = (s-s₀)^order * f(s).
/// Residue = g^{(order-1)}(s₀) / (order-1)!
///
/// IMPORTANT: We never evaluate g(s) at exactly s₀ because of 0*∞ issues.
/// Instead we use symmetric sample points offset from s₀.
fn meijer_g_numerical_residue(
  f: &dyn Fn(f64) -> f64,
  s0: f64,
  order: usize,
) -> f64 {
  let delta = 1e-4;

  let eval_g = |s: f64| -> f64 {
    let eps = s - s0;
    eps.powi(order as i32) * f(s)
  };

  let factorial = |n: usize| -> f64 {
    let mut fac = 1.0;
    for i in 2..=n {
      fac *= i as f64;
    }
    fac
  };

  if order == 1 {
    // Residue = g(s₀) = lim_{ε→0} ε * f(s₀+ε)
    // Use multiple points for Richardson extrapolation
    let g1 = eval_g(s0 + delta);
    let g2 = eval_g(s0 + delta / 2.0);
    let g3 = eval_g(s0 + delta / 4.0);
    // g should converge to the residue as δ→0
    // Use Richardson: 2*g2 - g1 (if linear error)
    let r1 = 2.0 * g2 - g1;
    let r2 = 2.0 * g3 - g2;
    // Second level Richardson
    let result = (4.0 * r2 - r1) / 3.0;
    if result.is_finite() { result } else { g3 }
  } else if order == 2 {
    // First derivative of g at s₀ using central difference (avoiding s₀)
    // g'(s₀) ≈ [g(s₀+δ) - g(s₀-δ)] / (2δ)
    // Use Richardson extrapolation for better accuracy
    let d1 = (eval_g(s0 + delta) - eval_g(s0 - delta)) / (2.0 * delta);
    let d2 = (eval_g(s0 + delta / 2.0) - eval_g(s0 - delta / 2.0)) / delta;
    // Richardson: (4*d2 - d1) / 3
    let result = (4.0 * d2 - d1) / 3.0;
    if result.is_finite() { result } else { d2 }
  } else {
    // Higher order: compute (order-1)-th derivative using finite differences
    // Use half-step offsets to avoid sampling at s₀ (where 0*∞ issues occur)
    let n_pts = order + 4;
    let values: Vec<f64> = (0..n_pts)
      .map(|i| {
        let s = s0 + (i as f64 - (n_pts as f64 - 1.0) / 2.0 + 0.5) * delta;
        eval_g(s)
      })
      .collect();

    let target_deriv = order - 1;
    let mut diff = values;
    for _ in 0..target_deriv {
      let new_len = diff.len() - 1;
      diff = (0..new_len)
        .map(|i| (diff[i + 1] - diff[i]) / delta)
        .collect();
    }

    let center = diff.len() / 2;
    diff[center] / factorial(target_deriv)
  }
}

/// StruveH[n, z] — Struve function H_n(z).
///
/// Series: H_n(z) = sum_{m=0}^{inf} (-1)^m / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_h_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StruveH expects exactly 2 arguments".into(),
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

  // Special case: StruveH[n, 0] for non-negative integer n => 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
    && *n >= 0
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = struve_h(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "StruveH".to_string(),
    args: args.to_vec(),
  })
}

/// Compute Struve H_n(z) using series expansion.
///
/// H_n(z) = sum_{m=0}^{inf} (-1)^m / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_h(n: f64, z: f64) -> f64 {
  // Special case: z = 0
  if z == 0.0 {
    if n >= -1.0 {
      return 0.0;
    }
    // For n < -1 with z=0, it may be divergent
    return f64::NAN;
  }

  let half_z = z / 2.0;

  // Series expansion
  let mut sum = 0.0;
  let gamma_3_2 = gamma_fn(1.5); // Gamma(3/2) = sqrt(pi)/2
  let first_gamma_denom = gamma_fn(n + 1.5);

  // First term (m=0): (z/2)^(n+1) / (Gamma(3/2) * Gamma(n + 3/2))
  let mut term = half_z.powf(n + 1.0) / (gamma_3_2 * first_gamma_denom);
  sum += term;

  for m in 1..300 {
    // Ratio of consecutive terms:
    // term_m / term_{m-1} = -half_z^2 / ((m + 0.5) * (m + n + 0.5))
    term *= -half_z * half_z / ((m as f64 + 0.5) * (m as f64 + n + 0.5));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }

  sum
}

/// StruveL[n, z] — Modified Struve function L_n(z).
///
/// Series: L_n(z) = sum_{m=0}^{inf} 1 / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_l_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "StruveL expects exactly 2 arguments".into(),
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

  // Special case: StruveL[n, 0] for non-negative integer n => 0
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = n_expr
    && *n >= 0
  {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = n_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(n_expr, Expr::Real(_)));

  if is_numeric_eval {
    let n = n_val.unwrap();
    let z = z_val.unwrap();
    let result = struve_l(n, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "StruveL".to_string(),
    args: args.to_vec(),
  })
}

/// Compute modified Struve L_n(z) using series expansion.
///
/// L_n(z) = sum_{m=0}^{inf} 1 / (Gamma(m + 3/2) * Gamma(m + n + 3/2)) * (z/2)^(2m+n+1)
pub fn struve_l(n: f64, z: f64) -> f64 {
  // Special case: z = 0
  if z == 0.0 {
    if n >= -1.0 {
      return 0.0;
    }
    // For n < -1 with z=0, it may be divergent
    return f64::NAN;
  }

  let half_z = z / 2.0;

  // Series expansion
  let mut sum = 0.0;
  let gamma_3_2 = gamma_fn(1.5); // Gamma(3/2) = sqrt(pi)/2
  let first_gamma_denom = gamma_fn(n + 1.5);

  // First term (m=0): (z/2)^(n+1) / (Gamma(3/2) * Gamma(n + 3/2))
  let mut term = half_z.powf(n + 1.0) / (gamma_3_2 * first_gamma_denom);
  sum += term;

  for m in 1..300 {
    // Ratio of consecutive terms (no alternating sign, unlike StruveH):
    // term_m / term_{m-1} = half_z^2 / ((m + 0.5) * (m + n + 0.5))
    term *= half_z * half_z / ((m as f64 + 0.5) * (m as f64 + n + 0.5));
    sum += term;
    if term.abs() < 1e-17 * sum.abs().max(1e-300) {
      break;
    }
  }

  sum
}

/// SquareWave[t] - square wave with period 1: +1 for frac(t) in [0,1/2), -1 for [1/2,1)
/// SquareWave[{d1, d2, ...}, t] - generalized multi-level square wave
pub fn square_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // SquareWave[t]
      if let Some(t) = expr_to_f64(&args[0]) {
        let frac = t - t.floor();
        if frac < 0.5 {
          return Ok(Expr::Integer(1));
        } else {
          return Ok(Expr::Integer(-1));
        }
      }
      // Exact rational input
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        let rem = n.rem_euclid(*d);
        // frac = rem/d, compare with 1/2 => 2*rem < d
        if 2 * rem < *d {
          return Ok(Expr::Integer(1));
        } else {
          return Ok(Expr::Integer(-1));
        }
      }
      Ok(Expr::FunctionCall {
        name: "SquareWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // SquareWave[{d1, d2, ...}, t]
      if let Expr::List(levels) = &args[0]
        && let Some(t) = expr_to_f64(&args[1])
      {
        let n = levels.len();
        if n == 0 {
          return Ok(Expr::Integer(0));
        }
        let frac = t - t.floor();
        let idx_raw = (frac * n as f64).floor() as usize;
        let idx = (n - 1).saturating_sub(idx_raw.min(n - 1));
        return Ok(levels[idx].clone());
      }
      Ok(Expr::FunctionCall {
        name: "SquareWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "SquareWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// TriangleWave[t] - triangle wave with period 1: linearly goes from 0 to 1 at t=1/4,
/// back to 0 at t=1/2, down to -1 at t=3/4, and back to 0 at t=1.
/// Formula: 4 * |frac(t + 3/4) - 1/2| - 1
/// TriangleWave[{min, max}, t] - scales output from [-1,1] to [min,max]
pub fn triangle_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Integer input: TriangleWave[n] = 0 for all integers
      if let Expr::Integer(_) = &args[0] {
        return Ok(Expr::Integer(0));
      }
      // Exact rational input
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        // Compute triangle wave for n/d exactly
        // shifted = n/d + 3/4 = (4n + 3d) / (4d)
        let num = 4 * n + 3 * d;
        let den = 4 * d;
        // frac = num mod den / den (using Euclidean remainder)
        let rem = num.rem_euclid(den);
        // val = 4 * |rem/den - 1/2| - 1 = 4 * |rem - den/2| / den - 1
        // = (4 * |2*rem - den|) / (2*den) - 1
        // = (2 * |2*rem - den| - den) / den
        let two_rem_minus_den = 2 * rem - den;
        let abs_val = two_rem_minus_den.abs();
        let result_num = 2 * abs_val - den;
        let result_den = den;
        // Simplify result_num / result_den
        let g = gcd(result_num, result_den);
        let sn = result_num / g;
        let sd = result_den / g;
        if sd == 1 {
          return Ok(Expr::Integer(sn));
        }
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sn), Expr::Integer(sd)],
        });
      }
      // Float input
      if let Some(t) = expr_to_f64(&args[0]) {
        let shifted = t + 0.75;
        let frac = shifted - shifted.floor();
        let val = 4.0 * (frac - 0.5).abs() - 1.0;
        return Ok(Expr::Real(val));
      }
      Ok(Expr::FunctionCall {
        name: "TriangleWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // TriangleWave[{min, max}, t]
      if let Expr::List(bounds) = &args[0]
        && bounds.len() == 2
      {
        // First compute base triangle wave value
        let base_args = [args[1].clone()];
        let base = triangle_wave_ast(&base_args)?;
        if let Some(v) = expr_to_f64(&base)
          && let (Some(lo), Some(hi)) =
            (expr_to_f64(&bounds[0]), expr_to_f64(&bounds[1]))
        {
          // Scale from [-1,1] to [min,max]: result = min + (max-min)*(v+1)/2
          let result = lo + (hi - lo) * (v + 1.0) / 2.0;
          return Ok(Expr::Real(result));
        }
      }
      Ok(Expr::FunctionCall {
        name: "TriangleWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "TriangleWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// SawtoothWave[x] - periodic sawtooth wave, fractional part of x
pub fn sawtooth_wave_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  match args.len() {
    1 => {
      // Integer input: SawtoothWave[n] = 0 for all integers
      if let Expr::Integer(_) = &args[0] {
        return Ok(Expr::Integer(0));
      }
      // Exact rational input: fractional part of n/d
      if let Expr::FunctionCall {
        name,
        args: rat_args,
      } = &args[0]
        && name == "Rational"
        && rat_args.len() == 2
        && let (Expr::Integer(n), Expr::Integer(d)) =
          (&rat_args[0], &rat_args[1])
      {
        // frac(n/d) = (n mod d) / d using Euclidean remainder
        let rem = n.rem_euclid(*d);
        if rem == 0 {
          return Ok(Expr::Integer(0));
        }
        let g = gcd(rem, *d);
        let sn = rem / g;
        let sd = d / g;
        if sd == 1 {
          return Ok(Expr::Integer(sn));
        }
        return Ok(Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(sn), Expr::Integer(sd)],
        });
      }
      // Float input
      if let Some(t) = expr_to_f64(&args[0]) {
        let frac = t - t.floor();
        return Ok(Expr::Real(frac));
      }
      Ok(Expr::FunctionCall {
        name: "SawtoothWave".to_string(),
        args: args.to_vec(),
      })
    }
    2 => {
      // SawtoothWave[{min, max}, t]
      if let Expr::List(bounds) = &args[0]
        && bounds.len() == 2
      {
        let base_args = [args[1].clone()];
        let base = sawtooth_wave_ast(&base_args)?;
        if let Some(v) = expr_to_f64(&base)
          && let (Some(lo), Some(hi)) =
            (expr_to_f64(&bounds[0]), expr_to_f64(&bounds[1]))
        {
          // Scale from [0,1] to [min,max]: result = min + (max-min)*v
          let result = lo + (hi - lo) * v;
          return Ok(Expr::Real(result));
        }
      }
      Ok(Expr::FunctionCall {
        name: "SawtoothWave".to_string(),
        args: args.to_vec(),
      })
    }
    _ => Err(InterpreterError::EvaluationError(
      "SawtoothWave expects 1 or 2 arguments".into(),
    )),
  }
}

/// ParabolicCylinderD[ν, z] - parabolic cylinder function D_ν(z)
pub fn parabolic_cylinder_d_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "ParabolicCylinderD expects exactly 2 arguments".into(),
    ));
  }

  // Numeric evaluation when both arguments are numeric
  if let (Some(nu), Some(z)) = (expr_to_f64(&args[0]), expr_to_f64(&args[1]))
    && (matches!(&args[0], Expr::Real(_)) || matches!(&args[1], Expr::Real(_)))
  {
    return Ok(Expr::Real(parabolic_cylinder_d(nu, z)));
  }

  Ok(Expr::FunctionCall {
    name: "ParabolicCylinderD".to_string(),
    args: args.to_vec(),
  })
}

/// Compute D_ν(z) using the relation to confluent hypergeometric functions:
/// D_ν(z) = 2^(ν/2) * exp(-z²/4) * [
///   √π / Γ((1-ν)/2) * 1F1(-ν/2, 1/2, z²/2)
///   - √2 * z / Γ(-ν/2) * 1F1((1-ν)/2, 3/2, z²/2)
/// ]
pub fn parabolic_cylinder_d(nu: f64, z: f64) -> f64 {
  use std::f64::consts::PI;
  let z2_half = z * z / 2.0;
  let prefactor = 2.0_f64.powf(nu / 2.0) * PI.sqrt() * (-z * z / 4.0).exp();

  let gamma_1 = gamma_fn((1.0 - nu) / 2.0);
  let gamma_2 = gamma_fn(-nu / 2.0);

  let term1 = if gamma_1.is_finite() && gamma_1.abs() > 1e-300 {
    hypergeometric_1f1(-nu / 2.0, 0.5, z2_half) / gamma_1
  } else {
    0.0
  };

  let term2 = if gamma_2.is_finite() && gamma_2.abs() > 1e-300 {
    -2.0_f64.sqrt() * z / gamma_2
      * hypergeometric_1f1((1.0 - nu) / 2.0, 1.5, z2_half)
  } else {
    0.0
  };

  prefactor * (term1 + term2)
}

/// AngerJ[nu, z] — Anger function.
///
/// For integer nu, AngerJ[n, z] = BesselJ[n, z].
/// For general nu, AngerJ[nu, z] = (1/Pi) * Integral[Cos[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn anger_j_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "AngerJ expects exactly 2 arguments".into(),
    ));
  }
  let nu_expr = &args[0];
  let z_expr = &args[1];

  // Extract numeric values
  let nu_val = match nu_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: AngerJ[n, 0] for integer n
  if matches!(z_expr, Expr::Integer(0))
    && let Expr::Integer(n) = nu_expr
  {
    return if *n == 0 {
      Ok(Expr::Integer(1))
    } else {
      Ok(Expr::Integer(0))
    };
  }

  // For integer nu, AngerJ[n, z] = BesselJ[n, z]
  if let Expr::Integer(_) = nu_expr {
    return bessel_j_ast(args);
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = nu_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(nu_expr, Expr::Real(_)));

  if is_numeric_eval {
    let nu = nu_val.unwrap();
    let z = z_val.unwrap();
    let result = anger_j(nu, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "AngerJ".to_string(),
    args: args.to_vec(),
  })
}

/// Compute AngerJ[nu, z] numerically using Gauss-Legendre quadrature.
///
/// AngerJ[nu, z] = (1/Pi) * Integral[Cos[nu*t - z*Sin[t]], {t, 0, Pi}]
/// For integer nu, this equals BesselJ[n, z].
pub fn anger_j(nu: f64, z: f64) -> f64 {
  // For integer nu, delegate to BesselJ
  if nu == nu.floor() && nu.is_finite() {
    return bessel_j(nu, z);
  }

  // For z = 0: AngerJ[nu, 0] = Sin[nu*Pi] / (nu*Pi)
  if z == 0.0 {
    let x = nu * std::f64::consts::PI;
    return x.sin() / x;
  }

  // Gauss-Legendre quadrature on [0, Pi]
  // Transform: t = (Pi/2) * (u + 1) where u in [-1, 1]
  gauss_legendre_anger_j(nu, z)
}

/// Gauss-Legendre quadrature for the Anger function integral.
fn gauss_legendre_anger_j(nu: f64, z: f64) -> f64 {
  // Use 64-point Gauss-Legendre quadrature
  // Nodes and weights for [-1, 1]
  let nodes_weights = gauss_legendre_64();
  let half_pi = std::f64::consts::PI / 2.0;
  let inv_pi = 1.0 / std::f64::consts::PI;

  let mut sum = 0.0;
  for &(node, weight) in &nodes_weights {
    let t = half_pi * (node + 1.0); // Map [-1,1] to [0, Pi]
    let integrand = (nu * t - z * t.sin()).cos();
    sum += weight * integrand;
  }
  sum * half_pi * inv_pi
}

/// WeberE[nu, z] — Weber function.
///
/// E_nu(z) = (1/Pi) * Integral[Sin[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn weber_e_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeberE expects exactly 2 arguments".into(),
    ));
  }
  let nu_expr = &args[0];
  let z_expr = &args[1];

  let nu_val = match nu_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let z_val = match z_expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };

  // Special case: WeberE[0, 0] = 0
  if matches!(z_expr, Expr::Integer(0)) && matches!(nu_expr, Expr::Integer(0)) {
    return Ok(Expr::Integer(0));
  }

  // Numeric evaluation when both args are numeric and at least one is Real
  let is_numeric_eval = nu_val.is_some()
    && z_val.is_some()
    && (matches!(z_expr, Expr::Real(_)) || matches!(nu_expr, Expr::Real(_)));

  if is_numeric_eval {
    let nu = nu_val.unwrap();
    let z = z_val.unwrap();
    let result = weber_e(nu, z);
    return Ok(Expr::Real(result));
  }

  // Return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeberE".to_string(),
    args: args.to_vec(),
  })
}

/// Compute WeberE[nu, z] numerically using Gauss-Legendre quadrature.
///
/// E_nu(z) = (1/Pi) * Integral[Sin[nu*t - z*Sin[t]], {t, 0, Pi}]
pub fn weber_e(nu: f64, z: f64) -> f64 {
  // For z = 0: WeberE[nu, 0] = (1 - Cos[nu*Pi]) / (nu*Pi)
  if z == 0.0 {
    if nu == 0.0 {
      return 0.0;
    }
    let x = nu * std::f64::consts::PI;
    return (1.0 - x.cos()) / x;
  }

  // Gauss-Legendre quadrature
  let nodes_weights = gauss_legendre_64();
  let half_pi = std::f64::consts::PI / 2.0;
  let inv_pi = 1.0 / std::f64::consts::PI;

  let mut sum = 0.0;
  for &(node, weight) in &nodes_weights {
    let t = half_pi * (node + 1.0);
    let integrand = (nu * t - z * t.sin()).sin();
    sum += weight * integrand;
  }
  sum * half_pi * inv_pi
}

/// WignerD[{j, m1, m2}, theta] — Wigner d-matrix element d^j_{m1,m2}(theta).
///
/// WignerD[{j, m1, m2}, phi, theta, psi] — full Wigner D-matrix element:
///   D^j_{m1,m2}(phi,theta,psi) = E^(-I*m1*phi) * d^j_{m1,m2}(theta) * E^(-I*m2*psi)
pub fn wigner_d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Supported forms:
  // WignerD[{j, m1, m2}, theta]
  // WignerD[{j, m1, m2}, phi, theta, psi]
  if args.len() != 2 && args.len() != 4 {
    return Ok(Expr::FunctionCall {
      name: "WignerD".to_string(),
      args: args.to_vec(),
    });
  }

  let (j_val, m1_val, m2_val) = match &args[0] {
    Expr::List(items) if items.len() == 3 => {
      let j = match try_eval_to_f64(&items[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let m1 = match try_eval_to_f64(&items[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let m2 = match try_eval_to_f64(&items[2]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
      };
      (j, m1, m2)
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WignerD".to_string(),
        args: args.to_vec(),
      });
    }
  };

  if args.len() == 2 {
    // WignerD[{j, m1, m2}, theta]
    let theta = match try_eval_to_f64(&args[1]) {
      Some(v) => v,
      None => {
        // Check if at least one is Real
        if !matches!(&args[1], Expr::Real(_)) {
          return Ok(Expr::FunctionCall {
            name: "WignerD".to_string(),
            args: args.to_vec(),
          });
        }
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let result = wigner_d_small(j_val, m1_val, m2_val, theta);
    Ok(Expr::Real(result))
  } else {
    // WignerD[{j, m1, m2}, phi, theta, psi]
    let phi = match try_eval_to_f64(&args[1]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let theta = match try_eval_to_f64(&args[2]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let psi = match try_eval_to_f64(&args[3]) {
      Some(v) => v,
      None => {
        return Ok(Expr::FunctionCall {
          name: "WignerD".to_string(),
          args: args.to_vec(),
        });
      }
    };
    let d = wigner_d_small(j_val, m1_val, m2_val, theta);
    // D = E^(-I*m1*phi) * d * E^(-I*m2*psi)
    // For real angles, this is: d * E^(-I*(m1*phi + m2*psi))
    // = d * (cos(m1*phi+m2*psi) - I*sin(m1*phi+m2*psi))
    let phase = m1_val * phi + m2_val * psi;
    let re = d * phase.cos();
    let im = -d * phase.sin();
    if im.abs() < 1e-15 {
      Ok(Expr::Real(re))
    } else {
      // Return Complex form
      Ok(Expr::FunctionCall {
        name: "Plus".to_string(),
        args: vec![
          Expr::Real(re),
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: vec![Expr::Real(im), Expr::Identifier("I".to_string())],
          },
        ],
      })
    }
  }
}

/// Compute the Wigner (small) d-matrix element d^j_{m1,m2}(theta).
fn wigner_d_small(j: f64, m1: f64, m2: f64, theta: f64) -> f64 {
  // Only handle half-integer/integer j, m1, m2
  let j2 = (2.0 * j).round() as i64;
  let m1_2 = (2.0 * m1).round() as i64;
  let m2_2 = (2.0 * m2).round() as i64;

  // Validate
  if (j2 as f64 - 2.0 * j).abs() > 1e-10
    || (m1_2 as f64 - 2.0 * m1).abs() > 1e-10
    || (m2_2 as f64 - 2.0 * m2).abs() > 1e-10
  {
    return f64::NAN;
  }

  if m1_2.abs() > j2 || m2_2.abs() > j2 {
    return 0.0;
  }

  // Use integer arithmetic for factorials
  // s ranges from max(0, m1-m2) to min(j+m1, j-m2)
  // Using half-integer labels: j+m1 = (j2+m1_2)/2, etc.
  let jpm1 = (j2 + m1_2) / 2;
  let jmm1 = (j2 - m1_2) / 2;
  let jpm2 = (j2 + m2_2) / 2;
  let jmm2 = (j2 - m2_2) / 2;
  let m1mm2 = (m1_2 - m2_2) / 2;

  let s_min = 0i64.max(m1mm2);
  let s_max = jpm1.min(jmm2);

  if s_min > s_max {
    return 0.0;
  }

  let half_theta = theta / 2.0;
  let cos_ht = half_theta.cos();
  let sin_ht = half_theta.sin();

  let prefactor = (factorial_f64(jpm1 as u64)
    * factorial_f64(jmm1 as u64)
    * factorial_f64(jpm2 as u64)
    * factorial_f64(jmm2 as u64))
  .sqrt();

  let mut sum = 0.0;
  for s in s_min..=s_max {
    let denom = factorial_f64((jpm1 - s) as u64)
      * factorial_f64(s as u64)
      * factorial_f64((s - m1mm2) as u64)
      * factorial_f64((jmm2 - s) as u64);

    // cos(theta/2)^(2j + m1 - m2 - 2s) * sin(theta/2)^(m2 - m1 + 2s)
    let cos_power = (2.0 * j + m1 - m2 - 2.0 * s as f64) as i64;
    let sin_power = (m2 - m1 + 2.0 * s as f64) as i64;

    let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
    let term =
      sign * cos_ht.powi(cos_power as i32) * sin_ht.powi(sin_power as i32)
        / denom;
    sum += term;
  }

  prefactor * sum
}

/// Factorial as f64 for moderate n.
fn factorial_f64(n: u64) -> f64 {
  if n <= 1 {
    return 1.0;
  }
  let mut result = 1.0;
  for i in 2..=n {
    result *= i as f64;
  }
  result
}

/// 64-point Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre_64() -> Vec<(f64, f64)> {
  // Compute nodes and weights using the Golub-Welsch algorithm
  let n = 64;
  let mut nodes = vec![0.0_f64; n];
  let mut weights = vec![0.0_f64; n];

  for i in 0..n {
    // Initial guess using Chebyshev nodes
    let mut x =
      -(std::f64::consts::PI * (4 * i + 3) as f64 / (4 * n + 2) as f64).cos();

    // Newton's method to find roots of P_n(x)
    for _ in 0..100 {
      let (p, dp) = legendre_p_and_deriv(n, x);
      let dx = -p / dp;
      x += dx;
      if dx.abs() < 1e-16 {
        break;
      }
    }
    nodes[i] = x;
    let (_, dp) = legendre_p_and_deriv(n, x);
    weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
  }

  nodes.into_iter().zip(weights).collect()
}

/// Evaluate Legendre polynomial P_n(x) and its derivative P_n'(x).
fn legendre_p_and_deriv(n: usize, x: f64) -> (f64, f64) {
  let mut p0 = 1.0;
  let mut p1 = x;
  let mut dp0 = 0.0;
  let mut dp1 = 1.0;

  for k in 1..n {
    let kf = k as f64;
    let p2 = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
    let dp2 = ((2.0 * kf + 1.0) * (p1 + x * dp1) - kf * dp0) / (kf + 1.0);
    p0 = p1;
    p1 = p2;
    dp0 = dp1;
    dp1 = dp2;
  }

  (p1, dp1)
}

/// NorlundB[n, a] - Nörlund generalized Bernoulli polynomial B_n^(a).
/// Computed via power series: (t/(e^t-1))^a = sum h_k t^k, B_n^(a) = n! * h_n.
/// Each h_k is a polynomial in a with rational coefficients.
pub fn norlund_b_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(Expr::FunctionCall {
      name: "NorlundB".to_string(),
      args: args.to_vec(),
    });
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "NorlundB".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let a_expr = &args[1];

  // Compute B_n^(a) as polynomial in a: vec of (num, den) pairs for a^0, a^1, ..., a^n
  let poly = norlund_b_poly(n);

  // If a is numeric, evaluate the polynomial
  if let Some(a_val) = try_eval_to_f64(a_expr) {
    // Check if a is an integer for exact computation
    if a_val == a_val.round() && a_val.abs() < 1e15 {
      let a_int = a_val as i128;
      // Evaluate polynomial at integer a using exact arithmetic
      let mut result_n: i128 = 0;
      let mut result_d: i128 = 1;
      let mut a_pow: i128 = 1; // a^k
      for &(cn, cd) in &poly {
        if cn != 0 {
          // result += cn/cd * a^k
          let term_n = cn.checked_mul(a_pow);
          if let Some(tn) = term_n {
            let new_n = result_n
              .checked_mul(cd)
              .and_then(|x| x.checked_add(tn.checked_mul(result_d)?));
            let new_d = result_d.checked_mul(cd);
            if let (Some(nn), Some(nd)) = (new_n, new_d) {
              let g = gcd(nn.abs(), nd.abs());
              result_n = nn / g;
              result_d = nd / g;
            } else {
              // Overflow - fall through to symbolic
              return evaluate_norlund_symbolic(&poly, a_expr);
            }
          } else {
            return evaluate_norlund_symbolic(&poly, a_expr);
          }
        }
        a_pow = match a_pow.checked_mul(a_int) {
          Some(v) => v,
          None => return evaluate_norlund_symbolic(&poly, a_expr),
        };
      }
      if result_d < 0 {
        result_n = -result_n;
        result_d = -result_d;
      }
      return Ok(crate::functions::math_ast::make_rational(
        result_n, result_d,
      ));
    }
  }

  // Symbolic case: build the polynomial expression
  evaluate_norlund_symbolic(&poly, a_expr)
}

/// Build the symbolic polynomial expression from coefficients.
fn evaluate_norlund_symbolic(
  poly: &[(i128, i128)],
  a: &Expr,
) -> Result<Expr, InterpreterError> {
  let mut terms: Vec<Expr> = Vec::new();
  for (k, &(cn, cd)) in poly.iter().enumerate() {
    if cn == 0 {
      continue;
    }
    let coeff = crate::functions::math_ast::make_rational(cn, cd);
    let term = if k == 0 {
      coeff
    } else {
      let a_pow = if k == 1 {
        a.clone()
      } else {
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![a.clone(), Expr::Integer(k as i128)],
        }
      };
      Expr::FunctionCall {
        name: "Times".to_string(),
        args: vec![coeff, a_pow],
      }
    };
    terms.push(term);
  }
  if terms.is_empty() {
    return Ok(Expr::Integer(0));
  }
  if terms.len() == 1 {
    return crate::evaluator::evaluate_expr_to_expr(&terms.pop().unwrap());
  }
  let sum = Expr::FunctionCall {
    name: "Plus".to_string(),
    args: terms,
  };
  crate::evaluator::evaluate_expr_to_expr(&sum)
}

/// Compute B_n^(a) as polynomial in a: returns vec of (num, den) for coefficients of a^0, a^1, ..., a^n.
/// Uses power series: f(t) = t/(e^t-1), h_k = coeff of t^k in f(t)^a, B_n^(a) = n! * h_n.
/// Recurrence: h_0 = 1, h_k = (1/k) * sum_{j=1}^{k} ((a+1)*j - k) * s_j * h_{k-j}
/// where s_j = B_j / j! (Bernoulli number divided by factorial).
fn norlund_b_poly(n: usize) -> Vec<(i128, i128)> {
  // s_j = B_j / j! as (numerator, denominator)
  let mut s: Vec<(i128, i128)> = Vec::with_capacity(n + 1);
  let mut factorial: i128 = 1;
  for j in 0..=n {
    if j > 0 {
      factorial *= j as i128;
    }
    if let Some((bn, bd)) = bernoulli_number(j) {
      let g = gcd(bn.abs(), factorial.abs());
      s.push((bn / g, bd * (factorial / g)));
    } else {
      s.push((0, 1));
    }
  }
  // Simplify s values
  for sval in &mut s {
    let g = gcd(sval.0.abs(), sval.1.abs());
    if g > 0 {
      sval.0 /= g;
      sval.1 /= g;
    }
    if sval.1 < 0 {
      sval.0 = -sval.0;
      sval.1 = -sval.1;
    }
  }

  // h_k is a polynomial in a of degree k, stored as Vec<(i128, i128)>
  // h_k[i] = coefficient of a^i
  let mut h: Vec<Vec<(i128, i128)>> = Vec::with_capacity(n + 1);
  h.push(vec![(1, 1)]); // h_0 = 1

  for k in 1..=n {
    // h_k = (1/k) * sum_{j=1}^{k} ((a+1)*j - k) * s_j * h_{k-j}
    //      = (1/k) * sum_{j=1}^{k} s_j * (j*a*h_{k-j} + (j-k)*h_{k-j})
    let max_deg = k;
    let mut hk = vec![(0i128, 1i128); max_deg + 1];

    for j in 1..=k {
      let (sn, sd) = s[j];
      if sn == 0 {
        continue;
      }
      let h_prev = &h[k - j];

      // Add s_j * (j-k) * h_{k-j} to hk (no shift in a)
      let scale = (j as i128) - (k as i128); // j - k
      if scale != 0 {
        for (i, &(pn, pd)) in h_prev.iter().enumerate() {
          if pn == 0 {
            continue;
          }
          // term = sn/sd * scale * pn/pd = (sn * scale * pn) / (sd * pd)
          let tn = sn * scale * pn;
          let td = sd * pd;
          rat_add_inplace(&mut hk[i], tn, td);
        }
      }

      // Add s_j * j * a * h_{k-j} to hk (shift by 1 in a)
      let j_val = j as i128;
      for (i, &(pn, pd)) in h_prev.iter().enumerate() {
        if pn == 0 {
          continue;
        }
        let tn = sn * j_val * pn;
        let td = sd * pd;
        rat_add_inplace(&mut hk[i + 1], tn, td);
      }
    }

    // Divide by k
    for coeff in &mut hk {
      coeff.1 *= k as i128;
      let g = gcd(coeff.0.abs(), coeff.1.abs());
      if g > 0 {
        coeff.0 /= g;
        coeff.1 /= g;
      }
      if coeff.1 < 0 {
        coeff.0 = -coeff.0;
        coeff.1 = -coeff.1;
      }
    }

    h.push(hk);
  }

  // B_n^(a) = n! * h_n
  let mut factorial: i128 = 1;
  for i in 1..=n {
    factorial *= i as i128;
  }
  let mut result = h[n].clone();
  for coeff in &mut result {
    coeff.0 *= factorial;
    let g = gcd(coeff.0.abs(), coeff.1.abs());
    if g > 0 {
      coeff.0 /= g;
      coeff.1 /= g;
    }
    if coeff.1 < 0 {
      coeff.0 = -coeff.0;
      coeff.1 = -coeff.1;
    }
  }
  result
}

/// Add rational tn/td to the value at *target in-place.
fn rat_add_inplace(target: &mut (i128, i128), tn: i128, td: i128) {
  let (rn, rd) = target;
  let new_n = *rn * td + tn * *rd;
  let new_d = *rd * td;
  let g = gcd(new_n.abs(), new_d.abs());
  if g > 0 {
    *rn = new_n / g;
    *rd = new_d / g;
  } else {
    *rn = new_n;
    *rd = new_d;
  }
  if *rd < 0 {
    *rn = -*rn;
    *rd = -*rd;
  }
}

/// AppellF1[a, b1, b2, c, x, y] - Appell hypergeometric function F1
/// F1(a, b1, b2; c; x, y) = Σ_{m,n≥0} (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
pub fn appell_f1_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 6 {
    return Err(InterpreterError::EvaluationError(
      "AppellF1 expects exactly 6 arguments".into(),
    ));
  }

  // Numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args.iter().map(expr_to_f64).collect();
  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a = vals[0].unwrap();
    let b1 = vals[1].unwrap();
    let b2 = vals[2].unwrap();
    let c = vals[3].unwrap();
    let x = vals[4].unwrap();
    let y = vals[5].unwrap();
    return Ok(Expr::Real(appell_f1_numeric(a, b1, b2, c, x, y)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "AppellF1".to_string(),
    args: args.to_vec(),
  })
}

/// Compute F1(a, b1, b2; c; x, y) using double series
fn appell_f1_numeric(a: f64, b1: f64, b2: f64, c: f64, x: f64, y: f64) -> f64 {
  // F1 = Σ_{m=0}^∞ Σ_{n=0}^∞ (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
  // We sum the outer m-loop, for each m summing over n
  let mut total = 0.0;

  // Pochhammer ratio terms for outer loop (m)
  let _a_m = 1.0; // (a)_m at current m... actually we need (a)_{m+n} which depends on n
  // Let's use a different approach: compute each term incrementally
  // term(m, n) = (a)_{m+n} (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
  // term(m, n+1) / term(m, n) = (a+m+n)(b2+n) / ((c+m+n)(n+1)) * y
  // term(m+1, n) / term(m, n) = (a+m+n)(b1+m) / ((c+m+n)(m+1)) * x

  // For each m, sum over n
  let mut coeff_m = 1.0; // (a)_m * (b1)_m / ((c)_m * m!) * x^m for the m-th outer term at n=0
  // But (a)_{m+n} = (a)_m * (a+m)_n, and (c)_{m+n} = (c)_m * (c+m)_n

  for m in 0..200 {
    // Inner sum over n for this m
    // term(m, n) = coeff_m * (a+m)_n * (b2)_n / ((c+m)_n * n!) * y^n
    let mut inner_sum = 1.0; // n=0 term is 1
    let mut coeff_n = 1.0;

    for n in 1..200 {
      coeff_n *= (a + m as f64 + n as f64 - 1.0) * (b2 + n as f64 - 1.0)
        / ((c + m as f64 + n as f64 - 1.0) * n as f64)
        * y;
      inner_sum += coeff_n;
      if coeff_n.abs() < 1e-16 * inner_sum.abs().max(1e-300) {
        break;
      }
    }

    total += coeff_m * inner_sum;

    // Update coeff_m for m+1
    if m < 199 {
      coeff_m *= (a + m as f64) * (b1 + m as f64)
        / ((c + m as f64) * (m as f64 + 1.0))
        * x;
      if coeff_m.abs() < 1e-16 * total.abs().max(1e-300) {
        break;
      }
    }
  }

  total
}

/// AppellF2[a, b1, b2, c1, c2, x, y] - Appell hypergeometric function F2
/// F2(a, b1, b2; c1, c2; x, y) = Σ_{m,n≥0} (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) x^m y^n
pub fn appell_f2_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 7 {
    return Err(InterpreterError::EvaluationError(
      "AppellF2 expects exactly 7 arguments".into(),
    ));
  }

  let a = &args[0];
  let b1 = &args[1];
  let b2 = &args[2];
  let c1 = &args[3];
  let c2 = &args[4];
  let x = &args[5];
  let y = &args[6];

  // a = 0 => 1
  if matches!(a, Expr::Integer(0)) || matches!(a, Expr::Real(v) if *v == 0.0) {
    return Ok(Expr::Integer(1));
  }

  // Check if x or y is zero
  let x_zero =
    matches!(x, Expr::Integer(0)) || matches!(x, Expr::Real(v) if *v == 0.0);
  let y_zero =
    matches!(y, Expr::Integer(0)) || matches!(y, Expr::Real(v) if *v == 0.0);

  // x = 0, y = 0 => 1
  if x_zero && y_zero {
    return Ok(Expr::Integer(1));
  }

  // b1 = 0 or x = 0 => Hypergeometric2F1[a, b2, c2, y]
  let b1_zero =
    matches!(b1, Expr::Integer(0)) || matches!(b1, Expr::Real(v) if *v == 0.0);
  if b1_zero || x_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a.clone(), b2.clone(), c2.clone(), y.clone()],
    );
  }

  // b2 = 0 or y = 0 => Hypergeometric2F1[a, b1, c1, x]
  let b2_zero =
    matches!(b2, Expr::Integer(0)) || matches!(b2, Expr::Real(v) if *v == 0.0);
  if b2_zero || y_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a.clone(), b1.clone(), c1.clone(), x.clone()],
    );
  }

  // Numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args.iter().map(expr_to_f64).collect();
  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a = vals[0].unwrap();
    let b1 = vals[1].unwrap();
    let b2 = vals[2].unwrap();
    let c1 = vals[3].unwrap();
    let c2 = vals[4].unwrap();
    let x = vals[5].unwrap();
    let y = vals[6].unwrap();
    return Ok(Expr::Real(appell_f2_numeric(a, b1, b2, c1, c2, x, y)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "AppellF2".to_string(),
    args: args.to_vec(),
  })
}

/// Compute F2(a, b1, b2; c1, c2; x, y) using double series
fn appell_f2_numeric(
  a: f64,
  b1: f64,
  b2: f64,
  c1: f64,
  c2: f64,
  x: f64,
  y: f64,
) -> f64 {
  // F2 = Σ_{m=0}^∞ Σ_{n=0}^∞ (a)_{m+n} (b1)_m (b2)_n / ((c1)_m (c2)_n m! n!) x^m y^n
  // term(m, n+1) / term(m, n) = (a+m+n)(b2+n) / ((c2+n)(n+1)) * y
  // For each m, the n=0 base term relative to (m-1, 0):
  //   coeff_m *= (a+m-1)(b1+m-1) / ((c1+m-1) * m) * x
  // But (a)_{m+n} = (a)_m * prod_{k=0..n-1}(a+m+k), so splitting:
  //   base_m = (a)_m (b1)_m / ((c1)_m m!) x^m
  //   inner_n = (a+m)_n (b2)_n / ((c2)_n n!) y^n

  let mut total = 0.0;
  let mut coeff_m = 1.0; // (a)_m (b1)_m / ((c1)_m m!) x^m

  for m in 0..200 {
    // Inner sum over n
    let mut inner_sum = 1.0; // n=0 term
    let mut coeff_n = 1.0;

    for n in 1..200 {
      coeff_n *= (a + m as f64 + n as f64 - 1.0) * (b2 + n as f64 - 1.0)
        / ((c2 + n as f64 - 1.0) * n as f64)
        * y;
      inner_sum += coeff_n;
      if coeff_n.abs() < 1e-16 * inner_sum.abs().max(1e-300) {
        break;
      }
    }

    total += coeff_m * inner_sum;

    // Update coeff_m for m+1
    if m < 199 {
      coeff_m *= (a + m as f64) * (b1 + m as f64)
        / ((c1 + m as f64) * (m as f64 + 1.0))
        * x;
      if coeff_m.abs() < 1e-16 * total.abs().max(1e-300) {
        break;
      }
    }
  }

  total
}

/// AppellF3[a1, a2, b1, b2, c, x, y] - Appell hypergeometric function F3
/// F3(a1, a2, b1, b2; c; x, y) = Σ_{m,n≥0} (a1)_m (a2)_n (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
pub fn appell_f3_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 7 {
    return Err(InterpreterError::EvaluationError(
      "AppellF3 expects exactly 7 arguments".into(),
    ));
  }

  let a1 = &args[0];
  let a2 = &args[1];
  let b1 = &args[2];
  let b2 = &args[3];
  let c = &args[4];
  let x = &args[5];
  let y = &args[6];

  let is_zero = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(v) if *v == 0.0)
  };

  let x_zero = is_zero(x);
  let y_zero = is_zero(y);

  // x = 0, y = 0 => 1
  if x_zero && y_zero {
    return Ok(Expr::Integer(1));
  }

  // a1 = 0 or b1 = 0 or x = 0 => Hypergeometric2F1[a2, b2, c, y]
  if is_zero(a1) || is_zero(b1) || x_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a2.clone(), b2.clone(), c.clone(), y.clone()],
    );
  }

  // a2 = 0 or b2 = 0 or y = 0 => Hypergeometric2F1[a1, b1, c, x]
  if is_zero(a2) || is_zero(b2) || y_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a1.clone(), b1.clone(), c.clone(), x.clone()],
    );
  }

  // Numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args.iter().map(expr_to_f64).collect();
  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a1 = vals[0].unwrap();
    let a2 = vals[1].unwrap();
    let b1 = vals[2].unwrap();
    let b2 = vals[3].unwrap();
    let c = vals[4].unwrap();
    let x = vals[5].unwrap();
    let y = vals[6].unwrap();
    return Ok(Expr::Real(appell_f3_numeric(a1, a2, b1, b2, c, x, y)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "AppellF3".to_string(),
    args: args.to_vec(),
  })
}

/// Compute F3(a1, a2, b1, b2; c; x, y) using double series
fn appell_f3_numeric(
  a1: f64,
  a2: f64,
  b1: f64,
  b2: f64,
  c: f64,
  x: f64,
  y: f64,
) -> f64 {
  // F3 = Σ_{m=0}^∞ Σ_{n=0}^∞ (a1)_m (a2)_n (b1)_m (b2)_n / ((c)_{m+n} m! n!) x^m y^n
  // The m-loop base (n=0): (a1)_m (b1)_m / ((c)_m m!) x^m
  // For each m, inner n: ratio = (a2+n-1)(b2+n-1) / ((c+m+n-1) n) * y

  let mut total = 0.0;
  let mut coeff_m = 1.0; // (a1)_m (b1)_m / ((c)_m m!) x^m

  for m in 0..200 {
    // Inner sum over n
    let mut inner_sum = 1.0;
    let mut coeff_n = 1.0;

    for n in 1..200 {
      coeff_n *= (a2 + n as f64 - 1.0) * (b2 + n as f64 - 1.0)
        / ((c + m as f64 + n as f64 - 1.0) * n as f64)
        * y;
      inner_sum += coeff_n;
      if coeff_n.abs() < 1e-16 * inner_sum.abs().max(1e-300) {
        break;
      }
    }

    total += coeff_m * inner_sum;

    // Update coeff_m for m+1
    if m < 199 {
      coeff_m *= (a1 + m as f64) * (b1 + m as f64)
        / ((c + m as f64) * (m as f64 + 1.0))
        * x;
      if coeff_m.abs() < 1e-16 * total.abs().max(1e-300) {
        break;
      }
    }
  }

  total
}

/// AppellF4[a, b, c1, c2, x, y] - Appell hypergeometric function F4
/// F4(a, b; c1, c2; x, y) = Σ_{m,n≥0} (a)_{m+n} (b)_{m+n} / ((c1)_m (c2)_n m! n!) x^m y^n
pub fn appell_f4_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 6 {
    return Err(InterpreterError::EvaluationError(
      "AppellF4 expects exactly 6 arguments".into(),
    ));
  }

  let a = &args[0];
  let b = &args[1];
  let c1 = &args[2];
  let c2 = &args[3];
  let x = &args[4];
  let y = &args[5];

  let is_zero = |e: &Expr| -> bool {
    matches!(e, Expr::Integer(0)) || matches!(e, Expr::Real(v) if *v == 0.0)
  };

  // a = 0 or b = 0 => 1
  if is_zero(a) || is_zero(b) {
    return Ok(Expr::Integer(1));
  }

  let x_zero = is_zero(x);
  let y_zero = is_zero(y);

  // x = 0, y = 0 => 1
  if x_zero && y_zero {
    return Ok(Expr::Integer(1));
  }

  // x = 0 => Hypergeometric2F1[a, b, c2, y]
  if x_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a.clone(), b.clone(), c2.clone(), y.clone()],
    );
  }

  // y = 0 => Hypergeometric2F1[a, b, c1, x]
  if y_zero {
    return crate::evaluator::evaluate_function_call_ast(
      "Hypergeometric2F1",
      &[a.clone(), b.clone(), c1.clone(), x.clone()],
    );
  }

  // Numeric evaluation when all args are numeric and at least one is Real
  let vals: Vec<Option<f64>> = args.iter().map(expr_to_f64).collect();
  let has_real = args.iter().any(|a| matches!(a, Expr::Real(_)));

  if vals.iter().all(|v| v.is_some()) && has_real {
    let a = vals[0].unwrap();
    let b = vals[1].unwrap();
    let c1 = vals[2].unwrap();
    let c2 = vals[3].unwrap();
    let x = vals[4].unwrap();
    let y = vals[5].unwrap();
    return Ok(Expr::Real(appell_f4_numeric(a, b, c1, c2, x, y)));
  }

  // Unevaluated
  Ok(Expr::FunctionCall {
    name: "AppellF4".to_string(),
    args: args.to_vec(),
  })
}

/// Compute F4(a, b; c1, c2; x, y) using double series
fn appell_f4_numeric(a: f64, b: f64, c1: f64, c2: f64, x: f64, y: f64) -> f64 {
  // F4 = Σ_{m=0}^∞ Σ_{n=0}^∞ (a)_{m+n} (b)_{m+n} / ((c1)_m (c2)_n m! n!) x^m y^n
  // Base at n=0: (a)_m (b)_m / ((c1)_m m!) x^m
  // Inner ratio n→n+1: (a+m+n)(b+m+n) / ((c2+n)(n+1)) * y

  let mut total = 0.0;
  let mut coeff_m = 1.0;

  for m in 0..200 {
    let mut inner_sum = 1.0;
    let mut coeff_n = 1.0;

    for n in 1..200 {
      coeff_n *= (a + m as f64 + n as f64 - 1.0)
        * (b + m as f64 + n as f64 - 1.0)
        / ((c2 + n as f64 - 1.0) * n as f64)
        * y;
      inner_sum += coeff_n;
      if coeff_n.abs() < 1e-16 * inner_sum.abs().max(1e-300) {
        break;
      }
    }

    total += coeff_m * inner_sum;

    if m < 199 {
      coeff_m *= (a + m as f64) * (b + m as f64)
        / ((c1 + m as f64) * (m as f64 + 1.0))
        * x;
      if coeff_m.abs() < 1e-16 * total.abs().max(1e-300) {
        break;
      }
    }
  }

  total
}

/// PolygonalNumber[n] - nth triangular number = n*(n+1)/2
/// PolygonalNumber[r, n] - nth r-gonal number = n*((r-2)*n - r + 4) / 2
pub fn polygonal_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (r, n) = match args.len() {
    1 => (Expr::Integer(3), args[0].clone()),
    2 => (args[0].clone(), args[1].clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "PolygonalNumber expects 1 or 2 arguments".into(),
      ));
    }
  };

  // Rewrite to: n * ((r - 2) * n - r + 4) / 2
  // Evaluate: Times[n, Plus[Times[Plus[r, -2], n], Times[-1, r], 4], Power[2, -1]]
  let r_minus_2 = crate::evaluator::evaluate_function_call_ast(
    "Plus",
    &[r.clone(), Expr::Integer(-2)],
  )?;
  let r_minus_2_times_n = crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[r_minus_2, n.clone()],
  )?;
  let neg_r = crate::evaluator::evaluate_function_call_ast(
    "Times",
    &[Expr::Integer(-1), r],
  )?;
  let inner = crate::evaluator::evaluate_function_call_ast(
    "Plus",
    &[r_minus_2_times_n, neg_r, Expr::Integer(4)],
  )?;
  let half = crate::evaluator::evaluate_function_call_ast(
    "Power",
    &[Expr::Integer(2), Expr::Integer(-1)],
  )?;
  crate::evaluator::evaluate_function_call_ast("Times", &[n, inner, half])
}

/// PerfectNumber[n] - gives the nth perfect number
/// Perfect numbers are 2^(p-1) * (2^p - 1) where 2^p - 1 is a Mersenne prime.
pub fn perfect_number_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "PerfectNumber expects exactly 1 argument".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 1 => *n as usize,
    Expr::Integer(_) | Expr::Real(_) => {
      return Ok(Expr::FunctionCall {
        name: "PerfectNumber".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PerfectNumber".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Known Mersenne prime exponents (sufficient for the first 51 known perfect numbers)
  let mersenne_exponents: &[u32] = &[
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657,
    37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
  ];

  if n > mersenne_exponents.len() {
    return Ok(Expr::FunctionCall {
      name: "PerfectNumber".to_string(),
      args: args.to_vec(),
    });
  }

  let p = mersenne_exponents[n - 1];

  // Compute 2^(p-1) * (2^p - 1) using BigInt
  let two = BigInt::from(2);
  let two_p = two.pow(p);
  let two_p_minus_1 = &two_p - BigInt::from(1);
  let two_p_minus_1_exp = two.pow(p - 1);
  let perfect = two_p_minus_1_exp * two_p_minus_1;

  // Try to fit in i128, otherwise use BigInteger
  use num_traits::ToPrimitive;
  if let Some(val) = perfect.to_i128() {
    Ok(Expr::Integer(val))
  } else {
    Ok(Expr::BigInteger(perfect))
  }
}

/// RamanujanTau[n] - Ramanujan tau function
/// τ(n) is the coefficient of q^n in q * ∏_{k=1}^∞ (1-q^k)^24
pub fn ramanujan_tau_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "RamanujanTau expects exactly 1 argument".into(),
    ));
  }

  match &args[0] {
    Expr::Integer(n) => {
      let n = *n;
      if n <= 0 {
        return Ok(Expr::Integer(0));
      }
      let n = n as usize;
      let tau = ramanujan_tau_compute(n);
      Ok(Expr::Integer(tau))
    }
    _ => Ok(Expr::FunctionCall {
      name: "RamanujanTau".to_string(),
      args: args.to_vec(),
    }),
  }
}

/// Compute τ(n) by expanding q * ∏_{k=1}^{n} (1-q^k)^24 as a power series
/// and extracting the coefficient of q^n.
fn ramanujan_tau_compute(n: usize) -> i128 {
  // We need coefficients of q * ∏(1-q^k)^24 up to q^n
  // Start with polynomial [1] (constant 1), multiply by (1-q^k)^24
  // for k = 1, 2, ..., n, tracking coefficients up to degree n-1
  // (since the final result has an extra factor of q, shifting by 1).
  //
  // Actually: Δ(q) = q * ∏_{k≥1} (1-q^k)^24
  // So coeff of q^n in Δ = coeff of q^{n-1} in ∏_{k≥1} (1-q^k)^24

  let target = n - 1; // We need the (n-1)th coeff of the product
  let mut coeffs = vec![0i128; target + 1];
  coeffs[0] = 1;

  // Multiply by (1-q^k)^24 for k = 1..n
  // (1-q^k)^24 = Σ_{j=0}^{24} C(24,j) (-1)^j q^{jk}
  let binom24: Vec<i128> = (0..=24)
    .map(|j| {
      let mut c: i128 = 1;
      for i in 0..j {
        c = c * (24 - i as i128) / (i as i128 + 1);
      }
      if j % 2 == 1 { -c } else { c }
    })
    .collect();

  for k in 1..=n {
    // Multiply coeffs by (1-q^k)^24
    // Process in place from high to low degree
    for i in (0..=target).rev() {
      let mut sum = 0i128;
      for (j, &bj) in binom24.iter().enumerate().skip(1) {
        let deg = j * k;
        if deg > i {
          break;
        }
        sum += bj * coeffs[i - deg];
      }
      coeffs[i] += sum;
    }
  }

  coeffs[target]
}

/// PowersRepresentations[n, k, p] gives all representations of n as a sum of
/// k non-negative integers each raised to the power p.
pub fn powers_representations_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Err(InterpreterError::EvaluationError(
      "PowersRepresentations expects 3 arguments".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(v) => *v,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PowersRepresentations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let k = match &args[1] {
    Expr::Integer(v) if *v >= 0 => *v as usize,
    Expr::Integer(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PowersRepresentations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let p = match &args[2] {
    Expr::Integer(v) if *v >= 1 => *v as u32,
    Expr::Integer(_) => {
      return Ok(Expr::FunctionCall {
        name: "PowersRepresentations".to_string(),
        args: args.to_vec(),
      });
    }
    _ => {
      return Ok(Expr::FunctionCall {
        name: "PowersRepresentations".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Negative numbers have no representations as sums of powers
  if n < 0 {
    return Ok(Expr::List(vec![]));
  }

  let n = n as u128;

  let mut results: Vec<Vec<u128>> = Vec::new();
  let mut current: Vec<u128> = Vec::new();
  powers_rep_search(n, k, p, 0, &mut current, &mut results);

  // Convert to Expr
  let expr_results: Vec<Expr> = results
    .into_iter()
    .map(|rep| {
      Expr::List(rep.into_iter().map(|v| Expr::Integer(v as i128)).collect())
    })
    .collect();

  Ok(Expr::List(expr_results))
}

/// Recursive search for power representations.
/// Find all non-decreasing sequences of `remaining` non-negative integers,
/// each >= `min_val`, whose `p`-th powers sum to `target`.
fn powers_rep_search(
  target: u128,
  remaining: usize,
  p: u32,
  min_val: u128,
  current: &mut Vec<u128>,
  results: &mut Vec<Vec<u128>>,
) {
  if remaining == 0 {
    if target == 0 {
      results.push(current.clone());
    }
    return;
  }

  // Maximum value that could contribute
  // val^p <= target, so val <= target^(1/p)
  let max_val = if target == 0 {
    0
  } else {
    // Integer p-th root via binary search
    let mut lo = min_val;
    let mut hi = if p == 1 {
      target
    } else if p == 2 {
      (target as f64).sqrt() as u128 + 2
    } else {
      (target as f64).powf(1.0 / p as f64) as u128 + 2
    };
    while lo < hi {
      let mid = lo + (hi - lo).div_ceil(2);
      if let Some(power) = mid.checked_pow(p) {
        if power <= target {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      } else {
        // Overflow means too large
        hi = mid - 1;
      }
    }
    lo
  };

  let start = min_val;
  let mut val = start;
  loop {
    if val > max_val {
      break;
    }
    let power = match val.checked_pow(p) {
      Some(v) if v <= target => v,
      _ => break,
    };
    // Pruning: remaining values are all >= val, so minimum sum is
    // remaining * val^p. If that exceeds target, stop.
    if let Some(min_remaining_sum) = power.checked_mul(remaining as u128)
      && min_remaining_sum > target
    {
      break;
    }
    current.push(val);
    powers_rep_search(target - power, remaining - 1, p, val, current, results);
    current.pop();
    val += 1;
  }
}
