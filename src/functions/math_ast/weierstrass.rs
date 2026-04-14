#[allow(unused_imports)]
use super::*;
use crate::InterpreterError;
use crate::syntax::Expr;

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

/// WeierstrassPPrime[u, {g2, g3}] - Derivative of Weierstrass elliptic function ℘'(u; g₂, g₃)
pub fn weierstrass_p_prime_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "WeierstrassPPrime expects exactly 2 arguments".into(),
    ));
  }

  let u = &args[0];
  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "WeierstrassPPrime".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole of order 3 at origin)
  if is_expr_zero(u) {
    return Ok(Expr::Identifier("ComplexInfinity".to_string()));
  }

  // Numeric evaluation
  if let (Some(u_f), Some(g2_f), Some(g3_f)) =
    (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_prime_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }

  // Symbolic: return unevaluated
  Ok(Expr::FunctionCall {
    name: "WeierstrassPPrime".to_string(),
    args: args.to_vec(),
  })
}

/// Compute WeierstrassPPrime numerically via central difference of WeierstrassP
pub fn weierstrass_p_prime_numeric(u: f64, g2: f64, g3: f64) -> f64 {
  // Use central difference with step size chosen for good accuracy
  let h = u.abs().max(1.0) * 1e-6;
  let p_plus = weierstrass_p_numeric(u + h, g2, g3);
  let p_minus = weierstrass_p_numeric(u - h, g2, g3);
  (p_plus - p_minus) / (2.0 * h)
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
    // One real root or degenerate — use duplication formula with Laurent series
    // The Laurent series converges only for small |u|, so we halve u
    // repeatedly until it's small enough, then apply the duplication formula.
    weierstrass_p_duplication(u, g2, g3)
  }
}

/// Compute ℘(u) using repeated argument halving + duplication formula.
/// The duplication formula is: ℘(2z) = -2℘(z) + (6℘(z)² - g₂/2)² / (4(4℘(z)³ - g₂℘(z) - g₃))
fn weierstrass_p_duplication(u: f64, g2: f64, g3: f64) -> f64 {
  // Halve u until it's small enough for the Laurent series to converge well
  let threshold = 0.3;
  let mut halvings = 0u32;
  let mut z = u;
  while z.abs() > threshold && halvings < 60 {
    z /= 2.0;
    halvings += 1;
  }

  // Compute ℘(z) using Laurent series (converges well for small z)
  let mut wp = weierstrass_p_laurent(z, g2, g3);

  // Apply duplication formula to double back: ℘(2z) from ℘(z)
  for _ in 0..halvings {
    let wp2 = wp * wp;
    let wp3 = wp2 * wp;
    let denom = 4.0 * wp3 - g2 * wp - g3; // = ℘'(z)²
    if denom.abs() < 1e-300 {
      return f64::INFINITY;
    }
    let num = 6.0 * wp2 - g2 / 2.0; // = ℘''(z)
    wp = -2.0 * wp + num * num / (4.0 * denom);
  }

  wp
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

/// InverseWeierstrassP[p, {g2, g3}] — inverse Weierstrass elliptic function.
///
/// Returns {u, ℘'(u)} where ℘(u; g₂, g₃) = p.
///
/// InverseWeierstrassP[{p, pp}, {g2, g3}] — returns u given both p and p'.
pub fn inverse_weierstrass_p_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "InverseWeierstrassP expects exactly 2 arguments".into(),
    ));
  }

  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Try to get g2, g3 as f64
  let g2_f = match try_eval_to_f64(g2) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };
  let g3_f = match try_eval_to_f64(g3) {
    Some(v) => v,
    None => {
      return Ok(Expr::FunctionCall {
        name: "InverseWeierstrassP".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Two forms:
  // 1. InverseWeierstrassP[p, {g2, g3}] where p is a number → {u, p'}
  // 2. InverseWeierstrassP[{p, pp}, {g2, g3}] → u
  match &args[0] {
    Expr::List(items) if items.len() == 2 => {
      // Form 2: {p, pp} given
      let p_f = match try_eval_to_f64(&items[0]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let pp_f = match try_eval_to_f64(&items[1]) {
        Some(v) => v,
        None => {
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      let u = inverse_weierstrass_p_with_prime(p_f, pp_f, g2_f, g3_f);
      Ok(Expr::Real(u))
    }
    _ => {
      // Form 1: just p given
      let p_f = match try_eval_to_f64(&args[0]) {
        Some(v) => v,
        None => {
          // Check if it's purely symbolic
          if !matches!(&args[0], Expr::Real(_))
            && !matches!(&args[0], Expr::Integer(_))
          {
            return Ok(Expr::FunctionCall {
              name: "InverseWeierstrassP".to_string(),
              args: args.to_vec(),
            });
          }
          return Ok(Expr::FunctionCall {
            name: "InverseWeierstrassP".to_string(),
            args: args.to_vec(),
          });
        }
      };
      // Need at least one Real for numeric eval
      let has_real = matches!(&args[0], Expr::Real(_))
        || matches!(g2, Expr::Real(_))
        || matches!(g3, Expr::Real(_));
      if !has_real {
        return Ok(Expr::FunctionCall {
          name: "InverseWeierstrassP".to_string(),
          args: args.to_vec(),
        });
      }
      let (u, pp) = inverse_weierstrass_p_numeric(p_f, g2_f, g3_f);
      Ok(Expr::List(vec![Expr::Real(u), Expr::Real(pp)]))
    }
  }
}

/// Compute InverseWeierstrassP[p, {g2, g3}] numerically.
/// Returns (u, p') where ℘(u) = p.
/// Uses Newton's method starting from an initial guess.
fn inverse_weierstrass_p_numeric(p: f64, g2: f64, g3: f64) -> (f64, f64) {
  // ℘'² = 4℘³ - g₂℘ - g₃
  let pp_sq = 4.0 * p * p * p - g2 * p - g3;
  let pp = if pp_sq >= 0.0 {
    -pp_sq.sqrt() // Take the negative branch (principal value, small positive u)
  } else {
    // Complex case — for now just use the magnitude
    0.0
  };
  let u = inverse_weierstrass_p_with_prime(p, pp, g2, g3);
  let actual_pp = weierstrass_p_prime_numeric(u, g2, g3);
  (u, actual_pp)
}

/// Compute u given p, p', g2, g3 such that ℘(u) = p and ℘'(u) = pp.
/// Uses Newton's method: u_{n+1} = u_n - (℘(u_n) - p) / ℘'(u_n)
fn inverse_weierstrass_p_with_prime(p: f64, pp: f64, g2: f64, g3: f64) -> f64 {
  // Initial guess: for large p, u ≈ 1/√p (from Laurent series ℘ ≈ 1/u²)
  let mut u = if p.abs() > 1.0 {
    let sign = if pp < 0.0 { 1.0 } else { -1.0 };
    sign / p.abs().sqrt()
  } else {
    // For smaller p, start with a moderate guess
    let sign = if pp < 0.0 { 1.0 } else { -1.0 };
    sign * 0.5
  };

  // Newton's method
  for _ in 0..200 {
    let wp = weierstrass_p_numeric(u, g2, g3);
    let wpp = weierstrass_p_prime_numeric(u, g2, g3);

    if wpp.abs() < 1e-300 {
      break;
    }

    let delta = (wp - p) / wpp;
    u -= delta;

    if delta.abs() < 1e-14 * u.abs().max(1e-14) {
      break;
    }
  }

  u
}
