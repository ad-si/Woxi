use super::weierstrass_lattice::{WeierstrassKind, weierstrass_complex};
#[allow(unused_imports)]
use super::*;

// Format a `Power::infy` warning in wolframscript's 2D layout, e.g.
// `0^-3` renders as:
//     "                                  -3\n\
//      Power::infy: Infinite expression 0   encountered."
pub(crate) fn format_power_infy_2d(base: &str, exp: &str) -> String {
  const PREFIX: &str = "Power::infy: Infinite expression ";
  let line1_width = PREFIX.len() + base.len() + exp.len();
  let trailing_spaces = " ".repeat(exp.len() + 1);
  format!("{exp:>line1_width$}\n{PREFIX}{base}{trailing_spaces}encountered.")
}

/// True if any of the given expressions contains an inexact (machine) number,
/// in which case the Weierstrass functions numericize; otherwise they stay
/// symbolic (matching wolframscript; `N[...]` makes the arguments inexact).
fn any_inexact(exprs: &[&Expr]) -> bool {
  exprs.iter().any(|e| contains_inexact_real(e))
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
      return Ok(unevaluated("WeierstrassP", args));
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole at origin). wolframscript
  // computes ℘ via 1/u² internally, so it emits a `Power::infy` warning
  // for the 0⁻² that arises along the way.
  if is_expr_zero(u) {
    crate::emit_message(&format_power_infy_2d("0", "-2"));
    return Ok(id_expr("ComplexInfinity"));
  }

  // Numeric evaluation only when an argument is inexact; exact arguments stay
  // symbolic, matching wolframscript.
  if any_inexact(&[u, g2, g3])
    && let (Some(u_f), Some(g2_f), Some(g3_f)) =
      (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }
  if let Some(value) = complex_weierstrass(WeierstrassKind::P, u, g2, g3) {
    return Ok(value);
  }

  // Symbolic: return unevaluated
  Ok(unevaluated("WeierstrassP", args))
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
      return Ok(unevaluated("WeierstrassPPrime", args));
    }
  };

  // Special case: u = 0 → ComplexInfinity (pole of order 3 at origin).
  // wolframscript's ℘′ implementation hits 0⁻³ along the way, so emit the
  // matching `Power::infy` warning.
  if is_expr_zero(u) {
    crate::emit_message(&format_power_infy_2d("0", "-3"));
    return Ok(id_expr("ComplexInfinity"));
  }

  // Numeric evaluation only when an argument is inexact; exact arguments stay
  // symbolic, matching wolframscript.
  if any_inexact(&[u, g2, g3])
    && let (Some(u_f), Some(g2_f), Some(g3_f)) =
      (try_eval_to_f64(u), try_eval_to_f64(g2), try_eval_to_f64(g3))
  {
    let result = weierstrass_p_prime_numeric(u_f, g2_f, g3_f);
    return Ok(Expr::Real(result));
  }
  if let Some(value) = complex_weierstrass(WeierstrassKind::PPrime, u, g2, g3) {
    return Ok(value);
  }

  // Symbolic: return unevaluated
  Ok(unevaluated("WeierstrassPPrime", args))
}

/// Compute WeierstrassPPrime numerically via central difference of WeierstrassP
fn weierstrass_p_prime_numeric(u: f64, g2: f64, g3: f64) -> f64 {
  // ℘'(z)² = 4℘³ - g2℘ - g3, so the magnitude follows directly from the
  // (accurate) value of ℘. A coarse central difference only fixes the sign,
  // which keeps full precision instead of relying on the difference's value.
  let p = weierstrass_p_numeric(u, g2, g3);
  let magnitude = (4.0 * p * p * p - g2 * p - g3).max(0.0).sqrt();
  let h = u.abs().max(1.0) * 1e-6;
  let slope =
    weierstrass_p_numeric(u + h, g2, g3) - weierstrass_p_numeric(u - h, g2, g3);
  if slope < 0.0 { -magnitude } else { magnitude }
}

/// Compute WeierstrassP numerically using cubic roots + Jacobi elliptic functions
fn weierstrass_p_numeric(u: f64, g2: f64, g3: f64) -> f64 {
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
fn weierstrass_p_laurent(u: f64, g2: f64, g3: f64) -> f64 {
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
    // Laurent recurrence (DLMF 23.9.3, re-indexed for c[k] = coeff of z^(2k)):
    // c[k] = 3 / ((2k+3)(k-2)) * Σ_{j=1}^{k-2} c[j] c[k-1-j].  The denominator
    // factor is (k-2), not (k-1): e.g. c[3] = g2²/1200, not g2²/2400.
    c[k] = 3.0 / ((2 * k + 3) as f64 * (k - 2) as f64) * sum;
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

/// InverseWeierstrassP[p, {g2, g3}] — inverse Weierstrass elliptic function:
/// the u with ℘(u; g₂, g₃) = p that WL returns.
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
      return Ok(unevaluated("InverseWeierstrassP", args));
    }
  };

  // Two forms:
  // 1. InverseWeierstrassP[p, {g2, g3}] → u with ℘(u) = p
  // 2. InverseWeierstrassP[{p, pp}, {g2, g3}] → u with ℘(u) = p, ℘'(u) = pp
  match &args[0] {
    Expr::List(items) if items.len() == 2 => {
      // Form 2: {p, pp} given
      let (Some(g2_f), Some(g3_f)) = (try_eval_to_f64(g2), try_eval_to_f64(g3))
      else {
        return Ok(unevaluated("InverseWeierstrassP", args));
      };
      let Some(p_f) = try_eval_to_f64(&items[0]) else {
        return Ok(unevaluated("InverseWeierstrassP", args));
      };
      let Some(pp_f) = try_eval_to_f64(&items[1]) else {
        return Ok(unevaluated("InverseWeierstrassP", args));
      };
      let u = inverse_weierstrass_p_with_prime(p_f, pp_f, g2_f, g3_f);
      Ok(Expr::Real(u))
    }
    _ => {
      // Form 1: just p given; numeric once an argument is inexact.
      if !any_inexact(&[&args[0], g2, g3]) {
        // ℘ has a double pole at 0: ℘^-1(0) for g2 = g3 = 0.
        if is_expr_zero(&args[0]) && is_expr_zero(g2) && is_expr_zero(g3) {
          return Ok(Expr::Identifier("ComplexInfinity".to_string()));
        }
        return Ok(unevaluated("InverseWeierstrassP", args));
      }
      let (Some(p), Some(a), Some(b)) = (
        try_extract_complex_float(&args[0]),
        try_extract_complex_float(g2),
        try_extract_complex_float(g3),
      ) else {
        return Ok(unevaluated("InverseWeierstrassP", args));
      };
      match super::weierstrass_lattice::inverse_weierstrass_p_complex(p, a, b) {
        Some((re, im)) => Ok(build_complex_float_expr(re, im)),
        None => Ok(unevaluated("InverseWeierstrassP", args)),
      }
    }
  }
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

/// A Weierstrass function at a machine-number point with machine invariants
/// (real or complex), through the lattice's theta function. `None` unless
/// some argument is inexact and all are numbers. Real arguments give a real
/// result.
fn complex_weierstrass(
  kind: WeierstrassKind,
  u: &Expr,
  g2: &Expr,
  g3: &Expr,
) -> Option<Expr> {
  if !any_inexact(&[u, g2, g3]) {
    return None;
  }
  let (z, a, b) = (
    try_extract_complex_float(u)?,
    try_extract_complex_float(g2)?,
    try_extract_complex_float(g3)?,
  );
  let (re, im) = weierstrass_complex(kind, z, a, b)?;
  if z.1 == 0.0 && a.1 == 0.0 && b.1 == 0.0 {
    return Some(Expr::Real(re));
  }
  Some(build_complex_float_expr(re, im))
}

/// WeierstrassZeta[u, {g2, g3}] and WeierstrassSigma[u, {g2, g3}] — numeric
/// for machine-number arguments, symbolic otherwise (`u = 0` gives the
/// exact 0).
pub fn weierstrass_zeta_sigma_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (g2, g3) = match &args[1] {
    Expr::List(items) if items.len() == 2 => (&items[0], &items[1]),
    _ => return Ok(unevaluated(name, args)),
  };
  let kind = if name == "WeierstrassZeta" {
    WeierstrassKind::Zeta
  } else {
    WeierstrassKind::Sigma
  };
  // ζ has its pole 1/u at the origin, where σ vanishes.
  if is_expr_zero(&args[0]) {
    return match kind {
      WeierstrassKind::Zeta => {
        crate::evaluator::evaluate_expr_to_expr(&pow2(
          args[0].clone(),
          Expr::Integer(-1),
        ))?;
        Ok(id_expr("ComplexInfinity"))
      }
      _ => Ok(args[0].clone()),
    };
  }
  if let Some(value) = complex_weierstrass(kind, &args[0], g2, g3) {
    return Ok(value);
  }
  Ok(unevaluated(name, args))
}
