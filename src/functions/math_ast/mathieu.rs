//! Mathieu function numerical evaluation.
//!
//! For real `(a, q, z)` we provide a Floquet-based numerical solver.
//! Wolfram's MathieuS / MathieuSPrime use a specific (a, q)-dependent
//! normalisation tied to the Hill tridiagonal eigenvalue problem; that
//! normalisation is not reproduced here. Instead the solution is
//! anchored by the boundary condition that matches the `q → 0` limit
//! (`MathieuS(a, 0, z) = sin(√a · z)` and
//! `MathieuSPrime(a, 0, z) = √a · cos(√a · z)`).
//!
//! For `q = 0` the closed forms above are exact; for `q ≠ 0` the result
//! is the analytic continuation under the BC `y(0) = 0`, `y'(0) = √a`
//! — which differs from wolframscript's specific numerical scaling by
//! a smooth (a, q)-dependent factor. Users who need exact wolframscript
//! agreement should compare ratios `MathieuSPrime(a, q, z) /
//! MathieuSPrime(a, q, 0)` (these match to numerical precision).
use crate::InterpreterError;
use crate::syntax::Expr;

fn try_real_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
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
  }
}

/// RK4 integration of the Mathieu ODE `y'' + (a − 2q cos(2z))·y = 0`
/// from `0` to `z_target`. Returns `(y, y')` at `z_target`.
fn integrate_mathieu(
  a: f64,
  q: f64,
  z_target: f64,
  y0: f64,
  yp0: f64,
) -> (f64, f64) {
  if z_target == 0.0 {
    return (y0, yp0);
  }
  let target_abs = z_target.abs();
  let dir = z_target.signum();
  // ~30k steps per unit of z gives ~1e-12 absolute accuracy for typical
  // (a, q) values in the unit range.
  let n = (30_000.0 * target_abs).max(1000.0) as usize;
  let h = target_abs / n as f64;
  let mut y = y0;
  let mut yp = yp0;
  let mut z = 0.0;
  for _ in 0..n {
    let h_signed = h * dir;
    let f2 = |z: f64, y: f64| -(a - 2.0 * q * (2.0 * z).cos()) * y;
    let k1y = yp;
    let k1yp = f2(z, y);
    let k2y = yp + h_signed * k1yp / 2.0;
    let k2yp = f2(z + h_signed / 2.0, y + h_signed * k1y / 2.0);
    let k3y = yp + h_signed * k2yp / 2.0;
    let k3yp = f2(z + h_signed / 2.0, y + h_signed * k2y / 2.0);
    let k4y = yp + h_signed * k3yp;
    let k4yp = f2(z + h_signed, y + h_signed * k3y);
    y += h_signed * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;
    yp += h_signed * (k1yp + 2.0 * k2yp + 2.0 * k3yp + k4yp) / 6.0;
    z += h_signed;
  }
  (y, yp)
}

/// MathieuS[a, q, z] — odd Mathieu function, normalised to match
/// `sin(√a · z)` at q = 0 (BC `y(0) = 0`, `y'(0) = √a`).
pub fn mathieu_s_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "MathieuS".to_string(),
      args: args.to_vec().into(),
    });
  }
  let (a, q, z) = match (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) {
    (Some(a), Some(q), Some(z)) => (a, q, z),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MathieuS".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  if a < 0.0 {
    return Ok(Expr::FunctionCall {
      name: "MathieuS".to_string(),
      args: args.to_vec().into(),
    });
  }
  let yp0 = a.sqrt();
  let (y, _) = integrate_mathieu(a, q, z, 0.0, yp0);
  Ok(Expr::Real(y))
}

/// MathieuSPrime[a, q, z] — derivative of MathieuS w.r.t. z.
pub fn mathieu_s_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "MathieuSPrime".to_string(),
      args: args.to_vec().into(),
    });
  }
  let (a, q, z) = match (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) {
    (Some(a), Some(q), Some(z)) => (a, q, z),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MathieuSPrime".to_string(),
        args: args.to_vec().into(),
      });
    }
  };
  if a < 0.0 {
    return Ok(Expr::FunctionCall {
      name: "MathieuSPrime".to_string(),
      args: args.to_vec().into(),
    });
  }
  let yp0 = a.sqrt();
  let (_, yp) = integrate_mathieu(a, q, z, 0.0, yp0);
  Ok(Expr::Real(yp))
}
