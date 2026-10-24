//! Mathieu function numerical evaluation.
//!
//! For real `(a, q, z)` we provide a Floquet-based numerical solver.
//! Wolfram's Mathieu functions use a specific (a, q)-dependent
//! normalisation tied to the Hill tridiagonal eigenvalue problem; that
//! normalisation is not reproduced here. Instead each solution is
//! anchored by the boundary condition that matches its `q → 0` limit:
//! `MathieuS(a, 0, z) = sin(√a · z)`, `MathieuSPrime(a, 0, z) = √a ·
//! cos(√a · z)`, `MathieuC(a, 0, z) = cos(√a · z)`, and
//! `MathieuCPrime(a, 0, z) = -√a · sin(√a · z)`.
//!
//! For `q = 0` the closed forms above are exact; for `q ≠ 0` the result
//! is the analytic continuation under the respective boundary condition
//! (`y(0) = 0, y'(0) = √a` for the S family; `y(0) = 1, y'(0) = 0` for
//! the C family) — which differs from wolframscript's specific
//! numerical scaling by a smooth (a, q)-dependent factor. Users who
//! need exact wolframscript agreement should compare ratios such as
//! `MathieuSPrime(a, q, z) / MathieuSPrime(a, q, 0)` (these match to
//! numerical precision).

#[allow(unused_imports)]
use super::*;

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

/// Value `(y, y')` at `z = i·t` of the solution of the Mathieu ODE with
/// `y(0) = y0`, `y'(0) = yp0`. With `Y(t) = y(i·t)` the ODE becomes
/// `Y'' = (a − 2q·cosh(2t))·Y`, which is integrated with RK4 like the real
/// case. Returns `(Y(t), Y'(t))`, so `y(i·t) = Y(t)` and `y'(i·t) = −i·Y'(t)`.
fn integrate_mathieu_imaginary(
  a: f64,
  q: f64,
  t_target: f64,
  y0: f64,
  yp0: f64,
) -> (f64, f64) {
  if t_target == 0.0 {
    return (y0, yp0);
  }
  let dir = t_target.signum();
  let n = (30_000.0 * t_target.abs()).max(1000.0) as usize;
  let h = t_target.abs() / n as f64 * dir;
  let (mut y, mut yp, mut t) = (y0, yp0, 0.0);
  let f2 = |t: f64, y: f64| (a - 2.0 * q * (2.0 * t).cosh()) * y;
  for _ in 0..n {
    let k1y = yp;
    let k1yp = f2(t, y);
    let k2y = yp + h * k1yp / 2.0;
    let k2yp = f2(t + h / 2.0, y + h * k1y / 2.0);
    let k3y = yp + h * k2yp / 2.0;
    let k3yp = f2(t + h / 2.0, y + h * k2y / 2.0);
    let k4y = yp + h * k3yp;
    let k4yp = f2(t + h, y + h * k3y);
    y += h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;
    yp += h * (k1yp + 2.0 * k2yp + 2.0 * k3yp + k4yp) / 6.0;
    t += h;
  }
  (y, yp)
}

/// The imaginary part `t` of an argument of the form `i·t` (machine
/// precision, zero real part), as used for the Mathieu functions of
/// imaginary argument.
fn pure_imaginary_f64(e: &Expr) -> Option<f64> {
  if matches!(e, Expr::Integer(_) | Expr::Real(_)) {
    return None;
  }
  let (re, im) = try_extract_complex_f64(e)?;
  (re == 0.0 && im != 0.0).then_some(im)
}

/// MathieuS[a, q, z] — odd Mathieu function, normalised to match
/// `sin(√a · z)` at q = 0 (BC `y(0) = 0`, `y'(0) = √a`).
pub fn mathieu_s_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("MathieuS", args));
  }
  if let (Some(a), Some(q), Some(t)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    pure_imaginary_f64(&args[2]),
  ) && a >= 0.0
  {
    // y(i·t) = Y(t) with Y(0) = 0 and Y'(0) = i·√a, so Y = i·(real solution).
    let (y, _) = integrate_mathieu_imaginary(a, q, t, 0.0, a.sqrt());
    return Ok(build_complex_float_expr(0.0, y));
  }
  let (Some(a), Some(q), Some(z)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) else {
    return Ok(unevaluated("MathieuS", args));
  };
  if a < 0.0 {
    return Ok(unevaluated("MathieuS", args));
  }
  let yp0 = a.sqrt();
  let (y, _) = integrate_mathieu(a, q, z, 0.0, yp0);
  Ok(Expr::Real(y))
}

/// MathieuSPrime[a, q, z] — derivative of MathieuS w.r.t. z.
pub fn mathieu_s_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("MathieuSPrime", args));
  }
  let (Some(a), Some(q), Some(z)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) else {
    return Ok(unevaluated("MathieuSPrime", args));
  };
  if a < 0.0 {
    return Ok(unevaluated("MathieuSPrime", args));
  }
  let yp0 = a.sqrt();
  let (_, yp) = integrate_mathieu(a, q, z, 0.0, yp0);
  Ok(Expr::Real(yp))
}

/// MathieuC[a, q, z] — even Mathieu function, normalised to match
/// `cos(√a · z)` at q = 0 (BC `y(0) = 1`, `y'(0) = 0`).
pub fn mathieu_c_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("MathieuC", args));
  }
  if let (Some(a), Some(q), Some(t)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    pure_imaginary_f64(&args[2]),
  ) {
    let (y, _) = integrate_mathieu_imaginary(a, q, t, 1.0, 0.0);
    return Ok(Expr::Real(y));
  }
  let (Some(a), Some(q), Some(z)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) else {
    return Ok(unevaluated("MathieuC", args));
  };
  // Unlike MathieuS, the y(0) = 1, y'(0) = 0 boundary condition is real for
  // any real `a` (it never involves `√a`), so negative `a` is fine here.
  let (y, _) = integrate_mathieu(a, q, z, 1.0, 0.0);
  Ok(Expr::Real(y))
}

/// MathieuCPrime[a, q, z] — derivative of MathieuC w.r.t. z.
pub fn mathieu_c_prime_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(unevaluated("MathieuCPrime", args));
  }
  let (Some(a), Some(q), Some(z)) = (
    try_real_f64(&args[0]),
    try_real_f64(&args[1]),
    try_real_f64(&args[2]),
  ) else {
    return Ok(unevaluated("MathieuCPrime", args));
  };
  // Same real boundary condition as MathieuC, so negative `a` is fine here.
  let (_, yp) = integrate_mathieu(a, q, z, 1.0, 0.0);
  Ok(Expr::Real(yp))
}

/// Number of eigenvalues below `x` of the symmetric tridiagonal matrix with
/// diagonal `d` and squared off-diagonal entries `e2` (Sturm sequence).
fn sturm_count_below(d: &[f64], e2: &[f64], x: f64) -> usize {
  let mut count = 0;
  let mut q = 1.0;
  for i in 0..d.len() {
    let off = if i == 0 { 0.0 } else { e2[i - 1] / q };
    q = d[i] - x - off;
    if q == 0.0 {
      q = f64::EPSILON;
    }
    if q < 0.0 {
      count += 1;
    }
  }
  count
}

/// The `k`-th smallest (0-based) eigenvalue of a symmetric tridiagonal
/// matrix, found by bisection on the Sturm count.
fn tridiagonal_eigenvalue(d: &[f64], e2: &[f64], k: usize) -> f64 {
  let n = d.len();
  let radius = |i: usize| {
    let l = if i > 0 { e2[i - 1].sqrt() } else { 0.0 };
    let r = if i + 1 < n { e2[i].sqrt() } else { 0.0 };
    l + r
  };
  let mut lo = (0..n)
    .map(|i| d[i] - radius(i))
    .fold(f64::INFINITY, f64::min);
  let mut hi = (0..n)
    .map(|i| d[i] + radius(i))
    .fold(f64::NEG_INFINITY, f64::max);
  for _ in 0..200 {
    let mid = 0.5 * (lo + hi);
    if mid == lo || mid == hi {
      break;
    }
    if sturm_count_below(d, e2, mid) > k {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  0.5 * (lo + hi)
}

/// Characteristic value of the Mathieu equation for integer order `r`:
/// `a_r(q)` (cosine-type) or `b_r(q)` (sine-type, `r ≥ 1`), as an eigenvalue
/// of the truncated Hill matrix of the matching Fourier basis.
fn mathieu_characteristic(r: usize, q: f64, sine: bool) -> f64 {
  let k = r / 2;
  // Enough Fourier terms for the eigenvalue to converge: the basis must
  // reach well past the turning point at roughly √(2|q|).
  let size = k + 40 + (3.0 * q.abs().sqrt()) as usize;
  let odd = r % 2 == 1;
  // For sine-type even order b_{2k+2} the basis starts at sin(2z).
  let shift = usize::from(odd) + if sine && !odd { 2 } else { 0 };
  let mut d: Vec<f64> = (0..size)
    .map(|i| {
      let m = (2 * i + shift) as f64;
      m * m
    })
    .collect();
  let mut e2 = vec![q * q; size - 1];
  match (sine, odd) {
    (false, false) => e2[0] = 2.0 * q * q,
    (false, true) => d[0] += q,
    (true, true) => d[0] -= q,
    (true, false) => {}
  }
  let index = if sine && !odd { k - 1 } else { k };
  tridiagonal_eigenvalue(&d, &e2, index)
}

fn mathieu_characteristic_ast(
  name: &str,
  sine: bool,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated(name, args));
  }
  let (Expr::Integer(r), Some(q)) = (&args[0], try_real_f64(&args[1])) else {
    return Ok(unevaluated(name, args));
  };
  // Only machine-precision `q` is evaluated numerically; exact `q` stays
  // symbolic, except for the trivial q = 0 where a_r = r² (b_r likewise).
  if matches!(args[1], Expr::Integer(0)) && *r >= i128::from(sine) {
    return Ok(Expr::Integer(r * r));
  }
  if !matches!(args[1], Expr::Real(_)) || *r < i128::from(sine) {
    return Ok(unevaluated(name, args));
  }
  Ok(Expr::Real(mathieu_characteristic(*r as usize, q, sine)))
}

/// MathieuCharacteristicA[r, q] — characteristic value `a_r(q)`.
pub fn mathieu_characteristic_a_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  mathieu_characteristic_ast("MathieuCharacteristicA", false, args)
}

/// MathieuCharacteristicB[r, q] — characteristic value `b_r(q)`.
pub fn mathieu_characteristic_b_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  mathieu_characteristic_ast("MathieuCharacteristicB", true, args)
}
