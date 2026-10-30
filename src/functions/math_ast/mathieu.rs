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
  // Beyond one period the equation's π-periodicity takes over: with Φ the
  // fundamental matrix and M = Φ(π), Φ(kπ + r) = Φ(r) M^k, so only the
  // remainder r ∈ [0, π) is integrated (a plot along z = 35 + 3x no longer
  // integrates thirty-odd periods per point).
  let pi = std::f64::consts::PI;
  if z_target.abs() > pi {
    let k = (z_target / pi).floor();
    let r = z_target - k * pi;
    let m = mathieu_monodromy(a, q);
    let mut power = if k >= 0.0 {
      m
    } else {
      // det M = 1 (the Wronskian), so the inverse is the adjugate.
      [[m[1][1], -m[0][1]], [-m[1][0], m[0][0]]]
    };
    let mut n = k.abs() as u64;
    let (mut v0, mut v1) = (y0, yp0);
    while n > 0 {
      if n & 1 == 1 {
        (v0, v1) = (
          power[0][0] * v0 + power[0][1] * v1,
          power[1][0] * v0 + power[1][1] * v1,
        );
      }
      power = [
        [
          power[0][0] * power[0][0] + power[0][1] * power[1][0],
          power[0][0] * power[0][1] + power[0][1] * power[1][1],
        ],
        [
          power[1][0] * power[0][0] + power[1][1] * power[1][0],
          power[1][0] * power[0][1] + power[1][1] * power[1][1],
        ],
      ];
      n >>= 1;
    }
    return integrate_mathieu_span(a, q, r, v0, v1);
  }
  integrate_mathieu_span(a, q, z_target, y0, yp0)
}

/// The monodromy matrix Φ(π) of the Mathieu ODE, cached per `(a, q)`.
fn mathieu_monodromy(a: f64, q: f64) -> [[f64; 2]; 2] {
  thread_local! {
    static CACHE: std::cell::RefCell<std::collections::HashMap<(u64, u64), [[f64; 2]; 2]>> =
      std::cell::RefCell::new(std::collections::HashMap::new());
  }
  let key = (a.to_bits(), q.to_bits());
  if let Some(m) = CACHE.with(|c| c.borrow().get(&key).copied()) {
    return m;
  }
  let pi = std::f64::consts::PI;
  let (y1, y1p) = integrate_mathieu_span(a, q, pi, 1.0, 0.0);
  let (y2, y2p) = integrate_mathieu_span(a, q, pi, 0.0, 1.0);
  let m = [[y1, y2], [y1p, y2p]];
  CACHE.with(|c| {
    let mut c = c.borrow_mut();
    if c.len() > 256 {
      c.clear();
    }
    c.insert(key, m);
  });
  m
}

/// RK4 over `[0, z_target]` (either sign).
fn integrate_mathieu_span(
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
    let mid = f64::midpoint(lo, hi);
    if mid == lo || mid == hi {
      break;
    }
    if sturm_count_below(d, e2, mid) > k {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  f64::midpoint(lo, hi)
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

fn mathieu_characteristic_ast(name: &str, sine: bool, args: &[Expr]) -> Expr {
  if args.len() != 2 {
    return unevaluated(name, args);
  }
  let (Expr::Integer(r), Some(q)) = (&args[0], try_real_f64(&args[1])) else {
    return unevaluated(name, args);
  };
  // Only machine-precision `q` is evaluated numerically; exact `q` stays
  // symbolic, except for the trivial q = 0 where a_r = r² (b_r likewise).
  if matches!(args[1], Expr::Integer(0)) && *r >= i128::from(sine) {
    return Expr::Integer(r * r);
  }
  if !matches!(args[1], Expr::Real(_)) || *r < i128::from(sine) {
    return unevaluated(name, args);
  }
  Expr::Real(mathieu_characteristic(*r as usize, q, sine))
}

/// MathieuCharacteristicA[r, q] — characteristic value `a_r(q)`.
pub fn mathieu_characteristic_a_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(mathieu_characteristic_ast(
    "MathieuCharacteristicA",
    false,
    args,
  ))
}

/// MathieuCharacteristicB[r, q] — characteristic value `b_r(q)`.
pub fn mathieu_characteristic_b_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(mathieu_characteristic_ast(
    "MathieuCharacteristicB",
    true,
    args,
  ))
}

/// The rotation number of the Mathieu ODE: the mean rate at which the
/// Prüfer angle φ of a solution (`y = r sin φ`, `y' = r cos φ`, so
/// `φ' = cos²φ + p sin²φ` with `p = a − 2q cos 2z`) turns, over `periods`
/// periods of π. It equals the real part of the characteristic exponent to
/// within `1/periods`.
fn mathieu_rotation_number(a: f64, q: f64, periods: usize) -> f64 {
  let per_period = 400.0 * ((a.abs() + 2.0 * q.abs()).sqrt() + 1.0);
  let n = (per_period as usize).max(200) * periods;
  let span = std::f64::consts::PI * periods as f64;
  let h = span / n as f64;
  let f = |z: f64, phi: f64| {
    let p = a - 2.0 * q * (2.0 * z).cos();
    let (s, c) = phi.sin_cos();
    c * c + p * s * s
  };
  let (mut phi, mut z) = (0.0, 0.0);
  for _ in 0..n {
    let k1 = f(z, phi);
    let k2 = f(z + h / 2.0, phi + h * k1 / 2.0);
    let k3 = f(z + h / 2.0, phi + h * k2 / 2.0);
    let k4 = f(z + h, phi + h * k3);
    phi += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    z += h;
  }
  phi / span
}

/// Half the monodromy trace, `(y₁(π) + y₂'(π))/2`, of the Mathieu ODE.
/// The potential is even, so it is `2 y₁(π/2) y₂'(π/2) − 1` (the
/// Wronskian is 1); each half-period integration is a fine-stepped RK4.
fn mathieu_half_trace(a: f64, q: f64) -> f64 {
  let half = std::f64::consts::FRAC_PI_2;
  let omega = (a.abs() + 2.0 * q.abs()).sqrt() + 1.0;
  let n = ((3000.0 * omega) as usize).clamp(2000, 200_000);
  let rk = |steps: usize, y0: f64, yp0: f64| -> (f64, f64) {
    let h = half / steps as f64;
    let f = |z: f64, y: f64| -(a - 2.0 * q * (2.0 * z).cos()) * y;
    let (mut y, mut yp) = (y0, yp0);
    for i in 0..steps {
      // The node itself, not a running sum, keeps the step points exact.
      let z = i as f64 * h;
      let k1y = yp;
      let k1p = f(z, y);
      let k2y = yp + h * k1p / 2.0;
      let k2p = f(z + h / 2.0, y + h * k1y / 2.0);
      let k3y = yp + h * k2p / 2.0;
      let k3p = f(z + h / 2.0, y + h * k2y / 2.0);
      let k4y = yp + h * k3p;
      let k4p = f(z + h, y + h * k3y);
      y += h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;
      yp += h * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0;
    }
    (y, yp)
  };
  let trace_at = |steps: usize| {
    let (y1, _) = rk(steps, 1.0, 0.0);
    let (_, y2p) = rk(steps, 0.0, 1.0);
    2.0 * y1 * y2p - 1.0
  };
  let coarse = trace_at(n);
  let fine = trace_at(2 * n);
  (16.0 * fine - coarse) / 15.0
}

/// `MathieuCharacteristicExponent[a, q]` for real machine `a` and `q`: the
/// Floquet exponent ν with `cos πν = (y₁(π) + y₂'(π))/2` from the
/// fundamental solutions, as `(Re ν, Im ν)`. A stable `(a, q)` has a real
/// ν ≥ 0, picked among `2m ± θ` by the rotation number; an unstable one
/// is `k + iμ`, its integer part `k` of the parity the trace's sign gives.
pub fn mathieu_characteristic_exponent(a: f64, q: f64) -> (f64, f64) {
  let pi = std::f64::consts::PI;
  let half_trace = mathieu_half_trace(a, q);
  if half_trace.abs() <= 1.0 {
    let theta = half_trace.acos() / pi;
    let gap = theta.min(1.0 - theta).max(1e-3);
    let periods = ((4.0 / gap).ceil() as usize).clamp(8, 400);
    let estimate = mathieu_rotation_number(a, q, periods);
    // Candidates 2m ± θ around the estimate.
    let m = (estimate / 2.0).round();
    let nu = [
      2.0 * m - theta,
      2.0 * m + theta,
      2.0 * m + 2.0 - theta,
      2.0 * m - 2.0 + theta,
    ]
    .into_iter()
    .filter(|v| *v >= 0.0)
    .min_by(|x, y| (x - estimate).abs().total_cmp(&(y - estimate).abs()))
    .unwrap_or(theta);
    (nu, 0.0)
  } else {
    let mu = half_trace.abs().acosh() / pi;
    let estimate = mathieu_rotation_number(a, q, 8);
    // Even k for a positive trace, odd for a negative one.
    let parity = if half_trace > 0.0 { 0.0 } else { 1.0 };
    let k = 2.0 * ((estimate - parity) / 2.0).round() + parity;
    (k.max(parity), mu)
  }
}

pub fn mathieu_characteristic_exponent_ast(args: &[Expr]) -> Expr {
  let unevaluated =
    || crate::helpers::unevaluated("MathieuCharacteristicExponent", args);
  let [a_e, q_e] = args else {
    return unevaluated();
  };
  // q = 0: the exponent is Sqrt[a] exactly (I Sqrt[-a] for a < 0).
  if matches!(q_e, Expr::Integer(0))
    && !crate::functions::math_ast::contains_inexact_real(a_e)
    && let Some(a) = try_real_f64(a_e)
  {
    let root = crate::evaluator::evaluate_function_call_ast(
      "Sqrt",
      std::slice::from_ref(a_e),
    );
    if a >= 0.0 {
      return root.unwrap_or_else(|_| unevaluated());
    }
    return crate::evaluator::evaluate_function_call_ast(
      "Times",
      &[
        Expr::Identifier("I".to_string()),
        crate::evaluator::evaluate_function_call_ast(
          "Sqrt",
          &[crate::evaluator::evaluate_function_call_ast(
            "Minus",
            std::slice::from_ref(a_e),
          )
          .unwrap_or_else(|_| unevaluated())],
        )
        .unwrap_or_else(|_| unevaluated()),
      ],
    )
    .unwrap_or_else(|_| unevaluated());
  }
  // Otherwise a machine number with real parameters.
  if !(crate::functions::math_ast::contains_inexact_real(a_e)
    || crate::functions::math_ast::contains_inexact_real(q_e))
  {
    return unevaluated();
  }
  let (Some(a), Some(q)) = (try_real_f64(a_e), try_real_f64(q_e)) else {
    return unevaluated();
  };
  let (re, im) = mathieu_characteristic_exponent(a, q);
  if im == 0.0 {
    Expr::Real(re)
  } else {
    crate::functions::math_ast::build_complex_float_expr_keep_real(re, im)
  }
}
