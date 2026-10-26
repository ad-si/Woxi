//! A stiff fallback for `NDSolve`'s first-order systems.
//!
//! The explicit RK4 integrator diverges on a stiff system — the Oregonator
//! model of the Belousov–Zhabotinsky reaction, with rate constants 2500
//! times apart, say — long before its step is small enough to be stable.
//! Such a leg is integrated with the linearly implicit Rosenbrock pair of
//! Shampine and Reichelt (MATLAB's `ode23s`): an order-2 step with an
//! order-3 error estimate, stable for any step size, whose Jacobian comes
//! from finite differences.

use super::*;

/// Relative and absolute local-error tolerances.
const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;

/// Accepted plus rejected steps a leg may take before giving up.
const MAX_STEPS: usize = 200_000;

/// The right-hand side `f(x, y)` of `y' = f(x, y)`; `None` where it cannot
/// be evaluated.
pub(super) type Rhs<'a> =
  dyn Fn(f64, &[f64]) -> Result<Option<Vec<f64>>, InterpreterError> + 'a;

/// Integrate `y' = f(x, y)` from `x_from` to `x_to` with adaptive
/// Rosenbrock steps. Returns every accepted point, starting at `x_from`, or
/// `None` when the right-hand side fails or the step size collapses.
pub(super) fn rosenbrock_leg(
  f: &Rhs,
  init: &[f64],
  x_from: f64,
  x_to: f64,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let n = init.len();
  let span = x_to - x_from;
  if span == 0.0 || n == 0 {
    return Ok(Some(vec![(x_from, init.to_vec())]));
  }
  let dir = span.signum();
  let d = 1.0 / (2.0 + std::f64::consts::SQRT_2);
  let e32 = 6.0 + std::f64::consts::SQRT_2;

  let mut x = x_from;
  let mut y = init.to_vec();
  let mut points = vec![(x, y.clone())];
  let Some(mut f0) = f(x, &y)? else {
    return Ok(None);
  };
  // A first step that changes `y` by about the tolerance.
  let scale0 = y
    .iter()
    .zip(&f0)
    .map(|(yi, fi)| fi.abs() / (ATOL + RTOL * yi.abs()))
    .fold(0.0_f64, f64::max);
  let mut h = if scale0 > 0.0 {
    (0.8 * RTOL.cbrt() / scale0).min(span.abs() / 10.0)
  } else {
    span.abs() / 100.0
  };
  let h_min = 16.0 * f64::EPSILON * x_from.abs().max(x_to.abs()).max(1.0);

  for _ in 0..MAX_STEPS {
    if (x_to - x) * dir <= h_min {
      return Ok(Some(points));
    }
    h = h.min((x_to - x).abs());
    let hs = h * dir;

    // The Jacobian and the time derivative, by forward differences.
    let Some(jacobian) = jacobian(f, x, &y, &f0)? else {
      return Ok(None);
    };
    let dt = f64::EPSILON.sqrt() * x.abs().max(h);
    let Some(ft) = f(x + dt * dir, &y)? else {
      return Ok(None);
    };
    let t: Vec<f64> = ft
      .iter()
      .zip(&f0)
      .map(|(a, b)| (a - b) / (dt * dir))
      .collect();

    // W = I - h d J.
    let w: Vec<Vec<f64>> = (0..n)
      .map(|i| {
        (0..n)
          .map(|j| f64::from(u8::from(i == j)) - hs * d * jacobian[i][j])
          .collect()
      })
      .collect();
    let Some(lu) = MassLu::factor(w) else {
      h /= 2.0;
      if h < h_min {
        return Ok(None);
      }
      continue;
    };
    // `MassLu::solve` answers `W·v = -c`.
    let solve = |b: Vec<f64>| -> Vec<f64> {
      lu.solve(&b.iter().map(|v| -v).collect::<Vec<_>>())
    };

    let k1 = solve(f0.iter().zip(&t).map(|(a, b)| a + hs * d * b).collect());
    let y1: Vec<f64> =
      y.iter().zip(&k1).map(|(a, k)| a + 0.5 * hs * k).collect();
    let Some(f1) = f(x + 0.5 * hs, &y1)? else {
      h /= 4.0;
      continue;
    };
    let k2: Vec<f64> = solve(f1.iter().zip(&k1).map(|(a, b)| a - b).collect())
      .iter()
      .zip(&k1)
      .map(|(a, b)| a + b)
      .collect();
    let y_new: Vec<f64> = y.iter().zip(&k2).map(|(a, k)| a + hs * k).collect();
    let Some(f2) = f(x + hs, &y_new)? else {
      h /= 4.0;
      continue;
    };
    let k3 = solve(
      (0..n)
        .map(|i| {
          f2[i] - e32 * (k2[i] - f1[i]) - 2.0 * (k1[i] - f0[i]) + hs * d * t[i]
        })
        .collect(),
    );
    let error = (0..n)
      .map(|i| {
        let e = (hs / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i])).abs();
        e / (ATOL + RTOL * y[i].abs().max(y_new[i].abs()))
      })
      .fold(0.0_f64, f64::max);
    if !error.is_finite() {
      h /= 4.0;
      if h < h_min {
        return Ok(None);
      }
      continue;
    }
    let factor = (0.8 * error.powf(-1.0 / 3.0)).clamp(0.2, 5.0);
    if error > 1.0 {
      h *= factor;
      if h < h_min {
        return Ok(None);
      }
      continue;
    }
    x += hs;
    if (x_to - x) * dir <= h_min {
      x = x_to;
    }
    y = y_new;
    f0 = f2;
    points.push((x, y.clone()));
    h *= factor;
  }
  Ok(None)
}

/// `∂f/∂y` at `(x, y)` by forward differences, given `f0 = f(x, y)`.
fn jacobian(
  f: &Rhs,
  x: f64,
  y: &[f64],
  f0: &[f64],
) -> Result<Option<Vec<Vec<f64>>>, InterpreterError> {
  let n = y.len();
  let mut jac = vec![vec![0.0; n]; n];
  let mut probe = y.to_vec();
  for j in 0..n {
    let delta = f64::EPSILON.sqrt() * y[j].abs().max(1e-5);
    probe[j] = y[j] + delta;
    let Some(fj) = f(x, &probe)? else {
      return Ok(None);
    };
    probe[j] = y[j];
    for i in 0..n {
      jac[i][j] = (fj[i] - f0[i]) / delta;
    }
  }
  Ok(Some(jac))
}
