//! A stiff fallback for `NDSolve`'s first-order systems.
//!
//! The explicit RK4 integrator diverges on a stiff system — the Oregonator
//! model of the Belousov–Zhabotinsky reaction, with rate constants 2500
//! times apart, say — long before its step is small enough to be stable.
//! Such a leg is integrated with Hairer & Wanner's Rodas4: a linearly
//! implicit, stiffly accurate order-4 Rosenbrock method with an embedded
//! order-3 error estimate, stable for any step size, whose Jacobian comes
//! from finite differences.

use super::*;

/// Relative and absolute local-error tolerances.
const RTOL: f64 = 1e-8;
const ATOL: f64 = 1e-10;

/// Accepted plus rejected steps a leg may take before giving up. The
/// order-2 method needs ~500 steps per unit of `x` on a stiff problem whose
/// slow solution varies on a unit scale (`y' = -10^5 (y - Cos[x])`), so a
/// leg of a few hundred units must fit.
const MAX_STEPS: usize = 2_000_000;

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
  rosenbrock_leg_limited(f, init, x_from, x_to, MAX_STEPS, RTOL, ATOL)
}

/// [`rosenbrock_leg`] with its own tolerances, giving up after `max_steps`
/// accepted plus rejected steps — for a right-hand side so expensive (a
/// Newton solve per evaluation) that a leg crawling towards a singular
/// point must stop early.
pub(super) fn rosenbrock_leg_limited(
  f: &Rhs,
  init: &[f64],
  x_from: f64,
  x_to: f64,
  max_steps: usize,
  rtol: f64,
  atol: f64,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  let n = init.len();
  let span = x_to - x_from;
  if span == 0.0 || n == 0 {
    return Ok(Some(vec![(x_from, init.to_vec())]));
  }
  let dir = span.signum();

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
    .map(|(yi, fi)| fi.abs() / (atol + rtol * yi.abs()))
    .fold(0.0_f64, f64::max);
  let mut h = if scale0 > 0.0 {
    (0.8 * rtol.cbrt() / scale0).min(span.abs() / 10.0)
  } else {
    span.abs() / 100.0
  };
  let h_min = 16.0 * f64::EPSILON * x_from.abs().max(x_to.abs()).max(1.0);
  let mut failures = 0;
  let mut sparsity = JacobianSparsity::default();

  for _ in 0..max_steps {
    if (x_to - x) * dir <= h_min {
      return Ok(Some(points));
    }
    h = h.min((x_to - x).abs());
    let hs = h * dir;

    // The Jacobian and the time derivative, by forward differences.
    let Some(jacobian) = sparsity.jacobian(f, x, &y, &f0)? else {
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

    // One Rodas4 step (Hairer & Wanner's stiffly accurate order-4
    // Rosenbrock method with an embedded order-3 solution): six linear
    // solves with W = I/(h γ) - J, the last stage's increment being the
    // error estimate.
    let inv_hg = 1.0 / (hs * RODAS_GAMMA);
    let w: Vec<Vec<f64>> = (0..n)
      .map(|i| {
        (0..n)
          .map(|j| f64::from(u8::from(i == j)) * inv_hg - jacobian[i][j])
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
    let mut ks: Vec<Vec<f64>> = Vec::with_capacity(6);
    let mut stage_failed = false;
    let mut u = y.clone();
    for stage in 0..6 {
      let (fu, tcoef) = if stage == 0 {
        (f0.clone(), RODAS_D[0])
      } else {
        // The stage point: y plus the a-combination of the previous
        // increments (stages 5 and 6 add the last increment to u).
        if stage <= 4 {
          u.clone_from(&y);
          for (k, kv) in ks.iter().enumerate() {
            let a = RODAS_A[stage - 1][k];
            if a != 0.0 {
              for i in 0..n {
                u[i] += a * kv[i];
              }
            }
          }
        } else {
          for i in 0..n {
            u[i] += ks[stage - 1][i];
          }
        }
        let tc = if stage <= 3 { RODAS_C[stage - 1] } else { 1.0 };
        let Some(fu) = f(x + tc * hs, &u)? else {
          stage_failed = true;
          break;
        };
        (fu, if stage <= 3 { RODAS_D[stage] } else { 0.0 })
      };
      let mut rhs: Vec<f64> =
        fu.iter().zip(&t).map(|(a, b)| a + hs * tcoef * b).collect();
      if stage > 0 {
        for (k, kv) in ks.iter().enumerate() {
          let c = RODAS_CC[stage - 1][k] / hs;
          if c != 0.0 {
            for i in 0..n {
              rhs[i] += c * kv[i];
            }
          }
        }
      }
      ks.push(solve(rhs));
    }
    if stage_failed {
      failures += 1;
      if failures > MAX_EVALUATION_FAILURES {
        return Ok(None);
      }
      h /= 4.0;
      continue;
    }
    let y_new: Vec<f64> = (0..n).map(|i| u[i] + ks[5][i]).collect();
    // The weighted root-mean-square norm of the error, as in Hairer &
    // Wanner's codes: a max norm lets one component's estimate hold the
    // whole step back.
    let error = ((0..n)
      .map(|i| {
        let e = ks[5][i] / (atol + rtol * y[i].abs().max(y_new[i].abs()));
        e * e
      })
      .sum::<f64>()
      / n as f64)
      .sqrt();
    if !error.is_finite() {
      h /= 4.0;
      if h < h_min {
        return Ok(None);
      }
      continue;
    }
    let factor = (0.9 * error.powf(-0.25)).clamp(0.2, 6.0);
    if error > 1.0 {
      h *= factor;
      if h < h_min {
        return Ok(None);
      }
      continue;
    }
    let Some(f_new) = f(x + hs, &y_new)? else {
      failures += 1;
      if failures > MAX_EVALUATION_FAILURES {
        return Ok(None);
      }
      h /= 4.0;
      continue;
    };
    x += hs;
    if (x_to - x) * dir <= h_min {
      x = x_to;
    }
    y = y_new;
    f0 = f_new;
    points.push((x, y.clone()));
    h *= factor;
  }
  Ok(None)
}

/// Rodas4 (Hairer & Wanner, as tabulated in DifferentialEquations.jl):
/// γ, the stage times c2..c4, the time-derivative weights d1..d4, the
/// stage combinations a and the increment couplings C.
const RODAS_GAMMA: f64 = 0.25;
const RODAS_C: [f64; 3] = [0.386, 0.21, 0.63];
const RODAS_D: [f64; 4] = [0.25, -0.1043, 0.1035, -0.0362];
const RODAS_A: [[f64; 4]; 4] = [
  [1.544, 0.0, 0.0, 0.0],
  [0.9466785280815826, 0.2557011698983284, 0.0, 0.0],
  [
    3.314825187068521,
    2.896124015972201,
    0.9986419139977817,
    0.0,
  ],
  [
    1.221224509226641,
    6.019134481288629,
    12.53708332932087,
    -0.687886036105895,
  ],
];
const RODAS_CC: [[f64; 5]; 5] = [
  [-5.6688, 0.0, 0.0, 0.0, 0.0],
  [-2.430093356833875, -0.2063599157091915, 0.0, 0.0, 0.0],
  [
    -0.1073529058151375,
    -9.594562251023355,
    -20.47028614809616,
    0.0,
    0.0,
  ],
  [
    7.496443313967647,
    -10.24680431464352,
    -33.99990352819905,
    11.7089089320616,
    0.0,
  ],
  [
    8.083246795921522,
    -7.981132988064893,
    -31.52159432874371,
    16.31930543123136,
    -6.058818238834054,
  ],
];

/// The sparsity of a large system's Jacobian, learned from full
/// Jacobians, and the column groups (Curtis, Powell & Reid) whose columns
/// touch disjoint rows, so that one evaluation of `f` differences a whole
/// group. A method-of-lines system, each equation coupling a few
/// neighbours, then costs a handful of evaluations per Jacobian instead of
/// one per unknown. The full Jacobian is recomputed every
/// `JACOBIAN_PATTERN_REFRESH` uses and its pattern merged in, so an entry
/// that happened to vanish where the pattern was taken is soon picked up.
#[derive(Default)]
struct JacobianSparsity {
  /// For each column, the rows where it has been seen nonzero.
  rows: Vec<Vec<usize>>,
  groups: Vec<Vec<usize>>,
  uses: usize,
}

const JACOBIAN_PATTERN_REFRESH: usize = 25;

impl JacobianSparsity {
  fn jacobian(
    &mut self,
    f: &Rhs,
    x: f64,
    y: &[f64],
    f0: &[f64],
  ) -> Result<Option<Vec<Vec<f64>>>, InterpreterError> {
    let n = y.len();
    let refresh = self.uses.is_multiple_of(JACOBIAN_PATTERN_REFRESH);
    self.uses += 1;
    if n < 16 || refresh || self.groups.is_empty() {
      let Some(jac) = jacobian(f, x, y, f0)? else {
        return Ok(None);
      };
      if n >= 16 {
        self.learn(&jac);
      }
      return Ok(Some(jac));
    }
    let mut jac = vec![vec![0.0; n]; n];
    let mut probe = y.to_vec();
    for group in &self.groups {
      let deltas: Vec<f64> = group
        .iter()
        .map(|&j| f64::EPSILON.sqrt() * y[j].abs().max(1e-5))
        .collect();
      for (&j, d) in group.iter().zip(&deltas) {
        probe[j] = y[j] + d;
      }
      let Some(fg) = f(x, &probe)? else {
        return Ok(None);
      };
      for (&j, d) in group.iter().zip(&deltas) {
        probe[j] = y[j];
        for &i in &self.rows[j] {
          jac[i][j] = (fg[i] - f0[i]) / d;
        }
      }
    }
    Ok(Some(jac))
  }

  /// Merge a full Jacobian's nonzero pattern in and regroup the columns
  /// greedily; when grouping saves little, keep differencing column by
  /// column (no groups).
  fn learn(&mut self, jac: &[Vec<f64>]) {
    let n = jac.len();
    if self.rows.len() != n {
      self.rows = vec![Vec::new(); n];
    }
    for (j, rows) in self.rows.iter_mut().enumerate() {
      for (i, row) in jac.iter().enumerate() {
        if row[j] != 0.0 && !rows.contains(&i) {
          rows.push(i);
        }
      }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut taken: Vec<Vec<bool>> = Vec::new();
    for j in 0..n {
      let slot = taken
        .iter()
        .position(|rows| self.rows[j].iter().all(|&i| !rows[i]));
      let g = if let Some(g) = slot {
        g
      } else {
        groups.push(Vec::new());
        taken.push(vec![false; n]);
        groups.len() - 1
      };
      groups[g].push(j);
      for &i in &self.rows[j] {
        taken[g][i] = true;
      }
    }
    self.groups = if groups.len() * 2 <= n {
      groups
    } else {
      Vec::new()
    };
  }
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

/// Steps the explicit adaptive fallback may take before handing a leg to
/// the stiff method.
const MAX_EXPLICIT_STEPS: usize = 1_000_000;

/// Steps of a leg whose right-hand side cannot be evaluated (the state
/// left the domain of a `Log`, or an algebraic constraint has no root
/// there) before the leg is given up. Shrinking the step and growing it
/// again after each success would otherwise probe the boundary for up to
/// the whole step budget — and each failed evaluation of a DAE's
/// constraints is a full failing Newton run.
const MAX_EVALUATION_FAILURES: usize = 20;

/// Integrate `y' = f(x, y)` from `x_from` to `x_to` with the adaptive
/// Dormand–Prince 5(4) pair — the explicit fallback for a leg the fixed RK4
/// grid cannot follow, such as a spiking membrane model whose fast phases
/// last a fraction of a nominal step. `None` when the right-hand side fails
/// or the steps the error control asks for collapse (a stiff system, left
/// to [`rosenbrock_leg`]).
pub(super) fn dormand_prince_leg(
  f: &Rhs,
  init: &[f64],
  x_from: f64,
  x_to: f64,
  detect_stiffness: bool,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  const C: [f64; 7] =
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
  const A: [[f64; 6]; 7] = [
    [0.0; 6],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
      19372.0 / 6561.0,
      -25360.0 / 2187.0,
      64448.0 / 6561.0,
      -212.0 / 729.0,
      0.0,
      0.0,
    ],
    [
      9017.0 / 3168.0,
      -355.0 / 33.0,
      46732.0 / 5247.0,
      49.0 / 176.0,
      -5103.0 / 18656.0,
      0.0,
    ],
    [
      35.0 / 384.0,
      0.0,
      500.0 / 1113.0,
      125.0 / 192.0,
      -2187.0 / 6784.0,
      11.0 / 84.0,
    ],
  ];
  // Fifth-order weights (the last stage's row) minus the fourth-order ones.
  const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
  ];

  let n = init.len();
  let span = x_to - x_from;
  if span == 0.0 || n == 0 {
    return Ok(Some(vec![(x_from, init.to_vec())]));
  }
  let dir = span.signum();
  let mut x = x_from;
  let mut y = init.to_vec();
  let mut points = vec![(x, y.clone())];
  let Some(mut k0) = f(x, &y)? else {
    return Ok(None);
  };
  let scale0 = y
    .iter()
    .zip(&k0)
    .map(|(yi, fi)| fi.abs() / (ATOL + RTOL * yi.abs()))
    .fold(0.0_f64, f64::max);
  let mut h = if scale0 > 0.0 {
    (0.1 * scale0.powf(-0.2)).min(span.abs() / 10.0)
  } else {
    span.abs() / 100.0
  };
  let h_min = 16.0 * f64::EPSILON * x_from.abs().max(x_to.abs()).max(1.0);
  let (mut stiff_steps, mut calm_steps) = (0, 0);
  let mut failures = 0;

  for _ in 0..MAX_EXPLICIT_STEPS {
    if (x_to - x) * dir <= h_min {
      return Ok(Some(points));
    }
    h = h.min((x_to - x).abs());
    if h < h_min {
      return Ok(None);
    }
    let hs = h * dir;
    let mut k: Vec<Vec<f64>> = Vec::with_capacity(7);
    k.push(k0.clone());
    let mut failed = false;
    // The sixth stage's argument, for the stiffness estimate below.
    let mut y_stage6 = Vec::new();
    for stage in 1..7 {
      let ys: Vec<f64> = (0..n)
        .map(|i| {
          y[i] + hs * (0..stage).map(|j| A[stage][j] * k[j][i]).sum::<f64>()
        })
        .collect();
      if stage == 5 {
        y_stage6.clone_from(&ys);
      }
      match f(x + C[stage] * hs, &ys)? {
        Some(ks) if ks.iter().all(|v| v.is_finite()) => k.push(ks),
        _ => {
          failed = true;
          break;
        }
      }
    }
    if failed {
      failures += 1;
      if failures > MAX_EVALUATION_FAILURES {
        return Ok(None);
      }
      h /= 4.0;
      continue;
    }
    // The fifth-order solution is the last stage's argument (FSAL).
    let y_new: Vec<f64> = (0..n)
      .map(|i| y[i] + hs * (0..6).map(|j| A[6][j] * k[j][i]).sum::<f64>())
      .collect();
    let error = (0..n)
      .map(|i| {
        let e = (hs * (0..7).map(|j| E[j] * k[j][i]).sum::<f64>()).abs();
        e / (ATOL + RTOL * y[i].abs().max(y_new[i].abs()))
      })
      .fold(0.0_f64, f64::max);
    if !error.is_finite() {
      h /= 4.0;
      continue;
    }
    let factor = (0.9 * error.powf(-0.2)).clamp(0.2, 5.0);
    if error > 1.0 {
      h *= factor;
      continue;
    }
    // Hairer's stiffness detection: the last two stages share the abscissa
    // x + h, so |k7 - k6| / |y7 - y6| estimates the Jacobian's dominant
    // eigenvalue. When h |λ| keeps reaching the stability boundary the
    // explicit method only crawls on, so the leg is left to the stiff one.
    // The step size hovers around that boundary, so the count of stiff
    // steps is only reset after six calm ones in a row.
    let num: f64 = (0..n).map(|i| (k[6][i] - k[5][i]).powi(2)).sum();
    let den: f64 = (0..n).map(|i| (y_new[i] - y_stage6[i]).powi(2)).sum();
    if detect_stiffness && den > 0.0 && h * h * num > 3.25 * 3.25 * den {
      calm_steps = 0;
      stiff_steps += 1;
      if stiff_steps >= 15 {
        return Ok(None);
      }
    } else {
      calm_steps += 1;
      if calm_steps >= 6 {
        stiff_steps = 0;
      }
    }
    x += hs;
    if (x_to - x) * dir <= h_min {
      x = x_to;
    }
    y = y_new;
    k0 = k.pop().unwrap_or_default();
    points.push((x, y.clone()));
    h *= factor;
  }
  Ok(None)
}

#[cfg(test)]
mod tests {
  use super::*;

  // The stiff method resolves a slow solution under a fast relaxation
  // (`y' = -10^5 (y - cos x)`) over a long leg without running out of steps.
  #[test]
  fn rodas4_is_accurate_and_takes_few_steps() {
    // y' = -y: e^-1 at 1, within the tolerance.
    let decay = |_: f64, y: &[f64]| Ok(Some(vec![-y[0]]));
    let points = rosenbrock_leg(&decay, &[1.0], 0.0, 1.0)
      .unwrap()
      .expect("integrates");
    let err = (points.last().unwrap().1[0] - (-1.0f64).exp()).abs();
    assert!(err < 1e-8, "{err}");
    // A stiff linear pair with eigenvalues -1 and -10^4: an order-4
    // method follows the slow mode in far fewer steps than an order-2 one
    // (about 3000 here).
    let pair = |_: f64, y: &[f64]| {
      Ok(Some(vec![-y[0] + 9999.0 * y[1], -10000.0 * y[1]]))
    };
    let points = rosenbrock_leg(&pair, &[1.0, 1.0], 0.0, 10.0)
      .unwrap()
      .expect("integrates");
    // y1 = (2 e^-x - e^(-10^4 x)), y2 = e^(-10^4 x).
    let y = &points.last().unwrap().1;
    assert!((y[0] - 2.0 * (-10.0f64).exp()).abs() < 1e-8, "{y:?}");
    assert!(points.len() < 600, "{} steps", points.len());
  }

  #[test]
  fn long_stiff_leg_completes() {
    let stiff = |x: f64, y: &[f64]| Ok(Some(vec![-1e5 * (y[0] - x.cos())]));
    let points = rosenbrock_leg(&stiff, &[0.0], 0.0, 300.0)
      .unwrap()
      .expect("integrates");
    let (x, y) = points.last().unwrap();
    assert_eq!(*x, 300.0);
    // y = cos x + 10^-5 sin x + O(10^-10).
    let err = (y[0] - (300.0f64.cos() + 1e-5 * 300.0f64.sin())).abs();
    assert!(err < 1e-7, "{err}");
  }

  // A right-hand side with no value past x = 1 makes both legs give up
  // after a bounded number of evaluations instead of probing the boundary
  // with ever smaller steps for the whole step budget.
  #[test]
  fn undefined_right_hand_side_gives_up_quickly() {
    let calls = std::cell::Cell::new(0usize);
    let undefined_past_one = |x: f64, y: &[f64]| {
      calls.set(calls.get() + 1);
      Ok((x <= 1.0).then(|| vec![-y[0]]))
    };
    assert!(
      dormand_prince_leg(&undefined_past_one, &[1.0], 0.0, 10.0, true)
        .unwrap()
        .is_none()
    );
    assert!(calls.get() < 5_000, "{} evaluations", calls.get());
    calls.set(0);
    assert!(
      rosenbrock_leg(&undefined_past_one, &[1.0], 0.0, 10.0)
        .unwrap()
        .is_none()
    );
    assert!(calls.get() < 5_000, "{} evaluations", calls.get());
  }

  // A stiff system makes the explicit leg give up quickly so the stiff
  // method takes over; a non-stiff one integrates to the end.
  #[test]
  fn explicit_leg_detects_stiffness() {
    let stiff = |x: f64, y: &[f64]| Ok(Some(vec![-1e5 * (y[0] - x.cos())]));
    assert!(
      dormand_prince_leg(&stiff, &[1.0], 0.0, 300.0, true)
        .unwrap()
        .is_none()
    );
    let decay = |_: f64, y: &[f64]| Ok(Some(vec![-y[0]]));
    let points = dormand_prince_leg(&decay, &[1.0], 0.0, 10.0, true)
      .unwrap()
      .expect("integrates");
    assert!((points.last().unwrap().1[0] - (-10.0f64).exp()).abs() < 1e-9);
  }
}
