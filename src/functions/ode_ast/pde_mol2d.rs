//! The method of lines for `NDSolve` PDEs of one unknown `u` in two space
//! variables and time, `NDSolve[eqns, u, {x, …}, {y, …}, {t, …}]`, on the
//! rectangle the space domains span:
//!
//! `m ∂u/∂t == f(u, u_x, u_y, u_xx, u_yy, x, y, t)`
//!
//! with an initial condition `u[x, y, t0] == g(x, y)` and, on each edge of
//! the rectangle, a Dirichlet condition (`DirichletCondition[u == g, x ==
//! x0]` or `u[x0, y, t] == g`), a flux condition (`NeumannValue[g, x ==
//! x0]`, `g` affine in `u`, with the finite element meaning `n·(c ∇u) = g`,
//! `c` the coefficient of `-u_xx`), or nothing: the zero natural flux of the
//! finite element method.
//!
//! The rectangle is divided into a uniform grid of cells whose centers
//! carry the unknowns, so no unknown sits on an edge: a coefficient that is
//! singular on an edge (the `1/r` of a radial Laplacian on a domain starting
//! at the axis) is never evaluated there, and a symmetry condition on that
//! edge stays consistent. Derivatives are second-order central differences;
//! a cell next to an edge reads a ghost value beyond it, fixed by the edge's
//! condition at the face between the two. The resulting stiff ODE system is
//! integrated with the Rodas4 Rosenbrock method on a banded Jacobian.

use super::pde_mol::{Band, MolOptions};
use super::stiff::{RODAS_A, RODAS_C, RODAS_CC, RODAS_D, RODAS_GAMMA};
use super::*;

const U: &str = "NDSolve$U";
const UT: &str = "NDSolve$Ut";
const UX: &str = "NDSolve$Ux";
const UY: &str = "NDSolve$Uy";
const UXX: &str = "NDSolve$Uxx";
const UYY: &str = "NDSolve$Uyy";
const XV: &str = "NDSolve$x";
const YV: &str = "NDSolve$y";
const TV: &str = "NDSolve$t";

/// The condition on one edge of the rectangle.
enum Edge {
  /// `c u_n = alpha + beta u` on the face (outward normal derivative), all
  /// three compiled over `[x, y, t]`. `None` for a zero natural flux.
  Flux(Option<(NExpr, NExpr, NExpr)>),
  /// `u = g` on the face, `g` compiled over `[x, y, t]`.
  Dirichlet(NExpr),
}

/// The discretized problem.
struct Grid {
  nx: usize,
  ny: usize,
  hx: f64,
  hy: f64,
  xc: Vec<f64>,
  yc: Vec<f64>,
  x_faces: [f64; 2],
  y_faces: [f64; 2],
  /// Right-hand side `f / m` compiled over `[U, Ux, Uy, Uxx, Uyy, x, y, t]`.
  rhs: NExpr,
  /// Edges `[x lo, x hi, y lo, y hi]`.
  edges: [Edge; 4],
}

impl Grid {
  /// The ghost value beyond edge `e` next to a cell holding `u0`, the face
  /// at `(xf, yf)`.
  fn ghost(&self, e: usize, u0: f64, xf: f64, yf: f64, t: f64) -> f64 {
    let h = if e < 2 { self.hx } else { self.hy };
    match &self.edges[e] {
      Edge::Flux(None) => u0,
      Edge::Flux(Some((c, alpha, beta))) => {
        // c (u_ghost - u0) / h = alpha + beta (u0 + u_ghost) / 2.
        let p = [xf, yf, t];
        let (c, a, b) = (c.eval(&p), alpha.eval(&p), beta.eval(&p));
        (h * a + (c + h * b / 2.0) * u0) / (c - h * b / 2.0)
      }
      Edge::Dirichlet(g) => 2.0 * g.eval(&[xf, yf, t]) - u0,
    }
  }

  /// The four neighbor values of cell `(i, j)`: left, right, below, above.
  fn neighbors(&self, y: &[f64], i: usize, j: usize, t: f64) -> [f64; 4] {
    let ny = self.ny;
    let u0 = y[i * ny + j];
    let (xi, yj) = (self.xc[i], self.yc[j]);
    [
      if i > 0 {
        y[(i - 1) * ny + j]
      } else {
        self.ghost(0, u0, self.x_faces[0], yj, t)
      },
      if i + 1 < self.nx {
        y[(i + 1) * ny + j]
      } else {
        self.ghost(1, u0, self.x_faces[1], yj, t)
      },
      if j > 0 {
        y[i * ny + j - 1]
      } else {
        self.ghost(2, u0, xi, self.y_faces[0], t)
      },
      if j + 1 < ny {
        y[i * ny + j + 1]
      } else {
        self.ghost(3, u0, xi, self.y_faces[1], t)
      },
    ]
  }

  fn residual(&self, t: f64, y: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; y.len()];
    let mut vars = [0.0; 8];
    for i in 0..self.nx {
      for j in 0..self.ny {
        let u0 = y[i * self.ny + j];
        let [l, r, b, a] = self.neighbors(y, i, j, t);
        vars[0] = u0;
        vars[1] = (r - l) / (2.0 * self.hx);
        vars[2] = (a - b) / (2.0 * self.hy);
        vars[3] = (r - 2.0 * u0 + l) / (self.hx * self.hx);
        vars[4] = (a - 2.0 * u0 + b) / (self.hy * self.hy);
        vars[5] = self.xc[i];
        vars[6] = self.yc[j];
        vars[7] = t;
        out[i * self.ny + j] = self.rhs.eval(&vars);
      }
    }
    out
  }

  /// `J = ∂F/∂y` by forward differences, nine colors of cells that no
  /// cell's stencil reads together.
  fn jacobian(&self, t: f64, y: &[f64], f0: &[f64]) -> Option<Band> {
    let (nx, ny) = (self.nx, self.ny);
    let mut jac = Band::new(nx * ny, ny, ny);
    let mut probe = y.to_vec();
    for ci in 0..3 {
      for cj in 0..3 {
        let cells: Vec<(usize, usize)> = (ci..nx)
          .step_by(3)
          .flat_map(|i| (cj..ny).step_by(3).map(move |j| (i, j)))
          .collect();
        if cells.is_empty() {
          continue;
        }
        let deltas: Vec<f64> = cells
          .iter()
          .map(|&(i, j)| f64::EPSILON.sqrt() * y[i * ny + j].abs().max(1e-5))
          .collect();
        for (&(i, j), d) in cells.iter().zip(&deltas) {
          probe[i * ny + j] += d;
        }
        let fp = self.residual(t, &probe);
        for (&(i, j), d) in cells.iter().zip(&deltas) {
          let col = i * ny + j;
          probe[col] = y[col];
          let mut readers = vec![(i, j)];
          if i > 0 {
            readers.push((i - 1, j));
          }
          if i + 1 < nx {
            readers.push((i + 1, j));
          }
          if j > 0 {
            readers.push((i, j - 1));
          }
          if j + 1 < ny {
            readers.push((i, j + 1));
          }
          for (ri, rj) in readers {
            let row = ri * ny + rj;
            let v = (fp[row] - f0[row]) / d;
            if !v.is_finite() {
              return None;
            }
            if v != 0.0 {
              jac.add(row, col, v);
            }
          }
        }
      }
    }
    Some(jac)
  }

  /// The values on the closed rectangle at state `y`: the cell centers
  /// framed by the faces, `out[i][j]` for `x` index `i` (0 the lower face)
  /// and `y` index `j`.
  fn closed_values(&self, y: &[f64], t: f64) -> Vec<Vec<f64>> {
    let (nx, ny) = (self.nx, self.ny);
    let mut out = vec![vec![0.0; ny + 2]; nx + 2];
    for i in 0..nx {
      for j in 0..ny {
        out[i + 1][j + 1] = y[i * ny + j];
      }
    }
    for j in 0..ny {
      let yj = self.yc[j];
      let (u_lo, u_hi) = (y[j], y[(nx - 1) * ny + j]);
      out[0][j + 1] =
        f64::midpoint(u_lo, self.ghost(0, u_lo, self.x_faces[0], yj, t));
      out[nx + 1][j + 1] =
        f64::midpoint(u_hi, self.ghost(1, u_hi, self.x_faces[1], yj, t));
    }
    for i in 0..nx {
      let xi = self.xc[i];
      let (u_lo, u_hi) = (y[i * ny], y[i * ny + ny - 1]);
      out[i + 1][0] =
        f64::midpoint(u_lo, self.ghost(2, u_lo, xi, self.y_faces[0], t));
      out[i + 1][ny + 1] =
        f64::midpoint(u_hi, self.ghost(3, u_hi, xi, self.y_faces[1], t));
    }
    // Corners: linear extrapolation from the two adjacent faces.
    for (ci, ii) in [(0, 1), (nx + 1, nx)] {
      for (cj, jj) in [(0, 1), (ny + 1, ny)] {
        out[ci][cj] = out[ci][jj] + out[ii][cj] - out[ii][jj];
      }
    }
    out
  }
}

/// Integrate `y' = F(t, y)` from `t0` to `t1` by Rodas4 steps. Returns the
/// accepted `(t, y)` points, or `None` when the integration fails.
fn integrate(
  grid: &Grid,
  y0: Vec<f64>,
  t0: f64,
  t1: f64,
  jacobian_constant: bool,
) -> Option<Vec<(f64, Vec<f64>)>> {
  const RTOL: f64 = 1e-5;
  const MAX_STEPS: usize = 20_000;
  let span = t1 - t0;
  let n = y0.len();
  let mut scale = y0.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
  if scale == 0.0 {
    scale = 1.0;
  }
  let h_max = span / 100.0;
  let h_min = 16.0 * f64::EPSILON * t0.abs().max(t1.abs()).max(span);
  let mut h = (span * 1e-5).max(h_min * 4.0);
  let mut t = t0;
  let mut y = y0;
  let mut f0 = grid.residual(t, &y);
  let mut points = vec![(t, y.clone())];
  let mut fixed_jac: Option<Band> = None;
  let mut steps = 0usize;
  let mut failures = 0usize;
  while t1 - t > h_min {
    steps += 1;
    if steps > MAX_STEPS {
      return None;
    }
    h = h.min(h_max).min(t1 - t);
    let jac = if let Some(j) = &fixed_jac {
      j.clone()
    } else {
      let j = grid.jacobian(t, &y, &f0)?;
      if jacobian_constant {
        fixed_jac = Some(j.clone());
      }
      j
    };
    let ft: Vec<f64> = if jacobian_constant {
      vec![0.0; n]
    } else {
      let dt = f64::EPSILON.sqrt() * t.abs().max(h);
      grid
        .residual(t + dt, &y)
        .iter()
        .zip(&f0)
        .map(|(a, b)| (a - b) / dt)
        .collect()
    };
    // W = I / (h γ) - J.
    let mut w = jac;
    w.negate();
    for i in 0..n {
      w.add(i, i, 1.0 / (h * RODAS_GAMMA));
    }
    if !w.factor() {
      h /= 2.0;
      if h < h_min {
        return None;
      }
      continue;
    }
    let mut ks: Vec<Vec<f64>> = Vec::with_capacity(6);
    let mut u = y.clone();
    let mut bad = false;
    for stage in 0..6 {
      let (fu, tcoef) = if stage == 0 {
        (f0.clone(), RODAS_D[0])
      } else {
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
        (
          grid.residual(t + tc * h, &u),
          if stage <= 3 { RODAS_D[stage] } else { 0.0 },
        )
      };
      let mut rhs: Vec<f64> =
        fu.iter().zip(&ft).map(|(a, b)| a + h * tcoef * b).collect();
      for (k, kv) in ks.iter().enumerate() {
        let c = RODAS_CC[stage - 1][k] / h;
        if c != 0.0 {
          for i in 0..n {
            rhs[i] += c * kv[i];
          }
        }
      }
      if rhs.iter().any(|v| !v.is_finite()) {
        bad = true;
        break;
      }
      w.solve(&mut rhs);
      ks.push(rhs);
    }
    if bad {
      failures += 1;
      if failures > 30 {
        return None;
      }
      h /= 4.0;
      continue;
    }
    let y_new: Vec<f64> = (0..n).map(|i| u[i] + ks[5][i]).collect();
    let atol = RTOL * scale;
    let error = ((0..n)
      .map(|i| {
        let e = ks[5][i] / (atol + RTOL * y[i].abs().max(y_new[i].abs()));
        e * e
      })
      .sum::<f64>()
      / n as f64)
      .sqrt();
    if !error.is_finite() {
      h /= 4.0;
      if h < h_min {
        return None;
      }
      continue;
    }
    let factor = (0.9 * error.powf(-0.25)).clamp(0.2, 6.0);
    if error > 1.0 {
      h *= factor;
      if h < h_min {
        return None;
      }
      continue;
    }
    t = if t1 - (t + h) <= h_min { t1 } else { t + h };
    y = y_new;
    f0 = grid.residual(t, &y);
    if f0.iter().any(|v| !v.is_finite()) {
      return None;
    }
    scale = scale.max(y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    points.push((t, y.clone()));
    h *= factor;
  }
  Some(points)
}

fn ev(e: &Expr) -> Option<Expr> {
  crate::evaluator::evaluate_expr_to_expr(e).ok()
}

fn ident(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

fn mentions_any(e: &Expr, names: &[&str]) -> bool {
  names.iter().any(|n| expr_contains_ident(e, n))
}

/// The arguments of the first call of `u` (plain or a derivative) in `e`.
fn first_u_args(e: &Expr, u: &str) -> Option<Vec<Expr>> {
  if let Expr::FunctionCall { name, args } = e
    && name == u
  {
    return Some(args.to_vec());
  }
  if let Expr::CurriedCall { func, args } = e
    && let Expr::CurriedCall { args: fname, .. } = func.as_ref()
    && matches!(fname.as_slice(), [Expr::Identifier(n)] if n == u)
  {
    return Some(args.clone());
  }
  expr_children(e).iter().find_map(|c| first_u_args(c, u))
}

/// `e` with every call of `u` (whatever its arguments) replaced by the
/// placeholder `U` and every supported derivative by its placeholder
/// (`slots` gives the argument positions of `t`, `x` and `y`). `None` if a
/// derivative is not one of `u_t`, `u_x`, `u_y`, `u_xx`, `u_yy`.
fn placeholderize(e: &Expr, u: &str, slots: [usize; 3]) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = e
    && name == u
    && args.len() == 3
  {
    return Some(ident(U));
  }
  if let Expr::CurriedCall { func, args } = e
    && args.len() == 3
    && let Expr::CurriedCall {
      func: head,
      args: fname,
    } = func.as_ref()
    && matches!(fname.as_slice(), [Expr::Identifier(n)] if n == u)
    && let Expr::FunctionCall { name, args: o } = head.as_ref()
    && name == "Derivative"
    && o.len() == 3
  {
    let order = |k: usize| match &o[slots[k]] {
      Expr::Integer(i) => Some(*i),
      _ => None,
    };
    let name = match (order(0)?, order(1)?, order(2)?) {
      (0, 0, 0) => U,
      (1, 0, 0) => UT,
      (0, 1, 0) => UX,
      (0, 0, 1) => UY,
      (0, 2, 0) => UXX,
      (0, 0, 2) => UYY,
      _ => return None,
    };
    return Some(ident(name));
  }
  if matches!(e, Expr::Identifier(n) if n == u) {
    return None;
  }
  let failed = std::cell::Cell::new(false);
  let mapped = map_children(e, &|c| {
    placeholderize(c, u, slots).unwrap_or_else(|| {
      failed.set(true);
      c.clone()
    })
  });
  (!failed.get()).then_some(mapped)
}

/// The highest order of the derivatives of `u` in `e` with respect to
/// argument `slot`, `None` if `u` is never differentiated or an order is
/// symbolic.
fn max_slot_order(e: &Expr, u: &str, slot: usize) -> Option<i64> {
  if let Expr::CurriedCall { func, .. } = e
    && let Expr::CurriedCall {
      func: head,
      args: fname,
    } = func.as_ref()
    && matches!(fname.as_slice(), [Expr::Identifier(n)] if n == u)
    && let Expr::FunctionCall { name, args: o } = head.as_ref()
    && name == "Derivative"
  {
    return match o.get(slot) {
      Some(Expr::Integer(k)) => Some(*k as i64),
      _ => None,
    };
  }
  expr_children(e)
    .iter()
    .filter_map(|c| max_slot_order(c, u, slot))
    .max()
}

/// The edges `x == x0` names, as `(axis, upper)` pairs: `axis` 0 for `x`,
/// 1 for `y`. `Or` of such equations names several.
fn edges_of(pred: &Expr, space: [&PdeDomain; 2]) -> Option<Vec<(usize, bool)>> {
  if let Expr::FunctionCall { name, args } = pred
    && name == "Or"
  {
    let mut out = Vec::new();
    for a in args {
      out.extend(edges_of(a, space)?);
    }
    return Some(out);
  }
  if let Expr::BinaryOp {
    op: BinaryOperator::Or,
    left,
    right,
  } = pred
  {
    let mut out = edges_of(left, space)?;
    out.extend(edges_of(right, space)?);
    return Some(out);
  }
  let (a, b) = as_equal_pair(pred)?;
  let (var, value) = match (a, b) {
    (Expr::Identifier(n), v) | (v, Expr::Identifier(n))
      if space.iter().any(|d| &d.name == n) =>
    {
      (n, v)
    }
    _ => return None,
  };
  let axis = space.iter().position(|d| &d.name == var)?;
  let v = nval_to_f64(value)?;
  let d = space[axis];
  let near = |target: f64| (v - target).abs() <= 1e-9 * target.abs().max(1.0);
  if near(d.min) {
    Some(vec![(axis, false)])
  } else if near(d.max) {
    Some(vec![(axis, true)])
  } else {
    None
  }
}

/// An unknown's call `u[a1, a2, a3]` as one side of `eq`, with the other
/// side free of `u`: `(args, other side)`.
fn u_call_side(eq: &Expr, u: &str) -> Option<(Vec<Expr>, Expr)> {
  let (a, b) = as_equal_pair(eq)?;
  for (call_side, other) in [(a, b), (b, a)] {
    if let Expr::FunctionCall { name, args } = call_side
      && name == u
      && args.len() == 3
      && !expr_contains_ident(other, u)
    {
      return Some((args.to_vec(), other.clone()));
    }
  }
  None
}

/// Solve `NDSolve[eqs, u, dom1, dom2, dom3]` for a scalar unknown in two
/// space variables and time (see the module documentation). `Ok(None)`
/// when the problem does not have the supported shape.
pub(super) fn solve_pde_mol_2d(
  eqs: &[Expr],
  rest: &[Expr],
  opts: &MolOptions,
) -> Result<Option<Expr>, InterpreterError> {
  Ok(solve(eqs, rest, opts))
}

fn solve(eqs: &[Expr], rest: &[Expr], opts: &MolOptions) -> Option<Expr> {
  let Expr::Identifier(u) = &rest[0] else {
    return None;
  };
  let u = u.as_str();
  let doms: Vec<PdeDomain> = rest[1..4]
    .iter()
    .map(parse_pde_domain)
    .collect::<Option<_>>()?;
  if doms[0].name == doms[1].name
    || doms[0].name == doms[2].name
    || doms[1].name == doms[2].name
  {
    return None;
  }

  // Classify the equations.
  let mut pde: Option<&Expr> = None;
  let mut dirichlet: Vec<(Expr, &Expr)> = Vec::new();
  let mut point_conditions: Vec<&Expr> = Vec::new();
  for eq in eqs {
    if let Expr::FunctionCall { name, args } = eq
      && name == "DirichletCondition"
      && args.len() == 2
    {
      dirichlet.push((args[0].clone(), &args[1]));
      continue;
    }
    as_equal_pair(eq)?;
    if mentions_head(eq, "NeumannValue") || mentions_head(eq, "Derivative") {
      if pde.is_some() {
        return None;
      }
      pde = Some(eq);
    } else {
      point_conditions.push(eq);
    }
  }
  let pde = pde?;

  // The argument slot of each domain's variable in the unknown's calls.
  let u_args = first_u_args(pde, u)?;
  if u_args.len() != 3 {
    return None;
  }
  let slot_of = |name: &str| {
    u_args
      .iter()
      .position(|a| matches!(a, Expr::Identifier(n) if n == name))
  };
  let dom_slots: Vec<usize> = doms
    .iter()
    .map(|d| slot_of(&d.name))
    .collect::<Option<_>>()?;

  // Split the PDE into its operator and the `NeumannValue` terms.
  let (lhs, rhs) = as_equal_pair(pde)?;
  let has_nv = |e: &Expr| {
    flatten_plus_terms(e)
      .iter()
      .any(|t| matches!(t, Expr::FunctionCall { name, .. } if name == "NeumannValue"))
  };
  let (lhs, rhs) = if has_nv(lhs) { (rhs, lhs) } else { (lhs, rhs) };
  if mentions_head(lhs, "NeumannValue") {
    return None;
  }
  let mut nvs: Vec<(Expr, Expr)> = Vec::new();
  let mut others = Vec::new();
  for term in flatten_plus_terms(rhs) {
    match &term {
      Expr::FunctionCall { name, args }
        if name == "NeumannValue" && args.len() == 2 =>
      {
        nvs.push((args[0].clone(), args[1].clone()));
      }
      _ => {
        if mentions_head(&term, "NeumannValue") {
          return None;
        }
        others.push(term);
      }
    }
  }
  let operator = minus2(
    lhs.clone(),
    if others.is_empty() {
      Expr::Integer(0)
    } else {
      plus(others)
    },
  );

  // The time variable: the domain the PDE differentiates once (and no
  // more) whose start an initial condition fixes.
  let mut found = None;
  for k in 0..3 {
    let t_dom = &doms[k];
    let t_slot = dom_slots[k];
    if max_slot_order(&operator, u, t_slot) != Some(1) {
      continue;
    }
    let ic = point_conditions.iter().position(|eq| {
      u_call_side(eq, u).is_some_and(|(args, _)| {
        args.iter().enumerate().all(|(s, a)| {
          if s == t_slot {
            nval_to_f64(a).is_some_and(|v| {
              (v - t_dom.min).abs() <= 1e-9 * t_dom.min.abs().max(1.0)
            })
          } else {
            matches!(a, Expr::Identifier(n) if Some(s) == slot_of(n))
          }
        })
      })
    });
    if let Some(ic) = ic {
      found = Some((k, ic));
      break;
    }
  }
  let (t_index, ic_index) = found?;
  let space_idx: Vec<usize> = (0..3).filter(|&k| k != t_index).collect();
  let (t_dom, x_dom, y_dom) =
    (&doms[t_index], &doms[space_idx[0]], &doms[space_idx[1]]);
  let slots = [
    dom_slots[t_index],
    dom_slots[space_idx[0]],
    dom_slots[space_idx[1]],
  ];
  // Rename the variables to private names so a value they might hold does
  // not interfere.
  let rename = |e: &Expr| -> Expr {
    let e = crate::syntax::substitute_variable(e, &t_dom.name, &ident(TV));
    let e = crate::syntax::substitute_variable(&e, &x_dom.name, &ident(XV));
    crate::syntax::substitute_variable(&e, &y_dom.name, &ident(YV))
  };
  let space_vars = vec![XV.to_string(), YV.to_string(), TV.to_string()];
  let rhs_vars: Vec<String> = [U, UX, UY, UXX, UYY, XV, YV, TV]
    .iter()
    .map(std::string::ToString::to_string)
    .collect();
  let placeholders = [U, UT, UX, UY, UXX, UYY];

  // The operator in placeholders, solved for `u_t`.
  let op = ev(&rename(&placeholderize(&operator, u, slots)?))?;
  if expr_contains_ident(&op, u) || !expr_contains_ident(&op, UT) {
    return None;
  }
  let mass = ev(&call("D", vec![op.clone(), ident(UT)]))?;
  if mentions_any(&mass, &placeholders) {
    return None;
  }
  let without_ut = ev(&crate::syntax::substitute_variable(
    &op,
    UT,
    &Expr::Integer(0),
  ))?;
  let f = ev(&times2(
    neg1(without_ut),
    call("Power", vec![mass, Expr::Integer(-1)]),
  ))?;
  let rhs_fn = compile_numeric(&f, &rhs_vars)?;
  // Linear with constant coefficients: the Jacobian need not be rebuilt.
  let mut jacobian_constant = !expr_contains_ident(&f, TV);
  for p in [U, UX, UY, UXX, UYY] {
    let d = ev(&call("D", vec![f.clone(), ident(p)]))?;
    if mentions_any(&d, &placeholders) {
      jacobian_constant = false;
    }
  }
  let diffusion = |p: &str| -> Option<Expr> {
    let c = ev(&neg1(call("D", vec![op.clone(), ident(p)])))?;
    (!mentions_any(&c, &placeholders)).then_some(c)
  };

  // Edge conditions.
  let space = [x_dom, y_dom];
  let mut flux: [Option<(Expr, Expr)>; 4] = Default::default();
  let mut dirichlet_values: [Option<Expr>; 4] = Default::default();
  let edge_index = |(axis, upper): (usize, bool)| 2 * axis + upper as usize;
  for (g, pred) in &nvs {
    let g = ev(&rename(&placeholderize(g, u, slots)?))?;
    if mentions_any(&g, &[UT, UX, UY, UXX, UYY]) {
      return None;
    }
    let alpha = ev(&crate::syntax::substitute_variable(
      &g,
      U,
      &Expr::Integer(0),
    ))?;
    let beta = ev(&call("D", vec![g.clone(), ident(U)]))?;
    if mentions_any(&beta, &placeholders) {
      return None;
    }
    for edge in edges_of(pred, space)? {
      let slot = &mut flux[edge_index(edge)];
      *slot = Some(match slot.take() {
        None => (alpha.clone(), beta.clone()),
        Some((a, b)) => (
          ev(&plus(vec![a, alpha.clone()]))?,
          ev(&plus(vec![b, beta.clone()]))?,
        ),
      });
    }
  }
  for (cond, pred) in &dirichlet {
    let (_, value) = u_call_side(cond, u)?;
    let value = ev(&rename(&value))?;
    for edge in edges_of(pred, space)? {
      dirichlet_values[edge_index(edge)] = Some(value.clone());
    }
  }
  let mut ic_value = None;
  for (k, eq) in point_conditions.iter().enumerate() {
    let (args, value) = u_call_side(eq, u)?;
    if k == ic_index {
      ic_value = Some(ev(&rename(&value))?);
      continue;
    }
    // `u[x0, y, t] == g`: a Dirichlet condition on the edge `x == x0`.
    let mut edge = None;
    for (axis, d) in space.iter().enumerate() {
      let s = slots[axis + 1];
      if let Some(v) = nval_to_f64(&args[s]) {
        let near =
          |target: f64| (v - target).abs() <= 1e-9 * target.abs().max(1.0);
        if edge.is_some() || !(near(d.min) || near(d.max)) {
          return None;
        }
        edge = Some((axis, near(d.max)));
      }
    }
    let edge = edge?;
    let other_ok = (0..3).filter(|&s| s != slots[edge.0 + 1]).all(
      |s| matches!(&args[s], Expr::Identifier(n) if slot_of(n) == Some(s)),
    );
    if !other_ok {
      return None;
    }
    dirichlet_values[edge_index(edge)] = Some(ev(&rename(&value))?);
  }
  let ic_fn = compile_numeric(&ic_value?, &space_vars)?;

  let mut edges: Vec<Edge> = Vec::with_capacity(4);
  for e in 0..4 {
    if let Some(v) = &dirichlet_values[e] {
      if expr_contains_ident(v, TV) {
        jacobian_constant = false;
      }
      edges.push(Edge::Dirichlet(compile_numeric(v, &space_vars)?));
      continue;
    }
    match &flux[e] {
      None => edges.push(Edge::Flux(None)),
      Some((alpha, beta)) => {
        let c = diffusion(if e < 2 { UXX } else { UYY })?;
        if mentions_any(&c, &[TV]) || mentions_any(beta, &[TV]) {
          jacobian_constant = false;
        }
        edges.push(Edge::Flux(Some((
          compile_numeric(&c, &space_vars)?,
          compile_numeric(alpha, &space_vars)?,
          compile_numeric(beta, &space_vars)?,
        ))));
      }
    }
  }
  let edges: [Edge; 4] = edges.try_into().ok()?;

  let mut n_cells = 32usize;
  if let Some(lo) = opts.min_points {
    n_cells = n_cells.max(lo);
  }
  if let Some(hi) = opts.max_points {
    n_cells = n_cells.min(hi);
  }
  let n_cells = n_cells.max(4);
  let (nx, ny) = (n_cells, n_cells);
  let hx = (x_dom.max - x_dom.min) / nx as f64;
  let hy = (y_dom.max - y_dom.min) / ny as f64;
  let grid = Grid {
    nx,
    ny,
    hx,
    hy,
    xc: (0..nx).map(|i| x_dom.min + (i as f64 + 0.5) * hx).collect(),
    yc: (0..ny).map(|j| y_dom.min + (j as f64 + 0.5) * hy).collect(),
    x_faces: [x_dom.min, x_dom.max],
    y_faces: [y_dom.min, y_dom.max],
    rhs: rhs_fn,
    edges,
  };
  let mut y0 = Vec::with_capacity(nx * ny);
  for i in 0..nx {
    for j in 0..ny {
      y0.push(ic_fn.eval(&[grid.xc[i], grid.yc[j], t_dom.min]));
    }
  }
  if y0.iter().any(|v| !v.is_finite()) {
    return None;
  }
  let points = integrate(&grid, y0, t_dom.min, t_dom.max, jacobian_constant)?;

  // The solution on the closed rectangle at every accepted time.
  let mut xs = vec![x_dom.min];
  xs.extend(&grid.xc);
  xs.push(x_dom.max);
  let mut ys = vec![y_dom.min];
  ys.extend(&grid.yc);
  ys.push(y_dom.max);
  // The first frame is the initial condition itself, edges included.
  let mut frames: Vec<Vec<Vec<f64>>> = vec![
    xs.iter()
      .map(|&x| ys.iter().map(|&y| ic_fn.eval(&[x, y, t_dom.min])).collect())
      .collect(),
  ];
  frames.extend(
    points
      .iter()
      .skip(1)
      .map(|(t, y)| grid.closed_values(y, *t)),
  );
  let ts: Vec<f64> = points.iter().map(|(t, _)| *t).collect();
  // Axis `k` of the result is the unknown's argument slot `k`.
  let role_of_slot: Vec<usize> = (0..3)
    .map(|s| slots.iter().position(|&x| x == s).unwrap())
    .collect();
  let axis_coords = |role: usize| -> &Vec<f64> {
    match role {
      0 => &ts,
      1 => &xs,
      _ => &ys,
    }
  };
  let axis_dom = |role: usize| -> &PdeDomain {
    match role {
      0 => t_dom,
      1 => x_dom,
      _ => y_dom,
    }
  };
  let reals =
    |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
  let value_at = |idx: [usize; 3]| -> f64 {
    let mut role_idx = [0usize; 3];
    for s in 0..3 {
      role_idx[role_of_slot[s]] = idx[s];
    }
    frames[role_idx[0]][role_idx[1]][role_idx[2]]
  };
  let lens: Vec<usize> =
    (0..3).map(|s| axis_coords(role_of_slot[s]).len()).collect();
  let values = Expr::List(
    (0..lens[0])
      .map(|a| {
        Expr::List(
          (0..lens[1])
            .map(|b| {
              Expr::List(
                (0..lens[2])
                  .map(|c| Expr::Real(value_at([a, b, c])))
                  .collect(),
              )
            })
            .collect(),
        )
      })
      .collect(),
  );
  let domain = Expr::List(
    (0..3)
      .map(|s| {
        let d = axis_dom(role_of_slot[s]);
        Expr::List(vec![Expr::Real(d.min), Expr::Real(d.max)].into())
      })
      .collect(),
  );
  let orders = Expr::List(vec![Expr::Integer(3); 3].into());
  let coords = Expr::List(
    (0..3)
      .map(|s| reals(axis_coords(role_of_slot[s])))
      .collect(),
  );
  let func = call(
    "InterpolatingFunction",
    vec![domain, values, orders, coords],
  );
  Some(Expr::List(
    vec![Expr::List(vec![rule_expr(ident(u), func)].into())].into(),
  ))
}
