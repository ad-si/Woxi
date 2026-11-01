//! The method of lines for `NDSolve` PDE systems first order in time and
//! one-dimensional in space.
//!
//! Every unknown `u_j[t, x]` is sampled on a uniform grid in `x`; spatial
//! derivatives become finite-difference stencils (Fornberg weights, of
//! `"DifferenceOrder"` 4 by default like Wolfram's `"TensorProductGrid"`,
//! centred in the interior and shifted one-sided towards the ends). Each
//! evolution equation may combine the time derivatives of several unknowns
//! linearly, `Σ_j m_ij(x) ∂u_j/∂t == f_i(u, u_x, u_xx, …, t, x)`, so the
//! semi-discrete system is `M y' == F(t, y)` with a block-diagonal mass
//! matrix `M`. A boundary condition replaces the evolution equation of the
//! unknown it constrains at that end of the grid by an algebraic row (`M`
//! row zero), whatever its form: Dirichlet, Neumann, Robin, a flux
//! condition coupling several unknowns. An unknown of second order in `x`
//! without a condition at one end keeps its evolution equation there,
//! discretized one-sidedly (Wolfram's "artificial boundary", announced by
//! `NDSolve::bcart`), and both ends tied together (`u[t, a] == u[t, b]`)
//! make the grid periodic.
//!
//! The resulting index-1 differential-algebraic system is integrated with
//! the stiffly accurate Rodas4 Rosenbrock method of
//! [`super::stiff`] (adaptive steps, a finite-difference Jacobian built
//! from a few colored evaluations), on banded matrices: the state is
//! ordered by grid point, interleaved for a periodic grid so that its
//! wrap-around stencils stay banded too.

use super::stiff::{RODAS_A, RODAS_C, RODAS_CC, RODAS_D, RODAS_GAMMA};
use super::*;

/// Options of `Method -> {"MethodOfLines", "SpatialDiscretization" ->
/// {"TensorProductGrid", …}}` the grid honours.
#[derive(Clone, Default)]
pub(super) struct MolOptions {
  pub min_points: Option<usize>,
  pub max_points: Option<usize>,
  pub difference_order: Option<usize>,
}

impl MolOptions {
  /// Read the grid options out of `NDSolve`'s trailing option rules.
  pub(super) fn from_rules(opts: &[Expr]) -> Self {
    let mut out = Self::default();
    fn key(e: &Expr) -> String {
      match e {
        Expr::String(s) => s.clone(),
        Expr::Identifier(s) => s.clone(),
        other => expr_to_string(other),
      }
    }
    fn walk(e: &Expr, out: &mut MolOptions) {
      match e {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => {
          let as_count = || match replacement.as_ref() {
            Expr::Integer(k) if *k > 1 => Some(*k as usize),
            _ => None,
          };
          match key(pattern).as_str() {
            "MinPoints" => out.min_points = as_count(),
            "MaxPoints" => out.max_points = as_count(),
            "DifferenceOrder" => {
              out.difference_order = as_count().filter(|p| *p <= 8);
            }
            _ => walk(replacement, out),
          }
        }
        Expr::List(items) => items.iter().for_each(|i| walk(i, out)),
        _ => {}
      }
    }
    for o in opts {
      walk(o, &mut out);
    }
    out
  }
}

/// One appearance of an unknown: `Derivative[dt, dx][u_j][t_arg, x_arg]`
/// (orders 0 for the plain call), with the arguments in time/space roles.
struct Occ {
  j: usize,
  dt: usize,
  dx: usize,
  t_arg: Expr,
  x_arg: Expr,
}

/// Every appearance of the unknowns in `expr`; `None` when an unknown is
/// used in any other way (bare, with the wrong arity, as an argument).
fn collect_occurrences(
  expr: &Expr,
  u_names: &[String],
  swap: bool,
  out: &mut Vec<Occ>,
) -> Option<()> {
  for (j, name) in u_names.iter().enumerate() {
    if let Some((dt, dx, t_arg, x_arg)) = match_pde_term_roles(expr, name, swap)
    {
      if expr_references_any(&t_arg, u_names)
        || expr_references_any(&x_arg, u_names)
      {
        return None;
      }
      out.push(Occ {
        j,
        dt,
        dx,
        t_arg,
        x_arg,
      });
      return Some(());
    }
  }
  match expr {
    Expr::Identifier(n) if u_names.contains(n) => return None,
    Expr::FunctionCall { name, .. } if u_names.contains(name) => return None,
    _ => {}
  }
  for c in expr_children(expr) {
    collect_occurrences(c, u_names, swap, out)?;
  }
  Some(())
}

/// Replace every appearance of an unknown by its placeholder: the time
/// derivative `Derivative[1, 0][u_j]` by `NDSolve$T{j}`, the `k`-th space
/// derivative (`k == 0` the value) by `NDSolve$D{k}U{j}`.
fn placeholderize(expr: &Expr, u_names: &[String], swap: bool) -> Expr {
  for (j, name) in u_names.iter().enumerate() {
    if let Some((dt, dx, _, _)) = match_pde_term_roles(expr, name, swap) {
      return Expr::Identifier(if dt == 1 {
        format!("NDSolve$T{j}")
      } else {
        format!("NDSolve$D{dx}U{j}")
      });
    }
  }
  map_children(expr, &|c| placeholderize(c, u_names, swap))
}

/// A maximum matching of the left vertices (`adj[l]` lists the right
/// vertices `l` may take) into `0..n_right`, by augmenting paths.
fn bipartite_match(adj: &[Vec<usize>], n_right: usize) -> Vec<Option<usize>> {
  fn augment(
    l: usize,
    adj: &[Vec<usize>],
    seen: &mut [bool],
    right_of: &mut [Option<usize>],
  ) -> bool {
    for &r in &adj[l] {
      if seen[r] {
        continue;
      }
      seen[r] = true;
      if right_of[r].is_none_or(|l2| augment(l2, adj, seen, right_of)) {
        right_of[r] = Some(l);
        return true;
      }
    }
    false
  }
  let mut right_of: Vec<Option<usize>> = vec![None; n_right];
  for l in 0..adj.len() {
    let mut seen = vec![false; n_right];
    augment(l, adj, &mut seen, &mut right_of);
  }
  let mut left_to = vec![None; adj.len()];
  for (r, l) in right_of.iter().enumerate() {
    if let Some(l) = l {
      left_to[*l] = Some(r);
    }
  }
  left_to
}

/// Fornberg's finite-difference weights for the `m`-th derivative at `z`
/// from the values at the nodes `xs`.
pub(super) fn fd_weights(z: f64, xs: &[f64], m: usize) -> Vec<f64> {
  let n = xs.len();
  let mut c = vec![vec![0.0; m + 1]; n];
  let mut c1 = 1.0;
  let mut c4 = xs[0] - z;
  c[0][0] = 1.0;
  for i in 1..n {
    let mn = i.min(m);
    let mut c2 = 1.0;
    let c5 = c4;
    c4 = xs[i] - z;
    for j in 0..i {
      let c3 = xs[i] - xs[j];
      c2 *= c3;
      if j == i - 1 {
        for k in (1..=mn).rev() {
          c[i][k] = c1 * (k as f64 * c[i - 1][k - 1] - c5 * c[i - 1][k]) / c2;
        }
        c[i][0] = -c1 * c5 * c[i - 1][0] / c2;
      }
      for k in (1..=mn).rev() {
        c[j][k] = (c4 * c[j][k] - k as f64 * c[j][k - 1]) / c3;
      }
      c[j][0] = c4 * c[j][0] / c3;
    }
    c1 = c2;
  }
  c.iter().map(|row| row[m]).collect()
}

/// A square banded matrix with `kl` sub- and `ku` super-diagonals, stored
/// by rows with room for the fill-in of partial pivoting, and its LU
/// factorization in place (LINPACK `dgbfa`/`dgbsl` order of operations).
struct Band {
  n: usize,
  kl: usize,
  ku: usize,
  ld: usize,
  a: Vec<f64>,
  piv: Vec<usize>,
}

impl Band {
  fn new(n: usize, kl: usize, ku: usize) -> Self {
    let ld = 2 * kl + ku + 1;
    Self {
      n,
      kl,
      ku,
      ld,
      a: vec![0.0; n * ld],
      piv: vec![0; n],
    }
  }

  fn at(&self, i: usize, j: usize) -> usize {
    i * self.ld + (j + self.kl - i)
  }

  fn get(&self, i: usize, j: usize) -> f64 {
    self.a[self.at(i, j)]
  }

  fn add(&mut self, i: usize, j: usize, v: f64) {
    let k = self.at(i, j);
    self.a[k] += v;
  }

  fn factor(&mut self) -> bool {
    let n = self.n;
    for k in 0..n {
      let last = (k + self.kl).min(n - 1);
      let mut p = k;
      let mut best = self.get(k, k).abs();
      for i in k + 1..=last {
        let v = self.get(i, k).abs();
        if v > best {
          best = v;
          p = i;
        }
      }
      if best == 0.0 || !best.is_finite() {
        return false;
      }
      self.piv[k] = p;
      let right = (k + self.kl + self.ku).min(n - 1);
      if p != k {
        for j in k..=right {
          let (a, b) = (self.at(k, j), self.at(p, j));
          self.a.swap(a, b);
        }
      }
      let pivot = self.get(k, k);
      for i in k + 1..=last {
        let l = self.get(i, k) / pivot;
        let ik = self.at(i, k);
        self.a[ik] = l;
        if l != 0.0 {
          for j in k + 1..=right {
            let kj = self.get(k, j);
            if kj != 0.0 {
              let ij = self.at(i, j);
              self.a[ij] -= l * kj;
            }
          }
        }
      }
    }
    true
  }

  fn solve(&self, b: &mut [f64]) {
    let n = self.n;
    for k in 0..n {
      let p = self.piv[k];
      if p != k {
        b.swap(k, p);
      }
      let bk = b[k];
      if bk != 0.0 {
        for i in k + 1..=(k + self.kl).min(n - 1) {
          b[i] -= self.get(i, k) * bk;
        }
      }
    }
    for k in (0..n).rev() {
      let mut s = b[k];
      for j in k + 1..=(k + self.kl + self.ku).min(n - 1) {
        s -= self.get(k, j) * b[j];
      }
      b[k] = s / self.get(k, k);
    }
  }
}

/// The finite-difference stencil of one derivative order at one grid
/// point: grid indices and weights.
struct Stencil {
  idx: Vec<usize>,
  w: Vec<f64>,
}

/// What one row of the semi-discrete system evaluates at its grid point.
#[derive(Clone, Copy)]
enum RowKind {
  /// The evolution equation with this index.
  Evolution(usize),
  /// The boundary condition with this index (an algebraic row).
  Boundary(usize),
  /// The zero natural flux of this unknown's evolution equation (finite
  /// element semantics of an end without a condition).
  NaturalFlux(usize),
}

/// The discretized system, ready for the integrator.
struct MolSystem {
  n: usize,
  n_pts: usize,
  periodic: bool,
  xs: Vec<f64>,
  /// Highest space-derivative order the rows read.
  k_max: usize,
  /// `stencils[k - 1][g]`: the `k`-th derivative at grid point `g`.
  stencils: Vec<Vec<Stencil>>,
  /// Compiled row functions over `[D{k}U{j}…, t, x]`.
  evolution_fns: Vec<NumFn>,
  boundary_fns: Vec<NumFn>,
  natural_fns: Vec<NumFn>,
  /// `rows[g][s]`: what row `s` of grid point `g` evaluates.
  rows: Vec<Vec<RowKind>>,
  /// `mass[g][s][j]`: the coefficient of `∂u_j/∂t` in row `s` at `g`.
  mass: Vec<Vec<Vec<f64>>>,
  /// Mass coefficients compiled over `[t, x]`, when they vary in time:
  /// `(evolution equation, unknown, coefficient)`.
  mass_t: Vec<(usize, usize, NumFn)>,
}

impl MolSystem {
  /// The state index of grid point `g`'s first unknown. A periodic grid
  /// interleaves its two halves (0, N-1, 1, N-2, …) so that every stencil,
  /// wrap-around included, stays within a narrow band.
  fn pos(&self, g: usize) -> usize {
    let p = if !self.periodic || 2 * g < self.n_pts {
      if self.periodic { 2 * g } else { g }
    } else {
      2 * (self.n_pts - 1 - g) + 1
    };
    p * self.n
  }

  /// `F(t, y)`, the right-hand sides of all rows.
  fn residual(&self, t: f64, y: &[f64]) -> Result<Vec<f64>, InterpreterError> {
    let n = self.n;
    let n_vars = (self.k_max + 1) * n + 2;
    let mut out = vec![0.0; y.len()];
    let mut vars = vec![0.0; n_vars];
    for g in 0..self.n_pts {
      let base = self.pos(g);
      vars[..n].copy_from_slice(&y[base..base + n]);
      for k in 1..=self.k_max {
        let st = &self.stencils[k - 1][g];
        for j in 0..n {
          let mut s = 0.0;
          for (&gi, &w) in st.idx.iter().zip(&st.w) {
            s += w * y[self.pos(gi) + j];
          }
          vars[k * n + j] = s;
        }
      }
      vars[n_vars - 2] = t;
      vars[n_vars - 1] = self.xs[g];
      for s in 0..n {
        out[base + s] = match self.rows[g][s] {
          RowKind::Evolution(e) => self.evolution_fns[e].eval(&vars)?,
          RowKind::Boundary(b) => self.boundary_fns[b].eval(&vars)?,
          RowKind::NaturalFlux(j) => self.natural_fns[j].eval(&vars)?,
        };
      }
    }
    Ok(out)
  }

  /// Refresh the time-dependent mass coefficients for time `t`.
  fn update_mass(&mut self, t: f64) -> Result<(), InterpreterError> {
    if self.mass_t.is_empty() {
      return Ok(());
    }
    for g in 0..self.n_pts {
      for s in 0..self.n {
        if let RowKind::Evolution(e) = self.rows[g][s] {
          for (e2, j, f) in &self.mass_t {
            if *e2 == e {
              self.mass[g][s][*j] = f.eval(&[t, self.xs[g]])?;
            }
          }
        }
      }
    }
    Ok(())
  }

  /// The grid points whose rows read grid point `g`'s values.
  fn readers(&self) -> Vec<Vec<usize>> {
    let mut readers: Vec<Vec<usize>> =
      (0..self.n_pts).map(|g| vec![g]).collect();
    for k in 1..=self.k_max {
      for (g, st) in self.stencils[k - 1].iter().enumerate() {
        for &gi in &st.idx {
          if !readers[gi].contains(&g) {
            readers[gi].push(g);
          }
        }
      }
    }
    readers
  }
}

/// The Jacobian's sparsity: column groups whose grid points no row reads
/// together, and the band widths of the interleaved state.
struct JacobianPlan {
  readers: Vec<Vec<usize>>,
  groups: Vec<Vec<usize>>,
  kl: usize,
  ku: usize,
}

fn jacobian_plan(sys: &MolSystem) -> JacobianPlan {
  let readers = sys.readers();
  let mut groups: Vec<Vec<usize>> = Vec::new();
  let mut taken: Vec<Vec<bool>> = Vec::new();
  for g in 0..sys.n_pts {
    let slot = taken
      .iter()
      .position(|rows| readers[g].iter().all(|&r| !rows[r]));
    let slot = slot.unwrap_or_else(|| {
      groups.push(Vec::new());
      taken.push(vec![false; sys.n_pts]);
      groups.len() - 1
    });
    groups[slot].push(g);
    for &r in &readers[g] {
      taken[slot][r] = true;
    }
  }
  let (mut kl, mut ku) = (0usize, 0usize);
  for (g, rs) in readers.iter().enumerate() {
    let col = sys.pos(g);
    for &r in rs {
      let row = sys.pos(r);
      if col > row {
        ku = ku.max(col - row);
      } else {
        kl = kl.max(row - col);
      }
    }
  }
  JacobianPlan {
    readers,
    groups,
    kl: kl + sys.n - 1,
    ku: ku + sys.n - 1,
  }
}

/// `W = M / (h γ) - J` in banded form, with `J = ∂F/∂y` by forward
/// differences over the column groups (all unknowns of a group's points at
/// once, one unknown at a time).
fn assemble_jacobian(
  sys: &MolSystem,
  plan: &JacobianPlan,
  t: f64,
  y: &[f64],
  f0: &[f64],
) -> Result<Option<Band>, InterpreterError> {
  let n = sys.n;
  let total = y.len();
  let mut jac = Band::new(total, plan.kl, plan.ku);
  let mut probe = y.to_vec();
  for group in &plan.groups {
    for j in 0..n {
      let deltas: Vec<f64> = group
        .iter()
        .map(|&g| f64::EPSILON.sqrt() * y[sys.pos(g) + j].abs().max(1e-5))
        .collect();
      for (&g, d) in group.iter().zip(&deltas) {
        probe[sys.pos(g) + j] += d;
      }
      let fp = sys.residual(t, &probe)?;
      for (&g, d) in group.iter().zip(&deltas) {
        let col = sys.pos(g) + j;
        probe[col] = y[col];
        for &r in &plan.readers[g] {
          let rb = sys.pos(r);
          for s in 0..n {
            let v = (fp[rb + s] - f0[rb + s]) / d;
            if !v.is_finite() {
              return Ok(None);
            }
            if v != 0.0 {
              jac.add(rb + s, col, v);
            }
          }
        }
      }
    }
  }
  Ok(Some(jac))
}

/// `-J + c M` as a fresh banded matrix.
fn shifted_matrix(sys: &MolSystem, jac: &Band, c: f64) -> Band {
  let mut w = Band::new(jac.n, jac.kl, jac.ku);
  for (dst, src) in w.a.iter_mut().zip(&jac.a) {
    *dst = -src;
  }
  if c != 0.0 {
    for g in 0..sys.n_pts {
      let b = sys.pos(g);
      for s in 0..sys.n {
        for j in 0..sys.n {
          let m = sys.mass[g][s][j];
          if m != 0.0 {
            w.add(b + s, b + j, c * m);
          }
        }
      }
    }
  }
  w
}

fn mass_times(sys: &MolSystem, v: &[f64]) -> Vec<f64> {
  let mut out = vec![0.0; v.len()];
  for g in 0..sys.n_pts {
    let b = sys.pos(g);
    for s in 0..sys.n {
      let mut acc = 0.0;
      for j in 0..sys.n {
        acc += sys.mass[g][s][j] * v[b + j];
      }
      out[b + s] = acc;
    }
  }
  out
}

/// Whether row `(g, s)` is algebraic (its mass row is zero).
fn is_algebraic(sys: &MolSystem, g: usize, s: usize) -> bool {
  sys.mass[g][s].iter().all(|m| *m == 0.0)
}

/// Make the initial state satisfy the algebraic rows (boundary conditions
/// the initial profile does not meet) by Newton iterations on the
/// algebraic rows' own unknowns, the differential values held fixed.
fn consistent_initial_state(
  sys: &MolSystem,
  plan: &JacobianPlan,
  t0: f64,
  y: &mut [f64],
) -> Result<bool, InterpreterError> {
  let alg: Vec<usize> = (0..sys.n_pts)
    .flat_map(|g| (0..sys.n).map(move |s| (g, s)))
    .filter(|&(g, s)| is_algebraic(sys, g, s))
    .map(|(g, s)| sys.pos(g) + s)
    .collect();
  if alg.is_empty() {
    return Ok(true);
  }
  let mut is_alg = vec![false; y.len()];
  for &i in &alg {
    is_alg[i] = true;
  }
  for _ in 0..20 {
    let f = sys.residual(t0, y)?;
    let worst = alg.iter().map(|&i| f[i].abs()).fold(0.0_f64, f64::max);
    let scale = y.iter().map(|v| v.abs()).fold(1e-300_f64, f64::max);
    if worst <= 1e-12 * scale.max(1.0) {
      return Ok(true);
    }
    let Some(jac) = assemble_jacobian(sys, plan, t0, y, &f)? else {
      return Ok(false);
    };
    // Rows of differential components become identity rows (no change).
    let mut a = Band::new(jac.n, jac.kl, jac.ku);
    for i in 0..jac.n {
      if is_alg[i] {
        let lo = i.saturating_sub(jac.kl);
        let hi = (i + jac.ku).min(jac.n - 1);
        for j in lo..=hi {
          if is_alg[j] {
            a.add(i, j, jac.get(i, j));
          }
        }
      } else {
        a.add(i, i, 1.0);
      }
    }
    if !a.factor() {
      return Ok(false);
    }
    let mut rhs: Vec<f64> = (0..y.len())
      .map(|i| if is_alg[i] { -f[i] } else { 0.0 })
      .collect();
    a.solve(&mut rhs);
    for i in 0..y.len() {
      if is_alg[i] {
        y[i] += rhs[i];
      }
    }
  }
  Ok(true)
}

/// Integrate `M y' = F(t, y)` from `t0` to `t1` by Rodas4 steps. Returns
/// the accepted `(t, y)` points (starting at `t0`), or `None` when the
/// integration fails.
fn integrate(
  sys: &mut MolSystem,
  plan: &JacobianPlan,
  y0: Vec<f64>,
  t0: f64,
  t1: f64,
) -> Result<Option<Vec<(f64, Vec<f64>)>>, InterpreterError> {
  const RTOL: f64 = 1e-6;
  const MAX_STEPS: usize = 20_000;
  let span = t1 - t0;
  let n_state = y0.len();
  // Absolute tolerances from each unknown's own scale.
  fn unknown_scale(sys: &MolSystem, y: &[f64], j: usize) -> f64 {
    (0..sys.n_pts)
      .map(|g| y[sys.pos(g) + j].abs())
      .fold(0.0_f64, f64::max)
  }
  let global = (0..sys.n)
    .map(|j| unknown_scale(sys, &y0, j))
    .fold(0.0_f64, f64::max)
    .max(1e-12);
  let mut scales: Vec<f64> = (0..sys.n)
    .map(|j| unknown_scale(sys, &y0, j).max(1e-3 * global))
    .collect();
  let mut atol = vec![0.0; n_state];
  let fill_atol = |atol: &mut [f64], scales: &[f64], sys: &MolSystem| {
    for g in 0..sys.n_pts {
      for j in 0..sys.n {
        atol[sys.pos(g) + j] = RTOL * scales[j];
      }
    }
  };
  fill_atol(&mut atol, &scales, sys);

  let h_max = span / 100.0;
  let h_min = 16.0 * f64::EPSILON * t0.abs().max(t1.abs()).max(span);
  let mut h = (span * 1e-6).max(h_min * 4.0);
  let mut t = t0;
  let mut y = y0;
  sys.update_mass(t)?;
  let mut f0 = sys.residual(t, &y)?;
  let mut points = vec![(t, y.clone())];
  let mut steps = 0usize;
  let mut failures = 0usize;
  while t1 - t > h_min {
    steps += 1;
    if steps > MAX_STEPS {
      return Ok(None);
    }
    h = h.min(h_max).min(t1 - t);
    let Some(jac) = assemble_jacobian(sys, plan, t, &y, &f0)? else {
      return Ok(None);
    };
    let dt = f64::EPSILON.sqrt() * t.abs().max(h);
    let ft: Vec<f64> = sys
      .residual(t + dt, &y)?
      .iter()
      .zip(&f0)
      .map(|(a, b)| (a - b) / dt)
      .collect();
    let mut w = shifted_matrix(sys, &jac, 1.0 / (h * RODAS_GAMMA));
    if !w.factor() {
      h /= 2.0;
      if h < h_min {
        return Ok(None);
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
              for i in 0..n_state {
                u[i] += a * kv[i];
              }
            }
          }
        } else {
          for i in 0..n_state {
            u[i] += ks[stage - 1][i];
          }
        }
        let tc = if stage <= 3 { RODAS_C[stage - 1] } else { 1.0 };
        let fu = sys.residual(t + tc * h, &u)?;
        (fu, if stage <= 3 { RODAS_D[stage] } else { 0.0 })
      };
      let mut rhs: Vec<f64> =
        fu.iter().zip(&ft).map(|(a, b)| a + h * tcoef * b).collect();
      if stage > 0 {
        let mut comb = vec![0.0; n_state];
        for (k, kv) in ks.iter().enumerate() {
          let c = RODAS_CC[stage - 1][k] / h;
          if c != 0.0 {
            for i in 0..n_state {
              comb[i] += c * kv[i];
            }
          }
        }
        for (r, m) in rhs.iter_mut().zip(mass_times(sys, &comb)) {
          *r += m;
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
        return Ok(None);
      }
      h /= 4.0;
      continue;
    }
    let y_new: Vec<f64> = (0..n_state).map(|i| u[i] + ks[5][i]).collect();
    let error = ((0..n_state)
      .map(|i| {
        let e = ks[5][i] / (atol[i] + RTOL * y[i].abs().max(y_new[i].abs()));
        e * e
      })
      .sum::<f64>()
      / n_state as f64)
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
    t = if t1 - (t + h) <= h_min { t1 } else { t + h };
    y = y_new;
    sys.update_mass(t)?;
    f0 = sys.residual(t, &y)?;
    if f0.iter().any(|v| !v.is_finite()) {
      return Ok(None);
    }
    points.push((t, y.clone()));
    // An unknown starting out at zero gets its tolerance from the scale
    // it reaches.
    let mut grew = false;
    for (j, sc) in scales.iter_mut().enumerate() {
      let now = unknown_scale(sys, &y, j);
      if now > *sc {
        *sc = now;
        grew = true;
      }
    }
    if grew {
      fill_atol(&mut atol, &scales, sys);
    }
    h *= factor;
  }
  Ok(Some(points))
}

/// The number of grid points: enough for every initial profile to be
/// resolved (its cubic interpolation between grid points within 10^-4 of
/// its own scale), at least 51 (64 periodic), at most 1025, and within
/// `MinPoints`/`MaxPoints`.
fn choose_points(
  ics: &[NumFn],
  x_min: f64,
  x_max: f64,
  periodic: bool,
  opts: &MolOptions,
) -> usize {
  let candidates: &[usize] = if periodic {
    &[64, 128, 256, 512, 1024]
  } else {
    &[51, 101, 201, 401, 801, 1025]
  };
  let resolved = |n_pts: usize| -> bool {
    let intervals = if periodic { n_pts } else { n_pts - 1 };
    let dx = (x_max - x_min) / intervals as f64;
    ics.iter().all(|ic| {
      let at = |x: f64| ic.eval(&[x]).unwrap_or(f64::NAN);
      let vals: Vec<f64> =
        (0..=intervals).map(|g| at(x_min + g as f64 * dx)).collect();
      let scale = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
      if scale == 0.0 || vals.iter().any(|v| !v.is_finite()) {
        return true;
      }
      (1..intervals.saturating_sub(1)).all(|g| {
        let mid = at(x_min + (g as f64 + 0.5) * dx);
        let cubic = (-vals[g - 1] + 9.0 * vals[g] + 9.0 * vals[g + 1]
          - vals[g + 2])
          / 16.0;
        (cubic - mid).abs() <= 1e-4 * scale
      })
    })
  };
  let mut n_pts = *candidates.last().unwrap();
  for &c in candidates {
    if resolved(c) {
      n_pts = c;
      break;
    }
  }
  if let Some(lo) = opts.min_points {
    n_pts = n_pts.max(lo);
  }
  if let Some(hi) = opts.max_points {
    n_pts = n_pts.min(hi + 1);
  }
  n_pts.max(5)
}

/// Solve the PDE system `eq_items` for `u_names` with `t_dom` as the
/// evolution variable and `x_dom` as the gridded one (`swap` when the
/// unknowns are called as `u[x, t]`). `natural_flux` gives an end without
/// a condition the zero-flux condition of the finite element method
/// instead of a one-sided equation. `Ok(None)` when the equations do not
/// have the supported shape.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_pde_mol(
  eq_items: &[Expr],
  all_names: &[String],
  t_dom: &PdeDomain,
  x_dom: &PdeDomain,
  dim0_dom: &PdeDomain,
  swap: bool,
  natural_flux: bool,
  opts: &MolOptions,
) -> Result<Option<Expr>, InterpreterError> {
  // A row that cannot be evaluated numerically (a complex coefficient, an
  // unknown function) means the system is not of this shape.
  match solve_pde_mol_inner(
    eq_items,
    all_names,
    t_dom,
    x_dom,
    dim0_dom,
    swap,
    natural_flux,
    opts,
  ) {
    Err(InterpreterError::EvaluationError(_)) => Ok(None),
    other => other,
  }
}

#[allow(clippy::too_many_arguments)]
fn solve_pde_mol_inner(
  eq_items: &[Expr],
  all_names: &[String],
  t_dom: &PdeDomain,
  x_dom: &PdeDomain,
  dim0_dom: &PdeDomain,
  swap: bool,
  natural_flux: bool,
  opts: &MolOptions,
) -> Result<Option<Expr>, InterpreterError> {
  let (t_name, x_name) = (t_dom.name.as_str(), x_dom.name.as_str());
  let near =
    |v: f64, target: f64| (v - target).abs() <= 1e-9 * target.abs().max(1.0);
  let is_ident = |e: &Expr, n: &str| matches!(e, Expr::Identifier(m) if m == n);

  // `DirichletCondition[u[…] == g, x == x0]` is the condition at that end.
  let mut eqs: Vec<Expr> = Vec::with_capacity(eq_items.len());
  for eq in eq_items {
    if let Expr::FunctionCall { name, args } = eq
      && name == "DirichletCondition"
    {
      let [cond, pred] = args.as_slice() else {
        return Ok(None);
      };
      let Some((Expr::Identifier(v), x0)) = as_equal_pair(pred) else {
        return Ok(None);
      };
      if v != x_name || nval_to_f64(x0).is_none() {
        return Ok(None);
      }
      eqs.push(crate::syntax::substitute_variable(cond, x_name, x0));
    } else {
      eqs.push(eq.clone());
    }
  }

  // Only the unknowns the equations mention are solved for; the others
  // come back as themselves.
  let mentioned: Vec<String> = all_names
    .iter()
    .filter(|u| {
      eqs
        .iter()
        .any(|e| mentions_head(e, u) || expr_contains_ident(e, u))
    })
    .cloned()
    .collect();
  if mentioned.is_empty() {
    return Ok(None);
  }
  let u_names = &mentioned[..];
  let n = u_names.len();

  enum Class {
    Evolution,
    Initial(usize, Expr),
    Bc(bool),
    Periodic(usize),
  }
  let mut classes: Vec<Class> = Vec::with_capacity(eqs.len());
  let mut occs_of: Vec<Vec<Occ>> = Vec::with_capacity(eqs.len());
  for eq in &eqs {
    let Some((lhs, rhs)) = as_equal_pair(eq) else {
      return Ok(None);
    };
    let mut occs = Vec::new();
    if collect_occurrences(eq, u_names, swap, &mut occs).is_none()
      || occs.is_empty()
    {
      return Ok(None);
    }
    let generic = occs
      .iter()
      .all(|o| is_ident(&o.t_arg, t_name) && is_ident(&o.x_arg, x_name));
    let class = if generic {
      if !occs.iter().any(|o| o.dt == 1)
        || occs.iter().any(|o| o.dt > 1 || (o.dt == 1 && o.dx > 0))
      {
        return Ok(None);
      }
      Class::Evolution
    } else if occs.iter().all(|o| {
      o.dt == 0
        && o.dx == 0
        && is_ident(&o.x_arg, x_name)
        && nval_to_f64(&o.t_arg).is_some_and(|v| near(v, t_dom.min))
    }) {
      let j = occs[0].j;
      let Some(value) =
        try_pde_initial_condition(eq, &u_names[j], x_name, t_dom.min, swap)
      else {
        return Ok(None);
      };
      if occs.len() != 1 || expr_references_any(&value, u_names) {
        return Ok(None);
      }
      Class::Initial(j, value)
    } else if occs.iter().all(|o| o.dt == 0 && is_ident(&o.t_arg, t_name)) {
      let mut ends = Vec::with_capacity(occs.len());
      for o in &occs {
        match nval_to_f64(&o.x_arg) {
          Some(v) if near(v, x_dom.min) => ends.push(false),
          Some(v) if near(v, x_dom.max) => ends.push(true),
          _ => return Ok(None),
        }
      }
      if ends.iter().all(|e| *e == ends[0]) {
        Class::Bc(ends[0])
      } else if occs.len() == 2
        && occs[0].j == occs[1].j
        && occs[0].dx == occs[1].dx
        && ends[0] != ends[1]
        && match_pde_term_roles(lhs, &u_names[occs[0].j], swap).is_some()
        && match_pde_term_roles(rhs, &u_names[occs[0].j], swap).is_some()
      {
        Class::Periodic(occs[0].j)
      } else {
        return Ok(None);
      }
    } else {
      return Ok(None);
    };
    classes.push(class);
    occs_of.push(occs);
  }

  // Each unknown's evolution equation (one with its time derivative).
  let evo_idx: Vec<usize> = (0..eqs.len())
    .filter(|&e| matches!(classes[e], Class::Evolution))
    .collect();
  if evo_idx.len() != n {
    return Ok(None);
  }
  let evo_adj: Vec<Vec<usize>> = (0..n)
    .map(|j| {
      (0..n)
        .filter(|&e| occs_of[evo_idx[e]].iter().any(|o| o.j == j && o.dt == 1))
        .collect()
    })
    .collect();
  let evo_of: Vec<usize> = match bipartite_match(&evo_adj, n)
    .into_iter()
    .collect::<Option<Vec<usize>>>()
  {
    Some(m) => m,
    None => return Ok(None),
  };

  // Initial conditions, exactly one per unknown.
  let mut ic_exprs: Vec<Option<Expr>> = vec![None; n];
  for c in &classes {
    if let Class::Initial(j, v) = c {
      if ic_exprs[*j].is_some() {
        return Ok(None);
      }
      ic_exprs[*j] = Some(v.clone());
    }
  }
  let Some(ic_exprs) = ic_exprs.into_iter().collect::<Option<Vec<Expr>>>()
  else {
    return Ok(None);
  };

  // Space orders: of each unknown in the evolution equations, overall.
  let mut s_order = vec![0usize; n];
  for &e in &evo_idx {
    for o in &occs_of[e] {
      s_order[o.j] = s_order[o.j].max(o.dx);
    }
  }
  let mut k_max = s_order.iter().copied().max().unwrap_or(0);
  for (e, c) in classes.iter().enumerate() {
    if matches!(c, Class::Bc(_)) {
      for o in &occs_of[e] {
        k_max = k_max.max(o.dx);
      }
    }
  }

  let periodic_set: Vec<usize> = classes
    .iter()
    .filter_map(|c| match c {
      Class::Periodic(j) => Some(*j),
      _ => None,
    })
    .collect();
  let bc_eqs = |hi: bool| -> Vec<usize> {
    (0..eqs.len())
      .filter(|&e| matches!(classes[e], Class::Bc(h) if h == hi))
      .collect()
  };
  let periodic = !periodic_set.is_empty();
  // `(row kinds at the low end, at the high end)` per unknown.
  let mut end_rows: [Vec<Option<RowKind>>; 2] = [vec![None; n], vec![None; n]];
  let mut missing_bc = false;
  if periodic {
    if (0..n).any(|j| s_order[j] > 0 && !periodic_set.contains(&j))
      || !bc_eqs(false).is_empty()
      || !bc_eqs(true).is_empty()
    {
      return Ok(None);
    }
  } else {
    if s_order.iter().any(|&s| s > 2) {
      return Ok(None);
    }
    for (side, hi) in [(0usize, false), (1usize, true)] {
      let bcs = bc_eqs(hi);
      let adj: Vec<Vec<usize>> = bcs
        .iter()
        .map(|&e| {
          (0..n)
            .filter(|&j| s_order[j] > 0 && occs_of[e].iter().any(|o| o.j == j))
            .collect()
        })
        .collect();
      let matched = bipartite_match(&adj, n);
      for (b, m) in matched.iter().enumerate() {
        let Some(j) = m else {
          return Ok(None);
        };
        end_rows[side][*j] = Some(RowKind::Boundary(bcs[b]));
      }
      for j in 0..n {
        if end_rows[side][j].is_none() && s_order[j] == 2 {
          if natural_flux {
            end_rows[side][j] = Some(RowKind::NaturalFlux(j));
          } else {
            missing_bc = true;
          }
        }
      }
    }
  }

  // Compile the rows over the placeholder variables.
  let mut var_names: Vec<String> = Vec::new();
  for k in 0..=k_max {
    for j in 0..n {
      var_names.push(format!("NDSolve$D{k}U{j}"));
    }
  }
  var_names.push(t_name.to_string());
  var_names.push(x_name.to_string());
  let ev = |e: Expr| crate::evaluator::evaluate_expr_to_expr(&e);
  // `∂e/∂v`; `D` of a piecewise coefficient leaves `If[cond, 0, 0]`
  // factors behind, collapsed so the result is recognised as constant.
  let partial = |e: &Expr, v: &str| -> Result<Expr, InterpreterError> {
    let d = ev(call("D", vec![e.clone(), Expr::Identifier(v.to_string())]))?;
    ev(collapse_identical_if_branches(&d))
  };
  let t_vars: Vec<String> = (0..n).map(|j| format!("NDSolve$T{j}")).collect();
  let has_placeholder = |e: &Expr| {
    t_vars.iter().any(|v| expr_contains_ident(e, v))
      || var_names[..var_names.len() - 2]
        .iter()
        .any(|v| expr_contains_ident(e, v))
  };
  let mut evolution_fns = Vec::with_capacity(n);
  let mut f_exprs = Vec::with_capacity(n);
  // `mass_exprs[e][j]`
  let mut mass_exprs: Vec<Vec<Expr>> = Vec::with_capacity(n);
  for &e in &evo_idx {
    let (lhs, rhs) = as_equal_pair(&eqs[e]).unwrap();
    let residual =
      placeholderize(&minus2(lhs.clone(), rhs.clone()), u_names, swap);
    let mut masses = Vec::with_capacity(n);
    for tv in &t_vars {
      let m = partial(&residual, tv)?;
      if has_placeholder(&m) {
        return Ok(None);
      }
      masses.push(m);
    }
    let mut r0 = residual;
    for tv in &t_vars {
      r0 = crate::syntax::substitute_variable(&r0, tv, &Expr::Integer(0));
    }
    let f = ev(neg1(r0))?;
    if t_vars.iter().any(|v| expr_contains_ident(&f, v)) {
      return Ok(None);
    }
    evolution_fns.push(NumFn::new(f.clone(), &var_names));
    f_exprs.push(f);
    mass_exprs.push(masses);
  }
  let mut boundary_fns = Vec::with_capacity(eqs.len());
  for (e, c) in classes.iter().enumerate() {
    if let Class::Bc(_) = c {
      let (lhs, rhs) = as_equal_pair(&eqs[e]).unwrap();
      let f = placeholderize(&minus2(rhs.clone(), lhs.clone()), u_names, swap);
      boundary_fns.push(NumFn::new(ev(f)?, &var_names));
    } else {
      boundary_fns.push(NumFn::new(Expr::Integer(0), &var_names));
    }
  }
  // The natural flux of unknown j's evolution equation: Σ_k c_k u_k,x with
  // c_k the coefficient of u_k,xx (evaluated at each end separately).
  let mut natural_fns = Vec::with_capacity(n);
  for j in 0..n {
    let needs = end_rows
      .iter()
      .any(|side| matches!(side[j], Some(RowKind::NaturalFlux(_))));
    if !needs {
      natural_fns.push(NumFn::new(Expr::Integer(0), &var_names));
      continue;
    }
    let f = &f_exprs[evo_of[j]];
    let mut terms = Vec::with_capacity(n);
    for k in 0..n {
      let c = partial(f, &format!("NDSolve$D2U{k}"))?;
      if has_placeholder(&c) {
        return Ok(None);
      }
      terms.push(times2(c, Expr::Identifier(format!("NDSolve$D1U{k}"))));
    }
    natural_fns.push(NumFn::new(ev(plus(terms))?, &var_names));
  }
  let ic_fns: Vec<NumFn> = ic_exprs
    .into_iter()
    .map(|e| NumFn::new(e, &[x_name.to_string()]))
    .collect();

  // The grid and its stencils.
  let n_pts = choose_points(&ic_fns, x_dom.min, x_dom.max, periodic, opts);
  let intervals = if periodic { n_pts } else { n_pts - 1 };
  let dx = (x_dom.max - x_dom.min) / intervals as f64;
  let xs: Vec<f64> = (0..n_pts).map(|g| x_dom.min + g as f64 * dx).collect();
  let p = opts.difference_order.unwrap_or(4);
  let mut stencils: Vec<Vec<Stencil>> = Vec::with_capacity(k_max);
  for k in 1..=k_max {
    let width = (2 * k.div_ceil(2) - 1 + p).min(n_pts);
    let half = width / 2;
    let mut per_point = Vec::with_capacity(n_pts);
    for g in 0..n_pts {
      let (idx, offsets): (Vec<usize>, Vec<f64>) = if periodic {
        (0..width)
          .map(|i| {
            let off = i as i64 - half as i64;
            let gi = (g as i64 + off).rem_euclid(n_pts as i64) as usize;
            (gi, off as f64 * dx)
          })
          .unzip()
      } else {
        let start = g.saturating_sub(half).min(n_pts - width);
        (start..start + width)
          .map(|gi| (gi, (gi as f64 - g as f64) * dx))
          .unzip()
      };
      let w = fd_weights(0.0, &offsets, k);
      per_point.push(Stencil { idx, w });
    }
    stencils.push(per_point);
  }

  // Rows per grid point and their mass coefficients.
  let mass_fns: Vec<Vec<NumFn>> = mass_exprs
    .iter()
    .map(|ms| {
      ms.iter()
        .map(|m| {
          NumFn::new(m.clone(), &[t_name.to_string(), x_name.to_string()])
        })
        .collect()
    })
    .collect();
  let mut mass_t = Vec::new();
  for (e, ms) in mass_exprs.iter().enumerate() {
    for (j, m) in ms.iter().enumerate() {
      if expr_contains_ident(m, t_name) {
        mass_t.push((
          e,
          j,
          NumFn::new(m.clone(), &[t_name.to_string(), x_name.to_string()]),
        ));
      }
    }
  }
  let mut rows = Vec::with_capacity(n_pts);
  let mut mass = Vec::with_capacity(n_pts);
  for g in 0..n_pts {
    let side = if periodic {
      None
    } else if g == 0 {
      Some(0)
    } else if g == n_pts - 1 {
      Some(1)
    } else {
      None
    };
    let mut row = Vec::with_capacity(n);
    let mut mrow = Vec::with_capacity(n);
    for s in 0..n {
      let kind = side
        .and_then(|sd| end_rows[sd][s])
        .unwrap_or(RowKind::Evolution(evo_of[s]));
      let m: Vec<f64> = match kind {
        RowKind::Evolution(e) => {
          let mut v = Vec::with_capacity(n);
          for j in 0..n {
            let val = mass_fns[e][j].eval(&[t_dom.min, xs[g]])?;
            if !val.is_finite() {
              return Ok(None);
            }
            v.push(val);
          }
          v
        }
        _ => vec![0.0; n],
      };
      row.push(kind);
      mrow.push(m);
    }
    rows.push(row);
    mass.push(mrow);
  }
  let mut sys = MolSystem {
    n,
    n_pts,
    periodic,
    xs,
    k_max,
    stencils,
    evolution_fns,
    boundary_fns,
    natural_fns,
    rows,
    mass,
    mass_t,
  };
  let plan = jacobian_plan(&sys);

  // The initial state, consistent with the algebraic rows.
  let mut y0 = vec![0.0; n_pts * n];
  for g in 0..n_pts {
    for j in 0..n {
      let v = ic_fns[j].eval(&[sys.xs[g]])?;
      if !v.is_finite() {
        return Ok(None);
      }
      y0[sys.pos(g) + j] = v;
    }
  }
  if !consistent_initial_state(&sys, &plan, t_dom.min, &mut y0)? {
    return Ok(None);
  }
  if missing_bc {
    crate::emit_message(&format!(
      "{}::bcart: Warning: an insufficient number of boundary conditions \
       have been specified for the direction of independent variable {}. \
       Artificial boundary effects may be present in the solution.",
      super::ndsolve_message_head(),
      x_name
    ));
  }
  let Some(points) = integrate(&mut sys, &plan, y0, t_dom.min, t_dom.max)?
  else {
    return Ok(None);
  };

  // Keep the rows an interpolation in time needs: about every hundredth of
  // the span, and wherever some unknown moved by 1% of its scale.
  let span = t_dom.max - t_dom.min;
  let scale_of = |y: &[f64], j: usize| {
    (0..n_pts)
      .map(|g| y[sys.pos(g) + j].abs())
      .fold(0.0_f64, f64::max)
  };
  let scales: Vec<f64> = (0..n)
    .map(|j| {
      points
        .iter()
        .map(|(_, y)| scale_of(y, j))
        .fold(0.0_f64, f64::max)
        .max(1e-300)
    })
    .collect();
  let mut kept: Vec<usize> = vec![0];
  for (i, (t, y)) in points.iter().enumerate().skip(1) {
    let (lt, ly) = &points[*kept.last().unwrap()];
    let moved = kept.len() < 1000
      && (0..n).any(|j| {
        (0..n_pts).any(|g| {
          let k = sys.pos(g) + j;
          (y[k] - ly[k]).abs() > 0.01 * scales[j]
        })
      });
    if i + 1 == points.len() || t - lt >= span / 100.0 * (1.0 - 1e-9) || moved {
      kept.push(i);
    }
  }
  let ts: Vec<f64> = kept.iter().map(|&i| points[i].0).collect();
  let mut out_xs = sys.xs.clone();
  if periodic {
    out_xs.push(x_dom.max);
  }
  let profile = |y: &[f64], j: usize| -> Vec<f64> {
    let mut row: Vec<f64> = (0..n_pts).map(|g| y[sys.pos(g) + j]).collect();
    if periodic {
      row.push(row[0]);
    }
    row
  };

  let t_is_dim0 = t_dom.name == dim0_dom.name;
  let (dim0, dim1) = if t_is_dim0 {
    (t_dom, x_dom)
  } else {
    (x_dom, t_dom)
  };
  let (c0, c1) = if t_is_dim0 {
    (&ts, &out_xs)
  } else {
    (&out_xs, &ts)
  };
  let reals =
    |v: &[f64]| Expr::List(v.iter().map(|x| Expr::Real(*x)).collect());
  let domain = Expr::List(
    vec![
      Expr::List(vec![Expr::Real(dim0.min), Expr::Real(dim0.max)].into()),
      Expr::List(vec![Expr::Real(dim1.min), Expr::Real(dim1.max)].into()),
    ]
    .into(),
  );
  let orders = Expr::List(vec![Expr::Integer(3), Expr::Integer(3)].into());
  let coords = Expr::List(vec![reals(c0), reals(c1)].into());
  let mut rules = Vec::with_capacity(all_names.len());
  for name in all_names {
    let Some(j) = u_names.iter().position(|u| u == name) else {
      rules.push(rule_expr(
        Expr::Identifier(name.clone()),
        Expr::Identifier(name.clone()),
      ));
      continue;
    };
    let tx: Vec<Vec<f64>> =
      kept.iter().map(|&i| profile(&points[i].1, j)).collect();
    let grid = if t_is_dim0 { tx } else { transpose_grid(&tx) };
    let grid_expr = Expr::List(grid.iter().map(|row| reals(row)).collect());
    rules.push(rule_expr(
      Expr::Identifier(name.clone()),
      call(
        "InterpolatingFunction",
        vec![domain.clone(), grid_expr, orders.clone(), coords.clone()],
      ),
    ));
  }
  Ok(Some(Expr::List(vec![Expr::List(rules.into())].into())))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fornberg_weights_reproduce_the_classic_stencils() {
    let w = fd_weights(0.0, &[-1.0, 0.0, 1.0], 2);
    assert!((w[0] - 1.0).abs() < 1e-12 && (w[1] + 2.0).abs() < 1e-12);
    let w = fd_weights(0.0, &[-2.0, -1.0, 0.0, 1.0, 2.0], 1);
    let want = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];
    for (a, b) in w.iter().zip(want) {
      assert!((a - b).abs() < 1e-12, "{w:?}");
    }
  }

  #[test]
  fn banded_lu_solves_with_pivoting() {
    // [[0, 1, 0], [2, 1, 1], [0, 3, 4]] x = [1, 5, 10] has x = [1, 1, 7/4].
    let mut a = Band::new(3, 1, 1);
    for (i, j, v) in [
      (0, 1, 1.0),
      (1, 0, 2.0),
      (1, 1, 1.0),
      (1, 2, 1.0),
      (2, 1, 3.0),
      (2, 2, 4.0),
    ] {
      a.add(i, j, v);
    }
    assert!(a.factor());
    let mut b = vec![1.0, 5.0, 10.0];
    a.solve(&mut b);
    // Check A x == b.
    let x = b;
    let r = [x[1], 2.0 * x[0] + x[1] + x[2], 3.0 * x[1] + 4.0 * x[2]];
    for (got, want) in r.iter().zip([1.0, 5.0, 10.0]) {
      assert!((got - want).abs() < 1e-12, "{x:?}");
    }
  }
}
