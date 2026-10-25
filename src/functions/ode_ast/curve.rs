//! Implicit curves for `NDSolve`.
//!
//! A system whose algebraic equations (no derivative in them) leave exactly
//! one degree of freedom, closed by equations in the first derivatives —
//! typically a unit-speed condition —
//!
//! ```text
//! {x[s]^2 + y[s]^2 == 1, x'[s]^2 + y'[s]^2 == 1,
//!  x[0] == 1, y[0] == 0, x'[0] == 0, y'[0] == 1}
//! ```
//!
//! traces a curve `g(y) == 0` parametrised by arclength. The tangent
//! `y'` solves `J_g(y) y' == 0` together with the derivative equations
//! (Newton's method, started from the previous tangent so the direction
//! stays continuous and the stated `y'(s0)` picks the orientation). The
//! curve is advanced with RK4 and every step is projected back onto
//! `g(y) == 0`.

use super::*;

/// Number of steps the whole requested interval is nominally cut into, as
/// for the ordinary integrator.
const NOMINAL_STEPS: f64 = 1000.0;
const MAX_NEWTON_ITERATIONS: usize = 50;

struct Curve {
  /// Residuals of the algebraic equations over `[x, y…, y'…]`.
  algebraic: Vec<NumFn>,
  /// Residuals of the derivative equations over `[x, y…, y'…]`.
  derivative: Vec<NumFn>,
  n: usize,
  /// `WhenEvent`s on the curve, with what they need to be evaluated.
  events: Vec<StateEvent>,
  funcs: Vec<SysFunc>,
  var_names: Vec<String>,
  x_name: String,
}

impl Curve {
  fn vars(&self, x: f64, y: &[f64], p: &[f64]) -> Vec<f64> {
    let mut v = Vec::with_capacity(1 + 2 * self.n);
    v.push(x);
    v.extend_from_slice(y);
    v.extend_from_slice(p);
    v
  }

  fn constraint(
    &self,
    x: f64,
    y: &[f64],
  ) -> Result<Vec<f64>, InterpreterError> {
    let vars = self.vars(x, y, &vec![0.0; self.n]);
    self.algebraic.iter().map(|g| g.eval(&vars)).collect()
  }

  /// Jacobian of the algebraic equations with respect to `y`.
  fn jacobian(
    &self,
    x: f64,
    y: &[f64],
  ) -> Result<Vec<Vec<f64>>, InterpreterError> {
    let mut jac = vec![vec![0.0; self.n]; self.algebraic.len()];
    let mut probe = y.to_vec();
    for j in 0..self.n {
      let h = 1e-6 * (1.0 + y[j].abs());
      probe[j] = y[j] + h;
      let hi = self.constraint(x, &probe)?;
      probe[j] = y[j] - h;
      let lo = self.constraint(x, &probe)?;
      probe[j] = y[j];
      for (i, row) in jac.iter_mut().enumerate() {
        row[j] = (hi[i] - lo[i]) / (2.0 * h);
      }
    }
    Ok(jac)
  }

  /// The tangent `y'` at `(x, y)`, found by Newton's method from `guess`.
  fn tangent(
    &self,
    x: f64,
    y: &[f64],
    guess: &[f64],
  ) -> Result<Option<Vec<f64>>, InterpreterError> {
    let jac = self.jacobian(x, y)?;
    let mut p = guess.to_vec();
    let residual = |p: &[f64]| -> Result<Vec<f64>, InterpreterError> {
      let mut r: Vec<f64> = jac
        .iter()
        .map(|row| row.iter().zip(p).map(|(a, b)| a * b).sum())
        .collect();
      let vars = self.vars(x, y, p);
      for h in &self.derivative {
        r.push(h.eval(&vars)?);
      }
      Ok(r)
    };
    for _ in 0..MAX_NEWTON_ITERATIONS {
      let r = residual(&p)?;
      if r.iter().any(|v| !v.is_finite()) {
        return Ok(None);
      }
      let norm = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
      if norm < 1e-13 {
        return Ok(Some(p));
      }
      // Jacobian of the residual: the constraint rows are linear in `p`,
      // the derivative rows are differenced.
      let mut m = jac.clone();
      let mut probe = p.clone();
      let mut cols = vec![vec![0.0; self.n]; self.derivative.len()];
      for j in 0..self.n {
        let h = 1e-6 * (1.0 + p[j].abs());
        probe[j] = p[j] + h;
        let hi = self.vars(x, y, &probe);
        probe[j] = p[j] - h;
        let lo = self.vars(x, y, &probe);
        probe[j] = p[j];
        for (i, hf) in self.derivative.iter().enumerate() {
          cols[i][j] = (hf.eval(&hi)? - hf.eval(&lo)?) / (2.0 * h);
        }
      }
      m.extend(cols);
      let Some(dp) = solve_dense_linear(m, r.iter().map(|v| -v).collect())
      else {
        return Ok(None);
      };
      for (pi, d) in p.iter_mut().zip(&dp) {
        *pi += d;
      }
      if dp.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 1e-14 {
        let r = residual(&p)?;
        let norm = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        return Ok((norm < 1e-9).then_some(p));
      }
    }
    Ok(None)
  }

  /// Move `y` onto `g(y) == 0` along the shortest Newton steps.
  fn project(&self, x: f64, y: &mut [f64]) -> Result<bool, InterpreterError> {
    for _ in 0..MAX_NEWTON_ITERATIONS {
      let g = self.constraint(x, y)?;
      if g.iter().any(|v| !v.is_finite()) {
        return Ok(false);
      }
      if g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 1e-13 {
        return Ok(true);
      }
      let jac = self.jacobian(x, y)?;
      // Minimum-norm correction `dy = -Jᵀ (J Jᵀ)⁻¹ g`.
      let k = jac.len();
      let gram: Vec<Vec<f64>> = (0..k)
        .map(|a| {
          (0..k)
            .map(|b| jac[a].iter().zip(&jac[b]).map(|(u, v)| u * v).sum())
            .collect()
        })
        .collect();
      let Some(lambda) =
        solve_dense_linear(gram, g.iter().map(|v| -v).collect())
      else {
        return Ok(false);
      };
      let mut moved = 0.0_f64;
      for (j, yj) in y.iter_mut().enumerate() {
        let d: f64 = (0..k).map(|i| jac[i][j] * lambda[i]).sum();
        *yj += d;
        moved = moved.max(d.abs());
      }
      if moved < 1e-15 {
        return Ok(true);
      }
    }
    let g = self.constraint(x, y)?;
    Ok(g.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < 1e-9)
  }

  /// Trace the curve from `(x0, y0)` with initial tangent `p0` to `x_end`.
  /// The points come back in the order visited, starting after `x0`. A
  /// curve that cannot be followed any further (it leaves the region where
  /// the equations are defined) ends there, like an `NDSolve` solution cut
  /// short at a singularity.
  fn trace(
    &self,
    x0: f64,
    y0: &[f64],
    p0: &[f64],
    x_end: f64,
    total_span: f64,
  ) -> Result<Vec<(f64, Vec<f64>)>, InterpreterError> {
    let span = x_end - x0;
    let mut points = Vec::new();
    if span == 0.0 {
      return Ok(points);
    }
    // The nominal step, as for the ordinary integrator; a step that fails
    // (the curve leaves the region where the equations are real, or a
    // Newton solve does not converge) or turns the tangent too sharply is
    // halved and retried, and the step grows back after successes. Where
    // even a tiny step fails, the curve ends, like an `NDSolve` solution
    // cut short at a singularity.
    let nominal =
      span / (NOMINAL_STEPS * span.abs() / total_span).ceil().max(1.0);
    let min_step = nominal.abs() * 1e-9;
    let mut h = nominal;
    let mut x = x0;
    let mut y = y0.to_vec();
    let mut guess = p0.to_vec();
    let mut successes = 0u32;
    while (x_end - x) * span.signum() > 0.0 {
      let step = if (x + h - x_end) * span.signum() > 0.0 {
        x_end - x
      } else {
        h
      };
      if let Some((y_new, tangent)) =
        self.controlled_step(x, &y, &guess, step)?
      {
        if let Some((x_ev, y_ev, stop)) =
          self.event_on_step(x, &y, &guess, step, &y_new)?
        {
          points.push((x_ev, y_ev.clone()));
          if stop {
            return Ok(points);
          }
        }
        x = if step == x_end - x { x_end } else { x + step };
        y = y_new;
        guess = tangent;
        points.push((x, y.clone()));
        successes += 1;
        if successes >= 2 && h.abs() < nominal.abs() {
          h = (h * 2.0).abs().min(nominal.abs()) * span.signum();
          successes = 0;
        }
      } else {
        successes = 0;
        h /= 2.0;
        if h.abs() < min_step {
          return Ok(points);
        }
      }
    }
    Ok(points)
  }

  /// The first active `WhenEvent` that fires on the step of length `h`
  /// from `(x, y)` (ending at `y_new`): its point, located by bisecting the
  /// step, after its action has run, and whether the action stops the
  /// integration.
  fn event_on_step(
    &self,
    x: f64,
    y: &[f64],
    guess: &[f64],
    h: f64,
    y_new: &[f64],
  ) -> Result<Option<(f64, Vec<f64>, bool)>, InterpreterError> {
    for ev in self.events.iter().filter(|e| e.active.get()) {
      let value =
        |x: f64, y: &[f64]| ev.value(x, y, &self.funcs, &self.var_names);
      let (Some(before), Some(after)) = (value(x, y), value(x + h, y_new))
      else {
        continue;
      };
      if !ev.fires(before, after) {
        continue;
      }
      // Bisect the step length for the firing point.
      let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
      let mut at = (x + h, y_new.to_vec());
      for _ in 0..50 {
        let mid = f64::midpoint(lo, hi);
        let Some((y_mid, _)) = self.rk4_step(x, y, guess, h * mid)? else {
          break;
        };
        match value(x + h * mid, &y_mid) {
          Some(v) if ev.fires(before, v) => {
            hi = mid;
            at = (x + h * mid, y_mid);
          }
          Some(_) => lo = mid,
          None => break,
        }
      }
      let offsets: Vec<usize> = (0..self.n).collect();
      let outcome = ev.run(
        at.0,
        &at.1,
        &self.funcs,
        &offsets,
        &self.var_names,
        &self.x_name,
      )?;
      return Ok(Some((at.0, outcome.state, outcome.stop)));
    }
    Ok(None)
  }

  /// A step of length `h` whose local error is within tolerance: two half
  /// steps, accepted when they agree with the whole step to the Richardson
  /// estimate `|whole - halves| / 15 <= atol + rtol |y|`. `None` asks the
  /// caller for a shorter step.
  fn controlled_step(
    &self,
    x: f64,
    y: &[f64],
    guess: &[f64],
    h: f64,
  ) -> Result<Option<(Vec<f64>, Vec<f64>)>, InterpreterError> {
    const RTOL: f64 = 1e-8;
    const ATOL: f64 = 1e-10;
    let Some((whole, _)) = self.rk4_step(x, y, guess, h)? else {
      return Ok(None);
    };
    let Some((mid, t_mid)) = self.rk4_step(x, y, guess, h / 2.0)? else {
      return Ok(None);
    };
    let Some((halves, tangent)) =
      self.rk4_step(x + h / 2.0, &mid, &t_mid, h / 2.0)?
    else {
      return Ok(None);
    };
    let within = whole
      .iter()
      .zip(&halves)
      .all(|(a, b)| (a - b).abs() / 15.0 <= ATOL + RTOL * a.abs().max(b.abs()));
    Ok(within.then_some((halves, tangent)))
  }

  /// One RK4 step of length `h` along the curve, projected back onto it.
  /// `None` when a stage fails or the tangent turns by more than ~25°
  /// across the step (too long a step to follow the curve faithfully).
  fn rk4_step(
    &self,
    x: f64,
    y: &[f64],
    guess: &[f64],
    h: f64,
  ) -> Result<Option<(Vec<f64>, Vec<f64>)>, InterpreterError> {
    let Some(k1) = self.tangent(x, y, guess)? else {
      return Ok(None);
    };
    let stage = |scale: f64, k: &[f64]| -> Vec<f64> {
      y.iter().zip(k).map(|(a, b)| a + scale * b).collect()
    };
    let Some(k2) = self.tangent(x + h / 2.0, &stage(h / 2.0, &k1), &k1)? else {
      return Ok(None);
    };
    let Some(k3) = self.tangent(x + h / 2.0, &stage(h / 2.0, &k2), &k2)? else {
      return Ok(None);
    };
    let Some(k4) = self.tangent(x + h, &stage(h, &k3), &k3)? else {
      return Ok(None);
    };
    let cosine = |a: &[f64], b: &[f64]| {
      let dot: f64 = a.iter().zip(b).map(|(u, v)| u * v).sum();
      let na: f64 = a.iter().map(|u| u * u).sum::<f64>().sqrt();
      let nb: f64 = b.iter().map(|u| u * u).sum::<f64>().sqrt();
      dot / (na * nb)
    };
    if !(cosine(&k1, &k4) > 0.9) {
      return Ok(None);
    }
    let mut y_new: Vec<f64> = (0..self.n)
      .map(|j| y[j] + h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]))
      .collect();
    if !self.project(x + h, &mut y_new)? {
      return Ok(None);
    }
    Ok(Some((y_new, k4)))
  }
}

/// `NDSolve[{g(y) == 0 …, h(y') == 0 …, conditions}, {y…}, {x, lo, hi}]` for
/// algebraic equations `g` and first-order derivative equations `h` that
/// together determine as many equations as there are unknowns.
pub(super) fn ndsolve_implicit_curve(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::List(domain) = &args[2] else {
    return Ok(None);
  };
  let (Expr::Identifier(x_name), 3) = (&domain[0], domain.len()) else {
    return Ok(None);
  };
  let (Some(lo), Some(hi)) = (nval_to_f64(&domain[1]), nval_to_f64(&domain[2]))
  else {
    return Ok(None);
  };
  if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
    return Ok(None);
  }

  let dep_items: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  let mut funcs: Vec<SysFunc> = Vec::new();
  for item in dep_items {
    match item {
      Expr::Identifier(name) => funcs.push(SysFunc {
        name: name.clone(),
        order: 0,
        function_form: true,
        ics: Vec::new(),
      }),
      Expr::FunctionCall { name, args: fargs }
        if fargs.len() == 1
          && matches!(&fargs[0], Expr::Identifier(a) if a == x_name) =>
      {
        funcs.push(SysFunc {
          name: name.clone(),
          order: 0,
          function_form: false,
          ics: Vec::new(),
        });
      }
      _ => return Ok(None),
    }
  }
  let n = funcs.len();
  if n < 2 {
    return Ok(None);
  }

  let mut eq_items = Vec::new();
  flatten_eq_list(&args[0], &mut eq_items);
  let eq_items = flatten_chained_equalities(&eq_items);
  let mut algebraic_eqs = Vec::new();
  let mut derivative_eqs = Vec::new();
  // Initial `[value, slope]` per function and the point they are given at.
  let mut start: Vec<[Option<f64>; 2]> = vec![[None, None]; n];
  let mut x0: Option<f64> = None;
  let mut raw_events = Vec::new();
  for eq in &eq_items {
    if let Expr::FunctionCall { name, args: wargs } = eq
      && name == "WhenEvent"
      && wargs.len() == 2
    {
      raw_events.push(RawStateEvent {
        condition: wargs[0].clone(),
        action: wargs[1].clone(),
      });
      continue;
    }
    if let Some(fi) = funcs
      .iter()
      .position(|f| is_initial_condition(eq, &f.name, x_name))
    {
      let Some((order, at, value)) =
        parse_numeric_initial_condition(eq, &funcs[fi].name)
      else {
        return Ok(None);
      };
      if order > 1 || x0.is_some_and(|p| (p - at).abs() > 1e-12) {
        return Ok(None);
      }
      x0 = Some(at);
      start[fi][order] = Some(value);
      continue;
    }
    let order = funcs
      .iter()
      .map(|f| max_derivative_order(eq, &f.name))
      .max()
      .unwrap_or(0);
    match order {
      0 => algebraic_eqs.push(eq.clone()),
      1 => derivative_eqs.push(eq.clone()),
      _ => return Ok(None),
    }
  }
  if algebraic_eqs.is_empty()
    || derivative_eqs.is_empty()
    || algebraic_eqs.len() + derivative_eqs.len() != n
  {
    return Ok(None);
  }
  let Some(x0) = x0 else {
    return Ok(None);
  };
  if x0 < lo || x0 > hi {
    return Ok(None);
  }
  // Every value has to be given; a slope may be left out (it only seeds the
  // tangent's Newton iteration, which the given slopes orient), but not all
  // of them.
  let Some(y0) = start.iter().map(|s| s[0]).collect::<Option<Vec<f64>>>()
  else {
    return Ok(None);
  };
  if start.iter().all(|s| s[1].is_none()) {
    return Ok(None);
  }
  let p0: Vec<f64> = start.iter().map(|s| s[1].unwrap_or(0.0)).collect();

  let mut var_names = vec![x_name.clone()];
  var_names.extend(funcs.iter().map(|f| placeholder_name(&f.name, 0)));
  var_names.extend(funcs.iter().map(|f| placeholder_name(&f.name, 1)));
  let compile = |eqs: &[Expr]| -> Result<Vec<NumFn>, InterpreterError> {
    eqs
      .iter()
      .map(|eq| {
        let residual =
          substitute_function_values(&normalize_equation(eq)?, &funcs, x_name);
        Ok(NumFn::new(residual, &var_names))
      })
      .collect()
  };
  let events = raw_events
    .into_iter()
    .map(|raw| StateEvent::new(raw, &funcs, x_name, &var_names))
    .collect();
  let algebraic = compile(&algebraic_eqs)?;
  let derivative = compile(&derivative_eqs)?;
  let curve = Curve {
    algebraic,
    derivative,
    n,
    events,
    funcs,
    var_names: var_names.clone(),
    x_name: x_name.clone(),
  };
  let funcs = &curve.funcs;

  let mut y_start = y0;
  if !curve.project(x0, &mut y_start)? {
    return Ok(None);
  }
  let Some(p_start) = curve.tangent(x0, &y_start, &p0)? else {
    return Ok(None);
  };
  let total = hi - lo;
  let mut before = curve.trace(x0, &y_start, &p_start, lo, total)?;
  let after = curve.trace(x0, &y_start, &p_start, hi, total)?;
  if before.is_empty() && after.is_empty() {
    return Ok(None);
  }
  before.reverse();
  let mut points = before;
  points.push((x0, y_start));
  points.extend(after);

  let x_lo = points.first().map_or(lo, |p| p.0);
  let x_hi = points.last().map_or(hi, |p| p.0);
  let mut rules = Vec::with_capacity(n);
  for (fi, f) in funcs.iter().enumerate() {
    let domain = Expr::List(
      vec![Expr::List(vec![Expr::Real(x_lo), Expr::Real(x_hi)].into())].into(),
    );
    let data = Expr::List(
      points
        .iter()
        .map(|(x, y)| {
          Expr::List(vec![Expr::Real(*x), Expr::Real(y[fi])].into())
        })
        .collect(),
    );
    let interp = call(
      "InterpolatingFunction",
      vec![domain, data, Expr::Integer(NDSOLVE_INTERPOLATION_ORDER)],
    );
    rules.push(if f.function_form {
      rule_expr(Expr::Identifier(f.name.clone()), interp)
    } else {
      rule_expr(
        Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(x_name.clone())].into(),
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(x_name.clone())],
        },
      )
    });
  }
  Ok(Some(Expr::List(vec![Expr::List(rules.into())].into())))
}
