//! Delay differential equations for `NDSolve`.
//!
//! A system such as
//!
//! ```text
//! {z'[t] == -z[t] + z[t - tau], z[t /; t <= 0] == 1}
//! ```
//!
//! states the solution's *history* for `t <= t0` as a `Condition`-wrapped
//! argument and lets right-hand sides look up earlier values `f[t - c]`
//! with a constant delay `c`. It is integrated with the classical RK4
//! method on a fixed grid; delayed values come from cubic Hermite
//! interpolation of the nodes already computed (or from the history
//! function while `t - c <= t0`). The step is chosen small enough that a
//! delayed lookup never reaches into the step being computed.

use super::*;
use std::cell::{Cell, RefCell};

/// One dependent function of a first-order delay system.
struct DdeFunc {
  name: String,
  function_form: bool,
}

/// A delayed lookup `f[t - delay]` replaced by a placeholder variable.
struct DelayedRef {
  func: usize,
  delay: f64,
}

fn state_name(fname: &str) -> String {
  format!("__dde$state${fname}")
}

fn delayed_name(k: usize) -> String {
  format!("__dde$delayed${k}")
}

/// `f[t /; t <= a] == rhs` — the history of `f` up to `a`. The condition's
/// variable has to be the independent variable `t_name`: wolframscript
/// rejects `f[s /; s <= a]` (`NDSolve::ndsv`). Returns the function name, the
/// history's end point `a` and the right-hand side.
fn parse_history_condition(
  eq: &Expr,
  t_name: &str,
) -> Option<(String, f64, Expr)> {
  let (lhs, rhs) = as_equal_pair(eq)?;
  let Expr::FunctionCall { name, args } = lhs else {
    return None;
  };
  if args.len() != 1 {
    return None;
  }
  let Expr::FunctionCall {
    name: cond_head,
    args: cond_args,
  } = &args[0]
  else {
    return None;
  };
  if cond_head != "Condition" || cond_args.len() != 2 {
    return None;
  }
  if !matches!(&cond_args[0], Expr::Identifier(v) if v == t_name) {
    return None;
  }
  let Expr::Comparison {
    operands,
    operators,
  } = &cond_args[1]
  else {
    return None;
  };
  if operands.len() != 2
    || operators.len() != 1
    || !matches!(operators[0], ComparisonOp::LessEqual | ComparisonOp::Less)
    || !matches!(&operands[0], Expr::Identifier(v) if v == t_name)
  {
    return None;
  }
  let end = nval_to_f64(&operands[1])?;
  Some((name.clone(), end, rhs.clone()))
}

fn contains_history_condition(eqs: &[Expr], t_name: &str) -> bool {
  eqs
    .iter()
    .any(|e| parse_history_condition(e, t_name).is_some())
}

/// Replace `f[t]` by its state placeholder and every `f[t - c]` (constant
/// `c`) by a delayed-value placeholder. Sets `failed` when a call to a
/// system function has an argument that is not `t` minus a nonnegative
/// constant, or a derivative of a system function occurs.
fn rewrite_delayed(
  expr: &Expr,
  funcs: &[DdeFunc],
  t_name: &str,
  refs: &RefCell<Vec<DelayedRef>>,
  failed: &Cell<bool>,
) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && args.len() == 1
    && let Some(fi) = funcs.iter().position(|f| &f.name == name)
  {
    if matches!(&args[0], Expr::Identifier(v) if v == t_name) {
      return Expr::Identifier(state_name(name));
    }
    let diff = crate::evaluator::evaluate_expr_to_expr(&minus2(
      id_expr(t_name),
      args[0].clone(),
    ))
    .ok()
    .and_then(|d| nval_to_f64(&d));
    let Some(delay) = diff.filter(|d| d.is_finite() && *d >= 0.0) else {
      failed.set(true);
      return expr.clone();
    };
    if delay == 0.0 {
      return Expr::Identifier(state_name(name));
    }
    let mut refs = refs.borrow_mut();
    let k = if let Some(k) =
      refs.iter().position(|r| r.func == fi && r.delay == delay)
    {
      k
    } else {
      refs.push(DelayedRef { func: fi, delay });
      refs.len() - 1
    };
    return Expr::Identifier(delayed_name(k));
  }
  if funcs
    .iter()
    .any(|f| extract_derivative_order_and_point(expr, &f.name).is_some())
  {
    failed.set(true);
    return expr.clone();
  }
  map_children(expr, &|child| {
    rewrite_delayed(child, funcs, t_name, refs, failed)
  })
}

/// Cubic Hermite value at `s` between nodes `(t0, y0, d0)` and
/// `(t1, y1, d1)`.
fn hermite(
  t0: f64,
  y0: f64,
  d0: f64,
  t1: f64,
  y1: f64,
  d1: f64,
  s: f64,
) -> f64 {
  let h = t1 - t0;
  let u = (s - t0) / h;
  let u2 = u * u;
  let u3 = u2 * u;
  (2.0 * u3 - 3.0 * u2 + 1.0) * y0
    + (u3 - 2.0 * u2 + u) * h * d0
    + (-2.0 * u3 + 3.0 * u2) * y1
    + (u3 - u2) * h * d1
}

/// The solution computed so far: nodes, states and state derivatives.
struct Trajectory {
  ts: Vec<f64>,
  ys: Vec<Vec<f64>>,
  ds: Vec<Vec<f64>>,
}

impl Trajectory {
  /// Value of function `fi` at time `s` (`s` is at or before the last node
  /// unless the delay is shorter than a step, in which case the last
  /// node's tangent extrapolates).
  fn value(&self, fi: usize, s: f64) -> f64 {
    let n = self.ts.len() - 1;
    if s >= self.ts[n] {
      return self.ys[n][fi] + (s - self.ts[n]) * self.ds[n][fi];
    }
    let k = self
      .ts
      .partition_point(|t| *t <= s)
      .saturating_sub(1)
      .min(n - 1);
    hermite(
      self.ts[k],
      self.ys[k][fi],
      self.ds[k][fi],
      self.ts[k + 1],
      self.ys[k + 1][fi],
      self.ds[k + 1][fi],
      s,
    )
  }
}

/// `NDSolve` for a first-order system with a stated history and constant
/// delays. `Ok(None)` when the system has no history condition or is not
/// of that shape.
pub(super) fn ndsolve_dde(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Expr::List(domain) = &args[2] else {
    return Ok(None);
  };
  let [Expr::Identifier(t_name), lo, hi] = domain.as_slice() else {
    return Ok(None);
  };
  let (Some(t0), Some(t1)) = (nval_to_f64(lo), nval_to_f64(hi)) else {
    return Ok(None);
  };
  if t1.partial_cmp(&t0) != Some(std::cmp::Ordering::Greater) {
    return Ok(None);
  }

  let mut eqs: Vec<Expr> = Vec::new();
  flatten_eq_list(&args[0], &mut eqs);
  let eqs = flatten_chained_equalities(&eqs);
  if !contains_history_condition(&eqs, t_name) {
    return Ok(None);
  }

  let dep_items: Vec<&Expr> = match &args[1] {
    Expr::List(items) => items.iter().collect(),
    other => vec![other],
  };
  let mut funcs: Vec<DdeFunc> = Vec::new();
  for item in dep_items {
    match item {
      Expr::Identifier(name) => funcs.push(DdeFunc {
        name: name.clone(),
        function_form: true,
      }),
      Expr::FunctionCall { name, args: fargs }
        if fargs.len() == 1
          && matches!(&fargs[0], Expr::Identifier(a) if a == t_name) =>
      {
        funcs.push(DdeFunc {
          name: name.clone(),
          function_form: false,
        });
      }
      _ => return Ok(None),
    }
  }
  if funcs.is_empty() {
    return Ok(None);
  }
  let nf = funcs.len();

  // Histories (one per function) and first-order explicit equations.
  let mut history: Vec<Option<(f64, Expr)>> = vec![None; nf];
  let mut rhs_exprs: Vec<Option<Expr>> = vec![None; nf];
  for eq in &eqs {
    if let Some((name, end, value)) = parse_history_condition(eq, t_name) {
      let Some(fi) = funcs.iter().position(|f| f.name == name) else {
        return Ok(None);
      };
      if history[fi].is_some() {
        return Ok(None);
      }
      history[fi] = Some((end, value));
      continue;
    }
    let Some((lhs, rhs)) = as_equal_pair(eq) else {
      return Ok(None);
    };
    // A plain initial value `f[t0] == v` is a history of length zero.
    if let Expr::FunctionCall { name, args: fargs } = lhs
      && fargs.len() == 1
      && let Some(fi) = funcs.iter().position(|f| &f.name == name)
      && let Some(at) = nval_to_f64(&fargs[0])
    {
      if history[fi].is_some() {
        return Ok(None);
      }
      history[fi] = Some((at, rhs.clone()));
      continue;
    }
    let mut placed = false;
    for (fi, f) in funcs.iter().enumerate() {
      for (side, other) in [(lhs, rhs), (rhs, lhs)] {
        if let Some((1, point)) =
          extract_derivative_order_and_point(side, &f.name)
          && matches!(&point, Expr::Identifier(v) if v == t_name)
        {
          if rhs_exprs[fi].is_some() {
            return Ok(None);
          }
          rhs_exprs[fi] = Some(other.clone());
          placed = true;
          break;
        }
      }
      if placed {
        break;
      }
    }
    if !placed {
      return Ok(None);
    }
  }
  if history.iter().any(Option::is_none)
    || rhs_exprs.iter().any(Option::is_none)
  {
    return Ok(None);
  }

  // Compile right-hand sides over [t, states…, delayed…].
  let refs = RefCell::new(Vec::new());
  let failed = Cell::new(false);
  let rewritten: Vec<Expr> = rhs_exprs
    .iter()
    .flatten()
    .map(|e| rewrite_delayed(e, &funcs, t_name, &refs, &failed))
    .collect();
  if failed.get() {
    return Ok(None);
  }
  let refs = refs.into_inner();
  let mut var_names: Vec<String> = vec![t_name.clone()];
  var_names.extend(funcs.iter().map(|f| state_name(&f.name)));
  var_names.extend((0..refs.len()).map(delayed_name));
  let rhs_fns: Vec<NumFn> = rewritten
    .into_iter()
    .map(|e| NumFn::new(e, &var_names))
    .collect();
  let history_fns: Vec<(f64, NumFn)> = history
    .into_iter()
    .flatten()
    .map(|(end, e)| (end, NumFn::new(e, std::slice::from_ref(t_name))))
    .collect();

  // The solution follows the histories to where they end and is integrated
  // from there: a history reaching past the domain's start (`f[t /; t <= 1]`
  // on `{t, 0, 10}`) is the solution on that stretch, as in wolframscript.
  let start = history_fns[0].0;
  if history_fns.iter().any(|(end, _)| *end != start)
    || start < t0
    || start >= t1
  {
    return Ok(None);
  }
  let mut y0 = Vec::with_capacity(nf);
  for (_, f) in &history_fns {
    y0.push(f.eval(&[start])?);
  }

  // Step size: fine enough that the shortest delay spans several steps.
  let span = t1 - start;
  let min_delay = refs.iter().map(|r| r.delay).fold(f64::INFINITY, f64::min);
  let steps = if min_delay.is_finite() {
    (span / (min_delay / 4.0)).ceil().clamp(1000.0, 20000.0) as usize
  } else {
    1000
  };
  let h = span / steps as f64;

  let lookup = |traj: &Trajectory,
                t: f64,
                r: &DelayedRef|
   -> Result<f64, InterpreterError> {
    let s = t - r.delay;
    if s <= start {
      history_fns[r.func].1.eval(&[s])
    } else {
      Ok(traj.value(r.func, s))
    }
  };
  let derivative = |traj: &Trajectory,
                    t: f64,
                    y: &[f64]|
   -> Result<Vec<f64>, InterpreterError> {
    let mut vars = Vec::with_capacity(var_names.len());
    vars.push(t);
    vars.extend_from_slice(y);
    for r in &refs {
      vars.push(lookup(traj, t, r)?);
    }
    rhs_fns.iter().map(|f| f.eval(&vars)).collect()
  };

  let mut traj = Trajectory {
    ts: vec![start],
    ys: vec![y0],
    ds: Vec::new(),
  };
  for n in 0..steps {
    let t = traj.ts[n];
    let y = traj.ys[n].clone();
    let k1 = derivative(&traj, t, &y)?;
    traj.ds.push(k1.clone());
    let shifted = |k: &[f64], c: f64| -> Vec<f64> {
      y.iter().zip(k).map(|(a, b)| a + b * c).collect()
    };
    let k2 = derivative(&traj, t + h / 2.0, &shifted(&k1, h / 2.0))?;
    let k3 = derivative(&traj, t + h / 2.0, &shifted(&k2, h / 2.0))?;
    let k4 = derivative(&traj, t + h, &shifted(&k3, h))?;
    let next: Vec<f64> = (0..nf)
      .map(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
      .collect();
    if next.iter().any(|v| !v.is_finite()) {
      return Ok(None);
    }
    // Node times from the step count, so the last one is `t1` exactly.
    traj.ts.push(if n + 1 == steps {
      t1
    } else {
      start + (n + 1) as f64 * h
    });
    traj.ys.push(next);
  }
  let last_t = *traj.ts.last().unwrap();
  let last_y = traj.ys.last().unwrap().clone();
  let last_d = derivative(&traj, last_t, &last_y)?;
  traj.ds.push(last_d);

  let domain = Expr::List(
    vec![Expr::List(vec![Expr::Real(t0), Expr::Real(last_t)].into())].into(),
  );
  // The stretch the histories cover, sampled at the integration step.
  let history_nodes = ((start - t0) / h).ceil() as usize;
  let mut rules: Vec<Expr> = Vec::with_capacity(nf);
  for (fi, f) in funcs.iter().enumerate() {
    let mut rows = Vec::with_capacity(history_nodes + traj.ts.len());
    for k in 0..history_nodes {
      let t = t0 + (start - t0) * k as f64 / history_nodes as f64;
      let v = history_fns[fi].1.eval(&[t])?;
      rows.push(Expr::List(vec![Expr::Real(t), Expr::Real(v)].into()));
    }
    rows.extend(traj.ts.iter().zip(&traj.ys).map(|(t, y)| {
      Expr::List(vec![Expr::Real(*t), Expr::Real(y[fi])].into())
    }));
    let data = Expr::List(rows.into());
    let interp = call(
      "InterpolatingFunction",
      vec![
        domain.clone(),
        data,
        Expr::Integer(NDSOLVE_INTERPOLATION_ORDER),
      ],
    );
    rules.push(if f.function_form {
      rule_expr(Expr::Identifier(f.name.clone()), interp)
    } else {
      rule_expr(
        Expr::FunctionCall {
          name: f.name.clone(),
          args: vec![Expr::Identifier(t_name.clone())].into(),
        },
        Expr::CurriedCall {
          func: Box::new(interp),
          args: vec![Expr::Identifier(t_name.clone())],
        },
      )
    });
  }
  Ok(Some(Expr::List(vec![Expr::List(rules.into())].into())))
}
