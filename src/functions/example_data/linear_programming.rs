//! `ExampleData[{"LinearProgramming", …}]` and
//! `ExampleData[{"LinearOptimization", …}]`: generated linear programs.
//!
//! Each problem has the recorded number of constraints and variables and
//! the recorded number of nonzero constraint coefficients, with objective and
//! coefficient values drawn from the recorded ranges. It is feasible and
//! bounded by construction: a feasible point is chosen first, the right-hand
//! sides are derived from it (equalities hold there, inequalities hold with
//! slack), and every variable has finite bounds. The same problem is served
//! in `LinearProgramming`'s form (`{c, m, b, bounds}`), in
//! `LinearOptimization`'s matrix form, and as equations.

use super::images::cached_opt;
use super::rng::Rng;
use super::tabular::{NumSpec, parse_column_numeric, quantize};
use super::{Collection, Entry, GENERATED_SOURCE};
use crate::functions::*;

/// A generated problem: minimize `c.x` subject to `m.x (≥ | = | ≤) b` and
/// `lo ≤ x ≤ hi`.
struct Problem {
  c: Vec<f64>,
  /// Rows of `(column, value)` entries.
  rows: Vec<Vec<(usize, f64)>>,
  /// Right-hand sides with the constraint type: 1 for ≥, 0 for =, -1 for ≤.
  b: Vec<(f64, i32)>,
  bounds: Vec<(f64, f64)>,
}

fn dims(entry: &Entry) -> Option<(usize, usize)> {
  match entry.exact("Dimensions")? {
    Expr::List(ref d) => match (d.first(), d.get(1)) {
      (Some(Expr::Integer(k)), Some(Expr::Integer(n))) => {
        Some((*k as usize, *n as usize))
      }
      _ => None,
    },
    _ => None,
  }
}

fn specs(entry: &Entry) -> (NumSpec, NumSpec) {
  match entry.stat("@Stats") {
    Some(Expr::List(ref s)) if s.len() == 2 => (
      parse_column_numeric(&s[0]).unwrap_or_default(),
      parse_column_numeric(&s[1]).unwrap_or_default(),
    ),
    _ => (NumSpec::default(), NumSpec::default()),
  }
}

/// A value drawn from a column's range and spread, never zero.
fn draw(rng: &mut Rng, spec: &NumSpec) -> f64 {
  let v = (spec.mean + spec.sd * rng.normal()).clamp(spec.min, spec.max);
  let v = match quantize(v, spec) {
    Expr::Integer(i) => i as f64,
    Expr::Real(r) => r,
    _ => v,
  };
  if v == 0.0 {
    if spec.max > 0.0 {
      spec.max.min(1.0)
    } else {
      -1.0
    }
  } else {
    v
  }
}

fn build(entry: &Entry) -> Option<Problem> {
  let (k, n) = dims(entry)?;
  if k == 0 || n == 0 {
    return None;
  }
  let nnz = match entry.stat("@Nonzeros") {
    Some(Expr::Integer(z)) => (z as usize).clamp(k.max(n), k * n),
    _ => k.max(n) * 3,
  };
  let (c_spec, m_spec) = specs(entry);
  let mut rng = entry.rng("lp");
  // Every row and every column gets at least one coefficient, the rest are
  // scattered; columns cluster around the row index like staircase LPs.
  let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); k];
  let mut seen = std::collections::HashSet::with_capacity(nnz);
  let mut place =
    |rows: &mut Vec<Vec<(usize, f64)>>, i: usize, j: usize, rng: &mut Rng| {
      if seen.insert((i, j)) {
        rows[i].push((j, draw(rng, &m_spec)));
        true
      } else {
        false
      }
    };
  let mut count = 0;
  for j in 0..n {
    let i = (j * k / n + rng.below(3)).min(k - 1);
    if place(&mut rows, i, j, &mut rng) {
      count += 1;
    }
  }
  for i in 0..k {
    if rows[i].is_empty()
      && place(&mut rows, i, (i * n / k).min(n - 1), &mut rng)
    {
      count += 1;
    }
  }
  let mut guard = 0;
  while count < nnz && guard < nnz * 20 + 1000 {
    guard += 1;
    let i = rng.below(k);
    let centre = (i * n / k) as i64;
    let spread = (n as f64 * 0.2).max(3.0) as i64;
    let j =
      (centre + rng.int_in(-spread, spread)).clamp(0, n as i64 - 1) as usize;
    let j = if rng.chance(0.1) { rng.below(n) } else { j };
    if place(&mut rows, i, j, &mut rng) {
      count += 1;
    }
  }
  for r in &mut rows {
    r.sort_by_key(|e| e.0);
  }
  // A feasible point inside the bounds, then right-hand sides from it.
  // The point is integral and equality rows get integral coefficients, so
  // equalities hold exactly (also in exact arithmetic); inequality
  // right-hand sides are rounded away from the point, keeping slack.
  let bounds: Vec<(f64, f64)> = (0..n)
    .map(|_| (0.0, (rng.int_in(1, 10) * 10) as f64))
    .collect();
  let x0: Vec<f64> = bounds
    .iter()
    .map(|&(_, hi)| rng.int_in(1, hi as i64 - 1) as f64)
    .collect();
  let mut b = Vec::with_capacity(k);
  for r in &mut rows {
    let kind = match rng.below(3) {
      0 => 0,
      1 => 1,
      _ => -1,
    };
    if kind == 0 {
      for e in r.iter_mut() {
        e.1 = if e.1.round() == 0.0 {
          e.1.signum()
        } else {
          e.1.round()
        };
      }
    }
    let ax: f64 = r.iter().map(|&(j, v)| v * x0[j]).sum();
    let slack = rng.range(0.01, 1.0 + ax.abs() * 0.1);
    b.push(match kind {
      0 => (ax, 0),
      1 => (((ax - slack) * 1000.0).floor() / 1000.0, 1),
      _ => (((ax + slack) * 1000.0).ceil() / 1000.0, -1),
    });
  }
  let c: Vec<f64> = (0..n)
    .map(|_| {
      if rng.chance(0.6) {
        0.0
      } else {
        draw(&mut rng, &c_spec)
      }
    })
    .collect();
  Some(Problem { c, rows, b, bounds })
}

fn real(v: f64) -> Expr {
  if v.fract() == 0.0 && v.abs() < 1e15 {
    Expr::Integer(v as i128)
  } else {
    Expr::Real(v)
  }
}

fn vector(v: &[f64]) -> Expr {
  Expr::List(v.iter().map(|&x| real(x)).collect::<Vec<_>>().into())
}

/// A `SparseArray` from rows of `(column, value)`.
fn sparse(rows: &[Vec<(usize, f64)>], ncols: usize) -> Expr {
  let mut ptr = vec![Expr::Integer(0)];
  let mut cols = Vec::new();
  let mut vals = Vec::new();
  for r in rows {
    for &(j, v) in r {
      cols.push(Expr::List(vec![Expr::Integer(j as i128 + 1)].into()));
      vals.push(Expr::Real(v));
    }
    ptr.push(Expr::Integer(cols.len() as i128));
  }
  let sa = call(
    "SparseArray",
    vec![
      id_expr("Automatic"),
      Expr::List(
        vec![
          Expr::Integer(rows.len() as i128),
          Expr::Integer(ncols as i128),
        ]
        .into(),
      ),
      Expr::Integer(0),
      Expr::List(
        vec![
          Expr::Integer(1),
          Expr::List(
            vec![Expr::List(ptr.into()), Expr::List(cols.into())].into(),
          ),
          Expr::List(vals.into()),
        ]
        .into(),
      ),
    ],
  );
  crate::evaluator::evaluate_expr_to_expr(&sa).unwrap_or(sa)
}

fn pair(a: f64, b: f64) -> Expr {
  Expr::List(vec![real(a), real(b)].into())
}

/// `{c, m, b, bounds}` for `LinearProgramming[c, m, b, bounds]`.
fn lp_data(p: &Problem) -> Expr {
  Expr::List(
    vec![
      vector(&p.c),
      sparse(&p.rows, p.c.len()),
      Expr::List(
        p.b
          .iter()
          .map(|&(v, s)| pair(v, s as f64))
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::List(
        p.bounds
          .iter()
          .map(|&(lo, hi)| pair(lo, hi))
          .collect::<Vec<_>>()
          .into(),
      ),
    ]
    .into(),
  )
}

/// `{c, {a, b}, {aeq, beq}}` for `LinearOptimization`: inequalities and
/// bounds as `a.x + b >= 0`, equalities as `aeq.x + beq == 0`.
fn lo_data(p: &Problem) -> Expr {
  let n = p.c.len();
  let mut ineq: Vec<Vec<(usize, f64)>> = Vec::new();
  let mut ineq_b = Vec::new();
  let mut eq: Vec<Vec<(usize, f64)>> = Vec::new();
  let mut eq_b = Vec::new();
  for (r, &(rhs, kind)) in p.rows.iter().zip(&p.b) {
    match kind {
      0 => {
        eq.push(r.clone());
        eq_b.push(-rhs);
      }
      1 => {
        ineq.push(r.clone());
        ineq_b.push(-rhs);
      }
      _ => {
        ineq.push(r.iter().map(|&(j, v)| (j, -v)).collect());
        ineq_b.push(rhs);
      }
    }
  }
  for (j, &(lo, hi)) in p.bounds.iter().enumerate() {
    ineq.push(vec![(j, 1.0)]);
    ineq_b.push(-lo);
    ineq.push(vec![(j, -1.0)]);
    ineq_b.push(hi);
  }
  Expr::List(
    vec![
      vector(&p.c),
      Expr::List(vec![sparse(&ineq, n), vector(&ineq_b)].into()),
      Expr::List(vec![sparse(&eq, n), vector(&eq_b)].into()),
    ]
    .into(),
  )
}

/// `{objective, constraints}` in variables `Subscript["Xj", "MPS"]`.
fn equations(p: &Problem) -> Expr {
  let var = |j: usize| {
    call(
      "Subscript",
      vec![str_expr(&format!("X{:02}", j + 1)), str_expr("MPS")],
    )
  };
  let linear = |terms: Vec<(usize, f64)>| -> Expr {
    let parts: Vec<Expr> = terms
      .into_iter()
      .map(|(j, v)| call("Times", vec![Expr::Real(v), var(j)]))
      .collect();
    if parts.is_empty() {
      Expr::Integer(0)
    } else {
      call("Plus", parts)
    }
  };
  let objective = linear(
    p.c
      .iter()
      .enumerate()
      .filter(|(_, v)| **v != 0.0)
      .map(|(j, v)| (j, *v))
      .collect(),
  );
  let mut constraints: Vec<Expr> = p
    .rows
    .iter()
    .zip(&p.b)
    .map(|(r, &(rhs, kind))| {
      let head = match kind {
        0 => "Equal",
        1 => "GreaterEqual",
        _ => "LessEqual",
      };
      call(head, vec![linear(r.clone()), Expr::Real(rhs)])
    })
    .collect();
  for (j, &(lo, hi)) in p.bounds.iter().enumerate() {
    constraints.push(call("LessEqual", vec![real(lo), var(j), real(hi)]));
  }
  let eqs = Expr::List(vec![objective, call("And", constraints)].into());
  crate::evaluator::evaluate_expr_to_expr(&eqs).unwrap_or(eqs)
}

pub struct LinearPrograms;

impl Collection for LinearPrograms {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let key = |p: &str| format!("{}/{}/{p}", entry.collection, entry.name);
    let lp = entry.collection == "LinearProgramming";
    let data = || {
      cached_opt(key("data"), || {
        let p = build(entry)?;
        Some(if lp { lp_data(&p) } else { lo_data(&p) })
      })
    };
    Some(match property {
      None | Some("LinearProgrammingData" | "LinearOptimizationData") => {
        data()?
      }
      Some("ConstraintMatrix") => cached_opt(key("matrix"), || {
        let p = build(entry)?;
        Some(sparse(&p.rows, p.c.len()))
      })?,
      Some("Equations") => {
        cached_opt(key("equations"), || Some(equations(&build(entry)?)))?
      }
      Some("Source") => str_expr(GENERATED_SOURCE),
      _ => return None,
    })
  }
}
