//! `NDEigensystem` and `NDEigenvalues` for one-dimensional Sturm–Liouville
//! operators `a u'' + c(x) u` (constant `a`) on an interval, with
//! homogeneous `DirichletCondition`s or, where none is given, the natural
//! (Neumann) condition at an end, as in wolframscript.
//!
//! The operator is discretized by finite volumes on a uniform grid, which
//! keeps the matrix symmetric (half cells at natural ends), and its
//! eigenvalues smallest in magnitude are found by Sturm bisection; the
//! eigenvectors follow by inverse iteration.

use crate::InterpreterError;
use crate::syntax::Expr;

fn ev(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn id(name: &str) -> Expr {
  Expr::Identifier(name.into())
}

/// Number of grid intervals.
const INTERVALS: usize = 2000;

/// A symmetric tridiagonal matrix: diagonal `d`, off-diagonal `e`.
struct Tridiagonal {
  d: Vec<f64>,
  e: Vec<f64>,
}

impl Tridiagonal {
  /// How many eigenvalues lie below `x` (the Sturm sequence count).
  fn count_below(&self, x: f64) -> usize {
    let mut count = 0;
    let mut q = 1.0;
    for i in 0..self.d.len() {
      let off = if i == 0 {
        0.0
      } else {
        self.e[i - 1] * self.e[i - 1] / q
      };
      q = self.d[i] - x - off;
      if q == 0.0 {
        q = -f64::EPSILON * (1.0 + x.abs());
      }
      if q < 0.0 {
        count += 1;
      }
    }
    count
  }

  /// The `k`-th smallest (0-based) eigenvalue, by bisection.
  fn eigenvalue(&self, k: usize) -> f64 {
    let n = self.d.len();
    let radius = |i: usize| {
      let l = if i > 0 { self.e[i - 1].abs() } else { 0.0 };
      let r = if i + 1 < n { self.e[i].abs() } else { 0.0 };
      l + r
    };
    let mut lo = (0..n)
      .map(|i| self.d[i] - radius(i))
      .fold(f64::INFINITY, f64::min);
    let mut hi = (0..n)
      .map(|i| self.d[i] + radius(i))
      .fold(f64::NEG_INFINITY, f64::max);
    for _ in 0..200 {
      let mid = f64::midpoint(lo, hi);
      if mid <= lo || mid >= hi {
        break;
      }
      if self.count_below(mid) > k {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    f64::midpoint(lo, hi)
  }

  /// The unit eigenvector of `lambda`, by inverse iteration.
  fn eigenvector(&self, lambda: f64) -> Vec<f64> {
    let n = self.d.len();
    let shift = lambda + 1e-10 * (1.0 + lambda.abs());
    let mut v: Vec<f64> = (0..n)
      .map(|i| 1.0 + 0.1 * ((i * 7919) % 13) as f64)
      .collect();
    for _ in 0..4 {
      // Solve (T - shift) w = v with the Thomas algorithm.
      let mut c = vec![0.0; n];
      let mut r = vec![0.0; n];
      let mut b0 = self.d[0] - shift;
      if b0 == 0.0 {
        b0 = 1e-300;
      }
      c[0] = if n > 1 { self.e[0] / b0 } else { 0.0 };
      r[0] = v[0] / b0;
      for i in 1..n {
        let mut m = self.d[i] - shift - self.e[i - 1] * c[i - 1];
        if m == 0.0 {
          m = 1e-300;
        }
        c[i] = if i + 1 < n { self.e[i] / m } else { 0.0 };
        r[i] = (v[i] - self.e[i - 1] * r[i - 1]) / m;
      }
      let mut w = vec![0.0; n];
      w[n - 1] = r[n - 1];
      for i in (0..n - 1).rev() {
        w[i] = r[i] - c[i] * w[i + 1];
      }
      let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
      if !norm.is_finite() || norm == 0.0 {
        break;
      }
      v = w.into_iter().map(|x| x / norm).collect();
    }
    v
  }

  /// The `count` eigenvalues smallest in magnitude, in that order.
  fn smallest_in_magnitude(&self, count: usize) -> Vec<f64> {
    let n = self.d.len();
    let negative = self.count_below(0.0);
    // Walk outwards from zero: below it the eigenvalues get more negative,
    // above it larger.
    let (mut below, mut above) = (negative as isize - 1, negative);
    let mut out = Vec::with_capacity(count);
    let mut next_below = (below >= 0).then(|| self.eigenvalue(below as usize));
    let mut next_above = (above < n).then(|| self.eigenvalue(above));
    while out.len() < count {
      match (next_below, next_above) {
        (Some(b), Some(a)) if b.abs() <= a.abs() => {
          out.push(b);
          below -= 1;
          next_below = (below >= 0).then(|| self.eigenvalue(below as usize));
        }
        (_, Some(a)) => {
          out.push(a);
          above += 1;
          next_above = (above < n).then(|| self.eigenvalue(above));
        }
        (Some(b), None) => {
          out.push(b);
          below -= 1;
          next_below = (below >= 0).then(|| self.eigenvalue(below as usize));
        }
        (None, None) => break,
      }
    }
    out
  }
}

/// A one-dimensional problem read from the arguments.
struct Problem {
  /// The coefficient `a` of `u''`.
  a: f64,
  /// The coefficient `c(x)` of `u`, as an expression in `x`.
  c: Expr,
  x: String,
  lo: f64,
  hi: f64,
  dirichlet_lo: bool,
  dirichlet_hi: bool,
}

/// `u[x]`, `u'[x]` and `u''[x]` replaced by placeholders, and the operator
/// split into their coefficients: `Some((a, b, c))` for `a u'' + b u' + c u`.
fn linear_coefficients(
  op: &Expr,
  u: &str,
) -> Result<Option<(Expr, Expr, Expr)>, InterpreterError> {
  let placeholders = ["Woxi`EigenU2", "Woxi`EigenU1", "Woxi`EigenU0"];
  // `u''`, `u'` and `u` become functions returning the placeholders, so
  // `u''[x]` reads `Woxi`EigenU2` and so on.
  let replaced = ev(&call(
    "ReplaceAll",
    vec![
      op.clone(),
      Expr::List(
        vec![
          Expr::Rule {
            pattern: Box::new(Expr::CurriedCall {
              func: Box::new(call("Derivative", vec![Expr::Integer(2)])),
              args: vec![id(u)],
            }),
            replacement: Box::new(call(
              "Function",
              vec![id("Woxi`EigenArg"), id(placeholders[0])],
            )),
          },
          Expr::Rule {
            pattern: Box::new(Expr::CurriedCall {
              func: Box::new(call("Derivative", vec![Expr::Integer(1)])),
              args: vec![id(u)],
            }),
            replacement: Box::new(call(
              "Function",
              vec![id("Woxi`EigenArg"), id(placeholders[1])],
            )),
          },
          Expr::Rule {
            pattern: Box::new(id(u)),
            replacement: Box::new(call(
              "Function",
              vec![id("Woxi`EigenArg"), id(placeholders[2])],
            )),
          },
        ]
        .into(),
      ),
    ],
  ))?;
  let coefficient = |p: &str| ev(&call("D", vec![replaced.clone(), id(p)]));
  let (a, b, c) = (
    coefficient(placeholders[0])?,
    coefficient(placeholders[1])?,
    coefficient(placeholders[2])?,
  );
  // Linear in u: what is left without the three terms is nothing, and the
  // coefficients do not involve u.
  let rest = ev(&call(
    "Expand",
    vec![call(
      "Plus",
      vec![
        replaced.clone(),
        call(
          "Times",
          vec![
            Expr::Integer(-1),
            call(
              "Plus",
              vec![
                call("Times", vec![a.clone(), id(placeholders[0])]),
                call("Times", vec![b.clone(), id(placeholders[1])]),
                call("Times", vec![c.clone(), id(placeholders[2])]),
              ],
            ),
          ],
        ),
      ],
    )],
  ))?;
  let free = |e: &Expr| {
    placeholders
      .iter()
      .all(|p| crate::functions::calculus_ast::is_constant_wrt(e, p))
      && crate::functions::calculus_ast::is_constant_wrt(e, u)
  };
  let zero = crate::functions::math_ast::try_eval_to_f64(&rest)
    .is_some_and(|v| v == 0.0);
  if !zero || !free(&a) || !free(&b) || !free(&c) {
    return Ok(None);
  }
  Ok(Some((a, b, c)))
}

/// Which ends a homogeneous `DirichletCondition[u[x] == 0, pred]` holds at.
fn dirichlet_ends(
  cond: &Expr,
  u: &str,
  x: &str,
  lo: f64,
  hi: f64,
) -> Result<Option<(bool, bool)>, InterpreterError> {
  let Expr::FunctionCall { name, args } = cond else {
    return Ok(None);
  };
  if name != "DirichletCondition" || args.len() != 2 {
    return Ok(None);
  }
  // The condition must be u[x] == 0.
  let Expr::Comparison {
    operands,
    operators,
  } = &args[0]
  else {
    return Ok(None);
  };
  if operands.len() != 2
    || !matches!(operators[0], crate::syntax::ComparisonOp::Equal)
  {
    return Ok(None);
  }
  let is_u = |e: &Expr| {
    matches!(e, Expr::FunctionCall { name, args }
      if name == u && matches!(args.as_slice(), [Expr::Identifier(v)] if v == x))
  };
  let zero = |e: &Expr| {
    crate::functions::math_ast::try_eval_to_f64(e).is_some_and(|v| v == 0.0)
  };
  if !((is_u(&operands[0]) && zero(&operands[1]))
    || (is_u(&operands[1]) && zero(&operands[0])))
  {
    return Ok(None);
  }
  let holds_at = |at: f64| -> Result<Option<bool>, InterpreterError> {
    let value = ev(&crate::syntax::substitute_variable(
      &args[1],
      x,
      &Expr::Real(at),
    ))?;
    Ok(match &value {
      Expr::Identifier(t) if t == "True" => Some(true),
      Expr::Identifier(f) if f == "False" => Some(false),
      _ => None,
    })
  };
  match (holds_at(lo)?, holds_at(hi)?) {
    (Some(a), Some(b)) => Ok(Some((a, b))),
    _ => Ok(None),
  }
}

/// The problem in `args` (`NDEigensystem[op or {op, conds…}, u[x] or u,
/// {x, xmin, xmax}, n, opts…]`), and whether `u` was given as `u[x]`.
fn read_problem(
  args: &[Expr],
) -> Result<Option<(Problem, bool, String, usize)>, InterpreterError> {
  if args.len() < 4 {
    return Ok(None);
  }
  let (u, applied) = match &args[1] {
    Expr::Identifier(u) => (u.clone(), false),
    Expr::FunctionCall { name, args: ua } if ua.len() == 1 => {
      (name.clone(), true)
    }
    _ => return Ok(None),
  };
  // `{x, xmin, xmax}`, or the region `Element[{x}, Line[{{xmin}, {xmax}}]]`.
  let (x, lo, hi) = match &args[2] {
    Expr::List(range) => match range.as_slice() {
      [Expr::Identifier(x), lo, hi] => (x.clone(), lo.clone(), hi.clone()),
      _ => return Ok(None),
    },
    Expr::FunctionCall { name, args: region }
      if name == "Element" && region.len() == 2 =>
    {
      let (
        Expr::List(vars),
        Expr::FunctionCall {
          name: line,
          args: ends,
        },
      ) = (&region[0], &region[1])
      else {
        return Ok(None);
      };
      let [Expr::Identifier(x)] = vars.as_slice() else {
        return Ok(None);
      };
      if line != "Line" || ends.len() != 1 {
        return Ok(None);
      }
      let Expr::List(points) = &ends[0] else {
        return Ok(None);
      };
      match points.as_slice() {
        [Expr::List(a), Expr::List(b)] if a.len() == 1 && b.len() == 1 => {
          (x.clone(), a[0].clone(), b[0].clone())
        }
        _ => return Ok(None),
      }
    }
    _ => return Ok(None),
  };
  let x = &x;
  let (lo, hi) = (&lo, &hi);
  let (Some(lo), Some(hi)) = (
    crate::functions::math_ast::try_eval_to_f64(lo),
    crate::functions::math_ast::try_eval_to_f64(hi),
  ) else {
    return Ok(None);
  };
  if hi <= lo {
    return Ok(None);
  }
  let Expr::Integer(count) = &args[3] else {
    return Ok(None);
  };
  if *count < 1 {
    return Ok(None);
  }
  let (op, conditions): (Expr, Vec<Expr>) = match &args[0] {
    Expr::List(items) if !items.is_empty() => {
      (items[0].clone(), items[1..].to_vec())
    }
    single => (single.clone(), Vec::new()),
  };
  // An operator written with inactive Div and Grad (as DiffusionPDETerm
  // gives it) is put in terms of derivatives first.
  let op = ev(&call("Activate", vec![op]))?;
  let Some((a, b, c)) = linear_coefficients(&op, &u)? else {
    return Ok(None);
  };
  let Some(a) =
    crate::functions::math_ast::try_eval_to_f64(&a).filter(|a| *a != 0.0)
  else {
    return Ok(None);
  };
  if !crate::functions::math_ast::try_eval_to_f64(&b).is_some_and(|b| b == 0.0)
  {
    return Ok(None);
  }
  let (mut dirichlet_lo, mut dirichlet_hi) = (false, false);
  for cond in &conditions {
    match dirichlet_ends(cond, &u, x, lo, hi)? {
      Some((l, h)) => {
        dirichlet_lo |= l;
        dirichlet_hi |= h;
      }
      None => return Ok(None),
    }
  }
  Ok(Some((
    Problem {
      a,
      c,
      x: x.clone(),
      lo,
      hi,
      dirichlet_lo,
      dirichlet_hi,
    },
    applied,
    x.clone(),
    *count as usize,
  )))
}

/// The eigenvalues and grid eigenfunctions (values at every grid node,
/// L²-normalized, positive just inside the left end).
fn solve(
  problem: &Problem,
  count: usize,
) -> Result<Option<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)>, InterpreterError> {
  let n = INTERVALS;
  let h = (problem.hi - problem.lo) / n as f64;
  let nodes: Vec<f64> = (0..=n).map(|i| problem.lo + i as f64 * h).collect();
  // c(x) at every node.
  let values = ev(&call(
    "Map",
    vec![
      Expr::Function {
        body: Box::new(crate::syntax::substitute_variable(
          &problem.c,
          &problem.x,
          &Expr::Slot(1),
        )),
      },
      Expr::List(nodes.iter().map(|v| Expr::Real(*v)).collect()),
    ],
  ))?;
  let Expr::List(values) = &values else {
    return Ok(None);
  };
  let c: Option<Vec<f64>> = values
    .iter()
    .map(crate::functions::math_ast::try_eval_to_f64)
    .collect();
  let Some(c) = c else { return Ok(None) };
  // The unknowns: every node but a Dirichlet end.
  let first = usize::from(problem.dirichlet_lo);
  let last = if problem.dirichlet_hi { n - 1 } else { n };
  if last < first {
    return Ok(None);
  }
  let unknowns: Vec<usize> = (first..=last).collect();
  let weight = |i: usize| if i == 0 || i == n { h / 2.0 } else { h };
  // Finite volumes: w_i (L u)_i = a (u_{i+1} - u_i)/h - a (u_i - u_{i-1})/h
  // + c_i u_i w_i, the flux through a natural end being zero.
  let m = unknowns.len();
  let mut d = vec![0.0; m];
  let mut e = vec![0.0; m.saturating_sub(1)];
  for (k, &i) in unknowns.iter().enumerate() {
    let neighbours = usize::from(i > 0) + usize::from(i < n);
    let w = weight(i);
    d[k] = (-problem.a * neighbours as f64 / h + c[i] * w) / w;
    if k + 1 < m {
      let j = unknowns[k + 1];
      e[k] = problem.a / h / (w * weight(j)).sqrt();
    }
  }
  let matrix = Tridiagonal { d, e };
  let mut lambdas = matrix.smallest_in_magnitude(count.min(m));
  let mut functions = Vec::with_capacity(lambdas.len());
  for lambda in &mut lambdas {
    let lambda_value = *lambda;
    let y = matrix.eigenvector(lambda_value);
    let mut full = vec![0.0; n + 1];
    for (k, &i) in unknowns.iter().enumerate() {
      full[i] = y[k] / weight(i).sqrt();
    }
    // The Rayleigh quotient in its energy form, (-a Σ (Δu)²/h + Σ c w u²) /
    // Σ w u², refines the eigenvalue: it is exact for a constant mode, where
    // the matrix form would leave rounding of the size of its norm.
    let mass: f64 = (0..=n).map(|i| weight(i) * full[i] * full[i]).sum();
    if mass > 0.0 {
      let gradient: f64 =
        (0..n).map(|i| (full[i + 1] - full[i]).powi(2) / h).sum();
      let potential: f64 =
        (0..=n).map(|i| c[i] * weight(i) * full[i] * full[i]).sum();
      *lambda = (-problem.a * gradient + potential) / mass;
    }
    // Normalize: Σ w_i u_i² = 1.
    let norm = mass.sqrt();
    if norm > 0.0 {
      for v in &mut full {
        *v /= norm;
      }
    }
    let peak = full.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if let Some(lead) = full.iter().find(|v| v.abs() > 1e-3 * peak)
      && *lead < 0.0
    {
      for v in &mut full {
        *v = -*v;
      }
    }
    functions.push(full);
  }
  Ok(Some((lambdas, nodes, functions)))
}

/// `NDEigensystem[…]`: `{eigenvalues, eigenfunctions}`.
pub fn nd_eigensystem_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some((problem, applied, x, count)) = read_problem(args)? else {
    return Ok(crate::helpers::unevaluated("NDEigensystem", args));
  };
  let Some((lambdas, nodes, functions)) = solve(&problem, count)? else {
    return Ok(crate::helpers::unevaluated("NDEigensystem", args));
  };
  let mut eigenfunctions = Vec::with_capacity(functions.len());
  for f in functions {
    let data = Expr::List(
      nodes
        .iter()
        .zip(&f)
        .map(|(x, v)| Expr::List(vec![Expr::Real(*x), Expr::Real(*v)].into()))
        .collect(),
    );
    let interpolation = ev(&call("Interpolation", vec![data]))?;
    eigenfunctions.push(if applied {
      Expr::CurriedCall {
        func: Box::new(interpolation),
        args: vec![id(&x)],
      }
    } else {
      interpolation
    });
  }
  Ok(Expr::List(
    vec![
      Expr::List(lambdas.into_iter().map(Expr::Real).collect()),
      Expr::List(eigenfunctions.into()),
    ]
    .into(),
  ))
}

/// `NDEigenvalues[…]`: the eigenvalues alone.
pub fn nd_eigenvalues_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some((problem, _, _, count)) = read_problem(args)? else {
    return Ok(crate::helpers::unevaluated("NDEigenvalues", args));
  };
  let Some((lambdas, _, _)) = solve(&problem, count)? else {
    return Ok(crate::helpers::unevaluated("NDEigenvalues", args));
  };
  Ok(Expr::List(lambdas.into_iter().map(Expr::Real).collect()))
}
