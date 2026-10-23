//! `Interpolation[{{{x1}, f1, f1', …}, {{x2}, f2, …}, …}]`: interpolation of
//! values together with derivatives (Hermite data), stored as
//! `InterpolatingFunction[{{xmin, xmax}}, rows, order]` with the rows kept in
//! their `{{x}, f, f', …}` form.
//!
//! As in wolframscript, the piece used on `[x_i, x_{i+1}]` is the polynomial
//! that matches every value and derivative given at both ends. While that is
//! fewer than `order + 1` conditions, whole neighbouring points join it, the
//! right-hand one first and then alternating sides. Plain values on four
//! points thus give the familiar centred cubic, and `f, f'` at each end give
//! the cubic Hermite piece whatever the `InterpolationOrder`.

use crate::InterpreterError;
use crate::evaluator::evaluate_function_call_ast;
use crate::functions::ode_ast::is_exact_number;
use crate::syntax::Expr;

/// One data point: its coordinate and `f, f', f'', …` there.
#[derive(Clone)]
pub struct Row {
  x: Expr,
  values: Vec<Expr>,
}

/// The `{{x}, f, f', …}` rows of `data`, sorted by `x`; `None` unless every
/// row has that shape with a numeric coordinate and at least one value.
pub fn parse_rows(data: &[Expr]) -> Option<Vec<Row>> {
  let mut rows: Vec<(f64, Row)> = data
    .iter()
    .map(|item| {
      let Expr::List(parts) = item else {
        return None;
      };
      let (Expr::List(coord), values) = parts.split_first()? else {
        return None;
      };
      let [x] = coord.as_slice() else {
        return None;
      };
      if values.is_empty() {
        return None;
      }
      let xf = to_f64(x)?;
      Some((
        xf,
        Row {
          x: x.clone(),
          values: values.to_vec(),
        },
      ))
    })
    .collect::<Option<_>>()?;
  rows
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
  Some(rows.into_iter().map(|(_, r)| r).collect())
}

/// Whether some row carries a derivative, i.e. the data needs this module
/// rather than plain value interpolation.
pub fn has_derivatives(rows: &[Row]) -> bool {
  rows.iter().any(|r| r.values.len() > 1)
}

pub fn rows_expr(rows: &[Row]) -> Expr {
  Expr::List(
    rows
      .iter()
      .map(|r| {
        let mut parts = vec![Expr::List(vec![r.x.clone()].into())];
        parts.extend(r.values.iter().cloned());
        Expr::List(parts.into())
      })
      .collect(),
  )
}

/// `InterpolatingFunction[{{xmin, xmax}}, rows, order]`.
pub fn build(rows: &[Row], order: usize) -> Expr {
  let domain = Expr::List(
    vec![Expr::List(
      vec![rows[0].x.clone(), rows[rows.len() - 1].x.clone()].into(),
    )]
    .into(),
  );
  Expr::FunctionCall {
    name: "InterpolatingFunction".into(),
    args: vec![domain, rows_expr(rows), Expr::Integer(order as i128)].into(),
  }
}

fn to_f64(e: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64(e)
}

/// The number type the polynomial is built in: machine reals, or exact
/// numbers through the evaluator.
trait Field: Clone {
  fn from_expr(e: &Expr) -> Option<Self>;
  fn int(n: i128) -> Self;
  fn add(&self, o: &Self) -> Self;
  fn sub(&self, o: &Self) -> Self;
  fn mul(&self, o: &Self) -> Self;
  fn div(&self, o: &Self) -> Self;
  fn same(&self, o: &Self) -> bool;
  fn into_expr(self) -> Expr;
}

impl Field for f64 {
  fn from_expr(e: &Expr) -> Option<Self> {
    to_f64(e)
  }
  fn int(n: i128) -> Self {
    n as f64
  }
  fn add(&self, o: &Self) -> Self {
    self + o
  }
  fn sub(&self, o: &Self) -> Self {
    self - o
  }
  fn mul(&self, o: &Self) -> Self {
    self * o
  }
  fn div(&self, o: &Self) -> Self {
    self / o
  }
  fn same(&self, o: &Self) -> bool {
    self == o
  }
  fn into_expr(self) -> Expr {
    Expr::Real(self)
  }
}

#[derive(Clone)]
struct Exact(Expr);

impl Exact {
  fn op(name: &str, a: &Expr, b: &Expr) -> Self {
    Exact(
      evaluate_function_call_ast(name, &[a.clone(), b.clone()])
        .unwrap_or(Expr::Identifier("Indeterminate".into())),
    )
  }
}

impl Field for Exact {
  fn from_expr(e: &Expr) -> Option<Self> {
    is_exact_number(e).then(|| Exact(e.clone()))
  }
  fn int(n: i128) -> Self {
    Exact(Expr::Integer(n))
  }
  fn add(&self, o: &Self) -> Self {
    Self::op("Plus", &self.0, &o.0)
  }
  fn sub(&self, o: &Self) -> Self {
    Self::op("Subtract", &self.0, &o.0)
  }
  fn mul(&self, o: &Self) -> Self {
    Self::op("Times", &self.0, &o.0)
  }
  fn div(&self, o: &Self) -> Self {
    Self::op("Divide", &self.0, &o.0)
  }
  fn same(&self, o: &Self) -> bool {
    matches!(
      evaluate_function_call_ast("SameQ", &[self.0.clone(), o.0.clone()]),
      Ok(Expr::Identifier(ref t)) if t == "True"
    )
  }
  fn into_expr(self) -> Expr {
    self.0
  }
}

/// The rows the piece around `x` is built from (see the module docs).
fn stencil(
  rows: &[Row],
  xs: &[f64],
  x: f64,
  order: usize,
) -> std::ops::Range<usize> {
  let n = rows.len();
  if n == 1 {
    return 0..1;
  }
  // Interval [idx, idx + 1] containing x (the end ones outside the data).
  let idx = xs
    .windows(2)
    .position(|w| x <= w[1])
    .unwrap_or(n - 2)
    .min(n - 2);
  let (mut lo, mut hi) = (idx, idx + 1);
  let conds = |lo: usize, hi: usize| -> usize {
    rows[lo..=hi].iter().map(|r| r.values.len()).sum()
  };
  let mut right_next = true;
  while conds(lo, hi) < order + 1 && (lo > 0 || hi + 1 < n) {
    if (right_next && hi + 1 < n) || lo == 0 {
      hi += 1;
    } else {
      lo -= 1;
    }
    right_next = !right_next;
  }
  lo..hi + 1
}

/// `f^(derivative)(x)` of the piece through `rows`, by confluent divided
/// differences and a shift of the Newton form to Taylor form at `x`.
fn piece_value<F: Field>(rows: &[Row], x: &F, derivative: usize) -> Option<F> {
  let mut z: Vec<F> = Vec::new();
  // For each expanded node, the row it came from.
  let mut owner: Vec<usize> = Vec::new();
  for (k, r) in rows.iter().enumerate() {
    let xr = F::from_expr(&r.x)?;
    for _ in 0..r.values.len() {
      z.push(xr.clone());
      owner.push(k);
    }
  }
  let m = z.len();
  // factorial(j) as F
  let fact = |j: usize| F::int((1..=j as i128).product::<i128>().max(1));
  let mut table: Vec<Vec<F>> = Vec::with_capacity(m);
  for i in 0..m {
    table.push(vec![F::from_expr(&rows[owner[i]].values[0])?]);
  }
  for j in 1..m {
    for i in (j..m).rev() {
      let v = if owner[i] == owner[i - j] {
        let d = rows[owner[i]].values.get(j)?;
        F::from_expr(d)?.div(&fact(j))
      } else {
        table[i][j - 1]
          .sub(&table[i - 1][j - 1])
          .div(&z[i].sub(&z[i - j]))
      };
      table[i].push(v);
    }
  }
  let coeff = |k: usize| table[k][k].clone();
  // p = c[m-1]; p = p*(t - z_k) + c_k, kept as coefficients in (t - x).
  let mut p: Vec<F> = vec![coeff(m - 1)];
  for k in (0..m - 1).rev() {
    let shift = x.sub(&z[k]);
    let mut next = vec![F::int(0); p.len() + 1];
    for (j, c) in p.iter().enumerate() {
      next[j + 1] = next[j + 1].add(c);
      next[j] = next[j].add(&c.mul(&shift));
    }
    next[0] = next[0].add(&coeff(k));
    p = next;
  }
  Some(match p.get(derivative) {
    Some(c) => c.mul(&fact(derivative)),
    None => F::int(0),
  })
}

/// The value (or `derivative`-th derivative) at `x_expr`; `None` when `x`
/// is not a number.
pub fn evaluate(
  rows: &[Row],
  order: usize,
  derivative: usize,
  x_expr: &Expr,
) -> Result<Option<Expr>, InterpreterError> {
  let Some(x) = to_f64(x_expr) else {
    return Ok(None);
  };
  let xs: Vec<f64> = rows
    .iter()
    .map(|r| to_f64(&r.x).unwrap_or(f64::NAN))
    .collect();
  if x < xs[0] || x > xs[xs.len() - 1] {
    crate::functions::ode_ast::emit_dmval(std::slice::from_ref(x_expr));
  }
  let piece = &rows[stencil(rows, &xs, x, order)];
  let exact = is_exact_number(x_expr)
    && piece
      .iter()
      .all(|r| is_exact_number(&r.x) && r.values.iter().all(is_exact_number));
  if exact && let Some(xe) = Exact::from_expr(x_expr) {
    // A data point's own value (or stored derivative) comes back as given.
    if let Some(r) = piece
      .iter()
      .find(|r| Exact::from_expr(&r.x).is_some_and(|v| v.same(&xe)))
      && let Some(v) = r.values.get(derivative)
    {
      return Ok(Some(v.clone()));
    }
    if let Some(v) = piece_value(piece, &xe, derivative) {
      return Ok(Some(v.into_expr()));
    }
  }
  Ok(piece_value(piece, &x, derivative).map(Field::into_expr))
}

/// `"Domain"`, `"Grid"`, `"ValuesOnGrid"`, `"Coordinates"` and
/// `"InterpolationOrder"`.
pub fn property(rows: &[Row], order: usize, prop: &str) -> Option<Expr> {
  let xs = || rows.iter().map(|r| r.x.clone());
  Some(match prop {
    "Domain" => Expr::List(
      vec![Expr::List(
        vec![rows[0].x.clone(), rows[rows.len() - 1].x.clone()].into(),
      )]
      .into(),
    ),
    "Grid" => Expr::List(xs().map(|x| Expr::List(vec![x].into())).collect()),
    "Coordinates" => Expr::List(vec![Expr::List(xs().collect())].into()),
    "ValuesOnGrid" => {
      Expr::List(rows.iter().map(|r| r.values[0].clone()).collect())
    }
    "InterpolationOrder" => {
      Expr::List(vec![Expr::Integer(order as i128)].into())
    }
    _ => return None,
  })
}
