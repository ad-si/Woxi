//! `Reduce[cond, vars, Integers]` when the constraints confine every
//! variable to a finite range: `x^2 + y^2 == 2`, `x + y == 3 && x > 0 &&
//! y > 0`, `0 <= x <= 10 && GCD[x, 6] == 2`.
//!
//! Bounds come from interval propagation through the polynomial
//! (in)equalities: each pure-power term `c v^k` must lie in the range left by
//! the other terms, which bounds `v`. When the resulting box is small enough
//! its integer points are enumerated; polynomial equations with integer
//! coefficients filter them exactly, and the survivors are checked against
//! the whole condition. The solutions are listed lexicographically, as
//! wolframscript does.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::{ComparisonOp, Expr};

/// Largest number of integer points the box may hold.
const MAX_POINTS: f64 = 200_000.0;

#[derive(Clone, Copy, PartialEq)]
struct Interval {
  lo: f64,
  hi: f64,
}

impl Interval {
  const ALL: Self = Self {
    lo: f64::NEG_INFINITY,
    hi: f64::INFINITY,
  };

  fn add(self, o: Self) -> Self {
    Self {
      lo: self.lo + o.lo,
      hi: self.hi + o.hi,
    }
  }

  fn neg(self) -> Self {
    Self {
      lo: -self.hi,
      hi: -self.lo,
    }
  }

  fn scale(self, c: f64) -> Self {
    let (a, b) = (mul(self.lo, c), mul(self.hi, c));
    Self {
      lo: a.min(b),
      hi: a.max(b),
    }
  }

  fn mul(self, o: Self) -> Self {
    let c = [
      mul(self.lo, o.lo),
      mul(self.lo, o.hi),
      mul(self.hi, o.lo),
      mul(self.hi, o.hi),
    ];
    Self {
      lo: c.iter().copied().fold(f64::INFINITY, f64::min),
      hi: c.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
  }

  fn pow(self, k: u32) -> Self {
    if k == 0 {
      return Self { lo: 1.0, hi: 1.0 };
    }
    let (a, b) = (self.lo.powi(k as i32), self.hi.powi(k as i32));
    if k % 2 == 1 {
      Self { lo: a, hi: b }
    } else if self.lo <= 0.0 && self.hi >= 0.0 {
      Self {
        lo: 0.0,
        hi: a.max(b),
      }
    } else {
      Self {
        lo: a.min(b),
        hi: a.max(b),
      }
    }
  }
}

/// `0 * ∞ = 0` for interval corners.
fn mul(a: f64, b: f64) -> f64 {
  if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
}

/// A polynomial constraint `Σ c_m x^m ∈ range`.
struct PolyConstraint {
  terms: Vec<(Vec<u32>, f64)>,
  /// Integer coefficients for exact evaluation, when all of them are.
  exact: Option<Vec<(Vec<u32>, i128)>>,
  range: Interval,
  is_equation: bool,
}

fn flatten_conjuncts(expr: &Expr, out: &mut Vec<Expr>) {
  match expr {
    Expr::FunctionCall { name, args } if name == "And" => {
      for a in args {
        flatten_conjuncts(a, out);
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() > 2 => {
      for i in 0..operators.len() {
        out.push(Expr::Comparison {
          operands: vec![operands[i].clone(), operands[i + 1].clone()],
          operators: vec![operators[i]],
        });
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Inequality" && args.len() >= 3 && args.len() % 2 == 1 =>
    {
      for i in (0..args.len() - 2).step_by(2) {
        let op = if let Expr::Identifier(s) = &args[i + 1] {
          match s.as_str() {
            "Less" => ComparisonOp::Less,
            "LessEqual" => ComparisonOp::LessEqual,
            "Greater" => ComparisonOp::Greater,
            "GreaterEqual" => ComparisonOp::GreaterEqual,
            "Equal" => ComparisonOp::Equal,
            _ => {
              out.push(expr.clone());
              return;
            }
          }
        } else {
          out.push(expr.clone());
          return;
        };
        out.push(Expr::Comparison {
          operands: vec![args[i].clone(), args[i + 2].clone()],
          operators: vec![op],
        });
      }
    }
    other => out.push(other.clone()),
  }
}

/// `lhs op rhs` as `Σ c_m x^m ∈ range` over `vars`, if it is polynomial.
fn poly_constraint(expr: &Expr, vars: &[String]) -> Option<PolyConstraint> {
  let Expr::Comparison {
    operands,
    operators,
  } = expr
  else {
    return None;
  };
  let ([lhs, rhs], [op]) = (operands.as_slice(), operators.as_slice()) else {
    return None;
  };
  let diff = evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Subtract".into(),
    args: vec![lhs.clone(), rhs.clone()].into(),
  })
  .ok()?;
  let var_list =
    Expr::List(vars.iter().cloned().map(Expr::Identifier).collect());
  let rules = evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "CoefficientRules".into(),
    args: vec![diff, var_list].into(),
  })
  .ok()?;
  let Expr::List(rules) = &rules else {
    return None;
  };
  let mut terms = Vec::new();
  let mut exact = Some(Vec::new());
  for r in rules {
    let Expr::Rule {
      pattern,
      replacement,
    } = r
    else {
      return None;
    };
    let Expr::List(exps) = pattern.as_ref() else {
      return None;
    };
    let exps: Vec<u32> = exps
      .iter()
      .map(|e| match e {
        Expr::Integer(k) if *k >= 0 && *k < 64 => Some(*k as u32),
        _ => None,
      })
      .collect::<Option<_>>()?;
    let c = try_eval_to_f64(replacement)?;
    if let Some(ex) = exact.as_mut() {
      match replacement.as_ref() {
        Expr::Integer(k) => ex.push((exps.clone(), *k)),
        _ => exact = None,
      }
    }
    terms.push((exps, c));
  }
  // `diff op 0`. Strict inequalities keep their endpoint here; the final
  // check against the whole condition drops it.
  let range = match op {
    ComparisonOp::Equal => Interval { lo: 0.0, hi: 0.0 },
    ComparisonOp::Less | ComparisonOp::LessEqual => Interval {
      lo: f64::NEG_INFINITY,
      hi: 0.0,
    },
    ComparisonOp::Greater | ComparisonOp::GreaterEqual => Interval {
      lo: 0.0,
      hi: f64::INFINITY,
    },
    _ => return None,
  };
  Some(PolyConstraint {
    terms,
    exact,
    range,
    is_equation: *op == ComparisonOp::Equal,
  })
}

fn term_interval(exps: &[u32], c: f64, boxes: &[Interval]) -> Interval {
  let mut iv = Interval { lo: c, hi: c };
  for (k, &e) in exps.iter().enumerate() {
    if e > 0 {
      iv = iv.mul(boxes[k].pow(e));
    }
  }
  iv
}

/// Tighten `boxes` with one constraint; returns whether anything changed.
fn propagate(con: &PolyConstraint, boxes: &mut [Interval]) -> bool {
  let ivs: Vec<Interval> = con
    .terms
    .iter()
    .map(|(e, c)| term_interval(e, *c, boxes))
    .collect();
  let mut changed = false;
  for (j, (exps, c)) in con.terms.iter().enumerate() {
    let nonzero: Vec<usize> =
      (0..exps.len()).filter(|&k| exps[k] > 0).collect();
    let [v] = nonzero.as_slice() else {
      continue;
    };
    let k = exps[*v];
    // c v^k ∈ range − Σ others.
    let others = ivs
      .iter()
      .enumerate()
      .filter(|&(i, _)| i != j)
      .fold(Interval { lo: 0.0, hi: 0.0 }, |acc, (_, iv)| acc.add(*iv));
    let allowed = con.range.add(others.neg()).scale(1.0 / c);
    let root = |x: f64| x.signum() * x.abs().powf(1.0 / k as f64);
    let bound = if k % 2 == 0 {
      if allowed.hi < 0.0 {
        Interval { lo: 1.0, hi: -1.0 }
      } else {
        let r = root(allowed.hi);
        Interval { lo: -r, hi: r }
      }
    } else {
      Interval {
        lo: root(allowed.lo),
        hi: root(allowed.hi),
      }
    };
    let b = &mut boxes[*v];
    let new = Interval {
      lo: b.lo.max(bound.lo),
      hi: b.hi.min(bound.hi),
    };
    // Integer points only: round inward (with slack for rounding error).
    let new = Interval {
      lo: (new.lo - 1e-9).ceil(),
      hi: (new.hi + 1e-9).floor(),
    };
    if new != *b {
      *b = new;
      changed = true;
    }
  }
  changed
}

/// Bounding box of a quadratic constraint whose quadratic part is definite:
/// `xᵀAx + bᵀx + k ∈ range` with `A` positive definite and a finite upper
/// end confines `x` to an ellipsoid around `x0 = -A⁻¹b/2`, whose extent
/// along axis i is `sqrt(R (A⁻¹)ᵢᵢ)` for `R = hi - min`. (A negative
/// definite form is handled by negating the constraint.)
fn quadratic_box(con: &PolyConstraint, n: usize) -> Option<Vec<Interval>> {
  if con.terms.iter().any(|(e, _)| e.iter().sum::<u32>() > 2) {
    return None;
  }
  let mut a = vec![vec![0.0; n]; n];
  let mut b = vec![0.0; n];
  let mut k = 0.0;
  for (e, c) in &con.terms {
    let nz: Vec<usize> = (0..n).filter(|&i| e[i] > 0).collect();
    match nz.as_slice() {
      [] => k += c,
      [i] if e[*i] == 2 => a[*i][*i] += c,
      [i] => b[*i] += c,
      [i, j] => {
        a[*i][*j] += c / 2.0;
        a[*j][*i] += c / 2.0;
      }
      _ => return None,
    }
  }
  let mut hi = con.range.hi;
  if a[0..n].iter().enumerate().all(|(i, row)| row[i] <= 0.0) {
    // Possibly negative definite: negate everything.
    for row in &mut a {
      for x in row.iter_mut() {
        *x = -*x;
      }
    }
    for x in &mut b {
      *x = -*x;
    }
    k = -k;
    hi = -con.range.lo;
  }
  if !hi.is_finite() {
    return None;
  }
  // Variables the constraint does not mention are left unbounded by it.
  let used: Vec<usize> = (0..n)
    .filter(|&i| a[i].iter().any(|&x| x != 0.0) || b[i] != 0.0)
    .collect();
  let m = used.len();
  if m == 0 {
    return None;
  }
  // Invert the used block by Gauss–Jordan; a singular or indefinite
  // matrix ends the attempt.
  let mut aug: Vec<Vec<f64>> = used
    .iter()
    .enumerate()
    .map(|(r, &i)| {
      let mut row: Vec<f64> = used.iter().map(|&j| a[i][j]).collect();
      row.extend((0..m).map(|c| if c == r { 1.0 } else { 0.0 }));
      row
    })
    .collect();
  for col in 0..m {
    let piv = (col..m).max_by(|&x, &y| {
      aug[x][col]
        .abs()
        .partial_cmp(&aug[y][col].abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    })?;
    if aug[piv][col].abs() < 1e-12 {
      return None;
    }
    aug.swap(piv, col);
    let p = aug[col][col];
    for x in &mut aug[col] {
      *x /= p;
    }
    for r in 0..m {
      if r != col {
        let f = aug[r][col];
        if f != 0.0 {
          for c in 0..2 * m {
            aug[r][c] -= f * aug[col][c];
          }
        }
      }
    }
  }
  let inv: Vec<Vec<f64>> = aug.iter().map(|row| row[m..].to_vec()).collect();
  // Positive definite ⇔ the inverse is too; check the diagonal and use it.
  if (0..m).any(|i| inv[i][i] <= 0.0) {
    return None;
  }
  let bu: Vec<f64> = used.iter().map(|&i| b[i]).collect();
  let x0: Vec<f64> = (0..m)
    .map(|i| -0.5 * (0..m).map(|j| inv[i][j] * bu[j]).sum::<f64>())
    .collect();
  let min = k + 0.5 * (0..m).map(|i| bu[i] * x0[i]).sum::<f64>();
  let r = hi - min;
  let mut boxes = vec![Interval::ALL; n];
  if r < 0.0 {
    return Some(vec![Interval { lo: 1.0, hi: -1.0 }; n]);
  }
  for (idx, &i) in used.iter().enumerate() {
    let ext = (r * inv[idx][idx]).sqrt();
    boxes[i] = Interval {
      lo: x0[idx] - ext,
      hi: x0[idx] + ext,
    };
  }
  Some(boxes)
}

fn eval_exact(terms: &[(Vec<u32>, i128)], point: &[i128]) -> Option<i128> {
  let mut sum: i128 = 0;
  for (exps, c) in terms {
    let mut t = *c;
    for (k, &e) in exps.iter().enumerate() {
      for _ in 0..e {
        t = t.checked_mul(point[k])?;
      }
    }
    sum = sum.checked_add(t)?;
  }
  Some(sum)
}

/// Linear factors `a·x + b` that each lie in `[-bound, bound]` and together
/// determine every variable bound each one: `x = A⁻¹ (L - b)`, so
/// `x^2 - y^2 == 15` confines `x` and `y` through `x - y` and `x + y`.
fn linear_factor_box(
  factors: &[Expr],
  bound: f64,
  vars: &[String],
  boxes: &mut [Interval],
) {
  let n = vars.len();
  let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
  for f in factors {
    let eq = Expr::Comparison {
      operands: vec![f.clone(), Expr::Integer(0)],
      operators: vec![ComparisonOp::Equal],
    };
    let Some(pc) = poly_constraint(&eq, vars) else {
      continue;
    };
    let mut a = vec![0.0; n];
    let mut b = 0.0;
    let mut linear = true;
    for (exps, c) in &pc.terms {
      match exps.iter().sum::<u32>() {
        0 => b += c,
        1 => a[exps.iter().position(|&e| e == 1).unwrap()] += c,
        _ => linear = false,
      }
    }
    if linear {
      rows.push((a, b));
    }
  }
  // n independent rows, picked greedily, then A⁻¹ by Gauss-Jordan.
  let mut chosen: Vec<(Vec<f64>, f64)> = Vec::new();
  for row in rows {
    let mut candidate = chosen.clone();
    candidate.push(row.clone());
    if rank(&candidate.iter().map(|r| r.0.clone()).collect::<Vec<_>>())
      == candidate.len()
    {
      chosen = candidate;
    }
    if chosen.len() == n {
      break;
    }
  }
  if chosen.len() < n {
    return;
  }
  let mut aug: Vec<Vec<f64>> = chosen
    .iter()
    .enumerate()
    .map(|(i, (a, _))| {
      let mut r = a.clone();
      r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
      r
    })
    .collect();
  for col in 0..n {
    let Some(p) =
      (col..n).max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
    else {
      return;
    };
    if aug[p][col] == 0.0 {
      return;
    }
    aug.swap(col, p);
    let pivot = aug[col][col];
    for v in &mut aug[col] {
      *v /= pivot;
    }
    for r in 0..n {
      if r != col {
        let factor = aug[r][col];
        if factor != 0.0 {
          for k in 0..2 * n {
            aug[r][k] -= factor * aug[col][k];
          }
        }
      }
    }
  }
  for (j, b) in boxes.iter_mut().enumerate() {
    let inv = &aug[j][n..];
    let center: f64 = -inv
      .iter()
      .zip(&chosen)
      .map(|(w, (_, c))| w * c)
      .sum::<f64>();
    let radius: f64 = bound * inv.iter().map(|w| w.abs()).sum::<f64>();
    b.lo = b.lo.max((center - radius - 1e-9).ceil());
    b.hi = b.hi.min((center + radius + 1e-9).floor());
  }
}

/// The rank of a small matrix, by Gaussian elimination.
fn rank(rows: &[Vec<f64>]) -> usize {
  let mut m: Vec<Vec<f64>> = rows.to_vec();
  let cols = m.first().map_or(0, Vec::len);
  let mut r = 0;
  for c in 0..cols {
    let Some(p) = (r..m.len()).find(|&i| m[i][c].abs() > 1e-12) else {
      continue;
    };
    m.swap(r, p);
    for i in r + 1..m.len() {
      let f = m[i][c] / m[r][c];
      for k in c..cols {
        m[i][k] -= f * m[r][k];
      }
    }
    r += 1;
    if r == m.len() {
      break;
    }
  }
  r
}

/// `Reduce[cond, vars, Integers]` by enumeration when `cond` bounds every
/// variable to a small box; `None` when it does not.
pub(super) fn reduce_bounded_integer_box(
  cond: &Expr,
  vars: &[String],
) -> Option<Result<Expr, InterpreterError>> {
  if vars.is_empty() {
    return None;
  }
  let mut conjuncts = Vec::new();
  flatten_conjuncts(cond, &mut conjuncts);
  let mut polys = Vec::new();
  for c in &conjuncts {
    if let Some(p) = poly_constraint(c, vars) {
      polys.push(p);
    }
  }
  if polys.is_empty() {
    return None;
  }
  let mut boxes = vec![Interval::ALL; vars.len()];
  for p in &polys {
    if let Some(q) = quadratic_box(p, vars.len()) {
      for (b, qb) in boxes.iter_mut().zip(q) {
        b.lo = b.lo.max((qb.lo - 1e-9).ceil());
        b.hi = b.hi.min((qb.hi + 1e-9).floor());
      }
    }
  }
  // A product of the variables equal to a nonzero integer, `x y == 6`,
  // bounds every factor by that integer's size: a divisor of it.
  for p in polys.iter().filter(|p| p.is_equation) {
    let Some(terms) = &p.exact else { continue };
    let (constant, monomials): (Vec<_>, Vec<_>) =
      terms.iter().partition(|(e, _)| e.iter().all(|&k| k == 0));
    let c: i128 = constant.iter().map(|(_, c)| *c).sum();
    if let [(exps, coeff)] = monomials.as_slice()
      && c != 0
      && *coeff != 0
      && c % coeff == 0
    {
      let bound = (c / coeff).unsigned_abs() as f64;
      for (b, &e) in boxes.iter_mut().zip(exps.iter()) {
        if e > 0 {
          b.lo = b.lo.max(-bound);
          b.hi = b.hi.min(bound);
        }
      }
    }
  }
  // More generally every factor of a polynomial equal to a nonzero
  // integer, `r (m - n) (m + n) == 60`, is a divisor of it and lies within
  // its size.
  let mut factor_bounds = Vec::new();
  for p in polys.iter().filter(|p| p.is_equation) {
    let Some(terms) = &p.exact else { continue };
    let (constant, monomials): (Vec<_>, Vec<_>) =
      terms.iter().partition(|(e, _)| e.iter().all(|&k| k == 0));
    let c: i128 = constant.iter().map(|(_, c)| *c).sum();
    if c == 0 || monomials.len() < 2 {
      continue;
    }
    let poly = Expr::FunctionCall {
      name: "Plus".into(),
      args: monomials
        .iter()
        .map(|(exps, coeff)| {
          let mut factors = vec![Expr::Integer(*coeff)];
          for (v, &e) in vars.iter().zip(exps.iter()) {
            if e > 0 {
              factors.push(Expr::FunctionCall {
                name: "Power".into(),
                args: vec![
                  Expr::Identifier(v.clone()),
                  Expr::Integer(e as i128),
                ]
                .into(),
              });
            }
          }
          Expr::FunctionCall {
            name: "Times".into(),
            args: factors.into(),
          }
        })
        .collect(),
    };
    let Ok(Expr::List(ref factor_list)) =
      evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "FactorList".into(),
        args: vec![poly].into(),
      })
    else {
      continue;
    };
    let mut content: i128 = 1;
    let mut polynomial_factors = Vec::new();
    for pair in factor_list {
      match pair {
        Expr::List(fe) => match fe.as_slice() {
          [Expr::Integer(k), Expr::Integer(e)] if *e >= 0 => {
            match k.checked_pow(*e as u32) {
              Some(v) => content = content.saturating_mul(v),
              None => content = 0,
            }
          }
          [f, Expr::Integer(e)] if *e > 0 => polynomial_factors.push(f.clone()),
          _ => {
            content = 0;
          }
        },
        _ => content = 0,
      }
    }
    if polynomial_factors.len() < 2 || content == 0 || c % content != 0 {
      continue;
    }
    let bound = (c / content).unsigned_abs() as i128;
    linear_factor_box(&polynomial_factors, bound as f64, vars, &mut boxes);
    for f in polynomial_factors {
      for (op, side) in [
        (ComparisonOp::GreaterEqual, -bound),
        (ComparisonOp::LessEqual, bound),
      ] {
        let constraint = Expr::Comparison {
          operands: vec![f.clone(), Expr::Integer(side)],
          operators: vec![op],
        };
        if let Some(pc) = poly_constraint(&constraint, vars) {
          factor_bounds.push(pc);
        }
      }
    }
  }
  polys.extend(factor_bounds);
  for _ in 0..50 {
    let mut changed = false;
    for p in &polys {
      changed |= propagate(p, &mut boxes);
    }
    if boxes.iter().any(|b| b.lo > b.hi) {
      return Some(Ok(Expr::Identifier("False".into())));
    }
    if !changed {
      break;
    }
  }
  for b in &boxes {
    if !(b.lo.is_finite() && b.hi.is_finite()) {
      return None;
    }
  }
  let exact_eqs: Vec<&Vec<(Vec<u32>, i128)>> = polys
    .iter()
    .filter(|p| p.is_equation)
    .filter_map(|p| p.exact.as_ref())
    .collect();
  // An exact equation of degree one or two in the last variable gives that
  // variable's values outright once the others are fixed, so only the
  // others are enumerated (`x^2 + y^2 + z^2 == 29^2` walks 59^2 points,
  // not 59^3).
  let last = vars.len() - 1;
  let solver = exact_eqs.iter().copied().find(|t| {
    let deg = t.iter().map(|(e, _)| e[last]).max().unwrap_or(0);
    (1..=2).contains(&deg)
  });
  let mut count = 1.0;
  for (k, b) in boxes.iter().enumerate() {
    if k == last && solver.is_some() {
      continue;
    }
    count *= b.hi - b.lo + 1.0;
  }
  if count > MAX_POINTS {
    return None;
  }

  let lo: Vec<i128> = boxes.iter().map(|b| b.lo as i128).collect();
  let hi: Vec<i128> = boxes.iter().map(|b| b.hi as i128).collect();
  let mut point = lo.clone();
  let mut solutions = Vec::new();
  loop {
    // The last variable's candidate values at this point of the others.
    let candidates: Vec<i128> = match solver {
      Some(terms) => {
        let mut coeffs = [0i128; 3];
        for (exps, c) in terms {
          let mut t = *c;
          for (k, &e) in exps.iter().enumerate() {
            if k == last {
              continue;
            }
            for _ in 0..e {
              t = t.checked_mul(point[k])?;
            }
          }
          let slot = &mut coeffs[exps[last] as usize];
          *slot = slot.checked_add(t)?;
        }
        integer_roots(coeffs[2], coeffs[1], coeffs[0], lo[last], hi[last])?
      }
      None => (lo[last]..=hi[last]).collect(),
    };
    for x in candidates {
      point[last] = x;
      if !exact_eqs
        .iter()
        .all(|t| eval_exact(t, &point).is_none_or(|v| v == 0))
      {
        continue;
      }
      let mut e = cond.clone();
      for (v, x) in vars.iter().zip(&point) {
        e = crate::functions::plot::substitute_var(&e, v, &Expr::Integer(*x));
      }
      match evaluate_expr_to_expr(&e) {
        Ok(Expr::Identifier(ref s)) if s == "True" => {
          solutions.push(point.clone());
        }
        Ok(Expr::Identifier(ref s)) if s == "False" => {}
        // A condition that does not decide at an integer point is beyond
        // this enumeration.
        Ok(_) => return None,
        Err(e) => return Some(Err(e)),
      }
    }
    // Next point of the other variables, the last of them fastest
    // (lexicographic order).
    let mut k = last;
    loop {
      if k == 0 {
        return Some(Ok(solutions_expr(vars, &solutions)));
      }
      k -= 1;
      if point[k] < hi[k] {
        point[k] += 1;
        point[k + 1..].copy_from_slice(&lo[k + 1..]);
        break;
      }
    }
  }
}

/// The integer roots of `a x^2 + b x + c` in `[lo, hi]`, ascending; every
/// integer there when the polynomial vanishes identically. None on
/// overflow.
fn integer_roots(
  a: i128,
  b: i128,
  c: i128,
  lo: i128,
  hi: i128,
) -> Option<Vec<i128>> {
  let in_range = |x: i128| (lo..=hi).contains(&x);
  if a == 0 {
    if b == 0 {
      return Some(if c == 0 { (lo..=hi).collect() } else { vec![] });
    }
    return Some(if c % b == 0 && in_range(-c / b) {
      vec![-c / b]
    } else {
      vec![]
    });
  }
  let disc = b
    .checked_mul(b)?
    .checked_sub(a.checked_mul(c)?.checked_mul(4)?)?;
  if disc < 0 {
    return Some(vec![]);
  }
  let mut s = (disc as f64).sqrt() as i128;
  while s * s > disc {
    s -= 1;
  }
  while (s + 1) * (s + 1) <= disc {
    s += 1;
  }
  if s * s != disc {
    return Some(vec![]);
  }
  let mut roots: Vec<i128> = [-b - s, -b + s]
    .into_iter()
    .filter(|n| n % (2 * a) == 0)
    .map(|n| n / (2 * a))
    .filter(|x| in_range(*x))
    .collect();
  roots.sort_unstable();
  roots.dedup();
  Some(roots)
}

fn solutions_expr(vars: &[String], solutions: &[Vec<i128>]) -> Expr {
  let conj = |p: &Vec<i128>| {
    let eqs: Vec<Expr> = vars
      .iter()
      .zip(p)
      .map(|(v, x)| Expr::Comparison {
        operands: vec![Expr::Identifier(v.clone()), Expr::Integer(*x)],
        operators: vec![ComparisonOp::Equal],
      })
      .collect();
    if eqs.len() == 1 {
      eqs.into_iter().next().unwrap()
    } else {
      Expr::FunctionCall {
        name: "And".into(),
        args: eqs.into(),
      }
    }
  };
  match solutions {
    [] => Expr::Identifier("False".into()),
    [one] => conj(one),
    many => Expr::FunctionCall {
      name: "Or".into(),
      args: many.iter().map(conj).collect(),
    },
  }
}
