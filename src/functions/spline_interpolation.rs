//! `Interpolation[data, Method -> "Spline"]`: an interpolating B-spline of
//! degree `InterpolationOrder` (3 by default), stored as
//! `InterpolatingFunction[{{xmin, xmax}}, {"Spline", xs, ys, knots, controls},
//! degree]`.
//!
//! As in wolframscript, the knots are clamped (the ends repeated `degree + 1`
//! times) and the interior knots are averages of `degree` consecutive data
//! sites (de Boor's choice); the control points make the spline pass through
//! every data point. A spline that wolframscript stored in a notebook (a
//! `BSplineFunction` inside the kernel layout) reads into the same form.

use crate::syntax::Expr;

pub struct Spline {
  pub xs: Vec<f64>,
  pub ys: Vec<f64>,
  pub knots: Vec<f64>,
  pub controls: Vec<f64>,
  pub degree: usize,
}

/// The clamped, averaged knot vector for sites `xs`.
fn averaged_knots(xs: &[f64], degree: usize) -> Vec<f64> {
  let n = xs.len();
  let mut knots = vec![xs[0]; degree + 1];
  for j in 1..n - degree {
    knots.push(xs[j..j + degree].iter().sum::<f64>() / degree as f64);
  }
  knots.extend(std::iter::repeat_n(xs[n - 1], degree + 1));
  knots
}

/// The knot span `[t_s, t_{s+1})` holding `x`, clamped to the valid spans so
/// a point outside the data extrapolates the end piece.
fn span(knots: &[f64], degree: usize, n_controls: usize, x: f64) -> usize {
  let mut s = degree;
  while s + 1 < n_controls && x >= knots[s + 1] {
    s += 1;
  }
  s
}

/// The `degree + 1` non-zero B-spline basis values at `x` on span `s`
/// (Cox–de Boor), for `N_{s-degree} … N_s`.
fn basis(knots: &[f64], degree: usize, s: usize, x: f64) -> Vec<f64> {
  let mut b = vec![0.0; degree + 1];
  b[0] = 1.0;
  let mut left = vec![0.0; degree + 1];
  let mut right = vec![0.0; degree + 1];
  for j in 1..=degree {
    left[j] = x - knots[s + 1 - j];
    right[j] = knots[s + j] - x;
    let mut saved = 0.0;
    for r in 0..j {
      let denom = right[r + 1] + left[j - r];
      let t = if denom == 0.0 { 0.0 } else { b[r] / denom };
      b[r] = saved + right[r + 1] * t;
      saved = left[j - r] * t;
    }
    b[j] = saved;
  }
  b
}

impl Spline {
  /// The interpolating spline through `(xs, ys)` (sorted by `xs`), with the
  /// degree lowered to `xs.len() - 1` when there are too few points.
  pub fn interpolate(
    xs: Vec<f64>,
    ys: Vec<f64>,
    degree: usize,
  ) -> Option<Spline> {
    let n = xs.len();
    if n < 2 || ys.len() != n {
      return None;
    }
    let degree = degree.clamp(1, n - 1);
    let knots = averaged_knots(&xs, degree);
    // Collocation matrix rows: row i is non-zero in columns s-degree..=s.
    // B-spline collocation matrices are totally positive, so elimination
    // without pivoting is stable; only the band is stored.
    let width = 2 * degree + 1;
    let mut band = vec![vec![0.0; width]; n];
    let mut rhs = ys.clone();
    for (i, &x) in xs.iter().enumerate() {
      let s = span(&knots, degree, n, x);
      for (k, v) in basis(&knots, degree, s, x).into_iter().enumerate() {
        let col = s - degree + k;
        // Column `col` of row `i` lives at band offset col + degree - i.
        let off = (col + degree).checked_sub(i)?;
        if off < width {
          band[i][off] = v;
        }
      }
    }
    let at = |band: &Vec<Vec<f64>>, i: usize, j: usize| -> f64 {
      (j + degree)
        .checked_sub(i)
        .filter(|&o| o < width)
        .map_or(0.0, |o| band[i][o])
    };
    for p in 0..n {
      let pivot = at(&band, p, p);
      if pivot == 0.0 {
        return None;
      }
      for r in p + 1..(p + degree + 1).min(n) {
        let factor = at(&band, r, p) / pivot;
        if factor == 0.0 {
          continue;
        }
        for c in p..(p + degree + 1).min(n) {
          let v = at(&band, p, c);
          if let Some(o) = (c + degree).checked_sub(r).filter(|&o| o < width) {
            band[r][o] -= factor * v;
          }
        }
        rhs[r] -= factor * rhs[p];
      }
    }
    let mut controls = vec![0.0; n];
    for p in (0..n).rev() {
      let mut v = rhs[p];
      for c in p + 1..(p + degree + 1).min(n) {
        v -= at(&band, p, c) * controls[c];
      }
      controls[p] = v / at(&band, p, p);
    }
    Some(Spline {
      xs,
      ys,
      knots,
      controls,
      degree,
    })
  }

  /// The `d`-th derivative at `x` (de Boor, on the derivative spline).
  pub fn eval(&self, x: f64, d: usize) -> f64 {
    let mut knots = self.knots.clone();
    let mut ctrl = self.controls.clone();
    let mut p = self.degree;
    for _ in 0..d {
      if p == 0 {
        return 0.0;
      }
      let next: Vec<f64> = (0..ctrl.len() - 1)
        .map(|i| {
          let dt = knots[i + p + 1] - knots[i + 1];
          if dt == 0.0 {
            0.0
          } else {
            p as f64 * (ctrl[i + 1] - ctrl[i]) / dt
          }
        })
        .collect();
      knots = knots[1..knots.len() - 1].to_vec();
      ctrl = next;
      p -= 1;
    }
    if ctrl.is_empty() {
      return 0.0;
    }
    let s = span(&knots, p, ctrl.len(), x);
    basis(&knots, p, s, x)
      .iter()
      .enumerate()
      .map(|(k, b)| b * ctrl[s - p + k])
      .sum()
  }

  pub fn to_data(&self) -> Expr {
    let list =
      |v: &[f64]| Expr::List(v.iter().map(|&x| Expr::Real(x)).collect());
    Expr::List(
      vec![
        Expr::String("Spline".into()),
        list(&self.xs),
        list(&self.ys),
        list(&self.knots),
        list(&self.controls),
      ]
      .into(),
    )
  }

  /// The stored form's data argument back as a spline.
  pub fn from_data(data: &Expr, degree: usize) -> Option<Spline> {
    let Expr::List(parts) = data else {
      return None;
    };
    let [Expr::String(tag), xs, ys, knots, controls] = parts.as_slice() else {
      return None;
    };
    if tag != "Spline" {
      return None;
    }
    let reals = |e: &Expr| -> Option<Vec<f64>> {
      match e {
        Expr::List(items) => items
          .iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect(),
        _ => None,
      }
    };
    let spline = Spline {
      xs: reals(xs)?,
      ys: reals(ys)?,
      knots: reals(knots)?,
      controls: reals(controls)?,
      degree,
    };
    (spline.knots.len() == spline.controls.len() + degree + 1
      && !spline.xs.is_empty())
    .then_some(spline)
  }

  /// A kernel-layout spline: `{BSplineFunction[1, {{a, b}}, {degree},
  /// {False}, {controls, {}}, {knots}, …], {}}` over the grid `xs`.
  pub fn from_kernel(values: &Expr, xs: &[Expr]) -> Option<Spline> {
    let Expr::List(items) = values else {
      return None;
    };
    let Some(Expr::FunctionCall { name, args }) = items.first() else {
      return None;
    };
    if name != "BSplineFunction" || args.len() < 6 {
      return None;
    }
    let first = |e: &Expr| -> Option<Expr> {
      match e {
        Expr::List(v) => v.first().cloned(),
        _ => None,
      }
    };
    let degree = match first(&args[2])? {
      Expr::Integer(k) if k >= 1 => k as usize,
      _ => return None,
    };
    let reals = |e: &Expr| -> Option<Vec<f64>> {
      match e {
        Expr::List(v) => v
          .iter()
          .map(crate::functions::math_ast::try_eval_to_f64)
          .collect(),
        _ => None,
      }
    };
    let controls = reals(&first(&args[4])?)?;
    let knots = reals(&first(&args[5])?)?;
    let xs: Vec<f64> = xs
      .iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect::<Option<_>>()?;
    if knots.len() != controls.len() + degree + 1 || xs.is_empty() {
      return None;
    }
    let mut spline = Spline {
      ys: Vec::new(),
      xs,
      knots,
      controls,
      degree,
    };
    spline.ys = spline.xs.iter().map(|&x| spline.eval(x, 0)).collect();
    Some(spline)
  }

  pub fn property(&self, prop: &str) -> Option<Expr> {
    let xs = || self.xs.iter().map(|&x| Expr::Real(x));
    let (a, b) = (self.xs[0], self.xs[self.xs.len() - 1]);
    Some(match prop {
      "Domain" => Expr::List(
        vec![Expr::List(vec![Expr::Real(a), Expr::Real(b)].into())].into(),
      ),
      "Grid" => Expr::List(xs().map(|x| Expr::List(vec![x].into())).collect()),
      "Coordinates" => Expr::List(vec![Expr::List(xs().collect())].into()),
      "ValuesOnGrid" => {
        Expr::List(self.ys.iter().map(|&y| Expr::Real(y)).collect())
      }
      "InterpolationOrder" => {
        Expr::List(vec![Expr::Integer(self.degree as i128)].into())
      }
      _ => return None,
    })
  }
}
