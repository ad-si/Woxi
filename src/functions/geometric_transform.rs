//! FindGeometricTransform[pts1, pts2] — the geometric transformation that
//! maps the points `pts2` onto `pts1` as closely as possible, with the mean
//! distance between the mapped and the target points: `{error,
//! TransformationFunction[m]}`.
//!
//! `TransformationClass` picks the family: "Translation", "Rigid" and
//! "Similarity" (orthogonal Procrustes), "Affine" (least squares) or
//! "Perspective" (the linear least-squares homography with `m[[3, 3]] = 1`).
//! `Automatic` takes the most general family the points determine: a
//! perspective map from four or more plane points, an affine one from
//! `dim + 1`, a similarity from two and a translation from one.

use crate::InterpreterError;
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn points(e: &Expr) -> Option<Vec<Vec<f64>>> {
  let Expr::List(rows) = e else { return None };
  rows
    .iter()
    .map(|r| match r {
      Expr::List(cs) => cs
        .iter()
        .map(crate::functions::math_ast::try_eval_to_f64)
        .collect(),
      _ => None,
    })
    .collect()
}

/// Solve the square system `a x = b` by Gaussian elimination with partial
/// pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
  let n = b.len();
  for col in 0..n {
    let pivot =
      (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
    if a[pivot][col].abs() < 1e-300 {
      return None;
    }
    a.swap(col, pivot);
    b.swap(col, pivot);
    for row in col + 1..n {
      let f = a[row][col] / a[col][col];
      for k in col..n {
        a[row][k] -= f * a[col][k];
      }
      b[row] -= f * b[col];
    }
  }
  let mut x = vec![0.0; n];
  for row in (0..n).rev() {
    let s: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
    x[row] = (b[row] - s) / a[row][row];
  }
  Some(x)
}

/// The least-squares solution of the rows `design · x ≈ rhs`.
fn least_squares(design: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
  let m = design.first()?.len();
  let mut ata = vec![vec![0.0; m]; m];
  let mut atb = vec![0.0; m];
  for (row, y) in design.iter().zip(rhs) {
    for i in 0..m {
      atb[i] += row[i] * y;
      for j in 0..m {
        ata[i][j] += row[i] * row[j];
      }
    }
  }
  solve(ata, atb)
}

/// The homogeneous matrix of the best map of `from` onto `to` in `class`.
fn fit(
  class: &str,
  from: &[Vec<f64>],
  to: &[Vec<f64>],
) -> Option<Vec<Vec<f64>>> {
  let d = from[0].len();
  let n = from.len() as f64;
  let mut m = vec![vec![0.0; d + 1]; d + 1];
  m[d][d] = 1.0;
  match class {
    "Translation" => {
      for k in 0..d {
        m[k][k] = 1.0;
        m[k][d] =
          to.iter().zip(from).map(|(t, f)| t[k] - f[k]).sum::<f64>() / n;
      }
    }
    "Affine" => {
      let design: Vec<Vec<f64>> = from
        .iter()
        .map(|p| p.iter().copied().chain(std::iter::once(1.0)).collect())
        .collect();
      for k in 0..d {
        let rhs: Vec<f64> = to.iter().map(|t| t[k]).collect();
        m[k] = least_squares(&design, &rhs)?;
      }
    }
    "Perspective" if d == 2 => {
      let mut design = Vec::new();
      let mut rhs = Vec::new();
      for (p, t) in from.iter().zip(to) {
        let (x, y, u, v) = (p[0], p[1], t[0], t[1]);
        design.push(vec![x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
        rhs.push(u);
        design.push(vec![0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
        rhs.push(v);
      }
      let h = least_squares(&design, &rhs)?;
      m = vec![h[0..3].to_vec(), h[3..6].to_vec(), vec![h[6], h[7], 1.0]];
    }
    "Rigid" | "Similarity" => {
      let mean = |ps: &[Vec<f64>]| -> Vec<f64> {
        (0..d)
          .map(|k| ps.iter().map(|p| p[k]).sum::<f64>() / n)
          .collect()
      };
      let (mf, mt) = (mean(from), mean(to));
      let centred = |ps: &[Vec<f64>], c: &[f64]| -> Vec<Vec<f64>> {
        ps.iter()
          .map(|p| p.iter().zip(c).map(|(a, b)| a - b).collect())
          .collect()
      };
      let (p, q) = (centred(from, &mf), centred(to, &mt));
      let spread: f64 = p.iter().flatten().map(|v| v * v).sum();
      let (rotation, trace) = if d == 2 {
        let (mut dot, mut cross) = (0.0, 0.0);
        for (a, b) in p.iter().zip(&q) {
          dot += a[0] * b[0] + a[1] * b[1];
          cross += a[0] * b[1] - a[1] * b[0];
        }
        let theta = cross.atan2(dot);
        let (s, c) = theta.sin_cos();
        (vec![vec![c, -s], vec![s, c]], dot.hypot(cross))
      } else {
        // Kabsch: H = P^T Q = U S V^T, R = V diag(1, …, det) U^T.
        let h: Vec<Vec<f64>> = (0..d)
          .map(|i| {
            (0..d)
              .map(|j| p.iter().zip(&q).map(|(a, b)| a[i] * b[j]).sum())
              .collect()
          })
          .collect();
        let to_expr = |mm: &[Vec<f64>]| {
          Expr::List(
            mm.iter()
              .map(|r| Expr::List(r.iter().map(|v| Expr::Real(*v)).collect()))
              .collect(),
          )
        };
        let svd = crate::evaluator::evaluate_expr_to_expr(&call(
          "SingularValueDecomposition",
          vec![to_expr(&h)],
        ))
        .ok()?;
        let Expr::List(parts) = &svd else { return None };
        let (u, s, v) =
          (points(&parts[0])?, points(&parts[1])?, points(&parts[2])?);
        let mut r = vec![vec![0.0; d]; d];
        let product = |sign: f64, r: &mut Vec<Vec<f64>>| {
          for i in 0..d {
            for j in 0..d {
              r[i][j] = (0..d)
                .map(|k| {
                  v[i][k] * u[j][k] * if k == d - 1 { sign } else { 1.0 }
                })
                .sum();
            }
          }
        };
        product(1.0, &mut r);
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
          - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
          + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        let sign = det.signum();
        product(sign, &mut r);
        let trace = (0..d)
          .map(|k| s[k][k] * if k == d - 1 { sign } else { 1.0 })
          .sum();
        (r, trace)
      };
      let scale = if class == "Similarity" && spread > 0.0 {
        trace / spread
      } else {
        1.0
      };
      for i in 0..d {
        for j in 0..d {
          m[i][j] = scale * rotation[i][j];
        }
        m[i][d] = mt[i] - (0..d).map(|j| m[i][j] * mf[j]).sum::<f64>();
      }
    }
    _ => return None,
  }
  Some(m)
}

fn apply(m: &[Vec<f64>], p: &[f64]) -> Vec<f64> {
  let d = p.len();
  let h: Vec<f64> = (0..=d)
    .map(|i| (0..d).map(|j| m[i][j] * p[j]).sum::<f64>() + m[i][d])
    .collect();
  (0..d).map(|i| h[i] / h[d]).collect()
}

pub fn find_geometric_transform_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let held = || Ok(crate::helpers::unevaluated("FindGeometricTransform", args));
  if args.len() < 2 {
    return held();
  }
  let (Some(to), Some(from)) = (points(&args[0]), points(&args[1])) else {
    return held();
  };
  let n = to.len();
  if n == 0 || n != from.len() {
    return held();
  }
  let d = to[0].len();
  if !(2..=3).contains(&d) || to.iter().chain(&from).any(|p| p.len() != d) {
    return held();
  }
  let mut class = "Automatic".to_string();
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(p) if p == "TransformationClass")
    {
      class = match replacement.as_ref() {
        Expr::String(s) | Expr::Identifier(s) => s.clone(),
        _ => return held(),
      };
    }
  }
  if class == "Automatic" {
    class = if d == 2 && n >= 4 {
      "Perspective"
    } else if n > d {
      "Affine"
    } else if n >= 2 {
      "Similarity"
    } else {
      "Translation"
    }
    .to_string();
  }
  let Some(m) = fit(&class, &from, &to) else {
    return held();
  };
  let error = from
    .iter()
    .zip(&to)
    .map(|(f, t)| {
      apply(&m, f)
        .iter()
        .zip(t)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
    })
    .sum::<f64>()
    / n as f64;
  let matrix = Expr::List(
    m.iter()
      .map(|r| Expr::List(r.iter().map(|v| Expr::Real(*v)).collect()))
      .collect(),
  );
  Ok(Expr::List(
    vec![
      Expr::Real(error),
      call("TransformationFunction", vec![matrix]),
    ]
    .into(),
  ))
}
