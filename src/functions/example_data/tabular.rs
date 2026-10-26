//! Generated tables: columns with a recorded type, range, spread, number of
//! distinct values and categories, and rank correlations between numeric
//! columns — used by `"Statistics"`, `"Dataset"` and `"MachineLearning"`.
//!
//! Numeric columns are drawn through a Gaussian copula: correlated normal
//! scores (from the recorded Spearman correlations) are mapped by rank onto
//! values drawn from each column's own distribution, so every column keeps
//! its range, mean and spread and the columns keep their correlations.
//! Categorical columns reproduce the recorded label frequencies exactly.

use super::rng::Rng;
use super::words::{Lexicon, prose};
use crate::syntax::Expr;

/// The recorded statistics of one column.
#[derive(Clone, Debug)]
pub enum Column {
  Numeric(NumSpec),
  /// Labels with their counts, most frequent first.
  Categorical(Vec<(Expr, usize)>),
  /// Free text (or other non-label values): mean and maximum length.
  Text {
    mean_len: f64,
  },
}

#[derive(Clone, Debug)]
pub struct NumSpec {
  pub min: f64,
  pub max: f64,
  pub mean: f64,
  pub sd: f64,
  pub distinct: usize,
  pub integer: bool,
  pub decimals: u32,
  /// How many of the recorded values were missing (non-numeric).
  pub missing: usize,
}

impl Default for NumSpec {
  fn default() -> Self {
    NumSpec {
      min: 0.0,
      max: 1.0,
      mean: 0.5,
      sd: 0.25,
      distinct: usize::MAX,
      integer: false,
      decimals: 3,
      missing: 0,
    }
  }
}

fn num(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(r) => Some(*r),
    _ => crate::functions::math_ast::try_eval_to_f64(e),
  }
}

/// Decode one entry of an `@Stats` list (see `colStats` in
/// `scripts/example_data_meta.wls`).
pub fn parse_column(e: &Expr) -> Option<Column> {
  let Expr::List(c) = e else { return None };
  let Some(Expr::String(kind)) = c.first() else {
    return None;
  };
  match kind.as_str() {
    "N" => {
      let Some(Expr::List(s)) = c.get(1) else {
        return Some(Column::Numeric(NumSpec::default()));
      };
      let get = |i: usize| s.get(i).and_then(num);
      let heads: Vec<String> = match s.get(6) {
        Some(Expr::List(h)) => h
          .iter()
          .filter_map(|x| match x {
            Expr::Identifier(n) => Some(n.clone()),
            _ => None,
          })
          .collect(),
        _ => Vec::new(),
      };
      Some(Column::Numeric(NumSpec {
        min: get(0).unwrap_or(0.0),
        max: get(1).unwrap_or(1.0),
        mean: get(2).unwrap_or(0.5),
        sd: get(3).unwrap_or(0.25),
        distinct: get(4).map_or(usize::MAX, |d| d as usize),
        integer: heads.iter().all(|h| h == "Integer") && !heads.is_empty(),
        decimals: get(7).map_or(3, |d| d as u32),
        missing: get(5).map_or(0, |m| m as usize),
      }))
    }
    "C" => {
      let Some(Expr::List(t)) = c.get(2) else {
        return None;
      };
      Some(Column::Categorical(
        t.iter()
          .filter_map(|p| match p {
            Expr::List(p) => match (p.first(), p.get(1)) {
              (Some(label), Some(Expr::Integer(n))) => {
                Some((label.clone(), *n as usize))
              }
              _ => None,
            },
            _ => None,
          })
          .collect(),
      ))
    }
    _ => {
      let mean_len = match c.get(4) {
        Some(Expr::List(l)) => l.first().and_then(num).unwrap_or(30.0),
        _ => 30.0,
      };
      Some(Column::Text { mean_len })
    }
  }
}

/// A bare `numStats` list `{min, max, mean, sd, distinct, missing, heads,
/// decimals}` as a numeric column.
pub fn parse_column_numeric(e: &Expr) -> Option<NumSpec> {
  match parse_column(&Expr::List(
    vec![Expr::String("N".into()), e.clone()].into(),
  ))? {
    Column::Numeric(s) => Some(s),
    _ => None,
  }
}

/// All columns of an `@Stats` list.
pub fn parse_columns(e: &Expr) -> Vec<Column> {
  match e {
    Expr::List(cols) => cols.iter().filter_map(parse_column).collect(),
    _ => Vec::new(),
  }
}

/// Decode `@Corr`: `{{numeric column positions (1-based)}, matrix}`.
pub fn parse_correlations(
  e: Option<&Expr>,
) -> Option<(Vec<usize>, Vec<Vec<f64>>)> {
  let Expr::List(c) = e? else { return None };
  let Expr::List(idx) = c.first()? else {
    return None;
  };
  let Expr::List(m) = c.get(1)? else {
    return None;
  };
  let idx: Vec<usize> = idx
    .iter()
    .filter_map(|x| num(x).map(|v| v as usize - 1))
    .collect();
  let m: Vec<Vec<f64>> = m
    .iter()
    .map(|row| match row {
      Expr::List(r) => r.iter().filter_map(num).collect(),
      _ => Vec::new(),
    })
    .collect();
  (m.len() == idx.len()).then_some((idx, m))
}

/// Round to `decimals` places, or to an integer.
pub fn quantize(v: f64, spec: &NumSpec) -> Expr {
  if spec.integer {
    Expr::Integer(v.round() as i128)
  } else {
    let f = 10f64.powi(spec.decimals.min(12) as i32);
    Expr::Real((v * f).round() / f)
  }
}

/// `n` values from the column's distribution, sorted ascending: a normal
/// with the recorded mean and spread clamped to the range (with both ends
/// attained), or a few evenly spaced levels when the column has only a
/// handful of distinct values.
pub fn sorted_draws(rng: &mut Rng, spec: &NumSpec, n: usize) -> Vec<f64> {
  let mut v: Vec<f64> = if spec.distinct <= 12 && spec.distinct >= 1 {
    let k = spec.distinct;
    let levels: Vec<f64> = (0..k)
      .map(|i| {
        if k == 1 {
          spec.min
        } else {
          spec.min + (spec.max - spec.min) * i as f64 / (k - 1) as f64
        }
      })
      .collect();
    // Weight levels by a normal density around the mean.
    let w: Vec<f64> = levels
      .iter()
      .map(|l| {
        (-(l - spec.mean).powi(2) / (2.0 * spec.sd.max(1e-9).powi(2))).exp()
          + 1e-3
      })
      .collect();
    let total: f64 = w.iter().sum();
    (0..n)
      .map(|_| {
        let mut u = rng.uniform() * total;
        for (l, wi) in levels.iter().zip(&w) {
          if u < *wi {
            return *l;
          }
          u -= wi;
        }
        *levels.last().unwrap_or(&spec.min)
      })
      .collect()
  } else {
    let below = spec.mean - spec.min;
    let above = spec.max - spec.mean;
    if spec.sd > 0.0 && below > 0.0 && below < 1.5 * spec.sd && above > below {
      // Right-skewed: a lognormal above the minimum with the recorded mean
      // and spread.
      let s2 = (1.0 + (spec.sd / below).powi(2)).ln();
      let mu = below.ln() - s2 / 2.0;
      (0..n)
        .map(|_| {
          (spec.min + (mu + s2.sqrt() * rng.normal()).exp()).min(spec.max)
        })
        .collect()
    } else if spec.sd > 0.0
      && above > 0.0
      && above < 1.5 * spec.sd
      && below > above
    {
      // Left-skewed: the mirror image.
      let s2 = (1.0 + (spec.sd / above).powi(2)).ln();
      let mu = above.ln() - s2 / 2.0;
      (0..n)
        .map(|_| {
          (spec.max - (mu + s2.sqrt() * rng.normal()).exp()).max(spec.min)
        })
        .collect()
    } else {
      (0..n)
        .map(|_| (spec.mean + spec.sd * rng.normal()).clamp(spec.min, spec.max))
        .collect()
    }
  };
  v.sort_by(f64::total_cmp);
  if n >= 2 && spec.distinct > 12 {
    v[0] = spec.min;
    v[n - 1] = spec.max;
  }
  v
}

/// Correlated standard normal scores for `k` variables (rows: samples),
/// from a correlation matrix via a (regularized) Cholesky factor.
fn correlated_normals(
  rng: &mut Rng,
  corr: &[Vec<f64>],
  n: usize,
) -> Vec<Vec<f64>> {
  let k = corr.len();
  // Spearman to Pearson for normals, and a diagonal nudge to stay positive
  // definite.
  let mut a = vec![vec![0.0; k]; k];
  for i in 0..k {
    for j in 0..k {
      a[i][j] = if i == j {
        1.0
      } else {
        0.98
          * 2.0
          * (std::f64::consts::PI / 6.0
            * corr[i].get(j).copied().unwrap_or(0.0))
          .sin()
      };
    }
  }
  let mut l = vec![vec![0.0; k]; k];
  for i in 0..k {
    for j in 0..=i {
      let s: f64 = (0..j).map(|p| l[i][p] * l[j][p]).sum();
      if i == j {
        l[i][j] = (a[i][i] - s).max(1e-6).sqrt();
      } else {
        l[i][j] = (a[i][j] - s) / l[j][j];
      }
    }
  }
  (0..n)
    .map(|_| {
      let z: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
      (0..k)
        .map(|i| (0..=i).map(|j| l[i][j] * z[j]).sum())
        .collect()
    })
    .collect()
}

/// Generate `n` rows for `columns`.
pub fn generate(
  rng: &mut Rng,
  columns: &[Column],
  n: usize,
  corr: Option<&(Vec<usize>, Vec<Vec<f64>>)>,
  lexicon: &Lexicon,
  temporal: bool,
) -> Vec<Vec<Expr>> {
  let m = columns.len();
  let mut table = vec![vec![Expr::Integer(0); m]; n];
  // Normal scores driving the numeric columns' ranks.
  let numeric: Vec<usize> = (0..m)
    .filter(|&j| matches!(columns[j], Column::Numeric(_)))
    .collect();
  let mut scores: Vec<Vec<f64>> = vec![Vec::new(); m];
  if let Some((idx, c)) = corr {
    let z = correlated_normals(rng, c, n);
    for (p, &j) in idx.iter().enumerate() {
      if j < m {
        scores[j] = z.iter().map(|row| row[p]).collect();
      }
    }
  }
  for &j in &numeric {
    if scores[j].is_empty() {
      scores[j] = (0..n).map(|_| rng.normal()).collect();
    }
  }
  // Rows of a time series are consecutive observations: smooth the scores
  // (an AR(1) process with a drift) so values wander instead of jumping.
  if temporal {
    let drift = rng.range(-0.02, 0.04);
    for &j in &numeric {
      let mut acc = 0.0;
      for (t, v) in scores[j].iter_mut().enumerate() {
        acc = 0.85 * acc + 0.5 * *v;
        *v = acc + drift * t as f64;
      }
    }
  }
  for (j, col) in columns.iter().enumerate() {
    match col {
      Column::Numeric(spec) => {
        let draws = sorted_draws(rng, spec, n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| scores[j][a].total_cmp(&scores[j][b]));
        for (rank, &row) in order.iter().enumerate() {
          table[row][j] = quantize(draws[rank], spec);
        }
        // As many missing values as the original had.
        let mut rows: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut rows);
        for &row in rows.iter().take(spec.missing.min(n)) {
          table[row][j] = Expr::FunctionCall {
            name: "Missing".into(),
            args: crate::ExprList::default(),
          };
        }
      }
      Column::Categorical(tally) => {
        let total: usize = tally.iter().map(|(_, c)| c).sum();
        let mut labels: Vec<Expr> = Vec::with_capacity(n);
        for (label, count) in tally {
          let k = if total == n {
            *count
          } else {
            (count * n).div_ceil(total.max(1))
          };
          labels.extend(std::iter::repeat_n(label.clone(), k));
        }
        while labels.len() < n {
          labels.push(
            tally
              .first()
              .map_or(Expr::String(String::new()), |t| t.0.clone()),
          );
        }
        labels.truncate(n);
        rng.shuffle(&mut labels);
        for (row, label) in labels.into_iter().enumerate() {
          table[row][j] = label;
        }
      }
      Column::Text { mean_len } => {
        for row in &mut table {
          let words = (mean_len / 6.0).round().max(1.0) as usize;
          row[j] = Expr::String(prose(lexicon, rng, words, words as f64));
        }
      }
    }
  }
  table
}
