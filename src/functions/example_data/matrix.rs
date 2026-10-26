//! `ExampleData[{"Matrix", …}]`: generated sparse matrices.
//!
//! Each matrix is a `SparseArray` with the recorded dimensions, number of
//! stored entries, element type, structure (general, symmetric,
//! skew-symmetric, Hermitian) and lower/upper bandwidth; entries cluster
//! near the diagonal like those of the finite-element and circuit matrices
//! the collection comes from. Matrices recorded as positive definite are
//! made diagonally dominant, so they really are. The other properties are
//! the recorded metadata.
//!
//! Matrices with more than [`MAX_ENTRIES`] stored entries are not generated
//! (they stay unevaluated): Woxi holds sparse arrays as ordinary
//! expressions, and tens of millions of entries do not fit.

use std::collections::HashSet;

use super::rng::Rng;
use super::{Collection, Entry, GENERATED_SOURCE};
use crate::functions::*;

/// The largest matrix generated, in stored entries.
pub const MAX_ENTRIES: usize = 4_000_000;

pub struct Matrices;

fn int(e: Option<Expr>) -> Option<usize> {
  match e? {
    Expr::Integer(n) if n >= 0 => Some(n as usize),
    _ => None,
  }
}

fn string(e: Option<&Expr>) -> String {
  match e {
    Some(Expr::String(s)) => s.clone(),
    _ => String::new(),
  }
}

#[derive(Clone, Copy, PartialEq)]
enum Structure {
  General,
  Symmetric,
  SkewSymmetric,
  Hermitian,
}

/// Distinct positions `(row, col)`, 0-based, sorted row-major.
fn pattern(
  rng: &mut Rng,
  rows: usize,
  cols: usize,
  nnz: usize,
  lower: usize,
  upper: usize,
  structure: Structure,
) -> Vec<(usize, usize)> {
  let mirrored = structure != Structure::General && rows == cols;
  let mut set: HashSet<(usize, usize)> = HashSet::with_capacity(nnz);
  let n = rows.min(cols);
  // Mirrored matrices: a diagonal (none for skew-symmetric) plus pairs.
  let (diag, off) = if mirrored {
    let d = if structure == Structure::SkewSymmetric {
      0
    } else {
      n.min(nnz)
    };
    let d = if (nnz - d) % 2 == 1 {
      d.saturating_sub(1)
    } else {
      d
    };
    (d, (nnz - d) / 2)
  } else {
    (n.min(nnz / 2), 0)
  };
  for i in 0..diag {
    set.insert((i, i));
  }
  let band_lo = lower
    .max(if mirrored { upper } else { 0 })
    .max(1)
    .min(rows.saturating_sub(1).max(1));
  let band_hi = if mirrored {
    0
  } else {
    upper.min(cols.saturating_sub(1))
  };
  let target = if mirrored { off } else { nnz };
  let capacity = |lo: usize, hi: usize| -> usize {
    // Positions within the band, roughly.
    (lo + hi + 1).saturating_mul(rows.max(cols)).max(1)
  };
  let mut added = if mirrored { 0 } else { set.len() };
  // Make the band reach its recorded width once on each side.
  let place = |set: &mut HashSet<(usize, usize)>, i: usize, j: usize| -> bool {
    if i < rows && j < cols && (!mirrored || i > j) {
      set.insert((i, j))
    } else {
      false
    }
  };
  if band_lo > 0 && rows > band_lo && place(&mut set, band_lo, 0) {
    added += 1;
  }
  if !mirrored
    && band_hi > 0
    && cols > band_hi
    && added < target
    && place(&mut set, 0, band_hi)
  {
    added += 1;
  }
  let dense = target as f64 > 0.5 * capacity(band_lo, band_hi) as f64;
  let mut guard = 0usize;
  while added < target && guard < target * 30 + 1000 {
    guard += 1;
    // Offsets cluster near the diagonal: d = width · u³.
    let below =
      band_hi == 0 || rng.chance(band_lo as f64 / (band_lo + band_hi) as f64);
    let width = if below { band_lo } else { band_hi };
    let u = rng.uniform();
    let d = 1
      + ((width.max(1) - 1) as f64 * if dense { u } else { u * u * u }).round()
        as usize;
    let (i, j) = if below {
      let j = rng.below(cols.max(1));
      (j + d, j)
    } else {
      let i = rng.below(rows.max(1));
      (i, i + d)
    };
    let (i, j) = if rows != cols && !mirrored {
      // Rectangular: spread the band across the shape.
      (i % rows, (j * cols / rows.max(1)).min(cols - 1))
    } else {
      (i, j)
    };
    if place(&mut set, i, j) {
      added += 1;
    }
  }
  // Anything still missing goes anywhere (a crowded band).
  while added < target {
    let (i, j) = (rng.below(rows), rng.below(cols));
    if place(
      &mut set,
      i.max(j).min(rows - 1),
      if mirrored { i.min(j) } else { j },
    ) {
      added += 1;
    }
    guard += 1;
    if guard > target * 100 + 10_000 {
      break;
    }
  }
  if mirrored {
    let lower_part: Vec<(usize, usize)> =
      set.iter().filter(|(i, j)| i > j).copied().collect();
    for (i, j) in lower_part {
      set.insert((j, i));
    }
  }
  let mut v: Vec<(usize, usize)> = set.into_iter().collect();
  v.sort_unstable();
  v
}

#[derive(Clone, Copy)]
enum Kind {
  Real,
  Integer,
  Complex,
}

fn build(entry: &Entry) -> Option<Expr> {
  let dims = match entry.exact("Dimensions")? {
    Expr::List(ref d) => d
      .iter()
      .filter_map(|x| match x {
        Expr::Integer(n) => Some(*n as usize),
        _ => None,
      })
      .collect::<Vec<_>>(),
    _ => return None,
  };
  let (rows, cols) = (*dims.first()?, *dims.get(1)?);
  let nnz = int(entry.exact("Entries"))?.min(rows.saturating_mul(cols));
  if nnz > MAX_ENTRIES {
    return None;
  }
  let lower = int(entry.exact("LowerBandwidth"))
    .or(int(entry.exact("Bandwidth")))
    .unwrap_or(rows / 10 + 1);
  let upper = int(entry.exact("UpperBandwidth")).unwrap_or(lower);
  let structure = match string(entry.exact("MatrixStructure").as_ref()).as_str()
  {
    "Symmetric" => Structure::Symmetric,
    "SkewSymmetric" => Structure::SkewSymmetric,
    "Hermitian" => Structure::Hermitian,
    _ => Structure::General,
  };
  let kind = match string(entry.exact("Type").as_ref())
    .to_ascii_lowercase()
    .as_str()
  {
    "integer" => Kind::Integer,
    "complex" => Kind::Complex,
    _ => Kind::Real,
  };
  let positive_definite = matches!(entry.exact("PositiveDefiniteQ"), Some(Expr::Identifier(ref t)) if t == "True");
  let mut rng = entry.rng("matrix");
  let pos = pattern(&mut rng, rows, cols, nnz, lower, upper, structure);

  // Values: off-diagonal ones first, mirrored per the structure, then the
  // diagonal (dominant when the matrix must be positive definite).
  let mut values: std::collections::HashMap<(usize, usize), (f64, f64)> =
    std::collections::HashMap::with_capacity(pos.len());
  let mut row_sum = vec![0.0f64; rows];
  for &(i, j) in &pos {
    if i == j || values.contains_key(&(i, j)) {
      continue;
    }
    let (re, im) = match kind {
      Kind::Integer => (
        (rng.int_in(-9, 9).max(1)) as f64
          * if rng.chance(0.5) { -1.0 } else { 1.0 },
        0.0,
      ),
      Kind::Real => (rng.normal(), 0.0),
      Kind::Complex => (rng.normal(), rng.normal()),
    };
    let (re, im) = if positive_definite {
      (-re.abs(), im)
    } else {
      (re, im)
    };
    values.insert((i, j), (re, im));
    if structure != Structure::General && i != j && i < cols && j < rows {
      let mirror = match structure {
        Structure::SkewSymmetric => (-re, -im),
        Structure::Hermitian => (re, -im),
        _ => (re, im),
      };
      values.insert((j, i), mirror);
      row_sum[j] += (re * re + im * im).sqrt();
    }
    row_sum[i] += (re * re + im * im).sqrt();
  }
  for &(i, j) in &pos {
    if i == j {
      let v = if positive_definite {
        row_sum[i] + 1.0 + rng.uniform()
      } else {
        match kind {
          Kind::Integer => rng.int_in(1, 9) as f64,
          _ => rng.normal() * 2.0,
        }
      };
      values.insert((i, j), (v, 0.0));
    }
  }
  let value_expr = |re: f64, im: f64| -> Expr {
    match kind {
      Kind::Integer => Expr::Integer(re.round() as i128),
      Kind::Real => Expr::Real(re),
      Kind::Complex => call("Complex", vec![Expr::Real(re), Expr::Real(im)]),
    }
  };
  // CSR: row pointers, column indices (1-based, as {j} lists), values.
  let mut row_ptr = vec![Expr::Integer(0)];
  let mut col_idx = Vec::with_capacity(pos.len());
  let mut vals = Vec::with_capacity(pos.len());
  let mut k = 0;
  for i in 0..rows {
    while k < pos.len() && pos[k].0 == i {
      let (_, j) = pos[k];
      let (re, im) = values.get(&(i, j)).copied().unwrap_or((1.0, 0.0));
      col_idx.push(Expr::List(vec![Expr::Integer(j as i128 + 1)].into()));
      vals.push(value_expr(re, im));
      k += 1;
    }
    row_ptr.push(Expr::Integer(k as i128));
  }
  let background = match kind {
    Kind::Integer => Expr::Integer(0),
    _ => Expr::Real(0.0),
  };
  let sparse = call(
    "SparseArray",
    vec![
      id_expr("Automatic"),
      Expr::List(
        vec![Expr::Integer(rows as i128), Expr::Integer(cols as i128)].into(),
      ),
      background,
      Expr::List(
        vec![
          Expr::Integer(1),
          Expr::List(
            vec![Expr::List(row_ptr.into()), Expr::List(col_idx.into())].into(),
          ),
          Expr::List(vals.into()),
        ]
        .into(),
      ),
    ],
  );
  Some(crate::evaluator::evaluate_expr_to_expr(&sparse).unwrap_or(sparse))
}

impl Collection for Matrices {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    match property {
      None | Some("Matrix") => {
        super::images::cached_opt(format!("Matrix/{}", entry.name), || {
          build(entry)
        })
      }
      Some("Source") => Some(str_expr(GENERATED_SOURCE)),
      _ => None,
    }
  }
}
