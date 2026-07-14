//! Forward and inverse discrete wavelet transforms (decimated, stationary,
//! packet, and lifting variants).
//!
//! Conventions (matching the Wolfram Language reference):
//! - Lowpass filters sum to 1; each analysis level multiplies by Sqrt[2],
//!   so `Sqrt[2] h` is the orthonormal filter.
//! - Wavelet indices are digit lists: at each level digit 0 refines to the
//!   coarse (lowpass) child and 1 to the detail child; 2D data uses digits
//!   0..3 (2*rowpass + columnpass).
//! - Decimated coefficient arrays at the next level have length
//!   ceil((n + fl - 2)/2) where fl is the analysis filter length.

use super::filters::{Filter, WaveletFilters, highpass_from};
use crate::syntax::Expr;

#[derive(Clone, Debug, PartialEq)]
pub enum Padding {
  Periodic,
  Reflected,
  Reversed,
  Fixed,
  Constant(f64),
}

impl Padding {
  pub fn parse(e: &Expr) -> Option<Padding> {
    match e {
      Expr::String(s) => match s.as_str() {
        "Periodic" => Some(Padding::Periodic),
        "Reflected" => Some(Padding::Reflected),
        "Reversed" => Some(Padding::Reversed),
        "Fixed" => Some(Padding::Fixed),
        _ => None,
      },
      _ => crate::functions::math_ast::expr_to_num(e).map(Padding::Constant),
    }
  }
  pub fn to_expr(&self) -> Expr {
    match self {
      Padding::Periodic => Expr::String("Periodic".into()),
      Padding::Reflected => Expr::String("Reflected".into()),
      Padding::Reversed => Expr::String("Reversed".into()),
      Padding::Fixed => Expr::String("Fixed".into()),
      Padding::Constant(c) => Expr::Real(*c),
    }
  }
}

/// Sample x at arbitrary index i, extending by the padding rule.
fn pad_sample(x: &[f64], i: i64, pad: &Padding) -> f64 {
  let n = x.len() as i64;
  if (0..n).contains(&i) {
    return x[i as usize];
  }
  match pad {
    Padding::Periodic => x[i.rem_euclid(n) as usize],
    Padding::Reflected => {
      // abc -> ...cb|abc|ba... (edge not repeated), period 2n-2
      if n == 1 {
        return x[0];
      }
      let m = i.rem_euclid(2 * n - 2);
      let k = if m < n { m } else { 2 * n - 2 - m };
      x[k as usize]
    }
    Padding::Reversed => {
      // abc -> ...cba|abc|cba..., period 2n
      let m = i.rem_euclid(2 * n);
      let k = if m < n { m } else { 2 * n - 1 - m };
      x[k as usize]
    }
    Padding::Fixed => x[i.clamp(0, n - 1) as usize],
    Padding::Constant(c) => *c,
  }
}

/// Coefficient array of rank 1 or 2.
#[derive(Clone, Debug)]
pub enum CoefArray {
  D1(Vec<f64>),
  D2(Vec<Vec<f64>>),
}

impl CoefArray {
  pub fn dims(&self) -> Vec<usize> {
    match self {
      CoefArray::D1(v) => vec![v.len()],
      CoefArray::D2(m) => vec![m.len(), m.first().map_or(0, |r| r.len())],
    }
  }
  pub fn energy(&self) -> f64 {
    match self {
      CoefArray::D1(v) => v.iter().map(|c| c * c).sum(),
      CoefArray::D2(m) => m.iter().flat_map(|r| r.iter()).map(|c| c * c).sum(),
    }
  }
  pub fn to_expr(&self) -> Expr {
    match self {
      CoefArray::D1(v) => {
        Expr::List(v.iter().map(|&c| Expr::Real(c)).collect::<Vec<_>>().into())
      }
      CoefArray::D2(m) => Expr::List(
        m.iter()
          .map(|r| {
            Expr::List(
              r.iter().map(|&c| Expr::Real(c)).collect::<Vec<_>>().into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      ),
    }
  }
  pub fn from_expr(e: &Expr) -> Option<CoefArray> {
    let Expr::List(items) = e else { return None };
    if items.is_empty() {
      return Some(CoefArray::D1(vec![]));
    }
    if matches!(&items[0], Expr::List(_)) {
      let mut rows = Vec::new();
      for row in items.iter() {
        let Expr::List(cells) = row else { return None };
        let mut r = Vec::new();
        for c in cells.iter() {
          r.push(crate::functions::math_ast::expr_to_num(c)?);
        }
        rows.push(r);
      }
      Some(CoefArray::D2(rows))
    } else {
      let mut v = Vec::new();
      for c in items.iter() {
        v.push(crate::functions::math_ast::expr_to_num(c)?);
      }
      Some(CoefArray::D1(v))
    }
  }
}

/// One decimated analysis convolution: out[t] = Sqrt[2] Sum_i f_i x[2t + i],
/// t running over the documented range (length ceil((n + fl - 2)/2)).
///
/// Odd-length data is first padded by one sample so the level has a proper
/// even circular structure; with the default periodic padding all indexing
/// then wraps modulo the evened length, which keeps the inverse exact.
fn dwt_step_1d(x: &[f64], filter: &Filter, pad: &Padding) -> Vec<f64> {
  let n = x.len() as i64;
  let n_even = n + (n % 2);
  let fl = filter.len() as i64;
  let s2 = std::f64::consts::SQRT_2;
  // Periodic padding (the default) gives the canonical circular transform
  // with exactly ceil(n/2) coefficients per band, so the energy norm is
  // conserved for orthogonal families. Other paddings emit the extra
  // boundary coefficients (length ceil((n + fl - 2)/2)).
  let (t0, cnt) = if *pad == Padding::Periodic {
    (0, n_even / 2)
  } else {
    let imax = filter.last().map_or(0, |p| p.0);
    (-(imax.div_euclid(2)), ((n + fl - 2 + 1) / 2).max(1))
  };
  let sample = |idx: i64| -> f64 {
    let idx = if *pad == Padding::Periodic {
      idx.rem_euclid(n_even)
    } else {
      idx
    };
    if (0..n).contains(&idx) {
      x[idx as usize]
    } else {
      pad_sample(x, idx, pad)
    }
  };
  (t0..t0 + cnt)
    .map(|t| {
      s2 * filter
        .iter()
        .map(|&(i, c)| c * sample(2 * t + i))
        .sum::<f64>()
    })
    .collect()
}

/// Canonical (circular) part of a decimated coefficient array: the slice of
/// length ceil(n/2) that corresponds to the circular transform of the
/// periodically padded data. `n` is the length of the parent data.
fn canonical_slice(
  coeffs: &[f64],
  n: usize,
  filter: &Filter,
  pad: &Padding,
) -> Vec<f64> {
  let m = n.div_ceil(2);
  if *pad == Padding::Periodic {
    return coeffs[..m.min(coeffs.len())].to_vec();
  }
  let imax = filter.last().map_or(0, |p| p.0);
  let t0 = -(imax.div_euclid(2));
  (0..m as i64)
    .map(|s| {
      let idx = s - t0;
      coeffs[(idx.rem_euclid(coeffs.len() as i64)) as usize]
    })
    .collect()
}

/// Inverse of one decimated level via circular synthesis:
/// x[j] = Sqrt[2] Sum_t (p[j-2t] a[t] + g[j-2t] d[t]) over the canonical
/// coefficients, reconstructing to length n (the parent length).
fn idwt_step_1d(
  approx: &[f64],
  detail: &[f64],
  n: usize,
  filters: &WaveletFilters,
  pad: &Padding,
) -> Vec<f64> {
  let analysis_lo = &filters.dual_lo;
  let analysis_hi = highpass_from(&filters.primal_lo);
  let synth_lo = &filters.primal_lo;
  let synth_hi = highpass_from(&filters.dual_lo);
  let s2 = std::f64::consts::SQRT_2;
  if *pad == Padding::Periodic {
    // Circular synthesis over the canonical coefficients: exact perfect
    // reconstruction for the biorthogonal filter-bank pairs.
    let a = canonical_slice(approx, n, analysis_lo, pad);
    let d = canonical_slice(detail, n, &analysis_hi, pad);
    let m = n.div_ceil(2);
    let n_even = 2 * m;
    let mut x = vec![0.0; n_even];
    for (t, (&at, &dt)) in a.iter().zip(d.iter()).enumerate() {
      for &(i, c) in synth_lo.iter() {
        let j = (2 * t as i64 + i).rem_euclid(n_even as i64) as usize;
        x[j] += s2 * c * at;
      }
      for &(i, c) in synth_hi.iter() {
        let j = (2 * t as i64 + i).rem_euclid(n_even as i64) as usize;
        x[j] += s2 * c * dt;
      }
    }
    x.truncate(n);
    return x;
  }
  // Non-periodic paddings store the extra boundary coefficients; synthesize
  // by the direct (non-circular) adjoint over the stored range. Interior
  // samples reconstruct exactly; the outermost samples are approximate.
  let mut x = vec![0.0; n];
  let mut accumulate = |coeffs: &[f64], analysis: &Filter, synth: &Filter| {
    let imax = analysis.last().map_or(0, |p| p.0);
    let t0 = -(imax.div_euclid(2));
    for (idx, &ct) in coeffs.iter().enumerate() {
      let t = t0 + idx as i64;
      for &(i, c) in synth.iter() {
        let j = 2 * t + i;
        if (0..n as i64).contains(&j) {
          x[j as usize] += s2 * c * ct;
        }
      }
    }
  };
  accumulate(approx, analysis_lo, synth_lo);
  accumulate(detail, &analysis_hi, &synth_hi);
  x
}

/// Stationary (a trous) analysis at dilation 2^level:
/// out[t] = Sqrt[2] Sum_i f_i x[(t + 2^level i) mod n].
fn swt_step_1d(x: &[f64], filter: &Filter, dilation: i64) -> Vec<f64> {
  let n = x.len() as i64;
  let s2 = std::f64::consts::SQRT_2;
  (0..n)
    .map(|t| {
      s2 * filter
        .iter()
        .map(|&(i, c)| c * x[(t + dilation * i).rem_euclid(n) as usize])
        .sum::<f64>()
    })
    .collect()
}

/// Stationary synthesis (adjoint averaged over the two phases):
/// x[t] = (1/Sqrt[2]) Sum_i (p_i a[(t - D i) mod n] + g_i d[(t - D i) mod n]).
fn iswt_step_1d(
  approx: &[f64],
  detail: &[f64],
  dilation: i64,
  filters: &WaveletFilters,
) -> Vec<f64> {
  let n = approx.len() as i64;
  let synth_lo = &filters.primal_lo;
  let synth_hi = highpass_from(&filters.dual_lo);
  let inv_s2 = 1.0 / std::f64::consts::SQRT_2;
  (0..n)
    .map(|t| {
      let mut acc = 0.0;
      for &(i, c) in synth_lo.iter() {
        acc += c * approx[(t - dilation * i).rem_euclid(n) as usize];
      }
      for &(i, c) in synth_hi.iter() {
        acc += c * detail[(t - dilation * i).rem_euclid(n) as usize];
      }
      inv_s2 * acc
    })
    .collect()
}

// ---------------------------------------------------------------------------
// Rank-generic single-level steps
// ---------------------------------------------------------------------------

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
  if m.is_empty() {
    return vec![];
  }
  (0..m[0].len())
    .map(|j| m.iter().map(|row| row[j]).collect())
    .collect()
}

/// One decimated level. Returns the children keyed by digit.
fn dwt_level(
  data: &CoefArray,
  filters: &WaveletFilters,
  pad: &Padding,
) -> Vec<(u8, CoefArray)> {
  let lo = &filters.dual_lo;
  let hi = highpass_from(&filters.primal_lo);
  match data {
    CoefArray::D1(x) => vec![
      (0, CoefArray::D1(dwt_step_1d(x, lo, pad))),
      (1, CoefArray::D1(dwt_step_1d(x, &hi, pad))),
    ],
    CoefArray::D2(m) => {
      // Filter along dim 2 (within each row) first, then along dim 1.
      // Digit = 2*(highpass along dim 1) + (highpass along dim 2).
      let rows_lo: Vec<Vec<f64>> =
        m.iter().map(|r| dwt_step_1d(r, lo, pad)).collect();
      let rows_hi: Vec<Vec<f64>> =
        m.iter().map(|r| dwt_step_1d(r, &hi, pad)).collect();
      let mut out = Vec::new();
      for (col_digit, half) in [(0u8, rows_lo), (1u8, rows_hi)] {
        let cols = transpose(&half);
        let cols_lo: Vec<Vec<f64>> =
          cols.iter().map(|c| dwt_step_1d(c, lo, pad)).collect();
        let cols_hi: Vec<Vec<f64>> =
          cols.iter().map(|c| dwt_step_1d(c, &hi, pad)).collect();
        out.push((col_digit, CoefArray::D2(transpose(&cols_lo))));
        out.push((col_digit + 2, CoefArray::D2(transpose(&cols_hi))));
      }
      out
    }
  }
}

/// Inverse of one decimated level for children keyed by digit; `dims` is the
/// parent dimensions.
fn idwt_level(
  children: &[(u8, CoefArray)],
  dims: &[usize],
  filters: &WaveletFilters,
  pad: &Padding,
) -> CoefArray {
  match dims.len() {
    1 => {
      let n = dims[0];
      let m = n.div_ceil(2);
      let zero = vec![0.0; m];
      let get = |digit: u8| -> Vec<f64> {
        children
          .iter()
          .find(|(d, _)| *d == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D1(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| zero.clone())
      };
      CoefArray::D1(idwt_step_1d(&get(0), &get(1), n, filters, pad))
    }
    _ => {
      let (n1, n2) = (dims[0], dims[1]);
      let (m1, m2) = (n1.div_ceil(2), n2.div_ceil(2));
      let get = |digit: u8| -> Vec<Vec<f64>> {
        children
          .iter()
          .find(|(d, _)| *d == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D2(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| vec![vec![0.0; m2]; m1])
      };
      // Undo dim-1 (rows) filtering per column, then dim-2 per row.
      let mut halves: Vec<Vec<Vec<f64>>> = Vec::new(); // [dim2-digit][rows][cols]
      for col_digit in [0u8, 1] {
        let lo = get(col_digit); // dim1 lowpass
        let hi = get(col_digit + 2); // dim1 highpass
        let lo_cols = transpose(&lo);
        let hi_cols = transpose(&hi);
        let rec_cols: Vec<Vec<f64>> = lo_cols
          .iter()
          .zip(hi_cols.iter())
          .map(|(a, d)| idwt_step_1d(a, d, n1, filters, pad))
          .collect();
        halves.push(transpose(&rec_cols));
      }
      let rec_rows: Vec<Vec<f64>> = halves[0]
        .iter()
        .zip(halves[1].iter())
        .map(|(a, d)| idwt_step_1d(a, d, n2, filters, pad))
        .collect();
      CoefArray::D2(rec_rows)
    }
  }
}

/// One stationary level at the given dilation.
fn swt_level(
  data: &CoefArray,
  filters: &WaveletFilters,
  dilation: i64,
) -> Vec<(u8, CoefArray)> {
  let lo = &filters.dual_lo;
  let hi = highpass_from(&filters.primal_lo);
  match data {
    CoefArray::D1(x) => vec![
      (0, CoefArray::D1(swt_step_1d(x, lo, dilation))),
      (1, CoefArray::D1(swt_step_1d(x, &hi, dilation))),
    ],
    CoefArray::D2(m) => {
      let rows_lo: Vec<Vec<f64>> =
        m.iter().map(|r| swt_step_1d(r, lo, dilation)).collect();
      let rows_hi: Vec<Vec<f64>> =
        m.iter().map(|r| swt_step_1d(r, &hi, dilation)).collect();
      let mut out = Vec::new();
      for (col_digit, half) in [(0u8, rows_lo), (1u8, rows_hi)] {
        let cols = transpose(&half);
        let cols_lo: Vec<Vec<f64>> =
          cols.iter().map(|c| swt_step_1d(c, lo, dilation)).collect();
        let cols_hi: Vec<Vec<f64>> =
          cols.iter().map(|c| swt_step_1d(c, &hi, dilation)).collect();
        out.push((col_digit, CoefArray::D2(transpose(&cols_lo))));
        out.push((col_digit + 2, CoefArray::D2(transpose(&cols_hi))));
      }
      out
    }
  }
}

fn iswt_level(
  children: &[(u8, CoefArray)],
  dims: &[usize],
  filters: &WaveletFilters,
  dilation: i64,
) -> CoefArray {
  match dims.len() {
    1 => {
      let n = dims[0];
      let zero = vec![0.0; n];
      let get = |digit: u8| -> Vec<f64> {
        children
          .iter()
          .find(|(d, _)| *d == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D1(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| zero.clone())
      };
      CoefArray::D1(iswt_step_1d(&get(0), &get(1), dilation, filters))
    }
    _ => {
      let (n1, n2) = (dims[0], dims[1]);
      let get = |digit: u8| -> Vec<Vec<f64>> {
        children
          .iter()
          .find(|(d, _)| *d == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D2(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| vec![vec![0.0; n2]; n1])
      };
      let mut halves: Vec<Vec<Vec<f64>>> = Vec::new();
      for col_digit in [0u8, 1] {
        let lo = get(col_digit);
        let hi = get(col_digit + 2);
        let lo_cols = transpose(&lo);
        let hi_cols = transpose(&hi);
        let rec_cols: Vec<Vec<f64>> = lo_cols
          .iter()
          .zip(hi_cols.iter())
          .map(|(a, d)| iswt_step_1d(a, d, dilation, filters))
          .collect();
        halves.push(transpose(&rec_cols));
      }
      let rec_rows: Vec<Vec<f64>> = halves[0]
        .iter()
        .zip(halves[1].iter())
        .map(|(a, d)| iswt_step_1d(a, d, dilation, filters))
        .collect();
      CoefArray::D2(rec_rows)
    }
  }
}

// ---------------------------------------------------------------------------
// Full transforms
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TransformKind {
  Dwt,
  Swt,
  Dwpt,
  Swpt,
  Lwt,
}

impl TransformKind {
  pub fn name(&self) -> &'static str {
    match self {
      TransformKind::Dwt => "DiscreteWaveletTransform",
      TransformKind::Swt => "StationaryWaveletTransform",
      TransformKind::Dwpt => "DiscreteWaveletPacketTransform",
      TransformKind::Swpt => "StationaryWaveletPacketTransform",
      TransformKind::Lwt => "LiftingWaveletTransform",
    }
  }
  pub fn from_name(s: &str) -> Option<TransformKind> {
    match s {
      "DiscreteWaveletTransform" => Some(TransformKind::Dwt),
      "StationaryWaveletTransform" => Some(TransformKind::Swt),
      "DiscreteWaveletPacketTransform" => Some(TransformKind::Dwpt),
      "StationaryWaveletPacketTransform" => Some(TransformKind::Swpt),
      "LiftingWaveletTransform" => Some(TransformKind::Lwt),
      _ => None,
    }
  }
  pub fn is_packet(&self) -> bool {
    matches!(self, TransformKind::Dwpt | TransformKind::Swpt)
  }
  pub fn is_stationary(&self) -> bool {
    matches!(self, TransformKind::Swt | TransformKind::Swpt)
  }
}

/// Default refinement level: floor(log2(n) + 1/2) on the minimum dimension,
/// capped at 4 for the packet transforms; the lifting transform uses the
/// 2-adic valuation of the (evened) length, capped at 4.
pub fn default_refinement(kind: TransformKind, dims: &[usize]) -> usize {
  let n = dims.iter().copied().min().unwrap_or(1).max(1);
  match kind {
    TransformKind::Lwt => {
      let r = dims
        .iter()
        .map(|&d| {
          let e = if d % 2 == 1 { d + 1 } else { d };
          (e.trailing_zeros() as usize).max(1)
        })
        .min()
        .unwrap_or(1);
      r.min(4)
    }
    _ => {
      let base = ((n as f64).log2() + 0.5).floor().max(1.0) as usize;
      if kind.is_packet() { base.min(4) } else { base }
    }
  }
}

/// Compute the full coefficient tree. Returns (wind -> coefficients) in
/// (level, lexicographic) order.
pub fn forward_transform(
  data: &CoefArray,
  filters: &WaveletFilters,
  kind: TransformKind,
  r: usize,
  pad: &Padding,
) -> Vec<(Vec<u8>, CoefArray)> {
  let mut out: Vec<(Vec<u8>, CoefArray)> = Vec::new();
  // For the lifting transform, pre-pad every dimension to a multiple of 2^r.
  let data_padded;
  let data = if kind == TransformKind::Lwt {
    data_padded = lwt_prepad(data, r, pad);
    &data_padded
  } else {
    data
  };
  let mut frontier: Vec<(Vec<u8>, CoefArray)> = vec![(vec![], data.clone())];
  for level in 0..r {
    let mut next: Vec<(Vec<u8>, CoefArray)> = Vec::new();
    for (wind, arr) in &frontier {
      let expand = kind.is_packet() || wind.iter().all(|&d| d == 0);
      if !expand {
        continue;
      }
      let children = match kind {
        TransformKind::Swt | TransformKind::Swpt => {
          swt_level(arr, filters, 1 << level)
        }
        TransformKind::Lwt => dwt_level_plain(arr, filters),
        _ => dwt_level(arr, filters, pad),
      };
      for (digit, carr) in children {
        let mut w = wind.clone();
        w.push(digit);
        next.push((w, carr));
      }
    }
    next.sort_by(|a, b| a.0.cmp(&b.0));
    out.extend(next.iter().cloned());
    frontier = next;
  }
  out
}

/// Decimated level without boundary coefficients (used by the lifting
/// transform, whose input is pre-padded to a multiple of 2^r): plain
/// circular convolution with downsampling.
fn dwt_level_plain(
  data: &CoefArray,
  filters: &WaveletFilters,
) -> Vec<(u8, CoefArray)> {
  let lo = &filters.dual_lo;
  // Lifting reports detail with the opposite sign of the decimated transform.
  let hi = super::filters::negate_filter(&highpass_from(&filters.primal_lo));
  let step = |x: &[f64], f: &Filter| -> Vec<f64> {
    let n = x.len() as i64;
    let s2 = std::f64::consts::SQRT_2;
    (0..n / 2)
      .map(|t| {
        s2 * f
          .iter()
          .map(|&(i, c)| c * x[(2 * t + i).rem_euclid(n) as usize])
          .sum::<f64>()
      })
      .collect()
  };
  match data {
    CoefArray::D1(x) => vec![
      (0, CoefArray::D1(step(x, lo))),
      (1, CoefArray::D1(step(x, &hi))),
    ],
    CoefArray::D2(m) => {
      let rows_lo: Vec<Vec<f64>> = m.iter().map(|r| step(r, lo)).collect();
      let rows_hi: Vec<Vec<f64>> = m.iter().map(|r| step(r, &hi)).collect();
      let mut out = Vec::new();
      for (col_digit, half) in [(0u8, rows_lo), (1u8, rows_hi)] {
        let cols = transpose(&half);
        let cols_lo: Vec<Vec<f64>> = cols.iter().map(|c| step(c, lo)).collect();
        let cols_hi: Vec<Vec<f64>> =
          cols.iter().map(|c| step(c, &hi)).collect();
        out.push((col_digit, CoefArray::D2(transpose(&cols_lo))));
        out.push((col_digit + 2, CoefArray::D2(transpose(&cols_hi))));
      }
      out
    }
  }
}

fn lwt_prepad(data: &CoefArray, r: usize, pad: &Padding) -> CoefArray {
  let mult = 1usize << r;
  let target = |n: usize| n.div_ceil(mult) * mult;
  match data {
    CoefArray::D1(x) => {
      let m = target(x.len());
      CoefArray::D1((0..m).map(|i| pad_sample(x, i as i64, pad)).collect())
    }
    CoefArray::D2(rows) => {
      let m1 = target(rows.len());
      let m2 = target(rows.first().map_or(0, |r| r.len()));
      let padded_rows: Vec<Vec<f64>> = rows
        .iter()
        .map(|row| (0..m2).map(|j| pad_sample(row, j as i64, pad)).collect())
        .collect();
      let get_row = |i: usize| -> Vec<f64> {
        let n1 = padded_rows.len() as i64;
        let idx = match pad {
          Padding::Periodic => (i as i64).rem_euclid(n1),
          Padding::Fixed => (i as i64).min(n1 - 1),
          Padding::Reflected => {
            let p = 2 * n1 - 2;
            let m = (i as i64).rem_euclid(p.max(1));
            if m < n1 { m } else { p - m }
          }
          Padding::Reversed => {
            let p = 2 * n1;
            let m = (i as i64).rem_euclid(p);
            if m < n1 { m } else { p - 1 - m }
          }
          Padding::Constant(_) => -1,
        };
        if idx < 0 {
          if let Padding::Constant(c) = pad {
            return vec![*c; m2];
          }
          unreachable!()
        }
        padded_rows[idx as usize].clone()
      };
      CoefArray::D2((0..m1).map(get_row).collect())
    }
  }
}

/// Dimensions of the wavelet coefficient array at each node, given the
/// original data dims. Follows the forward computation without doing it.
pub fn node_dims(
  wind: &[u8],
  data_dims: &[usize],
  filters: &WaveletFilters,
  kind: TransformKind,
  r: usize,
  pad: &Padding,
) -> Vec<usize> {
  if kind.is_stationary() {
    return data_dims.to_vec();
  }
  if kind == TransformKind::Lwt {
    let mult = 1usize << r;
    return data_dims
      .iter()
      .map(|&n| (n.div_ceil(mult) * mult) >> wind.len())
      .collect();
  }
  let lo_len = filters.dual_lo.len();
  let hi_len = filters.primal_lo.len();
  let mut dims = data_dims.to_vec();
  for (level, &digit) in wind.iter().enumerate() {
    let _ = level;
    dims = dims
      .iter()
      .enumerate()
      .map(|(axis, &n)| {
        // For rank-2 data, digit bit 1 (value 2) is dim-1 highpass and bit 0
        // is dim-2 highpass; for rank 1, the digit itself selects highpass.
        let is_high = if dims.len() == 1 {
          digit == 1
        } else if axis == 0 {
          digit >= 2
        } else {
          digit % 2 == 1
        };
        if *pad == Padding::Periodic {
          n.div_ceil(2)
        } else {
          let fl = if is_high { hi_len } else { lo_len };
          (n + fl - 1) / 2
        }
      })
      .collect();
  }
  dims
}

/// Reconstruct the root data from a set of coefficients. Missing nodes are
/// treated as zero. `include` is the set of winds whose coefficients are
/// used (a "basis"); interior nodes are reconstructed recursively.
pub fn inverse_transform(
  coeffs: &std::collections::BTreeMap<Vec<u8>, CoefArray>,
  data_dims: &[usize],
  filters: &WaveletFilters,
  kind: TransformKind,
  r: usize,
  pad: &Padding,
) -> CoefArray {
  #[allow(clippy::too_many_arguments)]
  fn reconstruct(
    wind: &[u8],
    coeffs: &std::collections::BTreeMap<Vec<u8>, CoefArray>,
    data_dims: &[usize],
    filters: &WaveletFilters,
    kind: TransformKind,
    r: usize,
    max_digit: u8,
    pad: &Padding,
  ) -> Option<CoefArray> {
    if let Some(c) = coeffs.get(wind) {
      return Some(c.clone());
    }
    if wind.len() >= r {
      return None;
    }
    let mut children: Vec<(u8, CoefArray)> = Vec::new();
    for digit in 0..=max_digit {
      let mut w = wind.to_vec();
      w.push(digit);
      if let Some(c) =
        reconstruct(&w, coeffs, data_dims, filters, kind, r, max_digit, pad)
      {
        children.push((digit, c));
      }
    }
    if children.is_empty() {
      return None;
    }
    let dims = node_dims(wind, data_dims, filters, kind, r, pad);
    Some(match kind {
      TransformKind::Swt | TransformKind::Swpt => {
        iswt_level(&children, &dims, filters, 1 << wind.len())
      }
      TransformKind::Lwt => idwt_level_plain(&children, &dims, filters),
      _ => idwt_level(&children, &dims, filters, pad),
    })
  }
  let max_digit = if data_dims.len() >= 2 { 3 } else { 1 };
  let result =
    reconstruct(&[], coeffs, data_dims, filters, kind, r, max_digit, pad)
      .unwrap_or_else(|| match data_dims.len() {
        1 => CoefArray::D1(vec![0.0; data_dims[0]]),
        _ => CoefArray::D2(vec![vec![0.0; data_dims[1]]; data_dims[0]]),
      });
  // The lifting transform pre-pads; truncate back to the original dims.
  match (result, data_dims.len()) {
    (CoefArray::D1(mut v), 1) => {
      v.truncate(data_dims[0]);
      CoefArray::D1(v)
    }
    (CoefArray::D2(mut m), _) => {
      m.truncate(data_dims[0]);
      for row in &mut m {
        row.truncate(data_dims[1]);
      }
      CoefArray::D2(m)
    }
    (other, _) => other,
  }
}

/// Inverse of `dwt_level_plain` (circular synthesis, exact halving).
fn idwt_level_plain(
  children: &[(u8, CoefArray)],
  dims: &[usize],
  filters: &WaveletFilters,
) -> CoefArray {
  let synth_lo = filters.primal_lo.clone();
  // Matches the flipped detail-sign convention of `dwt_level_plain`, so the
  // lifting transform still reconstructs exactly.
  let synth_hi =
    super::filters::negate_filter(&highpass_from(&filters.dual_lo));
  let step = |a: &[f64], d: &[f64], n: usize| -> Vec<f64> {
    let s2 = std::f64::consts::SQRT_2;
    let mut x = vec![0.0; n];
    for t in 0..a.len() {
      for &(i, c) in synth_lo.iter() {
        let j = (2 * t as i64 + i).rem_euclid(n as i64) as usize;
        x[j] += s2 * c * a[t];
      }
      for &(i, c) in synth_hi.iter() {
        let j = (2 * t as i64 + i).rem_euclid(n as i64) as usize;
        x[j] += s2 * c * d[t];
      }
    }
    x
  };
  match dims.len() {
    1 => {
      let n = dims[0];
      let m = n / 2;
      let zero = vec![0.0; m];
      let get = |digit: u8| -> Vec<f64> {
        children
          .iter()
          .find(|(dg, _)| *dg == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D1(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| zero.clone())
      };
      CoefArray::D1(step(&get(0), &get(1), n))
    }
    _ => {
      let (n1, n2) = (dims[0], dims[1]);
      let (m1, m2) = (n1 / 2, n2 / 2);
      let get = |digit: u8| -> Vec<Vec<f64>> {
        children
          .iter()
          .find(|(dg, _)| *dg == digit)
          .and_then(|(_, c)| match c {
            CoefArray::D2(v) => Some(v.clone()),
            _ => None,
          })
          .unwrap_or_else(|| vec![vec![0.0; m2]; m1])
      };
      let mut halves: Vec<Vec<Vec<f64>>> = Vec::new();
      for col_digit in [0u8, 1] {
        let lo = get(col_digit);
        let hi = get(col_digit + 2);
        let lo_cols = transpose(&lo);
        let hi_cols = transpose(&hi);
        let rec_cols: Vec<Vec<f64>> = lo_cols
          .iter()
          .zip(hi_cols.iter())
          .map(|(a, d)| step(a, d, n1))
          .collect();
        halves.push(transpose(&rec_cols));
      }
      let rec_rows: Vec<Vec<f64>> = halves[0]
        .iter()
        .zip(halves[1].iter())
        .map(|(a, d)| step(a, d, n2))
        .collect();
      CoefArray::D2(rec_rows)
    }
  }
}

/// Basis indices (the complete non-redundant set used by the inverse) for a
/// transform of refinement r over data of the given rank.
pub fn basis_index(kind: TransformKind, r: usize, rank: usize) -> Vec<Vec<u8>> {
  let digits: Vec<u8> = if rank >= 2 {
    vec![0, 1, 2, 3]
  } else {
    vec![0, 1]
  };
  let mut basis: Vec<Vec<u8>> = Vec::new();
  if kind.is_packet() {
    fn expand(
      prefix: Vec<u8>,
      depth: usize,
      digits: &[u8],
      out: &mut Vec<Vec<u8>>,
    ) {
      if depth == 0 {
        out.push(prefix);
        return;
      }
      for &d in digits {
        let mut p = prefix.clone();
        p.push(d);
        expand(p, depth - 1, digits, out);
      }
    }
    expand(vec![], r, &digits, &mut basis);
  } else {
    for level in 1..=r {
      let prefix = vec![0u8; level - 1];
      for &d in digits.iter().skip(1) {
        let mut w = prefix.clone();
        w.push(d);
        basis.push(w);
      }
      if level == r {
        let mut w = prefix;
        w.push(0);
        basis.push(w);
      }
    }
    // Natural construction order (finest detail first, coarse approximation
    // last) matches Wolfram's BasisIndex ordering; no sorting.
  }
  basis
}

/// LiftingFilterData[…] placeholder object produced by
/// WaveletFilterCoefficients[wave, "LiftingFilter"]: stores the wavelet so
/// filter properties and LiftingWaveletTransform can consume it.
pub fn lifting_filter_data(spec: &super::WaveletSpec) -> Expr {
  let wavelet = super::spec_to_expr(spec);
  Expr::FunctionCall {
    name: "LiftingFilterData".to_string(),
    args: vec![wavelet].into(),
  }
}
