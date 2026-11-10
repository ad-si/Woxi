//! `ListDeconvolve[ker, list]`: deconvolution of a numeric array of any rank.
//!
//! This follows Wolfram's own top-level implementation (the
//! `LinearAlgebra`DeconvolveDump`` package) step by step, so the results
//! agree to machine precision:
//!
//! - the data is scaled to a maximum absolute value of 1 (and scaled back at
//!   the end), and every axis shorter than 3 is padded by one reversed
//!   element on each side;
//! - a kernel axis of even length gets one trailing zero;
//! - the kernel is zero-padded to the data's size and rotated so that its
//!   center (`Ceiling[dims/2]`) sits at the origin, then transformed with
//!   `Fourier[…, FourierParameters -> {1, -1}]`, or for a symmetric kernel
//!   (except with `"Wiener"`) with the discrete cosine transform;
//! - the spectral methods are regularized inverse filters: `"DampedLS"`
//!   (Laplacian-damped least squares), `"Tikhonov"`, `"TSVD"` (truncated
//!   singular values) and `"Wiener"`. An `Automatic` regularization
//!   parameter is chosen by generalized cross-validation exactly as Wolfram
//!   does (Brent minimization for DampedLS/Tikhonov, the discrete GCV
//!   minimum for TSVD; Wiener uses 1/100).
//!
//! The iterative methods (`"Hybrid"`, `"RichardsonLucy"`,
//! `"SteepestDescent"`, `"TotalVariation"`) are not implemented: such calls
//! stay unevaluated.

use crate::InterpreterError;
use crate::helpers::unevaluated;
use crate::syntax::{Expr, expr_to_string};

/// `Sqrt[$MachineEpsilon]`, Wolfram's `$eps` for machine-precision input.
const EPS: f64 = 1.490_116_119_384_765_6e-8;

type C = (f64, f64);

/// A rectangular real array: its dimensions and row-major values.
struct Array {
  dims: Vec<usize>,
  data: Vec<f64>,
}

impl Array {
  fn len(&self) -> usize {
    self.data.len()
  }
}

/// Parses a nonempty rectangular array of real numbers of any rank.
fn parse_real_array(e: &Expr) -> Option<Array> {
  fn walk(
    e: &Expr,
    depth: usize,
    dims: &mut Vec<usize>,
    data: &mut Vec<f64>,
  ) -> bool {
    match e {
      Expr::List(items) => {
        if items.is_empty() {
          return false;
        }
        if dims.len() == depth {
          dims.push(items.len());
        } else if dims.len() < depth || dims[depth] != items.len() {
          return false;
        }
        items.iter().all(|it| walk(it, depth + 1, dims, data))
      }
      other => {
        if depth != dims.len() {
          return false;
        }
        match crate::functions::math_ast::try_eval_to_f64(other) {
          Some(v) if v.is_finite() => {
            data.push(v);
            true
          }
          _ => false,
        }
      }
    }
  }
  if !matches!(e, Expr::List(_)) {
    return None;
  }
  let mut dims = Vec::new();
  let mut data = Vec::new();
  if !walk(e, 0, &mut dims, &mut data)
    || data.len() != dims.iter().product::<usize>()
  {
    return None;
  }
  Some(Array { dims, data })
}

fn strides(dims: &[usize]) -> Vec<usize> {
  let mut s = vec![1; dims.len()];
  for k in (0..dims.len().saturating_sub(1)).rev() {
    s[k] = s[k + 1] * dims[k + 1];
  }
  s
}

/// The multi-index of flat position `i`.
fn unflatten(mut i: usize, dims: &[usize]) -> Vec<usize> {
  let mut idx = vec![0; dims.len()];
  for k in (0..dims.len()).rev() {
    idx[k] = i % dims[k];
    i /= dims[k];
  }
  idx
}

/// Applies a one-dimensional transform along every axis in turn.
fn separable<T: Copy>(
  data: &mut [T],
  dims: &[usize],
  mut core: impl FnMut(&[T]) -> Vec<T>,
) {
  let total = data.len();
  let mut stride = total;
  for &len in dims {
    stride /= len;
    let block = len * stride;
    for start in (0..total).step_by(block) {
      for offset in 0..stride {
        let line: Vec<T> = (0..len)
          .map(|i| data[start + offset + i * stride])
          .collect();
        for (i, v) in core(&line).into_iter().enumerate() {
          data[start + offset + i * stride] = v;
        }
      }
    }
  }
}

/// `Fourier[x, FourierParameters -> {1, -1}]` (or `InverseFourier`).
fn dft(data: &[C], dims: &[usize], inverse: bool) -> Vec<C> {
  let mut out = data.to_vec();
  separable(&mut out, dims, |line| {
    crate::functions::math_ast::dft_core(line, 1.0, -1.0, inverse)
  });
  out
}

/// Wolfram's `DCT` helper: `FourierDCT[x, 2]`, rescaled for rank 2.
fn dct(data: &[f64], dims: &[usize]) -> Vec<f64> {
  let mut out = data.to_vec();
  separable(&mut out, dims, |line| {
    crate::functions::math_ast::dct_core(line, 2)
  });
  if dims.len() == 2 {
    let sq2 = 2f64.sqrt();
    for (i, v) in out.iter_mut().enumerate() {
      *v *= 2.0;
      if i / dims[1] == 0 {
        *v /= sq2;
      }
      if i % dims[1] == 0 {
        *v /= sq2;
      }
    }
  }
  out
}

/// Wolfram's `IDCT` helper, the inverse of [`dct`].
fn idct(data: &[f64], dims: &[usize]) -> Vec<f64> {
  let mut out = data.to_vec();
  if dims.len() == 2 {
    let sq2 = 2f64.sqrt();
    for (i, v) in out.iter_mut().enumerate() {
      *v /= 2.0;
      if i / dims[1] == 0 {
        *v *= sq2;
      }
      if i % dims[1] == 0 {
        *v *= sq2;
      }
    }
  }
  separable(&mut out, dims, |line| {
    crate::functions::math_ast::dct_core(line, 3)
  });
  out
}

/// `PadRight[a, dims, 0]` (dims at least as large as `a`'s).
fn pad_right(a: &Array, dims: &[usize]) -> Array {
  let total: usize = dims.iter().product();
  let st = strides(dims);
  let mut data = vec![0.0; total];
  for (i, &v) in a.data.iter().enumerate() {
    let idx = unflatten(i, &a.dims);
    if idx.iter().zip(dims).all(|(i, d)| i < d) {
      data[idx.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()] = v;
    }
  }
  Array {
    dims: dims.to_vec(),
    data,
  }
}

/// `RotateLeft[a, shifts]`.
fn rotate_left(a: &Array, shifts: &[usize]) -> Vec<f64> {
  let st = strides(&a.dims);
  let mut out = vec![0.0; a.len()];
  for (i, slot) in out.iter_mut().enumerate() {
    let idx = unflatten(i, &a.dims);
    let src: usize = idx
      .iter()
      .zip(shifts)
      .zip(&a.dims)
      .zip(&st)
      .map(|(((i, s), d), st)| ((i + s) % d) * st)
      .sum();
    *slot = a.data[src];
  }
  out
}

/// `ArrayPad[a, 1, "Reversed"]`: every axis grows by one element on each
/// side, repeating the edge element.
fn pad_reversed(a: &Array) -> Array {
  let dims: Vec<usize> = a.dims.iter().map(|d| d + 2).collect();
  let st = strides(&a.dims);
  let total: usize = dims.iter().product();
  let data = (0..total)
    .map(|i| {
      let idx = unflatten(i, &dims);
      let src: usize = idx
        .iter()
        .zip(&a.dims)
        .zip(&st)
        .map(|((&i, &d), &s)| (i.saturating_sub(1)).min(d - 1) * s)
        .sum();
      a.data[src]
    })
    .collect();
  Array { dims, data }
}

/// `ArrayPad[a, -1]`: drops one element from both ends of every axis.
fn crop_one(data: &[f64], dims: &[usize]) -> Array {
  let inner: Vec<usize> = dims.iter().map(|d| d - 2).collect();
  let st = strides(dims);
  let total: usize = inner.iter().product();
  let out = (0..total)
    .map(|i| {
      let idx = unflatten(i, &inner);
      data[idx.iter().zip(&st).map(|(i, s)| (i + 1) * s).sum::<usize>()]
    })
    .collect();
  Array {
    dims: inner,
    data: out,
  }
}

/// Wolfram's `ThresholdAbs[x, level]`: values closer to zero than `level`
/// become `±level`.
fn threshold_abs(x: f64, level: f64) -> f64 {
  if x.abs() < level {
    if x >= 0.0 { level } else { -level }
  } else {
    x
  }
}

fn norm2(z: C) -> f64 {
  z.0 * z.0 + z.1 * z.1
}

/// Brent's one-dimensional minimization on `[a, b]` — a transcription of
/// Wolfram's `fmin` (the classic Forsythe–Malcolm–Moler routine).
fn fmin(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
  let (mut a2, mut b2) = (a, b);
  let c = 0.5 * (3.0 - 5f64.sqrt());
  let mut d = 0.0;
  let mut v = a2 + c * (b2 - a2);
  let mut w = v;
  let mut x = v;
  let mut e: f64 = 0.0;
  let mut fx = f(x);
  let mut fv = fx;
  let mut fw = fx;
  let tol3 = tol / 3.0;
  let mut xm = f64::midpoint(a2, b2);
  let mut tol1 = EPS * x.abs() + tol3;
  let mut t2 = 2.0 * tol1;
  while (x - xm).abs() > t2 - 0.5 * (b2 - a2) {
    let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
    if e.abs() > tol1 {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if q > 0.0 {
        p = -p;
      } else {
        q = -q;
      }
      r = e;
      e = d;
    }
    if p.abs() < (0.5 * q * r).abs() && p > q * (a2 - x) && p < q * (b2 - x) {
      d = p / q;
      let u = x + d;
      if u - a2 < t2 || b2 - u < t2 {
        d = tol1;
        if x >= xm {
          d = -d;
        }
      }
    } else {
      e = if x < xm { b2 - x } else { a2 - x };
      d = c * e;
    }
    let u = if d.abs() >= tol1 {
      x + d
    } else if d > 0.0 {
      x + tol1
    } else {
      x - tol1
    };
    let fu = f(u);
    if fx <= fu {
      if u < x {
        a2 = u;
      } else {
        b2 = u;
      }
    }
    if fu <= fx {
      if u < x {
        b2 = x;
      } else {
        a2 = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else if fu <= fw || w == x {
      v = w;
      fv = fw;
      w = u;
      fw = fu;
    } else if !(fu > fv && v != x && v != w) {
      v = u;
      fv = fu;
    }
    xm = f64::midpoint(a2, b2);
    tol1 = EPS * x.abs() + tol3;
    t2 = 2.0 * tol1;
  }
  x
}

fn min_max(v: &[f64]) -> (f64, f64) {
  v.iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
      (lo.min(x), hi.max(x))
    })
}

/// GCV choice of the DampedLS parameter (`gcvGtikhonovMachPrec`).
fn gcv_gtikhonov(sa: &[f64], sd: &[f64], b: &[f64]) -> f64 {
  let (smin, smax) = min_max(sa);
  let sa2: Vec<f64> = sa.iter().map(|s| threshold_abs(s * s, EPS)).collect();
  let sd2: Vec<f64> = sd.iter().map(|s| s * s).collect();
  fmin(
    |alpha| {
      let mut num = 0.0;
      let mut tr = 0.0;
      for i in 0..sa2.len() {
        let phid = sd2[i] / (sa2[i] + alpha * alpha * sd2[i]);
        num += (b[i] * phid).powi(2);
        tr += phid;
      }
      num / threshold_abs(tr * tr, EPS)
    },
    smin,
    smax,
    1e-4,
  )
}

/// GCV choice of the Tikhonov parameter (`gcvTikhonovMachPrec`).
fn gcv_tikhonov(s: &[f64], b: &[f64]) -> f64 {
  let (smin, smax) = min_max(s);
  let s2: Vec<f64> = s.iter().map(|s| threshold_abs(s * s, EPS)).collect();
  fmin(
    |alpha| {
      let mut num = 0.0;
      let mut tr = 0.0;
      for i in 0..s2.len() {
        let p = 1.0 / (s2[i] + alpha * alpha);
        num += (b[i] * p).powi(2);
        tr += p;
      }
      num / (tr * tr)
    },
    smin,
    smax,
    1e-4,
  )
}

/// GCV choice of the TSVD truncation level (`gcvTsvdMachPrec`).
fn gcv_tsvd(s: &[f64], b: &[f64]) -> f64 {
  let mut idx: Vec<usize> = (0..s.len()).collect();
  idx.sort_by(|&i, &j| s[i].total_cmp(&s[j]));
  let s2: Vec<f64> = idx.iter().rev().map(|&i| s[i]).collect();
  let n = s.len() as f64;
  let mut acc = 0.0;
  let mut g: Vec<f64> = idx
    .iter()
    .enumerate()
    .map(|(j, &i)| {
      acc += b[i] * b[i];
      // Table[(n - k)^2, {k, n - 1, 0, -1}]
      let k = n - 1.0 - j as f64;
      acc / ((n - k) * (n - k))
    })
    .collect();
  if g.len() > 1 {
    g.pop();
    g.reverse();
  }
  // Equal neighboring singular values cannot be told apart.
  for (k, w) in s2.windows(2).enumerate() {
    if w[1] - w[0] == 0.0 && k < g.len() {
      g[k] = f64::MAX;
    }
  }
  let best = g
    .iter()
    .enumerate()
    .fold(0, |best, (k, &v)| if v < g[best] { k } else { best });
  s2[best]
}

/// Wolfram's `isSymmetric[psf, center]`: whether the kernel, centered at
/// `center`, equals its reversal along every set of axes.
fn is_symmetric(psf: &Array, center: &[usize]) -> bool {
  let rank = psf.dims.len();
  let n = &psf.dims;
  // P = PadLeft[A, n + Max[n - 2 center + 1, 0]] then PadRight.
  let left: Vec<usize> = n
    .iter()
    .zip(center)
    .map(|(&n, &c)| (n as isize - 2 * c as isize + 1).max(0) as usize)
    .collect();
  let right: Vec<usize> = n
    .iter()
    .zip(center)
    .map(|(&n, &c)| (2 * c as isize - n as isize - 1).max(0) as usize)
    .collect();
  let dims: Vec<usize> = (0..rank).map(|k| n[k] + left[k] + right[k]).collect();
  let st = strides(&dims);
  let mut p = vec![0.0; dims.iter().product()];
  for (i, &v) in psf.data.iter().enumerate() {
    let idx = unflatten(i, n);
    p[idx
      .iter()
      .zip(&left)
      .zip(&st)
      .map(|((i, l), s)| (i + l) * s)
      .sum::<usize>()] = v;
  }
  // P plus its reversal along every nonempty set of axes.
  let mut ps = p.clone();
  for mask in 1..(1usize << rank) {
    for (i, slot) in ps.iter_mut().enumerate() {
      let idx = unflatten(i, &dims);
      let rev: usize = idx
        .iter()
        .enumerate()
        .map(|(k, &i)| {
          let j = if mask & (1 << k) != 0 {
            dims[k] - 1 - i
          } else {
            i
          };
          j * st[k]
        })
        .sum();
      *slot += p[rev];
    }
  }
  let scale = 2f64.powi(rank as i32);
  let diff: f64 = ps
    .iter()
    .zip(&p)
    .map(|(a, b)| (a / scale - b).powi(2))
    .sum::<f64>()
    .sqrt();
  let norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
  diff <= EPS * norm
}

/// The transformed kernel: either its DFT, or (symmetric kernels) its DCT
/// together with the DCT of a unit impulse.
enum Psf {
  Dft(Vec<C>),
  Dct(Vec<f64>, Vec<f64>),
}

fn construct_psf(psf: &Array, dims: &[usize], check_symmetry: bool) -> Psf {
  let center: Vec<usize> =
    psf.dims.iter().map(|&d| d.div_ceil(2).max(1)).collect();
  let a = pad_right(psf, dims);
  if check_symmetry && is_symmetric(psf, &center) {
    let mut e = vec![0.0; a.len()];
    e[0] = 1.0;
    let e = dct(&e, dims);
    let shifted = dct_shift(&a, &center);
    return Psf::Dct(dct(&shifted, dims), e);
  }
  let shifts: Vec<usize> = center.iter().map(|c| c - 1).collect();
  let rotated: Vec<C> = rotate_left(&a, &shifts)
    .into_iter()
    .map(|v| (v, 0.0))
    .collect();
  Psf::Dft(dft(&rotated, dims, false))
}

/// Wolfram's `dctShift[A, center]`: folds the part of `A` before the center
/// onto the part after it.
fn dct_shift(a: &Array, center: &[usize]) -> Vec<f64> {
  let rank = a.dims.len();
  // P = Take[A, {center, size}] for every axis.
  let pdims: Vec<usize> =
    (0..rank).map(|k| a.dims[k] + 1 - center[k]).collect();
  let ast = strides(&a.dims);
  let pst = strides(&pdims);
  let ptotal: usize = pdims.iter().product();
  let p: Vec<f64> = (0..ptotal)
    .map(|i| {
      let idx = unflatten(i, &pdims);
      a.data[idx
        .iter()
        .zip(center)
        .zip(&ast)
        .map(|((i, c), s)| (i + c - 1) * s)
        .sum::<usize>()]
    })
    .collect();
  let mut l = p.clone();
  // Every nonempty set of axes: Drop the first element along those axes,
  // PadRight back to P's size, and add.
  for mask in 1..(1usize << rank) {
    for (i, slot) in l.iter_mut().enumerate() {
      let idx = unflatten(i, &pdims);
      let src: Option<usize> = idx
        .iter()
        .enumerate()
        .map(|(k, &i)| {
          let j = if mask & (1 << (rank - 1 - k)) != 0 {
            i + 1
          } else {
            i
          };
          (j < pdims[k]).then_some(j * pst[k])
        })
        .sum();
      if let Some(src) = src {
        *slot += p[src];
      }
    }
  }
  let pa = Array {
    dims: pdims,
    data: l,
  };
  pad_right(&pa, &a.dims).data
}

/// The discrete Laplacian stencil of `generateLaplacian[rank]`, embedded in
/// a zero array of `dims` at the origin corner (each axis has length ≥ 3).
fn laplacian(dims: &[usize]) -> Vec<f64> {
  let rank = dims.len();
  let st = strides(dims);
  let mut s = vec![0.0; dims.iter().product()];
  let total = 3usize.pow(rank as u32);
  for i in 0..total {
    let idx = unflatten(i, &vec![3; rank]);
    // Sum over axes of L1[i_k] * Π_{j≠k} Id[i_j], L1 = {1,-2,1},
    // Id = {0,1,0}.
    let mut v = 0.0;
    for k in 0..rank {
      let l1 = [1.0, -2.0, 1.0][idx[k]];
      let others = (0..rank).filter(|&j| j != k).all(|j| idx[j] == 1);
      if others {
        v += l1;
      }
    }
    s[idx.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()] = v;
  }
  s
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
  DampedLs,
  Tikhonov,
  Tsvd,
  Wiener,
}

fn solve(
  psf: &Array,
  b: &Array,
  method: Method,
  param: Option<f64>,
) -> Vec<f64> {
  let dims = &b.dims;
  let n = b.len();
  let a = construct_psf(psf, dims, method != Method::Wiener);
  match a {
    Psf::Dct(a, e) => {
      let a: Vec<f64> = a
        .iter()
        .zip(&e)
        .map(|(a, e)| a / threshold_abs(*e, EPS))
        .collect();
      let x = dct(&b.data, dims);
      let out: Vec<f64> = match method {
        Method::DampedLs => {
          let s = dct(
            &dct_shift(
              &Array {
                dims: dims.clone(),
                data: laplacian(dims),
              },
              &vec![2; dims.len()],
            ),
            dims,
          );
          let s: Vec<f64> = s
            .iter()
            .zip(&e)
            .map(|(s, e)| s / threshold_abs(*e, EPS))
            .collect();
          let alpha = param.unwrap_or_else(|| gcv_gtikhonov(&a, &s, &x));
          (0..n)
            .map(|i| {
              let d =
                threshold_abs(a[i] * a[i] + alpha * alpha * s[i] * s[i], EPS);
              a[i] * x[i] / d
            })
            .collect()
        }
        Method::Tikhonov => {
          let alpha = param.unwrap_or_else(|| gcv_tikhonov(&a, &x));
          (0..n)
            .map(|i| {
              a[i] * x[i] / threshold_abs(a[i] * a[i] + alpha * alpha, EPS)
            })
            .collect()
        }
        Method::Tsvd => {
          let a: Vec<f64> = a.iter().map(|v| v.abs()).collect();
          let xa: Vec<f64> = x.iter().map(|v| v.abs()).collect();
          let alpha = param.unwrap_or_else(|| gcv_tsvd(&a, &xa)).max(EPS);
          (0..n)
            .map(|i| if a[i] > alpha { x[i] / a[i] } else { 0.0 })
            .collect()
        }
        Method::Wiener => unreachable!("Wiener never uses the DCT"),
      };
      idct(&out, dims)
    }
    Psf::Dft(a) => {
      let bc: Vec<C> = b.data.iter().map(|&v| (v, 0.0)).collect();
      let x = dft(&bc, dims, false);
      let filtered: Vec<C> = match method {
        Method::DampedLs => {
          let s0 = Array {
            dims: dims.clone(),
            data: laplacian(dims),
          };
          let s0 = rotate_left(&s0, &vec![1; dims.len()]);
          let s0: Vec<C> = s0.into_iter().map(|v| (v, 0.0)).collect();
          let s = dft(&s0, dims, false);
          let alpha = param.unwrap_or_else(|| {
            let sa: Vec<f64> = a.iter().map(|z| norm2(*z).sqrt()).collect();
            let sd: Vec<f64> = s.iter().map(|z| norm2(*z).sqrt()).collect();
            let xb: Vec<f64> = x.iter().map(|z| norm2(*z).sqrt()).collect();
            gcv_gtikhonov(&sa, &sd, &xb)
          });
          (0..n)
            .map(|i| {
              let d =
                threshold_abs(norm2(a[i]) + alpha * alpha * norm2(s[i]), EPS);
              conj_mul_div(a[i], x[i], d)
            })
            .collect()
        }
        Method::Tikhonov => {
          let alpha = param.unwrap_or_else(|| {
            let sa: Vec<f64> = a.iter().map(|z| norm2(*z).sqrt()).collect();
            let xb: Vec<f64> = x.iter().map(|z| norm2(*z).sqrt()).collect();
            gcv_tikhonov(&sa, &xb)
          });
          (0..n)
            .map(|i| {
              let d = threshold_abs(norm2(a[i]) + alpha * alpha, EPS);
              conj_mul_div(a[i], x[i], d)
            })
            .collect()
        }
        Method::Tsvd => {
          let sa: Vec<f64> = a.iter().map(|z| norm2(*z).sqrt()).collect();
          let alpha = param
            .unwrap_or_else(|| {
              let xb: Vec<f64> = x.iter().map(|z| norm2(*z).sqrt()).collect();
              gcv_tsvd(&sa, &xb)
            })
            .max(EPS);
          (0..n)
            .map(|i| {
              if sa[i] > alpha {
                (x[i].0 / sa[i], x[i].1 / sa[i])
              } else {
                (0.0, 0.0)
              }
            })
            .collect()
        }
        Method::Wiener => {
          let alpha = param.unwrap_or(0.01);
          (0..n)
            .map(|i| {
              let (ar, ai) = a[i];
              // S = Conjugate[A]/(A Conjugate[A] + alpha), or 1/A when
              // alpha is zero.
              let s = if alpha == 0.0 {
                let m = norm2(a[i]);
                if m == 0.0 {
                  (0.0, 0.0)
                } else {
                  (ar / m, -ai / m)
                }
              } else {
                let d = norm2(a[i]) + alpha;
                (ar / d, -ai / d)
              };
              let (xr, xi) = x[i];
              (xr * s.0 - xi * s.1, xr * s.1 + xi * s.0)
            })
            .collect()
        }
      };
      dft(&filtered, dims, true)
        .into_iter()
        .map(|z| z.0)
        .collect()
    }
  }
}

/// `Conjugate[a] * x / d`.
fn conj_mul_div(a: C, x: C, d: f64) -> C {
  ((a.0 * x.0 + a.1 * x.1) / d, (a.0 * x.1 - a.1 * x.0) / d)
}

fn build(data: &[f64], dims: &[usize]) -> Expr {
  match dims {
    [] => Expr::Real(data[0]),
    [n, rest @ ..] => {
      let sub: usize = rest.iter().product();
      Expr::List(
        (0..*n)
          .map(|i| build(&data[i * sub..(i + 1) * sub], rest))
          .collect(),
      )
    }
  }
}

fn option<'a>(opts: &'a [Expr], key: &str) -> Option<&'a Expr> {
  opts.iter().rev().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == key) => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

/// ListDeconvolve[ker, list, opts…]
pub fn list_deconvolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  deconvolve("ListDeconvolve", args)
}

/// The method and regularization parameter a deconvolution runs with, as
/// given by its `Method` option.
#[derive(Clone, Copy)]
pub(crate) struct Settings {
  method: Method,
  param: Option<f64>,
}

/// Reads the `Method` option (`"DampedLS"` by default, or `{name, param}`).
/// `None` when it is invalid (after a `head::mtd`/`head::param` message) or
/// names an iterative method, which is not implemented.
pub(crate) fn method_settings(head: &str, opts: &[Expr]) -> Option<Settings> {
  let msg = |text: String| crate::emit_message(&format!("{head}::{text}"));
  // ImageDeconvolve words its messages slightly differently.
  let image = head == "ImageDeconvolve";
  let (name, sub) = match option(opts, "Method") {
    None => (Expr::String("DampedLS".into()), Vec::new()),
    Some(Expr::List(items)) if !items.is_empty() => {
      (items[0].clone(), items[1..].to_vec())
    }
    Some(other) => (other.clone(), Vec::new()),
  };
  let method = match &name {
    Expr::String(s) => match s.as_str() {
      "DampedLS" => Some(Method::DampedLs),
      "Tikhonov" => Some(Method::Tikhonov),
      "TSVD" => Some(Method::Tsvd),
      "Wiener" => Some(Method::Wiener),
      // Iterative methods: not implemented.
      "Hybrid" | "RichardsonLucy" | "SteepestDescent" | "TotalVariation" => {
        return None;
      }
      _ => None,
    },
    _ => None,
  };
  let Some(method) = method else {
    msg(format!(
      "mtd: The value of the option Method -> {} should be one of \"DampedLS\", \"Tikhonov\", \"TSVD\", \"Wiener\", \"Hybrid\", \"RichardsonLucy\", \"SteepestDescent\" or \"TotalVariation\"{} with possible suboptions.",
      expr_to_string(&name),
      if image { "," } else { "" }
    ));
    return None;
  };
  let param = match sub.first() {
    None => None,
    Some(Expr::Identifier(s)) if s == "Automatic" => None,
    Some(p) => match crate::functions::math_ast::try_eval_to_f64(p) {
      Some(v) if v >= 0.0 => Some(v),
      _ => {
        msg(format!(
          "param: The value of regularization parameter {} should be a non-negative real number{} or Automatic.",
          expr_to_string(p),
          if image {
            ", a vector of such numbers (for multi-channel images)"
          } else {
            ""
          }
        ));
        return None;
      }
    },
  };

  Some(Settings { method, param })
}

/// Deconvolves `data` (dims `data_dims`, every axis at least 3 long) by
/// `psf` (odd or even dims, no larger than the data's) with the spectral
/// core shared by `ListDeconvolve` and `ImageDeconvolve`.
pub(crate) fn deconvolve_array(
  psf_dims: &[usize],
  psf: &[f64],
  data_dims: &[usize],
  data: &[f64],
  settings: Settings,
) -> Vec<f64> {
  let psf = Array {
    dims: psf_dims.to_vec(),
    data: psf.to_vec(),
  };
  let b = Array {
    dims: data_dims.to_vec(),
    data: data.to_vec(),
  };
  solve(&psf, &b, settings.method, settings.param)
}

fn deconvolve(head: &str, args: &[Expr]) -> Result<Expr, InterpreterError> {
  let fail = || Ok(unevaluated(head, args));
  let msg = |text: String| crate::emit_message(&format!("{head}::{text}"));
  // Options beyond the two arguments: rules or lists of rules.
  let mut opts: Vec<Expr> = Vec::new();
  for o in &args[2..] {
    match o {
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => opts.push(o.clone()),
      Expr::List(items)
        if items.iter().all(|r| {
          matches!(r, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        }) =>
      {
        opts.extend(items.iter().cloned());
      }
      other => {
        msg(format!(
          "nonopt: Options expected (instead of {}) beyond position 2 in {}. An option must be a rule or a list of rules.",
          expr_to_string(other),
          expr_to_string(&unevaluated(head, args))
        ));
        return fail();
      }
    }
  }
  let Some(psf) = parse_real_array(&args[0]) else {
    msg(format!(
      "arginv: Expecting a nonempty real numeric array instead of {}.",
      expr_to_string(&args[0])
    ));
    return fail();
  };
  let Some(mut data) = parse_real_array(&args[1]) else {
    msg(format!(
      "arginv: Expecting a nonempty real numeric array instead of {}.",
      expr_to_string(&args[1])
    ));
    return fail();
  };
  if psf.dims.len() != data.dims.len() {
    msg(format!(
      "dims: The kernel {} and list {} are not both nonempty numerical arrays with the same tensor rank.",
      expr_to_string(&args[0]),
      expr_to_string(&args[1])
    ));
    return fail();
  }
  if psf.dims.iter().zip(&data.dims).any(|(k, d)| k > d) {
    msg(
      "kldims: The dimensions of a kernel cannot be larger than the dimensions of a list."
        .to_string(),
    );
    return fail();
  }
  if let Some(p) = option(&opts, "Padding") {
    let ok = match p {
      Expr::Identifier(s) => s == "Automatic",
      Expr::String(s) => {
        matches!(s.as_str(), "Periodic" | "Fixed" | "Reflected" | "Reversed")
      }
      other => crate::functions::math_ast::try_eval_to_f64(other).is_some(),
    };
    if !ok {
      msg(format!(
        "padding: The value of the option Padding -> {} should be one of \"Fixed\", \"Periodic\", \"Reversed\", \"Reflected\" or c, where c is any machine-sized number.",
        expr_to_string(p)
      ));
      return fail();
    }
  }
  if let Some(it) = option(&opts, "MaxIterations") {
    let ok = match it {
      Expr::Identifier(s) => s == "Automatic",
      Expr::Integer(n) => *n > 0,
      _ => false,
    };
    if !ok {
      msg(format!(
        "iters: The value of the option MaxIterations -> {} should be a positive integer.",
        expr_to_string(it)
      ));
      return fail();
    }
  }
  let Some(Settings { method, param }) = method_settings(head, &opts) else {
    return fail();
  };

  let max_data = data.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
  let scaled = max_data != 1.0 && max_data != 0.0;
  if scaled {
    data.data.iter_mut().for_each(|v| *v /= max_data);
  }
  // Kernel axes of even length get a trailing zero.
  let psf_dims: Vec<usize> = psf
    .dims
    .iter()
    .map(|&d| if d % 2 == 0 { d + 1 } else { d })
    .collect();
  let psf = pad_right(&psf, &psf_dims);
  let padded = data.dims.iter().any(|&d| d < 3);
  let b = if padded {
    pad_reversed(&data)
  } else {
    Array {
      dims: data.dims.clone(),
      data: data.data.clone(),
    }
  };
  let x = solve(&psf, &b, method, param);
  let mut res = if padded {
    crop_one(&x, &b.dims)
  } else {
    Array {
      dims: b.dims.clone(),
      data: x,
    }
  };
  if scaled {
    res.data.iter_mut().for_each(|v| *v *= max_data);
  }
  Ok(build(&res.data, &res.dims))
}
