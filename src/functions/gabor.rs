//! `GaborMatrix` and `GaborFilter`: the Gabor kernel of radius `r` and wave
//! vector `k`, a Gaussian of standard deviation `σ` (default `r/2`)
//! normalized to unit sum, times `Cos[k . x - φ]`, over the offsets
//! `x ∈ {-n, …, n}^d` with `n = Floor[r + 1/2]`; and the convolution of an
//! array or image with it, the edges padded with their nearest value.

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::unevaluated;
use crate::syntax::Expr;

/// Half-width and standard deviation from `r` or `{r, σ}`.
fn radius_spec(spec: &Expr) -> Option<(usize, f64)> {
  let (r, sigma) = match spec {
    Expr::List(items) if items.len() == 2 => {
      (try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?)
    }
    other => {
      let r = try_eval_to_f64(other)?;
      (r, r / 2.0)
    }
  };
  if !(r >= 0.0 && sigma > 0.0) {
    return None;
  }
  Some(((r + 0.5).floor() as usize, sigma))
}

fn wave_vector(k: &Expr) -> Option<Vec<f64>> {
  let Expr::List(items) = k else { return None };
  if items.is_empty() || items.len() > 2 {
    return None;
  }
  items.iter().map(try_eval_to_f64).collect()
}

/// The kernel as a flat row-major array of side `2 n + 1` in `k.len()`
/// dimensions.
fn kernel(n: usize, sigma: f64, k: &[f64], phi: f64) -> Vec<f64> {
  let side = 2 * n + 1;
  let count = side.pow(k.len() as u32);
  let offsets = |mut flat: usize| -> Vec<f64> {
    let mut x = vec![0.0; k.len()];
    for d in (0..k.len()).rev() {
      x[d] = (flat % side) as f64 - n as f64;
      flat /= side;
    }
    x
  };
  let gauss: Vec<f64> = (0..count)
    .map(|i| {
      let x = offsets(i);
      (-x.iter().map(|v| v * v).sum::<f64>() / (2.0 * sigma * sigma)).exp()
    })
    .collect();
  let total: f64 = gauss.iter().sum();
  (0..count)
    .map(|i| {
      let x = offsets(i);
      let phase: f64 = x.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() - phi;
      gauss[i] / total * phase.cos()
    })
    .collect()
}

fn nest(values: &[f64], dims: &[usize]) -> Expr {
  if dims.len() <= 1 {
    return Expr::List(values.iter().map(|&v| Expr::Real(v)).collect());
  }
  let stride: usize = dims[1..].iter().product();
  Expr::List(
    values
      .chunks(stride)
      .map(|chunk| nest(chunk, &dims[1..]))
      .collect(),
  )
}

fn phase_arg(args: &[Expr], i: usize) -> Option<f64> {
  match args.get(i) {
    None => Some(0.0),
    Some(e) => try_eval_to_f64(e),
  }
}

/// `GaborMatrix[r, k]`, `GaborMatrix[{r, σ}, k, φ]`.
pub fn gabor_matrix_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Ok(unevaluated("GaborMatrix", args));
  if !(2..=3).contains(&args.len()) {
    return original();
  }
  let (Some((n, sigma)), Some(k), Some(phi)) = (
    radius_spec(&args[0]),
    wave_vector(&args[1]),
    phase_arg(args, 2),
  ) else {
    return original();
  };
  let side = 2 * n + 1;
  Ok(nest(&kernel(n, sigma, &k, phi), &vec![side; k.len()]))
}

/// A rectangular numeric array as its dimensions and row-major values.
fn numeric_array(e: &Expr) -> Option<(Vec<usize>, Vec<f64>)> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      if items.iter().all(|i| !matches!(i, Expr::List(_))) {
        let values: Option<Vec<f64>> =
          items.iter().map(try_eval_to_f64).collect();
        return Some((vec![items.len()], values?));
      }
      let mut dims = None;
      let mut values = Vec::new();
      for item in items {
        let (d, v) = numeric_array(item)?;
        if dims.as_ref().is_some_and(|prev| *prev != d) {
          return None;
        }
        dims = Some(d);
        values.extend(v);
      }
      let mut all = vec![items.len()];
      all.extend(dims?);
      Some((all, values))
    }
    _ => None,
  }
}

/// Convolve the leading `rank` dimensions of `values` (any trailing
/// dimensions, such as color channels, are filtered separately) with the
/// kernel, padding with the nearest edge value.
fn convolve(
  dims: &[usize],
  values: &[f64],
  rank: usize,
  n: usize,
  kern: &[f64],
) -> Vec<f64> {
  let side = 2 * n + 1;
  let inner: usize = dims[rank..].iter().product();
  let outer = &dims[..rank];
  let points: usize = outer.iter().product();
  let mut out = vec![0.0; values.len()];
  for p in 0..points {
    // The position of this point.
    let mut pos = vec![0usize; rank];
    let mut rest = p;
    for d in (0..rank).rev() {
      pos[d] = rest % outer[d];
      rest /= outer[d];
    }
    for (ki, &w) in kern.iter().enumerate() {
      // Convolution: the kernel offset x reads the data at pos - x.
      let mut flat = 0usize;
      let mut kr = ki;
      let mut offs = vec![0isize; rank];
      for d in (0..rank).rev() {
        offs[d] = (kr % side) as isize - n as isize;
        kr /= side;
      }
      for d in 0..rank {
        let q = (pos[d] as isize - offs[d]).clamp(0, outer[d] as isize - 1);
        flat = flat * outer[d] + q as usize;
      }
      for c in 0..inner {
        out[p * inner + c] += w * values[flat * inner + c];
      }
    }
  }
  out
}

/// `GaborFilter[data, r, k]`, `GaborFilter[data, {r, σ}, k, φ]` for a
/// numeric array whose rank is the length of `k`, or an image.
pub fn gabor_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || Ok(unevaluated("GaborFilter", args));
  if !(3..=4).contains(&args.len()) {
    return original();
  }
  let (Some((n, sigma)), Some(k), Some(phi)) = (
    radius_spec(&args[1]),
    wave_vector(&args[2]),
    phase_arg(args, 3),
  ) else {
    return original();
  };
  let kern = kernel(n, sigma, &k, phi);
  let is_image = matches!(&args[0], Expr::Image { .. });
  let data = if is_image {
    crate::evaluator::evaluate_function_call_ast(
      "ImageData",
      std::slice::from_ref(&args[0]),
    )?
  } else {
    args[0].clone()
  };
  let Some((dims, values)) = numeric_array(&data) else {
    return original();
  };
  if dims.len() < k.len() || (!is_image && dims.len() != k.len()) {
    return original();
  }
  let filtered = nest(&convolve(&dims, &values, k.len(), n, &kern), &dims);
  if is_image {
    return crate::evaluator::evaluate_function_call_ast("Image", &[filtered]);
  }
  Ok(filtered)
}
