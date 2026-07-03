//! Noise-removal filters: WienerFilter and TotalVariationFilter for 1D/2D
//! data and Audio objects, plus the Audio paths of LowpassFilter and
//! MeanFilter.

use super::{AudioData, make_audio, parse_audio, unevaluated};
use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// Numeric input for the data filters: a 1D list (one row) or a 2D
/// rectangular list of lists. The bool is true for 2D input.
fn numeric_rows(expr: &Expr) -> Option<(Vec<Vec<f64>>, bool)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
  if items.iter().all(|i| matches!(i, Expr::List(_))) {
    let mut rows = Vec::with_capacity(items.len());
    for item in items.iter() {
      let Expr::List(cols) = item else {
        unreachable!()
      };
      let mut row = Vec::with_capacity(cols.len());
      for c in cols.iter() {
        row.push(try_eval_to_f64(c)?);
      }
      rows.push(row);
    }
    let w = rows[0].len();
    if w == 0 || rows.iter().any(|r| r.len() != w) {
      return None;
    }
    Some((rows, true))
  } else {
    let mut row = Vec::with_capacity(items.len());
    for item in items.iter() {
      row.push(try_eval_to_f64(item)?);
    }
    Some((vec![row], false))
  }
}

/// Rebuild the result with the input's shape.
fn rows_to_expr(rows: Vec<Vec<f64>>, is_2d: bool) -> Expr {
  let row_expr = |r: Vec<f64>| {
    Expr::List(r.into_iter().map(Expr::Real).collect::<Vec<_>>().into())
  };
  if is_2d {
    Expr::List(rows.into_iter().map(row_expr).collect::<Vec<_>>().into())
  } else {
    row_expr(rows.into_iter().next().unwrap())
  }
}

// ---------------------------------------------------------------------------
// WienerFilter
// ---------------------------------------------------------------------------

/// Local means and variances over a clipped neighborhood of radius `r`
/// around each element of a matrix (square neighborhood for 2D input,
/// pass the data as one row for 1D).
fn local_stats(
  rows: &[Vec<f64>],
  ry: usize,
  rx: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
  let h = rows.len();
  let w = rows[0].len();
  let mut means = vec![vec![0.0; w]; h];
  let mut vars = vec![vec![0.0; w]; h];
  for y in 0..h {
    for x in 0..w {
      let y0 = y.saturating_sub(ry);
      let y1 = (y + ry + 1).min(h);
      let x0 = x.saturating_sub(rx);
      let x1 = (x + rx + 1).min(w);
      let count = ((y1 - y0) * (x1 - x0)) as f64;
      let mut sum = 0.0;
      let mut sum_sq = 0.0;
      for row in rows.iter().take(y1).skip(y0) {
        for v in row.iter().take(x1).skip(x0) {
          sum += v;
          sum_sq += v * v;
        }
      }
      let mean = sum / count;
      means[y][x] = mean;
      vars[y][x] = (sum_sq / count - mean * mean).max(0.0);
    }
  }
  (means, vars)
}

/// Core of the Wiener deconvolution smoother: each element becomes
/// μ + max(0, σ² − ν)/σ² (x − μ), where μ/σ² are the local mean/variance
/// and ν the noise power (estimated as the mean local variance when not
/// given).
fn wiener_rows(
  rows: &[Vec<f64>],
  ry: usize,
  rx: usize,
  noise: Option<f64>,
) -> Vec<Vec<f64>> {
  let (means, vars) = local_stats(rows, ry, rx);
  let noise = noise.unwrap_or_else(|| {
    let count = (vars.len() * vars[0].len()) as f64;
    vars.iter().flatten().sum::<f64>() / count
  });
  rows
    .iter()
    .enumerate()
    .map(|(y, row)| {
      row
        .iter()
        .enumerate()
        .map(|(x, &v)| {
          let var = vars[y][x];
          if var > 0.0 {
            means[y][x] + ((var - noise).max(0.0) / var) * (v - means[y][x])
          } else {
            means[y][x]
          }
        })
        .collect()
    })
    .collect()
}

/// WienerFilter[data, r] or WienerFilter[data, r, ν] — remove locally
/// stationary noise from 1D/2D data or an Audio object using a local
/// Wiener estimator with neighborhood radius r and noise power ν.
pub fn wiener_filter_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Ok(unevaluated("WienerFilter", args));
  }
  let radius = match try_eval_to_f64(&args[1]) {
    Some(r) if r >= 0.0 => r.round() as usize,
    _ => return Ok(unevaluated("WienerFilter", args)),
  };
  let noise = match args.get(2) {
    Some(e) => match try_eval_to_f64(e) {
      Some(v) => Some(v),
      None => return Ok(unevaluated("WienerFilter", args)),
    },
    None => None,
  };

  if let Some(audio) = parse_audio(&args[0]) {
    let channels = audio
      .channels
      .iter()
      .map(|c| {
        wiener_rows(std::slice::from_ref(c), 0, radius, noise)
          .pop()
          .unwrap()
      })
      .collect();
    return Ok(make_audio(&AudioData {
      channels,
      rate: audio.rate,
    }));
  }
  let Some((rows, is_2d)) = numeric_rows(&args[0]) else {
    return Ok(unevaluated("WienerFilter", args));
  };
  let ry = if is_2d { radius } else { 0 };
  Ok(rows_to_expr(wiener_rows(&rows, ry, radius, noise), is_2d))
}

// ---------------------------------------------------------------------------
// TotalVariationFilter
// ---------------------------------------------------------------------------

/// Rudin–Osher–Fatemi total-variation denoising by gradient descent on
/// min_u Σ|∇u| + 1/(2λ) Σ(u−f)², with a smoothed gradient magnitude.
fn tv_rows(rows: &[Vec<f64>], lambda: f64, is_2d: bool) -> Vec<Vec<f64>> {
  let h = rows.len();
  let w = rows[0].len();
  let eps = 1e-4;
  let tau = 0.05;
  let iterations = 100;
  let mut u: Vec<Vec<f64>> = rows.to_vec();
  let clamp_y = |y: i64| y.clamp(0, h as i64 - 1) as usize;
  let clamp_x = |x: i64| x.clamp(0, w as i64 - 1) as usize;
  for _ in 0..iterations {
    let mut next = u.clone();
    for y in 0..h {
      for x in 0..w {
        let (yi, xi) = (y as i64, x as i64);
        let dx = u[y][clamp_x(xi + 1)] - u[y][x];
        let dxm = u[y][x] - u[y][clamp_x(xi - 1)];
        let (dy, dym) = if is_2d {
          (
            u[clamp_y(yi + 1)][x] - u[y][x],
            u[y][x] - u[clamp_y(yi - 1)][x],
          )
        } else {
          (0.0, 0.0)
        };
        // Divergence of the normalized gradient (smoothed TV flow).
        let div = dx / (dx * dx + eps).sqrt() - dxm / (dxm * dxm + eps).sqrt()
          + dy / (dy * dy + eps).sqrt()
          - dym / (dym * dym + eps).sqrt();
        next[y][x] = u[y][x] + tau * (div - (u[y][x] - rows[y][x]) / lambda);
      }
    }
    u = next;
  }
  u
}

/// TotalVariationFilter[data] or TotalVariationFilter[data, param] —
/// total-variation denoising of 1D/2D data or an Audio object; larger
/// regularization parameters smooth more (default 0.1).
pub fn total_variation_filter_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("TotalVariationFilter", args));
  }
  let lambda = match args.get(1) {
    Some(e) => match try_eval_to_f64(e) {
      Some(v) if v > 0.0 => v,
      _ => return Ok(unevaluated("TotalVariationFilter", args)),
    },
    None => 0.1,
  };
  if let Some(audio) = parse_audio(&args[0]) {
    let channels = audio
      .channels
      .iter()
      .map(|c| {
        tv_rows(std::slice::from_ref(c), lambda, false)
          .pop()
          .unwrap()
      })
      .collect();
    return Ok(make_audio(&AudioData {
      channels,
      rate: audio.rate,
    }));
  }
  let Some((rows, is_2d)) = numeric_rows(&args[0]) else {
    return Ok(unevaluated("TotalVariationFilter", args));
  };
  Ok(rows_to_expr(tv_rows(&rows, lambda, is_2d), is_2d))
}

// ---------------------------------------------------------------------------
// Audio paths of the shared filters
// ---------------------------------------------------------------------------

/// MeanFilter[audio, r] — replace each sample by the mean of the samples in
/// a clipped neighborhood of radius r, per channel. Returns `None` when the
/// first argument is not an audio object (the list/image dispatch
/// continues).
pub fn mean_filter_audio_ast(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 2 || !super::is_audio_expr(&args[0]) {
    return None;
  }
  let audio = parse_audio(&args[0])?;
  let r = match try_eval_to_f64(&args[1]) {
    Some(v) if v >= 0.0 => v.round() as usize,
    _ => return Some(Ok(unevaluated("MeanFilter", args))),
  };
  let channels = audio
    .channels
    .iter()
    .map(|c| {
      let n = c.len();
      (0..n)
        .map(|i| {
          let lo = i.saturating_sub(r);
          let hi = (i + r + 1).min(n);
          c[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect()
    })
    .collect();
  Some(Ok(make_audio(&AudioData {
    channels,
    rate: audio.rate,
  })))
}

/// Kernel length used for LowpassFilter on audio (the list default — a
/// kernel as long as the data — would be quadratic in the sample count).
const AUDIO_FIR_TAPS: usize = 2049;

/// LowpassFilter[audio, ωc] — lowpass-filter each channel with the cutoff
/// ωc in radians per second (or Quantity[f, "Hertz"]). Delegates to the
/// windowed-sinc list implementation with the audio's sample rate. Returns
/// `None` when the first argument is not an audio object.
pub fn lowpass_filter_audio_ast(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() < 2 || !super::is_audio_expr(&args[0]) {
    return None;
  }
  let audio = parse_audio(&args[0])?;
  // Cutoff: radians per second, or a frequency quantity (ω = 2πf).
  let omega = (match &args[1] {
    Expr::FunctionCall { name, args: qargs }
      if name == "Quantity" && qargs.len() == 2 =>
    {
      let mag = try_eval_to_f64(&qargs[0])?;
      match &qargs[1] {
        Expr::String(u) | Expr::Identifier(u) if u == "Hertz" => {
          Some(2.0 * std::f64::consts::PI * mag)
        }
        _ => None,
      }
    }
    e => try_eval_to_f64(e),
  })?;
  if !(omega.is_finite() && omega > 0.0) {
    return Some(Ok(unevaluated("LowpassFilter", args)));
  }

  let taps = AUDIO_FIR_TAPS.min(audio.len());
  let mut channels = Vec::with_capacity(audio.channels.len());
  for ch in &audio.channels {
    let list =
      Expr::List(ch.iter().map(|&s| Expr::Real(s)).collect::<Vec<_>>().into());
    let call = crate::functions::math_ast::lowpass_filter_ast(&[
      list,
      Expr::Real(omega),
      Expr::Integer(taps as i128),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("SampleRate".to_string())),
        replacement: Box::new(Expr::Real(audio.rate)),
      },
    ]);
    match call {
      Ok(Expr::List(ref items)) => {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
          out.push(try_eval_to_f64(item)?);
        }
        channels.push(out);
      }
      Ok(_) => return Some(Ok(unevaluated("LowpassFilter", args))),
      Err(e) => return Some(Err(e)),
    }
  }
  Some(Ok(make_audio(&AudioData {
    channels,
    rate: audio.rate,
  })))
}
