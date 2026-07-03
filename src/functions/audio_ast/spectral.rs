//! Spectral analysis: ShortTimeFourier (and the ShortTimeFourierData
//! object), Spectrogram, Cepstrogram, and Periodogram.

use base64::Engine;

use super::{parse_audio, unevaluated};
use crate::InterpreterError;
use crate::functions::math_ast::numerical::{
  fft_pow2_in_place, fourier_result_to_expr,
};
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// FFT of a real frame zero-padded to the next power of two. Returns the
/// complex bins with Fourier's default 1/√n normalization (n = padded
/// length).
fn fft_padded(frame: &[f64]) -> Vec<(f64, f64)> {
  let n = frame.len().next_power_of_two().max(1);
  let mut data: Vec<(f64, f64)> = frame.iter().map(|&x| (x, 0.0)).collect();
  data.resize(n, (0.0, 0.0));
  fft_pow2_in_place(&mut data, -1.0);
  let scale = 1.0 / (n as f64).sqrt();
  for v in &mut data {
    v.0 *= scale;
    v.1 *= scale;
  }
  data
}

/// One-sided power spectrum of a signal: (frequencies in Hz, |X|²) for the
/// bins from DC up to the Nyquist frequency. The signal is zero-padded to a
/// power of two.
pub fn power_spectrum(xs: &[f64], rate: f64) -> (Vec<f64>, Vec<f64>) {
  let bins = fft_padded(xs);
  let n = bins.len();
  let half = n / 2;
  let freqs: Vec<f64> =
    (0..=half).map(|k| k as f64 * rate / n as f64).collect();
  let powers: Vec<f64> = bins[..=half]
    .iter()
    .map(|&(re, im)| re * re + im * im)
    .collect();
  (freqs, powers)
}

/// Numeric data for the spectral functions: an Audio/Sound object, or a
/// plain list of real samples (treated as sampled at 1 Hz, i.e. axes in
/// samples and frequency bins).
fn spectral_input(expr: &Expr) -> Option<(Vec<f64>, f64, bool)> {
  if let Some(audio) = parse_audio(expr) {
    return Some((audio.mixdown(), audio.rate, true));
  }
  if let Expr::List(items) = expr {
    let mut xs = Vec::with_capacity(items.len());
    for item in items.iter() {
      xs.push(try_eval_to_f64(item)?);
    }
    if xs.is_empty() {
      return None;
    }
    return Some((xs, 1.0, false));
  }
  None
}

/// Default STFT partition for a signal of n samples, per the Wolfram
/// documentation: window size m = 2^⌈log₂ √n⌉, offset ⌈m/3⌉.
pub fn default_partition(n: usize) -> (usize, usize) {
  let m = 2usize
    .pow((n as f64).sqrt().log2().ceil().max(0.0) as u32)
    .max(1);
  (m, m.div_ceil(3))
}

/// Parse optional partition arguments `…, n, d, …` following the data
/// argument: integers or time Quantities (seconds × sample rate).
fn partition_args(
  args: &[Expr],
  rate: f64,
  n: usize,
) -> Option<(usize, usize)> {
  let to_samples = |e: &Expr| -> Option<usize> {
    if let Expr::Integer(v) = e {
      return (*v > 0).then_some(*v as usize);
    }
    super::time_to_seconds(e)
      .filter(|t| *t > 0.0)
      .map(|t| ((t * rate).round() as usize).max(1))
  };
  let (dm, dd) = default_partition(n);
  let positional: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .collect();
  match positional.len() {
    0 => Some((dm, dd)),
    1 => {
      let m = to_samples(positional[0])?;
      Some((m, m.div_ceil(3)))
    }
    2 => Some((to_samples(positional[0])?, to_samples(positional[1])?)),
    _ => None,
  }
}

/// Split a signal into frames of length `m` with offset `o` (the trailing
/// partial frame is dropped; shorter signals give one zero-padded frame).
fn frames_of(xs: &[f64], m: usize, o: usize) -> Vec<Vec<f64>> {
  let n = xs.len();
  if n < m {
    let mut frame = xs.to_vec();
    frame.resize(m, 0.0);
    return vec![frame];
  }
  (0..=(n - m))
    .step_by(o.max(1))
    .map(|start| xs[start..start + m].to_vec())
    .collect()
}

/// Apply a Hann window to a frame in place. Used by the display plots
/// (Spectrogram, Cepstrogram) to suppress spectral leakage; the
/// ShortTimeFourier data function keeps the rectangular (Dirichlet)
/// default.
fn hann_window(frame: &mut [f64]) {
  let n = frame.len();
  if n < 2 {
    return;
  }
  for (i, x) in frame.iter_mut().enumerate() {
    let w = 0.5
      * (1.0
        - (2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0)).cos());
    *x *= w;
  }
}

// ---------------------------------------------------------------------------
// ShortTimeFourier
// ---------------------------------------------------------------------------

/// ShortTimeFourier[data] / ShortTimeFourier[data, n] /
/// ShortTimeFourier[data, n, d] — the short-time Fourier transform, as a
/// ShortTimeFourierData[frames, sampleRate, windowSize, offset] object.
/// Multichannel audio is mixed down to mono first.
pub fn short_time_fourier_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated("ShortTimeFourier", args));
  }
  let Some((xs, rate, _)) = spectral_input(&args[0]) else {
    return Ok(unevaluated("ShortTimeFourier", args));
  };
  let Some((m, o)) = partition_args(&args[1..], rate, xs.len()) else {
    return Ok(unevaluated("ShortTimeFourier", args));
  };
  // Frames are FFT'd at the window size (rounded up to a power of two).
  let frames: Vec<Expr> = frames_of(&xs, m, o)
    .iter()
    .map(|frame| {
      Expr::List(
        fft_padded(frame)
          .into_iter()
          .map(|(re, im)| fourier_result_to_expr(re, im, false))
          .collect::<Vec<_>>()
          .into(),
      )
    })
    .collect();
  let rate_expr = if rate.fract() == 0.0 {
    Expr::Integer(rate as i128)
  } else {
    Expr::Real(rate)
  };
  Ok(Expr::FunctionCall {
    name: "ShortTimeFourierData".to_string(),
    args: vec![
      Expr::List(frames.into()),
      rate_expr,
      Expr::Integer(m as i128),
      Expr::Integer(o as i128),
    ]
    .into(),
  })
}

/// Property access on a ShortTimeFourierData object:
/// `stfd["Data"]`, `stfd["SampleRate"]`, `stfd["WindowSize"]`,
/// `stfd["Offset"]`, `stfd["NumberOfFrames"]`, `stfd["Frequencies"]`,
/// `stfd["Times"]`, `stfd["Properties"]`. Returns `None` for anything that
/// is not such a property lookup.
pub fn stfd_property(head: &Expr, prop: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = head else {
    return None;
  };
  if name != "ShortTimeFourierData" || args.len() != 4 {
    return None;
  }
  let Expr::String(prop) = prop else {
    return None;
  };
  let frame_count = match &args[0] {
    Expr::List(frames) => frames.len(),
    _ => return None,
  };
  let rate = try_eval_to_f64(&args[1])?;
  let m = try_eval_to_f64(&args[2])? as usize;
  let o = try_eval_to_f64(&args[3])? as usize;
  let padded = m.next_power_of_two().max(1);
  match prop.as_str() {
    "Data" => Some(args[0].clone()),
    "SampleRate" => Some(args[1].clone()),
    "WindowSize" => Some(args[2].clone()),
    "Offset" => Some(args[3].clone()),
    "NumberOfFrames" => Some(Expr::Integer(frame_count as i128)),
    "Frequencies" => Some(Expr::List(
      (0..padded)
        .map(|k| Expr::Real(k as f64 * rate / padded as f64))
        .collect::<Vec<_>>()
        .into(),
    )),
    "Times" => Some(Expr::List(
      (0..frame_count)
        .map(|f| Expr::Real((f * o) as f64 / rate + m as f64 / rate / 2.0))
        .collect::<Vec<_>>()
        .into(),
    )),
    "Properties" => Some(Expr::List(
      [
        "Data",
        "Frequencies",
        "NumberOfFrames",
        "Offset",
        "SampleRate",
        "Times",
        "WindowSize",
      ]
      .iter()
      .map(|s| Expr::String((*s).to_string()))
      .collect::<Vec<_>>()
      .into(),
    )),
    _ => None,
  }
}

// ---------------------------------------------------------------------------
// Raster plots (Spectrogram, Cepstrogram)
// ---------------------------------------------------------------------------

/// Choose a "nice" tick step so that the range yields roughly `target`
/// ticks.
fn nice_step(range: f64, target: f64) -> f64 {
  let raw = range / target;
  let mag = 10f64.powf(raw.log10().floor());
  let norm = raw / mag;
  let step = if norm <= 1.5 {
    1.0
  } else if norm <= 3.5 {
    2.0
  } else if norm <= 7.5 {
    5.0
  } else {
    10.0
  };
  step * mag
}

/// Format an axis tick label compactly.
fn tick_label(v: f64) -> String {
  if v == 0.0 {
    return "0".to_string();
  }
  if v.fract() == 0.0 && v.abs() < 1e7 {
    return format!("{}", v as i64);
  }
  let s = format!("{v:.4}");
  s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Render an intensity matrix (rows bottom-to-top, `matrix[row][col]`,
/// values in [0, 1] where 1 is the most intense) as a framed SVG plot with
/// the raster embedded as a base64 PNG. Intensity is drawn white → black on
/// the light theme background.
fn raster_svg(
  matrix: &[Vec<f64>],
  x_max: f64,
  y_max: f64,
  x_label: &str,
  y_label: &str,
) -> String {
  let rows = matrix.len();
  let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
  let dark = crate::is_dark_mode();

  // PNG rows run top-to-bottom; our matrix rows run bottom-to-top.
  let mut img = image::RgbImage::new(cols.max(1) as u32, rows.max(1) as u32);
  for (row_idx, row) in matrix.iter().enumerate() {
    for (col_idx, &v) in row.iter().enumerate() {
      let v = v.clamp(0.0, 1.0);
      let level = if dark {
        (v * 255.0).round() as u8
      } else {
        ((1.0 - v) * 255.0).round() as u8
      };
      img.put_pixel(
        col_idx as u32,
        (rows - 1 - row_idx) as u32,
        image::Rgb([level, level, level]),
      );
    }
  }
  let mut png = Vec::new();
  image::DynamicImage::ImageRgb8(img)
    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
    .ok();
  let png_b64 = base64::engine::general_purpose::STANDARD.encode(&png);

  let (w, h) = (400.0, 250.0);
  let (left, right, top, bottom) = (52.0, 10.0, 10.0, 34.0);
  let (pw, ph) = (w - left - right, h - top - bottom);
  let (bg, axis, _, label_fill, _) = crate::functions::plot::plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  let frame = format!("rgb({},{},{})", axis.0, axis.1, axis.2);

  let mut svg = format!(
    "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n\
     <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n\
     <image x=\"{left}\" y=\"{top}\" width=\"{pw}\" height=\"{ph}\" \
     preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" \
     href=\"data:image/png;base64,{png_b64}\"/>\n\
     <rect x=\"{left}\" y=\"{top}\" width=\"{pw}\" height=\"{ph}\" \
     fill=\"none\" stroke=\"{frame}\" stroke-width=\"1\"/>\n"
  );

  let font = "font-family=\"sans-serif\" font-size=\"10\"";
  // X ticks
  if x_max > 0.0 {
    let step = nice_step(x_max, 5.0);
    let mut t = 0.0;
    while t <= x_max * 1.0001 {
      let x = left + t / x_max * pw;
      let y0 = top + ph;
      svg.push_str(&format!(
        "<line x1=\"{x}\" y1=\"{y0}\" x2=\"{x}\" y2=\"{}\" stroke=\"{frame}\" stroke-width=\"1\"/>\n\
         <text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" fill=\"{label_fill}\" {font}>{}</text>\n",
        y0 + 4.0,
        y0 + 14.0,
        tick_label(t)
      ));
      t += step;
    }
  }
  // Y ticks
  if y_max > 0.0 {
    let step = nice_step(y_max, 5.0);
    let mut t = 0.0;
    while t <= y_max * 1.0001 {
      let y = top + ph - t / y_max * ph;
      svg.push_str(&format!(
        "<line x1=\"{}\" y1=\"{y}\" x2=\"{left}\" y2=\"{y}\" stroke=\"{frame}\" stroke-width=\"1\"/>\n\
         <text x=\"{}\" y=\"{}\" text-anchor=\"end\" fill=\"{label_fill}\" {font}>{}</text>\n",
        left - 4.0,
        left - 6.0,
        y + 3.5,
        tick_label(t)
      ));
      t += step;
    }
  }
  if !x_label.is_empty() {
    svg.push_str(&format!(
      "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" fill=\"{label_fill}\" {font}>{x_label}</text>\n",
      left + pw / 2.0,
      h - 6.0
    ));
  }
  if !y_label.is_empty() {
    svg.push_str(&format!(
      "<text x=\"12\" y=\"{}\" text-anchor=\"middle\" fill=\"{label_fill}\" {font} \
       transform=\"rotate(-90 12 {})\">{y_label}</text>\n",
      top + ph / 2.0,
      top + ph / 2.0
    ));
  }
  svg.push_str("</svg>");
  svg
}

/// Map spectral magnitudes to display intensities in [0, 1] on a dB scale
/// with a 60 dB floor below the maximum.
fn db_intensity(mag: f64, max_mag: f64) -> f64 {
  if mag <= 0.0 || max_mag <= 0.0 {
    return 0.0;
  }
  let db = 20.0 * (mag / max_mag).log10();
  ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

/// Spectrogram[data] / Spectrogram[data, n] / Spectrogram[data, n, d] —
/// plot the magnitude of the short-time Fourier transform: time (seconds
/// for audio, samples for lists) against frequency (Hz for audio, bins for
/// lists), darker where the magnitude is larger.
pub fn spectrogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated("Spectrogram", args));
  }
  let Some((xs, rate, is_audio)) = spectral_input(&args[0]) else {
    return Ok(unevaluated("Spectrogram", args));
  };
  let Some((m, o)) = partition_args(&args[1..], rate, xs.len()) else {
    return Ok(unevaluated("Spectrogram", args));
  };
  let frames = frames_of(&xs, m, o);
  let spectra: Vec<Vec<f64>> = frames
    .into_iter()
    .map(|mut frame| {
      hann_window(&mut frame);
      let bins = fft_padded(&frame);
      let half = bins.len() / 2;
      bins[..=half]
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt())
        .collect()
    })
    .collect();
  let n_bins = spectra.first().map(|s| s.len()).unwrap_or(0);
  let max_mag = spectra.iter().flatten().cloned().fold(0.0f64, f64::max);
  // matrix[row][col]: row = frequency bin (bottom = DC), col = frame.
  let matrix: Vec<Vec<f64>> = (0..n_bins)
    .map(|bin| {
      spectra
        .iter()
        .map(|s| db_intensity(s[bin], max_mag))
        .collect()
    })
    .collect();
  let duration = xs.len() as f64 / rate;
  let nyquist = rate / 2.0;
  let svg = if is_audio {
    raster_svg(&matrix, duration, nyquist, "Time (s)", "Frequency (Hz)")
  } else {
    raster_svg(
      &matrix,
      xs.len() as f64,
      (m.next_power_of_two() / 2) as f64,
      "",
      "",
    )
  };
  Ok(crate::graphics_result(svg))
}

/// Cepstrogram[data] / Cepstrogram[data, n] / Cepstrogram[data, n, d] —
/// plot the power cepstrum of each partition: time against quefrency
/// (seconds for audio, bins for lists).
pub fn cepstrogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated("Cepstrogram", args));
  }
  let Some((xs, rate, is_audio)) = spectral_input(&args[0]) else {
    return Ok(unevaluated("Cepstrogram", args));
  };
  let Some((m, o)) = partition_args(&args[1..], rate, xs.len()) else {
    return Ok(unevaluated("Cepstrogram", args));
  };
  let frames = frames_of(&xs, m, o);
  let cepstra: Vec<Vec<f64>> = frames
    .into_iter()
    .map(|mut frame| {
      // Power cepstrum: |IFFT(log |FFT(x)|²)| over the lower quefrencies.
      hann_window(&mut frame);
      let bins = fft_padded(&frame);
      let mut log_power: Vec<(f64, f64)> = bins
        .iter()
        .map(|&(re, im)| ((re * re + im * im + 1e-12).ln(), 0.0))
        .collect();
      fft_pow2_in_place(&mut log_power, 1.0);
      let n = log_power.len();
      let half = n / 2;
      log_power[..=half]
        .iter()
        .map(|&(re, im)| (re * re + im * im).sqrt() / n as f64)
        .collect()
    })
    .collect();
  let n_bins = cepstra.first().map(|s| s.len()).unwrap_or(0);
  // Quefrency bin 0 (the overall level) dwarfs everything; skip it for
  // the intensity scaling like Wolfram's display does.
  let max_val = cepstra
    .iter()
    .flat_map(|s| s.iter().skip(1))
    .cloned()
    .fold(0.0f64, f64::max);
  let matrix: Vec<Vec<f64>> = (1..n_bins)
    .map(|bin| {
      cepstra
        .iter()
        .map(|s| {
          if max_val > 0.0 {
            (s[bin] / max_val).clamp(0.0, 1.0)
          } else {
            0.0
          }
        })
        .collect()
    })
    .collect();
  let duration = xs.len() as f64 / rate;
  let max_quefrency = n_bins as f64 / rate;
  let svg = if is_audio {
    raster_svg(
      &matrix,
      duration,
      max_quefrency,
      "Time (s)",
      "Quefrency (s)",
    )
  } else {
    raster_svg(&matrix, xs.len() as f64, n_bins as f64, "", "")
  };
  Ok(crate::graphics_result(svg))
}

// ---------------------------------------------------------------------------
// Periodogram
// ---------------------------------------------------------------------------

/// Cap on plotted periodogram points; longer spectra are reduced by
/// max-pooling so peaks stay visible.
const PERIODOGRAM_MAX_POINTS: usize = 2048;

/// Periodogram[data] — plot the power spectrum in decibels
/// (10 log₁₀ |X|²): against frequency in Hz for audio objects, against the
/// frequency bin index for plain lists. Rendered through ListLinePlot.
pub fn periodogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Ok(unevaluated("Periodogram", args));
  }
  let Some((xs, rate, is_audio)) = spectral_input(&args[0]) else {
    return Ok(unevaluated("Periodogram", args));
  };
  let (freqs, powers) = power_spectrum(&xs, rate);
  let n = xs.len() as f64;
  // dB values, floored at -120 dB so silent bins stay plottable.
  let mut points: Vec<(f64, f64)> = freqs
    .iter()
    .zip(&powers)
    .map(|(f, p)| {
      let x = if is_audio { *f } else { f * n };
      (x, 10.0 * (p / n).max(1e-12).log10())
    })
    .collect();
  if points.len() > PERIODOGRAM_MAX_POINTS {
    let bucket = points.len().div_ceil(PERIODOGRAM_MAX_POINTS);
    points = points
      .chunks(bucket)
      .map(|c| {
        c.iter()
          .cloned()
          .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
          .unwrap()
      })
      .collect();
  }
  let pairs = Expr::List(
    points
      .into_iter()
      .map(|(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
      .collect::<Vec<_>>()
      .into(),
  );
  let mut plot_args = vec![pairs];
  plot_args.extend(
    args[1..]
      .iter()
      .filter(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
      .cloned(),
  );
  crate::functions::list_plot::list_line_plot_ast(&plot_args)
}
