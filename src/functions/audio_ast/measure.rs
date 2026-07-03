//! Audio analysis: AudioMeasurements, AudioLocalMeasurements,
//! AudioIntervals, and the statistics functions (Mean, Median, Variance,
//! Quantile) applied to Audio objects.

use super::spectral::power_spectrum;
use super::{AudioData, parse_audio, quantity, unevaluated};
use crate::InterpreterError;
use crate::syntax::Expr;

/// One channel's worth of a measurement, as an expression.
/// Frequency-valued properties are Quantity["Hertz"], time-valued ones
/// Quantity["Seconds"], counts Integers, everything else Reals.
pub fn channel_property(prop: &str, xs: &[f64], rate: f64) -> Option<Expr> {
  let n = xs.len();
  if n == 0 {
    return None;
  }
  let nf = n as f64;
  let mean = xs.iter().sum::<f64>() / nf;
  let power = xs.iter().map(|x| x * x).sum::<f64>() / nf;
  let rms = power.sqrt();
  let max_abs = xs.iter().fold(0.0f64, |a, x| a.max(x.abs()));

  let real = |v: f64| Some(Expr::Real(v));
  match prop {
    "Max" => real(xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)),
    "Min" => real(xs.iter().cloned().fold(f64::INFINITY, f64::min)),
    "MaxAbs" => real(max_abs),
    "MinAbs" => real(xs.iter().fold(f64::INFINITY, |a, x| a.min(x.abs()))),
    "MinMax" => {
      let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
      let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      Some(Expr::List(vec![Expr::Real(min), Expr::Real(max)].into()))
    }
    "Mean" => real(mean),
    "Median" => {
      let mut sorted = xs.to_vec();
      sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
      real(if n % 2 == 1 {
        sorted[n / 2]
      } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
      })
    }
    "Total" => real(xs.iter().sum()),
    "Power" => real(power),
    "Energy" => real(xs.iter().map(|x| x * x).sum()),
    "RMSAmplitude" => real(rms),
    "StandardDeviation" | "Variance" => {
      if n < 2 {
        return None;
      }
      let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
      real(if prop == "Variance" { var } else { var.sqrt() })
    }
    "TotalVariation" => real(xs.windows(2).map(|w| (w[1] - w[0]).abs()).sum()),
    "CrestFactor" => {
      if rms == 0.0 {
        return None;
      }
      real(max_abs / rms)
    }
    "PeakToAveragePowerRatio" => {
      if power == 0.0 {
        return None;
      }
      real(max_abs * max_abs / power)
    }
    "ZeroCrossings" => Some(Expr::Integer(zero_crossings(xs) as i128)),
    "ZeroCrossingRate" => real(zero_crossings(xs) as f64 / (nf / rate)),
    "TemporalCentroid" => {
      let energy: f64 = xs.iter().map(|x| x * x).sum();
      if energy == 0.0 {
        return None;
      }
      let centroid = xs
        .iter()
        .enumerate()
        .map(|(i, x)| (i as f64 / rate) * x * x)
        .sum::<f64>()
        / energy;
      Some(quantity(centroid, "Seconds"))
    }
    "FundamentalFrequency" => Some(match fundamental_frequency(xs, rate) {
      Some(f) => quantity(f, "Hertz"),
      None => Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotAvailable".to_string())].into(),
      },
    }),
    "SpectralCentroid" | "SpectralSpread" | "SpectralSkewness"
    | "SpectralKurtosis" | "SpectralCrest" | "SpectralFlatness"
    | "SpectralRollOff" | "SpectralSlope" => spectral_property(prop, xs, rate),
    _ => None,
  }
}

/// Count sign changes in the sample sequence.
fn zero_crossings(xs: &[f64]) -> usize {
  xs.windows(2).filter(|w| w[0] * w[1] < 0.0).count()
}

/// Estimate the fundamental frequency of a frame via the peak of the
/// normalized autocorrelation in the 50–2000 Hz pitch range. Returns `None`
/// for frames without a clear periodicity (normalized peak below 0.5).
fn fundamental_frequency(xs: &[f64], rate: f64) -> Option<f64> {
  let n = xs.len();
  let min_lag = ((rate / 2000.0).floor() as usize).max(2);
  let max_lag = ((rate / 50.0).ceil() as usize).min(n / 2);
  if max_lag <= min_lag {
    return None;
  }
  let energy: f64 = xs.iter().map(|x| x * x).sum();
  if energy == 0.0 {
    return None;
  }
  let mut best = (0usize, 0.0f64);
  for lag in min_lag..=max_lag {
    let corr: f64 = (0..n - lag).map(|i| xs[i] * xs[i + lag]).sum();
    let norm = corr / energy;
    if norm > best.1 {
      best = (lag, norm);
    }
  }
  (best.1 > 0.5).then(|| rate / best.0 as f64)
}

/// Spectral-moment measurements from the one-sided power spectrum.
fn spectral_property(prop: &str, xs: &[f64], rate: f64) -> Option<Expr> {
  let (freqs, powers) = power_spectrum(xs, rate);
  let total: f64 = powers.iter().sum();
  if total == 0.0 || powers.is_empty() {
    return None;
  }
  let centroid =
    freqs.iter().zip(&powers).map(|(f, p)| f * p).sum::<f64>() / total;
  let moment = |k: i32| -> f64 {
    freqs
      .iter()
      .zip(&powers)
      .map(|(f, p)| (f - centroid).powi(k) * p)
      .sum::<f64>()
      / total
  };
  let spread = moment(2).sqrt();
  match prop {
    "SpectralCentroid" => Some(quantity(centroid, "Hertz")),
    "SpectralSpread" => Some(quantity(spread, "Hertz")),
    "SpectralSkewness" => {
      (spread > 0.0).then(|| Expr::Real(moment(3) / spread.powi(3)))
    }
    "SpectralKurtosis" => {
      (spread > 0.0).then(|| Expr::Real(moment(4) / spread.powi(4)))
    }
    "SpectralCrest" => {
      let max = powers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      Some(Expr::Real(max / (total / powers.len() as f64)))
    }
    "SpectralFlatness" => {
      let eps = 1e-30;
      let log_mean = powers.iter().map(|p| (p + eps).ln()).sum::<f64>()
        / powers.len() as f64;
      Some(Expr::Real(log_mean.exp() / (total / powers.len() as f64)))
    }
    "SpectralRollOff" => {
      // Frequency below which 85% of the spectral energy is contained.
      let threshold = 0.85 * total;
      let mut acc = 0.0;
      for (f, p) in freqs.iter().zip(&powers) {
        acc += p;
        if acc >= threshold {
          return Some(quantity(*f, "Hertz"));
        }
      }
      Some(quantity(*freqs.last().unwrap(), "Hertz"))
    }
    "SpectralSlope" => {
      // Least-squares slope of power vs frequency.
      let m = powers.len() as f64;
      let f_mean = freqs.iter().sum::<f64>() / m;
      let p_mean = total / m;
      let num: f64 = freqs
        .iter()
        .zip(&powers)
        .map(|(f, p)| (f - f_mean) * (p - p_mean))
        .sum();
      let den: f64 = freqs.iter().map(|f| (f - f_mean).powi(2)).sum();
      (den > 0.0).then(|| Expr::Real(num / den))
    }
    _ => None,
  }
}

/// Compute one measurement of a whole audio object: channel-independent
/// properties ("Duration") give a single value, everything else a scalar
/// for mono and a per-channel list for multichannel audio.
fn audio_property(prop: &str, audio: &AudioData) -> Option<Expr> {
  match prop {
    "Duration" => Some(quantity(audio.duration(), "Seconds")),
    _ => {
      let mut values = Vec::with_capacity(audio.channels.len());
      for ch in &audio.channels {
        values.push(channel_property(prop, ch, audio.rate)?);
      }
      Some(if values.len() == 1 {
        values.pop().unwrap()
      } else {
        Expr::List(values.into())
      })
    }
  }
}

/// Extract a property name (a string) from an expression.
fn prop_name(expr: &Expr) -> Option<&str> {
  match expr {
    Expr::String(s) => Some(s.as_str()),
    _ => None,
  }
}

/// AudioMeasurements[audio, "prop"] or AudioMeasurements[audio, {props…}].
pub fn audio_measurements_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("AudioMeasurements", args));
  }
  let Some(audio) = parse_audio(&args[0]) else {
    return Ok(unevaluated("AudioMeasurements", args));
  };
  match &args[1] {
    Expr::List(props) => {
      let mut out = Vec::with_capacity(props.len());
      for p in props.iter() {
        let Some(v) = prop_name(p).and_then(|p| audio_property(p, &audio))
        else {
          return Ok(unevaluated("AudioMeasurements", args));
        };
        out.push(v);
      }
      Ok(Expr::List(out.into()))
    }
    p => match prop_name(p).and_then(|p| audio_property(p, &audio)) {
      Some(v) => Ok(v),
      None => Ok(unevaluated("AudioMeasurements", args)),
    },
  }
}

/// Default analysis partition: 25 ms windows with 12.5 ms offset (in
/// samples, at least one sample each).
fn analysis_partition(rate: f64) -> (usize, usize) {
  let w = ((0.025 * rate).round() as usize).max(1);
  let hop = ((0.0125 * rate).round() as usize).max(1);
  (w, hop)
}

/// Window start indices for a partition of `n` samples (the final partial
/// window is dropped; signals shorter than one window get a single window
/// spanning everything).
fn window_starts(n: usize, w: usize, hop: usize) -> Vec<usize> {
  if n <= w {
    return vec![0];
  }
  (0..=(n - w)).step_by(hop).collect()
}

/// AudioLocalMeasurements[audio, "prop"] (or a list of properties) —
/// measurements over 25 ms partitions, as a TimeSeries whose times are the
/// partition centers in seconds.
pub fn audio_local_measurements_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("AudioLocalMeasurements", args));
  }
  let Some(audio) = parse_audio(&args[0]) else {
    return Ok(unevaluated("AudioLocalMeasurements", args));
  };

  let local_ts = |prop: &str| -> Option<Expr> {
    let n = audio.len();
    let (w, hop) = analysis_partition(audio.rate);
    let mut pairs = Vec::new();
    for start in window_starts(n, w, hop) {
      let end = (start + w).min(n);
      let t = (start + (end - start) / 2) as f64 / audio.rate;
      let mut values = Vec::with_capacity(audio.channels.len());
      for ch in &audio.channels {
        values.push(channel_property(prop, &ch[start..end], audio.rate)?);
      }
      let value = if values.len() == 1 {
        values.pop().unwrap()
      } else {
        Expr::List(values.into())
      };
      pairs.push(Expr::List(vec![Expr::Real(t), value].into()));
    }
    Some(Expr::FunctionCall {
      name: "TimeSeries".to_string(),
      args: vec![Expr::List(pairs.into())].into(),
    })
  };

  match &args[1] {
    Expr::List(props) => {
      let mut out = Vec::with_capacity(props.len());
      for p in props.iter() {
        let Some(ts) = prop_name(p).and_then(local_ts) else {
          return Ok(unevaluated("AudioLocalMeasurements", args));
        };
        out.push(ts);
      }
      Ok(Expr::List(out.into()))
    }
    p => match prop_name(p).and_then(local_ts) {
      Some(ts) => Ok(ts),
      None => Ok(unevaluated("AudioLocalMeasurements", args)),
    },
  }
}

/// Properties made available to AudioIntervals criterion functions
/// (as `#RMSAmplitude` etc. applied to an association per window).
const INTERVAL_PROPS: &[&str] = &[
  "Max",
  "Min",
  "MaxAbs",
  "MinAbs",
  "Mean",
  "Median",
  "Power",
  "RMSAmplitude",
  "StandardDeviation",
  "Total",
  "TotalVariation",
  "Energy",
  "CrestFactor",
  "ZeroCrossings",
  "ZeroCrossingRate",
];

/// RMS threshold below which a window counts as silent for
/// `AudioIntervals[audio]`.
const SILENCE_RMS: f64 = 1e-4;

/// AudioIntervals[audio] — time intervals (in seconds) of silence.
/// AudioIntervals[audio, crit] — intervals where the criterion holds; crit
/// is a function of per-window measurement properties, e.g.
/// `AudioIntervals[audio, #RMSAmplitude > 0.1 &]`.
pub fn audio_intervals_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("AudioIntervals", args));
  }
  let Some(audio) = parse_audio(&args[0]) else {
    return Ok(unevaluated("AudioIntervals", args));
  };
  let mono = audio.mixdown();
  let n = mono.len();
  let (w, hop) = analysis_partition(audio.rate);
  let starts = window_starts(n, w, hop);

  let mut flags = Vec::with_capacity(starts.len());
  for &start in &starts {
    let frame = &mono[start..(start + w).min(n)];
    let selected = if args.len() == 1 {
      let rms =
        (frame.iter().map(|x| x * x).sum::<f64>() / frame.len() as f64).sqrt();
      rms < SILENCE_RMS
    } else {
      let mut entries = Vec::with_capacity(INTERVAL_PROPS.len());
      for prop in INTERVAL_PROPS {
        if let Some(v) = channel_property(prop, frame, audio.rate) {
          entries.push((Expr::String((*prop).to_string()), v));
        }
      }
      let assoc = Expr::Association(entries);
      matches!(
        crate::evaluator::function_application::apply_curried_call(
          &args[1],
          &[assoc],
        ),
        Ok(Expr::Identifier(ref t)) if t == "True"
      )
    };
    flags.push(selected);
  }

  // Merge consecutive selected windows into time intervals.
  let duration = audio.duration();
  let mut intervals: Vec<Expr> = Vec::new();
  let mut i = 0;
  while i < flags.len() {
    if flags[i] {
      let begin = i;
      while i + 1 < flags.len() && flags[i + 1] {
        i += 1;
      }
      let t1 = starts[begin] as f64 / audio.rate;
      let t2 = ((starts[i] + w) as f64 / audio.rate).min(duration);
      intervals.push(Expr::List(vec![Expr::Real(t1), Expr::Real(t2)].into()));
    }
    i += 1;
  }
  Ok(Expr::List(intervals.into()))
}

/// Statistics functions applied to Audio objects: `Mean[audio]`,
/// `Median[audio]`, `Variance[audio]`, `Quantile[audio, q]` operate on the
/// samples of each channel (scalar result for mono, per-channel list
/// otherwise). Delegates to the list implementations so the numeric
/// conventions match exactly. Returns `None` when the first argument is
/// not an audio object (the regular dispatch continues).
pub fn audio_stat_ast(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.is_empty() || !super::is_audio_expr(&args[0]) {
    return None;
  }
  let audio = parse_audio(&args[0])?;
  let mut results = Vec::with_capacity(audio.channels.len());
  for ch in &audio.channels {
    let list =
      Expr::List(ch.iter().map(|&s| Expr::Real(s)).collect::<Vec<_>>().into());
    let mut call_args = vec![list];
    call_args.extend_from_slice(&args[1..]);
    match crate::evaluator::evaluate_function_call_ast(name, &call_args) {
      Ok(v) => results.push(v),
      Err(e) => return Some(Err(e)),
    }
  }
  Some(Ok(if results.len() == 1 {
    results.pop().unwrap()
  } else {
    Expr::List(results.into())
  }))
}
