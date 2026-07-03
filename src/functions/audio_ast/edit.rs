//! Audio editing: AudioAmplify, AudioTrim, AudioJoin, AudioPitchShift.

use super::{
  AudioData, make_audio, parse_audio, resample_to_len, time_to_seconds,
  unevaluated,
};
use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// AudioAmplify[audio, s] — multiply every sample by the amplification
/// factor s.
pub fn audio_amplify_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("AudioAmplify", args));
  }
  let (Some(mut audio), Some(s)) =
    (parse_audio(&args[0]), try_eval_to_f64(&args[1]))
  else {
    return Ok(unevaluated("AudioAmplify", args));
  };
  for ch in &mut audio.channels {
    for x in ch.iter_mut() {
      *x *= s;
    }
  }
  Ok(make_audio(&audio))
}

/// Convert a time in seconds to a sample index, clamped to [0, n].
fn sample_index(t: f64, rate: f64, n: usize) -> usize {
  ((t * rate).round().max(0.0) as usize).min(n)
}

/// Threshold below which a sample counts as silence for `AudioTrim[audio]`.
const SILENCE_THRESHOLD: f64 = 1e-4;

/// AudioTrim[audio] — trim leading and trailing silence.
/// AudioTrim[audio, t] — take the first t seconds.
/// AudioTrim[audio, {t1, t2}] — take the interval from t1 to t2 (negative
/// times count from the end of the audio).
pub fn audio_trim_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated("AudioTrim", args));
  }
  let Some(audio) = parse_audio(&args[0]) else {
    return Ok(unevaluated("AudioTrim", args));
  };
  let n = audio.len();
  let (start, end) = if args.len() == 1 {
    // Trim silence at both ends (a frame is silent when every channel is
    // below the threshold).
    let loud = |i: usize| {
      audio
        .channels
        .iter()
        .any(|c| c[i].abs() > SILENCE_THRESHOLD)
    };
    let first = (0..n).find(|&i| loud(i));
    match first {
      None => (0, 0),
      Some(first) => {
        let last = (0..n).rev().find(|&i| loud(i)).unwrap_or(first);
        (first, last + 1)
      }
    }
  } else {
    match &args[1] {
      Expr::List(spec) if spec.len() == 2 => {
        let (Some(mut t1), Some(mut t2)) =
          (time_to_seconds(&spec[0]), time_to_seconds(&spec[1]))
        else {
          return Ok(unevaluated("AudioTrim", args));
        };
        let dur = audio.duration();
        // Negative times count from the end.
        if t1 < 0.0 {
          t1 += dur;
        }
        if t2 < 0.0 {
          t2 += dur;
        }
        if t2 < t1 {
          return Ok(unevaluated("AudioTrim", args));
        }
        (
          sample_index(t1, audio.rate, n),
          sample_index(t2, audio.rate, n),
        )
      }
      spec => {
        let Some(t) = time_to_seconds(spec) else {
          return Ok(unevaluated("AudioTrim", args));
        };
        if t < 0.0 {
          return Ok(unevaluated("AudioTrim", args));
        }
        (0, sample_index(t, audio.rate, n))
      }
    }
  };

  if end <= start {
    return Ok(unevaluated("AudioTrim", args));
  }
  let trimmed = AudioData {
    channels: audio
      .channels
      .iter()
      .map(|c| c[start..end].to_vec())
      .collect(),
    rate: audio.rate,
  };
  Ok(make_audio(&trimmed))
}

/// Conform a list of audio objects to a common sample rate (the maximum)
/// and channel count (the maximum; missing channels repeat the last one).
fn conform(mut audios: Vec<AudioData>) -> Vec<AudioData> {
  let rate = audios.iter().fold(0.0f64, |a, x| a.max(x.rate));
  let channels = audios.iter().map(|a| a.channels.len()).max().unwrap_or(1);
  for a in &mut audios {
    if a.rate != rate {
      let new_len =
        ((a.len() as f64) * rate / a.rate).round().max(1.0) as usize;
      a.channels = a
        .channels
        .iter()
        .map(|c| resample_to_len(c, new_len))
        .collect();
      a.rate = rate;
    }
    while a.channels.len() < channels {
      let last = a.channels.last().unwrap().clone();
      a.channels.push(last);
    }
  }
  audios
}

/// AudioJoin[audio1, audio2, …] or AudioJoin[{audio1, …}] — concatenate
/// audio objects in time. Inputs are conformed to the highest sample rate
/// and channel count first.
pub fn audio_join_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let items: Vec<&Expr> = match args {
    [Expr::List(list)] => list.iter().collect(),
    _ => args.iter().collect(),
  };
  if items.is_empty() {
    return Ok(unevaluated("AudioJoin", args));
  }
  let mut audios = Vec::with_capacity(items.len());
  for item in items {
    let Some(a) = parse_audio(item) else {
      return Ok(unevaluated("AudioJoin", args));
    };
    audios.push(a);
  }
  let audios = conform(audios);
  let channels = audios[0].channels.len();
  let rate = audios[0].rate;
  let mut joined: Vec<Vec<f64>> = vec![Vec::new(); channels];
  for a in &audios {
    for (c, ch) in a.channels.iter().enumerate() {
      joined[c].extend_from_slice(ch);
    }
  }
  Ok(make_audio(&AudioData {
    channels: joined,
    rate,
  }))
}

/// Parse a pitch-shift specification: a plain frequency ratio, or
/// `Quantity[n, "Semitones"|"Cents"|"Octaves"]`.
fn pitch_ratio(expr: &Expr) -> Option<f64> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Quantity"
    && args.len() == 2
  {
    let mag = try_eval_to_f64(&args[0])?;
    let unit = match &args[1] {
      Expr::String(s) => s.as_str(),
      Expr::Identifier(s) => s.as_str(),
      _ => return None,
    };
    return match unit {
      "Semitones" => Some((mag / 12.0).exp2()),
      "Cents" => Some((mag / 1200.0).exp2()),
      "Octaves" => Some(mag.exp2()),
      _ => None,
    };
  }
  try_eval_to_f64(expr)
}

/// Time-stretch one channel by `factor` (output is `factor` times longer)
/// using windowed overlap-add with Hann windows.
fn time_stretch(xs: &[f64], factor: f64) -> Vec<f64> {
  let n = xs.len();
  let out_len = ((n as f64) * factor).round().max(1.0) as usize;
  // Frame size: large enough to span several pitch periods, small enough
  // that the input provides several overlapping frames.
  let w = 2048.min(((n / 4).max(1)).next_power_of_two() / 2).max(4);
  let hop_syn = (w / 4).max(1);
  if n <= w || out_len <= w {
    // Too short for overlap-add: plain resampling (changes pitch, but for
    // sub-frame inputs there is no meaningful stretch to perform).
    return resample_to_len(xs, out_len);
  }
  let hop_ana = hop_syn as f64 / factor;
  let window: Vec<f64> = (0..w)
    .map(|i| {
      let t = std::f64::consts::PI * 2.0 * i as f64 / w as f64;
      0.5 * (1.0 - t.cos())
    })
    .collect();
  let mut out = vec![0.0; out_len];
  let mut norm = vec![0.0; out_len];
  let frames = (out_len - w) / hop_syn + 1;
  for f in 0..frames {
    let out_pos = f * hop_syn;
    let in_pos =
      ((f as f64 * hop_ana).round() as usize).min(n.saturating_sub(w));
    for i in 0..w {
      out[out_pos + i] += xs[in_pos + i] * window[i];
      norm[out_pos + i] += window[i];
    }
  }
  for i in 0..out_len {
    if norm[i] > 1e-9 {
      out[i] /= norm[i];
    }
  }
  out
}

/// AudioPitchShift[audio, r] — shift the pitch by the frequency ratio r
/// (or by Quantity semitones/cents/octaves), preserving the duration:
/// the signal is time-stretched by r via overlap-add, then resampled back
/// to the original length.
pub fn audio_pitch_shift_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Ok(unevaluated("AudioPitchShift", args));
  }
  let (Some(audio), Some(r)) = (parse_audio(&args[0]), pitch_ratio(&args[1]))
  else {
    return Ok(unevaluated("AudioPitchShift", args));
  };
  if !(r.is_finite() && r > 0.0) {
    return Ok(unevaluated("AudioPitchShift", args));
  }
  let n = audio.len();
  let shifted = AudioData {
    channels: audio
      .channels
      .iter()
      .map(|c| resample_to_len(&time_stretch(c, r), n))
      .collect(),
    rate: audio.rate,
  };
  Ok(make_audio(&shifted))
}
