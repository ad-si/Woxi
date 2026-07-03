//! Audio processing functions (the Wolfram *Audio Processing* guide page):
//! editing (AudioAmplify, AudioTrim, AudioJoin, AudioPitchShift), analysis
//! (AudioMeasurements, AudioLocalMeasurements, AudioIntervals), the
//! short-time Fourier transform (ShortTimeFourier and its
//! ShortTimeFourierData object), the spectral plots (Spectrogram,
//! Periodogram, Cepstrogram), and noise-removal filters (WienerFilter,
//! TotalVariationFilter, plus Audio support for LowpassFilter/MeanFilter).
//!
//! Audio objects are handled in their canonical symbolic form
//! `Audio[data, SampleRate -> r]` (see [`crate::functions::sound`]): sample
//! data is a flat list (mono) or a list of per-channel lists, file-backed
//! audio is decoded when the file is a WAV, and `Sound`/`Play` expressions
//! are synthesized through the `Play` sampler.

pub mod edit;
pub mod filters;
pub mod measure;
pub mod spectral;

use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::sound;
use crate::syntax::Expr;

/// In-memory view of an audio object: per-channel samples plus the sample
/// rate in Hz. Channels are non-empty and equally long.
#[derive(Clone, Debug)]
pub struct AudioData {
  pub channels: Vec<Vec<f64>>,
  pub rate: f64,
}

impl AudioData {
  /// Number of sample frames per channel.
  pub fn len(&self) -> usize {
    self.channels.first().map(|c| c.len()).unwrap_or(0)
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Duration in seconds.
  pub fn duration(&self) -> f64 {
    self.len() as f64 / self.rate
  }

  /// Average all channels into a single mono channel (used by the analysis
  /// functions, which measure the mixed-down signal).
  pub fn mixdown(&self) -> Vec<f64> {
    if self.channels.len() == 1 {
      return self.channels[0].clone();
    }
    let n = self.len();
    let k = self.channels.len() as f64;
    (0..n)
      .map(|i| self.channels.iter().map(|c| c[i]).sum::<f64>() / k)
      .collect()
  }
}

/// Extract per-channel samples from an `Audio` data argument: a flat list of
/// samples (mono) or a list of per-channel lists. Unlike the playback path in
/// [`crate::functions::sound`], values are NOT clipped to [-1, 1] — analysis
/// and editing operate on the stored values.
fn samples_from_list(data: &Expr) -> Option<Vec<Vec<f64>>> {
  let Expr::List(items) = data else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
  let lists: Vec<Vec<Expr>> =
    if items.iter().all(|i| matches!(i, Expr::List(_))) {
      items
        .iter()
        .map(|i| match i {
          Expr::List(l) => l.to_vec(),
          _ => unreachable!(),
        })
        .collect()
    } else {
      vec![items.to_vec()]
    };
  let mut channels = Vec::with_capacity(lists.len());
  for list in &lists {
    let mut ch = Vec::with_capacity(list.len());
    for sample in list {
      ch.push(try_eval_to_f64(sample)?);
    }
    channels.push(ch);
  }
  let len = channels[0].len();
  if len == 0 || channels.iter().any(|c| c.len() != len) {
    return None;
  }
  Some(channels)
}

/// Parse an audio-like expression into sample data: `Audio[{…}, opts]`
/// (sample data), `Audio[File["x.wav"]]` / `Audio["x.wav"]` (decoded when
/// the file is a readable WAV), or a `Sound`/`Play` expression (synthesized
/// at 8000 Hz). Returns `None` for anything else.
pub fn parse_audio(expr: &Expr) -> Option<AudioData> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Audio" && !args.is_empty() =>
    {
      if let Some(channels) = samples_from_list(&args[0]) {
        return Some(AudioData {
          channels,
          rate: sound::audio_sample_rate(args),
        });
      }
      let path = sound::audio_file_source(&args[0])?;
      let bytes = std::fs::read(&path).ok()?;
      decode_wav(&bytes)
    }
    Expr::FunctionCall { name, .. } if name == "Sound" || name == "Play" => {
      let (samples, rate) = sound::sound_to_samples(expr)?;
      Some(AudioData {
        channels: vec![samples],
        rate: rate as f64,
      })
    }
    _ => None,
  }
}

/// True when the expression looks like an audio object (`Audio`, `Sound`,
/// or `Play` head) — used by shared functions (Mean, MeanFilter, …) to
/// decide whether to take the audio path.
pub fn is_audio_expr(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. }
    if name == "Audio" || name == "Sound" || name == "Play")
}

/// Build the canonical `Audio[data, SampleRate -> rate]` expression from
/// sample data: a flat list for mono, one list per channel otherwise.
pub fn make_audio(audio: &AudioData) -> Expr {
  let channel_list = |ch: &[f64]| -> Expr {
    Expr::List(ch.iter().map(|&s| Expr::Real(s)).collect::<Vec<_>>().into())
  };
  let data = if audio.channels.len() == 1 {
    channel_list(&audio.channels[0])
  } else {
    Expr::List(
      audio
        .channels
        .iter()
        .map(|c| channel_list(c))
        .collect::<Vec<_>>()
        .into(),
    )
  };
  let rate_expr = if audio.rate.fract() == 0.0 {
    Expr::Integer(audio.rate as i128)
  } else {
    Expr::Real(audio.rate)
  };
  Expr::FunctionCall {
    name: "Audio".to_string(),
    args: vec![
      data,
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("SampleRate".to_string())),
        replacement: Box::new(rate_expr),
      },
    ]
    .into(),
  }
}

/// The unevaluated form `name[args…]`, returned when arguments don't match
/// (wolframscript likewise leaves the expression symbolic).
pub fn unevaluated(name: &str, args: &[Expr]) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  }
}

/// Convert a time specification to seconds: a plain number, or
/// `Quantity[x, "Seconds"|"Milliseconds"|"Minutes"|"Hours"]`.
pub fn time_to_seconds(expr: &Expr) -> Option<f64> {
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
      "Seconds" => Some(mag),
      "Milliseconds" => Some(mag / 1000.0),
      "Minutes" => Some(mag * 60.0),
      "Hours" => Some(mag * 3600.0),
      _ => None,
    };
  }
  try_eval_to_f64(expr)
}

/// `Quantity[value, unit]` expression.
pub fn quantity(value: f64, unit: &str) -> Expr {
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![Expr::Real(value), Expr::String(unit.to_string())].into(),
  }
}

/// Linear-interpolation resampling of one channel to `new_len` samples,
/// mapping the first sample to the first and the last to the last.
pub fn resample_to_len(xs: &[f64], new_len: usize) -> Vec<f64> {
  if new_len == 0 || xs.is_empty() {
    return Vec::new();
  }
  if xs.len() == 1 {
    return vec![xs[0]; new_len];
  }
  if new_len == 1 {
    return vec![xs[0]];
  }
  let scale = (xs.len() - 1) as f64 / (new_len - 1) as f64;
  (0..new_len)
    .map(|i| {
      let pos = i as f64 * scale;
      let lo = pos.floor() as usize;
      let hi = (lo + 1).min(xs.len() - 1);
      let frac = pos - lo as f64;
      xs[lo] * (1.0 - frac) + xs[hi] * frac
    })
    .collect()
}

/// Import an audio file as an Audio object: WAV files are decoded into
/// sample data; other audio formats (no decoder available) become
/// file-backed `Audio[File["path"]]` objects, which visual hosts can still
/// play.
pub fn import_audio_file(path: &str) -> Result<Expr, crate::InterpreterError> {
  let ext = std::path::Path::new(path)
    .extension()
    .map(|e| e.to_string_lossy().to_lowercase())
    .unwrap_or_default();
  if matches!(ext.as_str(), "wav" | "wave") {
    let bytes = std::fs::read(path).map_err(|e| {
      crate::InterpreterError::EvaluationError(format!(
        "Import: cannot open \"{path}\": {e}"
      ))
    })?;
    if let Some(audio) = decode_wav(&bytes) {
      return Ok(make_audio(&audio));
    }
    return Err(crate::InterpreterError::EvaluationError(format!(
      "Import: \"{path}\" is not a valid WAV file"
    )));
  }
  if !std::path::Path::new(path).exists() {
    return Err(crate::InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{path}\": file not found"
    )));
  }
  Ok(Expr::FunctionCall {
    name: "Audio".to_string(),
    args: vec![Expr::FunctionCall {
      name: "File".to_string(),
      args: vec![Expr::String(path.to_string())].into(),
    }]
    .into(),
  })
}

// ---------------------------------------------------------------------------
// WAV decoding
// ---------------------------------------------------------------------------

/// Decode a RIFF/WAVE byte stream into per-channel samples in [-1, 1].
/// Supports PCM (8/16/24/32-bit integer) and IEEE-float (32/64-bit) data,
/// including WAVE_FORMAT_EXTENSIBLE headers.
pub fn decode_wav(bytes: &[u8]) -> Option<AudioData> {
  if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
    return None;
  }
  let mut format: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
  let mut data: Option<&[u8]> = None;
  let mut pos = 12usize;
  while pos + 8 <= bytes.len() {
    let id = &bytes[pos..pos + 4];
    let size =
      u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().ok()?) as usize;
    let body_end = (pos + 8 + size).min(bytes.len());
    let body = &bytes[pos + 8..body_end];
    match id {
      b"fmt " if body.len() >= 16 => {
        let mut tag = u16::from_le_bytes(body[0..2].try_into().ok()?);
        let channels = u16::from_le_bytes(body[2..4].try_into().ok()?);
        let rate = u32::from_le_bytes(body[4..8].try_into().ok()?);
        let bits = u16::from_le_bytes(body[14..16].try_into().ok()?);
        // WAVE_FORMAT_EXTENSIBLE: the real format tag is the first two
        // bytes of the SubFormat GUID.
        if tag == 0xFFFE && body.len() >= 26 {
          tag = u16::from_le_bytes(body[24..26].try_into().ok()?);
        }
        format = Some((tag, channels, rate, bits));
      }
      b"data" => data = Some(body),
      _ => {}
    }
    // Chunks are word-aligned: odd sizes are padded with one byte.
    pos = pos + 8 + size + (size & 1);
  }

  let (tag, channels, rate, bits) = format?;
  let data = data?;
  if channels == 0 || rate == 0 {
    return None;
  }
  let ch = channels as usize;
  let bytes_per_sample = (bits as usize).div_ceil(8);
  if bytes_per_sample == 0 {
    return None;
  }
  let frames = data.len() / (bytes_per_sample * ch);
  if frames == 0 {
    return None;
  }

  let read_sample = |idx: usize| -> Option<f64> {
    let start = idx * bytes_per_sample;
    let s = &data[start..start + bytes_per_sample];
    match (tag, bits) {
      (1, 8) => Some((s[0] as f64 - 128.0) / 128.0),
      (1, 16) => Some(i16::from_le_bytes(s.try_into().ok()?) as f64 / 32768.0),
      (1, 24) => {
        let v = i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8;
        Some(v as f64 / 8_388_608.0)
      }
      (1, 32) => {
        Some(i32::from_le_bytes(s.try_into().ok()?) as f64 / 2_147_483_648.0)
      }
      (3, 32) => Some(f32::from_le_bytes(s.try_into().ok()?) as f64),
      (3, 64) => Some(f64::from_le_bytes(s.try_into().ok()?)),
      _ => None,
    }
  };

  let mut out = vec![Vec::with_capacity(frames); ch];
  for f in 0..frames {
    for (c, channel) in out.iter_mut().enumerate() {
      channel.push(read_sample(f * ch + c)?);
    }
  }
  Some(AudioData {
    channels: out,
    rate: rate as f64,
  })
}
