//! Audio synthesis for `Play[f, {t, tmin, tmax}]` and playable rendering of
//! `Audio[…]` objects.
//!
//! `Play` builds a `Sound` object whose amplitude is the function `f` of the
//! time variable `t` (in seconds). The CLI cannot emit audio, so it renders
//! the object textually as `-Sound-`. In visual hosts (the Woxi Playground and
//! Woxi Studio) the waveform is sampled, encoded as a WAV file, and embedded in
//! an `<audio controls>` element so the sound can actually be played.
//!
//! `Audio[File["path"]]` / `Audio["path.flac"]` / `Audio[{samples…}]` objects
//! are likewise turned into a playable [`crate::AudioOutput`] so visual hosts
//! render a graphical audio player (the CLI keeps the symbolic form).

use base64::Engine;

use crate::AudioOutput;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::evaluate_at_point;
use crate::syntax::Expr;

/// Sample rate used for synthesized `Play` audio (Hz). Matches the default
/// `SampleRate` wolframscript uses for `Play` (8000 samples per second).
const SAMPLE_RATE: u32 = 8000;

/// Collect every `Play[f, {t, tmin, tmax}]` segment reachable inside a `Sound`
/// expression, recursing through nested lists (so `Sound[{Play[…], Play[…]}]`
/// yields both segments in order).
fn collect_play_segments<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
  match expr {
    Expr::FunctionCall { name, .. } if name == "Play" => out.push(expr),
    Expr::FunctionCall { args, .. } => {
      for a in args.iter() {
        collect_play_segments(a, out);
      }
    }
    Expr::List(items) => {
      for a in items.iter() {
        collect_play_segments(a, out);
      }
    }
    _ => {}
  }
}

/// Sample a single `Play[f, {t, tmin, tmax}]` segment into amplitude values
/// clipped to [-1, 1]. Returns `None` when the iterator is malformed or the
/// amplitude cannot be evaluated to a number anywhere on the interval.
fn sample_play(play: &Expr) -> Option<Vec<f64>> {
  let Expr::FunctionCall { name, args } = play else {
    return None;
  };
  if name != "Play" || args.len() != 2 {
    return None;
  }
  let Expr::List(iter) = &args[1] else {
    return None;
  };
  if iter.len() != 3 {
    return None;
  }
  let Expr::Identifier(var) = &iter[0] else {
    return None;
  };
  let tmin = try_eval_to_f64(&iter[1])?;
  let tmax = try_eval_to_f64(&iter[2])?;
  if !(tmax > tmin) {
    return None;
  }

  let body = &args[0];
  let count = ((tmax - tmin) * SAMPLE_RATE as f64).round() as usize;
  if count == 0 {
    return None;
  }

  let mut samples = Vec::with_capacity(count);
  let mut any_finite = false;
  for i in 0..count {
    let t = tmin + i as f64 / SAMPLE_RATE as f64;
    // Outside the defined range or where the amplitude is non-numeric, emit
    // silence rather than failing the whole segment.
    let amp = evaluate_at_point(body, var, t).unwrap_or(0.0);
    if amp.is_finite() {
      any_finite = true;
    }
    // wolframscript clips amplitudes to [-1, 1] before quantizing.
    samples.push(amp.clamp(-1.0, 1.0));
  }
  any_finite.then_some(samples)
}

/// Sample every `Play` segment inside a `Sound` (or bare `Play`) expression
/// into amplitude values clipped to [-1, 1], concatenated in order. Returns
/// the samples together with the sample rate in Hz, or `None` when the
/// expression contains no samplable `Play` segment.
pub fn sound_to_samples(sound_expr: &Expr) -> Option<(Vec<f64>, u32)> {
  let mut plays = Vec::new();
  collect_play_segments(sound_expr, &mut plays);
  let mut samples = Vec::new();
  for play in plays {
    if let Some(seg) = sample_play(play) {
      samples.extend(seg);
    }
  }
  (!samples.is_empty()).then_some((samples, SAMPLE_RATE))
}

/// Encode interleaved 16-bit PCM samples as a little-endian WAV byte stream.
fn encode_wav(samples: &[i16], channels: u16, sample_rate: u32) -> Vec<u8> {
  let num_samples = samples.len() as u32;
  let bits_per_sample: u16 = 16;
  let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
  let block_align = channels * (bits_per_sample / 8);
  let data_len = num_samples * (bits_per_sample / 8) as u32;
  let riff_len = 36 + data_len;

  let mut buf = Vec::with_capacity(44 + data_len as usize);
  buf.extend_from_slice(b"RIFF");
  buf.extend_from_slice(&riff_len.to_le_bytes());
  buf.extend_from_slice(b"WAVE");
  // fmt chunk
  buf.extend_from_slice(b"fmt ");
  buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
  buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
  buf.extend_from_slice(&channels.to_le_bytes());
  buf.extend_from_slice(&sample_rate.to_le_bytes());
  buf.extend_from_slice(&byte_rate.to_le_bytes());
  buf.extend_from_slice(&block_align.to_le_bytes());
  buf.extend_from_slice(&bits_per_sample.to_le_bytes());
  // data chunk
  buf.extend_from_slice(b"data");
  buf.extend_from_slice(&data_len.to_le_bytes());
  for s in samples {
    buf.extend_from_slice(&s.to_le_bytes());
  }
  buf
}

/// Render a `Sound` expression (containing one or more `Play` segments) into a
/// base64-encoded WAV byte stream. Returns `None` when the sound contains no
/// samplable `Play` segment.
pub fn sound_to_wav_base64(sound_expr: &Expr) -> Option<String> {
  // Concatenate the segments in order (matching how Sound plays a list of
  // sounds sequentially).
  let (samples, rate) = sound_to_samples(sound_expr)?;
  let pcm: Vec<i16> = samples
    .iter()
    .map(|s| (s * i16::MAX as f64).round() as i16)
    .collect();
  let wav = encode_wav(&pcm, 1, rate);
  Some(base64::engine::general_purpose::STANDARD.encode(&wav))
}

/// Default sample rate wolframscript assumes for `Audio[{samples…}]` built
/// from raw sample data (Hz).
pub(crate) const AUDIO_SAMPLE_RATE: f64 = 44100.0;

/// Extract the sample rate from an `Audio[data, opts…]` argument list
/// (`SampleRate -> r`), falling back to the 44100 Hz default.
pub(crate) fn audio_sample_rate(args: &[Expr]) -> f64 {
  for opt in args.iter().skip(1) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SampleRate")
    {
      let val = crate::evaluator::evaluate_expr_to_expr(replacement)
        .unwrap_or(*replacement.clone());
      if let Some(r) = try_eval_to_f64(&val)
        && r > 0.0
      {
        return r;
      }
    }
  }
  AUDIO_SAMPLE_RATE
}

/// File extensions recognized as audio sources for `Audio["path.ext"]`
/// (a bare string argument only counts as a file path when it looks like an
/// audio file; `Audio[File[…]]` always does).
const AUDIO_EXTENSIONS: &[&str] = &[
  "flac", "wav", "wave", "mp3", "ogg", "oga", "opus", "m4a", "mp4", "aac",
  "aif", "aiff",
];

/// MIME type for an audio file path, derived from its extension.
fn audio_mime_for_path(path: &str) -> &'static str {
  let ext = std::path::Path::new(path)
    .extension()
    .map(|e| e.to_string_lossy().to_lowercase())
    .unwrap_or_default();
  match ext.as_str() {
    "flac" => "audio/flac",
    "wav" | "wave" => "audio/wav",
    "mp3" => "audio/mpeg",
    "ogg" | "oga" | "opus" => "audio/ogg",
    "m4a" | "mp4" => "audio/mp4",
    "aac" => "audio/aac",
    "aif" | "aiff" => "audio/aiff",
    _ => "application/octet-stream",
  }
}

/// Extract the source path from a file-backed `Audio` first argument:
/// `File["path"]`, or a bare string with a recognized audio file extension.
fn audio_file_source(arg: &Expr) -> Option<String> {
  match arg {
    Expr::FunctionCall { name, args } if name == "File" && args.len() == 1 => {
      match &args[0] {
        Expr::String(s) => Some(s.clone()),
        _ => None,
      }
    }
    Expr::String(s) => {
      let ext = std::path::Path::new(s.as_str())
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
      AUDIO_EXTENSIONS.contains(&ext.as_str()).then(|| s.clone())
    }
    _ => None,
  }
}

/// Extract per-channel amplitude samples from `Audio`'s data argument:
/// `{s1, s2, …}` (mono) or `{{ch1…}, {ch2…}, …}` (one list per channel).
/// All channels must be non-empty, equally long, and numeric; samples are
/// clipped to [-1, 1] like wolframscript does before quantizing.
fn audio_sample_channels(data: &Expr) -> Option<Vec<Vec<f64>>> {
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
      ch.push(try_eval_to_f64(sample)?.clamp(-1.0, 1.0));
    }
    channels.push(ch);
  }
  let len = channels[0].len();
  if len == 0 || channels.iter().any(|c| c.len() != len) {
    return None;
  }
  Some(channels)
}

/// Turn an `Audio[…]` expression into a playable [`AudioOutput`] for visual
/// hosts (the Woxi Playground and Woxi Studio), which render it as a
/// graphical audio player.
///
/// - `Audio[File["path"]]` / `Audio["path.flac"]`: the file bytes are read
///   and base64-encoded with a MIME type derived from the extension. When the
///   file cannot be read (missing, or no filesystem in the browser
///   playground), the base64 payload is left empty — the player chrome is
///   still rendered, it just cannot play.
/// - `Audio[{samples…}, opts]`: the samples are quantized to 16-bit PCM and
///   encoded as a WAV (honoring `SampleRate -> r`, default 44100 Hz).
///
/// Returns `None` when the expression is not an `Audio` object with a
/// file source or numeric sample data.
pub fn audio_to_output(expr: &Expr) -> Option<AudioOutput> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Audio" || args.is_empty() {
    return None;
  }

  if let Some(path) = audio_file_source(&args[0]) {
    let mime = audio_mime_for_path(&path).to_string();
    let base64 = std::fs::read(&path)
      .map(|bytes| base64::engine::general_purpose::STANDARD.encode(&bytes))
      .unwrap_or_default();
    let label = std::path::Path::new(&path)
      .file_name()
      .map(|f| f.to_string_lossy().into_owned())
      .unwrap_or_else(|| path.clone());
    return Some(AudioOutput {
      base64,
      mime,
      label: Some(label),
    });
  }

  let channels = audio_sample_channels(&args[0])?;
  let rate = audio_sample_rate(args).round() as u32;
  let frames = channels[0].len();
  let mut pcm = Vec::with_capacity(frames * channels.len());
  for i in 0..frames {
    for ch in &channels {
      pcm.push((ch[i] * i16::MAX as f64).round() as i16);
    }
  }
  let wav = encode_wav(&pcm, channels.len() as u16, rate);
  Some(AudioOutput {
    base64: base64::engine::general_purpose::STANDARD.encode(&wav),
    mime: "audio/wav".to_string(),
    label: None,
  })
}
