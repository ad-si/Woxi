//! Audio synthesis for `Play[f, {t, tmin, tmax}]`.
//!
//! `Play` builds a `Sound` object whose amplitude is the function `f` of the
//! time variable `t` (in seconds). The CLI cannot emit audio, so it renders
//! the object textually as `-Sound-`. In visual hosts (the Woxi Playground and
//! Woxi Studio) the waveform is sampled, encoded as a WAV file, and embedded in
//! an `<audio controls>` element so the sound can actually be played.

use base64::Engine;

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

/// Sample a single `Play[f, {t, tmin, tmax}]` segment into 16-bit PCM samples.
/// Returns `None` when the iterator is malformed or the amplitude cannot be
/// evaluated to a number anywhere on the interval.
fn sample_play(play: &Expr) -> Option<Vec<i16>> {
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
    let clipped = amp.clamp(-1.0, 1.0);
    samples.push((clipped * i16::MAX as f64).round() as i16);
  }
  any_finite.then_some(samples)
}

/// Encode mono 16-bit PCM samples as a little-endian WAV byte stream.
fn encode_wav(samples: &[i16]) -> Vec<u8> {
  let num_samples = samples.len() as u32;
  let bits_per_sample: u16 = 16;
  let channels: u16 = 1;
  let byte_rate = SAMPLE_RATE * channels as u32 * (bits_per_sample / 8) as u32;
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
  buf.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
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
  let mut plays = Vec::new();
  collect_play_segments(sound_expr, &mut plays);
  if plays.is_empty() {
    return None;
  }

  // Concatenate the segments in order (matching how Sound plays a list of
  // sounds sequentially).
  let mut samples = Vec::new();
  for play in plays {
    if let Some(seg) = sample_play(play) {
      samples.extend(seg);
    }
  }
  if samples.is_empty() {
    return None;
  }

  let wav = encode_wav(&samples);
  Some(base64::engine::general_purpose::STANDARD.encode(&wav))
}
