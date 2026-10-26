//! A small audio synthesizer for generated example sounds: tones with
//! harmonics and ADSR envelopes, coloured noise, filters, formant speech,
//! chirps and percussion. Everything renders into `Vec<f64>` samples in
//! `[-1, 1]`.

use std::f64::consts::TAU;

use super::rng::Rng;

/// Attack–decay–sustain–release envelope (times in seconds, sustain level
/// relative to the peak) evaluated at time `t` of a note lasting `dur`.
pub fn adsr(t: f64, dur: f64, a: f64, d: f64, s: f64, r: f64) -> f64 {
  if t < 0.0 || t > dur {
    return 0.0;
  }
  let release_start = (dur - r).max(0.0);
  let level = if t < a {
    t / a.max(1e-9)
  } else if t < a + d {
    1.0 - (1.0 - s) * (t - a) / d.max(1e-9)
  } else {
    s
  };
  if t > release_start {
    level * (1.0 - (t - release_start) / r.max(1e-9)).max(0.0)
  } else {
    level
  }
}

/// A buffer of `n` samples at sample rate `sr`.
pub struct Buffer {
  pub sr: f64,
  pub samples: Vec<f64>,
}

impl Buffer {
  pub fn new(sr: f64, n: usize) -> Self {
    Buffer {
      sr,
      samples: vec![0.0; n],
    }
  }

  pub fn duration(&self) -> f64 {
    self.samples.len() as f64 / self.sr
  }

  /// Add a note starting at `start` (seconds) lasting `dur`, with
  /// fundamental `f0`, partials `(multiple, amplitude)`, an envelope and a
  /// slight vibrato.
  #[allow(clippy::too_many_arguments)]
  pub fn note(
    &mut self,
    start: f64,
    dur: f64,
    f0: f64,
    partials: &[(f64, f64)],
    env: (f64, f64, f64, f64),
    gain: f64,
    vibrato: f64,
  ) {
    let i0 = (start * self.sr) as usize;
    let n = (dur * self.sr) as usize;
    let mut phases = vec![0.0; partials.len()];
    for k in 0..n {
      let i = i0 + k;
      if i >= self.samples.len() {
        break;
      }
      let t = k as f64 / self.sr;
      let e = adsr(t, dur, env.0, env.1, env.2, env.3);
      let f = f0 * (1.0 + vibrato * (TAU * 5.5 * t).sin());
      let mut v = 0.0;
      for (j, &(m, a)) in partials.iter().enumerate() {
        phases[j] += TAU * f * m / self.sr;
        v += a * phases[j].sin();
      }
      self.samples[i] += gain * e * v;
    }
  }

  /// Add a noise burst with exponential decay (percussion, claps, pops).
  pub fn burst(
    &mut self,
    rng: &mut Rng,
    start: f64,
    decay: f64,
    gain: f64,
    tone: Option<f64>,
  ) {
    let i0 = (start * self.sr) as usize;
    let n = ((decay * 6.0) * self.sr) as usize;
    let mut lp = 0.0;
    for k in 0..n {
      let i = i0 + k;
      if i >= self.samples.len() {
        break;
      }
      let t = k as f64 / self.sr;
      let e = (-t / decay).exp();
      let white = rng.range(-1.0, 1.0);
      lp += 0.35 * (white - lp);
      let body = tone.map_or(0.0, |f| {
        (TAU * f * t * (1.0 - 0.3 * t / (decay * 6.0))).sin()
      });
      self.samples[i] += gain * e * (0.6 * lp + 0.8 * body);
    }
  }

  /// Add a frequency sweep from `f0` to `f1` (Hz).
  pub fn chirp(&mut self, start: f64, dur: f64, f0: f64, f1: f64, gain: f64) {
    let i0 = (start * self.sr) as usize;
    let n = (dur * self.sr) as usize;
    let mut phase = 0.0;
    for k in 0..n {
      let i = i0 + k;
      if i >= self.samples.len() {
        break;
      }
      let t = k as f64 / n.max(1) as f64;
      phase += TAU * (f0 + (f1 - f0) * t) / self.sr;
      let e = (std::f64::consts::PI * t).sin();
      self.samples[i] += gain * e * phase.sin();
    }
  }

  /// Add speech-like sound: a glottal pulse train at pitch `f0` shaped by
  /// two formants per syllable, with syllables of random vowels.
  pub fn speech(
    &mut self,
    rng: &mut Rng,
    start: f64,
    dur: f64,
    f0: f64,
    gain: f64,
  ) {
    // (F1, F2) of a, e, i, o, u.
    const VOWELS: [(f64, f64); 5] = [
      (730.0, 1090.0),
      (530.0, 1840.0),
      (270.0, 2290.0),
      (570.0, 840.0),
      (300.0, 870.0),
    ];
    let mut t = start;
    let end = (start + dur).min(self.duration());
    while t < end {
      let syl = rng.range(0.12, 0.28);
      let (f1, f2) = *rng.pick(&VOWELS);
      let pitch = f0 * rng.range(0.85, 1.2);
      let i0 = (t * self.sr) as usize;
      let n = (syl.min(end - t) * self.sr) as usize;
      let mut r1 = Resonator::new(f1, 80.0, self.sr);
      let mut r2 = Resonator::new(f2, 120.0, self.sr);
      let mut phase = 0.0;
      for k in 0..n {
        let i = i0 + k;
        if i >= self.samples.len() {
          break;
        }
        let tt = k as f64 / n.max(1) as f64;
        phase += pitch * (1.0 - 0.15 * tt) / self.sr;
        let pulse =
          if phase.fract() < 0.08 { 1.0 } else { 0.0 } + rng.range(-0.05, 0.05);
        if phase >= 1.0 {
          phase -= 1.0;
        }
        let v = r1.step(pulse) + 0.6 * r2.step(pulse);
        let e = (std::f64::consts::PI * tt).sin().powf(0.7);
        self.samples[i] += gain * e * v * 0.05;
      }
      // Pauses between words.
      t += syl
        + if rng.chance(0.3) {
          rng.range(0.05, 0.25)
        } else {
          0.02
        };
    }
  }

  /// Add filtered noise with a slow amplitude modulation (wind, water,
  /// crowds, machinery).
  #[allow(clippy::too_many_arguments)]
  pub fn ambience(
    &mut self,
    rng: &mut Rng,
    cutoff: f64,
    modulation_hz: f64,
    depth: f64,
    gain: f64,
    hum: Option<f64>,
  ) {
    let a = 1.0 - (-TAU * cutoff / self.sr).exp();
    let mut lp = 0.0;
    let phase0 = rng.range(0.0, TAU);
    for (i, s) in self.samples.iter_mut().enumerate() {
      let t = i as f64 / self.sr;
      lp += a * (rng.range(-1.0, 1.0) - lp);
      let m =
        1.0 - depth * 0.5 * (1.0 + (TAU * modulation_hz * t + phase0).sin());
      let h = hum.map_or(0.0, |f| {
        0.3 * (TAU * f * t).sin() + 0.1 * (TAU * 2.0 * f * t).sin()
      });
      *s += gain * m * (lp * 2.0 + h);
    }
  }

  /// Scale so the RMS level is `rms` (as recorded for the original), but
  /// never let the peak exceed `peak`.
  pub fn normalize(&mut self, rms: f64, peak: f64) {
    let n = self.samples.len().max(1) as f64;
    let cur = (self.samples.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    if cur < 1e-12 {
      return;
    }
    let mut g = rms / cur;
    let max = self.samples.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if max * g > peak {
      g = peak / max;
    }
    for v in &mut self.samples {
      *v = (*v * g).clamp(-1.0, 1.0);
    }
  }
}

/// A two-pole resonant band-pass filter.
pub struct Resonator {
  a1: f64,
  a2: f64,
  b0: f64,
  y1: f64,
  y2: f64,
}

impl Resonator {
  pub fn new(freq: f64, bandwidth: f64, sr: f64) -> Self {
    let r = (-std::f64::consts::PI * bandwidth / sr).exp();
    let theta = TAU * freq / sr;
    Resonator {
      a1: 2.0 * r * theta.cos(),
      a2: -r * r,
      b0: 1.0 - r,
      y1: 0.0,
      y2: 0.0,
    }
  }

  pub fn step(&mut self, x: f64) -> f64 {
    let y = self.b0 * x + self.a1 * self.y1 + self.a2 * self.y2;
    self.y2 = self.y1;
    self.y1 = y;
    y
  }
}

/// The colours of noise by spectral slope.
pub enum NoiseColor {
  White,
  Pink,
  Brown,
  Blue,
  Violet,
}

/// `n` samples of coloured noise, normalized to unit peak.
pub fn colored_noise(rng: &mut Rng, n: usize, color: &NoiseColor) -> Vec<f64> {
  let white: Vec<f64> = (0..n + 2).map(|_| rng.range(-1.0, 1.0)).collect();
  let mut out: Vec<f64> = match color {
    NoiseColor::White => white[..n].to_vec(),
    NoiseColor::Pink => {
      // Paul Kellet's economy pink filter.
      let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
      white[..n]
        .iter()
        .map(|&w| {
          b0 = 0.99765 * b0 + w * 0.0990460;
          b1 = 0.96300 * b1 + w * 0.2965164;
          b2 = 0.57000 * b2 + w * 1.0526913;
          b0 + b1 + b2 + w * 0.1848
        })
        .collect()
    }
    NoiseColor::Brown => {
      let mut acc = 0.0;
      white[..n]
        .iter()
        .map(|&w| {
          acc = (acc + 0.02 * w) * 0.998;
          acc
        })
        .collect()
    }
    NoiseColor::Blue => (0..n).map(|i| white[i + 1] - white[i]).collect(),
    NoiseColor::Violet => (0..n)
      .map(|i| white[i + 2] - 2.0 * white[i + 1] + white[i])
      .collect(),
  };
  let max = out.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
  for v in &mut out {
    *v /= max;
  }
  out
}
