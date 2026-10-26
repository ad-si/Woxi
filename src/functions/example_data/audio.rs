//! `ExampleData[{"Audio", …}]` and `ExampleData[{"Sound", …}]`:
//! synthesized recordings.
//!
//! Each recording is synthesized from its name (see [`synth`]) at the
//! recorded sample rate, channel count and length, then scaled to the
//! recorded loudness (RMS and peak): instruments by additive synthesis at
//! their typical pitch (single notes, or scales for `"…Scale"` entries),
//! speech as formant-filtered pulse trains, birds as chirps, percussion as
//! decaying noise bursts, room impulse responses as decaying noise, and the
//! test signals (noise colours, square waves, sweeps, bursts) exactly.

use std::f64::consts::TAU;

use super::images::{cached_opt, file_path, numbers};
use super::synth::{Buffer, NoiseColor, colored_noise};
use super::{Collection, Entry, GENERATED_SOURCE};
use crate::functions::*;

/// `(sample rate, channels, frames, stored sample type)` from the `"Audio"`
/// skeleton `Sk["Audio", Quantity[rate, "Hertz"], channels,
/// Quantity[frames, "Samples"], type]`.
fn shape(entry: &Entry) -> Option<(f64, usize, usize, String)> {
  let Expr::FunctionCall { ref args, .. } = entry.skeleton("Audio")? else {
    return None;
  };
  let magnitude = |e: Option<&Expr>| -> Option<f64> {
    match e? {
      Expr::FunctionCall { name, args } if name == "Quantity" => {
        numbers(args.first()?).first().copied()
      }
      other => numbers(other).first().copied(),
    }
  };
  let rate = magnitude(args.get(1))?;
  let channels = magnitude(args.get(2))? as usize;
  let frames = magnitude(args.get(3))? as usize;
  let ty = match args.get(4) {
    Some(Expr::String(t)) => t.clone(),
    _ => "SignedInteger16".to_string(),
  };
  Some((rate, channels, frames, ty))
}

/// Instrument families: (pitch in Hz, partial amplitudes, envelope).
fn instrument(
  base: &str,
) -> Option<(f64, Vec<(f64, f64)>, (f64, f64, f64, f64))> {
  let partials = |amps: &[f64]| {
    amps
      .iter()
      .enumerate()
      .map(|(i, a)| ((i + 1) as f64, *a))
      .collect::<Vec<_>>()
  };
  let odd = |n: usize| {
    (0..n)
      .map(|i| ((2 * i + 1) as f64, 1.0 / (2 * i + 1) as f64))
      .collect::<Vec<_>>()
  };
  let saw = |n: usize| {
    (1..=n)
      .map(|i| (i as f64, 1.0 / i as f64))
      .collect::<Vec<_>>()
  };
  let sustain = (0.05, 0.1, 0.8, 0.15);
  let pluck = (0.005, 0.3, 0.0, 0.05);
  Some(match base {
    "Flute" | "AltoFlute" | "BassFlute" => (
      match base {
        "Flute" => 523.0,
        "AltoFlute" => 392.0,
        _ => 262.0,
      },
      partials(&[1.0, 0.3, 0.1, 0.05]),
      sustain,
    ),
    "Clarinet" | "BassClarinet" => (
      if base == "Clarinet" { 294.0 } else { 147.0 },
      odd(7),
      sustain,
    ),
    "Oboe" => (440.0, partials(&[0.6, 1.0, 0.8, 0.5, 0.3, 0.2]), sustain),
    "Bassoon" => (117.0, partials(&[0.5, 1.0, 0.7, 0.4, 0.3]), sustain),
    "AltoSaxophone" | "SopranoSaxophone" => (
      if base == "AltoSaxophone" {
        311.0
      } else {
        466.0
      },
      saw(8),
      sustain,
    ),
    "Trumpet" => (466.0, saw(10), (0.03, 0.1, 0.85, 0.1)),
    "FrenchHorn" => (175.0, partials(&[1.0, 0.7, 0.4, 0.2]), sustain),
    "TenorTrombone" | "BassTrombone" => (
      if base == "TenorTrombone" { 117.0 } else { 87.0 },
      saw(9),
      sustain,
    ),
    "Tuba" => (58.0, saw(6), sustain),
    "Violin" => (440.0, saw(12), sustain),
    "Viola" => (294.0, saw(10), sustain),
    "Cello" => (131.0, saw(10), sustain),
    "DoubleBass" => (65.0, saw(8), sustain),
    "CelloPizzicato" => (131.0, saw(6), pluck),
    "DoubleBassPizzicato" => (65.0, saw(5), pluck),
    "Piano" => (
      262.0,
      partials(&[1.0, 0.5, 0.3, 0.15, 0.1]),
      (0.005, 0.6, 0.2, 0.3),
    ),
    "OrganChord" => (
      262.0,
      partials(&[1.0, 0.8, 0.5, 0.5, 0.3]),
      (0.05, 0.05, 1.0, 0.1),
    ),
    _ => return None,
  })
}

/// The synthesized samples of channel `ch` (in `[-1, 1]` before scaling).
fn synthesize(entry: &Entry, rate: f64, frames: usize, ch: usize) -> Vec<f64> {
  let name = entry.name;
  let base = name.trim_end_matches(|c: char| c.is_ascii_digit());
  let mut rng = entry.rng(&format!("audio{ch}"));
  let mut b = Buffer::new(rate, frames);
  let dur = frames as f64 / rate;
  let (scale_base, is_scale) = match base.strip_suffix("Scale") {
    Some(s) => (s, true),
    None => (base, false),
  };
  if let Some((f0, partials, env)) = instrument(scale_base) {
    if is_scale {
      // A major scale up and down, one note per step.
      let steps = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0];
      let note = dur / steps.len() as f64;
      for (k, s) in steps.iter().enumerate() {
        let f = f0 * 2f64.powf(*s as f64 / 12.0);
        b.note(k as f64 * note, note * 0.95, f, &partials, env, 0.5, 0.004);
      }
    } else if scale_base == "OrganChord" {
      for r in [1.0, 1.26, 1.5] {
        b.note(0.0, dur, f0 * r, &partials, env, 0.3, 0.0);
      }
    } else {
      b.note(0.0, dur, f0, &partials, env, 0.6, 0.005);
    }
    return b.samples;
  }
  match base {
    "NoiseWhite" | "NoisePink" | "NoiseBrown" | "NoiseBlue" | "NoiseViolet" => {
      let color = match base {
        "NoisePink" => NoiseColor::Pink,
        "NoiseBrown" => NoiseColor::Brown,
        "NoiseBlue" => NoiseColor::Blue,
        "NoiseViolet" => NoiseColor::Violet,
        _ => NoiseColor::White,
      };
      return colored_noise(&mut rng, frames, &color);
    }
    "Square" | "Burst" => {
      let f: f64 = name.trim_start_matches(base).parse().unwrap_or(100.0);
      return (0..frames)
        .map(|i| {
          let t = i as f64 / rate;
          let s = if base == "Square" {
            if (f * t).fract() < 0.5 { 1.0 } else { -1.0 }
          } else {
            // Tone bursts: ten cycles on, ten off.
            let on = ((f * t) / 10.0).floor() as i64 % 2 == 0;
            if on { (TAU * f * t).sin() } else { 0.0 }
          };
          0.9 * s
        })
        .collect();
    }
    "LinearSweep" => b.chirp(0.0, dur, 20.0, rate / 2.0 * 0.9, 1.0),
    "TestIntermodulationDistortion" => {
      return (0..frames)
        .map(|i| {
          let t = i as f64 / rate;
          0.6 * (TAU * 60.0 * t).sin() + 0.15 * (TAU * 7000.0 * t).sin()
        })
        .collect();
    }
    "Apollo" | "MaleVoice" | "NoisyTalk" | "Crowd" => {
      let voices = if base == "Crowd" { 6 } else { 1 };
      for v in 0..voices {
        b.speech(&mut rng, v as f64 * 0.1, dur, 110.0 + 20.0 * v as f64, 1.0);
      }
      if base != "MaleVoice" {
        b.ambience(
          &mut rng,
          2000.0,
          0.3,
          0.3,
          if base == "Apollo" { 0.05 } else { 0.2 },
          None,
        );
      }
    }
    "FemaleVoice" | "Laughing" | "Scream" => {
      let f0 = match base {
        "Scream" => 600.0,
        "Laughing" => 280.0,
        _ => 210.0,
      };
      b.speech(&mut rng, 0.0, dur, f0, 1.0);
    }
    "Bird" | "BlackcapWarbler" => {
      let mut t = 0.0;
      while t < dur {
        let f0 = rng.range(2500.0, 5000.0);
        let len = rng.range(0.03, 0.12);
        b.chirp(t, len, f0, f0 * rng.range(0.6, 1.5), 0.8);
        t += len + rng.range(0.02, if base == "Bird" { 0.4 } else { 0.1 });
      }
    }
    "Bee" => {
      return (0..frames)
        .map(|i| {
          let t = i as f64 / rate;
          let f = 220.0 + 15.0 * (TAU * 3.0 * t).sin();
          let saw = 2.0 * (f * t).fract() - 1.0;
          saw * (0.6 + 0.4 * (TAU * 1.3 * t).sin().abs())
        })
        .collect();
    }
    "Cat" => b.speech(&mut rng, 0.1, dur * 0.8, 500.0, 1.0),
    "DogBark" => {
      let mut t = 0.05;
      while t < dur {
        b.burst(&mut rng, t, 0.04, 1.0, Some(350.0));
        b.speech(&mut rng, t, 0.15, 300.0, 0.5);
        t += rng.range(0.3, 0.7);
      }
    }
    "BalloonPop" | "Clapping" | "Walking" | "Drums" | "CrashCymbal"
    | "RollingCoin" => {
      let (spacing, decay, tone) = match base {
        "BalloonPop" => (dur * 2.0, 0.02, None),
        "Clapping" => (0.12, 0.015, None),
        "Walking" => (0.55, 0.03, Some(80.0)),
        "Drums" => (0.25, 0.08, Some(60.0)),
        "CrashCymbal" => (dur * 2.0, 0.6, None),
        _ => (0.08, 0.01, Some(3000.0)),
      };
      let mut t = 0.02;
      while t < dur {
        b.burst(&mut rng, t, decay, 1.0, tone);
        t += spacing * rng.range(0.7, 1.3);
      }
    }
    "ChurchBell" | "ThaiBells" | "Sword" => {
      let f0 = match base {
        "ChurchBell" => 220.0,
        "ThaiBells" => 1200.0,
        _ => 2400.0,
      };
      // Inharmonic bell partials.
      let partials = [
        (0.5, 0.6),
        (1.0, 1.0),
        (1.19, 0.6),
        (1.5, 0.5),
        (2.0, 0.4),
        (2.5, 0.2),
        (2.67, 0.25),
        (3.0, 0.15),
      ];
      let mut t = 0.0;
      let gap = if base == "ThaiBells" { 0.8 } else { dur };
      while t < dur {
        b.note(
          t,
          (dur - t).min(4.0),
          f0,
          &partials,
          (0.002, 2.0, 0.0, 0.1),
          0.6,
          0.0,
        );
        t += gap;
      }
    }
    "CreakyDoor" => {
      let mut t = 0.0;
      while t < dur {
        let len = rng.range(0.2, 0.6);
        b.chirp(
          t,
          len,
          rng.range(300.0, 600.0),
          rng.range(500.0, 1200.0),
          0.6,
        );
        t += len + rng.range(0.05, 0.3);
      }
      b.ambience(&mut rng, 1500.0, 7.0, 0.8, 0.2, None);
    }
    "PowerSupply" => b.ambience(&mut rng, 4000.0, 0.1, 0.05, 0.1, Some(60.0)),
    "FogHorn" => b.note(
      0.1,
      dur - 0.2,
      90.0,
      &[(1.0, 1.0), (2.0, 0.6), (3.0, 0.4), (4.0, 0.2)],
      (0.3, 0.2, 0.9, 0.5),
      0.8,
      0.0,
    ),
    "SubwayTrain" | "JetSound" => b.ambience(
      &mut rng,
      if base == "JetSound" { 3000.0 } else { 600.0 },
      0.15,
      0.5,
      1.0,
      Some(if base == "JetSound" { 900.0 } else { 45.0 }),
    ),
    "Water" | "Wind" => b.ambience(
      &mut rng,
      if base == "Water" { 2500.0 } else { 400.0 },
      0.2,
      0.7,
      1.0,
      None,
    ),
    _ if base.starts_with("IR") => {
      // A room impulse response: a click, then decaying noise.
      let tail = dur * 0.25;
      return (0..frames)
        .map(|i| {
          let t = i as f64 / rate;
          if i == 0 {
            1.0
          } else {
            rng.range(-1.0, 1.0) * (-t / tail).exp() * 0.5
          }
        })
        .collect();
    }
    _ => b.ambience(&mut rng, 1000.0, 0.3, 0.5, 1.0, None),
  }
  b.samples
}

/// The recording's channels, scaled to the recorded RMS and peak, and
/// quantized to 16 bits like the files they stand in for.
fn channels(entry: &Entry) -> Option<(f64, Vec<Vec<f64>>, String)> {
  let (rate, n_ch, frames, ty) = shape(entry)?;
  let stats = entry
    .stat("@Stats")
    .map(|s| numbers(&s))
    .unwrap_or_default();
  let (rms, peak) = (
    stats.first().copied().unwrap_or(0.1),
    stats.get(1).copied().unwrap_or(0.9),
  );
  let mut chans = Vec::with_capacity(n_ch);
  for ch in 0..n_ch {
    let mut buf = Buffer {
      sr: rate,
      samples: synthesize(entry, rate, frames, ch),
    };
    buf.samples.resize(frames, 0.0);
    buf.normalize(rms, peak);
    let q = 32768.0;
    chans.push(buf.samples.iter().map(|v| (v * q).round() / q).collect());
  }
  Some((rate, chans, ty))
}

fn channel_lists(chans: &[Vec<f64>]) -> Expr {
  Expr::List(
    chans
      .iter()
      .map(|c| {
        Expr::List(c.iter().map(|v| Expr::Real(*v)).collect::<Vec<_>>().into())
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

fn rate_expr(rate: f64) -> Expr {
  if rate.fract() == 0.0 {
    Expr::Integer(rate as i128)
  } else {
    Expr::Real(rate)
  }
}

/// `Audio[{{…}, …}, type, SampleRate -> r]`.
fn audio(entry: &Entry) -> Option<Expr> {
  cached_opt(format!("{}/{}/audio", entry.collection, entry.name), || {
    let (rate, chans, ty) = channels(entry)?;
    // Stored like Wolfram's: integer samples in a NumericArray.
    let data = match crate::functions::audio_ast::integer_sample_bits(&ty) {
      Some(bits) => call(
        "NumericArray",
        vec![
          Expr::List(
            chans
              .iter()
              .map(|c| {
                Expr::List(
                  c.iter()
                    .map(|&v| {
                      Expr::Integer(
                        crate::functions::audio_ast::integer_sample(v, bits)
                          .into(),
                      )
                    })
                    .collect::<Vec<_>>()
                    .into(),
                )
              })
              .collect::<Vec<_>>()
              .into(),
          ),
          str_expr(&format!("Integer{bits}")),
        ],
      ),
      None => channel_lists(&chans),
    };
    Some(call(
      "Audio",
      vec![
        data,
        str_expr(&ty),
        rule_expr(id_expr("SampleRate"), rate_expr(rate)),
      ],
    ))
  })
}

fn data(entry: &Entry) -> Option<Expr> {
  let (_, chans, _) = channels(entry)?;
  Some(channel_lists(&chans))
}

fn cached_data(entry: &Entry) -> Option<Expr> {
  cached_opt(format!("{}/{}/data", entry.collection, entry.name), || {
    data(entry)
  })
}

fn sampled_sound_list(entry: &Entry) -> Option<Expr> {
  let rate = shape(entry)?.0;
  Some(call(
    "SampledSoundList",
    vec![cached_data(entry)?, rate_expr(rate)],
  ))
}

pub struct Recordings;

impl Collection for Recordings {
  fn value(&self, entry: &Entry, property: Option<&str>) -> Option<Expr> {
    let wav =
      || file_path(entry, "wav", || audio(entry).unwrap_or(Expr::Integer(0)));
    Some(match property? {
      "Audio" => audio(entry)?,
      "Data" => cached_data(entry)?,
      "FilePath" => wav(),
      "AudioFile" => {
        let rate = shape(entry)?.0;
        let path = wav();
        call(
          "Audio",
          vec![
            path,
            str_expr("Real32"),
            rule_expr(id_expr("Appearance"), id_expr("Automatic")),
            rule_expr(id_expr("AudioOutputDevice"), id_expr("Automatic")),
            rule_expr(id_expr("SampleRate"), rate_expr(rate)),
            rule_expr(id_expr("SoundVolume"), Expr::Integer(1)),
          ],
        )
      }
      "SampledSoundList" => sampled_sound_list(entry)?,
      "Sound" => call("Sound", vec![sampled_sound_list(entry)?]),
      // Text the original carries is written by Woxi.
      "Description" if entry.skeleton("Description").is_some() => str_expr(
        &format!("A synthesized stand-in for the recording {}.", entry.name),
      ),
      "Source" if entry.skeleton("Source").is_some() => {
        str_expr(GENERATED_SOURCE)
      }
      _ => return None,
    })
  }
}
