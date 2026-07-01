//! Standard MIDI File (SMF) export for the computational-music objects.
//!
//! `Export["score.mid", MusicScore[…]]` and friends serialize a music object to
//! a type-1 Standard MIDI File, matching the layout Wolfram Language 15 emits:
//! 480 ticks per quarter note, one track per voice, and each track prefixed with
//! a program-change, C-major key signature, 4/4 time signature and a 120 BPM
//! tempo. Notes sound at velocity 127 with a one-tick articulation gap, and a
//! voice's notes fill a 4/4 measure — the leading notes are quarter notes and
//! the last note is lengthened to the bar line (three pitches become
//! quarter/quarter/half, two become quarter/dotted-half, and so on).

use crate::functions::music_ast::{music_pitch_midi, resolve_pitch_name};
use crate::syntax::Expr;

/// Ticks per quarter note (the SMF division), as used by Wolfram.
const TICKS_PER_BEAT: i64 = 480;
/// Beats in the default 4/4 measure.
const BEATS_PER_MEASURE: i64 = 4;

/// A single sounding note: its MIDI number and duration in ticks.
struct Note {
  midi: u8,
  ticks: i64,
}

/// Encode a value as an SMF variable-length quantity (7 bits per byte, MSB set
/// on every byte but the last).
fn write_varlen(out: &mut Vec<u8>, mut value: u32) {
  let mut buffer = vec![(value & 0x7f) as u8];
  value >>= 7;
  while value > 0 {
    buffer.push((value & 0x7f) as u8 | 0x80);
    value >>= 7;
  }
  out.extend(buffer.iter().rev());
}

/// Resolve one voice element — a bare pitch name `"A"`, a `MusicPitch`, or a
/// `MusicNote` — to its MIDI note number, defaulting to octave 4. Returns `None`
/// for anything that is not a pitch.
fn element_midi(expr: &Expr) -> Option<i128> {
  match expr {
    // A bare pitch-name string: resolve through a `MusicPitch` so an octaveless
    // name such as "A" lands in the default register (A4 = 69).
    Expr::String(_) => music_pitch_midi(&Expr::FunctionCall {
      name: "MusicPitch".to_string(),
      args: vec![expr.clone()].into(),
    }),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "MusicPitch" => music_pitch_midi(expr),
      // A note carries its pitch as the first argument or as the `"Pitch"` entry
      // of its canonical association.
      "MusicNote" => match args.first()? {
        Expr::Association(pairs) => {
          let pitch = pairs.iter().find_map(|(k, v)| match k {
            Expr::String(s) if s == "Pitch" => Some(v),
            _ => None,
          })?;
          element_midi(pitch)
        }
        other => element_midi(other),
      },
      _ => None,
    },
    // A raw MIDI integer.
    Expr::Integer(n) => Some(*n),
    _ => resolve_pitch_name(expr)
      .and_then(|n| crate::functions::music_ast::pitch_name_to_midi(&n)),
  }
}

/// The note list of one voice, given as `MusicVoice[{…}]`, a bare `List`, or a
/// `MusicMeasure[{…}]`. Durations follow the 4/4 measure-fill convention.
fn voice_notes(expr: &Expr) -> Option<Vec<Note>> {
  let items: Vec<Expr> = match expr {
    Expr::List(items) => items.to_vec(),
    Expr::FunctionCall { name, args }
      if name == "MusicVoice" || name == "MusicMeasure" =>
    {
      match args.first()? {
        Expr::List(items) => items.to_vec(),
        _ => return None,
      }
    }
    _ => return None,
  };
  let midis: Vec<u8> = items
    .iter()
    .filter_map(element_midi)
    .map(|m| m.clamp(0, 127) as u8)
    .collect();
  if midis.is_empty() {
    return None;
  }
  let n = midis.len() as i64;
  Some(
    midis
      .into_iter()
      .enumerate()
      .map(|(i, midi)| {
        // Every note but the last is a quarter; the last fills the bar.
        let beats = if (i as i64) < n - 1 {
          1
        } else {
          BEATS_PER_MEASURE - ((n - 1) % BEATS_PER_MEASURE)
        };
        Note {
          midi,
          ticks: beats * TICKS_PER_BEAT,
        }
      })
      .collect(),
  )
}

/// Serialize the events of one voice into an `MTrk` chunk on the given channel.
fn write_track(out: &mut Vec<u8>, channel: u8, notes: &[Note]) {
  let mut data: Vec<u8> = Vec::new();
  // Program change (acoustic grand, program 0) on this channel.
  data.extend([0x00, 0xC0 | channel, 0x00]);
  // C-major key signature.
  data.extend([0x00, 0xFF, 0x59, 0x02, 0x00, 0x00]);
  // 4/4 time signature: 4 beats, denominator 2^2, 24 MIDI clocks/click,
  // 8 thirty-seconds per quarter.
  data.extend([0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
  // 120 BPM tempo (500000 microseconds per quarter note).
  data.extend([0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]);

  // Note events. Only the first note-on carries an explicit status byte; every
  // following event reuses it via running status. A note sounds for one tick
  // less than its slot, leaving a one-tick gap before the next note.
  for (i, note) in notes.iter().enumerate() {
    let on_delta = if i == 0 { 0 } else { 1 };
    write_varlen(&mut data, on_delta);
    if i == 0 {
      data.push(0x90 | channel);
    }
    data.extend([note.midi, 0x7F]);
    write_varlen(&mut data, (note.ticks - 1).max(0) as u32);
    data.extend([note.midi, 0x00]); // running-status note-off (velocity 0)
  }
  // End of track.
  data.extend([0x00, 0xFF, 0x2F, 0x00]);

  out.extend(b"MTrk");
  out.extend((data.len() as u32).to_be_bytes());
  out.extend(data);
}

/// Serialize a computational-music object to Standard MIDI File bytes, or `None`
/// when it carries no playable voice. `MusicScore[{voice, …}]` becomes one track
/// per voice; a lone `MusicVoice`/`MusicMeasure`/list becomes a single track.
pub fn music_to_midi(expr: &Expr) -> Option<Vec<u8>> {
  // Gather the voices: a score's elements, or the object itself as one voice.
  let voices: Vec<Vec<Note>> = match expr {
    Expr::FunctionCall { name, args } if name == "MusicScore" => {
      match args.first()? {
        Expr::List(items) => items.iter().filter_map(voice_notes).collect(),
        _ => return None,
      }
    }
    _ => voice_notes(expr).into_iter().collect(),
  };
  if voices.is_empty() {
    return None;
  }

  let mut out: Vec<u8> = Vec::new();
  // MThd: type-1 file, one track per voice, 480 ticks per quarter note.
  out.extend(b"MThd");
  out.extend(6u32.to_be_bytes());
  out.extend(1u16.to_be_bytes()); // format 1
  out.extend((voices.len() as u16).to_be_bytes());
  out.extend((TICKS_PER_BEAT as u16).to_be_bytes());

  for (i, notes) in voices.iter().enumerate() {
    write_track(&mut out, i as u8, notes);
  }
  Some(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn voice(pitches: &[&str]) -> Expr {
    Expr::FunctionCall {
      name: "MusicVoice".to_string(),
      args: vec![Expr::List(
        pitches
          .iter()
          .map(|p| Expr::String(p.to_string()))
          .collect::<Vec<_>>()
          .into(),
      )]
      .into(),
    }
  }

  #[test]
  fn varlen_matches_smf_examples() {
    let mut out = Vec::new();
    write_varlen(&mut out, 0);
    assert_eq!(out, vec![0x00]);
    out.clear();
    write_varlen(&mut out, 479);
    assert_eq!(out, vec![0x83, 0x5F]);
    out.clear();
    write_varlen(&mut out, 959);
    assert_eq!(out, vec![0x87, 0x3F]);
    out.clear();
    write_varlen(&mut out, 1439);
    assert_eq!(out, vec![0x8B, 0x1F]);
  }

  #[test]
  fn three_note_voice_fills_the_measure() {
    // A,G,E -> quarter, quarter, half (480, 480, 960 ticks).
    let notes = voice_notes(&voice(&["A", "G", "E"])).unwrap();
    let ticks: Vec<i64> = notes.iter().map(|n| n.ticks).collect();
    assert_eq!(ticks, vec![480, 480, 960]);
    let midis: Vec<u8> = notes.iter().map(|n| n.midi).collect();
    assert_eq!(midis, vec![69, 67, 64]); // A4, G4, E4
  }

  #[test]
  fn two_and_four_note_voices_fill_the_measure() {
    let two: Vec<i64> = voice_notes(&voice(&["A", "G"]))
      .unwrap()
      .iter()
      .map(|n| n.ticks)
      .collect();
    assert_eq!(two, vec![480, 1440]); // quarter, dotted half
    let four: Vec<i64> = voice_notes(&voice(&["A", "G", "E", "C"]))
      .unwrap()
      .iter()
      .map(|n| n.ticks)
      .collect();
    assert_eq!(four, vec![480, 480, 480, 480]); // four quarters
  }

  #[test]
  fn score_two_voices_matches_wolfram_bytes() {
    // MusicScore[{MusicVoice[{"A","G","E"}], MusicVoice[{"F","E","C"}]}] must
    // serialize byte-for-byte to the reference file Wolfram exports.
    let score = Expr::FunctionCall {
      name: "MusicScore".to_string(),
      args: vec![Expr::List(
        vec![voice(&["A", "G", "E"]), voice(&["F", "E", "C"])].into(),
      )]
      .into(),
    };
    let bytes = music_to_midi(&score).unwrap();
    let expected: &[u8] = &[
      0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x01, 0x00, 0x02,
      0x01, 0xe0, 0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x32, 0x00, 0xc0,
      0x00, 0x00, 0xff, 0x59, 0x02, 0x00, 0x00, 0x00, 0xff, 0x58, 0x04, 0x04,
      0x02, 0x18, 0x08, 0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20, 0x00, 0x90,
      0x45, 0x7f, 0x83, 0x5f, 0x45, 0x00, 0x01, 0x43, 0x7f, 0x83, 0x5f, 0x43,
      0x00, 0x01, 0x40, 0x7f, 0x87, 0x3f, 0x40, 0x00, 0x00, 0xff, 0x2f, 0x00,
      0x4d, 0x54, 0x72, 0x6b, 0x00, 0x00, 0x00, 0x32, 0x00, 0xc1, 0x00, 0x00,
      0xff, 0x59, 0x02, 0x00, 0x00, 0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18,
      0x08, 0x00, 0xff, 0x51, 0x03, 0x07, 0xa1, 0x20, 0x00, 0x91, 0x41, 0x7f,
      0x83, 0x5f, 0x41, 0x00, 0x01, 0x40, 0x7f, 0x83, 0x5f, 0x40, 0x00, 0x01,
      0x3c, 0x7f, 0x87, 0x3f, 0x3c, 0x00, 0x00, 0xff, 0x2f, 0x00,
    ];
    assert_eq!(bytes, expected);
  }
}
