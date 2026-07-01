//! Computational music objects from Wolfram Language 15.0
//! (the [ComputationalMusic guide](https://reference.wolfram.com/language/guide/ComputationalMusic.html)).
//!
//! Woxi represents the music *objects* — notes, chords, rests, pitches,
//! durations, scales, measures, voices, scores, … — as canonical symbolic
//! expressions, the same way `Sound` and `SoundNote` already work. That lets
//! them be constructed, nested, pattern-matched, and classified with
//! `MusicObjectQ`.
//!
//! `MusicPitch` additionally performs a genuine computation: it canonicalizes
//! any of its documented input forms to the named-pitch object
//! `MusicPitch["name"]`. Following the
//! [`MusicPitch` reference page](https://reference.wolfram.com/language/ref/MusicPitch.html),
//! these forms all denote the same pitch and therefore canonicalize alike:
//!
//! ```text
//! MusicPitch[55]                      (* MIDI note number,   middle C = 60 *)
//! MusicPitch[Quantity[200, "Hertz"]]  (* frequency,          A4 = 440 Hz   *)
//! MusicPitch[SoundNote[-5]]           (* SoundNote pitch,    middle C = 0  *)
//! MusicPitch[MusicNote["G3"]]         (* pitch of a note                   *)
//! ```
//!
//! all give `MusicPitch["G3"]`. The conversions use the standard convention
//! where middle C is MIDI 60 / C4 and A4 (MIDI 69) is 440 Hz.

use crate::syntax::Expr;

/// Heads that the Wolfram Language classifies as music objects — the "Music
/// Events", "Music Properties", and "Music Containers" of the
/// ComputationalMusic guide. `MusicObjectQ` returns `True` for these.
pub const MUSIC_OBJECT_HEADS: &[&str] = &[
  // Music Events
  "MusicNote",
  "MusicRest",
  "MusicChord",
  // Music Properties
  "MusicDuration",
  "MusicPitch",
  "MusicInterval",
  "MusicKeySignature",
  "MusicTimeSignature",
  "MusicScale",
  "MusicTempo",
  // Music Containers
  "MusicMeasure",
  "MusicVoice",
  "MusicScore",
];

/// `MusicObjectQ[expr]` — `True` when `expr` is a music object, i.e. an
/// expression whose head is one of [`MUSIC_OBJECT_HEADS`]; `False` otherwise.
pub fn music_object_q(args: &[Expr]) -> Expr {
  let is_object = matches!(
    args,
    [Expr::FunctionCall { name, .. }]
      if MUSIC_OBJECT_HEADS.contains(&name.as_str())
  );
  Expr::Identifier(if is_object { "True" } else { "False" }.to_string())
}

/// Chromatic note names (sharp spelling) indexed by pitch class `0..=11`,
/// with `C = 0`.
const PITCH_CLASS_NAMES: [&str; 12] = [
  "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Convert a MIDI note number to its scientific-pitch name, e.g. `60 -> "C4"`,
/// `69 -> "A4"`, `0 -> "C-1"`, `61 -> "C#4"`. Middle C (MIDI 60) is C4. Uses
/// floor division so it is well defined for every integer, not just the
/// nominal 0..127 MIDI range.
pub fn midi_to_pitch_name(midi: i128) -> String {
  let pitch_class = midi.rem_euclid(12) as usize;
  let octave = midi.div_euclid(12) - 1;
  format!("{}{}", PITCH_CLASS_NAMES[pitch_class], octave)
}

/// Parse a scientific-pitch name such as `"C4"`, `"A4"`, `"F#3"`, `"Eb5"`,
/// `"C-1"` back into its MIDI note number — the inverse of
/// [`midi_to_pitch_name`]. Accepts both sharp (`#`) and flat (`b`) accidentals
/// and a negative octave. Returns `None` for anything that is not a pitch
/// name (so callers can leave such specifications symbolic).
pub fn pitch_name_to_midi(name: &str) -> Option<i128> {
  let bytes = name.as_bytes();
  let letter = *bytes.first()?;
  // Base semitone of the natural note, C..B.
  let base = match letter.to_ascii_uppercase() {
    b'C' => 0,
    b'D' => 2,
    b'E' => 4,
    b'F' => 5,
    b'G' => 7,
    b'A' => 9,
    b'B' => 11,
    _ => return None,
  };
  let mut idx = 1;
  let mut accidental: i128 = 0;
  while let Some(&c) = bytes.get(idx) {
    match c {
      b'#' => accidental += 1,
      b'b' => accidental -= 1,
      _ => break,
    }
    idx += 1;
  }
  // The remainder must be the (possibly negative) octave number.
  let octave: i128 = name.get(idx..)?.parse().ok()?;
  Some((octave + 1) * 12 + base + accidental)
}

/// Convert a frequency in Hertz to the nearest MIDI note number in 12-tone
/// equal temperament, with A4 (MIDI 69) tuned to 440 Hz. Returns `None` for
/// non-positive or non-finite frequencies, which have no pitch.
pub fn frequency_to_midi(freq: f64) -> Option<i128> {
  if freq <= 0.0 || !freq.is_finite() {
    return None;
  }
  Some((69.0 + 12.0 * (freq / 440.0).log2()).round() as i128)
}

/// Extract the numeric magnitude of a `Quantity`/number argument as an `f64`.
fn magnitude_as_f64(expr: &Expr) -> Option<f64> {
  match expr {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  }
}

/// Resolve any documented `MusicPitch` input form to its canonical pitch-name
/// string, e.g. `Integer(55) -> "G3"`. Returns `None` when the specification
/// is not a recognized pitch form.
fn resolve_pitch_name(spec: &Expr) -> Option<String> {
  match spec {
    // MIDI note number (middle C = 60).
    Expr::Integer(midi) => Some(midi_to_pitch_name(*midi)),
    // A pitch name such as "C4" is already canonical.
    Expr::String(name) => Some(name.clone()),
    Expr::FunctionCall { name, args } => match name.as_str() {
      // MusicPitch[Quantity[f, "Hertz"]] — a frequency.
      "Quantity" if args.len() == 2 => {
        let is_hertz = matches!(
          &args[1],
          Expr::String(u) | Expr::Identifier(u) if u == "Hertz" || u == "Hz"
        );
        if !is_hertz {
          return None;
        }
        frequency_to_midi(magnitude_as_f64(&args[0])?).map(midi_to_pitch_name)
      }
      // MusicPitch[SoundNote[p, ...]] — SoundNote numbers pitches relative to
      // middle C (0), so the MIDI number is p + 60.
      "SoundNote" => match args.first()? {
        Expr::Integer(p) => Some(midi_to_pitch_name(p + 60)),
        Expr::String(s) => Some(s.clone()),
        _ => None,
      },
      // MusicPitch[MusicNote[pitch, ...]] / MusicPitch[MusicPitch[...]] — take
      // the pitch of the note, which uses MIDI/pitch-name conventions.
      "MusicNote" | "MusicPitch" => resolve_pitch_name(args.first()?),
      _ => None,
    },
    _ => None,
  }
}

/// `MusicPitch[spec]`. Canonicalizes any of the documented pitch
/// specifications — MIDI integer, `Quantity` frequency, `SoundNote`,
/// `MusicNote`, or a nested `MusicPitch` — to the named-pitch form
/// `MusicPitch["name"]`. A bare pitch string such as `"C4"` is already the
/// canonical symbolic object, so `None` is returned and the caller leaves it
/// unevaluated.
pub fn music_pitch(args: &[Expr]) -> Option<Expr> {
  let [spec] = args else {
    return None;
  };
  // A bare pitch name is already canonical; keep it unevaluated to avoid
  // pointless rewriting (MusicPitch["C4"] -> MusicPitch["C4"]).
  if matches!(spec, Expr::String(_)) {
    return None;
  }
  let name = resolve_pitch_name(spec)?;
  Some(Expr::FunctionCall {
    name: "MusicPitch".to_string(),
    args: vec![Expr::String(name)].into(),
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  // `Expr` does not implement `PartialEq`, so results are checked by matching
  // on structure / rendering to a string.

  fn head_with_c4(name: &str) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: vec![Expr::String("C4".to_string())].into(),
    }
  }

  fn is_true(expr: &Expr) -> bool {
    matches!(expr, Expr::Identifier(s) if s == "True")
  }

  fn is_false(expr: &Expr) -> bool {
    matches!(expr, Expr::Identifier(s) if s == "False")
  }

  #[test]
  fn music_object_q_recognizes_every_head() {
    for head in MUSIC_OBJECT_HEADS {
      assert!(
        is_true(&music_object_q(&[head_with_c4(head)])),
        "{head} should be a music object",
      );
    }
  }

  #[test]
  fn music_object_q_rejects_non_objects() {
    assert!(is_false(&music_object_q(&[Expr::Integer(3)])));
    assert!(is_false(&music_object_q(&[Expr::FunctionCall {
      name: "List".to_string(),
      args: vec![].into(),
    }])));
    // MusicPlot/MusicTransform/MusicMeasurements are operations, not objects.
    assert!(is_false(&music_object_q(&[Expr::FunctionCall {
      name: "MusicPlot".to_string(),
      args: vec![].into(),
    }])));
  }

  #[test]
  fn midi_to_pitch_name_known_values() {
    assert_eq!(midi_to_pitch_name(60), "C4");
    assert_eq!(midi_to_pitch_name(69), "A4");
    assert_eq!(midi_to_pitch_name(61), "C#4");
    assert_eq!(midi_to_pitch_name(12), "C0");
    assert_eq!(midi_to_pitch_name(0), "C-1");
    assert_eq!(midi_to_pitch_name(127), "G9");
    // Below the nominal MIDI range still resolves via floor division.
    assert_eq!(midi_to_pitch_name(-1), "B-2");
  }

  #[test]
  fn music_pitch_canonicalizes_midi_integer() {
    let result = music_pitch(&[Expr::Integer(60)]);
    match &result {
      Some(Expr::FunctionCall { name, args }) => {
        assert_eq!(name, "MusicPitch");
        assert!(matches!(&args[..], [Expr::String(s)] if s == "C4"));
      }
      other => panic!("expected MusicPitch[\"C4\"], got {other:?}"),
    }
  }

  #[test]
  fn music_pitch_leaves_string_spec_alone() {
    assert!(music_pitch(&[Expr::String("C4".to_string())]).is_none());
  }

  #[test]
  fn pitch_name_to_midi_inverts_midi_to_pitch_name() {
    // Round-trips for every MIDI number the naming scheme covers.
    for midi in -24..=127 {
      assert_eq!(
        pitch_name_to_midi(&midi_to_pitch_name(midi)),
        Some(midi),
        "round-trip failed for MIDI {midi}",
      );
    }
  }

  #[test]
  fn pitch_name_to_midi_accepts_flats_and_sharps() {
    assert_eq!(pitch_name_to_midi("C4"), Some(60));
    assert_eq!(pitch_name_to_midi("A4"), Some(69));
    assert_eq!(pitch_name_to_midi("C#4"), Some(61));
    // Db4 is enharmonically the same pitch as C#4.
    assert_eq!(pitch_name_to_midi("Db4"), Some(61));
    assert_eq!(pitch_name_to_midi("C-1"), Some(0));
    assert_eq!(pitch_name_to_midi("Cb4"), Some(59));
    // Not pitch names.
    assert_eq!(pitch_name_to_midi("H4"), None);
    assert_eq!(pitch_name_to_midi(""), None);
    assert_eq!(pitch_name_to_midi("C"), None);
  }

  #[test]
  fn frequency_to_midi_known_values() {
    assert_eq!(frequency_to_midi(440.0), Some(69)); // A4
    assert_eq!(frequency_to_midi(261.6255653), Some(60)); // middle C
    assert_eq!(frequency_to_midi(200.0), Some(55)); // ~G3, per the docs
    // Non-positive / non-finite frequencies have no pitch.
    assert_eq!(frequency_to_midi(0.0), None);
    assert_eq!(frequency_to_midi(-100.0), None);
    assert_eq!(frequency_to_midi(f64::INFINITY), None);
  }

  fn expect_music_pitch(spec: Expr, name: &str) {
    match &music_pitch(&[spec]) {
      Some(Expr::FunctionCall { name: head, args }) => {
        assert_eq!(head, "MusicPitch");
        assert!(
          matches!(&args[..], [Expr::String(s)] if s == name),
          "expected MusicPitch[{name:?}], got args {args:?}",
        );
      }
      other => panic!("expected MusicPitch[{name:?}], got {other:?}"),
    }
  }

  fn func(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: args.into(),
    }
  }

  #[test]
  fn music_pitch_canonicalizes_frequency() {
    expect_music_pitch(
      func(
        "Quantity",
        vec![Expr::Integer(440), Expr::String("Hertz".into())],
      ),
      "A4",
    );
    expect_music_pitch(
      func(
        "Quantity",
        vec![Expr::Integer(200), Expr::String("Hertz".into())],
      ),
      "G3",
    );
    // Identifier unit spelling and the "Hz" abbreviation are both accepted.
    expect_music_pitch(
      func(
        "Quantity",
        vec![Expr::Real(440.0), Expr::Identifier("Hertz".into())],
      ),
      "A4",
    );
    // A non-frequency Quantity has no pitch interpretation.
    assert!(
      music_pitch(&[func(
        "Quantity",
        vec![Expr::Integer(440), Expr::String("Meters".into())]
      )])
      .is_none()
    );
  }

  #[test]
  fn music_pitch_canonicalizes_soundnote() {
    // SoundNote numbers pitches relative to middle C (0), so 0 -> C4.
    expect_music_pitch(func("SoundNote", vec![Expr::Integer(0)]), "C4");
    expect_music_pitch(func("SoundNote", vec![Expr::Integer(-5)]), "G3");
    expect_music_pitch(
      func("SoundNote", vec![Expr::String("A4".into())]),
      "A4",
    );
  }

  #[test]
  fn music_pitch_extracts_note_pitch() {
    expect_music_pitch(
      func("MusicNote", vec![Expr::String("G3".into())]),
      "G3",
    );
    expect_music_pitch(
      func(
        "MusicNote",
        vec![func("MusicPitch", vec![Expr::Integer(55)])],
      ),
      "G3",
    );
    // Nested MusicPitch resolves through to the underlying pitch.
    expect_music_pitch(func("MusicPitch", vec![Expr::Integer(60)]), "C4");
  }
}
