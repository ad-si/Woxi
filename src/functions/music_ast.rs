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

use crate::syntax::{
  BinaryOperator, Expr, UnaryOperator, bool_expr, unevaluated,
};

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

/// `True` when `expr` is a non-empty `List` whose every element is a music
/// object (a head in [`MUSIC_OBJECT_HEADS`]). Such a list is a sequence of
/// musical events and is drawn as a single staff rather than as a bracketed
/// expression dump.
pub fn is_music_object_list(expr: &Expr) -> bool {
  matches!(expr, Expr::List(items)
    if !items.is_empty()
      && items.iter().all(|it| matches!(it,
        Expr::FunctionCall { name, .. }
          if MUSIC_OBJECT_HEADS.contains(&name.as_str()))))
}

/// `MusicObjectQ[expr]` — `True` when `expr` is a music object, i.e. an
/// expression whose head is one of [`MUSIC_OBJECT_HEADS`]; `False` otherwise.
pub fn music_object_q(args: &[Expr]) -> Expr {
  let is_object = matches!(
    args,
    [Expr::FunctionCall { name, .. }]
      if MUSIC_OBJECT_HEADS.contains(&name.as_str())
  );
  bool_expr(is_object)
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
fn frequency_to_midi(freq: f64) -> Option<i128> {
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
pub fn resolve_pitch_name(spec: &Expr) -> Option<String> {
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
      // MusicPitch[MusicNote[pitch, ...]] — take the pitch of the note. A
      // canonical note carries its pitch under the `"Pitch"` key; a raw
      // `MusicNote[spec, …]` carries it as the first argument.
      "MusicNote" => match args.first()? {
        Expr::Association(pairs) => {
          resolve_pitch_name(assoc_get(pairs, "Pitch")?)
        }
        other => resolve_pitch_name(other),
      },
      // MusicPitch[MusicPitch[...]] — resolve through to the inner pitch.
      "MusicPitch" => resolve_pitch_name(args.first()?),
      _ => None,
    },
    // The canonical `MusicPitch` association form,
    // `<|"Accidental" -> a, ["Octave" -> o,] "Key" -> k[, "MIDINumber" -> n]|>`.
    // A `MIDINumber`-only association (from a MIDI/frequency/SoundNote
    // specification) has no spelling of its own and reads straight off the
    // MIDI number.
    Expr::Association(pairs) => {
      if assoc_get(pairs, "Key").is_none() {
        return match assoc_get(pairs, "MIDINumber")? {
          Expr::Integer(m) => Some(midi_to_pitch_name(*m)),
          _ => None,
        };
      }
      let key = match assoc_get(pairs, "Key")? {
        Expr::String(s) => s.as_bytes().first()?.to_ascii_uppercase() as char,
        _ => return None,
      };
      let accidental = match assoc_get(pairs, "Accidental") {
        Some(Expr::Integer(n)) => *n,
        None => 0,
        _ => return None,
      };
      // Prefer an explicit octave; otherwise recover it from a MIDINumber, and
      // fall back to WL's default register (octave 4).
      let octave = match assoc_get(pairs, "Octave") {
        Some(Expr::Integer(n)) => *n,
        _ => match assoc_get(pairs, "MIDINumber") {
          Some(Expr::Integer(m)) => {
            let base = DIATONIC_BASE_SEMITONE
              [letter_diatonic_index(key as u8)? as usize];
            (m - base - accidental).div_euclid(12) - 1
          }
          _ => 4,
        },
      };
      let mut name = key.to_string();
      match accidental.cmp(&0) {
        std::cmp::Ordering::Greater => {
          name.push_str(&"#".repeat(accidental as usize))
        }
        std::cmp::Ordering::Less => {
          name.push_str(&"b".repeat((-accidental) as usize))
        }
        std::cmp::Ordering::Equal => {}
      }
      name.push_str(&octave.to_string());
      Some(name)
    }
    _ => None,
  }
}

/// The canonical MIDI-numbered pitch object `MusicPitch[<|"MIDINumber" -> n|>]`
/// that a MIDI / frequency / `SoundNote` specification resolves to. Unlike a
/// spelled name, these forms fix no letter spelling, so the Wolfram Language
/// keeps just the MIDI number.
fn midi_number_pitch(midi: i128) -> Expr {
  music_assoc("MusicPitch", vec![("MIDINumber", Expr::Integer(midi))])
}

/// `MusicPitch[spec]`. Canonicalizes any of the documented pitch
/// specifications to its association-form pitch object: a spelled name keeps
/// its letter/accidental (plus `Octave`/`Name` when an octave is given), while
/// a MIDI integer, `Quantity` frequency, or `SoundNote` — which fix no
/// spelling — keep only their `MIDINumber`. An association argument is already
/// the canonical object and is left untouched (`None`), which keeps the result
/// idempotent under re-evaluation.
pub fn music_pitch(args: &[Expr]) -> Option<Expr> {
  let [spec] = args else {
    return None;
  };
  match spec {
    // A spelled pitch name: `<|Accidental, [Octave,] Key[, Name]|>`.
    Expr::String(name) => {
      let (letter, accidental, octave) = parse_pitch_spelled(name)?;
      Some(note_pitch_object(letter, accidental, octave))
    }
    // MIDI note number (middle C = 60): no spelling, keep the number.
    Expr::Integer(midi) => Some(midi_number_pitch(*midi)),
    // Already canonical (any association form).
    Expr::Association(_) => None,
    Expr::FunctionCall { name, args: inner } => match name.as_str() {
      // MusicPitch[Quantity[f, "Hertz"]] — a frequency.
      "Quantity" if inner.len() == 2 => {
        let is_hertz = matches!(
          &inner[1],
          Expr::String(u) | Expr::Identifier(u) if u == "Hertz" || u == "Hz"
        );
        if !is_hertz {
          return None;
        }
        frequency_to_midi(magnitude_as_f64(&inner[0])?).map(midi_number_pitch)
      }
      // MusicPitch[SoundNote[p, …]] — SoundNote numbers pitches relative to
      // middle C (0), so the MIDI number is p + 60.
      "SoundNote" => match inner.first()? {
        Expr::Integer(p) => Some(midi_number_pitch(p + 60)),
        Expr::String(s) => pitch_name_to_midi(s).map(midi_number_pitch),
        _ => None,
      },
      // MusicPitch[MusicNote[…]] — the pitch of the note. A canonical note
      // carries its pitch object under the `"Pitch"` key; a raw
      // `MusicNote[spec, …]` resolves its first argument.
      "MusicNote" => match inner.first()? {
        Expr::Association(pairs) => assoc_get(pairs, "Pitch").cloned(),
        other => resolve_pitch_object(other),
      },
      // MusicPitch[MusicPitch[…]] — already a pitch object.
      "MusicPitch" => Some(spec.clone()),
      _ => None,
    },
    _ => None,
  }
}

// ---------------------------------------------------------------------------
// MusicPitch arithmetic
// ---------------------------------------------------------------------------
//
// Wolfram Language 15 lets `MusicPitch` objects be added and subtracted, e.g.
//
// ```text
// MusicPitch["Bb"] + MusicPitch["A#"] - MusicPitch["C"]
// (* MusicPitch[<|"Accidental" -> 1, "Key" -> "G", "MIDINumber" -> 80|>] *)
// ```
//
// A pitch is combined along two independent linear axes: its *diatonic staff
// position* (which letter, in which register) and its *MIDI number* (the actual
// sounding semitone). Both axes are summed with the summands' signs, then the
// result is decoded back into a (Key letter, Accidental, MIDINumber) triple.
// Summing the diatonic position — rather than re-spelling the final MIDI number
// — is what makes `Bb + A# - C` land on the *letter* `G` (B + A - C = 6 + 5 - 0
// = 11 ≡ G, one octave up) instead of an enharmonic `Ab`.
//
// Octaveless names such as `"C"` or `"Bb"` default to octave 4, the Wolfram
// Language default register for `MusicPitch`.

/// Diatonic scale-degree letters indexed `0..=6` (`C` … `B`).
const DIATONIC_LETTERS: [&str; 7] = ["C", "D", "E", "F", "G", "A", "B"];

/// Semitone of the natural note for each diatonic letter index `0..=6`.
const DIATONIC_BASE_SEMITONE: [i128; 7] = [0, 2, 4, 5, 7, 9, 11];

/// Diatonic scale-degree index (`C=0 … B=6`) of a note letter, or `None` for a
/// byte that is not a note letter.
fn letter_diatonic_index(letter: u8) -> Option<i128> {
  Some(match letter.to_ascii_uppercase() {
    b'C' => 0,
    b'D' => 1,
    b'E' => 2,
    b'F' => 3,
    b'G' => 4,
    b'A' => 5,
    b'B' => 6,
    _ => return None,
  })
}

/// Parse a pitch name such as `"Bb"`, `"A#"`, `"C"`, `"G3"`, `"C#4"` into its
/// `(diatonic index, accidental, octave)` components. A missing octave defaults
/// to 4. Returns `None` for anything that is not a pitch name.
fn parse_pitch_parts(name: &str) -> Option<(i128, i128, i128)> {
  let bytes = name.as_bytes();
  let diatonic = letter_diatonic_index(*bytes.first()?)?;
  let mut idx = 1;
  let mut accidental = 0i128;
  while let Some(&c) = bytes.get(idx) {
    match c {
      b'#' => accidental += 1,
      b'b' => accidental -= 1,
      _ => break,
    }
    idx += 1;
  }
  let octave = if idx >= name.len() {
    4
  } else {
    name.get(idx..)?.parse().ok()?
  };
  Some((diatonic, accidental, octave))
}

/// Combine `(diatonic index, accidental, octave)` into the two linear axes used
/// for pitch arithmetic: the absolute diatonic staff position and the MIDI
/// number.
fn parts_to_axes(
  diatonic: i128,
  accidental: i128,
  octave: i128,
) -> (i128, i128) {
  let position = octave * 7 + diatonic;
  let midi =
    (octave + 1) * 12 + DIATONIC_BASE_SEMITONE[diatonic as usize] + accidental;
  (position, midi)
}

/// Resolve a single `MusicPitch[...]` object to its `(diatonic position, MIDI
/// number)` axes. Handles the named form (`MusicPitch["Bb"]`), the canonical
/// association form produced by pitch arithmetic, and a bare MIDI integer.
fn music_pitch_axes(expr: &Expr) -> Option<(i128, i128)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicPitch" || args.len() != 1 {
    return None;
  }
  match &args[0] {
    Expr::String(name) => {
      let (diatonic, accidental, octave) = parse_pitch_parts(name)?;
      Some(parts_to_axes(diatonic, accidental, octave))
    }
    // A bare MIDI number would already have been canonicalized to a name by
    // `music_pitch`, but resolve it directly for robustness.
    Expr::Integer(midi) => {
      let (diatonic, accidental, octave) =
        parse_pitch_parts(&midi_to_pitch_name(*midi))?;
      Some(parts_to_axes(diatonic, accidental, octave))
    }
    // A canonical pitch association. With a `Key` spelling, the octave comes
    // from the `MIDINumber` when present (exact), else from an explicit
    // `Octave`, else WL's default register 4. A `MIDINumber`-only association
    // (a MIDI/frequency/SoundNote pitch) is spelled straight from its number.
    Expr::Association(pairs) => {
      let lookup = |key: &str| assoc_get(pairs, key);
      let Some(key) = lookup("Key") else {
        return match lookup("MIDINumber")? {
          Expr::Integer(m) => {
            let (diatonic, accidental, octave) =
              parse_pitch_parts(&midi_to_pitch_name(*m))?;
            Some(parts_to_axes(diatonic, accidental, octave))
          }
          _ => None,
        };
      };
      let key = match key {
        Expr::String(s) => s,
        _ => return None,
      };
      let diatonic = letter_diatonic_index(*key.as_bytes().first()?)?;
      let accidental = match lookup("Accidental") {
        Some(Expr::Integer(n)) => *n,
        None => 0,
        _ => return None,
      };
      match lookup("MIDINumber") {
        Some(Expr::Integer(midi)) => {
          let base = DIATONIC_BASE_SEMITONE[diatonic as usize];
          let octave = (midi - base - accidental).div_euclid(12) - 1;
          Some((octave * 7 + diatonic, *midi))
        }
        Some(_) => None,
        None => {
          let octave = match lookup("Octave") {
            Some(Expr::Integer(o)) => *o,
            None => 4,
            _ => return None,
          };
          Some(parts_to_axes(diatonic, accidental, octave))
        }
      }
    }
    _ => None,
  }
}

/// Decode a summed `(diatonic position, MIDI number)` back into the canonical
/// `MusicPitch[<|"Accidental" -> …, "Key" -> …, "MIDINumber" -> …|>]` object.
fn axes_to_music_pitch(position: i128, midi: i128) -> Expr {
  let octave = position.div_euclid(7);
  let letter_idx = position.rem_euclid(7) as usize;
  let natural_midi = (octave + 1) * 12 + DIATONIC_BASE_SEMITONE[letter_idx];
  let accidental = midi - natural_midi;
  // Keys are alphabetical (Accidental, Key, MIDINumber), matching Wolfram.
  Expr::FunctionCall {
    name: "MusicPitch".to_string(),
    args: vec![Expr::Association(vec![
      (Expr::String("Accidental".into()), Expr::Integer(accidental)),
      (
        Expr::String("Key".into()),
        Expr::String(DIATONIC_LETTERS[letter_idx].to_string()),
      ),
      (Expr::String("MIDINumber".into()), Expr::Integer(midi)),
    ])]
    .into(),
  }
}

/// Decode a MIDI number into the canonical arithmetic pitch object
/// `MusicPitch[<|"Accidental" -> …, "Key" -> …, "MIDINumber" -> …|>]`, spelled
/// the default (sharp) way straight from the MIDI number. This is the spelling
/// Wolfram uses when a pitch is transposed by a *bare-semitone* (chromatic)
/// `MusicInterval`, which carries no diatonic step of its own — e.g. `C +
/// MusicInterval[8]` is `G#`, not the diatonic `Ab`.
fn midi_to_music_pitch(midi: i128) -> Expr {
  let pitch_class = midi.rem_euclid(12) as usize;
  let spelling = PITCH_CLASS_NAMES[pitch_class]; // "C", "C#", … (sharp names)
  let key = spelling.as_bytes()[0] as char;
  // Every chromatic name is either natural or a single sharp.
  let accidental = spelling.len() as i128 - 1;
  Expr::FunctionCall {
    name: "MusicPitch".to_string(),
    args: vec![Expr::Association(vec![
      (Expr::String("Accidental".into()), Expr::Integer(accidental)),
      (Expr::String("Key".into()), Expr::String(key.to_string())),
      (Expr::String("MIDINumber".into()), Expr::Integer(midi)),
    ])]
    .into(),
  }
}

/// Whether a summand (possibly negated or integer-scaled) is a bare-semitone
/// *chromatic* `MusicInterval` — `MusicInterval[n]` or a `Semitones`-only
/// association with no spelled `"Name"`. Such an interval fixes no diatonic
/// step, so transposing by it spells the result straight from its MIDI number
/// rather than by summing staff positions.
fn is_chromatic_interval(term: &Expr) -> bool {
  match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_chromatic_interval(operand),
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.iter().any(is_chromatic_interval)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => is_chromatic_interval(left) || is_chromatic_interval(right),
    Expr::FunctionCall { name, args }
      if name == "MusicInterval" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::Integer(_) => true,
        Expr::Association(pairs) => assoc_get(pairs, "Name").is_none(),
        _ => false,
      }
    }
    _ => false,
  }
}

/// The `(diatonic step, semitone)` span of a named interval, or `None` for an
/// unrecognized name. The diatonic step is how many letter names the interval
/// covers (a third spans two letters), which fixes the *spelling* of a
/// transposed note independently of its sounding semitone.
fn interval_name_axes(name: &str) -> Option<(i128, i128)> {
  Some(match name {
    "PerfectUnison" | "Unison" => (0, 0),
    "MinorSecond" => (1, 1),
    "MajorSecond" => (1, 2),
    "AugmentedSecond" => (1, 3),
    "MinorThird" => (2, 3),
    "MajorThird" => (2, 4),
    "AugmentedThird" => (2, 5),
    "DiminishedFourth" => (3, 4),
    "PerfectFourth" => (3, 5),
    "AugmentedFourth" | "Tritone" => (3, 6),
    "DiminishedFifth" => (4, 6),
    "PerfectFifth" => (4, 7),
    "AugmentedFifth" => (4, 8),
    "MinorSixth" => (5, 8),
    "MajorSixth" => (5, 9),
    "AugmentedSixth" => (5, 10),
    "DiminishedSeventh" => (6, 9),
    "MinorSeventh" => (6, 10),
    "MajorSeventh" => (6, 11),
    "PerfectOctave" | "Octave" => (7, 12),
    "MinorNinth" => (8, 13),
    "MajorNinth" => (8, 14),
    _ => return None,
  })
}

/// The spelled name of a simple (within-octave) interval spanning `dia` letter
/// steps and `semi` semitones, or `None` for a spelling outside the standard
/// table (e.g. a doubly-augmented step).
fn simple_interval_name(dia: i128, semi: i128) -> Option<&'static str> {
  Some(match (dia, semi) {
    (0, 0) => "Unison",
    // Wolfram 15 names the augmented unison (C# - C) "DiminishedUnison".
    (0, 1) => "DiminishedUnison",
    (1, 0) => "DiminishedSecond",
    (1, 1) => "MinorSecond",
    (1, 2) => "MajorSecond",
    (1, 3) => "AugmentedSecond",
    (2, 2) => "DiminishedThird",
    (2, 3) => "MinorThird",
    (2, 4) => "MajorThird",
    (2, 5) => "AugmentedThird",
    (3, 4) => "DiminishedFourth",
    (3, 5) => "PerfectFourth",
    (3, 6) => "AugmentedFourth",
    (4, 6) => "DiminishedFifth",
    (4, 7) => "PerfectFifth",
    (4, 8) => "AugmentedFifth",
    (5, 7) => "DiminishedSixth",
    (5, 8) => "MinorSixth",
    (5, 9) => "MajorSixth",
    (5, 10) => "AugmentedSixth",
    (6, 9) => "DiminishedSeventh",
    (6, 10) => "MinorSeventh",
    (6, 11) => "MajorSeventh",
    (6, 12) => "AugmentedSeventh",
    (7, 11) => "DiminishedOctave",
    (7, 12) => "Octave",
    (7, 13) => "AugmentedOctave",
    _ => return None,
  })
}

/// The canonical `MusicInterval[<|"Semitones" -> …, "Name" -> …,
/// "CompoundOctaves" -> …|>]` object spanning the given diatonic-step and
/// semitone axes, as produced by a pitch difference. `Semitones` keeps the
/// direction sign while the spelled `Name` and the octave fold come from the
/// absolute span — `C4 - E4` is a `MajorThird` with `Semitones -> -4`, and
/// `D6 - C4` folds to `CompoundMajorSecond` with `CompoundOctaves -> 2`.
fn axes_to_music_interval(dia: i128, semi: i128) -> Expr {
  let (d, s) = (dia.abs(), semi.abs());
  let compound = if d > 7 { (d - 1).div_euclid(7) } else { 0 };
  let (d_simple, s_simple) = (d - 7 * compound, s - 12 * compound);
  let base = simple_interval_name(d_simple, s_simple).unwrap_or_else(|| {
    // A spelling outside the standard table falls back to the simplest
    // diatonic reading of its semitone class.
    let (dd, _) = semitone_interval_axes(s_simple);
    simple_interval_name(dd, s_simple).unwrap_or("Unison")
  });
  let name = if compound > 0 {
    format!("Compound{base}")
  } else {
    base.to_string()
  };
  music_assoc(
    "MusicInterval",
    vec![
      ("Semitones", Expr::Integer(semi)),
      ("Name", Expr::String(name)),
      ("CompoundOctaves", Expr::Integer(compound)),
    ],
  )
}

/// The `(diatonic step, semitone)` span of a bare-semitone interval such as
/// `MusicInterval[5]`, spelled with the simplest (most common) diatonic reading
/// of each semitone class. Handles compound (>octave) and descending intervals.
fn semitone_interval_axes(semi: i128) -> (i128, i128) {
  // Simplest diatonic step for each within-octave semitone 0..=11.
  const DIA: [i128; 12] = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6];
  let octave = semi.div_euclid(12);
  let simple = semi.rem_euclid(12) as usize;
  (octave * 7 + DIA[simple], semi)
}

/// The `(diatonic step, semitone)` span of a `MusicInterval` object in any of
/// its forms: a named interval `MusicInterval["MinorThird"]`, a bare-semitone
/// interval `MusicInterval[5]`, or the canonical
/// `MusicInterval[<|"Semitones" -> …, "Name" -> …, "CompoundOctaves" -> …|>]`
/// association. Returns `None` for anything that is not a `MusicInterval`.
fn music_interval_axes(expr: &Expr) -> Option<(i128, i128)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicInterval" || args.len() != 1 {
    return None;
  }
  match &args[0] {
    Expr::String(spec) => interval_name_axes(spec),
    Expr::Integer(semi) => Some(semitone_interval_axes(*semi)),
    Expr::Association(pairs) => {
      let lookup = |key: &str| {
        pairs.iter().find_map(|(k, v)| match k {
          Expr::String(s) if s == key => Some(v),
          _ => None,
        })
      };
      let compound = match lookup("CompoundOctaves") {
        Some(Expr::Integer(c)) => *c,
        _ => 0,
      };
      // Prefer the spelled `Name` for the diatonic step; fall back to the
      // simplest reading of the `Semitones` field.
      let (dia, semi) = match lookup("Name") {
        Some(Expr::String(n)) => interval_name_axes(n)?,
        _ => match lookup("Semitones") {
          Some(Expr::Integer(s)) => semitone_interval_axes(*s),
          _ => return None,
        },
      };
      Some((dia + 7 * compound, semi + 12 * compound))
    }
    _ => None,
  }
}

/// Interpret one summand of a music sum as `(coefficient, (diatonic, MIDI/step)
/// axes, is_pitch)`. A `MusicPitch` contributes its absolute staff position and
/// MIDI number; a `MusicInterval` contributes its diatonic-step and semitone
/// span so that `pitch + interval` transposes the pitch. Negation and integer
/// scaling (from subtraction / `k …`) carry the matching coefficient.
fn music_summand_axes(term: &Expr) -> Option<(i128, (i128, i128), bool)> {
  match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (coeff, axes, is_pitch) = music_summand_axes(operand)?;
      Some((-coeff, axes, is_pitch))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut coeff = 1i128;
      let mut inner = None;
      for arg in args.iter() {
        if let Expr::Integer(k) = arg {
          coeff *= *k;
        } else if inner.is_none() {
          inner = Some(music_summand_axes(arg)?);
        } else {
          return None;
        }
      }
      let (c, axes, is_pitch) = inner?;
      Some((coeff * c, axes, is_pitch))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => match (left.as_ref(), right.as_ref()) {
      (Expr::Integer(k), other) | (other, Expr::Integer(k)) => {
        let (c, axes, is_pitch) = music_summand_axes(other)?;
        Some((*k * c, axes, is_pitch))
      }
      _ => None,
    },
    _ => {
      if let Some(axes) = music_pitch_axes(term) {
        Some((1, axes, true))
      } else {
        music_interval_axes(term).map(|axes| (1, axes, false))
      }
    }
  }
}

/// Flatten nested `Plus` (built either as `FunctionCall{"Plus", …}` or chained
/// `BinaryOp::Plus`) into a single list of summands.
fn flatten_plus(args: &[Expr]) -> Vec<Expr> {
  let mut flat: Vec<Expr> = Vec::new();
  let mut stack: Vec<Expr> = args.iter().rev().cloned().collect();
  while let Some(term) = stack.pop() {
    match term {
      Expr::FunctionCall {
        ref name,
        args: ref inner,
      } if name == "Plus" => {
        for i in inner.iter().rev() {
          stack.push(i.clone());
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ref left,
        ref right,
      } => {
        stack.push((**right).clone());
        stack.push((**left).clone());
      }
      other => flat.push(other),
    }
  }
  flat
}

/// `Plus[…]` over `MusicPitch` and `MusicInterval` objects. Pitches only
/// combine *through intervals*: `pitch - pitch` is the `MusicInterval` between
/// them and `pitch + interval` transposes, but plain `pitch + pitch` has no
/// musical meaning and collects like ordinary terms (`p + p` is `2 p`, as in
/// Wolfram). Returns `None` whenever the sum is not such an arithmetic — a sum
/// mixing in a non-music term, a scaled pitch like `2 p`, or only plain
/// pitches — so ordinary `Plus` evaluation proceeds unchanged.
pub fn try_music_pitch_plus(args: &[Expr]) -> Option<Expr> {
  let flat = flatten_plus(args);

  let mut total = (0i128, 0i128);
  let mut count = 0;
  let mut pos_pitches = 0;
  let mut neg_pitches = 0;
  let mut interval_count = 0;
  let mut chromatic = false;
  for term in &flat {
    let (coeff, (position, midi), is_pitch) = music_summand_axes(term)?;
    total.0 += coeff * position;
    total.1 += coeff * midi;
    count += 1;
    if is_pitch {
      // Only a bare pitch or its negation takes part in pitch arithmetic; a
      // scaled pitch like `2 p` never does, so `2 p - p` collects to `p`.
      match coeff {
        1 => pos_pitches += 1,
        -1 => neg_pitches += 1,
        _ => return None,
      }
    } else {
      interval_count += 1;
    }
    if is_chromatic_interval(term) {
      chromatic = true;
    }
  }
  // A lone pitch is not arithmetic, and a sum with no pitch (e.g. two intervals)
  // is not a pitch — leave both for the normal path.
  if count < 2 || pos_pitches + neg_pitches == 0 {
    return None;
  }
  // Plain pitches don't combine with each other — without a subtraction or an
  // interval summand there is no pitch arithmetic, and Plus collects like
  // terms instead.
  if neg_pitches == 0 && interval_count == 0 {
    return None;
  }
  match pos_pitches - neg_pitches {
    // The pitches cancel pairwise into an interval: `A4 - A4` is a Unison.
    0 => Some(axes_to_music_interval(total.0, total.1)),
    // One pitch survives, transposed by the paired-off rest. A bare-semitone
    // interval carries no diatonic step, so its transposition is spelled
    // straight from the MIDI number; otherwise the summed staff position
    // fixes the letter.
    1 => Some(if chromatic {
      midi_to_music_pitch(total.1)
    } else {
      axes_to_music_pitch(total.0, total.1)
    }),
    _ => None,
  }
}

/// `MusicChord[{pitches…}] + MusicInterval[…] + …` — transpose every pitch of an
/// explicit-pitch-list chord by the summed interval span, keeping any trailing
/// chord arguments (e.g. a duration). Returns `None` unless exactly one summand
/// is a pitch-list `MusicChord` and every other summand is a `MusicInterval`
/// (whose pitches all resolve), so the sum is otherwise left symbolic.
pub fn try_music_chord_plus_interval(args: &[Expr]) -> Option<Expr> {
  let flat = flatten_plus(args);

  let mut chord: Option<&Expr> = None;
  let mut span = (0i128, 0i128);
  let mut interval_count = 0;
  let mut chromatic = false;
  for term in &flat {
    if let Expr::FunctionCall { name, args: cargs } = term
      && name == "MusicChord"
      && (matches!(cargs.first(), Some(Expr::List(_)))
        || matches!(cargs.first(), Some(Expr::Association(pairs))
          if matches!(assoc_get(pairs, "PitchList"), Some(Expr::List(_)))))
    {
      if chord.is_some() {
        return None; // two chords: not pitch arithmetic
      }
      chord = Some(term);
      continue;
    }
    // Every non-chord summand must be a (possibly scaled/negated) interval.
    let (coeff, axes, is_pitch) = music_summand_axes(term)?;
    if is_pitch {
      return None;
    }
    span.0 += coeff * axes.0;
    span.1 += coeff * axes.1;
    interval_count += 1;
    if is_chromatic_interval(term) {
      chromatic = true;
    }
  }

  let chord = chord?;
  if interval_count == 0 {
    return None;
  }
  let Expr::FunctionCall { args: cargs, .. } = chord else {
    return None;
  };
  // The tones live either in the raw list form or under the canonical
  // `"PitchList"` key.
  let items = match cargs.first() {
    Some(Expr::List(items)) => items,
    Some(Expr::Association(pairs)) => match assoc_get(pairs, "PitchList") {
      Some(Expr::List(items)) => items,
      _ => return None,
    },
    _ => return None,
  };
  // A bare-semitone interval carries no diatonic step, so every transposed tone
  // is spelled straight from its MIDI number; a named interval keeps the staff
  // position and its letter-based spelling.
  let transposed: Option<Vec<Expr>> = items
    .iter()
    .map(|p| {
      let (position, midi) = music_pitch_axes(p)?;
      Some(if chromatic {
        midi_to_music_pitch(midi + span.1)
      } else {
        axes_to_music_pitch(position + span.0, midi + span.1)
      })
    })
    .collect();
  // The canonical chord form keys its tones under `"PitchList"`.
  let pitch_list = Expr::Association(vec![(
    Expr::String("PitchList".into()),
    Expr::List(transposed?.into()),
  )]);
  let mut new_args = vec![pitch_list];
  new_args.extend(cargs.iter().skip(1).cloned());
  Some(Expr::FunctionCall {
    name: "MusicChord".to_string(),
    args: new_args.into(),
  })
}

/// Heads of the computational-music containers/events whose pitches are
/// transposed as a whole when a `MusicInterval` is added — `MusicNote`,
/// `MusicRest`, `MusicMeasure`, `MusicVoice`, `MusicScore`, `MusicScale`.
/// (`MusicChord`/`MusicPitch` summands have their own transposition paths.)
fn is_transposable_container(term: &Expr) -> bool {
  matches!(term, Expr::FunctionCall { name, .. }
    if matches!(name.as_str(),
      "MusicNote" | "MusicRest" | "MusicMeasure" | "MusicVoice"
        | "MusicScore" | "MusicScale"))
}

/// Transpose every `MusicPitch` nested anywhere in `expr` by the interval
/// `span` (a `(diatonic-position, MIDI)` shift), leaving all other structure —
/// durations, time signatures, container heads — intact. A `chromatic` span
/// (a bare-semitone interval, which fixes no diatonic step) spells each result
/// straight from its MIDI number; a diatonic span keeps the staff position so
/// the letter spelling follows the interval.
fn transpose_music(expr: &Expr, span: (i128, i128), chromatic: bool) -> Expr {
  match expr {
    Expr::FunctionCall { name, .. } if name == "MusicPitch" => {
      match resolve_pitch_name(expr).and_then(|n| parse_pitch_parts(&n)) {
        Some((diatonic, accidental, octave)) => {
          let (position, midi) = parts_to_axes(diatonic, accidental, octave);
          if chromatic {
            midi_to_music_pitch(midi + span.1)
          } else {
            axes_to_music_pitch(position + span.0, midi + span.1)
          }
        }
        None => expr.clone(),
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| transpose_music(a, span, chromatic))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| transpose_music(a, span, chromatic))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::Association(pairs) => Expr::Association(
      pairs
        .iter()
        .map(|(k, v)| (k.clone(), transpose_music(v, span, chromatic)))
        .collect(),
    ),
    other => other.clone(),
  }
}

/// `container + MusicInterval[…] + …` — transpose every pitch inside a music
/// container/event (`MusicNote`, `MusicMeasure`, `MusicVoice`, `MusicScore`,
/// `MusicScale`) by the summed interval span, so e.g. `voice + MusicInterval[5]`
/// shifts the whole voice up a perfect fourth. Returns `None` unless exactly one
/// summand is such a container and every other summand is a `MusicInterval`, so
/// any other sum is left symbolic.
pub fn try_music_container_plus_interval(args: &[Expr]) -> Option<Expr> {
  let flat = flatten_plus(args);
  let mut container: Option<&Expr> = None;
  let mut span = (0i128, 0i128);
  let mut interval_count = 0;
  let mut chromatic = false;
  for term in &flat {
    if is_transposable_container(term) {
      if container.is_some() {
        return None; // two containers: not interval transposition
      }
      container = Some(term);
      continue;
    }
    // Every other summand must be a (possibly scaled/negated) interval.
    let (coeff, axes, is_pitch) = music_summand_axes(term)?;
    if is_pitch {
      return None;
    }
    span.0 += coeff * axes.0;
    span.1 += coeff * axes.1;
    interval_count += 1;
    if is_chromatic_interval(term) {
      chromatic = true;
    }
  }
  let container = container?;
  if interval_count == 0 {
    return None;
  }
  Some(transpose_music(container, span, chromatic))
}

/// MIDI note number of a `MusicPitch` object in any of its forms. Octaveless
/// names default to octave 4, so `MusicPitch["C#"]` and `MusicPitch["Db"]` both
/// resolve to MIDI 61 — which is what makes them compare equal.
pub fn music_pitch_midi(expr: &Expr) -> Option<i128> {
  // The staff-position decoder handles named and MIDINumber-association forms;
  // fall back through the name resolver for the `Octave`-keyed association form
  // produced by chord spelling (which carries no explicit MIDINumber).
  if let Some((_position, midi)) = music_pitch_axes(expr) {
    return Some(midi);
  }
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicPitch" {
    return None;
  }
  pitch_name_to_midi(&resolve_pitch_name(args.first()?)?)
}

// ---------------------------------------------------------------------------
// MusicNote / MusicDuration / MusicChord canonicalization (WL 15)
// ---------------------------------------------------------------------------
//
// Following the WL 15 reference pages, `MusicNote[pitch, duration]`,
// `MusicDuration[number]` and `MusicChord["<root><quality>"]` canonicalize to
// association-valued objects that expose their parts by name, e.g.
//
// ```text
// MusicNote["A#", 1/2]
//   MusicNote[<|"Pitch"    -> MusicPitch[<|"Accidental" -> 1, "Key" -> "A"|>],
//               "Duration" -> MusicDuration[<|"Duration" -> 1/2|>]|>]
// MusicChord["GMajor"]
//   MusicChord[<|"Name" -> "Major",
//                "Root" -> MusicPitch[<|"Key" -> "G", "Accidental" -> 0|>]|>]
// ```

/// A small `Rational[n, d]` builder (Woxi stores rationals as `Rational[n, d]`).
fn rational(n: i128, d: i128) -> Expr {
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
  }
}

/// A `MusicPitch[<|…|>]` / `MusicDuration[<|…|>]` / … object wrapping a
/// key-ordered association.
fn music_assoc(head: &str, pairs: Vec<(&str, Expr)>) -> Expr {
  Expr::FunctionCall {
    name: head.to_string(),
    args: vec![Expr::Association(
      pairs
        .into_iter()
        .map(|(k, v)| (Expr::String(k.to_string()), v))
        .collect(),
    )]
    .into(),
  }
}

/// Look up a string key in an association's `(key, value)` pairs.
pub(crate) fn assoc_get<'a>(
  pairs: &'a [(Expr, Expr)],
  key: &str,
) -> Option<&'a Expr> {
  pairs.iter().find_map(|(k, v)| match k {
    Expr::String(s) if s == key => Some(v),
    _ => None,
  })
}

/// Parse a pitch name such as `"A#"`, `"C4"`, `"Db5"`, `"G-1"` into its
/// `(key letter, accidental, octave-if-present)` components. Unlike
/// [`parse_pitch_parts`], the octave is `None` when the name carries no octave
/// digit, so `MusicNote["A#", …]` can omit `"Octave"` from its pitch.
fn parse_pitch_spelled(name: &str) -> Option<(char, i128, Option<i128>)> {
  let bytes = name.as_bytes();
  let letter = bytes.first()?.to_ascii_uppercase();
  if !(b'A'..=b'G').contains(&letter) {
    return None;
  }
  let mut idx = 1;
  let mut accidental = 0i128;
  while let Some(&c) = bytes.get(idx) {
    match c {
      b'#' => accidental += 1,
      b'b' => accidental -= 1,
      _ => break,
    }
    idx += 1;
  }
  let octave = if idx >= name.len() {
    None
  } else {
    Some(name.get(idx..)?.parse().ok()?)
  };
  Some((letter as char, accidental, octave))
}

/// The canonical `MusicPitch` association for a note/chord tone, keyed in the
/// order `Accidental, [Octave,] Key` (the layout WL uses for note pitches and
/// chord `"PitchList"` entries). `"Octave"` is included only when known.
fn pitch_object(letter: char, accidental: i128, octave: Option<i128>) -> Expr {
  let mut pairs = vec![("Accidental", Expr::Integer(accidental))];
  if let Some(o) = octave {
    pairs.push(("Octave", Expr::Integer(o)));
  }
  pairs.push(("Key", Expr::String(letter.to_string())));
  music_assoc("MusicPitch", pairs)
}

/// The spelled name of a pitch — its key letter followed by an accidental
/// glyph, `♯` (sharp) or `♭` (flat), repeated for double accidentals. The
/// octave is not part of the name (`"C#4"` is named `"C♯"`).
fn pitch_spelled_name(letter: char, accidental: i128) -> String {
  let mut name = letter.to_string();
  match accidental.cmp(&0) {
    std::cmp::Ordering::Greater => {
      name.push_str(&"♯".repeat(accidental as usize))
    }
    std::cmp::Ordering::Less => {
      name.push_str(&"♭".repeat((-accidental) as usize))
    }
    std::cmp::Ordering::Equal => {}
  }
  name
}

/// The canonical `MusicPitch` association for a note's pitch, keyed
/// `Accidental, [Octave,] Key[, Name]`. When an octave is known the object also
/// carries its spelled `"Name"`, matching the Wolfram Language (`MusicNote["C4"]`
/// → `<|Accidental -> 0, Octave -> 4, Key -> C, Name -> C|>`); an octaveless
/// name stays `<|Accidental -> a, Key -> k|>`.
fn note_pitch_object(
  letter: char,
  accidental: i128,
  octave: Option<i128>,
) -> Expr {
  let mut pairs = vec![("Accidental", Expr::Integer(accidental))];
  if let Some(o) = octave {
    pairs.push(("Octave", Expr::Integer(o)));
  }
  pairs.push(("Key", Expr::String(letter.to_string())));
  if octave.is_some() {
    pairs.push(("Name", Expr::String(pitch_spelled_name(letter, accidental))));
  }
  music_assoc("MusicPitch", pairs)
}

/// Resolve a pitch specification (name string, `MusicPitch`, MIDI integer, …)
/// to the canonical note-pitch object, preserving whether an octave was given.
/// An already-canonical `MusicPitch[<|…|>]` object is kept verbatim — a note
/// built from a MIDI-numbered pitch stores `<|MIDINumber -> n|>` unspelled,
/// matching Wolfram.
fn resolve_pitch_object(spec: &Expr) -> Option<Expr> {
  if let Expr::FunctionCall { name, args } = spec
    && name == "MusicPitch"
    && matches!(args.first(), Some(Expr::Association(_)))
  {
    return Some(spec.clone());
  }
  let name = resolve_pitch_name(spec)?;
  let (letter, accidental, octave) = parse_pitch_spelled(&name)?;
  Some(note_pitch_object(letter, accidental, octave))
}

/// The rhythmic value of one of the named note durations, or `None` for an
/// unknown name.
fn named_duration_value(name: &str) -> Option<Expr> {
  let denom = match name {
    "Whole" => return Some(Expr::Integer(1)),
    "Half" => 2,
    "Quarter" => 4,
    "Eighth" => 8,
    "Sixteenth" => 16,
    "ThirtySecond" => 32,
    "SixtyFourth" => 64,
    _ => return None,
  };
  Some(rational(1, denom))
}

/// Extract the rhythmic value from any duration specification: a bare number,
/// a named duration string, or a `MusicDuration[…]` object (association, bare
/// number or name).
fn resolve_duration_value(spec: &Expr) -> Option<Expr> {
  match spec {
    Expr::Integer(_) | Expr::Real(_) => Some(spec.clone()),
    Expr::FunctionCall { name, .. } if name == "Rational" => Some(spec.clone()),
    Expr::String(s) => named_duration_value(s),
    Expr::FunctionCall { name, args } if name == "MusicDuration" => {
      match args.first()? {
        Expr::Association(pairs) => assoc_get(pairs, "Duration").cloned(),
        other => resolve_duration_value(other),
      }
    }
    _ => None,
  }
}

/// The canonical `MusicDuration[<|…|>]` object for any duration specification:
/// a bare number, a named duration string, or a `MusicDuration[…]` object in
/// any of those forms. A named duration keeps its spelled `"Name"` alongside
/// its rhythmic value (as in Wolfram, where `MusicDuration["Half"]` is
/// `MusicDuration[<|"Duration" -> 1/2, "Name" -> "Half"|>]`); the
/// already-canonical association form passes through unchanged.
fn resolve_duration_object(spec: &Expr) -> Option<Expr> {
  match spec {
    Expr::Integer(_) | Expr::Real(_) => Some(music_assoc(
      "MusicDuration",
      vec![("Duration", spec.clone())],
    )),
    Expr::FunctionCall { name, .. } if name == "Rational" => Some(music_assoc(
      "MusicDuration",
      vec![("Duration", spec.clone())],
    )),
    Expr::String(s) => {
      let value = named_duration_value(s)?;
      Some(music_assoc(
        "MusicDuration",
        vec![("Duration", value), ("Name", Expr::String(s.clone()))],
      ))
    }
    Expr::FunctionCall { name, args } if name == "MusicDuration" => {
      match args.first()? {
        Expr::Association(_) => Some(spec.clone()),
        other => resolve_duration_object(other),
      }
    }
    _ => None,
  }
}

/// `MusicDuration[spec]` — canonicalize a numeric or named duration to the
/// association form `MusicDuration[<|"Duration" -> value, …|>]`. The
/// already-canonical association form is left untouched (`None`).
pub fn music_duration(args: &[Expr]) -> Option<Expr> {
  let [spec] = args else {
    return None;
  };
  if matches!(spec, Expr::Association(_)) {
    return None;
  }
  resolve_duration_object(spec)
}

/// `MusicNote[pitch]` / `MusicNote[pitch, duration]` — canonicalize to the
/// association form. A note given only a pitch becomes
/// `MusicNote[<|"Pitch" -> MusicPitch[…]|>]`; a note with an explicit duration
/// adds `"Duration" -> MusicDuration[…]`. Returns `None` (leaving the note
/// symbolic) when a part is not a recognized specification.
pub fn music_note(args: &[Expr]) -> Option<Expr> {
  match args {
    [pitch_spec] => {
      let pitch = resolve_pitch_object(pitch_spec)?;
      Some(music_assoc("MusicNote", vec![("Pitch", pitch)]))
    }
    [pitch_spec, duration_spec] => {
      let pitch = resolve_pitch_object(pitch_spec)?;
      let duration = resolve_duration_object(duration_spec)?;
      Some(music_assoc(
        "MusicNote",
        vec![("Pitch", pitch), ("Duration", duration)],
      ))
    }
    _ => None,
  }
}

/// The stacked-thirds `(semitone, diatonic-step)` offsets from the root for
/// each recognized chord quality, or `None` for an unknown quality.
fn chord_quality_offsets(name: &str) -> Option<&'static [(i128, i128)]> {
  Some(match name {
    "Major" => &[(0, 0), (4, 2), (7, 4)],
    "Minor" => &[(0, 0), (3, 2), (7, 4)],
    "Diminished" => &[(0, 0), (3, 2), (6, 4)],
    "Augmented" => &[(0, 0), (4, 2), (8, 4)],
    "SuspendedSecond" => &[(0, 0), (2, 1), (7, 4)],
    "SuspendedFourth" => &[(0, 0), (5, 3), (7, 4)],
    "Sixth" => &[(0, 0), (4, 2), (7, 4), (9, 5)],
    "MinorSixth" => &[(0, 0), (3, 2), (7, 4), (9, 5)],
    "MajorSeventh" => &[(0, 0), (4, 2), (7, 4), (11, 6)],
    "MinorSeventh" => &[(0, 0), (3, 2), (7, 4), (10, 6)],
    "DominantSeventh" => &[(0, 0), (4, 2), (7, 4), (10, 6)],
    "DiminishedSeventh" => &[(0, 0), (3, 2), (6, 4), (9, 6)],
    "HalfDiminishedSeventh" => &[(0, 0), (3, 2), (6, 4), (10, 6)],
    "MinorMajorSeventh" => &[(0, 0), (3, 2), (7, 4), (11, 6)],
    "AugmentedSeventh" => &[(0, 0), (4, 2), (8, 4), (10, 6)],
    "DominantNinth" => &[(0, 0), (4, 2), (7, 4), (10, 6), (14, 8)],
    "MajorNinth" => &[(0, 0), (4, 2), (7, 4), (11, 6), (14, 8)],
    "MinorNinth" => &[(0, 0), (3, 2), (7, 4), (10, 6), (14, 8)],
    _ => return None,
  })
}

/// Normalize a written chord-quality suffix (everything after the root, e.g.
/// `"m7"`, `"maj7"`, `"dim"`, `"sus4"`, or the empty string for a bare triad)
/// to its canonical quality name. Matching is case-sensitive where it must be
/// (`"m"` is minor, `"M"` is major) and accepts the common jazz/pop symbols in
/// addition to the spelled-out names. A leading separator space (as in
/// `"G Major"`) is ignored. Returns `None` for an unrecognized suffix.
fn normalize_chord_quality(quality: &str) -> Option<&'static str> {
  Some(match quality.trim() {
    "" | "M" | "maj" | "Maj" | "major" | "Major" | "Ma" | "ma" | "Δ" => {
      "Major"
    }
    "m" | "min" | "Min" | "mi" | "Mi" | "minor" | "Minor" => "Minor",
    "aug" | "Aug" | "augmented" | "Augmented" | "Maug" | "M#5" | "M+5"
    | "M♯5" => "Augmented",
    "dim" | "Dim" | "diminished" | "Diminished" | "o" | "O" | "°" | "mb5"
    | "m♭5" | "mo5" => "Diminished",
    "sus2" | "Sus2" | "SuspendedSecond" => "SuspendedSecond",
    "sus" | "Sus" | "sus4" | "Sus4" | "SuspendedFourth" => "SuspendedFourth",
    "M6" | "maj6" | "add6" | "Sixth" => "Sixth",
    "m6" | "min6" | "MinorSixth" => "MinorSixth",
    "M7" | "maj7" | "Maj7" | "Ma7" | "major7" | "MajorSeventh" | "Δ7"
    | "MΔ7" => "MajorSeventh",
    "m7" | "min7" | "Min7" | "mi7" | "minor7" | "MinorSeventh" => {
      "MinorSeventh"
    }
    "dom7" | "Dom7" | "dominant7" | "Mm7" | "DominantSeventh" => {
      "DominantSeventh"
    }
    "dim7" | "Dim7" | "o7" | "O7" | "°7" | "DiminishedSeventh" => {
      "DiminishedSeventh"
    }
    "m7b5"
    | "m7♭5"
    | "m7-5"
    | "min7dim5"
    | "ø"
    | "ø7"
    | "Ø"
    | "Ø7"
    | "HalfDiminishedSeventh" => "HalfDiminishedSeventh",
    "mmaj7" | "minmaj7" | "m#7" | "mΔ7" | "MinorMajorSeventh" => {
      "MinorMajorSeventh"
    }
    "aug7" | "Aug7" | "7#5" | "7+5" | "7♯5" | "AugmentedSeventh" => {
      "AugmentedSeventh"
    }
    "DominantNinth" => "DominantNinth",
    "M9" | "maj9" | "Maj9" | "major9" | "Δ9" | "MajorNinth" => "MajorNinth",
    "m9" | "min9" | "minor9" | "MinorNinth" => "MinorNinth",
    _ => return None,
  })
}

/// Split a chord name such as `"GMajor"`, `"F#Minor"`, `"BbMajorSeventh"`,
/// `"Cm7"`, or `"G Major"` into its `(root letter, root accidental, quality)`
/// components. The root is the leading note letter plus any run of accidentals
/// (`#`/`♯` sharp, `b`/`♭` flat); the remainder is the quality suffix. A double
/// accidental like `"F##"` or `"Cbb"` is supported.
fn parse_chord_name(name: &str) -> Option<(char, i128, &str)> {
  let mut iter = name.char_indices().peekable();
  let letter = iter.next()?.1.to_ascii_uppercase();
  if !('A'..='G').contains(&letter) {
    return None;
  }
  let mut accidental = 0i128;
  let mut rest = name.len();
  while let Some(&(i, c)) = iter.peek() {
    match c {
      '#' | '♯' => accidental += 1,
      'b' | '♭' => accidental -= 1,
      _ => {
        rest = i;
        break;
      }
    }
    iter.next();
    rest = iter.peek().map_or(name.len(), |&(j, _)| j);
  }
  Some((letter, accidental, &name[rest..]))
}

/// The display spelling of a canonical chord quality: the CamelCase name with
/// a space before each interior capital, e.g. `"MinorSeventh"` →
/// `"Minor Seventh"` (the form Wolfram stores under a chord's `"Name"` key).
fn spaced_quality_name(quality: &str) -> String {
  let mut spaced = String::with_capacity(quality.len() + 3);
  for (i, c) in quality.chars().enumerate() {
    if i > 0 && c.is_ascii_uppercase() {
      spaced.push(' ');
    }
    spaced.push(c);
  }
  spaced
}

/// `MusicChord["<root><quality>"]` / `MusicChord[{pitches…}]` — canonicalize to
/// the association form.
///
/// A named chord becomes `MusicChord[<|"Name" -> quality, "Root" ->
/// MusicPitch[…]|>]`, with the quality stored in its spaced display spelling
/// (`"Minor Seventh"`). A bare root (`"G"`) stores no `"Name"` key at all —
/// which is what makes `MusicChord["G"] =!= MusicChord["GMajor"]`. A root
/// followed only by a digit (`"C7"`) is read as the note C in *octave* 7, not
/// a seventh chord: the root gains `Octave`/`Name` keys and the chord again
/// has no quality. An octave outside 0–9 emits `MusicChord::args` and leaves
/// the chord unevaluated.
///
/// An explicit pitch list canonicalizes to `MusicChord[<|"PitchList" ->
/// {…}|>]`, each tone spelled as its `<|Accidental, Octave, Key|>` object
/// (octaveless pitches default to register 4).
pub fn music_chord(args: &[Expr]) -> Option<Expr> {
  match args {
    [Expr::String(spec)] => {
      // Any spec that fails to parse — an invalid root letter, an unknown
      // quality suffix, an out-of-range octave — emits `MusicChord::args` and
      // stays unevaluated (matching Wolfram, which never leaves a chord
      // string silently symbolic).
      let invalid = || {
        crate::emit_message_to_stdout(
          "MusicChord::args: MusicChord called with invalid parameters.",
        );
        Some(unevaluated("MusicChord", args))
      };
      let Some((letter, accidental, quality)) = parse_chord_name(spec) else {
        return invalid();
      };
      // A trailing (optionally signed) number with no quality letters is the
      // root's octave: `"C7"` is the note C in octave 7, not a seventh chord,
      // and `"C+7"` reads the same (only octaves 0–9 exist, so `"C-7"` is
      // invalid). A lone sign (`"C-"`, `"C+"`) carries no octave at all and
      // is the bare root.
      let signed_octave = quality
        .strip_prefix(['+', '-'])
        .unwrap_or(quality)
        .bytes()
        .all(|b| b.is_ascii_digit());
      if !quality.is_empty() && signed_octave {
        let digits = quality.trim_start_matches(['+', '-']);
        if digits.is_empty() {
          // A bare trailing sign: just the root.
          let root = music_assoc(
            "MusicPitch",
            vec![
              ("Key", Expr::String(letter.to_string())),
              ("Accidental", Expr::Integer(accidental)),
            ],
          );
          return Some(music_assoc("MusicChord", vec![("Root", root)]));
        }
        let octave: i128 = match quality.trim_start_matches('+').parse() {
          Ok(o) => o,
          Err(_) => return invalid(),
        };
        if !(0..=9).contains(&octave) {
          return invalid();
        }
        // The octave-rooted pitch carries its full spelled name, octave digit
        // included (`"C♯7"`), keyed `Key, Accidental, Octave, Name`.
        let mut name = pitch_spelled_name(letter, accidental);
        name.push_str(digits);
        let root = music_assoc(
          "MusicPitch",
          vec![
            ("Key", Expr::String(letter.to_string())),
            ("Accidental", Expr::Integer(accidental)),
            ("Octave", Expr::Integer(octave)),
            ("Name", Expr::String(name)),
          ],
        );
        return Some(music_assoc("MusicChord", vec![("Root", root)]));
      }
      // The chord's Root pitch is keyed in the order `Key, Accidental`.
      let root = music_assoc(
        "MusicPitch",
        vec![
          ("Key", Expr::String(letter.to_string())),
          ("Accidental", Expr::Integer(accidental)),
        ],
      );
      // A bare root stores no quality — which is what makes
      // `MusicChord["G"] =!= MusicChord["GMajor"]`. A written suffix resolves
      // to its canonical quality and is stored in the spaced display
      // spelling.
      if quality.trim().is_empty() {
        return Some(music_assoc("MusicChord", vec![("Root", root)]));
      }
      let Some(quality) = normalize_chord_quality(quality) else {
        return invalid();
      };
      Some(music_assoc(
        "MusicChord",
        vec![
          ("Name", Expr::String(spaced_quality_name(quality))),
          ("Root", root),
        ],
      ))
    }
    // An explicit pitch list: spell every tone as its full pitch object.
    [Expr::List(items)] => {
      let tones: Option<Vec<Expr>> = items
        .iter()
        .map(|p| {
          if !matches!(p, Expr::FunctionCall { name, .. } if name == "MusicPitch")
          {
            return None;
          }
          let name = resolve_pitch_name(p)?;
          let (letter, accidental, octave) = parse_pitch_spelled(&name)?;
          Some(pitch_object(letter, accidental, Some(octave.unwrap_or(4))))
        })
        .collect();
      Some(music_assoc(
        "MusicChord",
        vec![("PitchList", Expr::List(tones?.into()))],
      ))
    }
    _ => None,
  }
}

/// Base MIDI number of a note letter in a given octave (natural, no accidental).
fn letter_natural_midi(letter_idx: usize, octave: i128) -> i128 {
  (octave + 1) * 12 + DIATONIC_BASE_SEMITONE[letter_idx]
}

/// Spell out the pitch objects of a canonical chord association's tones. Each
/// tone stacks thirds from the root, spelled on the correct staff letter so the
/// accidentals come out right (`GMajor` → G4, B4, D5).
pub fn chord_tones(pairs: &[(Expr, Expr)]) -> Option<Vec<Expr>> {
  // An explicit pitch list is already the chord's tones.
  if let Some(Expr::List(tones)) = assoc_get(pairs, "PitchList") {
    return Some(tones.to_vec());
  }
  // The stored quality is the spaced display spelling ("Minor Seventh"); a
  // chord with no stored quality (a bare root like MusicChord["G"]) defaults
  // to a major triad.
  let has_stored_name = assoc_get(pairs, "Name").is_some();
  let quality: String = match assoc_get(pairs, "Name") {
    Some(Expr::String(s)) => s.chars().filter(|c| *c != ' ').collect(),
    None => "Major".to_string(),
    _ => return None,
  };
  let offsets = chord_quality_offsets(&quality)?;
  let root = assoc_get(pairs, "Root")?;
  let Expr::FunctionCall { name, args } = root else {
    return None;
  };
  if name != "MusicPitch" {
    return None;
  }
  let Expr::Association(root_pairs) = args.first()? else {
    return None;
  };
  let root_letter = match assoc_get(root_pairs, "Key")? {
    Expr::String(s) => s.as_bytes().first()?.to_ascii_uppercase(),
    _ => return None,
  };
  let root_accidental = match assoc_get(root_pairs, "Accidental") {
    Some(Expr::Integer(n)) => *n,
    None => 0,
    _ => return None,
  };
  let root_octave = match assoc_get(root_pairs, "Octave") {
    Some(Expr::Integer(n)) => *n,
    _ => 4, // WL's default register for a chord root
  };
  let root_idx = letter_diatonic_index(root_letter)?;
  let root_midi =
    letter_natural_midi(root_idx as usize, root_octave) + root_accidental;

  // A named chord that stays within the root's octave prints its tones
  // without `Octave` keys; one that crosses an octave boundary — and any
  // chord with no stored quality (a bare/octave-numbered root) — spells the
  // register on every tone.
  let crosses_octave = offsets
    .iter()
    .any(|(_, step)| (root_idx + step).div_euclid(7) != 0);
  let show_octave = !has_stored_name || crosses_octave;
  offsets
    .iter()
    .map(|(semitone, step)| {
      let letter_pos = root_idx + step;
      let octave_delta = letter_pos.div_euclid(7);
      let letter_idx = letter_pos.rem_euclid(7) as usize;
      let octave = root_octave + octave_delta;
      let target_midi = root_midi + semitone;
      let accidental = target_midi - letter_natural_midi(letter_idx, octave);
      let letter = DIATONIC_LETTERS[letter_idx].chars().next()?;
      Some(pitch_object(
        letter,
        accidental,
        show_octave.then_some(octave),
      ))
    })
    .collect()
}

/// Diatonic interval name for a simple (within-octave) semitone distance.
fn interval_name(simple_semitones: i128) -> &'static str {
  match simple_semitones {
    0 => "PerfectUnison",
    1 => "MinorSecond",
    2 => "MajorSecond",
    3 => "MinorThird",
    4 => "MajorThird",
    5 => "PerfectFourth",
    6 => "Tritone",
    7 => "PerfectFifth",
    8 => "MinorSixth",
    9 => "MajorSixth",
    10 => "MinorSeventh",
    11 => "MajorSeventh",
    _ => "PerfectOctave",
  }
}

/// The `MusicInterval[<|…|>]` objects between successive chord tones.
fn chord_intervals(pairs: &[(Expr, Expr)]) -> Option<Vec<Expr>> {
  let tones = chord_tones(pairs)?;
  let midis: Option<Vec<i128>> = tones.iter().map(music_pitch_midi).collect();
  let midis = midis?;
  Some(
    midis
      .windows(2)
      .map(|w| {
        let distance = w[1] - w[0];
        let compound = distance.div_euclid(12);
        let simple = distance - compound * 12;
        music_assoc(
          "MusicInterval",
          vec![
            ("Semitones", Expr::Integer(simple)),
            ("Name", Expr::String(interval_name(simple).to_string())),
            ("CompoundOctaves", Expr::Integer(compound)),
          ],
        )
      })
      .collect(),
  )
}

/// Property access on a canonical music object, e.g. `MusicNote[<|…|>]["Pitch"]`
/// or `MusicChord[<|…|>]["PitchList"]`. Direct association keys are returned as
/// stored; chords additionally compute `"PitchList"` and `"IntervalList"`.
/// Returns `None` when the access is not a recognized music-object lookup.
pub fn music_property_access(
  head: &str,
  obj_args: &[Expr],
  prop_args: &[Expr],
) -> Option<Expr> {
  if !MUSIC_OBJECT_HEADS.contains(&head) {
    return None;
  }
  let [Expr::Association(pairs)] = obj_args else {
    return None;
  };
  let [Expr::String(key)] = prop_args else {
    return None;
  };
  // Stored keys (Pitch, Duration, Name, Root, …) take precedence.
  if let Some(value) = assoc_get(pairs, key) {
    return Some(value.clone());
  }
  if head == "MusicChord" {
    match key.as_str() {
      "PitchList" => return chord_tones(pairs).map(|t| Expr::List(t.into())),
      "IntervalList" => {
        return chord_intervals(pairs).map(|t| Expr::List(t.into()));
      }
      // A chord with no stored quality (a bare root) is a major triad.
      "Name" if assoc_get(pairs, "Root").is_some() => {
        return Some(Expr::String("Major".to_string()));
      }
      _ => {}
    }
  }
  None
}

/// Interpret one summand of a duration sum as `(coefficient, value)`, where the
/// value is the summand's `MusicDuration` rhythmic value and the coefficient is
/// whatever it is multiplied by. Returns `None` when the term is not a (scaled)
/// `MusicDuration`.
fn music_duration_summand(term: &Expr) -> Option<(Expr, Expr)> {
  match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (coeff, value) = music_duration_summand(operand)?;
      Some((
        Expr::UnaryOp {
          op: UnaryOperator::Minus,
          operand: Box::new(coeff),
        },
        value,
      ))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut coeff: Vec<Expr> = Vec::new();
      let mut value = None;
      for arg in args.iter() {
        if is_music_duration(arg) {
          if value.is_some() {
            return None;
          }
          value = Some(duration_value_of(arg)?);
        } else {
          coeff.push(arg.clone());
        }
      }
      Some((times_of(coeff), value?))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      let (l_dur, r_dur) = (is_music_duration(left), is_music_duration(right));
      match (l_dur, r_dur) {
        (false, true) => Some(((**left).clone(), duration_value_of(right)?)),
        (true, false) => Some(((**right).clone(), duration_value_of(left)?)),
        _ => None,
      }
    }
    _ if is_music_duration(term) => {
      Some((Expr::Integer(1), duration_value_of(term)?))
    }
    _ => None,
  }
}

/// Whether an expression is a `MusicDuration[…]` object.
fn is_music_duration(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. } if name == "MusicDuration")
}

/// The rhythmic value carried by a `MusicDuration[…]` object.
fn duration_value_of(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "MusicDuration" => {
      match args.first()? {
        Expr::Association(pairs) => assoc_get(pairs, "Duration").cloned(),
        other => resolve_duration_value(other),
      }
    }
    _ => None,
  }
}

/// Fold a list of factors into a single `Times[…]` (or the lone factor / `1`).
fn times_of(mut factors: Vec<Expr>) -> Expr {
  match factors.len() {
    0 => Expr::Integer(1),
    1 => factors.pop().unwrap(),
    _ => Expr::FunctionCall {
      name: "Times".to_string(),
      args: factors.into(),
    },
  }
}

/// Decompose a `Plus[…]` over `MusicDuration` objects into the per-summand
/// `Times[coeff, value]` products. Returns `None` when any summand is not a
/// (scaled) `MusicDuration`, so ordinary `Plus` handling proceeds. The caller
/// evaluates the products and wraps the total in a `MusicDuration`.
pub fn music_duration_plus_terms(args: &[Expr]) -> Option<Vec<Expr>> {
  let mut flat: Vec<Expr> = Vec::new();
  let mut stack: Vec<Expr> = args.iter().rev().cloned().collect();
  while let Some(term) = stack.pop() {
    match term {
      Expr::FunctionCall {
        ref name,
        args: ref inner,
      } if name == "Plus" => {
        for i in inner.iter().rev() {
          stack.push(i.clone());
        }
      }
      Expr::BinaryOp {
        op: BinaryOperator::Plus,
        ref left,
        ref right,
      } => {
        stack.push((**right).clone());
        stack.push((**left).clone());
      }
      other => flat.push(other),
    }
  }
  // A lone duration is not arithmetic; leave it for the normal path.
  if flat.len() < 2 {
    return None;
  }
  let mut products = Vec::with_capacity(flat.len());
  for term in &flat {
    let (coeff, value) = music_duration_summand(term)?;
    products.push(Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![coeff, value].into(),
    });
  }
  Some(products)
}

/// Wrap a computed total rhythmic value in the canonical duration object.
/// Duration arithmetic counts in the default quarter-note beat, so the sum
/// carries `BeatDuration -> 1/4` alongside its total (matching Wolfram, where
/// e.g. `MusicDuration[1] - MusicDuration[1/4]` is
/// `MusicDuration[<|Duration -> 3/4, BeatDuration -> 1/4|>]`).
pub fn music_duration_from_value(value: Expr) -> Expr {
  music_assoc(
    "MusicDuration",
    vec![("Duration", value), ("BeatDuration", rational(1, 4))],
  )
}

// ---------------------------------------------------------------------------
// MusicTimeSignature / MusicRest / MusicMeasure (WL 15)
// ---------------------------------------------------------------------------

/// `MusicTimeSignature[n, d]` — canonicalize to the association form
/// `MusicTimeSignature[<|"Numerator" -> n, "Denominator" -> d|>]`. Non-integer
/// or already-canonical arguments are left untouched (`None`).
pub fn music_time_signature(args: &[Expr]) -> Option<Expr> {
  let [Expr::Integer(n), Expr::Integer(d)] = args else {
    return None;
  };
  Some(music_assoc(
    "MusicTimeSignature",
    vec![
      ("Numerator", Expr::Integer(*n)),
      ("Denominator", Expr::Integer(*d)),
    ],
  ))
}

/// `MusicRest[]` / `MusicRest[duration]` — canonicalize to the association form.
/// A bare rest becomes `MusicRest[<||>]`; a rest with a duration becomes
/// `MusicRest[<|"Duration" -> MusicDuration[…]|>]`. Returns `None` when the
/// duration is not a recognized specification.
pub fn music_rest(args: &[Expr]) -> Option<Expr> {
  match args {
    [] => Some(music_assoc("MusicRest", vec![])),
    [spec] => {
      let duration = resolve_duration_object(spec)?;
      Some(music_assoc("MusicRest", vec![("Duration", duration)]))
    }
    _ => None,
  }
}

use crate::functions::math_ast::gcd;

/// Reduce `n/d` to lowest terms with a positive denominator, returning an
/// `Integer` when the denominator is 1 and a `Rational[n, d]` otherwise.
fn rational_reduced(mut n: i128, mut d: i128) -> Expr {
  if d < 0 {
    n = -n;
    d = -d;
  }
  let g = gcd(n.abs(), d).max(1);
  n /= g;
  d /= g;
  if d == 1 {
    Expr::Integer(n)
  } else {
    rational(n, d)
  }
}

/// A rational number as a reduced `(numerator, denominator)` pair with a
/// positive denominator.
type Ratio = (i128, i128);

/// Reduce a `(numerator, denominator)` pair.
fn reduce(n: i128, d: i128) -> Ratio {
  let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
  let g = gcd(n.abs(), d).max(1);
  (n / g, d / g)
}

fn ratio_mul((an, ad): Ratio, (bn, bd): Ratio) -> Ratio {
  reduce(an * bn, ad * bd)
}

fn ratio_add((an, ad): Ratio, (bn, bd): Ratio) -> Ratio {
  reduce(an * bd + bn * ad, ad * bd)
}

fn ratio_sub((an, ad): Ratio, (bn, bd): Ratio) -> Ratio {
  reduce(an * bd - bn * ad, ad * bd)
}

fn ratio_cmp((an, ad): Ratio, (bn, bd): Ratio) -> std::cmp::Ordering {
  (an * bd).cmp(&(bn * ad))
}

/// Render a `Ratio` as the `Expr` the Wolfram Language prints for it.
fn ratio_expr((n, d): Ratio) -> Expr {
  rational_reduced(n, d)
}

/// Parse a bare rhythmic value (`Integer` or `Rational`) into a `Ratio`.
fn duration_ratio(expr: &Expr) -> Option<Ratio> {
  match expr {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args } if name == "Rational" => {
      match &args[..] {
        [Expr::Integer(n), Expr::Integer(d)] if *d != 0 => Some(reduce(*n, *d)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// The rhythmic value of a `MusicDuration[…]` object as a `Ratio`.
fn music_duration_ratio(expr: &Expr) -> Option<Ratio> {
  match expr {
    Expr::FunctionCall { name, args } if name == "MusicDuration" => {
      match args.first()? {
        Expr::Association(pairs) => {
          duration_ratio(assoc_get(pairs, "Duration")?)
        }
        other => duration_ratio(other),
      }
    }
    _ => None,
  }
}

/// The `(numerator, denominator)` of a canonical `MusicTimeSignature[<|…|>]`.
fn time_signature_parts(expr: &Expr) -> Option<(i128, i128)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicTimeSignature" {
    return None;
  }
  let Expr::Association(pairs) = args.first()? else {
    return None;
  };
  let numer = match assoc_get(pairs, "Numerator")? {
    Expr::Integer(n) => *n,
    _ => return None,
  };
  let denom = match assoc_get(pairs, "Denominator")? {
    Expr::Integer(d) if *d > 0 => *d,
    _ => return None,
  };
  Some((numer, denom))
}

/// One event of a measure: a note (with its pitch object) or a rest, together
/// with its explicit rhythmic value if one was given (`None` for the default
/// one-beat duration).
enum MeasureEvent {
  Note(Expr, Option<Ratio>),
  Rest(Option<Ratio>),
}

impl MeasureEvent {
  /// Whether the event carries an explicit (rigid) duration. Default events are
  /// elastic: a trailing default note stretches to fill the measure.
  fn is_explicit(&self) -> bool {
    matches!(
      self,
      MeasureEvent::Note(_, Some(_)) | MeasureEvent::Rest(Some(_))
    )
  }
}

/// Extract an explicit rhythmic value from a canonical note/rest association,
/// distinguishing "no duration key" (`Some(None)` — a default event) from a
/// present-but-unparsable duration (`None` — bail out).
fn explicit_duration(pairs: &[(Expr, Expr)]) -> Option<Option<Ratio>> {
  match assoc_get(pairs, "Duration") {
    None => Some(None),
    Some(dur) => music_duration_ratio(dur).map(Some),
  }
}

/// Parse a measure event: a canonical `MusicNote[<|…|>]`/`MusicRest[<|…|>]`, or
/// a bare pitch specification (a name string, `MusicPitch`, MIDI integer, …),
/// which is taken as a default (one-beat) note.
fn parse_measure_event(expr: &Expr) -> Option<MeasureEvent> {
  if let Expr::FunctionCall { name, args } = expr
    && let Some(Expr::Association(pairs)) = args.first()
  {
    match name.as_str() {
      "MusicNote" => {
        let pitch = assoc_get(pairs, "Pitch")?.clone();
        return Some(MeasureEvent::Note(pitch, explicit_duration(pairs)?));
      }
      "MusicRest" => {
        return Some(MeasureEvent::Rest(explicit_duration(pairs)?));
      }
      _ => return None,
    }
  }
  // A bare pitch (e.g. `"C"`, `MusicPitch["C4"]`) is a default note.
  resolve_pitch_object(expr).map(|pitch| MeasureEvent::Note(pitch, None))
}

/// Build the beat-annotated `MusicDuration[<|…|>]` for a measure event. The
/// stored `"Duration"` key is kept only for events that were given an explicit
/// value; every event gains `"BeatDuration"` and `"Beats"`.
fn beat_annotated_duration(
  explicit: Option<Ratio>,
  beat_duration: Ratio,
  beats: Ratio,
) -> Expr {
  let mut pairs = Vec::new();
  if let Some(value) = explicit {
    pairs.push(("Duration", ratio_expr(value)));
  }
  pairs.push(("BeatDuration", ratio_expr(beat_duration)));
  pairs.push(("Beats", ratio_expr(beats)));
  music_assoc("MusicDuration", pairs)
}

/// Rebuild a measure event as its beat-annotated canonical object.
fn annotate_event(
  event: &MeasureEvent,
  beat_duration: Ratio,
  beats: Ratio,
) -> Expr {
  match event {
    MeasureEvent::Note(pitch, explicit) => music_assoc(
      "MusicNote",
      vec![
        ("Pitch", pitch.clone()),
        (
          "Duration",
          beat_annotated_duration(*explicit, beat_duration, beats),
        ),
      ],
    ),
    MeasureEvent::Rest(explicit) => music_assoc(
      "MusicRest",
      vec![(
        "Duration",
        beat_annotated_duration(*explicit, beat_duration, beats),
      )],
    ),
  }
}

/// `MusicMeasure[{events…}, MusicTimeSignature[n, d]]` — resolve the measure's
/// rhythm against its meter.
///
/// The beat unit follows the meter: a *compound* meter (numerator divisible by
/// three, with numerator ≥ 6 or denominator ≥ 8, e.g. 6/8, 9/8, 6/4, 3/8)
/// beats in dotted values (`BeatDuration = 3/d`, `numerator/3` beats per
/// measure); every other (*simple*) meter beats in `1/d` with `numerator`
/// beats per measure. A default note or rest occupies one beat.
///
/// When the events' total beats exceed the measure, the `MusicMeasure::measdur`
/// warning is emitted and the expression is returned in its non-associated
/// form. Otherwise the measure is packed into
/// `MusicMeasure[<|"NoteList" -> {…}, "TimeSignature" -> …|>]`, each event
/// annotated with its `BeatDuration`/`Beats`, the final event stretched (or a
/// padding rest appended) to fill the measure exactly. Anything that is not a
/// resolvable list of notes/rests with a canonical time signature is left
/// symbolic (`None`).
pub fn music_measure(args: &[Expr]) -> Option<Expr> {
  // A measure with no explicit meter defaults to common time; one with a meter
  // resolves against it. An empty `MusicMeasure[]`/`MusicMeasure[{}]` fills with
  // a whole-measure rest.
  let (events_raw, timesig, ts_defaulted): (crate::ExprList, Expr, bool) =
    match args {
      [] => (vec![].into(), default_time_signature(), true),
      [Expr::List(events)] => (events.clone(), default_time_signature(), true),
      [Expr::List(events), ts] => (events.clone(), ts.clone(), false),
      _ => return None,
    };
  let events_raw = &events_raw;
  let timesig = &timesig;
  let (numer, denom) = time_signature_parts(timesig)?;

  // Simple vs. compound meter fixes the beat unit and the beats per measure.
  let compound = numer % 3 == 0 && (numer >= 6 || denom >= 8);
  let beat_duration: Ratio = if compound {
    reduce(3, denom)
  } else {
    reduce(1, denom)
  };
  let capacity: Ratio = if compound {
    reduce(numer, 3)
  } else {
    (numer, 1)
  };

  let events: Vec<MeasureEvent> = events_raw
    .iter()
    .map(parse_measure_event)
    .collect::<Option<_>>()?;
  // An empty measure is filled with a single whole-measure rest. WL orders this
  // generated rest's duration keys `Beats`-then-`BeatDuration` (the reverse of a
  // note's) and gives it no stored `Duration`; a defaulted meter is reported as
  // the empty `MusicTimeSignature[<||>]`.
  if events.is_empty() {
    let fill_rest = music_assoc(
      "MusicRest",
      vec![(
        "Duration",
        music_assoc(
          "MusicDuration",
          vec![
            ("Beats", ratio_expr(capacity)),
            ("BeatDuration", ratio_expr(beat_duration)),
          ],
        ),
      )],
    );
    let reported_ts = if ts_defaulted {
      music_assoc("MusicTimeSignature", vec![])
    } else {
      timesig.clone()
    };
    return Some(music_assoc(
      "MusicMeasure",
      vec![
        ("NoteList", Expr::List(vec![fill_rest].into())),
        ("TimeSignature", reported_ts),
      ],
    ));
  }

  // Nominal beats: an explicit duration measures `duration / beatDuration`
  // beats; a default event is one beat.
  let nominal: Vec<Ratio> = events
    .iter()
    .map(|e| match e {
      MeasureEvent::Note(_, Some(v)) | MeasureEvent::Rest(Some(v)) => {
        ratio_mul(*v, (beat_duration.1, beat_duration.0))
      }
      _ => (1, 1),
    })
    .collect();
  let total = nominal.iter().fold((0, 1), |acc, &b| ratio_add(acc, b));

  // Overfull: warn and return the non-associated form unchanged.
  if ratio_cmp(total, capacity) == std::cmp::Ordering::Greater {
    crate::emit_message_to_stdout(&format!(
      "MusicMeasure::measdur: The total duration of beats {} exceeds the \
       allowed number of beats per measure {}.",
      format_ratio(total),
      capacity.0,
    ));
    return Some(Expr::FunctionCall {
      name: "MusicMeasure".to_string(),
      args: vec![Expr::List(events_raw.clone()), timesig.clone()].into(),
    });
  }

  // Fill the measure to exactly `capacity` beats.
  let deficit = ratio_sub(capacity, total);
  let last = events.len() - 1;
  let last_is_elastic = !events[last].is_explicit();

  let mut note_list: Vec<Expr> = Vec::with_capacity(events.len() + 1);
  for (i, event) in events.iter().enumerate() {
    let beats = if i == last && last_is_elastic {
      ratio_add(nominal[i], deficit)
    } else {
      nominal[i]
    };
    note_list.push(annotate_event(event, beat_duration, beats));
  }
  // A rigid final event cannot stretch, so pad the remainder with a rest.
  if deficit != (0, 1) && !last_is_elastic {
    let pad = MeasureEvent::Rest(Some(ratio_mul(deficit, beat_duration)));
    note_list.push(annotate_event(&pad, beat_duration, deficit));
  }

  Some(music_assoc(
    "MusicMeasure",
    vec![
      ("NoteList", Expr::List(note_list.into())),
      ("TimeSignature", timesig.clone()),
    ],
  ))
}

/// The auto-generated default meter (common time) for a measure/voice/score
/// with no explicit time signature. The Wolfram Language tags this generated
/// signature with a `BeatLength -> 1` marker that an explicitly-given meter
/// does not carry.
fn default_time_signature() -> Expr {
  music_assoc(
    "MusicTimeSignature",
    vec![
      ("Numerator", Expr::Integer(4)),
      ("Denominator", Expr::Integer(4)),
      ("BeatLength", Expr::Integer(1)),
    ],
  )
}

/// Whether `expr` is a resolved measure `MusicMeasure[<|…|>]`.
fn is_resolved_measure(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args }
    if name == "MusicMeasure"
      && matches!(args.first(), Some(Expr::Association(_))))
}

/// Whether `expr` is a resolved voice `MusicVoice[<|…|>]`.
fn is_resolved_voice(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args }
    if name == "MusicVoice"
      && matches!(args.first(), Some(Expr::Association(_))))
}

/// The `"TimeSignature"` entry of a resolved measure/voice association.
fn resolved_time_signature(expr: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { args, .. } = expr else {
    return None;
  };
  let Expr::Association(pairs) = args.first()? else {
    return None;
  };
  assoc_get(pairs, "TimeSignature").cloned()
}

/// `MusicVoice[{measures…}]` / `MusicVoice[{notes…}]` — resolve to the
/// association form `MusicVoice[<|"MeasureList" -> {…}, "TimeSignature" -> …|>]`.
/// A voice of measures keeps each already-resolved measure; a voice of loose
/// notes wraps them into a single default-meter measure. The voice's
/// `"TimeSignature"` is that of its first measure. An empty voice resolves to
/// `MusicVoice[<|"MeasureList" -> {}|>]`. Mixed measure/note content (which the
/// Wolfram Language rejects) is left symbolic (`None`).
pub fn music_voice(args: &[Expr]) -> Option<Expr> {
  let items: crate::ExprList = match args {
    [] => vec![].into(),
    [Expr::List(items)] => items.clone(),
    _ => return None,
  };
  if items.is_empty() {
    return Some(music_assoc(
      "MusicVoice",
      vec![("MeasureList", Expr::List(vec![].into()))],
    ));
  }
  let measures: Vec<Expr> = if items.iter().all(is_resolved_measure) {
    items.to_vec()
  } else if items.iter().any(is_resolved_measure) {
    return None; // mixed measures and loose events
  } else {
    // All loose events: pack them into one default-meter measure.
    vec![music_measure(&[Expr::List(items.clone())])?]
  };
  let time_signature = resolved_time_signature(measures.first()?)?;
  Some(music_assoc(
    "MusicVoice",
    vec![
      ("MeasureList", Expr::List(measures.into())),
      ("TimeSignature", time_signature),
    ],
  ))
}

/// `MusicScore[{voices…}]` — resolve to the association form
/// `MusicScore[<|"VoiceList" -> {…}, "TimeSignature" -> …|>]`, keeping each
/// already-resolved voice and taking its `"TimeSignature"` from the first
/// voice. An empty score resolves to `MusicScore[<|"VoiceList" -> {}|>]`. Any
/// non-voice element leaves the score symbolic (`None`).
pub fn music_score(args: &[Expr]) -> Option<Expr> {
  let items: crate::ExprList = match args {
    [] => vec![].into(),
    [Expr::List(items)] => items.clone(),
    _ => return None,
  };
  if items.is_empty() {
    return Some(music_assoc(
      "MusicScore",
      vec![("VoiceList", Expr::List(vec![].into()))],
    ));
  }
  if !items.iter().all(is_resolved_voice) {
    return None;
  }
  let time_signature = resolved_time_signature(items.first()?)?;
  Some(music_assoc(
    "MusicScore",
    vec![
      ("VoiceList", Expr::List(items.clone())),
      ("TimeSignature", time_signature),
    ],
  ))
}

/// Render an expression for a music error message the way wolframscript
/// does: a canonical music object (a music head wrapping an association)
/// prints as its summary form `-Head-`; everything else prints in OutputForm.
fn message_form(expr: &Expr) -> String {
  fn summarize(expr: &Expr) -> Expr {
    match expr {
      Expr::FunctionCall { name, args }
        if MUSIC_OBJECT_HEADS.contains(&name.as_str())
          && matches!(args.first(), Some(Expr::Association(_))) =>
      {
        Expr::Identifier(format!("-{name}-"))
      }
      Expr::FunctionCall { name, args } => Expr::FunctionCall {
        name: name.clone(),
        args: args.iter().map(summarize).collect::<Vec<_>>().into(),
      },
      Expr::List(items) => {
        Expr::List(items.iter().map(summarize).collect::<Vec<_>>().into())
      }
      other => other.clone(),
    }
  }
  crate::syntax::format_expr(&summarize(expr), crate::syntax::ExprForm::Output)
}

/// `MusicScale[…]` with a second argument. The Wolfram Language only accepts
/// a property *association* there; anything else (a pitch object, a name
/// string, …) emits `MusicScale::passc` and leaves the scale unevaluated.
pub fn music_scale(args: &[Expr]) -> Option<Expr> {
  match args {
    [_, second, ..] if !matches!(second, Expr::Association(_)) => {
      crate::emit_message_to_stdout(&format!(
        "MusicScale::passc: {} is not a valid property association.",
        message_form(second)
      ));
      Some(unevaluated("MusicScale", args))
    }
    _ => None,
  }
}

/// Whether `expr` is a `MusicScale` left unevaluated by [`music_scale`]'s
/// argument check — the form `MusicPlot` rejects with `MusicPlot::music`.
pub fn is_invalid_music_scale(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, args }
    if name == "MusicScale"
      && args.len() >= 2
      && !matches!(&args[1], Expr::Association(_)))
}

/// Emit the `MusicPlot::music` message for a non-music argument.
pub fn emit_music_plot_message(arg: &Expr) {
  crate::emit_message_to_stdout(&format!(
    "MusicPlot::music: Expecting a valid music object instead of {}.",
    message_form(arg)
  ));
}

/// Render a beat count the way the Wolfram Language prints it in the
/// `MusicMeasure::measdur` message (an integer, or `n/d` for a fraction).
fn format_ratio((n, d): Ratio) -> String {
  if d == 1 {
    n.to_string()
  } else {
    format!("{}/{}", n, d)
  }
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

  /// Assert that a `music_pitch` result is the MIDI-numbered association
  /// `MusicPitch[<|"MIDINumber" -> midi|>]`.
  fn expect_midi_pitch(result: Option<Expr>, midi: i128) {
    match &result {
      Some(Expr::FunctionCall { name, args }) if name == "MusicPitch" => {
        match &args[..] {
          [Expr::Association(pairs)] => {
            assert!(
              matches!(
                assoc_get(pairs, "MIDINumber"),
                Some(Expr::Integer(n)) if *n == midi
              ),
              "MIDINumber mismatch in {result:?}",
            );
            assert_eq!(pairs.len(), 1, "unexpected extra keys in {result:?}");
          }
          other => panic!("expected an association arg, got {other:?}"),
        }
      }
      other => {
        panic!("expected MusicPitch[<|MIDINumber -> …|>], got {other:?}")
      }
    }
  }

  #[test]
  fn music_pitch_canonicalizes_midi_integer() {
    expect_midi_pitch(music_pitch(&[Expr::Integer(60)]), 60);
  }

  #[test]
  fn music_pitch_spells_string_spec() {
    // A named pitch canonicalizes to its spelled association; the octave adds
    // `Octave`/`Name` keys, an octaveless name keeps only Accidental/Key.
    let result = music_pitch(&[Expr::String("C4".to_string())]);
    match &result {
      Some(Expr::FunctionCall { name, args }) if name == "MusicPitch" => {
        match &args[..] {
          [Expr::Association(pairs)] => {
            assert!(matches!(
              assoc_get(pairs, "Octave"),
              Some(Expr::Integer(4))
            ));
            assert!(
              matches!(assoc_get(pairs, "Key"), Some(Expr::String(k)) if k == "C")
            );
            assert!(
              matches!(assoc_get(pairs, "Name"), Some(Expr::String(n)) if n == "C")
            );
          }
          other => panic!("expected an association arg, got {other:?}"),
        }
      }
      other => panic!("expected MusicPitch[<|…|>], got {other:?}"),
    }
    // An association spec is already canonical and stays untouched.
    assert!(
      music_pitch(&[Expr::Association(vec![(
        Expr::String("MIDINumber".into()),
        Expr::Integer(60),
      )])])
      .is_none()
    );
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

  fn func(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: args.into(),
    }
  }

  #[test]
  fn music_pitch_canonicalizes_frequency() {
    // Frequencies fix no letter spelling and keep their MIDI number:
    // A4 = 440 Hz = MIDI 69; 200 Hz is nearest G3 = MIDI 55.
    expect_midi_pitch(
      music_pitch(&[func(
        "Quantity",
        vec![Expr::Integer(440), Expr::String("Hertz".into())],
      )]),
      69,
    );
    expect_midi_pitch(
      music_pitch(&[func(
        "Quantity",
        vec![Expr::Integer(200), Expr::String("Hertz".into())],
      )]),
      55,
    );
    // Identifier unit spelling and the "Hz" abbreviation are both accepted.
    expect_midi_pitch(
      music_pitch(&[func(
        "Quantity",
        vec![Expr::Real(440.0), Expr::Identifier("Hertz".into())],
      )]),
      69,
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
    // SoundNote numbers pitches relative to middle C (0), so 0 -> MIDI 60.
    expect_midi_pitch(
      music_pitch(&[func("SoundNote", vec![Expr::Integer(0)])]),
      60,
    );
    expect_midi_pitch(
      music_pitch(&[func("SoundNote", vec![Expr::Integer(-5)])]),
      55,
    );
    expect_midi_pitch(
      music_pitch(&[func("SoundNote", vec![Expr::String("A4".into())])]),
      69,
    );
  }

  #[test]
  fn music_pitch_extracts_note_pitch() {
    // A raw MusicNote spec resolves to the spelled note-pitch object …
    let result =
      music_pitch(&[func("MusicNote", vec![Expr::String("G3".into())])]);
    match &result {
      Some(Expr::FunctionCall { name, args }) if name == "MusicPitch" => {
        match &args[..] {
          [Expr::Association(pairs)] => {
            assert!(matches!(
              assoc_get(pairs, "Octave"),
              Some(Expr::Integer(3))
            ));
            assert!(
              matches!(assoc_get(pairs, "Key"), Some(Expr::String(k)) if k == "G")
            );
          }
          other => panic!("expected an association arg, got {other:?}"),
        }
      }
      other => panic!("expected MusicPitch[<|…|>], got {other:?}"),
    }
    // … while a canonical note returns its stored pitch object verbatim.
    let stored = func(
      "MusicNote",
      vec![Expr::Association(vec![(
        Expr::String("Pitch".into()),
        func(
          "MusicPitch",
          vec![Expr::Association(vec![(
            Expr::String("MIDINumber".into()),
            Expr::Integer(55),
          )])],
        ),
      )])],
    );
    expect_midi_pitch(music_pitch(&[stored]), 55);
  }

  fn named_pitch(name: &str) -> Expr {
    func("MusicPitch", vec![Expr::String(name.into())])
  }

  /// Assert that `try_music_pitch_plus` produced the canonical association form
  /// with the given accidental, key and MIDI number.
  fn expect_pitch_sum(result: Option<Expr>, acc: i128, key: &str, midi: i128) {
    match &result {
      Some(Expr::FunctionCall { name, args }) if name == "MusicPitch" => {
        match &args[..] {
          [Expr::Association(pairs)] => {
            let get = |k: &str| {
              pairs.iter().find_map(|(pk, pv)| match pk {
                Expr::String(s) if s == k => Some(pv),
                _ => None,
              })
            };
            assert!(
              matches!(get("Accidental"), Some(Expr::Integer(n)) if *n == acc),
              "accidental mismatch in {result:?}",
            );
            assert!(
              matches!(get("Key"), Some(Expr::String(s)) if s == key),
              "key mismatch in {result:?}",
            );
            assert!(
              matches!(get("MIDINumber"), Some(Expr::Integer(n)) if *n == midi),
              "MIDI mismatch in {result:?}",
            );
          }
          other => panic!("expected an association arg, got {other:?}"),
        }
      }
      other => panic!("expected MusicPitch[<|…|>], got {other:?}"),
    }
  }

  #[test]
  fn music_pitch_plus_matches_documented_example() {
    // MusicPitch["Bb"] + MusicPitch["A#"] - MusicPitch["C"]
    //   == MusicPitch[<|"Accidental" -> 1, "Key" -> "G", "MIDINumber" -> 80|>]
    let result = try_music_pitch_plus(&[
      named_pitch("Bb"),
      named_pitch("A#"),
      func("Times", vec![Expr::Integer(-1), named_pitch("C")]),
    ]);
    expect_pitch_sum(result, 1, "G", 80);
  }

  #[test]
  fn music_pitch_plus_is_associative_via_interval_form() {
    // The intermediate `Bb - C` decodes back exactly as a MusicInterval, so
    // adding `A#` afterwards gives the same answer as summing all three at
    // once.
    let partial = try_music_pitch_plus(&[
      named_pitch("Bb"),
      func("Times", vec![Expr::Integer(-1), named_pitch("C")]),
    ])
    .unwrap();
    let result = try_music_pitch_plus(&[named_pitch("A#"), partial]);
    expect_pitch_sum(result, 1, "G", 80);
  }

  #[test]
  fn music_pitch_plus_unary_minus_subtraction() {
    // A - A + A == A, exercising the UnaryOp::Minus summand path.
    let result = try_music_pitch_plus(&[
      named_pitch("A4"),
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand: Box::new(named_pitch("A4")),
      },
      named_pitch("A4"),
    ]);
    expect_pitch_sum(result, 0, "A", 69);
  }

  #[test]
  fn music_pitch_plus_plain_sum_declines() {
    // Plain pitches only combine through intervals — `p + p` is left for the
    // ordinary Plus path, which collects it to `2 p`.
    assert!(
      try_music_pitch_plus(&[named_pitch("C"), named_pitch("C")]).is_none()
    );
    // Distinct plain pitches don't combine either.
    assert!(
      try_music_pitch_plus(&[named_pitch("C"), named_pitch("D")]).is_none()
    );
    // A scaled pitch never takes part in pitch arithmetic (`2 p - p`
    // collects to `p` on the ordinary path).
    assert!(
      try_music_pitch_plus(&[
        func("Times", vec![Expr::Integer(2), named_pitch("C")]),
        func("Times", vec![Expr::Integer(-1), named_pitch("C")]),
      ])
      .is_none()
    );
  }

  #[test]
  fn music_pitch_plus_octaveless_names_default_to_octave_4() {
    // Octaveless `C` defaults to octave 4, so `C - C4` cancels to a Unison.
    let result = try_music_pitch_plus(&[
      named_pitch("C"),
      func("Times", vec![Expr::Integer(-1), named_pitch("C4")]),
    ]);
    match &result {
      Some(Expr::FunctionCall { name, args }) if name == "MusicInterval" => {
        match &args[..] {
          [Expr::Association(pairs)] => {
            let get = |k: &str| {
              pairs.iter().find_map(|(pk, pv)| match pk {
                Expr::String(s) if s == k => Some(pv),
                _ => None,
              })
            };
            assert!(
              matches!(get("Semitones"), Some(Expr::Integer(0))),
              "semitones mismatch in {result:?}",
            );
            assert!(
              matches!(get("Name"), Some(Expr::String(s)) if s == "Unison"),
              "name mismatch in {result:?}",
            );
          }
          other => panic!("expected an association arg, got {other:?}"),
        }
      }
      other => panic!("expected MusicInterval[<|…|>], got {other:?}"),
    }
  }

  #[test]
  fn music_pitch_plus_rejects_non_pitch_and_lone_terms() {
    // A sum mixing a pitch with a non-pitch is not pitch arithmetic.
    assert!(
      try_music_pitch_plus(&[named_pitch("C"), Expr::Integer(5)]).is_none()
    );
    // A single pitch is not arithmetic and is left for the normal path.
    assert!(try_music_pitch_plus(&[named_pitch("Bb")]).is_none());
  }

  fn named_interval(spec: &str) -> Expr {
    func("MusicInterval", vec![Expr::String(spec.into())])
  }

  #[test]
  fn interval_axes_cover_names_semitones_and_associations() {
    // Named interval: a minor third spans two letters and three semitones.
    assert_eq!(
      music_interval_axes(&named_interval("MinorThird")),
      Some((2, 3))
    );
    // Bare-semitone interval: 5 semitones is the simplest perfect fourth.
    assert_eq!(
      music_interval_axes(&func("MusicInterval", vec![Expr::Integer(5)])),
      Some((3, 5)),
    );
    // Compound (octave-plus) interval via the canonical association form.
    let assoc = func(
      "MusicInterval",
      vec![Expr::Association(
        vec![
          (Expr::String("Semitones".into()), Expr::Integer(4)),
          (
            Expr::String("Name".into()),
            Expr::String("MajorThird".into()),
          ),
          (Expr::String("CompoundOctaves".into()), Expr::Integer(1)),
        ]
        .into(),
      )],
    );
    assert_eq!(music_interval_axes(&assoc), Some((9, 16)));
  }

  #[test]
  fn music_pitch_plus_interval_transposes() {
    // C4 + a minor third -> Eb4 (accidental -1, MIDI 63).
    expect_pitch_sum(
      try_music_pitch_plus(&[named_pitch("C"), named_interval("MinorThird")]),
      -1,
      "E",
      63,
    );
    // C4 + a perfect fifth -> G4.
    expect_pitch_sum(
      try_music_pitch_plus(&[named_pitch("C"), named_interval("PerfectFifth")]),
      0,
      "G",
      67,
    );
  }

  #[test]
  fn music_pitch_plus_interval_only_is_not_a_pitch() {
    // A sum with no pitch (two intervals) is not a pitch and stays symbolic.
    assert!(
      try_music_pitch_plus(&[
        named_interval("MinorThird"),
        named_interval("MajorThird"),
      ])
      .is_none()
    );
  }

  #[test]
  fn music_chord_plus_interval_transposes_every_tone() {
    // MusicChord[{C4, E4, G4}] + MusicInterval[5] -> {F4, A4, C5}.
    let chord = func(
      "MusicChord",
      vec![Expr::List(
        vec![named_pitch("C4"), named_pitch("E4"), named_pitch("G4")].into(),
      )],
    );
    let result = try_music_chord_plus_interval(&[
      chord,
      func("MusicInterval", vec![Expr::Integer(5)]),
    ])
    .expect("chord + interval should transpose");
    let Expr::FunctionCall { name, args } = &result else {
      panic!("expected MusicChord[…], got {result:?}");
    };
    assert_eq!(name, "MusicChord");
    // The transposed chord is the canonical `<|"PitchList" -> {…}|>` form.
    let Some(Expr::Association(pairs)) = args.first() else {
      panic!("expected a PitchList association, got {args:?}");
    };
    let Some(Expr::List(items)) = assoc_get(pairs, "PitchList") else {
      panic!("expected a PitchList entry, got {pairs:?}");
    };
    let midis: Vec<i128> =
      items.iter().filter_map(|p| music_pitch_midi(p)).collect();
    assert_eq!(midis, vec![65, 69, 72]);
    // A bare-semitone interval spells the tones straight from MIDI: F, A, C.
    let keys: Vec<String> = items
      .iter()
      .filter_map(|p| match p {
        Expr::FunctionCall { args, .. } => match args.first()? {
          Expr::Association(pairs) => match assoc_get(pairs, "Key")? {
            Expr::String(s) => Some(s.clone()),
            _ => None,
          },
          _ => None,
        },
        _ => None,
      })
      .collect();
    assert_eq!(keys, vec!["F", "A", "C"]);
  }

  #[test]
  fn music_pitch_plus_chromatic_interval_spells_from_midi() {
    // A bare-semitone interval has no diatonic step: C + MusicInterval[8] is the
    // MIDI-default G#, not the diatonic Ab.
    expect_pitch_sum(
      try_music_pitch_plus(&[
        named_pitch("C"),
        func("MusicInterval", vec![Expr::Integer(8)]),
      ]),
      1,
      "G",
      68,
    );
    // Eb + MusicInterval[8] lands on MIDI 71 -> B, again spelled from MIDI.
    expect_pitch_sum(
      try_music_pitch_plus(&[
        named_pitch("Eb"),
        func("MusicInterval", vec![Expr::Integer(8)]),
      ]),
      0,
      "B",
      71,
    );
  }

  #[test]
  fn music_chord_plus_interval_rejects_non_chord_sums() {
    // No chord summand: not chord transposition.
    assert!(
      try_music_chord_plus_interval(&[
        named_pitch("C"),
        func("MusicInterval", vec![Expr::Integer(5)]),
      ])
      .is_none()
    );
    // A bare chord with no interval is left for the normal path.
    let chord = func(
      "MusicChord",
      vec![Expr::List(vec![named_pitch("C4")].into())],
    );
    assert!(try_music_chord_plus_interval(&[chord]).is_none());
  }
}
