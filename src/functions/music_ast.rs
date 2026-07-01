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

use crate::syntax::{BinaryOperator, Expr, UnaryOperator};

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
      // MusicPitch[MusicNote[pitch, ...]] / MusicPitch[MusicPitch[...]] — take
      // the pitch of the note, which uses MIDI/pitch-name conventions.
      "MusicNote" | "MusicPitch" => resolve_pitch_name(args.first()?),
      _ => None,
    },
    // The canonical `MusicPitch` association form,
    // `<|"Accidental" -> a, ["Octave" -> o,] "Key" -> k[, "MIDINumber" -> n]|>`.
    Expr::Association(pairs) => {
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
    // The canonical `<|"Accidental" -> a, "Key" -> k, "MIDINumber" -> n|>`
    // association: recover the octave from the MIDI number so the staff
    // position is exact.
    Expr::Association(pairs) => {
      let lookup = |key: &str| {
        pairs.iter().find_map(|(k, v)| match k {
          Expr::String(s) if s == key => Some(v),
          _ => None,
        })
      };
      let key = match lookup("Key")? {
        Expr::String(s) => s,
        _ => return None,
      };
      let diatonic = letter_diatonic_index(*key.as_bytes().first()?)?;
      let midi = match lookup("MIDINumber")? {
        Expr::Integer(n) => *n,
        _ => return None,
      };
      let accidental = match lookup("Accidental")? {
        Expr::Integer(n) => *n,
        _ => return None,
      };
      let base = DIATONIC_BASE_SEMITONE[diatonic as usize];
      let octave = (midi - base - accidental).div_euclid(12) - 1;
      Some((octave * 7 + diatonic, midi))
    }
    _ => None,
  }
}

/// Interpret one summand of a pitch sum as `(coefficient, (diatonic, MIDI))`.
/// Bare pitches carry coefficient `+1`; `-MusicPitch[…]` and `k MusicPitch[…]`
/// (integer `k`, from subtraction / scaling) carry the matching coefficient.
fn music_pitch_summand(term: &Expr) -> Option<(i128, (i128, i128))> {
  match term {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (coeff, axes) = music_pitch_summand(operand)?;
      Some((-coeff, axes))
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      let mut coeff = 1i128;
      let mut axes = None;
      for arg in args.iter() {
        if let Expr::Integer(k) = arg {
          coeff *= *k;
        } else if axes.is_none() {
          axes = Some(music_pitch_axes(arg)?);
        } else {
          return None;
        }
      }
      Some((coeff, axes?))
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => match (left.as_ref(), right.as_ref()) {
      (Expr::Integer(k), other) | (other, Expr::Integer(k)) => {
        Some((*k, music_pitch_axes(other)?))
      }
      _ => None,
    },
    _ => Some((1, music_pitch_axes(term)?)),
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

/// `Plus[…]` over `MusicPitch` objects. Returns the combined pitch when *every*
/// summand is a (possibly negated or integer-scaled) `MusicPitch`, and `None`
/// otherwise so ordinary `Plus` evaluation proceeds unchanged (a sum that mixes
/// a pitch with a non-pitch is left symbolic, as in Wolfram).
pub fn try_music_pitch_plus(args: &[Expr]) -> Option<Expr> {
  // Flatten nested `Plus` so a chain such as `a + b - c` — parsed as nested
  // binary operations — is treated as one sum.
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

  let mut total = (0i128, 0i128);
  let mut count = 0;
  for term in &flat {
    let (coeff, (position, midi)) = music_pitch_summand(term)?;
    total.0 += coeff * position;
    total.1 += coeff * midi;
    count += 1;
  }
  // A lone pitch is not arithmetic; leave it for the normal path.
  if count < 2 {
    return None;
  }
  Some(axes_to_music_pitch(total.0, total.1))
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
fn assoc_get<'a>(pairs: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
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

/// Resolve a pitch specification (name string, `MusicPitch`, MIDI integer, …)
/// to the canonical note-pitch object, preserving whether an octave was given.
fn resolve_pitch_object(spec: &Expr) -> Option<Expr> {
  let name = resolve_pitch_name(spec)?;
  let (letter, accidental, octave) = parse_pitch_spelled(&name)?;
  Some(pitch_object(letter, accidental, octave))
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

/// `MusicDuration[spec]` — canonicalize a numeric duration to the association
/// form `MusicDuration[<|"Duration" -> value|>]`. Named durations and the
/// already-canonical association form are left untouched (`None`).
pub fn music_duration(args: &[Expr]) -> Option<Expr> {
  let [spec] = args else {
    return None;
  };
  let value = match spec {
    Expr::Integer(_) | Expr::Real(_) => spec.clone(),
    Expr::FunctionCall { name, .. } if name == "Rational" => spec.clone(),
    _ => return None,
  };
  Some(music_assoc("MusicDuration", vec![("Duration", value)]))
}

/// `MusicNote[pitch, duration]` — canonicalize to the association form
/// `MusicNote[<|"Pitch" -> MusicPitch[…], "Duration" -> MusicDuration[…]|>]`.
/// Returns `None` (leaving the note symbolic) when either part is not a
/// recognized specification.
pub fn music_note(args: &[Expr]) -> Option<Expr> {
  let [pitch_spec, duration_spec] = args else {
    return None;
  };
  let pitch = resolve_pitch_object(pitch_spec)?;
  let duration_value = resolve_duration_value(duration_spec)?;
  let duration =
    music_assoc("MusicDuration", vec![("Duration", duration_value)]);
  Some(music_assoc(
    "MusicNote",
    vec![("Pitch", pitch), ("Duration", duration)],
  ))
}

/// The stacked-thirds `(semitone, diatonic-step)` offsets from the root for
/// each recognized chord quality, or `None` for an unknown quality.
fn chord_quality_offsets(name: &str) -> Option<&'static [(i128, i128)]> {
  Some(match name {
    "Major" => &[(0, 0), (4, 2), (7, 4)],
    "Minor" => &[(0, 0), (3, 2), (7, 4)],
    "Diminished" => &[(0, 0), (3, 2), (6, 4)],
    "Augmented" => &[(0, 0), (4, 2), (8, 4)],
    "MajorSeventh" => &[(0, 0), (4, 2), (7, 4), (11, 6)],
    "MinorSeventh" => &[(0, 0), (3, 2), (7, 4), (10, 6)],
    "DominantSeventh" => &[(0, 0), (4, 2), (7, 4), (10, 6)],
    _ => return None,
  })
}

/// Split a chord name such as `"GMajor"`, `"F#Minor"`, `"BbMajorSeventh"` into
/// its `(root letter, root accidental, quality)` components. The root is the
/// leading note letter plus any `#`/`b` accidentals; the remainder is the
/// quality name.
fn parse_chord_name(name: &str) -> Option<(char, i128, &str)> {
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
  Some((letter as char, accidental, name.get(idx..)?))
}

/// `MusicChord["<root><quality>"]` — canonicalize a named chord to the
/// association form `MusicChord[<|"Name" -> quality, "Root" -> MusicPitch[…]|>]`.
/// A chord given by an explicit pitch list is left symbolic.
pub fn music_chord(args: &[Expr]) -> Option<Expr> {
  let [Expr::String(spec)] = args else {
    return None;
  };
  let (letter, accidental, quality) = parse_chord_name(spec)?;
  // Only recognized qualities canonicalize; anything else stays symbolic.
  chord_quality_offsets(quality)?;
  // The chord's Root pitch is keyed in the order `Key, Accidental`.
  let root = music_assoc(
    "MusicPitch",
    vec![
      ("Key", Expr::String(letter.to_string())),
      ("Accidental", Expr::Integer(accidental)),
    ],
  );
  Some(music_assoc(
    "MusicChord",
    vec![("Name", Expr::String(quality.to_string())), ("Root", root)],
  ))
}

/// Base MIDI number of a note letter in a given octave (natural, no accidental).
fn letter_natural_midi(letter_idx: usize, octave: i128) -> i128 {
  (octave + 1) * 12 + DIATONIC_BASE_SEMITONE[letter_idx]
}

/// Spell out the pitch objects of a canonical chord association's tones. Each
/// tone stacks thirds from the root, spelled on the correct staff letter so the
/// accidentals come out right (`GMajor` → G4, B4, D5).
pub fn chord_tones(pairs: &[(Expr, Expr)]) -> Option<Vec<Expr>> {
  let quality = match assoc_get(pairs, "Name")? {
    Expr::String(s) => s.as_str(),
    _ => return None,
  };
  let offsets = chord_quality_offsets(quality)?;
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
      Some(pitch_object(letter, accidental, Some(octave)))
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
pub fn music_duration_from_value(value: Expr) -> Expr {
  music_assoc("MusicDuration", vec![("Duration", value)])
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
  fn music_pitch_plus_is_associative_via_association_form() {
    // The intermediate `Bb + A#` decodes back exactly, so subtracting `C`
    // afterwards gives the same answer as summing all three at once.
    let partial =
      try_music_pitch_plus(&[named_pitch("Bb"), named_pitch("A#")]).unwrap();
    let result = try_music_pitch_plus(&[
      partial,
      func("Times", vec![Expr::Integer(-1), named_pitch("C")]),
    ]);
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
  fn music_pitch_plus_octaveless_names_default_to_octave_4() {
    // C + C climbs one octave along the diatonic axis (C4 -> C5-ish MIDI 120).
    expect_pitch_sum(
      try_music_pitch_plus(&[named_pitch("C"), named_pitch("C")]),
      12,
      "C",
      120,
    );
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
}
