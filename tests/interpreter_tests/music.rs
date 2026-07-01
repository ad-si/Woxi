use super::*;

// ─── MusicObjectQ ────────────────────────────────────────────────────────────

#[test]
fn music_object_q_true_for_pitch() {
  assert_eq!(
    interpret("MusicObjectQ[MusicPitch[\"C4\"]]").unwrap(),
    "True"
  );
}

#[test]
fn music_object_q_true_for_note() {
  assert_eq!(
    interpret("MusicObjectQ[MusicNote[MusicPitch[\"C4\"]]]").unwrap(),
    "True"
  );
}

#[test]
fn music_object_q_true_for_chord_and_containers() {
  assert_eq!(
    interpret(
      "MusicObjectQ[MusicChord[{MusicPitch[\"C\"], MusicPitch[\"E\"]}]]"
    )
    .unwrap(),
    "True"
  );
  assert_eq!(interpret("MusicObjectQ[MusicScore[]]").unwrap(), "True");
  assert_eq!(interpret("MusicObjectQ[MusicMeasure[]]").unwrap(), "True");
}

#[test]
fn music_object_q_false_for_non_objects() {
  assert_eq!(interpret("MusicObjectQ[3]").unwrap(), "False");
  assert_eq!(interpret("MusicObjectQ[{1, 2, 3}]").unwrap(), "False");
  assert_eq!(interpret("MusicObjectQ[\"C4\"]").unwrap(), "False");
}

#[test]
fn music_object_q_false_for_operations() {
  // MusicPlot/MusicTransform/MusicMeasurements are operations, not objects.
  assert_eq!(interpret("MusicObjectQ[MusicPlot[x]]").unwrap(), "False");
  assert_eq!(
    interpret("MusicObjectQ[MusicTransform[x, y]]").unwrap(),
    "False"
  );
}

// ─── MusicPitch ──────────────────────────────────────────────────────────────

#[test]
fn music_pitch_canonicalizes_midi_number() {
  // Middle C is MIDI 60 / C4; A4 is MIDI 69.
  assert_eq!(interpret("MusicPitch[60]").unwrap(), "MusicPitch[C4]");
  assert_eq!(interpret("MusicPitch[69]").unwrap(), "MusicPitch[A4]");
  assert_eq!(interpret("MusicPitch[61]").unwrap(), "MusicPitch[C#4]");
  assert_eq!(interpret("MusicPitch[0]").unwrap(), "MusicPitch[C-1]");
}

#[test]
fn music_pitch_string_stays_symbolic() {
  assert_eq!(interpret("MusicPitch[\"C4\"]").unwrap(), "MusicPitch[C4]");
}

#[test]
fn music_pitch_canonicalizes_frequency() {
  // A4 is 440 Hz; 200 Hz is the nearest to G3 (per the MusicPitch docs).
  assert_eq!(
    interpret("MusicPitch[Quantity[440, \"Hertz\"]]").unwrap(),
    "MusicPitch[A4]"
  );
  assert_eq!(
    interpret("MusicPitch[Quantity[200, \"Hertz\"]]").unwrap(),
    "MusicPitch[G3]"
  );
}

#[test]
fn music_pitch_canonicalizes_soundnote() {
  // SoundNote numbers pitches relative to middle C (0 -> C4).
  assert_eq!(
    interpret("MusicPitch[SoundNote[0]]").unwrap(),
    "MusicPitch[C4]"
  );
  assert_eq!(
    interpret("MusicPitch[SoundNote[-5]]").unwrap(),
    "MusicPitch[G3]"
  );
}

#[test]
fn music_pitch_extracts_note_pitch() {
  assert_eq!(
    interpret("MusicPitch[MusicNote[\"G3\"]]").unwrap(),
    "MusicPitch[G3]"
  );
  assert_eq!(
    interpret("MusicPitch[MusicNote[MusicPitch[55]]]").unwrap(),
    "MusicPitch[G3]"
  );
}

#[test]
fn music_pitch_head_is_music_pitch() {
  assert_eq!(interpret("Head[MusicPitch[60]]").unwrap(), "MusicPitch");
}

// ─── Pitch arithmetic ────────────────────────────────────────────────────────

#[test]
fn music_pitch_arithmetic_documented_example() {
  // Adding/subtracting pitches combines them along the diatonic-position and
  // MIDI-number axes; octaveless names default to octave 4.
  assert_eq!(
    interpret("MusicPitch[\"Bb\"] + MusicPitch[\"A#\"] - MusicPitch[\"C\"]")
      .unwrap(),
    "MusicPitch[<|Accidental -> 1, Key -> G, MIDINumber -> 80|>]"
  );
}

#[test]
fn music_pitch_arithmetic_octave_climb() {
  // C4 + C4 rises one diatonic octave.
  assert_eq!(
    interpret("MusicPitch[\"C\"] + MusicPitch[\"C\"]").unwrap(),
    "MusicPitch[<|Accidental -> 12, Key -> C, MIDINumber -> 120|>]"
  );
}

#[test]
fn music_pitch_arithmetic_cancels() {
  // A - A + A == A (round-trips through the canonical association form).
  assert_eq!(
    interpret("MusicPitch[\"A4\"] - MusicPitch[\"A4\"] + MusicPitch[\"A4\"]")
      .unwrap(),
    "MusicPitch[<|Accidental -> 0, Key -> A, MIDINumber -> 69|>]"
  );
}

#[test]
fn music_pitch_mixed_sum_stays_symbolic() {
  // A pitch plus a plain number is not pitch arithmetic; it stays symbolic.
  assert_eq!(
    interpret("MusicPitch[\"C\"] + 5").unwrap(),
    "5 + MusicPitch[C]"
  );
}

// ─── Staff-notation rendering ────────────────────────────────────────────────

#[test]
fn export_music_note_to_svg() {
  // ExportString[obj, "SVG"] renders the object as musical-staff notation.
  let svg =
    interpret("ExportString[MusicNote[MusicPitch[\"C4\"]], \"SVG\"]").unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert!(svg.contains("class=\"notehead\"")); // a Leland note head
  assert!(svg.contains("class=\"clef\"")); // the Leland treble clef
  assert!(svg.contains("<line")); // staff lines
}

#[test]
fn export_music_chord_renders_three_heads() {
  let svg = interpret(
    "ExportString[MusicChord[{MusicPitch[\"C4\"], MusicPitch[\"E4\"], \
     MusicPitch[\"G4\"]}], \"SVG\"]",
  )
  .unwrap();
  assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
}

#[test]
fn bare_pitch_is_not_staff_notation() {
  // A `MusicPitch` carries no rhythmic value, so it is not drawn on a staff —
  // unlike a `MusicChord`, which is a musical event. The pitch's SVG therefore
  // has no staff glyphs (clef/note head), while the chord's does.
  let pitch = interpret("ExportString[MusicPitch[\"C4\"], \"SVG\"]").unwrap();
  assert!(!pitch.contains("class=\"notehead\""), "got: {pitch}");
  assert!(!pitch.contains("class=\"clef\""), "got: {pitch}");
  let chord =
    interpret("ExportString[MusicChord[{MusicPitch[\"C4\"]}], \"SVG\"]")
      .unwrap();
  assert!(chord.contains("class=\"clef\""), "got: {chord}");
}

#[test]
fn music_plot_returns_graphics() {
  // MusicPlot draws the object; in the CLI a Graphics prints as -Graphics-.
  assert_eq!(
    interpret("MusicPlot[MusicScale[\"Major\", MusicPitch[\"C4\"]]]").unwrap(),
    "-Graphics-"
  );
}

#[test]
fn bare_music_object_stays_symbolic_in_cli() {
  // Without ExportString/MusicPlot the CLI keeps the canonical symbolic form
  // (only the visual hosts auto-render it).
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"]]").unwrap(),
    "MusicNote[MusicPitch[C4]]"
  );
}

// ─── Symbolic music objects ──────────────────────────────────────────────────

#[test]
fn music_objects_stay_symbolic() {
  // A single-pitch note (no duration) round-trips as a canonical symbolic
  // object, and a chord given by an explicit pitch list stays symbolic.
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"]]").unwrap(),
    "MusicNote[MusicPitch[C4]]"
  );
  assert_eq!(
    interpret("Head[MusicChord[{MusicPitch[\"C\"]}]]").unwrap(),
    "MusicChord"
  );
}

// ─── MusicNote canonicalization (WL 15) ──────────────────────────────────────

#[test]
fn music_note_canonicalizes_pitch_and_duration() {
  // MusicNote[pitch, duration] canonicalizes to the association form exposing
  // its Pitch and Duration; an octaveless name omits "Octave".
  assert_eq!(
    interpret("MusicNote[\"A#\", 1/2]").unwrap(),
    "MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 1, Key -> A|>], \
     Duration -> MusicDuration[<|Duration -> 1/2|>]|>]"
  );
}

#[test]
fn music_note_from_pitch_and_duration_objects() {
  // Pitch/duration given as objects resolve the same way; a named duration maps
  // to its rhythmic value and an explicit octave is kept.
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"], MusicDuration[\"Half\"]]")
      .unwrap(),
    "MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C|>], \
     Duration -> MusicDuration[<|Duration -> 1/2|>]|>]"
  );
}

#[test]
fn music_note_property_access() {
  assert_eq!(
    interpret("MusicNote[\"A#\", 1/2][\"Pitch\"]").unwrap(),
    "MusicPitch[<|Accidental -> 1, Key -> A|>]"
  );
  assert_eq!(
    interpret("MusicNote[\"A#\", 1/2][\"Duration\"]").unwrap(),
    "MusicDuration[<|Duration -> 1/2|>]"
  );
}

// ─── MusicDuration (WL 15) ───────────────────────────────────────────────────

#[test]
fn music_duration_canonicalizes_number() {
  assert_eq!(
    interpret("MusicDuration[1/4]").unwrap(),
    "MusicDuration[<|Duration -> 1/4|>]"
  );
}

#[test]
fn music_duration_arithmetic() {
  // Durations add (scaled by any leading coefficient): 3·(1/2) + 1/4 = 7/4.
  assert_eq!(
    interpret("3 MusicDuration[<|\"Duration\" -> 1/2|>] + MusicDuration[1/4]")
      .unwrap(),
    "MusicDuration[<|Duration -> 7/4|>]"
  );
  // Subtraction works too: 1 - 1/4 = 3/4.
  assert_eq!(
    interpret("MusicDuration[1] - MusicDuration[1/4]").unwrap(),
    "MusicDuration[<|Duration -> 3/4|>]"
  );
}

// ─── Enharmonic MusicPitch equality (WL 15) ──────────────────────────────────

#[test]
fn music_pitch_enharmonic_equality() {
  // Enharmonic spellings denote the same pitch.
  assert_eq!(
    interpret("MusicPitch[\"C#\"] == MusicPitch[\"Db\"]").unwrap(),
    "True"
  );
  // Different pitches are not equal.
  assert_eq!(
    interpret("MusicPitch[\"C4\"] == MusicPitch[\"D4\"]").unwrap(),
    "False"
  );
  // Unequal is the negation.
  assert_eq!(
    interpret("MusicPitch[\"C#\"] != MusicPitch[\"Db\"]").unwrap(),
    "False"
  );
}

// ─── MusicChord canonicalization + properties (WL 15) ─────────────────────────

#[test]
fn music_chord_canonicalizes_named_chord() {
  assert_eq!(
    interpret("MusicChord[\"GMajor\"]").unwrap(),
    "MusicChord[<|Name -> Major, \
     Root -> MusicPitch[<|Key -> G, Accidental -> 0|>]|>]"
  );
}

#[test]
fn music_chord_pitch_list() {
  // G major triad, spelled on the correct staff letters: G4, B4, D5.
  assert_eq!(
    interpret("MusicChord[\"GMajor\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> B|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 5, Key -> D|>]}"
  );
}

#[test]
fn music_chord_interval_list() {
  // Successive intervals of a major triad: a major third then a minor third.
  assert_eq!(
    interpret("MusicChord[\"GMajor\"][\"IntervalList\"]").unwrap(),
    "{MusicInterval[<|Semitones -> 4, Name -> MajorThird, CompoundOctaves -> 0|>], \
     MusicInterval[<|Semitones -> 3, Name -> MinorThird, CompoundOctaves -> 0|>]}"
  );
}

#[test]
fn music_chord_minor_spelling() {
  // F minor triad spells the third as a flat A (Ab), not G#.
  assert_eq!(
    interpret("MusicChord[\"FMinor\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> F|>], \
     MusicPitch[<|Accidental -> -1, Octave -> 4, Key -> A|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 5, Key -> C|>]}"
  );
}

#[test]
fn music_chord_unknown_quality_stays_symbolic() {
  assert_eq!(interpret("MusicChord[\"Xyz\"]").unwrap(), "MusicChord[Xyz]");
}
