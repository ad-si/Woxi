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
  assert!(svg.contains("<ellipse")); // a note head
  assert!(svg.contains("<line")); // staff lines
}

#[test]
fn export_music_chord_renders_three_heads() {
  let svg = interpret(
    "ExportString[MusicChord[{MusicPitch[\"C4\"], MusicPitch[\"E4\"], \
     MusicPitch[\"G4\"]}], \"SVG\"]",
  )
  .unwrap();
  assert_eq!(svg.matches("<ellipse").count(), 3);
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
  // Constructors round-trip as canonical symbolic objects (like Sound/SoundNote).
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"], MusicDuration[\"Half\"]]")
      .unwrap(),
    "MusicNote[MusicPitch[C4], MusicDuration[Half]]"
  );
  assert_eq!(
    interpret("Head[MusicChord[{MusicPitch[\"C\"]}]]").unwrap(),
    "MusicChord"
  );
}
