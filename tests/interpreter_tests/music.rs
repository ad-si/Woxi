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
fn music_pitch_head_is_music_pitch() {
  assert_eq!(interpret("Head[MusicPitch[60]]").unwrap(), "MusicPitch");
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
