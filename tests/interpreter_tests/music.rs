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
  // A MIDI number fixes no letter spelling, so the canonical object keeps
  // just the number (middle C is MIDI 60; A4 is 69).
  assert_eq!(
    interpret("MusicPitch[60]").unwrap(),
    "MusicPitch[<|MIDINumber -> 60|>]"
  );
  assert_eq!(
    interpret("MusicPitch[69]").unwrap(),
    "MusicPitch[<|MIDINumber -> 69|>]"
  );
  assert_eq!(
    interpret("MusicPitch[0]").unwrap(),
    "MusicPitch[<|MIDINumber -> 0|>]"
  );
}

#[test]
fn music_pitch_string_canonicalizes_to_association() {
  // A spelled name with an octave carries Accidental/Octave/Key plus its
  // spelled Name; an octaveless name keeps only Accidental/Key.
  assert_eq!(
    interpret("MusicPitch[\"C4\"]").unwrap(),
    "MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C, Name -> C|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"C#4\"]").unwrap(),
    "MusicPitch[<|Accidental -> 1, Octave -> 4, Key -> C, Name -> C\u{266F}|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"C\"]").unwrap(),
    "MusicPitch[<|Accidental -> 0, Key -> C|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"Bb\"]").unwrap(),
    "MusicPitch[<|Accidental -> -1, Key -> B|>]"
  );
}

#[test]
fn music_pitch_canonicalizes_frequency() {
  // A4 is 440 Hz; 200 Hz is the nearest to G3 / MIDI 55 (per the MusicPitch
  // docs). Frequencies keep their MIDI number unspelled.
  assert_eq!(
    interpret("MusicPitch[Quantity[440, \"Hertz\"]]").unwrap(),
    "MusicPitch[<|MIDINumber -> 69|>]"
  );
  assert_eq!(
    interpret("MusicPitch[Quantity[200, \"Hertz\"]]").unwrap(),
    "MusicPitch[<|MIDINumber -> 55|>]"
  );
}

#[test]
fn music_pitch_canonicalizes_soundnote() {
  // SoundNote numbers pitches relative to middle C (0 -> MIDI 60).
  assert_eq!(
    interpret("MusicPitch[SoundNote[0]]").unwrap(),
    "MusicPitch[<|MIDINumber -> 60|>]"
  );
  assert_eq!(
    interpret("MusicPitch[SoundNote[-5]]").unwrap(),
    "MusicPitch[<|MIDINumber -> 55|>]"
  );
}

#[test]
fn music_pitch_extracts_note_pitch() {
  // The pitch of a note is returned exactly as the note stores it: a spelled
  // note carries the full association, a MIDI-numbered note stays unspelled.
  assert_eq!(
    interpret("MusicPitch[MusicNote[\"G3\"]]").unwrap(),
    "MusicPitch[<|Accidental -> 0, Octave -> 3, Key -> G, Name -> G|>]"
  );
  assert_eq!(
    interpret("MusicPitch[MusicNote[MusicPitch[55]]]").unwrap(),
    "MusicPitch[<|MIDINumber -> 55|>]"
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
fn music_pitch_plain_sum_collects_like_terms() {
  // Plain pitches only combine through intervals (`p - q`, `p + interval`);
  // `p + p` has no musical meaning and collects like ordinary terms.
  assert_eq!(
    interpret("MusicPitch[\"C\"] + MusicPitch[\"C\"]").unwrap(),
    "2*MusicPitch[<|Accidental -> 0, Key -> C|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"C\"] + MusicPitch[\"C\"] + MusicPitch[\"C\"]")
      .unwrap(),
    "3*MusicPitch[<|Accidental -> 0, Key -> C|>]"
  );
  // Distinct plain pitches don't combine either.
  assert_eq!(
    interpret("MusicPitch[\"C\"] + MusicPitch[\"D\"] + MusicPitch[\"C\"]")
      .unwrap(),
    "2*MusicPitch[<|Accidental -> 0, Key -> C|>] + \
     MusicPitch[<|Accidental -> 0, Key -> D|>]"
  );
  // A scaled pitch never takes part in pitch arithmetic, so `2 p - p`
  // collects to the canonical `p` rather than forming an interval.
  assert_eq!(
    interpret("2*MusicPitch[\"C\"] - MusicPitch[\"C\"]").unwrap(),
    "MusicPitch[<|Accidental -> 0, Key -> C|>]"
  );
}

#[test]
fn music_pitch_difference_is_interval() {
  // Subtracting pitches yields the MusicInterval between them.
  assert_eq!(
    interpret("MusicPitch[\"A4\"] - MusicPitch[\"A4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 0, Name -> Unison, CompoundOctaves -> 0|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"E4\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 4, Name -> MajorThird, \
     CompoundOctaves -> 0|>]"
  );
  // A descending interval keeps its signed semitone span but is named from
  // the absolute one.
  assert_eq!(
    interpret("MusicPitch[\"C4\"] - MusicPitch[\"E4\"]").unwrap(),
    "MusicInterval[<|Semitones -> -4, Name -> MajorThird, \
     CompoundOctaves -> 0|>]"
  );
  // The name follows the diatonic spelling, not just the semitone count.
  assert_eq!(
    interpret("MusicPitch[\"C#4\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 1, Name -> DiminishedUnison, \
     CompoundOctaves -> 0|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"Db4\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 1, Name -> MinorSecond, \
     CompoundOctaves -> 0|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"Gb4\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 6, Name -> DiminishedFifth, \
     CompoundOctaves -> 0|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"F#4\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 6, Name -> AugmentedFourth, \
     CompoundOctaves -> 0|>]"
  );
}

#[test]
fn music_pitch_difference_compound_intervals() {
  // A plain octave is not compound…
  assert_eq!(
    interpret("MusicPitch[\"C5\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 12, Name -> Octave, CompoundOctaves -> 0|>]"
  );
  // …but anything wider folds into a Compound name plus an octave count.
  assert_eq!(
    interpret("MusicPitch[\"D5\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 14, Name -> CompoundMajorSecond, \
     CompoundOctaves -> 1|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"C6\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 24, Name -> CompoundOctave, \
     CompoundOctaves -> 1|>]"
  );
  assert_eq!(
    interpret("MusicPitch[\"D6\"] - MusicPitch[\"C4\"]").unwrap(),
    "MusicInterval[<|Semitones -> 26, Name -> CompoundMajorSecond, \
     CompoundOctaves -> 2|>]"
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
  // A pitch plus a plain number is not pitch arithmetic; it stays symbolic
  // (with the pitch in its canonical association form).
  assert_eq!(
    interpret("MusicPitch[\"C\"] + 5").unwrap(),
    "5 + MusicPitch[<|Accidental -> 0, Key -> C|>]"
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
fn named_chord_list_renders_one_staff_per_chord() {
  // A list of named chords renders like a list of notes: each element gets its
  // own staff. Three major triads → three clefs (staves) and nine note heads,
  // with no symbolic `MusicChord[...]` text leaking into the drawing.
  let svg = interpret(
    "ExportString[{MusicChord[\"G\"], MusicChord[\"E\"], MusicChord[\"F\"]}, \
     \"SVG\"]",
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert_eq!(svg.matches("class=\"clef\"").count(), 3);
  assert_eq!(svg.matches("class=\"notehead\"").count(), 9);
  assert!(!svg.contains("MusicChord"), "symbolic chord leaked: {svg}");
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
  // MusicPlot draws a valid music object; in the CLI a Graphics prints as
  // -Graphics-.
  assert_eq!(
    interpret("MusicPlot[MusicNote[\"C4\"]]").unwrap(),
    "-Graphics-"
  );
}

#[test]
fn music_scale_rejects_non_association_second_argument() {
  // MusicScale's second argument must be a property association; a pitch
  // object there emits MusicScale::passc (showing the object's -MusicPitch-
  // summary form) and the scale stays unevaluated. MusicPlot then rejects the
  // invalid scale with MusicPlot::music instead of drawing it.
  clear_state();
  let r = woxi::interpret_with_stdout(
    "Head[MusicPlot[MusicScale[\"Major\", MusicPitch[\"C4\"]]]]",
  )
  .unwrap();
  assert_eq!(r.result, "MusicPlot");
  assert!(
    r.warnings.iter().any(|w| w.contains(
      "MusicScale::passc: -MusicPitch- is not a valid property association."
    )),
    "expected MusicScale::passc, got {:?}",
    r.warnings
  );
  assert!(
    r.warnings.iter().any(|w| w.contains(
      "MusicPlot::music: Expecting a valid music object instead of \
       MusicScale[Major, -MusicPitch-]."
    )),
    "expected MusicPlot::music, got {:?}",
    r.warnings
  );

  // A string second argument is reported in its plain form. (The textual
  // script-mode result comes from `interpret`; `interpret_with_stdout` runs
  // in visual mode, where music objects auto-render.)
  clear_state();
  assert_eq!(
    interpret("MusicScale[\"Major\", \"C4\"]").unwrap(),
    "MusicScale[Major, C4]"
  );
  clear_state();
  let r = woxi::interpret_with_stdout("MusicScale[\"Major\", \"C4\"]").unwrap();
  assert!(r.warnings.iter().any(|w| {
    w.contains("MusicScale::passc: C4 is not a valid property association.")
  }));
}

#[test]
fn bare_music_object_stays_symbolic_in_cli() {
  // Without ExportString/MusicPlot the CLI keeps the canonical (textual) form
  // rather than auto-rendering it as a staff — that only happens in visual
  // hosts. A single-argument note canonicalizes to its association, carrying
  // the pitch's spelled `Name` when it has an octave.
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"]]").unwrap(),
    "MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C, Name -> C|>]|>]"
  );
}

// ─── Symbolic music objects ──────────────────────────────────────────────────

#[test]
fn music_objects_stay_symbolic() {
  // A single-pitch note canonicalizes to its association form; a chord given by
  // an explicit pitch list stays symbolic.
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"]]").unwrap(),
    "MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C, Name -> C|>]|>]"
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
  // to its rhythmic value, and an explicit octave keeps the pitch's `Octave`
  // and spelled `Name`.
  assert_eq!(
    interpret("MusicNote[MusicPitch[\"C4\"], MusicDuration[\"Half\"]]")
      .unwrap(),
    "MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C, Name -> C|>], \
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
  // The sum counts in the default quarter-note beat, so it carries
  // BeatDuration -> 1/4.
  assert_eq!(
    interpret("3 MusicDuration[<|\"Duration\" -> 1/2|>] + MusicDuration[1/4]")
      .unwrap(),
    "MusicDuration[<|Duration -> 7/4, BeatDuration -> 1/4|>]"
  );
  // Subtraction works too: 1 - 1/4 = 3/4.
  assert_eq!(
    interpret("MusicDuration[1] - MusicDuration[1/4]").unwrap(),
    "MusicDuration[<|Duration -> 3/4, BeatDuration -> 1/4|>]"
  );
  // The beat unit is 1/4 regardless of the summands' values.
  assert_eq!(
    interpret("MusicDuration[1/2] + MusicDuration[1/8]").unwrap(),
    "MusicDuration[<|Duration -> 5/8, BeatDuration -> 1/4|>]"
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
fn music_chord_canonicalizes_explicit_pitch_list() {
  // An explicit pitch list becomes the canonical PitchList association, each
  // tone spelled as its full pitch object (octaveless pitches default to
  // register 4, without gaining a Name).
  assert_eq!(
    interpret(
      "MusicChord[{MusicPitch[\"C\"], MusicPitch[\"E\"], MusicPitch[\"G\"]}]"
    )
    .unwrap(),
    "MusicChord[<|PitchList -> {\
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>]}|>]"
  );
}

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
  // G major triad, spelled on the correct staff letters: G4, B4, D5. The
  // chord crosses an octave boundary, so every tone shows its Octave.
  assert_eq!(
    interpret("MusicChord[\"GMajor\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> B|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 5, Key -> D|>]}"
  );
}

#[test]
fn music_chord_pitch_list_omits_octave_within_register() {
  // A named chord whose tones all stay in the root's octave prints them
  // without Octave keys …
  assert_eq!(
    interpret("MusicChord[\"CMajor\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Key -> G|>]}"
  );
  // … while a bare root (no stored quality) always spells the register.
  assert_eq!(
    interpret("MusicChord[\"C\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>]}"
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
fn music_chord_unknown_quality_emits_args_message() {
  // Any chord string that fails to parse — invalid root letter or unknown
  // quality suffix — emits MusicChord::args and stays unevaluated.
  clear_state();
  let r = woxi::interpret_with_stdout("MusicChord[\"Xyz\"]").unwrap();
  assert_eq!(r.result, "MusicChord[Xyz]");
  assert!(r.warnings.iter().any(|w| {
    w.contains("MusicChord::args: MusicChord called with invalid parameters.")
  }));

  clear_state();
  let r = woxi::interpret_with_stdout("MusicChord[\"Cfoo\"]").unwrap();
  assert_eq!(r.result, "MusicChord[Cfoo]");
  assert!(r.warnings.iter().any(|w| {
    w.contains("MusicChord::args: MusicChord called with invalid parameters.")
  }));
}

// ─── MusicChord notation aliases ──────────────────────────────────────────────

/// A bare root stores no quality at all — which is what makes it unequal to
/// the explicitly-major forms — while the spelled-out, space-separated, and
/// letter qualities all store `Name -> Major`.
#[test]
fn music_chord_bare_root_and_spaced_forms() {
  assert_eq!(
    interpret("MusicChord[\"G\"]").unwrap(),
    "MusicChord[<|Root -> MusicPitch[<|Key -> G, Accidental -> 0|>]|>]"
  );
  let major = "MusicChord[<|Name -> Major, \
     Root -> MusicPitch[<|Key -> G, Accidental -> 0|>]|>]";
  assert_eq!(interpret("MusicChord[\"GMajor\"]").unwrap(), major);
  assert_eq!(interpret("MusicChord[\"G Major\"]").unwrap(), major);
  assert_eq!(interpret("MusicChord[\"GM\"]").unwrap(), major);
  assert_eq!(interpret("MusicChord[\"Gmaj\"]").unwrap(), major);
  // The bare root is therefore not identical to the named major chord, but
  // still *sounds* the default major triad through its properties.
  assert_eq!(
    interpret(
      "MusicChord[\"G\"] === MusicChord[\"GMajor\"] === MusicChord[\"G Major\"]"
    )
    .unwrap(),
    "False"
  );
  assert_eq!(interpret("MusicChord[\"G\"][\"Name\"]").unwrap(), "Major");
}

/// A trailing sign with no digits is dropped: `"C-"` and `"C+"` are the bare
/// root C, not minor/augmented chords.
#[test]
fn music_chord_trailing_sign_is_bare_root() {
  let bare =
    "MusicChord[<|Root -> MusicPitch[<|Key -> C, Accidental -> 0|>]|>]";
  assert_eq!(interpret("MusicChord[\"C-\"]").unwrap(), bare);
  assert_eq!(interpret("MusicChord[\"C+\"]").unwrap(), bare);
}

/// The short symbols map to the same canonical (spaced) names as the long
/// ones.
#[test]
fn music_chord_short_symbols_resolve() {
  let cases = [
    ("Cm", "Minor"),
    ("Cmin", "Minor"),
    ("Caug", "Augmented"),
    ("Cdim", "Diminished"),
    ("Co", "Diminished"),
    ("Cdom7", "Dominant Seventh"),
    ("Cmaj7", "Major Seventh"),
    ("CM7", "Major Seventh"),
    ("Cm7", "Minor Seventh"),
    ("Cdim7", "Diminished Seventh"),
    ("Cm7b5", "Half Diminished Seventh"),
    ("Caug7", "Augmented Seventh"),
    ("Cm6", "Minor Sixth"),
    ("CDominantNinth", "Dominant Ninth"),
    ("CM9", "Major Ninth"),
    ("Cm9", "Minor Ninth"),
    ("Csus2", "Suspended Second"),
    ("Csus4", "Suspended Fourth"),
    ("Csus", "Suspended Fourth"),
    ("CMinorMajorSeventh", "Minor Major Seventh"),
  ];
  for (input, name) in cases {
    assert_eq!(
      interpret(&format!("MusicChord[\"{input}\"][\"PitchList\"]")).is_ok(),
      true,
      "{input} should have a pitch list"
    );
    assert_eq!(
      interpret(&format!("MusicChord[\"{input}\"][\"Name\"]")).unwrap(),
      name,
      "quality of {input}"
    );
  }
}

/// A root followed only by a (possibly `+`-signed) digit is the note's
/// *octave*, not a chord quality: `"C7"` is the note C in octave 7 and its
/// pitch list is a major triad in that register. Octaves outside 0–9 (and a
/// negative sign) are invalid parameters.
#[test]
fn music_chord_trailing_digit_is_octave() {
  assert_eq!(
    interpret("MusicChord[\"C7\"]").unwrap(),
    "MusicChord[<|Root -> \
     MusicPitch[<|Key -> C, Accidental -> 0, Octave -> 7, Name -> C7|>]|>]"
  );
  assert_eq!(
    interpret("MusicChord[\"C7\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 7, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 7, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 7, Key -> G|>]}"
  );
  // The accidental is part of the root's spelled name (C♯7).
  assert_eq!(
    interpret("MusicChord[\"C#7\"]").unwrap(),
    "MusicChord[<|Root -> \
     MusicPitch[<|Key -> C, Accidental -> 1, Octave -> 7, Name -> C\u{266F}7|>]|>]"
  );
  // An explicit plus sign reads the same octave.
  assert_eq!(
    interpret("MusicChord[\"C+7\"]").unwrap(),
    "MusicChord[<|Root -> \
     MusicPitch[<|Key -> C, Accidental -> 0, Octave -> 7, Name -> C7|>]|>]"
  );
  // Out-of-range octaves and negative octaves are invalid parameters.
  for bad in ["C10", "C-7", "C13"] {
    clear_state();
    let r =
      woxi::interpret_with_stdout(&format!("MusicChord[\"{bad}\"]")).unwrap();
    assert_eq!(r.result, format!("MusicChord[{bad}]"));
    assert!(
      r.warnings.iter().any(|w| w.contains(
        "MusicChord::args: MusicChord called with invalid parameters."
      )),
      "expected MusicChord::args for {bad}"
    );
  }
}

/// Case matters where it must: `m` is minor, `M` is major.
#[test]
fn music_chord_case_distinguishes_major_from_minor() {
  assert_eq!(interpret("MusicChord[\"CM\"][\"Name\"]").unwrap(), "Major");
  assert_eq!(interpret("MusicChord[\"Cm\"][\"Name\"]").unwrap(), "Minor");
  assert_eq!(
    interpret("MusicChord[\"CM7\"][\"Name\"]").unwrap(),
    "Major Seventh"
  );
  assert_eq!(
    interpret("MusicChord[\"Cm7\"][\"Name\"]").unwrap(),
    "Minor Seventh"
  );
}

/// Sharp/flat roots parse (both ASCII and Unicode accidentals), including a
/// double flat, and combine with a quality suffix.
#[test]
fn music_chord_accidental_roots() {
  assert_eq!(
    interpret("MusicChord[\"Bbm7\"]").unwrap(),
    "MusicChord[<|Name -> Minor Seventh, \
     Root -> MusicPitch[<|Key -> B, Accidental -> -1|>]|>]"
  );
  assert_eq!(
    interpret("MusicChord[\"F#m\"]").unwrap(),
    "MusicChord[<|Name -> Minor, \
     Root -> MusicPitch[<|Key -> F, Accidental -> 1|>]|>]"
  );
  assert_eq!(
    interpret("MusicChord[\"F\u{266F}Major\"]").unwrap(),
    "MusicChord[<|Name -> Major, \
     Root -> MusicPitch[<|Key -> F, Accidental -> 1|>]|>]"
  );
  assert_eq!(
    interpret("MusicChord[\"Cbb\"][\"Root\"]").unwrap(),
    "MusicPitch[<|Key -> C, Accidental -> -2|>]"
  );
}

/// A dominant seventh spells a diminished-quality top: Cdom7 → C E G Bb (all
/// within the root's octave, so no Octave keys).
#[test]
fn music_chord_dominant_seventh_spelling() {
  assert_eq!(
    interpret("MusicChord[\"Cdom7\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Key -> G|>], \
     MusicPitch[<|Accidental -> -1, Key -> B|>]}"
  );
}

/// A suspended-fourth triad replaces the third with the fourth: C F G.
#[test]
fn music_chord_suspended_fourth_spelling() {
  assert_eq!(
    interpret("MusicChord[\"Csus4\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Key -> F|>], \
     MusicPitch[<|Accidental -> 0, Key -> G|>]}"
  );
}

/// A dominant ninth stacks the ninth an octave up: C E G Bb D5. Crossing the
/// octave boundary makes every tone spell its register.
#[test]
fn music_chord_dominant_ninth_spelling() {
  assert_eq!(
    interpret("MusicChord[\"CDominantNinth\"][\"PitchList\"]").unwrap(),
    "{MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> C|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> E|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 4, Key -> G|>], \
     MusicPitch[<|Accidental -> -1, Octave -> 4, Key -> B|>], \
     MusicPitch[<|Accidental -> 0, Octave -> 5, Key -> D|>]}"
  );
}

// ─── MusicTimeSignature / MusicRest ──────────────────────────────────────────

/// `MusicTimeSignature[n, d]` canonicalizes to its Numerator/Denominator
/// association.
#[test]
fn music_time_signature_canonicalizes() {
  assert_eq!(
    interpret("MusicTimeSignature[3, 4]").unwrap(),
    "MusicTimeSignature[<|Numerator -> 3, Denominator -> 4|>]"
  );
}

/// `MusicNote[pitch]` with no duration keeps only its `Pitch`.
#[test]
fn music_note_single_argument_canonicalizes() {
  assert_eq!(
    interpret("MusicNote[\"E\"]").unwrap(),
    "MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>]|>]"
  );
}

/// `MusicRest[duration]` wraps its duration; a bare `MusicRest[]` is an empty
/// association.
#[test]
fn music_rest_canonicalizes() {
  assert_eq!(
    interpret("MusicRest[1/2]").unwrap(),
    "MusicRest[<|Duration -> MusicDuration[<|Duration -> 1/2|>]|>]"
  );
  assert_eq!(interpret("MusicRest[]").unwrap(), "MusicRest[<||>]");
}

// ─── MusicMeasure ────────────────────────────────────────────────────────────

/// A measure whose beats fit packs into `<|NoteList, TimeSignature|>`, each
/// event annotated with its `BeatDuration`/`Beats`; the final default note is
/// stretched to fill the measure exactly (E, C fill one beat, D fills two).
#[test]
fn music_measure_fills_with_trailing_note() {
  assert_eq!(
    interpret(
      "MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"]}, \
       MusicTimeSignature[4, 4]]"
    )
    .unwrap(),
    "MusicMeasure[<|NoteList -> {\
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> C|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> D|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 2|>]|>]}, \
     TimeSignature -> MusicTimeSignature[<|Numerator -> 4, Denominator -> 4|>]|>]"
  );
}

/// An explicit trailing duration is rigid, so a padding rest fills the
/// remainder rather than the note stretching.
#[test]
fn music_measure_pads_rigid_tail_with_rest() {
  assert_eq!(
    interpret(
      "MusicMeasure[{MusicNote[\"E\"], MusicNote[\"D\", 1/4]}, \
       MusicTimeSignature[4, 4]]"
    )
    .unwrap(),
    "MusicMeasure[<|NoteList -> {\
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> D|>], \
     Duration -> MusicDuration[<|Duration -> 1/4, BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicRest[<|Duration -> MusicDuration[<|Duration -> 1/2, BeatDuration -> 1/4, Beats -> 2|>]|>]}, \
     TimeSignature -> MusicTimeSignature[<|Numerator -> 4, Denominator -> 4|>]|>]"
  );
}

/// A compound meter (6/8) beats in dotted quarters: two beats per measure, so
/// an explicit half note plus a default note also packs cleanly.
#[test]
fn music_measure_compound_meter_beat_unit() {
  assert_eq!(
    interpret("MusicMeasure[{MusicNote[\"E\"]}, MusicTimeSignature[9, 8]]")
      .unwrap(),
    "MusicMeasure[<|NoteList -> {\
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>], \
     Duration -> MusicDuration[<|BeatDuration -> 3/8, Beats -> 3|>]|>]}, \
     TimeSignature -> MusicTimeSignature[<|Numerator -> 9, Denominator -> 8|>]|>]"
  );
}

/// A measure resolved to its `<|NoteList, TimeSignature|>` association still
/// draws all of its notes on the staff, not just the meter. Regression: the
/// staff renderer previously only understood the unresolved list form, so a
/// `MusicMeasure[{…}, MusicTimeSignature[…]]` rendered its 3/4 glyph with no
/// note heads.
#[test]
fn music_measure_with_time_signature_renders_notes() {
  let svg = interpret(
    "ExportString[MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], \
     MusicNote[\"D\"]}, MusicTimeSignature[3, 4]], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
  assert_eq!(svg.matches("class=\"timesig\"").count(), 2);
}

/// A voice of mixed measures (a bare list measure in 4/4, then a
/// time-signatured measure in 3/4, then 4/4 again) draws every note and
/// reprints the meter on each change.
#[test]
fn music_voice_renders_mixed_measures() {
  let svg = interpret(
    "ExportString[MusicVoice[{\
     MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"], \
     MusicNote[\"G3\"]}], \
     MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"]}, \
     MusicTimeSignature[3, 4]], \
     MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"], \
     MusicNote[\"G3\"]}]}], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  // 4 + 3 + 4 note heads across the three measures.
  assert_eq!(svg.matches("class=\"notehead\"").count(), 11);
  // Meter reprinted on 4/4 → 3/4 → 4/4: three signatures, six digit glyphs.
  assert_eq!(svg.matches("class=\"timesig\"").count(), 6);
}

/// An overfull measure warns with `MusicMeasure::measdur` and returns its
/// non-associated form (a half note is two quarter beats, so E + C + D is four
/// beats in a three-beat measure). Regression for the reported example. The
/// textual (script-mode) result comes from `interpret`; `interpret_with_stdout`
/// runs in visual mode, where the measure auto-renders as a staff graphic.
#[test]
fn music_measure_overfull_warns() {
  clear_state();
  assert_eq!(
    interpret(
      "MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\", 1/2], \
       MusicNote[\"D\"]}, MusicTimeSignature[3, 4]]"
    )
    .unwrap(),
    "MusicMeasure[{\
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> C|>], \
     Duration -> MusicDuration[<|Duration -> 1/2|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> D|>]|>]}, \
     MusicTimeSignature[<|Numerator -> 3, Denominator -> 4|>]]"
  );

  clear_state();
  let result = interpret_with_stdout(
    "MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\", 1/2], MusicNote[\"D\"]}, \
     MusicTimeSignature[3, 4]]",
  )
  .unwrap();
  assert!(result.warnings.iter().any(|w| w.contains(
    "MusicMeasure::measdur: The total duration of beats 4 exceeds the \
     allowed number of beats per measure 3."
  )));
}

/// The overfull warning uses the compound-meter beat count: three default
/// notes are three beats in a two-beat 6/8 measure.
#[test]
fn music_measure_overfull_compound_meter() {
  clear_state();
  let result = interpret_with_stdout(
    "MusicMeasure[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"]}, \
     MusicTimeSignature[6, 8]]",
  )
  .unwrap();
  assert!(result.warnings.iter().any(|w| w.contains(
    "MusicMeasure::measdur: The total duration of beats 3 exceeds the \
     allowed number of beats per measure 2."
  )));
}

// ─── Voice / score resolution ────────────────────────────────────────────────

/// A bare `MusicMeasure[{…}]` (no meter) resolves against common time, tagging
/// the generated default signature with `BeatLength -> 1`; a lone note fills
/// the whole four-beat bar.
#[test]
fn music_measure_defaults_to_common_time() {
  assert_eq!(
    interpret("MusicMeasure[{MusicNote[\"C\"]}]").unwrap(),
    "MusicMeasure[<|NoteList -> {MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Key -> C|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 4|>]|>]}, \
     TimeSignature -> \
     MusicTimeSignature[<|Numerator -> 4, Denominator -> 4, BeatLength -> 1|>]|>]"
  );
}

/// A `MusicVoice` of measures resolves to `<|MeasureList, TimeSignature|>`, its
/// signature taken from the first measure.
#[test]
fn music_voice_resolves_to_measure_list() {
  assert_eq!(
    interpret(
      "MusicVoice[{MusicMeasure[{MusicNote[\"C\"], MusicNote[\"E\"], \
       MusicNote[\"D\"]}, MusicTimeSignature[3, 4]]}]"
    )
    .unwrap(),
    "MusicVoice[<|MeasureList -> {MusicMeasure[<|NoteList -> {\
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> C|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> E|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> MusicPitch[<|Accidental -> 0, Key -> D|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>]}, \
     TimeSignature -> MusicTimeSignature[<|Numerator -> 3, Denominator -> 4|>]|>]}, \
     TimeSignature -> MusicTimeSignature[<|Numerator -> 3, Denominator -> 4|>]|>]"
  );
}

/// A `MusicScore` resolves to `<|VoiceList, TimeSignature|>`, keeping each
/// resolved voice.
#[test]
fn music_score_resolves_to_voice_list() {
  let out = interpret("MusicScore[{MusicVoice[{MusicNote[\"C\"]}]}]").unwrap();
  assert!(
    out.starts_with("MusicScore[<|VoiceList -> {MusicVoice[<|MeasureList")
  );
  assert!(out.ends_with(
    "TimeSignature -> \
     MusicTimeSignature[<|Numerator -> 4, Denominator -> 4, BeatLength -> 1|>]|>]"
  ));
}

/// Empty containers resolve to their key-only association forms.
#[test]
fn music_empty_containers_resolve() {
  assert_eq!(
    interpret("MusicVoice[]").unwrap(),
    "MusicVoice[<|MeasureList -> {}|>]"
  );
  assert_eq!(
    interpret("MusicScore[]").unwrap(),
    "MusicScore[<|VoiceList -> {}|>]"
  );
}

// ─── Transposition by MusicInterval ──────────────────────────────────────────

/// Adding a `MusicInterval` to a `MusicNote` transposes its pitch. A
/// bare-semitone interval spells the result straight from the MIDI number
/// (C + 5 semitones = F); a named interval keeps the diatonic step
/// (C + a minor third = Eb, not D#).
#[test]
fn music_note_plus_interval_transposes() {
  assert_eq!(
    interpret("MusicNote[\"C\"] + MusicInterval[5]").unwrap(),
    "MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Key -> F, MIDINumber -> 65|>]|>]"
  );
  assert_eq!(
    interpret("MusicNote[\"C\"] + MusicInterval[\"MinorThird\"]").unwrap(),
    "MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> -1, Key -> E, MIDINumber -> 63|>]|>]"
  );
}

/// Adding an interval to a whole `MusicVoice` transposes every pitch it
/// contains (the voice resolves to its measure/note association, the pitches
/// shifted up a fourth: C→F, E→A).
#[test]
fn music_voice_plus_interval_transposes_every_pitch() {
  assert_eq!(
    interpret(
      "MusicVoice[{MusicNote[\"C\"], MusicNote[\"E\"]}] + MusicInterval[5]"
    )
    .unwrap(),
    "MusicVoice[<|MeasureList -> {MusicMeasure[<|NoteList -> {\
     MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Key -> F, MIDINumber -> 65|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 1|>]|>], \
     MusicNote[<|Pitch -> \
     MusicPitch[<|Accidental -> 0, Key -> A, MIDINumber -> 69|>], \
     Duration -> MusicDuration[<|BeatDuration -> 1/4, Beats -> 3|>]|>]}, \
     TimeSignature -> \
     MusicTimeSignature[<|Numerator -> 4, Denominator -> 4, BeatLength -> 1|>]\
     |>]}, TimeSignature -> \
     MusicTimeSignature[<|Numerator -> 4, Denominator -> 4, BeatLength -> 1|>]|>]"
  );
}

// ─── MusicScore rendering ────────────────────────────────────────────────────

/// A `MusicScore` overlays its voices on one shared staff: the voices sound
/// simultaneously, so a note from each at the same position stacks into a
/// chord. Here a voice and its transposition up a fourth print as three
/// two-note chords on a single staff.
#[test]
fn music_score_overlays_voices_on_one_staff() {
  let svg = interpret(
    "voice = MusicVoice[{MusicNote[\"E\"], MusicNote[\"C\"], MusicNote[\"D\"]}]; \
     ExportString[MusicScore[{voice, voice + MusicInterval[5]}], \"SVG\"]",
  )
  .unwrap();
  assert!(svg.starts_with("<svg"), "expected an SVG, got: {svg}");
  // One shared staff (one clef), with both voices' heads: 3 positions × 2 = 6.
  assert_eq!(svg.matches("class=\"clef\"").count(), 1);
  assert_eq!(svg.matches("class=\"notehead\"").count(), 6);
}
