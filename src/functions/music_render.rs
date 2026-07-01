//! SVG rendering of the Wolfram Language 15.0 ComputationalMusic objects on a
//! musical staff, the way Mathematica displays them.
//!
//! The music *objects* stay canonical symbolic expressions everywhere (so the
//! CLI still prints `MusicNote[MusicPitch[C4], …]`); only the visual hosts —
//! the Woxi Playground and Woxi Studio — turn them into notation. A single
//! entry point, [`music_to_svg`], produces a self-contained SVG that renders
//! identically in a browser and in Studio's `resvg` rasterizer.
//!
//! The proper musical symbols — treble clef, note heads, accidentals, rests
//! and flags — are the glyphs of MuseScore's [Leland](https://github.com/MuseScoreFonts/Leland)
//! SMuFL font, extracted as SVG `<path>`s by [`crate::functions::music_font`]
//! so no font need be embedded or shaped by the host. The staff lines, stems,
//! ledger lines and barlines are engraving rules, drawn directly as `<line>`s.

use crate::functions::graphics::theme;
use crate::functions::music_ast::{
  midi_to_pitch_name, pitch_name_to_midi, resolve_pitch_name,
};
use crate::functions::music_font::{self, glyph};
use crate::syntax::Expr;

// ── Staff geometry (internal user-space coordinates) ─────────────────────────

/// Vertical distance between two adjacent staff lines.
const GAP: f64 = 12.0;
/// Half a `GAP`: the vertical step between adjacent diatonic degrees (a line
/// and its neighbouring space).
const STEP: f64 = GAP / 2.0;
/// `y` of the bottom staff line (E4) in internal coordinates.
const BOTTOM_LINE_Y: f64 = 100.0;
/// Diatonic number of the bottom staff line, E4 (see [`diatonic_number`]).
const BOTTOM_LINE_DN: i32 = 30;
/// Horizontal advance between successive notes/rests.
const ADVANCE: f64 = 30.0;
/// `x` of the left edge of the staff.
const STAFF_X0: f64 = 6.0;
/// Stem length, in whole gaps.
const STEM_LEN: f64 = 3.0 * GAP;

/// `y` of a diatonic degree `dn` on the staff.
fn dn_y(dn: i32) -> f64 {
  BOTTOM_LINE_Y - (dn - BOTTOM_LINE_DN) as f64 * STEP
}

// ── Musical model ────────────────────────────────────────────────────────────

/// A rhythmic value, controlling note-head fill, stem and flags.
#[derive(Clone, Copy, PartialEq)]
enum Dur {
  Whole,
  Half,
  Quarter,
  Eighth,
  Sixteenth,
}

impl Dur {
  /// Whether the note carries a stem (everything but a whole note).
  fn has_stem(self) -> bool {
    self != Dur::Whole
  }
}

/// One note head: its diatonic staff position and accidental
/// (`-1` flat, `0` natural, `+1` sharp).
#[derive(Clone, Copy)]
struct Head {
  dn: i32,
  accidental: i32,
}

/// A drawable element laid out left to right on the staff.
enum Glyph {
  /// A note or (with several heads) a chord.
  Note {
    heads: Vec<Head>,
    dur: Dur,
  },
  Rest {
    dur: Dur,
  },
  /// A time signature drawn as stacked numerator/denominator digits.
  TimeSig {
    num: u32,
    den: u32,
  },
  Barline,
}

/// The `(numerator, denominator)` of a `MusicTimeSignature`, given either as
/// `MusicTimeSignature[n, d]` or the canonical
/// `MusicTimeSignature[<|"Numerator" -> n, "Denominator" -> d|>]`. Returns
/// `None` for anything that is not a time signature.
fn time_signature_of(expr: &Expr) -> Option<(u32, u32)> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "MusicTimeSignature" {
    return None;
  }
  match args.as_ref() {
    [Expr::Integer(n), Expr::Integer(d)] if *n > 0 && *d > 0 => {
      Some((*n as u32, *d as u32))
    }
    [Expr::Association(pairs)] => {
      let get = |key: &str| {
        pairs.iter().find_map(|(k, v)| match (k, v) {
          (Expr::String(s), Expr::Integer(n)) if s == key && *n > 0 => {
            Some(*n as u32)
          }
          _ => None,
        })
      };
      Some((get("Numerator")?, get("Denominator")?))
    }
    _ => None,
  }
}

/// The time signature carried by a container's arguments — an explicit
/// `MusicTimeSignature[…]` argument, or the `"TimeSignature"` entry of a
/// canonical `<|…|>` association. Returns `None` when none is specified.
fn container_time_signature(args: &[Expr]) -> Option<(u32, u32)> {
  args.iter().find_map(|a| match a {
    Expr::Association(pairs) => pairs.iter().find_map(|(k, v)| match k {
      Expr::String(s) if s == "TimeSignature" => time_signature_of(v),
      _ => None,
    }),
    other => time_signature_of(other),
  })
}

// ── Parsing music objects into glyphs ────────────────────────────────────────

/// Diatonic number of a scientific-pitch name: `octave * 7 + letter`, with
/// `C = 0 … B = 6`. E4 is `30` (the bottom treble-staff line), middle C (C4)
/// is `28` (one ledger line below). The accidental does not change the
/// diatonic position — it is drawn separately.
fn diatonic_number(name: &str) -> Option<(i32, i32)> {
  let bytes = name.as_bytes();
  let letter = *bytes.first()?;
  let letter_index = match letter.to_ascii_uppercase() {
    b'C' => 0,
    b'D' => 1,
    b'E' => 2,
    b'F' => 3,
    b'G' => 4,
    b'A' => 5,
    b'B' => 6,
    _ => return None,
  };
  let mut idx = 1;
  let mut accidental = 0;
  while let Some(&c) = bytes.get(idx) {
    match c {
      b'#' => accidental += 1,
      b'b' => accidental -= 1,
      _ => break,
    }
    idx += 1;
  }
  // An octaveless name such as "C" or "Eb" defaults to octave 4, the Wolfram
  // Language default register for `MusicPitch`.
  let octave: i32 = match name.get(idx..) {
    Some("") | None => 4,
    Some(rest) => rest.parse().ok()?,
  };
  Some((octave * 7 + letter_index, accidental))
}

/// Turn any pitch specification (a `MusicPitch`, `MusicNote`, name string or
/// MIDI integer) into a staff [`Head`].
fn pitch_head(spec: &Expr) -> Option<Head> {
  let name = resolve_pitch_name(spec)?;
  let (dn, accidental) = diatonic_number(&name)?;
  Some(Head { dn, accidental })
}

/// Parse a duration specification — `MusicDuration["Quarter"]`, the bare name
/// `"Half"`, the canonical `MusicDuration[<|"Duration" -> 1/2|>]` association,
/// or a bare rhythmic value — defaulting to a quarter note.
fn parse_duration(spec: &Expr) -> Dur {
  // The canonical association form (or a bare number) carries a rhythmic value.
  if let Some(value) = duration_value_f64(spec) {
    return if value >= 1.0 {
      Dur::Whole
    } else if value >= 0.5 {
      Dur::Half
    } else if value >= 0.25 {
      Dur::Quarter
    } else if value >= 0.125 {
      Dur::Eighth
    } else {
      Dur::Sixteenth
    };
  }
  let name = match spec {
    Expr::String(s) => Some(s.as_str()),
    Expr::FunctionCall { name, args }
      if name == "MusicDuration" && !args.is_empty() =>
    {
      match &args[0] {
        Expr::String(s) => Some(s.as_str()),
        _ => None,
      }
    }
    _ => None,
  };
  match name {
    Some("Whole") => Dur::Whole,
    Some("Half") => Dur::Half,
    Some("Eighth") => Dur::Eighth,
    Some("Sixteenth") => Dur::Sixteenth,
    _ => Dur::Quarter,
  }
}

/// The numeric rhythmic value of a duration given as a bare number or a
/// canonical `MusicDuration[<|"Duration" -> value|>]` association.
fn duration_value_f64(spec: &Expr) -> Option<f64> {
  fn value_of(expr: &Expr) -> Option<f64> {
    match expr {
      Expr::Integer(n) => Some(*n as f64),
      Expr::Real(f) => Some(*f),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => {
            Some(*n as f64 / *d as f64)
          }
          _ => None,
        }
      }
      _ => None,
    }
  }
  match spec {
    Expr::FunctionCall { name, args }
      if name == "MusicDuration" && args.len() == 1 =>
    {
      match &args[0] {
        Expr::Association(pairs) => {
          let get = |key: &str| {
            pairs.iter().find_map(|(k, v)| match k {
              Expr::String(s) if s == key => value_of(v),
              _ => None,
            })
          };
          // A plain `<|Duration -> v|>` carries the value directly; the
          // beat-annotated form a measure produces (`<|BeatDuration -> bd,
          // Beats -> n[, Duration -> v]|>`) may omit `Duration` for a default
          // or stretched note, in which case the sounding value is bd·n.
          get("Duration").or_else(|| Some(get("BeatDuration")? * get("Beats")?))
        }
        other => value_of(other),
      }
    }
    _ => None,
  }
}

/// Semitone offsets (from the root) of the scale degrees Woxi knows how to
/// spell out, or `None` for an unrecognized scale name.
fn scale_pattern(name: &str) -> Option<&'static [i32]> {
  Some(match name {
    "Major" | "Ionian" => &[0, 2, 4, 5, 7, 9, 11, 12],
    "Minor" | "NaturalMinor" | "Aeolian" => &[0, 2, 3, 5, 7, 8, 10, 12],
    "HarmonicMinor" => &[0, 2, 3, 5, 7, 8, 11, 12],
    "MelodicMinor" => &[0, 2, 3, 5, 7, 9, 11, 12],
    "Dorian" => &[0, 2, 3, 5, 7, 9, 10, 12],
    "Phrygian" => &[0, 1, 3, 5, 7, 8, 10, 12],
    "Lydian" => &[0, 2, 4, 6, 7, 9, 11, 12],
    "Mixolydian" => &[0, 2, 4, 5, 7, 9, 10, 12],
    "WholeTone" => &[0, 2, 4, 6, 8, 10, 12],
    "Chromatic" => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    _ => return None,
  })
}

/// Expand a `MusicScale[…]` argument list into its constituent pitch heads.
fn scale_heads(args: &[Expr]) -> Vec<Head> {
  // Locate the scale name and (optionally) the root pitch among the arguments.
  let mut pattern: &[i32] = &[0, 2, 4, 5, 7, 9, 11, 12];
  let mut root: Option<i128> = None;
  for arg in args {
    if let Expr::String(s) = arg
      && let Some(p) = scale_pattern(s)
    {
      pattern = p;
      continue;
    }
    if let Some(name) = resolve_pitch_name(arg)
      && let Some(midi) = pitch_name_to_midi(&name)
    {
      root = Some(midi);
    }
  }
  let root = root.unwrap_or(60); // default to middle C
  pattern
    .iter()
    .filter_map(|off| {
      let (dn, accidental) =
        diatonic_number(&midi_to_pitch_name(root + *off as i128))?;
      Some(Head { dn, accidental })
    })
    .collect()
}

/// Emit a time-signature glyph if it differs from the currently-shown one,
/// updating `ts`. This is what makes a `MusicMeasure`/`MusicVoice`/`MusicScore`
/// print its meter at the start, and re-print it whenever the meter changes.
fn show_time_signature(
  out: &mut Vec<Glyph>,
  ts: &mut Option<(u32, u32)>,
  value: (u32, u32),
) {
  if *ts != Some(value) {
    out.push(Glyph::TimeSig {
      num: value.0,
      den: value.1,
    });
    *ts = Some(value);
  }
}

/// Collect the drawable glyphs of a music object in reading order. Returns
/// `true` when `expr` was a container whose contents are conventionally closed
/// by a final barline. `ts` carries the time signature currently drawn on the
/// staff so meters are shown once and re-shown only on a change.
fn collect(
  expr: &Expr,
  out: &mut Vec<Glyph>,
  ts: &mut Option<(u32, u32)>,
) -> bool {
  match expr {
    Expr::List(items) => {
      for it in items.iter() {
        collect(it, out, ts);
      }
      false
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      // A bare `MusicPitch` is only a pitch — it carries no rhythmic value, so
      // it is not staff notation and is left symbolic (a pitch inside a
      // `MusicNote`/`MusicChord` is handled by those arms below). `MusicChord`,
      // by contrast, is a musical event and does render.
      "MusicPitch" => false,
      // Canonical association form: MusicNote[<|"Pitch" -> …, "Duration" -> …|>].
      "MusicNote" if matches!(args.first(), Some(Expr::Association(_))) => {
        if let Some(Expr::Association(pairs)) = args.first() {
          let get = |key: &str| {
            pairs.iter().find_map(|(k, v)| match k {
              Expr::String(s) if s == key => Some(v),
              _ => None,
            })
          };
          if let Some(h) = get("Pitch").and_then(pitch_head) {
            let dur =
              get("Duration").map(parse_duration).unwrap_or(Dur::Quarter);
            out.push(Glyph::Note {
              heads: vec![h],
              dur,
            });
          }
        }
        false
      }
      "MusicNote" if !args.is_empty() => {
        if let Some(h) = pitch_head(&args[0]) {
          let dur = args.get(1).map(parse_duration).unwrap_or(Dur::Quarter);
          out.push(Glyph::Note {
            heads: vec![h],
            dur,
          });
        }
        false
      }
      "MusicRest" => {
        let dur = args.first().map(parse_duration).unwrap_or(Dur::Quarter);
        out.push(Glyph::Rest { dur });
        false
      }
      "MusicChord" if !args.is_empty() => {
        let mut heads = Vec::new();
        match &args[0] {
          // Explicit pitch list: `MusicChord[{"C4", "E4", "G4"}]`.
          Expr::List(items) => {
            for it in items.iter() {
              if let Some(h) = pitch_head(it) {
                heads.push(h);
              }
            }
          }
          // Canonical chord association. Either an explicit
          // `<|"PitchList" -> {MusicPitch[…], …}|>` (produced by a pitch-list
          // chord or its transposition) or a named chord `MusicChord["GMajor"]`
          // → `<|"Name" -> …, "Root" -> MusicPitch[…]|>`, whose stacked-thirds
          // tones are spelled out. Either way the tones render as one chord.
          Expr::Association(pairs) => {
            let tones = pairs
              .iter()
              .find_map(|(k, v)| match (k, v) {
                (Expr::String(s), Expr::List(items)) if s == "PitchList" => {
                  Some(items.to_vec())
                }
                _ => None,
              })
              .or_else(|| crate::functions::music_ast::chord_tones(pairs));
            if let Some(tones) = tones {
              for t in &tones {
                if let Some(h) = pitch_head(t) {
                  heads.push(h);
                }
              }
            }
          }
          _ => {}
        }
        if !heads.is_empty() {
          heads.sort_by_key(|h| h.dn);
          let dur = args.get(1).map(parse_duration).unwrap_or(Dur::Quarter);
          out.push(Glyph::Note { heads, dur });
        }
        false
      }
      "MusicScale" => {
        for h in scale_heads(args) {
          out.push(Glyph::Note {
            heads: vec![h],
            dur: Dur::Quarter,
          });
        }
        true
      }
      // A time signature carried in the note stream (or as a container
      // argument) prints on the staff when it first appears or changes.
      "MusicTimeSignature" => {
        if let Some(value) = time_signature_of(expr) {
          show_time_signature(out, ts, value);
        }
        false
      }
      "MusicMeasure" => {
        // Each measure carries its own meter — its explicit `MusicTimeSignature`
        // or the 4/4 default — so a run of measures reprints the meter whenever
        // it differs from the previous one (e.g. 4/4, 3/4, back to 4/4), rather
        // than inheriting and sticking at the last change.
        let value = container_time_signature(args).unwrap_or((4, 4));
        show_time_signature(out, ts, value);
        // A resolved measure is the association form
        // `MusicMeasure[<|NoteList -> {…}, TimeSignature -> …|>]`; its notes live
        // under `"NoteList"`. The unresolved forms — a plain event list, or a
        // list plus a `MusicTimeSignature` — carry their events as arguments.
        if let Some(Expr::Association(pairs)) = args.first() {
          if let Some(Expr::List(items)) = pairs.iter().find_map(|(k, v)| {
            matches!(k, Expr::String(s) if s == "NoteList").then_some(v)
          }) {
            for it in items.iter() {
              collect(it, out, ts);
            }
          }
        } else {
          for arg in args {
            collect(arg, out, ts);
          }
        }
        out.push(Glyph::Barline);
        true
      }
      "MusicVoice" | "MusicScore" => {
        // A voice/score prints its meter at the start (4/4 unless specified).
        let value = container_time_signature(args).or(*ts).unwrap_or((4, 4));
        show_time_signature(out, ts, value);
        for arg in args {
          collect(arg, out, ts);
        }
        true
      }
      _ => false,
    },
    // A bare pitch-name string inside a measure/voice — as in
    // `MusicMeasure[{"C", "G", "A", "C"}]` — is a quarter note on that pitch.
    Expr::String(_) => {
      if let Some(h) = pitch_head(expr) {
        out.push(Glyph::Note {
          heads: vec![h],
          dur: Dur::Quarter,
        });
      }
      false
    }
    _ => false,
  }
}

// ── SVG drawing ──────────────────────────────────────────────────────────────

/// Accumulates SVG fragments while tracking the drawn bounding box so the final
/// `viewBox` can be fitted with padding.
struct Canvas {
  out: String,
  stroke: String,
  /// Font-unit → user-unit scale for placing Leland glyphs.
  scale: f64,
  minx: f64,
  miny: f64,
  maxx: f64,
  maxy: f64,
}

impl Canvas {
  fn new(stroke: &str, scale: f64) -> Self {
    Canvas {
      out: String::new(),
      stroke: stroke.to_string(),
      scale,
      minx: f64::INFINITY,
      miny: f64::INFINITY,
      maxx: f64::NEG_INFINITY,
      maxy: f64::NEG_INFINITY,
    }
  }

  fn bound(&mut self, x: f64, y: f64) {
    self.minx = self.minx.min(x);
    self.miny = self.miny.min(y);
    self.maxx = self.maxx.max(x);
    self.maxy = self.maxy.max(y);
  }

  fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, w: f64) {
    self.out.push_str(&format!(
      "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" \
       stroke=\"{}\" stroke-width=\"{w}\" stroke-linecap=\"round\"/>",
      self.stroke
    ));
    self.bound(x1.min(x2) - w, y1.min(y2) - w);
    self.bound(x1.max(x2) + w, y1.max(y2) + w);
  }

  /// Draw the Leland glyph `ch` so its origin `(0, 0)` lands at `(ox, oy)`,
  /// tagging the emitted `<path>` with `class` so hosts (and tests) can
  /// recognise it.
  fn glyph_at(&mut self, ch: char, ox: f64, oy: f64, class: &str) {
    let Some(d) = music_font::glyph_path_d(ch, self.scale, ox, oy) else {
      return;
    };
    self.out.push_str(&format!(
      "<path class=\"{class}\" d=\"{d}\" fill=\"{}\"/>",
      self.stroke
    ));
    if let Some(bb) = music_font::glyph_bbox(ch) {
      self.bound(ox + bb.x_min * self.scale, oy - bb.y_max * self.scale);
      self.bound(ox + bb.x_max * self.scale, oy - bb.y_min * self.scale);
    }
  }

  /// Draw a glyph horizontally centred on `cx`, with its vertical origin (the
  /// SMuFL staff reference) at `cy`.
  fn glyph_centered(&mut self, ch: char, cx: f64, cy: f64, class: &str) {
    let cxoff = music_font::glyph_bbox(ch)
      .map(|bb| bb.center_x(self.scale))
      .unwrap_or(0.0);
    self.glyph_at(ch, cx - cxoff, cy, class);
  }

  /// Half the width of the notehead glyph used for `dur`, in user units — the
  /// horizontal offset from a head's centre to where its stem attaches.
  fn notehead_half_width(&self, dur: Dur) -> f64 {
    music_font::glyph_bbox(notehead_glyph(dur))
      .map(|bb| bb.half_width(self.scale))
      .unwrap_or(6.0)
  }

  /// Half the width of an arbitrary glyph, in user units.
  fn glyph_half_width(&self, ch: char) -> f64 {
    music_font::glyph_bbox(ch)
      .map(|bb| bb.half_width(self.scale))
      .unwrap_or(6.0)
  }

  /// Total ink width (user units) of a run of digit glyphs laid out adjacently
  /// with a small gap between them.
  fn digits_width(&self, digits: &[char]) -> f64 {
    const DIGIT_GAP: f64 = 1.0;
    let mut w = 0.0;
    for (i, &c) in digits.iter().enumerate() {
      if let Some(bb) = music_font::glyph_bbox(c) {
        w += (bb.x_max - bb.x_min) * self.scale;
        if i + 1 < digits.len() {
          w += DIGIT_GAP;
        }
      }
    }
    w
  }

  /// Half the width of a stacked time signature (the wider of its two rows).
  fn time_signature_half_width(&self, num: u32, den: u32) -> f64 {
    self
      .digits_width(&time_sig_digits(num))
      .max(self.digits_width(&time_sig_digits(den)))
      / 2.0
  }

  /// Draw a run of digits with the group horizontally centred on `cx` and each
  /// glyph's bounding box vertically centred on `cy`.
  fn draw_digits_centered(&mut self, cx: f64, cy: f64, digits: &[char]) {
    const DIGIT_GAP: f64 = 1.0;
    let mut left = cx - self.digits_width(digits) / 2.0;
    for &c in digits {
      let Some(bb) = music_font::glyph_bbox(c) else {
        continue;
      };
      // Left ink edge at `left`; bounding box vertically centred on `cy`.
      let ox = left - bb.x_min * self.scale;
      let oy = cy + (bb.y_min + bb.y_max) / 2.0 * self.scale;
      self.glyph_at(c, ox, oy, "timesig");
      left += (bb.x_max - bb.x_min) * self.scale + DIGIT_GAP;
    }
  }

  /// How far left of a note/chord's head centre its ink reaches, including any
  /// accidentals (mirrors the column staggering in [`draw_note`]).
  fn note_left_extent(&self, heads: &[Head], dur: Dur) -> f64 {
    let hw = self.notehead_half_width(dur);
    // A down-stem second is displaced a head-width to the left; accidentals then
    // sit left of that displaced head.
    let leftmost = second_offsets(heads, stem_up(heads), hw)
      .iter()
      .cloned()
      .fold(0.0, f64::min);
    let acc_base = -leftmost + hw;
    let mut reach = hw - leftmost;
    // Mirror the column packing in `draw_note`: the leftmost accidental sits in
    // the furthest column, one uniform slot per column beyond the base gap.
    let (columns, slot) = accidental_columns(heads, self.scale);
    for (i, h) in heads.iter().enumerate() {
      if let Some(col) = columns[i] {
        let w = accidental_width(h.accidental, self.scale);
        reach = reach.max(acc_base + 3.0 + col as f64 * slot + w);
      }
    }
    reach
  }

  /// How far right of a note/chord's head centre its ink reaches, including an
  /// up-stem flag and any second displaced to the right of the stem.
  fn note_right_extent(&self, heads: &[Head], dur: Dur) -> f64 {
    let hw = self.notehead_half_width(dur);
    let rightmost = second_offsets(heads, stem_up(heads), hw)
      .iter()
      .cloned()
      .fold(0.0, f64::max);
    let mut reach = hw + rightmost;
    if dur.has_stem()
      && stem_up(heads)
      && let Some(fc) = flag_glyph(dur, true)
      && let Some(bb) = music_font::glyph_bbox(fc)
    {
      reach = reach.max(hw - 0.7 + bb.x_max * self.scale);
    }
    reach
  }
}

/// Horizontal notehead displacements (in user units, one per entry of `heads`
/// in the given order) that place notes a *second* apart on opposite sides of
/// the stem so their heads do not overlap. Displaced heads move a full
/// head-width across the stem — to the right for up-stems, to the left for
/// down-stems — and a run of clustered seconds alternates sides.
fn second_offsets(heads: &[Head], up: bool, hw: f64) -> Vec<f64> {
  // Read the cluster from the stem's starting note: bottom-up for up-stems,
  // top-down for down-stems.
  let mut order: Vec<usize> = (0..heads.len()).collect();
  order.sort_by_key(|&i| heads[i].dn);
  if !up {
    order.reverse();
  }
  let step = if up { 2.0 * hw } else { -2.0 * hw };
  let mut offsets = vec![0.0; heads.len()];
  let mut prev_dn: Option<i32> = None;
  let mut prev_displaced = false;
  for &i in &order {
    let dn = heads[i].dn;
    if prev_dn.is_some_and(|p| (p - dn).abs() == 1) && !prev_displaced {
      offsets[i] = step;
      prev_displaced = true;
    } else {
      prev_displaced = false;
    }
    prev_dn = Some(dn);
  }
  offsets
}

/// The Leland notehead glyph for a rhythmic value.
fn notehead_glyph(dur: Dur) -> char {
  match dur {
    Dur::Whole => glyph::NOTEHEAD_WHOLE,
    Dur::Half => glyph::NOTEHEAD_HALF,
    _ => glyph::NOTEHEAD_BLACK,
  }
}

/// The Leland accidental glyph for a signed accidental count: single or double
/// sharp for positive, single or double flat for negative, `None` for natural.
fn accidental_glyph(accidental: i32) -> Option<char> {
  Some(match accidental {
    n if n >= 2 => glyph::DOUBLE_SHARP,
    1 => glyph::SHARP,
    -1 => glyph::FLAT,
    n if n <= -2 => glyph::DOUBLE_FLAT,
    _ => return None,
  })
}

/// Vertical half-height (user units) of an accidental glyph, used to decide
/// whether two accidentals in the same column would overlap.
fn accidental_half_height(accidental: i32, scale: f64) -> f64 {
  accidental_glyph(accidental)
    .and_then(music_font::glyph_bbox)
    .map(|bb| (bb.y_max - bb.y_min) * scale / 2.0)
    .unwrap_or(STEP)
}

/// Full width (user units) of an accidental glyph.
fn accidental_width(accidental: i32, scale: f64) -> f64 {
  accidental_glyph(accidental)
    .and_then(music_font::glyph_bbox)
    .map(|bb| (bb.x_max - bb.x_min) * scale)
    .unwrap_or(0.0)
}

/// Stack a chord's accidentals into columns to the left of the note heads so
/// none overlap. Returns, for each head, the column its accidental occupies
/// (`None` for naturals; `0` is the column nearest the heads) together with the
/// uniform column pitch. Following standard engraving, the accidentals are
/// placed top to bottom, each in the closest column whose previous (higher)
/// accidental clears it vertically — so a stack of thirds alternates between two
/// columns rather than marching ever further left.
fn accidental_columns(heads: &[Head], scale: f64) -> (Vec<Option<usize>>, f64) {
  let mut order: Vec<usize> = (0..heads.len())
    .filter(|&i| heads[i].accidental != 0)
    .collect();
  order.sort_by_key(|&i| std::cmp::Reverse(heads[i].dn));

  let mut columns = vec![None; heads.len()];
  // The lowest occupied edge (largest y) in each column so far.
  let mut column_bottom: Vec<f64> = Vec::new();
  let mut max_width = 0.0f64;
  for i in order {
    let acc = heads[i].accidental;
    let hh = accidental_half_height(acc, scale);
    max_width = max_width.max(accidental_width(acc, scale));
    let cy = dn_y(heads[i].dn);
    let top = cy - hh;
    // The first column whose last (higher) accidental clears this one's top.
    let col = column_bottom
      .iter()
      .position(|&bottom| bottom <= top - 1.0)
      .unwrap_or(column_bottom.len());
    if col == column_bottom.len() {
      column_bottom.push(cy + hh);
    } else {
      column_bottom[col] = cy + hh;
    }
    columns[i] = Some(col);
  }
  (columns, max_width + 3.0)
}

/// Draw a Leland sharp or flat glyph to the left of a note head, its right edge
/// a small gap left of `head_left` and vertically centred on `cy`. `column` and
/// `slot` place stacked chord accidentals in uniform-width columns so their
/// signs do not collide horizontally regardless of glyph width.
fn draw_accidental(
  cv: &mut Canvas,
  head_left: f64,
  cy: f64,
  accidental: i32,
  column: usize,
  slot: f64,
) {
  let Some(ch) = accidental_glyph(accidental) else {
    return;
  };
  let Some(bb) = music_font::glyph_bbox(ch) else {
    return;
  };
  // Where this accidental's right edge should sit.
  let right = head_left - 3.0 - column as f64 * slot;
  // Place the origin so the glyph's own right extent lands on `right`.
  let ox = right - bb.x_max * cv.scale;
  cv.glyph_at(ch, ox, cy, "accidental");
}

/// Draw ledger lines through a head at diatonic position `dn` centred at `cx`,
/// extending `head_half_width` plus a small overhang either side of the head.
fn draw_ledger_lines(cv: &mut Canvas, cx: f64, dn: i32, head_half_width: f64) {
  let half = head_half_width + 2.5;
  if dn <= BOTTOM_LINE_DN - 2 {
    let mut e = BOTTOM_LINE_DN - 2; // first ledger below the staff (C4 = 28)
    while e >= dn {
      let y = dn_y(e);
      cv.line(cx - half, y, cx + half, y, 1.2);
      e -= 2;
    }
  }
  let top_line_dn = BOTTOM_LINE_DN + 8; // F5 = 38
  if dn >= top_line_dn + 2 {
    let mut e = top_line_dn + 2; // first ledger above the staff (A5 = 40)
    while e <= dn {
      let y = dn_y(e);
      cv.line(cx - half, y, cx + half, y, 1.2);
      e += 2;
    }
  }
}

/// Draw a note or chord centred at `x`.
fn draw_note(cv: &mut Canvas, x: f64, heads: &[Head], dur: Dur) {
  let nh = notehead_glyph(dur);
  let hw = cv.notehead_half_width(dur);
  let up = stem_up(heads);
  // Displace notes a second apart to the opposite side of the stem so their
  // heads do not overlap.
  let offsets = second_offsets(heads, up, hw);
  // Accidentals sit in a column to the left of the whole cluster, so anchor
  // them at the leftmost head edge (a down-stem second can reach further left).
  let leftmost = offsets.iter().cloned().fold(0.0, f64::min);
  let acc_anchor = x + leftmost - hw;
  // Ledger lines behind the heads.
  for (i, h) in heads.iter().enumerate() {
    draw_ledger_lines(cv, x + offsets[i], h.dn, hw);
  }
  // Pack the accidentals into non-overlapping columns to the left of the heads.
  let (columns, slot) = accidental_columns(heads, cv.scale);
  // Heads (top-down), each accidental placed in its assigned column.
  for i in (0..heads.len()).rev() {
    let h = &heads[i];
    cv.glyph_centered(nh, x + offsets[i], dn_y(h.dn), "notehead");
    if let Some(col) = columns[i] {
      draw_accidental(cv, acc_anchor, dn_y(h.dn), h.accidental, col, slot);
    }
  }
  if !dur.has_stem() {
    return;
  }
  let top_dn = heads.iter().map(|h| h.dn).max().unwrap();
  let bottom_dn = heads.iter().map(|h| h.dn).min().unwrap();
  // The stem attaches at the right edge of the head (up) or the left edge
  // (down), just inside the notehead so it meets the outline cleanly.
  let (sx, y_near, y_far) = if up {
    (x + hw - 0.7, dn_y(bottom_dn), dn_y(top_dn) - STEM_LEN)
  } else {
    (x - hw + 0.7, dn_y(top_dn), dn_y(bottom_dn) + STEM_LEN)
  };
  cv.line(sx, y_near, sx, y_far, 1.3);
  // A flag glyph for eighth / sixteenth notes, hung at the stem tip. Leland's
  // 16th-note flag glyph already carries both flags.
  if let Some(fc) = flag_glyph(dur, up) {
    cv.glyph_at(fc, sx, y_far, "flag");
  }
}

/// Whether a note or chord's stem points up: notes centred on the lower half of
/// the staff stem up, the rest stem down.
fn stem_up(heads: &[Head]) -> bool {
  let top = heads.iter().map(|h| h.dn).max().unwrap();
  let bottom = heads.iter().map(|h| h.dn).min().unwrap();
  let mid = BOTTOM_LINE_DN + 4; // middle staff line, B4 = 34
  (top + bottom) as f64 / 2.0 < mid as f64
}

/// The Leland flag glyph for an eighth / sixteenth note, or `None` otherwise.
fn flag_glyph(dur: Dur, up: bool) -> Option<char> {
  Some(match (dur, up) {
    (Dur::Eighth, true) => glyph::FLAG_8TH_UP,
    (Dur::Eighth, false) => glyph::FLAG_8TH_DOWN,
    (Dur::Sixteenth, true) => glyph::FLAG_16TH_UP,
    (Dur::Sixteenth, false) => glyph::FLAG_16TH_DOWN,
    _ => return None,
  })
}

/// Draw a rest centred at `x`. The whole rest hangs from the fourth staff line
/// (D5) and the half rest sits on the middle line; the shorter rests are
/// registered on the middle line, matching SMuFL's glyph origins.
fn draw_rest(cv: &mut Canvas, x: f64, dur: Dur) {
  // The whole rest hangs from the fourth line (D5); the others are registered
  // on the middle line.
  let oy = match dur {
    Dur::Whole => dn_y(BOTTOM_LINE_DN + 6),
    _ => dn_y(BOTTOM_LINE_DN + 4),
  };
  cv.glyph_centered(rest_glyph(dur), x, oy, "rest");
}

/// The Leland rest glyph for a rhythmic value.
fn rest_glyph(dur: Dur) -> char {
  match dur {
    Dur::Whole => glyph::REST_WHOLE,
    Dur::Half => glyph::REST_HALF,
    Dur::Quarter => glyph::REST_QUARTER,
    Dur::Eighth => glyph::REST_8TH,
    Dur::Sixteenth => glyph::REST_16TH,
  }
}

/// The `timeSig0`…`timeSig9` glyphs spelling a (positive) number's decimal
/// digits, left to right.
fn time_sig_digits(n: u32) -> Vec<char> {
  n.to_string()
    .bytes()
    .map(|b| music_font::glyph::time_sig_digit(b - b'0'))
    .collect()
}

/// Draw a stacked time signature centred horizontally on `cx`: the numerator on
/// the upper half of the staff (the fourth line) and the denominator on the
/// lower half (the second line), matching standard notation.
fn draw_time_signature(cv: &mut Canvas, cx: f64, num: u32, den: u32) {
  cv.draw_digits_centered(cx, dn_y(BOTTOM_LINE_DN + 6), &time_sig_digits(num));
  cv.draw_digits_centered(cx, dn_y(BOTTOM_LINE_DN + 2), &time_sig_digits(den));
}

/// Draw the treble (G) clef with its left edge at `x`. Leland's `gClef` is
/// registered so its origin sits on the G4 staff line, which is exactly where
/// the clef's curl must centre.
fn draw_treble_clef(cv: &mut Canvas, x: f64) {
  let g = dn_y(BOTTOM_LINE_DN + 2); // G4 line
  cv.glyph_at(glyph::G_CLEF, x, g, "clef");
}

/// Render a computational-music object to a standalone SVG, or `None` when the
/// expression is not a music object that carries notation.
pub fn music_to_svg(expr: &Expr) -> Option<String> {
  let mut glyphs = Vec::new();
  let mut ts = None;
  let container = collect(expr, &mut glyphs, &mut ts);
  if glyphs.is_empty() {
    return None;
  }

  let stroke = theme().text_primary;
  let scale = music_font::scale_for_gap(GAP);
  let mut cv = Canvas::new(stroke, scale);

  // The clef sits at the left; `right_edge` tracks the rightmost inked point so
  // each following glyph can be nudged clear of whatever precedes it.
  let clef_x = STAFF_X0 + 4.0;
  draw_treble_clef(&mut cv, clef_x);
  let clef_right = clef_x
    + music_font::glyph_bbox(glyph::G_CLEF)
      .map(|bb| bb.x_max * scale)
      .unwrap_or(28.0);
  let first_note_x = clef_right + 14.0;

  // Lay the glyphs out left to right. `cursor` is the preferred centre of the
  // next note (a steady rhythmic advance); `right_edge` is the rightmost ink so
  // far. A note whose left extent (its accidentals) would collide with the
  // previous ink is pushed right just enough to clear it, `PAD` apart.
  const PAD: f64 = 4.0;
  let mut cursor = first_note_x;
  let mut right_edge = clef_right;
  for g in &glyphs {
    match g {
      Glyph::Note { heads, dur } => {
        let left = cv.note_left_extent(heads, *dur);
        let center = cursor.max(right_edge + PAD + left);
        draw_note(&mut cv, center, heads, *dur);
        right_edge = center + cv.note_right_extent(heads, *dur);
        cursor = center + ADVANCE;
      }
      Glyph::Rest { dur } => {
        let hw = cv.glyph_half_width(rest_glyph(*dur));
        let center = cursor.max(right_edge + PAD + hw);
        draw_rest(&mut cv, center, *dur);
        right_edge = center + hw;
        cursor = center + ADVANCE;
      }
      Glyph::TimeSig { num, den } => {
        let hw = cv.time_signature_half_width(*num, *den);
        let center = (right_edge + PAD + hw).max(clef_right + 8.0 + hw);
        draw_time_signature(&mut cv, center, *num, *den);
        right_edge = center + hw;
        cursor = (center + hw + 18.0).max(cursor);
      }
      Glyph::Barline => {
        let bx = (right_edge + PAD).max(cursor - ADVANCE / 2.0);
        cv.line(bx, dn_y(BOTTOM_LINE_DN + 8), bx, dn_y(BOTTOM_LINE_DN), 1.3);
        right_edge = bx;
        cursor = bx + ADVANCE * 0.5;
      }
    }
  }
  let staff_x_end = (right_edge + 12.0).max(first_note_x + 8.0);

  // The five staff lines (drawn first, conceptually behind the glyphs — SVG
  // paint order does not matter here since strokes are opaque and thin).
  let mut staff = String::new();
  for i in 0..5 {
    let y = BOTTOM_LINE_Y - i as f64 * GAP;
    staff.push_str(&format!(
      "<line x1=\"{STAFF_X0:.2}\" y1=\"{y:.2}\" x2=\"{staff_x_end:.2}\" \
       y2=\"{y:.2}\" stroke=\"{stroke}\" stroke-width=\"1\"/>"
    ));
  }
  cv.bound(STAFF_X0, dn_y(BOTTOM_LINE_DN + 8));
  cv.bound(staff_x_end, dn_y(BOTTOM_LINE_DN));

  // A closing barline for whole containers (measures/voices/scores/scales).
  if container {
    staff.push_str(&format!(
      "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x1:.2}\" y2=\"{y2:.2}\" \
       stroke=\"{stroke}\" stroke-width=\"1.6\"/>",
      x1 = staff_x_end,
      y1 = dn_y(BOTTOM_LINE_DN + 8),
      y2 = dn_y(BOTTOM_LINE_DN),
    ));
  }

  // Fit the viewBox around everything with a small margin.
  let pad = 6.0;
  let minx = cv.minx.min(STAFF_X0) - pad;
  let miny = cv.miny - pad;
  let width = (cv.maxx.max(staff_x_end) - minx) + pad;
  let height = (cv.maxy - miny) + pad;

  Some(format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.0}\" \
     height=\"{h:.0}\" viewBox=\"{minx:.2} {miny:.2} {width:.2} {height:.2}\">\
     {staff}{body}</svg>",
    w = width,
    h = height,
    body = cv.out,
  ))
}

/// Extract a numeric attribute (e.g. `width`, `height`) from the opening tag
/// of an SVG string produced by [`music_to_svg`].
fn svg_attr(svg: &str, attr: &str) -> Option<f64> {
  let key = format!("{attr}=\"");
  let start = svg.find(&key)? + key.len();
  let rest = &svg[start..];
  let end = rest.find('"')?;
  rest[..end].parse().ok()
}

/// Turn each `<svg …>…</svg>` produced by [`music_to_svg`] into a nested SVG
/// element positioned at `(x, y)` by injecting the coordinates into its
/// opening tag. The inner `width`/`height`/`viewBox` keep it at natural size.
fn nest_svg_at(svg: &str, x: f64, y: f64) -> String {
  svg.replacen("<svg ", &format!("<svg x=\"{x:.2}\" y=\"{y:.2}\" "), 1)
}

/// Render a plain list of music events (each a [`MUSIC_OBJECT_HEADS`] object)
/// as `{ <staff>, <staff>, … }` — every element drawn as its own staff,
/// laid out horizontally with brace and comma separators, matching how the
/// bracketed list would otherwise appear as text.
///
/// Returns `None` unless `expr` is a non-empty music-object list whose elements
/// all render to a staff.
pub fn music_list_to_svg(expr: &Expr) -> Option<String> {
  let items = match expr {
    Expr::List(items) if !items.is_empty() => items,
    _ => return None,
  };

  // Render every element to its own staff first; bail if any element is not
  // drawable so the caller can fall back to the text form.
  let staves: Vec<String> =
    items.iter().map(music_to_svg).collect::<Option<_>>()?;

  let stroke = theme().text_primary;
  const FONT: f64 = 14.0;
  const CHAR_W: f64 = FONT * 0.62; // monospace advance for `{`, `,`, `}`
  const SEP_GAP: f64 = 4.0; // padding around separators

  // Overall height is the tallest staff; separators are centred within it.
  let height = staves
    .iter()
    .filter_map(|s| svg_attr(s, "height"))
    .fold(0.0_f64, f64::max)
    .max(FONT);
  let mid_y = height / 2.0;
  let text_baseline = mid_y + FONT * 0.35;

  let mut body = String::new();
  let mut x = 0.0_f64;

  let text = |body: &mut String, x: &mut f64, s: &str| {
    body.push_str(&format!(
      "<text x=\"{x:.2}\" y=\"{text_baseline:.2}\" font-family=\"monospace\" \
       font-size=\"{FONT:.1}\" fill=\"{stroke}\" stroke=\"none\">{s}</text>"
    ));
    *x += CHAR_W;
  };

  text(&mut body, &mut x, "{");
  x += SEP_GAP;
  for (i, staff) in staves.iter().enumerate() {
    if i > 0 {
      text(&mut body, &mut x, ",");
      x += SEP_GAP;
    }
    let w = svg_attr(staff, "width").unwrap_or(0.0);
    let h = svg_attr(staff, "height").unwrap_or(height);
    body.push_str(&nest_svg_at(staff, x, mid_y - h / 2.0));
    x += w + SEP_GAP;
  }
  text(&mut body, &mut x, "}");

  let width = x;
  Some(format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" \
     height=\"{height:.0}\" viewBox=\"0 0 {width:.2} {height:.2}\">{body}</svg>"
  ))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A bare `MusicPitch` (pitch only, no rhythmic value).
  fn note(name: &str) -> Expr {
    Expr::FunctionCall {
      name: "MusicPitch".to_string(),
      args: vec![Expr::String(name.to_string())].into(),
    }
  }

  /// A `MusicNote` wrapping a pitch — a rhythmic event that does render.
  fn music_note(name: &str) -> Expr {
    Expr::FunctionCall {
      name: "MusicNote".to_string(),
      args: vec![note(name)].into(),
    }
  }

  #[test]
  fn bare_pitch_does_not_render() {
    // A `MusicPitch` has no length, so it is not staff notation and stays
    // symbolic; a `MusicNote` built from the same pitch does render.
    assert!(music_to_svg(&note("C4")).is_none());
    let svg = music_to_svg(&music_note("C4")).expect("a note should render");
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("class=\"notehead\"")); // a Leland note head
    assert!(svg.contains("class=\"clef\"")); // the Leland treble clef
    assert!(svg.contains("<line")); // staff lines
  }

  #[test]
  fn diatonic_number_places_middle_c_below_the_staff() {
    // E4 is the bottom line (30); middle C (C4) is two steps lower.
    assert_eq!(diatonic_number("E4").unwrap().0, 30);
    assert_eq!(diatonic_number("C4").unwrap().0, 28);
    assert_eq!(diatonic_number("F5").unwrap().0, 38); // top line
    // Accidentals are reported but do not shift the diatonic position.
    assert_eq!(diatonic_number("C#4").unwrap(), (28, 1));
    assert_eq!(diatonic_number("Db4").unwrap(), (29, -1));
  }

  #[test]
  fn non_music_expressions_do_not_render() {
    assert!(music_to_svg(&Expr::Integer(3)).is_none());
    assert!(
      music_to_svg(&Expr::FunctionCall {
        name: "MusicDuration".to_string(),
        args: vec![Expr::String("Half".to_string())].into(),
      })
      .is_none()
    );
  }

  #[test]
  fn chord_renders_multiple_heads() {
    let chord = Expr::FunctionCall {
      name: "MusicChord".to_string(),
      args: vec![Expr::List(vec![note("C4"), note("E4"), note("G4")].into())]
        .into(),
    };
    let svg = music_to_svg(&chord).unwrap();
    assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
  }

  #[test]
  fn named_chord_association_renders_multiple_heads() {
    // `MusicChord["GMajor"]` canonicalizes to the association form
    // `MusicChord[<|"Name" -> "Major", "Root" -> MusicPitch[…]|>]`; it must
    // still spell its stacked-thirds tones (G/B/D) and render on a staff.
    let chord = crate::functions::music_ast::music_chord(&[Expr::String(
      "GMajor".to_string(),
    )])
    .expect("GMajor should canonicalize");
    assert!(matches!(
      &chord,
      Expr::FunctionCall { name, args }
        if name == "MusicChord" && matches!(args.first(), Some(Expr::Association(_)))
    ));
    let svg = music_to_svg(&chord).expect("a named chord should render");
    assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
    assert!(svg.contains("class=\"clef\"")); // treble clef
  }

  #[test]
  fn pitch_list_association_renders_multiple_heads() {
    // A transposed pitch-list chord is returned as the canonical
    // `MusicChord[<|"PitchList" -> {MusicPitch[…], …}|>]`; it must still render
    // every tone (the playground draws it as one chord).
    let chord = crate::functions::plus_ast(&[
      Expr::FunctionCall {
        name: "MusicChord".to_string(),
        args: vec![Expr::List(vec![note("C4"), note("E4"), note("G4")].into())]
          .into(),
      },
      Expr::FunctionCall {
        name: "MusicInterval".to_string(),
        args: vec![Expr::Integer(5)].into(),
      },
    ])
    .expect("chord + interval should evaluate");
    assert!(matches!(
      &chord,
      Expr::FunctionCall { name, args }
        if name == "MusicChord" && matches!(args.first(), Some(Expr::Association(_)))
    ));
    let svg = music_to_svg(&chord).expect("a pitch-list chord should render");
    assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
  }

  #[test]
  fn seconds_are_displaced_to_alternating_sides() {
    let hw = 6.0;
    // A cluster of stacked seconds (C, D, E) alternates sides so no two adjacent
    // heads share a column.
    let cluster = [
      Head {
        dn: 28,
        accidental: 0,
      }, // C4
      Head {
        dn: 29,
        accidental: 0,
      }, // D4
      Head {
        dn: 30,
        accidental: 0,
      }, // E4
    ];
    // Up-stem: the bottom head is on the normal (left) side, the second above it
    // is displaced right, the third returns to the normal side.
    let up = second_offsets(&cluster, true, hw);
    assert_eq!(up, vec![0.0, 2.0 * hw, 0.0]);
    // Down-stem: read top-down, displaced heads move left instead.
    let down = second_offsets(&cluster, false, hw);
    assert_eq!(down, vec![0.0, -2.0 * hw, 0.0]);
    // Thirds never collide, so nothing is displaced.
    let thirds = [
      Head {
        dn: 28,
        accidental: 0,
      },
      Head {
        dn: 30,
        accidental: 0,
      },
      Head {
        dn: 32,
        accidental: 0,
      },
    ];
    assert_eq!(second_offsets(&thirds, true, hw), vec![0.0, 0.0, 0.0]);
  }

  #[test]
  fn stacked_accidentals_alternate_between_two_columns() {
    // The C-minor-third stack C,Eb,Gb,Bbb,Dbb has four (double) flats a third
    // apart. They must pack into two alternating columns, not one overlapping
    // pile or an ever-widening staircase.
    let heads = [
      Head {
        dn: 28,
        accidental: 0,
      }, // C4 (natural, no accidental)
      Head {
        dn: 30,
        accidental: -1,
      }, // Eb4
      Head {
        dn: 32,
        accidental: -1,
      }, // Gb4
      Head {
        dn: 34,
        accidental: -2,
      }, // Bbb4
      Head {
        dn: 36,
        accidental: -2,
      }, // Dbb5
    ];
    let (columns, _slot) = accidental_columns(&heads, 0.02);
    assert_eq!(columns[0], None); // the natural C has no accidental
    // Top-down (Dbb, Bbb, Gb, Eb) alternate col 0,1,0,1.
    assert_eq!(columns[4], Some(0)); // Dbb (top)
    assert_eq!(columns[3], Some(1)); // Bbb
    assert_eq!(columns[2], Some(0)); // Gb
    assert_eq!(columns[1], Some(1)); // Eb (bottom)
    // Only two columns are ever used.
    let max_col = columns.iter().flatten().copied().max().unwrap();
    assert_eq!(max_col, 1);
  }

  #[test]
  fn scale_expands_to_eight_notes() {
    let scale = Expr::FunctionCall {
      name: "MusicScale".to_string(),
      args: vec![Expr::String("Major".to_string()), note("C4")].into(),
    };
    let mut glyphs = Vec::new();
    collect(&scale, &mut glyphs, &mut None);
    let notes = glyphs
      .iter()
      .filter(|g| matches!(g, Glyph::Note { .. }))
      .count();
    assert_eq!(notes, 8);
  }

  #[test]
  fn plain_list_of_notes_renders_one_staff_per_element() {
    // A bare list of music events keeps its list shape `{ <staff>, … }`: one
    // nested staff (its own clef) per element, joined by brace/comma text.
    let list = Expr::List(
      vec![music_note("C4"), music_note("G4"), music_note("F4")].into(),
    );
    assert!(crate::functions::music_ast::is_music_object_list(&list));
    let svg = music_list_to_svg(&list).expect("a note list should render");
    // Three separate staves: three clefs, three noteheads, three nested SVGs.
    assert_eq!(svg.matches("class=\"clef\"").count(), 3);
    assert_eq!(svg.matches("class=\"notehead\"").count(), 3);
    assert_eq!(svg.matches("<svg").count(), 4); // 1 outer + 3 nested
    // Brace and comma separators keep the list structure visible.
    assert!(svg.contains(">{</text>"));
    assert!(svg.contains(">}</text>"));
    assert_eq!(svg.matches(">,</text>").count(), 2);
  }

  #[test]
  fn measure_appends_a_barline() {
    let measure = Expr::FunctionCall {
      name: "MusicMeasure".to_string(),
      args: vec![Expr::List(vec![music_note("C4"), music_note("D4")].into())]
        .into(),
    };
    let mut glyphs = Vec::new();
    assert!(collect(&measure, &mut glyphs, &mut None));
    // A time signature, the two notes, then a closing barline.
    assert!(matches!(
      glyphs.first(),
      Some(Glyph::TimeSig { num: 4, den: 4 })
    ));
    assert!(
      glyphs
        .iter()
        .filter(|g| matches!(g, Glyph::Note { .. }))
        .count()
        == 2
    );
    assert!(matches!(glyphs.last(), Some(Glyph::Barline)));
  }

  #[test]
  fn voice_shows_meter_and_reprints_on_change() {
    // A voice prints 4/4 at the start, then a new time signature when the meter
    // changes mid-stream, but not a repeated identical one.
    let voice = Expr::FunctionCall {
      name: "MusicVoice".to_string(),
      args: vec![Expr::List(
        vec![
          Expr::String("C".to_string()),
          Expr::FunctionCall {
            name: "MusicTimeSignature".to_string(),
            args: vec![Expr::Integer(3), Expr::Integer(4)].into(),
          },
          Expr::String("D".to_string()),
          Expr::FunctionCall {
            name: "MusicTimeSignature".to_string(),
            args: vec![Expr::Integer(3), Expr::Integer(4)].into(),
          },
          Expr::String("E".to_string()),
        ]
        .into(),
      )]
      .into(),
    };
    let mut glyphs = Vec::new();
    collect(&voice, &mut glyphs, &mut None);
    let sigs: Vec<(u32, u32)> = glyphs
      .iter()
      .filter_map(|g| match g {
        Glyph::TimeSig { num, den } => Some((*num, *den)),
        _ => None,
      })
      .collect();
    // 4/4 default at the start, then 3/4 once — the repeated 3/4 is suppressed.
    assert_eq!(sigs, vec![(4, 4), (3, 4)]);
  }

  #[test]
  fn measures_reprint_meter_returning_to_the_default() {
    // A voice of measures 4/4, 3/4, 4/4 must reprint the meter at every change,
    // including the return to 4/4 — a measure without its own time signature is
    // 4/4, not an inheritance of the previous 3/4.
    let measure = |ts: Option<(i128, i128)>| {
      let mut args =
        vec![Expr::List(vec![music_note("C4"), music_note("D4")].into())];
      if let Some((n, d)) = ts {
        args.push(Expr::FunctionCall {
          name: "MusicTimeSignature".to_string(),
          args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
        });
      }
      Expr::FunctionCall {
        name: "MusicMeasure".to_string(),
        args: args.into(),
      }
    };
    let voice = Expr::FunctionCall {
      name: "MusicVoice".to_string(),
      args: vec![Expr::List(
        vec![measure(None), measure(Some((3, 4))), measure(None)].into(),
      )]
      .into(),
    };
    let mut glyphs = Vec::new();
    collect(&voice, &mut glyphs, &mut None);
    let sigs: Vec<(u32, u32)> = glyphs
      .iter()
      .filter_map(|g| match g {
        Glyph::TimeSig { num, den } => Some((*num, *den)),
        _ => None,
      })
      .collect();
    assert_eq!(sigs, vec![(4, 4), (3, 4), (4, 4)]);
  }
}
