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
  Barline,
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
  let octave: i32 = name.get(idx..)?.parse().ok()?;
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
        Expr::Association(pairs) => pairs.iter().find_map(|(k, v)| match k {
          Expr::String(s) if s == "Duration" => value_of(v),
          _ => None,
        }),
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

/// Collect the drawable glyphs of a music object in reading order. Returns
/// `true` when `expr` was a container whose contents are conventionally closed
/// by a final barline.
fn collect(expr: &Expr, out: &mut Vec<Glyph>) -> bool {
  match expr {
    Expr::List(items) => {
      for it in items.iter() {
        collect(it, out);
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
        if let Expr::List(items) = &args[0] {
          for it in items.iter() {
            if let Some(h) = pitch_head(it) {
              heads.push(h);
            }
          }
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
      "MusicMeasure" => {
        for arg in args {
          collect(arg, out);
        }
        out.push(Glyph::Barline);
        true
      }
      "MusicVoice" | "MusicScore" => {
        for arg in args {
          collect(arg, out);
        }
        true
      }
      _ => false,
    },
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

  /// How far left of a note/chord's head centre its ink reaches, including any
  /// accidentals (mirrors the column staggering in [`draw_note`]).
  fn note_left_extent(&self, heads: &[Head], dur: Dur) -> f64 {
    let hw = self.notehead_half_width(dur);
    let mut reach = hw;
    let mut column = 0usize;
    let mut prev_dn: Option<i32> = None;
    for h in heads.iter().rev() {
      if h.accidental == 0 {
        continue;
      }
      if prev_dn.is_some_and(|p| (p - h.dn).abs() <= 2) {
        column += 1;
      } else {
        column = 0;
      }
      let ch = if h.accidental > 0 {
        glyph::SHARP
      } else {
        glyph::FLAT
      };
      if let Some(bb) = music_font::glyph_bbox(ch) {
        let w = (bb.x_max - bb.x_min) * self.scale;
        reach = reach.max(hw + 3.0 + column as f64 * (w + 2.0) + w);
      }
      prev_dn = Some(h.dn);
    }
    reach
  }

  /// How far right of a note/chord's head centre its ink reaches, including an
  /// up-stem flag.
  fn note_right_extent(&self, heads: &[Head], dur: Dur) -> f64 {
    let hw = self.notehead_half_width(dur);
    if dur.has_stem()
      && stem_up(heads)
      && let Some(fc) = flag_glyph(dur, true)
      && let Some(bb) = music_font::glyph_bbox(fc)
    {
      return (hw - 0.7 + bb.x_max * self.scale).max(hw);
    }
    hw
  }
}

/// The Leland notehead glyph for a rhythmic value.
fn notehead_glyph(dur: Dur) -> char {
  match dur {
    Dur::Whole => glyph::NOTEHEAD_WHOLE,
    Dur::Half => glyph::NOTEHEAD_HALF,
    _ => glyph::NOTEHEAD_BLACK,
  }
}

/// Draw a Leland sharp or flat glyph to the left of a note head, its right edge
/// a small gap left of `head_left` and vertically centred on `cy`. `column`
/// shifts stacked chord accidentals further left so their signs do not collide.
fn draw_accidental(
  cv: &mut Canvas,
  head_left: f64,
  cy: f64,
  accidental: i32,
  column: usize,
) {
  let ch = if accidental > 0 {
    glyph::SHARP
  } else if accidental < 0 {
    glyph::FLAT
  } else {
    return;
  };
  let Some(bb) = music_font::glyph_bbox(ch) else {
    return;
  };
  let width = (bb.x_max - bb.x_min) * cv.scale;
  // Where this accidental's right edge should sit.
  let right = head_left - 3.0 - column as f64 * (width + 2.0);
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
  // Ledger lines behind the heads.
  for h in heads {
    draw_ledger_lines(cv, x, h.dn, hw);
  }
  // Heads (top-down), each with any accidental staggered into columns so
  // adjacent signs do not overlap.
  let mut column = 0usize;
  let mut prev_dn: Option<i32> = None;
  for h in heads.iter().rev() {
    cv.glyph_centered(nh, x, dn_y(h.dn), "notehead");
    if h.accidental != 0 {
      if prev_dn.is_some_and(|p| (p - h.dn).abs() <= 2) {
        column += 1;
      } else {
        column = 0;
      }
      draw_accidental(cv, x - hw, dn_y(h.dn), h.accidental, column);
      prev_dn = Some(h.dn);
    }
  }
  if !dur.has_stem() {
    return;
  }
  let up = stem_up(heads);
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
  let container = collect(expr, &mut glyphs);
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
  fn scale_expands_to_eight_notes() {
    let scale = Expr::FunctionCall {
      name: "MusicScale".to_string(),
      args: vec![Expr::String("Major".to_string()), note("C4")].into(),
    };
    let mut glyphs = Vec::new();
    collect(&scale, &mut glyphs);
    let notes = glyphs
      .iter()
      .filter(|g| matches!(g, Glyph::Note { .. }))
      .count();
    assert_eq!(notes, 8);
  }

  #[test]
  fn measure_appends_a_barline() {
    let measure = Expr::FunctionCall {
      name: "MusicMeasure".to_string(),
      args: vec![Expr::List(vec![music_note("C4"), music_note("D4")].into())]
        .into(),
    };
    let mut glyphs = Vec::new();
    assert!(collect(&measure, &mut glyphs));
    // Two notes then a closing barline.
    assert!(matches!(glyphs.first(), Some(Glyph::Note { .. })));
    assert!(matches!(glyphs.last(), Some(Glyph::Barline)));
  }
}
