//! SVG rendering of the Wolfram Language 15.0 ComputationalMusic objects on a
//! musical staff, the way Mathematica displays them.
//!
//! The music *objects* stay canonical symbolic expressions everywhere (so the
//! CLI still prints `MusicNote[MusicPitch[C4], …]`); only the visual hosts —
//! the Woxi Playground and Woxi Studio — turn them into notation. A single
//! entry point, [`music_to_svg`], produces a self-contained SVG that renders
//! identically in a browser and in Studio's `resvg` rasterizer, so all the
//! notation (staff, treble clef, note heads, stems, flags, accidentals,
//! ledger lines, rests and barlines) is drawn geometrically rather than with
//! musical Unicode glyphs, which the bundled fonts do not carry.

use crate::functions::graphics::theme;
use crate::functions::music_ast::{
  midi_to_pitch_name, pitch_name_to_midi, resolve_pitch_name,
};
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
/// `x` of the first note head, leaving room for the clef.
const FIRST_NOTE_X: f64 = 46.0;
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
  /// Whether the note head is filled (`true`) or open (`false`).
  fn filled(self) -> bool {
    matches!(self, Dur::Quarter | Dur::Eighth | Dur::Sixteenth)
  }
  /// Whether the note carries a stem (everything but a whole note).
  fn has_stem(self) -> bool {
    self != Dur::Whole
  }
  /// Number of flags on the stem (eighth = 1, sixteenth = 2).
  fn flags(self) -> u32 {
    match self {
      Dur::Eighth => 1,
      Dur::Sixteenth => 2,
      _ => 0,
    }
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
      "MusicPitch" => {
        if let Some(h) = pitch_head(expr) {
          out.push(Glyph::Note {
            heads: vec![h],
            dur: Dur::Quarter,
          });
        }
        false
      }
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

/// Accumulates SVG path/shape fragments while tracking the drawn bounding box
/// so the final `viewBox` can be fitted with padding.
struct Canvas {
  out: String,
  stroke: String,
  minx: f64,
  miny: f64,
  maxx: f64,
  maxy: f64,
}

impl Canvas {
  fn new(stroke: &str) -> Self {
    Canvas {
      out: String::new(),
      stroke: stroke.to_string(),
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

  /// A note head: an ellipse tilted like an engraved note, filled or open.
  fn note_head(&mut self, cx: f64, cy: f64, filled: bool) {
    let (rx, ry) = (6.3, 4.6);
    let fill = if filled {
      format!("fill=\"{}\"", self.stroke)
    } else {
      format!(
        "fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"",
        self.stroke
      )
    };
    self.out.push_str(&format!(
      "<ellipse cx=\"{cx:.2}\" cy=\"{cy:.2}\" rx=\"{rx}\" ry=\"{ry}\" \
       transform=\"rotate(-22 {cx:.2} {cy:.2})\" {fill}/>"
    ));
    self.bound(cx - rx - 1.0, cy - ry - 1.0);
    self.bound(cx + rx + 1.0, cy + ry + 1.0);
  }

  /// A small filled disc (clef terminal, augmentation/rest dots).
  fn dot(&mut self, cx: f64, cy: f64, r: f64) {
    self.out.push_str(&format!(
      "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r}\" fill=\"{}\"/>",
      self.stroke
    ));
    self.bound(cx - r, cy - r);
    self.bound(cx + r, cy + r);
  }

  fn filled_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
    self.out.push_str(&format!(
      "<rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" \
       fill=\"{}\"/>",
      self.stroke
    ));
    self.bound(x, y);
    self.bound(x + w, y + h);
  }

  fn path(&mut self, d: &str, filled: bool, w: f64) {
    let style = if filled {
      format!("fill=\"{}\"", self.stroke)
    } else {
      format!(
        "fill=\"none\" stroke=\"{}\" stroke-width=\"{w}\"",
        self.stroke
      )
    };
    self.out.push_str(&format!(
      "<path d=\"{d}\" {style} stroke-linejoin=\"round\"/>"
    ));
  }
}

/// Draw a sharp or flat sign to the left of a note head centred at `(cx, cy)`.
/// `column` shifts the sign further left so stacked chord accidentals do not
/// collide.
fn draw_accidental(
  cv: &mut Canvas,
  cx: f64,
  cy: f64,
  accidental: i32,
  column: usize,
) {
  let x = cx - 12.0 - column as f64 * 9.0;
  match accidental {
    a if a > 0 => {
      // Sharp: two vertical strokes crossed by two slightly rising bars.
      cv.line(x - 1.5, cy - 6.0, x - 1.5, cy + 5.0, 1.1);
      cv.line(x + 1.8, cy - 5.0, x + 1.8, cy + 6.0, 1.1);
      cv.line(x - 4.0, cy - 1.6, x + 4.0, cy - 2.8, 1.6);
      cv.line(x - 4.0, cy + 2.4, x + 4.0, cy + 1.2, 1.6);
    }
    a if a < 0 => {
      // Flat: a vertical stem with a small bowl on its lower right.
      cv.line(x - 1.5, cy - 8.0, x - 1.5, cy + 4.5, 1.1);
      cv.path(
        &format!(
          "M {:.2} {:.2} Q {:.2} {:.2} {:.2} {:.2} Q {:.2} {:.2} {:.2} {:.2}",
          x - 1.5,
          cy - 1.5,
          x + 4.5,
          cy - 3.0,
          x + 3.0,
          cy + 2.0,
          x + 1.5,
          cy + 4.2,
          x - 1.5,
          cy + 4.5,
        ),
        false,
        1.1,
      );
    }
    _ => {}
  }
}

/// Draw ledger lines through a head at diatonic position `dn` centred at `cx`.
fn draw_ledger_lines(cv: &mut Canvas, cx: f64, dn: i32) {
  let half = 8.5;
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
  // Ledger lines and heads.
  for h in heads {
    draw_ledger_lines(cv, x, h.dn);
  }
  // Stagger accidentals of adjacent heads into separate columns so their
  // signs do not overlap (top head nearest the note, lower heads further left).
  let mut column = 0usize;
  let mut prev_dn: Option<i32> = None;
  for h in heads.iter().rev() {
    cv.note_head(x, dn_y(h.dn), dur.filled());
    if h.accidental != 0 {
      if prev_dn.is_some_and(|p| (p - h.dn).abs() <= 2) {
        column += 1;
      } else {
        column = 0;
      }
      draw_accidental(cv, x, dn_y(h.dn), h.accidental, column);
      prev_dn = Some(h.dn);
    }
  }
  if !dur.has_stem() {
    return;
  }
  // Stem direction from the middle head: notes on the upper half stem down.
  let top_dn = heads.iter().map(|h| h.dn).max().unwrap();
  let bottom_dn = heads.iter().map(|h| h.dn).min().unwrap();
  let mid = BOTTOM_LINE_DN + 4; // middle staff line, B4 = 34
  let stem_up = (top_dn + bottom_dn) as f64 / 2.0 < mid as f64;
  let (sx, y_near, y_far) = if stem_up {
    let sx = x + 5.7;
    (sx, dn_y(bottom_dn), dn_y(top_dn) - STEM_LEN)
  } else {
    let sx = x - 5.7;
    (sx, dn_y(top_dn), dn_y(bottom_dn) + STEM_LEN)
  };
  cv.line(sx, y_near, sx, y_far, 1.4);
  // Flags for eighth / sixteenth notes.
  let flags = dur.flags();
  for i in 0..flags {
    let fy = y_far + i as f64 * 7.0;
    if stem_up {
      cv.path(
        &format!(
          "M {sx:.2} {fy:.2} C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
          sx + 9.0,
          fy + 3.0,
          sx + 9.0,
          fy + 10.0,
          sx + 2.0,
          fy + 14.0,
        ),
        false,
        1.6,
      );
    } else {
      cv.path(
        &format!(
          "M {sx:.2} {fy:.2} C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}",
          sx + 9.0,
          fy - 3.0,
          sx + 9.0,
          fy - 10.0,
          sx + 2.0,
          fy - 14.0,
        ),
        false,
        1.6,
      );
    }
  }
}

/// Draw a rest centred at `x`.
fn draw_rest(cv: &mut Canvas, x: f64, dur: Dur) {
  let mid = dn_y(BOTTOM_LINE_DN + 4); // middle line, B4
  match dur {
    Dur::Whole => {
      // A filled block hanging below the second line from the top (D5).
      cv.filled_rect(x - 5.5, dn_y(BOTTOM_LINE_DN + 6) - 5.0, 11.0, 5.0);
    }
    Dur::Half => {
      // A filled block sitting on the middle line.
      cv.filled_rect(x - 5.5, mid, 11.0, 5.0);
    }
    _ => {
      // Quarter (and shorter): a stylized zig-zag rest.
      cv.path(
        &format!(
          "M {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} \
           Q {:.2} {:.2} {:.2} {:.2}",
          x - 3.5,
          mid - 10.0,
          x + 2.5,
          mid - 3.0,
          x - 3.0,
          mid + 2.5,
          x + 3.5,
          mid + 9.0,
          x - 3.5,
          mid + 4.0,
          x - 1.0,
          mid + 12.0,
        ),
        false,
        1.8,
      );
      if dur == Dur::Eighth || dur == Dur::Sixteenth {
        // A small dot to hint at the shorter value.
        cv.dot(x + 4.0, mid - 8.0, 1.8);
      }
    }
  }
}

/// Draw the treble (G) clef, its curl wrapped around the G4 line at `x`.
fn draw_treble_clef(cv: &mut Canvas, x: f64) {
  let g = dn_y(BOTTOM_LINE_DN + 2); // G4 line
  // A stylized G-clef: a tall spine, an upper hook, the big belly spiralling
  // in to the G line, and a tail with a terminal dot below the staff.
  let d = format!(
    "M {x0:.2} {tail:.2} \
     C {c1x:.2} {c1y:.2} {c2x:.2} {c2y:.2} {topx:.2} {topy:.2} \
     C {c3x:.2} {c3y:.2} {c4x:.2} {c4y:.2} {rightx:.2} {righty:.2} \
     C {c5x:.2} {c5y:.2} {c6x:.2} {c6y:.2} {gx:.2} {gy:.2} \
     C {c7x:.2} {c7y:.2} {c8x:.2} {c8y:.2} {leftx:.2} {lefty:.2} \
     C {c9x:.2} {c9y:.2} {c10x:.2} {c10y:.2} {cx:.2} {cy:.2}",
    x0 = x - 1.0,
    tail = g + 30.0,
    c1x = x - 10.0,
    c1y = g + 20.0,
    c2x = x - 9.0,
    c2y = g + 2.0,
    topx = x + 1.0,
    topy = g - 34.0,
    c3x = x + 9.0,
    c3y = g - 20.0,
    c4x = x + 9.0,
    c4y = g - 6.0,
    rightx = x + 8.0,
    righty = g + 2.0,
    c5x = x + 7.0,
    c5y = g + 12.0,
    c6x = x - 8.0,
    c6y = g + 12.0,
    gx = x - 8.0,
    gy = g,
    c7x = x - 8.0,
    c7y = g - 8.0,
    c8x = x + 3.0,
    c8y = g - 8.0,
    leftx = x + 3.0,
    lefty = g,
    c9x = x + 3.0,
    c9y = g + 5.0,
    c10x = x - 1.0,
    c10y = g + 5.0,
    cx = x - 1.5,
    cy = g + 3.0,
  );
  cv.path(&d, false, 2.0);
  // Terminal dot on the clef's tail.
  cv.dot(x - 4.0, g + 32.0, 2.6);
  cv.bound(x - 11.0, g - 36.0);
  cv.bound(x + 10.0, g + 36.0);
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
  let mut cv = Canvas::new(stroke);

  // Lay the glyphs out left to right, remembering where the staff must extend.
  let mut x = FIRST_NOTE_X;
  for g in &glyphs {
    match g {
      Glyph::Note { heads, dur } => {
        draw_note(&mut cv, x, heads, *dur);
        x += ADVANCE;
      }
      Glyph::Rest { dur } => {
        draw_rest(&mut cv, x, *dur);
        x += ADVANCE;
      }
      Glyph::Barline => {
        let bx = x - ADVANCE / 2.0;
        cv.line(bx, dn_y(BOTTOM_LINE_DN + 8), bx, dn_y(BOTTOM_LINE_DN), 1.3);
        x += ADVANCE * 0.5;
      }
    }
  }
  let staff_x_end = x - ADVANCE + FIRST_NOTE_X - STAFF_X0;
  let staff_x_end = staff_x_end.max(FIRST_NOTE_X + 8.0);

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

  draw_treble_clef(&mut cv, STAFF_X0 + 20.0);

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

  fn note(name: &str) -> Expr {
    Expr::FunctionCall {
      name: "MusicPitch".to_string(),
      args: vec![Expr::String(name.to_string())].into(),
    }
  }

  #[test]
  fn renders_a_single_pitch() {
    let svg = music_to_svg(&note("C4")).expect("pitch should render");
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("<ellipse")); // a note head
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
    assert_eq!(svg.matches("<ellipse").count(), 3);
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
      args: vec![Expr::List(vec![note("C4"), note("D4")].into())].into(),
    };
    let mut glyphs = Vec::new();
    assert!(collect(&measure, &mut glyphs));
    assert!(matches!(glyphs.last(), Some(Glyph::Barline)));
  }
}
