//! Extraction of music-notation glyphs from the bundled Leland font.
//!
//! [Leland](https://github.com/MuseScoreFonts/Leland) is MuseScore's SMuFL
//! (Standard Music Font Layout) font, licensed under the SIL Open Font License
//! 1.1 (see `resources/leland/LICENSE.txt`). Rather than embed the whole font
//! in every rendered SVG — or rely on a host `@font-face`/text-shaping pipeline
//! that Studio's `resvg` rasterizer and browsers implement differently — we
//! read each glyph's outline once with `ttf-parser` and emit it as a plain SVG
//! `<path>`. The resulting notation is self-contained and renders identically
//! everywhere.
//!
//! In a SMuFL font one staff space equals a quarter of the em square, so the
//! font-unit → user-unit scale for a staff whose line spacing is `gap` user
//! units is `gap * 4 / unitsPerEm` (see [`scale_for_gap`]).

use std::sync::OnceLock;
use ttf_parser::{Face, OutlineBuilder};

/// The bundled Leland font (SIL OFL 1.1, `resources/leland/`).
static FONT_BYTES: &[u8] = include_bytes!("../../resources/leland/Leland.otf");

/// The parsed font face, built lazily and cached for the process lifetime.
fn face() -> Option<&'static Face<'static>> {
  static FACE: OnceLock<Option<Face<'static>>> = OnceLock::new();
  FACE
    .get_or_init(|| Face::parse(FONT_BYTES, 0).ok())
    .as_ref()
}

/// SMuFL codepoints for the glyphs the staff renderer draws. The values are the
/// standard Private-Use-Area assignments shared by every SMuFL-compliant font.
pub mod glyph {
  /// Treble (G) clef; its origin sits on the G4 staff line.
  pub const G_CLEF: char = '\u{E050}';
  pub const NOTEHEAD_WHOLE: char = '\u{E0A2}';
  pub const NOTEHEAD_HALF: char = '\u{E0A3}';
  pub const NOTEHEAD_BLACK: char = '\u{E0A4}';
  pub const FLAT: char = '\u{E260}';
  pub const NATURAL: char = '\u{E261}';
  pub const SHARP: char = '\u{E262}';
  pub const REST_WHOLE: char = '\u{E4E3}';
  pub const REST_HALF: char = '\u{E4E4}';
  pub const REST_QUARTER: char = '\u{E4E5}';
  pub const REST_8TH: char = '\u{E4E6}';
  pub const REST_16TH: char = '\u{E4E7}';
  pub const FLAG_8TH_UP: char = '\u{E240}';
  pub const FLAG_8TH_DOWN: char = '\u{E241}';
  pub const FLAG_16TH_UP: char = '\u{E242}';
  pub const FLAG_16TH_DOWN: char = '\u{E243}';
}

/// The font-unit → user-unit scale for a staff with `gap` user units between
/// adjacent staff lines. One SMuFL staff space is a quarter of the em square.
pub fn scale_for_gap(gap: f64) -> f64 {
  let upm = face().map(|f| f.units_per_em()).unwrap_or(1000) as f64;
  gap * 4.0 / upm
}

/// A glyph's bounding box, in font units (font `y` points up).
#[derive(Clone, Copy)]
pub struct FontBBox {
  pub x_min: f64,
  pub y_min: f64,
  pub x_max: f64,
  pub y_max: f64,
}

impl FontBBox {
  /// Half the box width, scaled to user units.
  pub fn half_width(&self, scale: f64) -> f64 {
    (self.x_max - self.x_min) / 2.0 * scale
  }
  /// The horizontal centre of the box, scaled to user units.
  pub fn center_x(&self, scale: f64) -> f64 {
    (self.x_min + self.x_max) / 2.0 * scale
  }
}

/// The bounding box of a glyph, or `None` if the font lacks it or it is empty.
pub fn glyph_bbox(ch: char) -> Option<FontBBox> {
  let face = face()?;
  let gid = face.glyph_index(ch)?;
  let r = face.glyph_bounding_box(gid)?;
  Some(FontBBox {
    x_min: r.x_min as f64,
    y_min: r.y_min as f64,
    x_max: r.x_max as f64,
    y_max: r.y_max as f64,
  })
}

/// Builds SVG path data from a glyph outline, mapping each font-unit point
/// `(x, y)` to user space as `(ox + x·scale, oy − y·scale)` — the `y` flip
/// converts the font's upward axis to SVG's downward one.
struct SvgPathBuilder {
  scale: f64,
  ox: f64,
  oy: f64,
  d: String,
}

impl SvgPathBuilder {
  fn map(&self, x: f32, y: f32) -> (f64, f64) {
    (
      self.ox + x as f64 * self.scale,
      self.oy - y as f64 * self.scale,
    )
  }
}

impl OutlineBuilder for SvgPathBuilder {
  fn move_to(&mut self, x: f32, y: f32) {
    let (px, py) = self.map(x, y);
    self.d.push_str(&format!("M{px:.2} {py:.2}"));
  }
  fn line_to(&mut self, x: f32, y: f32) {
    let (px, py) = self.map(x, y);
    self.d.push_str(&format!("L{px:.2} {py:.2}"));
  }
  fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
    let (c1x, c1y) = self.map(x1, y1);
    let (px, py) = self.map(x, y);
    self
      .d
      .push_str(&format!("Q{c1x:.2} {c1y:.2} {px:.2} {py:.2}"));
  }
  fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
    let (c1x, c1y) = self.map(x1, y1);
    let (c2x, c2y) = self.map(x2, y2);
    let (px, py) = self.map(x, y);
    self.d.push_str(&format!(
      "C{c1x:.2} {c1y:.2} {c2x:.2} {c2y:.2} {px:.2} {py:.2}"
    ));
  }
  fn close(&mut self) {
    self.d.push('Z');
  }
}

/// SVG path data for a glyph whose origin `(0, 0)` maps to `(ox, oy)` at the
/// given font-unit → user-unit `scale`. Returns `None` when the glyph is
/// missing or has no outline.
pub fn glyph_path_d(ch: char, scale: f64, ox: f64, oy: f64) -> Option<String> {
  let face = face()?;
  let gid = face.glyph_index(ch)?;
  let mut b = SvgPathBuilder {
    scale,
    ox,
    oy,
    d: String::new(),
  };
  face.outline_glyph(gid, &mut b)?;
  if b.d.is_empty() { None } else { Some(b.d) }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn font_parses_and_reports_a_1000_unit_em() {
    assert_eq!(face().unwrap().units_per_em(), 1000);
    // GAP = 12 → one staff space is 250 font units → 0.048 user units each.
    assert!((scale_for_gap(12.0) - 0.048).abs() < 1e-9);
  }

  #[test]
  fn every_used_glyph_has_an_outline() {
    for ch in [
      glyph::G_CLEF,
      glyph::NOTEHEAD_WHOLE,
      glyph::NOTEHEAD_HALF,
      glyph::NOTEHEAD_BLACK,
      glyph::FLAT,
      glyph::NATURAL,
      glyph::SHARP,
      glyph::REST_WHOLE,
      glyph::REST_HALF,
      glyph::REST_QUARTER,
      glyph::REST_8TH,
      glyph::REST_16TH,
      glyph::FLAG_8TH_UP,
      glyph::FLAG_8TH_DOWN,
      glyph::FLAG_16TH_UP,
      glyph::FLAG_16TH_DOWN,
    ] {
      assert!(
        glyph_bbox(ch).is_some(),
        "missing bbox for U+{:04X}",
        ch as u32
      );
      assert!(
        glyph_path_d(ch, 0.048, 0.0, 0.0).is_some(),
        "missing outline for U+{:04X}",
        ch as u32
      );
    }
  }

  #[test]
  fn black_notehead_path_is_transformed_to_user_space() {
    let d = glyph_path_d(glyph::NOTEHEAD_BLACK, 0.048, 100.0, 50.0).unwrap();
    assert!(d.starts_with('M'));
    // With a ~1.3 staff-space-wide head the extent stays within a few user
    // units of the placement point — i.e. the scale was actually applied.
    let bb = glyph_bbox(glyph::NOTEHEAD_BLACK).unwrap();
    assert!(bb.half_width(0.048) > 3.0 && bb.half_width(0.048) < 10.0);
  }
}
