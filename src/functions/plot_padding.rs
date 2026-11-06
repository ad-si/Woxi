//! The automatic `ImagePadding` of a 2D plot or an axed `Graphics`: the
//! room the Wolfram Language front end leaves around the plotting area for
//! the frame, the axes and their labels.
//!
//! The front end measures every decoration as a box in Arial metrics (the
//! tick labels at 10 pt, `FrameLabel`/`AxesLabel` at 10 pt, `PlotLabel` at
//! 12 pt; a box is one line height, 2288/2048 em, tall) and pads each side
//! of the plotting area by however far the boxes reach past it:
//!
//! - frame tick labels sit 3 pt outside the frame (the y labels right
//!   aligned, the x labels hanging below), centred on their ticks;
//! - axis tick labels sit 3 pt beside their axis the same way, so they
//!   only need room where the axis runs close to the plot's edge;
//! - a `FrameLabel` sits 6 pt beyond the tick labels of its edge;
//! - the x `AxesLabel` starts 8 pt past the right end of the x axis,
//!   centred on it; the y `AxesLabel` sits 6 pt above the y axis, centred
//!   on it;
//! - every side keeps at least 0.5 pt once a frame or an axis is drawn,
//!   and the right and bottom sides one more point;
//! - a `PlotLabel` adds its own band above all of that.
//!
//! The plotting area is what the padding leaves of the image width, its
//! height follows from the aspect ratio, and the image height is the sum
//! rounded up to a whole point. Tick labels near the ends of an axis reach
//! further past the edge the narrower the plot is, so the padding is solved
//! as a fixed point.

/// Tick label font size, in points.
pub(crate) const TICK_FONT: f64 = 10.0;
/// `FrameLabel` / `AxesLabel` font size, in points.
pub(crate) const LABEL_FONT: f64 = 10.0;
/// `PlotLabel` font size, in points.
pub(crate) const TITLE_FONT: f64 = 12.0;
/// Gap between a tick label and its frame edge or axis.
pub(crate) const TICK_LABEL_GAP: f64 = 3.0;
/// Gap between a `FrameLabel` and the tick labels of its edge.
pub(crate) const FRAME_LABEL_GAP: f64 = 6.0;
/// Gap between the right end of the x axis and the x `AxesLabel`.
pub(crate) const AXES_LABEL_GAP_X: f64 = 8.0;
/// Gap between the top end of the y axis and the y `AxesLabel`.
pub(crate) const AXES_LABEL_GAP_Y: f64 = 6.0;
/// Gap between the right edge of the plotting area and `PlotLabels`.
pub(crate) const PLOT_LABELS_GAP: f64 = 23.3;
/// Room `PlotLabels` keep past their text.
const PLOT_LABELS_TAIL: f64 = 10.6;
/// The padding every side keeps once a frame or an axis is drawn.
const MIN_PADDING: f64 = 0.5;
/// The extra point the right and bottom sides keep.
const FAR_SIDE_EXTRA: f64 = 1.0;

/// Arial's line height (ascent + descent), in em.
const LINE_HEIGHT_EM: f64 = 2288.0 / 2048.0;
/// Arial's ascent, in em.
const ASCENT_EM: f64 = 1854.0 / 2048.0;
/// Pitch of the lines of a multi-line label, in line heights.
const LINE_PITCH: f64 = 1.25;

/// Height of one line of text at `size` points.
pub(crate) fn line_height(size: f64) -> f64 {
  size * LINE_HEIGHT_EM
}

/// Distance from the top of a line box to its baseline at `size` points.
pub(crate) fn ascent(size: f64) -> f64 {
  size * ASCENT_EM
}

/// Distance from the baseline to the bottom of a line box.
pub(crate) fn descent(size: f64) -> f64 {
  line_height(size) - ascent(size)
}

/// Height of a block of `lines` lines at `size` points.
pub(crate) fn block_height(lines: usize, size: f64) -> f64 {
  if lines == 0 {
    0.0
  } else {
    line_height(size) * (1.0 + LINE_PITCH * (lines - 1) as f64)
  }
}

/// Pitch between the baselines of a multi-line label at `size` points.
pub(crate) fn line_pitch(size: f64) -> f64 {
  line_height(size) * LINE_PITCH
}

/// Arial advance widths of the printable ASCII characters, in 1/2048 em.
const ARIAL_ASCII: [u16; 95] = [
  569, 569, 727, 1139, 1139, 1821, 1366, 391, 682, 682, 797, 1196, 569, 682,
  569, 569, 1139, 1139, 1139, 1139, 1139, 1139, 1139, 1139, 1139, 1139, 569,
  569, 1196, 1196, 1196, 1139, 2079, 1366, 1366, 1479, 1479, 1366, 1251, 1593,
  1479, 569, 1024, 1366, 1139, 1706, 1479, 1593, 1366, 1593, 1479, 1366, 1251,
  1479, 1366, 1933, 1366, 1366, 1251, 569, 569, 569, 961, 1139, 682, 1139,
  1139, 1024, 1139, 1139, 569, 1139, 1139, 455, 455, 1024, 455, 1706, 1139,
  1139, 1139, 1139, 682, 1024, 569, 1139, 1024, 1479, 1024, 1024, 1024, 684,
  532, 684, 1196,
];

/// Width of a numeric tick label's minus sign, in em (a little wider than
/// Arial's U+2212).
const TICK_MINUS_EM: f64 = 0.63027;
/// The thin space Wolfram sets between digit blocks of a long integer part
/// (`10 000`), in em.
const DIGIT_BLOCK_GAP_EM: f64 = 285.0 / 2048.0;

/// Advance widths of the printable ASCII characters in Atkinson Hyperlegible
/// Next (weight 400), the face `Export` embeds for every SVG text, in
/// 1/1000 em.
const ATKINSON_ASCII: [u16; 95] = [
  280, 286, 301, 651, 592, 919, 731, 160, 299, 299, 485, 606, 210, 362, 210,
  387, 648, 407, 548, 576, 614, 588, 598, 510, 615, 595, 210, 210, 553, 628,
  553, 480, 778, 626, 613, 645, 669, 569, 552, 694, 694, 392, 512, 631, 541,
  824, 693, 714, 592, 724, 614, 592, 558, 690, 593, 856, 623, 575, 607, 317,
  387, 344, 552, 372, 243, 525, 566, 485, 564, 538, 314, 558, 550, 252, 245,
  492, 262, 845, 550, 548, 566, 574, 349, 471, 326, 540, 461, 684, 481, 449,
  460, 344, 237, 344, 555,
];

fn char_width_em(c: char) -> f64 {
  match c {
    ' '..='~' => ARIAL_ASCII[c as usize - 0x20] as f64 / 2048.0,
    '\u{2212}' => 1196.0 / 2048.0,
    '\u{00D7}' => 1196.0 / 2048.0,
    // Narrow combining marks and zero-width characters.
    '\u{200B}'..='\u{200D}' | '\u{2061}'..='\u{2063}' => 0.0,
    _ => 1139.0 / 2048.0,
  }
}

/// Left side bearings of the printable ASCII characters in Atkinson
/// Hyperlegible Next (weight 400), in 1/1000 em: the blank the glyph leaves
/// before its ink.
const ATKINSON_ASCII_LSB: [i16; 95] = [
  0, 78, 40, 36, 38, 36, 59, 40, 55, 35, 70, 55, 40, 35, 40, 20, 60, 35, 29,
  44, 38, 59, 65, 15, 60, 57, 40, 40, 40, 66, 40, 30, 44, 10, 78, 44, 78, 78,
  78, 44, 78, 55, 15, 78, 78, 78, 78, 44, 78, 44, 78, 38, 10, 76, 12, 18, 10,
  0, 44, 78, 20, 60, 53, -2, 40, 31, 72, 36, 36, 36, 15, 36, 72, 25, -61, 72,
  68, 72, 72, 36, 72, 36, 72, 31, 20, 70, 10, 15, 10, 10, 40, 38, 78, 34, 41,
];

fn atkinson_lsb_em(c: char) -> f64 {
  match c {
    ' '..='~' => ATKINSON_ASCII_LSB[c as usize - 0x20] as f64 / 1000.0,
    '\u{2212}' | '\u{00D7}' => 0.055,
    _ => 0.0,
  }
}

/// Width of `c` in the face the exported SVG actually draws with.
fn atkinson_width_em(c: char) -> f64 {
  match c {
    ' '..='~' => ATKINSON_ASCII[c as usize - 0x20] as f64 / 1000.0,
    '\u{2212}' => 0.606,
    '\u{00D7}' => 0.532,
    _ => char_width_em(c),
  }
}

/// The visible text of SVG label markup: tags dropped, entities decoded
/// to one character each.
fn visible_chars(markup: &str) -> Vec<char> {
  let mut out = Vec::new();
  let mut chars = markup.chars().peekable();
  while let Some(c) = chars.next() {
    match c {
      '<' => {
        for d in chars.by_ref() {
          if d == '>' {
            break;
          }
        }
      }
      '&' => {
        let mut entity = String::new();
        for d in chars.by_ref() {
          if d == ';' {
            break;
          }
          entity.push(d);
        }
        out.push(match entity.as_str() {
          "amp" => '&',
          "lt" => '<',
          "gt" => '>',
          "quot" => '"',
          "apos" => '\'',
          "#215" | "times" => '\u{00D7}',
          "#8722" | "minus" => '\u{2212}',
          e => e
            .strip_prefix("#x")
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .or_else(|| e.strip_prefix('#').and_then(|d| d.parse().ok()))
            .and_then(char::from_u32)
            .unwrap_or('?'),
        });
      }
      _ => out.push(c),
    }
  }
  out
}

/// Width of a line of plain text (or SVG label markup) at `size` points.
/// A multi-line text measures its widest line.
pub(crate) fn text_width(markup: &str, size: f64) -> f64 {
  markup
    .split('\n')
    .map(|line| {
      visible_chars(line)
        .into_iter()
        .map(char_width_em)
        .sum::<f64>()
    })
    .fold(0.0, f64::max)
    * size
}

/// Finite LabelingSize uses the front end's Text-style (14 pt) measure.
/// Keep at least two characters before the three-dot truncation marker,
/// as TruncateStringToWidth does even for very small positive widths.
pub(crate) fn truncate_label(markup: &str, width: f64) -> String {
  if !width.is_finite() || text_width(markup, 14.0) <= width {
    return markup.to_string();
  }
  if width <= 0.0 {
    return String::new();
  }
  let chars = visible_chars(markup);
  let mut count = chars.len().min(2);
  while count < chars.len() {
    let candidate =
      format!("{}...", chars[..=count].iter().collect::<String>());
    if text_width(&candidate, 14.0) > width {
      break;
    }
    count += 1;
  }
  crate::functions::graphics::svg_escape(&format!(
    "{}...",
    chars[..count].iter().collect::<String>()
  ))
}

/// Width of a tick label at `size` points. A number is measured the way
/// the front end typesets it: a leading minus a little wider than a
/// hyphen, and an integer part of five or more digits split into blocks
/// of three by thin spaces.
pub(crate) fn tick_label_width(markup: &str, size: f64) -> f64 {
  let chars = visible_chars(markup);
  let numeric = !chars.is_empty()
    && chars
      .iter()
      .all(|c| c.is_ascii_digit() || matches!(c, '-' | '.' | '\u{2212}'))
    && chars.iter().any(char::is_ascii_digit);
  if !numeric {
    return text_width(markup, size);
  }
  let mut em = 0.0;
  let mut int_digits = 0usize;
  let mut in_int = true;
  for &c in &chars {
    match c {
      '-' | '\u{2212}' => em += TICK_MINUS_EM,
      '.' => {
        in_int = false;
        em += char_width_em('.');
      }
      d => {
        if in_int {
          int_digits += 1;
        }
        em += char_width_em(d);
      }
    }
  }
  if int_digits >= 5 {
    em += ((int_digits - 1) / 3) as f64 * DIGIT_BLOCK_GAP_EM;
  }
  em * size
}

/// The horizontal ink of a one-line text in the face exported graphics draw
/// with (Atkinson Hyperlegible Next): from the ink of its first glyph to the
/// end of its advance, as `(lead, width)` at `size`, `lead` being the blank
/// before the first glyph's ink.
fn exported_text_extent(markup: &str, size: f64) -> (f64, f64) {
  let chars = visible_chars(markup);
  let lead = chars.first().map_or(0.0, |&c| atkinson_lsb_em(c));
  let width = chars.iter().copied().map(atkinson_width_em).sum::<f64>();
  (lead * size, width * size)
}

/// Shift every label of a flat plot SVG that would run past the left or
/// right edge of its `viewBox` back inside it.
///
/// The automatic padding is laid out in the Arial metrics the Wolfram front
/// end uses, so a plot's geometry matches wolframscript's; but the text is
/// drawn in Atkinson Hyperlegible, whose digits are wider (`0` is 0.648 em
/// against Arial's 0.556 em). A y tick label that exactly fills its room in
/// Arial would lose its first digit at the image edge. Only `<text>` outside
/// any transformed group is considered, and only its single `x`.
pub(crate) fn keep_text_in_view(svg: &str) -> std::borrow::Cow<'_, str> {
  use std::borrow::Cow;
  let Some(width) = svg_canvas_width(svg) else {
    return Cow::Borrowed(svg);
  };
  // Every unrotated top-level label: where its `x` value sits, the value,
  // its anchor, and how far it must move to stay inside.
  struct Label {
    at: usize,
    len: usize,
    x: f64,
    anchor: u8,
    shift: f64,
  }
  let mut labels: Vec<Label> = Vec::new();
  // One entry per open `<g>` / `<svg>`: whether it sets up coordinates of
  // its own.
  let mut groups: Vec<bool> = Vec::new();
  let mut svg_depth = 0usize;
  let mut pos = 0usize;
  while let Some(rel) = svg[pos..].find('<') {
    let start = pos + rel;
    let Some(tag_end) = svg[start..].find('>').map(|e| start + e + 1) else {
      break;
    };
    let tag = &svg[start..tag_end];
    pos = tag_end;
    let is_group = tag.starts_with("<g")
      && tag
        .as_bytes()
        .get(2)
        .is_none_or(|b| !b.is_ascii_alphanumeric());
    if is_group {
      if !tag.ends_with("/>") {
        groups.push(tag.contains("transform="));
      }
      continue;
    }
    // A nested `<svg>` (a cell of a list of graphics) has its own
    // coordinates, like a transformed group; the outermost one is the
    // canvas.
    let is_svg = tag.starts_with("<svg")
      && tag
        .as_bytes()
        .get(4)
        .is_none_or(|b| !b.is_ascii_alphanumeric());
    if is_svg {
      if !tag.ends_with("/>") {
        groups.push(svg_depth > 0);
        svg_depth += 1;
      }
      continue;
    }
    if tag.starts_with("</g") || tag.starts_with("</svg") {
      groups.pop();
      continue;
    }
    if !tag.starts_with("<text") || tag.ends_with("/>") {
      continue;
    }
    let Some(close) = svg[tag_end..].find("</text>").map(|c| tag_end + c)
    else {
      continue;
    };
    pos = close;
    let content = svg[tag_end..close].trim();
    if groups.iter().any(|&t| t)
      || tag.contains("transform=")
      || content.contains('<')
      || content.contains('\n')
    {
      continue;
    }
    let Some((x_at, len, x)) = attr_number(tag, "x") else {
      continue;
    };
    let size = attr_number(tag, "font-size").map_or(16.0, |(_, _, v)| v);
    let (lead, w) = exported_text_extent(content, size);
    if w <= 0.0 || w - lead > width {
      continue;
    }
    let (anchor, left) = if tag.contains("text-anchor=\"end\"") {
      (2, x - w)
    } else if tag.contains("text-anchor=\"middle\"") {
      (1, x - w / 2.0)
    } else {
      (0, x)
    };
    let ink_left = left + lead;
    let shift = if ink_left < 0.0 {
      -ink_left
    } else if left + w > width {
      (width - (left + w)).max(-ink_left)
    } else {
      0.0
    };
    labels.push(Label {
      at: start + x_at,
      len,
      x,
      anchor,
      shift,
    });
  }
  if labels.iter().all(|l| l.shift == 0.0) {
    return Cow::Borrowed(svg);
  }
  // A column of right-aligned labels (the y tick labels) moves as one, so
  // it stays aligned.
  let column_shift = |l: &Label| {
    if l.anchor != 2 || l.shift < 0.0 {
      return l.shift;
    }
    labels
      .iter()
      .filter(|o| o.anchor == 2 && (o.x - l.x).abs() < 1e-6)
      .map(|o| o.shift)
      .fold(0.0, f64::max)
  };
  let mut out = String::with_capacity(svg.len());
  let mut copied = 0usize;
  for l in &labels {
    let shift = column_shift(l);
    if shift == 0.0 {
      continue;
    }
    out.push_str(&svg[copied..l.at]);
    out.push_str(&format!("{:.1}", l.x + shift));
    copied = l.at + l.len;
  }
  out.push_str(&svg[copied..]);
  Cow::Owned(out)
}

/// The width of the outermost `<svg>`'s `viewBox` (else its `width`).
fn svg_canvas_width(svg: &str) -> Option<f64> {
  let open = svg.find("<svg")?;
  let tag = &svg[open..open + svg[open..].find('>')?];
  if let Some(i) = tag.find("viewBox=\"") {
    let v = &tag[i + 9..];
    let v = &v[..v.find('"')?];
    return v
      .split([' ', ','])
      .filter(|t| !t.is_empty())
      .nth(2)?
      .parse()
      .ok();
  }
  attr_number(tag, "width").map(|(_, _, w)| w)
}

/// A numeric attribute of a tag: the byte offset and length of its value
/// within `tag`, and the value.
fn attr_number(tag: &str, name: &str) -> Option<(usize, usize, f64)> {
  let pat = format!(" {name}=\"");
  let at = tag.find(&pat)? + pat.len();
  let len = tag[at..].find('"')?;
  let value = tag[at..at + len].trim().parse().ok()?;
  Some((at, len, value))
}

/// A tick label: where its tick falls along the axis (0 at the low end,
/// 1 at the high end), and its size in points.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LabelBox {
  pub frac: f64,
  pub width: f64,
  pub height: f64,
  /// An axis label at the axes crossing, drawn only while the other axis
  /// stops short of it (see `plot_axes::crossing_label_blocked`).
  pub at_crossing: bool,
}

/// Everything around the plotting area that the padding has to hold.
#[derive(Clone, Debug, Default)]
pub(crate) struct Decorations {
  /// Whether a frame (any edge of it) is drawn.
  pub frame: bool,
  /// Which axes are drawn (unframed plots).
  pub axes: (bool, bool),
  /// Where the axes cross, as fractions of the plot width (the y axis)
  /// and height (the x axis, from the bottom).
  pub origin: (f64, f64),
  /// The tick labels of the frame's bottom/left/top/right edges.
  pub frame_ticks: [Vec<LabelBox>; 4],
  /// How far the tick marks of each frame edge reach out of the frame, as
  /// a fraction of the plot width; the labels sit beyond them.
  pub frame_tick_out: [f64; 4],
  /// The tick labels of the x and y axes.
  pub axis_ticks: (Vec<LabelBox>, Vec<LabelBox>),
  /// The heights of the bottom/left/top/right `FrameLabel`s.
  pub frame_labels: [Option<f64>; 4],
  /// Width and height of the x and y `AxesLabel`.
  pub axes_labels: (Option<(f64, f64)>, Option<(f64, f64)>),
  /// Height of the `PlotLabel` band.
  pub title: f64,
  /// Width of the widest of the `PlotLabels` beside the curve ends.
  pub curve_labels: Option<f64>,
  /// Room the labels placed before / after their curves take left and
  /// right of the plotting area.
  pub side_labels: (f64, f64),
  /// The widest tick label of each `MultiaxisArrangement` axis standing
  /// outside the frame, left and right, in order outward.
  pub extra_axes: [Vec<f64>; 2],
}

/// How far a `MultiaxisArrangement` axis outside the frame stands beyond
/// the widest tick label of the axis inside it, as in wolframscript's SVGs.
pub(crate) const MULTIAXIS_SPACING: f64 = 23.0;

/// The padding around the plotting area, in points.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Padding {
  pub left: f64,
  pub right: f64,
  pub bottom: f64,
  /// Above the plotting area, below the `PlotLabel` band.
  pub top: f64,
  /// The `PlotLabel` band above `top`.
  pub title: f64,
}

impl Padding {
  /// Everything above the plotting area.
  pub(crate) fn above(&self) -> f64 {
    self.top + self.title
  }
}

impl Decorations {
  /// Room the tick marks and labels of frame edge `edge` take outside it
  /// (with the labels' gap) on a plot `w` wide, 0 without either.
  pub(crate) fn frame_tick_room(&self, edge: usize, w: f64) -> f64 {
    let out = self.frame_tick_out[edge] * w;
    let ticks = &self.frame_ticks[edge];
    if ticks.is_empty() {
      return out;
    }
    let size = |b: &LabelBox| {
      if edge.is_multiple_of(2) {
        b.height
      } else {
        b.width
      }
    };
    out + TICK_LABEL_GAP + ticks.iter().map(size).fold(0.0, f64::max)
  }

  /// How far each extra axis of the left (`side` 0) or right (1) frame
  /// edge stands from that edge, in points, in order outward.
  pub(crate) fn extra_axis_offsets(&self, side: usize) -> Vec<f64> {
    let edge = if side == 0 { 1 } else { 3 };
    let mut inner = self.frame_ticks[edge]
      .iter()
      .map(|b| b.width)
      .fold(0.0, f64::max);
    let mut offset = 0.0;
    self.extra_axes[side]
      .iter()
      .map(|&width| {
        offset += inner + MULTIAXIS_SPACING;
        inner = width;
        offset
      })
      .collect()
  }

  /// The extents past the left/right/bottom/top edges of a `w` × `h`
  /// plotting area.
  fn extents(&self, w: f64, h: f64) -> [f64; 4] {
    let mut e = [0.0f64; 4];
    let mut reach = |side: usize, v: f64| e[side] = e[side].max(v);
    const L: usize = 0;
    const R: usize = 1;
    const B: usize = 2;
    const T: usize = 3;
    // Horizontal labels centred on ticks along a horizontal edge, and
    // vertically centred ones along a vertical edge, overhang the ends.
    let along_x = |reach: &mut dyn FnMut(usize, f64), b: &LabelBox| {
      reach(L, b.width / 2.0 - b.frac * w);
      reach(R, b.width / 2.0 - (1.0 - b.frac) * w);
    };
    let along_y = |reach: &mut dyn FnMut(usize, f64), b: &LabelBox| {
      reach(B, b.height / 2.0 - b.frac * h);
      reach(T, b.height / 2.0 - (1.0 - b.frac) * h);
    };
    if self.frame {
      // Edge order bottom, left, top, right; sides L R B T.
      let side_of = [B, L, T, R];
      for edge in 0..4 {
        let mut room = self.frame_tick_room(edge, w);
        for b in &self.frame_ticks[edge] {
          if edge % 2 == 0 {
            along_x(&mut reach, b);
          } else {
            along_y(&mut reach, b);
          }
        }
        if let Some(lh) = self.frame_labels[edge] {
          room += FRAME_LABEL_GAP + lh;
        }
        reach(side_of[edge], room);
      }
      // Each axis outside the frame takes its own labels' room beyond it.
      for (side, reach_side) in [(0, L), (1, R)] {
        if let (Some(&offset), Some(&width)) = (
          self.extra_axis_offsets(side).last(),
          self.extra_axes[side].last(),
        ) {
          reach(reach_side, offset + TICK_LABEL_GAP + width);
        }
      }
    }
    if let Some(lw) = self.curve_labels {
      reach(R, PLOT_LABELS_GAP + lw + PLOT_LABELS_TAIL);
    }
    reach(L, self.side_labels.0);
    reach(R, self.side_labels.1);
    let (ox, oy) = (self.origin.0 * w, self.origin.1 * h);
    let blocked = crate::functions::plot_axes::crossing_label_blocked;
    if self.axes.0 {
      // The y axis runs `oy` below the crossing.
      for b in self
        .axis_ticks
        .0
        .iter()
        .filter(|b| !(b.at_crossing && self.axes.1 && blocked(oy)))
      {
        along_x(&mut reach, b);
        reach(B, TICK_LABEL_GAP + b.height - oy);
      }
      if let Some((lw, lh)) = self.axes_labels.0 {
        reach(R, AXES_LABEL_GAP_X + lw);
        reach(T, lh / 2.0 - (h - oy));
        reach(B, lh / 2.0 - oy);
      }
    }
    if self.axes.1 {
      // The x axis runs `ox` left of the crossing.
      for b in self
        .axis_ticks
        .1
        .iter()
        .filter(|b| !(b.at_crossing && self.axes.0 && blocked(ox)))
      {
        along_y(&mut reach, b);
        reach(L, TICK_LABEL_GAP + b.width - ox);
      }
      if let Some((lw, lh)) = self.axes_labels.1 {
        reach(T, AXES_LABEL_GAP_Y + lh);
        reach(L, lw / 2.0 - ox);
        reach(R, lw / 2.0 - (w - ox));
      }
    }
    e
  }

  /// Solve the padding for an image `image_w` points wide whose plotting
  /// area is `height_of(w)` tall when it is `w` wide.
  pub(crate) fn solve(
    &self,
    image_w: f64,
    height_of: impl Fn(f64) -> f64,
  ) -> Padding {
    let decorated = self.frame || self.axes.0 || self.axes.1;
    let (min, extra) = if decorated {
      (MIN_PADDING, FAR_SIDE_EXTRA)
    } else {
      (0.0, 0.0)
    };
    let mut pad = Padding {
      left: min,
      right: min + extra,
      bottom: min + extra,
      top: min,
      title: self.title,
    };
    for _ in 0..12 {
      let w = (image_w - pad.left - pad.right).max(1.0);
      let h = height_of(w).max(1.0);
      let e = self.extents(w, h);
      let next = Padding {
        left: e[0].max(min),
        right: e[1].max(min) + extra,
        bottom: e[2].max(min) + extra,
        top: e[3].max(min),
        title: self.title,
      };
      let converged = (next.left - pad.left).abs() < 1e-6
        && (next.right - pad.right).abs() < 1e-6
        && (next.bottom - pad.bottom).abs() < 1e-6
        && (next.top - pad.top).abs() < 1e-6;
      pad = next;
      if converged {
        break;
      }
    }
    pad
  }
}

/// Where [`crate::functions::plot::plot_labels_svg`] places the labels
/// around the plotting area.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LabelLayout {
  /// Room the tick labels of the bottom/left/top/right frame edges take
  /// outside the frame, in points; a `FrameLabel` sits 6 pt beyond it.
  pub tick_room: [f64; 4],
  /// Where the `PlotLabel` band starts, in render units from the top.
  pub title_top: f64,
}

impl Decorations {
  /// The [`LabelLayout`] tick rooms of these decorations.
  /// `w` is the plot width, in points.
  pub(crate) fn tick_rooms(&self, w: f64) -> [f64; 4] {
    if self.frame {
      [0, 1, 2, 3].map(|edge| self.frame_tick_room(edge, w))
    } else {
      [0.0; 4]
    }
  }
}

/// Lines in a label string (`"a\nb"` has two).
fn label_lines(text: &str) -> usize {
  text.split('\n').count()
}

/// Width of a label string (box markup allowed) at `size` points.
fn label_width(text: &str, size: f64) -> f64 {
  text
    .split('\n')
    .map(|line| {
      text_width(
        &crate::functions::graphics::box_string_to_svg_scaled(line, 1.0),
        size,
      )
    })
    .fold(0.0, f64::max)
}

/// The tick font of a plot, in points.
pub(crate) fn plot_tick_font(
  opts: &crate::functions::plot::PlotOptions,
) -> f64 {
  opts
    .tick_label_style
    .and_then(|s| s.font_size)
    // `LabelStyle` sizes the frame's tick labels too (the origin axes draw
    // theirs at the fixed `AXIS_TICK_FONT`).
    .or(
      opts
        .label_style
        .filter(|_| opts.frame)
        .and_then(|s| s.font_size),
    )
    .unwrap_or(TICK_FONT)
}

/// The `FrameLabel` / `AxesLabel` font of a plot, in points.
pub(crate) fn plot_label_font(
  opts: &crate::functions::plot::PlotOptions,
) -> f64 {
  opts
    .label_style
    .and_then(|s| s.font_size)
    .unwrap_or(LABEL_FONT)
}

/// The `PlotLabel` font of a plot, in points.
pub(crate) fn plot_title_font(
  opts: &crate::functions::plot::PlotOptions,
) -> f64 {
  opts
    .plot_label
    .as_ref()
    .and_then(|sl| sl.font_size)
    .or(opts.label_style.and_then(|s| s.font_size))
    .unwrap_or(TITLE_FONT)
}

/// Height of a plot's `PlotLabel` band, 0 without one.
pub(crate) fn plot_title_band(
  opts: &crate::functions::plot::PlotOptions,
) -> f64 {
  match &opts.plot_label {
    Some(sl) if !sl.text.is_empty() => {
      let lines = 1 + sl.extra_line_count().max(label_lines(&sl.text) - 1);
      block_height(lines, plot_title_font(opts))
    }
    _ => 0.0,
  }
}

/// What a 2D plot draws around its plotting area, for [`Decorations::solve`].
///
/// `x_ticks`/`y_ticks` are the labelled ticks of the bottom/left edge (or
/// the axes), `x`/`y` the axis scales, `origin` where the axes cross and
/// `bottom_graphic_h` the height of a picture `FrameLabel` under the frame.
pub(crate) fn plot_decorations(
  opts: &crate::functions::plot::PlotOptions,
  x_ticks: &crate::functions::plot_axes::AxisTicks,
  y_ticks: &crate::functions::plot_axes::AxisTicks,
  (x, y): (
    crate::functions::plot_axes::AxisScale,
    crate::functions::plot_axes::AxisScale,
  ),
  origin: (f64, f64),
  bottom_graphic_h: f64,
) -> Decorations {
  let tick_font = plot_tick_font(opts);
  let label_font = plot_label_font(opts);
  let show_ticks = opts.ticks_visible();
  let boxes = |ticks: &[(f64, String)],
               scale: crate::functions::plot_axes::AxisScale,
               skip: Option<f64>| {
    ticks
      .iter()
      .filter(|(_, l)| !l.is_empty())
      .map(|(v, l)| LabelBox {
        at_crossing: skip.is_some_and(|o| {
          (v - o).abs() <= (scale.max - scale.min).abs() * 1e-9
        }),
        frac: scale.frac(*v),
        width: l
          .split('\n')
          .map(|line| tick_label_width(line, tick_font))
          .fold(0.0, f64::max),
        height: block_height(label_lines(l), tick_font),
      })
      .filter(|b| b.frac.is_finite())
      .collect::<Vec<_>>()
  };
  let frame = opts.frame;
  let mut d = Decorations {
    frame,
    axes: if frame { (false, false) } else { opts.axes },
    origin: (
      x.frac(origin.0).clamp(0.0, 1.0),
      y.frac(origin.1).clamp(0.0, 1.0),
    ),
    title: plot_title_band(opts),
    ..Default::default()
  };
  if !d.origin.0.is_finite() {
    d.origin.0 = 0.0;
  }
  if !d.origin.1.is_finite() {
    d.origin.1 = 0.0;
  }
  let edges = opts.frame_edges;
  if frame {
    if show_ticks {
      use crate::functions::plot::FrameEdgeTicks;
      let present = [edges.bottom, edges.left, edges.top, edges.right];
      for (edge, spec) in opts.frame_tick_edges.iter().enumerate() {
        if !present[edge] {
          continue;
        }
        let horizontal = edge % 2 == 0;
        let (scale, axis_ticks) = if horizontal {
          (x, x_ticks)
        } else {
          (y, y_ticks)
        };
        d.frame_ticks[edge] = match spec {
          // The bottom and left edge label what the axis ticks hold (an
          // explicit `Ticks` list included); the far edges the automatic
          // ticks.
          FrameEdgeTicks::Labeled if edge < 2 => {
            boxes(&axis_ticks.majors, scale, None)
          }
          FrameEdgeTicks::Labeled => boxes(
            &crate::functions::plot_axes::AxisTicks::new(
              scale,
              horizontal && opts.date_axis,
              None,
            )
            .majors,
            scale,
            None,
          ),
          FrameEdgeTicks::Explicit(ticks) => {
            d.frame_tick_out[edge] = ticks
              .iter()
              .filter(|t| scale.contains(t.pos))
              .filter_map(|t| t.len.map(|(_, out)| out))
              .fold(0.0, f64::max);
            let listed: Vec<(f64, String)> = ticks
              .iter()
              .filter(|t| scale.contains(t.pos))
              .map(|t| (t.pos, t.label.clone()))
              .collect();
            boxes(&listed, scale, None)
          }
          FrameEdgeTicks::None | FrameEdgeTicks::Marks => Vec::new(),
        };
      }
    }
  } else if show_ticks {
    // A label at the crossing is dropped when the other axis runs past it
    // (decided in `extents`, which knows the plot size).
    let (sx, sy) = opts.axes;
    d.axis_ticks = (
      boxes(&x_ticks.majors, x, sy.then_some(origin.0)),
      boxes(&y_ticks.majors, y, sx.then_some(origin.1)),
    );
  }
  let text_label = |t: &Option<String>| {
    t.as_ref()
      .filter(|t| !t.is_empty())
      .map(|t| block_height(label_lines(t), label_font))
  };
  d.frame_labels = [
    text_label(&opts.frame_label_bottom).map(|h| {
      if bottom_graphic_h > 0.0 {
        h + FRAME_LABEL_GAP + bottom_graphic_h
      } else {
        h
      }
    }),
    text_label(&opts.frame_label_left),
    text_label(&opts.frame_label_top),
    text_label(&opts.frame_label_right),
  ];
  if d.frame_labels[0].is_none() && bottom_graphic_h > 0.0 {
    d.frame_labels[0] = Some(bottom_graphic_h);
  }
  if !frame {
    d.frame_labels = [None; 4];
  }
  d.curve_labels = opts
    .plot_labels
    .iter()
    .flatten()
    .map(|l| label_width(l, label_font))
    .reduce(f64::max);
  d.side_labels =
    crate::functions::plot_curve_labels::side_room(&opts.curve_label_specs);
  if frame && show_ticks {
    for axis in &opts.extra_frame_axes {
      let width = axis
        .ticks
        .iter()
        .filter(|t| y.contains(t.pos) && !t.label.is_empty())
        .map(|t| tick_label_width(&t.label, tick_font))
        .fold(0.0, f64::max);
      d.extra_axes[usize::from(axis.edge == 3)].push(width);
    }
  }
  if let Some((lx, ly)) = &opts.axes_label
    && !frame
  {
    let measure = |t: &str| {
      (!t.is_empty()).then(|| {
        (
          label_width(t, label_font),
          block_height(label_lines(t), label_font),
        )
      })
    };
    d.axes_labels = (
      measure(lx).filter(|_| opts.axes.0),
      measure(ly).filter(|_| opts.axes.1),
    );
  }
  d
}

/// The image height, in whole points, for a plotting area `plot_h` tall
/// inside `pad`: Wolfram rounds the sum up.
pub(crate) fn image_height(plot_h: f64, pad: &Padding) -> f64 {
  let total = plot_h + pad.bottom + pad.above();
  // Tolerate sampling noise just above a whole number (a sampled ellipse
  // spans a hair more than its exact range).
  (total - 4e-3).ceil().max(1.0)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tick_label_widths_follow_arial() {
    assert!((tick_label_width("100", 10.0) - 16.6846).abs() < 1e-3);
    assert!((tick_label_width("0.0010", 10.0) - 30.586).abs() < 1e-3);
    // A minus a little wider than Arial's U+2212.
    assert!((tick_label_width("-1.0", 10.0) - 20.204).abs() < 1e-3);
    // `10 000`: a thin space between the digit blocks.
    assert!((tick_label_width("10000", 10.0) - 29.199).abs() < 1e-3);
    assert!((text_width("time", 10.0) - 18.892).abs() < 1e-3);
  }

  #[test]
  fn labels_running_off_the_canvas_are_pulled_back_in() {
    // `0.08` right-aligned 19.5 pt from the left edge: Arial's width, but
    // Atkinson's ink reaches 20.61 pt left of the anchor.
    let svg = "<svg viewBox=\"0 0 400 200\"><text x=\"19.5\" y=\"9\" \
               text-anchor=\"end\" font-size=\"10\">0.08</text></svg>";
    let out = keep_text_in_view(svg);
    assert!(out.contains("x=\"20.6\""), "{out}");
    // The column of y labels moves together, so it stays right-aligned.
    let svg = "<svg viewBox=\"0 0 400 200\"><text x=\"19.5\" y=\"9\" \
               text-anchor=\"end\" font-size=\"10\">0.08</text><text \
               x=\"19.5\" y=\"29\" text-anchor=\"end\" font-size=\"10\">\
               0.10</text></svg>";
    let out = keep_text_in_view(svg);
    assert_eq!(out.matches("x=\"20.6\"").count(), 2, "{out}");
    // A centred label overhanging the right edge moves left.
    let svg = "<svg viewBox=\"0 0 400 200\"><text x=\"398\" y=\"9\" \
               text-anchor=\"middle\" font-size=\"10\">100</text></svg>";
    let out = keep_text_in_view(svg);
    assert!(out.contains("x=\"391.5\""), "{out}");
    // Labels already inside, and labels in a transformed group, stay.
    let svg = "<svg viewBox=\"0 0 400 200\"><text x=\"30\" \
               text-anchor=\"end\" font-size=\"10\">0.08</text>\
               <g transform=\"translate(50,0)\"><text x=\"1\" \
               text-anchor=\"end\" font-size=\"10\">0.08</text></g></svg>";
    assert!(matches!(
      keep_text_in_view(svg),
      std::borrow::Cow::Borrowed(_)
    ));
    // Nor do labels of a nested `<svg>` (a cell of a list of graphics),
    // which lie in that cell's own, wider coordinates.
    let svg = "<svg viewBox=\"0 0 389 101\"><svg x=\"7.8\" width=\"180\" \
               viewBox=\"0 0 1800 1010\"><text x=\"1684.7\" \
               text-anchor=\"middle\" font-size=\"100\">6</text></svg></svg>";
    assert!(matches!(
      keep_text_in_view(svg),
      std::borrow::Cow::Borrowed(_)
    ));
  }

  fn frame_sin() -> Decorations {
    // Plot[Sin[x], {x, 0, 10}, Frame -> True]: ticks 0..10 by 2 on x,
    // -1..1 by 0.5 on y, ranges padded by 1/50 (x) and 1/20 (y) of the
    // padded span on each side.
    let lh = line_height(TICK_FONT);
    let x = (0..=5)
      .map(|i| LabelBox {
        frac: (1.0 + 48.0 * i as f64 / 5.0) / 50.0,
        width: tick_label_width(&(2 * i).to_string(), TICK_FONT),
        height: lh,
        at_crossing: false,
      })
      .collect();
    let y = ["-1.0", "-0.5", "0.0", "0.5", "1.0"]
      .iter()
      .enumerate()
      .map(|(i, l)| LabelBox {
        frac: (1.0 + 18.0 * i as f64 / 4.0) / 20.0,
        width: tick_label_width(l, TICK_FONT),
        height: lh,
        at_crossing: false,
      })
      .collect();
    Decorations {
      frame: true,
      frame_ticks: [x, y, Vec::new(), Vec::new()],
      ..Default::default()
    }
  }

  #[test]
  fn framed_plot_padding_matches_wolfram() {
    let ar = 1.0 / f64::midpoint(1.0, 5f64.sqrt());
    let pad = frame_sin().solve(360.0, |w| w * ar);
    // AbsoluteOptions[…, ImagePadding] in WL 15.
    assert!((pad.left - 23.204).abs() < 1e-3, "{pad:?}");
    assert!((pad.right - 1.5).abs() < 1e-9);
    assert!((pad.bottom - 15.172).abs() < 1e-3);
    assert!((pad.top - 0.5).abs() < 1e-9);
    let w = 360.0 - pad.left - pad.right;
    assert_eq!(image_height(w * ar, &pad), 223.0);
  }

  #[test]
  fn small_frame_pads_for_overhanging_end_labels() {
    let ar = 1.0 / f64::midpoint(1.0, 5f64.sqrt());
    let pad = frame_sin().solve(180.0, |w| w * ar);
    assert!((pad.right - 3.496).abs() < 2e-3, "{pad:?}");
    assert!((pad.top - 0.849).abs() < 2e-3, "{pad:?}");
  }
}
