#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{DEFAULT_HEIGHT, DEFAULT_WIDTH, parse_image_size};
use crate::syntax::expr_to_output;

/// Dash length for the "Small" named size in Dashing directives.
/// This is the default dash segment length used by Dashed, Dotted, etc.
/// `Small`, the default dash length, in pixels (`Dashed` is `4,4`).
const SMALL_DASH_PX: f64 = 4.0;

/// Convert a named size (Tiny, Small, Medium, Large) to a dash length.
/// A named dash size. Wolfram's named sizes are *absolute* lengths — a
/// `Dashed` line is 4 pixels on and 4 off whatever the image size, while a
/// numeric `Dashing[{0.05, …}]` is a fraction of the image width. The two
/// are told apart by sign, as `symbolic_thickness` already does: a negative
/// length is absolute pixels. Measured from wolframscript's SVG export.
/// The radius in pixels of a point drawn at `point_size`: a fraction of the
/// image width, or — stored negative — an absolute size in printer's points
/// (`AbsolutePointSize[6]` is a 6-pixel dot at every image size, measured
/// from wolframscript's own SVG export).
fn point_radius(point_size: f64, svg_w: f64) -> f64 {
  if point_size < 0.0 {
    -point_size * 0.5
  } else {
    point_size * svg_w * 0.5
  }
}

/// A named point size. `Small`/`Medium`/… name absolute sizes, like the
/// named dash lengths do.
fn symbolic_point_size(expr: &Expr) -> Option<f64> {
  let Expr::Identifier(s) = expr else {
    return None;
  };
  match s.as_str() {
    "Tiny" => Some(-1.0),
    "Small" => Some(-2.0),
    "Medium" => Some(-4.5),
    "Large" => Some(-7.0),
    _ => None,
  }
}

fn dash_size_to_f64(expr: &Expr) -> Option<f64> {
  if let Expr::Identifier(s) = expr {
    match s.as_str() {
      "Tiny" => Some(-2.0),
      "Small" => Some(-SMALL_DASH_PX),
      "Medium" => Some(-8.0),
      "Large" => Some(-16.0),
      _ => None,
    }
  } else {
    None
  }
}

// ── Color ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Color {
  pub(crate) r: f64,
  pub(crate) g: f64,
  pub(crate) b: f64,
  pub(crate) a: f64,
}

impl Color {
  pub(crate) fn new(r: f64, g: f64, b: f64) -> Self {
    Self { r, g, b, a: 1.0 }
  }

  pub(crate) fn with_alpha(mut self, a: f64) -> Self {
    self.a = a;
    self
  }

  pub(crate) fn to_svg_rgb(self) -> String {
    let r = (self.r.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (self.g.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (self.b.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("rgb({r},{g},{b})")
  }

  fn opacity_attr(&self) -> String {
    if self.a < 1.0 {
      format!(" opacity=\"{}\"", self.a)
    } else {
      String::new()
    }
  }

  fn darker(self, amount: f64) -> Self {
    let f = 1.0 - amount;
    Self::new(self.r * f, self.g * f, self.b * f).with_alpha(self.a)
  }

  fn lighter(self, amount: f64) -> Self {
    let f = amount;
    Self::new(
      self.r + (1.0 - self.r) * f,
      self.g + (1.0 - self.g) * f,
      self.b + (1.0 - self.b) * f,
    )
    .with_alpha(self.a)
  }

  pub(crate) fn gray(level: f64) -> Self {
    Self::new(level, level, level)
  }

  /// Convert to an Expr (RGBColor or GrayLevel) for embedding in Graphics expressions.
  pub(crate) fn to_expr(self) -> Expr {
    if (self.r - self.g).abs() < 1e-14 && (self.g - self.b).abs() < 1e-14 {
      call1("GrayLevel", Expr::Real(self.r))
    } else {
      call(
        "RGBColor",
        vec![Expr::Real(self.r), Expr::Real(self.g), Expr::Real(self.b)],
      )
    }
  }

  pub(crate) fn from_hue(h: f64, s: f64, b: f64) -> Self {
    // HSB to RGB conversion
    let h = ((h % 1.0) + 1.0) % 1.0;
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f64;
    let p = b * (1.0 - s);
    let q = b * (1.0 - f * s);
    let t = b * (1.0 - (1.0 - f) * s);
    let (r, g, bl) = match i % 6 {
      0 => (b, t, p),
      1 => (q, b, p),
      2 => (p, b, t),
      3 => (p, q, b),
      4 => (t, p, b),
      _ => (b, p, q),
    };
    Self::new(r, g, bl)
  }
}

const BLACK: Color = Color {
  r: 0.0,
  g: 0.0,
  b: 0.0,
  a: 1.0,
};

// ── Theme colors for light/dark mode ────────────────────────────────────

pub struct ThemeColors {
  pub text_primary: &'static str,
  pub text_secondary: &'static str,
  pub text_muted: &'static str,
  pub stroke_default: &'static str,
  pub axis_stroke: &'static str,
  pub tick_label_fill: &'static str,
  pub table_header_bg: &'static str,
  pub table_row_num_bg: &'static str,
  pub table_row_num_header_bg: &'static str,
  pub table_border_strong: &'static str,
  pub table_border_light: &'static str,
  pub framed_border: &'static str,
  pub highlighted_bg: &'static str,
}

const LIGHT_THEME: ThemeColors = ThemeColors {
  text_primary: "#333",
  text_secondary: "#555",
  text_muted: "#888",
  stroke_default: "black",
  axis_stroke: "#b3b3b3",
  tick_label_fill: "#555555",
  table_header_bg: "#f0f0f0",
  table_row_num_bg: "#eef2f7",
  table_row_num_header_bg: "#dde4ed",
  table_border_strong: "#999",
  table_border_light: "#ccc",
  framed_border: "rgb(190,190,190)",
  highlighted_bg: "rgb(255,245,155)",
};

const DARK_THEME: ThemeColors = ThemeColors {
  text_primary: "#e0e0e0",
  text_secondary: "#b0b0b0",
  text_muted: "#777",
  stroke_default: "#e0e0e0",
  axis_stroke: "#555",
  tick_label_fill: "#a0a0a0",
  table_header_bg: "#2a2a2a",
  table_row_num_bg: "#1e2830",
  table_row_num_header_bg: "#252d35",
  table_border_strong: "#555",
  table_border_light: "#3a3a3a",
  framed_border: "rgb(80,80,80)",
  highlighted_bg: "rgb(102,92,20)",
};

pub fn theme() -> &'static ThemeColors {
  if crate::is_dark_mode() {
    &DARK_THEME
  } else {
    &LIGHT_THEME
  }
}

pub(crate) fn named_color(name: &str) -> Option<Color> {
  Some(match name {
    // Basic colors (matching Wolfram Language values)
    "Red" => Color::new(1.0, 0.0, 0.0),
    "Green" => Color::new(0.0, 1.0, 0.0),
    "Blue" => Color::new(0.0, 0.0, 1.0),
    "Black" => Color::new(0.0, 0.0, 0.0),
    "White" => Color::new(1.0, 1.0, 1.0),
    "Gray" => Color::new(0.5, 0.5, 0.5),
    "Cyan" => Color::new(0.0, 1.0, 1.0),
    "Magenta" => Color::new(1.0, 0.0, 1.0),
    "Yellow" => Color::new(1.0, 1.0, 0.0),
    "Brown" => Color::new(0.6, 0.4, 0.2),
    "Orange" => Color::new(1.0, 0.5, 0.0),
    "Pink" => Color::new(1.0, 0.5, 0.5),
    "Purple" => Color::new(0.5, 0.0, 0.5),
    // Light colors (matching Wolfram Language values)
    "LightRed" => Color::new(1.0, 0.85, 0.85),
    "LightBlue" => Color::new(0.87, 0.94, 1.0),
    "LightGreen" => Color::new(0.88, 1.0, 0.88),
    "LightGray" => Color::new(0.85, 0.85, 0.85),
    "LightOrange" => Color::new(1.0, 0.9, 0.8),
    "LightYellow" => Color::new(1.0, 1.0, 0.85),
    "LightPurple" => Color::new(0.94, 0.88, 0.94),
    "LightCyan" => Color::new(0.9, 1.0, 1.0),
    "LightMagenta" => Color::new(1.0, 0.9, 1.0),
    "LightBrown" => Color::new(0.94, 0.91, 0.88),
    "LightPink" => Color::new(1.0, 0.925, 0.925),
    _ => return None,
  })
}

// ── Style State ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StyleState {
  color: Color,
  /// `FaceForm[…]` sets only this — the fill color of area primitives
  /// (Disk, Polygon, Rectangle, …) — leaving `color` (which Text, Line,
  /// Point and everything else read) untouched. `None` means those
  /// primitives fall back to `color` too, matching a bare color directive.
  face_color: Option<Color>,
  opacity: f64,
  thickness: f64, // fraction of plot width; negative = absolute pixels
  point_size: f64, // fraction of plot width, default ~0.012
  dashing: Option<Vec<f64>>, // dash lengths in coordinate-space fractions
  edge_form: Option<EdgeForm>,
  halo: Option<Halo>, // Haloing[...] contrasting outline behind primitives
  drop_shadow: Option<DropShadow>, // DropShadowing[...] shadow behind primitives
  font_size: f64,
  font_weight: String,
  font_style: String,
  font_family: String, // empty string means SVG default
  /// `Background -> colour` carried by a `Style` around a label, as in
  /// `Style[Text[…], Background -> White]`. It paints the panel behind the
  /// text — the same panel `Text[…, Background -> colour]` asks for — and
  /// means nothing to the other primitives, which never read it.
  text_background: Option<Color>,
  /// `Arrowheads[…]`: where the heads of the arrows that follow sit, how
  /// big they are and — for a custom head — what to draw there. `None`
  /// keeps Wolfram's default: one head at the tip.
  arrowheads: Option<Vec<ArrowHead>>,
}

/// One entry of an `Arrowheads` specification.
#[derive(Debug, Clone)]
struct ArrowHead {
  /// Head length as a fraction of the plot width. Negative points the head
  /// backwards along the arrow, as Wolfram's negative sizes do.
  size: f64,
  /// Where the head sits on the arrow: 0 at the tail, 1 at the tip.
  position: f64,
  /// A custom head — the graphic drawn in place of the triangle.
  graphic: Option<Expr>,
}

#[derive(Debug, Clone)]
struct EdgeForm {
  color: Option<Color>,
  thickness: Option<f64>,
}

/// `Haloing[…]` directive: draws a contrasting outline (halo) behind a
/// primitive so it stays visible against any background.  The halo is a
/// wider stroke of `color` extending `radius` pixels beyond the primitive.
#[derive(Debug, Clone)]
struct Halo {
  color: Color,
  radius: f64, // extra pixel radius beyond the primitive
}

/// `DropShadowing[…]` directive: renders primitives with a drop shadow.
/// Offsets are stored in Wolfram graphics orientation (y up, in display
/// px); the SVG emission flips the y sign. `radius` is the blur radius;
/// the SVG Gaussian `stdDeviation` is radius/2.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DropShadow {
  pub(crate) dx: f64,
  pub(crate) dy: f64,
  pub(crate) radius: f64,
  pub(crate) color: Color,
}

/// Format a shadow parameter without trailing zeros (2 → "2", 1.5 → "1.5").
fn fmt_shadow_num(x: f64) -> String {
  let x = if x == 0.0 { 0.0 } else { x }; // normalize -0
  let s = format!("{x:.2}");
  s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl DropShadow {
  /// Deterministic SVG filter id derived from the shadow parameters, so
  /// identical shadows share one `<defs>` entry without threading an id
  /// map through the render pipeline.
  pub(crate) fn filter_id(&self) -> String {
    fn enc(x: f64) -> String {
      format!("{x:.2}").replace('-', "m").replace('.', "p")
    }
    format!(
      "ds_{}_{}_{}_{:02x}{:02x}{:02x}{:02x}",
      enc(self.dx),
      enc(self.dy),
      enc(self.radius),
      (self.color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (self.color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (self.color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      (self.color.a.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  }

  /// `<filter>` definition for this shadow. `scale` converts display px
  /// to the target SVG's user units (1.0 for Graphics, RESOLUTION_SCALE
  /// for the plotters-based Plot backend). The generous filter region
  /// keeps large offsets/blurs from being clipped.
  pub(crate) fn filter_def(&self, scale: f64) -> String {
    format!(
      "<filter id=\"{}\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">\
       <feDropShadow dx=\"{}\" dy=\"{}\" stdDeviation=\"{}\" \
       flood-color=\"{}\" flood-opacity=\"{}\"/></filter>",
      self.filter_id(),
      fmt_shadow_num(self.dx * scale),
      fmt_shadow_num(-self.dy * scale),
      fmt_shadow_num(self.radius / 2.0 * scale),
      self.color.to_svg_rgb(),
      fmt_shadow_num(self.color.a.clamp(0.0, 1.0)),
    )
  }
}

/// Parse the arguments of a `DropShadowing[…]` directive into a
/// `DropShadow`, using the same positional slot logic as
/// `drop_shadowing_ast` (offset 2-list, radius number, color) and the
/// same defaults ({-3, -3}, 2, foreground at opacity 1/3). Returns
/// `None` for `DropShadowing[…, None]` (shadow disabled) and for
/// argument lists that don't fit the pattern.
pub(crate) fn parse_drop_shadowing(args: &[Expr]) -> Option<DropShadow> {
  let (mut offset, mut radius, mut color) = (None, None, None);
  for arg in args {
    let as_offset = |e: &Expr| -> Option<(f64, f64)> {
      if let Expr::List(items) = e
        && items.len() == 2
        && let Some(x) = expr_to_f64(&items[0])
        && let Some(y) = expr_to_f64(&items[1])
      {
        return Some((x, y));
      }
      None
    };
    if offset.is_none()
      && radius.is_none()
      && color.is_none()
      && let Some(o) = as_offset(arg)
    {
      offset = Some(o);
    } else if radius.is_none()
      && color.is_none()
      && !matches!(arg, Expr::List(_) | Expr::FunctionCall { .. })
      && let Some(r) = expr_to_f64(arg)
    {
      radius = Some(r);
    } else if color.is_none() {
      if matches!(arg, Expr::Identifier(s) if s == "None") {
        return None; // shadow explicitly disabled
      }
      color = Some(parse_shadow_color(arg)?);
    } else {
      return None;
    }
  }
  let (dx, dy) = offset.unwrap_or((-3.0, -3.0));
  Some(DropShadow {
    dx,
    dy,
    radius: radius.unwrap_or(2.0),
    color: color.unwrap_or(BLACK.with_alpha(1.0 / 3.0)),
  })
}

/// Parse a shadow color spec: a plain color, or `Opacity[a]` /
/// `Opacity[a, color]` (the canonical default uses
/// `Opacity[1/3, ThemeColor[Foreground]]`, whose inner ThemeColor falls
/// back to the foreground black).
fn parse_shadow_color(expr: &Expr) -> Option<Color> {
  if let Some(c) = parse_color(expr) {
    return Some(c);
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Opacity"
    && !args.is_empty()
    && let Some(a) = expr_to_f64(&args[0])
  {
    let base = args.get(1).and_then(parse_color).unwrap_or(BLACK);
    return Some(base.with_alpha(a.clamp(0.0, 1.0)));
  }
  None
}

impl Default for StyleState {
  fn default() -> Self {
    Self {
      color: BLACK,
      face_color: None,
      opacity: 1.0,
      // Wolfram strokes an undirected primitive 1 pixel wide whatever the
      // image size — measured from wolframscript at four sizes — so the
      // default is absolute (the negative encoding), not a fraction of the
      // width the way an explicit `Thickness` is.
      thickness: -1.0,
      point_size: 0.012,
      dashing: None,
      edge_form: None,
      halo: None,
      drop_shadow: None,
      font_size: 14.0,
      font_weight: "normal".to_string(),
      font_style: "normal".to_string(),
      font_family: String::new(),
      text_background: None,
      arrowheads: None,
    }
  }
}

/// The named arrowhead sizes, as a fraction of the plot width — the same
/// units an explicit numeric size is given in.
fn named_arrowhead_size(name: &str) -> Option<f64> {
  match name {
    "Tiny" => Some(0.01),
    "Small" => Some(0.02),
    "Medium" => Some(0.04),
    "Large" => Some(0.06),
    _ => None,
  }
}

/// An arrowhead size: a number, a named size, or either negated (a
/// negative size points the head back down the arrow). `-Medium` reaches
/// here as `Times[-1, Medium]`.
fn arrowhead_size(expr: &Expr) -> Option<f64> {
  if let Expr::Identifier(s) = expr {
    return named_arrowhead_size(s);
  }
  if let Expr::FunctionCall { name, args } = expr
    && name == "Times"
    && args.len() == 2
    && matches!(&args[0], Expr::Integer(-1))
  {
    return arrowhead_size(&args[1]).map(|v| -v);
  }
  expr_to_f64(expr)
}

/// Parse one entry of an `Arrowheads` list: a bare size, `{size, pos}`, or
/// `{size, pos, graphic}`. `default_position` applies to a bare size.
fn parse_arrowhead(spec: &Expr, default_position: f64) -> Option<ArrowHead> {
  match spec {
    Expr::List(items) if !items.is_empty() => Some(ArrowHead {
      size: arrowhead_size(&items[0])?,
      position: items
        .get(1)
        .and_then(expr_to_f64)
        .unwrap_or(default_position),
      graphic: items.get(2).cloned(),
    }),
    other => Some(ArrowHead {
      size: arrowhead_size(other)?,
      position: default_position,
      graphic: None,
    }),
  }
}

/// Parse the argument of `Arrowheads[…]`. Entries that do not name their
/// own position are spread evenly from tail to tip, so `Arrowheads[{-s, s}]`
/// is the usual double-headed arrow. `None` removes the heads entirely.
fn parse_arrowheads(spec: &Expr) -> Option<Vec<ArrowHead>> {
  match spec {
    Expr::Identifier(s) if s == "None" => Some(Vec::new()),
    Expr::List(items) => {
      let n = items.len();
      items
        .iter()
        .enumerate()
        .map(|(i, e)| {
          let pos = if n <= 1 {
            1.0
          } else {
            i as f64 / (n - 1) as f64
          };
          parse_arrowhead(e, pos)
        })
        .collect()
    }
    other => parse_arrowhead(other, 1.0).map(|h| vec![h]),
  }
}

impl StyleState {
  fn effective_color(&self) -> Color {
    self.color.with_alpha(self.color.a * self.opacity)
  }

  /// The fill color for area primitives (Disk, Polygon, Rectangle, …):
  /// `FaceForm`'s color when one was set, otherwise the general color —
  /// the same fallback a bare color directive gives every primitive.
  fn effective_face_color(&self) -> Color {
    let c = self.face_color.unwrap_or(self.color);
    c.with_alpha(c.a * self.opacity)
  }
}

// ── Bounding Box ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct BBox {
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
}

impl BBox {
  fn empty() -> Self {
    Self {
      x_min: f64::INFINITY,
      x_max: f64::NEG_INFINITY,
      y_min: f64::INFINITY,
      y_max: f64::NEG_INFINITY,
    }
  }

  fn include_point(&mut self, x: f64, y: f64) {
    if x.is_finite() && y.is_finite() {
      self.x_min = self.x_min.min(x);
      self.x_max = self.x_max.max(x);
      self.y_min = self.y_min.min(y);
      self.y_max = self.y_max.max(y);
    }
  }

  fn merge(&mut self, other: &Self) {
    self.x_min = self.x_min.min(other.x_min);
    self.x_max = self.x_max.max(other.x_max);
    self.y_min = self.y_min.min(other.y_min);
    self.y_max = self.y_max.max(other.y_max);
  }

  fn is_empty(&self) -> bool {
    self.x_min > self.x_max || self.y_min > self.y_max
  }

  fn with_padding(self, frac: f64) -> Self {
    if self.is_empty() {
      return self;
    }
    let dx = (self.x_max - self.x_min) * frac;
    let dy = (self.y_max - self.y_min) * frac;
    // Ensure non-zero range
    let dx = if dx < 1e-10 { 0.5 } else { dx };
    let dy = if dy < 1e-10 { 0.5 } else { dy };
    Self {
      x_min: self.x_min - dx,
      x_max: self.x_max + dx,
      y_min: self.y_min - dy,
      y_max: self.y_max + dy,
    }
  }

  fn width(&self) -> f64 {
    self.x_max - self.x_min
  }

  fn height(&self) -> f64 {
    self.y_max - self.y_min
  }
}

// ── Primitives ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Primitive {
  PointSingle {
    x: f64,
    y: f64,
    style: StyleState,
  },
  PointMulti {
    points: Vec<(f64, f64)>,
    style: StyleState,
  },
  Line {
    segments: Vec<Vec<(f64, f64)>>,
    style: StyleState,
  },
  CircleArc {
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    /// Angular extent `(theta1, theta2)` in radians for a partial circle
    /// (`Circle[c, r, {t1, t2}]`). `None` draws the full circle; a range that
    /// is not a full turn draws only that open arc (stroked on one side, not a
    /// closed sector).
    angles: Option<(f64, f64)>,
    style: StyleState,
  },
  Disk {
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    style: StyleState,
  },
  DiskSector {
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
    angle1: f64,
    angle2: f64,
    style: StyleState,
  },
  RectPrim {
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    style: StyleState,
  },
  PolygonPrim {
    points: Vec<(f64, f64)>,
    /// Boundaries cut out of the polygon (`Polygon[outer -> holes]`).
    /// Empty for an ordinary polygon.
    holes: Vec<Vec<(f64, f64)>>,
    style: StyleState,
  },
  ArrowPrim {
    points: Vec<(f64, f64)>,
    setback: (f64, f64),
    style: StyleState,
  },
  TextPrim {
    text: String,
    x: f64,
    y: f64,
    /// `Text[expr, pos, offset]`: which point of the label's own box sits
    /// at `pos`, in units running from -1 (left/bottom) to 1 (right/top).
    /// `(0, 0)` centres it, as Wolfram's default does.
    offset: (f64, f64),
    /// `Background -> colour`: a panel painted behind the label, so it
    /// stays readable over whatever it is placed on.
    background: Option<Color>,
    /// `Framed[…]`: the colour of the border drawn around the label, or
    /// `None` for an unframed one (including `FrameStyle -> None`).
    frame: Option<Color>,
    /// `x`/`y` are `Scaled[…]` fractions of the plot range, resolved when
    /// the range is known — see [`resolve_anchor`].
    scaled: bool,
    /// `Text[expr, pos, offset, direction]`'s fourth argument: a data-space
    /// vector the label's baseline is rotated to run parallel to (e.g. the
    /// local tangent of a curve it annotates). `None` for the unrotated
    /// (horizontal) default — including when the argument is omitted or
    /// written `Automatic`.
    direction: Option<(f64, f64)>,
    style: StyleState,
  },
  BezierCurvePrim {
    points: Vec<(f64, f64)>,
    style: StyleState,
  },
  /// A whole rendered picture placed inside this one by `Inset[obj, pos]`.
  /// The anchor is in data coordinates; the size is the object's own, in
  /// pixels, since an inset keeps its natural size whatever the plot range.
  InsetGraphic {
    svg: String,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    /// `x`/`y` are `Scaled[…]` fractions of the plot range, resolved when
    /// the range is known — see [`resolve_anchor`].
    scaled: bool,
  },
  RasterPrim {
    /// rows x cols grid of RGBA colors (row 0 = bottom in Wolfram coords)
    data: Vec<Vec<Color>>,
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
  },
  /// HalfPlane (fill on the `w` side of the line through `p` along `v`) or,
  /// with `full`, InfinitePlane covering the whole viewport. The actual fill
  /// polygon is built at render time so it always reaches past the visible
  /// plot range.
  HalfPlanePrim {
    p: (f64, f64),
    v: (f64, f64),
    w: (f64, f64),
    full: bool,
    style: StyleState,
  },
  /// A fixed-pixel-size marker (e.g. a `Locator`'s appearance graphic)
  /// centered on a data-space point. The pre-rendered SVG is embedded at
  /// `w`×`h` screen pixels regardless of the plot's coordinate scale.
  MarkerPrim {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    svg: String,
  },
}

impl Primitive {
  /// The style captured when the primitive was collected (None for
  /// Raster, which carries no style).
  fn style(&self) -> Option<&StyleState> {
    match self {
      Self::PointSingle { style, .. }
      | Self::PointMulti { style, .. }
      | Self::Line { style, .. }
      | Self::CircleArc { style, .. }
      | Self::Disk { style, .. }
      | Self::DiskSector { style, .. }
      | Self::RectPrim { style, .. }
      | Self::PolygonPrim { style, .. }
      | Self::ArrowPrim { style, .. }
      | Self::TextPrim { style, .. }
      | Self::BezierCurvePrim { style, .. }
      | Self::HalfPlanePrim { style, .. } => Some(style),
      Self::RasterPrim { .. }
      | Self::MarkerPrim { .. }
      | Self::InsetGraphic { .. } => None,
    }
  }
}

// ── Parsing helpers ──────────────────────────────────────────────────────

pub(crate) fn expr_to_f64(expr: &Expr) -> Option<f64> {
  try_eval_to_f64(expr)
}

fn expr_to_point(expr: &Expr) -> Option<(f64, f64)> {
  if let Expr::List(items) = expr
    && items.len() == 2
  {
    let x = expr_to_f64(&items[0])?;
    let y = expr_to_f64(&items[1])?;
    return Some((x, y));
  }
  None
}

/// Where a `Text`/`Inset` anchor sits, as `(x, y, scaled)`. With `scaled`
/// set, `x`/`y` are fractions of the final plot range rather than
/// coordinates in it: `Scaled[{0.9, 0.775}]` pins a label near the top
/// right whatever the data turns out to span. Such an anchor cannot be
/// resolved while primitives are collected — the range it refers to is only
/// known once every other primitive has been seen — so the flag travels
/// with the primitive and [`resolve_anchor`] applies it at render time.
///
/// `ImageScaled` measures the same fractions against the image rather than
/// the plot range; for a picture whose image *is* its plot range the two
/// coincide, so both read as scaled here.
fn expr_to_anchor(expr: &Expr) -> Option<(f64, f64, bool)> {
  if let Expr::FunctionCall { name, args } = expr
    && (name == "Scaled" || name == "ImageScaled")
    && args.len() == 1
    && let Some((x, y)) = expr_to_point(&args[0])
  {
    return Some((x, y, true));
  }
  expr_to_point(expr).map(|(x, y)| (x, y, false))
}

/// A primitive's anchor in data coordinates, turning a scaled fraction into
/// the point of the plot range it names.
fn resolve_anchor(x: f64, y: f64, scaled: bool, bb: &BBox) -> (f64, f64) {
  if !scaled {
    return (x, y);
  }
  (
    bb.x_min + x * (bb.x_max - bb.x_min),
    bb.y_min + y * (bb.y_max - bb.y_min),
  )
}

fn expr_to_point_list(expr: &Expr) -> Option<Vec<(f64, f64)>> {
  if let Expr::List(items) = expr {
    let mut pts = Vec::with_capacity(items.len());
    for item in items {
      pts.push(expr_to_point(item)?);
    }
    if !pts.is_empty() {
      return Some(pts);
    }
  }
  None
}

// ── Color parsing ────────────────────────────────────────────────────────

/// Parse a hex color string like "#RRGGBB" or "#RGB" into a Color.
fn parse_hex_color(s: &str) -> Option<Color> {
  let s = s.strip_prefix('#')?;
  match s.len() {
    6 => {
      let r = u8::from_str_radix(&s[0..2], 16).ok()?;
      let g = u8::from_str_radix(&s[2..4], 16).ok()?;
      let b = u8::from_str_radix(&s[4..6], 16).ok()?;
      Some(Color::new(
        r as f64 / 255.0,
        g as f64 / 255.0,
        b as f64 / 255.0,
      ))
    }
    3 => {
      let r = u8::from_str_radix(&s[0..1], 16).ok()?;
      let g = u8::from_str_radix(&s[1..2], 16).ok()?;
      let b = u8::from_str_radix(&s[2..3], 16).ok()?;
      Some(Color::new(
        (r * 17) as f64 / 255.0,
        (g * 17) as f64 / 255.0,
        (b * 17) as f64 / 255.0,
      ))
    }
    _ => None,
  }
}

/// Derive the automatic counterpart of a color for the opposite appearance
/// (`LightDarkSwitched[c]`, `LightDarkSwitched[c, Automatic]`,
/// `LightDarkSwitched[Automatic, c]`): hue and saturation are kept while
/// the HSL lightness is flipped, so the color stays legible when the
/// background switches between light and dark.
fn auto_light_dark_variant(c: Color) -> Color {
  let max = c.r.max(c.g).max(c.b);
  let min = c.r.min(c.g).min(c.b);
  let l = f64::midpoint(max, min);
  let flipped = 1.0 - l;
  let d = max - min;
  if d < 1e-12 {
    return Color::new(flipped, flipped, flipped).with_alpha(c.a);
  }
  let s = if l > 0.5 {
    d / (2.0 - max - min)
  } else {
    d / (max + min)
  };
  let h = if max == c.r {
    ((c.g - c.b) / d + if c.g < c.b { 6.0 } else { 0.0 }) / 6.0
  } else if max == c.g {
    ((c.b - c.r) / d + 2.0) / 6.0
  } else {
    ((c.r - c.g) / d + 4.0) / 6.0
  };
  let q = if flipped < 0.5 {
    flipped * (1.0 + s)
  } else {
    flipped + s - flipped * s
  };
  let p = 2.0 * flipped - q;
  let channel = |t: f64| {
    let t = ((t % 1.0) + 1.0) % 1.0;
    if t < 1.0 / 6.0 {
      p + (q - p) * 6.0 * t
    } else if t < 0.5 {
      q
    } else if t < 2.0 / 3.0 {
      p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
      p
    }
  };
  Color::new(channel(h + 1.0 / 3.0), channel(h), channel(h - 1.0 / 3.0))
    .with_alpha(c.a)
}

/// (light, dark) rendering values for `ThemeColor` names. The front end
/// resolves these at render time; Woxi's SVG renderer plays that role using
/// its own palette (accents follow the ColorData[97] plot colors, lightened
/// for dark mode).
fn theme_color_pair(name: &str) -> Option<(&'static str, &'static str)> {
  Some(match name {
    "Foreground" => ("#333333", "#e0e0e0"),
    "Background" => ("#ffffff", "#1e1e1e"),
    "Accent1" => ("#5e81b5", "#7a9bc9"),
    "Accent2" => ("#e19c24", "#eab04a"),
    "Accent3" => ("#8fb032", "#a3c455"),
    "Accent4" => ("#eb6235", "#ef7f58"),
    "Accent5" => ("#8778b3", "#9f92c4"),
    "Accent6" => ("#c56e1a", "#d68a40"),
    "Accent7" => ("#5d9ec7", "#7db3d4"),
    "Accent8" => ("#ffbf00", "#ffcc33"),
    "Accent9" => ("#a5609d", "#b87eb1"),
    "Syntax1" => ("#2e5f9e", "#7aa6d9"),
    "Syntax2" => ("#3c7d3c", "#7dbb7d"),
    "Syntax3" => ("#2e8b8b", "#66c2c2"),
    "Syntax4" => ("#666666", "#999999"),
    "Syntax5" => ("#8250a8", "#b48ad6"),
    "Syntax6" => ("#a8642a", "#cc9257"),
    "Syntax7" => ("#3a3a3a", "#d0d0d0"),
    "Syntax8" => ("#888888", "#808080"),
    "SyntaxError1" => ("#cc0000", "#ff6666"),
    "SyntaxError2" => ("#d94f00", "#ff8c4d"),
    "SyntaxError3" => ("#b8860b", "#e0b040"),
    "SyntaxError4" => ("#cc3366", "#e07a9e"),
    "SyntaxError5" => ("#993399", "#c273c2"),
    "SyntaxError6" => ("#8b4513", "#c47a4d"),
    _ => return None,
  })
}

/// (light, dark) rendering values for `SystemColor` names — the named UI
/// element colors of the windowing system, resolved with Woxi's palette.
fn system_color_pair(name: &str) -> Option<(&'static str, &'static str)> {
  Some(match name {
    "Accent" | "Highlight" => ("#3875d7", "#4d8de0"),
    "HighlightText" => ("#ffffff", "#ffffff"),
    "Hotlight" => ("#0066cc", "#66aaff"),
    "InactiveHighlight" => ("#c0c0c0", "#4a4a4a"),
    "InactiveHighlightText" => ("#333333", "#cccccc"),
    "Window" | "ModalDialog" | "ModelessDialog" => ("#f5f5f5", "#1e1e1e"),
    "Menu" => ("#ffffff", "#2a2a2a"),
    "Toolbar" | "Status" => ("#ececec", "#2d2d2d"),
    "Palette" | "PanelBackground" => ("#f0f0f0", "#262626"),
    "DialogButton" | "PaletteButton" => ("#e8e8e8", "#3a3a3a"),
    "StatusFrame" => ("#cccccc", "#3a3a3a"),
    "Tooltip" => ("#ffffe1", "#3a3a2e"),
    "TooltipFrame" => ("#c9c98f", "#55553a"),
    "WindowText" | "MenuText" | "DialogText" | "ButtonText" | "StatusText"
    | "TooltipText" | "TabButtonText" | "DefaultButtonText"
    | "CancelButtonText" => ("#333333", "#e0e0e0"),
    "PressedButtonText"
    | "PressedCancelButtonText"
    | "PressedDefaultButtonText"
    | "TabButtonTextPressed" => ("#000000", "#ffffff"),
    "TabButtonTextHover" => ("#111111", "#f5f5f5"),
    "DialogTextDisabled" | "TabButtonTextDisabled" => ("#999999", "#6e6e6e"),
    _ => return None,
  })
}

pub(crate) fn parse_color(expr: &Expr) -> Option<Color> {
  match expr {
    Expr::Identifier(name) => named_color(name),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "RGBColor" => {
        if args.len() >= 3 {
          let r = expr_to_f64(&args[0])?;
          let g = expr_to_f64(&args[1])?;
          let b = expr_to_f64(&args[2])?;
          let a = if args.len() >= 4 {
            expr_to_f64(&args[3]).unwrap_or(1.0)
          } else {
            1.0
          };
          Some(Color::new(r, g, b).with_alpha(a))
        } else if args.len() == 1 {
          // RGBColor["#hex"] or RGBColor[gray]
          if let Expr::String(s) = &args[0] {
            return parse_hex_color(s);
          }
          let g = expr_to_f64(&args[0])?;
          Some(Color::new(g, g, g))
        } else {
          None
        }
      }
      "Hue" => {
        if args.len() >= 3 {
          let h = expr_to_f64(&args[0])?;
          let s = expr_to_f64(&args[1])?;
          let b = expr_to_f64(&args[2])?;
          let a = if args.len() >= 4 {
            expr_to_f64(&args[3]).unwrap_or(1.0)
          } else {
            1.0
          };
          Some(Color::from_hue(h, s, b).with_alpha(a))
        } else if args.len() == 2 {
          let h = expr_to_f64(&args[0])?;
          let s = expr_to_f64(&args[1])?;
          Some(Color::from_hue(h, s, 1.0))
        } else if args.len() == 1 {
          let h = expr_to_f64(&args[0])?;
          Some(Color::from_hue(h, 1.0, 1.0))
        } else {
          None
        }
      }
      "GrayLevel" => {
        if args.is_empty() {
          None
        } else {
          let g = expr_to_f64(&args[0])?;
          let a = if args.len() >= 2 {
            expr_to_f64(&args[1]).unwrap_or(1.0)
          } else {
            1.0
          };
          Some(Color::new(g, g, g).with_alpha(a))
        }
      }
      "Darker" => {
        if args.is_empty() {
          return None;
        }
        let base = parse_color(&args[0])?;
        let amount = if args.len() >= 2 {
          expr_to_f64(&args[1]).unwrap_or(1.0 / 3.0)
        } else {
          1.0 / 3.0
        };
        Some(base.darker(amount))
      }
      "Lighter" => {
        if args.is_empty() {
          return None;
        }
        let base = parse_color(&args[0])?;
        let amount = if args.len() >= 2 {
          expr_to_f64(&args[1]).unwrap_or(1.0 / 3.0)
        } else {
          1.0 / 3.0
        };
        Some(base.lighter(amount))
      }
      "Blend" => {
        if args.is_empty() {
          return None;
        }
        if let Expr::List(colors) = &args[0] {
          if colors.len() < 2 {
            return None;
          }
          let parsed: Vec<Color> =
            colors.iter().map(parse_color).collect::<Option<Vec<_>>>()?;
          let n = parsed.len() as f64;
          if args.len() == 1 {
            // Equal blend (average)
            let r = parsed.iter().map(|c| c.r).sum::<f64>() / n;
            let g = parsed.iter().map(|c| c.g).sum::<f64>() / n;
            let b = parsed.iter().map(|c| c.b).sum::<f64>() / n;
            Some(Color::new(r, g, b))
          } else {
            // Weighted blend: Blend[{c1, c2, ...}, t]
            let t = expr_to_f64(&args[1])?.clamp(0.0, 1.0);
            let nc = parsed.len();
            if nc == 2 {
              let c1 = &parsed[0];
              let c2 = &parsed[1];
              Some(Color::new(
                c1.r * (1.0 - t) + c2.r * t,
                c1.g * (1.0 - t) + c2.g * t,
                c1.b * (1.0 - t) + c2.b * t,
              ))
            } else {
              let segments = (nc - 1) as f64;
              let pos = t * segments;
              let seg_idx = (pos as usize).min(nc - 2);
              let local_t = pos - seg_idx as f64;
              let c1 = &parsed[seg_idx];
              let c2 = &parsed[seg_idx + 1];
              Some(Color::new(
                c1.r * (1.0 - local_t) + c2.r * local_t,
                c1.g * (1.0 - local_t) + c2.g * local_t,
                c1.b * (1.0 - local_t) + c2.b * local_t,
              ))
            }
          }
        } else {
          None
        }
      }
      // Kernel evaluation keeps these symbolic; the front end resolves them
      // when rendering. Woxi's renderer resolves them here from the current
      // light/dark mode.
      "LightDarkSwitched" if !args.is_empty() && args.len() <= 2 => {
        let is_auto =
          |e: &Expr| matches!(e, Expr::Identifier(s) if s == "Automatic");
        let light = &args[0];
        let dark = args.get(1);
        if crate::is_dark_mode() {
          match dark {
            Some(d) if !is_auto(d) => parse_color(d),
            // Missing/Automatic dark variant: derive it from the light color
            _ if !is_auto(light) => {
              parse_color(light).map(auto_light_dark_variant)
            }
            _ => None,
          }
        } else if !is_auto(light) {
          parse_color(light)
        } else {
          // LightDarkSwitched[Automatic, dark]: derive the light variant
          dark
            .filter(|d| !is_auto(d))
            .and_then(parse_color)
            .map(auto_light_dark_variant)
        }
      }
      "ThemeColor" if args.len() == 1 => {
        if let Expr::String(n) = &args[0] {
          let (light, dark) = theme_color_pair(n)?;
          parse_hex_color(if crate::is_dark_mode() { dark } else { light })
        } else {
          None
        }
      }
      "SystemColor" if args.len() == 1 => {
        if let Expr::String(n) = &args[0] {
          let (light, dark) = system_color_pair(n)?;
          parse_hex_color(if crate::is_dark_mode() { dark } else { light })
        } else {
          None
        }
      }
      _ => None,
    },
    _ => None,
  }
}

/// Generate a single 16×16 SVG swatch for a color.
pub(crate) fn color_swatch_svg(color: &Color) -> String {
  format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"16\" height=\"16\" \
     viewBox=\"0 0 16 16\">\
     <rect width=\"16\" height=\"16\" rx=\"2\" fill=\"{}\"{}/>\
     </svg>",
    color.to_svg_rgb(),
    color.opacity_attr(),
  )
}

// ── Directive parsing ────────────────────────────────────────────────────

fn apply_directive(expr: &Expr, style: &mut StyleState) -> bool {
  // Named color
  if let Some(color) = parse_color(expr) {
    style.color = color;
    return true;
  }

  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // `Specularity` only means something in 3D. In 2D it is inert — but
      // it still has to be consumed here, since the colour it carries is a
      // highlight colour and letting it fall through to the generic
      // recursion would repaint the primitives that follow
      // (`{GrayLevel[.25], Specularity[White, 10], Disk[]}`).
      "Specularity" if !args.is_empty() => true,
      "Opacity" if !args.is_empty() => {
        if let Some(o) = expr_to_f64(&args[0]) {
          style.opacity = o.clamp(0.0, 1.0);
          // If a color follows as second arg, apply it too
          if args.len() >= 2
            && let Some(c) = parse_color(&args[1])
          {
            style.color = c;
          }
        }
        true
      }
      "Thickness" if args.len() == 1 => {
        // Named sizes (`Thickness[Large]`, what `Thick` evaluates to) are
        // absolute widths; anything else is a fraction of the plot width.
        if let Some(t) = symbolic_thickness_arg(&args[0]) {
          style.thickness = t;
        } else if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = t;
        }
        true
      }
      "AbsoluteThickness" if args.len() == 1 => {
        // AbsoluteThickness gives pixel-level thickness
        // We'll store it as a negative number to distinguish from relative
        if let Some(t) = symbolic_thickness_arg(&args[0]) {
          style.thickness = t;
        } else if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = -t; // negative = absolute pixels
        }
        true
      }
      "PointSize" if args.len() == 1 => {
        if let Some(s) = symbolic_point_size(&args[0]) {
          style.point_size = s;
        } else if let Some(s) = expr_to_f64(&args[0]) {
          style.point_size = s;
        }
        true
      }
      // `AbsolutePointSize[n]` is n printer's points across whatever the
      // image size, where `PointSize` is a fraction of the width. Stored
      // negative, as absolute thickness and dash lengths already are.
      "AbsolutePointSize" if args.len() == 1 => {
        if let Some(s) = symbolic_point_size(&args[0]) {
          style.point_size = s;
        } else if let Some(s) = expr_to_f64(&args[0]) {
          style.point_size = -s;
        }
        true
      }
      // `Arrowheads[…]` sets where the heads of the arrows that follow sit
      // and how big they are; an entry may also carry its own graphic to
      // draw there instead of a triangle.
      "Arrowheads" if args.len() == 1 => {
        style.arrowheads = parse_arrowheads(&args[0]);
        true
      }
      "Dashing" if !args.is_empty() => {
        // Dashing[{d1, d2, ...}] or Dashing[d]
        // Supports named sizes: Tiny, Small, Medium, Large
        match &args[0] {
          Expr::List(items) => {
            let dashes: Vec<f64> = items
              .iter()
              .filter_map(|e| dash_size_to_f64(e).or_else(|| expr_to_f64(e)))
              .collect();
            if !dashes.is_empty() {
              style.dashing = Some(dashes);
            }
          }
          _ => {
            if let Some(d) =
              dash_size_to_f64(&args[0]).or_else(|| expr_to_f64(&args[0]))
            {
              style.dashing = Some(vec![d, d]);
            }
          }
        }
        true
      }
      "EdgeForm" => {
        if args.is_empty() {
          style.edge_form = Some(EdgeForm {
            color: None,
            thickness: None,
          });
        } else {
          let mut ef = EdgeForm {
            color: None,
            thickness: None,
          };
          // Unwrap a single List argument: EdgeForm[{GrayLevel[0, 0.5]}]
          let directives: &[Expr] =
            if args.len() == 1 && matches!(&args[0], Expr::List(_)) {
              if let Expr::List(items) = &args[0] {
                items
              } else {
                args
              }
            } else {
              args
            };
          for a in directives {
            if let Some(c) = parse_color(a) {
              ef.color = Some(c);
            } else if let Expr::FunctionCall { name: n2, args: a2 } = a {
              // `EdgeForm[Thin]` arrives as `EdgeForm[Thickness[Tiny]]`.
              if n2 == "Thickness" && a2.len() == 1 {
                ef.thickness = symbolic_thickness_arg(&a2[0])
                  .or_else(|| expr_to_f64(&a2[0]));
              } else if n2 == "AbsoluteThickness" && a2.len() == 1 {
                ef.thickness = symbolic_thickness_arg(&a2[0])
                  .or_else(|| expr_to_f64(&a2[0]).map(|t| -t));
              }
            } else if let Expr::Identifier(sym) = a
              && let Some(t) = symbolic_thickness(sym)
            {
              // `EdgeForm[None]` names neither colour nor width, so it
              // draws no edge at all.
              ef.thickness = Some(t);
            }
          }
          style.edge_form = Some(ef);
        }
        true
      }
      "FaceForm" if !args.is_empty() => {
        // FaceForm[color] sets the fill color of area primitives only —
        // unlike a bare color directive, it must not recolor Text, Line
        // or Point primitives that follow it.
        if let Some(c) = parse_color(&args[0]) {
          style.face_color = Some(c);
        }
        true
      }
      "Haloing" => {
        // Haloing[]           → white halo, default radius
        // Haloing[color]      → colored halo, default radius
        // Haloing[color, r]   → colored halo of pixel radius r
        // Haloing[None]       → disable haloing
        if args.len() == 1
          && matches!(&args[0], Expr::Identifier(s) if s == "None")
        {
          style.halo = None;
        } else {
          let color = args
            .first()
            .and_then(parse_color)
            .unwrap_or(Color::new(1.0, 1.0, 1.0));
          let radius = args.get(1).and_then(expr_to_f64).unwrap_or(2.0);
          style.halo = Some(Halo { color, radius });
        }
        true
      }
      "Directive" => {
        for a in args {
          apply_directive(a, style);
        }
        true
      }
      "DropShadowing" => {
        // DropShadowing[offset, radius, color]; DropShadowing[…, None]
        // (or an unparseable spec) disables the shadow.
        style.drop_shadow = parse_drop_shadowing(args);
        true
      }
      // Darker/Lighter/RGBColor/Hue already handled by parse_color above
      _ => false,
    },
    // `Thin` / `Thick` name absolute widths (negative = absolute pixels).
    Expr::Identifier(s) if symbolic_thickness(s).is_some() => {
      style.thickness = symbolic_thickness(s).unwrap();
      true
    }
    // Dashed is equivalent to Dashing[{Small, Small}]
    Expr::Identifier(s) if s == "Dashed" => {
      style.dashing = Some(vec![-SMALL_DASH_PX, -SMALL_DASH_PX]);
      true
    }
    // Dotted is equivalent to Dashing[{0, Small}]
    Expr::Identifier(s) if s == "Dotted" => {
      style.dashing = Some(vec![0.0, -SMALL_DASH_PX]);
      true
    }
    // DotDashed is equivalent to Dashing[{0, Small, Small, Small}]
    Expr::Identifier(s) if s == "DotDashed" => {
      style.dashing =
        Some(vec![0.0, -SMALL_DASH_PX, -SMALL_DASH_PX, -SMALL_DASH_PX]);
      true
    }
    _ => false,
  }
}

/// Apply a single Style directive that affects text (font size, weight,
/// family, style). Handles bare `Bold`/`Italic` identifiers, plain
/// numeric font sizes, and Rule forms like `FontSize -> 24`,
/// `FontFamily -> "Consolas"`, `FontWeight -> "Medium"`,
/// `FontSlant -> "Italic"`. Returns `true` if the directive was
/// recognised so callers can avoid double-applying via other paths.
/// The font size and colour a named style carries in Wolfram's default
/// stylesheet — `Style[expr, "Section"]` is large and orange, `"Label"`
/// small and black. Measured from wolframscript's own rendering.
fn named_style_appearance(name: &str) -> Option<(f64, Option<(u8, u8, u8)>)> {
  Some(match name {
    "Title" => (44.0, Some((204, 12, 2))),
    "Subtitle" => (24.0, Some((89, 89, 89))),
    "Section" => (28.0, Some((202, 81, 25))),
    "Subsection" => (20.0, Some((199, 108, 41))),
    "Subsubsection" => (19.0, Some((203, 72, 20))),
    "Text" => (15.0, None),
    "Label" => (9.0, None),
    _ => return None,
  })
}

/// The point size a named font size stands for. `Tiny`, `Small`, `Medium`
/// and `Large` are absolute in Wolfram's default stylesheet; `Larger` and
/// `Smaller` scale whatever size is in force. Measured from wolframscript's
/// own rendering of `Text[Style["z", …]]` against numeric sizes.
fn named_font_size(name: &str, current: f64) -> Option<f64> {
  Some(match name {
    "Tiny" => 6.0,
    "Small" => 9.0,
    "Medium" => 12.0,
    "Large" => 24.0,
    "Larger" => current * 1.25,
    "Smaller" => current * 0.8,
    _ => return None,
  })
}

/// Whether a `Style` directive sets the font's face — its weight or slant.
/// These are the ones a `Text[…]` wrapper resets rather than inherits.
fn is_font_face_directive(d: &Expr) -> bool {
  match d {
    Expr::Identifier(s) => {
      matches!(s.as_str(), "Bold" | "Italic" | "Plain" | "Underlined")
    }
    Expr::Rule {
      pattern,
      replacement: _,
    } => matches!(pattern.as_ref(), Expr::Identifier(k)
      if matches!(k.as_str(), "FontWeight" | "FontSlant" | "FontVariations")),
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      matches!(&args[0], Expr::Identifier(k)
        if matches!(k.as_str(), "FontWeight" | "FontSlant" | "FontVariations"))
    }
    _ => false,
  }
}

/// The glyph a mathematical constant is typeset with. Wolfram shows these
/// wherever text is set rather than printed — `Infinity` inside a picture's
/// `Text` is `∞`, not the word — while `Print` keeps the name. Arithmetic
/// leaves a constant as either `Constant("Pi")` or `Identifier("Pi")`
/// depending on which side of a product it started on, so callers must try
/// both spellings.
fn typeset_constant_glyph(name: &str) -> Option<&'static str> {
  Some(match name {
    "Infinity" => "\u{221E}",    // ∞
    "Pi" => "\u{03C0}",          // π
    "E" => "\u{2147}",           // ⅇ
    "I" => "\u{2148}",           // ⅈ
    "Degree" => "\u{00B0}",      // °
    "EulerGamma" => "\u{03B3}",  // γ
    "GoldenRatio" => "\u{03D5}", // ϕ
    _ => return None,
  })
}

/// Whether an expression is a plain-assignment target: a bare symbol, or a
/// (possibly nested) list of them — `{{xmin, xmax}, {ymin, ymax}}` is the
/// left side of the list-destructuring form `{{xmin, xmax}, {ymin, ymax}} =
/// {{-2, 2}, {-2, 2}}`, which `Set` threads element-wise the same way it
/// does a plain `{a, b} = {1, 2}`.
fn is_assignment_target(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(_) => true,
    Expr::List(items) => items.iter().all(is_assignment_target),
    _ => false,
  }
}

/// The run of plain assignments a Manipulate body opens with. A body that
/// sets up its own bounds does so first (`tmin = 0; tmax = 2 Pi; …`), and
/// stopping at the first statement that is not a `Set` keeps this to
/// definitions — no plotting, no side effects beyond the names themselves.
fn leading_assignments(body: &Expr) -> Vec<&Expr> {
  let statements: &[Expr] = match body {
    Expr::CompoundExpr(items) => items,
    _ => return Vec::new(),
  };
  statements
    .iter()
    .take_while(|stmt| {
      matches!(stmt, Expr::FunctionCall { name, args }
        if name == "Set" && args.len() == 2
          && is_assignment_target(&args[0]))
    })
    .collect()
}

/// Whether a Manipulate argument is a `ButtonBar[…]` — a row of buttons
/// whose labels and actions are computed from a list.
fn is_button_bar(spec: &Expr) -> bool {
  matches!(spec, Expr::FunctionCall { name, args }
    if name == "ButtonBar" && !args.is_empty())
}

/// A `Style[expr, …]` directive list in the order Wolfram applies it: a
/// named style ("Label", "Section", …) supplies the base appearance and the
/// explicit directives sit on top of it, whichever side of it they were
/// written. `Style["A", 20, "Label"]` is 20-point text, not the 9-point the
/// "Label" stylesheet entry gives on its own.
fn style_directives_in_application_order(directives: &[Expr]) -> Vec<&Expr> {
  let (named, explicit): (Vec<&Expr>, Vec<&Expr>) =
    directives.iter().partition(
      |d| matches!(d, Expr::String(s) if named_style_appearance(s).is_some()),
    );
  named.into_iter().chain(explicit).collect()
}

fn apply_text_style_directive(d: &Expr, style: &mut StyleState) -> bool {
  match d {
    Expr::Identifier(s) if s == "Bold" => {
      style.font_weight = "bold".to_string();
      true
    }
    Expr::Identifier(s) if s == "Italic" => {
      style.font_style = "italic".to_string();
      true
    }
    Expr::Identifier(s) if s == "Plain" => {
      style.font_weight = "normal".to_string();
      style.font_style = "normal".to_string();
      true
    }
    // `Style[expr, Large]` — a named size, the same way a number is one.
    Expr::Identifier(s) | Expr::Constant(s)
      if named_font_size(s, style.font_size).is_some() =>
    {
      style.font_size = named_font_size(s, style.font_size).unwrap();
      true
    }
    Expr::Integer(n) => {
      style.font_size = *n as f64;
      true
    }
    Expr::Real(f) => {
      style.font_size = *f;
      true
    }
    // A named style ("Section", "Label", …) brings its own size and
    // colour from the stylesheet.
    Expr::String(name) => match named_style_appearance(name) {
      Some((size, color)) => {
        style.font_size = size;
        if let Some((r, g, b)) = color {
          style.color =
            Color::new(r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0);
        }
        true
      }
      None => false,
    },
    Expr::Rule {
      pattern,
      replacement,
    } => apply_text_style_rule(pattern, replacement, style),
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      apply_text_style_rule(&args[0], &args[1], style)
    }
    _ => false,
  }
}

fn apply_text_style_rule(
  pattern: &Expr,
  replacement: &Expr,
  style: &mut StyleState,
) -> bool {
  let key = match pattern {
    Expr::Identifier(s) => s.as_str(),
    _ => return false,
  };
  match key {
    "FontSize" => {
      if let Some(sz) = expr_to_f64(replacement) {
        style.font_size = sz;
        return true;
      }
      // `FontSize -> Large` names its size instead of giving a number.
      if let Expr::Identifier(s) | Expr::Constant(s) = replacement
        && let Some(sz) = named_font_size(s, style.font_size)
      {
        style.font_size = sz;
        return true;
      }
      false
    }
    "Background" => match parse_background(replacement) {
      Some(color) => {
        style.text_background = Some(color);
        true
      }
      None => false,
    },
    "FontFamily" => match replacement {
      Expr::String(s) => {
        style.font_family.clone_from(s);
        true
      }
      Expr::Identifier(s) => {
        style.font_family.clone_from(s);
        true
      }
      _ => false,
    },
    "FontWeight" => {
      let v = match replacement {
        Expr::String(s) => Some(s.as_str()),
        Expr::Identifier(s) => Some(s.as_str()),
        _ => None,
      };
      if let Some(s) = v {
        style.font_weight = match s {
          "Bold" | "bold" => "bold".to_string(),
          "Plain" | "Normal" | "normal" => "normal".to_string(),
          // Pass through SVG-recognised names/numbers (Light, Medium, ...)
          other => other.to_lowercase(),
        };
        return true;
      }
      false
    }
    "FontSlant" | "FontStyle" => {
      let v = match replacement {
        Expr::String(s) => Some(s.as_str()),
        Expr::Identifier(s) => Some(s.as_str()),
        _ => None,
      };
      if let Some(s) = v {
        style.font_style = match s {
          "Italic" | "italic" => "italic".to_string(),
          "Oblique" | "oblique" => "oblique".to_string(),
          "Plain" | "Normal" | "normal" => "normal".to_string(),
          other => other.to_lowercase(),
        };
        return true;
      }
      false
    }
    _ => false,
  }
}

// ── AST walker ───────────────────────────────────────────────────────────

fn collect_primitives(
  expr: &Expr,
  style: &mut StyleState,
  prims: &mut Vec<Primitive>,
  errors: &mut Vec<String>,
) {
  match expr {
    Expr::List(items) => {
      // Nested list scopes style changes
      let saved = style.clone();
      for item in items {
        collect_primitives(item, style, prims, errors);
      }
      *style = saved;
    }
    // A primitive whose *argument* is `Dynamic[…]` draws the value that
    // argument currently has: `Line[Dynamic[{p1, p2, p3}]]` is the shape a
    // Demonstration's draggable vertices trace out, and a front end shows it
    // as the line those points make. `Dynamic` is HoldFirst, so the content
    // arrives unexpanded and has to be released here; without this the
    // primitive reads as malformed and the whole picture becomes an error box.
    Expr::FunctionCall { name, args }
      if args.iter().any(|a| {
        matches!(a, Expr::FunctionCall { name, args }
          if name == "Dynamic" && args.len() == 1)
      }) && !matches!(name.as_str(), "Dynamic" | "Style") =>
    {
      let released: Vec<Expr> = args
        .iter()
        .map(|a| match a {
          Expr::FunctionCall { name, args: inner }
            if name == "Dynamic" && inner.len() == 1 =>
          {
            crate::evaluator::evaluate_expr_to_expr(&inner[0])
              .unwrap_or_else(|_| a.clone())
          }
          other => other.clone(),
        })
        .collect();
      collect_primitives(
        &Expr::FunctionCall {
          name: name.clone(),
          args: released.into(),
        },
        style,
        prims,
        errors,
      );
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        // Style directives are handled by apply_directive
        "Style" if args.len() >= 2 => {
          let saved = style.clone();
          // Apply directives (everything after first arg)
          for directive in style_directives_in_application_order(&args[1..]) {
            apply_directive(directive, style);
            apply_text_style_directive(directive, style);
          }
          collect_primitives(&args[0], style, prims, errors);
          *style = saved;
        }

        // Geometric primitives
        "Point" if !args.is_empty() => {
          let before = prims.len();
          parse_point(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "Line" if !args.is_empty() => {
          let before = prims.len();
          parse_line(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "Circle" => {
          parse_circle(args, style, prims);
        }
        "Disk" => {
          parse_disk(args, style, prims);
        }
        "Sphere" | "Ball" => {
          parse_sphere(name == "Ball", args, style, prims);
        }
        "Rectangle" => {
          parse_rectangle(args, style, prims);
        }
        "Polygon" | "Triangle" if !args.is_empty() => {
          let before = prims.len();
          parse_polygon(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "Arrow" if !args.is_empty() => {
          let before = prims.len();
          parse_arrow(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "Text" if !args.is_empty() => {
          // `Text[picture, pos]` embeds an already-rendered `Graphics` or
          // `Image` the same way `Inset` does — either primitive can hold a
          // picture, not just a string label — instead of printing the
          // object's `-Graphics-`/`-Image-` short form as literal text.
          match peel_style_wrapper(&args[0]) {
            Expr::Graphics { .. } | Expr::Image { .. } => {
              match inset_primitives(args, errors) {
                Some(inner) => prims.extend(inner),
                None => parse_text(args, style, prims),
              }
            }
            _ => parse_text(args, style, prims),
          }
        }
        "BezierCurve" if !args.is_empty() => {
          let before = prims.len();
          parse_bezier(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "BSplineCurve" if !args.is_empty() => {
          let before = prims.len();
          parse_bspline(args, style, prims);
          if prims.len() == before {
            errors.push(format!("Coordinate {} should be a pair of numbers, or a list of pairs of numbers.", expr_to_string(&args[0])));
          }
        }
        "PolarCurve" if args.len() >= 2 => {
          parse_polar_curve(args, style, prims, false);
        }
        "FilledPolarCurve" if !args.is_empty() => {
          // FilledPolarCurve[PolarCurve[r, {t, t0, t1}]] wraps a curve;
          // also accept the direct FilledPolarCurve[r, {t, t0, t1}] form.
          if let Expr::FunctionCall {
            name: inner_name,
            args: inner_args,
          } = &args[0]
            && inner_name == "PolarCurve"
            && inner_args.len() >= 2
          {
            parse_polar_curve(inner_args, style, prims, true);
          } else if args.len() >= 2 {
            parse_polar_curve(args, style, prims, true);
          }
        }
        "Inset" if !args.is_empty() => {
          // `Inset[graphic, pos, …]` draws a picture inside this one; any
          // other object (a string, a number) is placed like `Text`.
          match inset_primitives(args, errors) {
            Some(inner) => prims.extend(inner),
            None => parse_text(args, style, prims),
          }
        }
        // `RasterBox` is the box form of `Raster` — the shape a stored
        // Demonstration image arrives in.
        "Raster" | "RasterBox" if !args.is_empty() => {
          parse_raster(args, prims);
        }
        "GraphicsComplex" if args.len() >= 2 => {
          if let Some(coords) = expr_to_point_list(&args[0]) {
            // Resolve integer indices to coordinates and process normally
            let resolved = resolve_graphics_complex_indices(&args[1], &coords);
            collect_primitives(&resolved, style, prims, errors);
          }
        }
        "RegularPolygon" if !args.is_empty() => {
          parse_regular_polygon(args, style, prims);
        }
        "Parallelogram" => {
          parse_parallelogram(args, style, prims);
        }
        // JoinedCurve[{c1, c2, …}] draws its curve components as one path;
        // stroke-rendering each component in order is visually equivalent.
        "JoinedCurve" if !args.is_empty() => {
          collect_primitives(&args[0], style, prims, errors);
        }
        "HalfPlane" => {
          parse_half_plane(args, style, prims);
        }
        "InfinitePlane" => {
          parse_infinite_plane(args, style, prims);
        }
        // Rotate[g, θ] rotates g by θ radians counterclockwise about the
        // center of its bounding box; Rotate[g, θ, {x, y}] about the point
        // {x, y}. Collect the inner primitives, then rotate their coordinates.
        "Rotate" if args.len() >= 2 => {
          let mut inner_style = style.clone();
          let mut inner = Vec::new();
          collect_primitives(&args[0], &mut inner_style, &mut inner, errors);
          match expr_to_f64(&args[1]) {
            Some(angle) => {
              let (cx, cy) =
                args.get(2).and_then(expr_to_point).unwrap_or_else(|| {
                  let mut bb = BBox::empty();
                  for p in &inner {
                    bb.merge(&primitive_bbox(p));
                  }
                  if bb.is_empty() {
                    (0.0, 0.0)
                  } else {
                    (
                      f64::midpoint(bb.x_min, bb.x_max),
                      f64::midpoint(bb.y_min, bb.y_max),
                    )
                  }
                });
              for p in &inner {
                prims.push(rotate_primitive(p, cx, cy, angle));
              }
            }
            // Non-numeric angle: draw the content unrotated as a fallback.
            None => prims.extend(inner),
          }
        }
        // Translate[g, {dx, dy}] translates g by the given vector;
        // Translate[g, {{dx1, dy1}, {dx2, dy2}, …}] draws one translated
        // copy of g per vector.
        "Translate" if args.len() >= 2 => {
          let mut inner_style = style.clone();
          let mut inner = Vec::new();
          collect_primitives(&args[0], &mut inner_style, &mut inner, errors);
          let offsets: Vec<(f64, f64)> =
            if let Some(v) = expr_to_point(&args[1]) {
              vec![v]
            } else {
              expr_to_point_list(&args[1]).unwrap_or_default()
            };
          if offsets.is_empty() {
            // Non-numeric offset: draw the content untranslated as a fallback.
            prims.extend(inner);
          } else {
            for &(dx, dy) in &offsets {
              for p in &inner {
                prims.push(translate_primitive(p, dx, dy));
              }
            }
          }
        }
        // Scale[g, s] scales g by s about the center of its bounding box;
        // Scale[g, {sx, sy}] scales each axis separately and
        // Scale[g, s, {x, y}] scales about the point {x, y}.
        "Scale" if args.len() >= 2 => {
          let mut inner_style = style.clone();
          let mut inner = Vec::new();
          collect_primitives(&args[0], &mut inner_style, &mut inner, errors);
          let factors = match &args[1] {
            Expr::List(_) => expr_to_point(&args[1]),
            other => expr_to_f64(other).map(|s| (s, s)),
          };
          match factors {
            Some((sx, sy)) => {
              let (cx, cy) =
                args.get(2).and_then(expr_to_point).unwrap_or_else(|| {
                  let mut bb = BBox::empty();
                  for p in &inner {
                    bb.merge(&primitive_bbox(p));
                  }
                  if bb.is_empty() {
                    (0.0, 0.0)
                  } else {
                    (
                      f64::midpoint(bb.x_min, bb.x_max),
                      f64::midpoint(bb.y_min, bb.y_max),
                    )
                  }
                });
              for p in &inner {
                prims.push(scale_primitive(p, cx, cy, sx, sy));
              }
            }
            // Non-numeric factor: draw the content unscaled as a fallback.
            None => prims.extend(inner),
          }
        }
        // `GeometricTransformation[g, t]` draws `g` mapped through the
        // affine transform `t` — a `{matrix, vector}` pair, a bare matrix,
        // a `TransformationFunction[…]`, or a list of any of those (one
        // copy per transform). A Demonstration mirrors a curve about the
        // line `y = x` this way to draw a function's inverse.
        "GeometricTransformation" if args.len() == 2 => {
          let mut inner_style = style.clone();
          let mut inner = Vec::new();
          collect_primitives(&args[0], &mut inner_style, &mut inner, errors);
          let transforms = parse_affine_transforms(&args[1]);
          if transforms.is_empty() {
            // Nothing numeric to map through: draw the content as it is.
            prims.extend(inner);
          } else {
            for (m, v) in &transforms {
              for p in &inner {
                prims.push(affine_primitive(p, *m, *v));
              }
            }
          }
        }

        // `Dynamic[expr]` displays as the current value of `expr`: release
        // the hold and render the result (Dynamic is HoldFirst, so the
        // content arrives unevaluated). Evaluation also runs any
        // assignments the Dynamic performs (the Demonstrations pattern
        // computes shared values inside a graphic's Dynamic), which later
        // display items read. Any graphics the evaluation captures are
        // embedded here, not standalone outputs — drop them from the
        // capture buffer.
        "Dynamic" if !args.is_empty() => {
          let captured = crate::captured_graphics_count();
          if let Ok(inner) = crate::evaluator::evaluate_expr_to_expr(&args[0]) {
            crate::truncate_captured_graphics(captured);
            collect_primitives(&inner, style, prims, errors);
          } else {
            crate::truncate_captured_graphics(captured);
          }
        }
        // `Tooltip[g, label]` draws `g`; the hover label has no static SVG
        // form. Only the first argument is rendered — the label must not
        // leak into the graphic.
        "Tooltip" if !args.is_empty() => {
          collect_primitives(&args[0], style, prims, errors);
        }
        // `Locator[pt]` / `Locator[Dynamic[pt, …], appearance]`: a marker
        // drawn at the point's current position. A custom appearance
        // graphic keeps its own ImageSize in screen pixels.
        "Locator" if !args.is_empty() => {
          parse_locator(args, prims);
        }
        _ => {
          // Try as directive first
          if !apply_directive(expr, style) {
            // Not recognized - could be a nested graphics expression
            for a in args {
              collect_primitives(a, style, prims, errors);
            }
          }
        }
      }
    }
    Expr::Identifier(name) => {
      // Try as named color directive
      apply_directive(expr, &mut *style);
      let _ = name;
    }
    _ => {}
  }
}

/// Resolve integer indices within a GraphicsComplex to actual coordinate pairs.
/// In GraphicsComplex, integer indices (1-based) refer to the coordinate list.
/// This function walks the expression tree and replaces:
/// - Single integers inside primitives → coordinate pair {x, y}
/// - Lists of integers inside primitives → lists of coordinate pairs
fn resolve_graphics_complex_indices(
  expr: &Expr,
  coords: &[(f64, f64)],
) -> Expr {
  match expr {
    Expr::List(items) => {
      // A list of integers → resolve each to a coordinate pair
      if !items.is_empty()
        && items.iter().all(|e| matches!(e, Expr::Integer(_)))
      {
        Expr::List(
          items
            .iter()
            .map(|e| {
              if let Expr::Integer(idx) = e {
                index_to_coord(*idx, coords)
              } else {
                e.clone()
              }
            })
            .collect(),
        )
      } else {
        Expr::List(
          items
            .iter()
            .map(|e| resolve_graphics_complex_indices(e, coords))
            .collect(),
        )
      }
    }
    Expr::FunctionCall { name, args } => {
      // For primitives that take point arguments, resolve integer indices
      match name.as_str() {
        "Point" | "Line" | "Polygon" | "Triangle" | "Arrow" | "BezierCurve"
        | "BSplineCurve" => Expr::FunctionCall {
          name: name.clone(),
          args: args
            .iter()
            .map(|a| resolve_primitive_arg(a, coords))
            .collect(),
        },
        "Circle" | "Disk" | "Rectangle" => {
          // First arg is center/position (single index), rest stay
          let mut new_args = Vec::with_capacity(args.len());
          for (i, a) in args.iter().enumerate() {
            if i == 0 {
              if let Expr::Integer(idx) = a {
                new_args.push(index_to_coord(*idx, coords));
              } else {
                new_args.push(resolve_graphics_complex_indices(a, coords));
              }
            } else {
              new_args.push(a.clone());
            }
          }
          Expr::FunctionCall {
            name: name.clone(),
            args: new_args.into(),
          }
        }
        "Text" | "Inset" => {
          // Second arg (if present) is position
          let mut new_args = args.clone();
          if new_args.len() >= 2
            && let Expr::Integer(idx) = &new_args[1]
          {
            new_args[1] = index_to_coord(*idx, coords);
          }
          Expr::FunctionCall {
            name: name.clone(),
            args: new_args,
          }
        }
        _ => {
          // For everything else (Style, directives, etc.), recurse
          Expr::FunctionCall {
            name: name.clone(),
            args: args
              .iter()
              .map(|a| resolve_graphics_complex_indices(a, coords))
              .collect(),
          }
        }
      }
    }
    _ => expr.clone(),
  }
}

/// Resolve a primitive argument that expects point(s).
/// An integer becomes a coordinate, a list of integers becomes a list of coordinates,
/// a list of lists of integers becomes a list of list of coordinates.
fn resolve_primitive_arg(arg: &Expr, coords: &[(f64, f64)]) -> Expr {
  match arg {
    Expr::Integer(idx) => index_to_coord(*idx, coords),
    Expr::List(items) => {
      if !items.is_empty()
        && items.iter().all(|e| matches!(e, Expr::Integer(_)))
      {
        // List of integer indices → list of coordinate pairs
        Expr::List(
          items
            .iter()
            .map(|e| {
              if let Expr::Integer(idx) = e {
                index_to_coord(*idx, coords)
              } else {
                e.clone()
              }
            })
            .collect(),
        )
      } else {
        // Could be list of lists (multi-segment line) or mixed
        Expr::List(
          items
            .iter()
            .map(|e| resolve_primitive_arg(e, coords))
            .collect(),
        )
      }
    }
    _ => arg.clone(),
  }
}

/// Convert a 1-based index to a coordinate pair expression {x, y}.
fn index_to_coord(idx: i128, coords: &[(f64, f64)]) -> Expr {
  let i = (idx as usize).wrapping_sub(1);
  if i < coords.len() {
    let (x, y) = coords[i];
    Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
  } else {
    // Out of bounds — return as-is
    Expr::Integer(idx)
  }
}

// ── Primitive parsers ────────────────────────────────────────────────────

fn parse_point(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  if let Some(pt) = expr_to_point(&args[0]) {
    prims.push(Primitive::PointSingle {
      x: pt.0,
      y: pt.1,
      style: style.clone(),
    });
  } else if let Some(pts) = expr_to_point_list(&args[0]) {
    prims.push(Primitive::PointMulti {
      points: pts,
      style: style.clone(),
    });
  }
}

fn parse_line(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  // Line[{{x1,y1},{x2,y2},...}] or Line[{seg1, seg2, ...}] for multiple segments
  if let Some(pts) = expr_to_point_list(&args[0]) {
    prims.push(Primitive::Line {
      segments: vec![pts],
      style: style.clone(),
    });
  } else if let Expr::List(items) = &args[0] {
    // Multi-segment: each item is a point list
    let mut segments = Vec::new();
    for item in items {
      if let Some(pts) = expr_to_point_list(item) {
        segments.push(pts);
      }
    }
    if !segments.is_empty() {
      prims.push(Primitive::Line {
        segments,
        style: style.clone(),
      });
    }
  }
}

fn parse_circle(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  let (cx, cy) = if args.is_empty() {
    (0.0, 0.0)
  } else {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  };
  let (rx, ry) = if args.len() >= 2 {
    if let Some((a, b)) = expr_to_point(&args[1]) {
      (a, b)
    } else {
      let r = expr_to_f64(&args[1]).unwrap_or(1.0);
      (r, r)
    }
  } else {
    (1.0, 1.0)
  };
  // Circle[center, r, {theta1, theta2}] draws only the arc over that angular
  // range (an open curve stroked on one side), not the whole circle.
  let angles = args.get(2).and_then(expr_to_point);
  prims.push(Primitive::CircleArc {
    cx,
    cy,
    rx,
    ry,
    angles,
    style: style.clone(),
  });
}

fn parse_disk(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  let (cx, cy) = if args.is_empty() {
    (0.0, 0.0)
  } else {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  };
  let (rx, ry) = if args.len() >= 2 {
    if let Some((a, b)) = expr_to_point(&args[1]) {
      (a, b)
    } else {
      let r = expr_to_f64(&args[1]).unwrap_or(1.0);
      (r, r)
    }
  } else {
    (1.0, 1.0)
  };
  // Disk[center, r, {angle1, angle2}] creates a sector
  if args.len() >= 3
    && let Some((a1, a2)) = expr_to_point(&args[2])
  {
    prims.push(Primitive::DiskSector {
      cx,
      cy,
      rx,
      ry,
      angle1: a1,
      angle2: a2,
      style: style.clone(),
    });
    return;
  }
  prims.push(Primitive::Disk {
    cx,
    cy,
    rx,
    ry,
    style: style.clone(),
  });
}

/// `Sphere[…]` and `Ball[…]` inside a two-dimensional `Graphics`. The region
/// functions hand back spheres and balls whatever the dimension —
/// `Circumsphere` of three points in the plane is a `Sphere` — and in the
/// plane a sphere is the circle bounding it and a ball the filled disk, so
/// each draws as its planar namesake. A centre with any other number of
/// coordinates belongs to a `Graphics3D` and draws nothing here.
fn parse_sphere(
  filled: bool,
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  // `Sphere[n]` / `Ball[n]` is the unit sphere/ball at the origin in `n`
  // dimensions; only the planar one has anything to draw.
  if let [Expr::Integer(dimension)] = args {
    if *dimension == 2 {
      emit_sphere(filled, (0.0, 0.0), 1.0, style, prims);
    }
    return;
  }
  let Some(radius) = (match args.get(1) {
    Some(r) => expr_to_f64(r),
    None => Some(1.0),
  }) else {
    return;
  };
  // One centre, or a list of them — `Sphere[{p1, p2}, r]` draws one sphere
  // of radius `r` around each point.
  let centers: Vec<(f64, f64)> = match args.first() {
    Some(Expr::List(items))
      if !items.is_empty()
        && items.iter().all(|i| matches!(i, Expr::List(_))) =>
    {
      items.iter().filter_map(expr_to_point).collect()
    }
    Some(single) => expr_to_point(single).into_iter().collect(),
    None => Vec::new(),
  };
  for center in centers {
    emit_sphere(filled, center, radius, style, prims);
  }
}

fn emit_sphere(
  filled: bool,
  (cx, cy): (f64, f64),
  r: f64,
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  prims.push(if filled {
    Primitive::Disk {
      cx,
      cy,
      rx: r,
      ry: r,
      style: style.clone(),
    }
  } else {
    Primitive::CircleArc {
      cx,
      cy,
      rx: r,
      ry: r,
      angles: None,
      style: style.clone(),
    }
  });
}

fn parse_rectangle(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  let (x_min, y_min) = if args.is_empty() {
    (0.0, 0.0)
  } else {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  };
  let (x_max, y_max) = if args.len() >= 2 {
    expr_to_point(&args[1]).unwrap_or((1.0, 1.0))
  } else {
    (x_min + 1.0, y_min + 1.0)
  };
  // Wolfram accepts the two corners in any order; normalize so the primitive
  // always has min <= max (a reversed pair would otherwise render as a rect
  // with negative width/height, which SVG drops entirely).
  prims.push(Primitive::RectPrim {
    x_min: x_min.min(x_max),
    y_min: y_min.min(y_max),
    x_max: x_min.max(x_max),
    y_max: y_min.max(y_max),
    style: style.clone(),
  });
}

fn parse_polygon(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  // Polygon[{p1, p2, …}] — one polygon — or Polygon[{poly1, poly2, …}],
  // a list of them, which is what mapping a face-index table over a vertex
  // list produces.
  if let Some(pts) = expr_to_point_list(&args[0]) {
    prims.push(Primitive::PolygonPrim {
      points: pts,
      holes: Vec::new(),
      style: style.clone(),
    });
  } else if let Some((outer, holes)) =
    crate::functions::polygon_holes::split_holes(&args[0], &expr_to_point_list)
  {
    // Polygon[outer -> holes] — a polygon with the hole boundaries cut
    // out of it.
    prims.push(Primitive::PolygonPrim {
      points: outer,
      holes,
      style: style.clone(),
    });
  } else if let Expr::List(items) = &args[0] {
    for item in items {
      if let Some(pts) = expr_to_point_list(item) {
        prims.push(Primitive::PolygonPrim {
          points: pts,
          holes: Vec::new(),
          style: style.clone(),
        });
      } else if let Some((outer, holes)) =
        crate::functions::polygon_holes::split_holes(item, &expr_to_point_list)
      {
        prims.push(Primitive::PolygonPrim {
          points: outer,
          holes,
          style: style.clone(),
        });
      }
    }
  }
}

/// Parallelogram[p, {v1, v2}] (default: unit square {0,0} + {{0,1},{1,0}})
/// — a filled quadrilateral with corners p, p+v1, p+v1+v2, p+v2.
fn parse_parallelogram(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  let (p, v1, v2) = if args.is_empty() {
    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
  } else if args.len() == 2 {
    let Some(p) = expr_to_point(&args[0]) else {
      return;
    };
    let Expr::List(vecs) = &args[1] else {
      return;
    };
    if vecs.len() != 2 {
      return;
    }
    let (Some(v1), Some(v2)) =
      (expr_to_point(&vecs[0]), expr_to_point(&vecs[1]))
    else {
      return;
    };
    (p, v1, v2)
  } else {
    return;
  };
  let points = vec![
    p,
    (p.0 + v1.0, p.1 + v1.1),
    (p.0 + v1.0 + v2.0, p.1 + v1.1 + v2.1),
    (p.0 + v2.0, p.1 + v2.1),
  ];
  prims.push(Primitive::PolygonPrim {
    points,
    holes: Vec::new(),
    style: style.clone(),
  });
}

/// HalfPlane[{p1, p2}, w] — the half plane swept by translating the line
/// through p1 and p2 along w. HalfPlane[p, v, w] — the same with the line
/// given as point p and direction v.
fn parse_half_plane(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  let (p, v, w) = match args.len() {
    2 => {
      let Expr::List(pts) = &args[0] else {
        return;
      };
      if pts.len() != 2 {
        return;
      }
      let (Some(p1), Some(p2)) =
        (expr_to_point(&pts[0]), expr_to_point(&pts[1]))
      else {
        return;
      };
      let Some(w) = expr_to_point(&args[1]) else {
        return;
      };
      (p1, (p2.0 - p1.0, p2.1 - p1.1), w)
    }
    3 => {
      let (Some(p), Some(v), Some(w)) = (
        expr_to_point(&args[0]),
        expr_to_point(&args[1]),
        expr_to_point(&args[2]),
      ) else {
        return;
      };
      (p, v, w)
    }
    _ => return,
  };
  if (v.0 == 0.0 && v.1 == 0.0) || (w.0 == 0.0 && w.1 == 0.0) {
    return;
  }
  prims.push(Primitive::HalfPlanePrim {
    p,
    v,
    w,
    full: false,
    style: style.clone(),
  });
}

/// InfinitePlane[{p1, p2, p3}] / InfinitePlane[p, {v1, v2}] — with 2D
/// coordinates the plane covers the entire viewport.
fn parse_infinite_plane(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  let p = match args.len() {
    1 => {
      let Expr::List(pts) = &args[0] else {
        return;
      };
      if pts.len() != 3 {
        return;
      }
      let Some(p1) = expr_to_point(&pts[0]) else {
        return;
      };
      if expr_to_point(&pts[1]).is_none() || expr_to_point(&pts[2]).is_none() {
        return;
      }
      p1
    }
    2 => {
      let Some(p) = expr_to_point(&args[0]) else {
        return;
      };
      p
    }
    _ => return,
  };
  prims.push(Primitive::HalfPlanePrim {
    p,
    v: (1.0, 0.0),
    w: (0.0, 1.0),
    full: true,
    style: style.clone(),
  });
}

fn parse_regular_polygon(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  // RegularPolygon[n] — unit circumradius at origin
  // RegularPolygon[{cx, cy}, r, n] — at center with circumradius r
  let (cx, cy, r, n) = match args.len() {
    1 => {
      let n = expr_to_f64(&args[0]).unwrap_or(0.0) as usize;
      (0.0, 0.0, 1.0, n)
    }
    3 => {
      let center = expr_to_point(&args[0]).unwrap_or((0.0, 0.0));
      let r = expr_to_f64(&args[1]).unwrap_or(1.0);
      let n = expr_to_f64(&args[2]).unwrap_or(0.0) as usize;
      (center.0, center.1, r, n)
    }
    _ => return,
  };
  if n < 3 {
    return;
  }
  // Generate vertices starting from top (Pi/2), going counterclockwise
  let pts: Vec<(f64, f64)> = (0..n)
    .map(|k| {
      let angle = std::f64::consts::FRAC_PI_2
        + 2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
      (cx + r * angle.cos(), cy + r * angle.sin())
    })
    .collect();
  prims.push(Primitive::PolygonPrim {
    points: pts,
    holes: Vec::new(),
    style: style.clone(),
  });
}

fn parse_arrow(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  // Arrow[{seg1, seg2, …}] — a list of paths — draws one arrow per path.
  if expr_to_point_list(&args[0]).is_none()
    && let Expr::List(items) = &args[0]
  {
    for item in items {
      if expr_to_point_list(item).is_some() {
        let mut sub: Vec<Expr> = vec![item.clone()];
        sub.extend(args[1..].iter().cloned());
        parse_arrow(&sub, style, prims);
      }
    }
    return;
  }
  // Arrow[{{x1,y1},{x2,y2},...}] or Arrow[{{x1,y1},...}, {s1, s2}]
  if let Some(pts) = expr_to_point_list(&args[0])
    && pts.len() >= 2
  {
    let setback = if args.len() >= 2 {
      match &args[1] {
        Expr::List(items) if items.len() == 2 => {
          let s1 = expr_to_f64(&items[0]).unwrap_or(0.0);
          let s2 = expr_to_f64(&items[1]).unwrap_or(0.0);
          (s1, s2)
        }
        other => {
          let s = expr_to_f64(other).unwrap_or(0.0);
          (s, s)
        }
      }
    } else {
      (0.0, 0.0)
    };
    // A head that carries its own graphic (a label, a marker) is drawn as
    // that graphic, centred on the point of the arrow it sits at — the
    // renderer's triangle is only for the plain sizes.
    for head in style.arrowheads.iter().flatten() {
      let Some(graphic) = &head.graphic else {
        continue;
      };
      let (x, y) = point_along_path(&pts, head.position);
      let mut inner_style = style.clone();
      inner_style.arrowheads = None;
      let mut inner = Vec::new();
      collect_primitives(
        graphic,
        &mut inner_style,
        &mut inner,
        &mut Vec::new(),
      );
      for p in &inner {
        prims.push(translate_primitive(p, x, y));
      }
    }
    prims.push(Primitive::ArrowPrim {
      points: pts,
      setback,
      style: style.clone(),
    });
  }
}

/// The point a fraction `t` of the way along a polyline, measured by arc
/// length. `t` outside `[0, 1]` clamps to the ends.
fn point_along_path(pts: &[(f64, f64)], t: f64) -> (f64, f64) {
  if pts.is_empty() {
    return (0.0, 0.0);
  }
  let seg_len: Vec<f64> = pts
    .windows(2)
    .map(|w| ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt())
    .collect();
  let total: f64 = seg_len.iter().sum();
  if total <= 0.0 {
    return pts[0];
  }
  let mut want = t.clamp(0.0, 1.0) * total;
  for (i, &len) in seg_len.iter().enumerate() {
    if want <= len || i + 1 == seg_len.len() {
      let f = if len > 0.0 {
        (want / len).clamp(0.0, 1.0)
      } else {
        0.0
      };
      return (
        pts[i].0 + f * (pts[i + 1].0 - pts[i].0),
        pts[i].1 + f * (pts[i + 1].1 - pts[i].1),
      );
    }
    want -= len;
  }
  pts[pts.len() - 1]
}

/// Render the text content of a `Text`/`Inset` label. Inside a graphic the
/// label is typeset, so display wrappers resolve to their formatted text:
/// `Style` unwraps, `Row` concatenates, and `NumberForm` (and friends)
/// render their formatted number — e.g.
/// `Row[{Style[NumberForm[50., {3, 1}], 18], Style["% shaded", 18]}]`
/// becomes "50.0% shaded". Plain strings pass through verbatim; anything
/// else falls back to `ToString`'s default form.
fn graphics_text_content(expr: &Expr) -> String {
  match expr {
    // A string carrying inline `\!\(\*…\)` box notation — the front end's
    // linear-syntax form for a typeset sub-expression, e.g. the label a
    // Demonstration writes as `Text["\!\(\*SubscriptBox[\(X\), \(3\)]\)",
    // pos]` for "X₃". Fold each box segment into the Unicode glyphs it
    // typesets to, the same as `Subscript`/`Superscript` below, instead of
    // drawing the private-use markers and box source literally.
    Expr::String(s) if s.contains(crate::functions::string_ast::BOX_START) => {
      inline_box_label_runs(s, false).map_or_else(|| s.clone(), |runs| flatten_label_runs(&runs))
    }
    Expr::String(s) => s.clone(),
    // Text inside a picture is typeset, not printed: a mathematical
    // constant shows as its glyph there, the way it does in a notebook.
    // (Script-mode output still writes the name, which is what
    // wolframscript prints.)
    Expr::Identifier(name) | Expr::Constant(name)
      if typeset_constant_glyph(name).is_some() =>
    {
      typeset_constant_glyph(name).unwrap().to_string()
    }
    Expr::FunctionCall { name, args }
      if is_style_wrapper(name) && !args.is_empty() =>
    {
      graphics_text_content(&args[0])
    }
    // `TraditionalForm[expr]` typesets `expr`, so the text it contributes
    // is the flattened traditional-notation box tree — the same thing
    // `expr_to_svg_markup` draws — rather than `expr`'s InputForm. Keeping
    // the two in step is what makes the measured width match the drawing.
    Expr::FunctionCall { name, args }
      if name == "TraditionalForm" && args.len() == 1 =>
    {
      box_expr_to_plain(
        &crate::evaluator::dispatch::complex_and_special::
          expr_to_box_form_traditional(&args[0]),
      )
    }
    // `StandardForm[expr]` / `OutputForm[expr]` ask for `expr` to be set in
    // that form; inside a picture everything is typeset already, so the
    // wrapper contributes no text of its own. Without this the box markup
    // it serializes to leaked into the picture as literal source.
    Expr::FunctionCall { name, args }
      if matches!(
        name.as_str(),
        "StandardForm" | "OutputForm" | "DisplayForm"
      ) && args.len() == 1 =>
    {
      graphics_text_content(&args[0])
    }
    Expr::FunctionCall { name, args }
      if name == "Row"
        && !args.is_empty()
        && matches!(args.first(), Some(Expr::List(_))) =>
    {
      let Some(Expr::List(items)) = args.first() else {
        unreachable!()
      };
      // A `Spacer[n]` among the items is a gap, not something to print.
      let parts: Vec<String> = items
        .iter()
        .map(|item| match spacer_gap_text(item) {
          Some(gap) => gap,
          None => graphics_text_content(item),
        })
        .collect();
      match args.get(1) {
        // `Spacer[n]` separates with a gap rather than printing itself.
        Some(sep) if spacer_gap_text(sep).is_some() => {
          parts.join(&spacer_gap_text(sep).unwrap_or_default())
        }
        Some(sep) => parts.join(&graphics_text_content(sep)),
        None => parts.concat(),
      }
    }
    // `Subscript`/`Superscript` typeset as scripts, not as the two-line
    // OutputForm box `ToString` would give: a label reading `N` over ` D`
    // is not what the picture is meant to show. `expr_to_label` already
    // folds them into the Unicode script characters for plot labels, so a
    // `Text` label written the same way reads the same way.
    Expr::FunctionCall { name, args }
      if (name == "Subscript" || name == "Superscript") && args.len() >= 2 =>
    {
      crate::functions::chart::expr_to_label(expr)
        .unwrap_or_else(|| crate::syntax::expr_to_string(expr))
    }
    _ => {
      let text = match crate::functions::string_ast::to_string_ast(
        std::slice::from_ref(expr),
      ) {
        Ok(Expr::String(ref s)) => s.clone(),
        _ => crate::syntax::expr_to_string(expr),
      };
      typeset_constants_in_text(&text)
    }
  }
}

/// Typeset the named constants inside a flattened label. A constant on its
/// own is already drawn as its glyph; one inside a larger expression reaches
/// here as its name, so `Text[50 Degree]` would read "50 Degree" where
/// Wolfram draws "50 °". Only whole words are replaced, and only the
/// protected constants of [`typeset_constant_glyph`] — no user symbol can
/// carry one of those names.
fn typeset_constants_in_text(text: &str) -> String {
  const NAMES: [&str; 7] = [
    "Infinity",
    "Degree",
    "EulerGamma",
    "GoldenRatio",
    "Pi",
    "E",
    "I",
  ];
  if !NAMES.iter().any(|n| text.contains(n)) {
    return text.to_string();
  }
  let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
  let bytes: Vec<char> = text.chars().collect();
  let mut out = String::with_capacity(text.len());
  let mut i = 0;
  'outer: while i < bytes.len() {
    if i == 0 || !is_word(bytes[i - 1]) {
      for name in NAMES {
        let n: Vec<char> = name.chars().collect();
        if bytes[i..].starts_with(&n[..])
          && bytes.get(i + n.len()).is_none_or(|c| !is_word(*c))
          && let Some(glyph) = typeset_constant_glyph(name)
        {
          out.push_str(glyph);
          i += n.len();
          continue 'outer;
        }
      }
    }
    out.push(bytes[i]);
    i += 1;
  }
  out
}

/// The blank a `Spacer[n]` stands for when a layout is flattened to text,
/// `None` for anything else. Used both for a spacer among a `Row`'s items
/// and for one given as its separator, so the two agree.
fn spacer_gap_text(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Spacer" {
    return None;
  }
  let width = args.first().and_then(expr_to_f64).unwrap_or(1.0);
  Some(" ".repeat((width.max(0.0).round() as usize).max(1)))
}

/// The primitives an `Inset[obj, pos, opos, size, dirs]` contributes when
/// `obj` is itself a picture: the object's own primitives, scaled into the
/// inset's box, turned to face `dirs`, and moved to `pos`. Returns `None`
/// for anything else, which `Inset` then places as text.
///
/// `pos` defaults to the origin, `opos` (the point *of the object* that
/// lands on `pos`) to its centre, and `size` to the object's own extent.
/// The pixel size of the plate a `Button[label, …]` draws as: the label at
/// the standard 14-point text metric, with Wolfram's padding around it.
fn button_plate_size(label: &str) -> (f64, f64) {
  const CHAR_W: f64 = 8.4;
  const PAD_X: f64 = 12.0;
  let w = label.chars().count() as f64 * CHAR_W + 2.0 * PAD_X;
  (w.max(28.0), 26.0)
}

/// A `Button`'s plate, as a standalone SVG to be inset into a picture.
fn button_plate_svg(label: &str) -> String {
  let (w, h) = button_plate_size(label);
  format!(
    "<svg width=\"{w:.0}\" height=\"{h:.0}\" viewBox=\"0 0 {w:.0} {h:.0}\" xmlns=\"http://www.w3.org/2000/svg\">\n\
     <rect x=\"0.5\" y=\"0.5\" width=\"{:.1}\" height=\"{:.1}\" rx=\"4\" ry=\"4\" fill=\"#FDFDFD\" stroke=\"#BFBFBF\" stroke-width=\"1\"/>\n\
     <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"Atkinson Hyperlegible Next, sans-serif\" font-size=\"14\" fill=\"#000000\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n\
     </svg>\n",
    w - 1.0,
    h - 1.0,
    w / 2.0,
    h / 2.0,
    svg_escape(label)
  )
}

/// Peel a top-level `Style[content, dirs…]`/`StyleForm[…]` wrapper so
/// callers can pattern-match the payload underneath — e.g. `Inset[Style[img,
/// Magnification -> .2], pos]` still embeds `img` as a picture rather than
/// falling through to the plain-text path just because it is styled. The
/// style directives themselves (font, magnification, …) are not applied;
/// getting the picture on screen at all matters more than honoring them.
fn peel_style_wrapper(expr: &Expr) -> &Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if is_style_wrapper(name) && !args.is_empty() =>
    {
      peel_style_wrapper(&args[0])
    }
    other => other,
  }
}

fn inset_primitives(
  args: &[Expr],
  errors: &mut Vec<String>,
) -> Option<Vec<Primitive>> {
  // A graphic that has already been rendered — a `Plot`, a `Show`, any
  // picture whose symbolic content was not kept — is embedded whole, at
  // its own size. That is what an inset is: the object keeps the size it
  // would have on its own, wherever the plot range puts its anchor.
  //
  // A three-dimensional object goes the same way: its projection is fixed
  // by its own `ViewPoint`, so there is nothing to re-project into this
  // picture's coordinates. `Graphics3D[…]` that has not been rendered yet
  // is rendered here first (a Demonstration builds its little inset scene
  // inside the body and insets the variable), which is also what keeps it
  // from falling through to the text path and printing `-Graphics3D-`.
  let anchor = args.get(1).and_then(expr_to_anchor);
  let rendered;
  let image_svg;
  let embedded = match peel_style_wrapper(&args[0]) {
    Expr::Graphics {
      svg,
      structure: None,
      ..
    } => Some(svg),
    Expr::Graphics {
      svg, is_3d: true, ..
    } => Some(svg),
    // A rasterized picture (e.g. from `Rasterize[…]` or `Import`) draws at
    // its own pixel size, the same as a rendered `Graphics` above — there is
    // no symbolic content to fold into this picture's coordinate system.
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => {
      image_svg = crate::functions::image_ast::image_to_html_img(
        *width, *height, *channels, data,
      );
      Some(&image_svg)
    }
    // A picture given symbolically normally has its primitives folded into
    // this one (below), which is what lets it share the coordinate system.
    // That cannot be done from a `Scaled` anchor — the range it names is
    // only known once every primitive is in — so such an inset is rendered
    // to its own picture and embedded whole instead.
    call @ Expr::FunctionCall { name, .. }
      if name == "Graphics3D"
        || name == "Graphics3DBox"
        || (anchor.is_some_and(|(_, _, scaled)| scaled)
          && (name == "Graphics" || name == "GraphicsBox")) =>
    {
      rendered = crate::evaluator::expr_to_svg(call);
      (!rendered.is_empty()).then_some(&rendered)
    }
    _ => None,
  };
  if let Some(svg) = embedded
    && let Some(dims) = parse_svg_dimensions(svg)
    && let Some((x, y, scaled)) = anchor
  {
    return Some(vec![Primitive::InsetGraphic {
      svg: svg.clone(),
      x,
      y,
      w: dims.nat_w,
      h: dims.nat_h,
      scaled,
    }]);
  }
  // `Inset[Button[label, action], pos]` — the control a puzzle
  // Demonstration puts inside its picture. It draws as the button itself:
  // a rounded white plate with a thin border and the label centred,
  // sized to the label the way Wolfram sizes one.
  if let Expr::FunctionCall { name, args: bargs } = &args[0]
    && name == "Button"
    && !bargs.is_empty()
    && let Some((x, y)) = args.get(1).and_then(expr_to_point)
  {
    let label = graphics_text_content(&bargs[0]);
    return Some(vec![Primitive::InsetGraphic {
      svg: button_plate_svg(&label),
      x,
      y,
      w: button_plate_size(&label).0,
      h: button_plate_size(&label).1,
      scaled: false,
    }]);
  }
  // The object: `Graphics[…]` (or its box form), a rendered graphic that
  // kept its symbolic content, or a bare `Raster`/`Image`.
  let content: Expr = match &args[0] {
    Expr::FunctionCall { name, args: gargs }
      if (name == "Graphics" || name == "GraphicsBox") && !gargs.is_empty() =>
    {
      gargs[0].clone()
    }
    Expr::Graphics {
      structure: Some(inner),
      is_3d: false,
      ..
    } => match inner.as_ref() {
      Expr::FunctionCall { name, args: gargs }
        if (name == "Graphics" || name == "GraphicsBox")
          && !gargs.is_empty() =>
      {
        gargs[0].clone()
      }
      other => other.clone(),
    },
    obj @ (Expr::Image { .. } | Expr::FunctionCall { .. }) => {
      // `Raster[…]`/`RasterBox[…]`/`Image[…]` draw as themselves.
      match obj {
        Expr::Image { .. } => obj.clone(),
        Expr::FunctionCall { name, .. }
          if name == "Raster" || name == "RasterBox" || name == "Image" =>
        {
          obj.clone()
        }
        _ => return None,
      }
    }
    _ => return None,
  };

  let mut inner_style = StyleState::default();
  let mut inner: Vec<Primitive> = Vec::new();
  collect_primitives(&content, &mut inner_style, &mut inner, errors);
  if inner.is_empty() {
    return None;
  }

  let mut bb = BBox::empty();
  for p in &inner {
    bb.merge(&primitive_bbox(p));
  }
  if bb.is_empty() {
    return None;
  }
  let (cx, cy) = (
    f64::midpoint(bb.x_min, bb.x_max),
    f64::midpoint(bb.y_min, bb.y_max),
  );
  let (w, h) = (bb.x_max - bb.x_min, bb.y_max - bb.y_min);

  // `size` may be `{w, h}`, a single number (both sides), or Automatic.
  let (target_w, target_h) = match args.get(3) {
    Some(spec) => match expr_to_point(spec) {
      Some(wh) => (Some(wh.0), Some(wh.1)),
      None => match expr_to_f64(spec) {
        Some(v) => (Some(v), Some(v)),
        None => (None, None),
      },
    },
    None => (None, None),
  };
  let sx = match target_w {
    Some(tw) if w > 0.0 => tw / w,
    _ => 1.0,
  };
  let sy = match target_h {
    Some(th) if h > 0.0 => th / h,
    _ => 1.0,
  };

  // `dirs` is `{Automatic, {dx, dy}}` (or a bare direction vector): the
  // object turns so its x axis points that way.
  let angle = args
    .get(4)
    .and_then(|spec| match spec {
      Expr::List(items) if items.len() == 2 => {
        expr_to_point(&items[1]).or_else(|| expr_to_point(spec))
      }
      other => expr_to_point(other),
    })
    .map(|(dx, dy)| dy.atan2(dx))
    .filter(|a| a.is_finite() && *a != 0.0);

  let pos = args.get(1).and_then(expr_to_point).unwrap_or((0.0, 0.0));
  // `opos` names the point of the object that lands on `pos`, in the
  // object's own coordinates (Automatic = its centre).
  let (ox, oy) = args.get(2).and_then(expr_to_point).unwrap_or((cx, cy));

  let placed = inner
    .iter()
    .map(|p| {
      let p = scale_primitive(p, cx, cy, sx, sy);
      let p = match angle {
        Some(a) => rotate_primitive(&p, cx, cy, a),
        None => p,
      };
      // After scaling about the centre, the alignment point moved with it.
      let (ax, ay) = (cx + (ox - cx) * sx, cy + (oy - cy) * sy);
      translate_primitive(&p, pos.0 - ax, pos.1 - ay)
    })
    .collect();
  Some(placed)
}

/// Fold the font directives of a `Row[{Style[…], …}]` label into the
/// label's own style, but only the ones every part agrees on. See the call
/// site in [`parse_text`].
fn apply_agreed_row_part_styles(content: &Expr, style: &mut StyleState) {
  let Expr::FunctionCall { name, args } = content else {
    return;
  };
  if name != "Row" || args.is_empty() {
    return;
  }
  let Expr::List(items) = &args[0] else {
    return;
  };
  if items.is_empty() {
    return;
  }
  let mut parts = Vec::with_capacity(items.len());
  for item in items {
    let Expr::FunctionCall {
      name: iname,
      args: sargs,
    } = item
    else {
      return; // an unstyled part keeps the label's own setting
    };
    if !is_style_wrapper(iname) || sargs.is_empty() {
      return;
    }
    let mut st = style.clone();
    for d in style_directives_in_application_order(&sargs[1..]) {
      apply_directive(d, &mut st);
      apply_text_style_directive(d, &mut st);
    }
    parts.push(st);
  }
  let first = &parts[0];
  if parts.iter().all(|p| p.font_size == first.font_size) {
    style.font_size = first.font_size;
  }
  if parts.iter().all(|p| p.font_weight == first.font_weight) {
    style.font_weight.clone_from(&first.font_weight);
  }
  if parts.iter().all(|p| p.font_style == first.font_style) {
    style.font_style.clone_from(&first.font_style);
  }
  if parts.iter().all(|p| p.color == first.color) {
    style.color = first.color;
  }
}

fn parse_text(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  // Text[str, {x, y}] or Text[Style[str, ...], {x, y}]
  let mut local_style = style.clone();
  // `Framed[content, opts…]` boxes the label: a border around it and, with
  // `Background -> colour`, a panel behind it. Peel it off first so the
  // `Style` directives it usually wraps still reach the primitive.
  let (framed_body, frame_opts) = match &args[0] {
    Expr::FunctionCall { name, args: fargs }
      if name == "Framed" && !fargs.is_empty() =>
    {
      (&fargs[0], &fargs[1..])
    }
    other => (other, &[] as &[Expr]),
  };
  // A top-level `Style[content, dirs…]` carries text directives (font size,
  // weight, color) that apply to the whole label; peel it off and apply them
  // to the primitive, then render the inner content.
  let content = match framed_body {
    Expr::FunctionCall { name, args: sargs }
      if is_style_wrapper(name) && !sargs.is_empty() =>
    {
      for d in style_directives_in_application_order(&sargs[1..]) {
        apply_directive(d, &mut local_style);
        apply_text_style_directive(d, &mut local_style);
      }
      &sargs[0]
    }
    other => other,
  };
  // A label written as a `Row` of styled parts — the `f(x)` a
  // Demonstration writes beside a point — carries its font directives on
  // the parts rather than on the label itself. The label is drawn as one
  // run, so take the directives every part agrees on: a row whose parts
  // all ask for 20 point is a 20-point label, while parts that disagree
  // (an italic letter next to an upright bracket) leave the label's own
  // setting alone.
  apply_agreed_row_part_styles(content, &mut local_style);
  let text = graphics_text_content(content);

  let (x, y, scaled) = if args.len() >= 2 {
    expr_to_anchor(&args[1]).unwrap_or((0.0, 0.0, false))
  } else {
    (0.0, 0.0, false)
  };
  let offset = args.get(2).and_then(text_offset).unwrap_or((0.0, 0.0));
  // `Text[expr, pos, offset, direction]`: rotate the label so its baseline
  // runs parallel to `direction`, a vector in the same data coordinates as
  // `pos`. Left `None` for the common `Automatic`/omitted case.
  let direction = args.get(3).and_then(text_direction);
  // `Background -> colour`, written either as a trailing option of the
  // `Text` or as a directive of the `Style` it wraps.
  fn option_value<'a>(opts: &'a [Expr], key: &str) -> Option<&'a Expr> {
    opts.iter().find_map(|o| match o {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(k) if k == key) => {
        Some(replacement.as_ref())
      }
      _ => None,
    })
  }
  let background_of =
    |opts: &[Expr]| option_value(opts, "Background").and_then(parse_color);
  // A `Style` the label is written inside — around the `Text` or around its
  // content — leaves its background in the style state, which is how
  // `Style[Text[…], Background -> White]` keeps a distance label legible over
  // the line it sits on.
  let background = background_of(frame_opts)
    .or_else(|| background_of(&args[1..]))
    .or(local_style.text_background);
  // Wolfram draws a `Framed` box with a thin border unless the label asks
  // for none; an explicit `FrameStyle -> colour` recolours it.
  let is_framed = matches!(
    &args[0],
    Expr::FunctionCall { name, args: fargs } if name == "Framed" && !fargs.is_empty()
  );
  let frame = if is_framed {
    match option_value(frame_opts, "FrameStyle") {
      Some(Expr::Identifier(s)) if s == "None" => None,
      Some(spec) => parse_color(spec).or(Some(Color::new(0.0, 0.0, 0.0))),
      None => Some(Color::new(0.0, 0.0, 0.0)),
    }
  } else {
    None
  };

  prims.push(Primitive::TextPrim {
    text,
    x,
    y,
    offset,
    background,
    frame,
    scaled,
    direction,
    style: local_style,
  });
}

/// The rotation direction of `Text[expr, pos, offset, direction]`: a plain
/// `{dx, dy}` vector, or `None` for `Automatic` (no rotation) or anything
/// else that isn't a 2-vector.
fn text_direction(spec: &Expr) -> Option<(f64, f64)> {
  match spec {
    Expr::List(items) if items.len() == 2 => {
      Some((expr_to_f64(&items[0])?, expr_to_f64(&items[1])?))
    }
    _ => None,
  }
}

/// The alignment offset of `Text[expr, pos, offset]` / `Inset[…]`: a pair
/// running from -1 (left/bottom) to 1 (right/top), written either as
/// numbers or with the alignment symbols `Left`/`Center`/`Right` and
/// `Bottom`/`Center`/`Top`.
fn text_offset(spec: &Expr) -> Option<(f64, f64)> {
  fn component(e: &Expr, horizontal: bool) -> Option<f64> {
    if let Expr::Identifier(s) = e {
      return match (s.as_str(), horizontal) {
        ("Left", true) | ("Bottom", false) => Some(-1.0),
        ("Center" | "Automatic" | "Axis" | "Baseline", _) => Some(0.0),
        ("Right", true) | ("Top", false) => Some(1.0),
        _ => None,
      };
    }
    expr_to_f64(e)
  }
  match spec {
    Expr::List(items) if items.len() == 2 => {
      Some((component(&items[0], true)?, component(&items[1], false)?))
    }
    _ => None,
  }
}

fn parse_bezier(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  if let Some(pts) = expr_to_point_list(&args[0])
    && pts.len() >= 2
  {
    prims.push(Primitive::BezierCurvePrim {
      points: pts,
      style: style.clone(),
    });
  }
}

/// Parse PolarCurve[r, {t, t0, t1}] into a stroked curve (`filled` =
/// false) or the region it encloses (`filled` = true, used for
/// FilledPolarCurve). The radius expression is sampled numerically over
/// the angle range and converted to Cartesian coordinates.
/// FilledPolarCurve[r, t] (bare variable, no range) fills the region
/// enclosed over the full period {t, 0, 2 Pi}.
fn parse_polar_curve(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
  filled: bool,
) {
  let (var, t_min, t_max) = match &args[1] {
    Expr::List(iter) if iter.len() == 3 => {
      let Expr::Identifier(var) = &iter[0] else {
        return;
      };
      let (Some(t_min), Some(t_max)) =
        (expr_to_f64(&iter[1]), expr_to_f64(&iter[2]))
      else {
        return;
      };
      (var, t_min, t_max)
    }
    // FilledPolarCurve[r, t] spans the full period 0…2π.
    Expr::Identifier(var) if filled => (var, 0.0, 2.0 * std::f64::consts::PI),
    _ => return,
  };
  if !t_min.is_finite() || !t_max.is_finite() || t_min == t_max {
    return;
  }

  const SAMPLES: usize = 300;
  let step = (t_max - t_min) / (SAMPLES - 1) as f64;
  let mut points = Vec::with_capacity(SAMPLES);
  for i in 0..SAMPLES {
    let t = t_min + i as f64 * step;
    if let Some(r) = crate::functions::plot::evaluate_at_point(&args[0], var, t)
      && r.is_finite()
    {
      points.push((r * t.cos(), r * t.sin()));
    }
  }
  if points.len() < 2 {
    return;
  }
  if filled {
    prims.push(Primitive::PolygonPrim {
      points,
      holes: Vec::new(),
      style: style.clone(),
    });
  } else {
    prims.push(Primitive::Line {
      segments: vec![points],
      style: style.clone(),
    });
  }
}

/// Render a top-level `PolarCurve[…]` / `FilledPolarCurve[…]` call as a
/// Graphics expression. Visual hosts (playground, studio, jupyter) display
/// curve objects graphically like Wolfram notebooks; the CLI keeps the
/// symbolic echo. Returns `None` when the arguments don't describe a
/// renderable curve (symbolic bounds etc.), so those stay symbolic.
pub fn polar_curve_to_graphics(name: &str, args: &[Expr]) -> Option<Expr> {
  let expr = unevaluated(name, args);
  // Check that the arguments actually parse into a drawable primitive
  // before rendering — otherwise an invalid call would show up as an
  // empty graphic instead of its symbolic form.
  let mut style = StyleState::default();
  let mut prims = Vec::new();
  let mut errors = Vec::new();
  collect_primitives(&expr, &mut style, &mut prims, &mut errors);
  if prims.is_empty() {
    return None;
  }
  let rendered = graphics_ast(std::slice::from_ref(&expr)).ok()?;
  if let Expr::Graphics {
    svg, is_3d, source, ..
  } = &rendered
  {
    // Report the curve head (like Region does) while rendering
    // identically to the wrapping Graphics.
    Some(Expr::Graphics {
      svg: svg.clone(),
      is_3d: *is_3d,
      source: source.clone(),
      head: Some(name.to_string()),
      structure: None,
    })
  } else {
    None
  }
}

/// Parse BSplineCurve[{pts...}] or BSplineCurve[{pts...}, SplineClosed -> True].
/// Evaluates the B-spline and converts to a Line primitive.
fn parse_bspline(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  if let Some(pts) = expr_to_point_list(&args[0])
    && pts.len() >= 2
  {
    // Check for SplineClosed -> True option
    let closed = args.iter().skip(1).any(|arg| {
      matches!(arg,
        Expr::Rule { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "SplineClosed")
          && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True")
      )
    });

    let control = if closed {
      // For closed splines, wrap the first (degree) points to the end
      let degree = 3usize.min(pts.len() - 1);
      let mut cp = pts.clone();
      for i in 0..degree {
        cp.push(pts[i]);
      }
      cp
    } else {
      pts
    };

    let sampled = evaluate_bspline(&control, 200);
    prims.push(Primitive::Line {
      segments: vec![sampled],
      style: style.clone(),
    });
  }
}

/// Evaluate a uniform B-spline curve of degree min(3, n-1) at `num_samples` points.
fn evaluate_bspline(
  control_points: &[(f64, f64)],
  num_samples: usize,
) -> Vec<(f64, f64)> {
  let n = control_points.len();
  if n < 2 {
    return control_points.to_vec();
  }

  let degree = 3usize.min(n - 1);
  let num_knots = n + degree + 1;

  // Clamped uniform knot vector
  let mut knots = Vec::with_capacity(num_knots);
  knots.extend(std::iter::repeat_n(0.0, degree + 1));
  let num_internal = num_knots - 2 * (degree + 1);
  for i in 1..=num_internal {
    knots.push(i as f64);
  }
  let max_knot = (num_internal + 1) as f64;
  knots.extend(std::iter::repeat_n(max_knot, degree + 1));

  let t_min = knots[degree];
  let t_max = knots[n];

  let mut result = Vec::with_capacity(num_samples);
  for i in 0..num_samples {
    let t = t_min + (t_max - t_min) * i as f64 / (num_samples - 1) as f64;
    let (mut x, mut y) = (0.0, 0.0);
    for j in 0..n {
      let b = bspline_basis(j, degree, t, &knots);
      x += b * control_points[j].0;
      y += b * control_points[j].1;
    }
    result.push((x, y));
  }
  result
}

/// Cox-de Boor recursion for B-spline basis function.
pub(crate) fn bspline_basis(i: usize, k: usize, t: f64, knots: &[f64]) -> f64 {
  if k == 0 {
    return if knots[i] <= t && t < knots[i + 1] {
      1.0
    } else if (t - knots[i + 1]).abs() < 1e-12
      && knots[i] < knots[i + 1]
      && (i + 2 >= knots.len() || (knots[i + 1] - knots[i + 2]).abs() < 1e-12)
    {
      // Handle the last real knot boundary (t == t_max at last non-degenerate interval)
      1.0
    } else {
      0.0
    };
  }

  let denom1 = knots[i + k] - knots[i];
  let term1 = if denom1 > 0.0 {
    (t - knots[i]) / denom1 * bspline_basis(i, k - 1, t, knots)
  } else {
    0.0
  };

  let denom2 = knots[i + k + 1] - knots[i + 1];
  let term2 = if denom2 > 0.0 {
    (knots[i + k + 1] - t) / denom2 * bspline_basis(i + 1, k - 1, t, knots)
  } else {
    0.0
  };

  term1 + term2
}

fn parse_raster(args: &[Expr], prims: &mut Vec<Primitive>) {
  // Raster[data], Raster[data, {{xmin, ymin}, {xmax, ymax}}], or
  // Raster[data, rect, {vmin, vmax}] — data is a 2D array of grayscale
  // values or {r,g,b}/{r,g,b,a} lists, scaled from the {vmin, vmax} range
  // (default {0, 1}). RasterBox cells carry their pixel data as
  // `RawArray["UnsignedInteger8", …]` (from `CompressedData`); unwrap it.
  let data_expr = match &args[0] {
    Expr::FunctionCall { name, args: inner }
      if name == "RawArray" && inner.len() >= 2 =>
    {
      &inner[1]
    }
    Expr::FunctionCall { name, args: inner }
      if name == "NumericArray" && !inner.is_empty() =>
    {
      &inner[0]
    }
    other => other,
  };
  let Expr::List(rows) = data_expr else { return };
  if rows.is_empty() {
    return;
  }

  // Raster[data, rect, {vmin, vmax}]: pixel values are scaled from this
  // range instead of the default 0–1 (e.g. byte data uses {0, 255}).
  let (v_min, v_max) = if args.len() >= 3
    && let Expr::List(range) = &args[2]
    && range.len() == 2
    && let Some(lo) = expr_to_f64(&range[0])
    && let Some(hi) = expr_to_f64(&range[1])
    && hi != lo
  {
    (lo, hi)
  } else {
    (0.0, 1.0)
  };
  let scale = |v: f64| ((v - v_min) / (v_max - v_min)).clamp(0.0, 1.0);

  let mut grid: Vec<Vec<Color>> = Vec::with_capacity(rows.len());
  for row in rows {
    let Expr::List(cols) = row else { return };
    let mut row_colors: Vec<Color> = Vec::with_capacity(cols.len());
    for cell in cols {
      if let Expr::List(components) = cell
        && (components.len() == 3 || components.len() == 4)
      {
        // RGB or RGBA list
        let r = scale(expr_to_f64(&components[0]).unwrap_or(0.0));
        let g = scale(expr_to_f64(&components[1]).unwrap_or(0.0));
        let b = scale(expr_to_f64(&components[2]).unwrap_or(0.0));
        let a = if components.len() == 4 {
          scale(expr_to_f64(&components[3]).unwrap_or(1.0))
        } else {
          1.0
        };
        row_colors.push(Color::new(r, g, b).with_alpha(a));
      } else if let Some(v) = expr_to_f64(cell) {
        // Grayscale value: single number maps to gray (0=black, 1=white)
        let v = scale(v);
        row_colors.push(Color::new(v, v, v));
      } else {
        row_colors.push(Color::new(0.0, 0.0, 0.0));
      }
    }
    grid.push(row_colors);
  }

  let nrows = grid.len();
  let ncols = grid.iter().map(std::vec::Vec::len).max().unwrap_or(0);
  if ncols == 0 {
    return;
  }

  // Parse optional coordinate range: Raster[data, {{xmin, ymin}, {xmax, ymax}}]
  let (x_min, y_min, x_max, y_max) = if args.len() >= 2 {
    if let Expr::List(range) = &args[1]
      && range.len() == 2
      && let Some((x0, y0)) = expr_to_point(&range[0])
      && let Some((x1, y1)) = expr_to_point(&range[1])
    {
      (x0, y0, x1, y1)
    } else {
      (0.0, 0.0, ncols as f64, nrows as f64)
    }
  } else {
    (0.0, 0.0, ncols as f64, nrows as f64)
  };

  // A reversed range mirrors the image: {{0, h}, {w, 0}} (the RasterBox
  // convention for top-down pixel rows) flips vertically. Row 0 renders at
  // the bottom, so flipping rows/columns here and normalizing the rect
  // reproduces the mirrored orientation.
  if y_min > y_max {
    grid.reverse();
  }
  if x_min > x_max {
    for row in &mut grid {
      row.reverse();
    }
  }

  prims.push(Primitive::RasterPrim {
    data: grid,
    x_min: x_min.min(x_max),
    y_min: y_min.min(y_max),
    x_max: x_min.max(x_max),
    y_max: y_min.max(y_max),
  });
}

/// `Locator[pt]` / `Locator[Dynamic[pt, …], appearance]`: a marker drawn at
/// the point's current position. The position may be held inside `Dynamic`
/// (whose second argument, the write-back callback, only matters to
/// interactive front-ends). A custom appearance graphic is embedded at its
/// own ImageSize in screen pixels, centered on the position — Wolfram
/// treats the appearance as a screen-space icon, not data-space geometry.
fn parse_locator(args: &[Expr], prims: &mut Vec<Primitive>) {
  let pos_expr = match &args[0] {
    Expr::FunctionCall { name, args: dargs }
      if name == "Dynamic" && !dargs.is_empty() =>
    {
      &dargs[0]
    }
    other => other,
  };
  let captured = crate::captured_graphics_count();
  let evaluated = crate::evaluator::evaluate_expr_to_expr(pos_expr)
    .unwrap_or_else(|_| pos_expr.clone());
  let points: Vec<(f64, f64)> = if let Some(p) = expr_to_point(&evaluated) {
    vec![p]
  } else {
    expr_to_point_list(&evaluated).unwrap_or_default()
  };
  if points.is_empty() {
    crate::truncate_captured_graphics(captured);
    return;
  }

  // The appearance graphic, pre-rendered at its own ImageSize. `Automatic`
  // (or no appearance) uses the default crosshair marker; `None` hides the
  // marker entirely.
  let appearance = args
    .get(1)
    .filter(|a| !matches!(a, Expr::Identifier(s) if s == "Automatic"));
  if appearance.is_some_and(|a| matches!(a, Expr::Identifier(s) if s == "None"))
  {
    crate::truncate_captured_graphics(captured);
    return;
  }
  let marker: Option<(String, f64, f64)> = appearance.and_then(|a| {
    // Render the appearance to a graphic. A held `Graphics[…]` call is
    // rendered directly (Graphics evaluates lazily at display time);
    // anything else is evaluated first.
    let rendered = match a {
      Expr::Graphics { .. } => Some(a.clone()),
      Expr::FunctionCall { name, args: gargs } if name == "Graphics" => {
        graphics_ast(gargs).ok()
      }
      other => match crate::evaluator::evaluate_expr_to_expr(other) {
        Ok(ev) => match &ev {
          Expr::Graphics { .. } => Some(ev.clone()),
          Expr::FunctionCall { name, args: gargs } if name == "Graphics" => {
            graphics_ast(&gargs.iter().cloned().collect::<Vec<_>>()).ok()
          }
          _ => None,
        },
        Err(_) => None,
      },
    };
    if let Some(Expr::Graphics { svg, .. }) = &rendered {
      let (w, h) = parse_svg_wh(svg);
      if w > 0.0 && h > 0.0 {
        return Some((svg.clone(), w, h));
      }
    }
    None
  });
  // Sub-evaluations may have rendered graphics (the appearance itself, or
  // anything a Dynamic position computed); those are embedded here, not
  // standalone outputs.
  crate::truncate_captured_graphics(captured);

  for (x, y) in points {
    let (svg, w, h) = if let Some((svg, w, h)) = &marker {
      (svg.clone(), *w, *h)
    } else {
      let (svg, size) = default_locator_marker_svg();
      (svg, size, size)
    };
    prims.push(Primitive::MarkerPrim { x, y, w, h, svg });
  }
}

/// Wolfram's default Locator appearance: a small circled crosshair.
fn default_locator_marker_svg() -> (String, f64) {
  let size = 16.0;
  let c = size / 2.0;
  let svg = format!(
    "<svg width=\"{size}\" height=\"{size}\" viewBox=\"0 0 {size} {size}\" xmlns=\"http://www.w3.org/2000/svg\">\
     <circle cx=\"{c}\" cy=\"{c}\" r=\"{r:.1}\" fill=\"rgba(180,180,180,0.35)\" stroke=\"#606060\" stroke-width=\"1\"/>\
     <line x1=\"{c}\" y1=\"1\" x2=\"{c}\" y2=\"{size}\" stroke=\"#606060\" stroke-width=\"1\"/>\
     <line x1=\"1\" y1=\"{c}\" x2=\"{size}\" y2=\"{c}\" stroke=\"#606060\" stroke-width=\"1\"/>\
     </svg>",
    r = c - 1.0,
  );
  (svg, size)
}

// ── Bounding box computation ─────────────────────────────────────────────

fn primitive_bbox(prim: &Primitive) -> BBox {
  let mut bb = BBox::empty();
  match prim {
    Primitive::PointSingle { x, y, .. } => {
      bb.include_point(*x, *y);
    }
    Primitive::PointMulti { points, .. }
    | Primitive::BezierCurvePrim { points, .. } => {
      for &(x, y) in points {
        bb.include_point(x, y);
      }
    }
    Primitive::Line { segments, .. } => {
      for seg in segments {
        for &(x, y) in seg {
          bb.include_point(x, y);
        }
      }
    }
    Primitive::CircleArc { cx, cy, rx, ry, .. }
    | Primitive::Disk { cx, cy, rx, ry, .. } => {
      bb.include_point(cx - rx, cy - ry);
      bb.include_point(cx + rx, cy + ry);
    }
    // A marker is a screen-space icon: only its anchor point occupies data
    // space.
    Primitive::MarkerPrim { x, y, .. } => {
      bb.include_point(*x, *y);
    }
    Primitive::DiskSector {
      cx,
      cy,
      rx,
      ry,
      angle1,
      angle2,
      ..
    } => {
      // Include center point (sector always connects to center)
      bb.include_point(*cx, *cy);
      // Include the two endpoint arcs
      bb.include_point(cx + rx * angle1.cos(), cy + ry * angle1.sin());
      bb.include_point(cx + rx * angle2.cos(), cy + ry * angle2.sin());
      // Include axis-aligned extremes if the arc crosses them
      let mut a = *angle1 % (2.0 * std::f64::consts::PI);
      if a < 0.0 {
        a += 2.0 * std::f64::consts::PI;
      }
      let span = angle2 - angle1;
      // Check each cardinal direction: 0, PI/2, PI, 3PI/2
      for k in 0..4 {
        let cardinal = k as f64 * std::f64::consts::FRAC_PI_2;
        let mut diff = cardinal - a;
        if diff < 0.0 {
          diff += 2.0 * std::f64::consts::PI;
        }
        if diff < span {
          bb.include_point(cx + rx * cardinal.cos(), cy + ry * cardinal.sin());
        }
      }
    }
    Primitive::RectPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      ..
    } => {
      bb.include_point(*x_min, *y_min);
      bb.include_point(*x_max, *y_max);
    }
    Primitive::PolygonPrim { points, .. } => {
      for &(x, y) in points {
        bb.include_point(x, y);
      }
    }
    Primitive::ArrowPrim { points, .. } => {
      for &(x, y) in points {
        bb.include_point(x, y);
      }
    }
    Primitive::TextPrim { x, y, .. } => {
      bb.include_point(*x, *y);
    }
    Primitive::RasterPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      ..
    } => {
      bb.include_point(*x_min, *y_min);
      bb.include_point(*x_max, *y_max);
    }
    // An inset keeps its own pixel size, so it anchors the range at its
    // position and nothing more.
    Primitive::InsetGraphic { x, y, .. } => bb.include_point(*x, *y),
    // An unbounded fill only anchors the plot range at its defining
    // points; the fill itself extends past whatever range results.
    Primitive::HalfPlanePrim { p, v, w, .. } => {
      bb.include_point(p.0, p.1);
      bb.include_point(p.0 + v.0, p.1 + v.1);
      bb.include_point(p.0 + w.0, p.1 + w.1);
    }
  }
  bb
}

/// Rotate a point (x, y) by `angle` radians counterclockwise about (cx, cy).
fn rotate_point(
  x: f64,
  y: f64,
  cx: f64,
  cy: f64,
  cos: f64,
  sin: f64,
) -> (f64, f64) {
  let dx = x - cx;
  let dy = y - cy;
  (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
}

/// Return a copy of `prim` rotated by `angle` radians about (cx, cy).
///
/// An axis-aligned rectangle becomes a (generally non-axis-aligned) polygon.
/// A circular disk/arc keeps its radius and only moves its center. An
/// elliptical disk (rx != ry) cannot be represented tilted in this model, so
/// its axes are kept axis-aligned as a best-effort approximation.
/// The affine transforms a `GeometricTransformation` second argument names,
/// each as a `({{a, b}, {c, d}}, {e, f})` pair mapping `p` to `m.p + v`.
/// Accepts the `{matrix, vector}` pair Wolfram normalizes a
/// `TransformationFunction` to, a bare matrix (no translation), the
/// homogeneous `TransformationFunction[…]` itself, and a list of any of
/// those — a list draws one transformed copy per entry. Empty when nothing
/// numeric can be read out.
pub(crate) fn parse_affine_transforms(
  expr: &Expr,
) -> Vec<([[f64; 2]; 2], (f64, f64))> {
  // {{a, b}, {c, d}} — a 2×2 linear map.
  fn matrix2(expr: &Expr) -> Option<[[f64; 2]; 2]> {
    let Expr::List(rows) = expr else { return None };
    if rows.len() != 2 {
      return None;
    }
    let r0 = expr_to_point(&rows[0])?;
    let r1 = expr_to_point(&rows[1])?;
    Some([[r0.0, r0.1], [r1.0, r1.1]])
  }
  // The 3×3 homogeneous matrix a TransformationFunction carries.
  fn homogeneous(expr: &Expr) -> Option<([[f64; 2]; 2], (f64, f64))> {
    let Expr::List(rows) = expr else { return None };
    if rows.len() != 3 {
      return None;
    }
    let mut m = [[0.0; 2]; 2];
    let mut v = [0.0; 2];
    for (i, row) in rows.iter().take(2).enumerate() {
      let Expr::List(cells) = row else { return None };
      if cells.len() != 3 {
        return None;
      }
      m[i][0] = expr_to_f64(&cells[0])?;
      m[i][1] = expr_to_f64(&cells[1])?;
      v[i] = expr_to_f64(&cells[2])?;
    }
    Some((m, (v[0], v[1])))
  }
  fn single(expr: &Expr) -> Option<([[f64; 2]; 2], (f64, f64))> {
    if let Expr::FunctionCall { name, args } = expr
      && name == "TransformationFunction"
      && args.len() == 1
    {
      return homogeneous(&args[0]);
    }
    if let Expr::List(items) = expr
      && items.len() == 2
      && let Some(m) = matrix2(&items[0])
      && let Some(v) = expr_to_point(&items[1])
    {
      return Some((m, v));
    }
    matrix2(expr).map(|m| (m, (0.0, 0.0)))
  }
  if let Some(one) = single(expr) {
    return vec![one];
  }
  match expr {
    Expr::List(items) => items.iter().filter_map(single).collect(),
    _ => Vec::new(),
  }
}

/// Map a primitive through the affine transform `p ↦ m.p + v`.
///
/// The matrix is factored (a closed-form 2×2 SVD) into
/// `rotate(φ) ∘ scale(sx, sy) ∘ rotate(θ)`, so the transform is carried out
/// by the same primitive-level rotate/scale/translate the explicit
/// `Rotate`/`Scale`/`Translate` directives use — including their handling of
/// the shapes an affine map does not leave in its own family (a circle
/// becoming an ellipse, an arc reversing sweep under a reflection).
fn affine_primitive(
  prim: &Primitive,
  m: [[f64; 2]; 2],
  v: (f64, f64),
) -> Primitive {
  let (a, b, c, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);
  let (e, f, g, h) = (
    f64::midpoint(a, d),
    (a - d) / 2.0,
    f64::midpoint(c, b),
    (c - b) / 2.0,
  );
  let (q, r) = ((e * e + h * h).sqrt(), (f * f + g * g).sqrt());
  let (sx, sy) = (q + r, q - r);
  let (a1, a2) = (g.atan2(f), h.atan2(e));
  let (theta, phi) = ((a2 - a1) / 2.0, f64::midpoint(a2, a1));
  let rotated = rotate_primitive(prim, 0.0, 0.0, theta);
  let scaled = scale_primitive(&rotated, 0.0, 0.0, sx, sy);
  let rotated = rotate_primitive(&scaled, 0.0, 0.0, phi);
  translate_primitive(&rotated, v.0, v.1)
}

fn rotate_primitive(
  prim: &Primitive,
  cx: f64,
  cy: f64,
  angle: f64,
) -> Primitive {
  let cos = angle.cos();
  let sin = angle.sin();
  let rp = |x: f64, y: f64| rotate_point(x, y, cx, cy, cos, sin);
  // Rotate a direction vector (no translation).
  let rv = |x: f64, y: f64| (x * cos - y * sin, x * sin + y * cos);
  match prim {
    // An inset keeps its own orientation and size; only its anchor moves.
    Primitive::InsetGraphic {
      svg,
      x,
      y,
      w,
      h,
      scaled,
    } => {
      // A scaled anchor names a place in the finished frame, not a point in
      // the data, so a transform of the coordinates leaves it where it is.
      let (nx, ny) = if *scaled { (*x, *y) } else { rp(*x, *y) };
      Primitive::InsetGraphic {
        svg: svg.clone(),
        x: nx,
        y: ny,
        w: *w,
        h: *h,
        scaled: *scaled,
      }
    }
    Primitive::PointSingle { x, y, style } => {
      let (nx, ny) = rp(*x, *y);
      Primitive::PointSingle {
        x: nx,
        y: ny,
        style: style.clone(),
      }
    }
    Primitive::PointMulti { points, style } => Primitive::PointMulti {
      points: points.iter().map(|&(x, y)| rp(x, y)).collect(),
      style: style.clone(),
    },
    Primitive::Line { segments, style } => Primitive::Line {
      segments: segments
        .iter()
        .map(|seg| seg.iter().map(|&(x, y)| rp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::PolygonPrim {
      points,
      holes,
      style,
    } => Primitive::PolygonPrim {
      points: points.iter().map(|&(x, y)| rp(x, y)).collect(),
      holes: holes
        .iter()
        .map(|h| h.iter().map(|&(x, y)| rp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::ArrowPrim {
      points,
      setback,
      style,
    } => Primitive::ArrowPrim {
      points: points.iter().map(|&(x, y)| rp(x, y)).collect(),
      setback: *setback,
      style: style.clone(),
    },
    Primitive::BezierCurvePrim { points, style } => {
      Primitive::BezierCurvePrim {
        points: points.iter().map(|&(x, y)| rp(x, y)).collect(),
        style: style.clone(),
      }
    }
    // A rotated rectangle is no longer axis-aligned → emit a polygon of its
    // four rotated corners.
    Primitive::RectPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      style,
    } => Primitive::PolygonPrim {
      points: [
        (*x_min, *y_min),
        (*x_max, *y_min),
        (*x_max, *y_max),
        (*x_min, *y_max),
      ]
      .iter()
      .map(|&(x, y)| rp(x, y))
      .collect(),
      holes: Vec::new(),
      style: style.clone(),
    },
    Primitive::Disk {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      style,
    } => {
      let (nx, ny) = rp(*dcx, *dcy);
      Primitive::Disk {
        cx: nx,
        cy: ny,
        rx: *rx,
        ry: *ry,
        style: style.clone(),
      }
    }
    Primitive::CircleArc {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      angles,
      style,
    } => {
      let (nx, ny) = rp(*dcx, *dcy);
      Primitive::CircleArc {
        cx: nx,
        cy: ny,
        rx: *rx,
        ry: *ry,
        // Rotating the circle rotates its arc's angular range too.
        angles: angles.map(|(a1, a2)| (a1 + angle, a2 + angle)),
        style: style.clone(),
      }
    }
    Primitive::DiskSector {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      angle1,
      angle2,
      style,
    } => {
      let (nx, ny) = rp(*dcx, *dcy);
      Primitive::DiskSector {
        cx: nx,
        cy: ny,
        rx: *rx,
        ry: *ry,
        angle1: angle1 + angle,
        angle2: angle2 + angle,
        style: style.clone(),
      }
    }
    Primitive::TextPrim {
      text,
      x,
      y,
      offset,
      background,
      frame,
      scaled,
      direction,
      style,
    } => {
      let (nx, ny) = if *scaled { (*x, *y) } else { rp(*x, *y) };
      Primitive::TextPrim {
        text: text.clone(),
        x: nx,
        y: ny,
        offset: *offset,
        background: *background,
        frame: *frame,
        scaled: *scaled,
        direction: direction.map(|(dx, dy)| rv(dx, dy)),
        style: style.clone(),
      }
    }
    // Rasters aren't re-sampled here; keep them in place.
    Primitive::RasterPrim {
      data,
      x_min,
      y_min,
      x_max,
      y_max,
    } => Primitive::RasterPrim {
      data: data.clone(),
      x_min: *x_min,
      y_min: *y_min,
      x_max: *x_max,
      y_max: *y_max,
    },
    Primitive::HalfPlanePrim {
      p,
      v,
      w,
      full,
      style,
    } => Primitive::HalfPlanePrim {
      p: rp(p.0, p.1),
      v: rv(v.0, v.1),
      w: rv(w.0, w.1),
      full: *full,
      style: style.clone(),
    },
    // A marker is a screen-space icon anchored on a data point: transforms
    // move the anchor and leave the icon itself untouched.
    Primitive::MarkerPrim { x, y, w, h, svg } => {
      let (x, y) = rp(*x, *y);
      Primitive::MarkerPrim {
        x,
        y,
        w: *w,
        h: *h,
        svg: svg.clone(),
      }
    }
  }
}

/// Return a copy of `prim` translated by (dx, dy).
fn translate_primitive(prim: &Primitive, dx: f64, dy: f64) -> Primitive {
  let tp = |x: f64, y: f64| (x + dx, y + dy);
  match prim {
    Primitive::InsetGraphic {
      svg,
      x,
      y,
      w,
      h,
      scaled,
    } => Primitive::InsetGraphic {
      svg: svg.clone(),
      x: if *scaled { *x } else { x + dx },
      y: if *scaled { *y } else { y + dy },
      w: *w,
      h: *h,
      scaled: *scaled,
    },
    Primitive::PointSingle { x, y, style } => Primitive::PointSingle {
      x: x + dx,
      y: y + dy,
      style: style.clone(),
    },
    Primitive::PointMulti { points, style } => Primitive::PointMulti {
      points: points.iter().map(|&(x, y)| tp(x, y)).collect(),
      style: style.clone(),
    },
    Primitive::Line { segments, style } => Primitive::Line {
      segments: segments
        .iter()
        .map(|seg| seg.iter().map(|&(x, y)| tp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::PolygonPrim {
      points,
      holes,
      style,
    } => Primitive::PolygonPrim {
      points: points.iter().map(|&(x, y)| tp(x, y)).collect(),
      holes: holes
        .iter()
        .map(|h| h.iter().map(|&(x, y)| tp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::ArrowPrim {
      points,
      setback,
      style,
    } => Primitive::ArrowPrim {
      points: points.iter().map(|&(x, y)| tp(x, y)).collect(),
      setback: *setback,
      style: style.clone(),
    },
    Primitive::BezierCurvePrim { points, style } => {
      Primitive::BezierCurvePrim {
        points: points.iter().map(|&(x, y)| tp(x, y)).collect(),
        style: style.clone(),
      }
    }
    Primitive::RectPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      style,
    } => Primitive::RectPrim {
      x_min: x_min + dx,
      y_min: y_min + dy,
      x_max: x_max + dx,
      y_max: y_max + dy,
      style: style.clone(),
    },
    Primitive::Disk {
      cx,
      cy,
      rx,
      ry,
      style,
    } => Primitive::Disk {
      cx: cx + dx,
      cy: cy + dy,
      rx: *rx,
      ry: *ry,
      style: style.clone(),
    },
    Primitive::CircleArc {
      cx,
      cy,
      rx,
      ry,
      angles,
      style,
    } => Primitive::CircleArc {
      cx: cx + dx,
      cy: cy + dy,
      rx: *rx,
      ry: *ry,
      angles: *angles,
      style: style.clone(),
    },
    Primitive::DiskSector {
      cx,
      cy,
      rx,
      ry,
      angle1,
      angle2,
      style,
    } => Primitive::DiskSector {
      cx: cx + dx,
      cy: cy + dy,
      rx: *rx,
      ry: *ry,
      angle1: *angle1,
      angle2: *angle2,
      style: style.clone(),
    },
    Primitive::TextPrim {
      text,
      x,
      y,
      offset,
      background,
      frame,
      scaled,
      direction,
      style,
    } => Primitive::TextPrim {
      text: text.clone(),
      x: if *scaled { *x } else { x + dx },
      y: if *scaled { *y } else { y + dy },
      offset: *offset,
      background: *background,
      frame: *frame,
      scaled: *scaled,
      direction: *direction,
      style: style.clone(),
    },
    Primitive::RasterPrim {
      data,
      x_min,
      y_min,
      x_max,
      y_max,
    } => Primitive::RasterPrim {
      data: data.clone(),
      x_min: x_min + dx,
      y_min: y_min + dy,
      x_max: x_max + dx,
      y_max: y_max + dy,
    },
    Primitive::HalfPlanePrim {
      p,
      v,
      w,
      full,
      style,
    } => Primitive::HalfPlanePrim {
      p: tp(p.0, p.1),
      v: *v,
      w: *w,
      full: *full,
      style: style.clone(),
    },
    Primitive::MarkerPrim { x, y, w, h, svg } => Primitive::MarkerPrim {
      x: x + dx,
      y: y + dy,
      w: *w,
      h: *h,
      svg: svg.clone(),
    },
  }
}

/// Return a copy of `prim` scaled by (sx, sy) about the fixed point (cx, cy).
///
/// Ellipse radii scale per-axis by |sx| and |sy| (a radius cannot be
/// negative); for negative factors the arc angles are mirrored accordingly.
/// Point sizes, stroke widths, and text sizes are absolute (not part of the
/// coordinate system) and stay unchanged.
fn scale_primitive(
  prim: &Primitive,
  cx: f64,
  cy: f64,
  sx: f64,
  sy: f64,
) -> Primitive {
  let sp = |x: f64, y: f64| (cx + (x - cx) * sx, cy + (y - cy) * sy);
  // Map an ellipse angle through the axis scaling: a point at angle θ on the
  // ellipse lands at angle θ' on the scaled ellipse with
  // cos θ' = sign(sx)·cos θ and sin θ' = sign(sy)·sin θ.
  let sa = |a: f64| -> f64 {
    match (sx < 0.0, sy < 0.0) {
      (false, false) => a,
      (true, false) => std::f64::consts::PI - a,
      (false, true) => -a,
      (true, true) => a + std::f64::consts::PI,
    }
  };
  // A single mirror flips the sweep direction, so the endpoints swap to keep
  // the arc's counterclockwise orientation.
  let sr = |a1: f64, a2: f64| -> (f64, f64) {
    if (sx < 0.0) == (sy < 0.0) {
      (sa(a1), sa(a2))
    } else {
      (sa(a2), sa(a1))
    }
  };
  match prim {
    Primitive::InsetGraphic {
      svg,
      x,
      y,
      w,
      h,
      scaled,
    } => {
      let (nx, ny) = if *scaled { (*x, *y) } else { sp(*x, *y) };
      Primitive::InsetGraphic {
        svg: svg.clone(),
        x: nx,
        y: ny,
        w: *w,
        h: *h,
        scaled: *scaled,
      }
    }
    Primitive::PointSingle { x, y, style } => {
      let (nx, ny) = sp(*x, *y);
      Primitive::PointSingle {
        x: nx,
        y: ny,
        style: style.clone(),
      }
    }
    Primitive::PointMulti { points, style } => Primitive::PointMulti {
      points: points.iter().map(|&(x, y)| sp(x, y)).collect(),
      style: style.clone(),
    },
    Primitive::Line { segments, style } => Primitive::Line {
      segments: segments
        .iter()
        .map(|seg| seg.iter().map(|&(x, y)| sp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::PolygonPrim {
      points,
      holes,
      style,
    } => Primitive::PolygonPrim {
      points: points.iter().map(|&(x, y)| sp(x, y)).collect(),
      holes: holes
        .iter()
        .map(|h| h.iter().map(|&(x, y)| sp(x, y)).collect())
        .collect(),
      style: style.clone(),
    },
    Primitive::ArrowPrim {
      points,
      setback,
      style,
    } => Primitive::ArrowPrim {
      points: points.iter().map(|&(x, y)| sp(x, y)).collect(),
      setback: *setback,
      style: style.clone(),
    },
    Primitive::BezierCurvePrim { points, style } => {
      Primitive::BezierCurvePrim {
        points: points.iter().map(|&(x, y)| sp(x, y)).collect(),
        style: style.clone(),
      }
    }
    Primitive::RectPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      style,
    } => {
      // Negative factors swap the corners; renormalize to min/max form.
      let (x1, y1) = sp(*x_min, *y_min);
      let (x2, y2) = sp(*x_max, *y_max);
      Primitive::RectPrim {
        x_min: x1.min(x2),
        y_min: y1.min(y2),
        x_max: x1.max(x2),
        y_max: y1.max(y2),
        style: style.clone(),
      }
    }
    Primitive::Disk {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      style,
    } => {
      let (nx, ny) = sp(*dcx, *dcy);
      Primitive::Disk {
        cx: nx,
        cy: ny,
        rx: rx * sx.abs(),
        ry: ry * sy.abs(),
        style: style.clone(),
      }
    }
    Primitive::CircleArc {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      angles,
      style,
    } => {
      let (nx, ny) = sp(*dcx, *dcy);
      Primitive::CircleArc {
        cx: nx,
        cy: ny,
        rx: rx * sx.abs(),
        ry: ry * sy.abs(),
        angles: angles.map(|(a1, a2)| sr(a1, a2)),
        style: style.clone(),
      }
    }
    Primitive::DiskSector {
      cx: dcx,
      cy: dcy,
      rx,
      ry,
      angle1,
      angle2,
      style,
    } => {
      let (nx, ny) = sp(*dcx, *dcy);
      let (a1, a2) = sr(*angle1, *angle2);
      Primitive::DiskSector {
        cx: nx,
        cy: ny,
        rx: rx * sx.abs(),
        ry: ry * sy.abs(),
        angle1: a1,
        angle2: a2,
        style: style.clone(),
      }
    }
    Primitive::TextPrim {
      text,
      x,
      y,
      offset,
      background,
      frame,
      scaled,
      direction,
      style,
    } => {
      let (nx, ny) = if *scaled { (*x, *y) } else { sp(*x, *y) };
      Primitive::TextPrim {
        text: text.clone(),
        x: nx,
        y: ny,
        offset: *offset,
        background: *background,
        frame: *frame,
        scaled: *scaled,
        direction: direction.map(|(dx, dy)| (dx * sx, dy * sy)),
        style: style.clone(),
      }
    }
    Primitive::RasterPrim {
      data,
      x_min,
      y_min,
      x_max,
      y_max,
    } => {
      let (x1, y1) = sp(*x_min, *y_min);
      let (x2, y2) = sp(*x_max, *y_max);
      Primitive::RasterPrim {
        data: data.clone(),
        x_min: x1.min(x2),
        y_min: y1.min(y2),
        x_max: x1.max(x2),
        y_max: y1.max(y2),
      }
    }
    Primitive::HalfPlanePrim {
      p,
      v,
      w,
      full,
      style,
    } => Primitive::HalfPlanePrim {
      p: sp(p.0, p.1),
      v: (v.0 * sx, v.1 * sy),
      w: (w.0 * sx, w.1 * sy),
      full: *full,
      style: style.clone(),
    },
    Primitive::MarkerPrim { x, y, w, h, svg } => {
      let (nx, ny) = sp(*x, *y);
      Primitive::MarkerPrim {
        x: nx,
        y: ny,
        w: *w,
        h: *h,
        svg: svg.clone(),
      }
    }
  }
}

/// Trim a polyline by `setback.0` from the start and `setback.1` from the end,
/// measured in coordinate-space distance along the path.
fn apply_setback(
  points: &[(f64, f64)],
  setback: (f64, f64),
) -> Vec<(f64, f64)> {
  if points.len() < 2 {
    return points.to_vec();
  }
  let (s_start, s_end) = setback;
  if s_start <= 0.0 && s_end <= 0.0 {
    return points.to_vec();
  }

  // Compute cumulative distances
  let n = points.len();
  let mut cum = vec![0.0_f64; n];
  for i in 1..n {
    let dx = points[i].0 - points[i - 1].0;
    let dy = points[i].1 - points[i - 1].1;
    cum[i] = cum[i - 1] + (dx * dx + dy * dy).sqrt();
  }
  let total = cum[n - 1];

  if s_start + s_end >= total {
    return Vec::new();
  }

  let start_dist = s_start;
  let end_dist = total - s_end;

  let mut result = Vec::new();

  // Find new start point
  let mut start_seg = 0;
  for i in 1..n {
    if cum[i] >= start_dist {
      start_seg = i;
      break;
    }
  }
  // Interpolate start point on segment [start_seg-1, start_seg]
  let seg_len = cum[start_seg] - cum[start_seg - 1];
  if seg_len > 0.0 {
    let t = (start_dist - cum[start_seg - 1]) / seg_len;
    let (x0, y0) = points[start_seg - 1];
    let (x1, y1) = points[start_seg];
    result.push((x0 + t * (x1 - x0), y0 + t * (y1 - y0)));
  } else {
    result.push(points[start_seg]);
  }

  // Add intermediate points between start and end
  for i in start_seg..n {
    if cum[i] > start_dist && cum[i] < end_dist {
      result.push(points[i]);
    }
  }

  // Find new end point
  let mut end_seg = n - 1;
  for i in (1..n).rev() {
    if cum[i - 1] <= end_dist {
      end_seg = i;
      break;
    }
  }
  // Interpolate end point on segment [end_seg-1, end_seg]
  let seg_len = cum[end_seg] - cum[end_seg - 1];
  if seg_len > 0.0 {
    let t = (end_dist - cum[end_seg - 1]) / seg_len;
    let (x0, y0) = points[end_seg - 1];
    let (x1, y1) = points[end_seg];
    let end_pt = (x0 + t * (x1 - x0), y0 + t * (y1 - y0));
    // Avoid duplicate if end point equals last pushed point
    if result.last() != Some(&end_pt) {
      result.push(end_pt);
    }
  } else if result.last() != Some(&points[end_seg]) {
    result.push(points[end_seg]);
  }

  result
}

// ── SVG generation ───────────────────────────────────────────────────────

fn coord_x(x: f64, bb: &BBox, svg_w: f64) -> f64 {
  (x - bb.x_min) / bb.width() * svg_w
}

fn coord_y(y: f64, bb: &BBox, svg_h: f64) -> f64 {
  // Flip y: Wolfram is y-up, SVG is y-down
  (bb.y_max - y) / bb.height() * svg_h
}

/// The width named by a thickness symbol, in the negative encoding this
/// module uses for `AbsoluteThickness`. `Thin` evaluates to
/// `Thickness[Tiny]` and `Thick` to `Thickness[Large]`, and Wolfram
/// strokes those 0.2 and 2 wide respectively.
fn symbolic_thickness(name: &str) -> Option<f64> {
  match name {
    "Tiny" | "Thin" => Some(-0.2),
    "Small" => Some(-0.5),
    "Medium" => Some(-1.0),
    "Large" | "Thick" => Some(-2.0),
    _ => None,
  }
}

/// `symbolic_thickness` for a directive argument (`Thickness[Large]`).
fn symbolic_thickness_arg(arg: &Expr) -> Option<f64> {
  match arg {
    Expr::Identifier(s) => symbolic_thickness(s),
    _ => None,
  }
}

/// The stroke an `EdgeForm` puts around a filled primitive: its colour
/// and width in SVG units, or `None` when no edge is drawn.
///
/// Wolfram draws no edge for `EdgeForm[]` / `EdgeForm[None]`; an
/// `EdgeForm` naming only a thickness draws it in black, and one naming
/// only a colour draws it at the default width of 1.
fn edge_stroke(
  edge_form: Option<&EdgeForm>,
  bb: &BBox,
  svg_w: f64,
) -> Option<(Color, f64)> {
  let ef = edge_form.as_ref()?;
  if ef.color.is_none() && ef.thickness.is_none() {
    return None;
  }
  Some((
    ef.color.unwrap_or(Color::new(0.0, 0.0, 0.0)),
    ef.thickness.map_or(1.0, |t| thickness_px(t, bb, svg_w)),
  ))
}

fn thickness_px(t: f64, bb: &BBox, svg_w: f64) -> f64 {
  if t < 0.0 {
    // Absolute thickness (stored as negative)
    -t
  } else {
    // A `Thickness` is a fraction of the *image* width, the same measure a
    // `Dashing` length uses — `Thickness[0.05]` on a 200-pixel picture is
    // 10 px whether the picture is taller or wider than it is broad
    // (measured from wolframscript in both aspects). Scaling it by the data
    // range's taller side made a portrait picture's lines twice too heavy.
    let _ = bb;
    t * svg_w
  }
}

/// The `stroke-dasharray` for a dashing spec. Wolfram states dash lengths as
/// fractions of the *image* width — `Dashing[{0.05, 0.05}]` on a 400-pixel
/// picture is `20,20`, and `Dashed` (`Dashing[{Small, Small}]`, Small =
/// 0.01) is `4,4`, both measured from wolframscript's own SVG export. A
/// zero-length dash is drawn as one pixel, which is how Wolfram makes
/// `Dotted` (`Dashing[{0, Small}]`) visible as `1,4`; scaling it by the
/// data width instead left dotted lines invisible.
fn dash_attr(dashing: Option<&Vec<f64>>, _bb: &BBox, svg_w: f64) -> String {
  if let Some(dashes) = dashing {
    let px: Vec<String> = dashes
      .iter()
      .map(|d| {
        // Negative = absolute pixels (a named size); positive = a fraction
        // of the image width; zero = the dot Wolfram draws as one pixel.
        let px = if *d < 0.0 { -*d } else { *d * svg_w };
        format!("{:.1}", if px <= 0.0 { 1.0 } else { px })
      })
      .collect();
    format!(" stroke-dasharray=\"{}\"", px.join(","))
  } else {
    String::new()
  }
}

fn format_tick_value(v: f64) -> String {
  if v.abs() < 1e-10 {
    return "0".to_string();
  }
  if (v - v.round()).abs() < 1e-10 {
    return format!("{}", v.round() as i64);
  }
  let mut s = format!("{v:.6}");
  while s.contains('.') && s.ends_with('0') {
    s.pop();
  }
  if s.ends_with('.') {
    s.pop();
  }
  s
}

/// The spacing of a tick sequence, or `None` when there are too few ticks
/// to tell. Every label of a set carries the decimals the spacing needs,
/// so this is what decides whether an axis reads `-1.0, -0.5, 0.0` or
/// `-1, -0.5, 0`.
fn tick_sequence_step(values: &[f64]) -> Option<f64> {
  let step = values
    .windows(2)
    .map(|w| (w[1] - w[0]).abs())
    .fold(f64::INFINITY, f64::min);
  (step.is_finite() && step > 0.0).then_some(step)
}

/// Label one tick of a `step`-spaced set, the way the plot renderer does:
/// all the labels of a set get the same number of decimals.
fn format_tick_in_sequence(v: f64, step: Option<f64>) -> String {
  match step {
    Some(step) => crate::functions::plot::format_tick_with_step(v, step),
    None => format_tick_value(v),
  }
}

fn nice_tick_step(min: f64, max: f64, target_count: usize) -> f64 {
  let range = (max - min).abs();
  if !range.is_finite() || range <= 0.0 {
    return 1.0;
  }
  let raw_step = range / (target_count.max(2) as f64);
  let magnitude = 10f64.powf(raw_step.log10().floor());
  let normalized = raw_step / magnitude;
  let nice = if normalized < 1.5 {
    1.0
  } else if normalized < 3.0 {
    2.0
  } else if normalized < 7.0 {
    5.0
  } else {
    10.0
  };
  nice * magnitude
}

fn generate_ticks(min: f64, max: f64, target_count: usize) -> Vec<f64> {
  let step = nice_tick_step(min, max, target_count);
  if !step.is_finite() || step <= 0.0 {
    return vec![];
  }
  let start = (min / step).ceil() * step;
  let end = (max / step).floor() * step;
  if !start.is_finite() || !end.is_finite() || start > end {
    return vec![];
  }

  let mut ticks = Vec::new();
  let mut t = start;
  let eps = step * 1e-8;
  while t <= end + eps && ticks.len() < 200 {
    ticks.push(if t.abs() < eps { 0.0 } else { t });
    t += step;
  }
  ticks
}

/// `Ticks` for one axis: automatic positions, none at all, or an explicit
/// list of positions (each optionally carrying its own label).
#[derive(Clone)]
enum TickSpec {
  None,
  Automatic,
  Explicit(Vec<(f64, Option<String>)>),
}

/// Parse one side of `Ticks -> {xspec, yspec}`.
fn parse_tick_spec(expr: &Expr) -> TickSpec {
  match expr {
    Expr::Identifier(s) if s == "None" || s == "False" => TickSpec::None,
    Expr::Identifier(s) if s == "Automatic" || s == "All" => {
      TickSpec::Automatic
    }
    Expr::List(entries) => TickSpec::Explicit(
      entries
        .iter()
        .filter_map(|entry| match entry {
          // `{pos, label}` gives the mark its own text.
          Expr::List(pair) if pair.len() >= 2 => {
            Some((expr_to_f64(&pair[0])?, Some(expr_to_svg_markup(&pair[1]))))
          }
          // A bare position is labelled with the expression standing at
          // it, so `Ticks -> {{-Pi, 0, Pi}, …}` reads "-π", "0", "π"
          // rather than the numeric value it evaluates to.
          other => {
            let pos = expr_to_f64(other)?;
            Some((
              pos,
              Some(crate::functions::plot::bare_tick_label(other, pos)),
            ))
          }
        })
        .collect(),
    ),
    _ => match expr_to_f64(expr) {
      Some(p) => TickSpec::Explicit(vec![(p, None)]),
      Option::None => TickSpec::None,
    },
  }
}

/// The tick positions (and any explicit labels) an axis draws.
fn axis_ticks(
  spec: &TickSpec,
  min: f64,
  max: f64,
) -> Vec<(f64, Option<String>)> {
  match spec {
    TickSpec::None => Vec::new(),
    TickSpec::Automatic => generate_ticks(min, max, 6)
      .into_iter()
      .map(|t| (t, None))
      .collect(),
    TickSpec::Explicit(entries) => entries.clone(),
  }
}

/// Horizontal centre for a bottom-axis tick label, shifted inwards when
/// centring it on the tick would push part of the text outside the picture.
/// `margins` is the room available left of `0` and right of `svg_w`; the
/// width estimate assumes the 14px monospace face the labels are drawn in.
fn clamp_tick_label_x(
  x: f64,
  label: &str,
  svg_w: f64,
  margins: (f64, f64),
) -> f64 {
  let half = label.chars().count() as f64 * 14.0 * 0.6 / 2.0;
  let (left, right) = margins;
  let lo = -left + half;
  let hi = svg_w + right - half;
  // A label wider than the picture cannot be placed without overflow; leave
  // it centred rather than swinging it to an arbitrary side.
  if lo > hi {
    return x;
  }
  x.clamp(lo, hi)
}

fn render_axes(
  svg: &mut String,
  axes: (bool, bool),
  bb: &BBox,
  svg_w: f64,
  svg_h: f64,
  axes_label: Option<&(String, String)>,
  ticks: (&TickSpec, &TickSpec),
  x_margins: (f64, f64),
) {
  let t = theme();
  let axis_stroke = t.axis_stroke;
  let tick_label_fill = t.tick_label_fill;

  if !axes.0 && !axes.1 {
    return;
  }

  let axis_y_data = if bb.y_min <= 0.0 && 0.0 <= bb.y_max {
    0.0
  } else {
    bb.y_min
  };
  let axis_x_data = if bb.x_min <= 0.0 && 0.0 <= bb.x_max {
    0.0
  } else {
    bb.x_min
  };
  let axis_y_px = coord_y(axis_y_data, bb, svg_h);
  let axis_x_px = coord_x(axis_x_data, bb, svg_w);

  if axes.0 {
    svg.push_str(&format!(
      "<line x1=\"0.00\" y1=\"{axis_y_px:.2}\" x2=\"{svg_w:.2}\" y2=\"{axis_y_px:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n"
    ));
    let entries = axis_ticks(ticks.0, bb.x_min, bb.x_max);
    let step =
      tick_sequence_step(&entries.iter().map(|(t, _)| *t).collect::<Vec<_>>());
    for (t, tick_label) in entries {
      let x = coord_x(t, bb, svg_w);
      if !x.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{x:.2}\" y2=\"{:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n",
        axis_y_px - 4.0,
        axis_y_px + 4.0
      ));
      let label = tick_label
        .unwrap_or_else(|| svg_escape(&format_tick_in_sequence(t, step)));
      // The two axes cross at the origin, so only one of them labels it.
      if axes.1 && t.abs() < step.unwrap_or(1.0) * 1e-6 {
        continue;
      }
      // A tick sitting on the edge of the drawing area centres its label
      // half outside the picture, where it is cut off — `-1.0` at the left
      // edge would read as `1.0`. Nudge such a label inwards just far enough
      // to clear the margin, the way Wolfram's image padding keeps the
      // outermost labels whole.
      let x = clamp_tick_label_x(x, &label, svg_w, x_margins);
      svg.push_str(&format!(
        "<text x=\"{x:.2}\" y=\"{:.2}\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"monospace\" text-anchor=\"middle\" dominant-baseline=\"hanging\">{label}</text>\n",
        axis_y_px + 6.0,
      ));
    }
  }

  if axes.1 {
    svg.push_str(&format!(
      "<line x1=\"{axis_x_px:.2}\" y1=\"0.00\" x2=\"{axis_x_px:.2}\" y2=\"{svg_h:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n"
    ));
    let entries = axis_ticks(ticks.1, bb.y_min, bb.y_max);
    let step =
      tick_sequence_step(&entries.iter().map(|(t, _)| *t).collect::<Vec<_>>());
    for (t, tick_label) in entries {
      let y = coord_y(t, bb, svg_h);
      if !y.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{y:.2}\" x2=\"{:.2}\" y2=\"{y:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n",
        axis_x_px - 4.0,
        axis_x_px + 4.0
      ));
      let label = tick_label
        .unwrap_or_else(|| svg_escape(&format_tick_in_sequence(t, step)));
      if axes.0 && t.abs() < step.unwrap_or(1.0) * 1e-6 {
        continue;
      }
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{y:.2}\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"monospace\" text-anchor=\"end\" dominant-baseline=\"middle\">{label}</text>\n",
        axis_x_px - 6.0,
      ));
    }
  }

  // Wolfram writes an AxesLabel at the *end* of its axis: the x label just
  // past the right edge, level with the axis, and the y label above the
  // top of the vertical axis.
  if let Some((x_label, y_label)) = axes_label {
    if axes.0 && !x_label.is_empty() {
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"sans-serif\" text-anchor=\"start\" dominant-baseline=\"middle\">{x_label}</text>\n",
        svg_w + 8.0,
        axis_y_px,
      ));
    }
    if axes.1 && !y_label.is_empty() {
      svg.push_str(&format!(
        "<text x=\"{axis_x_px:.2}\" y=\"-8.00\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"sans-serif\" text-anchor=\"middle\">{y_label}</text>\n",
      ));
    }
  }
}

/// Render a rectangular frame around the plot area with tick marks and labels
/// on the bottom and left edges, and minor ticks on the top and right edges.
fn render_frame(
  svg: &mut String,
  bb: &BBox,
  svg_w: f64,
  svg_h: f64,
  ticks: bool,
) {
  let t = theme();
  let frame_stroke = t.framed_border;
  let tick_label_fill = t.tick_label_fill;

  // Draw the rectangular border
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{svg_w:.2}\" height=\"{svg_h:.2}\" fill=\"none\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n"
  ));
  if !ticks {
    return;
  }

  let x_ticks = generate_ticks(bb.x_min, bb.x_max, 6);
  let y_ticks = generate_ticks(bb.y_min, bb.y_max, 6);
  let x_step = tick_sequence_step(&x_ticks);
  let y_step = tick_sequence_step(&y_ticks);

  // Bottom edge: ticks + labels
  for &t_val in &x_ticks {
    let x = coord_x(t_val, bb, svg_w);
    if !x.is_finite() {
      continue;
    }
    // Tick mark inward from bottom edge
    svg.push_str(&format!(
      "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{x:.2}\" y2=\"{svg_h:.2}\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n",
      svg_h - 5.0
    ));
    // Label below the bottom edge
    let label = format_tick_in_sequence(t_val, x_step);
    svg.push_str(&format!(
      "<text x=\"{x:.2}\" y=\"{:.2}\" fill=\"{tick_label_fill}\" font-size=\"12\" font-family=\"monospace\" text-anchor=\"middle\" dominant-baseline=\"hanging\">{}</text>\n",
      svg_h + 4.0,
      svg_escape(&label),
    ));
  }

  // Top edge: ticks only (no labels)
  for &t_val in &x_ticks {
    let x = coord_x(t_val, bb, svg_w);
    if !x.is_finite() {
      continue;
    }
    svg.push_str(&format!(
      "<line x1=\"{x:.2}\" y1=\"0\" x2=\"{x:.2}\" y2=\"{:.2}\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n",
      5.0
    ));
  }

  // Left edge: ticks + labels
  for &t_val in &y_ticks {
    let y = coord_y(t_val, bb, svg_h);
    if !y.is_finite() {
      continue;
    }
    // Tick mark inward from left edge
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.2}\" x2=\"{:.2}\" y2=\"{y:.2}\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n",
      5.0
    ));
    // Label to the left of the frame
    let label = format_tick_in_sequence(t_val, y_step);
    svg.push_str(&format!(
      "<text x=\"{:.2}\" y=\"{y:.2}\" fill=\"{tick_label_fill}\" font-size=\"12\" font-family=\"monospace\" text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>\n",
      -4.0,
      svg_escape(&label),
    ));
  }

  // Right edge: ticks only (no labels)
  for &t_val in &y_ticks {
    let y = coord_y(t_val, bb, svg_h);
    if !y.is_finite() {
      continue;
    }
    svg.push_str(&format!(
      "<line x1=\"{:.2}\" y1=\"{y:.2}\" x2=\"{svg_w:.2}\" y2=\"{y:.2}\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n",
      svg_w - 5.0
    ));
  }
}

/// A single explicit grid-line position with an optional per-line style
/// override (`{pos, style}` form).
struct GridLine {
  pos: f64,
  style: Option<StyleState>,
}

/// Per-axis grid-line specification (one side of `GridLines -> {x, y}`).
enum GridSpec {
  /// No grid lines on this axis.
  None,
  /// Automatic tick positions.
  Automatic,
  /// Explicit positions (each with an optional style).
  Explicit(Vec<GridLine>),
}

impl GridSpec {
  fn is_active(&self) -> bool {
    !matches!(self, Self::None)
  }
}

/// Parse one side of `GridLines -> {xspec, yspec}` (`Automatic`, `None`, or a
/// list of positions; a position may be `{pos, style}` for a per-line style).
fn parse_grid_spec(expr: &Expr) -> GridSpec {
  match expr {
    Expr::Identifier(s) if s == "None" => GridSpec::None,
    Expr::Identifier(s) if s == "Automatic" || s == "All" => {
      GridSpec::Automatic
    }
    Expr::List(entries) => {
      GridSpec::Explicit(entries.iter().filter_map(parse_grid_line).collect())
    }
    // A bare number → a single grid line.
    _ => match expr_to_f64(expr) {
      Some(p) => GridSpec::Explicit(vec![GridLine {
        pos: p,
        style: None,
      }]),
      Option::None => GridSpec::None,
    },
  }
}

/// Parse one explicit grid-line entry: a bare position or `{pos, style}`.
fn parse_grid_line(entry: &Expr) -> Option<GridLine> {
  if let Expr::List(pair) = entry
    && !pair.is_empty()
  {
    let pos = expr_to_f64(&pair[0])?;
    let style = pair.get(1).map(|s| {
      let mut st = StyleState::default();
      apply_directive(s, &mut st);
      st
    });
    return Some(GridLine { pos, style });
  }
  expr_to_f64(entry).map(|pos| GridLine { pos, style: None })
}

/// Resolve a `GridSpec` to the list of (position, style) pairs to draw, using
/// the automatic tick positions and `default_style` where appropriate.
fn grid_positions<'a>(
  spec: &'a GridSpec,
  ticks: &[f64],
  default_style: &'a StyleState,
) -> Vec<(f64, &'a StyleState)> {
  match spec {
    GridSpec::None => Vec::new(),
    GridSpec::Automatic => ticks.iter().map(|&p| (p, default_style)).collect(),
    GridSpec::Explicit(lines) => lines
      .iter()
      .map(|l| (l.pos, l.style.as_ref().unwrap_or(default_style)))
      .collect(),
  }
}

/// Draw grid lines spanning the plot. Vertical lines sit at the `grid_x`
/// positions, horizontal lines at the `grid_y` positions; `Automatic` uses the
/// tick positions. `default_style` (from `GridLinesStyle`) applies to lines
/// without a per-line override.
#[allow(clippy::too_many_arguments)]
fn render_grid_lines(
  svg: &mut String,
  bb: &BBox,
  svg_w: f64,
  svg_h: f64,
  grid_x: &GridSpec,
  grid_y: &GridSpec,
  default_style: &StyleState,
) {
  let x_ticks = generate_ticks(bb.x_min, bb.x_max, 6);
  let y_ticks = generate_ticks(bb.y_min, bb.y_max, 6);

  for (pos, style) in grid_positions(grid_x, &x_ticks, default_style) {
    let x = coord_x(pos, bb, svg_w);
    if !x.is_finite() {
      continue;
    }
    let color = style.effective_color();
    svg.push_str(&format!(
      "<line x1=\"{x:.2}\" y1=\"0\" x2=\"{x:.2}\" y2=\"{svg_h:.2}\" \
       stroke=\"{}\" stroke-width=\"{:.2}\"{}{}/>\n",
      color.to_svg_rgb(),
      thickness_px(style.thickness, bb, svg_w).max(0.5),
      color.opacity_attr(),
      dash_attr(style.dashing.as_ref(), bb, svg_w),
    ));
  }
  for (pos, style) in grid_positions(grid_y, &y_ticks, default_style) {
    let y = coord_y(pos, bb, svg_h);
    if !y.is_finite() {
      continue;
    }
    let color = style.effective_color();
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.2}\" x2=\"{svg_w:.2}\" y2=\"{y:.2}\" \
       stroke=\"{}\" stroke-width=\"{:.2}\"{}{}/>\n",
      color.to_svg_rgb(),
      thickness_px(style.thickness, bb, svg_w).max(0.5),
      color.opacity_attr(),
      dash_attr(style.dashing.as_ref(), bb, svg_w),
    ));
  }
}

/// Truncate a BigFloat digit string to `prec` significant digits for graphical display.
/// E.g. digits="0.84147098480789650665" with prec=3 → "0.841"
fn truncate_bigfloat_digits(digits: &str, prec: usize) -> String {
  if prec == 0 {
    return digits.to_string();
  }
  let negative = digits.starts_with('-');
  let d = if negative { &digits[1..] } else { digits };

  // Count leading zeros after decimal point (they are not significant)
  // e.g. "0.00123" has 2 leading zeros
  let mut sig_seen = 0;
  let mut cut_pos = d.len();
  let mut past_dot = false;
  let mut leading_zeros = true;
  for (i, ch) in d.char_indices() {
    if ch == '.' {
      past_dot = true;
      continue;
    }
    if !ch.is_ascii_digit() {
      cut_pos = i;
      break;
    }
    if leading_zeros && past_dot && ch == '0' {
      continue; // leading fractional zeros are not significant
    }
    if ch != '0' || !leading_zeros {
      leading_zeros = false;
      sig_seen += 1;
      if sig_seen == prec {
        cut_pos = i + ch.len_utf8();
        break;
      }
    }
  }

  let truncated = &d[..cut_pos];
  // Remove trailing dot if nothing follows
  let truncated = truncated.strip_suffix('.').unwrap_or(truncated);
  if negative {
    format!("-{truncated}")
  } else {
    truncated.to_string()
  }
}

/// Information about how a BigFloat should be displayed graphically.
/// For normal numbers, only `mantissa` is set.
/// For scientific notation, `exponent` contains the power of 10.
pub(crate) struct BigFloatDisplay {
  pub(crate) mantissa: String,
  pub(crate) exponent: Option<i64>,
}

/// Prepare BigFloat digits for graphical display, using scientific notation
/// when the number is very large (>= 1e6) or very small (< 1e-5).
/// Returns a struct with the truncated mantissa and optional exponent.
fn bigfloat_display_parts(digits: &str, prec: f64) -> BigFloatDisplay {
  let negative = digits.starts_with('-');
  let d = if negative { &digits[1..] } else { digits };
  let prefix = if negative { "-" } else { "" };

  let dot_pos = d.find('.');
  let int_part = if let Some(dp) = dot_pos { &d[..dp] } else { d };
  let frac_part = if let Some(dp) = dot_pos {
    if dp + 1 < d.len() { &d[dp + 1..] } else { "" }
  } else {
    ""
  };

  let int_nonzero_len = int_part.trim_start_matches('0').len();

  // Large number (6+ integer digits) → scientific notation
  if int_part.len() >= 6 && int_nonzero_len > 0 {
    let all_digits: String =
      int_part.chars().chain(frac_part.chars()).collect();
    let sig_digits = all_digits.trim_end_matches('0');
    if sig_digits.is_empty() {
      return BigFloatDisplay {
        mantissa: format!("{prefix}0."),
        exponent: Some(0),
      };
    }
    let exp = int_part.len() as i64 - 1;
    // Truncate to prec significant digits
    let prec_usize = (prec.ceil() as usize).max(1);
    let trunc_len = prec_usize.min(sig_digits.len());
    let trunc = &sig_digits[..trunc_len];
    let mantissa = if trunc.len() > 1 {
      format!("{}{}.{}", prefix, &trunc[..1], &trunc[1..])
    } else {
      format!("{}{}.", prefix, &trunc[..1])
    };
    return BigFloatDisplay {
      mantissa,
      exponent: Some(exp),
    };
  }

  // Very small number (5+ leading fractional zeros) → scientific notation
  if (int_part == "0" || int_part.is_empty()) && !frac_part.is_empty() {
    let leading_zeros = frac_part.chars().take_while(|&c| c == '0').count();
    if leading_zeros >= 5 {
      let sig_part = &frac_part[leading_zeros..];
      let sig_digits = sig_part.trim_end_matches('0');
      if sig_digits.is_empty() {
        return BigFloatDisplay {
          mantissa: format!("{prefix}0."),
          exponent: Some(0),
        };
      }
      let exp = -(leading_zeros as i64 + 1);
      let prec_usize = (prec.ceil() as usize).max(1);
      let trunc_len = prec_usize.min(sig_digits.len());
      let trunc = &sig_digits[..trunc_len];
      let mantissa = if trunc.len() > 1 {
        format!("{}{}.{}", prefix, &trunc[..1], &trunc[1..])
      } else {
        format!("{}{}.", prefix, &trunc[..1])
      };
      return BigFloatDisplay {
        mantissa,
        exponent: Some(exp),
      };
    }
  }

  // Normal range — just truncate
  BigFloatDisplay {
    mantissa: truncate_bigfloat_digits(digits, (prec.ceil() as usize).max(1)),
    exponent: None,
  }
}

/// Graphical display for a machine-precision real: 6 significant figures,
/// switching to scientific notation when the magnitude is `>= 1e6` or `< 1e-5`,
/// matching Wolfram's StandardForm (`4.086947…` → `4.08695`, `1234567.89` →
/// `1.23457×10^6`, `0.000001234` → `1.234×10^-6`). Unlike the CLI / `eval`
/// InputForm (which keeps the full round-trip precision to match
/// `wolframscript -code`), the notebook front end rounds machine reals for
/// display, so grid cells and typeset labels apply this. Returns the mantissa
/// and an optional power-of-ten exponent (rendered as a superscript by the
/// caller). Non-finite values (Infinity / NaN) fall back to the plain form.
pub(crate) fn machine_real_display_parts(f: f64) -> BigFloatDisplay {
  if !f.is_finite() {
    return BigFloatDisplay {
      mantissa: crate::syntax::format_real(f),
      exponent: None,
    };
  }
  if f == 0.0 {
    return BigFloatDisplay {
      mantissa: "0.".to_string(),
      exponent: None,
    };
  }
  let negative = f.is_sign_negative();
  let sign = if negative { "-" } else { "" };
  // Round to 6 significant figures: `{:.5e}` yields one integer digit plus five
  // fractional digits (six total), with proper round-half-to-even carrying that
  // can bump the exponent (`999999.6` → `1.00000e6`).
  let sci = format!("{:.5e}", f.abs());
  let (mantissa_s, exp_s) = sci.split_once('e').unwrap_or((&sci, "0"));
  let exp: i64 = exp_s.parse().unwrap_or(0);
  let digits: String =
    mantissa_s.chars().filter(char::is_ascii_digit).collect();

  // Scientific notation outside the [-5, 5] decimal exponent window.
  if exp >= 6 || exp <= -6 {
    let sig = {
      let t = digits.trim_end_matches('0');
      if t.is_empty() { "0" } else { t }
    };
    let mantissa = if sig.len() > 1 {
      format!("{}{}.{}", sign, &sig[..1], &sig[1..])
    } else {
      format!("{sign}{sig}.")
    };
    return BigFloatDisplay {
      mantissa,
      exponent: Some(exp),
    };
  }

  // Decimal form: the point sits `exp + 1` digits from the left.
  let dot = exp + 1;
  let mut s = String::from(sign);
  if dot <= 0 {
    s.push_str("0.");
    for _ in 0..(-dot) {
      s.push('0');
    }
    s.push_str(&digits);
  } else {
    let dp = dot as usize;
    if dp >= digits.len() {
      s.push_str(&digits);
      for _ in 0..(dp - digits.len()) {
        s.push('0');
      }
      s.push('.');
    } else {
      s.push_str(&digits[..dp]);
      s.push('.');
      s.push_str(&digits[dp..]);
    }
  }
  // Drop trailing fractional zeros while keeping the decimal point (`4.00000` →
  // `4.`, `12.5000` → `12.5`, `100000.` → `100000.`).
  let mantissa = match s.split_once('.') {
    Some((int_s, frac_s)) => {
      format!("{}.{}", int_s, frac_s.trim_end_matches('0'))
    }
    None => s,
  };
  BigFloatDisplay {
    mantissa,
    exponent: None,
  }
}

pub(crate) fn svg_escape(s: &str) -> String {
  let s = crate::syntax::substitute_private_use_glyphs(s);
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

fn render_primitive(
  prim: &Primitive,
  bb: &BBox,
  svg_w: f64,
  svg_h: f64,
  out: &mut String,
) {
  match prim {
    // The inset is embedded whole, at its own size, centred on its anchor.
    Primitive::InsetGraphic {
      svg,
      x,
      y,
      w,
      h,
      scaled,
    } => {
      let (ax, ay) = resolve_anchor(*x, *y, *scaled, bb);
      out.push_str(&embed_svg_centered(
        svg,
        coord_x(ax, bb, svg_w),
        coord_y(ay, bb, svg_h),
        *w,
        *h,
      ));
    }
    Primitive::PointSingle { x, y, style } => {
      let cx = coord_x(*x, bb, svg_w);
      let cy = coord_y(*y, bb, svg_h);
      let r = point_radius(style.point_size, svg_w);
      let color = style.effective_color();
      if let Some(ref halo) = style.halo {
        out.push_str(&format!(
          "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{:.2}\" fill=\"{}\"{}/>\n",
          r + halo.radius,
          halo.color.to_svg_rgb(),
          halo.color.opacity_attr(),
        ));
      }
      out.push_str(&format!(
        "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r:.2}\" fill=\"{}\"{}/>\n",
        color.to_svg_rgb(),
        color.opacity_attr(),
      ));
    }
    Primitive::PointMulti { points, style } => {
      let r = point_radius(style.point_size, svg_w);
      let color = style.effective_color();
      for &(x, y) in points {
        let cx = coord_x(x, bb, svg_w);
        let cy = coord_y(y, bb, svg_h);
        if let Some(ref halo) = style.halo {
          out.push_str(&format!(
            "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{:.2}\" fill=\"{}\"{}/>\n",
            r + halo.radius,
            halo.color.to_svg_rgb(),
            halo.color.opacity_attr(),
          ));
        }
        out.push_str(&format!(
          "<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{r:.2}\" fill=\"{}\"{}/>\n",
          color.to_svg_rgb(),
          color.opacity_attr(),
        ));
      }
    }
    Primitive::Line { segments, style } => {
      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(style.dashing.as_ref(), bb, svg_w);
      for seg in segments {
        let pts: Vec<String> = seg
          .iter()
          .map(|&(x, y)| {
            format!("{:.2},{:.2}", coord_x(x, bb, svg_w), coord_y(y, bb, svg_h))
          })
          .collect();
        // Draw the halo (contrasting outline) behind the line first.
        if let Some(ref halo) = style.halo {
          let hw = sw + 2.0 * halo.radius;
          out.push_str(&format!(
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{hw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"round\"{}/>\n",
            pts.join(" "),
            halo.color.to_svg_rgb(),
            halo.color.opacity_attr(),
          ));
        }
        out.push_str(&format!(
          "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"butt\"{}{}/>\n",
          pts.join(" "),
          color.to_svg_rgb(),
          color.opacity_attr(),
          dash,
        ));
      }
    }
    Primitive::CircleArc {
      cx,
      cy,
      rx,
      ry,
      angles,
      style,
    } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *rx / bb.width() * svg_w;
      let sry = *ry / bb.height() * svg_h;
      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(style.dashing.as_ref(), bb, svg_w);
      // A partial angular range draws only that open arc (stroked on one
      // side); a full turn (or no range) draws the whole circle as an ellipse.
      let partial = angles
        .filter(|(a1, a2)| (a2 - a1).abs() < std::f64::consts::TAU - 1e-9);
      if let Some((a1, a2)) = partial {
        // SVG y is flipped, so negate the sine component; sweep-flag 0 then
        // traces the arc in the mathematical (counter-clockwise) direction.
        let x1 = scx + srx * a1.cos();
        let y1 = scy - sry * a1.sin();
        let x2 = scx + srx * a2.cos();
        let y2 = scy - sry * a2.sin();
        let large_arc = i32::from((a2 - a1).abs() > std::f64::consts::PI);
        out.push_str(&format!(
          "<path d=\"M {x1:.2},{y1:.2} A {srx:.2},{sry:.2} 0 {large_arc} 0 {x2:.2},{y2:.2}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\"{}{}/>\n",
          color.to_svg_rgb(),
          color.opacity_attr(),
          dash,
        ));
      } else {
        out.push_str(&format!(
          "<ellipse cx=\"{scx:.2}\" cy=\"{scy:.2}\" rx=\"{srx:.2}\" ry=\"{sry:.2}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\"{}{}/>\n",
          color.to_svg_rgb(),
          color.opacity_attr(),
          dash,
        ));
      }
    }
    Primitive::Disk {
      cx,
      cy,
      rx,
      ry,
      style,
    } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *rx / bb.width() * svg_w;
      let sry = *ry / bb.height() * svg_h;
      let color = style.effective_face_color();
      // Edge form for stroke
      let (stroke_color, stroke_width) =
        match edge_stroke(style.edge_form.as_ref(), bb, svg_w) {
          Some((sc, sw)) => (Some(sc), sw),
          None => (None, 0.0),
        };
      let stroke_attr = if let Some(sc) = stroke_color {
        let so = if sc.a < 1.0 {
          format!(" stroke-opacity=\"{}\"", sc.a)
        } else {
          String::new()
        };
        format!(
          " stroke=\"{}\" stroke-width=\"{stroke_width:.2}\"{so}",
          sc.to_svg_rgb()
        )
      } else {
        String::new()
      };
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      out.push_str(&format!(
        "<ellipse cx=\"{scx:.2}\" cy=\"{scy:.2}\" rx=\"{srx:.2}\" ry=\"{sry:.2}\" fill=\"{}\"{}{}/>\n",
        color.to_svg_rgb(),
        fill_opacity,
        stroke_attr,
      ));
    }
    Primitive::DiskSector {
      cx,
      cy,
      rx,
      ry,
      angle1,
      angle2,
      style,
    } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *rx / bb.width() * svg_w;
      let sry = *ry / bb.height() * svg_h;
      // Start point of arc (in SVG coords: negate y because SVG y is flipped)
      let x1 = scx + srx * angle1.cos();
      let y1 = scy - sry * angle1.sin();
      // End point of arc
      let x2 = scx + srx * angle2.cos();
      let y2 = scy - sry * angle2.sin();
      // large-arc flag: 1 if arc spans more than PI
      let sweep_angle = angle2 - angle1;
      let large_arc = i32::from(sweep_angle.abs() > std::f64::consts::PI);
      // Because we negate the sine component when computing arc points
      // (to flip y), the arc geometry is already mirrored.  We therefore
      // need sweep-flag=0 (counter-clockwise in SVG y-down) to trace the
      // correct half of the ellipse.
      let sweep_flag = 0;
      let color = style.effective_face_color();
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      // Edge form for stroke
      let stroke_attr = match edge_stroke(style.edge_form.as_ref(), bb, svg_w) {
        Some((sc, sw)) => {
          let so = if sc.a < 1.0 {
            format!(" stroke-opacity=\"{}\"", sc.a)
          } else {
            String::new()
          };
          format!(
            " stroke=\"{}\" stroke-width=\"{sw:.2}\"{so}",
            sc.to_svg_rgb()
          )
        }
        None => String::new(),
      };
      // Path: move to center, line to arc start, arc to arc end, close
      out.push_str(&format!(
        "<path d=\"M {scx:.2},{scy:.2} L {x1:.2},{y1:.2} A {srx:.2},{sry:.2} 0 {large_arc} {sweep_flag} {x2:.2},{y2:.2} Z\" fill=\"{}\"{}{}/>\n",
        color.to_svg_rgb(),
        fill_opacity,
        stroke_attr,
      ));
    }
    Primitive::RectPrim {
      x_min,
      y_min,
      x_max,
      y_max,
      style,
    } => {
      let sx = coord_x(*x_min, bb, svg_w);
      let sy = coord_y(*y_max, bb, svg_h); // y_max maps to top (lower SVG y)
      let sw = (*x_max - *x_min) / bb.width() * svg_w;
      let sh = (*y_max - *y_min) / bb.height() * svg_h;
      let color = style.effective_face_color();
      // Edge form
      let (stroke_color, stroke_width) =
        match edge_stroke(style.edge_form.as_ref(), bb, svg_w) {
          Some((sc, sw)) => (Some(sc), sw),
          None => (None, 0.0),
        };
      let stroke_attr = if let Some(sc) = stroke_color {
        let so = if sc.a < 1.0 {
          format!(" stroke-opacity=\"{}\"", sc.a)
        } else {
          String::new()
        };
        format!(
          " stroke=\"{}\" stroke-width=\"{stroke_width:.2}\"{so}",
          sc.to_svg_rgb()
        )
      } else {
        String::new()
      };
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      out.push_str(&format!(
        "<rect x=\"{sx:.2}\" y=\"{sy:.2}\" width=\"{sw:.2}\" height=\"{sh:.2}\" fill=\"{}\"{}{}/>\n",
        color.to_svg_rgb(),
        fill_opacity,
        stroke_attr,
      ));
    }
    Primitive::PolygonPrim {
      points,
      holes,
      style,
    } => {
      let color = style.effective_face_color();
      let pts: Vec<String> = points
        .iter()
        .map(|&(x, y)| {
          format!("{:.2},{:.2}", coord_x(x, bb, svg_w), coord_y(y, bb, svg_h))
        })
        .collect();
      // Edge form
      let (stroke_color, stroke_width) =
        match edge_stroke(style.edge_form.as_ref(), bb, svg_w) {
          Some((sc, sw)) => (Some(sc), sw),
          None => (None, 0.0),
        };
      let stroke_attr = if let Some(sc) = stroke_color {
        let so = if sc.a < 1.0 {
          format!(" stroke-opacity=\"{}\"", sc.a)
        } else {
          String::new()
        };
        format!(
          " stroke=\"{}\" stroke-width=\"{stroke_width:.2}\"{so}",
          sc.to_svg_rgb()
        )
      } else {
        String::new()
      };
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      if holes.is_empty() {
        out.push_str(&format!(
          "<polygon points=\"{}\" fill=\"{}\"{}{}/>\n",
          pts.join(" "),
          color.to_svg_rgb(),
          fill_opacity,
          stroke_attr,
        ));
      } else {
        // A polygon with holes becomes one path per boundary, filled with
        // the even-odd rule so the hole subpaths are cut out.
        let subpath = |ring: &[(f64, f64)]| {
          let mut d = String::new();
          for (i, &(x, y)) in ring.iter().enumerate() {
            d.push_str(&format!(
              "{}{:.2},{:.2}",
              if i == 0 { "M" } else { "L" },
              coord_x(x, bb, svg_w),
              coord_y(y, bb, svg_h)
            ));
            d.push(' ');
          }
          d.push_str("Z ");
          d
        };
        let mut d = subpath(points);
        for hole in holes {
          d.push_str(&subpath(hole));
        }
        out.push_str(&format!(
          "<path d=\"{}\" fill=\"{}\" fill-rule=\"evenodd\"{}{}/>\n",
          d.trim_end(),
          color.to_svg_rgb(),
          fill_opacity,
          stroke_attr,
        ));
      }
    }
    Primitive::HalfPlanePrim {
      p,
      v,
      w,
      full,
      style,
    } => {
      // Build a parallelogram that extends far past the visible plot range
      // in every relevant direction; the SVG viewport clips it.
      let ext = 10.0 * (bb.width() + bb.height());
      let corners: Vec<(f64, f64)> = if *full {
        vec![
          (bb.x_min - ext, bb.y_min - ext),
          (bb.x_max + ext, bb.y_min - ext),
          (bb.x_max + ext, bb.y_max + ext),
          (bb.x_min - ext, bb.y_max + ext),
        ]
      } else {
        let norm = |(x, y): (f64, f64)| {
          let len = (x * x + y * y).sqrt();
          (x / len * ext, y / len * ext)
        };
        let (vx, vy) = norm(*v);
        let (wx, wy) = norm(*w);
        vec![
          (p.0 - vx, p.1 - vy),
          (p.0 + vx, p.1 + vy),
          (p.0 + vx + wx, p.1 + vy + wy),
          (p.0 - vx + wx, p.1 - vy + wy),
        ]
      };
      let color = style.effective_color();
      let pts: Vec<String> = corners
        .iter()
        .map(|&(x, y)| {
          format!("{:.2},{:.2}", coord_x(x, bb, svg_w), coord_y(y, bb, svg_h))
        })
        .collect();
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      out.push_str(&format!(
        "<polygon points=\"{}\" fill=\"{}\"{}/>\n",
        pts.join(" "),
        color.to_svg_rgb(),
        fill_opacity,
      ));
    }
    Primitive::ArrowPrim {
      points,
      setback,
      style,
    } => {
      let trimmed = apply_setback(points, *setback);
      if trimmed.len() < 2 {
        // Setback consumed the entire path; nothing to draw
        return;
      }

      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(style.dashing.as_ref(), bb, svg_w);

      // Draw the line
      let pts: Vec<String> = trimmed
        .iter()
        .map(|&(x, y)| {
          format!("{:.2},{:.2}", coord_x(x, bb, svg_w), coord_y(y, bb, svg_h))
        })
        .collect();
      out.push_str(&format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"butt\"{}{}/>\n",
        pts.join(" "),
        color.to_svg_rgb(),
        color.opacity_attr(),
        dash,
      ));

      // Draw the arrowheads. Without an `Arrowheads` directive there is
      // one at the tip, sized to the arrow; with one, there is a head at
      // every position it names — except the ones carrying their own
      // graphic, which `parse_arrow` already placed.
      if trimmed.len() >= 2 {
        // The path in screen pixels, with its cumulative arc length, so a
        // head's position along the arrow resolves to a point and a
        // direction there.
        let screen: Vec<(f64, f64)> = trimmed
          .iter()
          .map(|&(x, y)| (coord_x(x, bb, svg_w), coord_y(y, bb, svg_h)))
          .collect();
        let seg_len: Vec<f64> = screen
          .windows(2)
          .map(|w| {
            ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt()
          })
          .collect();
        let total_len_px: f64 = seg_len.iter().sum();

        // The shape's bounding box, so a self-loop gets a head in
        // proportion to the size it is drawn at rather than its arc length.
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in &screen {
          min_x = min_x.min(x);
          max_x = max_x.max(x);
          min_y = min_y.min(y);
          max_y = max_y.max(y);
        }
        let bbox_diag =
          ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();

        // Wolfram's `Arrowheads[Automatic]` is a head 4 % of the plot's
        // width long, the same measure an explicit size gives — every
        // arrow in a picture carries the same head, however short it is
        // and however thick its shaft. Measured from wolframscript against
        // `Arrowheads[0.1]` in frames of two different aspects.
        //
        // Two caps remain, for shapes a fixed head would swallow: 45 % of
        // the path length, and 40 % of the shape's bounding-box diagonal
        // (which keeps a self-loop's head in proportion to its visible
        // size rather than its arc length).
        let default_head = (svg_w * 0.04)
          .min(total_len_px * 0.45)
          .min(bbox_diag * 0.4)
          .max(1.0);

        // Where a head at fraction `t` sits, and the unit direction of the
        // path there (pointing towards the tip).
        let at = |t: f64| -> Option<((f64, f64), (f64, f64))> {
          if total_len_px <= 0.0 {
            return None;
          }
          let mut want = t.clamp(0.0, 1.0) * total_len_px;
          for (i, &len) in seg_len.iter().enumerate() {
            if (want <= len || i + 1 == seg_len.len()) && len > 0.0 {
              let f = (want / len).clamp(0.0, 1.0);
              let (ax, ay) = screen[i];
              let (bx, by) = screen[i + 1];
              return Some((
                (ax + f * (bx - ax), ay + f * (by - ay)),
                ((bx - ax) / len, (by - ay) / len),
              ));
            }
            want -= len;
          }
          None
        };

        // (position, head length in px, direction sign) for each head.
        let heads: Vec<(f64, f64, f64)> = match &style.arrowheads {
          None => vec![(1.0, default_head, 1.0)],
          Some(specs) => specs
            .iter()
            .filter(|h| h.graphic.is_none())
            .map(|h| {
              // A size is a fraction of the plot width, as Wolfram's is —
              // uncapped, because the whole point of naming it is to fix
              // the head's size. A vector field draws hundreds of arrows
              // barely longer than their heads, and Wolfram lets the head
              // take up nearly all of each one.
              let px = h.size.abs() * svg_w;
              (h.position, px.max(1.0), h.size.signum())
            })
            .collect(),
        };

        for (t, head_len, dir) in heads {
          let Some(((tip_x, tip_y), (ux, uy))) = at(t) else {
            continue;
          };
          let (ux, uy) = (ux * dir, uy * dir);
          // Wolfram's head is a good deal narrower than it is long: its
          // corners sit 0.28 of the length either side of the shaft
          // (measured across sizes and frames from wolframscript).
          let head_half_w = head_len * 0.28;
          // Perpendicular
          let (px, py) = (-uy, ux);
          let base_l_x = tip_x - ux * head_len + px * head_half_w;
          let base_l_y = tip_y - uy * head_len + py * head_half_w;
          let base_r_x = tip_x - ux * head_len - px * head_half_w;
          let base_r_y = tip_y - uy * head_len - py * head_half_w;
          out.push_str(&format!(
            "<polygon points=\"{tip_x:.2},{tip_y:.2} {base_l_x:.2},{base_l_y:.2} {base_r_x:.2},{base_r_y:.2}\" fill=\"{}\"{}/>\n",
            color.to_svg_rgb(),
            color.opacity_attr(),
          ));
        }
      }
    }
    Primitive::TextPrim {
      text,
      x,
      y,
      offset,
      background,
      frame,
      scaled,
      direction,
      style,
    } => {
      let color = style.effective_color();
      let fs = style.font_size;
      // The offset names which point of the label's box sits at the
      // coordinate, so the box moves the other way: half its width per
      // unit horizontally, half its height per unit vertically (and the
      // vertical sign flips, SVG counting y downwards).
      let longest = text
        .split('\n')
        .map(str::chars)
        .map(Iterator::count)
        .max()
        .unwrap_or(0);
      let text_w = longest as f64 * fs * 0.6;
      let text_h = text.split('\n').count() as f64 * fs;
      let (ax, ay) = resolve_anchor(*x, *y, *scaled, bb);
      let sx = coord_x(ax, bb, svg_w) - offset.0 * text_w / 2.0;
      let sy = coord_y(ay, bb, svg_h) + offset.1 * text_h / 2.0;
      // A fourth `direction` argument tilts the label's baseline to match
      // that vector — carried in data coordinates, so it has to go through
      // the same x/y pixel-per-unit scaling `coord_x`/`coord_y` apply (and
      // the same y-flip) before it becomes a screen-space angle for SVG's
      // `rotate()`.
      let rotate_attr = match direction {
        Some((dx, dy)) if *dx != 0.0 || *dy != 0.0 => {
          let px = dx * svg_w / bb.width();
          let py = -dy * svg_h / bb.height();
          format!(
            " transform=\"rotate({:.3} {sx:.2} {sy:.2})\"",
            py.atan2(px).to_degrees()
          )
        }
        _ => String::new(),
      };
      // `Background -> colour` paints a panel behind the label, which is
      // what keeps a value readable over whatever it is placed on; a
      // `Framed` label additionally gets the border drawn around it.
      if background.is_some() || frame.is_some() {
        let (fill, fill_opacity) = match background {
          Some(bg) => (bg.to_svg_rgb(), bg.opacity_attr()),
          None => ("none".to_string(), String::new()),
        };
        let stroke = match frame {
          Some(fc) => format!(" stroke=\"{}\"", fc.to_svg_rgb()),
          None => String::new(),
        };
        out.push_str(&format!(
          "<rect x=\"{:.2}\" y=\"{:.2}\" width=\"{text_w:.2}\" height=\"{text_h:.2}\" fill=\"{fill}\"{fill_opacity}{stroke}/>\n",
          sx - text_w / 2.0,
          sy - text_h / 2.0,
        ));
      }
      let ff_attr = if style.font_family.is_empty() {
        String::new()
      } else {
        format!(" font-family=\"{}\"", svg_escape(&style.font_family))
      };

      if text.contains('\n') {
        // Multi-line text with tspan
        let lines: Vec<&str> = text.split('\n').collect();
        out.push_str(&format!(
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\"{ff_attr} text-anchor=\"middle\" dominant-baseline=\"central\"{}{rotate_attr}>",
          color.to_svg_rgb(),
          style.font_weight,
          style.font_style,
          color.opacity_attr(),
        ));
        for (i, line) in lines.iter().enumerate() {
          if i == 0 {
            out.push_str(&format!(
              "<tspan x=\"{sx:.2}\" dy=\"0\">{}</tspan>",
              svg_escape(line)
            ));
          } else {
            out.push_str(&format!(
              "<tspan x=\"{sx:.2}\" dy=\"{fs}\">{}</tspan>",
              svg_escape(line)
            ));
          }
        }
        out.push_str("</text>\n");
      } else {
        out.push_str(&format!(
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\"{ff_attr} text-anchor=\"middle\" dominant-baseline=\"central\"{}{rotate_attr}>{}</text>\n",
          color.to_svg_rgb(),
          style.font_weight,
          style.font_style,
          color.opacity_attr(),
          svg_escape(text),
        ));
      }
    }
    Primitive::BezierCurvePrim { points, style } => {
      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(style.dashing.as_ref(), bb, svg_w);

      if points.len() < 2 {
        return;
      }

      let mut d = String::new();
      let (x0, y0) = points[0];
      d.push_str(&format!(
        "M{:.2},{:.2}",
        coord_x(x0, bb, svg_w),
        coord_y(y0, bb, svg_h)
      ));

      if points.len() == 2 {
        let (x1, y1) = points[1];
        d.push_str(&format!(
          " L{:.2},{:.2}",
          coord_x(x1, bb, svg_w),
          coord_y(y1, bb, svg_h)
        ));
      } else if points.len() == 3 {
        // Quadratic bezier
        let (x1, y1) = points[1];
        let (x2, y2) = points[2];
        d.push_str(&format!(
          " Q{:.2},{:.2} {:.2},{:.2}",
          coord_x(x1, bb, svg_w),
          coord_y(y1, bb, svg_h),
          coord_x(x2, bb, svg_w),
          coord_y(y2, bb, svg_h),
        ));
      } else if points.len() == 4 {
        // Cubic bezier
        let (x1, y1) = points[1];
        let (x2, y2) = points[2];
        let (x3, y3) = points[3];
        d.push_str(&format!(
          " C{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}",
          coord_x(x1, bb, svg_w),
          coord_y(y1, bb, svg_h),
          coord_x(x2, bb, svg_w),
          coord_y(y2, bb, svg_h),
          coord_x(x3, bb, svg_w),
          coord_y(y3, bb, svg_h),
        ));
      } else {
        // For more points, chain cubic segments (every 3 after the first)
        let mut i = 1;
        while i + 2 < points.len() {
          let (x1, y1) = points[i];
          let (x2, y2) = points[i + 1];
          let (x3, y3) = points[i + 2];
          d.push_str(&format!(
            " C{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}",
            coord_x(x1, bb, svg_w),
            coord_y(y1, bb, svg_h),
            coord_x(x2, bb, svg_w),
            coord_y(y2, bb, svg_h),
            coord_x(x3, bb, svg_w),
            coord_y(y3, bb, svg_h),
          ));
          i += 3;
        }
        // Handle remaining points
        let remaining = &points[i..];
        if remaining.len() == 2 {
          let (x1, y1) = remaining[0];
          let (x2, y2) = remaining[1];
          d.push_str(&format!(
            " Q{:.2},{:.2} {:.2},{:.2}",
            coord_x(x1, bb, svg_w),
            coord_y(y1, bb, svg_h),
            coord_x(x2, bb, svg_w),
            coord_y(y2, bb, svg_h),
          ));
        } else if remaining.len() == 1 {
          let (x1, y1) = remaining[0];
          d.push_str(&format!(
            " L{:.2},{:.2}",
            coord_x(x1, bb, svg_w),
            coord_y(y1, bb, svg_h),
          ));
        }
      }

      out.push_str(&format!(
        "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"butt\"{}{}/>\n",
        d,
        color.to_svg_rgb(),
        color.opacity_attr(),
        dash,
      ));
    }
    Primitive::RasterPrim {
      data,
      x_min,
      y_min,
      x_max,
      y_max,
    } => {
      let nrows = data.len();
      if nrows == 0 {
        return;
      }
      let ncols = data.iter().map(std::vec::Vec::len).max().unwrap_or(0);
      if ncols == 0 {
        return;
      }

      // Large rasters (photos) are embedded as one PNG <image> instead of
      // one <rect> per pixel: a 400x520 image would otherwise emit 208k
      // rects, which SVG renderers cannot handle interactively. Small
      // rasters keep the exact rect fills (crisp pixel edges at any zoom).
      if nrows * ncols > 4096 {
        use base64::Engine;
        let mut img = image::RgbaImage::new(ncols as u32, nrows as u32);
        for (ri, row) in data.iter().enumerate() {
          for ci in 0..ncols {
            let color =
              row.get(ci).copied().unwrap_or(Color::new(0.0, 0.0, 0.0));
            let px = image::Rgba([
              (color.r * 255.0).round().clamp(0.0, 255.0) as u8,
              (color.g * 255.0).round().clamp(0.0, 255.0) as u8,
              (color.b * 255.0).round().clamp(0.0, 255.0) as u8,
              (color.a * 255.0).round().clamp(0.0, 255.0) as u8,
            ]);
            // Row 0 in Wolfram is at the bottom; PNG row 0 is at the top.
            img.put_pixel(ci as u32, (nrows - 1 - ri) as u32, px);
          }
        }
        let mut png = Vec::new();
        if image::DynamicImage::ImageRgba8(img)
          .write_to(
            &mut std::io::Cursor::new(&mut png),
            image::ImageFormat::Png,
          )
          .is_ok()
        {
          let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
          let sx = coord_x(*x_min, bb, svg_w);
          let sy = coord_y(*y_max, bb, svg_h);
          let sw = (x_max - x_min) / bb.width() * svg_w;
          let sh = (y_max - y_min) / bb.height() * svg_h;
          out.push_str(&format!(
            "<image x=\"{sx:.2}\" y=\"{sy:.2}\" width=\"{sw:.2}\" height=\"{sh:.2}\" preserveAspectRatio=\"none\" href=\"data:image/png;base64,{b64}\"/>\n"
          ));
          return;
        }
        // PNG encoding failed: fall through to the per-pixel rects.
      }

      let cell_w = (x_max - x_min) / ncols as f64;
      let cell_h = (y_max - y_min) / nrows as f64;

      // Row 0 in Wolfram is at the bottom (y_min), so iterate bottom-to-top
      for (ri, row) in data.iter().enumerate() {
        let y = y_min + ri as f64 * cell_h;
        for (ci, color) in row.iter().enumerate() {
          let x = x_min + ci as f64 * cell_w;

          let sx = coord_x(x, bb, svg_w);
          let sy = coord_y(y + cell_h, bb, svg_h); // top edge in SVG
          let sw = cell_w / bb.width() * svg_w;
          let sh = cell_h / bb.height() * svg_h;

          let opacity_attr = if color.a < 1.0 {
            format!(" fill-opacity=\"{}\"", color.a)
          } else {
            String::new()
          };
          out.push_str(&format!(
            "<rect x=\"{sx:.2}\" y=\"{sy:.2}\" width=\"{sw:.2}\" height=\"{sh:.2}\" fill=\"{}\"{}/>\n",
            color.to_svg_rgb(),
            opacity_attr,
          ));
        }
      }
    }
    // A screen-space icon (e.g. a Locator's appearance) centered on its
    // data-space anchor: embed the pre-rendered SVG at its pixel size.
    Primitive::MarkerPrim { x, y, w, h, svg } => {
      let scx = coord_x(*x, bb, svg_w);
      let scy = coord_y(*y, bb, svg_h);
      out.push_str(&format!(
        "<svg x=\"{:.2}\" y=\"{:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" viewBox=\"0 0 {w:.2} {h:.2}\">\n",
        scx - w / 2.0,
        scy - h / 2.0,
      ));
      out.push_str(strip_svg_wrapper(svg));
      out.push_str("</svg>\n");
    }
  }
}

// ── Options parsing ──────────────────────────────────────────────────────

fn parse_plot_range(
  expr: &Expr,
) -> (
  std::option::Option<(f64, f64)>,
  std::option::Option<(f64, f64)>,
) {
  match expr {
    Expr::Identifier(s) if s == "All" || s == "Automatic" => (None, None),
    Expr::List(items) if items.len() == 2 => {
      let x_range = parse_range_spec(&items[0]);
      let y_range = parse_range_spec(&items[1]);
      (x_range, y_range)
    }
    _ => {
      // Single range applies to both axes
      let r = parse_range_spec(expr);
      (r, r)
    }
  }
}

fn parse_range_spec(expr: &Expr) -> Option<(f64, f64)> {
  match expr {
    Expr::List(items) if items.len() == 2 => {
      let lo = expr_to_f64(&items[0])?;
      let hi = expr_to_f64(&items[1])?;
      Some((lo, hi))
    }
    Expr::Identifier(s) if s == "All" || s == "Automatic" => None,
    _ => {
      // Single number n means {-n, n}
      expr_to_f64(expr).map(|v| (-v, v))
    }
  }
}

fn parse_background(expr: &Expr) -> Option<Color> {
  parse_color(expr)
}

fn parse_axes(expr: &Expr) -> Option<(bool, bool)> {
  fn parse_bool(expr: &Expr) -> Option<bool> {
    match expr {
      // `Automatic` shows the axis (positioned automatically), e.g. the
      // common `Axes -> {Automatic, False}` form.
      Expr::Identifier(s) if s == "True" || s == "Automatic" => Some(true),
      Expr::Identifier(s) if s == "False" => Some(false),
      _ => None,
    }
  }

  match expr {
    Expr::Identifier(s) if s == "True" => Some((true, true)),
    Expr::Identifier(s) if s == "False" => Some((false, false)),
    Expr::List(items) if items.len() == 2 => {
      let x_axis = parse_bool(&items[0])?;
      let y_axis = parse_bool(&items[1])?;
      Some((x_axis, y_axis))
    }
    _ => None,
  }
}

// ── GraphicsBox generation ───────────────────────────────────────────────

use crate::functions::graphicsbox as gbox;

/// Track style changes and emit corresponding box directives.
struct BoxStyleTracker {
  color: (f64, f64, f64),
  opacity: f64,
  thickness: f64,
}

impl Default for BoxStyleTracker {
  fn default() -> Self {
    Self {
      color: (0.0, 0.0, 0.0), // Black
      opacity: 1.0,
      thickness: -1.0,
    }
  }
}

impl BoxStyleTracker {
  /// Emit directives needed to switch to the given style, returning any new directives.
  fn emit_style_changes(&mut self, style: &StyleState) -> Vec<String> {
    let mut directives = Vec::new();
    let new_color = (style.color.r, style.color.g, style.color.b);
    if (new_color.0 - self.color.0).abs() > 1e-6
      || (new_color.1 - self.color.1).abs() > 1e-6
      || (new_color.2 - self.color.2).abs() > 1e-6
    {
      directives.push(gbox::rgbcolor_box(
        new_color.0,
        new_color.1,
        new_color.2,
      ));
      self.color = new_color;
    }
    if (style.opacity - self.opacity).abs() > 1e-6 {
      directives.push(gbox::opacity_box(style.opacity));
      self.opacity = style.opacity;
    }
    if (style.thickness - self.thickness).abs() > 1e-6 {
      directives.push(gbox::abs_thickness_box(style.thickness));
      self.thickness = style.thickness;
    }
    directives
  }
}

/// Convert a list of primitives into GraphicsBox element strings.
fn primitives_to_box_elements(primitives: &[Primitive]) -> Vec<String> {
  let mut elements = Vec::new();
  let mut tracker = BoxStyleTracker::default();

  for prim in primitives {
    match prim {
      Primitive::PointSingle { x, y, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::point_box(*x, *y));
      }
      Primitive::PointMulti { points, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::point_box_multi(points));
      }
      Primitive::Line { segments, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.extend(gbox::line_box(segments));
      }
      Primitive::CircleArc {
        cx, cy, rx, style, ..
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::circle_box(*cx, *cy, *rx));
      }
      Primitive::Disk {
        cx, cy, rx, style, ..
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::disk_box(*cx, *cy, *rx));
      }
      Primitive::DiskSector {
        cx,
        cy,
        rx,
        angle1,
        angle2,
        style,
        ..
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::disk_sector_box(*cx, *cy, *rx, *angle1, *angle2));
      }
      Primitive::RectPrim {
        x_min,
        y_min,
        x_max,
        y_max,
        style,
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::rectangle_box(*x_min, *y_min, *x_max, *y_max));
      }
      Primitive::PolygonPrim {
        points,
        holes,
        style,
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(if holes.is_empty() {
          gbox::polygon_box(points)
        } else {
          gbox::polygon_with_holes_box(points, holes)
        });
      }
      Primitive::ArrowPrim {
        points,
        setback,
        style,
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::arrow_box(points, *setback));
      }
      Primitive::TextPrim {
        text, x, y, style, ..
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::inset_box(text, *x, *y));
      }
      Primitive::BezierCurvePrim { points, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::bezier_curve_box(points));
      }
      Primitive::RasterPrim { .. } => {
        // RasterBox is not yet supported in .nb export; skip
      }
      Primitive::InsetGraphic { .. } => {
        // An embedded rendering has no fixed-coordinate box form; skip
      }
      Primitive::HalfPlanePrim { .. } => {
        // Unbounded fills have no fixed-coordinate box form; skip
      }
      Primitive::MarkerPrim { .. } => {
        // Screen-space marker icons have no box form; skip
      }
    }
  }

  elements
}

// ── Entry point ──────────────────────────────────────────────────────────

/// Splice an option *list* in the option slots into individual rules.
/// `Graphics[prims, {ImageSize -> 100, Frame -> True}]` is the shape the
/// Wolfram front end stores a picture in — and what `ColorData[name, "Image"]`
/// hands back — and it means exactly the same as the flat
/// `Graphics[prims, ImageSize -> 100, Frame -> True]`.
///
/// Only a list made up entirely of rules is spliced, so a list of primitives
/// that follows the content is left where it is.
pub fn splice_option_lists(args: &[Expr]) -> Vec<Expr> {
  let mut out = Vec::with_capacity(args.len());
  for (i, arg) in args.iter().enumerate() {
    match arg {
      Expr::List(items)
        if i > 0
          && !items.is_empty()
          && items.iter().all(|item| {
            matches!(item, Expr::Rule { .. } | Expr::RuleDelayed { .. })
          }) =>
      {
        out.extend(items.iter().cloned());
      }
      _ => out.push(arg.clone()),
    }
  }
  out
}

pub fn graphics_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &splice_option_lists(args)[..];
  // First arg is the content (primitives + directives)
  // Evaluate it so that Table/Map/etc. produce concrete lists
  // Remaining args are options as Rule expressions
  let content = evaluate_expr_to_expr(&args[0])?;

  // `Graphics[g]` where `g` is itself a finished picture — a plot, or
  // another `Graphics` — is just `g`: the wrapper adds nothing and the
  // inner picture keeps its own options (in Wolfram an outer `Frame ->
  // True` on such a wrapper has no effect). Demonstrations write
  // `Graphics[ContourPlot[…]]` when collecting layers for a later `Show`,
  // and the plot would otherwise be dropped for not being a primitive.
  match &content {
    Expr::Graphics { is_3d: false, .. } => return Ok(content),
    // An inner `Graphics[…]` call is still a call at this point (it only
    // renders at the output stage), so render that one instead of taking
    // its primitives and losing its own options.
    Expr::FunctionCall { name, args: inner }
      if name == "Graphics" && !inner.is_empty() =>
    {
      let inner: Vec<Expr> = inner.iter().cloned().collect();
      return graphics_ast(&inner);
    }
    _ => {}
  }

  // Parse options
  // Whether an `ImageSize` was asked for. Wolfram's is the whole picture,
  // axes and labels included, so the margins come out of it rather than
  // being added around it.
  let mut explicit_size = false;
  let mut svg_width: u32 = 360;
  let mut svg_height: u32 = 225;
  let mut explicit_height = false;
  let mut full_width = false;
  let mut plot_range_x: Option<(f64, f64)> = None;
  let mut plot_range_y: Option<(f64, f64)> = None;
  let mut background: Option<Color> = None;
  let mut plot_label: Option<Vec<String>> = None;
  // `AxesLabel -> {xlabel, ylabel}`, already typeset as SVG markup.
  let mut axes_label: Option<(String, String)> = None;
  // `FrameLabel -> {bottom, left}` (or the nested four-edge form), as SVG
  // markup: the captions that sit outside the frame, in the order
  // bottom, left, top, right.
  let mut frame_label: Option<(String, String, String, String)> = None;
  // `Ticks -> {xspec, yspec}`: which tick marks each axis carries.
  let mut ticks_x = TickSpec::Automatic;
  let mut ticks_y = TickSpec::Automatic;
  let mut axes = (false, false);
  let mut frame = false;
  // `FrameTicks -> False | None` keeps the border but drops the tick
  // marks and their labels, so the frame becomes a plain box.
  let mut frame_ticks = true;
  let mut grid_x = GridSpec::None;
  let mut grid_y = GridSpec::None;
  let mut grid_style: Option<StyleState> = None;
  // When true, skip uniform scaling so x and y axes scale independently
  // (needed for plots where data aspect ≠ image aspect).
  let mut aspect_ratio_full = false;
  // `AspectRatio -> r`: height/width of the drawing, applied after the option
  // list is read so it does not depend on where `ImageSize` sits in it.
  let mut aspect_ratio: Option<f64> = None;
  // `ImagePadding -> {{left, right}, {bottom, top}}` (or a single number for
  // all four sides): the room reserved around the drawing area, replacing the
  // automatic margins that the frame and its tick labels would ask for.
  let mut image_padding: Option<[f64; 4]> = None;
  // `Prolog -> …` / `Epilog -> …`: extra primitives drawn under and over
  // the picture's own content. Neither takes part in the plot range, so
  // they are collected only once the range is settled.
  let mut prolog: Option<Expr> = None;
  let mut epilog: Option<Expr> = None;

  for raw_opt in &args[1..] {
    let opt =
      evaluate_expr_to_expr(raw_opt).unwrap_or_else(|_| raw_opt.clone());
    if let Some((name, replacement)) = option_name_value(&opt) {
      let replacement = &*replacement;
      match name {
        "ImageSize" => {
          explicit_size = true;
          // A height was named only when the `{w, h}` form gives a number
          // for it; `{w, Automatic}` still follows the data aspect.
          if let Expr::List(items) = replacement
            && items.len() == 2
            && !matches!(&items[1], Expr::Identifier(n) if n == "Automatic")
          {
            explicit_height = true;
          }
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        "PlotRange" => {
          let (xr, yr) = parse_plot_range(replacement);
          plot_range_x = xr;
          plot_range_y = yr;
        }
        "Prolog" => prolog = Some(replacement.clone()),
        "Epilog" => epilog = Some(replacement.clone()),
        "PlotLabel" => {
          // Any expression can label the plot (`Row[…]`, a string, a
          // symbol). It is typeset like the rest of the graphic, so machine
          // reals show 6 significant figures and `Subscript`/`Superscript`
          // (and a string's inline linear-syntax boxes) become shifted
          // tspans — the label is drawn, not printed.
          match replacement {
            Expr::Identifier(s) if s == "None" => {}
            other => {
              let lines = expr_to_svg_markup_lines(other);
              if lines.iter().any(|l| !l.is_empty()) {
                plot_label = Some(lines);
              }
            }
          }
        }
        "Background" => {
          background = parse_background(replacement);
        }
        "Axes" => {
          if let Some(parsed_axes) = parse_axes(replacement) {
            axes = parsed_axes;
          }
        }
        // `Ticks -> {xspec, yspec}`: `None` draws none, a list gives the
        // positions to mark (each optionally `{pos, label}`).
        "Ticks" => match replacement {
          Expr::Identifier(s) if s == "None" || s == "False" => {
            ticks_x = TickSpec::None;
            ticks_y = TickSpec::None;
          }
          Expr::Identifier(s) if s == "Automatic" || s == "All" => {
            ticks_x = TickSpec::Automatic;
            ticks_y = TickSpec::Automatic;
          }
          // `Ticks -> {xspec}` states the x ticks only; the y axis keeps
          // its default.
          Expr::List(items) if (1..=2).contains(&items.len()) => {
            ticks_x = parse_tick_spec(&items[0]);
            if let Some(y) = items.get(1) {
              ticks_y = parse_tick_spec(y);
            }
          }
          _ => {}
        },
        // `AxesLabel -> {x, y}` (or a single label for the x axis).
        "AxesLabel" => match replacement {
          Expr::List(items) if items.len() == 2 => {
            let label = |e: &Expr| match e {
              Expr::Identifier(s) if s == "None" => String::new(),
              other => expr_to_svg_markup(other),
            };
            axes_label = Some((label(&items[0]), label(&items[1])));
          }
          Expr::Identifier(s) if s == "None" => axes_label = None,
          other => {
            axes_label = Some((expr_to_svg_markup(other), String::new()));
          }
        },
        // `FrameLabel -> {bottom, left}` / `{{left, right}, {bottom, top}}`
        // captions the frame edges.
        "FrameLabel" => {
          let fl = crate::functions::plot::parse_frame_label(replacement);
          if !fl.bottom.is_empty()
            || !fl.left.is_empty()
            || !fl.top.is_empty()
            || !fl.right.is_empty()
          {
            frame_label = Some((
              svg_escape(&fl.bottom),
              svg_escape(&fl.left),
              svg_escape(&fl.top),
              svg_escape(&fl.right),
            ));
          }
        }
        "Frame" => {
          if let Expr::Identifier(s) = replacement {
            if s == "True" {
              frame = true;
            }
          } else if let Expr::FunctionCall { name: fn_name, .. } = replacement
            && fn_name == "True"
          {
            frame = true;
          }
        }
        "FrameTicks" => {
          if matches!(replacement, Expr::Identifier(s) if s == "False" || s == "None")
          {
            frame_ticks = false;
          }
        }
        "ImagePadding" => {
          image_padding =
            crate::functions::plot::parse_image_padding(replacement);
        }
        "GridLines" => match replacement {
          Expr::Identifier(s)
            if s == "Automatic" || s == "True" || s == "All" =>
          {
            grid_x = GridSpec::Automatic;
            grid_y = GridSpec::Automatic;
          }
          Expr::Identifier(s) if s == "None" || s == "False" => {
            grid_x = GridSpec::None;
            grid_y = GridSpec::None;
          }
          // {xspec, yspec}: each side is Automatic, None, or an explicit list
          // of positions (a position may be `{pos, style}` for a per-line
          // style).
          Expr::List(items) if items.len() == 2 => {
            grid_x = parse_grid_spec(&items[0]);
            grid_y = parse_grid_spec(&items[1]);
          }
          _ => {}
        },
        "GridLinesStyle" => {
          let mut st = StyleState::default();
          apply_directive(replacement, &mut st);
          grid_style = Some(st);
        }
        "AspectRatio" => {
          // AspectRatio -> Full: skip uniform scaling (used by plots)
          if let Expr::Identifier(s) = replacement {
            if s == "Full" {
              aspect_ratio_full = true;
            }
          } else if let Some(r) = expr_to_f64(replacement)
            && r > 0.0
          {
            aspect_ratio = Some(r);
            aspect_ratio_full = true;
          }
        }
        _ => {}
      }
    }
  }

  // The height an `AspectRatio` asks for follows from the *final* width, so it
  // is worked out once the whole option list has been read: options mean the
  // same thing in either order, and `Graphics[…, AspectRatio -> 1/8,
  // ImageSize -> 100]` draws the same strip as with the two swapped. An
  // `ImageSize -> {w, h}` names the height outright and keeps it.
  if let Some(r) = aspect_ratio
    && !explicit_height
  {
    svg_height = ((svg_width as f64 * r).round() as u32).max(1);
    explicit_height = true;
  }

  // Collect primitives
  let mut style = StyleState::default();
  let mut primitives = Vec::new();
  let mut errors: Vec<String> = Vec::new();
  collect_primitives(&content, &mut style, &mut primitives, &mut errors);

  // Compute bounding box
  let mut bb = BBox::empty();
  for prim in &primitives {
    bb.merge(&primitive_bbox(prim));
  }

  if bb.is_empty() {
    // Default range if nothing to draw
    bb = BBox {
      x_min: -1.0,
      x_max: 1.0,
      y_min: -1.0,
      y_max: 1.0,
    };
  }

  // Apply 4% padding
  bb = bb.with_padding(0.04);

  // Apply PlotRange overrides
  if let Some((lo, hi)) = plot_range_x {
    bb.x_min = lo;
    bb.x_max = hi;
  }
  if let Some((lo, hi)) = plot_range_y {
    bb.y_min = lo;
    bb.y_max = hi;
  }

  // The plot range is settled, so the Prolog's and Epilog's own primitives
  // can join now — under and over the content respectively — without
  // having stretched the range they are placed against.
  if let Some(under) = &prolog {
    let mut under_prims = Vec::new();
    collect_primitives(
      under,
      &mut StyleState::default(),
      &mut under_prims,
      &mut errors,
    );
    under_prims.append(&mut primitives);
    primitives = under_prims;
  }
  if let Some(over) = &epilog {
    collect_primitives(
      over,
      &mut StyleState::default(),
      &mut primitives,
      &mut errors,
    );
  }

  // Adjust aspect ratio to match data unless explicitly set via ImageSize -> {w, h}
  if !explicit_height {
    let data_aspect = bb.height() / bb.width();
    if data_aspect.is_finite() && data_aspect > 0.0 {
      svg_height = (svg_width as f64 * data_aspect).round() as u32;
    }
  }

  // Generate SVG
  let svg_w = svg_width as f64;
  let svg_h = svg_height as f64;

  // Ensure uniform scaling: expand the bounding box so that
  // bb.width()/bb.height() == svg_w/svg_h.  This guarantees that
  // 1 data-unit maps to the same number of pixels in both x and y,
  // so circles are always rendered round.
  // Skipped when AspectRatio -> Full (plots need independent axis scaling).
  let svg_aspect = svg_w / svg_h;
  let data_aspect_wh = bb.width() / bb.height();
  if !aspect_ratio_full
    && svg_aspect.is_finite()
    && data_aspect_wh.is_finite()
    && (svg_aspect - data_aspect_wh).abs() > 1e-9
  {
    if svg_aspect > data_aspect_wh {
      // SVG is wider than data: expand bb width, centering horizontally
      let new_width = bb.height() * svg_aspect;
      let extra = new_width - bb.width();
      bb.x_min -= extra / 2.0;
      bb.x_max += extra / 2.0;
    } else {
      // SVG is taller than data: expand bb height, centering vertically
      let new_height = bb.width() / svg_aspect;
      let extra = new_height - bb.height();
      bb.y_min -= extra / 2.0;
      bb.y_max += extra / 2.0;
    }
  }

  // Compute margins for axis/frame tick labels. A PlotLabel reserves an
  // extra strip above the drawing area for its centered title text.
  // Without tick labels the frame needs no gutter, just room for its stroke.
  let frame_gutter = frame && frame_ticks;
  let has_bottom_caption =
    frame_label.as_ref().is_some_and(|(b, ..)| !b.is_empty());
  let has_left_caption =
    frame_label.as_ref().is_some_and(|(_, l, ..)| !l.is_empty());
  let has_top_caption = frame_label
    .as_ref()
    .is_some_and(|(_, _, t, _)| !t.is_empty());
  let has_right_caption = frame_label
    .as_ref()
    .is_some_and(|(_, _, _, r)| !r.is_empty());
  // An axis whose range spans zero is drawn through the middle of the
  // picture and carries its tick labels there too, so it needs no gutter
  // beside the drawing area — only the small padding Wolfram leaves all
  // round. Reserving the full gutter for one pushed the drawing hard
  // against the opposite edge and clipped whatever sat at the far side.
  let y_axis_interior = axes.1 && bb.x_min <= 0.0 && 0.0 <= bb.x_max;
  let x_axis_interior = axes.0 && bb.y_min <= 0.0 && 0.0 <= bb.y_max;
  let margin_left: f64 = if frame_gutter || (axes.1 && !y_axis_interior) {
    50.0
  } else if frame {
    10.0
  } else if y_axis_interior {
    6.0
  } else {
    0.0
  } + if has_left_caption { 20.0 } else { 0.0 };
  let margin_bottom: f64 = if frame_gutter || (axes.0 && !x_axis_interior) {
    25.0
  } else if frame {
    10.0
  } else if x_axis_interior {
    6.0
  } else {
    0.0
  } + if has_bottom_caption { 20.0 } else { 0.0 };
  // An AxesLabel sits at the end of its axis (Wolfram's placement), so
  // the x label needs room to the right and the y label room above.
  let has_x_axis_label =
    axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_y_axis_label =
    axes_label.as_ref().is_some_and(|(_, y)| !y.is_empty());
  let margin_right: f64 = if frame {
    10.0
  } else if y_axis_interior {
    // Balance the padding an interior y axis leaves on the left, so the
    // drawing area sits centred instead of running off the right edge.
    6.0
  } else {
    0.0
  } + if has_x_axis_label { 24.0 } else { 0.0 }
    + if has_right_caption { 20.0 } else { 0.0 };
  // A multi-line title (a `Grid`/`Column` label) claims one further line
  // height per extra row on top of the single-line strip.
  let label_strip: f64 = match &plot_label {
    Some(lines) => 26.0 + 18.0 * (lines.len().saturating_sub(1)) as f64,
    None => 0.0,
  };
  let margin_top: f64 = if frame { 10.0 } else { 0.0 }
    + label_strip
    + if has_y_axis_label && label_strip == 0.0 {
      20.0
    } else {
      0.0
    }
    + if has_top_caption { 20.0 } else { 0.0 };
  // `ImagePadding` states the room around the drawing area outright, so it
  // replaces the margins the frame and its labels would otherwise claim
  // (Wolfram draws the tick labels inside that padding).
  let (margin_left, margin_right, margin_bottom, margin_top) =
    match image_padding {
      Some([left, right, bottom, top]) => (left, right, bottom, top),
      None => (margin_left, margin_right, margin_bottom, margin_top),
    };
  // The size asked for is the whole picture, so the drawing area is what
  // is left of it once the axes and their labels have taken their room.
  let (svg_w, svg_h) = if explicit_size {
    (
      (svg_w - margin_left - margin_right).max(1.0),
      (svg_h - margin_bottom - margin_top).max(1.0),
    )
  } else {
    (svg_w, svg_h)
  };
  let total_width = svg_w + margin_left + margin_right;
  let total_height = svg_h + margin_bottom + margin_top;

  let mut svg = String::with_capacity(4096);

  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {total_width:.0} {total_height:.0}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{total_width:.0}\" height=\"{total_height:.0}\" viewBox=\"0 0 {total_width:.0} {total_height:.0}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    ));
  }

  // Drop-shadow filter definitions (one per distinct shadow in use)
  let shadow_defs: Vec<&DropShadow> = {
    let mut seen: Vec<&DropShadow> = Vec::new();
    for prim in &primitives {
      if let Some(ds) = prim.style().and_then(|s| s.drop_shadow.as_ref())
        && !seen.contains(&ds)
      {
        seen.push(ds);
      }
    }
    seen
  };
  if !shadow_defs.is_empty() {
    svg.push_str("<defs>\n");
    for ds in &shadow_defs {
      svg.push_str(&ds.filter_def(1.0));
      svg.push('\n');
    }
    svg.push_str("</defs>\n");
  }

  // Background (covers the full SVG including margins)
  if let Some(bg) = background {
    svg.push_str(&format!(
      "<rect width=\"{total_width:.0}\" height=\"{total_height:.0}\" fill=\"{}\"/>\n",
      bg.to_svg_rgb(),
    ));
  }

  // PlotLabel: centered above the drawing area, in the reserved strip. It
  // arrives already typeset as SVG markup (sub/superscript tspans included).
  if let Some(lines) = &plot_label {
    let cx = margin_left + svg_w / 2.0;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"17\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"16\" fill=\"#333333\">",
    ));
    for (i, line) in lines.iter().enumerate() {
      if i == 0 {
        svg.push_str(line);
      } else {
        svg.push_str(&format!("<tspan x=\"{cx:.1}\" dy=\"18\">{line}</tspan>"));
      }
    }
    svg.push_str("</text>\n");
  }

  // Offset the drawing area so axes/frame labels fit in the margins
  let has_margin = margin_left > 0.0 || margin_bottom > 0.0 || margin_top > 0.0;
  if has_margin {
    svg.push_str(&format!(
      "<g transform=\"translate({margin_left:.0},{margin_top:.0})\">\n"
    ));
  }

  // Render error indicator (red background + border + message) if primitives had invalid args
  if !errors.is_empty() {
    svg.push_str(&format!(
      "<rect width=\"{svg_width}\" height=\"{svg_height}\" fill=\"rgb(100%,33%,33%)\" fill-opacity=\"0.08\"/>\n"
    ));
    svg.push_str(&format!(
      "<rect x=\"0.6\" y=\"0.6\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"rgb(100%,33%,33%)\" stroke-width=\"1.2\"/>\n",
      svg_width as f64 - 1.2,
      svg_height as f64 - 1.2
    ));
    let title_text = errors
      .iter()
      .map(|m| svg_escape(m))
      .collect::<Vec<_>>()
      .join("\n");
    svg.push_str(&format!(
      "<rect width=\"{svg_width}\" height=\"{svg_height}\" fill=\"transparent\" stroke=\"none\"><title>{title_text}</title></rect>\n"
    ));
  }

  // Grid lines render behind the axes and primitives.
  if grid_x.is_active() || grid_y.is_active() {
    let default_style = grid_style.clone().unwrap_or_else(|| StyleState {
      color: Color::gray(0.8),
      ..StyleState::default()
    });
    render_grid_lines(
      &mut svg,
      &bb,
      svg_w,
      svg_h,
      &grid_x,
      &grid_y,
      &default_style,
    );
  }

  render_axes(
    &mut svg,
    axes,
    &bb,
    svg_w,
    svg_h,
    axes_label.as_ref(),
    (&ticks_x, &ticks_y),
    (margin_left, margin_right),
  );

  // Render primitives. A primitive with a drop shadow is wrapped in a
  // <g> that applies the shadow filter, so each primitive casts its own
  // shadow (overlapping shadows stack, giving the depth effect).
  for prim in &primitives {
    let shadow = prim.style().and_then(|s| s.drop_shadow.as_ref());
    if let Some(ds) = shadow {
      svg.push_str(&format!("<g filter=\"url(#{})\">\n", ds.filter_id()));
    }
    render_primitive(prim, &bb, svg_w, svg_h, &mut svg);
    if shadow.is_some() {
      svg.push_str("</g>\n");
    }
  }

  if frame {
    render_frame(&mut svg, &bb, svg_w, svg_h, frame_ticks);
  }

  // Frame captions: the bottom/top ones centred outside their edge, the
  // left/right ones rotated along theirs.
  if let Some((bottom, left, top, right)) = &frame_label {
    if !bottom.is_empty() {
      svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{}\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\">{bottom}</text>\n",
        svg_w / 2.0,
        svg_h + margin_bottom - 6.0,
        theme().tick_label_fill,
      ));
    }
    if !left.is_empty() {
      let lx = -(margin_left - 12.0);
      let ly = svg_h / 2.0;
      svg.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{ly:.1}\" fill=\"{}\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(-90,{lx:.1},{ly:.1})\">{left}</text>\n",
        theme().tick_label_fill,
      ));
    }
    if !top.is_empty() {
      svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{}\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\">{top}</text>\n",
        svg_w / 2.0,
        -(margin_top - 12.0),
        theme().tick_label_fill,
      ));
    }
    if !right.is_empty() {
      let rx = svg_w + margin_right - 12.0;
      let ry = svg_h / 2.0;
      svg.push_str(&format!(
        "<text x=\"{rx:.1}\" y=\"{ry:.1}\" fill=\"{}\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(90,{rx:.1},{ry:.1})\">{right}</text>\n",
        theme().tick_label_fill,
      ));
    }
  }

  if has_margin {
    svg.push_str("</g>\n");
  }

  svg.push_str("</svg>");

  // Generate and store GraphicsBox expression for .nb export
  let box_elements = primitives_to_box_elements(&primitives);
  let graphicsbox = gbox::graphics_box(&box_elements);
  crate::capture_graphicsbox(&graphicsbox);

  // Keep the symbolic `Graphics[prims, opts…]` alongside the rendering.
  // In Wolfram a `Graphics` expression stays an expression, so a picture
  // held in a variable can be layered by a later `Show` — without this the
  // primitives are gone and `Show[…, g, …]` silently drops `g`.
  let structure = Expr::FunctionCall {
    name: "Graphics".to_string(),
    args: std::iter::once(content)
      .chain(args[1..].iter().cloned())
      .collect(),
  };
  Ok(crate::graphics_result_with_structure(svg, structure))
}

// ── Grid SVG rendering ──────────────────────────────────────────────────

/// Extract the base and exponent from a Power expression (either BinaryOp or FunctionCall form).
/// Public accessor for `as_power` — used by `expr_to_box_form` for unit handling.
pub fn as_power(expr: &Expr) -> Option<(&Expr, &Expr)> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => Some((left.as_ref(), right.as_ref())),
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

/// Check if expression is an additive form (Plus/Minus) for parenthesization.
fn is_additive_expr(e: &Expr) -> bool {
  matches!(
    e,
    Expr::BinaryOp {
      op: BinaryOperator::Plus | BinaryOperator::Minus,
      ..
    }
  ) || matches!(e, Expr::FunctionCall { name, .. } if name == "Plus")
}

/// Determine the separator between two adjacent factors in Times SVG rendering.
/// Returns `""` (no separator) or `" "` (space) — never `"*"`.
fn times_svg_separator(_left: &Expr, right: &Expr) -> &'static str {
  // Right side is additive → will be wrapped in parens → no separator (e.g. 9(x + y))
  if is_additive_expr(right) {
    return "";
  }
  // Right is Power with additive base → rendered starting with "(" → no separator
  if let Some((base, _)) = as_power(right)
    && is_additive_expr(base)
  {
    return "";
  }
  // Default: space (implicit multiplication)
  " "
}

/// Render a stacked fraction (numerator over denominator) as SVG tspan markup.
/// Uses `<tspan>` elements with `dy`/`dx` positioning in `ch` units so that
/// the layout adapts to the actual monospace character width of the browser,
/// avoiding compounding drift from hard-coded pixel offsets.
/// Legacy stacked fraction for the old `expr_to_svg_markup` / `boxes_to_svg`
/// text-based paths (used by Grid cell rendering). Renders as "num/den" inline.
fn stacked_fraction_svg(
  num_markup: &str,
  den_markup: &str,
  _num_w: f64,
  _den_w: f64,
) -> String {
  format!("{num_markup}/{den_markup}")
}

/// A rendered box layout node. Each node carries its pixel dimensions,
/// the vertical offset of the baseline from the top, and the SVG elements
/// needed to draw it (positioned relative to (0, 0) of the node).
#[derive(Clone)]
pub struct BoxLayout {
  pub width: f64,
  pub height: f64,
  /// Distance from top of the box to the text baseline.
  pub baseline: f64,
  /// SVG elements as a string, positioned relative to (0, baseline).
  /// Can contain `<text>`, `<line>`, nested `<g>`, etc.
  pub elements: String,
}

/// Character advance of the monospace font used to display typeset text in the
/// visual hosts, as a fraction of the font size. The Playground/Studio map the
/// SVG `<text font-family="monospace">` we emit onto Atkinson Hyperlegible Mono
/// (see the host CSS), whose glyphs advance 632/1000 em. The per-atom width
/// estimate must match this, or successive atoms (e.g. a function name and its
/// opening `[`) overlap.
pub(crate) const MONO_ADVANCE: f64 = 0.632;

/// If `s` is a single n-ary/large operator glyph (∑ ∏ ∫ …), return the font
/// scale factor at which it should be drawn so it reads as a display-size
/// operator. Returns `None` for ordinary atoms.
fn large_operator_scale(s: &str) -> Option<f64> {
  match s {
    "\u{2211}" | "\u{220F}" | "\u{2210}" => Some(1.9), // ∑ ∏ ∐
    "\u{22C3}" | "\u{22C2}" | "\u{2A01}" | "\u{2A02}" | "\u{2A00}" => Some(1.8), // ⋃ ⋂ ⨁ ⨂ ⨀
    "\u{222B}" | "\u{222C}" | "\u{222D}" | "\u{222E}" => Some(1.8), // ∫ ∬ ∭ ∮
    _ => None,
  }
}

/// True if `s` is a single Latin letter, which TraditionalForm renders as an
/// italic math variable.
fn is_math_italic_atom(s: &str) -> bool {
  let mut chars = s.chars();
  match (chars.next(), chars.next()) {
    (Some(c), None) => c.is_ascii_alphabetic(),
    _ => false,
  }
}

/// The set of single-character bracket/bar glyphs that can be vertically
/// stretched to enclose tall content.
fn stretchy_delim_kind(s: &str) -> Option<char> {
  match s {
    "(" | ")" | "[" | "]" | "|" | "{" | "}" => s.chars().next(),
    _ => None,
  }
}

/// Whether `open`/`close` form a matching stretchy-delimiter pair. Braces
/// are intentionally excluded: lists keep ordinary `{`/`}` glyphs.
fn delim_pair_matches(open: char, close: char) -> bool {
  matches!((open, close), ('(', ')') | ('[', ']') | ('|', '|'))
}

/// Horizontal space (in character advances) placed on each side of a binary
/// operator or relation token; `0.0` for non-operators.
fn operator_space(s: &str) -> f64 {
  match s {
    "=" | "\u{2260}" | "<" | ">" | "\u{2264}" | "\u{2265}" | "\u{2261}"
    | "\u{2262}" | "\u{2192}" | "\u{29F4}" | "\u{2248}" | "\u{221D}"
    | "\u{21D2}" | "\u{27F9}" => 0.44,
    "+" | "-" | "\u{00B1}" | "\u{2213}" => 0.36,
    "\u{2227}" | "\u{2228}" | "\u{22C5}" => 0.34,
    _ => 0.0,
  }
}

/// Draw a bracket/bar/paren glyph as a vector path stretched vertically to
/// enclose content of the given ascent/descent. The delimiter's baseline is
/// aligned to the inner content's baseline so the surrounding row stays on the
/// math axis.
fn render_stretchy_delim(
  kind: char,
  inner_ascent: f64,
  inner_descent: f64,
  font_size: f64,
) -> BoxLayout {
  let pad = font_size * 0.14;
  let h = inner_ascent + inner_descent + pad * 2.0;
  let baseline = inner_ascent + pad;
  let sw = (font_size * 0.055).max(0.7);
  // Width reserved for the delimiter, plus a hair of side bearing.
  let (body_w, bearing) = match kind {
    '|' => (font_size * 0.10, font_size * 0.10),
    '[' | ']' => (font_size * 0.22, font_size * 0.08),
    '{' | '}' => (font_size * 0.28, font_size * 0.08),
    _ => (font_size * 0.30, font_size * 0.08), // ( )
  };
  let w = body_w + bearing * 2.0;
  let x0 = bearing;
  let x1 = bearing + body_w;
  let path = match kind {
    '|' => {
      let cx = f64::midpoint(x0, x1);
      format!(
        "<line x1=\"{cx:.2}\" y1=\"0\" x2=\"{cx:.2}\" y2=\"{h:.2}\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>"
      )
    }
    '[' => format!(
      "<path d=\"M {x1:.2} 0 L {x0:.2} 0 L {x0:.2} {h:.2} L {x1:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>"
    ),
    ']' => format!(
      "<path d=\"M {x0:.2} 0 L {x1:.2} 0 L {x1:.2} {h:.2} L {x0:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>"
    ),
    '{' => {
      let midy = h / 2.0;
      let xm = f64::midpoint(x0, x1);
      format!(
        "<path d=\"M {x1:.2} 0 Q {x0:.2} 0 {x0:.2} {q:.2} L {x0:.2} {a:.2} Q {x0:.2} {midy:.2} {xm:.2} {midy:.2} Q {x0:.2} {midy:.2} {x0:.2} {b:.2} L {x0:.2} {c:.2} Q {x0:.2} {h:.2} {x1:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>",
        q = h * 0.12,
        a = midy - h * 0.06,
        b = midy + h * 0.06,
        c = h * 0.88,
      )
    }
    '}' => {
      let midy = h / 2.0;
      let xm = f64::midpoint(x0, x1);
      format!(
        "<path d=\"M {x0:.2} 0 Q {x1:.2} 0 {x1:.2} {q:.2} L {x1:.2} {a:.2} Q {x1:.2} {midy:.2} {xm:.2} {midy:.2} Q {x1:.2} {midy:.2} {x1:.2} {b:.2} L {x1:.2} {c:.2} Q {x1:.2} {h:.2} {x0:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>",
        q = h * 0.12,
        a = midy - h * 0.06,
        b = midy + h * 0.06,
        c = h * 0.88,
      )
    }
    '(' => {
      let cx = x1 + body_w * 0.15;
      format!(
        "<path d=\"M {cx:.2} 0 Q {x0:.2} {midy:.2} {cx:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>",
        midy = h / 2.0,
      )
    }
    ')' => {
      let cx = x0 - body_w * 0.15;
      format!(
        "<path d=\"M {cx:.2} 0 Q {x1:.2} {midy:.2} {cx:.2} {h:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\"/>",
        midy = h / 2.0,
      )
    }
    _ => String::new(),
  };
  BoxLayout {
    width: w,
    height: h,
    baseline,
    elements: path,
  }
}

/// Lay out a square-root: the radical hook and the vinculum over `content`
/// are emitted as one connected polyline that scales to the content height,
/// so the sign and overbar read as a single object. `left_offset` reserves
/// space to the left of the hook (used by RadicalBox for its index). Returns
/// the hook width and the composed layout.
fn sqrt_radical(
  content: &BoxLayout,
  left_offset: f64,
  font_size: f64,
) -> (f64, BoxLayout) {
  let ch = content.height;
  let sw = (font_size * 0.06).max(0.9);
  let gap_top = font_size * 0.18;
  let hook_w = font_size * 0.55;
  let line_y = sw; // keep the vinculum fully inside the viewport
  let content_x = left_offset + hook_w;
  let content_top = line_y + gap_top;
  let bottom = content_top + ch;
  let h = bottom + sw;
  let w = content_x + content.width + font_size * 0.10;

  // Radical polyline: enter mid-left, dip to the bottom vertex, rise along
  // the long diagonal to the top-left corner, then run across as the vinculum.
  let x0 = left_offset;
  let p0 = (x0 + hook_w * 0.04, content_top + ch * 0.55);
  let p1 = (x0 + hook_w * 0.30, content_top + ch * 0.72);
  let p2 = (x0 + hook_w * 0.52, h - sw * 0.5);
  let p3 = (content_x, line_y);
  let p4 = (w, line_y);
  let path = format!(
    "<path d=\"M {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{sw:.2}\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>",
    p0.0, p0.1, p1.0, p1.1, p2.0, p2.1, p3.0, p3.1, p4.0, p4.1,
  );
  let elements =
    format!("{}{}", path, content.translate(content_x, content_top));
  (
    hook_w,
    BoxLayout {
      width: w,
      height: h,
      baseline: content_top + content.baseline,
      elements,
    },
  )
}

impl BoxLayout {
  /// Create a layout for a simple text atom.
  fn text(s: &str, font_size: f64) -> Self {
    let ch = font_size * MONO_ADVANCE; // monospace char advance
    let ascent = font_size * 0.8; // approximate ascent
    let descent = font_size * 0.25; // approximate descent
    let height = ascent + descent;
    // Large (n-ary) operators — ∑ ∏ ∫ … — are drawn oversized and vertically
    // centered on the math axis so they read as display-size operators with
    // limits stacked above/below (Sum, Product) or as scripts (Integrate),
    // matching conventional math typesetting.
    if let Some(scale) = large_operator_scale(s) {
      let fs = font_size * scale;
      let glyph_ascent = fs * 0.72;
      let glyph_descent = fs * 0.28;
      let w = fs * 0.62;
      let escaped = svg_escape(s);
      return Self {
        width: w,
        height: glyph_ascent + glyph_descent,
        baseline: glyph_ascent,
        elements: format!(
          "<text x=\"{cx:.2}\" y=\"{glyph_ascent:.2}\" font-family=\"serif\" font-size=\"{fs:.2}\" stroke=\"none\" text-anchor=\"middle\">{escaped}</text>",
          cx = w / 2.0,
        ),
      };
    }
    // The multiplication separator " × " reserves a full monospace space on
    // each side of the small sign, which reads as too wide. Render it in a
    // tighter slot (~half a space per side) with the sign centered, without
    // changing the underlying box string (which must stay " × " to match
    // wolframscript's MakeBoxes/ToString output).
    if s == " \u{00d7} " {
      let w = ch * 1.8;
      // Nudge the sign slightly right of the slot's geometric center: the ×
      // glyph sits a touch left within its advance box, so a perfectly
      // centered anchor leaves the right gap looking larger than the left.
      let cx = w / 2.0 + ch * 0.1;
      let escaped = svg_escape("\u{00d7}");
      return Self {
        width: w,
        height,
        baseline: ascent,
        elements: format!(
          "<text x=\"{cx:.2}\" y=\"{ascent:.1}\" font-family=\"monospace\" font-size=\"{font_size:.1}\" stroke=\"none\" text-anchor=\"middle\">{escaped}</text>"
        ),
      };
    }
    // Digit-group separator (thin space U+2009, inserted by `group_digits_str`):
    // render each 3-digit group as its own atom separated by a narrow gap, so
    // the spacing is thinner than a full monospace advance and matches the
    // Wolfram notebook's grouping (a mono thin-space glyph advances a full em,
    // which would look too wide).
    if s.contains('\u{2009}') {
      let ascent = font_size * 0.8;
      let descent = font_size * 0.25;
      let gap = ch * 0.32;
      let mut x = 0.0_f64;
      let mut elements = String::new();
      for (i, seg) in s.split('\u{2009}').enumerate() {
        if i > 0 {
          x += gap;
        }
        let escaped = svg_escape(seg);
        elements.push_str(&format!(
          "<text x=\"{x:.2}\" y=\"{ascent:.1}\" font-family=\"monospace\" font-size=\"{font_size:.1}\" stroke=\"none\" xml:space=\"preserve\">{escaped}</text>"
        ));
        x += seg.chars().count() as f64 * ch;
      }
      return Self {
        width: x,
        height: ascent + descent,
        baseline: ascent,
        elements,
      };
    }
    // Map Wolfram private-use operator glyphs to standard Unicode. The box
    // form emits `\[Rule]` / `\[RuleDelayed]` as their FrontEnd private-use
    // codepoints (U+F522 / U+F51F), which only have glyphs in Mathematica's
    // bundled fonts — in a normal monospace font they render as a
    // missing-glyph box (▢). Substitute the public Unicode arrows so the
    // SVG output displays correctly everywhere. Each maps one char to one
    // char, so the width estimate below is unaffected.
    let mapped: String = s
      .chars()
      .map(|c| match c {
        '\u{f522}' => '\u{2192}', // \[Rule] → →
        '\u{f51f}' => '\u{29f4}', // \[RuleDelayed] → ⧴
        other => other,
      })
      .collect();
    let s = mapped.as_str();
    let w = s.chars().count() as f64 * ch;
    let escaped = svg_escape(s);
    // Single Latin letters are math variables — render them italic for a
    // conventional TeX-like look. The advance width is unchanged (same font,
    // slanted), so surrounding layout is unaffected.
    let style_attr = if is_math_italic_atom(s) {
      " font-style=\"italic\""
    } else {
      ""
    };
    Self {
      width: w,
      height,
      baseline: ascent,
      // `xml:space="preserve"` keeps leading/trailing spaces from collapsing,
      // so separator atoms like " × " (space, sign, space) stay centered in
      // their allocated width rather than the glyph sticking to the previous
      // token. The width above already counts those spaces.
      elements: format!(
        "<text x=\"0\" y=\"{ascent:.1}\" font-family=\"monospace\" font-size=\"{font_size:.1}\" stroke=\"none\"{style_attr} xml:space=\"preserve\">{escaped}</text>"
      ),
    }
  }

  /// Translate this layout by (dx, dy) by wrapping in a `<g transform>`.
  fn translate(&self, dx: f64, dy: f64) -> String {
    format!(
      "<g transform=\"translate({dx:.1},{dy:.1})\">{}</g>",
      self.elements
    )
  }
}

/// Recursively lay out a box expression into a `BoxLayout`.
/// This is the main bottom-up tree renderer for the box language.
pub fn layout_box(expr: &Expr, font_size: f64) -> BoxLayout {
  let ch = font_size * MONO_ADVANCE;

  match expr {
    Expr::String(s) => BoxLayout::text(strip_precision_marker(s), font_size),
    Expr::Identifier(s) => {
      BoxLayout::text(strip_precision_marker(s), font_size)
    }
    Expr::Integer(n) => {
      BoxLayout::text(&group_digits_str(&n.to_string()), font_size)
    }
    Expr::BigInteger(n) => {
      BoxLayout::text(&group_digits_str(&n.to_string()), font_size)
    }

    Expr::FunctionCall { name, args } => match name.as_str() {
      // RowBox: lay out children left-to-right, align baselines
      "RowBox" if args.len() == 1 => {
        let items = match &args[0] {
          Expr::List(items) => items.as_slice(),
          other => return layout_box(other, font_size),
        };
        if items.is_empty() {
          return BoxLayout::text("", font_size);
        }
        let mut children: Vec<BoxLayout> =
          items.iter().map(|e| layout_box(e, font_size)).collect();

        // Stretchy delimiters: when the row is bracketed — its first child is
        // an opening (or bar) delimiter and its last child is the matching
        // closing (or bar) delimiter — grow both to enclose the inner content
        // (matrices, determinants, tall fractions). Function-call parens are
        // unaffected because the head token precedes the `(`.
        if items.len() >= 3 {
          let open = match &items[0] {
            Expr::String(s) => stretchy_delim_kind(s),
            _ => None,
          };
          let close = match items.last() {
            Some(Expr::String(s)) => stretchy_delim_kind(s),
            _ => None,
          };
          if let (Some(o), Some(c)) = (open, close)
            && delim_pair_matches(o, c)
          {
            let inner_ascent = children[1..children.len() - 1]
              .iter()
              .map(|c| c.baseline)
              .fold(0.0_f64, f64::max);
            let inner_descent = children[1..children.len() - 1]
              .iter()
              .map(|c| c.height - c.baseline)
              .fold(0.0_f64, f64::max);
            let natural = children[0].height;
            // Only stretch for genuinely tall content (fractions, grids,
            // nested radicals) — a lone superscript (~1.3×) stays with plain
            // glyphs so ordinary parenthesized expressions are unaffected.
            if inner_ascent + inner_descent > natural * 1.5 {
              let last = children.len() - 1;
              children[0] = render_stretchy_delim(
                o,
                inner_ascent,
                inner_descent,
                font_size,
              );
              children[last] = render_stretchy_delim(
                c,
                inner_ascent,
                inner_descent,
                font_size,
              );
            }
          }
        }

        // Find the maximum baseline and maximum below-baseline
        let max_baseline =
          children.iter().map(|c| c.baseline).fold(0.0_f64, f64::max);
        let max_below = children
          .iter()
          .map(|c| c.height - c.baseline)
          .fold(0.0_f64, f64::max);

        let mut elements = String::new();
        let mut x = 0.0_f64;
        for (i, child) in children.iter().enumerate() {
          // Medium space around binary operators / relations (but not around a
          // leading unary sign, which has no left operand).
          let op_space = match &items[i] {
            Expr::String(s) if i > 0 && i + 1 < items.len() => {
              operator_space(s) * ch
            }
            _ => 0.0,
          };
          // A small space before a bare comma so it does not crowd the
          // preceding token (e.g. the trailing digit of a list element).
          if i > 0 && matches!(&items[i], Expr::String(s) if s == ",") {
            x += ch * 0.2;
          }
          x += op_space;
          let dy = max_baseline - child.baseline;
          elements.push_str(&child.translate(x, dy));
          x += child.width;
          x += op_space;
          // Add space after bare comma
          if matches!(&items[i], Expr::String(s) if s == ",")
            && i + 1 < items.len()
          {
            x += ch * 0.4;
          }
        }
        BoxLayout {
          width: x,
          height: max_baseline + max_below,
          baseline: max_baseline,
          elements,
        }
      }

      // FractionBox: numerator above line above denominator
      "FractionBox" if args.len() == 2 => {
        let num = layout_box(&args[0], font_size * 0.75);
        let den = layout_box(&args[1], font_size * 0.75);
        let frac_w = num.width.max(den.width) + 4.0;
        let gap = 3.0;
        let line_thickness = 0.8;

        // Numerator centered above line
        let num_x = (frac_w - num.width) / 2.0;
        let num_y = 0.0;
        // Line below numerator
        let line_y = num.height + gap;
        // Denominator centered below line
        let den_x = (frac_w - den.width) / 2.0;
        let den_y = line_y + line_thickness + gap;

        let total_h = den_y + den.height;
        // Baseline of the fraction = at the line (so it aligns with surrounding text)
        let baseline = line_y;

        let elements = format!(
          "{}\
           <line x1=\"0\" y1=\"{line_y:.1}\" x2=\"{frac_w:.1}\" y2=\"{line_y:.1}\" stroke=\"currentColor\" stroke-width=\"{line_thickness}\"/>\
           {}",
          num.translate(num_x, num_y),
          den.translate(den_x, den_y),
        );
        BoxLayout {
          width: frac_w,
          height: total_h,
          baseline,
          elements,
        }
      }

      // SuperscriptBox: base with raised exponent
      "SuperscriptBox" if args.len() == 2 => {
        let base = layout_box(&args[0], font_size);
        let sup = layout_box(&args[1], font_size * 0.7);
        // Superscript top aligns with top of base
        let sup_y = 0.0;
        let base_y = sup.height * 0.4; // base shifted down so sup overlaps top
        let elements = format!(
          "{}{}",
          base.translate(0.0, base_y),
          sup.translate(base.width, sup_y),
        );
        BoxLayout {
          width: base.width + sup.width,
          height: (base_y + base.height).max(sup.height),
          baseline: base_y + base.baseline,
          elements,
        }
      }

      // SubscriptBox: base with lowered subscript
      "SubscriptBox" if args.len() == 2 => {
        let base = layout_box(&args[0], font_size);
        let sub = layout_box(&args[1], font_size * 0.7);
        let sub_y = base.height * 0.4;
        let elements = format!(
          "{}{}",
          base.translate(0.0, 0.0),
          sub.translate(base.width, sub_y),
        );
        BoxLayout {
          width: base.width + sub.width,
          height: (sub_y + sub.height).max(base.height),
          baseline: base.baseline,
          elements,
        }
      }

      // SubsuperscriptBox: base with both
      "SubsuperscriptBox" if args.len() == 3 => {
        let base = layout_box(&args[0], font_size);
        let sub = layout_box(&args[1], font_size * 0.7);
        let sup = layout_box(&args[2], font_size * 0.7);
        // A large-operator base (∫) carries its limits like a display
        // integral: the upper bound at the top-right tip and the lower bound
        // at the bottom, nudged left to sit under the slanted stem (the sign's
        // lower hook is on the left). Other bases use ordinary tight
        // sub/superscripts to the right.
        let large_op = matches!(&args[0],
          Expr::String(s) if large_operator_scale(s).is_some());
        if large_op {
          // Set the limits beside the sign, clear of its ink: the upper bound
          // at the top-right tip and the lower bound at the bottom, a little
          // left of the upper (following the sign's slant). Both start at (or
          // past) the sign's right edge so neither overlaps the stroke.
          let gap = font_size * 0.08;
          let sup_x = base.width + gap;
          let sup_y = 0.0;
          let sub_x = base.width * 0.72 + gap;
          let sub_y = (base.height - sub.height).max(base.baseline);
          let width =
            base.width.max(sup_x + sup.width).max(sub_x + sub.width) + gap;
          let elements = format!(
            "{}{}{}",
            base.translate(0.0, 0.0),
            sup.translate(sup_x, sup_y),
            sub.translate(sub_x, sub_y),
          );
          return BoxLayout {
            width,
            height: (sub_y + sub.height).max(base.height),
            baseline: base.baseline,
            elements,
          };
        }
        let sup_y = 0.0;
        let base_y = sup.height * 0.4;
        let sub_y = base_y + base.height * 0.4;
        let script_x = base.width;
        let script_w = sub.width.max(sup.width);
        let elements = format!(
          "{}{}{}",
          base.translate(0.0, base_y),
          sup.translate(script_x, sup_y),
          sub.translate(script_x, sub_y),
        );
        BoxLayout {
          width: base.width + script_w,
          height: (sub_y + sub.height).max(base_y + base.height),
          baseline: base_y + base.baseline,
          elements,
        }
      }

      // SqrtBox: the radical hook and its vinculum (overline) are drawn as a
      // single connected stroke that scales to the content height, so the
      // sign and bar read as one object instead of a fixed glyph butted
      // against a separate line.
      "SqrtBox" if args.len() == 1 => {
        let content = layout_box(&args[0], font_size);
        sqrt_radical(&content, 0.0, font_size).1
      }

      // RadicalBox: like SqrtBox but with a small index tucked into the hook.
      "RadicalBox" if args.len() == 2 => {
        let content = layout_box(&args[0], font_size);
        let index = layout_box(&args[1], font_size * 0.6);
        let index_w = index.width + font_size * 0.05;
        let (hook_w, body) = sqrt_radical(&content, index_w, font_size);
        // Place the index over the low part of the hook.
        let index_x = (index_w + hook_w - index.width).max(0.0) * 0.4;
        let index_y = body.height * 0.18;
        let elements =
          format!("{}{}", index.translate(index_x, index_y), body.elements);
        BoxLayout { elements, ..body }
      }

      // OverscriptBox / UnderscriptBox / UnderoverscriptBox — same as super/sub for now
      "OverscriptBox" if args.len() >= 2 => {
        let base = layout_box(&args[0], font_size);
        let over = layout_box(&args[1], font_size * 0.7);
        let base_y = over.height;
        let elements = format!(
          "{}{}",
          over.translate((base.width - over.width) / 2.0, 0.0),
          base.translate(0.0, base_y)
        );
        BoxLayout {
          width: base.width.max(over.width),
          height: base_y + base.height,
          baseline: base_y + base.baseline,
          elements,
        }
      }
      "UnderscriptBox" if args.len() >= 2 => {
        let base = layout_box(&args[0], font_size);
        let under = layout_box(&args[1], font_size * 0.7);
        let under_y = base.height;
        let elements = format!(
          "{}{}",
          base.translate(0.0, 0.0),
          under.translate((base.width - under.width) / 2.0, under_y)
        );
        BoxLayout {
          width: base.width.max(under.width),
          height: under_y + under.height,
          baseline: base.baseline,
          elements,
        }
      }
      "UnderoverscriptBox" if args.len() >= 3 => {
        let base = layout_box(&args[0], font_size);
        let under = layout_box(&args[1], font_size * 0.7);
        let over = layout_box(&args[2], font_size * 0.7);
        let base_y = over.height;
        let under_y = base_y + base.height;
        let w = base.width.max(under.width).max(over.width);
        let elements = format!(
          "{}{}{}",
          over.translate((w - over.width) / 2.0, 0.0),
          base.translate((w - base.width) / 2.0, base_y),
          under.translate((w - under.width) / 2.0, under_y),
        );
        BoxLayout {
          width: w,
          height: under_y + under.height,
          baseline: base_y + base.baseline,
          elements,
        }
      }

      // FrameBox
      "FrameBox" if !args.is_empty() => {
        let content = layout_box(&args[0], font_size);
        let pad = 4.0;
        let stroke_w = 0.5;
        let margin = stroke_w; // keep border fully inside the SVG viewport
        let inner_w = content.width + pad * 2.0;
        let inner_h = content.height + pad * 2.0;
        let w = inner_w + margin * 2.0;
        let h = inner_h + margin * 2.0;
        let elements = format!(
          "<rect x=\"{margin:.1}\" y=\"{margin:.1}\" width=\"{inner_w:.1}\" height=\"{inner_h:.1}\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"{stroke_w}\"/>\
           {}",
          content.translate(margin + pad, margin + pad),
        );
        BoxLayout {
          width: w,
          height: h,
          baseline: margin + pad + content.baseline,
          elements,
        }
      }

      // TagBox, FormBox, InterpretationBox — delegate to content.
      // TagBox[boxes, tag, opts...] may carry trailing options such as
      // `Editable -> True` (e.g. from TraditionalForm), so accept 2+ args.
      "TagBox" if args.len() >= 2 => layout_box(&args[0], font_size),
      // FormBox[boxes, TraditionalForm] — the form marker is display-only.
      "FormBox" if !args.is_empty() => layout_box(&args[0], font_size),
      // InterpretationBox[boxes, expr, opts...] — render the boxes; the
      // interpretation expression and any trailing options (e.g.
      // `AutoDelete -> True`) are display pass-throughs.
      "InterpretationBox" if args.len() >= 2 => layout_box(&args[0], font_size),

      // StyleBox — apply FontSize, FontColor, and Background from options
      "StyleBox" if !args.is_empty() => {
        let mut effective_font_size = font_size;
        let mut font_color: Option<Color> = None;
        let mut background: Option<Color> = None;
        // Scan style options (Rule expressions) in args[1..]
        for opt in &args[1..] {
          let (key, val) = match opt {
            Expr::Rule {
              pattern,
              replacement,
            } => (pattern.as_ref(), replacement.as_ref()),
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rule" && ra.len() == 2 =>
            {
              (&ra[0], &ra[1])
            }
            _ => continue,
          };
          if let Expr::Identifier(k) = key {
            match k.as_str() {
              "FontSize" => {
                if let Some(sz) = expr_to_f64(val) {
                  effective_font_size = sz;
                }
              }
              "FontColor" => {
                font_color = parse_color(val);
              }
              "Background" => {
                background = parse_color(val);
              }
              _ => {}
            }
          }
        }
        let content = layout_box(&args[0], effective_font_size);
        let mut elements = String::new();
        // Background rectangle behind content
        if let Some(bg) = background {
          elements.push_str(&format!(
            "<rect x=\"0\" y=\"0\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"{}/>",
            content.width, content.height, bg.to_svg_rgb(), bg.opacity_attr(),
          ));
        }
        if let Some(color) = font_color {
          elements.push_str(&format!(
            "<g fill=\"{}\"{}>{}</g>",
            color.to_svg_rgb(),
            color.opacity_attr(),
            content.elements,
          ));
        } else {
          elements.push_str(&content.elements);
        }
        BoxLayout {
          elements,
          ..content
        }
      }

      // GridBox
      "GridBox" if !args.is_empty() => {
        if let Expr::List(rows) = &args[0] {
          let gap_x = ch;
          let gap_y = font_size * 0.4;
          let laid_out: Vec<Vec<BoxLayout>> = rows
            .iter()
            .map(|row| {
              if let Expr::List(cells) = row {
                cells.iter().map(|c| layout_box(c, font_size)).collect()
              } else {
                vec![layout_box(row, font_size)]
              }
            })
            .collect();

          let n_cols =
            laid_out.iter().map(std::vec::Vec::len).max().unwrap_or(0);
          // Column widths
          let col_widths: Vec<f64> = (0..n_cols)
            .map(|c| {
              laid_out
                .iter()
                .filter_map(|r| r.get(c))
                .map(|l| l.width)
                .fold(0.0_f64, f64::max)
            })
            .collect();
          // Row heights and baselines
          let row_metrics: Vec<(f64, f64)> = laid_out
            .iter()
            .map(|r| {
              let bl = r.iter().map(|c| c.baseline).fold(0.0_f64, f64::max);
              let below = r
                .iter()
                .map(|c| c.height - c.baseline)
                .fold(0.0_f64, f64::max);
              (bl, bl + below)
            })
            .collect();

          let total_w: f64 =
            col_widths.iter().sum::<f64>() + gap_x * (n_cols.max(1) - 1) as f64;
          let total_h: f64 = row_metrics.iter().map(|(_, h)| h).sum::<f64>()
            + gap_y * (row_metrics.len().max(1) - 1) as f64;

          let mut elements = String::new();
          let mut y = 0.0;
          for (ri, row) in laid_out.iter().enumerate() {
            let (row_bl, row_h) = row_metrics[ri];
            let mut x = 0.0;
            for (ci, cell) in row.iter().enumerate() {
              let dy = row_bl - cell.baseline;
              elements.push_str(&cell.translate(x, y + dy));
              x += col_widths.get(ci).unwrap_or(&0.0) + gap_x;
            }
            y += row_h + gap_y;
          }
          let first_bl = row_metrics.first().map_or(font_size, |(bl, _)| *bl);
          BoxLayout {
            width: total_w,
            height: total_h,
            baseline: first_bl,
            elements,
          }
        } else {
          layout_box(&args[0], font_size)
        }
      }

      // TemplateBox[{label, uri}, "HyperlinkURL"] — clickable hyperlink.
      // The label is unwrapped from its surrounding `"…"` if it boxed
      // from a literal string (matching wolframscript's MakeBoxes,
      // which bakes the quotes into the box content). Any other
      // template falls through to the text fallback.
      "TemplateBox"
        if args.len() == 2
          && matches!(&args[1], Expr::String(t) if t == "HyperlinkURL") =>
      {
        if let Expr::List(items) = &args[0]
          && items.len() == 2
          && let Expr::String(uri) = &items[1]
        {
          // Strip surrounding quotes from a string label (the box form
          // of `"Woxi"` is the literal text `"Woxi"` — show `Woxi`).
          let label_box = match &items[0] {
            Expr::String(s)
              if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') =>
            {
              Expr::String(s[1..s.len() - 1].to_string())
            }
            other => other.clone(),
          };
          let label = layout_box(&label_box, font_size);
          let underline_y = label.baseline + font_size * 0.12;
          let stroke_w = (font_size * 0.05).max(0.6);
          let elements = format!(
            "<a href=\"{href}\" target=\"_blank\" rel=\"noopener\">\
             <g fill=\"#1a73e8\" stroke=\"none\">{inner}</g>\
             <line x1=\"0\" y1=\"{uy:.1}\" x2=\"{w:.1}\" y2=\"{uy:.1}\" stroke=\"#1a73e8\" stroke-width=\"{sw:.2}\"/>\
             </a>",
            href = svg_escape(uri),
            inner = label.elements,
            uy = underline_y,
            w = label.width,
            sw = stroke_w,
          );
          return BoxLayout {
            width: label.width,
            height: label.height.max(underline_y + stroke_w),
            baseline: label.baseline,
            elements,
          };
        }
        let text = crate::syntax::expr_to_output(expr);
        BoxLayout::text(&text, font_size)
      }

      // Unknown function: render as text
      _ => {
        let text = crate::syntax::expr_to_output(expr);
        BoxLayout::text(&text, font_size)
      }
    },

    Expr::List(items) => {
      // Concatenate like RowBox
      let children: Vec<BoxLayout> =
        items.iter().map(|e| layout_box(e, font_size)).collect();
      let max_bl = children.iter().map(|c| c.baseline).fold(0.0_f64, f64::max);
      let max_below = children
        .iter()
        .map(|c| c.height - c.baseline)
        .fold(0.0_f64, f64::max);
      let mut elements = String::new();
      let mut x = 0.0;
      for child in &children {
        let dy = max_bl - child.baseline;
        elements.push_str(&child.translate(x, dy));
        x += child.width;
      }
      BoxLayout {
        width: x,
        height: max_bl + max_below,
        baseline: max_bl,
        elements,
      }
    }

    _ => {
      let text = crate::syntax::expr_to_output(expr);
      BoxLayout::text(&text, font_size)
    }
  }
}

/// Group the integer part of a decimal number string into blocks of three
/// digits separated by a thin space (U+2009), matching the Wolfram notebook's
/// digit grouping (`10000000000` → `10 000 000 000`, `100000.` → `100 000.`).
/// Grouping only kicks in at five or more integer digits, so `1000` stays
/// `1000` and `10000` becomes `10 000`. The fractional part and any leading
/// sign are left untouched; a non-numeric string is returned unchanged.
///
/// `BoxLayout::text` renders the U+2009 separators as narrow gaps (thinner than
/// a full monospace advance) so the spacing reads like the notebook's.
pub(crate) fn group_digits_str(s: &str) -> String {
  // The box form usually splits a sign into its own token, but the
  // Integer/BigInteger layout arms pass a signed magnitude — handle both.
  let (sign, rest) = match s.strip_prefix('-') {
    Some(r) => ("-", r),
    None => ("", s),
  };
  let (int_part, frac) = match rest.find('.') {
    Some(i) => (&rest[..i], &rest[i..]),
    None => (rest, ""),
  };
  // Only a plain decimal number is grouped: all-digit integer part and an
  // optional `.` followed by digits.
  if int_part.is_empty()
    || !int_part.bytes().all(|b| b.is_ascii_digit())
    || !frac.bytes().all(|b| b == b'.' || b.is_ascii_digit())
    || int_part.len() < 5
  {
    return s.to_string();
  }
  let n = int_part.len();
  let first = match n % 3 {
    0 => 3,
    r => r,
  };
  let mut grouped = String::with_capacity(n + n / 3);
  grouped.push_str(&int_part[..first]);
  let mut i = first;
  while i < n {
    grouped.push('\u{2009}');
    grouped.push_str(&int_part[i..i + 3]);
    i += 3;
  }
  format!("{sign}{grouped}{frac}")
}

/// Render a `BoxLayout` into a complete SVG string.
pub fn layout_to_svg(layout: &BoxLayout, fill: &str) -> String {
  let width = layout.width.ceil().max(1.0) as usize;
  let height = layout.height.ceil().max(1.0) as usize;
  format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">\
     <g fill=\"{fill}\" stroke=\"{fill}\">{}</g>\
     </svg>",
    layout.elements,
  )
}

/// Estimate the display width of a stacked fraction in parent character units.
fn stacked_fraction_width(num_w: f64, den_w: f64) -> f64 {
  let frac_chars = num_w.max(den_w).ceil().max(1.0);
  frac_chars * 0.7 + 0.5
}

/// Check if an expression contains a Rational (stacked fraction),
/// which requires extra vertical space.
pub fn has_fraction(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      true
    }
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::FunctionCall { name: rn, args: ra }
          if rn == "Rational" && ra.len() == 2
            && matches!(&ra[0], Expr::Integer(1))
            && matches!(&ra[1], Expr::Integer(d) if *d > 0)) =>
    {
      true
    }
    // FullForm renders as plain text, no stacked fractions
    Expr::FunctionCall { name, args }
      if name == "FullForm" && args.len() == 1 =>
    {
      false
    }
    Expr::FunctionCall { args, .. } => args.iter().any(has_fraction),
    Expr::List(items) => items.iter().any(has_fraction),
    Expr::BinaryOp { left, right, .. } => {
      has_fraction(left) || has_fraction(right)
    }
    Expr::UnaryOp { operand, .. } => has_fraction(operand),
    Expr::Comparison { operands, .. } => operands.iter().any(has_fraction),
    Expr::Rule {
      pattern,
      replacement,
    } => has_fraction(pattern) || has_fraction(replacement),
    Expr::Association(pairs) => pairs
      .iter()
      .any(|(k, v)| has_fraction(k) || has_fraction(v)),
    _ => false,
  }
}

/// Convert a Quantity unit expression to its abbreviated SVG form.
/// E.g. `"Meters"/"Seconds"` → `m/s`, `"Meters"^2` → `m²` (with superscript).
fn quantity_unit_to_svg_abbrev(unit: &Expr) -> String {
  use crate::functions::quantity_ast::unit_to_abbreviation;

  // Handle Power in both BinaryOp and FunctionCall form
  if let Some((base, exp)) = as_power(unit) {
    let base_str = quantity_unit_to_svg_abbrev(base);
    let exp_str = expr_to_svg_markup(exp);
    return format!(
      "{base_str}<tspan baseline-shift=\"super\" font-size=\"70%\">{exp_str}</tspan>"
    );
  }

  match unit {
    Expr::Identifier(s) | Expr::String(s) => {
      let abbr = unit_to_abbreviation(s).unwrap_or(s.as_str());
      svg_escape(abbr)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      format!(
        "{}/{}",
        quantity_unit_to_svg_abbrev(left),
        quantity_unit_to_svg_abbrev(right)
      )
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      format!(
        "{}\u{22c5}{}",
        quantity_unit_to_svg_abbrev(left),
        quantity_unit_to_svg_abbrev(right)
      )
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      // Check for fraction form: Times[..., Power[den, -n]]
      let mut numer_parts: Vec<String> = Vec::new();
      let mut denom_parts: Vec<String> = Vec::new();
      for a in args {
        if let Some((base, neg_exp)) = crate::syntax::extract_neg_power_info(a)
        {
          let base_str = quantity_unit_to_svg_abbrev(base);
          if neg_exp == -1 {
            denom_parts.push(base_str);
          } else {
            // For SVG, use superscript for the positive exponent
            denom_parts.push(format!(
              "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
              base_str, -neg_exp
            ));
          }
        } else {
          numer_parts.push(quantity_unit_to_svg_abbrev(a));
        }
      }
      if denom_parts.is_empty() {
        numer_parts.join("\u{22c5}")
      } else {
        let numer = if numer_parts.is_empty() {
          "1".to_string()
        } else {
          numer_parts.join("\u{22c5}")
        };
        let denom = denom_parts.join("\u{22c5}");
        format!("{numer}/{denom}")
      }
    }
    _ => expr_to_svg_markup(unit),
  }
}

/// Group digits of a number string in threes (from the right) with thin spaces
/// for SVG display, matching Wolfram's graphical output.
/// Only applies to numbers with 5 or more digits.
/// Handles an optional leading minus sign.
fn group_digits_svg(s: &str) -> String {
  let (sign, digits) = if let Some(rest) = s.strip_prefix('-') {
    ("−", rest) // use Unicode minus for display
  } else {
    ("", s)
  };

  if digits.len() < 5 || !digits.chars().all(|c| c.is_ascii_digit()) {
    return svg_escape(s);
  }

  // Group from the right in threes.
  // Wrap every group after the first in a <tspan dx="0.3ch"> so that
  // the dx offset actually shifts the characters inside the tspan.
  let remainder = digits.len() % 3;
  let mut result = String::with_capacity(s.len() + 20);
  result.push_str(sign);

  if remainder > 0 {
    result.push_str(&digits[..remainder]);
  }
  for (i, chunk) in digits.as_bytes()[remainder..].chunks(3).enumerate() {
    let chunk_str = std::str::from_utf8(chunk).unwrap();
    if i > 0 || remainder > 0 {
      // Thin space between groups (~30% of a monospace character width)
      result.push_str("<tspan dx=\"0.3ch\">");
      result.push_str(chunk_str);
      result.push_str("</tspan>");
    } else {
      result.push_str(chunk_str);
    }
  }
  result
}

/// Calculate the extra display width added by digit grouping.
/// Returns the number of thin-space separators × 0.3 character widths.
fn digit_group_extra_width(digit_count: usize) -> f64 {
  if digit_count < 5 {
    return 0.0;
  }
  let remainder = digit_count % 3;
  let num_groups = digit_count / 3 + usize::from(remainder > 0);
  (num_groups - 1) as f64 * 0.3
}

/// Read an optional precision/`n` argument of a display form, defaulting to
/// `default` when it is absent or non-integer.
fn format_precision_arg(arg: Option<&Expr>, default: i64) -> i64 {
  match arg {
    Some(Expr::Integer(n)) => *n as i64,
    _ => default,
  }
}

/// The digits `BaseForm[x, base]` displays, via the one renderer that
/// knows every value kind (integers, machine reals, and
/// arbitrary-precision reals shown to their own precision).
fn base_form_digits(x: &Expr, base: &Expr) -> Option<String> {
  let Expr::Integer(b) = base else {
    return None;
  };
  crate::functions::string_ast::base_form_digits(x, *b)
}

/// Convert an `Expr` into SVG text markup (inner content of a `<text>` element).
/// Recursively handles all expression types so that Power expressions
/// anywhere in the tree are rendered with `<tspan>` superscripts.
/// The SVG text attributes a `Style[…]` directive list asks for — colour,
/// slant, weight and size — for the `tspan` the styled content goes into.
/// Directives with no textual meaning (or none we render) are skipped.
fn style_directives_to_svg_attrs(directives: &[Expr]) -> String {
  // One slot per SVG attribute, last directive wins: `Style[expr, "Label",
  // 12]` names a stylesheet style *and* overrides its size, exactly as
  // Wolfram reads it. Appending both would emit the attribute twice, which
  // is not valid SVG and leaves the size up to the renderer's tie-break.
  let mut attrs: Vec<(&'static str, String)> = Vec::new();
  let mut set = |name: &'static str, value: String| match attrs
    .iter_mut()
    .find(|(n, _)| *n == name)
  {
    Some(slot) => slot.1 = value,
    None => attrs.push((name, value)),
  };
  for d in directives {
    match d {
      Expr::Identifier(s) if s == "Italic" => {
        set("font-style", "italic".to_string());
      }
      Expr::Identifier(s) if s == "Bold" => {
        set("font-weight", "bold".to_string());
      }
      // A bare number is the font size (`Style[expr, 12]`).
      Expr::Integer(_) | Expr::Real(_) => {
        if let Some(size) = expr_to_f64(d) {
          set("font-size", format!("{size:.0}"));
        }
      }
      // A named style brings its size and colour from the stylesheet.
      Expr::String(name) => {
        if let Some((size, color)) = named_style_appearance(name) {
          set("font-size", format!("{size:.0}"));
          if let Some((r, g, b)) = color {
            set("fill", format!("rgb({r},{g},{b})"));
          }
        }
      }
      Expr::Rule {
        pattern,
        replacement,
      } => match option_name(pattern) {
        Some("FontSize") => {
          if let Some(size) = expr_to_f64(replacement) {
            set("font-size", format!("{size:.0}"));
          }
        }
        Some("FontColor") => {
          if let Some(c) = parse_color(replacement) {
            set("fill", c.to_svg_rgb());
          }
        }
        Some("FontSlant") => {
          if matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Italic")
          {
            set("font-style", "italic".to_string());
          }
        }
        Some("FontWeight") => {
          if matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Bold")
          {
            set("font-weight", "bold".to_string());
          }
        }
        _ => {}
      },
      other => {
        if let Some(c) = parse_color(other) {
          set("fill", c.to_svg_rgb());
        }
      }
    }
  }
  use std::fmt::Write;
  attrs.iter().fold(String::new(), |mut out, (name, value)| {
    let _ = write!(out, " {name}=\"{value}\"");
    out
  })
}

/// The root index `n` when `exp` is the unit fraction `1/n` — the shape a
/// radical is written from (`Sqrt[x]` is `x^(1/2)` after evaluation).
fn unit_fraction_root_index(exp: &Expr) -> Option<i128> {
  let (num, den) = match exp {
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) => (*n, *d),
        _ => return None,
      }
    }
    _ => return None,
  };
  (num == 1 && (2..=9).contains(&den)).then_some(den)
}

/// The markup for `-term` when a `Plus` term carries a negative *coefficient*
/// other than -1 (`Times[-5, x]`), so the sum reads `-5 - 5 x` rather than
/// `-5 + -5 x`. The -1 case is handled by the caller, which drops the
/// coefficient entirely.
fn negated_markup_term(arg: &Expr) -> Option<String> {
  let flip_first = |first: &Expr| match first {
    Expr::Integer(n) if *n < 0 => Some(Expr::Integer(-n)),
    Expr::Real(f) if *f < 0.0 => Some(Expr::Real(-f)),
    _ => None,
  };
  let (first, rest): (&Expr, &[Expr]) = match arg {
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      (&args[0], &args[1..])
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => (left.as_ref(), std::slice::from_ref(right.as_ref())),
    Expr::Real(f) if *f < 0.0 => {
      return Some(expr_to_svg_markup(&Expr::Real(-f)));
    }
    _ => return None,
  };
  let positive = flip_first(first)?;
  let mut factors = vec![positive];
  factors.extend(rest.iter().cloned());
  Some(expr_to_svg_markup(&unevaluated("Times", &factors)))
}

/// Typeset a label that may occupy several lines, one entry per line.
/// `Grid[{{a}, {b}}]` and `Column[{a, b}]` stack their items — Demonstrations
/// build multi-line plot titles that way — and any line that itself carries
/// newlines (the 2D text `ToString[…, TraditionalForm]` returns) splits
/// further. Everything else is a single line.
pub fn expr_to_svg_markup_lines(expr: &Expr) -> Vec<String> {
  let rows: Vec<String> = match expr {
    Expr::FunctionCall { name, args } if name == "Grid" && !args.is_empty() => {
      match &args[0] {
        Expr::List(rows) => rows
          .iter()
          .map(|row| match row {
            Expr::List(cells) => cells
              .iter()
              .map(expr_to_svg_markup)
              .collect::<Vec<_>>()
              .join(" "),
            cell => expr_to_svg_markup(cell),
          })
          .collect(),
        other => vec![expr_to_svg_markup(other)],
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Column" && !args.is_empty() =>
    {
      match &args[0] {
        Expr::List(items) => items.iter().map(expr_to_svg_markup).collect(),
        other => vec![expr_to_svg_markup(other)],
      }
    }
    other => vec![expr_to_svg_markup(other)],
  };
  rows
    .iter()
    .flat_map(|line| line.split('\n').map(str::to_string).collect::<Vec<_>>())
    .collect()
}

pub fn expr_to_svg_markup(expr: &Expr) -> String {
  // A unit-fraction power is a radical, not a superscript: `Sqrt[2]`
  // (which is `2^(1/2)`) typesets as √2 under its vinculum, and a cube
  // root carries its index in the hook.
  if let Some((base, exp)) = as_power(expr)
    && let Some(index) = unit_fraction_root_index(exp)
  {
    let content = expr_to_svg_markup(base);
    return if index == 2 {
      format!("\u{221A}<tspan text-decoration=\"overline\">{content}</tspan>")
    } else {
      format!(
        "<tspan baseline-shift=\"super\" font-size=\"70%\">{index}</tspan>\u{221A}<tspan text-decoration=\"overline\">{content}</tspan>"
      )
    };
  }

  // Power → superscript (handles both BinaryOp and FunctionCall forms)
  if let Some((base, exp)) = as_power(expr) {
    let base_markup = expr_to_svg_markup(base);
    let exp_markup = expr_to_svg_markup(exp);
    // Wrap base in parens if it's a lower-precedence additive expression
    let base_fmt = if is_additive_expr(base) {
      format!("({base_markup})")
    } else {
      base_markup
    };
    return format!(
      "{base_fmt}<tspan baseline-shift=\"super\" font-size=\"70%\">{exp_markup}</tspan>"
    );
  }

  match expr {
    // ── Atoms ──
    // A string may embed Wolfram linear-syntax boxes
    // (`"area = 0.68 \!\(\*SuperscriptBox[\(AU\), \(2\)]\)"`); those typeset
    // into sub/superscript tspans. For ordinary text this is a plain escape.
    Expr::String(s) => box_string_to_svg(s),
    // A mathematical constant is set as its glyph wherever an expression is
    // typeset — `π`, not the word — the same way a `Text` label sets it.
    // Arithmetic leaves a constant as either spelling, so both are matched.
    Expr::Identifier(name) | Expr::Constant(name)
      if typeset_constant_glyph(name).is_some() =>
    {
      typeset_constant_glyph(name).unwrap().to_string()
    }
    Expr::Identifier(s) => svg_escape(s),
    Expr::BigFloat(digits, prec) => {
      // Graphical output shows `prec` significant digits with ×10^exp for large/small numbers
      let parts = bigfloat_display_parts(digits, *prec);
      if let Some(exp) = parts.exponent {
        format!(
          "{}×10<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          svg_escape(&parts.mantissa),
          exp
        )
      } else {
        svg_escape(&parts.mantissa)
      }
    }
    Expr::Integer(n) => group_digits_svg(&n.to_string()),
    Expr::BigInteger(n) => group_digits_svg(&n.to_string()),
    Expr::Real(f) => {
      // Machine reals display at 6 significant figures in the notebook front
      // end (unlike the full-precision InputForm), with scientific notation for
      // very large / small magnitudes rendered as a superscript exponent.
      let parts = machine_real_display_parts(*f);
      match parts.exponent {
        Some(exp) => format!(
          "{}×10<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          svg_escape(&parts.mantissa),
          exp
        ),
        None => svg_escape(&parts.mantissa),
      }
    }
    Expr::Constant(_) | Expr::Slot(_) => svg_escape(&expr_to_output(expr)),

    // ── List → {a, b, c} ──
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_svg_markup).collect();
      format!("{{{}}}", parts.join(", "))
    }

    // ── UnaryOp ──
    Expr::UnaryOp { op, operand } => {
      let inner = expr_to_svg_markup(operand);
      match op {
        UnaryOperator::Minus => format!("-{inner}"),
        UnaryOperator::Not => format!("!{inner}"),
      }
    }

    // ── BinaryOp (Power already handled above) ──
    Expr::BinaryOp { op, left, right } => {
      // Power should already be caught by as_power() above, but handle
      // it gracefully as a superscript instead of panicking.
      if matches!(op, BinaryOperator::Power) {
        let base_markup = expr_to_svg_markup(left);
        let exp_markup = expr_to_svg_markup(right);
        let base_fmt = if is_additive_expr(left) {
          format!("({base_markup})")
        } else {
          base_markup
        };
        return format!(
          "{base_fmt}<tspan baseline-shift=\"super\" font-size=\"70%\">{exp_markup}</tspan>"
        );
      }
      let (op_str, needs_space) = match op {
        BinaryOperator::Plus => ("+", true),
        BinaryOperator::Minus => ("-", true),
        BinaryOperator::Times => (times_svg_separator(left, right), false),
        BinaryOperator::Divide => ("/", false),
        BinaryOperator::Power => unreachable!(),
        BinaryOperator::And => ("&amp;&amp;", true),
        BinaryOperator::Or => ("||", true),
        BinaryOperator::StringJoin => ("&lt;&gt;", false),
        BinaryOperator::Alternatives => ("|", true),
      };
      let is_mult =
        matches!(op, BinaryOperator::Times | BinaryOperator::Divide);
      let left_str = expr_to_svg_markup(left);
      let right_str = expr_to_svg_markup(right);
      let left_fmt = if is_mult && is_additive_expr(left) {
        format!("({left_str})")
      } else {
        left_str
      };
      let right_fmt = if is_mult && is_additive_expr(right.as_ref()) {
        format!("({right_str})")
      } else {
        right_str
      };
      if needs_space {
        format!("{left_fmt} {op_str} {right_fmt}")
      } else {
        format!("{left_fmt}{op_str}{right_fmt}")
      }
    }

    // ── Comparison → a == b, a < b, etc. ──
    Expr::Comparison {
      operands,
      operators,
    } => {
      let mut result = expr_to_svg_markup(&operands[0]);
      for (i, op) in operators.iter().enumerate() {
        let op_str = match op {
          ComparisonOp::Equal => " == ",
          ComparisonOp::NotEqual => " != ",
          ComparisonOp::Less => " &lt; ",
          ComparisonOp::LessEqual => " &lt;= ",
          ComparisonOp::Greater => " &gt; ",
          ComparisonOp::GreaterEqual => " &gt;= ",
          ComparisonOp::SameQ => " === ",
          ComparisonOp::UnsameQ => " =!= ",
        };
        result.push_str(op_str);
        result.push_str(&expr_to_svg_markup(&operands[i + 1]));
      }
      result
    }

    // ── Rule → pattern -> replacement ──
    Expr::Rule {
      pattern,
      replacement,
    } => {
      format!(
        "{} -&gt; {}",
        expr_to_svg_markup(pattern),
        expr_to_svg_markup(replacement)
      )
    }

    // ── Association → <|k1 -> v1, ...|> ──
    Expr::Association(items) => {
      let parts: Vec<String> = items
        .iter()
        .map(|(k, v)| {
          format!("{} -&gt; {}", expr_to_svg_markup(k), expr_to_svg_markup(v))
        })
        .collect();
      format!("&lt;|{}|&gt;", parts.join(", "))
    }

    // ── FunctionCall ──
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        // Missing[...] → rendered as a dash
        "Missing" => "-".to_string(),

        // Plus[a, b, ...] with negative-term handling
        "Plus" if args.len() >= 2 => {
          let mut result = expr_to_svg_markup(&args[0]);
          for arg in &args[1..] {
            if let Expr::UnaryOp {
              op: UnaryOperator::Minus,
              operand,
            } = arg
            {
              result.push_str(" - ");
              result.push_str(&expr_to_svg_markup(operand));
            } else if let Expr::BinaryOp {
              op: BinaryOperator::Times,
              left,
              right,
            } = arg
              && matches!(left.as_ref(), Expr::Integer(-1))
            {
              result.push_str(" - ");
              result.push_str(&expr_to_svg_markup(right));
            } else if let Expr::FunctionCall {
              name: fn_name,
              args: fn_args,
            } = arg
              && fn_name == "Times"
              && fn_args.len() >= 2
              && matches!(&fn_args[0], Expr::Integer(-1))
            {
              result.push_str(" - ");
              if fn_args.len() == 2 {
                result.push_str(&expr_to_svg_markup(&fn_args[1]));
              } else {
                result.push_str(&expr_to_svg_markup(&unevaluated(
                  "Times",
                  &fn_args[1..],
                )));
              }
            } else if let Expr::Integer(n) = arg
              && *n < 0
            {
              result.push_str(" - ");
              result.push_str(&expr_to_svg_markup(&Expr::Integer(-n)));
            } else if let Some(positive) = negated_markup_term(arg) {
              result.push_str(" - ");
              result.push_str(&positive);
            } else {
              result.push_str(" + ");
              result.push_str(&expr_to_svg_markup(arg));
            }
          }
          result
        }

        // Times[a, b, ...] with -1 coefficient and Rational handling
        "Times" if args.len() >= 2 => {
          // Times[Rational[1, d], expr] → stacked fraction expr/d
          if args.len() == 2
            && let Expr::FunctionCall {
              name: rname,
              args: rargs,
            } = &args[0]
            && rname == "Rational"
            && rargs.len() == 2
            && matches!(&rargs[0], Expr::Integer(1))
            && matches!(&rargs[1], Expr::Integer(d) if *d > 0)
          {
            let num_markup = expr_to_svg_markup(&args[1]);
            let den_markup = expr_to_svg_markup(&rargs[1]);
            let num_w = estimate_display_width(&args[1]);
            let den_w = estimate_display_width(&rargs[1]);
            return stacked_fraction_svg(
              &num_markup,
              &den_markup,
              num_w,
              den_w,
            );
          }
          // Times[-1, x, ...] → -x...
          if matches!(&args[0], Expr::Integer(-1)) {
            let rest_args = &args[1..];
            let rest: Vec<String> = rest_args
              .iter()
              .map(|a| {
                let s = expr_to_svg_markup(a);
                if is_additive_expr(a) {
                  format!("({s})")
                } else {
                  s
                }
              })
              .collect();
            let mut joined = rest[0].clone();
            for i in 1..rest.len() {
              joined.push_str(times_svg_separator(
                &rest_args[i - 1],
                &rest_args[i],
              ));
              joined.push_str(&rest[i]);
            }
            return format!("-{joined}");
          }
          // General: implicit multiplication (no * symbol)
          let parts: Vec<String> = args
            .iter()
            .map(|a| {
              let s = expr_to_svg_markup(a);
              if is_additive_expr(a) {
                format!("({s})")
              } else {
                s
              }
            })
            .collect();
          let mut result = parts[0].clone();
          for i in 1..parts.len() {
            result.push_str(times_svg_separator(&args[i - 1], &args[i]));
            result.push_str(&parts[i]);
          }
          result
        }

        // Rational[n, d] → stacked fraction
        "Rational" if args.len() == 2 => {
          let num_markup = expr_to_svg_markup(&args[0]);
          let den_markup = expr_to_svg_markup(&args[1]);
          let num_w = estimate_display_width(&args[0]);
          let den_w = estimate_display_width(&args[1]);
          stacked_fraction_svg(&num_markup, &den_markup, num_w, den_w)
        }

        // FullForm[expr] → render in canonical notation
        "FullForm" if args.len() == 1 => {
          let full_form =
            crate::functions::predicate_ast::expr_to_full_form(&args[0]);
          svg_escape(&full_form)
        }

        // Quantity[magnitude, unit] → "magnitude abbreviation"
        "Quantity" if args.len() == 2 => {
          let mag = expr_to_svg_markup(&args[0]);
          let unit = quantity_unit_to_svg_abbrev(&args[1]);
          let unit = crate::syntax::singularize_unit_if_one(&args[0], &unit);
          format!("{mag} {unit}")
        }

        // CForm/TeXForm/FortranForm → display converted text
        "CForm" if args.len() == 1 => {
          svg_escape(&crate::functions::string_ast::expr_to_c(&args[0]))
        }
        "TeXForm" if args.len() == 1 => {
          svg_escape(&crate::functions::string_ast::expr_to_tex(&args[0]))
        }
        "FortranForm" if args.len() == 1 => {
          svg_escape(&crate::functions::string_ast::expr_to_fortran(&args[0]))
        }

        // ScientificForm[x] / [x, n] → mantissa × 10^exp with a superscript
        // exponent (matching the number's notebook typesetting).
        "ScientificForm" if !args.is_empty() => {
          let n = format_precision_arg(args.get(1), 6);
          match crate::functions::string_ast::scientific_form_parts(&args[0], n)
          {
            Some((mantissa, Some(exp))) => format!(
              "{}×10<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
              svg_escape(&mantissa),
              exp
            ),
            Some((mantissa, None)) => svg_escape(&mantissa),
            None => expr_to_svg_markup(&args[0]),
          }
        }

        // EngineeringForm[x] / [x, n] → like ScientificForm but exp is a
        // multiple of 3.
        "EngineeringForm" if !args.is_empty() => {
          let n = format_precision_arg(args.get(1), 6);
          match crate::functions::string_ast::engineering_form_parts(
            &args[0], n,
          ) {
            Some((mantissa, Some(exp))) => format!(
              "{}×10<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
              svg_escape(&mantissa),
              exp
            ),
            Some((mantissa, None)) => svg_escape(&mantissa),
            None => expr_to_svg_markup(&args[0]),
          }
        }

        // BaseForm[x, b] → digits of x in base b with a subscript base.
        "BaseForm" if args.len() == 2 => {
          if let (Some(digits), Expr::Integer(base)) =
            (base_form_digits(&args[0], &args[1]), &args[1])
          {
            format!(
              "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>",
              svg_escape(&digits),
              base
            )
          } else {
            expr_to_svg_markup(&args[0])
          }
        }

        // PaddedForm[BaseForm[x, b], n, opts…] → the base-b digits padded to
        // the n+1 field with the NumberPadding character, with a subscript
        // base (e.g. 00000101₂). Leading spaces from the default padding are
        // dropped (the grid handles alignment); visible pad characters such
        // as "0" are kept.
        "PaddedForm"
          if crate::functions::string_ast::padded_form_base_parts(args)
            .is_some() =>
        {
          let (digits, base) =
            crate::functions::string_ast::padded_form_base_parts(args).unwrap();
          let shown = digits.trim_start_matches(' ');
          if base == 10 {
            svg_escape(shown)
          } else {
            format!(
              "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>",
              svg_escape(shown),
              base
            )
          }
        }

        // NumberForm / PaddedForm / AccountingForm → the formatted number text
        // (padding/grouping is folded into the string; the grid handles
        // alignment). Multi-line 2-D forms are flattened to a single line. When
        // the wrapper can't format its argument (e.g. a nested BaseForm that
        // ToString leaves intact), fall back to rendering the inner value so a
        // grid of numbers still appears rather than raw wrapper text.
        "NumberForm" | "PaddedForm" | "AccountingForm" if !args.is_empty() => {
          match crate::functions::string_ast::number_form_family_to_string(
            name, args,
          ) {
            Some(s) => svg_escape(s.replace('\n', " ").trim()),
            None => expr_to_svg_markup(&args[0]),
          }
        }

        // A box expression typesets rather than printing itself: a
        // stored `UnderscriptBox["1", "_"]` is the digit `1̲`, not the
        // text `UnderscriptBox[1, _]`.
        n if is_typeset_box_head(n) => boxes_to_svg(expr),

        // Style[content, directives...] → the content, wrapped in a
        // tspan carrying the colour, slant, weight and size the
        // directives ask for (a label reads `Style["P", Blue, Italic]`).
        "Style" if !args.is_empty() => {
          let content = expr_to_svg_markup(&args[0]);
          // `Text[…]` sets its content in the graphic's own text style, so
          // the weight and slant it would inherit are reset: Wolfram draws
          // `Style[Text["a"], Bold]` plain, while the colour and size still
          // reach through. A `Style` written inside the `Text` still applies.
          let directives: Vec<Expr> = if matches!(&args[0],
            Expr::FunctionCall { name, args: ta } if name == "Text" && ta.len() == 1)
          {
            args[1..]
              .iter()
              .filter(|d| !is_font_face_directive(d))
              .cloned()
              .collect()
          } else {
            args[1..].to_vec()
          };
          let attrs = style_directives_to_svg_attrs(&directives);
          if attrs.is_empty() {
            content
          } else {
            format!("<tspan{attrs}>{content}</tspan>")
          }
        }

        // HoldForm[expr] → render content
        "HoldForm" if args.len() == 1 => expr_to_svg_markup(&args[0]),

        // Tooltip[content, tip] → render content (the tip only shows on
        // hover in the Wolfram FrontEnd, which static SVG can't do)
        "Tooltip" if !args.is_empty() => expr_to_svg_markup(&args[0]),

        // `TraditionalForm[expr]` asks for conventional mathematical
        // notation, so typeset the expression instead of printing its
        // InputForm: `Sum[…, {n, 0, ∞}]` becomes a ∑ with limits and
        // `LegendreP[n, x]` becomes `Pₙ(x)`.
        "TraditionalForm" if args.len() == 1 => boxes_to_svg(
          &crate::evaluator::dispatch::complex_and_special::
            expr_to_box_form_traditional(&args[0]),
        ),

        // Presentation wrappers display their content only.
        "Text" | "DisplayForm" | "StandardForm" if args.len() == 1 => {
          expr_to_svg_markup(&args[0])
        }

        // Subscript[base, i, …] / Superscript[base, e, …] typeset as
        // shifted tspans — a Demonstrations plot label reads
        // `Row[{α, " = ", K, "(", Subscript[p, 0], ")"}]`, which must show
        // `p₀`, not the 1D `Subscript[p, 0]` that OutputForm keeps.
        "Subscript" | "Superscript" if args.len() >= 2 => {
          let shift = if name == "Subscript" { "sub" } else { "super" };
          let scripts: String = args[1..]
            .iter()
            .map(expr_to_svg_markup)
            .collect::<Vec<_>>()
            .join(",");
          format!(
            "{}<tspan baseline-shift=\"{shift}\" font-size=\"70%\">{}</tspan>",
            expr_to_svg_markup(&args[0]),
            scripts
          )
        }

        // `Spacer[n]` is a gap n printer's points wide, wherever it
        // appears — among a `Row`'s items as readily as as its separator.
        // The width is absolute, not relative to the font: `Spacer[25]`
        // is the same gap next to 8-point text as next to 30-point text.
        "Spacer" => {
          let pts = args.first().and_then(expr_to_f64).unwrap_or(1.0);
          format!("<tspan style=\"letter-spacing:{pts:.2}px\"> </tspan>")
        }

        // Framed[content, …] / Highlighted[content, …] set inline text —
        // a `PlotLabel`/`AxesLabel` markup line has no room to draw the
        // frame's border or highlight fill around it, so just typeset the
        // content (matching `Tooltip`, whose hover-only chrome the same
        // static context can't show either).
        "Framed" | "Highlighted" if !args.is_empty() => {
          expr_to_svg_markup(&args[0])
        }

        // Row[{a, b, …}] concatenates its parts; Row[{…}, sep] joins
        // them with the separator.
        "Row" if !args.is_empty() => match &args[0] {
          Expr::List(parts) => {
            let sep = args
              .get(1)
              .map(|s| match s {
                // `Spacer[n]` separates with a gap n ems wide, rather
                // than printing itself.
                Expr::FunctionCall { name, args: sargs }
                  if name == "Spacer" =>
                {
                  let pts = sargs.first().and_then(expr_to_f64).unwrap_or(1.0);
                  format!(
                    "<tspan style=\"letter-spacing:{pts:.2}px\"> </tspan>"
                  )
                }
                other => expr_to_svg_markup(other),
              })
              .unwrap_or_default();
            parts
              .iter()
              .map(expr_to_svg_markup)
              .collect::<Vec<_>>()
              .join(&sep)
          }
          other => expr_to_svg_markup(other),
        },

        // General FunctionCall: name[arg1, arg2, ...]
        _ => {
          let parts: Vec<String> =
            args.iter().map(expr_to_svg_markup).collect();
          if args.is_empty() {
            format!("{}[]", svg_escape(name))
          } else {
            format!("{}[{}]", svg_escape(name), parts.join(", "))
          }
        }
      }
    }

    // ── Expr::Image → placeholder text (actual embedding happens in grid) ──
    Expr::Image { width, height, .. } => {
      format!("-Image ({width}×{height})-")
    }

    // ── Curried call f[a][b] → head markup + bracketed args, so display
    // wrappers in the head resolve (e.g. HoldForm[f][x] shows as f[x]) ──
    Expr::CurriedCall { func, args } => {
      let parts: Vec<String> = args.iter().map(expr_to_svg_markup).collect();
      format!("{}[{}]", expr_to_svg_markup(func), parts.join(", "))
    }

    // ── Everything else → fallback to expr_to_output ──
    _ => svg_escape(&expr_to_output(expr)),
  }
}

/// Estimate the display width of an expression in character units,
/// accounting for superscript sizing (exponents rendered at ~70% width).
/// Recursively mirrors `expr_to_svg_markup` structure.
pub fn estimate_display_width(expr: &Expr) -> f64 {
  if let Some((base, exp)) = as_power(expr) {
    let parens = if is_additive_expr(base) { 2.0 } else { 0.0 };
    return estimate_display_width(base)
      + parens
      + estimate_display_width(exp) * 0.7;
  }

  match expr {
    // Atoms
    // A string carrying inline linear syntax
    // (`\!\(\*SubscriptBox[\(S\), \(ABC\)]\)`) displays as the typeset
    // formula, which is far shorter than the markup it is written with —
    // measure what will actually be seen, or a grid column holding one
    // comes out several times too wide.
    Expr::String(s) if s.contains(crate::functions::string_ast::BOX_START) => {
      box_string_visible_len(s) as f64
    }
    Expr::String(s) => s.len() as f64,
    Expr::Identifier(s) => s.len() as f64,
    Expr::BigFloat(digits, prec) => {
      let parts = bigfloat_display_parts(digits, *prec);
      if let Some(exp) = parts.exponent {
        // mantissa + "×10" (3 chars) + superscript exponent at 70% width
        let exp_str = exp.to_string();
        parts.mantissa.len() as f64 + 3.0 + exp_str.len() as f64 * 0.7
      } else {
        parts.mantissa.len() as f64
      }
    }
    Expr::Integer(n) => {
      let s = n.to_string();
      let digit_count = s.trim_start_matches('-').len();
      s.len() as f64 + digit_group_extra_width(digit_count)
    }
    Expr::BigInteger(n) => {
      let s = n.to_string();
      let digit_count = s.trim_start_matches('-').len();
      s.len() as f64 + digit_group_extra_width(digit_count)
    }
    Expr::Real(_) | Expr::Constant(_) | Expr::Slot(_) => {
      expr_to_output(expr).len() as f64
    }

    // List → {a, b, c}: 2 for braces + items + separators
    Expr::List(items) => {
      let inner: f64 = items.iter().map(estimate_display_width).sum();
      let seps = if items.len() > 1 {
        (items.len() - 1) as f64 * 2.0
      } else {
        0.0
      };
      2.0 + inner + seps
    }

    // UnaryOp: 1 char prefix + operand
    Expr::UnaryOp { operand, .. } => 1.0 + estimate_display_width(operand),

    // BinaryOp
    Expr::BinaryOp { op, left, right } => {
      // Power should already be caught by as_power() above, but handle
      // it gracefully instead of panicking.
      if matches!(op, BinaryOperator::Power) {
        let parens = if is_additive_expr(left) { 2.0 } else { 0.0 };
        return estimate_display_width(left)
          + parens
          + estimate_display_width(right) * 0.7;
      }
      let is_mult =
        matches!(op, BinaryOperator::Times | BinaryOperator::Divide);
      let op_len: f64 = match op {
        BinaryOperator::Plus | BinaryOperator::Minus => 3.0,
        BinaryOperator::Times => times_svg_separator(left, right).len() as f64,
        BinaryOperator::Divide => 1.0,
        BinaryOperator::Power => unreachable!(),
        BinaryOperator::And => 4.0,
        BinaryOperator::Or => 4.0,
        BinaryOperator::StringJoin => 2.0,
        BinaryOperator::Alternatives => 3.0,
      };
      let left_parens = if is_mult && is_additive_expr(left) {
        2.0
      } else {
        0.0
      };
      let right_parens = if is_mult && is_additive_expr(right) {
        2.0
      } else {
        0.0
      };
      estimate_display_width(left)
        + left_parens
        + op_len
        + estimate_display_width(right)
        + right_parens
    }

    // Comparison: operands + operators
    Expr::Comparison {
      operands,
      operators,
    } => {
      let ops_width: f64 = operators
        .iter()
        .map(|_| 4.0_f64) // approximate: " == ", " < ", etc.
        .sum();
      let operands_width: f64 =
        operands.iter().map(estimate_display_width).sum();
      operands_width + ops_width
    }

    // Rule: pattern -> replacement (4 chars for " -> ")
    Expr::Rule {
      pattern,
      replacement,
    } => {
      estimate_display_width(pattern)
        + 4.0
        + estimate_display_width(replacement)
    }

    // Association: <|...|> (4 chars overhead + items)
    Expr::Association(items) => {
      let inner: f64 = items
        .iter()
        .map(|(k, v)| {
          estimate_display_width(k) + 4.0 + estimate_display_width(v)
        })
        .sum();
      let seps = if items.len() > 1 {
        (items.len() - 1) as f64 * 2.0
      } else {
        0.0
      };
      4.0 + inner + seps
    }

    // FunctionCall
    Expr::FunctionCall { name, args } => match name.as_str() {
      "FullForm" if args.len() == 1 => {
        let full_form =
          crate::functions::predicate_ast::expr_to_full_form(&args[0]);
        full_form.len() as f64
      }
      "Plus" if args.len() >= 2 => {
        let terms: f64 = args.iter().map(estimate_display_width).sum();
        terms + (args.len() - 1) as f64 * 3.0
      }
      "Times" if args.len() >= 2 => {
        // Times[Rational[1, d], expr] → stacked fraction expr/d
        if args.len() == 2
          && let Expr::FunctionCall {
            name: rname,
            args: rargs,
          } = &args[0]
          && rname == "Rational"
          && rargs.len() == 2
          && matches!(&rargs[0], Expr::Integer(1))
          && matches!(&rargs[1], Expr::Integer(d) if *d > 0)
        {
          return stacked_fraction_width(
            estimate_display_width(&args[1]),
            estimate_display_width(&rargs[1]),
          );
        }
        if matches!(&args[0], Expr::Integer(-1)) {
          let rest_args = &args[1..];
          let rest: f64 = rest_args
            .iter()
            .map(|a| {
              let w = estimate_display_width(a);
              if is_additive_expr(a) { w + 2.0 } else { w }
            })
            .sum();
          let sep_width: f64 = rest_args
            .windows(2)
            .map(|w| times_svg_separator(&w[0], &w[1]).len() as f64)
            .sum();
          1.0 + rest + sep_width
        } else {
          let factors: f64 = args
            .iter()
            .map(|a| {
              let w = estimate_display_width(a);
              if is_additive_expr(a) { w + 2.0 } else { w }
            })
            .sum();
          let sep_width: f64 = args
            .windows(2)
            .map(|w| times_svg_separator(&w[0], &w[1]).len() as f64)
            .sum();
          factors + sep_width
        }
      }
      "Rational" if args.len() == 2 => stacked_fraction_width(
        estimate_display_width(&args[0]),
        estimate_display_width(&args[1]),
      ),
      "Quantity" if args.len() == 2 => {
        // "magnitude unit_abbrev" — 1 space between
        estimate_display_width(&args[0])
          + 1.0
          + estimate_unit_abbrev_width(&args[1])
      }
      // Style[content, ...] → width of content
      "Style" | "StyleForm" if !args.is_empty() => {
        estimate_display_width(&args[0])
      }
      // HoldForm[expr] → width of content
      "HoldForm" if args.len() == 1 => estimate_display_width(&args[0]),
      // The presentation wrappers `expr_to_svg_markup` types out as their
      // content only — measuring the `Head[…]` source instead would leave a
      // Row cell several times too wide for the glyphs drawn in it.
      // `Invisible[content]` is one of them: it reserves exactly the space
      // `content` takes, it just isn't painted.
      "Text" | "TraditionalForm" | "DisplayForm" | "StandardForm"
      | "Invisible"
        if args.len() == 1 =>
      {
        estimate_display_width(&args[0])
      }
      // Subscript[base, i, …] / Superscript[base, e, …] typeset as shifted
      // tspans at 70% size, the scripts comma-separated.
      "Subscript" | "Superscript" if args.len() >= 2 => {
        let scripts: f64 = args[1..].iter().map(estimate_display_width).sum();
        let seps = (args.len() - 2) as f64;
        estimate_display_width(&args[0]) + (scripts + seps) * 0.7
      }
      // Row[{a, b, …}] concatenates its parts, joined by the separator.
      "Row" if !args.is_empty() => match &args[0] {
        Expr::List(parts) => {
          let sep = args.get(1).map_or(0.0, |s| match s {
            Expr::FunctionCall { name, args: sargs } if name == "Spacer" => {
              sargs.first().and_then(expr_to_f64).unwrap_or(1.0)
            }
            other => estimate_display_width(other),
          });
          let inner: f64 = parts.iter().map(estimate_display_width).sum();
          inner + sep * parts.len().saturating_sub(1) as f64
        }
        other => estimate_display_width(other),
      },
      // Tooltip[content, tip] → width of content (tip is hover-only)
      "Tooltip" if !args.is_empty() => estimate_display_width(&args[0]),
      // Number-display wrappers estimate the width of their *rendered* form
      // (mantissa × 10^exp, base digits, padded number) rather than the raw
      // `Head[...]` text, so table columns aren't wildly over-sized.
      "ScientificForm" | "EngineeringForm" if !args.is_empty() => {
        let n = format_precision_arg(args.get(1), 6);
        let parts = if name == "ScientificForm" {
          crate::functions::string_ast::scientific_form_parts(&args[0], n)
        } else {
          crate::functions::string_ast::engineering_form_parts(&args[0], n)
        };
        match parts {
          Some((mantissa, Some(exp))) => {
            // mantissa + "×10" (3) + exponent as a 70%-width superscript
            mantissa.len() as f64 + 3.0 + exp.to_string().len() as f64 * 0.7
          }
          Some((mantissa, None)) => mantissa.len() as f64,
          None => estimate_display_width(&args[0]),
        }
      }
      "BaseForm" if args.len() == 2 => {
        match base_form_digits(&args[0], &args[1]) {
          // digits + subscript base at 70% width
          Some(digits) => {
            let base_len = match &args[1] {
              Expr::Integer(b) => b.to_string().len(),
              _ => 1,
            };
            digits.len() as f64 + base_len as f64 * 0.7
          }
          None => estimate_display_width(&args[0]),
        }
      }
      // Padded base-b digits + subscript base at 70% width, mirroring the
      // markup branch above (leading space padding is not rendered).
      "PaddedForm"
        if crate::functions::string_ast::padded_form_base_parts(args)
          .is_some() =>
      {
        let (digits, base) =
          crate::functions::string_ast::padded_form_base_parts(args).unwrap();
        let shown = digits.trim_start_matches(' ');
        let base_len = if base == 10 {
          0
        } else {
          base.to_string().len()
        };
        shown.chars().count() as f64 + base_len as f64 * 0.7
      }
      "NumberForm" | "PaddedForm" | "AccountingForm" if !args.is_empty() => {
        match crate::functions::string_ast::number_form_family_to_string(
          name, args,
        ) {
          Some(s) => s.replace('\n', " ").trim().chars().count() as f64,
          None => estimate_display_width(&args[0]),
        }
      }
      _ => {
        let args_width: f64 = args.iter().map(estimate_display_width).sum();
        let seps = if args.len() > 1 {
          (args.len() - 1) as f64 * 2.0
        } else {
          0.0
        };
        name.len() as f64 + 2.0 + args_width + seps
      }
    },

    // Expr::Image → width in character units, capped at standard display size.
    // Mathematica's default image display width is ~180pt (= 240 CSS px at 96 DPI).
    Expr::Image { width, .. } => (*width as f64).min(240.0) / 8.4,

    // Curried call f[a][b] → head width + brackets + args, mirroring the
    // markup branch (so HoldForm[f][x] is sized as f[x]).
    Expr::CurriedCall { func, args } => {
      let args_width: f64 = args.iter().map(estimate_display_width).sum();
      let seps = if args.len() > 1 {
        (args.len() - 1) as f64 * 2.0
      } else {
        0.0
      };
      estimate_display_width(func) + 2.0 + args_width + seps
    }

    // Fallback
    _ => expr_to_output(expr).len() as f64,
  }
}

/// Estimate the display width of an abbreviated unit expression.
fn estimate_unit_abbrev_width(unit: &Expr) -> f64 {
  use crate::functions::quantity_ast::unit_to_abbreviation;

  if let Some((base, exp)) = as_power(unit) {
    return estimate_unit_abbrev_width(base)
      + estimate_display_width(exp) * 0.7;
  }

  match unit {
    Expr::Identifier(s) | Expr::String(s) => {
      let abbr = unit_to_abbreviation(s).unwrap_or(s.as_str());
      abbr.len() as f64
    }
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      estimate_unit_abbrev_width(left) + 1.0 + estimate_unit_abbrev_width(right)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      // · separator = 1 char
      estimate_unit_abbrev_width(left) + 1.0 + estimate_unit_abbrev_width(right)
    }
    Expr::FunctionCall { name, args }
      if name == "Times" && !args.is_empty() =>
    {
      let parts: f64 = args.iter().map(estimate_unit_abbrev_width).sum();
      parts + (args.len() - 1) as f64 // · separators
    }
    _ => estimate_display_width(unit),
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Box-form → SVG rendering
// ═══════════════════════════════════════════════════════════════════════

/// Convert a box-form expression (produced by `expr_to_box_form()`) to SVG
/// text markup.  This mirrors `expr_to_svg_markup()` but operates on the
/// intermediate box representation (RowBox, SuperscriptBox, FractionBox, …)
/// rather than raw Expr trees.
/// Whether `name` is a box head that [`boxes_to_svg`] typesets — the
/// forms a stored notebook expression carries its 2-D layout in.
fn is_typeset_box_head(name: &str) -> bool {
  matches!(
    name,
    "SubscriptBox"
      | "SuperscriptBox"
      | "SubsuperscriptBox"
      | "UnderscriptBox"
      | "OverscriptBox"
      | "UnderoverscriptBox"
      | "FractionBox"
      | "SqrtBox"
      | "RadicalBox"
  )
}

/// A box atom holding a machine-precision real carries the `` ` `` precision
/// marker — `ToBoxes[1.5]` is the box string ``"1.5`"``. The marker is
/// notation, not a glyph: the FrontEnd never draws it, so strip it before
/// the number reaches a picture.
fn strip_precision_marker(s: &str) -> &str {
  match s.strip_suffix('`') {
    Some(head)
      if !head.is_empty()
        && head
          .chars()
          .all(|c| c.is_ascii_digit() || c == '.' || c == '-') =>
    {
      head
    }
    _ => s,
  }
}

pub fn boxes_to_svg(expr: &Expr) -> String {
  match expr {
    // Atoms: in box form, atoms are always Expr::String
    Expr::String(s) => svg_escape(strip_precision_marker(s)),
    // Identifiers can appear for fallback cases
    Expr::Identifier(s) => svg_escape(strip_precision_marker(s)),
    Expr::Integer(n) => group_digits_svg(&n.to_string()),
    Expr::BigInteger(n) => group_digits_svg(&n.to_string()),

    Expr::FunctionCall { name, args } => match name.as_str() {
      // RowBox[{e1, e2, ...}] → concatenate children
      // Commas get a trailing space for readability (matching Wolfram rendering).
      "RowBox" if args.len() == 1 => {
        if let Expr::List(items) = &args[0] {
          let mut result = String::new();
          for (i, item) in items.iter().enumerate() {
            let rendered = boxes_to_svg(item);
            result.push_str(&rendered);
            // Add space after comma separators (bare "," strings)
            if rendered == "," && i + 1 < items.len() {
              result.push(' ');
            }
          }
          result
        } else {
          // Single non-list arg: just render it
          boxes_to_svg(&args[0])
        }
      }

      // SuperscriptBox[base, exp]
      "SuperscriptBox" if args.len() == 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let exp_svg = boxes_to_svg(&args[1]);
        format!(
          "{base_svg}<tspan baseline-shift=\"super\" font-size=\"70%\">{exp_svg}</tspan>"
        )
      }

      // SubscriptBox[base, sub]
      "SubscriptBox" if args.len() == 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let sub_svg = boxes_to_svg(&args[1]);
        format!(
          "{base_svg}<tspan baseline-shift=\"sub\" font-size=\"70%\">{sub_svg}</tspan>"
        )
      }

      // SubsuperscriptBox[base, sub, sup]
      "SubsuperscriptBox" if args.len() == 3 => {
        let base_svg = boxes_to_svg(&args[0]);
        let sub_svg = boxes_to_svg(&args[1]);
        let sup_svg = boxes_to_svg(&args[2]);
        format!(
          "{base_svg}<tspan baseline-shift=\"sub\" font-size=\"70%\">{sub_svg}</tspan>\
           <tspan baseline-shift=\"super\" font-size=\"70%\">{sup_svg}</tspan>"
        )
      }

      // FractionBox[num, den] → stacked fraction
      "FractionBox" if args.len() == 2 => {
        let num_svg = boxes_to_svg(&args[0]);
        let den_svg = boxes_to_svg(&args[1]);
        let num_w = estimate_box_display_width(&args[0]);
        let den_w = estimate_box_display_width(&args[1]);
        stacked_fraction_svg(&num_svg, &den_svg, num_w, den_w)
      }

      // SqrtBox[expr] → √content with overline
      "SqrtBox" if args.len() == 1 => {
        let content = boxes_to_svg(&args[0]);
        format!("\u{221A}<tspan text-decoration=\"overline\">{content}</tspan>")
      }

      // RadicalBox[expr, n] → index√content with overline
      "RadicalBox" if args.len() == 2 => {
        let content = boxes_to_svg(&args[0]);
        let index = boxes_to_svg(&args[1]);
        format!(
          "<tspan baseline-shift=\"super\" font-size=\"70%\">{index}</tspan>\u{221A}<tspan text-decoration=\"overline\">{content}</tspan>"
        )
      }

      // OverscriptBox[base, over] → base with overscript
      "OverscriptBox" if args.len() >= 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let over_svg = boxes_to_svg(&args[1]);
        format!(
          "{base_svg}<tspan baseline-shift=\"super\" font-size=\"70%\">{over_svg}</tspan>"
        )
      }

      // UnderscriptBox[base, under] → base with underscript. An
      // underscript that is just a rule (`"_"`) sits *under* the base
      // rather than beside it — the balanced-ternary digit `1̲` — so it
      // reads as an underline.
      "UnderscriptBox" if args.len() >= 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        if matches!(&args[1], Expr::String(u) if u == "_" || u == "\u{23df}") {
          return format!(
            "<tspan text-decoration=\"underline\">{base_svg}</tspan>"
          );
        }
        let under_svg = boxes_to_svg(&args[1]);
        format!(
          "{base_svg}<tspan baseline-shift=\"sub\" font-size=\"70%\">{under_svg}</tspan>"
        )
      }

      // UnderoverscriptBox[base, under, over] → base with both
      "UnderoverscriptBox" if args.len() >= 3 => {
        let base_svg = boxes_to_svg(&args[0]);
        let under_svg = boxes_to_svg(&args[1]);
        let over_svg = boxes_to_svg(&args[2]);
        format!(
          "{base_svg}<tspan baseline-shift=\"sub\" font-size=\"70%\">{under_svg}</tspan>\
           <tspan baseline-shift=\"super\" font-size=\"70%\">{over_svg}</tspan>"
        )
      }

      // FrameBox[content, ...] → content with frame markers
      "FrameBox" if !args.is_empty() => {
        let content = boxes_to_svg(&args[0]);
        format!("[{content}]")
      }

      // TagBox[boxes, tag, opts...] → render boxes, ignore tag and options
      "TagBox" if args.len() >= 2 => boxes_to_svg(&args[0]),

      // FormBox[boxes, form] → render boxes, ignore the form marker
      "FormBox" if !args.is_empty() => boxes_to_svg(&args[0]),

      // InterpretationBox[display, interpretation] → render display part only
      "InterpretationBox" if args.len() >= 2 => boxes_to_svg(&args[0]),

      // StyleBox[content, ...] → render content with style attributes
      "StyleBox" if !args.is_empty() => {
        let content = boxes_to_svg(&args[0]);
        let mut font_size_attr = String::new();
        let mut color_attr = String::new();
        for opt in &args[1..] {
          let (key, val) = match opt {
            Expr::Rule {
              pattern,
              replacement,
            } => (pattern.as_ref(), replacement.as_ref()),
            Expr::FunctionCall { name: rn, args: ra }
              if rn == "Rule" && ra.len() == 2 =>
            {
              (&ra[0], &ra[1])
            }
            _ => continue,
          };
          if let Expr::Identifier(k) = key {
            match k.as_str() {
              "FontSize" => {
                if let Some(sz) = expr_to_f64(val) {
                  font_size_attr = format!(" font-size=\"{sz}\"");
                }
              }
              "FontColor" => {
                if let Some(color) = parse_color(val) {
                  color_attr = format!(" fill=\"{}\"", color.to_svg_rgb());
                }
              }
              _ => {}
            }
          }
        }
        if font_size_attr.is_empty() && color_attr.is_empty() {
          content
        } else {
          format!("<tspan{font_size_attr}{color_attr}>{content}</tspan>")
        }
      }

      // GridBox[{{...}, ...}] → simple text rendering
      "GridBox" if !args.is_empty() => {
        if let Expr::List(rows) = &args[0] {
          let row_strs: Vec<String> = rows
            .iter()
            .map(|row| {
              if let Expr::List(cells) = row {
                cells
                  .iter()
                  .map(boxes_to_svg)
                  .collect::<Vec<_>>()
                  .join("\t")
              } else {
                boxes_to_svg(row)
              }
            })
            .collect();
          row_strs.join("\n")
        } else {
          boxes_to_svg(&args[0])
        }
      }

      // Unknown box type: render as Name[arg1, arg2, ...]
      _ => {
        let parts: Vec<String> = args.iter().map(boxes_to_svg).collect();
        if args.is_empty() {
          format!("{}[]", svg_escape(name))
        } else {
          format!("{}[{}]", svg_escape(name), parts.join(", "))
        }
      }
    },

    Expr::List(items) => {
      // Lists in box form (e.g. inside RowBox) – just concatenate
      items.iter().map(boxes_to_svg).collect::<String>()
    }

    // Fallback: use expr_to_output for anything else
    _ => svg_escape(&crate::syntax::expr_to_output(expr)),
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Inline box-notation strings (Wolfram "linear syntax") → SVG
// ═══════════════════════════════════════════════════════════════════════
//
// Label strings such as PlotLegends/FrameLabel entries can embed Wolfram
// "linear syntax" box notation, e.g.
//   "C\!\(\*SubscriptBox[\(\),\(2\)]\)=9.78 GeV\!\(\*SuperscriptBox[\(\),\(-3\)]\)"
// where `\!\(...\)` wraps a displayed box and `\*Head[\(..\),\(..\)]` is the
// explicit box function form. After string-literal parsing these escapes are
// stored either as the literal two-character sequences (`\(`, `\*`, …) or as
// the private-use marker codepoints. `box_string_to_svg` resolves both into
// SVG `<text>` content with proper sub/superscript/sqrt `<tspan>`s.

/// Normalize private-use box marker codepoints back to their literal
/// escape-sequence form (`\!`, `\(`, `\*`, `\)`) so a single parser handles
/// both representations.
fn normalize_box_markers(s: &str) -> String {
  use crate::functions::string_ast::{BOX_CLOSE, BOX_OPEN, BOX_SEP, BOX_START};
  let mut out = String::with_capacity(s.len());
  for c in s.chars() {
    match c {
      _ if c == BOX_START => out.push_str("\\!"),
      _ if c == BOX_OPEN => out.push_str("\\("),
      _ if c == BOX_SEP => out.push_str("\\*"),
      _ if c == BOX_CLOSE => out.push_str("\\)"),
      _ => out.push(c),
    }
  }
  out
}

/// Find the index of the backslash of the `\)` that matches the `\(` whose
/// content starts at `start` (depth already 1). Returns `None` if unbalanced.
fn find_box_group_close(cs: &[char], start: usize) -> Option<usize> {
  let mut depth = 1usize;
  let mut j = start;
  while j < cs.len() {
    if cs[j] == '\\' && j + 1 < cs.len() {
      match cs[j + 1] {
        '(' => {
          depth += 1;
          j += 2;
          continue;
        }
        ')' => {
          depth -= 1;
          if depth == 0 {
            return Some(j);
          }
          j += 2;
          continue;
        }
        _ => {
          j += 1;
          continue;
        }
      }
    }
    j += 1;
  }
  None
}

/// Trim leading/trailing whitespace from a char slice. The whitespace
/// *between* box arguments (e.g. after a comma, before `\(`) is syntactic and
/// must not become rendered text; whitespace inside a `\(...\)` group is kept
/// because it lives within the group's own slice.
fn trim_char_slice(cs: &[char]) -> &[char] {
  let mut start = 0;
  let mut end = cs.len();
  while start < end && cs[start].is_whitespace() {
    start += 1;
  }
  while end > start && cs[end - 1].is_whitespace() {
    end -= 1;
  }
  &cs[start..end]
}

/// Parse a `\*Head[\(arg\),\(arg\),…]` explicit box, with `cs[pos..]` pointing
/// just past the `\*`. Returns the resulting box Expr and the index past the
/// consumed tokens.
fn parse_explicit_box(cs: &[char], pos: usize) -> (Expr, usize) {
  // Read the head name (letters/digits/`$`).
  let mut i = pos;
  let name_start = i;
  while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '$') {
    i += 1;
  }
  let name: String = cs[name_start..i].iter().collect();
  if i >= cs.len() || cs[i] != '[' {
    // Bare box symbol (no bracketed args) — render as an atom.
    return (Expr::Identifier(name), i);
  }
  // Parse bracketed, comma-separated args. Commas inside nested `\(...\)`
  // groups or `[...]` brackets do not split.
  i += 1; // consume '['
  let mut args: Vec<Expr> = Vec::new();
  let mut arg_start = i;
  let mut gdepth = 0usize; // `\(` group depth
  let mut bdepth = 0usize; // `[` bracket depth
  while i < cs.len() {
    let c = cs[i];
    if c == '\\' && i + 1 < cs.len() && cs[i + 1] == '(' {
      gdepth += 1;
      i += 2;
      continue;
    }
    if c == '\\' && i + 1 < cs.len() && cs[i + 1] == ')' {
      gdepth = gdepth.saturating_sub(1);
      i += 2;
      continue;
    }
    if gdepth > 0 {
      i += 1;
      continue;
    }
    match c {
      '[' => {
        bdepth += 1;
        i += 1;
      }
      ']' if bdepth == 0 => {
        // End of the argument list.
        if !(args.is_empty() && arg_start == i) {
          args.push(parse_box_to_expr(trim_char_slice(&cs[arg_start..i])));
        }
        i += 1; // consume ']'
        return (
          Expr::FunctionCall {
            name,
            args: args.into(),
          },
          i,
        );
      }
      ']' => {
        bdepth -= 1;
        i += 1;
      }
      ',' if bdepth == 0 => {
        args.push(parse_box_to_expr(trim_char_slice(&cs[arg_start..i])));
        i += 1;
        arg_start = i;
      }
      _ => i += 1,
    }
  }
  // Unbalanced — consume what we have.
  if arg_start < cs.len() {
    args.push(parse_box_to_expr(trim_char_slice(&cs[arg_start..])));
  }
  (
    Expr::FunctionCall {
      name,
      args: args.into(),
    },
    cs.len(),
  )
}

/// Parse a sequence of box-notation units (plain runs, `\(...\)` groups and
/// `\*Head[...]` explicit boxes) into a list of box Exprs.
fn parse_box_units(cs: &[char]) -> Vec<Expr> {
  let mut res: Vec<Expr> = Vec::new();
  let mut plain = String::new();
  // A linear-syntax group may open with the form it is typeset in —
  // `\!\(TraditionalForm\`16 …\)`. The name says how to read the rest and
  // shows nothing itself. (`\`` inside a string parses to its own marker,
  // U+F7CD, so it cannot be confused with a literal backtick.)
  let mut i = match cs.iter().position(|c| *c == '\u{f7cd}') {
    Some(tick)
      if tick > 0
        && cs[..tick].iter().collect::<String>().ends_with("Form")
        && cs[..tick].iter().all(char::is_ascii_alphanumeric) =>
    {
      tick + 1
    }
    _ => 0,
  };
  while i < cs.len() {
    if cs[i] == '\\' && i + 1 < cs.len() {
      match cs[i + 1] {
        // An escaped space is a space: linear syntax writes the gaps in a
        // formula that way, since a bare space would be ignored.
        ' ' => {
          plain.push(' ');
          i += 2;
          continue;
        }
        '*' => {
          if !plain.is_empty() {
            res.push(Expr::String(std::mem::take(&mut plain)));
          }
          let (e, ni) = parse_explicit_box(cs, i + 2);
          res.push(e);
          i = ni;
          continue;
        }
        '(' => {
          if !plain.is_empty() {
            res.push(Expr::String(std::mem::take(&mut plain)));
          }
          match find_box_group_close(cs, i + 2) {
            Some(close) => {
              res.push(parse_box_to_expr(&cs[i + 2..close]));
              i = close + 2;
            }
            None => i += 1,
          }
          continue;
        }
        // Lone `\!` interpret marker (the following `\(` is handled next
        // iteration) and a stray `\)` — skip the two-char marker.
        '!' | ')' => {
          i += 2;
          continue;
        }
        _ => {}
      }
    }
    plain.push(cs[i]);
    i += 1;
  }
  if !plain.is_empty() {
    res.push(Expr::String(plain));
  }
  res
}

/// Parse box-notation content into a single box Expr (wrapping multiple units
/// in a `RowBox`).
fn parse_box_to_expr(cs: &[char]) -> Expr {
  let mut units = parse_box_units(cs);
  match units.len() {
    0 => Expr::String(String::new()),
    1 => units.pop().unwrap(),
    _ => call1("RowBox", Expr::List(units.into())),
  }
}

/// Render a label string that may contain inline Wolfram box notation into
/// SVG `<text>` content. Plain strings (no box notation) are simply
/// SVG-escaped, so this is a safe drop-in for `svg_escape`.
pub fn box_string_to_svg(s: &str) -> String {
  let norm = normalize_box_markers(s);
  let cs: Vec<char> = norm.chars().collect();
  parse_box_units(&cs).iter().map(boxes_to_svg).collect()
}

/// Plain-text projection of a box-notation Expr, used for layout width
/// estimation (sub/superscripts contribute their content length).
fn box_expr_to_plain(e: &Expr) -> String {
  match e {
    Expr::String(s) | Expr::Identifier(s) => {
      strip_precision_marker(s).to_string()
    }
    Expr::Integer(n) => n.to_string(),
    Expr::BigInteger(n) => n.to_string(),
    Expr::List(items) => items.iter().map(box_expr_to_plain).collect(),
    Expr::FunctionCall { name, args } => match name.as_str() {
      "SqrtBox" | "RadicalBox" => format!(
        "\u{221A}{}",
        args.first().map(box_expr_to_plain).unwrap_or_default()
      ),
      _ => args.iter().map(box_expr_to_plain).collect(),
    },
    _ => String::new(),
  }
}

/// Number of visible characters a box-notation label occupies, ignoring the
/// box markup itself. Used to size legends/labels instead of the raw byte
/// length (which over-counts the `\!\(\*…\)` scaffolding).
pub fn box_string_visible_len(s: &str) -> usize {
  let norm = normalize_box_markers(s);
  let cs: Vec<char> = norm.chars().collect();
  parse_box_units(&cs)
    .iter()
    .map(|u| box_expr_to_plain(u).chars().count())
    .sum()
}

/// Estimate the display width of a box-form expression in character units.
/// Assemble box markup into a complete SVG string.
/// Handles fraction markers by splitting text around nested `<svg>` elements
/// with `<line>` for fraction bars.
pub fn estimate_box_display_width(expr: &Expr) -> f64 {
  match expr {
    Expr::String(s) => s.len() as f64,
    Expr::Identifier(s) => s.len() as f64,
    Expr::Integer(n) => {
      let s = n.to_string();
      let digit_count = s.trim_start_matches('-').len();
      s.len() as f64 + digit_group_extra_width(digit_count)
    }
    Expr::BigInteger(n) => {
      let s = n.to_string();
      let digit_count = s.trim_start_matches('-').len();
      s.len() as f64 + digit_group_extra_width(digit_count)
    }

    Expr::FunctionCall { name, args } => match name.as_str() {
      "RowBox" if args.len() == 1 => {
        if let Expr::List(items) = &args[0] {
          items.iter().map(estimate_box_display_width).sum()
        } else {
          estimate_box_display_width(&args[0])
        }
      }
      "SuperscriptBox" if args.len() == 2 => {
        estimate_box_display_width(&args[0])
          + estimate_box_display_width(&args[1]) * 0.7
      }
      "SubscriptBox" if args.len() == 2 => {
        estimate_box_display_width(&args[0])
          + estimate_box_display_width(&args[1]) * 0.7
      }
      "SubsuperscriptBox" if args.len() == 3 => {
        let base = estimate_box_display_width(&args[0]);
        let sub = estimate_box_display_width(&args[1]) * 0.7;
        let sup = estimate_box_display_width(&args[2]) * 0.7;
        base + sub.max(sup)
      }
      "FractionBox" if args.len() == 2 => stacked_fraction_width(
        estimate_box_display_width(&args[0]),
        estimate_box_display_width(&args[1]),
      ),
      "SqrtBox" if args.len() == 1 => {
        // √( + content + )
        3.0 + estimate_box_display_width(&args[0])
      }
      "RadicalBox" if args.len() == 2 => {
        estimate_box_display_width(&args[1]) * 0.7
          + 2.0
          + estimate_box_display_width(&args[0])
      }
      "StyleBox" if !args.is_empty() => estimate_box_display_width(&args[0]),
      "OverscriptBox" if args.len() >= 2 => {
        estimate_box_display_width(&args[0])
          + estimate_box_display_width(&args[1]) * 0.7
      }
      "UnderscriptBox" if args.len() >= 2 => {
        estimate_box_display_width(&args[0])
          + estimate_box_display_width(&args[1]) * 0.7
      }
      "UnderoverscriptBox" if args.len() >= 3 => {
        let base = estimate_box_display_width(&args[0]);
        let under = estimate_box_display_width(&args[1]) * 0.7;
        let over = estimate_box_display_width(&args[2]) * 0.7;
        base + under.max(over)
      }
      "FrameBox" if !args.is_empty() => {
        estimate_box_display_width(&args[0]) + 2.0
      }
      "TagBox" if args.len() >= 2 => estimate_box_display_width(&args[0]),
      "FormBox" if !args.is_empty() => estimate_box_display_width(&args[0]),
      "InterpretationBox" if args.len() >= 2 => {
        estimate_box_display_width(&args[0])
      }
      _ => {
        let args_width: f64 = args.iter().map(estimate_box_display_width).sum();
        let seps = if args.len() > 1 {
          (args.len() - 1) as f64 * 2.0
        } else {
          0.0
        };
        name.len() as f64 + 2.0 + args_width + seps
      }
    },

    Expr::List(items) => items.iter().map(estimate_box_display_width).sum(),

    _ => crate::syntax::expr_to_output(expr).len() as f64,
  }
}

/// Check whether a box-form expression contains a FractionBox anywhere,
/// which requires extra vertical space in the SVG wrapper.
pub fn box_has_fraction(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } if name == "FractionBox" => true,
    Expr::FunctionCall { args, .. } => args.iter().any(box_has_fraction),
    Expr::List(items) => items.iter().any(box_has_fraction),
    _ => false,
  }
}

/// The `(name, value)` of an option, written either way round.
///
/// `name :> value` means the same as `name -> value` for an option: the
/// right-hand side is evaluated where the option is *used*, and rendering a
/// graphic uses it once. Demonstrations reach for the delayed form so a
/// label or style tracks the controls, e.g. `PlotLabel :> Which[t == 1100,
/// "…", True, ""]`. Every option reader goes through here so both spellings
/// stay supported together.
pub(crate) fn option_name_value(
  opt: &Expr,
) -> Option<(&str, std::borrow::Cow<'_, Expr>)> {
  let (pattern, replacement, delayed) = match opt {
    Expr::Rule {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref(), false),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => (pattern.as_ref(), replacement.as_ref(), true),
    _ => return None,
  };
  let name = match pattern {
    Expr::Identifier(name) | Expr::Constant(name) => name.as_str(),
    _ => return None,
  };
  if delayed {
    // `:>` holds its right-hand side until the option is used — which is
    // now, so evaluate it against the current bindings.
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| replacement.clone());
    Some((name, std::borrow::Cow::Owned(value)))
  } else {
    Some((name, std::borrow::Cow::Borrowed(replacement)))
  }
}

/// Extract the option name from a Rule pattern (e.g. Identifier("ImageSize") -> "ImageSize")
fn option_name(expr: &Expr) -> Option<&str> {
  if let Expr::Identifier(name) = expr {
    Some(name.as_str())
  } else {
    None
  }
}

/// Merge an option into a list, replacing any existing option with the same name.
fn merge_option(opts: &mut Vec<Expr>, opt: &Expr) {
  if let Some((opt_name, replacement)) = option_name_value(opt) {
    // For PlotRange, compute the union (min of mins, max of maxes)
    // so that all merged graphics remain visible.
    if opt_name == "PlotRange"
      && let Some(pos) = opts
        .iter()
        .position(|e| matches!(option_name_value(e), Some(("PlotRange", _))))
      && let Some((_, existing_repl)) = option_name_value(&opts[pos])
    {
      let merged = merge_plot_ranges(&existing_repl, &replacement);
      opts[pos] = Expr::Rule {
        pattern: Box::new(Expr::Identifier("PlotRange".to_string())),
        replacement: Box::new(merged),
      };
      return;
    }

    opts.retain(|existing| {
      option_name_value(existing).map(|(n, _)| n) != Some(opt_name)
    });
  }
  opts.push(opt.clone());
}

/// Merge two PlotRange values by taking the union (min of mins, max of maxes).
fn merge_plot_ranges(a: &Expr, b: &Expr) -> crate::syntax::Expr {
  let (ax, ay) = parse_plot_range(a);
  let (bx, by) = parse_plot_range(b);

  let merge_range =
    |r1: Option<(f64, f64)>, r2: Option<(f64, f64)>| -> Option<(f64, f64)> {
      match (r1, r2) {
        (Some((lo1, hi1)), Some((lo2, hi2))) => {
          Some((lo1.min(lo2), hi1.max(hi2)))
        }
        (Some(r), None) | (None, Some(r)) => Some(r),
        (None, None) => None,
      }
    };

  let mx = merge_range(ax, bx);
  let my = merge_range(ay, by);

  let range_to_expr = |r: Option<(f64, f64)>| -> Expr {
    match r {
      Some((lo, hi)) => Expr::List(vec![Expr::Real(lo), Expr::Real(hi)].into()),
      None => Expr::Identifier("All".to_string()),
    }
  };

  Expr::List(vec![range_to_expr(mx), range_to_expr(my)].into())
}

/// Implementation of Show[g1, g2, ..., opts...].
/// Convert MeshRegion vertex/polygon data to Graphics primitives (Polygon with coordinates).
pub(crate) fn mesh_region_to_graphics_prims(
  vertices_expr: &Expr,
  primitives_expr: &Expr,
) -> Option<Vec<Expr>> {
  let Expr::List(vertices_list) = vertices_expr else {
    return None;
  };
  let mut vertices: Vec<(f64, f64)> = Vec::new();
  for v in vertices_list {
    if let Expr::List(coords) = v
      && coords.len() == 2
      && let (Some(x), Some(y)) = (
        crate::functions::math_ast::try_eval_to_f64(&coords[0]),
        crate::functions::math_ast::try_eval_to_f64(&coords[1]),
      )
    {
      vertices.push((x, y));
      continue;
    }
    return None;
  }

  let Expr::List(prims) = primitives_expr else {
    return None;
  };

  let mut result = Vec::new();
  // Add default styling
  result.push(call1("EdgeForm", Color::gray(0.4).to_expr()));
  result.push(call1("FaceForm", Color::new(0.626, 0.836, 0.919).to_expr()));

  for prim in prims {
    if let Expr::FunctionCall { name, args } = prim
      && name == "Polygon"
      && args.len() == 1
      && let Expr::List(index_lists) = &args[0]
    {
      for idx_list in index_lists {
        if let Expr::List(indices) = idx_list {
          let points: Vec<Expr> = indices
            .iter()
            .filter_map(|idx| {
              crate::functions::math_ast::try_eval_to_f64(idx).and_then(|i| {
                let i = i as usize;
                if i >= 1 && i <= vertices.len() {
                  let (x, y) = vertices[i - 1];
                  Some(Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
                } else {
                  None
                }
              })
            })
            .collect();
          if points.len() >= 3 {
            result.push(call1("Polygon", Expr::List(points.into())));
          }
        }
      }
    }
  }
  Some(result)
}

/// Merges multiple Graphics[...] calls into a single Graphics[...] call,
/// combining their primitives and options. Arguments are kept unevaluated
/// (Show is in the held-args list) so Graphics[...] expressions arrive as
/// FunctionCall nodes rather than being rendered to `-Graphics-`.
/// Whether a `Graphics[…]` argument is a finished picture rather than a
/// list of primitives — an already-rendered graphic, another `Graphics`,
/// or a call (a plot) that evaluates to one.
pub fn wraps_rendered_graphic(content: &Expr) -> bool {
  match content {
    Expr::Graphics { .. } => true,
    Expr::FunctionCall { name, .. } if name == "Graphics" => true,
    // Lists, primitives and directives are content, not pictures; only a
    // call worth evaluating can turn out to be one.
    Expr::FunctionCall { .. } => {
      matches!(evaluate_expr_to_expr(content), Ok(Expr::Graphics { .. }))
    }
    _ => false,
  }
}

/// The drawing primitives a rendered plot's sampled series stand for —
/// filled regions, the series colour and thickness, and the `Line` /
/// `Point` the samples make up. `Show` merges these with the primitives of
/// the graphics it is layering, and a structural rule (`plot /. L_Line :>
/// …`) matches against them, which is what makes a plot's curve reachable
/// by pattern replacement at all.
pub fn plot_source_primitives(ps: &crate::syntax::PlotSource) -> Vec<Expr> {
  let mut series_prims: Vec<Expr> = Vec::new();
  for sd in &ps.series {
    // Filled region (Filling -> …) as a Polygon underneath the curve,
    // wrapped in its own list so the fill color/opacity directives
    // don't leak onto the line drawn after it. FillingStyle appearance
    // travels on the series (fill_color/fill_opacity); the defaults
    // match the standalone plot render (series color at 0.2 opacity).
    if !sd.is_scatter
      && let Some(ref_y) = sd.filling.reference_y(ps.y_range.0, ps.y_range.1)
    {
      let (fr, fg, fb) = sd.fill_color.unwrap_or(sd.color);
      let mut fill_prims: Vec<Expr> = vec![
        call1("Opacity", Expr::Real(sd.fill_opacity.unwrap_or(0.2))),
        call(
          "RGBColor",
          vec![
            Expr::Real(fr as f64 / 255.0),
            Expr::Real(fg as f64 / 255.0),
            Expr::Real(fb as f64 / 255.0),
          ],
        ),
      ];
      for seg in &crate::functions::plot::split_into_segments(&sd.points) {
        if seg.len() < 2 {
          continue;
        }
        let mut coords: Vec<Expr> = seg
          .iter()
          .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
          .collect();
        let (x_last, _) = seg[seg.len() - 1];
        let (x_first, _) = seg[0];
        coords.push(Expr::List(
          vec![Expr::Real(x_last), Expr::Real(ref_y)].into(),
        ));
        coords.push(Expr::List(
          vec![Expr::Real(x_first), Expr::Real(ref_y)].into(),
        ));
        fill_prims.push(call1("Polygon", Expr::List(coords.into())));
      }
      series_prims.push(Expr::List(fill_prims.into()));
    }
    // Color directive
    series_prims.push(call(
      "RGBColor",
      vec![
        Expr::Real(sd.color.0 as f64 / 255.0),
        Expr::Real(sd.color.1 as f64 / 255.0),
        Expr::Real(sd.color.2 as f64 / 255.0),
      ],
    ));
    if sd.is_scatter {
      series_prims.push(call1("PointSize", Expr::Real(0.012)));
      let coords: Vec<Expr> = sd
        .points
        .iter()
        .filter(|(_, y)| y.is_finite())
        .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
        .collect();
      if !coords.is_empty() {
        series_prims.push(call1("Point", Expr::List(coords.into())));
      }
    } else {
      series_prims.push(call1(
        "AbsoluteThickness",
        Expr::Real(sd.thickness.unwrap_or(1.5)),
      ));
      let segments = crate::functions::plot::split_into_segments(&sd.points);
      for seg in &segments {
        let coords: Vec<Expr> = seg
          .iter()
          .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
          .collect();
        if coords.len() >= 2 {
          series_prims.push(call1("Line", Expr::List(coords.into())));
        }
      }
    }
  }
  series_prims
}

pub fn show_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut merged_primitives: Vec<Expr> = Vec::new();
  let mut merged_options: Vec<Expr> = Vec::new();
  let mut is_3d = false;
  // Pre-rendered Graphics objects (e.g. from Plot[], Plot3D[])
  let mut rendered_graphics: Vec<Expr> = Vec::new();
  // Plot source data for re-rendering via plotters
  let mut plot_sources: Vec<crate::syntax::PlotSource> = Vec::new();
  // The layers in the order `Show` was given them, so a mixture of plots
  // and primitive graphics stacks the way Wolfram stacks it — the last
  // argument on top. Each entry indexes either `merged_primitives` or
  // `plot_sources`.
  enum Layer {
    Prims(usize),
    Source(usize),
  }
  let mut layers: Vec<Layer> = Vec::new();
  // Whether the first graphic argument was a plot (carried a PlotSource).
  // Wolfram takes the merged result's options from the first graphic, so
  // plot defaults (axes, 1/GoldenRatio aspect) apply only in that case.
  let mut first_graphic_is_plot: Option<bool> = None;
  // Options carried over from the graphics being shown (see the
  // `Expr::Graphics` arm below).
  let mut inherited_options: Vec<Expr> = Vec::new();
  // The single graphic of a `Show[g]`, and whether `Show` was given any
  // options of its own — with neither a second graphic nor an option to
  // apply, `Show[g]` is `g`.
  let mut only_graphic: Option<Expr> = None;
  let mut graphic_count = 0usize;
  let mut has_own_options = false;

  // `Show[{g1, g2, …}, opts…]` — flatten a leading List argument into
  // multiple graphics args (Wolfram convention; not Listable but accepts
  // a list-of-graphics form alongside the variadic form).
  let flat_args_owned: Vec<Expr>;
  let args: &[Expr] = if let Some((first, rest)) = args.split_first()
    && let Expr::List(items) = first
  {
    flat_args_owned = items.iter().chain(rest.iter()).cloned().collect();
    &flat_args_owned
  } else {
    args
  };

  // `Show[image, opts…]` — a raster Image argument passes through: Show
  // of an image just displays it (sizing options like `ImageSize -> 100`
  // don't alter the pixel data), e.g. the `Show[ColorData[name, "Image"],
  // ImageSize -> 100]` gradient swatches of the Demonstrations site.
  if let Some((first, rest)) = args.split_first()
    && rest
      .iter()
      .all(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
  {
    let evaled_first;
    let first_ref = match first {
      Expr::FunctionCall { name, .. }
        if name == "Graphics" || name == "Graphics3D" =>
      {
        first
      }
      Expr::Image { .. } => first,
      _ => {
        evaled_first =
          evaluate_expr_to_expr(first).unwrap_or_else(|_| first.clone());
        &evaled_first
      }
    };
    if matches!(first_ref, Expr::Image { .. }) {
      return Ok(first_ref.clone());
    }
  }

  // Walk the args with an explicit work list: an argument that evaluates
  // to a *list* of graphics (e.g. `Show[{g, {h1, h2}}]`, or a variable
  // holding a collected list of Graphics) is spliced in place so its
  // elements merge like ordinary arguments instead of being dropped.
  let mut pending: Vec<Expr> = args.to_vec();
  let mut idx = 0;
  while idx < pending.len() {
    let arg = pending[idx].clone();
    // If the arg is not already a Graphics/Graphics3D expression,
    // try evaluating it (e.g. it could be a variable or function call)
    let expr_owned = match &arg {
      // A `Graphics[…]` call stays symbolic so its primitives can merge
      // with the other arguments' — unless it wraps a finished picture
      // (`Graphics[ContourPlot[…]]`), in which case its content is no
      // primitive at all and the wrapper has to be rendered to get the
      // picture it stands for.
      Expr::FunctionCall { name, args: gargs }
        if (name == "Graphics" || name == "Graphics3D")
          && !gargs.is_empty()
          && wraps_rendered_graphic(&gargs[0]) =>
      {
        let gargs: Vec<Expr> = gargs.iter().cloned().collect();
        if name == "Graphics" {
          graphics_ast(&gargs).unwrap_or_else(|_| arg.clone())
        } else {
          crate::functions::plot3d::graphics3d_ast(&gargs)
            .unwrap_or_else(|_| arg.clone())
        }
      }
      Expr::FunctionCall { name, .. }
        if name == "Graphics" || name == "Graphics3D" =>
      {
        arg.clone()
      }
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => arg.clone(),
      _ => evaluate_expr_to_expr(&arg).unwrap_or_else(|_| arg.clone()),
    };
    // The same wrapper can arrive through a *variable* — the Demonstrations
    // idiom `g1 = Graphics[Plot[…]]; Show[g1, …]` — in which case only the
    // evaluated value has the shape, and the arm above never saw it. Render
    // it here as well, or its finished picture would be merged as if it
    // were a drawing primitive and nothing would come out.
    let expr_owned = match &expr_owned {
      Expr::FunctionCall { name, args: gargs }
        if (name == "Graphics" || name == "Graphics3D")
          && !gargs.is_empty()
          && wraps_rendered_graphic(&gargs[0]) =>
      {
        let gargs: Vec<Expr> = gargs.iter().cloned().collect();
        if name == "Graphics" {
          graphics_ast(&gargs).unwrap_or_else(|_| expr_owned.clone())
        } else {
          crate::functions::plot3d::graphics3d_ast(&gargs)
            .unwrap_or_else(|_| expr_owned.clone())
        }
      }
      _ => expr_owned,
    };
    if let Expr::List(items) = &expr_owned {
      let items: Vec<Expr> = items.iter().cloned().collect();
      pending.splice(idx..=idx, items);
      continue;
    }
    idx += 1;
    let expr_ref = &expr_owned;

    match expr_ref {
      Expr::FunctionCall { name, args: gargs } if name == "Graphics" => {
        graphic_count += 1;
        first_graphic_is_plot.get_or_insert(false);
        if !gargs.is_empty() {
          layers.push(Layer::Prims(merged_primitives.len()));
          merged_primitives.push(gargs[0].clone());
        }
        // Wolfram gives the combined graphic the options of the *first*
        // graphic only: an `AxesLabel` that just one of the later layers
        // carries does not label the merged picture.
        if graphic_count == 1 {
          for opt in gargs.iter().skip(1) {
            merge_option(&mut merged_options, opt);
          }
        }
      }
      Expr::FunctionCall { name, args: gargs }
        if name == "MeshRegion" && gargs.len() == 2 =>
      {
        // Convert MeshRegion to Graphics primitives for Show merging
        first_graphic_is_plot.get_or_insert(false);
        if let Some(graphics_prims) =
          mesh_region_to_graphics_prims(&gargs[0], &gargs[1])
        {
          layers.push(Layer::Prims(merged_primitives.len()));
          merged_primitives.push(Expr::List(graphics_prims.into()));
        }
      }
      Expr::FunctionCall { name, args: gargs } if name == "Graphics3D" => {
        graphic_count += 1;
        first_graphic_is_plot.get_or_insert(false);
        is_3d = true;
        if !gargs.is_empty() {
          layers.push(Layer::Prims(merged_primitives.len()));
          merged_primitives.push(gargs[0].clone());
        }
        if graphic_count == 1 {
          for opt in gargs.iter().skip(1) {
            merge_option(&mut merged_options, opt);
          }
        }
      }
      Expr::Graphics {
        is_3d: g_is_3d,
        source,
        structure,
        ..
      } => {
        graphic_count += 1;
        if graphic_count == 1 {
          only_graphic = Some(expr_ref.clone());
        }
        first_graphic_is_plot.get_or_insert(source.is_some());
        is_3d = *g_is_3d;
        // A rendering that kept its symbolic form (a ContourPlot's
        // contour lines, say) merges as those primitives, so it can be
        // drawn together with whatever else `Show` was given instead of
        // being dropped for not being a plot.
        if source.is_none()
          && let Some(structure) = structure
          && let Expr::FunctionCall {
            name: sname,
            args: sargs,
          } = structure.as_ref()
          && (sname == "Graphics" || sname == "Graphics3D")
          && !sargs.is_empty()
        {
          layers.push(Layer::Prims(merged_primitives.len()));
          merged_primitives.push(sargs[0].clone());
          if graphic_count == 1 {
            for opt in sargs.iter().skip(1) {
              merge_option(&mut merged_options, opt);
            }
          }
        } else if let Some(src) = source {
          // Wolfram gives the merged graphic the options of the first
          // graphic it was given; an option given to `Show` itself still
          // overrides them (applied after the walk).
          for opt in src.options.iter().filter(|_| graphic_count == 1) {
            let name = option_name_value(opt).map(|(n, _)| n);
            if name.is_some()
              && !inherited_options.iter().any(|existing| {
                option_name_value(existing).map(|(n, _)| n) == name
              })
            {
              inherited_options.push(opt.clone());
            }
          }
          layers.push(Layer::Source(plot_sources.len()));
          plot_sources.push(src.as_ref().clone());
        } else {
          // No source data — collect as opaque pre-rendered graphic
          rendered_graphics.push(expr_ref.clone());
        }
      }
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => {
        has_own_options = true;
        merge_option(&mut merged_options, expr_ref);
      }
      _ => {}
    }
  }

  // `Show[g]` is `g`: hand back the rendering it already has instead of
  // rebuilding one from the series, which would lose everything the
  // graphic was drawn with.
  if graphic_count == 1
    && !has_own_options
    && let Some(graphic) = &only_graphic
  {
    return Ok(graphic.clone());
  }

  // Options of the shown graphics fill in whatever `Show` was not told
  // explicitly.
  for opt in inherited_options {
    let name = option_name_value(&opt).map(|(n, _)| n);
    if !merged_options
      .iter()
      .any(|existing| option_name_value(existing).map(|(n, _)| n) == name)
    {
      merged_options.push(opt);
    }
  }

  // If we have plot sources (from Plot/ListPlot) and no other Graphics
  // primitives, merge them and re-render via plotters so the output
  // looks identical to standalone plots.
  if !plot_sources.is_empty() && merged_primitives.is_empty() {
    // Merge all series and compute the union of ranges
    let mut all_series = Vec::new();
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut image_size = plot_sources[0].image_size;

    for ps in &plot_sources {
      all_series.extend(ps.series.iter().cloned());
      x_min = x_min.min(ps.x_range.0);
      x_max = x_max.max(ps.x_range.1);
      y_min = y_min.min(ps.y_range.0);
      y_max = y_max.max(ps.y_range.1);
      // Use the largest image size
      if ps.image_size.0 > image_size.0 {
        image_size = ps.image_size;
      }
    }

    // If there are also Graphics[...] primitives, render them as an
    // overlay by converting to plot source entries is not feasible,
    // so we render the plot sources alone for now.
    let merged = crate::syntax::PlotSource {
      series: all_series,
      x_range: (x_min, x_max),
      y_range: (y_min, y_max),
      image_size,
      options: merged_options.clone(),
    };

    let svg = crate::functions::plot::render_merged_plot_source(&merged)?;
    return Ok(crate::graphics_result_with_source(svg, merged));
  }

  // Mixed case: plot sources + Graphics primitives.
  // Convert plot source series to Line/Point primitives so they can be
  // merged with the other Graphics primitives via graphics_ast.
  if !plot_sources.is_empty() {
    // One primitive list per plot source, kept aside so the layers can be
    // stacked back in the order `Show` was given them.
    let mut source_prims: Vec<Expr> = Vec::with_capacity(plot_sources.len());
    for ps in &plot_sources {
      source_prims.push(Expr::List(plot_source_primitives(ps).into()));

      // Deliberately do NOT force a PlotRange from the plot source here: the
      // series are emitted as real Line/Point primitives, so the renderer's
      // automatic range already covers the curve. Forcing the source's tight
      // range would crop any other Graphics primitives that extend beyond it
      // (e.g. a control polygon), whereas Wolfram shows the union of all
      // primitives. Leaving PlotRange unset yields that union.
    }

    // Stack the layers in argument order: a translucent region given last
    // covers the curve given before it, as it does in Wolfram. (Without a
    // complete record of the order — a layer kind that did not register —
    // fall back to primitives first, curves on top.)
    let base_prims = std::mem::take(&mut merged_primitives);
    let ordered_covers_all = layers
      .iter()
      .filter(|l| matches!(l, Layer::Prims(_)))
      .count()
      == base_prims.len()
      && layers
        .iter()
        .filter(|l| matches!(l, Layer::Source(_)))
        .count()
        == source_prims.len();
    if ordered_covers_all {
      for layer in &layers {
        merged_primitives.push(match layer {
          Layer::Prims(i) => base_prims[*i].clone(),
          Layer::Source(i) => source_prims[*i].clone(),
        });
      }
    } else {
      merged_primitives = base_prims;
      merged_primitives.extend(source_prims);
    }

    // Defaults inherited from the plots when the *first* graphic was a plot
    // (Wolfram takes the result's options from the first graphic): axes on,
    // and that plot's own shape. Explicit options passed to Show (already
    // collected in `merged_options`) win — without the AspectRatio default
    // a wide PlotRange collapses the render to a sliver, since the graphics
    // renderer otherwise derives the height from the data aspect. When a
    // raw Graphics comes first, its uniform scaling stays in charge
    // (circles must render round).
    let has_option = |opts: &[Expr], name: &str| {
      opts.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } if option_name(pattern) == Some(name))
      })
    };
    if first_graphic_is_plot == Some(true) {
      // A framed plot draws no interior axes, so the merged graphic must
      // not grow a set of them either: `Show[ParametricPlot[…, Frame ->
      // True], …]` keeps the frame it was given and nothing more.
      let framed = merged_options.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, replacement }
          if option_name(pattern) == Some("Frame")
            && !matches!(replacement.as_ref(),
              Expr::Identifier(s) if s == "False" || s == "None"))
      });
      if !has_option(&merged_options, "Axes") && !framed {
        merged_options.push(Expr::Rule {
          pattern: Box::new(Expr::Identifier("Axes".to_string())),
          replacement: Box::new(bool_expr(true)),
        });
      }
      // An `ImageSize -> {w, h}` already fixes the height, so the plot
      // aspect must not be filled in over it.
      let sized_both_ways = merged_options.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, replacement }
          if option_name(pattern) == Some("ImageSize")
            && matches!(replacement.as_ref(), Expr::List(v) if v.len() == 2))
      });
      if !has_option(&merged_options, "AspectRatio") && !sized_both_ways {
        // The shape the leading plot drew itself in. `Plot`/`ListPlot`
        // default to 1/GoldenRatio, but `ParametricPlot` and friends
        // default to `AspectRatio -> Automatic` and size themselves from
        // the data, so a circle stays a circle once `Show` layers other
        // graphics on top of one.
        let aspect = plot_sources
          .first()
          .map(|ps| ps.image_size.1 as f64 / ps.image_size.0 as f64)
          .filter(|r| r.is_finite() && *r > 0.0)
          .unwrap_or(1.0 / 1.618_033_988_749_895);
        merged_options.push(Expr::Rule {
          pattern: Box::new(Expr::Identifier("AspectRatio".to_string())),
          replacement: Box::new(Expr::Real(aspect)),
        });
      }
    }
  }

  // If we have pre-rendered Graphics but no primitives from Graphics[...],
  // return the rendered result directly. Single-arg Show just passes through.
  if merged_primitives.is_empty() && !rendered_graphics.is_empty() {
    return Ok(rendered_graphics[0].clone());
  }

  if merged_primitives.is_empty() {
    return Ok(unevaluated("Show", args));
  }

  let content = Expr::List(merged_primitives.into());
  let mut graphics_args = vec![content];
  graphics_args.extend(merged_options);

  if is_3d {
    crate::functions::plot3d::graphics3d_ast(&graphics_args)
  } else {
    graphics_ast(&graphics_args)
  }
}

pub fn grid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  grid_ast_with_gaps(args, &[])
}

/// Render a grid with default styles inherited from an outer Style wrapper.
pub fn grid_ast_styled(
  args: &[Expr],
  style: &GridStyle,
) -> Result<Expr, InterpreterError> {
  let svg = grid_svg_styled_internal(args, &[], false, style)?;
  Ok(crate::graphics_result(svg))
}

/// Render a grid enclosed with large parentheses (for MatrixForm).
pub fn grid_ast_with_parens(args: &[Expr]) -> Result<Expr, InterpreterError> {
  grid_ast_internal(args, &[], true)
}

/// Render a parenthesized grid (MatrixForm) with an inherited outer Style.
pub fn grid_ast_styled_with_parens(
  args: &[Expr],
  style: &GridStyle,
) -> Result<Expr, InterpreterError> {
  let svg = grid_svg_styled_internal(args, &[], true, style)?;
  Ok(crate::graphics_result(svg))
}

/// Render a grid and return the raw SVG string.
pub fn grid_svg_with_gaps(
  args: &[Expr],
  group_gaps: &[usize],
) -> Result<String, InterpreterError> {
  grid_svg_internal(args, group_gaps, false)
}

/// Render a grid with optional extra vertical gaps before certain rows.
/// `group_gaps` lists row indices that should have extra spacing before them.
pub fn grid_ast_with_gaps(
  args: &[Expr],
  group_gaps: &[usize],
) -> Result<Expr, InterpreterError> {
  grid_ast_internal(args, group_gaps, false)
}

fn grid_ast_internal(
  args: &[Expr],
  group_gaps: &[usize],
  parens: bool,
) -> Result<Expr, InterpreterError> {
  // A grid whose cells are all graphics is a layout of pictures, not of
  // text: lay the renderings out side by side at their own sizes, the
  // way `GraphicsGrid` does, instead of printing `-Graphics-` per cell.
  if !parens
    && let Some(rows) = grid_of_graphics_svgs(args)
    && let Some(combined) = combine_graphics_svgs(&rows)
  {
    crate::clear_captured_graphics();
    return Ok(crate::graphics_result(combined));
  }
  let svg = grid_svg_internal(args, group_gaps, parens)?;
  Ok(crate::graphics_result(svg))
}

/// The rendered SVG of every cell, when `Grid`'s first argument is a
/// matrix in which *every* cell is a rendered graphic. `None` as soon as
/// one cell is anything else, so mixed text/graphics grids keep the
/// ordinary text layout.
fn grid_of_graphics_svgs(args: &[Expr]) -> Option<Vec<Vec<String>>> {
  let Expr::List(rows) = args.first()? else {
    return None;
  };
  let mut out: Vec<Vec<String>> = Vec::with_capacity(rows.len());
  for row in rows {
    let cells = match row {
      Expr::List(cells) => cells.iter().cloned().collect::<Vec<_>>(),
      single => vec![single.clone()],
    };
    let mut row_svgs = Vec::with_capacity(cells.len());
    for cell in &cells {
      let svg = match cell {
        Expr::Graphics { svg, .. } => svg.clone(),
        // A cell may still be the unevaluated call that produces the
        // picture (`Grid` holds its argument), so evaluate the heads
        // that are known to render.
        Expr::FunctionCall { name, .. } if is_graphics_producing_head(name) => {
          let evaluated = evaluate_expr_to_expr(cell).ok()?;
          crate::evaluator::expr_to_svg(&evaluated)
        }
        // A display wrapper that resolves to a picture (`Labeled[…]`,
        // `Pane[…]`, `LocatorPane[…]`, `Dynamic[…]`) is drawn through the
        // same path, which unwraps it. Without this the cell printed as
        // source.
        Expr::FunctionCall { .. }
          if crate::evaluator::lays_out_a_graphic(cell) =>
        {
          crate::evaluator::expr_to_svg(cell)
        }
        _ => return None,
      };
      if svg.is_empty() {
        return None;
      }
      row_svgs.push(svg);
    }
    if row_svgs.is_empty() {
      return None;
    }
    out.push(row_svgs);
  }
  (!out.is_empty()).then_some(out)
}

/// Default style inherited from an outer Style[Grid[...], directives...].
#[derive(Clone, Default)]
pub struct GridStyle {
  pub font_weight: Option<&'static str>,
  pub font_style: Option<&'static str>,
  pub font_size: Option<f64>,
  /// The font family a `FontFamily -> "…"` directive asks for.
  pub font_family: Option<String>,
  pub(crate) color: Option<Color>,
}

/// `SpanFromLeft` in a `Grid` cell means "the cell to my left continues into
/// this column": it draws nothing itself, and its neighbour is laid out
/// across the merged span. `SpanFromBoth` continues a cell that spans both
/// ways, so it is a left continuation too.
fn is_span_from_left(cell: &Expr) -> bool {
  matches!(cell, Expr::Identifier(s) if s == "SpanFromLeft" || s == "SpanFromBoth")
}

/// `SpanFromAbove` means "the cell above me continues into this row" — the
/// vertical counterpart of `SpanFromLeft`, with `SpanFromBoth` marking the
/// inside of a block that is merged in both directions.
fn is_span_from_above(cell: &Expr) -> bool {
  matches!(cell, Expr::Identifier(s) if s == "SpanFromAbove" || s == "SpanFromBoth")
}

/// A cell that only continues a merged one, in either direction: it holds no
/// content of its own and so is skipped by every sizing and drawing pass.
fn is_span_placeholder(cell: &Expr) -> bool {
  is_span_from_left(cell) || is_span_from_above(cell)
}

/// How many columns the cell at `j` covers — itself plus every
/// `SpanFromLeft` that follows it.
fn span_width(row: &[Expr], j: usize) -> usize {
  1 + row[j + 1..]
    .iter()
    .take_while(|c| is_span_from_left(c))
    .count()
}

/// How many rows the cell at `(i, j)` covers — itself plus every row below
/// whose cell in that column continues it.
fn span_height(rows: &[Vec<Expr>], i: usize, j: usize) -> usize {
  1 + rows[i + 1..]
    .iter()
    .take_while(|row| row.get(j).is_some_and(is_span_from_above))
    .count()
}

/// `StyleForm` is the Wolfram Language's older spelling of `Style`; the
/// front end renders the two identically (both stay unevaluated and are
/// interpreted at display time), so every display path treats them alike.
pub(crate) fn is_style_wrapper(name: &str) -> bool {
  name == "Style" || name == "StyleForm"
}

/// Heads that a `Manipulate` argument may carry while still being a static
/// annotation row rather than a variable specification: a styled or plain
/// text label sitting between the controls, which Wolfram tags
/// ``Manipulate`Dump`ThisIsNotAControl`` instead of reporting as malformed.
/// `Row`/`Column` cover the Demonstrations idiom for a captioned section
/// header (`Row[{Style["assets", Bold], " in ($K)"}]`) — a bare `Row`/
/// `Column` reaches here only once any actual controls it grouped have
/// already been flattened out by `control_group_items`, so what is left is
/// always plain text layout, never a control panel.
pub(crate) fn is_manipulate_annotation_head(name: &str) -> bool {
  is_style_wrapper(name) || matches!(name, "Text" | "Row" | "Column")
}

/// Whether an annotation row (`Style[…]`, `Row[{…}]`, `Column[{…}]`, …) has a
/// `Dynamic[…]` anywhere inside it — a Demonstration's live step counter is
/// often written `Row[{Style["moves: "], Dynamic[moves]}]` right in the
/// control-panel argument list. Such a row is not "plain text layout" (see
/// [`is_manipulate_annotation_head`]): the `Dynamic` part must track its
/// variable and update every frame, so it needs the same live "display"
/// treatment as a bare `Dynamic[…]` argument rather than being frozen into a
/// static `Heading` at parse time.
fn annotation_contains_dynamic(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      name == "Dynamic" || args.iter().any(annotation_contains_dynamic)
    }
    Expr::List(items) => items.iter().any(annotation_contains_dynamic),
    _ => false,
  }
}

/// Peel a display-only `Invisible[expr]` wrapper, returning the content it
/// hides. `Invisible` keeps exactly the space `expr` would take but paints
/// nothing — Demonstrations use it to hold a layout's shape steady while an
/// item is toggled off (`If[show, Identity, Invisible] @ …`), so a renderer
/// measures the content as usual and only suppresses its fill.
pub(crate) fn peel_invisible(expr: &Expr) -> Option<&Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Invisible" && args.len() == 1 =>
    {
      Some(&args[0])
    }
    _ => None,
  }
}

/// The appearance a `Style[content, directives…]` cell asks for, as used by
/// the `Grid`/`Column`/`Row` layout renderers.
pub(crate) struct CellTextStyle<'a> {
  /// The styled content, with the `Style`/`Invisible` wrappers peeled off.
  pub content: &'a Expr,
  pub font_size: Option<f64>,
  pub font_weight: &'static str,
  pub font_style: &'static str,
  /// The font family a `FontFamily -> "…"` directive asks for.
  pub font_family: Option<String>,
  pub color: Option<Color>,
  /// Set by an `Invisible[…]` wrapper: the cell is laid out but not painted.
  pub hidden: bool,
}

/// Extract style info from a Style[content, directives...] cell.
fn extract_cell_style(cell: &Expr) -> CellTextStyle<'_> {
  // `Invisible[Style[…]]` — the wrapper outside the styling.
  if let Some(inner) = peel_invisible(cell) {
    return CellTextStyle {
      hidden: true,
      ..extract_cell_style(inner)
    };
  }
  if let Expr::FunctionCall { name, args } = cell
    && is_style_wrapper(name)
    && !args.is_empty()
  {
    // `Style[Invisible[…], …]` — the styling outside the wrapper. Either
    // nesting order hides the cell while keeping its footprint.
    let (content, invisible) = match peel_invisible(&args[0]) {
      Some(inner) => (inner, true),
      None => (&args[0], false),
    };
    let mut fs: Option<f64> = None;
    let mut fw = "normal";
    let mut fst = "normal";
    let mut ff: Option<String> = None;
    let mut color: Option<Color> = None;
    for directive in &args[1..] {
      match directive {
        Expr::Identifier(s) if s == "Bold" => fw = "bold",
        Expr::Identifier(s) if s == "Italic" => fst = "italic",
        Expr::Integer(n) => fs = Some(*n as f64),
        Expr::Real(f) => fs = Some(*f),
        // The long option spellings, which `StyleForm` cells are written
        // with: `FontWeight -> Bold`, `FontSlant -> Italic`,
        // `FontColor -> GrayLevel[1]`, …
        Expr::Rule {
          pattern,
          replacement,
        } => {
          let Expr::Identifier(k) = pattern.as_ref() else {
            continue;
          };
          match k.as_str() {
            "FontSize" => match replacement.as_ref() {
              Expr::Integer(n) => fs = Some(*n as f64),
              Expr::Real(f) => fs = Some(*f),
              _ => {}
            },
            "FontWeight" => {
              if matches!(replacement.as_ref(),
                Expr::Identifier(v) | Expr::String(v) if v == "Bold")
              {
                fw = "bold";
              }
            }
            "FontSlant" => {
              if matches!(replacement.as_ref(),
                Expr::Identifier(v) | Expr::String(v) if v == "Italic")
              {
                fst = "italic";
              }
            }
            "FontFamily" => {
              if let Expr::String(v) | Expr::Identifier(v) =
                replacement.as_ref()
              {
                ff = Some(v.clone());
              }
            }
            "FontColor" => {
              if let Some(c) = parse_color(replacement) {
                color = Some(c);
              }
            }
            _ => {}
          }
        }
        _ => {
          if let Some(c) = parse_color(directive) {
            color = Some(c);
          }
        }
      }
    }
    return CellTextStyle {
      content,
      font_size: fs,
      font_weight: fw,
      font_style: fst,
      font_family: ff,
      color,
      hidden: invisible,
    };
  }
  CellTextStyle {
    content: cell,
    font_size: None,
    font_weight: "normal",
    font_style: "normal",
    font_family: None,
    color: None,
    hidden: false,
  }
}

/// Parse Style directives into a GridStyle.
pub fn parse_grid_style(directives: &[Expr]) -> GridStyle {
  let mut gs = GridStyle::default();
  for d in directives {
    match d {
      Expr::Identifier(s) if s == "Bold" => gs.font_weight = Some("bold"),
      Expr::Identifier(s) if s == "Italic" => gs.font_style = Some("italic"),
      // Named font sizes (relative to the ~12 pt default cell text).
      Expr::Identifier(s) if s == "Tiny" => gs.font_size = Some(7.0),
      Expr::Identifier(s) if s == "Small" => gs.font_size = Some(9.0),
      Expr::Identifier(s) if s == "Medium" => gs.font_size = Some(12.0),
      Expr::Identifier(s) if s == "Large" => gs.font_size = Some(18.0),
      Expr::Identifier(s) if s == "Huge" => gs.font_size = Some(26.0),
      Expr::Integer(n) => gs.font_size = Some(*n as f64),
      Expr::Real(f) => gs.font_size = Some(*f),
      Expr::Rule {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(k) = pattern.as_ref() {
          match k.as_str() {
            "FontSize" => match replacement.as_ref() {
              Expr::Integer(n) => gs.font_size = Some(*n as f64),
              Expr::Real(f) => gs.font_size = Some(*f),
              _ => {}
            },
            "FontFamily" => {
              if let Expr::String(f) | Expr::Identifier(f) =
                replacement.as_ref()
              {
                gs.font_family = Some(f.clone());
              }
            }
            _ => {}
          }
        }
      }
      // Directives grouped in a list, e.g. Style[expr, {Large, Bold, Orange}].
      Expr::List(items) => {
        let inner = parse_grid_style(items);
        if inner.font_weight.is_some() {
          gs.font_weight = inner.font_weight;
        }
        if inner.font_style.is_some() {
          gs.font_style = inner.font_style;
        }
        if inner.font_size.is_some() {
          gs.font_size = inner.font_size;
        }
        if inner.font_family.is_some() {
          gs.font_family = inner.font_family;
        }
        if inner.color.is_some() {
          gs.color = inner.color;
        }
      }
      _ => {
        if let Some(c) = parse_color(d) {
          gs.color = Some(c);
        }
      }
    }
  }
  gs
}

/// The rendered SVG of a grid cell that is a graphic, and its natural
/// size. `Grid` lays such a cell out as a picture instead of printing the
/// expression's text form.
fn grid_cell_graphic(cell: &Expr) -> Option<(String, f64, f64)> {
  let svg = match cell {
    Expr::Graphics { svg, .. } => svg.clone(),
    // A styled graphic is still a graphic.
    Expr::FunctionCall { name, args }
      if is_style_wrapper(name) && !args.is_empty() =>
    {
      return grid_cell_graphic(&args[0]);
    }
    // A cell may be the unevaluated call that draws the picture: only the
    // rendering path turns `Graphics[…]` into a rendered graphic, so ask
    // for its SVG directly.
    Expr::FunctionCall { name, .. } if is_graphics_producing_head(name) => {
      crate::evaluator::expr_to_svg(cell)
    }
    // A sound cell shows the sound box a notebook draws for it — a play
    // button and the waveform — rather than the `Play[…]` source text.
    Expr::FunctionCall { name, .. } if name == "Sound" || name == "Play" => {
      crate::functions::sound::sound_svg(cell)?
    }
    // As above, a display wrapper that resolves to a picture is drawn
    // rather than printed as source.
    Expr::FunctionCall { .. } if crate::evaluator::lays_out_a_graphic(cell) => {
      crate::evaluator::expr_to_svg(cell)
    }
    // A cell may itself be a block layout, which the text pass cannot
    // set: lay it out on its own and place the result as a picture.
    Expr::FunctionCall { name, args } => {
      let args: Vec<Expr> = args.iter().cloned().collect();
      match name.as_str() {
        "Grid" if !args.is_empty() => grid_svg_with_gaps(&args, &[]).ok()?,
        "Column" if !args.is_empty() => column_to_svg(&args)?,
        _ => return None,
      }
    }
    _ => return None,
  };
  if svg.is_empty() {
    return None;
  }
  let parsed = parse_svg_dimensions(&svg)?;
  Some((svg, parsed.nat_w, parsed.nat_h))
}

/// The per-position half of a `Spacings` spec: `{i1 -> s1, i2 -> s2, …}`
/// names the gap at individual positions, while a plain `{s1, s2, …}` gives
/// them in order from the first. The result is indexed by position - 1, with
/// `None` wherever the spec says nothing and the default gap applies.
fn parse_position_spacings(spec: &Expr) -> Vec<Option<f64>> {
  let Expr::List(items) = spec else {
    return Vec::new();
  };
  let value = |e: &Expr| match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(f) => Some(*f),
    _ => None,
  };
  let mut out: Vec<Option<f64>> = Vec::new();
  let mut put = |index: usize, v: f64| {
    if out.len() <= index {
      out.resize(index + 1, None);
    }
    out[index] = Some(v);
  };
  for (i, item) in items.iter().enumerate() {
    match item {
      Expr::Rule {
        pattern,
        replacement,
      } => {
        // Positions are 1-based, and a spec may name them out of order.
        if let (Expr::Integer(pos), Some(v)) =
          (pattern.as_ref(), value(replacement))
          && *pos >= 1
        {
          put(*pos as usize - 1, v);
        }
      }
      other => {
        if let Some(v) = value(other) {
          put(i, v);
        }
      }
    }
  }
  out
}

/// Check if a cell is or contains an Expr::Image (unwrapping Style).
fn unwrap_to_image(cell: &Expr) -> Option<&Expr> {
  match cell {
    Expr::Image { .. } => Some(cell),
    Expr::FunctionCall { name, args }
      if is_style_wrapper(name) && !args.is_empty() =>
    {
      unwrap_to_image(&args[0])
    }
    _ => None,
  }
}

/// Convert a WL alignment identifier to SVG text-anchor value.
/// The `"."` string maps to the pseudo-anchor `"decimal"`, handled specially
/// so numbers in the column line up on their decimal point.
fn alignment_to_anchor(expr: &Expr) -> &'static str {
  match expr {
    Expr::Identifier(val) => match val.as_str() {
      "Left" => "start",
      "Right" => "end",
      _ => "middle",
    },
    Expr::String(s) if s == "." => "decimal",
    _ => "middle",
  }
}

/// Split a plain (unmarked-up) cell string into its integer-part and
/// fractional-part widths, in character units, for decimal-point alignment.
/// The integer part includes any sign and integer digits; the fractional part
/// includes the decimal point and following digits. A string without a decimal
/// point has zero fractional width (it aligns with the dot at its right edge).
fn split_decimal_str(s: &str) -> (f64, f64) {
  match s.find('.') {
    Some(pos) => (
      s[..pos].chars().count() as f64,
      s[pos..].chars().count() as f64,
    ),
    None => (s.chars().count() as f64, 0.0),
  }
}

/// Compute the (integer-width, fractional-width) split for a decimal-aligned
/// cell, or `None` when the cell doesn't render as a plain number (e.g. a
/// fraction or scientific-notation form containing SVG markup), in which case
/// the caller falls back to centering.
fn decimal_split_width(cell: &Expr) -> Option<(f64, f64)> {
  let s = expr_to_svg_markup(cell);
  if s.contains('<') {
    return None;
  }
  Some(split_decimal_str(&s))
}

/// Parse a divider entry: a color expression means "draw with this color",
/// False/None means "don't draw".
fn parse_divider_entry(expr: &Expr) -> Option<Color> {
  match expr {
    Expr::Identifier(n) if n == "False" || n == "None" => None,
    Expr::FunctionCall { name, .. } if name == "False" || name == "None" => {
      None
    }
    Expr::Identifier(n) if n == "True" || n == "All" => {
      // True means draw with default color — use a sentinel black
      Some(Color::new(0.0, 0.0, 0.0))
    }
    _ => parse_color(expr),
  }
}

/// Parse a color from a Background list entry, treating "None" as None.
fn parse_bg_color(expr: &Expr) -> Option<Color> {
  if let Expr::Identifier(n) = expr
    && n == "None"
  {
    return None;
  }
  parse_color(expr)
}

fn grid_svg_internal(
  args: &[Expr],
  group_gaps: &[usize],
  parens: bool,
) -> Result<String, InterpreterError> {
  grid_svg_styled_internal(args, group_gaps, parens, &GridStyle::default())
}

fn grid_svg_styled_internal(
  args: &[Expr],
  group_gaps: &[usize],
  parens: bool,
  default_style: &GridStyle,
) -> Result<String, InterpreterError> {
  // Extract rows from args[0]
  let data = evaluate_expr_to_expr(&args[0])?;
  let mut rows: Vec<Vec<Expr>> = match &data {
    Expr::List(items) => {
      // Check if it's a list of lists (matrix) or a flat list (single row)
      if items.iter().all(|item| matches!(item, Expr::List(_))) {
        items
          .iter()
          .map(|row| {
            if let Expr::List(cells) = row {
              cells.to_vec()
            } else {
              vec![row.clone()]
            }
          })
          .collect()
      } else {
        // 1D list → single row
        vec![items.to_vec()]
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Grid: argument must be a list".into(),
      ));
    }
  };

  // A cell may arrive inside a wrapper that only says how to set it —
  // `Pane[…]` reserving an area, an `Item[…]` cell, `Text[…]` choosing a
  // font. What the cell *shows* is what they hold, so peel them before
  // any sizing or drawing pass: otherwise the cell printed as source
  // (`Pane[Grid[…]]`), which is how a Demonstration's readout panel used
  // to render.
  for row in &mut rows {
    for cell in row.iter_mut() {
      *cell = unwrap_display_wrappers(cell);
    }
  }

  // Parse options from remaining args
  let mut frame_outer = false; // Frame -> True: outer border only
  let mut frame_all = false; // Frame -> All: all gridlines
  let mut frame_color: Option<Color> = None; // custom frame color
  let mut row_headings: Vec<Expr> = Vec::new();
  let mut col_headings: Vec<Expr> = Vec::new();
  let mut spacings_h: Option<f64> = None; // horizontal spacing override
  let mut spacings_v: Option<f64> = None; // vertical spacing override
  // `Spacings -> {{i -> s, …}, …}` sets the gap at individual column
  // positions: position `i` is the gap to the left of column `i`, and
  // position `ncols + 1` the margin after the last column. Positions left
  // out keep the default gap.
  let mut col_spacings: Vec<Option<f64>> = Vec::new();
  // TableForm's TableSpacing maps directly to pixels (see the option below).
  let mut table_pad_x: Option<f64> = None; // per-cell horizontal padding (px)
  let mut table_row_gap: Option<f64> = None; // gap between rows (px)
  let mut dividers_col = false; // vertical divider lines between columns
  let mut dividers_row = false; // horizontal divider lines between rows
  // `Dividers -> All` rules *every* position, the outer edges included, so
  // a grid drawn that way is a closed box; `Dividers -> Center` rules only
  // the boundaries between cells. These flags record which of the two was
  // asked for, per direction.
  let mut dividers_col_border = false; // left/right edges from `All`
  let mut dividers_row_border = false; // top/bottom edges from `All`
  // Per-position divider specs: Some(color) = draw with color, None = don't draw
  // These use the same repeating-list pattern as backgrounds
  let mut col_dividers: Vec<Option<Color>> = Vec::new(); // vertical lines (ncols+1 positions)
  let mut row_dividers: Vec<Option<Color>> = Vec::new(); // horizontal lines (nrows+1 positions)
  let mut col_div_explicit_start: Vec<Option<Color>> = Vec::new();
  let mut col_div_repeating: Vec<Option<Color>> = Vec::new();
  let mut col_div_explicit_end: Vec<Option<Color>> = Vec::new();
  let mut col_div_has_repeating = false;
  let mut row_div_explicit_start: Vec<Option<Color>> = Vec::new();
  let mut row_div_repeating: Vec<Option<Color>> = Vec::new();
  let mut row_div_explicit_end: Vec<Option<Color>> = Vec::new();
  let mut row_div_has_repeating = false;
  let mut background_color: Option<Color> = None; // uniform background
  let mut col_backgrounds: Vec<Option<Color>> = Vec::new(); // per-column bg
  let mut row_backgrounds: Vec<Option<Color>> = Vec::new(); // per-row bg
  // For WL repeating-list patterns like {first, {repeat1, repeat2}, last}
  let mut row_bg_explicit_start: Vec<Option<Color>> = Vec::new();
  let mut row_bg_repeating: Vec<Option<Color>> = Vec::new();
  let mut row_bg_explicit_end: Vec<Option<Color>> = Vec::new();
  let mut row_bg_has_repeating = false;
  let mut alignment_h: &str = "middle"; // SVG text-anchor value (default)
  let mut col_alignments: Vec<&str> = Vec::new(); // per-column alignments
  let mut col_align_explicit_start: Vec<&str> = Vec::new();
  let mut col_align_repeating: Vec<&str> = Vec::new();
  let mut col_align_has_repeating = false;
  for raw_opt in &args[1..] {
    let opt =
      evaluate_expr_to_expr(raw_opt).unwrap_or_else(|_| raw_opt.clone());
    if let Expr::Rule {
      pattern,
      replacement,
    } = &opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "Frame" => match replacement.as_ref() {
          Expr::Identifier(val) if val == "All" => frame_all = true,
          Expr::Identifier(val) if val == "True" => frame_outer = true,
          Expr::FunctionCall { name: fn_name, .. } if fn_name == "True" => {
            frame_outer = true;
          }
          expr => {
            // A color expression means Frame -> True with that color
            if let Some(color) = parse_color(expr) {
              frame_outer = true;
              frame_color = Some(color);
            }
          }
        },
        "Dividers" => match replacement.as_ref() {
          Expr::Identifier(val) if val == "All" || val == "True" => {
            dividers_col = true;
            dividers_row = true;
            dividers_col_border = true;
            dividers_row_border = true;
          }
          Expr::Identifier(val) if val == "Center" => {
            dividers_col = true;
            dividers_row = true;
          }
          Expr::List(items) => {
            // Dividers -> {col_spec, row_spec}
            // Each spec can be: True/All, or a list with optional repeating pattern
            for (idx, spec) in items.iter().enumerate() {
              match spec {
                Expr::Identifier(v) if v == "All" || v == "True" => {
                  if idx == 0 {
                    dividers_col = true;
                    dividers_col_border = true;
                  } else {
                    dividers_row = true;
                    dividers_row_border = true;
                  }
                }
                // `Center` rules the boundaries between cells only, so it
                // leaves the outer edges of the grid open.
                Expr::Identifier(v) if v == "Center" => {
                  if idx == 0 {
                    dividers_col = true;
                  } else {
                    dividers_row = true;
                  }
                }
                Expr::List(positions) => {
                  // Per-position spec with optional repeating pattern
                  let has_nested =
                    positions.iter().any(|c| matches!(c, Expr::List(_)));
                  let (
                    target_dividers,
                    explicit_start,
                    repeating,
                    explicit_end,
                    has_rep_flag,
                  ) = if idx == 0 {
                    (
                      &mut col_dividers,
                      &mut col_div_explicit_start,
                      &mut col_div_repeating,
                      &mut col_div_explicit_end,
                      &mut col_div_has_repeating,
                    )
                  } else {
                    (
                      &mut row_dividers,
                      &mut row_div_explicit_start,
                      &mut row_div_repeating,
                      &mut row_div_explicit_end,
                      &mut row_div_has_repeating,
                    )
                  };
                  if has_nested {
                    *has_rep_flag = true;
                    let mut before_repeat = true;
                    for p in positions {
                      if let Expr::List(rep_items) = p {
                        before_repeat = false;
                        *repeating =
                          rep_items.iter().map(parse_divider_entry).collect();
                      } else if before_repeat {
                        explicit_start.push(parse_divider_entry(p));
                      } else {
                        explicit_end.push(parse_divider_entry(p));
                      }
                    }
                  } else {
                    *target_dividers =
                      positions.iter().map(parse_divider_entry).collect();
                  }
                }
                _ => {}
              }
            }
          }
          _ => {}
        },
        "Background" => match replacement.as_ref() {
          expr if parse_color(expr).is_some() => {
            background_color = parse_color(expr);
          }
          Expr::List(items) => {
            // Background -> {{col_colors...}, {row_colors...}}
            if !items.is_empty() {
              if let Expr::List(cols) = &items[0] {
                col_backgrounds = cols.iter().map(parse_bg_color).collect();
              } else if parse_color(&items[0]).is_some() {
                // Background -> {color} (single color in list)
                background_color = parse_color(&items[0]);
              }
            }
            if items.len() >= 2
              && let Expr::List(row_cols) = &items[1]
            {
              // Check for repeating-list pattern: {first..., {repeat...}, last...}
              let has_nested =
                row_cols.iter().any(|c| matches!(c, Expr::List(_)));
              if has_nested {
                row_bg_has_repeating = true;
                let mut before_repeat = true;
                for c in row_cols {
                  if let Expr::List(repeat_items) = c {
                    before_repeat = false;
                    row_bg_repeating =
                      repeat_items.iter().map(parse_bg_color).collect();
                  } else if before_repeat {
                    row_bg_explicit_start.push(parse_bg_color(c));
                  } else {
                    row_bg_explicit_end.push(parse_bg_color(c));
                  }
                }
              } else {
                row_backgrounds = row_cols.iter().map(parse_bg_color).collect();
              }
            }
          }
          _ => {}
        },
        "Alignment" => match replacement.as_ref() {
          Expr::String(s) if s == "." => alignment_h = "decimal",
          Expr::Identifier(val) => match val.as_str() {
            "Left" => alignment_h = "start",
            "Right" => alignment_h = "end",
            _ => alignment_h = "middle",
          },
          Expr::List(items) => {
            // Alignment -> {col_spec} or Alignment -> {col_spec, row_spec}
            // col_spec can be: Left, {Left, Right, {Left}}, etc.
            if let Some(first) = items.first() {
              match first {
                Expr::String(s) if s == "." => alignment_h = "decimal",
                Expr::Identifier(val) => match val.as_str() {
                  "Left" => alignment_h = "start",
                  "Right" => alignment_h = "end",
                  _ => alignment_h = "middle",
                },
                Expr::List(col_specs) => {
                  // Per-column alignment with optional repeating pattern
                  let has_nested =
                    col_specs.iter().any(|c| matches!(c, Expr::List(_)));
                  if has_nested {
                    col_align_has_repeating = true;
                    let mut before_repeat = true;
                    for spec in col_specs {
                      if let Expr::List(rep_items) = spec {
                        before_repeat = false;
                        col_align_repeating =
                          rep_items.iter().map(alignment_to_anchor).collect();
                      } else if before_repeat {
                        col_align_explicit_start
                          .push(alignment_to_anchor(spec));
                      }
                      // Note: trailing explicit not common for alignment
                    }
                  } else {
                    col_alignments =
                      col_specs.iter().map(alignment_to_anchor).collect();
                  }
                }
                _ => {}
              }
            }
          }
          _ => {}
        },
        "Spacings" => {
          // Spacings -> {h, v} or Spacings -> n
          match replacement.as_ref() {
            Expr::Integer(n) => {
              spacings_h = Some(*n as f64);
              spacings_v = Some(*n as f64);
            }
            Expr::Real(f) => {
              spacings_h = Some(*f);
              spacings_v = Some(*f);
            }
            Expr::List(items) => {
              if !items.is_empty() {
                match &items[0] {
                  Expr::Integer(n) => spacings_h = Some(*n as f64),
                  Expr::Real(f) => spacings_h = Some(*f),
                  // A list gives the gap at each position individually.
                  Expr::List(_) => {
                    col_spacings = parse_position_spacings(&items[0]);
                  }
                  _ => {}
                }
              }
              if items.len() >= 2 {
                match &items[1] {
                  Expr::Integer(n) => spacings_v = Some(*n as f64),
                  Expr::Real(f) => spacings_v = Some(*f),
                  _ => {}
                }
              }
            }
            _ => {}
          }
        }
        "TableHeadings" => match replacement.as_ref() {
          // TableHeadings -> {{row_h...}, {col_h...}} (either may be None)
          Expr::List(lists) => {
            if !lists.is_empty()
              && let Expr::List(rh) = &lists[0]
            {
              row_headings = rh.to_vec();
            }
            if lists.len() >= 2
              && let Expr::List(ch) = &lists[1]
            {
              col_headings = ch.to_vec();
            }
          }
          // TableHeadings -> Automatic: label rows and columns with their
          // 1-based indices.
          Expr::Identifier(v) if v == "Automatic" => {
            let n_rows = rows.len();
            let n_cols = rows.iter().map(std::vec::Vec::len).max().unwrap_or(0);
            row_headings =
              (1..=n_rows).map(|i| Expr::Integer(i as i128)).collect();
            col_headings =
              (1..=n_cols).map(|i| Expr::Integer(i as i128)).collect();
          }
          _ => {}
        },
        "TableSpacing" => {
          // TableForm's TableSpacing -> {rows, cols}: the first value is the
          // gap between rows, the second the gap between columns (the opposite
          // order from Grid's Spacings -> {h, v}). Its units differ from
          // Grid's ems, so map them to pixels directly rather than routing
          // through spacings_h / spacings_v. The default column spacing is 3
          // (→ the standard 12 px cell padding), and rows touch at spacing 1.
          let num = |e: &Expr| -> Option<f64> {
            match e {
              Expr::Integer(n) => Some(*n as f64),
              Expr::Real(f) => Some(*f),
              _ => None,
            }
          };
          if let Expr::List(items) = replacement.as_ref() {
            if let Some(r) = items.first().and_then(num) {
              table_row_gap = Some((r - 1.0).max(0.0) * 3.5);
            }
            if let Some(c) = items.get(1).and_then(num) {
              table_pad_x = Some(c * 4.0);
            }
          }
        }
        _ => {}
      }
    }
  }

  // Inject TableHeadings into the grid data. Wolfram renders the headings in
  // the same (plain) style as the body cells, separated from it by a thin
  // rule — not bold. Track whether a heading row/column was added so those
  // separator lines can be drawn later.
  let mut has_col_heading_row = false;
  let mut has_row_heading_col = false;
  if !col_headings.is_empty() {
    // Add column headings as the first row
    let mut heading_row: Vec<Expr> = col_headings;
    if !row_headings.is_empty() {
      // Insert empty top-left corner cell
      heading_row.insert(0, Expr::Identifier(String::new()));
    }
    rows.insert(0, heading_row);
    has_col_heading_row = true;
  }
  if !row_headings.is_empty() {
    // Add row headings as the first column
    let start = usize::from(has_col_heading_row);
    for (i, row) in rows.iter_mut().enumerate() {
      if i >= start {
        let idx = i - start;
        if let Some(h) = row_headings.get(idx) {
          row.insert(0, h.clone());
        } else {
          row.insert(0, Expr::Identifier(String::new()));
        }
      }
    }
    has_row_heading_col = true;
  }

  // Convert cells to text
  let num_rows = rows.len();
  let num_cols = rows.iter().map(std::vec::Vec::len).max().unwrap_or(0);
  if num_cols == 0 {
    return Err(InterpreterError::EvaluationError("Grid: empty data".into()));
  }

  // Compute column widths based on estimated display width
  let char_width: f64 = 8.4; // approximate monospace char width at font-size 14
  let font_size: f64 = 14.0;
  // Apply Spacings option: values are in ems (multiples of char_width / font_size)
  let pad_x: f64 = match (table_pad_x, spacings_h) {
    (Some(px), _) => px, // TableSpacing → pixels directly
    (None, Some(h)) => h * char_width, // Spacings h in ems → pixel padding
    (None, None) => 12.0, // default horizontal padding per cell
  };
  let pad_y: f64 = 2.0; // vertical padding per cell (each side = 1)
  let row_gap: f64 = match (table_row_gap, spacings_v) {
    (Some(g), _) => g, // TableSpacing → pixels directly
    (None, Some(v)) => v * font_size, // Spacings v in ems → pixel gap
    (None, None) => 0.0, // default: no extra row gap
  };
  let group_gap: f64 = 6.0; // extra spacing between groups
  let base_row_height = font_size + pad_y;
  let frac_row_height = font_size + pad_y + 10.0; // taller for stacked fractions

  // The padding a column carries on each side. A gap between two columns is
  // one gap, so it is shared half-and-half by the columns it separates,
  // while the gaps at the two ends belong entirely to the first and last
  // column. Positions the spec leaves out fall back to the uniform layout:
  // `pad_x` between columns and `pad_x / 2` at each end, i.e. `pad_x` per
  // column however the grid is spaced.
  let (col_pad_left, col_pad_right): (Vec<f64>, Vec<f64>) = {
    let spacing_at = |position: usize| -> Option<f64> {
      col_spacings
        .get(position - 1)
        .copied()
        .flatten()
        .map(|ems| ems * char_width)
    };
    let inner_gap = |position: usize| spacing_at(position).unwrap_or(pad_x);
    let edge_gap =
      |position: usize| spacing_at(position).unwrap_or(pad_x / 2.0);
    let mut left = Vec::with_capacity(num_cols);
    let mut right = Vec::with_capacity(num_cols);
    for j in 0..num_cols {
      // Column `j` (0-based) sits between positions `j + 1` and `j + 2`.
      left.push(if j == 0 {
        edge_gap(1)
      } else {
        inner_gap(j + 1) / 2.0
      });
      right.push(if j + 1 == num_cols {
        edge_gap(num_cols + 1)
      } else {
        inner_gap(j + 2) / 2.0
      });
    }
    (left, right)
  };
  let col_pad = |j: usize| -> f64 {
    col_pad_left.get(j).copied().unwrap_or(pad_x / 2.0)
      + col_pad_right.get(j).copied().unwrap_or(pad_x / 2.0)
  };

  let mut col_widths: Vec<f64> = vec![0.0; num_cols];
  // Cells that span several columns are held back: their columns are sized
  // by the ordinary cells first, and only what a span still needs is added
  // afterwards, so one wide heading does not stretch a single column.
  let mut spans: Vec<(usize, usize, f64)> = Vec::new();
  for row in &rows {
    for (j, cell) in row.iter().enumerate() {
      if is_span_placeholder(cell) {
        continue;
      }
      let w = match grid_cell_graphic(cell) {
        Some((_, nat_w, _)) => nat_w + col_pad(j),
        None => estimate_display_width(cell) * char_width + col_pad(j),
      };
      let cols = span_width(row, j);
      if cols > 1 {
        spans.push((j, cols, w));
      } else if w > col_widths[j] {
        col_widths[j] = w;
      }
    }
  }
  for (j, cols, w) in spans {
    let end = (j + cols).min(num_cols);
    let have: f64 = col_widths[j..end].iter().sum();
    if w > have && end > j {
      col_widths[end - 1] += w - have;
    }
  }

  // Decimal-point alignment: for each column aligned on ".", find the widest
  // integer part and the widest fractional part across its cells. The dot sits
  // at `max_int` chars from the content's left edge, so the column must be
  // `max_int + max_frac` chars wide to hold the widest number on either side of
  // the point. `col_decimal_dims[j]` carries the per-column split for the
  // render pass; `None` means the column isn't decimal-aligned.
  let col_decimal_dims: Vec<Option<(f64, f64)>> = (0..num_cols)
    .map(|j| {
      let is_decimal =
        col_alignments.get(j).copied().unwrap_or(alignment_h) == "decimal";
      if !is_decimal {
        return None;
      }
      let (mut max_int, mut max_frac) = (0.0_f64, 0.0_f64);
      for row in &rows {
        if let Some(cell) = row.get(j)
          && let Some((iw, fw)) = decimal_split_width(cell)
        {
          max_int = max_int.max(iw);
          max_frac = max_frac.max(fw);
        }
      }
      Some((max_int, max_frac))
    })
    .collect();
  for (j, dims) in col_decimal_dims.iter().enumerate() {
    if let Some((max_int, max_frac)) = dims {
      col_widths[j] = (max_int + max_frac) * char_width + col_pad(j);
    }
  }

  // Compute per-row heights (taller for fractions or images). Cells that
  // span several rows are held back the way column spans are: the rows are
  // sized by the ordinary cells first, and only what a span still needs is
  // added afterwards, so one tall picture does not stretch a single row.
  let mut row_spans: Vec<(usize, usize, f64)> = Vec::new();
  let mut row_heights: Vec<f64> = Vec::with_capacity(num_rows);
  for (i, row) in rows.iter().enumerate() {
    let mut max_h = base_row_height;
    for (j, cell) in row.iter().enumerate() {
      if is_span_placeholder(cell) {
        continue;
      }
      let mut cell_h = if has_fraction(cell) {
        frac_row_height
      } else {
        base_row_height
      };
      // A graphic cell keeps its own height.
      if let Some((_, _, nat_h)) = grid_cell_graphic(cell) {
        cell_h = cell_h.max(nat_h + pad_y);
      }
      // An Image cell is scaled to fit the column width, which fixes its height.
      if let Some(img) = unwrap_to_image(cell)
        && let Expr::Image {
          width: iw,
          height: ih,
          ..
        } = img
      {
        let col_w = if j < col_widths.len() {
          col_widths[j] - col_pad(j)
        } else {
          200.0
        };
        let scale = col_w / (*iw as f64);
        cell_h = cell_h.max((*ih as f64) * scale + pad_y);
      }
      let span = span_height(&rows, i, j);
      if span > 1 {
        row_spans.push((i, span, cell_h));
      } else {
        max_h = max_h.max(cell_h);
      }
    }
    row_heights.push(max_h);
  }
  for (i, span, h) in row_spans {
    let end = (i + span).min(num_rows);
    let have: f64 = row_heights[i..end].iter().sum();
    if h > have && end > i {
      row_heights[end - 1] += h - have;
    }
  }

  let grid_width: f64 = col_widths.iter().sum();
  let total_gap: f64 = group_gaps.len() as f64 * group_gap;
  // Add row_gap between each pair of adjacent rows, plus half-gap padding
  // at top and bottom so that all rows (including first/last) have equal
  // visual height.
  let row_gaps_total: f64 = if num_rows > 1 {
    (num_rows - 1) as f64 * row_gap
  } else {
    0.0
  };
  let edge_pad: f64 = if num_rows > 1 { row_gap } else { 0.0 };
  let total_height: f64 =
    row_heights.iter().sum::<f64>() + total_gap + row_gaps_total + edge_pad;

  // Expand repeating row background pattern into flat row_backgrounds
  if row_bg_has_repeating && !row_bg_repeating.is_empty() {
    let start_len = row_bg_explicit_start.len();
    let end_len = row_bg_explicit_end.len();
    let repeat_len = row_bg_repeating.len();
    row_backgrounds = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
      if i < start_len {
        row_backgrounds.push(row_bg_explicit_start[i]);
      } else if end_len > 0 && i >= num_rows - end_len {
        let end_idx = i - (num_rows - end_len);
        row_backgrounds.push(row_bg_explicit_end[end_idx]);
      } else {
        let repeat_idx = (i - start_len) % repeat_len;
        row_backgrounds.push(row_bg_repeating[repeat_idx]);
      }
    }
  }

  // Expand repeating divider patterns
  // Row dividers have num_rows+1 positions (top, between each row, bottom)
  if row_div_has_repeating && !row_div_repeating.is_empty() {
    let n = num_rows + 1;
    let start_len = row_div_explicit_start.len();
    let end_len = row_div_explicit_end.len();
    let rep_len = row_div_repeating.len();
    row_dividers = Vec::with_capacity(n);
    for i in 0..n {
      if i < start_len {
        row_dividers.push(row_div_explicit_start[i]);
      } else if end_len > 0 && i >= n - end_len {
        let end_idx = i - (n - end_len);
        row_dividers.push(row_div_explicit_end[end_idx]);
      } else {
        let rep_idx = (i - start_len) % rep_len;
        row_dividers.push(row_div_repeating[rep_idx]);
      }
    }
  }
  // Column dividers have num_cols+1 positions (left, between each col, right)
  if col_div_has_repeating && !col_div_repeating.is_empty() {
    let n = num_cols + 1;
    let start_len = col_div_explicit_start.len();
    let end_len = col_div_explicit_end.len();
    let rep_len = col_div_repeating.len();
    col_dividers = Vec::with_capacity(n);
    for i in 0..n {
      if i < start_len {
        col_dividers.push(col_div_explicit_start[i]);
      } else if end_len > 0 && i >= n - end_len {
        let end_idx = i - (n - end_len);
        col_dividers.push(col_div_explicit_end[end_idx]);
      } else {
        let rep_idx = (i - start_len) % rep_len;
        col_dividers.push(col_div_repeating[rep_idx]);
      }
    }
  }

  // Expand repeating column alignment pattern
  if col_align_has_repeating && !col_align_repeating.is_empty() {
    let start_len = col_align_explicit_start.len();
    let rep_len = col_align_repeating.len();
    col_alignments = Vec::with_capacity(num_cols);
    for j in 0..num_cols {
      if j < start_len {
        col_alignments.push(col_align_explicit_start[j]);
      } else {
        let rep_idx = (j - start_len) % rep_len;
        col_alignments.push(col_align_repeating[rep_idx]);
      }
    }
  }

  let has_per_pos_dividers =
    !row_dividers.is_empty() || !col_dividers.is_empty();

  // When parentheses are enabled, reserve space on left and right
  let paren_margin: f64 = if parens { 12.0 } else { 0.0 };
  let total_width: f64 = grid_width + 2.0 * paren_margin;

  // Build SVG — add padding when frame borders are drawn so strokes aren't clipped
  let has_frame = frame_all
    || frame_outer
    || has_per_pos_dividers
    || dividers_col_border
    || dividers_row_border;
  let frame_pad: f64 = if has_frame { 0.5 } else { 0.0 };
  let svg_w = (total_width + 2.0 * frame_pad).ceil() as u32;
  let svg_h = (total_height + 2.0 * frame_pad).ceil() as u32;
  let mut svg = String::with_capacity(2048);
  if has_frame {
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"-0.5 -0.5 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  }

  // Draw round parentheses if enabled
  if parens {
    let h = total_height;
    let inset = 8.0; // how far the curve bows inward
    let stroke_w = 1.2;
    let stroke_color = theme().stroke_default;
    // Left parenthesis: smooth arc from top to bottom, bowing left
    // Cubic Bézier: start at (margin, 0), control points pull left, end at (margin, h)
    let lx = paren_margin;
    svg.push_str(&format!(
      "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_w}\"/>\n",
      lx, 0.0,
      lx - inset, h * 0.33,
      lx - inset, h * 0.67,
      lx, h
    ));
    // Right parenthesis: smooth arc from top to bottom, bowing right
    let rx = paren_margin + grid_width;
    svg.push_str(&format!(
      "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_w}\"/>\n",
      rx, 0.0,
      rx + inset, h * 0.33,
      rx + inset, h * 0.67,
      rx, h
    ));
  }

  // Precompute divider/frame drawing flags (needed for visual bounds below)
  let draw_outer = frame_all || frame_outer;
  // `Dividers -> All` closes the grid on that axis: the top/bottom (or
  // left/right) edge is ruled as well as the boundaries between cells.
  let draw_outer_h = draw_outer || dividers_row_border;
  let draw_outer_v = draw_outer || dividers_col_border;
  let draw_inner_h = frame_all || dividers_row;
  let draw_inner_v = frame_all || dividers_col;
  let has_row_div = !row_dividers.is_empty();
  let has_col_div = !col_dividers.is_empty();

  // Precompute per-row visual bounds for backgrounds and text centering.
  // Compute content y-start and divider y for each row position.
  // Start with row_gap/2 top padding so first/last rows have equal visual height.
  let mut content_y_starts: Vec<f64> = Vec::with_capacity(num_rows);
  let mut divider_ys: Vec<f64> = Vec::with_capacity(num_rows + 1);
  {
    let mut y = row_gap / 2.0;
    for i in 0..=num_rows {
      divider_ys.push(y);
      if i > 0 && i < num_rows {
        y += row_gap;
        if group_gaps.contains(&i) {
          y += group_gap;
        }
      }
      if i < num_rows {
        content_y_starts.push(y);
        y += row_heights[i];
      }
    }
  }
  // Visual top/bottom for each row — backgrounds always split gaps at midpoint
  let mut visual_tops: Vec<f64> = Vec::with_capacity(num_rows);
  let mut visual_bottoms: Vec<f64> = Vec::with_capacity(num_rows);
  for i in 0..num_rows {
    let top = if i == 0 {
      0.0
    } else {
      // Midpoint of gap between row i-1 and row i
      let prev_bottom = content_y_starts[i - 1] + row_heights[i - 1];
      f64::midpoint(prev_bottom, content_y_starts[i])
    };
    let bottom = if i == num_rows - 1 {
      total_height
    } else {
      let this_bottom = content_y_starts[i] + row_heights[i];
      f64::midpoint(this_bottom, content_y_starts[i + 1])
    };
    visual_tops.push(top);
    visual_bottoms.push(bottom);
  }

  // Draw cell backgrounds
  for i in 0..num_rows {
    let bg_y = visual_tops[i];
    let bg_h = visual_bottoms[i] - visual_tops[i];
    // Every column, not just the cells this row happens to have: a ragged
    // row still sits on the grid's background for its full width, the way
    // the Wolfram Language paints it. Columns that share a colour are
    // painted as one rectangle — abutting rectangles antialias against the
    // page along their shared edge, which shows as a seam on a dark
    // background.
    let colour_at = |j: usize| {
      row_backgrounds
        .get(i % row_backgrounds.len().max(1))
        .and_then(|c| c.as_ref())
        .or_else(|| {
          col_backgrounds
            .get(j % col_backgrounds.len().max(1))
            .and_then(|c| c.as_ref())
        })
        .or(background_color.as_ref())
    };
    let mut x_offset: f64 = paren_margin;
    let mut j = 0;
    while j < num_cols {
      let bg = colour_at(j);
      let mut run_w = col_widths[j];
      let mut k = j + 1;
      while k < num_cols
        && colour_at(k).map(|c| c.to_svg_rgb()) == bg.map(|c| c.to_svg_rgb())
      {
        run_w += col_widths[k];
        k += 1;
      }
      if let Some(color) = bg {
        svg.push_str(&format!(
          "<rect x=\"{x_offset:.1}\" y=\"{bg_y:.1}\" width=\"{run_w:.1}\" height=\"{bg_h:.1}\" fill=\"{}\"{}/>\n",
          color.to_svg_rgb(),
          color.opacity_attr()
        ));
      }
      x_offset += run_w;
      j = k;
    }
  }

  // Draw cell contents — text is centered within visual row bounds
  for (i, row) in rows.iter().enumerate() {
    let mut x_offset: f64 = paren_margin;
    for (j, cell) in row.iter().enumerate() {
      // A span placeholder draws nothing; the cell it continues was already
      // laid out across this column or row.
      if is_span_placeholder(cell) {
        x_offset += col_widths[j];
        continue;
      }
      let col_w: f64 = {
        let end = (j + span_width(row, j)).min(num_cols);
        col_widths[j..end].iter().sum()
      };
      // A row-spanning cell is centred over all the rows it covers.
      let last_row = (i + span_height(&rows, i, j)).min(num_rows) - 1;
      let (cell_top, cell_bottom) = (visual_tops[i], visual_bottoms[last_row]);
      let col_align = col_alignments.get(j).copied().unwrap_or(alignment_h);
      // Decimal columns anchor each cell so its decimal point lands at a shared
      // `dot_x`; the whole number is start-anchored at `dot_x - int_width`.
      // Cells that don't render as plain numbers fall back to centering.
      let (anchor, cx): (&str, f64) = if col_align == "decimal" {
        let (max_int, _) = col_decimal_dims
          .get(j)
          .copied()
          .flatten()
          .unwrap_or((0.0, 0.0));
        let dot_x = x_offset + col_pad_left[j] + max_int * char_width;
        match decimal_split_width(cell) {
          Some((iw, _)) => ("start", dot_x - iw * char_width),
          None => ("middle", x_offset + col_w / 2.0),
        }
      } else {
        let span_end = (j + span_width(row, j)).min(num_cols) - 1;
        let cx = match col_align {
          "start" => x_offset + col_pad_left[j],
          "end" => x_offset + col_w - col_pad_right[span_end],
          // Centred on the cell's own area, which is the column run minus
          // the gaps on its two outer sides — the same as the middle of the
          // run whenever those gaps are equal.
          _ => {
            let (l, r) = (col_pad_left[j], col_pad_right[span_end]);
            x_offset + l + (col_w - l - r) / 2.0
          }
        };
        (col_align, cx)
      };
      // Shift text down slightly to compensate for ascenders being taller
      // than descenders, which makes mathematical centering look top-heavy.
      let cy = f64::midpoint(cell_top, cell_bottom) - 1.0;

      // A graphic cell is drawn as a nested <svg> at its own size,
      // centred in the cell.
      if let Some((cell_svg, nat_w, nat_h)) = grid_cell_graphic(cell)
        && let Some(parsed) = parse_svg_dimensions(&cell_svg)
      {
        let gx = x_offset + (col_w - nat_w) / 2.0;
        let vis_h = cell_bottom - cell_top;
        let gy = cell_top + (vis_h - nat_h) / 2.0;
        svg.push_str(&format!(
          "<svg x=\"{gx:.1}\" y=\"{gy:.1}\" width=\"{nat_w:.1}\" height=\"{nat_h:.1}\" viewBox=\"{}\" preserveAspectRatio=\"xMidYMid meet\">\n{}</svg>\n",
          parsed.view_box, parsed.inner_content
        ));
        x_offset += col_w;
        continue;
      }

      // Check if the cell (possibly inside Style) is an Image
      if let Some(img) = unwrap_to_image(cell) {
        if let Expr::Image {
          width: iw,
          height: ih,
          channels,
          data,
          ..
        } = img
        {
          let avail_w = col_w - col_pad(j);
          let scale = avail_w / (*iw as f64);
          let draw_w = avail_w;
          let draw_h = (*ih as f64) * scale;
          let ix = x_offset + col_pad_left[j];
          let vis_h = cell_bottom - cell_top;
          let iy = cell_top + (vis_h - draw_h) / 2.0;

          // Encode image as base64 PNG
          let dyn_img = crate::functions::image_ast::expr_to_dynamic_image(
            *iw, *ih, *channels, data,
          );
          let mut buf = Vec::new();
          dyn_img
            .write_to(
              &mut std::io::Cursor::new(&mut buf),
              image::ImageFormat::Png,
            )
            .expect("PNG encoding failed");
          let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &buf,
          );
          svg.push_str(&format!(
            "<image x=\"{ix:.1}\" y=\"{iy:.1}\" width=\"{draw_w:.1}\" height=\"{draw_h:.1}\" href=\"data:image/png;base64,{b64}\" preserveAspectRatio=\"xMidYMid meet\"/>\n"
          ));
        }
      } else {
        // Text cell — extract optional Style attributes
        let CellTextStyle {
          content,
          font_size: cell_fs,
          font_weight: cell_fw,
          font_style: cell_fst,
          font_family: cell_ff,
          color: cell_color,
          hidden: cell_hidden,
        } = extract_cell_style(cell);

        // Detect `Hyperlink[displayText, url]` cells so the grid can render
        // them as clickable SVG anchors. The display text and href are kept
        // separate: callers pass a stripped-down label (e.g. without the
        // `https://` prefix) while the anchor target stays canonical.
        let (text_content, link_href): (&Expr, Option<&str>) = match content {
          Expr::FunctionCall { name, args }
            if name == "Hyperlink" && args.len() == 2 =>
          {
            let href = match &args[1] {
              Expr::String(s) => Some(s.as_str()),
              _ => None,
            };
            (&args[0], href)
          }
          other => (other, None),
        };

        let fs = cell_fs.or(default_style.font_size).unwrap_or(font_size);
        // Cell style overrides default style; default style overrides "normal"
        let eff_fw = if cell_fw == "normal" {
          default_style.font_weight.unwrap_or("normal")
        } else {
          cell_fw
        };
        let eff_fst = if cell_fst == "normal" {
          default_style.font_style.unwrap_or("normal")
        } else {
          cell_fst
        };
        let fw_attr = if eff_fw == "normal" {
          String::new()
        } else {
          format!(" font-weight=\"{eff_fw}\"")
        };
        let fst_attr = if eff_fst == "normal" {
          String::new()
        } else {
          format!(" font-style=\"{eff_fst}\"")
        };
        // `Style[…, FontFamily -> "Times"]` on the cell, or on the whole
        // grid, picks the face the text is set in; everything else stays
        // with the sans-serif default.
        let ff = cell_ff
          .as_deref()
          .or(default_style.font_family.as_deref())
          .unwrap_or("sans-serif");
        // Hyperlink cells default to a link-blue fill (overridable via
        // explicit Style[..., color]). Plain cells use the cell/default/theme
        // colors as before.
        // An `Invisible[…]` cell is laid out like any other but painted with
        // no fill, so the grid keeps its shape while the cell reads blank.
        let text_fill = if cell_hidden {
          "none".to_string()
        } else if let Some(ref c) = cell_color {
          c.to_svg_rgb()
        } else if link_href.is_some() {
          "#1a73e8".to_string()
        } else if let Some(ref c) = default_style.color {
          c.to_svg_rgb()
        } else {
          theme().text_primary.to_string()
        };
        let markup = expr_to_svg_markup(text_content);
        // `Row[{…, "  ", …}]` spaces its parts with the string it was
        // given, and Wolfram draws every one of those spaces. SVG collapses
        // runs of whitespace unless the element asks it not to.
        let space_attr = if markup.contains("  ") {
          " xml:space=\"preserve\""
        } else {
          ""
        };
        let text_elem = format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"{ff}\" font-size=\"{fs}\"{fw_attr}{fst_attr} fill=\"{text_fill}\" text-anchor=\"{anchor}\" dominant-baseline=\"central\"{space_attr}>{markup}</text>\n",
          ff = svg_escape(ff),
        );
        if let Some(href) = link_href {
          svg.push_str(&format!(
            "<a href=\"{href}\" target=\"_blank\" rel=\"noopener\">{text_elem}</a>\n",
            href = svg_escape(href),
            text_elem = text_elem,
          ));
        } else {
          svg.push_str(&text_elem);
        }
      }
      x_offset += col_w;
    }
  }

  // Draw frame and divider lines
  // Frame color takes priority, then outer Style color, then theme default
  let default_stroke = frame_color
    .as_ref()
    .or(default_style.color.as_ref())
    .map_or_else(|| theme().stroke_default.to_string(), |c| c.to_svg_rgb());

  {
    // Horizontal lines (row dividers)
    // Divider position i is between row i-1 and row i (at the row boundary).
    // Frame borders (i=0 / i=num_rows) are drawn at the grid edges (0 / total_height).
    for i in 0..=num_rows {
      let is_border = i == 0 || i == num_rows;
      // Check per-position divider spec first, then fall back to boolean flags
      let (should_draw, stroke) = if has_row_div {
        if let Some(Some(color)) = row_dividers.get(i) {
          (true, color.to_svg_rgb())
        } else if is_border && draw_outer_h {
          (true, default_stroke.clone())
        } else {
          (false, String::new())
        }
      } else if (is_border && draw_outer_h) || (!is_border && draw_inner_h) {
        (true, default_stroke.clone())
      } else {
        (false, String::new())
      };
      if should_draw {
        // Frame borders at 0 / total_height; inner dividers at visual row boundaries
        let draw_y = if i == 0 {
          0.0
        } else if i == num_rows {
          total_height
        } else {
          visual_tops[i]
        };
        // A divider is interrupted wherever it would cut through a cell
        // that spans across it (`SpanFromAbove`), so it is drawn per column
        // and contiguous runs are merged back into one line.
        let mut x_offset: f64 = paren_margin;
        let mut run_start: Option<f64> = None;
        for j in 0..num_cols {
          let spanned = !is_border
            && rows
              .get(i)
              .and_then(|row| row.get(j))
              .is_some_and(is_span_from_above);
          if spanned {
            if let Some(x0) = run_start.take() {
              svg.push_str(&format!(
                "<line x1=\"{x0:.1}\" y1=\"{draw_y:.1}\" x2=\"{x_offset:.1}\" y2=\"{draw_y:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n"
              ));
            }
          } else if run_start.is_none() {
            run_start = Some(x_offset);
          }
          x_offset += col_widths[j];
        }
        if let Some(x0) = run_start {
          svg.push_str(&format!(
            "<line x1=\"{x0:.1}\" y1=\"{draw_y:.1}\" x2=\"{:.1}\" y2=\"{draw_y:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n",
            paren_margin + grid_width
          ));
        }
      }
    }
  }
  {
    // Vertical lines (column dividers)
    let mut x_offset: f64 = paren_margin;
    for j in 0..=num_cols {
      let is_border = j == 0 || j == num_cols;
      let (should_draw, stroke) = if has_col_div {
        if let Some(Some(color)) = col_dividers.get(j) {
          (true, color.to_svg_rgb())
        } else if is_border && draw_outer_v {
          (true, default_stroke.clone())
        } else {
          (false, String::new())
        }
      } else if (is_border && draw_outer_v) || (!is_border && draw_inner_v) {
        (true, default_stroke.clone())
      } else {
        (false, String::new())
      };
      if should_draw {
        // A divider is interrupted wherever it would cut through a cell
        // that spans across it (`SpanFromLeft`), so it is drawn per row and
        // contiguous runs are merged back into one line.
        let mut run_start: Option<f64> = None;
        for i in 0..num_rows {
          let spanned = !is_border
            && rows
              .get(i)
              .and_then(|row| row.get(j))
              .is_some_and(is_span_from_left);
          if spanned {
            if let Some(y0) = run_start.take() {
              svg.push_str(&format!(
                "<line x1=\"{x_offset:.1}\" y1=\"{y0:.1}\" x2=\"{x_offset:.1}\" y2=\"{:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n",
                visual_tops[i]
              ));
            }
          } else if run_start.is_none() {
            run_start = Some(if i == 0 { 0.0 } else { visual_tops[i] });
          }
        }
        if let Some(y0) = run_start {
          svg.push_str(&format!(
            "<line x1=\"{x_offset:.1}\" y1=\"{y0:.1}\" x2=\"{x_offset:.1}\" y2=\"{total_height:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n"
          ));
        }
      }
      if j < num_cols {
        x_offset += col_widths[j];
      }
    }
  }

  // TableHeadings separators: a thin rule below the heading row and to the
  // right of the heading column, matching how Wolfram sets headings apart
  // from the table body.
  if has_col_heading_row || has_row_heading_col {
    let heading_stroke = theme().stroke_default;
    if has_col_heading_row && num_rows > 1 {
      // Horizontal rule at the boundary below the first (heading) row.
      let y = visual_bottoms[0];
      svg.push_str(&format!(
        "<line x1=\"{paren_margin:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{heading_stroke}\" stroke-width=\"1\"/>\n",
        paren_margin + grid_width
      ));
    }
    if has_row_heading_col && num_cols > 1 {
      // Vertical rule at the boundary to the right of the first (heading) col.
      let x = paren_margin + col_widths[0];
      svg.push_str(&format!(
        "<line x1=\"{x:.1}\" y1=\"0\" x2=\"{x:.1}\" y2=\"{total_height:.1}\" stroke=\"{heading_stroke}\" stroke-width=\"1\"/>\n"
      ));
    }
  }

  svg.push_str("</svg>");

  Ok(svg)
}

/// Render a 3D MatrixForm: a 2D grid of parenthesized column vectors,
/// all wrapped in outer parentheses.
///
/// Input: list of rows, each row is a list of sub-lists.
/// Each sub-list `{a, b, c}` is rendered as a parenthesized column vector.
/// The grid of these column vectors is wrapped in outer parentheses.
pub fn matrixform_3d_ast(
  outer_rows: &[Vec<Expr>],
) -> Result<Expr, InterpreterError> {
  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 12.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;
  let paren_w: f64 = 10.0; // width reserved for each sub-paren pair
  let paren_inset: f64 = 5.0; // how far parens bow
  let outer_paren_margin: f64 = 12.0;
  let outer_paren_inset: f64 = 8.0;
  let cell_gap_x: f64 = 14.0; // horizontal gap between cells
  let cell_gap_y: f64 = 10.0; // vertical gap between rows

  let num_outer_rows = outer_rows.len();
  let num_outer_cols =
    outer_rows.iter().map(std::vec::Vec::len).max().unwrap_or(0);
  if num_outer_cols == 0 {
    return Err(InterpreterError::EvaluationError(
      "MatrixForm: empty 3D data".into(),
    ));
  }

  // For each cell, determine: max element display width and number of sub-rows
  // cell_info[i][j] = (sub_row_count, max_elem_width_chars)
  let mut cell_info: Vec<Vec<(usize, f64)>> = Vec::new();
  for row in outer_rows {
    let mut row_info = Vec::new();
    for cell in row {
      match cell {
        Expr::List(items) => {
          let count = items.len().max(1);
          let max_w: f64 = items
            .iter()
            .map(estimate_display_width)
            .fold(0.0_f64, f64::max);
          row_info.push((count, max_w));
        }
        _ => {
          row_info.push((1, estimate_display_width(cell)));
        }
      }
    }
    // Pad to num_outer_cols
    while row_info.len() < num_outer_cols {
      row_info.push((1, 1.0));
    }
    cell_info.push(row_info);
  }

  // For each outer column, find max sub-cell width
  let mut col_inner_widths: Vec<f64> = vec![0.0; num_outer_cols];
  for row_info in &cell_info {
    for (j, &(_, max_w)) in row_info.iter().enumerate() {
      let w = max_w * char_width + pad_x;
      if w > col_inner_widths[j] {
        col_inner_widths[j] = w;
      }
    }
  }

  // Each cell's total width = inner_width + 2 * paren_w (for sub-parens)
  let col_total_widths: Vec<f64> =
    col_inner_widths.iter().map(|w| w + 2.0 * paren_w).collect();

  // For each outer row, find max sub-row count (determines row height)
  let outer_row_sub_counts: Vec<usize> = cell_info
    .iter()
    .map(|ri| ri.iter().map(|&(c, _)| c).max().unwrap_or(1))
    .collect();
  let outer_row_heights: Vec<f64> = outer_row_sub_counts
    .iter()
    .map(|&c| c as f64 * row_height)
    .collect();

  let grid_width: f64 = col_total_widths.iter().sum::<f64>()
    + (num_outer_cols as f64 - 1.0) * cell_gap_x;
  let grid_height: f64 = outer_row_heights.iter().sum::<f64>()
    + (num_outer_rows as f64 - 1.0) * cell_gap_y;

  let total_width = grid_width + 2.0 * outer_paren_margin;
  let total_height = grid_height;

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  // Draw outer parentheses
  let lx = outer_paren_margin;
  let h = total_height;
  let stroke_w = 1.2;
  let stroke_color = theme().stroke_default;
  svg.push_str(&format!(
    "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_w}\"/>\n",
    lx, 0.0,
    lx - outer_paren_inset, h * 0.33,
    lx - outer_paren_inset, h * 0.67,
    lx, h
  ));
  let rx = outer_paren_margin + grid_width;
  svg.push_str(&format!(
    "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{stroke_w}\"/>\n",
    rx, 0.0,
    rx + outer_paren_inset, h * 0.33,
    rx + outer_paren_inset, h * 0.67,
    rx, h
  ));

  // Draw each cell
  let mut y_off = 0.0_f64;
  for (i, row) in outer_rows.iter().enumerate() {
    let row_h = outer_row_heights[i];
    let mut x_off = outer_paren_margin;
    for (j, cell) in row.iter().enumerate() {
      let cell_w = col_total_widths[j];
      let inner_w = col_inner_widths[j];

      // Get sub-items for this cell
      let sub_items: Vec<&Expr> = match cell {
        Expr::List(items) => items.iter().collect(),
        _ => vec![cell],
      };
      let sub_count = sub_items.len();
      let sub_h = sub_count as f64 * row_height;

      // Center sub-vector vertically within cell
      let sub_y_start = y_off + (row_h - sub_h) / 2.0;

      // Draw sub-parentheses around this cell's column vector
      let sub_lx = x_off + paren_w;
      let sub_rx = x_off + paren_w + inner_w;
      let sub_top = sub_y_start;
      let sub_bot = sub_y_start + sub_h;
      let sub_stroke = 1.0;

      svg.push_str(&format!(
        "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{sub_stroke}\"/>\n",
        sub_lx, sub_top,
        sub_lx - paren_inset, sub_top + sub_h * 0.33,
        sub_lx - paren_inset, sub_top + sub_h * 0.67,
        sub_lx, sub_bot
      ));
      svg.push_str(&format!(
        "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"{stroke_color}\" stroke-width=\"{sub_stroke}\"/>\n",
        sub_rx, sub_top,
        sub_rx + paren_inset, sub_top + sub_h * 0.33,
        sub_rx + paren_inset, sub_top + sub_h * 0.67,
        sub_rx, sub_bot
      ));

      // Draw sub-items as text, vertically stacked
      let text_fill = theme().text_primary;
      for (k, item) in sub_items.iter().enumerate() {
        let cx = x_off + cell_w / 2.0;
        let cy = sub_y_start + k as f64 * row_height + row_height / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"sans-serif\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(item)
        ));
      }

      x_off += cell_w + cell_gap_x;
    }
    y_off += row_h + cell_gap_y;
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Render a Dataset expression as an SVG table.
/// Dataset[<|k1 -> v1, ...|>, type, meta] → transposed table (keys left, values right)
/// Dataset[{<|...|>, <|...|>, ...}, type, meta] → multi-row table with column headers
pub fn dataset_to_svg(data: &Expr) -> Option<String> {
  match data {
    Expr::Association(pairs) => dataset_assoc_to_svg(pairs),
    Expr::List(items) => dataset_list_to_svg(items),
    _ => None,
  }
}

/// Single association: transposed two-column table with keys on the left (bold,
/// with background) and values on the right.
fn dataset_assoc_to_svg(pairs: &[(Expr, Expr)]) -> Option<String> {
  if pairs.is_empty() {
    return None;
  }

  let num_rows = pairs.len();
  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 16.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;

  // Compute key column and value column widths
  let mut key_col_w: f64 = 0.0;
  let mut val_col_w: f64 = 0.0;
  let keys: Vec<String> =
    pairs.iter().map(|(k, _)| expr_to_svg_markup(k)).collect();
  for (i, (_, v)) in pairs.iter().enumerate() {
    let kw = keys[i].len() as f64 * char_width + pad_x;
    if kw > key_col_w {
      key_col_w = kw;
    }
    let vw = estimate_display_width(v) * char_width + pad_x;
    if vw > val_col_w {
      val_col_w = vw;
    }
  }

  let total_width = key_col_w + val_col_w;
  let total_height = num_rows as f64 * row_height;

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let t = theme();

  // Key column background
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{key_col_w:.1}\" height=\"{total_height:.1}\" fill=\"{}\"/>\n",
    t.table_header_bg
  ));

  // Rows
  let mut y_offset: f64 = 0.0;
  let text_fill = t.text_primary;
  for (i, (_, v)) in pairs.iter().enumerate() {
    let cy = y_offset + row_height / 2.0;
    // Key (bold, in left column)
    let kx = key_col_w / 2.0;
    svg.push_str(&format!(
      "<text x=\"{kx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      keys[i]
    ));
    // Value (in right column)
    let vx = key_col_w + val_col_w / 2.0;
    svg.push_str(&format!(
      "<text x=\"{vx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(v)
    ));
    y_offset += row_height;
  }

  // Grid lines
  let border_color = t.table_border_strong;
  let light_color = t.table_border_light;
  // Horizontal lines
  let mut y = 0.0_f64;
  for i in 0..=num_rows {
    let stroke_width = if i == 0 || i == num_rows {
      "1.5"
    } else {
      "0.5"
    };
    let color = if i == 0 || i == num_rows {
      border_color
    } else {
      light_color
    };
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>\n"
    ));
    y += row_height;
  }
  // Vertical lines: outer borders + separator between key and value columns
  svg.push_str(&format!(
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{key_col_w:.1}\" y1=\"0\" x2=\"{key_col_w:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));

  svg.push_str("</svg>");
  Some(svg)
}

/// Plain list of values: single-column table with no header.
fn dataset_plain_list_to_svg(items: &[Expr]) -> Option<String> {
  if items.is_empty() {
    return None;
  }

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 16.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;
  let num_rows = items.len();

  // Compute column width from data
  let mut col_w: f64 = 0.0;
  for item in items {
    let w = estimate_display_width(item) * char_width + pad_x;
    if w > col_w {
      col_w = w;
    }
  }

  let total_width = col_w;
  let total_height = num_rows as f64 * row_height;

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(2048);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let t = theme();
  let text_fill = t.text_primary;

  // Data rows
  let mut y_offset: f64 = 0.0;
  for item in items {
    let cx = col_w / 2.0;
    let cy = y_offset + row_height / 2.0;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(item)
    ));
    y_offset += row_height;
  }

  // Grid lines
  let border_color = t.table_border_strong;
  let light_color = t.table_border_light;
  // Horizontal lines
  let mut y = 0.0_f64;
  for i in 0..=num_rows {
    let stroke_width = if i == 0 || i == num_rows {
      "1.5"
    } else {
      "0.5"
    };
    let color = if i == 0 || i == num_rows {
      border_color
    } else {
      light_color
    };
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>\n"
    ));
    y += row_height;
  }
  // Vertical lines (outer borders)
  svg.push_str(&format!(
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));

  svg.push_str("</svg>");
  Some(svg)
}

/// List of associations: multi-row table with column headers on top.
fn dataset_list_to_svg(items: &[Expr]) -> Option<String> {
  if items.is_empty() {
    return None;
  }
  // Check if this is a list of associations or a plain list
  let is_assoc_list = items
    .iter()
    .all(|item| matches!(item, Expr::Association(_)));
  if !is_assoc_list {
    return dataset_plain_list_to_svg(items);
  }
  // Collect all unique keys in order of first appearance
  let mut headers: Vec<String> = Vec::new();
  let mut header_set = std::collections::HashSet::new();
  for item in items {
    if let Expr::Association(pairs) = item {
      for (k, _) in pairs {
        let key_str = expr_to_svg_markup(k);
        if header_set.insert(key_str.clone()) {
          headers.push(key_str);
        }
      }
    }
  }
  // Build rows aligned to headers
  let rows: Vec<Vec<Expr>> = items
    .iter()
    .map(|item| {
      if let Expr::Association(pairs) = item {
        headers
          .iter()
          .map(|h| {
            pairs
              .iter()
              .find(|(k, _)| expr_to_svg_markup(k) == *h)
              .map_or(call0("Missing"), |(_, v)| v.clone())
          })
          .collect()
      } else {
        vec![]
      }
    })
    .collect();

  if headers.is_empty() {
    return None;
  }

  let num_cols = headers.len();
  let num_data_rows = rows.len();
  let num_total_rows = num_data_rows + 1; // +1 for header row

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 16.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;
  let header_row_height = font_size + pad_y + 2.0;

  // Compute column widths from headers and data
  let mut col_widths: Vec<f64> = headers
    .iter()
    .map(|h| h.len() as f64 * char_width + pad_x)
    .collect();
  for row in &rows {
    for (j, cell) in row.iter().enumerate() {
      if j < num_cols {
        let w = estimate_display_width(cell) * char_width + pad_x;
        if w > col_widths[j] {
          col_widths[j] = w;
        }
      }
    }
  }

  let total_width: f64 = col_widths.iter().sum();
  let total_height: f64 =
    header_row_height + (num_data_rows as f64) * row_height;

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let t = theme();

  // Header background
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{total_width:.1}\" height=\"{header_row_height:.1}\" fill=\"{}\"/>\n",
    t.table_header_bg
  ));

  // Header text (bold)
  let text_fill = t.text_primary;
  {
    let mut x_offset: f64 = 0.0;
    for (j, header) in headers.iter().enumerate() {
      let col_w = col_widths[j];
      let cx = x_offset + col_w / 2.0;
      let cy = header_row_height / 2.0;
      svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{header}</text>\n"
      ));
      x_offset += col_w;
    }
  }

  // Data rows
  let mut y_offset: f64 = header_row_height;
  for row in &rows {
    let mut x_offset: f64 = 0.0;
    for (j, cell) in row.iter().enumerate() {
      if j < num_cols {
        let col_w = col_widths[j];
        let cx = x_offset + col_w / 2.0;
        let cy = y_offset + row_height / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(cell)
        ));
        x_offset += col_w;
      }
    }
    y_offset += row_height;
  }

  // Grid lines
  let border_color = t.table_border_strong;
  let light_color = t.table_border_light;
  // Horizontal lines
  let mut y = 0.0_f64;
  for i in 0..=num_total_rows {
    let stroke_width = if i == 0 || i == 1 || i == num_total_rows {
      "1.5"
    } else {
      "0.5"
    };
    let color = if i == 0 || i == 1 || i == num_total_rows {
      border_color
    } else {
      light_color
    };
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>\n"
    ));
    if i == 0 {
      y += header_row_height;
    } else if i < num_total_rows {
      y += row_height;
    }
  }
  // Vertical lines (only outer borders)
  svg.push_str(&format!(
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));

  svg.push_str("</svg>");
  Some(svg)
}

// ── Combine multiple Graphics SVGs into a grid ─────────────────────────

/// Parsed metadata from an SVG element
struct ParsedSvg {
  view_box: String,
  inner_content: String,
  /// Natural display width from the root `width="..."` attribute, if present.
  /// Falls back to the viewBox width when the attribute is missing. Used by
  /// `combine_svgs_grid` to pick a default total width that lets each cell
  /// render near its native size instead of being scaled down to illegibility.
  nat_w: f64,
  /// Natural display height from the root `height="..."` attribute, if
  /// present. Falls back to the viewBox height when the attribute is missing.
  nat_h: f64,
}

/// Parse a numeric attribute value like `width="360"` or `height="225px"` from
/// the root `<svg ...>` tag. Trailing unit suffixes (px, pt) are stripped.
/// Find an attribute's value in `header`, accepting either quote style
/// (image SVG wrappers use single quotes). Returns the raw value text.
fn find_svg_attr<'a>(header: &'a str, attr: &str) -> Option<&'a str> {
  for quote in ['"', '\''] {
    let needle = format!("{attr}={quote}");
    if let Some(start) = header.find(&needle) {
      let start = start + needle.len();
      let rel_end = header[start..].find(quote)?;
      return Some(&header[start..start + rel_end]);
    }
  }
  None
}

/// The first `<svg ...` opening tag's header (attributes text). Skips any
/// leading `<?xml ...?>` declaration, whose `?>` would otherwise be taken
/// for the end of the root tag.
fn svg_root_header(svg: &str) -> Option<&str> {
  let tag_start = svg.find("<svg")?;
  let rel_end = svg[tag_start..].find('>')?;
  Some(&svg[tag_start..tag_start + rel_end])
}

fn parse_svg_numeric_attr(svg: &str, attr: &str) -> Option<f64> {
  // Only consider the first `<svg ...>` opening tag to avoid matching
  // attributes on nested cells.
  let header = svg_root_header(svg)?;
  let raw = find_svg_attr(header, attr)?.trim();
  let numeric_end = raw
    .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
    .unwrap_or(raw.len());
  raw[..numeric_end].parse().ok()
}

/// The natural display size of a rendered picture, in pixels. That is the
/// size an inset of it keeps whatever the enclosing plot range turns out to
/// be, so it is what a caller needs before embedding one.
pub(crate) fn svg_natural_size(svg: &str) -> Option<(f64, f64)> {
  parse_svg_dimensions(svg).map(|p| (p.nat_w, p.nat_h))
}

/// An already-rendered picture nested inside another one: a `<svg>` element
/// of `w`×`h` pixels centred on (`cx`, `cy`) in the enclosing picture's
/// pixel coordinates, drawing the original through its own viewBox.
pub(crate) fn embed_svg_centered(
  svg: &str,
  cx: f64,
  cy: f64,
  w: f64,
  h: f64,
) -> String {
  let view_box = parse_svg_dimensions(svg)
    .map_or_else(|| format!("0 0 {w} {h}"), |p| p.view_box);
  format!(
    "<svg x=\"{:.2}\" y=\"{:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" viewBox=\"{view_box}\" preserveAspectRatio=\"xMidYMid meet\">\n{}</svg>\n",
    cx - w / 2.0,
    cy - h / 2.0,
    strip_svg_wrapper(svg),
  )
}

/// Parse width, height, and viewBox from an SVG string
fn parse_svg_dimensions(svg: &str) -> Option<ParsedSvg> {
  // Extract the viewBox attribute; an SVG without one (e.g. the base64-PNG
  // wrapper produced for `Image[…]`) synthesizes it from width/height so
  // raster images can take part in GraphicsRow/Column/Grid layouts.
  let header = svg_root_header(svg)?;
  let view_box = if let Some(vb) = find_svg_attr(header, "viewBox") {
    vb.to_string()
  } else {
    let w = parse_svg_numeric_attr(svg, "width")?;
    let h = parse_svg_numeric_attr(svg, "height")?;
    format!("0 0 {w} {h}")
  };

  // Parse viewBox to get dimensions: "x y w h"
  let parts: Vec<f64> = view_box
    .split_whitespace()
    .filter_map(|s| s.parse().ok())
    .collect();
  if parts.len() < 4 {
    return None;
  }
  let (vb_w, vb_h) = (parts[2], parts[3]);

  // Extract inner content (everything between the root tag's > and the
  // last </svg>)
  let root_start = svg.find("<svg")?;
  let inner_start = root_start + svg[root_start..].find('>')? + 1;
  // Skip past the newline after the opening tag if present
  let inner_start = if svg[inner_start..].starts_with('\n') {
    inner_start + 1
  } else {
    inner_start
  };
  let inner_end = svg.rfind("</svg>")?;
  let inner_content = svg[inner_start..inner_end].to_string();

  // Prefer the root width/height attributes (the natural display size);
  // fall back to the viewBox dimensions when absent.
  let nat_w = parse_svg_numeric_attr(svg, "width").unwrap_or(vb_w);
  let nat_h = parse_svg_numeric_attr(svg, "height").unwrap_or(vb_h);

  Some(ParsedSvg {
    view_box,
    inner_content,
    nat_w,
    nat_h,
  })
}

/// Combine multiple SVG strings arranged as rows of cells into a single SVG.
/// `rows` is a Vec of rows, each row is a Vec of SVG strings. Used for 2-D
/// and 3-D lists of graphics at the top level of an expression.
///
/// Uses the same natural-dimension layout as `GraphicsRow`/`GraphicsGrid`
/// so each cell renders near its native size (instead of being crammed
/// into fixed 80-pixel squares that make plots illegible).
pub fn combine_graphics_svgs(rows: &[Vec<String>]) -> Option<String> {
  combine_svgs_grid(rows, &default_layout_options())
}

/// Render a 1-D list of SVGs as `{ svg₁, svg₂, … }` with brace/comma text
/// interleaved between the nested graphic cells.
///
/// Uses the same natural-dimension layout as `GraphicsRow` so each cell
/// renders near its native size, with brace and comma decorations sized
/// proportionally to the row height.
pub fn graphics_list_svg(svgs: &[String]) -> Option<String> {
  if svgs.is_empty() {
    return None;
  }

  // Lay out the cells as a single row using the shared grid engine.
  let rows = vec![svgs.to_vec()];
  let layout = compute_grid_layout(&rows, &default_layout_options())?;
  let row = layout.rows.first()?;
  if row.cells.is_empty() {
    return None;
  }

  // Decoration sizes scale with the row height so braces/commas match
  // the visual weight of the contained graphics.
  let row_h = row.row_h;
  let font_size = (row_h * 0.18).max(12.0);
  let brace_w = (row_h * 0.12).max(10.0);
  let comma_w = (row_h * 0.08).max(6.0);
  let text_y = row_h / 2.0;

  // Extra horizontal space needed for braces and per-gap commas. The
  // grid layout already placed cells at x = Σ(prev cell_w + h_gap); we
  // shift everything right by brace_w and inject extra comma_w slots
  // into each gap between cells.
  let n = row.cells.len();
  let n_gaps = (n - 1) as f64;
  let total_width = layout.total_width + 2.0 * brace_w + n_gaps * comma_w;

  let mut out = String::with_capacity(4096);
  out.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n",
    total_width.ceil() as u32,
    row_h.ceil() as u32,
    total_width.ceil() as u32,
    row_h.ceil() as u32,
  ));

  let text_fill = theme().text_primary;

  // Opening brace
  out.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size:.1}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">{{</text>\n",
    brace_w / 2.0,
  ));

  // Place each cell at its layout-computed x, shifted by brace_w plus
  // one comma_w for every preceding gap.
  for (i, cell) in row.cells.iter().enumerate() {
    let shift = brace_w + (i as f64) * comma_w;
    let cell_x = cell.x + shift;

    if i > 0 {
      // Comma goes in the middle of the slot between the previous cell
      // and this one, i.e. just left of the current shifted cell x.
      let comma_center = cell_x - comma_w / 2.0;
      out.push_str(&format!(
        "<text x=\"{comma_center:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
         font-size=\"{font_size:.1}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
         dominant-baseline=\"central\">,</text>\n",
      ));
    }

    out.push_str(&format!(
      "<svg x=\"{:.0}\" y=\"{:.0}\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"{}\">\n",
      cell_x,
      cell.y_off,
      cell.w,
      cell.h,
      cell.view_box,
    ));
    out.push_str(&cell.inner);
    out.push_str("</svg>\n");
  }

  // Closing brace — sits just past the last cell's right edge.
  let close_x = total_width - brace_w / 2.0;
  out.push_str(&format!(
    "<text x=\"{close_x:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size:.1}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">}}</text>\n",
  ));

  out.push_str("</svg>");
  Some(out)
}

// ── GraphicsRow / GraphicsColumn / GraphicsGrid ────────────────────────

/// Extract SVG strings from a list of evaluated expressions.
/// Items that are already Expr::Graphics are used directly; other
/// expressions (Graph, TreeForm, Dataset, plain values, ...) are
/// converted to SVG via the shared `expr_to_svg` helper so they render
/// alongside native graphics in GraphicsRow/Column/Grid layouts.
fn extract_svgs_from_list(items: &[Expr]) -> Vec<String> {
  items
    .iter()
    .filter_map(|item| {
      let svg = crate::evaluator::expr_to_svg(item);
      if svg.is_empty() { None } else { Some(svg) }
    })
    .collect()
}

/// Spacing specification: either absolute printer's points or scaled fraction.
#[derive(Clone, Copy)]
enum SpacingSpec {
  /// Absolute spacing in printer's points (1 pt = 4/3 px at 96 dpi)
  Points(f64),
  /// Fraction of item size (e.g. Scaled[0.1] = 10% of cell dimension)
  Scaled(f64),
}

impl SpacingSpec {
  /// Default: Scaled[0.1] per Mathematica docs
  fn default_val() -> Self {
    Self::Scaled(0.1)
  }

  /// Resolve to pixels given a cell dimension (width or height)
  fn to_px(self, cell_dim: f64) -> f64 {
    match self {
      Self::Points(pts) => pts * (4.0 / 3.0), // pt → px at 96 dpi
      Self::Scaled(frac) => frac * cell_dim,
    }
  }
}

/// Parse a single spacing value from an expression.
/// - Numeric → Points(n)
/// - Scaled[s] → Scaled(s)
fn parse_spacing_expr(expr: &Expr) -> Option<SpacingSpec> {
  // Scaled[s]
  if let Expr::FunctionCall { name, args } = expr
    && name == "Scaled"
    && args.len() == 1
    && let Some(val) = try_eval_to_f64(&args[0])
  {
    return Some(SpacingSpec::Scaled(val));
  }
  // Numeric value → printer's points
  if let Some(val) = try_eval_to_f64(expr) {
    return Some(SpacingSpec::Points(val));
  }
  None
}

/// Frame setting for GraphicsRow/Column/Grid.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameSetting {
  /// No frame
  None,
  /// Outer frame only (Frame -> True)
  Outer,
  /// All cell boundaries (Frame -> All)
  All,
}

/// Parsed layout options for GraphicsRow/Column/Grid.
struct LayoutOptions {
  h_spacing: SpacingSpec,
  v_spacing: SpacingSpec,
  /// Total width constraint (from ImageSize -> n or ImageSize -> {w, h})
  target_width: Option<f64>,
  /// Total height constraint (only from ImageSize -> {w, h})
  target_height: Option<f64>,
  /// Frame setting (Frame -> None | True | All)
  frame: FrameSetting,
  /// True if Spacings was explicitly given (so Frame -> All shouldn't
  /// override the user's choice).
  spacings_explicit: bool,
}

/// Parse Spacings and ImageSize options from rule arguments.
fn parse_layout_options(args: &[Expr]) -> LayoutOptions {
  let mut opts = LayoutOptions {
    h_spacing: SpacingSpec::default_val(),
    v_spacing: SpacingSpec::default_val(),
    target_width: None,
    target_height: None,
    frame: FrameSetting::None,
    spacings_explicit: false,
  };

  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "Spacings" => {
          opts.spacings_explicit = true;
          match replacement.as_ref() {
            // {h, v} pair
            Expr::List(pair) if pair.len() == 2 => {
              if let Some(h) = parse_spacing_expr(&pair[0]) {
                opts.h_spacing = h;
              }
              if let Some(v) = parse_spacing_expr(&pair[1]) {
                opts.v_spacing = v;
              }
            }
            // Single value → both directions
            other => {
              if let Some(spec) = parse_spacing_expr(other) {
                opts.h_spacing = spec;
                opts.v_spacing = spec;
              }
            }
          }
        }
        Expr::Identifier(name) if name == "Frame" => {
          opts.frame = match replacement.as_ref() {
            Expr::Identifier(s) if s == "All" => FrameSetting::All,
            Expr::Identifier(s) if s == "True" => FrameSetting::Outer,
            Expr::Identifier(s) if s == "None" || s == "False" => {
              FrameSetting::None
            }
            _ => opts.frame,
          };
        }
        Expr::Identifier(name) if name == "ImageSize" => {
          match replacement.as_ref() {
            // {w, h} explicit pair
            Expr::List(pair) if pair.len() == 2 => {
              if let Some(w) = try_eval_to_f64(&pair[0]) {
                opts.target_width = Some(w);
              }
              if let Some(h) = try_eval_to_f64(&pair[1]) {
                opts.target_height = Some(h);
              }
            }
            // Single number → total width only
            other => {
              if let Some(w) = try_eval_to_f64(other) {
                opts.target_width = Some(w);
              } else if let Some((w, _, _)) =
                parse_image_size(other, DEFAULT_WIDTH, DEFAULT_HEIGHT)
              {
                // Named sizes (Small, Medium, Large, etc.)
                opts.target_width = Some(w as f64);
              }
            }
          }
        }
        _ => {}
      }
    }
  }

  opts
}

/// A single laid-out cell in a grid: position within its row plus
/// scaled display dimensions and the raw SVG fragment to emit.
struct LayoutCell {
  x: f64,
  y_off: f64,
  w: f64,
  h: f64,
  view_box: String,
  inner: String,
}

struct GridRowLayout {
  cells: Vec<LayoutCell>,
  row_h: f64,
}

/// Computed layout for a grid of graphics. All coordinates are in
/// final pixel space (already scaled). Row `y` positions must be derived
/// by walking `rows` in order and adding `v_gap` between them.
struct GridLayout {
  rows: Vec<GridRowLayout>,
  total_width: f64,
  total_height: f64,
  v_gap: f64,
}

/// Compute a natural-dimension grid layout for a 2-D array of SVG cells.
///
/// Each cell keeps its natural width and height (parsed from the root
/// `width="..."` / `height="..."` attributes of each input SVG), scaled by a
/// single uniform factor so the widest row fits the target total width.
/// Within a row, cells with shorter natural heights are vertically centered,
/// which keeps widths consistent across cells of different aspect ratios —
/// e.g. a NumberLinePlot (natively 360×105) and a Plot (natively 360×225)
/// both render at 360 wide instead of the NumberLinePlot ballooning out.
fn compute_grid_layout(
  rows: &[Vec<String>],
  opts: &LayoutOptions,
) -> Option<GridLayout> {
  if rows.is_empty() {
    return None;
  }
  if rows.iter().all(std::vec::Vec::is_empty) {
    return None;
  }

  // When Frame is set and the user didn't pick a spacing, default to 0
  // so cells abut the frame lines (matching wolframscript's behaviour).
  let opts_owned;
  let opts: &LayoutOptions =
    if opts.frame != FrameSetting::None && !opts.spacings_explicit {
      opts_owned = LayoutOptions {
        h_spacing: SpacingSpec::Points(0.0),
        v_spacing: SpacingSpec::Points(0.0),
        target_width: opts.target_width,
        target_height: opts.target_height,
        frame: opts.frame,
        spacings_explicit: opts.spacings_explicit,
      };
      &opts_owned
    } else {
      opts
    };

  // Parse all SVGs: (viewBox, inner_content, nat_w, nat_h).
  let parsed_rows: Vec<Vec<(String, String, f64, f64)>> = rows
    .iter()
    .map(|row| {
      row
        .iter()
        .filter_map(|svg| {
          let p = parse_svg_dimensions(svg)?;
          Some((p.view_box, p.inner_content, p.nat_w, p.nat_h))
        })
        .collect()
    })
    .collect();

  if parsed_rows.iter().all(std::vec::Vec::is_empty) {
    return None;
  }

  // Natural row widths (sum of child nat_w) and natural row heights
  // (max of child nat_h). These drive the default layout before any
  // target-width scaling.
  let row_nat_dims: Vec<(f64, f64)> = parsed_rows
    .iter()
    .map(|row| {
      let w: f64 = row.iter().map(|(_, _, nw, _)| *nw).sum();
      let h: f64 = row.iter().map(|(_, _, _, nh)| *nh).fold(0.0_f64, f64::max);
      (w, h)
    })
    .collect();
  let max_nat_row_w = row_nat_dims
    .iter()
    .map(|(w, _)| *w)
    .fold(0.0_f64, f64::max)
    .max(10.0);

  // Default total width: widest natural row + padding for Scaled[0.1]
  // gaps so cells aren't compressed below their native resolution.
  let max_cols = parsed_rows
    .iter()
    .map(std::vec::Vec::len)
    .max()
    .unwrap_or(1);
  let gap_pad = if max_cols > 1 {
    1.0 + 0.1 * (max_cols as f64 - 1.0) / max_cols as f64
  } else {
    1.0
  };
  let default_total_w = max_nat_row_w * gap_pad;
  let target_w = opts.target_width.unwrap_or(default_total_w);

  // Uniform scale: the same factor applies to every cell so relative
  // proportions across rows stay intact.
  let mut scale = target_w / default_total_w;

  // If an explicit height is also given, shrink further so the whole
  // grid fits. The natural total height is sum of row maxes plus the
  // per-row v_gap estimated from the average natural row height.
  if let Some(total_h) = opts.target_height {
    let nat_total_h: f64 = row_nat_dims.iter().map(|(_, h)| *h).sum();
    let num_nonempty = row_nat_dims.iter().filter(|(_, h)| *h > 0.0).count();
    if nat_total_h > 0.0 && num_nonempty > 0 {
      let avg_row_h = nat_total_h / num_nonempty as f64;
      let v_gap_nat = opts.v_spacing.to_px(avg_row_h);
      let nat_total_h_with_gaps =
        nat_total_h + (num_nonempty as f64 - 1.0).max(0.0) * v_gap_nat;
      let scale_h = total_h / nat_total_h_with_gaps.max(1e-6);
      // Use whichever constraint is tighter so both dimensions fit.
      scale = scale.min(scale_h);
    }
  }

  // Scaled per-cell dimensions (enforce a minimum so pathological
  // zero-sized inputs don't vanish entirely).
  let scaled_rows: Vec<Vec<(f64, f64)>> = parsed_rows
    .iter()
    .map(|row| {
      row
        .iter()
        .map(|(_, _, nw, nh)| ((nw * scale).max(1.0), (nh * scale).max(1.0)))
        .collect()
    })
    .collect();

  // A column is as wide as its widest cell, in every row — that is what
  // makes a `Grid`'s columns line up when the rows hold different things
  // (a picture in one row, a caption under it in the next). Rows of
  // uniform cells, the `GraphicsGrid` case, are unaffected.
  let mut col_w: Vec<f64> = vec![0.0; max_cols];
  for row in &scaled_rows {
    for (j, (w, _)) in row.iter().enumerate() {
      col_w[j] = col_w[j].max(*w);
    }
  }

  // Horizontal gap: `Scaled` is resolved against the average cell width
  // over the whole grid, so every row uses the same column pitch.
  let cell_count = scaled_rows.iter().map(Vec::len).sum::<usize>().max(1);
  let avg_cell_w: f64 =
    scaled_rows.iter().flatten().map(|(w, _)| *w).sum::<f64>()
      / cell_count as f64;
  let h_gap = opts.h_spacing.to_px(avg_cell_w);
  let col_x: Vec<f64> = col_w
    .iter()
    .scan(0.0_f64, |x, w| {
      let here = *x;
      *x += w + h_gap;
      Some(here)
    })
    .collect();

  // Per-cell layout: keep natural aspect ratios, centre each cell in its
  // column and vertically centre shorter cells within their row.
  let mut row_layouts: Vec<GridRowLayout> = Vec::new();

  for (parsed_row, cell_dims) in parsed_rows.iter().zip(scaled_rows.iter()) {
    if parsed_row.is_empty() {
      row_layouts.push(GridRowLayout {
        cells: Vec::new(),
        row_h: 0.0,
      });
      continue;
    }

    let row_h = cell_dims
      .iter()
      .map(|(_, h)| *h)
      .fold(0.0_f64, f64::max)
      .max(10.0);

    let mut cells = Vec::with_capacity(parsed_row.len());
    for (j, ((vb, inner, _, _), (cw, ch))) in
      parsed_row.iter().zip(cell_dims.iter()).enumerate()
    {
      let y_off = ((row_h - ch) / 2.0).max(0.0);
      let x = col_x[j] + (col_w[j] - cw) / 2.0;
      cells.push(LayoutCell {
        x,
        y_off,
        w: *cw,
        h: *ch,
        view_box: vb.clone(),
        inner: inner.clone(),
      });
    }

    row_layouts.push(GridRowLayout { cells, row_h });
  }

  // Compute total dimensions. The grid is as wide as its columns, which
  // a row that stops short of the last column does not shorten.
  let total_width = col_x
    .iter()
    .zip(col_w.iter())
    .map(|(x, w)| x + w)
    .fold(0.0_f64, f64::max);

  let v_gap = if row_layouts.is_empty() {
    0.0
  } else {
    let avg_h = row_layouts.iter().map(|r| r.row_h).sum::<f64>()
      / row_layouts.len().max(1) as f64;
    opts.v_spacing.to_px(avg_h)
  };
  let total_height: f64 = row_layouts.iter().map(|r| r.row_h).sum::<f64>()
    + (row_layouts.iter().filter(|r| r.row_h > 0.0).count() as f64 - 1.0)
      .max(0.0)
      * v_gap;

  Some(GridLayout {
    rows: row_layouts,
    total_width,
    total_height,
    v_gap,
  })
}

/// Write a single `<svg>` cell element to `out`.
fn write_cell_svg(out: &mut String, cell: &LayoutCell, y: f64) {
  out.push_str(&format!(
    "<svg x=\"{:.0}\" y=\"{:.0}\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"{}\">\n",
    cell.x,
    y + cell.y_off,
    cell.w,
    cell.h,
    cell.view_box,
  ));
  out.push_str(&cell.inner);
  out.push_str("</svg>\n");
}

/// Combine SVG strings in a grid layout with configurable spacing and size.
/// See `compute_grid_layout` for the sizing rules.
fn combine_svgs_grid(
  rows: &[Vec<String>],
  opts: &LayoutOptions,
) -> Option<String> {
  let layout = compute_grid_layout(rows, opts)?;

  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    layout.total_width.ceil() as u32,
    layout.total_height.ceil() as u32,
    layout.total_width.ceil() as u32,
    layout.total_height.ceil() as u32,
  ));

  // Track per-row vertical positions so frame lines can be drawn at
  // the correct y boundaries even when rows have different heights.
  let mut row_y_starts: Vec<f64> = Vec::with_capacity(layout.rows.len());
  let mut y = 0.0_f64;
  for row in &layout.rows {
    if row.row_h <= 0.0 {
      row_y_starts.push(y);
      continue;
    }
    row_y_starts.push(y);
    for cell in &row.cells {
      write_cell_svg(&mut svg, cell, y);
    }
    y += row.row_h + layout.v_gap;
  }

  if opts.frame != FrameSetting::None {
    draw_frame_lines(
      &mut svg,
      &layout,
      &row_y_starts,
      opts.frame == FrameSetting::All,
    );
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// Draw frame lines for a GraphicsRow/Column/Grid. When `all` is true,
/// draw lines on every cell boundary (Frame -> All); otherwise only the
/// outer rectangle (Frame -> True).
fn draw_frame_lines(
  out: &mut String,
  layout: &GridLayout,
  row_y_starts: &[f64],
  all: bool,
) {
  let total_w = layout.total_width;
  let total_h = layout.total_height;
  let stroke = "rgb(0,0,0)";
  let sw = 1.0_f64;

  let line = |out: &mut String, x1: f64, y1: f64, x2: f64, y2: f64| {
    out.push_str(&format!(
      "<line x1=\"{x1:.2}\" y1=\"{y1:.2}\" x2=\"{x2:.2}\" y2=\"{y2:.2}\" \
       stroke=\"{stroke}\" stroke-width=\"{sw}\" stroke-linecap=\"square\"/>\n"
    ));
  };

  // Outer border
  line(out, 0.0, 0.0, total_w, 0.0);
  line(out, 0.0, total_h, total_w, total_h);
  line(out, 0.0, 0.0, 0.0, total_h);
  line(out, total_w, 0.0, total_w, total_h);

  if !all {
    return;
  }

  // Inner row dividers: draw at the top of each row after the first
  for (i, row) in layout.rows.iter().enumerate() {
    if i == 0 || row.row_h <= 0.0 {
      continue;
    }
    let y = row_y_starts[i];
    line(out, 0.0, y, total_w, y);
  }

  // Inner column dividers: use the widest row to place vertical lines
  // at every cell boundary. Cells share an x-coordinate scheme inside
  // a row, so taking max-cells row gives the most granular boundaries.
  let widest_row = layout
    .rows
    .iter()
    .max_by_key(|r| r.cells.len())
    .map_or(&[][..], |r| &r.cells[..]);
  for (i, cell) in widest_row.iter().enumerate() {
    if i == 0 {
      continue;
    }
    let x = cell.x;
    line(out, x, 0.0, x, total_h);
  }
}

/// Default layout options (Scaled[0.1] spacing, natural sizing).
fn default_layout_options() -> LayoutOptions {
  LayoutOptions {
    h_spacing: SpacingSpec::default_val(),
    v_spacing: SpacingSpec::default_val(),
    target_width: None,
    target_height: None,
    frame: FrameSetting::None,
    spacings_explicit: false,
  }
}

/// Maximum default total row width (in pixels) before cells are
/// re-rendered at a smaller per-cell size. With `DEFAULT_WIDTH = 360`,
/// this gives 3 cells at native size before shrinking kicks in.
const GRID_ROW_CAP_WIDTH: f64 = 1080.0;

/// Compute the per-cell pixel width for a row of `n` cells. When an
/// explicit total width is given, divide it evenly; otherwise use the
/// natural cell width up to `GRID_ROW_CAP_WIDTH` total.
///
/// This is used to pre-render each child Plot/BarChart/etc. at a size
/// that matches its final display footprint, so text and strokes stay
/// at their intended pixel dimensions instead of being scaled down to
/// sub-legible sizes when the row is packed with many items.
fn compute_per_cell_width(n: usize, explicit_total: Option<f64>) -> i128 {
  let n_f = n.max(1) as f64;
  let natural = DEFAULT_WIDTH as f64;
  let total =
    explicit_total.unwrap_or_else(|| (natural * n_f).min(GRID_ROW_CAP_WIDTH));
  let per = (total / n_f).round() as i128;
  per.max(1)
}

/// Whitelist of function heads that are known to produce graphics and
/// honor the `ImageSize` option. Used to avoid injecting `ImageSize`
/// into arbitrary user functions (which might error or behave oddly on
/// unknown options) while still catching the common plot/chart cases.
fn is_graphics_producing_head(name: &str) -> bool {
  matches!(
    name,
    // Core graphics primitives
    "Graphics"
      | "Graphics3D"
      | "Image"
      // 2-D / 3-D plots
      | "Plot"
      | "Plot3D"
      | "LogPlot"
      | "LogLogPlot"
      | "LogLinearPlot"
      | "ParametricPlot"
      | "ParametricPlot3D"
      | "PolarPlot"
      | "ContourPlot"
      | "ContourPlot3D"
      | "DensityPlot"
      | "DensityPlot3D"
      | "RegionPlot"
      | "RegionPlot3D"
      | "DiscretePlot"
      | "DiscretePlot3D"
      | "StreamPlot"
      | "VectorPlot"
      | "VectorPlot3D"
      | "NumberLinePlot"
      | "TimelinePlot"
      | "Dendrogram"
      | "ComplexPlot"
      | "ComplexPlot3D"
      | "ComplexListPlot"
      | "ComplexArrayPlot"
      | "ComplexContourPlot"
      | "ComplexRegionPlot"
      | "ComplexVectorPlot"
      | "ComplexStreamPlot"
      // List plots
      | "ListPlot"
      | "ListLinePlot"
      | "ListLogPlot"
      | "ListLogLogPlot"
      | "ListLogLinearPlot"
      | "ListStepPlot"
      | "StackedListPlot"
      | "ListContourPlot"
      | "ListDensityPlot"
      | "ListPolarPlot"
      | "TernaryListPlot"
      | "ListStreamPlot"
      | "ListVectorPlot"
      | "ListPlot3D"
      | "ListLinePlot3D"
      | "DateListPlot"
      | "DateListLogPlot"
      | "DateListStepPlot"
      // Charts
      | "BarChart"
      | "BarChart3D"
      | "PieChart"
      | "PieChart3D"
      | "Histogram"
      | "Histogram3D"
      | "DensityHistogram"
      | "DateHistogram"
      | "BubbleChart"
      | "BubbleChart3D"
      | "BubbleHistogram"
      | "BoxWhiskerChart"
      | "DistributionChart"
      | "SectorChart"
      | "CandlestickChart"
      // Arrays / matrices
      | "ArrayPlot"
      | "ArrayPlot3D"
      | "MatrixPlot"
      // Graphs / trees / meshes
      | "Graph"
      | "TreeForm"
      | "TreePlot"
      | "TreeGraph"
      | "VoronoiMesh"
      | "DelaunayMesh"
      // Misc
      | "BodePlot"
      | "AbsArgPlot"
      | "MoleculePlot"
      | "Framed"
  )
}

/// Return a copy of `expr` with `ImageSize -> size` appended if `expr`
/// is a FunctionCall with a known graphics-producing head that doesn't
/// already carry an `ImageSize` option. Other expression shapes
/// (identifiers, literals, already-evaluated Graphics, user functions)
/// are returned unchanged so we never override user options or break
/// unrelated calls.
fn with_default_image_size(expr: &Expr, size: i128) -> Expr {
  let Expr::FunctionCall { name, args } = expr else {
    return expr.clone();
  };
  if !is_graphics_producing_head(name) {
    return expr.clone();
  }
  let has_image_size = args.iter().any(|a| {
    matches!(a, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ImageSize"))
  });
  if has_image_size {
    return expr.clone();
  }
  let mut new_args = args.clone();
  new_args.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("ImageSize".to_string())),
    replacement: Box::new(Expr::Integer(size)),
  });
  Expr::FunctionCall {
    name: name.clone(),
    args: new_args,
  }
}

/// If `expr` is a top-level list that looks like a list of graphics
/// (1-D `{p1, p2, ...}` or 2-D `{{p1, p2}, {p3, p4}}`), return a new
/// expression with `ImageSize -> per_cell_w` injected on each child so
/// the items get re-rendered at the correct per-cell size during
/// evaluation. Returns `None` for anything else so the caller can
/// evaluate the original expression unchanged.
pub fn inject_image_size_for_list_of_graphics(expr: &Expr) -> Option<Expr> {
  // 2-D list (grid): rows × cells
  if let Expr::List(rows) = expr
    && !rows.is_empty()
    && rows.iter().all(|r| matches!(r, Expr::List(_)))
  {
    let any_graphic = rows.iter().any(|r| {
      if let Expr::List(items) = r {
        items.iter().any(|it| {
          matches!(it, Expr::FunctionCall { name, .. }
            if is_graphics_producing_head(name))
        })
      } else {
        false
      }
    });
    if !any_graphic {
      return None;
    }
    let max_cols = rows
      .iter()
      .map(|r| {
        if let Expr::List(items) = r {
          items.len()
        } else {
          0
        }
      })
      .max()
      .unwrap_or(0);
    if max_cols == 0 {
      return None;
    }
    let per_cell_w = compute_per_cell_width(max_cols, None);
    let new_rows: Vec<Expr> = rows
      .iter()
      .map(|row| {
        if let Expr::List(items) = row {
          Expr::List(
            items
              .iter()
              .map(|it| with_default_image_size(it, per_cell_w))
              .collect(),
          )
        } else {
          row.clone()
        }
      })
      .collect();
    return Some(Expr::List(new_rows.into()));
  }

  // 1-D list
  if let Expr::List(items) = expr
    && !items.is_empty()
  {
    let any_graphic = items.iter().any(|it| {
      matches!(it, Expr::FunctionCall { name, .. }
        if is_graphics_producing_head(name))
    });
    if !any_graphic {
      return None;
    }
    let per_cell_w = compute_per_cell_width(items.len(), None);
    let new_items: Vec<Expr> = items
      .iter()
      .map(|it| with_default_image_size(it, per_cell_w))
      .collect();
    return Some(Expr::List(new_items.into()));
  }

  None
}

/// Evaluate each item with `ImageSize -> per_cell_w` injected (when the
/// item is a rewritable FunctionCall) and collect the resulting SVGs.
/// Items that are already evaluated (variables, literals) pass through
/// unchanged and are rendered at their natural size.
fn render_items_at_size(items: &[Expr], per_cell_w: i128) -> Vec<String> {
  items
    .iter()
    .filter_map(|item| {
      let rewritten = with_default_image_size(item, per_cell_w);
      let evaluated = evaluate_expr_to_expr(&rewritten).ok()?;
      let svg = crate::evaluator::expr_to_svg(&evaluated);
      (!svg.is_empty()).then_some(svg)
    })
    .collect()
}

/// GraphicsRow[{g1, g2, ...}] or GraphicsRow[{g1, g2, ...}, opts...]
/// Arranges graphics side-by-side in a single row.
///
/// When the first argument is a literal list of function calls, each
/// child is re-rendered with `ImageSize -> per_cell_w` injected so text
/// and strokes come out at their intended pixel sizes instead of being
/// scaled down to illegibility when the row is packed with many items.
pub fn graphics_row_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let opts = parse_layout_options(&args[1..]);

  // Prefer rewriting the unevaluated items so we can re-render each at
  // the final per-cell size. Fall back to post-evaluation scaling when
  // the argument is a variable / computed list whose items are already
  // Graphics objects and can't be re-rendered.
  let svgs = if let Expr::List(items) = &args[0] {
    if items.is_empty() {
      return Ok(crate::graphics_result(
        "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
      ));
    }
    let per_cell_w = compute_per_cell_width(items.len(), opts.target_width);
    render_items_at_size(items, per_cell_w)
  } else {
    let list_expr = evaluate_expr_to_expr(&args[0])?;
    let items = match &list_expr {
      Expr::List(items) => items.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "GraphicsRow expects a list as its first argument".into(),
        ));
      }
    };
    extract_svgs_from_list(&items)
  };

  if svgs.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let row = vec![svgs];
  match combine_svgs_grid(&row, &opts) {
    Some(combined) => {
      crate::clear_captured_graphics();
      Ok(crate::graphics_result(combined))
    }
    None => Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    )),
  }
}

/// GraphicsColumn[{g1, g2, ...}] or GraphicsColumn[{g1, g2, ...}, opts...]
/// Arranges graphics vertically in a single column.
pub fn graphics_column_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let opts = parse_layout_options(&args[1..]);

  // A column has one cell per row, so each cell takes the full column
  // width. Re-render at DEFAULT_WIDTH (or the explicit ImageSize) so
  // text stays legible regardless of how many rows there are.
  let svgs = if let Expr::List(items) = &args[0] {
    if items.is_empty() {
      return Ok(crate::graphics_result(
        "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
      ));
    }
    let per_cell_w = compute_per_cell_width(1, opts.target_width);
    render_items_at_size(items, per_cell_w)
  } else {
    let list_expr = evaluate_expr_to_expr(&args[0])?;
    let items = match &list_expr {
      Expr::List(items) => items.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "GraphicsColumn expects a list as its first argument".into(),
        ));
      }
    };
    extract_svgs_from_list(&items)
  };

  if svgs.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  // Each SVG becomes its own row (single-column layout)
  let rows: Vec<Vec<String>> = svgs.into_iter().map(|s| vec![s]).collect();
  match combine_svgs_grid(&rows, &opts) {
    Some(combined) => {
      crate::clear_captured_graphics();
      Ok(crate::graphics_result(combined))
    }
    None => Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    )),
  }
}

/// GraphicsGrid[{{g1, g2}, {g3, g4}}, opts...]
/// Arranges graphics in a 2D grid.
pub fn graphics_grid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let opts = parse_layout_options(&args[1..]);

  // GraphicsGrid is held by the dispatcher, so args[0] arrives
  // unevaluated. Resolve it to a list-of-lists before laying out the
  // grid so that built-up forms like Table[...] expand into individual
  // cells we can re-render at per-cell size.
  let grid_expr_owned;
  let grid_list_ref: &Expr = if let Expr::List(_) = &args[0] {
    &args[0]
  } else {
    grid_expr_owned = evaluate_expr_to_expr(&args[0])?;
    &grid_expr_owned
  };
  let Expr::List(outer_items) = grid_list_ref else {
    return Err(InterpreterError::EvaluationError(
      "GraphicsGrid expects a list of lists as its first argument".into(),
    ));
  };

  // Determine the widest row so every cell in the grid is re-rendered
  // at the same per-cell width (grids typically expect uniform cells).
  let max_cols = outer_items
    .iter()
    .map(|item| match item {
      Expr::List(row_items) => row_items.len(),
      _ => 1,
    })
    .max()
    .unwrap_or(0);
  if max_cols == 0 {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let per_cell_w = compute_per_cell_width(max_cols, opts.target_width);
  let rows: Vec<Vec<String>> = outer_items
    .iter()
    .map(|item| match item {
      Expr::List(row_items) => render_items_at_size(row_items, per_cell_w),
      other => render_items_at_size(std::slice::from_ref(other), per_cell_w),
    })
    .collect();

  // Check if we have any SVGs at all
  if rows.iter().all(std::vec::Vec::is_empty) {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  match combine_svgs_grid(&rows, &opts) {
    Some(combined) => {
      crate::clear_captured_graphics();
      Ok(crate::graphics_result(combined))
    }
    None => Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    )),
  }
}

/// PlotGrid[{{p1, p2}, {p3, p4}}, opts...]
/// Arranges a matrix of plots in a shared grid. Like GraphicsGrid, each
/// cell is re-rendered at a uniform per-cell width so the plots stay
/// legible; the resulting composite renders as a single `-Graphics-`
/// object. PlotGrid is tailored to plots (which already carry their own
/// frames and axes), so the layout logic is shared with GraphicsGrid.
pub fn plot_grid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  graphics_grid_ast(args)
}

// ── Tabular SVG rendering ────────────────────────────────────────────

/// Convert a Tabular[data, schema] to an SVG table.
/// The data can be a list of lists, a list of associations, or a
/// column-oriented association.
pub fn tabular_to_svg(data: &Expr, schema: &Expr) -> Option<String> {
  // Extract column keys from schema
  let col_keys = extract_tabular_column_keys(schema);

  match data {
    Expr::List(rows) if !rows.is_empty() => {
      if rows.iter().all(|r| matches!(r, Expr::Association(_))) {
        tabular_list_of_assocs_to_svg(rows, &col_keys)
      } else if rows.iter().all(|r| matches!(r, Expr::List(_))) {
        tabular_list_of_lists_to_svg(rows, &col_keys)
      } else {
        // Flat list — single column
        tabular_flat_list_to_svg(rows, &col_keys)
      }
    }
    Expr::Association(pairs) if !pairs.is_empty() => {
      tabular_column_assoc_to_svg(pairs, &col_keys)
    }
    _ => None,
  }
}

/// Extract column keys from a TabularSchema expression.
fn extract_tabular_column_keys(schema: &Expr) -> Vec<String> {
  if let Expr::FunctionCall { name, args } = schema
    && name == "TabularSchema"
    && !args.is_empty()
    && let Expr::Association(pairs) = &args[0]
  {
    for (k, v) in pairs {
      let key_str = match k {
        Expr::String(s) => s.as_str(),
        Expr::Identifier(s) => s.as_str(),
        _ => continue,
      };
      if key_str == "ColumnKeys"
        && let Expr::List(keys) = v
      {
        return keys.iter().map(expr_to_svg_markup).collect();
      }
    }
  }
  vec![]
}

/// Render Tabular from a list of associations as SVG.
fn tabular_list_of_assocs_to_svg(
  rows: &[Expr],
  col_keys: &[String],
) -> Option<String> {
  if col_keys.is_empty() {
    return None;
  }

  // Build data grid aligned to column keys
  let grid: Vec<Vec<Expr>> = rows
    .iter()
    .map(|item| {
      if let Expr::Association(pairs) = item {
        col_keys
          .iter()
          .map(|h| {
            pairs
              .iter()
              .find(|(k, _)| expr_to_svg_markup(k) == *h)
              .map_or(call0("Missing"), |(_, v)| v.clone())
          })
          .collect()
      } else {
        vec![]
      }
    })
    .collect();

  render_tabular_svg_grid(col_keys, &grid, true)
}

/// Render Tabular from a list of lists as SVG.
fn tabular_list_of_lists_to_svg(
  rows: &[Expr],
  col_keys: &[String],
) -> Option<String> {
  let grid: Vec<Vec<Expr>> = rows
    .iter()
    .map(|r| {
      if let Expr::List(items) = r {
        items.to_vec()
      } else {
        vec![]
      }
    })
    .collect();

  let has_named_cols = !col_keys.is_empty()
    && !col_keys
      .iter()
      .enumerate()
      .all(|(i, k)| k == &format!("{}", i + 1));

  render_tabular_svg_grid(col_keys, &grid, has_named_cols)
}

/// Render Tabular from a flat list as SVG (single column).
fn tabular_flat_list_to_svg(
  items: &[Expr],
  col_keys: &[String],
) -> Option<String> {
  let grid: Vec<Vec<Expr>> = items.iter().map(|e| vec![e.clone()]).collect();
  let has_named_cols = !col_keys.is_empty()
    && !col_keys
      .iter()
      .enumerate()
      .all(|(i, k)| k == &format!("{}", i + 1));
  render_tabular_svg_grid(col_keys, &grid, has_named_cols)
}

/// Render Tabular from a column-oriented association as SVG.
/// <|"a" -> {1,2,3}, "b" -> {4,5,6}|>
fn tabular_column_assoc_to_svg(
  pairs: &[(Expr, Expr)],
  col_keys: &[String],
) -> Option<String> {
  // Determine number of rows from the longest column
  let num_rows = pairs
    .iter()
    .map(|(_, v)| {
      if let Expr::List(items) = v {
        items.len()
      } else {
        1
      }
    })
    .max()
    .unwrap_or(0);

  // Build grid by transposing column data to row data
  let mut grid: Vec<Vec<Expr>> = Vec::with_capacity(num_rows);
  for i in 0..num_rows {
    let row: Vec<Expr> = pairs
      .iter()
      .map(|(_, v)| {
        if let Expr::List(items) = v {
          items.get(i).cloned().unwrap_or(call0("Missing"))
        } else if i == 0 {
          v.clone()
        } else {
          call0("Missing")
        }
      })
      .collect();
    grid.push(row);
  }

  render_tabular_svg_grid(col_keys, &grid, true)
}

/// Core SVG rendering for Tabular: a table with optional column headers,
/// row numbers in a left column, and grid lines.
fn render_tabular_svg_grid(
  col_keys: &[String],
  grid: &[Vec<Expr>],
  show_headers: bool,
) -> Option<String> {
  if grid.is_empty() {
    return None;
  }

  let num_data_rows = grid.len();
  let num_cols = if col_keys.is_empty() {
    grid.iter().map(std::vec::Vec::len).max().unwrap_or(0)
  } else {
    col_keys.len()
  };

  if num_cols == 0 {
    return None;
  }

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 16.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;
  let header_row_height = font_size + pad_y + 2.0;

  // Row-number column width (for 1-based row indices)
  let max_row_digits = format!("{num_data_rows}").len().max(1) as f64;
  let row_num_col_w = max_row_digits * char_width + pad_x;

  // Compute data column widths from headers and data
  let mut col_widths: Vec<f64> = if show_headers && !col_keys.is_empty() {
    col_keys
      .iter()
      .map(|h| h.len() as f64 * char_width + pad_x)
      .collect()
  } else {
    vec![pad_x; num_cols]
  };

  for row in grid {
    for (j, cell) in row.iter().enumerate() {
      if j < num_cols && j < col_widths.len() {
        let w = estimate_display_width(cell) * char_width + pad_x;
        if w > col_widths[j] {
          col_widths[j] = w;
        }
      }
    }
  }

  let data_width: f64 = col_widths.iter().sum();
  let total_width = row_num_col_w + data_width;
  let num_header_rows = usize::from(show_headers && !col_keys.is_empty());
  let total_height: f64 = if num_header_rows > 0 {
    header_row_height + (num_data_rows as f64) * row_height
  } else {
    (num_data_rows as f64) * row_height
  };

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let t = theme();

  // Row-number column background (light blue-gray)
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{row_num_col_w:.1}\" height=\"{total_height:.1}\" fill=\"{}\"/>\n",
    t.table_row_num_bg
  ));

  // Header row background (if applicable)
  if num_header_rows > 0 {
    svg.push_str(&format!(
      "<rect x=\"{row_num_col_w:.1}\" y=\"0\" width=\"{data_width:.1}\" height=\"{header_row_height:.1}\" fill=\"{}\"/>\n",
      t.table_header_bg
    ));
    // Also extend the row-number column header background
    svg.push_str(&format!(
      "<rect x=\"0\" y=\"0\" width=\"{row_num_col_w:.1}\" height=\"{header_row_height:.1}\" fill=\"{}\"/>\n",
      t.table_row_num_header_bg
    ));
  }

  // Header text (bold)
  let text_fill = t.text_primary;
  if num_header_rows > 0 && !col_keys.is_empty() {
    let mut x_offset: f64 = row_num_col_w;
    for (j, header) in col_keys.iter().enumerate() {
      if j >= col_widths.len() {
        break;
      }
      let col_w = col_widths[j];
      let cx = x_offset + col_w / 2.0;
      let cy = header_row_height / 2.0;
      svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{header}</text>\n"
      ));
      x_offset += col_w;
    }
  }

  // Data rows
  let y_start: f64 = if num_header_rows > 0 {
    header_row_height
  } else {
    0.0
  };
  let row_num_fill = t.text_muted;
  let mut y_offset: f64 = y_start;
  for (i, row) in grid.iter().enumerate() {
    // Row number (1-based, in left column)
    let row_num = format!("{}", i + 1);
    let rx = row_num_col_w / 2.0;
    let cy = y_offset + row_height / 2.0;
    svg.push_str(&format!(
      "<text x=\"{rx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{row_num_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{row_num}</text>\n"
    ));

    // Data cells
    let mut x_offset: f64 = row_num_col_w;
    for (j, cell) in row.iter().enumerate() {
      if j < num_cols && j < col_widths.len() {
        let col_w = col_widths[j];
        let cx = x_offset + col_w / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(cell)
        ));
        x_offset += col_w;
      }
    }
    y_offset += row_height;
  }

  // Grid lines
  let border_color = t.table_border_strong;
  let light_color = t.table_border_light;
  // Horizontal lines
  let num_total_rows = num_header_rows + num_data_rows;
  let mut y = 0.0_f64;
  for i in 0..=num_total_rows {
    let is_border =
      i == 0 || i == num_total_rows || (num_header_rows > 0 && i == 1);
    let stroke_width = if is_border { "1.5" } else { "0.5" };
    let color = if is_border { border_color } else { light_color };
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>\n"
    ));
    if num_header_rows > 0 && i == 0 {
      y += header_row_height;
    } else if i < num_total_rows {
      y += row_height;
    }
  }

  // Vertical lines: outer borders + separator after row-number column
  svg.push_str(&format!(
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{row_num_col_w:.1}\" y1=\"0\" x2=\"{row_num_col_w:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"{border_color}\" stroke-width=\"1.5\"/>\n"
  ));

  svg.push_str("</svg>");
  Some(svg)
}

/// Resolve a `Column`/`Row` display item for rendering: release a held
/// `Dynamic[…]` (a static rendering shows its current value; front-ends
/// re-evaluate the whole display on interaction) and unwrap a top-level
/// `Text[…]` wrapper, which displays as its content. Any graphics that the
/// `Dynamic` evaluation captures are embedded in the surrounding layout, not
/// standalone outputs, so they are dropped from the capture buffer.
/// Peel the wrappers that say how to *set* what they hold rather than what
/// to show: `Text[…]`, an `Item[…]` cell, a `Pane[…]` box, and the form
/// wrappers. They nest in any order — `Item[Text@TraditionalForm@Framed[…],
/// …]` — so peel until what is left is the thing that actually draws.
/// `Pane[content, opts…]` only reserves an area for `content`; what is
/// shown is `content`, so a display pass has to reach through it (a grid
/// cell holding one used to print as `Pane[…]` source).
///
/// A `TraditionalForm` around a picture or a layout only asks for its
/// contents to be set traditionally, and the layout renderer takes it from
/// there. Around an ordinary expression the form *is* what to show —
/// `p(x) = 1 + x`, not `p(x) == 1 + x` — so the wrapper is put back, for
/// the caller to typeset through the traditional box builder.
pub fn unwrap_display_wrappers(expr: &Expr) -> Expr {
  let mut current = expr.clone();
  let mut traditional = false;
  while let Expr::FunctionCall { name, args } = &current {
    let pass_through = match name.as_str() {
      "Text" | "TraditionalForm" | "StandardForm" | "DisplayForm" => {
        args.len() == 1
      }
      // `Deploy[expr]` only makes its content non-selectable; it draws
      // exactly as the content does.
      "Deploy" => args.len() == 1,
      "Item" | "Pane" => !args.is_empty(),
      _ => false,
    };
    if !pass_through {
      break;
    }
    traditional |= name == "TraditionalForm";
    let inner = args[0].clone();
    current = inner;
  }
  if traditional
    && !crate::evaluator::lays_out_a_graphic(&current)
    && !matches!(&current, Expr::FunctionCall { name, .. }
      if matches!(name.as_str(), "Grid" | "Column" | "Row" | "Framed"))
  {
    current = call1("TraditionalForm", current);
  }
  current
}

fn resolve_display_item(expr: &Expr) -> Expr {
  let mut current = expr.clone();
  if let Expr::FunctionCall { name, args } = &current
    && name == "Dynamic"
    && !args.is_empty()
  {
    let captured = crate::captured_graphics_count();
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&args[0]);
    crate::truncate_captured_graphics(captured);
    if let Ok(inner) = evaluated {
      current = inner;
    }
  }
  current = unwrap_display_wrappers(&current);
  // `InputForm[expr]` displays as the InputForm text of `expr`.
  if let Expr::FunctionCall { name, args } = &current
    && name == "InputForm"
    && args.len() == 1
  {
    current = Expr::Raw(crate::syntax::expr_to_input_form(&args[0]));
  }
  current
}

/// Distribute a `Style[…]`'s directives over the items of the layout it
/// wraps, so `Style[Row[{a, b}], Bold]` becomes `Row[{Style[a, Bold],
/// Style[b, Bold]}]`. Only the item list is rewritten; a separator or
/// option argument keeps its place. `None` when the wrapped expression is
/// not a layout with a list of items.
pub(crate) fn style_pushed_into_layout(
  inner: &Expr,
  directives: &[Expr],
) -> Option<Expr> {
  if directives.is_empty() {
    return None;
  }
  let Expr::FunctionCall { name, args } = inner else {
    return None;
  };
  if !matches!(name.as_str(), "Row" | "Column" | "Grid") || args.is_empty() {
    return None;
  }
  let Expr::List(items) = &args[0] else {
    return None;
  };
  // A picture takes no text style: wrapping it would only hide it from the
  // row/column renderer, which recognises a graphic by its own head and
  // would fall back to the `-Graphics-` placeholder text.
  let restyle = |e: &Expr| {
    if matches!(e, Expr::Graphics { .. })
      || crate::evaluator::lays_out_a_graphic(e)
    {
      return e.clone();
    }
    Expr::FunctionCall {
      name: "Style".to_string(),
      args: std::iter::once(e.clone())
        .chain(directives.iter().cloned())
        .collect::<Vec<_>>()
        .into(),
    }
  };
  // A `Grid`'s items are rows of cells; everything else is a flat list.
  let styled: Vec<Expr> = items
    .iter()
    .map(|item| match item {
      Expr::List(cells) if name == "Grid" => {
        Expr::List(cells.iter().map(&restyle).collect())
      }
      other => restyle(other),
    })
    .collect();
  let mut new_args = vec![Expr::List(styled.into())];
  new_args.extend(args[1..].iter().cloned());
  Some(Expr::FunctionCall {
    name: name.clone(),
    args: new_args.into(),
  })
}

/// Typeset `expr` in TraditionalForm as a standalone SVG, through the same
/// box builder and box layout the top-level display uses. `None` when the
/// boxes lay out to nothing.
fn form_box_svg(expr: &Expr) -> Option<String> {
  let boxes =
    crate::evaluator::dispatch::complex_and_special::expr_to_box_form_traditional(
      expr,
    );
  let boxes = crate::strip_number_precision_markers(&boxes);
  let layout = layout_box(&boxes, 14.0);
  if layout.width <= 0.0 || layout.height <= 0.0 {
    return None;
  }
  Some(layout_to_svg(&layout, theme().text_primary))
}

/// Render a resolved `Column`/`Row` item that is itself a layout construct
/// (`Row[…]` / `Column[…]`) to a nested SVG, so mixed text/graphics rows
/// compose instead of printing as raw InputForm text. `None` for other
/// expressions (rendered as a plain text line by the caller).
fn nested_layout_svg(expr: &Expr) -> Option<String> {
  if let Expr::FunctionCall { name, args } = expr {
    let args: Vec<Expr> = args.iter().cloned().collect();
    if args.is_empty() {
      return None;
    }
    match name.as_str() {
      "Row" => return row_to_svg(&args),
      "Column" => return column_to_svg(&args),
      // A `Grid` nested in a layout is laid out too, rather than printed
      // as its own expression text.
      "Grid" => return grid_svg_with_gaps(&args, &[]).ok(),
      // A `Framed[…]` box draws its border and its content; a `Column`
      // item that is one used to print as source.
      "Framed" if !args.is_empty() => return framed_to_svg(&args),
      // A sound item shows the sound box a notebook draws for it — a play
      // button and the waveform — rather than the `Play[…]` source text.
      "Sound" | "Play" => return crate::functions::sound::sound_svg(expr),
      // `TraditionalForm[expr]` is typeset in conventional notation —
      // `=` for `Equal`, `≤` for `LessEqual`, `sin(x)` for `Sin[x]`,
      // stacked fractions — by the same box pipeline the top-level
      // display uses.
      "TraditionalForm" if args.len() == 1 => {
        return form_box_svg(&args[0]);
      }
      // A styled layout keeps its layout, and its style: `Style[Row[{…}],
      // Bold, 20]` sets every item of the row, since a `Style` is inherited
      // by what it wraps. Pushing the directives inwards is what lets the
      // row renderer, which reads each item's own style, apply them.
      "Style" | "StyleForm" => {
        let inner = style_pushed_into_layout(&args[0], &args[1..])
          .unwrap_or_else(|| args[0].clone());
        return nested_layout_svg(&inner);
      }
      // A display wrapper that resolves to a picture (`Labeled[…]`,
      // `LocatorPane[…]`, `Dynamic[…]`) is drawn through the export path,
      // which unwraps it. Without this a `Column` item holding one was
      // written out as a line of source text.
      _ if crate::evaluator::lays_out_a_graphic(expr) => {
        let svg = crate::evaluator::expr_to_svg(expr);
        if svg.starts_with("<svg") {
          return Some(svg);
        }
      }
      _ => {}
    }
  }
  None
}

/// Render `Column[{expr1, expr2, ...}]` as an SVG with items stacked vertically.
/// Optionally accepts an alignment argument (Left, Center, Right); defaults to Left.
pub fn column_to_svg(args: &[Expr]) -> Option<String> {
  if args.is_empty() {
    return None;
  }

  // Extract items from the first argument (must be a List)
  let items = match &args[0] {
    Expr::List(items) => items.clone(),
    _ => return None,
  };

  if items.is_empty() {
    return None;
  }

  // Parse optional alignment: positional (`Column[{…}, Center]`) or the
  // option form (`Column[{…}, Alignment -> Center]`). Default: Left.
  let mut alignment = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(s) if s == "Center" => "middle",
      Expr::Identifier(s) if s == "Right" => "end",
      _ => "start", // Left or anything else
    }
  } else {
    "start"
  };
  for arg in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Alignment")
      && let Expr::Identifier(spec) = replacement.as_ref()
    {
      alignment = match spec.as_str() {
        "Center" => "middle",
        "Right" => "end",
        _ => "start",
      };
    }
  }

  // Parse optional spacing from third arg in ems (default: 0)
  let spacing_ems: f64 = if args.len() >= 3 {
    match &args[2] {
      Expr::Integer(n) => *n as f64,
      Expr::Real(f) => *f,
      _ => 0.0,
    }
  } else {
    0.0
  };

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 12.0;
  let pad_y: f64 = 8.0;
  let gap = spacing_ems * font_size;

  // An item is either a pre-rendered SVG (e.g. nested TableForm/Framed/Grid)
  // or a plain expression rendered as a single text line. A text item keeps
  // the appearance its own `Style[…]` asks for — a `Column`'s heading is
  // written `Style[…, Bold, 20]` and used to come out at the default size.
  enum Cell {
    Svg {
      svg: String,
      width: f64,
      height: f64,
    },
    Text(Expr),
  }

  let cells: Vec<Cell> = items
    .iter()
    .map(|item| {
      let resolved = resolve_display_item(item);
      match &resolved {
        Expr::Graphics { svg, .. } => {
          let (w, h) = parse_svg_wh(svg);
          Cell::Svg {
            svg: svg.clone(),
            width: w,
            height: h,
          }
        }
        _ => match nested_layout_svg(&resolved) {
          Some(svg) => {
            let (w, h) = parse_svg_wh(&svg);
            Cell::Svg {
              svg,
              width: w,
              height: h,
            }
          }
          None => Cell::Text(resolved),
        },
      }
    })
    .collect();

  // The appearance a text item's own `Style[…]` asks for; its size drives
  // both the cell's width and the row's height.
  fn text_style(e: &Expr, default_size: f64) -> (CellTextStyle<'_>, f64) {
    let style = extract_cell_style(e);
    let size = style.font_size.unwrap_or(default_size);
    (style, size)
  }

  // Compute column width from widest item
  let col_width: f64 = cells
    .iter()
    .map(|c| match c {
      Cell::Svg { width, .. } => *width,
      Cell::Text(e) => {
        let (style, fs) = text_style(e, font_size);
        estimate_display_width(style.content) * char_width * (fs / font_size)
          + pad_x
      }
    })
    .fold(0.0_f64, f64::max);

  // Per-row heights
  let row_heights: Vec<f64> = cells
    .iter()
    .map(|c| match c {
      Cell::Svg { height, .. } => *height,
      Cell::Text(e) => text_style(e, font_size).1 + pad_y,
    })
    .collect();

  let n = cells.len();
  let total_height: f64 =
    row_heights.iter().sum::<f64>() + gap * (n.saturating_sub(1) as f64);
  let total_width = col_width;

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;

  let mut svg = String::with_capacity(1024);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  // Compute text x-coordinate based on alignment
  let text_x: f64 = match alignment {
    "middle" => total_width / 2.0,
    "end" => total_width - pad_x / 2.0,
    _ => pad_x / 2.0, // "start"
  };

  let text_fill = theme().text_primary;
  let mut y_cursor: f64 = 0.0;
  for (i, cell) in cells.iter().enumerate() {
    let h = row_heights[i];
    match cell {
      Cell::Text(expr) => {
        let cy = y_cursor + h / 2.0;
        let (style, fs) = text_style(expr, font_size);
        let (fw, fst) = (style.font_weight, style.font_style);
        // An `Invisible[…]` row is measured and placed like any other but
        // painted with no fill, so the column keeps its height and width
        // while the row itself reads blank.
        let fill = if style.hidden {
          "none".to_string()
        } else {
          style
            .color
            .map_or_else(|| text_fill.to_string(), Color::to_svg_rgb)
        };
        // `Style[…, FontFamily -> "Times"]` picks the face; without one the
        // column keeps its monospace default.
        let ff = style.font_family.as_deref().unwrap_or("monospace");
        svg.push_str(&format!(
          "<text x=\"{text_x:.1}\" y=\"{cy:.1}\" font-family=\"{ff}\" font-size=\"{fs}\" font-weight=\"{fw}\" font-style=\"{fst}\" fill=\"{fill}\" text-anchor=\"{alignment}\" dominant-baseline=\"central\">{content}</text>\n",
          ff = svg_escape(ff),
          content = expr_to_svg_markup(style.content),
        ));
      }
      Cell::Svg {
        svg: child,
        width: cw,
        height: ch,
      } => {
        let x_off: f64 = match alignment {
          "middle" => (total_width - cw) / 2.0,
          "end" => total_width - cw,
          _ => 0.0,
        };
        // The child keeps its own coordinate space: a plot draws at a
        // multiple of its display size, so the nested `<svg>` needs the
        // child's viewBox or none of it lands inside the cell.
        let view_box = parse_svg_dimensions(child)
          .map_or_else(|| format!("0 0 {cw} {ch}"), |p| p.view_box);
        svg.push_str(&format!(
          "<svg x=\"{x_off:.1}\" y=\"{y_cursor:.1}\" width=\"{cw:.1}\" height=\"{ch:.1}\" viewBox=\"{view_box}\" preserveAspectRatio=\"xMidYMid meet\">\n"
        ));
        svg.push_str(strip_svg_wrapper(child));
        svg.push_str("</svg>\n");
      }
    }
    y_cursor += h + gap;
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// Render `Row[{items...}]` or `Row[{items...}, sep]` as a horizontal SVG layout.
/// When `sep` is `Spacer[n]`, uses `n` points of horizontal space between items.
/// When `sep` is any other expression, renders it as text between items.
pub fn row_to_svg(args: &[Expr]) -> Option<String> {
  if args.is_empty() {
    return None;
  }

  // First argument must be a list
  let Expr::List(items) = &args[0] else {
    return None;
  };

  if items.is_empty() {
    return None;
  }

  // Split the tail into an optional separator (a non-rule second argument)
  // and trailing option rules. A rule in separator position is an option,
  // not a separator; any further non-rule argument keeps the expression
  // symbolic, matching wolframscript (e.g. Row[{1, 2}, "|", "x"]).
  let (sep_expr, opt_args) = match args.get(1) {
    Some(a) if !crate::syntax::is_rule_expr(a) => (Some(a), &args[2..]),
    _ => (None, &args[1..]),
  };
  if !opt_args.iter().all(crate::syntax::is_rule_expr) {
    return None;
  }

  // Parse Alignment and ImageSize options. ImageSize widens the canvas;
  // the horizontal alignment places the content block inside it (Wolfram
  // typesets a per-item alignment list left-packed, so a list spec keeps
  // the default). The vertical alignment positions items of different
  // heights relative to each other.
  fn h_of(spec: &Expr) -> Option<&'static str> {
    match spec {
      Expr::Identifier(s) if s == "Left" => Some("left"),
      Expr::Identifier(s) if s == "Center" => Some("center"),
      Expr::Identifier(s) if s == "Right" => Some("right"),
      _ => None,
    }
  }
  fn v_of(spec: &Expr) -> Option<&'static str> {
    match spec {
      Expr::Identifier(s) if s == "Top" => Some("top"),
      Expr::Identifier(s) if s == "Center" => Some("center"),
      Expr::Identifier(s) if s == "Bottom" => Some("bottom"),
      _ => None,
    }
  }
  let mut target_width: Option<f64> = None;
  let mut h_align = "left";
  let mut v_align = "center";
  for opt in opt_args {
    let Some((key, val)) = option_kv(opt) else {
      continue;
    };
    match key {
      "ImageSize" => {
        target_width = match val {
          Expr::List(pair) if !pair.is_empty() => expr_to_f64(&pair[0]),
          other => expr_to_f64(other).or_else(|| {
            parse_image_size(other, DEFAULT_WIDTH, DEFAULT_HEIGHT)
              .map(|(w, _, _)| w as f64)
          }),
        };
      }
      "Alignment" => match val {
        Expr::List(pair) if pair.len() == 2 => {
          if let Some(h) = h_of(&pair[0]) {
            h_align = h;
          }
          if let Some(v) = v_of(&pair[1]) {
            v_align = v;
          }
        }
        single => {
          if let Some(h) = h_of(single) {
            h_align = h;
          } else if let Some(v) = v_of(single) {
            v_align = v;
          }
        }
      },
      _ => {}
    }
  }

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_y: f64 = 8.0;

  // An item is either a pre-rendered graphic embedded as a sub-SVG or an
  // expression rendered as text (with `Style` font/color directives applied).
  enum Cell {
    Svg {
      svg: String,
      width: f64,
      height: f64,
    },
    Text {
      markup: String,
      width: f64,
      height: f64,
      fill: String,
      size: f64,
      weight: String,
      slant: String,
      /// The face `FontFamily -> "…"` asked for; empty keeps the default.
      family: String,
    },
  }

  let make_text_cell = |item: &Expr| -> Cell {
    let mut st = StyleState::default();
    let mut color: Option<Color> = None;
    // An `Invisible[…]` item takes its content's width but is painted with
    // no fill, so the row keeps its shape while the item reads blank.
    let (item, hidden) = match peel_invisible(item) {
      Some(inner) => (inner, true),
      None => (item, false),
    };
    if let Expr::FunctionCall { name, args: sargs } = item
      && is_style_wrapper(name)
      && !sargs.is_empty()
    {
      for d in style_directives_in_application_order(&sargs[1..]) {
        if let Some(c) = parse_color(d) {
          color = Some(c);
        } else {
          apply_text_style_directive(d, &mut st);
        }
      }
    }
    let scale = st.font_size / font_size;
    Cell::Text {
      markup: expr_to_svg_markup(item),
      width: estimate_display_width(item) * char_width * scale,
      height: st.font_size + pad_y,
      fill: if hidden {
        "none".to_string()
      } else {
        color
          .map_or_else(|| theme().text_primary.to_string(), Color::to_svg_rgb)
      },
      size: st.font_size,
      weight: st.font_weight,
      slant: st.font_style,
      family: st.font_family,
    }
  };

  let cells: Vec<Cell> = items
    .iter()
    .map(|item| {
      let resolved = resolve_display_item(item);
      match &resolved {
        Expr::Graphics { svg, .. } => {
          let (w, h) = parse_svg_wh(svg);
          Cell::Svg {
            svg: svg.clone(),
            width: w,
            height: h,
          }
        }
        _ => match nested_layout_svg(&resolved) {
          Some(svg) => {
            let (w, h) = parse_svg_wh(&svg);
            Cell::Svg {
              svg,
              width: w,
              height: h,
            }
          }
          // `Spacer[n]` between items is blank horizontal space, the same
          // as in separator position — not something to print.
          None => match crate::syntax::spacer_width_pts(&resolved) {
            Some(width) => Cell::Text {
              markup: String::new(),
              width,
              height: font_size + pad_y,
              fill: theme().text_primary.to_string(),
              size: font_size,
              weight: "normal".to_string(),
              slant: "normal".to_string(),
              family: String::new(),
            },
            None => make_text_cell(&resolved),
          },
        },
      }
    })
    .collect();

  // Determine separator: either Spacer[n] (pixel gap) or rendered expression
  enum Separator {
    Gap(f64),          // pixel gap (from Spacer[n])
    Text(String, f64), // rendered text and its width
  }

  let separator = match sep_expr {
    Some(sep) => {
      if let Some(pts) = crate::syntax::spacer_width_pts(sep) {
        Separator::Gap(pts)
      } else {
        let text = expr_to_svg_markup(sep);
        let w = estimate_display_width(sep) * char_width;
        Separator::Text(text, w)
      }
    }
    None => Separator::Gap(0.0), // no separator
  };

  let sep_width = match &separator {
    Separator::Gap(g) => *g,
    Separator::Text(_, w) => *w,
  };

  let cell_width = |c: &Cell| match c {
    Cell::Svg { width, .. } | Cell::Text { width, .. } => *width,
  };
  let cell_height = |c: &Cell| match c {
    Cell::Svg { height, .. } | Cell::Text { height, .. } => *height,
  };

  let items_width: f64 = cells.iter().map(&cell_width).sum();
  let seps_total = if cells.len() > 1 {
    (cells.len() - 1) as f64 * sep_width
  } else {
    0.0
  };
  let content_w = items_width + seps_total;
  let total_h = cells
    .iter()
    .map(&cell_height)
    .fold(font_size + pad_y, f64::max);
  let canvas_w = target_width.map_or(content_w, |w| w.max(content_w));

  let svg_w = canvas_w.ceil().max(1.0) as u32;
  let svg_h = total_h.ceil() as u32;

  let mut svg = String::with_capacity(1024);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let mid_y = total_h / 2.0;
  let text_fill = theme().text_primary;
  let cell_top = |h: f64| match v_align {
    "top" => 0.0,
    "bottom" => total_h - h,
    _ => (total_h - h) / 2.0,
  };

  let mut x: f64 = match h_align {
    "center" => (canvas_w - content_w) / 2.0,
    "right" => canvas_w - content_w,
    _ => 0.0,
  };
  for (i, cell) in cells.iter().enumerate() {
    if i > 0 {
      match &separator {
        Separator::Gap(g) => x += g,
        Separator::Text(text, w) => {
          let cx = x + w / 2.0;
          svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{text}</text>\n"
          ));
          x += w;
        }
      }
    }

    match cell {
      Cell::Text {
        markup,
        width,
        height,
        fill,
        size,
        weight,
        slant,
        family,
      } => {
        let cx = x + width / 2.0;
        let cy = cell_top(*height) + height / 2.0;
        let weight_attr = if weight == "normal" {
          String::new()
        } else {
          format!(" font-weight=\"{weight}\"")
        };
        let slant_attr = if slant == "normal" {
          String::new()
        } else {
          format!(" font-style=\"{slant}\"")
        };
        // `Style[…, FontFamily -> "Times"]` picks the face; without one the
        // row keeps its monospace default.
        let ff = if family.is_empty() {
          "monospace"
        } else {
          family.as_str()
        };
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"{ff}\" font-size=\"{size}\" fill=\"{fill}\"{weight_attr}{slant_attr} text-anchor=\"middle\" dominant-baseline=\"central\">{markup}</text>\n",
          ff = svg_escape(ff),
        ));
        x += width;
      }
      Cell::Svg {
        svg: child,
        width,
        height,
      } => {
        let y_off = cell_top(*height);
        // The child keeps its own coordinate space: a plot draws at a
        // multiple of its display size, so the nested `<svg>` needs the
        // child's viewBox or none of it lands inside the cell.
        let view_box = parse_svg_dimensions(child)
          .map_or_else(|| format!("0 0 {width} {height}"), |p| p.view_box);
        svg.push_str(&format!(
          "<svg x=\"{x:.1}\" y=\"{y_off:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" viewBox=\"{view_box}\" preserveAspectRatio=\"xMidYMid meet\">\n"
        ));
        svg.push_str(strip_svg_wrapper(child));
        svg.push_str("</svg>\n");
        x += width;
      }
    }
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// Extract an option's key and value from a rule expression in either
/// the dedicated `Expr::Rule`/`Expr::RuleDelayed` AST variants or the
/// `Rule`/`RuleDelayed` FunctionCall forms.
fn option_kv(expr: &Expr) -> Option<(&str, &Expr)> {
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => option_name(pattern).map(|k| (k, replacement.as_ref())),
    Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2 =>
    {
      option_name(&args[0]).map(|k| (k, &args[1]))
    }
    _ => None,
  }
}

/// Render `Framed[expr]` as an SVG box with a rectangular border around the content.
/// Handles nested Framed by recursively rendering inner content as embedded SVG.
pub fn framed_to_svg(args: &[Expr]) -> Option<String> {
  if args.is_empty() {
    return None;
  }

  let content = &args[0];

  // Optional `Background -> color` option fills the inside of the frame.
  let bg_fill = args[1..]
    .iter()
    .filter_map(option_kv)
    .find(|(k, _)| *k == "Background")
    .and_then(|(_, v)| parse_color(v))
    .map(Color::to_svg_rgb);
  let rect_fill = bg_fill.as_deref().unwrap_or("none");

  // Layout constants
  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let margin: f64 = 6.0; // padding between content and frame border
  let stroke_width: f64 = 1.0;
  let rounding: f64 = 3.0;

  // Check if content is itself a Framed (nested) or already a Graphics
  let (inner_svg, inner_w, inner_h): (Option<String>, f64, f64) =
    if let Expr::FunctionCall {
      name,
      args: inner_args,
    } = content
    {
      if name == "Framed" {
        // Recursively render inner Framed
        if let Some(svg) = framed_to_svg(inner_args) {
          let (w, h) = parse_svg_wh(&svg);
          (Some(svg), w, h)
        } else {
          (None, 0.0, 0.0)
        }
      } else {
        (None, 0.0, 0.0)
      }
    } else if let Expr::Graphics { svg, .. } = content {
      let (w, h) = parse_svg_wh(svg);
      (Some(svg.clone()), w, h)
    } else {
      (None, 0.0, 0.0)
    };

  // A picture that has not been drawn yet — `Framed[Graphics[…]]`, where the
  // `Graphics` is still a call because it only renders at the output stage —
  // is drawn and framed, rather than printed as its own source text.
  let (inner_svg, inner_w, inner_h) = match inner_svg {
    None if crate::evaluator::lays_out_a_graphic(content) => {
      let svg = crate::evaluator::expr_to_svg(content);
      if svg.starts_with("<svg") {
        let (w, h) = parse_svg_wh(&svg);
        (Some(svg), w, h)
      } else {
        (None, 0.0, 0.0)
      }
    }
    other => (other, inner_w, inner_h),
  };

  if let Some(ref child_svg) = inner_svg {
    // Embed child SVG inside a frame
    let total_w = inner_w + 2.0 * margin;
    let total_h = inner_h + 2.0 * margin;
    let svg_w = total_w.ceil() as u32;
    let svg_h = total_h.ceil() as u32;

    let mut svg = String::with_capacity(child_svg.len() + 512);
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
    // Border rectangle
    let framed_border = theme().framed_border;
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"{rect_fill}\" stroke=\"{framed_border}\" stroke-width=\"{stroke_width}\"/>\n",
      stroke_width / 2.0, stroke_width / 2.0,
      total_w - stroke_width, total_h - stroke_width,
    ));
    // Embed child SVG
    svg.push_str(&format!(
      "<svg x=\"{margin:.1}\" y=\"{margin:.1}\" width=\"{inner_w:.1}\" height=\"{inner_h:.1}\">\n"
    ));
    // Strip outer <svg> and </svg> tags from child to embed its content
    let inner_content = strip_svg_wrapper(child_svg);
    svg.push_str(inner_content);
    svg.push_str("</svg>\n");
    svg.push_str("</svg>");
    Some(svg)
  } else {
    // Text content — measure and render
    let content_w = estimate_display_width(content) * char_width;
    let frac_extra = if has_fraction(content) { 10.0 } else { 0.0 };
    let content_h = font_size + frac_extra;

    let total_w = content_w + 2.0 * margin;
    let total_h = content_h + 2.0 * margin;
    let svg_w = total_w.ceil() as u32;
    let svg_h = total_h.ceil() as u32;

    let mut svg = String::with_capacity(512);
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
    // Border rectangle
    let framed_border = theme().framed_border;
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"{rect_fill}\" stroke=\"{framed_border}\" stroke-width=\"{stroke_width}\"/>\n",
      stroke_width / 2.0, stroke_width / 2.0,
      total_w - stroke_width, total_h - stroke_width,
    ));
    // Text centered inside
    let cx = total_w / 2.0;
    let cy = total_h / 2.0;
    let text_fill = theme().text_primary;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(content)
    ));
    svg.push_str("</svg>");
    Some(svg)
  }
}

/// Render `Highlighted[expr]` (or `Highlighted[expr, color]`) as an SVG box
/// with a filled, colored background behind the content. Without an explicit
/// color the theme's default highlight color (a light yellow) is used.
/// Handles nested `Highlighted`/`Framed` and embedded `Graphics` recursively.
pub fn highlighted_to_svg(args: &[Expr]) -> Option<String> {
  if args.is_empty() {
    return None;
  }

  let content = &args[0];

  // Optional second argument: a color for the highlight background.
  let bg_fill = args
    .get(1)
    .and_then(parse_color)
    .map_or_else(|| theme().highlighted_bg.to_string(), Color::to_svg_rgb);

  // Layout constants (mirrors framed_to_svg)
  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let margin: f64 = 6.0; // padding between content and highlight edge
  let rounding: f64 = 3.0;

  // Check whether the content is itself renderable as a nested SVG.
  let (inner_svg, inner_w, inner_h): (Option<String>, f64, f64) =
    if let Expr::FunctionCall {
      name,
      args: inner_args,
    } = content
    {
      match name.as_str() {
        "Highlighted" => {
          highlighted_to_svg(inner_args).map_or((None, 0.0, 0.0), |svg| {
            let (w, h) = parse_svg_wh(&svg);
            (Some(svg), w, h)
          })
        }
        "Framed" => framed_to_svg(inner_args).map_or((None, 0.0, 0.0), |svg| {
          let (w, h) = parse_svg_wh(&svg);
          (Some(svg), w, h)
        }),
        _ => (None, 0.0, 0.0),
      }
    } else if let Expr::Graphics { svg, .. } = content {
      let (w, h) = parse_svg_wh(svg);
      (Some(svg.clone()), w, h)
    } else {
      (None, 0.0, 0.0)
    };

  if let Some(ref child_svg) = inner_svg {
    // Embed child SVG on top of a highlighted background.
    let total_w = inner_w + 2.0 * margin;
    let total_h = inner_h + 2.0 * margin;
    let svg_w = total_w.ceil() as u32;
    let svg_h = total_h.ceil() as u32;

    let mut svg = String::with_capacity(child_svg.len() + 512);
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
    // Highlighted background rectangle (filled, no border).
    svg.push_str(&format!(
      "<rect x=\"0\" y=\"0\" width=\"{total_w:.1}\" height=\"{total_h:.1}\" rx=\"{rounding:.1}\" fill=\"{bg_fill}\"/>\n"
    ));
    // Embed child SVG.
    svg.push_str(&format!(
      "<svg x=\"{margin:.1}\" y=\"{margin:.1}\" width=\"{inner_w:.1}\" height=\"{inner_h:.1}\">\n"
    ));
    svg.push_str(strip_svg_wrapper(child_svg));
    svg.push_str("</svg>\n");
    svg.push_str("</svg>");
    Some(svg)
  } else {
    // Text content — measure and render on a highlighted background.
    let content_w = estimate_display_width(content) * char_width;
    let frac_extra = if has_fraction(content) { 10.0 } else { 0.0 };
    let content_h = font_size + frac_extra;

    let total_w = content_w + 2.0 * margin;
    let total_h = content_h + 2.0 * margin;
    let svg_w = total_w.ceil() as u32;
    let svg_h = total_h.ceil() as u32;

    let mut svg = String::with_capacity(512);
    svg.push_str(&format!(
      "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
    // Highlighted background rectangle (filled, no border).
    svg.push_str(&format!(
      "<rect x=\"0\" y=\"0\" width=\"{total_w:.1}\" height=\"{total_h:.1}\" rx=\"{rounding:.1}\" fill=\"{bg_fill}\"/>\n"
    ));
    // Text centered inside.
    let cx = total_w / 2.0;
    let cy = total_h / 2.0;
    let text_fill = theme().text_primary;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(content)
    ));
    svg.push_str("</svg>");
    Some(svg)
  }
}

/// Parse width and height from an SVG's root element attributes.
fn parse_svg_wh(svg: &str) -> (f64, f64) {
  let w = svg
    .find("width=\"")
    .and_then(|i| {
      let start = i + 7;
      svg[start..].find('"').map(|end| &svg[start..start + end])
    })
    .and_then(|s| s.parse::<f64>().ok())
    .unwrap_or(100.0);
  let h = svg
    .find("height=\"")
    .and_then(|i| {
      let start = i + 8;
      svg[start..].find('"').map(|end| &svg[start..start + end])
    })
    .and_then(|s| s.parse::<f64>().ok())
    .unwrap_or(30.0);
  (w, h)
}

/// Strip the outer <svg ...> and </svg> tags, returning only the inner content.
fn strip_svg_wrapper(svg: &str) -> &str {
  let start = svg.find('>').map_or(0, |i| i + 1);
  let end = svg.rfind("</svg>").unwrap_or(svg.len());
  &svg[start..end]
}

/// Render a list that contains Framed or Highlighted elements as a horizontal
/// row SVG. Plain items are rendered as text; Framed/Highlighted items are
/// fully rendered via `framed_to_svg` / `highlighted_to_svg` (each handling
/// arbitrary nesting) and embedded as child SVGs. The result looks like
/// `{x, |a|, ||b||}` with visual brackets and commas.
pub fn row_with_framed_to_svg(items: &[Expr]) -> Option<String> {
  if items.is_empty() {
    return None;
  }

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let sep_width: f64 = 2.0 * char_width; // ", " between items
  let brace_width: f64 = char_width; // "{" and "}"

  // Pre-compute each item: either a pre-rendered SVG or plain text metrics
  enum CellContent {
    /// Pre-rendered SVG string (for Framed items)
    Svg(String),
    /// Plain expression rendered as text
    Text(Expr),
  }

  struct CellInfo {
    width: f64,
    height: f64,
    content: CellContent,
  }

  let mut cells: Vec<CellInfo> = Vec::with_capacity(items.len());
  for item in items {
    if let Expr::FunctionCall { name, args } = item
      && !args.is_empty()
    {
      // Render the entire Framed / Highlighted (with any nesting) as SVG
      let child_svg = match name.as_str() {
        "Framed" => framed_to_svg(args),
        "Highlighted" => highlighted_to_svg(args),
        _ => None,
      };
      if let Some(child_svg) = child_svg {
        let (w, h) = parse_svg_wh(&child_svg);
        cells.push(CellInfo {
          width: w,
          height: h,
          content: CellContent::Svg(child_svg),
        });
        continue;
      }
    }
    let content_w = estimate_display_width(item) * char_width;
    let frac_extra = if has_fraction(item) { 10.0 } else { 0.0 };
    cells.push(CellInfo {
      width: content_w,
      height: font_size + frac_extra,
      content: CellContent::Text(item.clone()),
    });
  }

  // Total width: { + items + separators + }
  let items_width: f64 = cells.iter().map(|c| c.width).sum::<f64>();
  let seps_width = if cells.len() > 1 {
    (cells.len() - 1) as f64 * sep_width
  } else {
    0.0
  };
  let total_w = brace_width + items_width + seps_width + brace_width;
  let max_h = cells.iter().map(|c| c.height).fold(font_size, f64::max);
  // Add vertical padding so text items are not cramped
  let total_h = max_h + 4.0;

  let svg_w = total_w.ceil() as u32;
  let svg_h = total_h.ceil() as u32;

  let mut svg = String::with_capacity(2048);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let mid_y = total_h / 2.0;

  let text_fill = theme().text_primary;

  // Opening brace
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{{</text>\n",
    brace_width / 2.0
  ));

  let mut x = brace_width;
  for (i, cell) in cells.iter().enumerate() {
    if i > 0 {
      // Draw comma separator
      svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">,</text>\n",
        x + sep_width / 2.0
      ));
      x += sep_width;
    }

    match &cell.content {
      CellContent::Svg(child_svg) => {
        // Embed pre-rendered Framed SVG, vertically centered
        let ey = (total_h - cell.height) / 2.0;
        svg.push_str(&format!(
          "<svg x=\"{x:.1}\" y=\"{ey:.1}\" width=\"{:.1}\" height=\"{:.1}\">\n",
          cell.width, cell.height
        ));
        svg.push_str(strip_svg_wrapper(child_svg));
        svg.push_str("</svg>\n");
      }
      CellContent::Text(expr) => {
        let cx = x + cell.width / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(expr)
        ));
      }
    }
    x += cell.width;
  }

  // Closing brace
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">}}</text>\n",
    x + brace_width / 2.0
  ));

  svg.push_str("</svg>");
  Some(svg)
}

// ─── KochCurve ──────────────────────────────────────────────────────

/// KochCurve[n] - returns a Line representing the Koch curve at level n
pub fn koch_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 2 {
    return Err(InterpreterError::EvaluationError(
      "KochCurve called with wrong number of arguments; 1 or 2 arguments are expected.".into(),
    ));
  }

  let n = match &args[0] {
    Expr::Integer(n) if *n >= 0 => *n as usize,
    _ => {
      return Ok(unevaluated("KochCurve", args));
    }
  };

  // Start with a line from (0,0) to (1,0)
  let mut points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0)];

  for _ in 0..n {
    let mut new_points: Vec<(f64, f64)> =
      Vec::with_capacity(points.len() * 4 - 3);
    for i in 0..points.len() - 1 {
      let (x1, y1) = points[i];
      let (x2, y2) = points[i + 1];
      let dx = x2 - x1;
      let dy = y2 - y1;

      // Point at 1/3
      let p1 = (x1 + dx / 3.0, y1 + dy / 3.0);
      // Peak of equilateral triangle
      let p2 = (
        x1 + dx / 2.0 - dy * (3.0_f64.sqrt() / 6.0),
        y1 + dy / 2.0 + dx * (3.0_f64.sqrt() / 6.0),
      );
      // Point at 2/3
      let p3 = (x1 + 2.0 * dx / 3.0, y1 + 2.0 * dy / 3.0);

      new_points.push(points[i]);
      new_points.push(p1);
      new_points.push(p2);
      new_points.push(p3);
    }
    new_points.push(*points.last().unwrap());
    points = new_points;
  }

  // Build Line[{{x1, y1}, {x2, y2}, ...}]
  let point_exprs: Vec<Expr> = points
    .iter()
    .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into()))
    .collect();

  Ok(call1("Line", Expr::List(point_exprs.into())))
}

// ─── LinearGradientFilling ──────────────────────────────────────────

/// Check if an expression is a color specification (RGBColor, GrayLevel, Hue,
/// CMYKColor, or a theme-resolved color like LightDarkSwitched/ThemeColor/
/// SystemColor)
fn is_color_expr(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "RGBColor"
        | "GrayLevel"
        | "Hue"
        | "CMYKColor"
        | "LightDarkSwitched"
        | "ThemeColor"
        | "SystemColor"
    ),
    _ => false,
  }
}

/// Generate evenly spaced stops from 0 to 1 for n colors as exact fractions
fn evenly_spaced_stops(n: usize) -> Vec<Expr> {
  if n <= 1 {
    return vec![Expr::Integer(0), Expr::Integer(1)];
  }
  let denom = (n - 1) as i128;
  (0..n)
    .map(|i| crate::functions::make_rational(i as i128, denom))
    .collect()
}

// ─── DropShadowing ──────────────────────────────────────────────────

/// DropShadowing[...] — canonicalize to the full three-argument form
/// DropShadowing[offset, radius, color], filling in the defaults
/// {-3, -3}, 2 and Opacity[1/3, ThemeColor[Foreground]].
/// Arguments are matched positionally in the order offset (2-element
/// numeric list), radius (number), color (color directive or None);
/// each slot is optional but the order is fixed. Argument lists that
/// don't fit this pattern stay unevaluated.
pub fn drop_shadowing_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  fn is_number(e: &Expr) -> bool {
    match e {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _) => true,
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        is_number(&args[0]) && is_number(&args[1])
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => is_number(operand),
      _ => false,
    }
  }
  fn is_offset(e: &Expr) -> bool {
    matches!(e, Expr::List(items) if items.len() == 2 && items.iter().all(is_number))
  }
  fn is_color(e: &Expr) -> bool {
    match e {
      Expr::Identifier(name) => name == "None",
      Expr::FunctionCall { name, .. } => matches!(
        name.as_str(),
        "RGBColor"
          | "GrayLevel"
          | "Hue"
          | "CMYKColor"
          | "XYZColor"
          | "LABColor"
          | "LUVColor"
          | "LCHColor"
          | "Opacity"
          | "ThemeColor"
      ),
      _ => false,
    }
  }

  let (mut offset, mut radius, mut color) = (None, None, None);
  let mut valid = true;
  for arg in args {
    if offset.is_none() && radius.is_none() && color.is_none() && is_offset(arg)
    {
      offset = Some(arg.clone());
    } else if radius.is_none() && color.is_none() && is_number(arg) {
      radius = Some(arg.clone());
    } else if color.is_none() && is_color(arg) {
      color = Some(arg.clone());
    } else {
      valid = false;
      break;
    }
  }

  if !valid {
    return Ok(unevaluated("DropShadowing", args));
  }

  Ok(Expr::FunctionCall {
    name: "DropShadowing".to_string(),
    args: vec![
      offset.unwrap_or_else(|| {
        Expr::List(vec![Expr::Integer(-3), Expr::Integer(-3)].into())
      }),
      radius.unwrap_or(Expr::Integer(2)),
      color.unwrap_or_else(|| Expr::FunctionCall {
        name: "Opacity".to_string(),
        args: vec![
          crate::functions::make_rational(1, 3),
          call1("ThemeColor", Expr::Identifier("Foreground".to_string())),
        ]
        .into(),
      }),
    ]
    .into(),
  })
}

/// LinearGradientFilling[...] - normalizes gradient color specifications
pub fn linear_gradient_filling_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (stops, colors, angle, space) = if args.is_empty() {
    // LinearGradientFilling[] → default black to white
    let stops = vec![Expr::Integer(0), Expr::Integer(1)];
    let colors = vec![
      call1("GrayLevel", Expr::Integer(0)),
      call1("GrayLevel", Expr::Integer(1)),
    ];
    (
      stops,
      colors,
      Expr::Integer(0),
      Expr::String("Fixed".to_string()),
    )
  } else {
    // Parse angle (2nd arg) and space (3rd arg)
    let angle = if args.len() >= 2 {
      args[1].clone()
    } else {
      Expr::Integer(0)
    };
    let space = if args.len() >= 3 {
      args[2].clone()
    } else {
      Expr::String("Fixed".to_string())
    };

    match &args[0] {
      Expr::List(items) if !items.is_empty() => {
        // Check if items are {pos, color} pairs or plain colors
        let has_stop_pairs = items.iter().all(|item| {
          matches!(item, Expr::List(pair) if pair.len() == 2 && !is_color_expr(&pair[0]))
        });

        if has_stop_pairs {
          // {{pos1, color1}, {pos2, color2}, ...}
          let mut stops = Vec::new();
          let mut colors = Vec::new();
          for item in items {
            if let Expr::List(pair) = item {
              stops.push(pair[0].clone());
              colors.push(pair[1].clone());
            }
          }
          (stops, colors, angle, space)
        } else if items.len() == 1 {
          // Single color → duplicate it
          let stops = vec![Expr::Integer(0), Expr::Integer(1)];
          let colors = vec![items[0].clone(), items[0].clone()];
          (stops, colors, angle, space)
        } else {
          // Plain list of colors
          let stops = evenly_spaced_stops(items.len());
          let colors = items.to_vec();
          (stops, colors, angle, space)
        }
      }
      // Single non-list color arg
      other => {
        return Ok(call(
          "LinearGradientFilling",
          vec![other.clone(), angle, space],
        ));
      }
    }
  };

  // Build: LinearGradientFilling[{stops} -> {colors}, angle, space]
  let rule = Expr::Rule {
    pattern: Box::new(Expr::List(stops.into())),
    replacement: Box::new(Expr::List(colors.into())),
  };

  Ok(call("LinearGradientFilling", vec![rule, angle, space]))
}

/// Manipulate[expr, {u, umin, umax}, …] — interactive control construct.
///
/// In a text front-end (wolframscript CLI), Manipulate echoes itself back
/// with its body and variable specs preserved. Inside Woxi we treat
/// Manipulate as held (see `core_eval.rs`) so the body is not prematurely
/// evaluated with free control variables.
///
/// Supported variable-spec forms:
///   {u, umin, umax}                    — continuous
///   {u, umin, umax, du}                — stepped
///   {{u, uinit}, umin, umax, …}        — with initial value
///   {{u, uinit, ulbl}, umin, umax, …}  — with initial value and label
///   {u, {u1, u2, …}}                   — discrete values
///
/// Bounds inside a well-formed spec list are evaluated (so e.g.
/// `{x, 0, 2 Pi}` works), but the body expression and variable symbols
/// stay unevaluated. A non-list spec triggers a `Manipulate::vsform`
/// message (matching wolframscript) and the expression is still echoed
/// back as-is.
pub fn manipulate_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Process variable specs (args[1..]); args[0] is the held body.
  let mut out_args: Vec<Expr> = Vec::with_capacity(args.len());
  if let Some(body) = args.first() {
    out_args.push(body.clone());
  }

  for spec in args.iter().skip(1) {
    match spec {
      Expr::List(items) if !items.is_empty() => {
        out_args.push(process_manipulate_var_spec(items));
      }
      Expr::List(_) => {
        // Empty list — echo as-is.
        out_args.push(spec.clone());
      }
      // Manipulate options like `Initialization :> …`, `TrackedSymbols :> {a}`,
      // `SaveDefinitions -> True` are passed through unchanged. They are not
      // variable specifications and must not trigger `Manipulate::vsform`.
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        // `Initialization :> …` defines helper symbols that should be
        // visible to subsequent expressions in the same session, matching
        // Mathematica's behavior of running Initialization at notebook
        // evaluation time. Evaluate its body in the current (global)
        // scope so SetDelayed definitions register before the Manipulate
        // expression itself is returned.
        if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Initialization")
        {
          let _ = evaluate_expr_to_expr(replacement);
        }
        out_args.push(spec.clone());
      }
      // A trailing `Dynamic[…]` is an additional displayed expression, not
      // a variable specification, so it passes through without a vsform
      // message (matching wolframscript).
      Expr::FunctionCall { name, .. } if name == "Dynamic" => {
        out_args.push(spec.clone());
      }
      // `Delimiter`, a bare string, `Style[…]` and `Text[…]` are static
      // annotation rows between controls (Wolfram tags them
      // Manipulate`Dump`ThisIsNotAControl), not malformed variable specs —
      // they pass through with no message, matching wolframscript.
      Expr::Identifier(s) if s == "Delimiter" => out_args.push(spec.clone()),
      Expr::String(_) => out_args.push(spec.clone()),
      Expr::FunctionCall { name, .. }
        if is_manipulate_annotation_head(name) =>
      {
        out_args.push(spec.clone());
      }
      // Control objects and layout containers grouping them — `Control[…]`,
      // `Button[…]`, `Row[{Control[…], Spacer[…], …}]` and friends — are
      // valid Manipulate arguments (the Demonstrations layout pattern), not
      // malformed variable specs. They pass through with no message.
      // `PaneSelector[{v -> controls, …}, sel]` is the same pattern one
      // level up: a Demonstration whose modes need different controls
      // swaps whole control panels as `sel` changes. `Item[Column[…], opts]`
      // is the same layout pattern wrapped in a grid-alignment `Item[…]`
      // (a Demonstration lining up its whole control panel inside an outer
      // `Grid`), not a control itself.
      Expr::FunctionCall { name, .. }
        if matches!(
          name.as_str(),
          "Row"
            | "Column"
            | "Grid"
            | "Control"
            | "Button"
            | "ButtonBar"
            | "Spacer"
            | "PaneSelector"
            | "TabView"
            | "Item"
        ) =>
      {
        out_args.push(spec.clone());
      }
      _ => {
        // Non-list variable specification: emit Manipulate::vsform
        // message but still return the expression unchanged, matching
        // wolframscript's behavior.
        crate::emit_message(&format!(
          "Manipulate::vsform: Manipulate argument {} does not have the correct form for a variable specification.",
          crate::syntax::expr_to_string(spec)
        ));
        out_args.push(spec.clone());
      }
    }
  }

  Ok(call("Manipulate", out_args))
}

/// Whether `expr` still mentions a free symbol after evaluation — i.e. it
/// did not reduce to a concrete value. A control bound evaluated on its own
/// (outside the Manipulate's own variable scope) stays symbolic when it
/// depends on another control's variable (`Range[y]`); a call like
/// `RGBColor[0.49, 0, 0]` that evaluates to itself has no such dependency.
/// Named mathematical constants don't count as free — they're already
/// concrete values in disguise.
fn expr_is_symbolic(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(name) => !matches!(
      name.as_str(),
      "Pi"
        | "E"
        | "Degree"
        | "I"
        | "Infinity"
        | "ComplexInfinity"
        | "True"
        | "False"
        | "None"
        | "Automatic"
        | "All"
        | "Null"
        | "GoldenRatio"
        | "EulerGamma"
        | "Catalan"
    ),
    Expr::FunctionCall { args, .. } => args.iter().any(expr_is_symbolic),
    Expr::List(items) => items.iter().any(expr_is_symbolic),
    Expr::Association(pairs) => pairs
      .iter()
      .any(|(k, v)| expr_is_symbolic(k) || expr_is_symbolic(v)),
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => expr_is_symbolic(pattern) || expr_is_symbolic(replacement),
    _ => false,
  }
}

/// Process a single Manipulate/Control variable specification list,
/// evaluating trailing bounds/step/discrete values while keeping the head
/// (variable symbol or `{u, uinit, ulbl}`) intact. A 2-item spec
/// `{var, range}` whose range is still symbolic is wrapped in `Dynamic[…]`
/// to match wolframscript's echoed form.
fn process_manipulate_var_spec(items: &[Expr]) -> Expr {
  // Preserve the head as-is; evaluate any trailing bounds/step/values.
  let mut new_items: Vec<Expr> = Vec::with_capacity(items.len());
  new_items.push(items[0].clone());
  for item in &items[1..] {
    // `Enabled -> cond` / `TrackingFunction -> f` must stay held: `cond`
    // references other controls' variables (and often symbols an
    // `Initialization` block — processed after every spec here — hasn't
    // defined yet), so evaluating it now would freeze it at whatever it
    // happens to fold to with nothing bound, instead of the live condition
    // `parse_manipulate_control` re-resolves on every frame.
    if let Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } = item
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Enabled" || s == "TrackingFunction")
    {
      new_items.push(item.clone());
      continue;
    }
    // Try to evaluate bounds; if evaluation fails, keep the original so
    // the echoed form still round-trips. A bound stated in terms of another
    // control's variable (`{{p, init}, {r[[1]], s[[1]]}, …}`) cannot resolve
    // here — Wolfram holds the whole Manipulate and only ever sees such a
    // bound with the control variables in scope — so this speculative pass
    // must stay silent rather than report `Part::partd` on the free symbol.
    let snapshot = crate::snapshot_warnings();
    crate::push_quiet();
    let evaluated =
      evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone());
    crate::pop_quiet();
    crate::restore_warnings(snapshot);
    new_items.push(evaluated);
  }
  // A 2-item spec `{var, range}` whose `range` doesn't reduce to a concrete
  // numeric value or list (e.g. it still contains a free symbol like
  // `Range[y]`) is wrapped in Dynamic[…] so the menu updates as the host
  // variable changes.
  if new_items.len() == 2
    && let needs_dynamic = match &new_items[1] {
      Expr::Integer(_) | Expr::Real(_) | Expr::List(_) => false,
      Expr::FunctionCall { name, .. } if name == "Dynamic" => false,
      // Any other call (e.g. `RGBColor[0.49, 0, 0]`) only needs wrapping
      // when it stayed symbolic after evaluation — i.e. it still mentions a
      // free symbol such as another control's variable. A call that
      // evaluated down to a concrete value is already stable and needs no
      // live re-resolution.
      Expr::FunctionCall { args, .. } => args.iter().any(expr_is_symbolic),
      // A trailing control option such as `ControlType -> None` is not a
      // range, so it must not be wrapped in Dynamic[…].
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => false,
      // A bare control-type shorthand in the range position (`{{p, init},
      // Locator}`, `{u, Slider}` …) selects the control; it is not a range.
      Expr::Identifier(s)
        if matches!(
          s.as_str(),
          "Locator"
            | "Slider"
            | "Slider2D"
            | "VerticalSlider"
            | "Manipulator"
            | "InputField"
            | "PopupMenu"
            | "SetterBar"
            | "RadioButton"
            | "RadioButtonBar"
            | "TogglerBar"
            | "Checkbox"
            | "ColorSlider"
            | "ColorSetter"
            | "IntervalSlider"
            | "Animator"
            | "Trigger"
            | "None"
            | "Automatic"
        ) =>
      {
        false
      }
      _ => true,
    }
    && needs_dynamic
  {
    let range = new_items.pop().unwrap();
    new_items.push(call1("Dynamic", range));
  }
  Expr::List(new_items.into())
}

/// Held evaluation of a standalone `Control[…]` expression. Like Manipulate,
/// Control holds its argument and, in a text front-end, echoes itself back
/// with the variable spec's bounds evaluated (`Control[{x, 0, 2 Pi}]` →
/// `Control[{x, 0, 2 Pi}]` with `2 Pi` reduced). The Playground / Studio
/// front-ends detect the held `Control[…]` and render an interactive
/// control widget (see `extract_control_spec`).
pub fn control_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut out_args: Vec<Expr> = Vec::with_capacity(args.len());
  if let Some(first) = args.first() {
    match first {
      Expr::List(items) if !items.is_empty() => {
        out_args.push(process_manipulate_var_spec(items));
      }
      other => out_args.push(other.clone()),
    }
  }
  // Any trailing options pass through unchanged.
  for extra in args.iter().skip(1) {
    out_args.push(extra.clone());
  }
  Ok(call("Control", out_args))
}

// ─────────────────────────────────────────────────────────────────
// GeometricScene
// ─────────────────────────────────────────────────────────────────

/// Pull the bound symbol name out of a point-definition rule's
/// left-hand side (`sym -> value`); anything else isn't a point rule.
fn geometric_scene_point_name(pattern: &Expr) -> Option<String> {
  match pattern {
    Expr::Identifier(name) => Some(name.clone()),
    _ => None,
  }
}

/// Held evaluation of `GeometricScene[{sym -> value, ...}, {primitives...}]`
/// (an optional third constraints-list argument is accepted and, like the
/// primitives, held for later use). The point-definition rules are
/// evaluated in order, left to right, with every earlier point symbol
/// substituted into later right-hand sides first — so a later point may be
/// derived from earlier ones, e.g. `centroid -> TriangleCenter[{a, b, c},
/// "Centroid"]`. The primitives (and constraints) stay symbolic, still
/// referencing the point *names*, until `["Graphics"]` resolves them.
pub fn geometric_scene_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut out_args: Vec<Expr> = Vec::with_capacity(args.len());

  let mut bindings: Vec<(String, Expr)> = Vec::new();
  match args.first() {
    Some(Expr::List(items)) => {
      let mut evaluated_rules: Vec<Expr> = Vec::with_capacity(items.len());
      for item in items {
        let rule_parts = match item {
          Expr::Rule {
            pattern,
            replacement,
          }
          | Expr::RuleDelayed {
            pattern,
            replacement,
          } => Some((pattern.as_ref(), replacement.as_ref())),
          Expr::FunctionCall { name, args: rargs }
            if (name == "Rule" || name == "RuleDelayed")
              && rargs.len() == 2 =>
          {
            Some((&rargs[0], &rargs[1]))
          }
          _ => None,
        };
        if let Some((name, replacement)) =
          rule_parts.and_then(|(pattern, replacement)| {
            geometric_scene_point_name(pattern).map(|name| (name, replacement))
          })
        {
          let binding_refs: Vec<(&str, &Expr)> =
            bindings.iter().map(|(n, v)| (n.as_str(), v)).collect();
          let substituted =
            crate::syntax::substitute_variables(replacement, &binding_refs);
          let value = evaluate_expr_to_expr(&substituted)?;
          evaluated_rules.push(Expr::Rule {
            pattern: Box::new(Expr::Identifier(name.clone())),
            replacement: Box::new(value.clone()),
          });
          bindings.push((name, value));
        } else {
          // Not a recognizable `symbol -> value` point rule; evaluate it
          // on its own (with points bound so far in scope) and keep it as-is.
          let binding_refs: Vec<(&str, &Expr)> =
            bindings.iter().map(|(n, v)| (n.as_str(), v)).collect();
          let substituted =
            crate::syntax::substitute_variables(item, &binding_refs);
          evaluated_rules.push(evaluate_expr_to_expr(&substituted)?);
        }
      }
      out_args.push(Expr::List(evaluated_rules.into()));
    }
    Some(other) => out_args.push(other.clone()),
    None => {}
  }

  // Primitives (args[1]) and any optional constraints (args[2]) reference
  // the point symbols by name rather than by value, so they stay held
  // until a `["Graphics"]` (or similar) property substitutes them in.
  for extra in args.iter().skip(1) {
    out_args.push(extra.clone());
  }

  // A scene always reports itself in the same three-argument shape:
  // `GeometricScene[{points, quantities}, primitives, constraints]`, with
  // the slots a two-argument call leaves open filled by empty lists.
  let empty = || Expr::List(Vec::new().into());
  let points = out_args.first().cloned().unwrap_or_else(empty);
  let canonical = vec![
    Expr::List(vec![points, empty()].into()),
    out_args.get(1).cloned().unwrap_or_else(empty),
    out_args.get(2).cloned().unwrap_or_else(empty),
  ];

  Ok(call("GeometricScene", canonical))
}

/// `GeometricScene[{sym -> value, ...}, {primitives...}][ "Graphics" ]` —
/// substitute every point symbol occurring anywhere in the primitives
/// (including inside wrappers like `Style[...]`/`Directive[...]`) with its
/// bound coordinate value, then evaluate the result as an ordinary
/// `Graphics[{...}]` expression.
pub fn geometric_scene_graphics(
  func_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let bindings: Vec<(String, Expr)> = match scene_point_rules(func_args) {
    Some(items) => items
      .iter()
      .filter_map(|item| match item {
        Expr::Rule {
          pattern,
          replacement,
        } => geometric_scene_point_name(pattern)
          .map(|name| (name, replacement.as_ref().clone())),
        _ => None,
      })
      .collect(),
    _ => Vec::new(),
  };
  let binding_refs: Vec<(&str, &Expr)> =
    bindings.iter().map(|(n, v)| (n.as_str(), v)).collect();

  let primitives = func_args
    .get(1)
    .cloned()
    .unwrap_or(Expr::List(Vec::new().into()));
  let substituted =
    crate::syntax::substitute_variables(&primitives, &binding_refs);
  evaluate_expr_to_expr(&call("Graphics", vec![substituted]))
}

/// The point-definition rules of a scene: the first element of the
/// `{points, quantities}` list a canonical `GeometricScene` keeps in its
/// first slot.
fn scene_point_rules(func_args: &[Expr]) -> Option<&crate::ExprList> {
  match func_args.first() {
    Some(Expr::List(slots)) => match slots.first() {
      Some(Expr::List(points)) => Some(points),
      _ => None,
    },
    _ => None,
  }
}

/// `GeometricScene[{{sym -> value, ...}, {}}, ...][ "Points" ]` — the
/// list of point definitions the scene was built from.
pub fn geometric_scene_points(
  func_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(scene_point_rules(func_args).map_or_else(
    || Expr::List(Vec::new().into()),
    |points| Expr::List(points.clone()),
  ))
}

// ─────────────────────────────────────────────────────────────────
// Interactive Manipulate support (for Woxi Playground / Woxi Studio)
// ─────────────────────────────────────────────────────────────────

/// A styled run of a control label. A label is a sequence of these so the
/// UI can render `Style["t", Italic]` as an italic `t` while leaving the
/// rest upright. Italic, bold and color are tracked — the styling Wolfram
/// labels and Demonstration captions use in practice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LabelRun {
  pub text: String,
  pub italic: bool,
  pub bold: bool,
  /// The run's color as `(r, g, b)` in 0..1, when `Style` gave it one.
  pub color: Option<(f32, f32, f32)>,
}

/// A single control inside a Manipulate expression.
///
/// Continuous controls correspond to `{u, umin, umax}` or
/// `{u, umin, umax, du}` (optionally wrapped in `{{u, uinit}, …}` /
/// `{{u, uinit, ulbl}, …}`). Discrete controls correspond to
/// `{u, {u1, u2, …}}` and are rendered as a dropdown / pick list.
#[derive(Debug, Clone)]
pub enum ManipulateControl {
  Continuous {
    name: String,
    min: f64,
    max: f64,
    /// Optional explicit step size (`du`). When `None`, the UI picks a
    /// reasonable default (e.g. (max - min) / 100).
    step: Option<f64>,
    initial: f64,
    /// Plain-text label (all runs concatenated) — used for JSON and any
    /// consumer that can't render styling.
    label: String,
    /// The label split into styled runs for rich-text rendering.
    label_runs: Vec<LabelRun>,
    /// Whether the control's variable is machine-real by construction — any
    /// of its `min`/`max`/step/initial was written as an inexact number
    /// (e.g. `{{rq, 0, "RQ"}, 0, 1, 0.01}`). Wolfram keeps such a variable
    /// real-valued even while the slider sits at a "round" value (`0.`, not
    /// `0`), which matters once a caption formats it with `NumberForm[…, {n,
    /// f}]` — that wrapper pads a real's fraction but leaves an exact
    /// integer unchanged. A control whose whole spec is exact integers
    /// (`{{n, 5, "n"}, 1, 10, 1}`) keeps binding exact integers throughout.
    is_real: bool,
  },
  Discrete {
    name: String,
    /// Each discrete value rendered as InputForm — echoed back into the
    /// variable binding. For a rule-form choice `value -> "label"` this is
    /// the left side (the actual value), never the whole rule.
    values: Vec<String>,
    /// The display label for each choice, parallel to `values`. For a plain
    /// choice this equals the value's InputForm; for a rule-form choice it is
    /// the (unquoted) right side of the rule. A rule label that is itself a
    /// graphic falls back to the value's display text here (the rendered
    /// icon travels in `value_label_svgs`).
    value_labels: Vec<String>,
    /// Rendered SVG for each choice whose rule label is a `Graphics[…]`
    /// icon (e.g. the crosshair pickers of the Demonstrations site),
    /// parallel to `values`. `None` for plain text labels.
    value_label_svgs: Vec<Option<String>>,
    initial_index: usize,
    label: String,
    label_runs: Vec<LabelRun>,
    /// `ControlType -> PopupMenu`: always render a dropdown, even when the
    /// choice count is small enough for a SetterBar.
    popup: bool,
    /// `ControlType -> SetterBar` or `-> RadioButtonBar`: always render the
    /// row of buttons, even when there are more choices than the automatic
    /// split would put in a bar. The heuristic only decides for a spec that
    /// stays silent.
    setter_bar: bool,
    /// `ControlType -> Slider`: render a slider that steps through the
    /// choices by index, the way Wolfram draws a slider over a discrete
    /// domain. Without it a twenty-entry list would become a dropdown.
    slider: bool,
    /// `Appearance -> "Vertical"` (or the bare symbol `Vertical`): stack the
    /// choice row in a column instead of Wolfram's default horizontal bar.
    /// Only affects the SetterBar/RadioButtonBar bar layout; a dropdown or
    /// index slider ignores it.
    vertical: bool,
  },
  /// A 2D control (`ControlType -> Slider2D`, or a 2D range spec
  /// `{u, {xmin, ymin}, {xmax, ymax}}`). Binds its variable to a 2-vector
  /// `{x, y}`.
  Slider2D {
    name: String,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    x_initial: f64,
    y_initial: f64,
    label: String,
    /// InputForm of a write-back function (the second argument of a
    /// `Locator[Dynamic[var, cb], …]` this control was promoted from).
    /// Candidate values pass through it — `(cb)[{x, y}]` — before the
    /// variable is read back, so e.g. `Clip[Round[#], …]` validation runs
    /// exactly as Wolfram would.
    write_callback: Option<String>,
  },
  /// An interval control (`ControlType -> IntervalSlider`). Binds its
  /// variable to a 2-vector `{low, high}` describing the selected range.
  IntervalSlider {
    name: String,
    min: f64,
    max: f64,
    /// Optional explicit step size. When `None`, the UI picks a default.
    step: Option<f64>,
    low_initial: f64,
    high_initial: f64,
    label: String,
  },
  /// A `Locator` control bound to a *list* of draggable 2D points (e.g.
  /// the vertices of a polygon). Rendered as one X/Y slider pair per
  /// point; `auto_create` (`LocatorAutoCreate -> True`) additionally
  /// offers adding and removing points.
  Locator {
    name: String,
    points: Vec<(f64, f64)>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    auto_create: bool,
    label: String,
  },
  /// A `ControlType -> Trigger` control: a play/pause button pair that
  /// sweeps its variable from `min` towards `max` in `step` increments
  /// while running (the Demonstrations "run/stop simulation" control).
  /// `max` may be `f64::INFINITY` (`{time, 0, Infinity, 1, …}`), in which
  /// case the sweep never wraps.
  Trigger {
    name: String,
    min: f64,
    max: f64,
    step: f64,
    initial: f64,
    /// `AnimationRunning -> True`: start sweeping immediately. Wolfram's
    /// Trigger sits paused until pressed by default.
    running: bool,
    label: String,
    label_runs: Vec<LabelRun>,
  },
  /// A `Button[label, action]` control row. Pressing it evaluates `action`
  /// (InputForm) against the live bindings — typically resetting state
  /// variables (`time = 0; {U, V} = {Uinit, Vinit}`). Binds no variable.
  Button {
    label: String,
    label_runs: Vec<LabelRun>,
    action: String,
  },
  /// A static heading row between controls: a bare string or `Style[…]`
  /// Manipulate argument (Wolfram's `ThisIsNotAControl` annotations, e.g.
  /// the "signal 1" / "signal 2" captions of the oscilloscope
  /// demonstration). Binds no variable.
  Heading {
    label: String,
    label_runs: Vec<LabelRun>,
  },
  /// A `Delimiter` argument: a horizontal separator row between control
  /// groups. Binds no variable.
  Divider,
}

impl ManipulateControl {
  /// The bound variable name for this control. Annotation rows
  /// (`Heading` / `Divider`) bind no variable and return `""`.
  pub fn name(&self) -> &str {
    match self {
      Self::Continuous { name, .. }
      | Self::Discrete { name, .. }
      | Self::Slider2D { name, .. }
      | Self::IntervalSlider { name, .. }
      | Self::Trigger { name, .. }
      | Self::Locator { name, .. } => name,
      Self::Button { .. } | Self::Heading { .. } | Self::Divider => "",
    }
  }
}

/// A parsed Manipulate expression ready for interactive rendering.
#[derive(Debug, Clone)]
pub struct ManipulateSpec {
  /// The body expression as an InputForm-compatible string, ready to be
  /// substituted into a `Block[{…}, body]` for re-evaluation.
  pub body_code: String,
  pub controls: Vec<ManipulateControl>,
  /// Mutable state variables that have no visible slider/picker widget
  /// (`ControlType -> None`) but are shared between the body and any
  /// interactive display element (e.g. a `Checkbox` grid writing back into
  /// the variable). Each entry is `(name, initial value as InputForm)`; the
  /// value is evaluated once so it does not re-randomize on every frame.
  /// Unlike `Locator`-style fixed bindings (which are baked into `body_code`),
  /// these are passed live in the binding set so a control can rewrite them.
  pub state: Vec<(String, String)>,
  /// Extra display expressions that trail the control specs, e.g. a
  /// `Dynamic[Panel[Grid[…]]]` of `Checkbox`es. Stored as InputForm so the
  /// frontend can re-render them (via `render_manipulate_display`) on every
  /// state change. Empty when the Manipulate has no extra display.
  pub displays: Vec<String>,
  /// Initialization code from `Initialization :> …`. Runs once before the
  /// first evaluation of the body so that helper definitions (e.g.
  /// `d[t_] := …`) are in scope. `None` when the Manipulate has no
  /// `Initialization` option.
  pub initialization: Option<String>,
  /// Per-control `Enabled -> Dynamic[cond]` gating, as `(control name,
  /// condition code)`. Each condition is a boolean expression in the control
  /// variables, re-evaluated against the live bindings so a control can grey
  /// itself out (e.g. the Yin-Yang demonstration disables the curve sliders
  /// while `YinYang` is `True`). Controls with no `Enabled` option (or the
  /// trivial `Enabled -> True`) do not appear here and stay always enabled.
  pub control_enabled: Vec<(String, String)>,
  /// Per-control visibility gating, as `(control name, condition code)`.
  /// A `PaneSelector[{v -> controls, …}, sel]` argument swaps whole control
  /// panels as `sel` changes, so each pane's controls are shown only while
  /// `sel` holds that pane's value (Kepler's-conjecture packing shows one
  /// angle slider for the disk view, four controls for the sphere view and
  /// none for the cannonball view). Woxi's control panel is one flat list,
  /// so the panes become conditions the frontend re-evaluates against the
  /// live bindings, hiding the rows that do not apply. Controls outside any
  /// pane do not appear here and stay always visible.
  pub control_visible: Vec<(String, String)>,
  /// Continuous-control bounds that reference other control variables, as
  /// `(control name, min code, max code)`. A Demonstration like Kepler's
  /// Second Law bounds its time slider by the orbital-period variable
  /// (`{{t, 0, …}, 0, P, .01}`), so the numeric `min`/`max` stored on the
  /// control are only the values at build time — the frontend re-evaluates
  /// these code fragments against the live bindings after every change and
  /// updates the slider range to follow. A `None` side is static.
  pub dynamic_bounds: Vec<(String, Option<String>, Option<String>)>,
  /// Discrete-control choice lists that reference other control variables,
  /// as `(control name, values code)`. A Demonstration whose level setter
  /// reads `Range[1, If[flat, 3, 6], 1]` offers six levels flat and three
  /// in 3D, so the choices stored on the control are only the ones the
  /// initial values produce — the frontend re-evaluates this code fragment
  /// against the live bindings after every change and rebuilds the choices.
  pub dynamic_values: Vec<(String, String)>,
  /// The variable animated by a `ControlType -> Trigger`/`Animator` control
  /// spec. Wolfram renders those as play buttons that sweep the variable
  /// over its range; the widget's animation targets this variable instead
  /// of defaulting to the first continuous control.
  pub animation_var: Option<String>,
  /// Whether this spec should auto-play, i.e. it came from `Animate[…]` or
  /// `ListAnimate[…]`. An animated widget advances its first continuous
  /// control on a timer (with a play/pause toggle) instead of sitting still
  /// until the user drags a slider. Plain `Manipulate`/`Control` widgets leave
  /// this `false`.
  pub animated: bool,
  /// Whether an animated widget starts playing. Wolfram's default is
  /// `AnimationRunning -> True`; `AnimationRunning -> False` builds the
  /// widget paused until the user presses play.
  pub animation_running: bool,
  /// `Appearance -> None`: the widget shows no visible control rows (the
  /// animation just runs), matching Wolfram's control-free appearance.
  pub appearance_none: bool,
  /// `TrackedSymbols :> {a, b}`: the only variables whose change re-runs the
  /// body. A control outside the list still moves — it just does not
  /// re-render until a tracked variable changes too (Descartes's Rule of
  /// Signs tracks only the seed and the zoom, so picking a new polynomial
  /// degree takes effect when the "new polynomial" button reseeds). `None`
  /// means every variable is tracked, which is Wolfram's default and also
  /// what `TrackedSymbols -> All` / `-> Manipulate` ask for.
  pub tracked_symbols: Option<Vec<String>>,
  /// Per-control `TrackingFunction -> f` callbacks, as `(control name,
  /// function code)`. Whenever the named control's value changes, `f` runs
  /// with the new value as `#` — typically to reset a *different* control
  /// (an Electrophilic Aromatic Substitution demonstration resets its time
  /// slider to `0` whenever the reaction step picker changes: `{{a, 1, ""},
  /// choices, TrackingFunction -> (a = #; t = 0; &)}`). Controls with no
  /// `TrackingFunction` option do not appear here.
  pub tracking: Vec<(String, String)>,
}

/// Result of parsing a single list-shaped Manipulate argument.
enum ParsedControl {
  /// A control that renders a UI element (slider or pick list). `enabled` is
  /// an optional `Enabled` condition (InputForm code) that gates the widget:
  /// when it evaluates to `False` against the live bindings the control is
  /// shown greyed-out and non-interactive. `min_code`/`max_code` carry a
  /// continuous bound that references other control variables (re-resolved
  /// live by the frontend); `animate` is set for a `ControlType ->
  /// Trigger`/`Animator` spec, with the flag telling whether the animation
  /// starts running (`Animator`) or paused (`Trigger`).
  Visible {
    control: ManipulateControl,
    enabled: Option<String>,
    min_code: Option<String>,
    max_code: Option<String>,
    /// A discrete choice list that references other control variables
    /// (re-resolved live by the frontend, like `min_code`/`max_code`).
    values_code: Option<String>,
    animate: Option<bool>,
    /// `TrackingFunction -> f` (InputForm code), run with the control's new
    /// value whenever it changes.
    tracking: Option<String>,
  },
  /// A `Locator` control with no widget. It contributes a fixed `name =
  /// value` binding that is baked directly into the body so the variable is
  /// in scope while the visible controls drive the plot.
  Fixed { name: String, value: String },
  /// A `ControlType -> None` variable: no widget, but a *mutable* binding
  /// passed live in the binding set so an interactive display element (a
  /// `Checkbox`, `Setter`, …) can write back into it.
  State { name: String, value: String },
  /// A multiple-choice control (`ControlType -> CheckboxBar`/`TogglerBar`):
  /// the variable binds the *list* of chosen values, and the bar is carried
  /// as a display element — the same widget an in-body
  /// `TogglerBar[Dynamic[v], …]` produces, which every frontend draws.
  StateWithDisplay {
    name: String,
    value: String,
    display: String,
  },
  /// A custom control — `{{u, uinit, ulbl}, func}`, where `func` builds the
  /// widget. Both parts are needed: the widget row *and* a mutable binding
  /// for `u`, since the widget's action writes back into it.
  StateWithControl {
    name: String,
    value: String,
    control: ManipulateControl,
  },
}

/// Whether a bare `TrackedSymbols -> sym` value asks for the default of
/// tracking every variable rather than naming a single tracked variable.
fn is_track_everything_symbol(sym: &str) -> bool {
  matches!(sym, "All" | "Full" | "Manipulate" | "Automatic" | "True")
}

/// Attempt to extract a `ManipulateSpec` from a held `Manipulate[…]` or
/// `Animate[…]` expression. `Animate` shares `Manipulate`'s argument shape
/// (a body followed by `{u, umin, umax}`-style control specs) but auto-plays,
/// so it produces the same spec with `animated` set. Returns `None` if the
/// expression is not a well-formed Manipulate/Animate (e.g. `Manipulate[]`,
/// `Manipulate[expr]`, or a spec that isn't a list). In those cases the caller
/// should fall back to the standard text/graphics output path.
pub fn extract_manipulate_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if (name != "Manipulate" && name != "Animate") || args.len() < 2 {
    return None;
  }
  let mut animated = name == "Animate";
  let mut animation_running = true;

  // A Manipulate whose body is itself an `Animate[…]` (the Demonstrations
  // "oscilloscope" pattern) nests an auto-playing animation inside the outer
  // control panel. A single widget can render both: flatten the inner
  // Animate into this spec — its body becomes the combined body and its
  // animation variable becomes the leading (animated) control, followed by
  // the outer controls.
  let inner = if name == "Manipulate" {
    match &args[0] {
      Expr::FunctionCall { name: n, .. } if n == "Animate" => {
        extract_manipulate_spec(&args[0])
      }
      _ => None,
    }
  } else {
    None
  };
  // Kept so a pick list the body draws can be lifted out of it once the
  // control specs say which variable it drives (see `body_popups` below).
  let mut body_expr_kept: Option<Expr> = None;
  let (
    body_code,
    mut controls,
    mut state,
    mut displays,
    mut initialization,
    mut control_enabled,
    mut control_visible,
    mut dynamic_bounds,
    mut dynamic_values,
    mut animation_var,
    mut tracking,
  ) = if let Some(inner) = inner {
    animated = true;
    animation_running = inner.animation_running;
    (
      inner.body_code,
      inner.controls,
      inner.state,
      inner.displays,
      inner.initialization,
      inner.control_enabled,
      inner.control_visible,
      inner.dynamic_bounds,
      inner.dynamic_values,
      inner.animation_var,
      inner.tracking,
    )
  } else {
    // `DynamicModule[{locals…}, …]` wrapping the body: unlike `Module`, a
    // DynamicModule's locals live for the widget's whole lifetime, so a
    // `Button` inside it that writes one (a Demonstration's "throw"/"step"
    // action) must have that write survive the next re-render. Hoist them
    // into the widget's hidden state — the same mechanism a Manipulate's
    // own `ControlType -> None` variables use — and drop the wrapper, so
    // `reevaluate` installs them as ordinary globals instead of Module
    // re-creating fresh ones every frame.
    let mut dynamic_module_state = Vec::new();
    let unwrapped =
      unwrap_dynamic_module_locals(&args[0], &mut dynamic_module_state);
    // `TogglerBar[Dynamic[var], …]` and a bare `Button[label, action]`
    // inside the body move into the display list (replaced by `Nothing` in
    // the body): a front-end renders displays as live widgets, whereas
    // inside the rendered output they would only be an inert picture.
    let mut body_displays = Vec::new();
    let body_expr = extract_body_togglerbars(&unwrapped, &mut body_displays);
    let body_code = crate::syntax::expr_to_input_form(&body_expr);
    body_expr_kept = Some(body_expr);
    (
      body_code,
      Vec::with_capacity(args.len() - 1),
      dynamic_module_state,
      body_displays,
      None,
      Vec::new(),
      Vec::new(),
      Vec::new(),
      Vec::new(),
      None,
      Vec::new(),
    )
  };
  // `Locator[Dynamic[var, cb], …]` markers inside the body drive their
  // variable interactively: a hidden `ControlType -> None` spec for such a
  // variable is promoted to a visible Locator-style control below (with
  // `cb` as its write-back callback).
  let body_locators = collect_body_locator_callbacks(&args[0]);
  // `PopupMenu[Dynamic[var], choices]` drawn by the body likewise drives its
  // variable: a hidden `ControlType -> None` spec for such a variable turns
  // into a visible pick list, with the body's own choice list behind it.
  // Only this Manipulate's own body qualifies — a promoted pick list has to
  // be taken back out of the body it was drawn in, and a body flattened from
  // a nested `Animate` arrives already serialized.
  let body_popups = match &body_expr_kept {
    Some(body) => collect_body_popup_menus(body),
    None => Vec::new(),
  };
  let mut promoted_popups: Vec<String> = Vec::new();
  // `Locator` bindings are baked into the body (never rewritten by a
  // display); `ControlType -> None` bindings become live mutable state.
  let mut fixed: Vec<(String, String)> = Vec::new();
  let mut appearance_none = false;
  let mut tracked_symbols: Option<Vec<String>> = None;
  // Compound (non-symbol) control variables such as `Subscript[signal, 1]`
  // cannot be bound by name; each is renamed to a synthesized plain symbol,
  // and every occurrence in the body (and related code fragments) is
  // rewritten below via `(original InputForm, synthesized name)` pairs.
  let mut renames: Vec<(String, String)> = Vec::new();
  // Controls grouped in a `Row[…]`/`Column[…]`/`Grid[…]` argument (the
  // Demonstrations layout pattern `Row[{Control[…], Spacer[20], Button[…]}]`)
  // flatten into their items so each inner control becomes its own row, and
  // a `Control[spec, opts…]` wrapper unwraps to its ordinary variable
  // specification so it parses through the standard path.
  let mut arg_items: Vec<Expr> =
    Vec::with_capacity(args.len().saturating_sub(1));
  for spec in &args[1..] {
    // A `PaneSelector` argument shows one pane's controls at a time; the
    // flattened list holds every pane's, so each pane's controls also pick
    // up the condition under which they are on screen.
    collect_pane_visibility(spec, &mut control_visible);
    match control_group_items(spec) {
      Some(items) => arg_items.extend(items),
      None => arg_items.push(spec.clone()),
    }
  }
  let arg_items: Vec<Expr> =
    arg_items.into_iter().map(unwrap_control_wrapper).collect();
  let pane_governed_names = pane_or_tab_governed_names(&args[1..]);
  // A `ControlType -> …` given to the Manipulate itself sets the type of every
  // control that does not choose one; push it into the specs now that they are
  // flattened, so they parse through the single per-spec path below.
  let arg_items = apply_global_control_type(arg_items);
  // A control's bounds may reference *other* control variables — Kepler's
  // Second Law bounds its time sliders by the orbital period (`{{t, 0, …},
  // 0, P, .01}` with `{{P, 20, …}, .1, 50, .01}` further down). Collect
  // every control's initial value first and install them as scoped globals
  // while the specs are parsed, so those bounds resolve to their build-time
  // numbers regardless of declaration order.
  let initial_bindings = manipulate_initial_value_bindings(&arg_items);
  // `Sequence@@If[cond, ctrlSpec, {}]` (the Demonstrations idiom for a
  // control that only appears under some condition on another control, e.g.
  // an extra slider shown only in one mode) is not itself a control spec —
  // it is code that *produces* zero or more of them once evaluated. Resolve
  // every such entry against the controls' initial values, the way Wolfram
  // evaluates the spec list once up front, and splice whatever list it
  // produces into the flat control list in its place.
  let arg_items: Vec<Expr> =
    expand_conditional_control_items(arg_items, &initial_bindings)
      .into_iter()
      .map(unwrap_control_wrapper)
      .collect();
  // The same names, as the set a control's choice list is checked against
  // to tell a static list from one that follows another control.
  let sibling_names: Vec<String> =
    initial_bindings.iter().map(|(n, _)| n.clone()).collect();
  // Filled in on demand by the spec loop below, when a spec only parses
  // once the body has run (see the retry there).
  let mut post_body_bindings: Option<Vec<(String, String)>> = None;
  // A slider's bounds may be symbols the body assigns before doing
  // anything else — `Manipulate[tmin = 0; tmax = 2 Pi; …, {{t, 0}, tmin,
  // tmax}]`. Wolfram evaluates the body before laying the controls out, so
  // those names are bound by the time the slider is built. Running the
  // leading run of plain assignments is enough to resolve them, and cannot
  // do anything the body would not have done anyway.
  //
  // The control variables are installed while those assignments run, since
  // Wolfram evaluates the body at their initial values. Without them a
  // leading assignment that calls a recursive helper on a control variable
  // — `u = {…, ss[-1., 1., n, dc]}` in front of a nested-circles
  // Demonstration — recurses on a symbolic depth that never reaches the
  // base case, and the widget never appears.
  if let Some(body) = args.first() {
    // Quietly: this is a probe of the body's leading run only, so an
    // assignment in it may well complain about a symbol that the full
    // evaluation would have in hand (`ring = Table[…, {i, n}]` reports an
    // iterator without bounds for an `n` that is not a control variable).
    // Wolfram, which evaluates the whole body, says nothing.
    crate::push_quiet();
    crate::with_scoped_globals(&initial_bindings, || {
      for stmt in leading_assignments(body) {
        let _ = evaluate_expr_to_expr(stmt);
      }
    });
    crate::pop_quiet();
  }
  // A `ButtonBar`'s buttons are computed — `ButtonBar[Table[…]]` builds one
  // per vertex of a graph, labelling each from a list the `Initialization`
  // option defines. Those definitions therefore have to exist before the
  // control can be built at all, so run the initialization here rather than
  // only in the frontend. Nothing else needs it, so nothing else pays for
  // it.
  if arg_items.iter().any(is_button_bar) {
    for spec in &arg_items {
      if let Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } = spec
        && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Initialization")
      {
        let _ = evaluate_expr_to_expr(replacement);
      }
    }
  }
  for spec in &arg_items {
    // Options such as `Initialization :> …` or `TrackedSymbols :> …`
    // are not variable specs; extract what we understand and ignore
    // the rest rather than failing the whole extraction.
    if let Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } = spec
    {
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Initialization")
      {
        initialization = Some(crate::syntax::expr_to_input_form(replacement));
      }
      // `Appearance -> None` hides the control rows; the animation just
      // runs (an animated widget keeps its play/pause toggle).
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Appearance")
        && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "None")
      {
        appearance_none = true;
      }
      // `AnimationRunning -> False` builds the widget paused.
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "AnimationRunning")
        && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "False")
      {
        animation_running = false;
      }
      // `TrackedSymbols :> {a, b}` narrows re-evaluation to those
      // variables; a single symbol may be given bare. `All` / `Full` /
      // `Automatic` / `True` / `Manipulate` (and anything else) keep the
      // default of tracking everything — a bare `True` in particular must
      // not be mistaken for a variable named `True`, which would leave
      // every control untracked and the display frozen.
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "TrackedSymbols")
      {
        tracked_symbols = match replacement.as_ref() {
          Expr::List(items) => items
            .iter()
            .map(|it| match it {
              Expr::Identifier(s) => Some(s.clone()),
              _ => None,
            })
            .collect::<Option<Vec<_>>>(),
          // `TrackedSymbols -> None`: no variable re-runs the body.
          Expr::Identifier(s) if s == "None" => Some(Vec::new()),
          Expr::Identifier(s) if !is_track_everything_symbol(s) => {
            Some(vec![s.clone()])
          }
          _ => None,
        };
      }
      continue;
    }
    // `Delimiter` and string / `Style[…]` / `Text[…]` arguments are static
    // annotation rows between controls (Wolfram's `ThisIsNotAControl`),
    // keeping their position among the control rows.
    match spec {
      Expr::Identifier(s) if s == "Delimiter" => {
        controls.push(ManipulateControl::Divider);
        continue;
      }
      Expr::String(_) => {
        let label_runs = manipulate_label_runs(spec, false);
        controls.push(ManipulateControl::Heading {
          label: flatten_label_runs(&label_runs),
          label_runs,
        });
        continue;
      }
      Expr::FunctionCall { name, .. }
        if is_manipulate_annotation_head(name)
          && !annotation_contains_dynamic(spec) =>
      {
        let label_runs = manipulate_label_runs(spec, false);
        controls.push(ManipulateControl::Heading {
          label: flatten_label_runs(&label_runs),
          label_runs,
        });
        continue;
      }
      // A `Row`/`Column`/`Style`/`Text` annotation with a `Dynamic[…]`
      // somewhere inside (a live step counter, e.g.
      // `Row[{Style["moves: "], Dynamic[moves]}]`) is not static text: fall
      // through to the generic "extra display element" handling below so it
      // re-renders from the live bindings every frame instead of being
      // frozen into a `Heading` with the `Dynamic` wrapper's bare source.
      Expr::FunctionCall { name, .. }
        if is_manipulate_annotation_head(name)
          && annotation_contains_dynamic(spec) =>
      {
        displays.push(crate::syntax::expr_to_input_form(spec));
        continue;
      }
      // `Button[label, action, opts…]`: a pressable control row whose
      // action code runs against the live bindings (e.g. a reset button).
      Expr::FunctionCall { name, args }
        if name == "Button" && args.len() >= 2 =>
      {
        let label_runs = manipulate_label_runs(&args[0], false);
        controls.push(ManipulateControl::Button {
          label: flatten_label_runs(&label_runs),
          label_runs,
          action: crate::syntax::expr_to_input_form(&args[1]),
        });
        continue;
      }
      // `ButtonBar[{label :> action, …}, opts…]`: a row of pressable
      // buttons, one per rule. The list is usually computed (a `Table` over
      // the things being offered), so it is evaluated first; each rule then
      // becomes its own button, keeping its action held.
      Expr::FunctionCall { name, args }
        if name == "ButtonBar" && !args.is_empty() =>
      {
        let entries =
          evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
        if let Expr::List(items) = &entries {
          let mut any = false;
          for item in items {
            let (Expr::Rule {
              pattern,
              replacement,
            }
            | Expr::RuleDelayed {
              pattern,
              replacement,
            }) = item
            else {
              continue;
            };
            let label_runs = manipulate_label_runs(pattern, false);
            controls.push(ManipulateControl::Button {
              label: flatten_label_runs(&label_runs),
              label_runs,
              action: crate::syntax::expr_to_input_form(replacement),
            });
            any = true;
          }
          if any {
            continue;
          }
        }
      }
      // `Spacer[…]` is pure layout between grouped controls — skip it.
      Expr::FunctionCall { name, .. } if name == "Spacer" => continue,
      // The span markers of a `Grid` control panel say that the cell to
      // their left (or above) stretches into this slot. Once the grid has
      // been flattened into a list of control rows they mark nothing at
      // all, so they are dropped rather than mistaken for display
      // elements — which would put a literal `SpanFromLeft` under the
      // widget, one per marker.
      Expr::Identifier(s)
        if matches!(
          s.as_str(),
          "SpanFromLeft" | "SpanFromAbove" | "SpanFromBoth" | "Null"
        ) =>
      {
        continue;
      }
      _ => {}
    }
    // Only list-shaped arguments are control specs (layout containers of
    // controls were already flattened into `arg_items` above). Any other
    // trailing argument (e.g. a `Dynamic[Panel[…]]` of checkboxes) is an
    // extra display element: capture it so the frontend can render it
    // live.
    if !matches!(spec, Expr::List(_)) {
      displays.push(crate::syntax::expr_to_input_form(spec));
      continue;
    }
    let (spec, rename) = rewrite_compound_control_var(spec);
    let parsed = if let Some(parsed) =
      crate::with_scoped_globals(&initial_bindings, || {
        parse_manipulate_control(&spec, &sibling_names)
      }) {
      parsed
    } else {
      // Wolfram evaluates the body once before laying the controls out, so
      // a control whose choice list is a symbol the *body* fills in —
      // `{{k, 9, " "}, choices, ControlType -> PopupMenu}` next to a
      // `{{choices, {}}, ControlType -> None}` state variable — already has
      // its choices in hand by then. The leading-assignment probe above
      // only reaches assignments the body opens with, so retry against the
      // bindings a full body run leaves behind. That run is expensive, so
      // it happens lazily: only after a spec has actually failed, and only
      // once per Manipulate.
      let after_body = post_body_bindings
        .get_or_insert_with(|| {
          manipulate_post_body_bindings(args.first(), &initial_bindings)
        })
        .clone();
      crate::with_scoped_globals(&after_body, || {
        parse_manipulate_control(&spec, &sibling_names)
      })?
    };
    match parsed {
      ParsedControl::Visible {
        control: mut c,
        enabled,
        min_code,
        max_code,
        values_code,
        animate,
        tracking: tracking_fn,
      } => {
        if let Some((orig, orig_form, synth)) = &rename {
          patch_default_label(&mut c, orig, synth);
          renames.push((orig_form.clone(), synth.clone()));
        }
        // A `Trigger`/`Animator` control spec animates its variable: the
        // widget auto-plays (paused for a Trigger, running for an Animator)
        // and the animation targets this variable.
        if let Some(running) = animate {
          animated = true;
          animation_running = running;
          animation_var = Some(c.name().to_string());
        }
        // A second spec for an already-bound variable merges into the
        // earlier row — contributing only the animation/enabled semantics
        // captured above — in the two cases Wolfram itself collapses: a
        // `Trigger` pairing with an existing slider (Kepler's time slider
        // plus a `Trigger` on the same `t`), and a variable declared inside
        // a `PaneSelector`/`TabView` pane (only one pane is ever on screen,
        // so a shared widget — or a per-pane variant of one, with its own
        // bounds or choice list — still gets a single row; see
        // `pane_or_tab_governed_names`). Two *different* ordinary specs
        // sharing a variable outside any pane (e.g. a coarse and a fine
        // SetterBar preset row for the same count, as in "Polypath
        // Iterations") are a real Wolfram pattern instead: both stay
        // visible, independently interactive, and read/write the same
        // binding, so those get their own rows.
        if (animate.is_some() || pane_governed_names.contains(c.name()))
          && !c.name().is_empty()
          && controls.iter().any(|prev| prev.name() == c.name())
        {
          continue;
        }
        if let Some(cond) = enabled {
          control_enabled.push((c.name().to_string(), cond));
        }
        if let Some(code) = tracking_fn {
          tracking.push((c.name().to_string(), code));
        }
        if min_code.is_some() || max_code.is_some() {
          dynamic_bounds.push((c.name().to_string(), min_code, max_code));
        }
        if let Some(code) = values_code {
          dynamic_values.push((c.name().to_string(), code));
        }
        controls.push(c);
      }
      ParsedControl::Fixed { name, value } => {
        if let Some((_, orig_form, synth)) = &rename {
          renames.push((orig_form.clone(), synth.clone()));
        }
        fixed.push((name, value));
      }
      ParsedControl::StateWithDisplay {
        name,
        value,
        display,
      } => {
        if let Some((_, orig_form, synth)) = &rename {
          renames.push((orig_form.clone(), synth.clone()));
        }
        state.push((name, value));
        displays.push(display);
      }
      ParsedControl::State { name, value } => {
        if let Some((_, orig_form, synth)) = &rename {
          renames.push((orig_form.clone(), synth.clone()));
        }
        // A hidden variable driven by an in-body `Locator[Dynamic[…]]`
        // becomes a visible Locator-style control: re-parse its spec with
        // the `ControlType -> None` option swapped for a `Locator` marker,
        // and carry the Dynamic's write-back callback so candidate values
        // are validated the way Wolfram would.
        if let Some((_, callback)) =
          body_locators.iter().find(|(n, _)| *n == name)
          && let Expr::List(items) = &spec
        {
          let promoted: Vec<Expr> = items
            .iter()
            .filter(|it| !is_control_type_rule(it))
            .cloned()
            .chain(std::iter::once(Expr::Identifier("Locator".to_string())))
            .collect();
          if let Some(ParsedControl::Visible {
            control: mut c,
            enabled: enabled2,
            ..
          }) = parse_manipulate_control(&Expr::List(promoted.into()), &[])
          {
            if let ManipulateControl::Slider2D { write_callback, .. } = &mut c {
              write_callback.clone_from(callback);
            }
            if let Some(cond) = enabled2 {
              control_enabled.push((c.name().to_string(), cond));
            }
            controls.push(c);
            continue;
          }
        }
        // Likewise for a hidden variable a `PopupMenu[Dynamic[…]]` in the
        // body drives: it becomes a real pick list, built from the choice
        // list the body computes. The list is re-resolved on every frame
        // (through `dynamic_values`) because it usually depends on the
        // other controls — and the body draws it that way too.
        if let Some((_, choices_code)) =
          body_popups.iter().find(|(n, _)| *n == name)
          && let Some(choices) =
            crate::with_scoped_globals(&initial_bindings, || {
              crate::interpret_to_expr(choices_code)
                .ok()
                .and_then(|e| evaluate_expr_to_expr(&e).ok())
            })
          && matches!(choices, Expr::List(_))
        {
          let default = crate::interpret_to_expr(&value)
            .unwrap_or_else(|_| Expr::Identifier(value.clone()));
          let promoted = Expr::List(
            vec![
              Expr::List(vec![Expr::Identifier(name.clone()), default].into()),
              choices,
              Expr::Rule {
                pattern: Box::new(Expr::Identifier("ControlType".to_string())),
                replacement: Box::new(Expr::Identifier(
                  "PopupMenu".to_string(),
                )),
              },
            ]
            .into(),
          );
          if let Some(ParsedControl::Visible {
            control: c,
            enabled: enabled2,
            ..
          }) = parse_manipulate_control(&promoted, &[])
          {
            if let Some(cond) = enabled2 {
              control_enabled.push((c.name().to_string(), cond));
            }
            dynamic_values.push((name.clone(), choices_code.clone()));
            promoted_popups.push(name.clone());
            controls.push(c);
            continue;
          }
        }
        state.push((name, value));
      }
      ParsedControl::StateWithControl {
        name,
        value,
        control,
      } => {
        if let Some((_, orig_form, synth)) = &rename {
          renames.push((orig_form.clone(), synth.clone()));
        }
        state.push((name, value));
        controls.push(control);
      }
    }
  }

  // A Manipulate with no controls or state at all (e.g. `Manipulate[x^2,
  // badspec]`, where `badspec` is neither a spec nor an option) isn't
  // renderable as an interactive widget — fall back to the plain path.
  // Annotation rows alone don't make a widget either.
  if !controls.iter().any(|c| !c.name().is_empty())
    && fixed.is_empty()
    && state.is_empty()
  {
    return None;
  }

  // Rewrite every occurrence of a renamed compound variable in the code
  // fragments that reference it. All fragments were printed by
  // `expr_to_input_form`, so the original variable appears verbatim
  // (including as a call head: `Subscript[signal, 1][x]`). Longest
  // original first so no rename can match inside a longer one.
  let mut body_code = body_code;
  // A pick list that became a control is no longer part of the body — left
  // in, it would print as its own source next to the control that replaced
  // it.
  if !promoted_popups.is_empty()
    && let Some(body_expr) = &body_expr_kept
  {
    body_code = crate::syntax::expr_to_input_form(&strip_body_popup_menus(
      body_expr,
      &promoted_popups,
    ));
  }
  if !renames.is_empty() {
    renames.sort_by_key(|(orig, _)| std::cmp::Reverse(orig.len()));
    let rewrite = |code: &mut String| {
      for (orig, synth) in &renames {
        *code = code.replace(orig.as_str(), synth.as_str());
      }
    };
    rewrite(&mut body_code);
    for d in &mut displays {
      rewrite(d);
    }
    if let Some(init) = &mut initialization {
      rewrite(init);
    }
    for (_, cond) in
      control_enabled.iter_mut().chain(control_visible.iter_mut())
    {
      rewrite(cond);
    }
    for (_, min_code, max_code) in &mut dynamic_bounds {
      if let Some(code) = min_code {
        rewrite(code);
      }
      if let Some(code) = max_code {
        rewrite(code);
      }
    }
    for (_, code) in &mut dynamic_values {
      rewrite(code);
    }
    for (_, value) in fixed.iter_mut().chain(state.iter_mut()) {
      rewrite(value);
    }
  }

  // Bake fixed (Locator) bindings into the body so they remain in scope on
  // every re-evaluation, independent of the visible control state. Mutable
  // `state` bindings are not baked — they travel live in the binding set.
  let body_code = if fixed.is_empty() {
    body_code
  } else {
    manipulate_block_code(&body_code, &fixed)
  };

  Some(ManipulateSpec {
    body_code,
    controls,
    state,
    displays,
    initialization,
    control_enabled,
    control_visible,
    dynamic_bounds,
    dynamic_values,
    animation_var,
    animated,
    animation_running,
    appearance_none,
    tracked_symbols,
    tracking,
  })
}

/// Whether `s` names one of Wolfram's control types, i.e. a value
/// `ControlType -> …` accepts (which a variable spec may also carry as a bare
/// marker, as in `{u, {1, 2, 3}, SetterBar}`).
fn is_control_type_name(s: &str) -> bool {
  matches!(
    s,
    "Locator"
      | "Slider"
      | "Slider2D"
      | "VerticalSlider"
      | "Manipulator"
      | "InputField"
      | "PopupMenu"
      | "Setter"
      | "SetterBar"
      | "RadioButton"
      | "RadioButtonBar"
      | "Toggler"
      | "TogglerBar"
      | "Checkbox"
      | "CheckboxBar"
      | "Opener"
      | "OpenerBar"
      | "ColorSlider"
      | "ColorSetter"
      | "IntervalSlider"
      | "Animator"
      | "Trigger"
      | "Automatic"
  )
}

/// Whether a variable spec picks its own control type, either as a
/// `ControlType -> …` option or as a bare marker after the head.
fn spec_declares_control_type(items: &[Expr]) -> bool {
  items.iter().any(is_control_type_rule)
    || items
      .iter()
      .skip(1)
      .any(|it| matches!(it, Expr::Identifier(s) if is_control_type_name(s)))
}

/// Push a Manipulate-level `ControlType -> …` option down into the variable
/// specs, which is where the control type is read from.
///
/// `Manipulate[body, {u, …}, {v, …}, ControlType -> PopupMenu]` gives *every*
/// control that does not choose a type of its own a popup menu — the
/// Demonstrations idiom for laying controls out in a `Grid[…]` and then
/// setting their type once. A list value assigns one type per control, in the
/// order the specs appear (`ControlType -> {Slider, PopupMenu}`); controls
/// past the end of the list keep their default. `Automatic` means "decide as
/// usual", so it is left off entirely.
fn apply_global_control_type(items: Vec<Expr>) -> Vec<Expr> {
  let global = items.iter().find_map(|it| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = it
    else {
      return None;
    };
    matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ControlType")
      .then(|| replacement.as_ref().clone())
  });
  let Some(global) = global else {
    return items;
  };
  let per_spec = match &global {
    Expr::List(types) => Some(types.to_vec()),
    Expr::Identifier(_) => None,
    // Anything else is not a control type at all — leave the specs alone.
    _ => return items,
  };
  let mut spec_index = 0usize;
  items
    .into_iter()
    .map(|item| {
      let Expr::List(spec) = &item else {
        return item;
      };
      let index = spec_index;
      spec_index += 1;
      let control_type = match &per_spec {
        Some(types) => match types.get(index) {
          Some(t) => t.clone(),
          None => return item,
        },
        None => global.clone(),
      };
      if !matches!(&control_type, Expr::Identifier(s) if is_control_type_name(s) && s != "Automatic")
        || spec_declares_control_type(spec)
      {
        return item;
      }
      let extended: Vec<Expr> = spec
        .iter()
        .cloned()
        .chain(std::iter::once(Expr::Rule {
          pattern: Box::new(Expr::Identifier("ControlType".to_string())),
          replacement: Box::new(control_type),
        }))
        .collect();
      Expr::List(extended.into())
    })
    .collect()
}

/// Whether a control-spec item is a `ControlType -> …` option rule.
fn is_control_type_rule(item: &Expr) -> bool {
  matches!(
    item,
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ControlType")
  )
}

/// The variables driven by `Locator[Dynamic[var, cb], …]` markers inside a
/// Manipulate body, each with the InputForm of its write-back callback (the
/// Dynamic's second argument), in first-seen order.
fn collect_body_locator_callbacks(
  expr: &Expr,
) -> Vec<(String, Option<String>)> {
  fn walk(expr: &Expr, found: &mut Vec<(String, Option<String>)>) {
    match expr {
      Expr::FunctionCall { name, args } => {
        if name == "Locator"
          && let Some(Expr::FunctionCall {
            name: dname,
            args: dargs,
          }) = args.first()
          && dname == "Dynamic"
          && let Some(Expr::Identifier(var)) = dargs.first()
          && !found.iter().any(|(n, _)| n == var)
        {
          let callback = dargs.get(1).map(crate::syntax::expr_to_input_form);
          found.push((var.clone(), callback));
        }
        for a in args {
          walk(a, found);
        }
      }
      Expr::List(items) => {
        for it in items {
          walk(it, found);
        }
      }
      Expr::CompoundExpr(items) => {
        for it in items {
          walk(it, found);
        }
      }
      _ => {}
    }
  }
  let mut found = Vec::new();
  walk(expr, &mut found);
  found
}

/// The variable and choice-list code of every `PopupMenu[Dynamic[var], …]`
/// written inside a Manipulate body.
///
/// Putting a pick list in the body rather than in the control panel is how a
/// Demonstration places it inside its own layout. The choice list normally
/// depends on locals the body itself introduces (`With[{choices = …}, …
/// PopupMenu[Dynamic[an], choices] …]`), so the code returned here re-wraps
/// the list in whatever `With`/`Module`/`Block` scopes enclose it — that
/// makes it evaluable on its own, outside the body, which is what promoting
/// the pick list to a real control needs.
fn collect_body_popup_menus(expr: &Expr) -> Vec<(String, String)> {
  fn walk(
    expr: &Expr,
    scopes: &mut Vec<(String, Expr)>,
    found: &mut Vec<(String, String)>,
  ) {
    match expr {
      Expr::FunctionCall { name, args } => {
        if matches!(name.as_str(), "With" | "Module" | "Block")
          && args.len() == 2
        {
          scopes.push((name.clone(), args[0].clone()));
          walk(&args[1], scopes, found);
          scopes.pop();
          return;
        }
        if name == "PopupMenu"
          && args.len() >= 2
          && let Some(Expr::FunctionCall {
            name: dname,
            args: dargs,
          }) = args.first()
          && dname == "Dynamic"
          && let Some(Expr::Identifier(var)) = dargs.first()
          && !found.iter().any(|(n, _)| n == var)
        {
          let mut code = args[1].clone();
          for (head, binds) in scopes.iter().rev() {
            code = Expr::FunctionCall {
              name: head.clone(),
              args: vec![binds.clone(), code].into(),
            };
          }
          found.push((var.clone(), crate::syntax::expr_to_input_form(&code)));
        }
        for a in args {
          walk(a, scopes, found);
        }
      }
      Expr::List(items) => {
        for it in items {
          walk(it, scopes, found);
        }
      }
      Expr::CompoundExpr(items) => {
        for it in items {
          walk(it, scopes, found);
        }
      }
      _ => {}
    }
  }
  let mut found = Vec::new();
  walk(expr, &mut Vec::new(), &mut found);
  found
}

/// Replace each `PopupMenu[Dynamic[var], …]` whose `var` is listed in
/// `promoted` with `Nothing`, so the pick list is not also printed as source
/// inside the body it was lifted out of.
fn strip_body_popup_menus(expr: &Expr, promoted: &[String]) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "PopupMenu"
        && matches!(
          args.first(),
          Some(Expr::FunctionCall { name: dname, args: dargs })
            if dname == "Dynamic"
              && matches!(
                dargs.first(),
                Some(Expr::Identifier(v)) if promoted.contains(v)
              )
        ) =>
    {
      Expr::Identifier("Nothing".to_string())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| strip_body_popup_menus(a, promoted))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|it| strip_body_popup_menus(it, promoted))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::CompoundExpr(items) => Expr::CompoundExpr(
      items
        .iter()
        .map(|it| strip_body_popup_menus(it, promoted))
        .collect(),
    ),
    other => other.clone(),
  }
}

/// Replace every `TogglerBar[Dynamic[var], …]` in a Manipulate body with
/// `Nothing`, pushing each one's InputForm onto `displays` so the front-end
/// renders it as a live widget instead of a static picture.
fn extract_body_togglerbars(expr: &Expr, displays: &mut Vec<String>) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "TogglerBar"
        && matches!(
          args.first(),
          Some(Expr::FunctionCall { name: dname, args: dargs })
            if dname == "Dynamic"
              && matches!(dargs.first(), Some(Expr::Identifier(_)))
        ) =>
    {
      displays.push(crate::syntax::expr_to_input_form(expr));
      Expr::Identifier("Nothing".to_string())
    }
    // A bare `Button[label, action, opts…]` drawn directly by the body (a
    // Demonstration's "throw"/"step" action mixed into a `Column` of
    // graphics and captions), distinct from a `Button` given as its own
    // Manipulate control-spec argument (handled in `manipulate_controls`).
    // Moves into the display list, replaced by `Nothing`, so it renders as
    // a live, clickable element instead of an inert picture.
    Expr::FunctionCall { name, args }
      if name == "Button" && args.len() >= 2 =>
    {
      displays.push(crate::syntax::expr_to_input_form(expr));
      Expr::Identifier("Nothing".to_string())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| extract_body_togglerbars(a, displays))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|it| extract_body_togglerbars(it, displays))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::CompoundExpr(items) => Expr::CompoundExpr(
      items
        .iter()
        .map(|it| extract_body_togglerbars(it, displays))
        .collect(),
    ),
    other => other.clone(),
  }
}

/// Apply a control's write-back callback to a candidate value: evaluates
/// `(cb)[value]` under the current bindings (the callback usually assigns
/// the control variable itself, possibly transformed or rejected), then
/// reads the variable back. Returns its new InputForm value, or `None` when
/// evaluation fails.
pub fn apply_manipulate_callback(
  bindings: &[(String, String)],
  callback: &str,
  value: &str,
  var: &str,
) -> Option<String> {
  let body = format!("({callback})[{value}]; {var}");
  let code = manipulate_block_code(&body, bindings);
  crate::interpret_to_expr(&code)
    .ok()
    .map(|e| crate::syntax::expr_to_input_form(&e))
}

/// Parse an InputForm `{x, y}` point (as stored in a Manipulate binding)
/// back into coordinates.
pub fn parse_manipulate_point(code: &str) -> Option<(f64, f64)> {
  let expr = crate::interpret_to_expr(code).ok()?;
  list2_f64(&expr)
}

/// A Manipulate body wrapped in `Dynamic[…]` displays the Dynamic's first
/// argument — the wrapper only adds FrontEnd tracking hints (a second
/// update-function argument, `TrackedSymbols :> …`) — so evaluation uses
/// that inner expression.
fn unwrap_dynamic_body(body: &Expr) -> &Expr {
  match body {
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && !args.is_empty() =>
    {
      &args[0]
    }
    other => other,
  }
}

/// Peel `Dynamic[…]` wrappers and any top-level `DynamicModule[{locals…},
/// …]` off a Manipulate body, pushing each local's `(name, initial value)`
/// onto `state`. A `DynamicModule` local without an initializer (`{tf,
/// soln}`, computed fresh each frame) is seeded as `Null`.
///
/// Wolfram keeps a `DynamicModule`'s locals alive for the widget's whole
/// lifetime — that is the entire point of `DynamicModule` over `Module` —
/// so a `Button` inside it that writes one (a Demonstration's "throw" or
/// "step" action) must have that write survive the next re-render. Hoisting
/// the locals into the widget's hidden state, the same mechanism a
/// Manipulate's own `ControlType -> None` variables use, gets that for
/// free: `reevaluate` installs `state` as scoped globals before every
/// render, and writes the body (or a button action) makes to any of those
/// names get read back afterwards.
fn unwrap_dynamic_module_locals(
  body: &Expr,
  state: &mut Vec<(String, String)>,
) -> Expr {
  let mut cur = body;
  loop {
    match cur {
      Expr::FunctionCall { name, args }
        if name == "Dynamic" && !args.is_empty() =>
      {
        cur = &args[0];
      }
      // Only a `DynamicModule` whose own body is *itself* an explicit
      // `Dynamic[…]` gets its locals hoisted: that inner `Dynamic` is the
      // author's own re-render boundary, marking everything outside it —
      // the locals — as evaluated once and persisted, exactly what
      // `DynamicModule` (vs. `Module`) is for. A `DynamicModule` whose body
      // is plain code (`DynamicModule[{p = a x^2+b x+c}, RegionPlot[…]]`,
      // no inner `Dynamic`) has no such boundary: Manipulate's own
      // (implicit, outer) dynamic wrapping re-evaluates the whole
      // `DynamicModule` — including the local's initializer — on every
      // control change, so `p` must keep tracking `a`/`b`/`c` fresh each
      // frame rather than freezing at its first value.
      Expr::FunctionCall { name, args }
        if name == "DynamicModule"
          && args.len() >= 2
          && matches!(
            &args[1],
            Expr::FunctionCall { name: inner, args: iargs }
              if inner == "Dynamic" && !iargs.is_empty()
          ) =>
      {
        if let Expr::List(locals) = &args[0] {
          for local in locals {
            match local {
              Expr::FunctionCall {
                name: set_name,
                args: set_args,
              } if set_name == "Set" && set_args.len() == 2 => {
                if let Expr::Identifier(var_name) = &set_args[0] {
                  state.push((
                    var_name.clone(),
                    crate::syntax::expr_to_input_form(&set_args[1]),
                  ));
                }
              }
              Expr::Rule {
                pattern,
                replacement,
              } => {
                if let Expr::Identifier(var_name) = pattern.as_ref() {
                  state.push((
                    var_name.clone(),
                    crate::syntax::expr_to_input_form(replacement),
                  ));
                }
              }
              Expr::Identifier(var_name) => {
                state.push((var_name.clone(), "Null".to_string()));
              }
              _ => {}
            }
          }
        }
        cur = &args[1];
      }
      _ => return cur.clone(),
    }
  }
}

/// The flattened control items of a `Row[…]`/`Column[…]`/`Grid[…]`
/// Manipulate argument that lays several controls out in one row (the
/// Wolfram Demonstrations pattern `Row[{Control[…], Spacer[20],
/// Button[…]}]`). Returns `None` when the argument contains no
/// `Control[…]`/`Button[…]` anywhere, so plain `Row[…]` display
/// expressions keep flowing to the display path.
fn control_group_items(spec: &Expr) -> Option<Vec<Expr>> {
  fn contains_control(e: &Expr) -> bool {
    match e {
      Expr::FunctionCall { name, args } => {
        name == "Control"
          || name == "Button"
          || args.iter().any(contains_control)
      }
      Expr::List(items) => items.iter().any(contains_control),
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => contains_control(pattern) || contains_control(replacement),
      // `Sequence@@If[cond, Control[…], {}]`: the condition/branches may
      // hide the only control a Dynamic-wrapped list declares, so look
      // inside the `f@@expr` the same way a plain function call is scanned.
      Expr::Apply { func, list } => {
        contains_control(func) || contains_control(list)
      }
      _ => false,
    }
  }
  // `Item[content, opts…]` is a grid-alignment wrapper (the Demonstrations
  // idiom for lining up a whole control panel inside an outer `Grid`), not
  // a layout container in its own right — unwrap it to reach the container
  // it dresses up.
  let spec = match spec {
    Expr::FunctionCall { name, args } if name == "Item" && !args.is_empty() => {
      &args[0]
    }
    _ => spec,
  };
  let Expr::FunctionCall { name, args } = spec else {
    return None;
  };
  // `Dynamic[{ctrl1, ctrl2, …}]` (the Demonstrations idiom for a control
  // panel whose row set itself needs to react to another control, e.g. a
  // control only shown for some mode) flattens the same way a `Column` of
  // controls would — the `Dynamic` just marks that the list is recomputed
  // reactively, which the panel already does on every re-evaluation. A
  // `Dynamic` wrapping anything other than a bare list (a live picture, a
  // `Panel[…]` of checkboxes, …) is a display element instead and falls
  // through via the `contains_control` guard below.
  if !matches!(
    name.as_str(),
    "Row" | "Column" | "Grid" | "TabView" | "PaneSelector" | "Dynamic"
  ) || args.is_empty()
  {
    return None;
  }
  if !contains_control(spec) {
    return None;
  }
  let Expr::List(items) = &args[0] else {
    return None;
  };
  let mut out = Vec::new();
  for item in items {
    // A `TabView` lists its tabs as `label -> content`, and a
    // `PaneSelector` its panes as `value -> content`; only the content
    // holds controls. (Woxi's control panel is one flat list, so every
    // pane's controls are collected into it, and a variable declared in
    // more than one pane is registered once, from the first pane that
    // declares it. `collect_pane_visibility` records which pane each one
    // came from, so the frontend can still show one panel at a time.)
    let item = match (name.as_str(), item) {
      (
        "TabView" | "PaneSelector",
        Expr::Rule { replacement, .. } | Expr::RuleDelayed { replacement, .. },
      ) => {
        // A pane holding no control at all contributes nothing to the flat
        // list — a placeholder pane (`3 -> " "`, the Demonstrations idiom
        // for "this mode has no controls") must not leave a blank heading
        // row behind.
        if !contains_control(replacement) {
          continue;
        }
        replacement.as_ref()
      }
      _ => item,
    };
    match item {
      // A Grid's first level is its list of layout rows; their elements
      // are the actual items.
      Expr::List(row) if name == "Grid" => out.extend(row.iter().cloned()),
      // Nested layout containers flatten recursively.
      other => match control_group_items(other) {
        Some(nested) => out.extend(nested),
        None => out.push(other.clone()),
      },
    }
  }
  Some(out)
}

/// `Dynamic[Control[…]]` (the Demonstrations idiom `Dynamic@Control@{…}`)
/// is the `Control` it wraps — the `Dynamic` only adds FrontEnd update
/// hints — and a bare `Control[spec, opts…]` wrapper is its ordinary
/// variable specification, so both unwrap to the plain spec that parses
/// through the standard control path. Anything else passes through
/// unchanged. Applied both to the top-level control-spec arguments and to
/// whatever `expand_conditional_control_items` splices in, since a spliced
/// control is just as likely to arrive `Control`-wrapped.
fn unwrap_control_wrapper(spec: Expr) -> Expr {
  let spec = match &spec {
    Expr::FunctionCall { name, args }
      if name == "Dynamic"
        && matches!(
          args.first(),
          Some(Expr::FunctionCall { name: inner, .. }) if inner == "Control"
        ) =>
    {
      args[0].clone()
    }
    _ => spec,
  };
  match &spec {
    Expr::FunctionCall { name, args }
      if name == "Control" && !args.is_empty() =>
    {
      args[0].clone()
    }
    _ => spec,
  }
}

/// Resolve `Sequence@@expr` entries in a flattened control list (the
/// Demonstrations idiom `Sequence@@If[cond, ctrlSpec, {}]` for a control
/// only shown under some condition on another control) by evaluating each
/// one against `bindings` — the other controls' initial values — and
/// splicing whatever list the evaluation produces in its place, the way
/// Wolfram evaluates the whole spec list once before laying out the panel.
/// A splice may itself be a layout container or contain further
/// `Sequence@@…` entries (nested conditions), so both are re-resolved
/// recursively. Non-`Sequence` entries pass through unchanged.
fn expand_conditional_control_items(
  items: Vec<Expr>,
  bindings: &[(String, String)],
) -> Vec<Expr> {
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    let Expr::Apply { func, list } = &item else {
      out.push(item);
      continue;
    };
    if !matches!(func.as_ref(), Expr::Identifier(s) if s == "Sequence") {
      out.push(item);
      continue;
    }
    let evaluated =
      crate::with_scoped_globals(bindings, || evaluate_expr_to_expr(list))
        .unwrap_or_else(|_| (**list).clone());
    let spliced: Vec<Expr> = match &evaluated {
      Expr::List(inner) => inner.iter().cloned().collect(),
      _ => vec![evaluated.clone()],
    };
    let flattened: Vec<Expr> = spliced
      .into_iter()
      .flat_map(|s| match control_group_items(&s) {
        Some(nested) => nested,
        None => vec![s],
      })
      .collect();
    out.extend(expand_conditional_control_items(flattened, bindings));
  }
  out
}

/// Record, for every control declared inside a `PaneSelector[{v -> content,
/// …}, sel]` argument, the condition under which its pane is on screen
/// (`sel == v`). Woxi lays every pane's controls out in one flat list, so
/// these conditions are what let a frontend hide the rows belonging to the
/// panes the selector is not showing.
///
/// A variable declared in several panes (a Demonstration reusing one angle
/// slider across two modes) is visible in all of them, so its conditions
/// are or-ed together. Only the outermost `PaneSelector` of an argument is
/// honoured — a pane nested inside another pane keeps its parent's
/// condition rather than gaining its own.
fn collect_pane_visibility(spec: &Expr, out: &mut Vec<(String, String)>) {
  let Expr::FunctionCall { name, args } = spec else {
    return;
  };
  if name != "PaneSelector" {
    // The `PaneSelector` may sit inside a layout container.
    for arg in args {
      collect_pane_visibility(arg, out);
    }
    return;
  }
  let (Some(Expr::List(panes)), Some(selector)) = (args.first(), args.get(1))
  else {
    return;
  };
  let selector = crate::syntax::expr_to_input_form(selector);
  for pane in panes {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = pane
    else {
      continue;
    };
    let cond = format!(
      "({}) == ({})",
      selector,
      crate::syntax::expr_to_input_form(pattern)
    );
    for var in pane_control_variables(replacement) {
      match out.iter_mut().find(|(n, _)| *n == var) {
        Some((_, existing)) => *existing = format!("{existing} || {cond}"),
        None => out.push((var, cond.clone())),
      }
    }
  }
}

/// The Manipulate variable names declared inside a `PaneSelector`/`TabView`
/// pane or tab, anywhere among `args` (a Manipulate's control-spec
/// arguments). Only one pane/tab is ever on screen at a time, so a
/// duplicate spec for one of these names — the same widget shared across
/// panes, or a per-pane variant of it (different bounds, different choice
/// list) — must still collapse to a single row; see the merge check where
/// this is used, alongside `collect_pane_visibility` which computes the
/// same panes' *display* condition for the row that does get built.
fn pane_or_tab_governed_names(
  args: &[Expr],
) -> std::collections::HashSet<String> {
  fn walk(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
      Expr::FunctionCall { name, args } => {
        if (name == "PaneSelector" || name == "TabView")
          && let Some(Expr::List(panes)) = args.first()
        {
          for pane in panes {
            if let Expr::Rule { replacement, .. }
            | Expr::RuleDelayed { replacement, .. } = pane
            {
              out.extend(pane_control_variables(replacement));
            }
          }
        }
        for a in args {
          walk(a, out);
        }
      }
      // A `PaneSelector`/`TabView` may sit inside a `Row[{…}]`/`Column[{…}]`
      // layout, whose single argument is itself a list of the grouped
      // items — walk has to descend into that list too, not just a
      // function call's own arguments, or a pane nested that way is missed.
      Expr::List(items) => {
        for it in items {
          walk(it, out);
        }
      }
      _ => {}
    }
  }
  let mut out = std::collections::HashSet::new();
  for a in args {
    walk(a, &mut out);
  }
  out
}

/// The control variables a `PaneSelector` pane declares: the variable of
/// every `Control[…]` in it, plus — when the pane *is* a bare variable
/// specification — that spec's own variable.
fn pane_control_variables(pane: &Expr) -> Vec<String> {
  fn walk(e: &Expr, out: &mut Vec<String>) {
    match e {
      Expr::FunctionCall { name, args } => {
        if name == "Control"
          && let Some(spec) = args.first()
          && let Some(var) = control_spec_variable(spec)
        {
          out.push(var);
          return;
        }
        for a in args {
          walk(a, out);
        }
      }
      Expr::List(items) => {
        for it in items {
          walk(it, out);
        }
      }
      _ => {}
    }
  }
  let mut out = Vec::new();
  if let Some(var) = control_spec_variable(pane) {
    out.push(var);
  } else {
    walk(pane, &mut out);
  }
  out
}

/// The variable a control specification binds: `{u, …}` or `{{u, init, …},
/// …}`. `None` for anything that is not a variable specification.
fn control_spec_variable(spec: &Expr) -> Option<String> {
  let Expr::List(items) = spec else {
    return None;
  };
  match items.first()? {
    Expr::List(head) => match head.first()? {
      Expr::Identifier(name) => Some(name.clone()),
      _ => None,
    },
    Expr::Identifier(name) if items.len() >= 2 => Some(name.clone()),
    _ => None,
  }
}

/// Collect `(name, initial value as InputForm)` for every control spec that
/// declares one: an explicit `{{u, uinit, …}, …}` head, or a plain-symbol
/// `{u, umin, …}` head whose lower bound is statically numeric (the default
/// initial value). Installed as scoped globals while the specs are parsed so
/// a bound referencing another control variable resolves to its build-time
/// value regardless of declaration order.
fn manipulate_initial_value_bindings(specs: &[Expr]) -> Vec<(String, String)> {
  specs
    .iter()
    .filter_map(|spec| {
      let Expr::List(items) = spec else { return None };
      match items.first()? {
        Expr::List(head) => {
          let Expr::Identifier(name) = head.first()? else {
            return None;
          };
          let init = head.get(1)?;
          Some((name.clone(), crate::syntax::expr_to_input_form(init)))
        }
        Expr::Identifier(name) => {
          let min = items.get(1)?;
          crate::functions::math_ast::try_eval_to_f64(min)?;
          Some((name.clone(), crate::syntax::expr_to_input_form(min)))
        }
        _ => None,
      }
    })
    .collect()
}

/// The expression a Manipulate bound really states, with a `Dynamic[…]`
/// wrapper stripped, plus whether such a wrapper was there. A bound written
/// `Dynamic[expr]` (`{{k, 1}, 1, Dynamic[Binomial[n, 3]], 1}` — a counter
/// whose end follows another control) is re-read by the front end whenever
/// anything it names changes, so it is dynamic by construction even when it
/// happens to be constant right now.
fn manipulate_bound_expr(expr: &Expr) -> (&Expr, bool) {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && !args.is_empty() =>
    {
      (&args[0], true)
    }
    other => (other, false),
  }
}

/// Re-read a Manipulate's control/state variables after evaluating its body
/// once, the way Wolfram initializes a `Manipulate` before laying out its
/// controls. Only the variables that already have an initial binding are
/// reported, each replaced by whatever the body left it holding, so the
/// result can stand in for `initial` wholesale. A variable the body does not
/// touch keeps its initial value, and a body that fails contributes nothing.
fn manipulate_post_body_bindings(
  body: Option<&Expr>,
  initial: &[(String, String)],
) -> Vec<(String, String)> {
  let mut out = initial.to_vec();
  let Some(body) = body else { return out };
  let body = unwrap_dynamic_body(body);
  // The body is being run outside its own controls' feedback loop, so any
  // complaint it makes here would be a duplicate of one the frontend
  // reports when it evaluates the body for real.
  crate::push_quiet();
  crate::with_scoped_globals(initial, || {
    let _ = evaluate_expr_to_expr(body);
    for (name, value) in &mut out {
      let symbol = Expr::Identifier(name.clone());
      if let Ok(evaluated) = evaluate_expr_to_expr(&symbol)
        && !matches!(&evaluated, Expr::Identifier(s) if s == name)
      {
        *value = crate::syntax::expr_to_input_form(&evaluated);
      }
    }
  });
  crate::pop_quiet();
  out
}

/// Evaluate a Manipulate bound expression to a number. A literal (`2 Pi`)
/// resolves statically; a bound referencing another control variable (`P`)
/// resolves through the evaluator against the initial-value globals the
/// caller installed. The flag reports whether the bound is dynamic — either
/// because the environment was needed or because it is wrapped in
/// `Dynamic[…]` — in which case it must be re-resolved against live bindings.
fn eval_manipulate_bound(expr: &Expr) -> Option<(f64, bool)> {
  let (expr, is_dynamic) = manipulate_bound_expr(expr);
  if let Some(v) =
    crate::functions::math_ast::try_eval_to_f64_with_infinity(expr)
  {
    return Some((v, is_dynamic));
  }
  let evaluated = crate::evaluator::evaluate_expr_to_expr(expr).ok()?;
  crate::functions::math_ast::try_eval_to_f64_with_infinity(&evaluated)
    .map(|v| (v, true))
}

/// Whether a Manipulate bound expression (a `min`/`max`/step/initial-value
/// term) is an inexact (machine-real) number rather than an exact integer or
/// rational — `0.01` or `N[Pi]`, not `1` or `1/2`. Mirrors
/// [`eval_manipulate_bound`]'s own resolution (static first, then against
/// the live environment) so a bound naming another control's variable
/// (`{{t, 0, …}, 0, P, .01}`) sees the same value. Used to decide whether a
/// continuous control's variable stays real-valued even at a "round" slider
/// position (see [`ManipulateControl::Continuous::is_real`]).
fn manipulate_bound_is_inexact(expr: &Expr) -> bool {
  let (expr, _) = manipulate_bound_expr(expr);
  // Fast path for a literal: no need to round-trip through the evaluator.
  match expr {
    Expr::Real(_) | Expr::BigFloat(_, _) => return true,
    Expr::Integer(_) | Expr::BigInteger(_) => return false,
    _ => {}
  }
  matches!(
    crate::evaluator::evaluate_expr_to_expr(expr),
    Ok(Expr::Real(_) | Expr::BigFloat(_, _))
  )
}

/// Re-resolve a dynamic bound's code fragment against the interpreter's
/// current globals (the caller installs the live bindings via
/// `with_scoped_globals`). Returns `None` when the code doesn't evaluate to
/// a finite number, in which case the control keeps its previous bound.
pub fn manipulate_eval_bound_code(code: &str) -> Option<f64> {
  let expr = crate::interpret_to_expr(code).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&expr).filter(|v| v.is_finite())
}

/// Whether `expr` mentions any of `names` as a symbol. Used to tell a
/// choice list that merely needs evaluating (`Range[20]`) from one that
/// follows another control (`Range[1, If[flat, 3, 6], 1]`) and so has to be
/// rebuilt on every change.
fn expr_references_any(expr: &Expr, names: &[String]) -> bool {
  names
    .iter()
    .any(|n| crate::functions::plot::expr_mentions_var(expr, n))
}

/// Split a discrete control's choice list into the three parallel columns
/// the frontends consume: the value bound to the variable, its display
/// label, and (for a rule label that is itself a graphic) a rendered icon.
///
/// A choice may be given as a rule `value -> "label"` (e.g. a SetterBar
/// spec `{True -> "Yin-Yang", False -> "alternate image"}`). In that case
/// the left side is the value bound to the variable and the right side is
/// only the display label, so the binding never sees the whole rule.
fn discrete_choice_columns(
  items: &[Expr],
) -> (Vec<String>, Vec<String>, Vec<Option<String>>) {
  let mut values = Vec::with_capacity(items.len());
  let mut labels = Vec::with_capacity(items.len());
  let mut svgs = Vec::with_capacity(items.len());
  for item in items {
    if let Some((value, label)) = discrete_choice_rule(item) {
      values.push(crate::syntax::expr_to_input_form(value));
      // A rule label that is itself a graphic (the crosshair icons of
      // the Demonstrations site) renders as an SVG icon; its text column
      // falls back to the bound value so a non-graphical frontend still
      // shows something short and meaningful.
      let svg = discrete_choice_label_svg(label);
      labels.push(if svg.is_some() {
        discrete_choice_label(value)
      } else {
        discrete_choice_label(label)
      });
      svgs.push(svg);
    } else if let Some(color) = crate::functions::graphics::parse_color(item) {
      // A plain colour choice (no Rule label) renders as a swatch icon —
      // the ColorSetter idiom — rather than its `RGBColor[…]` InputForm.
      values.push(crate::syntax::expr_to_input_form(item));
      labels.push(discrete_choice_label(item));
      svgs.push(Some(color_swatch_svg(&color)));
    } else {
      values.push(crate::syntax::expr_to_input_form(item));
      labels.push(discrete_choice_label(item));
      svgs.push(None);
    }
  }
  (values, labels, svgs)
}

/// Re-resolve a discrete control's choice-list code against the
/// interpreter's current globals (installed by the caller via
/// `with_scoped_globals`), returning the same three columns
/// [`discrete_choice_columns`] builds. `None` when the code no longer
/// evaluates to a non-empty list, in which case the control keeps the
/// choices it already has.
pub fn manipulate_eval_values_code(
  code: &str,
) -> Option<(Vec<String>, Vec<String>, Vec<Option<String>>)> {
  let expr = crate::interpret_to_expr(code).ok()?;
  let evaluated = crate::evaluator::evaluate_expr_to_expr(&expr).ok()?;
  let Expr::List(items) = &evaluated else {
    return None;
  };
  let columns = discrete_choice_columns(items);
  (!columns.0.is_empty()).then_some(columns)
}

/// Synthesize a plain symbol name for a compound (non-Identifier) control
/// variable. `Subscript[signal, 1]` → `signal$1`; any other compound head
/// sanitizes its whole InputForm, mapping non-symbol characters to `$`.
/// Returns `None` for heads that already are plain symbols (or head lists).
fn synthesize_var_name(expr: &Expr) -> Option<String> {
  if !matches!(expr, Expr::FunctionCall { .. }) {
    return None;
  }
  let sanitize = |s: &str| -> String {
    s.chars()
      .map(|c| {
        if c.is_alphanumeric() || c == '_' {
          c
        } else {
          '$'
        }
      })
      .collect::<String>()
      .split('$')
      .filter(|p| !p.is_empty())
      .collect::<Vec<_>>()
      .join("$")
  };
  let mut name = match expr {
    Expr::FunctionCall { name, args } if name == "Subscript" => args
      .iter()
      .map(|a| sanitize(&crate::syntax::expr_to_input_form(a)))
      .collect::<Vec<_>>()
      .join("$"),
    other => sanitize(&crate::syntax::expr_to_input_form(other)),
  };
  if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
    name.insert(0, '$');
  }
  Some(name)
}

/// If the control spec's variable is a compound expression (e.g.
/// `{{Subscript[signal, 1], SquareWave, ""}, …}`), return the spec with the
/// variable replaced by a synthesized plain symbol, together with
/// `(original expr, original InputForm, synthesized name)` so the caller can
/// rewrite the body and patch the default label. Specs with plain symbol
/// variables come back unchanged.
fn rewrite_compound_control_var(
  spec: &Expr,
) -> (Expr, Option<(Expr, String, String)>) {
  let Expr::List(items) = spec else {
    return (spec.clone(), None);
  };
  let Some(head) = items.first() else {
    return (spec.clone(), None);
  };
  // The variable is either the head itself or the first element of a
  // `{var, init, lbl}` head list.
  let var = match head {
    Expr::List(head_items) => match head_items.first() {
      Some(v) => v,
      None => return (spec.clone(), None),
    },
    other => other,
  };
  let Some(synth) = synthesize_var_name(var) else {
    return (spec.clone(), None);
  };
  let orig_form = crate::syntax::expr_to_input_form(var);
  let replacement = Expr::Identifier(synth.clone());
  let new_head = match head {
    Expr::List(head_items) => {
      let mut new_items: Vec<Expr> = head_items.iter().cloned().collect();
      new_items[0] = replacement;
      Expr::List(new_items.into())
    }
    _ => replacement,
  };
  let mut new_spec: Vec<Expr> = items.iter().cloned().collect();
  new_spec[0] = new_head;
  (
    Expr::List(new_spec.into()),
    Some((var.clone(), orig_form, synth)),
  )
}

/// After renaming a compound control variable, a control that fell back to
/// the default label (the bare synthesized name) gets a pretty label
/// rendered from the original expression instead — `Subscript[signal, 1]`
/// shows as `signal₁`, not `signal$1`.
fn patch_default_label(
  control: &mut ManipulateControl,
  original: &Expr,
  synth: &str,
) {
  let pretty_runs = manipulate_label_runs(original, false);
  let pretty = flatten_label_runs(&pretty_runs);
  match control {
    ManipulateControl::Continuous {
      label, label_runs, ..
    }
    | ManipulateControl::Discrete {
      label, label_runs, ..
    }
    | ManipulateControl::Trigger {
      label, label_runs, ..
    } => {
      if label == synth {
        *label = pretty;
        *label_runs = pretty_runs;
      }
    }
    ManipulateControl::Slider2D { label, .. }
    | ManipulateControl::IntervalSlider { label, .. }
    | ManipulateControl::Locator { label, .. } => {
      if label == synth {
        *label = pretty;
      }
    }
    ManipulateControl::Button { .. }
    | ManipulateControl::Heading { .. }
    | ManipulateControl::Divider => {}
  }
}

/// Attempt to extract a `ManipulateSpec` from a held `ListAnimate[{e1, …, en}]`
/// expression. Each element is one animation frame; the widget cycles through
/// them by binding an integer frame index and displaying the selected element.
/// It renders as an auto-playing single-slider Manipulate whose body is
/// `Part[{e1, …, en}, i]`. Returns `None` when the argument is not a non-empty
/// list literal (e.g. `ListAnimate[expr]` or `ListAnimate[{}]`), so the caller
/// falls back to the plain output path.
pub fn extract_list_animate_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "ListAnimate" || args.is_empty() {
    return None;
  }
  let frames = match &args[0] {
    Expr::List(items) if !items.is_empty() => items,
    _ => return None,
  };
  let n = frames.len();
  let list_code = crate::syntax::expr_to_input_form(&args[0]);
  // Frame index `i` runs 1..n in unit steps; the body picks that element.
  // `Round` guards against any float drift the slider might introduce.
  let body_code = format!("Part[{list_code}, Round[i]]");
  let control = ManipulateControl::Continuous {
    name: "i".to_string(),
    min: 1.0,
    max: n as f64,
    step: Some(1.0),
    initial: 1.0,
    label: "i".to_string(),
    label_runs: vec![LabelRun {
      text: "i".to_string(),
      italic: false,
      ..Default::default()
    }],
    is_real: false,
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    control_visible: Vec::new(),
    dynamic_bounds: Vec::new(),
    dynamic_values: Vec::new(),
    animation_var: None,
    animated: true,
    animation_running: true,
    appearance_none: false,
    tracked_symbols: None,
    tracking: Vec::new(),
  })
}

/// Attempt to extract a `ManipulateSpec` from a held `Animator[…]` expression.
/// `Animator` is a standalone auto-playing control: it sweeps a value over a
/// range and (like `Control`) displays the bound variable so its effect is
/// visible. Supported forms: `Animator[{min, max}]`, `Animator[{min, max,
/// step}]`, `Animator[Dynamic[v], {min, max}[, step]]`, `Animator[x]` (the
/// default 0..1 range with initial value `x`), and `Animator[]`. Returns
/// `None` when the range doesn't resolve to numbers.
pub fn extract_animator_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Animator" {
    return None;
  }
  // An optional leading `Dynamic[v]` names the bound variable; else `u`.
  let mut var = "u".to_string();
  let mut idx = 0;
  if let Some(Expr::FunctionCall { name: dn, args: da }) = args.first()
    && dn == "Dynamic"
    && da.len() == 1
    && let Expr::Identifier(s) = &da[0]
  {
    var.clone_from(s);
    idx = 1;
  }
  let (min, max, step, initial, is_real) = match args.get(idx) {
    Some(Expr::List(items)) if !items.is_empty() => {
      let min = crate::functions::math_ast::try_eval_to_f64(&items[0])?;
      let max = items
        .get(1)
        .and_then(crate::functions::math_ast::try_eval_to_f64)
        .unwrap_or(min + 1.0);
      let step = items
        .get(2)
        .and_then(crate::functions::math_ast::try_eval_to_f64);
      let is_real = manipulate_bound_is_inexact(&items[0])
        || items.get(1).is_some_and(manipulate_bound_is_inexact)
        || items.get(2).is_some_and(manipulate_bound_is_inexact);
      (min, max, step, min, is_real)
    }
    // A single number is the initial value over the default 0..1 range.
    Some(other) if idx == 0 => {
      let init = crate::functions::math_ast::try_eval_to_f64(other)?;
      (0.0, 1.0, None, init, manipulate_bound_is_inexact(other))
    }
    // `Animator[]` / `Animator[Dynamic[v]]`: default 0..1 range.
    None => (0.0, 1.0, None, 0.0, false),
    _ => return None,
  };
  let control = ManipulateControl::Continuous {
    name: var.clone(),
    min,
    max,
    step,
    initial,
    label: var.clone(),
    label_runs: vec![LabelRun {
      text: var.clone(),
      italic: false,
      ..Default::default()
    }],
    is_real,
  };
  Some(ManipulateSpec {
    body_code: var,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    control_visible: Vec::new(),
    dynamic_bounds: Vec::new(),
    dynamic_values: Vec::new(),
    animation_var: None,
    animated: true,
    animation_running: true,
    appearance_none: false,
    tracked_symbols: None,
    tracking: Vec::new(),
  })
}

/// Interpret an optional trailing `{{xmin, ymin}, {xmax, ymax}}` range
/// argument, defaulting to the unit square when absent or malformed. Shared by
/// the `LocatorPane`/`ClickPane` pane extractors.
fn pane_range(arg: Option<&Expr>) -> ((f64, f64), (f64, f64)) {
  match arg {
    Some(Expr::List(corners)) if corners.len() == 2 => {
      match (list2_f64(&corners[0]), list2_f64(&corners[1])) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => ((0.0, 0.0), (1.0, 1.0)),
      }
    }
    _ => ((0.0, 0.0), (1.0, 1.0)),
  }
}

/// Attempt to extract a `ManipulateSpec` from a held `LocatorPane[…]`.
/// A locator pane shows a graphic with a draggable point that drives it.
/// Supported forms: `LocatorPane[Dynamic[p], body]` and `LocatorPane[p0,
/// body]`, each with an optional trailing coordinate range `{{xmin, ymin},
/// {xmax, ymax}}` (default: the unit square). Renders as a 2D pad — the
/// draggable locator — beside the live `body` graphic. Returns `None` if the
/// arguments don't fit.
pub fn extract_locator_pane_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "LocatorPane" || args.len() < 2 {
    return None;
  }
  // arg0: `Dynamic[p]` (named variable) or a literal initial point `{x, y}`.
  let (var, explicit_init) = match &args[0] {
    Expr::FunctionCall { name: dn, args: da }
      if dn == "Dynamic" && da.len() == 1 =>
    {
      match &da[0] {
        Expr::Identifier(s) => (s.clone(), None),
        _ => return None,
      }
    }
    pt => ("p".to_string(), Some(list2_f64(pt)?)),
  };
  let body_code = crate::syntax::expr_to_input_form(&args[1]);
  let ((x_min, y_min), (x_max, y_max)) = pane_range(args.get(2));
  // Start the locator at the given point, else the range centre.
  let (x_initial, y_initial) = explicit_init
    .unwrap_or((f64::midpoint(x_min, x_max), f64::midpoint(y_min, y_max)));
  let control = ManipulateControl::Slider2D {
    name: var.clone(),
    x_min,
    x_max,
    y_min,
    y_max,
    x_initial,
    y_initial,
    label: var,
    write_callback: None,
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    control_visible: Vec::new(),
    dynamic_bounds: Vec::new(),
    dynamic_values: Vec::new(),
    animation_var: None,
    animated: false,
    animation_running: true,
    appearance_none: false,
    tracked_symbols: None,
    tracking: Vec::new(),
  })
}

/// Attempt to extract a `ManipulateSpec` from a held `ClickPane[…]`.
/// A click pane applies a handler to the coordinates of each click. We model
/// it as a 2D pad whose position feeds the handler: `ClickPane[expr, func]`
/// (and `ClickPane[expr, {{xmin, ymin}, {xmax, ymax}}, func]`) render as a
/// clickable/draggable pad with the live `func[{x, y}]` result shown beside
/// it. Returns `None` if there's no handler to apply.
pub fn extract_click_pane_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "ClickPane" || args.len() < 2 {
    return None;
  }
  // The handler is the last argument; a 3-argument form carries an explicit
  // coordinate range in the middle.
  let func = args.last()?;
  let range_arg = if args.len() >= 3 { args.get(1) } else { None };
  let ((x_min, y_min), (x_max, y_max)) = pane_range(range_arg);
  // Bind the click position `pos` and show the handler applied to it; the body
  // re-evaluates `func[pos]` on every pad move.
  let func_code = crate::syntax::expr_to_input_form(func);
  let body_code = format!("({func_code})[pos]");
  let control = ManipulateControl::Slider2D {
    name: "pos".to_string(),
    x_min,
    x_max,
    y_min,
    y_max,
    x_initial: f64::midpoint(x_min, x_max),
    y_initial: f64::midpoint(y_min, y_max),
    label: "pos".to_string(),
    write_callback: None,
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    control_visible: Vec::new(),
    dynamic_bounds: Vec::new(),
    dynamic_values: Vec::new(),
    animation_var: None,
    animated: false,
    animation_running: true,
    appearance_none: false,
    tracked_symbols: None,
    tracking: Vec::new(),
  })
}

/// Attempt to extract a `ManipulateSpec` from a held standalone
/// `Control[{…}]` expression. A bare `Control` renders a single interactive
/// control whose bound variable has no body to display, so the "body" is
/// synthesized as the variable itself — dragging the control then shows the
/// current bound value (a number, a discrete choice, a 2-vector, …).
///
/// Returns `None` if the expression is not a well-formed `Control` (e.g. the
/// argument is not a variable-spec list, or resolves to a hidden control).
pub fn extract_control_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Control" || args.is_empty() {
    return None;
  }
  // The first argument is the variable specification; any trailing options
  // are ignored for rendering purposes.
  if !matches!(&args[0], Expr::List(items) if !items.is_empty()) {
    return None;
  }
  let (control, enabled, animate) =
    match parse_manipulate_control(&args[0], &[])? {
      ParsedControl::Visible {
        control,
        enabled,
        animate,
        ..
      } => (control, enabled, animate),
      // A hidden control (`ControlType -> None` / Locator) has no widget and
      // nothing to display on its own — fall back to the plain output path.
      ParsedControl::Fixed { .. }
      | ParsedControl::State { .. }
      | ParsedControl::StateWithDisplay { .. }
      | ParsedControl::StateWithControl { .. } => return None,
    };
  // Display the bound variable so the control's effect is visible.
  let body_code = control.name().to_string();
  let control_enabled = match enabled {
    Some(cond) => vec![(control.name().to_string(), cond)],
    None => Vec::new(),
  };
  // A standalone `Control[{…, ControlType -> Trigger/Animator}]` animates
  // its own variable.
  let animation_var = animate.map(|_| control.name().to_string());
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled,
    control_visible: Vec::new(),
    dynamic_bounds: Vec::new(),
    dynamic_values: Vec::new(),
    animation_var,
    animated: animate.is_some(),
    animation_running: animate.unwrap_or(true),
    appearance_none: false,
    tracked_symbols: None,
    tracking: Vec::new(),
  })
}

/// Evaluate `expr` and render the result as InputForm. Falls back to the
/// unevaluated form if evaluation fails. Used to freeze a hidden control's
/// initial value (e.g. `RandomInteger[…]`) to a concrete literal so it does
/// not change on every re-evaluation.
fn manipulate_value_to_input_form(expr: &Expr) -> String {
  match crate::evaluator::evaluate_expr_to_expr(expr) {
    Ok(evaluated) => crate::syntax::expr_to_input_form(&evaluated),
    Err(_) => crate::syntax::expr_to_input_form(expr),
  }
}

/// Interpret an expression as a 2-element numeric list `{a, b}`, evaluating
/// each element to an `f64`. Returns `None` for anything that isn't a
/// 2-vector of numbers.
///
/// A corner point may name a symbol a leading body assignment sets rather
/// than carry a literal number — `{{u, {1, 1}, ""}, {xmin, ymin}, {xmax,
/// ymax}, ControlType -> Slider2D}` bounds a 2D control by the same `xmin`
/// leading assignments resolve for a 1D slider (`eval_manipulate_bound`
/// falls back to a full evaluation for exactly this reason). Each element
/// gets the same fallback here so a `Slider2D` corner point resolves a
/// symbolic bound the way a plain slider's `min`/`max` already does.
fn list2_f64(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::List(l) if l.len() == 2 => {
      let a = eval_manipulate_bound(&l[0])?.0;
      let b = eval_manipulate_bound(&l[1])?.0;
      Some((a, b))
    }
    _ => None,
  }
}

/// Interpret an expression as a non-empty list of 2D numeric points
/// `{{x1, y1}, {x2, y2}, …}`. Returns `None` when any element isn't a
/// numeric 2-vector.
fn point_list_f64(e: &Expr) -> Option<Vec<(f64, f64)>> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      items.iter().map(list2_f64).collect()
    }
    _ => None,
  }
}

/// Coordinate range for a Locator control: the explicit
/// `{xmin, ymin}, {xmax, ymax}` corner bounds when the spec gives both,
/// otherwise the bounding box of the initial points padded by half its
/// span per axis (at least 1), so every point stays draggable.
fn locator_range(
  corner_bounds: &[(f64, f64)],
  points: &[(f64, f64)],
) -> (f64, f64, f64, f64) {
  if corner_bounds.len() >= 2 {
    let (x0, y0) = corner_bounds[0];
    let (x1, y1) = corner_bounds[1];
    return (x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1));
  }
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  for (x, y) in points {
    x_min = x_min.min(*x);
    x_max = x_max.max(*x);
    y_min = y_min.min(*y);
    y_max = y_max.max(*y);
  }
  let x_pad = ((x_max - x_min) / 2.0).max(1.0);
  let y_pad = ((y_max - y_min) / 2.0).max(1.0);
  (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
}

/// Render a control-label expression into styled runs for the interactive
/// widget. Wolfram labels are frequently wrapped in presentation heads
/// (`Style[…, Italic]`, `Text[…]`) or use `Subscript`, none of which should
/// appear as literal source next to a slider.
///
/// The heavy lifting is delegated to the OutputForm renderer
/// (`format_expr(_, Output)`), which already unwraps `Style`, concatenates
/// `Row`, renders `Rational`, etc. The arms handled explicitly here are the
/// label-specific bits OutputForm intentionally does *not* do: it keeps
/// `Subscript`/`Superscript` in 1D structural form (`Subscript[m, 1]`, to
/// match wolframscript's 1D text output) and leaves `Text[…]` wrapped. So we
/// recurse through those heads to reach a nested `Subscript`/`Superscript`
/// (folding it into Unicode) and to carry `Style[…, Italic]` down to the
/// individual runs, giving e.g. `Text[Subscript[Style["m", Italic], 1]]` →
/// an italic `m` followed by an upright `₁`, and `Style["t", Italic]` → an
/// italic `t`. `italic` is the style inherited from an enclosing `Style`.
/// The items a `Row`/`Column` lays out. Written literally they are already a
/// list; a Demonstration more often computes them — `Row[Flatten[{" ",
/// Riffle[Subscript[Style["N", Italic], #] & /@ {"D", "U"}, " and "], " "}]]`
/// builds a setter's caption that way — so a non-literal argument is
/// evaluated first. Anything that does not come back as a list (a symbol
/// with no value, say) yields `None`, leaving the label on the OutputForm
/// path rather than printing the source of the computation.
fn layout_parts(arg: Option<&Expr>) -> Option<Vec<Expr>> {
  match arg? {
    Expr::List(parts) => Some(parts.to_vec()),
    other => match evaluate_expr_to_expr(other) {
      Ok(ref evaluated) => match evaluated {
        Expr::List(parts) => Some(parts.to_vec()),
        _ => None,
      },
      Err(_) => None,
    },
  }
}

/// Renders a `Grid`/`TableForm`/`TextGrid` row list as label runs: each
/// row's non-blank cells joined by a space, rows joined by a newline.
fn grid_label_runs(rows: &[Expr], italic: bool) -> Vec<LabelRun> {
  let mut out = Vec::new();
  for row in rows {
    let cells = layout_parts(Some(row)).unwrap_or_else(|| vec![row.clone()]);
    let mut row_runs: Vec<LabelRun> = Vec::new();
    for cell in &cells {
      let cell_runs = manipulate_label_runs(cell, italic);
      if flatten_label_runs(&cell_runs).trim().is_empty() {
        continue;
      }
      if !row_runs.is_empty() {
        row_runs.push(LabelRun {
          text: " ".to_string(),
          italic,
          ..Default::default()
        });
      }
      row_runs.extend(cell_runs);
    }
    if row_runs.is_empty() {
      continue;
    }
    if !out.is_empty() {
      out.push(LabelRun {
        text: "\n".to_string(),
        italic,
        ..Default::default()
      });
    }
    out.extend(row_runs);
  }
  out
}

fn manipulate_label_runs(expr: &Expr, italic: bool) -> Vec<LabelRun> {
  let mut runs = manipulate_label_runs_inner(expr, italic);
  // A label is text a widget draws, so the private-use code points Wolfram
  // stores its characters as (`\[WarningSign]` is U+F725) give way to the
  // glyphs a normal font has. Idempotent, so the recursive calls inside
  // `manipulate_label_runs_inner` may pass through here too.
  for run in &mut runs {
    if let std::borrow::Cow::Owned(text) =
      crate::syntax::substitute_private_use_glyphs(&run.text)
    {
      run.text = text;
    }
  }
  runs
}

fn manipulate_label_runs_inner(expr: &Expr, italic: bool) -> Vec<LabelRun> {
  let output_run = |italic: bool| {
    let text =
      crate::syntax::format_expr(expr, crate::syntax::ExprForm::Output);
    if text.is_empty() {
      vec![]
    } else {
      vec![LabelRun {
        text,
        italic,
        ..Default::default()
      }]
    }
  };
  // `Derivative[n][f]` — a slider labelled `y'(0)` writes its `y'` this way.
  // The primes are upright even when the function they mark is italic.
  if let Some((func, order)) = as_derivative_of(expr) {
    let mut runs = manipulate_label_runs(func, italic);
    runs.push(LabelRun {
      text: derivative_prime_marks(order),
      italic: false,
      ..Default::default()
    });
    return runs;
  }
  // A square root — the notebook's `\[Sqrt]…` radical, which survives as the
  // `Sqrt` head inside a held expression and normalizes to `x^(1/2)`
  // elsewhere. Either spelling sets as the radical sign over its radicand, so
  // the widget reads `√(eᶻ+1)` rather than the source of a head or of a
  // fractional exponent. Checked before the structural match so both
  // spellings reach here.
  if let Some(radicand) = radical_radicand(expr) {
    let mut runs = vec![LabelRun {
      text: "\u{221A}".to_string(),
      italic: false,
      ..Default::default()
    }];
    runs.extend(grouped_label_runs(radicand, italic));
    return runs;
  }
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Style[expr, dir…] — render `expr`, turning italic on if any directive
      // asks for it (bare `Italic` or `FontSlant -> "Italic"`), and applying
      // any bold/color directive to the runs it produces. A nested `Style`
      // has already set its own color by then and keeps it.
      "Style" | "StyleForm" => {
        let styled = italic || args.iter().skip(1).any(is_italic_directive);
        let bold = args.iter().skip(1).any(is_bold_directive);
        let color = args
          .iter()
          .skip(1)
          .find_map(parse_color)
          .map(|c| (c.r as f32, c.g as f32, c.b as f32));
        let mut runs = args
          .first()
          .map(|a| manipulate_label_runs(a, styled))
          .unwrap_or_default();
        for run in &mut runs {
          run.bold |= bold;
          if run.color.is_none() {
            run.color = color;
          }
        }
        runs
      }
      // Presentation wrappers whose content may nest styling/subscripts —
      // recurse rather than defer to OutputForm. `Tooltip[label, tip]`
      // displays its label (the tip only appears on hover in the Wolfram
      // FrontEnd), so a control spec like
      // `{{v, True, Tooltip["source", "Show source"]}, {True, False}}`
      // labels the control "source". `Framed[content, …]` keeps only its
      // content — the frame itself has no text to typeset.
      // `Dynamic[content]` inside a label (a Demonstration's step counter
      // buttons often frame one) only exists to keep `content` live; a
      // static label typesets `content` itself rather than the `Dynamic[…]`
      // wrapper's own source.
      "Text" | "DisplayForm" | "TraditionalForm" | "Tooltip" | "Framed"
      | "Dynamic" => args
        .first()
        .map(|a| manipulate_label_runs(a, italic))
        .unwrap_or_default(),
      // `Labeled[content, label, …]` draws its label alongside the content
      // (by default below it); a Demonstration's setter option can pair a
      // picture with a caption this way. Placement specs (a 3rd argument, or
      // a list of labels) are ignored — only the first label is shown.
      "Labeled" => {
        let mut runs = args
          .first()
          .map(|a| manipulate_label_runs(a, italic))
          .unwrap_or_default();
        if let Some(label_arg) = args.get(1) {
          let label_runs = manipulate_label_runs(label_arg, italic);
          if !flatten_label_runs(&label_runs).trim().is_empty() {
            if !runs.is_empty() {
              runs.push(LabelRun {
                text: "\n".to_string(),
                italic,
                ..Default::default()
              });
            }
            runs.extend(label_runs);
          }
        }
        runs
      }
      // Row[{a, b, …}] — concatenate the (recursively rendered) parts.
      "Row" => match layout_parts(args.first()) {
        Some(parts) => parts
          .iter()
          .flat_map(|p| manipulate_label_runs(p, italic))
          .collect(),
        _ => output_run(italic),
      },
      // Column[{a, b, …}] — the same, but one part per line. Demonstrations
      // label a hypothesis-test setter with a two-line column.
      "Column" => match layout_parts(args.first()) {
        Some(parts) => parts
          .iter()
          .enumerate()
          .flat_map(|(i, p)| {
            let mut runs = Vec::new();
            if i > 0 {
              runs.push(LabelRun {
                text: "\n".to_string(),
                italic,
                ..Default::default()
              });
            }
            runs.extend(manipulate_label_runs(p, italic));
            runs
          })
          .collect(),
        _ => output_run(italic),
      },
      // Grid[{{a, b, …}, …}, …] / TableForm / TextGrid — a Demonstration
      // uses a small grid as a setter's per-choice appearance function (e.g.
      // marking which edge of a patch a rule reflects). Cells render as
      // their own label runs, joined by a space within a row and by a
      // newline between rows; blank cells (Grid pads a ragged table with
      // `""`) drop out rather than leaving stray spaces.
      "Grid" | "TableForm" | "TextGrid" => match layout_parts(args.first()) {
        Some(rows) => grid_label_runs(&rows, italic),
        _ => output_run(italic),
      },
      "Subscript" => script_runs(args, italic, false),
      "Superscript" => script_runs(args, italic, true),
      // `Underscript[base, mark]` — the limit written under a base, e.g. a
      // sum's index. No diacritic reads as an under-mark in practice, so
      // the mark simply sits in subscript position like `Subscript` does.
      "Underscript" if args.len() == 2 => script_runs(args, italic, false),
      // `Overscript[base, mark]`, the evaluable form `OverscriptBox` reads
      // as. A diacritic mark (a hat, bar, tilde, dot, …) draws directly on
      // the base's last glyph — `Overscript[u, "_"]` is the antiquark ū in
      // a Demonstration's quark-content picker; anything else (a rate
      // constant over a reaction arrow) hangs above the base instead.
      "Overscript" if args.len() == 2 => overscript_runs(args, italic),
      // `Power[b, e]`, which is what a label's typeset `SuperscriptBox`
      // reads as: a Demonstration writes its gravity slider's unit as
      // `m/\!\(\*SuperscriptBox[\(s\), \(2\)]\)`, and it must show as
      // `m/s²`, not `m/s^2`.
      "Power" if args.len() == 2 => script_runs(args, italic, true),
      // The evaluated shapes of the same arithmetic the `BinaryOp` arms
      // below handle: a quotient of typeset pieces reaches here as
      // `Times[a, b^-1]`. Only worth recursing into when there is something
      // to typeset — see `contains_presentation_head`.
      "Times" if contains_presentation_head(expr) => {
        product_label_runs(args, italic)
      }
      "Plus" if contains_presentation_head(expr) => {
        joined_label_runs(args, " + ", false, italic)
      }
      _ => output_run(italic),
    },
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => script_runs(
      &[left.as_ref().clone(), right.as_ref().clone()],
      italic,
      true,
    ),
    // Arithmetic joining typeset label pieces — a Demonstrations setter
    // writes a quotient of two typeset rows as `Row[…]/Row[…]`. Recurse into
    // the operands so their `Row`/`Style`/`Superscript` structure keeps
    // rendering as styled runs; OutputForm would print the operands' source
    // next to the operator. Operands with nothing to typeset stay on the
    // OutputForm path, which already knows this arithmetic's precedence and
    // sign conventions.
    Expr::BinaryOp { op, left, right }
      if arithmetic_label_operator(*op).is_some()
        && contains_presentation_head(expr) =>
    {
      let (text, group) = arithmetic_label_operator(*op).unwrap_or_default();
      let side = |e: &Expr| {
        if group {
          grouped_label_runs(e, italic)
        } else {
          manipulate_label_runs(e, italic)
        }
      };
      let mut runs = side(left);
      runs.push(LabelRun {
        text: text.to_string(),
        italic: false,
        ..Default::default()
      });
      runs.extend(side(right));
      runs
    }
    // A label written in the notebook FrontEnd carries its typeset bits as
    // inline linear syntax inside the string: `"value to test against
    // \!\(\*SubscriptBox[\(p\), \(0\)]\)"`. Render each box segment as the
    // expression it typesets so the widget shows `value to test against p₀`
    // rather than the private-use box markers.
    Expr::String(s) => {
      inline_box_label_runs(s, italic).unwrap_or_else(|| output_run(italic))
    }
    _ => output_run(italic),
  }
}

/// The radicand of a square root written either way: as the `Sqrt[x]` head
/// (which is what survives inside a held expression, e.g. the body of a
/// `Manipulate`) or as the `x^(1/2)` power it normalizes to elsewhere.
fn radical_radicand(expr: &Expr) -> Option<&Expr> {
  /// Is this exponent the rational 1/2 — `Rational[1, 2]` or the
  /// `Divide`/`Times` shapes an unevaluated `1/2` can take?
  fn is_one_half(exponent: &Expr) -> bool {
    match exponent {
      Expr::FunctionCall { name, args } if name == "Rational" => {
        matches!(
          (args.first(), args.get(1)),
          (Some(Expr::Integer(1)), Some(Expr::Integer(2)))
        )
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::Divide,
        left,
        right,
      } => {
        matches!(left.as_ref(), Expr::Integer(1))
          && matches!(right.as_ref(), Expr::Integer(2))
      }
      _ => false,
    }
  }
  match expr {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      args.first()
    }
    Expr::FunctionCall { name, args }
      if name == "Power" && args.len() == 2 && is_one_half(&args[1]) =>
    {
      args.first()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if is_one_half(right) => Some(left),
    _ => None,
  }
}

/// The base of a reciprocal factor — `x^-1`, which is how a quotient's
/// denominator reaches a `Times`.
fn reciprocal_base(expr: &Expr) -> Option<&Expr> {
  // A denominator's exponent reaches here as the literal `-1` or, in an
  // unevaluated expression, as a negated `1`.
  let is_minus_one = |e: &Expr| {
    matches!(e, Expr::Integer(-1))
      || matches!(e, Expr::UnaryOp { op: crate::syntax::UnaryOperator::Minus, operand }
        if matches!(operand.as_ref(), Expr::Integer(1)))
  };
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Power" && args.len() == 2 && is_minus_one(&args[1]) =>
    {
      args.first()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } if is_minus_one(right) => Some(left),
    _ => None,
  }
}

/// Runs for a `Times[…]` of typeset pieces. Reciprocal factors are the
/// quotient's denominator, so `Row[…] Row[…]^-1` — what `Row[…]/Row[…]`
/// evaluates to — sets as a quotient rather than as a product with a
/// negative exponent.
fn product_label_runs(factors: &[Expr], italic: bool) -> Vec<LabelRun> {
  let (denominators, numerators): (Vec<&Expr>, Vec<&Expr>) =
    factors.iter().partition(|f| reciprocal_base(f).is_some());
  let numerators: Vec<Expr> = numerators.into_iter().cloned().collect();
  let mut runs = joined_label_runs(&numerators, "*", true, italic);
  if denominators.is_empty() {
    return runs;
  }
  let denominators: Vec<Expr> = denominators
    .into_iter()
    .filter_map(|f| reciprocal_base(f).cloned())
    .collect();
  runs.push(LabelRun {
    text: "/".to_string(),
    italic: false,
    ..Default::default()
  });
  runs.extend(joined_label_runs(&denominators, "*", true, italic));
  runs
}

/// Runs for a list of label pieces joined by an operator, each piece
/// parenthesized when `group` is set and the piece is compound.
fn joined_label_runs(
  parts: &[Expr],
  separator: &str,
  group: bool,
  italic: bool,
) -> Vec<LabelRun> {
  let mut runs = Vec::new();
  for (i, part) in parts.iter().enumerate() {
    if i > 0 {
      runs.push(LabelRun {
        text: separator.to_string(),
        italic: false,
        ..Default::default()
      });
    }
    if group {
      runs.extend(grouped_label_runs(part, italic));
    } else {
      runs.extend(manipulate_label_runs(part, italic));
    }
  }
  runs
}

/// Whether a label expression contains something only the structural
/// renderer sets properly — a layout, a style, or a script. Arithmetic over
/// pieces that have none of these gains nothing from recursing, so it stays
/// on the OutputForm path that already handles precedence and signs.
fn contains_presentation_head(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, args } => {
      matches!(
        name.as_str(),
        "Row"
          | "Column"
          | "Grid"
          | "Style"
          | "StyleForm"
          | "Text"
          | "Tooltip"
          | "DisplayForm"
          | "TraditionalForm"
          | "Subscript"
          | "Superscript"
          | "Subsuperscript"
      ) || args.iter().any(contains_presentation_head)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_presentation_head(left) || contains_presentation_head(right)
    }
    Expr::List(items) => items.iter().any(contains_presentation_head),
    _ => false,
  }
}

/// How an arithmetic operator joining two label pieces is written, and
/// whether its operands bind tightly enough to need parentheses when they
/// are compound. The spellings match the OutputForm renderer this falls
/// back to, so a label reads the same whichever path builds it.
fn arithmetic_label_operator(
  op: crate::syntax::BinaryOperator,
) -> Option<(&'static str, bool)> {
  use crate::syntax::BinaryOperator as Op;
  Some(match op {
    Op::Plus => (" + ", false),
    Op::Minus => (" - ", false),
    Op::Times => ("*", true),
    Op::Divide => ("/", true),
    _ => return None,
  })
}

/// Label runs for a piece sitting under a radical or on one side of a
/// product or quotient, parenthesized when it is a compound piece so the
/// grouping stays unambiguous: `√(eᶻ+1)`, not `√eᶻ+1`.
fn grouped_label_runs(expr: &Expr, italic: bool) -> Vec<LabelRun> {
  let runs = manipulate_label_runs(expr, italic);
  if !label_is_compound(expr) {
    return runs;
  }
  let mut grouped = vec![LabelRun {
    text: "(".to_string(),
    italic: false,
    ..Default::default()
  }];
  grouped.extend(runs);
  grouped.push(LabelRun {
    text: ")".to_string(),
    italic: false,
    ..Default::default()
  });
  grouped
}

/// Whether a label piece sets more than one term, so that using it as a
/// radicand or a factor needs parentheses around it.
fn label_is_compound(expr: &Expr) -> bool {
  // A radical brackets its own radicand, however it is spelled.
  if radical_radicand(expr).is_some() {
    return false;
  }
  match expr {
    Expr::BinaryOp { op, .. } => arithmetic_label_operator(*op).is_some(),
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Presentation wrappers group exactly what they wrap.
      "Style" | "StyleForm" | "Text" | "DisplayForm" | "TraditionalForm"
      | "Tooltip" => args.first().is_some_and(label_is_compound),
      // A layout of several items reads as several terms.
      "Row" | "Column" => {
        matches!(args.first(), Some(Expr::List(parts)) if parts.len() > 1)
      }
      "Plus" | "Subtract" | "Times" | "Divide" => args.len() > 1,
      _ => false,
    },
    _ => false,
  }
}

/// Label runs for a string with inline `\!\(\*…\)` box segments: the prose
/// stays literal and each box segment is converted to the expression it
/// typesets and rendered by the same label machinery (so `SubscriptBox[p, 0]`
/// folds into the Unicode `p₀`). Returns None when the string has no box
/// segment, leaving plain strings on the normal path.
fn inline_box_label_runs(s: &str, italic: bool) -> Option<Vec<LabelRun>> {
  let segments = crate::functions::string_ast::split_inline_boxes(s);
  if !segments.iter().any(|seg| seg.is_box) {
    return None;
  }
  let mut runs = Vec::new();
  for seg in segments {
    if !seg.is_box {
      runs.push(LabelRun {
        text: seg.text,
        italic,
        ..Default::default()
      });
      continue;
    }
    // Parsed, not evaluated: a box like `OverscriptBox[x, "_"]` is a frozen
    // typesetting form, and `x` here names the base glyph to draw, not a
    // reference to whatever a same-named Manipulate control is currently
    // bound to. Evaluating it would substitute that control's live value
    // (e.g. `x -> 1`) into every choice label that happens to mention `x`.
    let rendered = crate::notebook::box_source_to_expression(&seg.text)
      .and_then(|code| crate::parse_to_expr(&code).ok())
      .map(|expr| manipulate_label_runs(&expr, italic));
    match rendered {
      // An unrecognised box head keeps its source text rather than
      // vanishing from the label.
      None => runs.push(LabelRun {
        text: seg.text,
        italic,
        ..Default::default()
      }),
      Some(box_runs) => runs.extend(box_runs),
    }
  }
  Some(runs)
}

/// Build the runs for a `Subscript`/`Superscript`: the base rendered in the
/// inherited style, followed by each remaining argument folded into Unicode
/// sub-/superscript glyphs (which have no italic variant, so they stay
/// upright).
fn script_runs(
  args: &[Expr],
  italic: bool,
  superscript: bool,
) -> Vec<LabelRun> {
  let mut runs = args
    .first()
    .map(|a| manipulate_label_runs(a, italic))
    .unwrap_or_default();
  let script: String = args
    .iter()
    .skip(1)
    .map(|a| {
      to_unicode_script(
        &flatten_label_runs(&manipulate_label_runs(a, false)),
        superscript,
      )
    })
    .collect();
  if !script.is_empty() {
    runs.push(LabelRun {
      text: script,
      italic: false,
      ..Default::default()
    });
  }
  runs
}

/// Build the runs for an `Overscript[base, mark]`. A diacritic mark
/// (recognized by [`crate::notebook::combining_accent`]) is appended as a
/// Unicode combining character onto the base's last run, so `Overscript[u,
/// "_"]` reads as `ū` rather than `u_`. Anything else is a script the base
/// carries above it — a rate constant over a reaction arrow — which has no
/// combining form, so it is written out in parentheses instead.
fn overscript_runs(args: &[Expr], italic: bool) -> Vec<LabelRun> {
  let mut runs = args
    .first()
    .map(|a| manipulate_label_runs(a, italic))
    .unwrap_or_default();
  let Some(mark_arg) = args.get(1) else {
    return runs;
  };
  let mark = flatten_label_runs(&manipulate_label_runs(mark_arg, false));
  match crate::notebook::combining_accent(&mark) {
    Some(combining) => match runs.last_mut() {
      Some(last) => last.text.push_str(combining),
      None => runs.push(LabelRun {
        text: combining.to_string(),
        italic,
        ..Default::default()
      }),
    },
    None if !mark.is_empty() => runs.push(LabelRun {
      text: format!("^({mark})"),
      italic: false,
      ..Default::default()
    }),
    None => {}
  }
  runs
}

/// True when a `Style` directive requests italic: bare `Italic` or
/// `FontSlant -> "Italic" | Italic`.
/// True when a `Style` directive requests bold: bare `Bold` or
/// `FontWeight -> "Bold"`.
fn is_bold_directive(dir: &Expr) -> bool {
  match dir {
    Expr::Identifier(s) => s == "Bold",
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      matches!(pattern.as_ref(), Expr::Identifier(s) if s == "FontWeight")
        && match replacement.as_ref() {
          Expr::String(s) => s == "Bold",
          Expr::Identifier(s) => s == "Bold",
          _ => false,
        }
    }
    _ => false,
  }
}

fn is_italic_directive(dir: &Expr) -> bool {
  match dir {
    Expr::Identifier(s) => s == "Italic",
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      matches!(pattern.as_ref(), Expr::Identifier(s) if s == "FontSlant")
        && match replacement.as_ref() {
          Expr::String(s) => s == "Italic",
          Expr::Identifier(s) => s == "Italic",
          _ => false,
        }
    }
    _ => false,
  }
}

/// Concatenate the text of a run sequence, discarding styling. Used where a
/// plain string is needed (JSON export, Unicode-script folding).
fn flatten_label_runs(runs: &[LabelRun]) -> String {
  runs.iter().map(|r| r.text.as_str()).collect()
}

/// The mark the Wolfram Language typesets `Derivative[n][f]` with: a prime
/// per order up to three (`y′`, `y″`, `y‴`) and a parenthesised superscript
/// order beyond that (`y⁽⁴⁾`).
pub(crate) fn derivative_prime_marks(order: u32) -> String {
  match order {
    0 => String::new(),
    1 => "\u{2032}".to_string(),
    2 => "\u{2033}".to_string(),
    3 => "\u{2034}".to_string(),
    n => to_unicode_script(&format!("({n})"), true),
  }
}

/// Split a derivative application into the function it differentiates and
/// the order. Covers both shapes the parser leaves behind: the curried
/// `Derivative[n][f]` and the flattened `Derivative[n, f]` that
/// `expr_to_output` prints back as `Derivative[n][f]`.
pub(crate) fn as_derivative_of(expr: &Expr) -> Option<(&Expr, u32)> {
  let (head, args): (&Expr, &[Expr]) = match expr {
    Expr::CurriedCall { func, args } => (func.as_ref(), args),
    Expr::FunctionCall { name, args } if name == "Derivative" => {
      return match args.len() {
        2 => order_of(&args[0]).map(|n| (&args[1], n)),
        _ => None,
      };
    }
    _ => return None,
  };
  if args.len() != 1 {
    return None;
  }
  match head {
    Expr::FunctionCall { name, args: hargs }
      if name == "Derivative" && hargs.len() == 1 =>
    {
      order_of(&hargs[0]).map(|n| (&args[0], n))
    }
    _ => None,
  }
}

/// The integer order of a `Derivative[n]` index.
fn order_of(expr: &Expr) -> Option<u32> {
  match expr {
    Expr::Integer(n) if *n >= 0 => u32::try_from(*n).ok(),
    _ => None,
  }
}

/// Map the characters of `s` to their Unicode sub-/superscript form when a
/// mapping exists, leaving other characters unchanged. Used to render
/// `Subscript`/`Superscript` control labels inline.
pub(crate) fn to_unicode_script(s: &str, superscript: bool) -> String {
  s.chars()
    .map(|c| unicode_script_char(c, superscript).unwrap_or(c))
    .collect()
}

/// Like [`to_unicode_script`], but only for the digits and signs
/// (U+2080–U+208E and U+2070–U+207E). Letters keep their full size: the
/// Latin sub-/superscript letters live in blocks most text fonts omit, so
/// mapping them draws a row of missing-glyph boxes where the plain letters
/// would have read fine. Used where the text is handed to a font renderer
/// rather than typeset, e.g. a plot label.
pub(crate) fn to_unicode_script_digits(s: &str, superscript: bool) -> String {
  s.chars()
    .map(|c| {
      if c.is_alphabetic() {
        c
      } else {
        unicode_script_char(c, superscript).unwrap_or(c)
      }
    })
    .collect()
}

/// Unicode sub-/superscript for a single character, if one exists.
fn unicode_script_char(c: char, superscript: bool) -> Option<char> {
  let mapped = if superscript {
    match c {
      '0' => '\u{2070}',
      '1' => '\u{00B9}',
      '2' => '\u{00B2}',
      '3' => '\u{00B3}',
      '4' => '\u{2074}',
      '5' => '\u{2075}',
      '6' => '\u{2076}',
      '7' => '\u{2077}',
      '8' => '\u{2078}',
      '9' => '\u{2079}',
      '+' => '\u{207A}',
      '-' => '\u{207B}',
      '(' => '\u{207D}',
      ')' => '\u{207E}',
      '=' => '\u{207C}',
      // Unicode's modifier-letter superscripts. `q` is the one Latin
      // letter with no superscript form.
      'a' => '\u{1D43}',
      'b' => '\u{1D47}',
      'c' => '\u{1D9C}',
      'd' => '\u{1D48}',
      'e' => '\u{1D49}',
      'f' => '\u{1DA0}',
      'g' => '\u{1D4D}',
      'h' => '\u{02B0}',
      'i' => '\u{2071}',
      'j' => '\u{02B2}',
      'k' => '\u{1D4F}',
      'l' => '\u{02E1}',
      'm' => '\u{1D50}',
      'n' => '\u{207F}',
      'o' => '\u{1D52}',
      'p' => '\u{1D56}',
      'r' => '\u{02B3}',
      's' => '\u{02E2}',
      't' => '\u{1D57}',
      'u' => '\u{1D58}',
      'v' => '\u{1D5B}',
      'w' => '\u{02B7}',
      'x' => '\u{02E3}',
      'y' => '\u{02B8}',
      'z' => '\u{1DBB}',
      _ => return None,
    }
  } else {
    match c {
      '0' => '\u{2080}',
      '1' => '\u{2081}',
      '2' => '\u{2082}',
      '3' => '\u{2083}',
      '4' => '\u{2084}',
      '5' => '\u{2085}',
      '6' => '\u{2086}',
      '7' => '\u{2087}',
      '8' => '\u{2088}',
      '9' => '\u{2089}',
      '+' => '\u{208A}',
      '-' => '\u{208B}',
      '(' => '\u{208D}',
      ')' => '\u{208E}',
      '=' => '\u{208C}',
      // Unicode only defines subscripts for these Latin letters; the rest
      // (b, c, d, …) fall back to the plain character.
      'a' => '\u{2090}',
      'e' => '\u{2091}',
      'h' => '\u{2095}',
      'i' => '\u{1D62}',
      'j' => '\u{2C7C}',
      'k' => '\u{2096}',
      'l' => '\u{2097}',
      'm' => '\u{2098}',
      'n' => '\u{2099}',
      'o' => '\u{2092}',
      'p' => '\u{209A}',
      'r' => '\u{1D63}',
      's' => '\u{209B}',
      't' => '\u{209C}',
      'u' => '\u{1D64}',
      'v' => '\u{1D65}',
      'x' => '\u{2093}',
      _ => return None,
    }
  };
  Some(mapped)
}

/// A discrete control's choice list may be written `Dynamic[expr, opts…]`
/// (e.g. `Dynamic[# -> data[[#]] & /@ Range[Length[data]], SynchronousUpdating -> False]`)
/// so the front end refreshes it as other state changes. The trailing
/// options only govern *when* Wolfram redraws the popup, not what the list
/// contains, so unwrapping to `expr` and evaluating it once already matches
/// what Wolfram renders. `Dynamic` is `HoldFirst` and has no evaluation rule
/// of its own, so without this the wrapped expression is never reached and
/// the control fails to parse.
fn unwrap_dynamic_choices(expr: &Expr) -> &Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && !args.is_empty() =>
    {
      &args[0]
    }
    _ => expr,
  }
}

/// Parse a single variable-spec list into a `ParsedControl`. `siblings`
/// names the Manipulate's other control variables, so a choice list built
/// from one of them can be marked for live re-resolution; pass an empty
/// slice for a standalone `Control[…]`, which has no siblings.
/// Drop every `NCache[exact, approx]` wrapper, keeping the exact value.
///
/// `NCache` is inert in Wolfram — it is how the front end remembers a
/// numeric approximation of an exact slider bound — but the control parser
/// wants plain numbers, and a Demonstration saved from the front end writes
/// its bounds as `{u, NCache[Pi/4, 0.785…], 1}`.
fn strip_ncache(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "NCache" && args.len() == 2 =>
    {
      strip_ncache(&args[0])
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(strip_ncache).collect::<Vec<_>>().into(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(strip_ncache).collect::<Vec<_>>().into())
    }
    other => other.clone(),
  }
}

fn parse_manipulate_control(
  spec: &Expr,
  siblings: &[String],
) -> Option<ParsedControl> {
  let spec = &strip_ncache(spec);
  let Expr::List(items) = spec else { return None };
  if items.is_empty() {
    return None;
  }

  // Head can be either a plain symbol `u` or `{u, uinit}` / `{u, uinit, ulbl}`.
  let plain_run = |s: String| {
    vec![LabelRun {
      text: s,
      italic: false,
      ..Default::default()
    }]
  };
  let (name, explicit_initial, label_runs) = match &items[0] {
    Expr::Identifier(n) => (n.clone(), None, plain_run(n.clone())),
    Expr::List(head_items) if !head_items.is_empty() => {
      let n = match &head_items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => return None,
      };
      let init = head_items.get(1).cloned();
      // A string label still goes through the label renderer: it may carry
      // inline `\!\(\*SubscriptBox[…]\)` typesetting.
      let lbl = match head_items.get(2) {
        Some(other) => manipulate_label_runs(other, false),
        None => plain_run(n.clone()),
      };
      (n, init, lbl)
    }
    _ => return None,
  };
  let label = flatten_label_runs(&label_runs);

  // `Enabled -> cond` / `Enabled :> cond` gates the control. `Dynamic[expr]`
  // unwraps to its live condition `expr`; a plain value is used as-is. The
  // default `Enabled -> True` needs no gating and yields `None` so the control
  // stays unconditionally enabled.
  let enabled: Option<String> = items.iter().find_map(|it| match it {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Enabled") => {
      let cond = match replacement.as_ref() {
        Expr::FunctionCall { name, args }
          if name == "Dynamic" && args.len() == 1 =>
        {
          &args[0]
        }
        other => other,
      };
      if matches!(cond, Expr::Identifier(s) if s == "True") {
        None
      } else {
        Some(crate::syntax::expr_to_input_form(cond))
      }
    }
    _ => None,
  });

  // `TrackingFunction -> f` / `:> f` runs `f[newValue]` whenever this
  // control's value changes, so a Demonstration can reset a companion
  // control (e.g. rewinding a time slider when the reaction step changes).
  let tracking: Option<String> = items.iter().find_map(|it| match it {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "TrackingFunction") =>
    {
      Some(crate::syntax::expr_to_input_form(replacement))
    }
    _ => None,
  });

  // `Locator` controls (`{{p, init}, pmin, pmax, Locator}`, as a bare
  // marker or `ControlType -> Locator`) and hidden `ControlType -> None`
  // variables (`{{v, init}, ControlType -> None}`).
  let is_locator = items.iter().any(|it| {
    matches!(it, Expr::Identifier(s) if s == "Locator")
      || matches!(
        it,
        Expr::Rule { pattern, replacement }
        | Expr::RuleDelayed { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ControlType")
            && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Locator")
      )
  });
  // `{{x, 1}, None}` states the control's domain as `None`, which is the
  // positional spelling of `ControlType -> None`: the variable is bound but
  // no widget is drawn (verified against wolframscript, which shows no
  // slider for it). A bare `None` in the control-type slot after the bounds
  // — `{{v, init}, {xmin, ymin}, {xmax, ymax}, None}`, how a puzzle
  // Demonstration declares a state variable its buttons drive — says the
  // same thing.
  let is_hidden = items[1..]
    .iter()
    .any(|it| matches!(it, Expr::Identifier(s) if s == "None"))
    || items.iter().any(|it| {
      matches!(
        it,
        Expr::Rule { pattern, replacement }
        | Expr::RuleDelayed { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ControlType")
            && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "None")
      )
    });
  if is_locator || is_hidden {
    let value_expr = explicit_initial
      .clone()
      .or_else(|| items.get(1).cloned())
      .unwrap_or(Expr::Identifier("Null".to_string()));
    if is_hidden {
      // A `ControlType -> None` variable stays a live, mutable binding so
      // an interactive display can rewrite it. Without an explicit initial
      // (`{v, domain, ControlType -> None}` rather than `{{v, init}, …}`)
      // the second element is the variable's *domain*, exactly as it is for
      // a visible control: a list of choices, whose first entry is where the
      // variable starts. Taking the list itself would bind one level too
      // deep — `{aa, {{1, 1, 1, 1}}, ControlType -> None}` starts `aa` at
      // `{1, 1, 1, 1}`, not at `{{1, 1, 1, 1}}`.
      let value = match (&explicit_initial, &value_expr) {
        (None, domain) => match crate::evaluator::evaluate_expr_to_expr(domain)
        {
          Ok(Expr::List(ref choices)) if !choices.is_empty() => {
            crate::syntax::expr_to_input_form(&choices[0])
          }
          Ok(evaluated) => crate::syntax::expr_to_input_form(&evaluated),
          Err(_) => crate::syntax::expr_to_input_form(domain),
        },
        _ => manipulate_value_to_input_form(&value_expr),
      };
      return Some(ParsedControl::State { name, value });
    }
    // A Locator drives its variable interactively: a single `{x, y}` point
    // becomes a 2D slider (like `LocatorPane`), a list of points becomes a
    // multi-point Locator control (one X/Y pair per point, with add/remove
    // when `LocatorAutoCreate -> True`). The coordinate range comes from
    // the `{xmin, ymin}, {xmax, ymax}` bounds when given, else from the
    // points' bounding box. A non-numeric initial value keeps the previous
    // behavior: a fixed binding baked into the body.
    let evaluated = crate::evaluator::evaluate_expr_to_expr(&value_expr)
      .unwrap_or_else(|_| value_expr.clone());
    // A corner bound is often written in terms of *other* control variables
    // (`{{p, {0, 1}}, {xrange[[1]], yrange[[1]]}, {xrange[[2]],
    // yrange[[2]]}, Locator}` reuses two `ControlType -> None` ranges), so
    // each corner is evaluated against the initial-value bindings the caller
    // installed before it is read as a numeric pair. Without this the corners
    // look non-numeric and the locator falls back to a padded box around its
    // starting point.
    let corner_bounds: Vec<(f64, f64)> = items[1..]
      .iter()
      .filter(|it| {
        !matches!(it, Expr::Rule { .. } | Expr::RuleDelayed { .. })
          && !matches!(it, Expr::Identifier(s) if s == "Locator")
      })
      .filter_map(|it| {
        list2_f64(it).or_else(|| {
          list2_f64(&crate::evaluator::evaluate_expr_to_expr(it).ok()?)
        })
      })
      .collect();
    let auto_create = items.iter().any(|it| {
      matches!(
        it,
        Expr::Rule { pattern, replacement }
        | Expr::RuleDelayed { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "LocatorAutoCreate")
            && match replacement.as_ref() {
              Expr::Identifier(s) => {
                s == "True" || s == "Automatic" || s == "All"
              }
              // `LocatorAutoCreate -> {min, max}` bounds how many points
              // the user may add; any such range still means "adding is
              // allowed".
              Expr::List(bounds) => !bounds.is_empty(),
              _ => false,
            }
      )
    });
    if let Some((x, y)) = list2_f64(&evaluated) {
      let (x_min, x_max, y_min, y_max) =
        locator_range(&corner_bounds, &[(x, y)]);
      return Some(ParsedControl::Visible {
        control: ManipulateControl::Slider2D {
          name,
          x_min,
          x_max,
          y_min,
          y_max,
          x_initial: x,
          y_initial: y,
          label,
          write_callback: None,
        },
        enabled,
        min_code: None,
        max_code: None,
        values_code: None,
        animate: None,
        tracking: tracking.clone(),
      });
    }
    if let Some(points) = point_list_f64(&evaluated) {
      let (x_min, x_max, y_min, y_max) = locator_range(&corner_bounds, &points);
      return Some(ParsedControl::Visible {
        control: ManipulateControl::Locator {
          name,
          points,
          x_min,
          x_max,
          y_min,
          y_max,
          auto_create,
          label,
        },
        enabled,
        min_code: None,
        max_code: None,
        values_code: None,
        animate: None,
        tracking: tracking.clone(),
      });
    }
    let value = manipulate_value_to_input_form(&value_expr);
    return Some(ParsedControl::Fixed { name, value });
  }

  // A `ControlType -> Slider2D` / `ControlType -> IntervalSlider` option
  // selects a compound control. The control type may also appear as a bare
  // identifier in the spec (`{u, {1, 2, 3}, SetterBar}`). The bounds are
  // the non-option, non-control-type items after the head.
  let control_type = items
    .iter()
    .find_map(|it| match it {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "ControlType") => {
        match replacement.as_ref() {
          Expr::Identifier(s) => Some(s.clone()),
          _ => None,
        }
      }
      _ => None,
    })
    .or_else(|| {
      items[1..].iter().find_map(|it| match it {
        Expr::Identifier(s) if is_control_type_name(s) => Some(s.clone()),
        _ => None,
      })
    });
  let bounds: Vec<&Expr> = items[1..]
    .iter()
    .filter(|it| {
      !matches!(it, Expr::Rule { .. } | Expr::RuleDelayed { .. })
        && !matches!(it, Expr::Identifier(s) if is_control_type_name(s))
    })
    .collect();

  // `Appearance -> "Vertical"` (or the bare symbol `Vertical`) stacks a
  // SetterBar/RadioButtonBar/CheckboxBar in a column instead of Wolfram's
  // default horizontal bar.
  let appearance_vertical = items.iter().any(|it| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = it
    else {
      return false;
    };
    matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Appearance")
      && (matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Vertical")
        || matches!(replacement.as_ref(), Expr::String(s) if s == "Vertical"))
  });

  // A `ControlType -> Trigger` (or bare `Trigger` marker) becomes a
  // dedicated play/pause control sweeping its variable from `min` towards
  // `max` — whether that end is infinite (`{time, 0, Infinity, 1, …}`, the
  // Demonstrations "run/stop simulation" control) or finite (`{{t, 0, ""},
  // 0, tmax, .01, Trigger, DefaultDuration -> tmax}`, the "play once over a
  // fixed duration" pattern). When this variable already has a visible row
  // from an earlier spec (Kepler pairs a `Trigger` with a plain slider on
  // the same `t`), the row built here is dropped by the caller's dedup
  // check and only this control's `animate` field takes effect, so building
  // the dedicated widget here is safe either way.
  if control_type.as_deref() == Some("Trigger") {
    let min = bounds
      .first()
      .and_then(|e| {
        crate::functions::math_ast::try_eval_to_f64_with_infinity(e)
      })
      .unwrap_or(0.0);
    let max = bounds
      .get(1)
      .and_then(|e| {
        crate::functions::math_ast::try_eval_to_f64_with_infinity(e)
      })
      .unwrap_or(f64::INFINITY);
    let step = bounds
      .get(2)
      .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e))
      .unwrap_or(1.0);
    let initial = explicit_initial
      .as_ref()
      .and_then(crate::functions::math_ast::try_eval_to_f64)
      .unwrap_or(min);
    // Wolfram's Trigger sits paused until pressed; only an explicit
    // `AnimationRunning -> True` starts it sweeping immediately.
    let running = items.iter().any(|it| {
      matches!(
        it,
        Expr::Rule { pattern, replacement }
        | Expr::RuleDelayed { pattern, replacement }
          if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "AnimationRunning")
            && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True")
      )
    });
    return Some(ParsedControl::Visible {
      control: ManipulateControl::Trigger {
        name,
        min,
        max,
        step,
        initial,
        running,
        label,
        label_runs,
      },
      enabled,
      min_code: None,
      max_code: None,
      values_code: None,
      animate: Some(running),
      tracking: tracking.clone(),
    });
  }

  // 2D control: either an explicit `ControlType -> Slider2D`, or a range
  // given as two corner points `{u, {xmin, ymin}, {xmax, ymax}}`.
  let is_2d_range = bounds.len() >= 2
    && list2_f64(bounds[0]).is_some()
    && list2_f64(bounds[1]).is_some();
  if control_type.as_deref() == Some("Slider2D") || is_2d_range {
    let (x_min, x_max, y_min, y_max) = if is_2d_range {
      let (x0, y0) = list2_f64(bounds[0])?;
      let (x1, y1) = list2_f64(bounds[1])?;
      (x0, x1, y0, y1)
    } else {
      // Scalar bounds `{u, min, max}` apply to both axes.
      let mn = bounds.first().and_then(|e| eval_manipulate_bound(e))?.0;
      let mx = bounds
        .get(1)
        .and_then(|e| eval_manipulate_bound(e))
        .map_or(mn + 1.0, |(v, _)| v);
      (mn, mx, mn, mx)
    };
    let (x_initial, y_initial) =
      match explicit_initial.as_ref().and_then(list2_f64) {
        Some((a, b)) => (a, b),
        None => (x_min, y_min),
      };
    return Some(ParsedControl::Visible {
      control: ManipulateControl::Slider2D {
        name,
        x_min,
        x_max,
        y_min,
        y_max,
        x_initial,
        y_initial,
        label,
        write_callback: None,
      },
      enabled,
      min_code: None,
      max_code: None,
      values_code: None,
      animate: None,
      tracking: tracking.clone(),
    });
  }

  // Interval control: `{u, min, max, ControlType -> IntervalSlider}` binds
  // `u` to a `{low, high}` pair.
  if control_type.as_deref() == Some("IntervalSlider") {
    let min = bounds.first().and_then(|e| eval_manipulate_bound(e))?.0;
    let max = bounds
      .get(1)
      .and_then(|e| eval_manipulate_bound(e))
      .map_or(min + 1.0, |(v, _)| v);
    let step = bounds
      .get(2)
      .and_then(|e| eval_manipulate_bound(e))
      .map(|(v, _)| v);
    let (low_initial, high_initial) =
      match explicit_initial.as_ref().and_then(list2_f64) {
        Some((a, b)) => (a, b),
        None => (min, max),
      };
    return Some(ParsedControl::Visible {
      control: ManipulateControl::IntervalSlider {
        name,
        min,
        max,
        step,
        low_initial,
        high_initial,
        label,
      },
      enabled,
      min_code: None,
      max_code: None,
      values_code: None,
      animate: None,
      tracking: tracking.clone(),
    });
  }

  // Discrete form: `{u, {u1, u2, …}}` or `{{u, uinit, …}, {u1, u2, …}}`,
  // possibly with trailing options (`{{u, uinit, ""}, {u1, …},
  // ControlType -> PopupMenu}`) — hence matched on the option-free `bounds`.
  // The value list may also be given as an expression that evaluates to a
  // list (e.g. `{g, PolyhedronData[All]}`), so evaluate a non-literal.
  // A control type that offers one widget per value (`RadioButton`,
  // `SetterBar`, …) turns a numeric range into that list of values:
  // `{{n, 1, "frequency"}, 1, 2, 1, RadioButton}` offers 1 and 2, which is
  // what Wolfram draws a radio button for — not a slider.
  let picks_one_value_per_choice = matches!(
    control_type.as_deref(),
    Some(
      "RadioButton"
        | "RadioButtonBar"
        | "Setter"
        | "SetterBar"
        | "Toggler"
        | "TogglerBar"
        | "PopupMenu"
    )
  );

  // A bar of checkboxes (or of togglers) picks *several* of its choices at
  // once: the variable binds the list of them, so the widget is the
  // multiple-choice bar rather than a one-of picker.
  if matches!(control_type.as_deref(), Some("CheckboxBar" | "TogglerBar"))
    && bounds.len() == 1
    && let Some(choices) = match unwrap_dynamic_choices(bounds[0]) {
      l @ Expr::List(_) => Some(l.clone()),
      other => crate::evaluator::evaluate_expr_to_expr(other).ok(),
    }
    && matches!(choices, Expr::List(_))
  {
    let value = match &explicit_initial {
      Some(init) => manipulate_value_to_input_form(init),
      None => "{}".to_string(),
    };
    let display = if appearance_vertical {
      format!(
        "TogglerBar[Dynamic[{name}], {}, Appearance -> \"Vertical\"]",
        crate::syntax::expr_to_input_form(&choices)
      )
    } else {
      format!(
        "TogglerBar[Dynamic[{name}], {}]",
        crate::syntax::expr_to_input_form(&choices)
      )
    };
    return Some(ParsedControl::StateWithDisplay {
      name,
      value,
      display,
    });
  }
  {
    let value_items: Option<Vec<Expr>> = if bounds.len() == 1 {
      match unwrap_dynamic_choices(bounds[0]) {
        Expr::List(vs) => Some(vs.iter().cloned().collect()),
        other => match crate::evaluator::evaluate_expr_to_expr(other) {
          Ok(Expr::List(ref vs)) => Some(vs.iter().cloned().collect()),
          _ => None,
        },
      }
    } else if picks_one_value_per_choice && (2..=3).contains(&bounds.len()) {
      enumerate_range(&bounds)
    } else {
      None
    };
    if let Some(value_items) = value_items {
      let (values, value_labels, value_label_svgs) =
        discrete_choice_columns(&value_items);
      if values.is_empty() {
        return None;
      }
      let initial_index = match explicit_initial {
        Some(init) => {
          let init_code = crate::syntax::expr_to_input_form(&init);
          values
            .iter()
            .position(|v| *v == init_code)
            .or_else(|| {
              // The variable spec is held (so the symbol survives) while the
              // choice list arrives evaluated, so an initial value written as
              // an expression — `{{fac, 10^6, …}, {1, 10, 10^6}}` — only
              // matches once it is evaluated too.
              let evaluated =
                crate::evaluator::evaluate_expr_to_expr(&init).ok()?;
              let code = crate::syntax::expr_to_input_form(&evaluated);
              values.iter().position(|v| *v == code)
            })
            .unwrap_or(0)
        }
        None => 0,
      };
      // A choice list built from another control's variable (`Range[1,
      // If[flat, 3, 6], 1]`) only holds for that variable's current value;
      // keep its code so the frontend can rebuild the choices whenever the
      // other control moves. A `Dynamic[…]` wrapper is stripped first (see
      // `unwrap_dynamic_choices`) so the kept code re-evaluates cleanly too.
      let values_code = (bounds.len() == 1
        && expr_references_any(unwrap_dynamic_choices(bounds[0]), siblings))
      .then(|| {
        crate::syntax::expr_to_input_form(unwrap_dynamic_choices(bounds[0]))
      });
      return Some(ParsedControl::Visible {
        control: ManipulateControl::Discrete {
          name,
          values,
          value_labels,
          value_label_svgs,
          initial_index,
          label,
          label_runs,
          popup: control_type.as_deref() == Some("PopupMenu"),
          // `SetterBar` and `RadioButtonBar` both always draw the full row
          // of buttons (a row of highlighted setters, or of radio dots)
          // regardless of choice count — unlike a spec that stays silent,
          // which the automatic SetterBar/PopupMenu split
          // (`renders_as_setter_bar`, in woxi-studio) only applies to.
          setter_bar: matches!(
            control_type.as_deref(),
            Some("SetterBar" | "RadioButtonBar")
          ),
          slider: matches!(
            control_type.as_deref(),
            Some("Slider" | "VerticalSlider" | "Manipulator")
          ),
          vertical: appearance_vertical,
        },
        enabled,
        min_code: None,
        max_code: None,
        values_code,
        animate: None,
        tracking: tracking.clone(),
      });
    }
  }

  // Custom-control form: `{{u, uinit, ulbl}, func}`, where `func` builds
  // the widget rather than describing a range. Wolfram applies it to the
  // variable's `Dynamic`; the Demonstrations idiom is a `Button` that
  // resets the variable, so apply the function and take the widget it
  // returns. The variable itself becomes live state so the action can
  // write into it.
  // The spec list arrives with a bare `&` function wrapped in `Dynamic`.
  let custom_builder = match bounds.first() {
    Some(Expr::Function { .. }) => bounds.first().copied(),
    Some(Expr::FunctionCall { name: n, args: a })
      if n == "Dynamic"
        && a.len() == 1
        && matches!(&a[0], Expr::Function { .. }) =>
    {
      Some(&a[0])
    }
    _ => None,
  };
  if bounds.len() == 1
    && let Some(Expr::Function { body }) = custom_builder
    // Substitute the slot rather than applying the function: `Button` holds
    // its action, and evaluating here would *run* the reset instead of
    // storing it.
    && let built = crate::syntax::substitute_slots(
      body,
      &[call1("Dynamic", Expr::Identifier(name.clone()))],
    )
    && let Expr::FunctionCall {
      name: built_name,
      args: built_args,
    } = &built
    && built_name == "Button"
    && built_args.len() >= 2
  {
    let button_runs = manipulate_label_runs(&built_args[0], false);
    let value = explicit_initial
      .as_ref()
      .map_or_else(|| "Null".to_string(), crate::syntax::expr_to_input_form);
    return Some(ParsedControl::StateWithControl {
      name,
      value,
      control: ManipulateControl::Button {
        label: flatten_label_runs(&button_runs),
        label_runs: button_runs,
        action: crate::syntax::expr_to_input_form(&built_args[1]),
      },
    });
  }

  // Colour form: `{{u, uinit, ulbl}, colour}` — wolframscript renders a
  // ColorSetter whose swatches are the initial colour and the listed
  // alternate(s), matching the Demonstrations idiom of toggling between a
  // couple of fixed colours. With an explicit initial colour that differs
  // from `colour`, the two form a real 2-swatch choice; without one there is
  // only a single possible value, so the variable stays fixed at it (there
  // is nothing to pick between).
  if bounds.len() == 1
    && crate::functions::graphics::parse_color(bounds[0]).is_some()
  {
    let alt_code = crate::syntax::expr_to_input_form(bounds[0]);
    if let Some(init) = explicit_initial.as_ref().filter(|init| {
      parse_color(init).is_some()
        && crate::syntax::expr_to_input_form(init) != alt_code
    }) {
      let value_items = vec![init.clone(), bounds[0].clone()];
      let (values, value_labels, value_label_svgs) =
        discrete_choice_columns(&value_items);
      return Some(ParsedControl::Visible {
        control: ManipulateControl::Discrete {
          name,
          values,
          value_labels,
          value_label_svgs,
          initial_index: 0,
          label,
          label_runs,
          popup: false,
          // A ColorSetter always shows its swatches as a row of buttons,
          // never a dropdown, and that row is always horizontal.
          setter_bar: true,
          slider: false,
          vertical: appearance_vertical,
        },
        enabled,
        min_code: None,
        max_code: None,
        values_code: None,
        animate: None,
        tracking: tracking.clone(),
      });
    }
    let value = crate::syntax::expr_to_input_form(bounds[0]);
    return Some(ParsedControl::Fixed { name, value });
  }

  // Continuous form: {u, umin, umax} or {u, umin, umax, du}
  // (or with labelled head: {{u, uinit, ulbl}, umin, umax, …}),
  // possibly with trailing options (`ImageSize -> Tiny`, …) — hence
  // matched on the option-free `bounds`.
  if bounds.len() < 2 {
    return None;
  }
  let (mut min, min_dynamic) = eval_manipulate_bound(bounds[0])?;
  let (mut max, max_dynamic) = eval_manipulate_bound(bounds[1])?;
  // A bound that only resolved through the environment references another
  // control variable (Kepler's `{{t, 0, …}, 0, P, .01}`), as does one
  // written `Dynamic[…]`; keep its code — the bare expression, without the
  // `Dynamic` wrapper — so the frontend can re-resolve it against the live
  // bindings and let the slider range follow the other control.
  let min_code = min_dynamic
    .then(|| {
      crate::syntax::expr_to_input_form(manipulate_bound_expr(bounds[0]).0)
    })
    .filter(|_| min.is_finite());
  let max_code = max_dynamic
    .then(|| {
      crate::syntax::expr_to_input_form(manipulate_bound_expr(bounds[1]).0)
    })
    .filter(|_| max.is_finite());
  // An infinite bound (`Animate[…, {ϕ, 0, Infinity}]` runs forever in
  // Wolfram) cannot drive a finite slider; substitute a 2π looping window
  // so the default sine-based demonstrations wrap seamlessly.
  match (min.is_finite(), max.is_finite()) {
    (true, false) => max = min + 2.0 * std::f64::consts::PI,
    (false, true) => min = max - 2.0 * std::f64::consts::PI,
    (false, false) => {
      min = 0.0;
      max = 2.0 * std::f64::consts::PI;
    }
    (true, true) => {}
  }
  let step = bounds
    .get(2)
    .and_then(|e| eval_manipulate_bound(e))
    .map(|(v, _)| v);
  let initial = match explicit_initial.as_ref() {
    Some(init) => eval_manipulate_bound(init).map_or(min, |(v, _)| v),
    None => min,
  };
  // The variable stays machine-real for the widget's whole lifetime once any
  // of its spec terms was inexact — even a slider that happens to sit at a
  // "round" position (e.g. `rq = 0` on a `0, 1, 0.01` range) binds `0.`, not
  // the exact integer `0`.
  let is_real = manipulate_bound_is_inexact(bounds[0])
    || manipulate_bound_is_inexact(bounds[1])
    || bounds
      .get(2)
      .is_some_and(|e| manipulate_bound_is_inexact(e))
    || explicit_initial
      .as_ref()
      .is_some_and(manipulate_bound_is_inexact);

  // A `Trigger`/`Animator` control is a play button sweeping its variable
  // over the range: the widget animates that variable (a Trigger starts
  // paused, an Animator running).
  let animate = match control_type.as_deref() {
    Some("Trigger") => Some(false),
    Some("Animator") => Some(true),
    _ => None,
  };

  // `{u, umin, umax}` with umin > umax is a documented reversed-direction
  // slider (e.g. an angle control counting down from a positive start to a
  // negative end). `initial` above is already resolved against the original
  // umin/umax order, so it is safe to sort the pair now — every downstream
  // consumer (the slider widget, dynamic-bounds re-resolution) expects
  // `min <= max`.
  if min > max {
    std::mem::swap(&mut min, &mut max);
  }

  Some(ParsedControl::Visible {
    control: ManipulateControl::Continuous {
      name,
      min,
      max,
      step,
      initial,
      label,
      label_runs,
      is_real,
    },
    enabled,
    min_code,
    max_code,
    values_code: None,
    animate,
    tracking,
  })
}

/// The values a `{min, max}` / `{min, max, step}` control range covers,
/// for the control types that offer one widget per value. `step` defaults
/// to 1, as it does for an integer range in Wolfram. Returns `None` when
/// the bounds are not numeric or the range does not terminate.
fn enumerate_range(bounds: &[&Expr]) -> Option<Vec<Expr>> {
  let value = |e: &Expr| expr_to_f64(e);
  let min = value(bounds.first()?)?;
  let max = value(bounds.get(1)?)?;
  let step = match bounds.get(2) {
    Some(e) => value(e)?,
    None => 1.0,
  };
  if step <= 0.0 || !(max - min).is_finite() {
    return None;
  }
  let integral = |v: f64| (v - v.round()).abs() < 1e-9;
  let exact = integral(min) && integral(step);
  let mut values = Vec::new();
  let mut v = min;
  // The half-step slack keeps a range whose end is only reachable up to
  // floating-point error (`0, 1, 0.1`) from losing its last value.
  while v <= max + step / 2.0 {
    values.push(if exact {
      Expr::Integer(v.round() as i128)
    } else {
      Expr::Real(v)
    });
    v += step;
  }
  (!values.is_empty()).then_some(values)
}

/// If `item` is a rule `lhs -> rhs` (in either `Expr::Rule` or
/// `Rule[…]` function-call form), return `(lhs, rhs)`. Used to split a
/// discrete-choice spec like `True -> "Yin-Yang"` into its bound value and
/// its display label.
fn discrete_choice_rule(item: &Expr) -> Option<(&Expr, &Expr)> {
  match item {
    Expr::Rule {
      pattern,
      replacement,
    } => Some((pattern, replacement)),
    Expr::FunctionCall { name, args } if name == "Rule" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

/// Render a discrete-choice label. A string label is shown without its
/// surrounding quotes; presentation wrappers (`Style["P", Italic]`,
/// `Row[{…}]`) render as their display text via the label-run renderer;
/// anything that renders empty falls back to its InputForm.
fn discrete_choice_label(expr: &Expr) -> String {
  match expr {
    // A string may carry inline typeset boxes the FrontEnd wrote as
    // `\!\(\*…\)` — an antiquark's `\!\(\*OverscriptBox[\(u\), \(_\)]\)`
    // reads as `ū`, not the private-use box markers it's stored with.
    // A plain string just needs its remaining private-use code points
    // (e.g. `\[WarningSign]`) swapped for real glyphs.
    Expr::String(s) => match inline_box_label_runs(s, false) {
      Some(runs) => {
        crate::syntax::substitute_private_use_glyphs(&flatten_label_runs(&runs))
          .into_owned()
      }
      None => crate::syntax::substitute_private_use_glyphs(s).into_owned(),
    },
    other => {
      let flat = flatten_label_runs(&manipulate_label_runs(other, false));
      // A structural head (Grid/Row/Column/…) can legitimately typeset to
      // nothing — e.g. a Grid whose cells are all `""`, marking "no flag
      // set" among a family of choices that each flip one cell on. Only an
      // unrecognized head's empty result means "couldn't render", which
      // falls back to its source so the choice still shows something.
      let structural = matches!(other, Expr::FunctionCall { name, .. } if is_text_layout_head(name));
      if flat.is_empty() && !structural {
        crate::syntax::expr_to_input_form(other)
      } else {
        flat
      }
    }
  }
}

/// Heads that arrange or annotate *text*, so a label built from them is
/// rendered by `manipulate_label_runs` rather than as a graphical icon.
fn is_text_layout_head(name: &str) -> bool {
  matches!(
    name,
    "Row"
      | "Column"
      | "Grid"
      | "TableForm"
      | "TextGrid"
      | "Style"
      | "Text"
      | "Framed"
      | "Labeled"
      | "Tooltip"
      | "DisplayForm"
      | "TraditionalForm"
      | "StandardForm"
      | "Subscript"
      | "Superscript"
      | "Subsuperscript"
  )
}

/// Rendered SVG for a discrete-choice label that is a graphic (e.g.
/// `"+" -> myIcon[2]` in a Demonstrations crosshair picker). A held
/// `Graphics[…]` call or a call producing one is evaluated; anything
/// text-like yields `None`.
fn discrete_choice_label_svg(label: &Expr) -> Option<String> {
  match label {
    Expr::Graphics { svg, .. } => Some(svg.clone()),
    // A raster image label (e.g. the `ColorData[…, "Image"]` swatches of a
    // gradient picker) wraps its pixels as an SVG document.
    Expr::Image {
      width,
      height,
      channels,
      data,
      ..
    } => Some(crate::functions::image_ast::image_to_svg_document(
      *width, *height, *channels, data,
    )),
    // Evaluating through the interpreter (not `evaluate_expr_to_expr`)
    // is what renders a held `Graphics[…]` call — or a user-defined icon
    // function like `myIcon[2]` — to SVG. A call producing an `Image`
    // (e.g. `Show[ColorData[name, "Image"], ImageSize -> 100]`) recurses
    // into the raster arm above.
    //
    // Text layout heads never become icons: `Column[{"…", "…"}]` has no
    // graphical meaning, and evaluating it only yields the typeset echo of
    // its own source — which would put the label's InputForm on the button.
    // `manipulate_label_runs` renders these structurally instead.
    Expr::FunctionCall { name, .. } if is_text_layout_head(name) => None,
    Expr::FunctionCall { .. } => {
      let code = crate::syntax::expr_to_input_form(label);
      match crate::interpret_with_stdout(&code) {
        Ok(result) => {
          if result.graphics.is_some() {
            return result.graphics;
          }
          match crate::interpret_to_expr(&code) {
            Ok(img @ Expr::Image { .. }) => discrete_choice_label_svg(&img),
            _ => None,
          }
        }
        Err(_) => None,
      }
    }
    _ => None,
  }
}

/// Pick a reasonable current value for each control. For continuous
/// controls this is the `initial`; for discrete controls it is the value
/// at `initial_index`. Returns `(variable_name, input_form_value)` pairs.
pub fn manipulate_initial_bindings(
  spec: &ManipulateSpec,
) -> Vec<(String, String)> {
  spec
    .controls
    .iter()
    .filter_map(|c| match c {
      // Annotation and button rows bind no variable.
      ManipulateControl::Heading { .. }
      | ManipulateControl::Divider
      | ManipulateControl::Button { .. } => None,
      ManipulateControl::Continuous {
        name,
        initial,
        is_real,
        ..
      } => Some((
        name.clone(),
        if *is_real {
          format_f64_real(*initial)
        } else {
          format_f64_input(*initial)
        },
      )),
      ManipulateControl::Trigger { name, initial, .. } => {
        Some((name.clone(), format_f64_input(*initial)))
      }
      ManipulateControl::Discrete {
        name,
        values,
        initial_index,
        ..
      } => Some((
        name.clone(),
        values
          .get(*initial_index)
          .cloned()
          .unwrap_or_else(|| "Null".to_string()),
      )),
      ManipulateControl::Slider2D {
        name,
        x_initial,
        y_initial,
        ..
      } => Some((
        name.clone(),
        format!(
          "{{{}, {}}}",
          format_f64_input(*x_initial),
          format_f64_input(*y_initial)
        ),
      )),
      ManipulateControl::IntervalSlider {
        name,
        low_initial,
        high_initial,
        ..
      } => Some((
        name.clone(),
        format!(
          "{{{}, {}}}",
          format_f64_input(*low_initial),
          format_f64_input(*high_initial)
        ),
      )),
      ManipulateControl::Locator { name, points, .. } => {
        Some((name.clone(), format_point_list_input(points)))
      }
    })
    // Mutable `ControlType -> None` state variables travel in the binding
    // set alongside the visible controls so displays can read/write them.
    .chain(spec.state.iter().cloned())
    .collect()
}

/// Format a f64 in a round-trip-safe way as Wolfram input code.
/// Integers are rendered without a decimal point so that e.g. Factor[x^n + 1]
/// with n = 10 substitutes as 10 (Integer) rather than 10. (Real).
fn format_f64_input(v: f64) -> String {
  if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
    format!("{}", v as i64)
  } else {
    format!("{v}")
  }
}

/// Format a list of 2D points as Wolfram input code, e.g.
/// `{{2., 2.}, {8., 2.}, {8., 8.}}`. The binding value for a `Locator`
/// control. Locator positions are machine reals (dragging produces
/// fractional coordinates), so integral values keep a trailing dot.
pub fn format_point_list_input(points: &[(f64, f64)]) -> String {
  let parts: Vec<String> = points
    .iter()
    .map(|(x, y)| {
      format!("{{{}, {}}}", format_f64_real(*x), format_f64_real(*y))
    })
    .collect();
  format!("{{{}}}", parts.join(", "))
}

/// Format an f64 as a Wolfram machine-real literal: integral values keep a
/// trailing dot (`2.`) so they substitute as Real, not Integer.
fn format_f64_real(v: f64) -> String {
  if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
    format!("{}.", v as i64)
  } else {
    format!("{v}")
  }
}

/// Build a `Block[{a = val, b = val}, body]` expression as a source-code
/// string, ready to hand to `interpret_with_stdout`.
pub fn manipulate_block_code(
  body_code: &str,
  bindings: &[(String, String)],
) -> String {
  if bindings.is_empty() {
    return body_code.to_string();
  }
  let binding_parts: Vec<String> = bindings
    .iter()
    .map(|(name, value)| format!("{name} = {value}"))
    .collect();
  format!("Block[{{{}}}, {}]", binding_parts.join(", "), body_code)
}

/// Evaluate a control's `Enabled` condition and report whether the control
/// should be interactive. The condition is a boolean expression in the
/// Manipulate variables; it is evaluated against whatever bindings are
/// currently installed as globals (the caller wraps this in
/// [`crate::with_scoped_globals`]). The control is disabled only when the
/// condition evaluates to the literal `False`; a symbolic or errored result
/// fails open (enabled) so a control never becomes permanently stuck.
pub fn manipulate_condition_enabled(condition: &str) -> bool {
  match crate::interpret(condition) {
    Ok(result) => result.trim() != "False",
    Err(_) => true,
  }
}

/// Evaluate each condition against the given bindings and return one flag per
/// condition. A `None` condition (control with no `Enabled` option) is always
/// enabled. Installs the bindings as globals once for the whole batch.
pub fn manipulate_enabled_states(
  conditions: &[Option<String>],
  bindings: &[(String, String)],
) -> Vec<bool> {
  if conditions.iter().all(Option::is_none) {
    return vec![true; conditions.len()];
  }
  crate::with_scoped_globals(bindings, || {
    conditions
      .iter()
      .map(|c| match c {
        Some(cond) => manipulate_condition_enabled(cond),
        None => true,
      })
      .collect()
  })
}

/// Parse a very small JSON object `{"name": "value", …}` where every
/// value is a string (an InputForm fragment), into an ordered list of
/// `(name, value)` pairs. Non-string values are coerced to their textual
/// form. Kept minimal to avoid pulling in a JSON dependency — the caller
/// (the Playground worker or JupyterLite kernel) always provides string
/// values on purpose.
///
/// Correctly decodes multi-byte UTF-8 in keys and values, so Manipulate
/// variable names like `ω` or `ϕ` round-trip without being mangled.
pub fn parse_manipulate_bindings(s: &str) -> Vec<(String, String)> {
  let bytes = s.as_bytes();
  let mut i = 0;
  let mut out: Vec<(String, String)> = Vec::new();

  // Skip leading whitespace and the opening brace.
  while i < bytes.len() && bytes[i].is_ascii_whitespace() {
    i += 1;
  }
  if i >= bytes.len() || bytes[i] != b'{' {
    return out;
  }
  i += 1;

  loop {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i >= bytes.len() || bytes[i] == b'}' {
      break;
    }

    // Key must be a JSON string.
    if bytes[i] != b'"' {
      break;
    }
    let Some((key, next)) = parse_json_string(bytes, i) else {
      break;
    };
    i = next;

    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i >= bytes.len() || bytes[i] != b':' {
      break;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i >= bytes.len() {
      break;
    }

    // Value: string, number, true/false, or null. We stringify each.
    let (value, next) = if bytes[i] == b'"' {
      match parse_json_string(bytes, i) {
        Some(v) => v,
        None => break,
      }
    } else {
      let start = i;
      while i < bytes.len() && bytes[i] != b',' && bytes[i] != b'}' {
        i += 1;
      }
      let slice = s[start..i].trim().to_string();
      (slice, i)
    };
    i = next;
    out.push((key, value));

    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i < bytes.len() && bytes[i] == b',' {
      i += 1;
      continue;
    }
    break;
  }

  out
}

/// Parse a small JSON array of string literals `["a", "b", …]` into a
/// `Vec<String>`. Non-string elements are skipped. Used to decode the
/// `displays` and `mutations` arguments handed to `evaluate_manipulate_full`
/// (each a list of InputForm code fragments), so their embedded brackets and
/// commas survive without a full JSON dependency.
pub fn parse_json_string_array(s: &str) -> Vec<String> {
  let bytes = s.as_bytes();
  let mut i = 0;
  let mut out: Vec<String> = Vec::new();
  while i < bytes.len() && bytes[i] != b'[' {
    i += 1;
  }
  if i >= bytes.len() {
    return out;
  }
  i += 1; // past '['
  loop {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i >= bytes.len() || bytes[i] == b']' {
      break;
    }
    if bytes[i] == b'"' {
      match parse_json_string(bytes, i) {
        Some((val, next)) => {
          out.push(val);
          i = next;
        }
        None => break,
      }
    } else {
      // Skip an unexpected non-string token up to the next separator.
      while i < bytes.len() && bytes[i] != b',' && bytes[i] != b']' {
        i += 1;
      }
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
      i += 1;
    }
    if i < bytes.len() && bytes[i] == b',' {
      i += 1;
      continue;
    }
    break;
  }
  out
}

/// Apply a set of interactive write-back mutations — each an assignment like
/// `data[[3, 5]] = 1` produced by toggling a `Checkbox` — to the current
/// `bindings`, and return the updated InputForm value of every mutated
/// variable. The target variable of a mutation is the leading symbol of its
/// left-hand side (`data` for `data[[3, 5]] = 1`). All mutations run in one
/// `Block` so later ones see earlier writes.
pub fn apply_manipulate_mutations(
  bindings: &[(String, String)],
  mutations: &[String],
) -> Vec<(String, String)> {
  if mutations.is_empty() {
    return Vec::new();
  }
  let mut vars: Vec<String> = Vec::new();
  for m in mutations {
    if let Some(v) = mutation_target_symbol(m)
      && !vars.contains(&v)
    {
      vars.push(v);
    }
  }
  if vars.is_empty() {
    return Vec::new();
  }
  let body = format!("{}; {{{}}}", mutations.join("; "), vars.join(", "));
  let code = manipulate_block_code(&body, bindings);
  match crate::interpret_to_expr(&code) {
    Ok(Expr::List(ref vals)) if vals.len() == vars.len() => vars
      .into_iter()
      .zip(vals.iter())
      .map(|(name, v)| (name, crate::syntax::expr_to_input_form(v)))
      .collect(),
    _ => Vec::new(),
  }
}

/// Run a Manipulate `Button[…]` action against the current bindings and
/// return the updated InputForm value of every bound variable. The action's
/// writes to unbound globals (e.g. `{U, V} = {Uinit, Vinit}`) persist as
/// ordinary global side effects; writes to the bound control variables are
/// captured through the returned list so the caller can move its widgets
/// (e.g. `time = 0` rewinds a Trigger control).
pub fn apply_manipulate_button_action(
  bindings: &[(String, String)],
  action: &str,
) -> Vec<(String, String)> {
  if bindings.is_empty() {
    let _ = crate::interpret_with_stdout(action);
    return Vec::new();
  }
  let names: Vec<&str> = bindings.iter().map(|(n, _)| n.as_str()).collect();
  let body = format!("{action}; {{{}}}", names.join(", "));
  let code = manipulate_block_code(&body, bindings);
  match crate::interpret_to_expr(&code) {
    Ok(Expr::List(ref vals)) if vals.len() == names.len() => names
      .into_iter()
      .map(str::to_string)
      .zip(vals.iter().map(crate::syntax::expr_to_input_form))
      .collect(),
    _ => Vec::new(),
  }
}

/// Read the current value of each named Manipulate variable, as InputForm.
///
/// Called inside the scope the body was just evaluated in, so it reports
/// what the body left behind: a Manipulate body is free to assign to the
/// widget's own variables (`{v, e} = ve[[n]]`), and those assignments are
/// part of the widget's state in Wolfram, not throwaway locals. Names that
/// no longer have a value are skipped.
pub fn read_manipulate_state(names: &[String]) -> Vec<(String, String)> {
  names
    .iter()
    .filter_map(|name| {
      let value = crate::interpret_to_expr(name).ok()?;
      // An unset symbol evaluates to itself; there is nothing to record.
      if matches!(&value, Expr::Identifier(s) if s == name) {
        return None;
      }
      Some((name.clone(), crate::syntax::expr_to_input_form(&value)))
    })
    .collect()
}

/// The distinct target variables of a set of write-back assignments, in first-
/// seen order. Used to read back the mutated state values after applying the
/// assignments to the (globally-installed) bindings.
pub fn mutation_target_symbols(mutations: &[String]) -> Vec<String> {
  let mut vars: Vec<String> = Vec::new();
  for m in mutations {
    if let Some(v) = mutation_target_symbol(m)
      && !vars.contains(&v)
    {
      vars.push(v);
    }
  }
  vars
}

/// The target variable of a write-back assignment: the leading identifier of
/// its left-hand side, up to the first `[`, whitespace, or `=`.
fn mutation_target_symbol(m: &str) -> Option<String> {
  let end = m.find(|c: char| c == '[' || c == '=' || c.is_whitespace())?;
  let sym = m[..end].trim();
  if sym.is_empty() {
    None
  } else {
    Some(sym.to_string())
  }
}

/// Parse a JSON string literal starting at `start` (which must point at
/// an opening `"`). Returns the decoded string and the index after the
/// closing quote.
///
/// Non-escaped, non-quote bytes are accumulated verbatim and decoded as
/// UTF-8 at the end, so multi-byte characters (e.g. Greek letters used
/// as Manipulate variable names like `ω` or `ϕ`) round-trip correctly.
fn parse_json_string(bytes: &[u8], start: usize) -> Option<(String, usize)> {
  if start >= bytes.len() || bytes[start] != b'"' {
    return None;
  }
  let mut i = start + 1;
  let mut out: Vec<u8> = Vec::new();
  while i < bytes.len() {
    let c = bytes[i];
    if c == b'"' {
      return String::from_utf8(out).ok().map(|s| (s, i + 1));
    }
    if c == b'\\' && i + 1 < bytes.len() {
      let esc = bytes[i + 1];
      match esc {
        b'"' => out.push(b'"'),
        b'\\' => out.push(b'\\'),
        b'/' => out.push(b'/'),
        b'n' => out.push(b'\n'),
        b'r' => out.push(b'\r'),
        b't' => out.push(b'\t'),
        b'b' => out.push(0x08),
        b'f' => out.push(0x0C),
        // Unicode escapes and others: pass through raw (best-effort).
        _ => out.push(esc),
      }
      i += 2;
      continue;
    }
    out.push(c);
    i += 1;
  }
  None
}

/// JSON-escape a string. Shared with the wasm output builder but kept
/// private here so `ManipulateSpec` can be serialized without pulling in
/// an extra dependency.
fn json_escape_manipulate(s: &str) -> String {
  let mut out = String::with_capacity(s.len() + 16);
  for ch in s.chars() {
    match ch {
      '"' => out.push_str("\\\""),
      '\\' => out.push_str("\\\\"),
      '\n' => out.push_str("\\n"),
      '\r' => out.push_str("\\r"),
      '\t' => out.push_str("\\t"),
      c if (c as u32) < 0x20 => {
        out.push_str(&format!("\\u{:04x}", c as u32));
      }
      c => out.push(c),
    }
  }
  out
}

/// Serialize the styled runs of a label as a JSON array, e.g.
/// `[{"text":"m","italic":true},{"text":"₁","italic":false}]`. The playground
/// renders each run as a span, applying italic where flagged.
fn label_runs_to_json(runs: &[LabelRun]) -> String {
  let parts: Vec<String> = runs
    .iter()
    .map(|r| {
      let color = match r.color {
        Some((cr, cg, cb)) => format!(
          r#","color":"rgb({},{},{})""#,
          (cr.clamp(0.0, 1.0) * 255.0).round() as u8,
          (cg.clamp(0.0, 1.0) * 255.0).round() as u8,
          (cb.clamp(0.0, 1.0) * 255.0).round() as u8,
        ),
        None => String::new(),
      };
      format!(
        r#"{{"text":"{}","italic":{},"bold":{}{}}}"#,
        json_escape_manipulate(&r.text),
        r.italic,
        r.bold,
        color,
      )
    })
    .collect();
  format!("[{}]", parts.join(","))
}

/// Serialize a `ManipulateSpec` to a JSON object string (no surrounding
/// braces for an output-item wrapper — the caller adds `"type":"manipulate"`
/// etc. around it).
pub fn manipulate_spec_to_json(spec: &ManipulateSpec) -> String {
  let mut ctrl_parts: Vec<String> = Vec::with_capacity(spec.controls.len());
  for c in &spec.controls {
    match c {
      ManipulateControl::Continuous {
        name,
        min,
        max,
        step,
        initial,
        label,
        label_runs,
        ..
      } => {
        let step_json = match step {
          Some(s) => format!(r#","step":{s}"#),
          None => String::new(),
        };
        ctrl_parts.push(format!(
          r#"{{"kind":"continuous","name":"{}","label":"{}","labelRuns":{},"min":{},"max":{},"initial":{}{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
          min,
          max,
          initial,
          step_json,
        ));
      }
      ManipulateControl::Discrete {
        name,
        values,
        value_labels,
        value_label_svgs,
        initial_index,
        label,
        label_runs,
        popup,
        setter_bar,
        slider,
        vertical,
      } => {
        let value_parts: Vec<String> = values
          .iter()
          .map(|v| format!(r#""{}""#, json_escape_manipulate(v)))
          .collect();
        let label_parts: Vec<String> = value_labels
          .iter()
          .map(|v| format!(r#""{}""#, json_escape_manipulate(v)))
          .collect();
        let popup_json = if *popup { r#","popup":true"# } else { "" };
        let setter_bar_json = if *setter_bar {
          r#","setterBar":true"#
        } else {
          ""
        };
        let slider_json = if *slider { r#","slider":true"# } else { "" };
        let vertical_json = if *vertical { r#","vertical":true"# } else { "" };
        // Icon labels (rule right sides that are graphics) ride along as
        // rendered SVG, parallel to `values`; omitted when all-text.
        let svg_json = if value_label_svgs.iter().any(Option::is_some) {
          let svg_parts: Vec<String> = value_label_svgs
            .iter()
            .map(|s| match s {
              Some(svg) => {
                format!(r#""{}""#, json_escape_manipulate(svg))
              }
              None => "null".to_string(),
            })
            .collect();
          format!(r#","valueLabelSvgs":[{}]"#, svg_parts.join(","))
        } else {
          String::new()
        };
        ctrl_parts.push(format!(
          r#"{{"kind":"discrete","name":"{}","label":"{}","labelRuns":{},"values":[{}],"valueLabels":[{}],"initialIndex":{}{}{}{}{}{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
          value_parts.join(","),
          label_parts.join(","),
          initial_index,
          popup_json,
          setter_bar_json,
          slider_json,
          vertical_json,
          svg_json,
        ));
      }
      ManipulateControl::Slider2D {
        name,
        x_min,
        x_max,
        y_min,
        y_max,
        x_initial,
        y_initial,
        label,
        ..
      } => {
        ctrl_parts.push(format!(
          r#"{{"kind":"slider2d","name":"{}","label":"{}","xMin":{},"xMax":{},"yMin":{},"yMax":{},"xInit":{},"yInit":{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          x_min,
          x_max,
          y_min,
          y_max,
          x_initial,
          y_initial,
        ));
      }
      ManipulateControl::IntervalSlider {
        name,
        min,
        max,
        step,
        low_initial,
        high_initial,
        label,
      } => {
        let step_json = match step {
          Some(s) => format!(r#","step":{s}"#),
          None => String::new(),
        };
        ctrl_parts.push(format!(
          r#"{{"kind":"interval","name":"{}","label":"{}","min":{},"max":{},"lowInit":{},"highInit":{}{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          min,
          max,
          low_initial,
          high_initial,
          step_json,
        ));
      }
      ManipulateControl::Locator {
        name,
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        auto_create,
        label,
      } => {
        let point_parts: Vec<String> =
          points.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
        ctrl_parts.push(format!(
          r#"{{"kind":"locator","name":"{}","label":"{}","xMin":{},"xMax":{},"yMin":{},"yMax":{},"points":[{}],"autoCreate":{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          x_min,
          x_max,
          y_min,
          y_max,
          point_parts.join(","),
          auto_create,
        ));
      }
      ManipulateControl::Trigger {
        name,
        min,
        max,
        step,
        initial,
        running,
        label,
        label_runs,
      } => {
        // An infinite sweep end (`{time, 0, Infinity, 1}`) serializes as
        // `null` — JSON has no Infinity literal.
        let max_json = if max.is_finite() {
          format!("{max}")
        } else {
          "null".to_string()
        };
        ctrl_parts.push(format!(
          r#"{{"kind":"trigger","name":"{}","label":"{}","labelRuns":{},"min":{},"max":{},"step":{},"initial":{},"running":{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
          min,
          max_json,
          step,
          initial,
          running,
        ));
      }
      ManipulateControl::Button {
        label,
        label_runs,
        action,
      } => {
        ctrl_parts.push(format!(
          r#"{{"kind":"button","label":"{}","labelRuns":{},"action":"{}"}}"#,
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
          json_escape_manipulate(action),
        ));
      }
      ManipulateControl::Heading { label, label_runs } => {
        ctrl_parts.push(format!(
          r#"{{"kind":"heading","label":"{}","labelRuns":{}}}"#,
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
        ));
      }
      ManipulateControl::Divider => {
        ctrl_parts.push(r#"{"kind":"delimiter"}"#.to_string());
      }
    }
  }

  // Inject each control's `Enabled` condition (when present) into its JSON
  // object so the frontend can re-evaluate it and grey the control out.
  for (c, part) in spec.controls.iter().zip(ctrl_parts.iter_mut()) {
    if let Some((_, cond)) =
      spec.control_enabled.iter().find(|(n, _)| n == c.name())
      && part.ends_with('}')
    {
      let field =
        format!(r#","enabledWhen":"{}""#, json_escape_manipulate(cond));
      part.truncate(part.len() - 1);
      part.push_str(&field);
      part.push('}');
    }
    // A control belonging to a `PaneSelector` pane rides along with the
    // condition under which its pane is on screen, so the frontend can hide
    // the rows of the panes the selector is not showing.
    if let Some((_, cond)) =
      spec.control_visible.iter().find(|(n, _)| n == c.name())
      && part.ends_with('}')
    {
      let field =
        format!(r#","visibleWhen":"{}""#, json_escape_manipulate(cond));
      part.truncate(part.len() - 1);
      part.push_str(&field);
      part.push('}');
    }
    // Dynamic bounds (a slider range following another control's variable)
    // ride along as code fragments for the frontend to re-resolve.
    if let Some((_, min_code, max_code)) =
      spec.dynamic_bounds.iter().find(|(n, _, _)| n == c.name())
      && part.ends_with('}')
    {
      let mut field = String::new();
      if let Some(code) = min_code {
        field.push_str(&format!(
          r#","minCode":"{}""#,
          json_escape_manipulate(code)
        ));
      }
      if let Some(code) = max_code {
        field.push_str(&format!(
          r#","maxCode":"{}""#,
          json_escape_manipulate(code)
        ));
      }
      part.truncate(part.len() - 1);
      part.push_str(&field);
      part.push('}');
    }
    // A choice list that follows another control rides along the same way.
    if let Some((_, code)) =
      spec.dynamic_values.iter().find(|(n, _)| n == c.name())
      && part.ends_with('}')
    {
      let field =
        format!(r#","valuesCode":"{}""#, json_escape_manipulate(code));
      part.truncate(part.len() - 1);
      part.push_str(&field);
      part.push('}');
    }
  }

  let state_parts: Vec<String> = spec
    .state
    .iter()
    .map(|(name, value)| {
      format!(
        r#""{}":"{}""#,
        json_escape_manipulate(name),
        json_escape_manipulate(value),
      )
    })
    .collect();
  let display_parts: Vec<String> = spec
    .displays
    .iter()
    .map(|d| format!(r#""{}""#, json_escape_manipulate(d)))
    .collect();

  let animated_json = if spec.animated {
    if spec.animation_running {
      r#","animated":true"#
    } else {
      // Animated but built paused (`AnimationRunning -> False`).
      r#","animated":true,"animationRunning":false"#
    }
  } else {
    ""
  };
  let appearance_json = if spec.appearance_none {
    r#","appearanceNone":true"#
  } else {
    ""
  };
  let animation_var_json = match &spec.animation_var {
    Some(var) => {
      format!(r#","animationVar":"{}""#, json_escape_manipulate(var))
    }
    None => String::new(),
  };
  // Absent when every variable is tracked (the default), so a frontend that
  // does not know the option keeps re-rendering on every change.
  let tracked_json = match &spec.tracked_symbols {
    Some(names) => {
      let parts: Vec<String> = names
        .iter()
        .map(|n| format!(r#""{}""#, json_escape_manipulate(n)))
        .collect();
      format!(r#","trackedSymbols":[{}]"#, parts.join(","))
    }
    None => String::new(),
  };

  format!(
    r#""body":"{}","controls":[{}],"state":{{{}}},"displays":[{}]{}{}{}{}"#,
    json_escape_manipulate(&spec.body_code),
    ctrl_parts.join(","),
    state_parts.join(","),
    display_parts.join(","),
    animated_json,
    appearance_json,
    animation_var_json,
    tracked_json,
  )
}

/// A node in a rendered Manipulate extra-display widget tree. Both frontends
/// consume this: the Playground via `render_manipulate_display` (JSON), the
/// Studio via `build_manipulate_display` (this enum directly).
#[derive(Debug, Clone)]
pub enum DisplayNode {
  /// A framed container wrapping a single child (`Panel`, `Framed`, …).
  Panel(Box<Self>),
  /// A 2D grid of cells (`Grid`).
  Grid(Vec<Vec<Self>>),
  /// A vertical stack (`Column`, or a bare list).
  Column(Vec<Self>),
  /// A horizontal stack (`Row`).
  Row(Vec<Self>),
  /// A checkbox. `target` is the InputForm of the write-back lvalue (e.g.
  /// `data[[3, 5]]`), `None` for a non-interactive checkbox; `checked` is its
  /// current state; `on`/`off` are the InputForm values a toggle writes back.
  Checkbox {
    target: Option<String>,
    checked: bool,
    on: String,
    off: String,
  },
  /// One choice of a `TogglerBar[Dynamic[var], …]`: a toggle button that
  /// adds `value` to (or removes it from) the list variable. `mutation` is
  /// the ready-to-evaluate write-back assignment; `selected` is whether the
  /// value is currently a member of the list.
  Toggler {
    label: Box<Self>,
    mutation: String,
    selected: bool,
  },
  /// A `Button[label, action]`: pressing it evaluates `action` (InputForm)
  /// against the live bindings, exactly like a `Button` written as a
  /// Manipulate control argument. Demonstrations use these inside a
  /// `Dynamic[…]` caption to step a variable (`n++`, `n = 1`, …).
  Button { label: Box<Self>, action: String },
  /// A `Spacer[w]`: `w` printer's points of horizontal space.
  Spacer { width: f64 },
  /// A text leaf with its styled runs, so `Style["…", Bold, Red]` renders
  /// bold and red rather than as the literal `Style[…]` source.
  Text { runs: Vec<LabelRun> },
  /// Any unrecognized leaf, rendered to SVG (graphics) or text.
  Static { svg: Option<String>, text: String },
}

/// Render one extra-display expression (its InputForm in `display_code`) in
/// the scope of the current variable `bindings` into a JSON widget tree the
/// Playground can lay out and wire up interactively.
pub fn render_manipulate_display(
  display_code: &str,
  bindings: &[(String, String)],
) -> String {
  display_node_to_json(&build_manipulate_display(display_code, bindings))
}

/// Render one extra-display expression into a native `DisplayNode` tree.
///
/// Every checkbox's current on/off state is read in a *single* batched
/// evaluation rather than one interpreter call per cell — the difference
/// between 1 and (rows × cols) `Block` evaluations for a large grid, which is
/// what keeps a toggle responsive.
pub fn build_manipulate_display(
  display_code: &str,
  bindings: &[(String, String)],
) -> DisplayNode {
  let Ok(expr) = crate::interpret_to_expr(display_code) else {
    return DisplayNode::Static {
      svg: None,
      text: String::new(),
    };
  };

  // First pass: build the tree, collecting each checkbox's value-probe
  // expression (deferring `checked`). `Dynamic` layout wrappers still expand
  // eagerly (one call each), but the many leaf reads are collected here.
  let mut probes: Vec<String> = Vec::new();
  let mut ons: Vec<String> = Vec::new();
  let mut tree = display_expr_to_node(&expr, bindings, &mut probes, &mut ons);

  // Second pass: evaluate all probes at once, then fill in `checked`.
  if !probes.is_empty() {
    let list_code = format!("{{{}}}", probes.join(", "));
    let flags: Vec<bool> = match eval_display_in_scope_str(&list_code, bindings)
    {
      Some(Expr::List(ref vals)) if vals.len() == ons.len() => vals
        .iter()
        .zip(ons.iter())
        .map(|(v, on)| crate::syntax::expr_to_input_form(v) == *on)
        .collect(),
      _ => vec![false; probes.len()],
    };
    let mut idx = 0;
    assign_checkbox_state(&mut tree, &flags, &mut idx);
  }
  tree
}

/// Evaluate `expr` inside `Block[{bindings}, expr]` and return the resulting
/// expression. Used to release a held `Dynamic[…]`, expanding the layout it
/// wraps while inner `Dynamic[lval]` stays held.
fn eval_display_in_scope(
  expr: &Expr,
  bindings: &[(String, String)],
) -> Option<Expr> {
  eval_display_in_scope_str(&crate::syntax::expr_to_input_form(expr), bindings)
}

/// Like `eval_display_in_scope` but takes the InputForm code directly (used
/// for the batched checkbox-value probe list).
fn eval_display_in_scope_str(
  code: &str,
  bindings: &[(String, String)],
) -> Option<Expr> {
  crate::interpret_to_expr(&manipulate_block_code(code, bindings)).ok()
}

/// Recursively convert a display expression into a `DisplayNode`. Each
/// checkbox pushes its value-probe expression to `probes` (and the "on" value
/// to `ons`) instead of evaluating it inline; `checked` is filled in later
/// from a single batched evaluation.
fn display_expr_to_node(
  expr: &Expr,
  bindings: &[(String, String)],
  probes: &mut Vec<String>,
  ons: &mut Vec<String>,
) -> DisplayNode {
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // `Dynamic` is HoldFirst, so its content arrives unexpanded. Release
      // the hold under the current bindings so the layout (Grid/Outer/…)
      // expands, while any nested `Dynamic[lval]` stays held (keeping its
      // write-back target). Then render the expanded content.
      "Dynamic" if !args.is_empty() => {
        match eval_display_in_scope(&args[0], bindings) {
          Some(inner) => display_expr_to_node(&inner, bindings, probes, ons),
          None => static_leaf_node(expr, bindings),
        }
      }
      "Panel" | "Framed" | "Deploy" | "Item" | "Pane" | "Labeled"
        if !args.is_empty() =>
      {
        DisplayNode::Panel(Box::new(display_expr_to_node(
          &args[0], bindings, probes, ons,
        )))
      }
      "Grid" | "GridBox" | "TableForm" if !args.is_empty() => match &args[0] {
        Expr::List(rows) => DisplayNode::Grid(
          rows
            .iter()
            .map(|row| match row {
              Expr::List(cs) => cs
                .iter()
                .map(|c| display_expr_to_node(c, bindings, probes, ons))
                .collect(),
              other => {
                vec![display_expr_to_node(other, bindings, probes, ons)]
              }
            })
            .collect(),
        ),
        _ => static_leaf_node(expr, bindings),
      },
      "Column" if !args.is_empty() => {
        DisplayNode::Column(list_children(&args[0], bindings, probes, ons))
      }
      "Row" if !args.is_empty() => {
        DisplayNode::Row(list_children(&args[0], bindings, probes, ons))
      }
      "Checkbox" => checkbox_node(args, probes, ons),
      // `Button[label, action]` — the action is held, so its source is
      // taken verbatim and evaluated only when the button is pressed.
      "Button" if args.len() >= 2 => DisplayNode::Button {
        label: Box::new(display_expr_to_node(&args[0], bindings, probes, ons)),
        action: crate::syntax::expr_to_input_form(&args[1]),
      },
      "Spacer" if !args.is_empty() => DisplayNode::Spacer {
        width: spacer_width(&args[0]),
      },
      // A styled caption fragment: rendered as rich text, not as source.
      "Style" | "StyleForm" if !args.is_empty() => {
        styled_text_node(expr, bindings)
      }
      // `TogglerBar[Dynamic[var], {v1 -> label1, …}]`: a row of toggle
      // buttons; clicking one adds/removes its value from the list `var`.
      "TogglerBar" if args.len() >= 2 => {
        match togglerbar_node(args, bindings, probes, ons) {
          Some(node) => node,
          None => static_leaf_node(expr, bindings),
        }
      }
      // `PaneSelector[{v1 -> content1, v2 -> content2, …}, sel]` used as a
      // caption/heading row (e.g. a "set the isothermal temperature" label
      // that swaps to "choose a nonisothermal temperature profile" as a
      // toggle flips): evaluate `sel` against the live bindings and render
      // only the matching pane's content, like the Wolfram front end does,
      // instead of falling through to the raw source text.
      "PaneSelector" if args.len() >= 2 => {
        match pane_selector_content(args, bindings) {
          Some(content) => display_expr_to_node(content, bindings, probes, ons),
          None => DisplayNode::Column(Vec::new()),
        }
      }
      _ => static_leaf_node(expr, bindings),
    },
    // A bare list of display elements stacks vertically, like `Column`.
    Expr::List(_) => {
      DisplayNode::Column(list_children(expr, bindings, probes, ons))
    }
    // Literal prose in a caption row.
    Expr::String(_) => styled_text_node(expr, bindings),
    // Releasing a `Dynamic[…]` hold (above) can fully evaluate its content
    // into an already-rendered graphic rather than a layout container (e.g.
    // `Dynamic[GraphicsRow[…]]` used directly as a Specifications entry, a
    // common Demonstrations idiom for a live preview beside the sliders).
    // Route the SVG straight through instead of falling into the generic
    // leaf path below, which round-trips through `InputForm` text — and
    // `Expr::Graphics` has no source form, only the `-Graphics-` /
    // `-Graphics3D-` display placeholder, which does not re-parse.
    Expr::Graphics { svg, .. } => DisplayNode::Static {
      svg: Some(svg.clone()),
      text: String::new(),
    },
    _ => static_leaf_node(expr, bindings),
  }
}

/// `Spacer[w]` / `Spacer[{w, h}]` — the horizontal size it reserves, in
/// printer's points. Anything unreadable falls back to Wolfram's default.
fn spacer_width(arg: &Expr) -> f64 {
  match arg {
    Expr::List(items) if !items.is_empty() => {
      expr_to_f64(&items[0]).unwrap_or(0.0)
    }
    other => expr_to_f64(other).unwrap_or(0.0),
  }
}

/// A text leaf rendered through the label machinery, so `Style[…, Bold,
/// Red]` keeps its styling and a variable inside it shows its current
/// value. Falls back to the generic leaf when the fragment does not
/// evaluate (e.g. it is really a graphic).
fn styled_text_node(expr: &Expr, bindings: &[(String, String)]) -> DisplayNode {
  match eval_display_in_scope(expr, bindings) {
    Some(evaluated) => DisplayNode::Text {
      runs: manipulate_label_runs(&evaluated, false),
    },
    None => static_leaf_node(expr, bindings),
  }
}

/// The content of the pane a `PaneSelector[{v1 -> content1, …}, sel]`
/// display element is currently showing: `sel` is evaluated against the
/// live bindings and matched against each rule's (unevaluated) pattern by
/// its InputForm text, mirroring the equality test the Wolfram front end
/// performs. `None` when the panes list is malformed or no rule matches
/// (Wolfram shows nothing in that case, absent a `Default` option).
fn pane_selector_content<'a>(
  args: &'a [Expr],
  bindings: &[(String, String)],
) -> Option<&'a Expr> {
  let Expr::List(panes) = args.first()? else {
    return None;
  };
  let selector = eval_display_in_scope(&args[1], bindings).map_or_else(
    || crate::syntax::expr_to_input_form(&args[1]),
    |e| crate::syntax::expr_to_input_form(&e),
  );
  panes.iter().find_map(|pane| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = pane
    else {
      return None;
    };
    (crate::syntax::expr_to_input_form(pattern) == selector)
      .then_some(replacement.as_ref())
  })
}

/// Render the children of a `Column[{…}]` / `Row[{…}]` (or a bare list).
fn list_children(
  list: &Expr,
  bindings: &[(String, String)],
  probes: &mut Vec<String>,
  ons: &mut Vec<String>,
) -> Vec<DisplayNode> {
  match list {
    Expr::List(items) => items
      .iter()
      .map(|c| display_expr_to_node(c, bindings, probes, ons))
      .collect(),
    other => vec![display_expr_to_node(other, bindings, probes, ons)],
  }
}

/// Build a `Checkbox[…]` leaf node. An interactive checkbox is
/// `Checkbox[Dynamic[lval], {off, on}]` (the value list defaults to
/// `{False, True}`); its `target` is the InputForm of `lval`, its `checked`
/// state is `lval == on` under the current bindings, and `on`/`off` are the
/// values a toggle writes back. A non-`Dynamic` `Checkbox[val, …]` renders
/// the same but non-interactively (no `target`).
///
/// The value that decides `checked` is not evaluated here — its InputForm is
/// pushed onto `probes` (with the matching `on` value onto `ons`) so the
/// caller can evaluate every checkbox in one batched call. The returned node
/// carries a provisional `checked = false`, patched afterwards.
fn checkbox_node(
  args: &[Expr],
  probes: &mut Vec<String>,
  ons: &mut Vec<String>,
) -> DisplayNode {
  // Extract the {off, on} value pair (InputForm), defaulting to False/True.
  let (off, on) = match args.get(1) {
    Some(Expr::List(vs)) if vs.len() == 2 => (
      crate::syntax::expr_to_input_form(&vs[0]),
      crate::syntax::expr_to_input_form(&vs[1]),
    ),
    _ => ("False".to_string(), "True".to_string()),
  };

  // A `Dynamic[lval]` first argument is an interactive, write-back target.
  let dynamic_lval = match args.first() {
    Some(Expr::FunctionCall { name, args: dargs })
      if name == "Dynamic" && !dargs.is_empty() =>
    {
      Some(&dargs[0])
    }
    _ => None,
  };

  // The expression whose value determines `checked`: the held lvalue for an
  // interactive checkbox, or the (static) first argument otherwise.
  let probe = match dynamic_lval.or_else(|| args.first()) {
    Some(e) => crate::syntax::expr_to_input_form(e),
    None => "False".to_string(),
  };
  probes.push(probe);
  ons.push(on.clone());

  DisplayNode::Checkbox {
    checked: false,
    target: dynamic_lval.map(crate::syntax::expr_to_input_form),
    on,
    off,
  }
}

/// Build a `TogglerBar[Dynamic[var], choices]` display: a Row of Toggler
/// buttons. Each choice is `value -> label` (or a plain value, labelled by
/// itself); clicking a button toggles the value's membership in the list
/// `var`. Returns `None` when the arguments don't have that shape (the
/// caller falls back to a static rendering).
fn togglerbar_node(
  args: &[Expr],
  bindings: &[(String, String)],
  probes: &mut Vec<String>,
  ons: &mut Vec<String>,
) -> Option<DisplayNode> {
  let var = match args.first() {
    Some(Expr::FunctionCall { name, args: dargs })
      if name == "Dynamic" && !dargs.is_empty() =>
    {
      match &dargs[0] {
        Expr::Identifier(v) => v.clone(),
        _ => return None,
      }
    }
    _ => return None,
  };
  // The choice list may be held (e.g. `Thread[Range[1, 4] -> {…}]`).
  let choices_expr = match &args[1] {
    l @ Expr::List(_) => l.clone(),
    other => crate::evaluator::evaluate_expr_to_expr(other).ok()?,
  };
  let Expr::List(choices) = &choices_expr else {
    return None;
  };
  // The current selection, for the per-choice `selected` state.
  let current =
    crate::evaluator::evaluate_expr_to_expr(&Expr::Identifier(var.clone()))
      .ok();
  let mut buttons = Vec::with_capacity(choices.len());
  for choice in choices {
    let (value, label) = match choice {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      other => (other, other),
    };
    let value_code = crate::syntax::expr_to_input_form(value);
    let selected = match &current {
      Some(Expr::List(items)) => items
        .iter()
        .any(|it| crate::syntax::expr_to_input_form(it) == value_code),
      Some(single) => crate::syntax::expr_to_input_form(single) == value_code,
      None => false,
    };
    let mutation = format!(
      "{var} = If[MemberQ[{var}, {value_code}], DeleteCases[{var}, \
       {value_code}], Append[{var}, {value_code}]]"
    );
    buttons.push(DisplayNode::Toggler {
      label: Box::new(display_expr_to_node(label, bindings, probes, ons)),
      mutation,
      selected,
    });
  }
  // A trailing `Appearance -> "Vertical"` (added by the CheckboxBar/TogglerBar
  // branch of `parse_manipulate_control`) stacks the toggles in a column
  // instead of Wolfram's default horizontal bar.
  let vertical = args[2..].iter().any(|it| {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = it
    else {
      return false;
    };
    matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Appearance")
      && (matches!(replacement.as_ref(), Expr::Identifier(s) if s == "Vertical")
        || matches!(replacement.as_ref(), Expr::String(s) if s == "Vertical"))
  });
  Some(if vertical {
    DisplayNode::Column(buttons)
  } else {
    DisplayNode::Row(buttons)
  })
}

/// Fill in each checkbox's `checked` flag from the batched probe results, in
/// the same pre-order the probes were collected.
fn assign_checkbox_state(
  node: &mut DisplayNode,
  flags: &[bool],
  idx: &mut usize,
) {
  match node {
    DisplayNode::Panel(child) => assign_checkbox_state(child, flags, idx),
    DisplayNode::Grid(rows) => {
      for row in rows {
        for cell in row {
          assign_checkbox_state(cell, flags, idx);
        }
      }
    }
    DisplayNode::Column(children) | DisplayNode::Row(children) => {
      for c in children {
        assign_checkbox_state(c, flags, idx);
      }
    }
    DisplayNode::Toggler { label, .. } | DisplayNode::Button { label, .. } => {
      assign_checkbox_state(label, flags, idx);
    }
    DisplayNode::Checkbox { checked, .. } => {
      if let Some(f) = flags.get(*idx) {
        *checked = *f;
      }
      *idx += 1;
    }
    DisplayNode::Spacer { .. }
    | DisplayNode::Text { .. }
    | DisplayNode::Static { .. } => {}
  }
}

/// Render an unrecognized display leaf by evaluating it in scope and
/// capturing its SVG (graphics) or text output.
fn static_leaf_node(expr: &Expr, bindings: &[(String, String)]) -> DisplayNode {
  let code =
    manipulate_block_code(&crate::syntax::expr_to_input_form(expr), bindings);
  match crate::interpret_with_stdout(&code) {
    Ok(r) => {
      if let Some(svg) = r.graphics {
        DisplayNode::Static {
          svg: Some(svg),
          text: String::new(),
        }
      } else {
        let text = r
          .result
          .replace("-Graphics-", "")
          .replace("-Graphics3D-", "")
          .replace("-Image-", "");
        DisplayNode::Static {
          svg: None,
          text: text.trim().to_string(),
        }
      }
    }
    Err(_) => DisplayNode::Static {
      svg: None,
      text: String::new(),
    },
  }
}

/// Serialize a `DisplayNode` tree to the JSON the Playground consumes.
fn display_node_to_json(node: &DisplayNode) -> String {
  match node {
    DisplayNode::Panel(child) => {
      format!(
        r#"{{"kind":"panel","child":{}}}"#,
        display_node_to_json(child)
      )
    }
    DisplayNode::Grid(rows) => {
      let row_json: Vec<String> = rows
        .iter()
        .map(|row| {
          let cells: Vec<String> =
            row.iter().map(display_node_to_json).collect();
          format!("[{}]", cells.join(","))
        })
        .collect();
      format!(r#"{{"kind":"grid","rows":[{}]}}"#, row_json.join(","))
    }
    DisplayNode::Column(children) => {
      let cs: Vec<String> = children.iter().map(display_node_to_json).collect();
      format!(r#"{{"kind":"column","children":[{}]}}"#, cs.join(","))
    }
    DisplayNode::Row(children) => {
      let cs: Vec<String> = children.iter().map(display_node_to_json).collect();
      format!(r#"{{"kind":"row","children":[{}]}}"#, cs.join(","))
    }
    DisplayNode::Checkbox {
      target,
      checked,
      on,
      off,
    } => match target {
      Some(t) => format!(
        r#"{{"kind":"checkbox","target":"{}","checked":{},"on":"{}","off":"{}"}}"#,
        json_escape_manipulate(t),
        checked,
        json_escape_manipulate(on),
        json_escape_manipulate(off),
      ),
      None => format!(
        r#"{{"kind":"checkbox","checked":{},"on":"{}","off":"{}"}}"#,
        checked,
        json_escape_manipulate(on),
        json_escape_manipulate(off),
      ),
    },
    DisplayNode::Toggler {
      label,
      mutation,
      selected,
    } => format!(
      r#"{{"kind":"toggler","label":{},"mutation":"{}","selected":{}}}"#,
      display_node_to_json(label),
      json_escape_manipulate(mutation),
      selected,
    ),
    DisplayNode::Button { label, action } => format!(
      r#"{{"kind":"button","label":{},"action":"{}"}}"#,
      display_node_to_json(label),
      json_escape_manipulate(action),
    ),
    DisplayNode::Spacer { width } => {
      format!(r#"{{"kind":"spacer","width":{width}}}"#)
    }
    DisplayNode::Text { runs } => {
      format!(r#"{{"kind":"text","runs":{}}}"#, label_runs_to_json(runs))
    }
    DisplayNode::Static { svg, text } => match svg {
      Some(svg) => format!(
        r#"{{"kind":"static","svg":"{}"}}"#,
        json_escape_manipulate(svg)
      ),
      None => format!(
        r#"{{"kind":"static","text":"{}"}}"#,
        json_escape_manipulate(text)
      ),
    },
  }
}

// ─── HilbertCurve / PeanoCurve ─────────────────────────────────────

/// Shared plumbing for the integer-grid space-filling curves: parses the
/// order (::intpm for anything but a positive machine integer) and an
/// optional DataRange -> {{xmin, xmax}, {ymin, ymax}} rule that affinely
/// maps the grid to real coordinates, then wraps the points in Line[…].
fn space_filling_curve(
  name: &str,
  args: &[Expr],
  side: i64,
  points: &[(i64, i64)],
) -> Expr {
  let mut range: Option<[(f64, f64); 2]> = None;
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "DataRange")
      && let Expr::List(pair) = replacement.as_ref()
      && pair.len() == 2
    {
      let parse_pair = |e: &Expr| -> Option<(f64, f64)> {
        if let Expr::List(mm) = e
          && mm.len() == 2
        {
          Some((
            crate::functions::graphics::expr_to_f64(&mm[0])?,
            crate::functions::graphics::expr_to_f64(&mm[1])?,
          ))
        } else {
          None
        }
      };
      if let (Some(xr), Some(yr)) = (parse_pair(&pair[0]), parse_pair(&pair[1]))
      {
        range = Some([xr, yr]);
      }
    }
  }
  let point_exprs: Vec<Expr> = points
    .iter()
    .map(|&(x, y)| match &range {
      Some([xr, yr]) => {
        let denom = (side - 1).max(1) as f64;
        Expr::List(
          vec![
            Expr::Real(xr.0 + x as f64 * (xr.1 - xr.0) / denom),
            Expr::Real(yr.0 + y as f64 * (yr.1 - yr.0) / denom),
          ]
          .into(),
        )
      }
      None => Expr::List(
        vec![Expr::Integer(x as i128), Expr::Integer(y as i128)].into(),
      ),
    })
    .collect();
  let _ = name;
  call1("Line", Expr::List(point_exprs.into()))
}

/// The curve order, or None (after emitting ::intpm) for invalid input.
fn curve_order(name: &str, args: &[Expr], max_n: i128) -> Option<i128> {
  match args.first() {
    Some(Expr::Integer(n)) if *n >= 1 && *n <= max_n => Some(*n),
    Some(other) => {
      crate::emit_message(&format!(
        "{}::intpm: Positive machine-sized integer expected at position 1 in {}[{}].",
        name,
        name,
        crate::syntax::expr_to_output(other)
      ));
      None
    }
    None => None,
  }
}

/// HilbertCurve[n] — one Line through all 4^n cells of the 2^n × 2^n grid
/// in Hilbert order (the classic table-driven index-to-coordinate walk,
/// which reproduces wolframscript's orientation exactly: order 1 runs
/// (0,0) → (0,1) → (1,1) → (1,0)).
pub fn hilbert_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("HilbertCurve", args));
  let Some(n) = curve_order("HilbertCurve", args, 10) else {
    return unevaluated();
  };
  let n = n as u32;
  let side = 1i64 << n;
  let total = 1u64 << (2 * n);
  let mut points = Vec::with_capacity(total as usize);
  for d in 0..total {
    let (mut x, mut y) = (0i64, 0i64);
    let mut t = d;
    let mut s = 1i64;
    while s < side {
      let rx = (1 & (t / 2)) as i64;
      let ry = (1 & (t ^ (rx as u64))) as i64;
      if ry == 0 {
        if rx == 1 {
          x = s - 1 - x;
          y = s - 1 - y;
        }
        std::mem::swap(&mut x, &mut y);
      }
      x += s * rx;
      y += s * ry;
      t /= 4;
      s *= 2;
    }
    points.push((x, y));
  }
  Ok(space_filling_curve("HilbertCurve", args, side, &points))
}

/// PeanoCurve[n] — one Line through all 9^n cells of the 3^n × 3^n grid in
/// Peano order. Coordinates come from Peano's digit construction: index
/// digits alternate y/x roles (y first), and each digit is complemented
/// (d → 2 - d) when the sum of the preceding other-coordinate digits is
/// odd — which reproduces wolframscript's serpentine orientation.
pub fn peano_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PeanoCurve", args));
  let Some(n) = curve_order("PeanoCurve", args, 7) else {
    return unevaluated();
  };
  let n = n as u32;
  let digits_len = (2 * n) as usize;
  let side = 3i64.pow(n);
  let total = 9u64.pow(n);
  let mut points = Vec::with_capacity(total as usize);
  let mut digits = vec![0u8; digits_len];
  for index in 0..total {
    let mut t = index;
    for slot in (0..digits_len).rev() {
      digits[slot] = (t % 3) as u8;
      t /= 3;
    }
    let (mut x, mut y) = (0i64, 0i64);
    for (i, &d) in digits.iter().enumerate() {
      let flip: u32 = digits[..i]
        .iter()
        .enumerate()
        .filter(|(j, _)| (j % 2) != (i % 2))
        .map(|(_, &v)| v as u32)
        .sum();
      let e = if flip % 2 == 1 {
        2 - d as i64
      } else {
        d as i64
      };
      if i % 2 == 0 {
        y = y * 3 + e;
      } else {
        x = x * 3 + e;
      }
    }
    points.push((x, y));
  }
  Ok(space_filling_curve("PeanoCurve", args, side, &points))
}

/// SierpinskiCurve[n] — the closed Sierpiński square curve as a Line,
/// generated by Wirth's classic four-procedure recursion
///   A: A↘B→D↗A   B: B↙C↓A↘B   C: C↖D←B↙C   D: D↗A↑C↖D
/// glued as A↘B↙C↖D↗ (closed), with a fixed half-step of 32 — diagonal
/// moves are (±32, ±32) and axis moves 64 — matching wolframscript's
/// absolute integer coordinates at every order.
pub fn sierpinski_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("SierpinskiCurve", args));
  let Some(n) = curve_order("SierpinskiCurve", args, 9) else {
    return unevaluated();
  };
  const H: i64 = 32;
  struct Gen {
    pos: (i64, i64),
    points: Vec<(i64, i64)>,
  }
  impl Gen {
    fn step(&mut self, dx: i64, dy: i64) {
      self.pos = (self.pos.0 + dx, self.pos.1 + dy);
      self.points.push(self.pos);
    }
    // Moves named after their compass direction on wolframscript's
    // y-downward-negative layout.
    fn se(&mut self) {
      self.step(H, -H);
    }
    fn sw(&mut self) {
      self.step(-H, -H);
    }
    fn ne(&mut self) {
      self.step(H, H);
    }
    fn nw(&mut self) {
      self.step(-H, H);
    }
    fn east(&mut self) {
      self.step(2 * H, 0);
    }
    fn west(&mut self) {
      self.step(-2 * H, 0);
    }
    fn north(&mut self) {
      self.step(0, 2 * H);
    }
    fn south(&mut self) {
      self.step(0, -2 * H);
    }
    fn a(&mut self, k: u32) {
      if k == 0 {
        return;
      }
      self.a(k - 1);
      self.se();
      self.b(k - 1);
      self.east();
      self.d(k - 1);
      self.ne();
      self.a(k - 1);
    }
    fn b(&mut self, k: u32) {
      if k == 0 {
        return;
      }
      self.b(k - 1);
      self.sw();
      self.c(k - 1);
      self.south();
      self.a(k - 1);
      self.se();
      self.b(k - 1);
    }
    fn c(&mut self, k: u32) {
      if k == 0 {
        return;
      }
      self.c(k - 1);
      self.nw();
      self.d(k - 1);
      self.west();
      self.b(k - 1);
      self.sw();
      self.c(k - 1);
    }
    fn d(&mut self, k: u32) {
      if k == 0 {
        return;
      }
      self.d(k - 1);
      self.ne();
      self.a(k - 1);
      self.north();
      self.c(k - 1);
      self.nw();
      self.d(k - 1);
    }
  }
  let mut g = Gen {
    pos: (0, 0),
    points: vec![(0, 0)],
  };
  let k = n as u32;
  g.a(k);
  g.se();
  g.b(k);
  g.sw();
  g.c(k);
  g.nw();
  g.d(k);
  g.ne();
  let point_exprs: Vec<Expr> = g
    .points
    .iter()
    .map(|&(x, y)| {
      Expr::List(
        vec![Expr::Integer(x as i128), Expr::Integer(y as i128)].into(),
      )
    })
    .collect();
  Ok(call1("Line", Expr::List(point_exprs.into())))
}

#[cfg(test)]
mod manipulate_label_tests {
  use super::*;

  fn runs(expr: &Expr) -> Vec<LabelRun> {
    manipulate_label_runs(expr, false)
  }

  fn run(text: &str, italic: bool) -> LabelRun {
    LabelRun {
      text: text.to_string(),
      italic,
      ..Default::default()
    }
  }

  #[test]
  fn style_italic_string_is_one_italic_run() {
    let label = call(
      "Style",
      vec![Expr::String("t".into()), Expr::Identifier("Italic".into())],
    );
    assert_eq!(runs(&label), vec![run("t", true)]);
  }

  #[test]
  fn style_fontslant_rule_is_italic() {
    let label = call(
      "Style",
      vec![
        Expr::String("t".into()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("FontSlant".into())),
          replacement: Box::new(Expr::String("Italic".into())),
        },
      ],
    );
    assert_eq!(runs(&label), vec![run("t", true)]);
  }

  #[test]
  fn text_subscript_style_renders_italic_base_and_upright_subscript() {
    // Text[Subscript[Style["m", Italic], 1]]  ->  italic "m", upright "₁"
    let styled = call(
      "Style",
      vec![Expr::String("m".into()), Expr::Identifier("Italic".into())],
    );
    let subscript = call("Subscript", vec![styled, Expr::Integer(1)]);
    let label = call1("Text", subscript);
    assert_eq!(runs(&label), vec![run("m", true), run("\u{2081}", false)]);
  }

  #[test]
  fn plain_identifier_passthrough() {
    let label = Expr::Identifier("\u{03B8}".into());
    assert_eq!(runs(&label), vec![run("\u{03B8}", false)]);
    assert_eq!(flatten_label_runs(&runs(&label)), "\u{03B8}");
  }

  #[test]
  fn superscript_renders_unicode() {
    let label = call(
      "Superscript",
      vec![Expr::Identifier("x".into()), Expr::Integer(2)],
    );
    assert_eq!(runs(&label), vec![run("x", false), run("\u{00B2}", false)]);
  }

  #[test]
  fn row_concatenates_parts_preserving_style() {
    let italic_a = call(
      "Style",
      vec![Expr::String("a".into()), Expr::Identifier("Italic".into())],
    );
    let row = call1(
      "Row",
      Expr::List(vec![italic_a, Expr::String("b".into())].into()),
    );
    assert_eq!(runs(&row), vec![run("a", true), run("b", false)]);
    assert_eq!(flatten_label_runs(&runs(&row)), "ab");
  }

  /// `Derivative[n][f]` — a slider labelled `y'(0)` in a Demonstration.
  fn derivative(order: i128, func: Expr) -> Expr {
    Expr::CurriedCall {
      func: Box::new(call1("Derivative", Expr::Integer(order))),
      args: vec![func],
    }
  }

  #[test]
  fn derivative_of_an_italic_style_primes_an_italic_base() {
    let italic_y = call(
      "Style",
      vec![Expr::String("y".into()), Expr::Identifier("Italic".into())],
    );
    let label = call1(
      "Text",
      call1(
        "Row",
        Expr::List(
          vec![derivative(1, italic_y), Expr::String("(0)".into())].into(),
        ),
      ),
    );
    assert_eq!(
      runs(&label),
      vec![run("y", true), run("\u{2032}", false), run("(0)", false),]
    );
    assert_eq!(flatten_label_runs(&runs(&label)), "y\u{2032}(0)");
  }

  #[test]
  fn higher_derivative_orders_get_their_own_marks() {
    let y = || Expr::Identifier("y".into());
    let marks = |order| flatten_label_runs(&runs(&derivative(order, y())));
    assert_eq!(marks(2), "y\u{2033}");
    assert_eq!(marks(3), "y\u{2034}");
    // Past three primes the order is written as a superscript in parens.
    assert_eq!(marks(4), "y\u{207D}\u{2074}\u{207E}");
  }

  /// The evaluator hands back `Derivative[n][f]` flattened to
  /// `Derivative[n, f]`; both shapes must label the same.
  #[test]
  fn flattened_derivative_labels_like_the_curried_one() {
    let flat = call(
      "Derivative",
      vec![Expr::Integer(1), Expr::Identifier("y".into())],
    );
    assert_eq!(flatten_label_runs(&runs(&flat)), "y\u{2032}");
  }
}

#[cfg(test)]
mod manipulate_dynamic_control_list_tests {
  use super::*;

  fn spec(code: &str) -> ManipulateSpec {
    let expr = crate::parse_to_expr(code).expect("parse");
    extract_manipulate_spec(&expr).expect("extract spec")
  }

  fn names(spec: &ManipulateSpec) -> Vec<&str> {
    spec.controls.iter().map(ManipulateControl::name).collect()
  }

  /// The whole control-spec list wrapped in `Dynamic[…]` (the
  /// Demonstrations idiom for a panel that reacts to another control)
  /// flattens like a plain `Column` of controls instead of being
  /// mistaken for a static display element.
  #[test]
  fn dynamic_wrapped_control_list_flattens_to_controls() {
    let s = spec("Manipulate[x, Dynamic[{Control[{{x, 0}, -1, 1}]}]]");
    assert_eq!(names(&s), vec!["x"]);
    assert!(s.displays.is_empty());
  }

  /// `Sequence@@If[cond, ctrlSpec, {}]` inside a Dynamic control list
  /// splices in the extra control when the condition — evaluated against
  /// the other controls' initial values — holds.
  #[test]
  fn sequence_apply_if_splices_control_when_condition_holds() {
    let s = spec(
      "Manipulate[x + y, Dynamic[{Control[{{mode, 1}, -1, 1}], \
       Sequence@@If[mode == 1, {Control[{{y, 0}, -1, 1}]}, {}]}]]",
    );
    assert_eq!(names(&s), vec!["mode", "y"]);
  }

  /// The same conditional control is omitted when its condition — the
  /// other control's initial value — does not hold.
  #[test]
  fn sequence_apply_if_omits_control_when_condition_fails() {
    let s = spec(
      "Manipulate[x + y, Dynamic[{Control[{{mode, 0}, -1, 1}], \
       Sequence@@If[mode == 1, {Control[{{y, 0}, -1, 1}]}, {}]}]]",
    );
    assert_eq!(names(&s), vec!["mode"]);
  }

  /// A discrete control's own *choice list* (not the whole control-spec
  /// list) can be written `Dynamic[expr, opts…]` — the shape a PopupMenu
  /// built from a lookup table uses so the front end can refresh it as
  /// other state changes (e.g. `Dynamic[# -> data[[#, 2]] & /@
  /// Range[Length[data]], SynchronousUpdating -> False]`). `Dynamic` is
  /// `HoldFirst` with no evaluation rule of its own, so the wrapper must be
  /// stripped before the wrapped expression is evaluated to a list of
  /// choices; previously the control silently failed to parse at all.
  #[test]
  fn dynamic_wrapped_choice_list_still_offers_its_options() {
    let _ = crate::interpret_with_stdout(
      "presetTable = {{1, \"a\"}, {2, \"b\"}, {3, \"c\"}};",
    );
    let s = spec(
      "Manipulate[pick, {{pick, 2, \"preset\"}, \
       Dynamic[#\
        -> presetTable[[#, 2]] & /@ Range[Length[presetTable]], \
        SynchronousUpdating -> False], ControlType -> PopupMenu}]",
    );
    assert_eq!(names(&s), vec!["pick"]);
    match &s.controls[0] {
      ManipulateControl::Discrete {
        values,
        value_labels,
        initial_index,
        popup,
        ..
      } => {
        assert_eq!(values, &["1", "2", "3"]);
        assert_eq!(value_labels, &["a", "b", "c"]);
        assert_eq!(*initial_index, 1);
        assert!(popup, "ControlType -> PopupMenu must render as a dropdown");
      }
      other => panic!("expected a Discrete control, got {other:?}"),
    }
  }
}

#[cfg(test)]
mod manipulate_display_pane_selector_tests {
  use super::*;

  /// A `PaneSelector[…]` caption row (the Demonstrations idiom for a label
  /// that swaps text as a toggle flips) renders only the pane matching the
  /// selector's current value, as styled text — not the raw
  /// `PaneSelector[…]` source.
  #[test]
  fn renders_matching_pane_as_text() {
    let code = r#"PaneSelector[{True -> Style["on", Bold], False -> Style["off", Bold]}, flag]"#;
    let on = build_manipulate_display(code, &[("flag".into(), "True".into())]);
    match on {
      DisplayNode::Text { runs } => {
        assert_eq!(flatten_label_runs(&runs), "on");
      }
      other => panic!("expected a text node, got {other:?}"),
    }

    let off =
      build_manipulate_display(code, &[("flag".into(), "False".into())]);
    match off {
      DisplayNode::Text { runs } => {
        assert_eq!(flatten_label_runs(&runs), "off");
      }
      other => panic!("expected a text node, got {other:?}"),
    }
  }

  /// A pane's content can itself be a layout container (the shelf-life
  /// Demonstration wraps its label in `Row[{Spacer[…], Style[…], …}]`);
  /// the selected pane recurses through the normal display machinery
  /// instead of being treated as an opaque leaf.
  #[test]
  fn renders_matching_pane_as_row() {
    let code = r#"PaneSelector[{1 -> Row[{Style["a"], Style["b"]}], 2 -> Style["c"]}, mode]"#;
    let node = build_manipulate_display(code, &[("mode".into(), "1".into())]);
    match node {
      DisplayNode::Row(children) => assert_eq!(children.len(), 2),
      other => panic!("expected a row node, got {other:?}"),
    }
  }

  /// No pane matches the selector's current value: nothing is shown,
  /// rather than the unevaluated `PaneSelector[…]` source leaking through.
  #[test]
  fn no_matching_pane_renders_nothing() {
    let code = r#"PaneSelector[{1 -> Style["a"], 2 -> Style["b"]}, mode]"#;
    let node = build_manipulate_display(code, &[("mode".into(), "3".into())]);
    match node {
      DisplayNode::Column(children) => assert!(children.is_empty()),
      other => panic!("expected an empty column, got {other:?}"),
    }
  }
}

#[cfg(test)]
mod manipulate_reversed_range_tests {
  use super::*;

  fn first_continuous(code: &str) -> (f64, f64, f64) {
    let expr = crate::parse_to_expr(code).expect("parse");
    let spec = extract_manipulate_spec(&expr).expect("extract spec");
    match spec.controls.first() {
      Some(ManipulateControl::Continuous {
        min, max, initial, ..
      }) => (*min, *max, *initial),
      other => panic!("expected a continuous control, got {other:?}"),
    }
  }

  /// `{u, umin, umax}` with `umin > umax` is a documented reversed-direction
  /// slider (Wolfram draws it counting down from left to right). The parsed
  /// `min`/`max` fields must still satisfy `min <= max` — every downstream
  /// consumer (the slider widget's `RangeInclusive`, dynamic-bounds
  /// re-resolution) assumes that order, and an unsorted pair makes the
  /// iced slider clamp the initial value to the wrong end of the range.
  #[test]
  fn reversed_bounds_are_sorted_for_the_control() {
    let (min, max, initial) = first_continuous(
      r#"Manipulate[u, {{u, -1.2, "angle"}, 0.02, -2.5, 0.02}]"#,
    );
    assert!(min <= max, "min ({min}) must not exceed max ({max})");
    assert_eq!(min, -2.5);
    assert_eq!(max, 0.02);
    // The initial value is still the one written in the spec, not clamped
    // to either sorted bound.
    assert_eq!(initial, -1.2);
  }

  /// The ordinary (already increasing) case is unaffected by the sort.
  #[test]
  fn forward_bounds_are_unchanged() {
    let (min, max, initial) =
      first_continuous(r#"Manipulate[u, {{u, 1.3, "radius"}, 0.01, 2}]"#);
    assert_eq!(min, 0.01);
    assert_eq!(max, 2.0);
    assert_eq!(initial, 1.3);
  }
}
