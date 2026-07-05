use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{DEFAULT_HEIGHT, DEFAULT_WIDTH, parse_image_size};
use crate::syntax::{Expr, expr_to_string};

/// Dash length for the "Small" named size in Dashing directives.
/// This is the default dash segment length used by Dashed, Dotted, etc.
const SMALL_DASH: f64 = 0.01;

/// Convert a named size (Tiny, Small, Medium, Large) to a dash length.
fn dash_size_to_f64(expr: &Expr) -> Option<f64> {
  if let Expr::Identifier(s) = expr {
    match s.as_str() {
      "Tiny" => Some(0.005),
      "Small" => Some(SMALL_DASH),
      "Medium" => Some(0.02),
      "Large" => Some(0.04),
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

  pub(crate) fn to_svg_rgb(&self) -> String {
    let r = (self.r.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (self.g.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (self.b.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("rgb({},{},{})", r, g, b)
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
  pub(crate) fn to_expr(&self) -> Expr {
    if (self.r - self.g).abs() < 1e-14 && (self.g - self.b).abs() < 1e-14 {
      Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![Expr::Real(self.r)].into(),
      }
    } else {
      Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(self.r), Expr::Real(self.g), Expr::Real(self.b)]
          .into(),
      }
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
  opacity: f64,
  thickness: f64,  // fraction of plot width, default ~0.004
  point_size: f64, // fraction of plot width, default ~0.012
  dashing: Option<Vec<f64>>, // dash lengths in coordinate-space fractions
  edge_form: Option<EdgeForm>,
  halo: Option<Halo>, // Haloing[...] contrasting outline behind primitives
  drop_shadow: Option<DropShadow>, // DropShadowing[...] shadow behind primitives
  font_size: f64,
  font_weight: String,
  font_style: String,
  font_family: String, // empty string means SVG default
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
  let s = format!("{:.2}", x);
  s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl DropShadow {
  /// Deterministic SVG filter id derived from the shadow parameters, so
  /// identical shadows share one `<defs>` entry without threading an id
  /// map through the render pipeline.
  pub(crate) fn filter_id(&self) -> String {
    fn enc(x: f64) -> String {
      format!("{:.2}", x).replace('-', "m").replace('.', "p")
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
      opacity: 1.0,
      thickness: 0.004,
      point_size: 0.012,
      dashing: None,
      edge_form: None,
      halo: None,
      drop_shadow: None,
      font_size: 14.0,
      font_weight: "normal".to_string(),
      font_style: "normal".to_string(),
      font_family: String::new(),
    }
  }
}

impl StyleState {
  fn effective_color(&self) -> Color {
    self.color.with_alpha(self.color.a * self.opacity)
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
    style: StyleState,
  },
  BezierCurvePrim {
    points: Vec<(f64, f64)>,
    style: StyleState,
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
}

impl Primitive {
  /// The style captured when the primitive was collected (None for
  /// Raster, which carries no style).
  fn style(&self) -> Option<&StyleState> {
    match self {
      Primitive::PointSingle { style, .. }
      | Primitive::PointMulti { style, .. }
      | Primitive::Line { style, .. }
      | Primitive::CircleArc { style, .. }
      | Primitive::Disk { style, .. }
      | Primitive::DiskSector { style, .. }
      | Primitive::RectPrim { style, .. }
      | Primitive::PolygonPrim { style, .. }
      | Primitive::ArrowPrim { style, .. }
      | Primitive::TextPrim { style, .. }
      | Primitive::BezierCurvePrim { style, .. }
      | Primitive::HalfPlanePrim { style, .. } => Some(style),
      Primitive::RasterPrim { .. } => None,
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
  let l = (max + min) / 2.0;
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
        if !args.is_empty() {
          let g = expr_to_f64(&args[0])?;
          let a = if args.len() >= 2 {
            expr_to_f64(&args[1]).unwrap_or(1.0)
          } else {
            1.0
          };
          Some(Color::new(g, g, g).with_alpha(a))
        } else {
          None
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
        // Handle named sizes: Thickness[Large] → same as Thick, etc.
        if let Expr::Identifier(s) = &args[0] {
          match s.as_str() {
            "Large" => style.thickness = -2.0, // AbsoluteThickness[2]
            "Tiny" => style.thickness = -0.5,  // AbsoluteThickness[0.5]
            _ => {
              if let Some(t) = expr_to_f64(&args[0]) {
                style.thickness = t;
              }
            }
          }
        } else if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = t;
        }
        true
      }
      "AbsoluteThickness" if args.len() == 1 => {
        // AbsoluteThickness gives pixel-level thickness
        // We'll store it as a negative number to distinguish from relative
        if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = -t; // negative = absolute pixels
        }
        true
      }
      "PointSize" if args.len() == 1 => {
        if let Some(s) = expr_to_f64(&args[0]) {
          style.point_size = s;
        }
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
              if n2 == "Thickness" && a2.len() == 1 {
                ef.thickness = expr_to_f64(&a2[0]);
              } else if n2 == "AbsoluteThickness" && a2.len() == 1 {
                ef.thickness = expr_to_f64(&a2[0]).map(|t| -t);
              }
            }
          }
          style.edge_form = Some(ef);
        }
        true
      }
      "FaceForm" if !args.is_empty() => {
        // FaceForm[color] sets fill color
        if let Some(c) = parse_color(&args[0]) {
          style.color = c;
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
    // Thick is equivalent to AbsoluteThickness[2]
    Expr::Identifier(s) if s == "Thick" => {
      style.thickness = -2.0; // negative = absolute pixels
      true
    }
    // Thin is equivalent to AbsoluteThickness[0.5]
    Expr::Identifier(s) if s == "Thin" => {
      style.thickness = -0.5;
      true
    }
    // Dashed is equivalent to Dashing[{Small, Small}]
    Expr::Identifier(s) if s == "Dashed" => {
      style.dashing = Some(vec![SMALL_DASH, SMALL_DASH]);
      true
    }
    // Dotted is equivalent to Dashing[{0, Small}]
    Expr::Identifier(s) if s == "Dotted" => {
      style.dashing = Some(vec![0.0, SMALL_DASH]);
      true
    }
    // DotDashed is equivalent to Dashing[{0, Small, Small, Small}]
    Expr::Identifier(s) if s == "DotDashed" => {
      style.dashing = Some(vec![0.0, SMALL_DASH, SMALL_DASH, SMALL_DASH]);
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
    Expr::Integer(n) => {
      style.font_size = *n as f64;
      true
    }
    Expr::Real(f) => {
      style.font_size = *f;
      true
    }
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
      false
    }
    "FontFamily" => match replacement {
      Expr::String(s) => {
        style.font_family = s.clone();
        true
      }
      Expr::Identifier(s) => {
        style.font_family = s.clone();
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
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        // Style directives are handled by apply_directive
        "Style" if args.len() >= 2 => {
          let saved = style.clone();
          // Apply directives (everything after first arg)
          for directive in &args[1..] {
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
          parse_text(args, style, prims);
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
          // Inset[text, pos] is similar to Text
          parse_text(args, style, prims);
        }
        "Raster" if !args.is_empty() => {
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
                    ((bb.x_min + bb.x_max) / 2.0, (bb.y_min + bb.y_max) / 2.0)
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
  let (cx, cy) = if !args.is_empty() {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  } else {
    (0.0, 0.0)
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
  let (cx, cy) = if !args.is_empty() {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  } else {
    (0.0, 0.0)
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

fn parse_rectangle(
  args: &[Expr],
  style: &StyleState,
  prims: &mut Vec<Primitive>,
) {
  let (x_min, y_min) = if !args.is_empty() {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  } else {
    (0.0, 0.0)
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
  if let Some(pts) = expr_to_point_list(&args[0]) {
    prims.push(Primitive::PolygonPrim {
      points: pts,
      style: style.clone(),
    });
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
    style: style.clone(),
  });
}

fn parse_arrow(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
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
    prims.push(Primitive::ArrowPrim {
      points: pts,
      setback,
      style: style.clone(),
    });
  }
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
    Expr::String(s) => s.clone(),
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
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
      let parts: Vec<String> =
        items.iter().map(graphics_text_content).collect();
      match args.get(1) {
        Some(sep) => parts.join(&graphics_text_content(sep)),
        None => parts.concat(),
      }
    }
    _ => match crate::functions::string_ast::to_string_ast(
      std::slice::from_ref(expr),
    ) {
      Ok(Expr::String(ref s)) => s.clone(),
      _ => crate::syntax::expr_to_string(expr),
    },
  }
}

fn parse_text(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  // Text[str, {x, y}] or Text[Style[str, ...], {x, y}]
  let mut local_style = style.clone();
  // A top-level `Style[content, dirs…]` carries text directives (font size,
  // weight, color) that apply to the whole label; peel it off and apply them
  // to the primitive, then render the inner content.
  let content = match &args[0] {
    Expr::FunctionCall { name, args: sargs }
      if name == "Style" && !sargs.is_empty() =>
    {
      for d in &sargs[1..] {
        apply_directive(d, &mut local_style);
        apply_text_style_directive(d, &mut local_style);
      }
      &sargs[0]
    }
    other => other,
  };
  let text = graphics_text_content(content);

  let (x, y) = if args.len() >= 2 {
    expr_to_point(&args[1]).unwrap_or((0.0, 0.0))
  } else {
    (0.0, 0.0)
  };

  prims.push(Primitive::TextPrim {
    text,
    x,
    y,
    style: local_style,
  });
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
  let expr = Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  };
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
  for _ in 0..=degree {
    knots.push(0.0);
  }
  let num_internal = num_knots - 2 * (degree + 1);
  for i in 1..=num_internal {
    knots.push(i as f64);
  }
  let max_knot = (num_internal + 1) as f64;
  for _ in 0..=degree {
    knots.push(max_knot);
  }

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
  // Raster[data] or Raster[data, {{xmin, ymin}, {xmax, ymax}}]
  // data is a 2D array of grayscale values (0-1) or {r,g,b}/{r,g,b,a} lists
  let data_expr = &args[0];
  let rows = match data_expr {
    Expr::List(rows) => rows,
    _ => return,
  };
  if rows.is_empty() {
    return;
  }

  let mut grid: Vec<Vec<Color>> = Vec::with_capacity(rows.len());
  for row in rows {
    let cols = match row {
      Expr::List(cols) => cols,
      _ => return,
    };
    let mut row_colors: Vec<Color> = Vec::with_capacity(cols.len());
    for cell in cols {
      if let Expr::List(components) = cell
        && (components.len() == 3 || components.len() == 4)
      {
        // RGB or RGBA list
        let r = expr_to_f64(&components[0]).unwrap_or(0.0).clamp(0.0, 1.0);
        let g = expr_to_f64(&components[1]).unwrap_or(0.0).clamp(0.0, 1.0);
        let b = expr_to_f64(&components[2]).unwrap_or(0.0).clamp(0.0, 1.0);
        let a = if components.len() == 4 {
          expr_to_f64(&components[3]).unwrap_or(1.0).clamp(0.0, 1.0)
        } else {
          1.0
        };
        row_colors.push(Color::new(r, g, b).with_alpha(a));
      } else if let Some(v) = expr_to_f64(cell) {
        // Grayscale value: single number maps to gray (0=black, 1=white)
        let v = v.clamp(0.0, 1.0);
        row_colors.push(Color::new(v, v, v));
      } else {
        row_colors.push(Color::new(0.0, 0.0, 0.0));
      }
    }
    grid.push(row_colors);
  }

  let nrows = grid.len();
  let ncols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
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

  prims.push(Primitive::RasterPrim {
    data: grid,
    x_min,
    y_min,
    x_max,
    y_max,
  });
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
    Primitive::PolygonPrim { points, style } => Primitive::PolygonPrim {
      points: points.iter().map(|&(x, y)| rp(x, y)).collect(),
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
    Primitive::TextPrim { text, x, y, style } => {
      let (nx, ny) = rp(*x, *y);
      Primitive::TextPrim {
        text: text.clone(),
        x: nx,
        y: ny,
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

fn thickness_px(t: f64, bb: &BBox, svg_w: f64) -> f64 {
  if t < 0.0 {
    // Absolute thickness (stored as negative)
    -t
  } else {
    t * svg_w / bb.width() * bb.width().max(bb.height())
  }
}

fn dash_attr(dashing: &Option<Vec<f64>>, bb: &BBox, svg_w: f64) -> String {
  if let Some(dashes) = dashing {
    let px: Vec<String> = dashes
      .iter()
      .map(|d| format!("{:.1}", d / bb.width() * svg_w))
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

fn render_axes(
  svg: &mut String,
  axes: (bool, bool),
  bb: &BBox,
  svg_w: f64,
  svg_h: f64,
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
    for t in generate_ticks(bb.x_min, bb.x_max, 6) {
      let x = coord_x(t, bb, svg_w);
      if !x.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{x:.2}\" y2=\"{:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n",
        axis_y_px - 4.0,
        axis_y_px + 4.0
      ));
      let label = format_tick_value(t);
      if axes.1 && label == "0" {
        continue;
      }
      svg.push_str(&format!(
        "<text x=\"{x:.2}\" y=\"{:.2}\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"monospace\" text-anchor=\"middle\" dominant-baseline=\"hanging\">{}</text>\n",
        axis_y_px + 6.0,
        svg_escape(&label),
      ));
    }
  }

  if axes.1 {
    svg.push_str(&format!(
      "<line x1=\"{axis_x_px:.2}\" y1=\"0.00\" x2=\"{axis_x_px:.2}\" y2=\"{svg_h:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n"
    ));
    for t in generate_ticks(bb.y_min, bb.y_max, 6) {
      let y = coord_y(t, bb, svg_h);
      if !y.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{y:.2}\" x2=\"{:.2}\" y2=\"{y:.2}\" stroke=\"{axis_stroke}\" stroke-width=\"1\"/>\n",
        axis_x_px - 4.0,
        axis_x_px + 4.0
      ));
      let label = format_tick_value(t);
      if axes.0 && label == "0" {
        continue;
      }
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{y:.2}\" fill=\"{tick_label_fill}\" font-size=\"14\" font-family=\"monospace\" text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>\n",
        axis_x_px - 6.0,
        svg_escape(&label),
      ));
    }
  }
}

/// Render a rectangular frame around the plot area with tick marks and labels
/// on the bottom and left edges, and minor ticks on the top and right edges.
fn render_frame(svg: &mut String, bb: &BBox, svg_w: f64, svg_h: f64) {
  let t = theme();
  let frame_stroke = t.framed_border;
  let tick_label_fill = t.tick_label_fill;

  // Draw the rectangular border
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{svg_w:.2}\" height=\"{svg_h:.2}\" fill=\"none\" stroke=\"{frame_stroke}\" stroke-width=\"1\"/>\n"
  ));

  let x_ticks = generate_ticks(bb.x_min, bb.x_max, 6);
  let y_ticks = generate_ticks(bb.y_min, bb.y_max, 6);

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
    let label = format_tick_value(t_val);
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
    let label = format_tick_value(t_val);
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
    !matches!(self, GridSpec::None)
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
      dash_attr(&style.dashing, bb, svg_w),
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
      dash_attr(&style.dashing, bb, svg_w),
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
    format!("-{}", truncated)
  } else {
    truncated.to_string()
  }
}

/// Information about how a BigFloat should be displayed graphically.
/// For normal numbers, only `mantissa` is set.
/// For scientific notation, `exponent` contains the power of 10.
struct BigFloatDisplay {
  mantissa: String,
  exponent: Option<i64>,
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
        mantissa: format!("{}0.", prefix),
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
          mantissa: format!("{}0.", prefix),
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

pub(crate) fn svg_escape(s: &str) -> String {
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
    Primitive::PointSingle { x, y, style } => {
      let cx = coord_x(*x, bb, svg_w);
      let cy = coord_y(*y, bb, svg_h);
      let r = style.point_size * svg_w * 0.5;
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
      let r = style.point_size * svg_w * 0.5;
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
      let dash = dash_attr(&style.dashing, bb, svg_w);
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
      let dash = dash_attr(&style.dashing, bb, svg_w);
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
        let large_arc = if (a2 - a1).abs() > std::f64::consts::PI {
          1
        } else {
          0
        };
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
      let color = style.effective_color();
      // Edge form for stroke
      let (stroke_color, stroke_width) = if let Some(ref ef) = style.edge_form {
        let sc = ef.color.unwrap_or(color);
        let sw = ef
          .thickness
          .map(|t| thickness_px(t, bb, svg_w).max(0.5))
          .unwrap_or(0.5);
        (Some(sc), sw)
      } else {
        (None, 0.0)
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
      let large_arc = if sweep_angle.abs() > std::f64::consts::PI {
        1
      } else {
        0
      };
      // Because we negate the sine component when computing arc points
      // (to flip y), the arc geometry is already mirrored.  We therefore
      // need sweep-flag=0 (counter-clockwise in SVG y-down) to trace the
      // correct half of the ellipse.
      let sweep_flag = 0;
      let color = style.effective_color();
      let fill_opacity = if color.a < 1.0 {
        format!(" fill-opacity=\"{}\"", color.a)
      } else {
        String::new()
      };
      // Edge form for stroke
      let stroke_attr = if let Some(ref ef) = style.edge_form {
        let sc = ef.color.unwrap_or(color);
        let sw = ef
          .thickness
          .map(|t| thickness_px(t, bb, svg_w).max(0.5))
          .unwrap_or(0.5);
        let so = if sc.a < 1.0 {
          format!(" stroke-opacity=\"{}\"", sc.a)
        } else {
          String::new()
        };
        format!(
          " stroke=\"{}\" stroke-width=\"{sw:.2}\"{so}",
          sc.to_svg_rgb()
        )
      } else {
        String::new()
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
      let color = style.effective_color();
      // Edge form
      let (stroke_color, stroke_width) = if let Some(ref ef) = style.edge_form {
        let sc = ef.color.unwrap_or(color);
        let st = ef
          .thickness
          .map(|t| thickness_px(t, bb, svg_w).max(0.5))
          .unwrap_or(0.5);
        (Some(sc), st)
      } else {
        (None, 0.0)
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
    Primitive::PolygonPrim { points, style } => {
      let color = style.effective_color();
      let pts: Vec<String> = points
        .iter()
        .map(|&(x, y)| {
          format!("{:.2},{:.2}", coord_x(x, bb, svg_w), coord_y(y, bb, svg_h))
        })
        .collect();
      // Edge form
      let (stroke_color, stroke_width) = if let Some(ref ef) = style.edge_form {
        let sc = ef.color.unwrap_or(color);
        let st = ef
          .thickness
          .map(|t| thickness_px(t, bb, svg_w).max(0.5))
          .unwrap_or(0.5);
        (Some(sc), st)
      } else {
        (None, 0.0)
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
        "<polygon points=\"{}\" fill=\"{}\"{}{}/>\n",
        pts.join(" "),
        color.to_svg_rgb(),
        fill_opacity,
        stroke_attr,
      ));
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
      let dash = dash_attr(&style.dashing, bb, svg_w);

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

      // Draw arrowhead at the end
      if trimmed.len() >= 2 {
        let n = trimmed.len();
        let (x1, y1) = trimmed[n - 2];
        let (x2, y2) = trimmed[n - 1];
        let sx1 = coord_x(x1, bb, svg_w);
        let sy1 = coord_y(y1, bb, svg_h);
        let sx2 = coord_x(x2, bb, svg_w);
        let sy2 = coord_y(y2, bb, svg_h);

        let dx = sx2 - sx1;
        let dy = sy2 - sy1;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 0.0 {
          // Compute the path length AND bounding-box diagonal in SVG
          // pixels so short edges and small self-loops (common in small
          // or multi-component graphs) get proportionally smaller
          // arrowheads instead of the absolute 9 px floor swallowing the
          // whole shape. The path length captures straight-line edges,
          // while the bbox diagonal captures curved shapes like self-
          // loops whose total arc length is large but whose visual size
          // (= loop diameter) is small.
          let mut total_len_px = 0.0_f64;
          let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
          let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
          for (i, w) in trimmed.windows(2).enumerate() {
            let a = (coord_x(w[0].0, bb, svg_w), coord_y(w[0].1, bb, svg_h));
            let b = (coord_x(w[1].0, bb, svg_w), coord_y(w[1].1, bb, svg_h));
            total_len_px += ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
            if i == 0 {
              min_x = a.0.min(b.0);
              max_x = a.0.max(b.0);
              min_y = a.1.min(b.1);
              max_y = a.1.max(b.1);
            } else {
              if b.0 < min_x {
                min_x = b.0;
              }
              if b.0 > max_x {
                max_x = b.0;
              }
              if b.1 < min_y {
                min_y = b.1;
              }
              if b.1 > max_y {
                max_y = b.1;
              }
            }
          }
          let bbox_w = max_x - min_x;
          let bbox_h = max_y - min_y;
          let bbox_diag = (bbox_w * bbox_w + bbox_h * bbox_h).sqrt();

          // Three caps: 45 % of total path length (keeps straight arrows
          // from being dominated), 40 % of the shape's bbox diagonal
          // (keeps self-loops and curved arrows proportional to their
          // visible size), and the usual default of max(sw*6, 9).
          let path_cap = total_len_px * 0.45;
          let bbox_cap = bbox_diag * 0.4;
          let default_head = (sw * 6.0).max(9.0);
          let head_len = default_head.min(path_cap).min(bbox_cap).max(1.0);
          let head_half_w = head_len * 0.45;
          let ux = dx / len;
          let uy = dy / len;
          // Perpendicular
          let px = -uy;
          let py = ux;
          // Arrowhead triangle vertices
          let tip_x = sx2;
          let tip_y = sy2;
          let base_l_x = sx2 - ux * head_len + px * head_half_w;
          let base_l_y = sy2 - uy * head_len + py * head_half_w;
          let base_r_x = sx2 - ux * head_len - px * head_half_w;
          let base_r_y = sy2 - uy * head_len - py * head_half_w;
          out.push_str(&format!(
            "<polygon points=\"{tip_x:.2},{tip_y:.2} {base_l_x:.2},{base_l_y:.2} {base_r_x:.2},{base_r_y:.2}\" fill=\"{}\"{}/>\n",
            color.to_svg_rgb(),
            color.opacity_attr(),
          ));
        }
      }
    }
    Primitive::TextPrim {
      text, x, y, style, ..
    } => {
      let sx = coord_x(*x, bb, svg_w);
      let sy = coord_y(*y, bb, svg_h);
      let color = style.effective_color();
      let fs = style.font_size;
      let ff_attr = if style.font_family.is_empty() {
        String::new()
      } else {
        format!(" font-family=\"{}\"", svg_escape(&style.font_family))
      };

      if text.contains('\n') {
        // Multi-line text with tspan
        let lines: Vec<&str> = text.split('\n').collect();
        out.push_str(&format!(
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\"{ff_attr} text-anchor=\"middle\" dominant-baseline=\"central\"{}>",
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
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\"{ff_attr} text-anchor=\"middle\" dominant-baseline=\"central\"{}>{}</text>\n",
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
      let dash = dash_attr(&style.dashing, bb, svg_w);

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
      let ncols = data.iter().map(|r| r.len()).max().unwrap_or(0);
      if ncols == 0 {
        return;
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
  }
}

// ── Options parsing ──────────────────────────────────────────────────────

fn parse_plot_range(
  expr: &Expr,
) -> Option<(Option<(f64, f64)>, Option<(f64, f64)>)> {
  match expr {
    Expr::Identifier(s) if s == "All" || s == "Automatic" => Some((None, None)),
    Expr::List(items) if items.len() == 2 => {
      let x_range = parse_range_spec(&items[0]);
      let y_range = parse_range_spec(&items[1]);
      Some((x_range, y_range))
    }
    _ => {
      // Single range applies to both axes
      let r = parse_range_spec(expr);
      Some((r, r))
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
      Expr::Identifier(s) if s == "True" => Some(true),
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
  current_color: (f64, f64, f64),
  current_opacity: f64,
  current_thickness: f64,
}

impl Default for BoxStyleTracker {
  fn default() -> Self {
    Self {
      current_color: (0.0, 0.0, 0.0), // Black
      current_opacity: 1.0,
      current_thickness: 0.004,
    }
  }
}

impl BoxStyleTracker {
  /// Emit directives needed to switch to the given style, returning any new directives.
  fn emit_style_changes(&mut self, style: &StyleState) -> Vec<String> {
    let mut directives = Vec::new();
    let new_color = (style.color.r, style.color.g, style.color.b);
    if (new_color.0 - self.current_color.0).abs() > 1e-6
      || (new_color.1 - self.current_color.1).abs() > 1e-6
      || (new_color.2 - self.current_color.2).abs() > 1e-6
    {
      directives.push(gbox::rgbcolor_box(
        new_color.0,
        new_color.1,
        new_color.2,
      ));
      self.current_color = new_color;
    }
    if (style.opacity - self.current_opacity).abs() > 1e-6 {
      directives.push(gbox::opacity_box(style.opacity));
      self.current_opacity = style.opacity;
    }
    if (style.thickness - self.current_thickness).abs() > 1e-6 {
      directives.push(gbox::abs_thickness_box(style.thickness));
      self.current_thickness = style.thickness;
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
      Primitive::PolygonPrim { points, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::polygon_box(points));
      }
      Primitive::ArrowPrim {
        points,
        setback,
        style,
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::arrow_box(points, *setback));
      }
      Primitive::TextPrim { text, x, y, style } => {
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
      Primitive::HalfPlanePrim { .. } => {
        // Unbounded fills have no fixed-coordinate box form; skip
      }
    }
  }

  elements
}

// ── Entry point ──────────────────────────────────────────────────────────

pub fn graphics_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // First arg is the content (primitives + directives)
  // Evaluate it so that Table/Map/etc. produce concrete lists
  // Remaining args are options as Rule expressions
  let content = evaluate_expr_to_expr(&args[0])?;

  // Parse options
  let mut svg_width: u32 = 360;
  let mut svg_height: u32 = 225;
  let mut explicit_height = false;
  let mut full_width = false;
  let mut plot_range_x: Option<(f64, f64)> = None;
  let mut plot_range_y: Option<(f64, f64)> = None;
  let mut background: Option<Color> = None;
  let mut axes = (false, false);
  let mut frame = false;
  let mut grid_x = GridSpec::None;
  let mut grid_y = GridSpec::None;
  let mut grid_style: Option<StyleState> = None;
  // When true, skip uniform scaling so x and y axes scale independently
  // (needed for plots where data aspect ≠ image aspect).
  let mut aspect_ratio_full = false;

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
        "ImageSize" => {
          // Check if {w, h} form was used (explicit height)
          if let Expr::List(items) = replacement.as_ref()
            && items.len() == 2
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
          if let Some((xr, yr)) = parse_plot_range(replacement) {
            plot_range_x = xr;
            plot_range_y = yr;
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
        "Frame" => {
          if let Expr::Identifier(s) = replacement.as_ref() {
            if s == "True" {
              frame = true;
            }
          } else if let Expr::FunctionCall { name: fn_name, .. } =
            replacement.as_ref()
            && fn_name == "True"
          {
            frame = true;
          }
        }
        "GridLines" => match replacement.as_ref() {
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
          apply_directive(replacement.as_ref(), &mut st);
          grid_style = Some(st);
        }
        "AspectRatio" => {
          // AspectRatio -> Full: skip uniform scaling (used by plots)
          if let Expr::Identifier(s) = replacement.as_ref() {
            if s == "Full" {
              aspect_ratio_full = true;
            }
          } else if let Some(r) = expr_to_f64(replacement)
            && r > 0.0
          {
            svg_height = (svg_width as f64 * r).round() as u32;
            explicit_height = true;
            aspect_ratio_full = true;
          }
        }
        _ => {}
      }
    }
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

  // Compute margins for axis/frame tick labels.
  let margin_left: f64 = if frame {
    50.0
  } else if axes.1 {
    50.0
  } else {
    0.0
  };
  let margin_bottom: f64 = if frame {
    25.0
  } else if axes.0 {
    25.0
  } else {
    0.0
  };
  let margin_right: f64 = if frame { 10.0 } else { 0.0 };
  let margin_top: f64 = if frame { 10.0 } else { 0.0 };
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
      "<rect width=\"{}\" height=\"{}\" fill=\"rgb(100%,33%,33%)\" fill-opacity=\"0.08\"/>\n",
      svg_width, svg_height
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
      "<rect width=\"{}\" height=\"{}\" fill=\"transparent\" stroke=\"none\"><title>{}</title></rect>\n",
      svg_width, svg_height, title_text
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

  render_axes(&mut svg, axes, &bb, svg_w, svg_h);

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
    render_frame(&mut svg, &bb, svg_w, svg_h);
  }

  if has_margin {
    svg.push_str("</g>\n");
  }

  svg.push_str("</svg>");

  // Generate and store GraphicsBox expression for .nb export
  let box_elements = primitives_to_box_elements(&primitives);
  let graphicsbox = gbox::graphics_box(&box_elements);
  crate::capture_graphicsbox(&graphicsbox);

  Ok(crate::graphics_result(svg))
}

// ── Grid SVG rendering ──────────────────────────────────────────────────

/// Extract the base and exponent from a Power expression (either BinaryOp or FunctionCall form).
/// Public accessor for `as_power` — used by `expr_to_box_form` for unit handling.
pub fn as_power_pub(expr: &Expr) -> Option<(&Expr, &Expr)> {
  as_power(expr)
}

fn as_power(expr: &Expr) -> Option<(&Expr, &Expr)> {
  match expr {
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
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
  use crate::syntax::BinaryOperator;
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
  format!("{}/{}", num_markup, den_markup)
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
      let cx = (x0 + x1) / 2.0;
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
      let xm = (x0 + x1) / 2.0;
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
      let xm = (x0 + x1) / 2.0;
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
      return BoxLayout {
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
      return BoxLayout {
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
      return BoxLayout {
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
    BoxLayout {
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
    Expr::String(s) => BoxLayout::text(s, font_size),
    Expr::Identifier(s) => BoxLayout::text(s, font_size),
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

      // TagBox, InterpretationBox — delegate to content
      "TagBox" if args.len() == 2 => layout_box(&args[0], font_size),
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

          let n_cols = laid_out.iter().map(|r| r.len()).max().unwrap_or(0);
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
          let first_bl =
            row_metrics.first().map(|(bl, _)| *bl).unwrap_or(font_size);
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
  use crate::syntax::BinaryOperator;

  // Handle Power in both BinaryOp and FunctionCall form
  if let Some((base, exp)) = as_power(unit) {
    let base_str = quantity_unit_to_svg_abbrev(base);
    let exp_str = expr_to_svg_markup(exp);
    return format!(
      "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
      base_str, exp_str
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
        format!("{}/{}", numer, denom)
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
  for (i, chunk) in digits[remainder..].as_bytes().chunks(3).enumerate() {
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
  let num_groups = digit_count / 3 + if remainder > 0 { 1 } else { 0 };
  (num_groups - 1) as f64 * 0.3
}

/// Convert an `Expr` into SVG text markup (inner content of a `<text>` element).
/// Recursively handles all expression types so that Power expressions
/// anywhere in the tree are rendered with `<tspan>` superscripts.
pub fn expr_to_svg_markup(expr: &Expr) -> String {
  use crate::syntax::{
    BinaryOperator, ComparisonOp, UnaryOperator, expr_to_output,
  };

  // Power → superscript (handles both BinaryOp and FunctionCall forms)
  if let Some((base, exp)) = as_power(expr) {
    let base_markup = expr_to_svg_markup(base);
    let exp_markup = expr_to_svg_markup(exp);
    // Wrap base in parens if it's a lower-precedence additive expression
    let base_fmt = if is_additive_expr(base) {
      format!("({})", base_markup)
    } else {
      base_markup
    };
    return format!(
      "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
      base_fmt, exp_markup
    );
  }

  match expr {
    // ── Atoms ──
    Expr::String(s) => svg_escape(s),
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
    Expr::Real(_) | Expr::Constant(_) | Expr::Slot(_) => {
      svg_escape(&expr_to_output(expr))
    }

    // ── List → {a, b, c} ──
    Expr::List(items) => {
      let parts: Vec<String> = items.iter().map(expr_to_svg_markup).collect();
      format!("{{{}}}", parts.join(", "))
    }

    // ── UnaryOp ──
    Expr::UnaryOp { op, operand } => {
      let inner = expr_to_svg_markup(operand);
      match op {
        UnaryOperator::Minus => format!("-{}", inner),
        UnaryOperator::Not => format!("!{}", inner),
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
          format!("({})", base_markup)
        } else {
          base_markup
        };
        return format!(
          "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          base_fmt, exp_markup
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
        format!("({})", left_str)
      } else {
        left_str
      };
      let right_fmt = if is_mult && is_additive_expr(right.as_ref()) {
        format!("({})", right_str)
      } else {
        right_str
      };
      if needs_space {
        format!("{} {} {}", left_fmt, op_str, right_fmt)
      } else {
        format!("{}{}{}", left_fmt, op_str, right_fmt)
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
                result.push_str(&expr_to_svg_markup(&Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: fn_args[1..].to_vec().into(),
                }));
              }
            } else if let Expr::Integer(n) = arg
              && *n < 0
            {
              result.push_str(" - ");
              result.push_str(&expr_to_svg_markup(&Expr::Integer(-n)));
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
                  format!("({})", s)
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
            return format!("-{}", joined);
          }
          // General: implicit multiplication (no * symbol)
          let parts: Vec<String> = args
            .iter()
            .map(|a| {
              let s = expr_to_svg_markup(a);
              if is_additive_expr(a) {
                format!("({})", s)
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
          format!("{} {}", mag, unit)
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

        // Style[content, directives...] → render content only
        "Style" if !args.is_empty() => expr_to_svg_markup(&args[0]),

        // HoldForm[expr] → render content
        "HoldForm" if args.len() == 1 => expr_to_svg_markup(&args[0]),

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
      format!("-Image ({}×{})-", width, height)
    }

    // ── Everything else → fallback to expr_to_output ──
    _ => svg_escape(&expr_to_output(expr)),
  }
}

/// Estimate the display width of an expression in character units,
/// accounting for superscript sizing (exponents rendered at ~70% width).
/// Recursively mirrors `expr_to_svg_markup` structure.
pub fn estimate_display_width(expr: &Expr) -> f64 {
  use crate::syntax::{BinaryOperator, expr_to_output};

  if let Some((base, exp)) = as_power(expr) {
    let parens = if is_additive_expr(base) { 2.0 } else { 0.0 };
    return estimate_display_width(base)
      + parens
      + estimate_display_width(exp) * 0.7;
  }

  match expr {
    // Atoms
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
      "Style" if !args.is_empty() => estimate_display_width(&args[0]),
      // HoldForm[expr] → width of content
      "HoldForm" if args.len() == 1 => estimate_display_width(&args[0]),
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

    // Fallback
    _ => expr_to_output(expr).len() as f64,
  }
}

/// Estimate the display width of an abbreviated unit expression.
fn estimate_unit_abbrev_width(unit: &Expr) -> f64 {
  use crate::functions::quantity_ast::unit_to_abbreviation;
  use crate::syntax::BinaryOperator;

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
pub fn boxes_to_svg(expr: &Expr) -> String {
  match expr {
    // Atoms: in box form, atoms are always Expr::String
    Expr::String(s) => svg_escape(s),
    // Identifiers can appear for fallback cases
    Expr::Identifier(s) => svg_escape(s),
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
          "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          base_svg, exp_svg
        )
      }

      // SubscriptBox[base, sub]
      "SubscriptBox" if args.len() == 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let sub_svg = boxes_to_svg(&args[1]);
        format!(
          "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>",
          base_svg, sub_svg
        )
      }

      // SubsuperscriptBox[base, sub, sup]
      "SubsuperscriptBox" if args.len() == 3 => {
        let base_svg = boxes_to_svg(&args[0]);
        let sub_svg = boxes_to_svg(&args[1]);
        let sup_svg = boxes_to_svg(&args[2]);
        format!(
          "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>\
           <tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          base_svg, sub_svg, sup_svg
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
        format!(
          "\u{221A}<tspan text-decoration=\"overline\">{}</tspan>",
          content
        )
      }

      // RadicalBox[expr, n] → index√content with overline
      "RadicalBox" if args.len() == 2 => {
        let content = boxes_to_svg(&args[0]);
        let index = boxes_to_svg(&args[1]);
        format!(
          "<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>\u{221A}<tspan text-decoration=\"overline\">{}</tspan>",
          index, content
        )
      }

      // OverscriptBox[base, over] → base with overscript
      "OverscriptBox" if args.len() >= 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let over_svg = boxes_to_svg(&args[1]);
        format!(
          "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          base_svg, over_svg
        )
      }

      // UnderscriptBox[base, under] → base with underscript
      "UnderscriptBox" if args.len() >= 2 => {
        let base_svg = boxes_to_svg(&args[0]);
        let under_svg = boxes_to_svg(&args[1]);
        format!(
          "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>",
          base_svg, under_svg
        )
      }

      // UnderoverscriptBox[base, under, over] → base with both
      "UnderoverscriptBox" if args.len() >= 3 => {
        let base_svg = boxes_to_svg(&args[0]);
        let under_svg = boxes_to_svg(&args[1]);
        let over_svg = boxes_to_svg(&args[2]);
        format!(
          "{}<tspan baseline-shift=\"sub\" font-size=\"70%\">{}</tspan>\
           <tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
          base_svg, under_svg, over_svg
        )
      }

      // FrameBox[content, ...] → content with frame markers
      "FrameBox" if !args.is_empty() => {
        let content = boxes_to_svg(&args[0]);
        format!("[{}]", content)
      }

      // TagBox[boxes, tag] → render boxes, ignore tag
      "TagBox" if args.len() == 2 => boxes_to_svg(&args[0]),

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
                  font_size_attr = format!(" font-size=\"{}\"", sz);
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
          format!("<tspan{}{}>{}</tspan>", font_size_attr, color_attr, content)
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
      items.iter().map(boxes_to_svg).collect::<Vec<_>>().join("")
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
  let mut i = 0;
  while i < cs.len() {
    if cs[i] == '\\' && i + 1 < cs.len() {
      match cs[i + 1] {
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
    _ => Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(units.into())].into(),
    },
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
    Expr::String(s) | Expr::Identifier(s) => s.clone(),
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
      "TagBox" if args.len() == 2 => estimate_box_display_width(&args[0]),
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
  if let Expr::Rule {
    pattern,
    replacement,
  } = opt
    && let Some(opt_name) = option_name(pattern)
  {
    // For PlotRange, compute the union (min of mins, max of maxes)
    // so that all merged graphics remain visible.
    if opt_name == "PlotRange"
      && let Some(pos) = opts.iter().position(|existing| {
        if let Expr::Rule { pattern: ep, .. } = existing {
          option_name(ep) == Some("PlotRange")
        } else {
          false
        }
      })
      && let Expr::Rule {
        replacement: ref existing_repl,
        ..
      } = opts[pos]
      && let Some(merged) = merge_plot_ranges(existing_repl, replacement)
    {
      opts[pos] = Expr::Rule {
        pattern: Box::new(Expr::Identifier("PlotRange".to_string())),
        replacement: Box::new(merged),
      };
      return;
    }

    opts.retain(|existing| {
      if let Expr::Rule { pattern: ep, .. } = existing {
        option_name(ep) != Some(opt_name)
      } else {
        true
      }
    });
  }
  opts.push(opt.clone());
}

/// Merge two PlotRange values by taking the union (min of mins, max of maxes).
fn merge_plot_ranges(a: &Expr, b: &Expr) -> Option<Expr> {
  let (ax, ay) = parse_plot_range(a)?;
  let (bx, by) = parse_plot_range(b)?;

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

  Some(Expr::List(
    vec![range_to_expr(mx), range_to_expr(my)].into(),
  ))
}

/// Implementation of Show[g1, g2, ..., opts...].
/// Convert MeshRegion vertex/polygon data to Graphics primitives (Polygon with coordinates).
pub(crate) fn mesh_region_to_graphics_prims(
  vertices_expr: &Expr,
  primitives_expr: &Expr,
) -> Option<Vec<Expr>> {
  let vertices_list = match vertices_expr {
    Expr::List(v) => v,
    _ => return None,
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

  let prims = match primitives_expr {
    Expr::List(v) => v,
    _ => return None,
  };

  let mut result = Vec::new();
  // Add default styling
  result.push(Expr::FunctionCall {
    name: "EdgeForm".to_string(),
    args: vec![Color::gray(0.4).to_expr()].into(),
  });
  result.push(Expr::FunctionCall {
    name: "FaceForm".to_string(),
    args: vec![Color::new(0.626, 0.836, 0.919).to_expr()].into(),
  });

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
            result.push(Expr::FunctionCall {
              name: "Polygon".to_string(),
              args: vec![Expr::List(points.into())].into(),
            });
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
pub fn show_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut merged_primitives: Vec<Expr> = Vec::new();
  let mut merged_options: Vec<Expr> = Vec::new();
  let mut is_3d = false;
  // Pre-rendered Graphics objects (e.g. from Plot[], Plot3D[])
  let mut rendered_graphics: Vec<Expr> = Vec::new();
  // Plot source data for re-rendering via plotters
  let mut plot_sources: Vec<crate::syntax::PlotSource> = Vec::new();

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

  for arg in args {
    // If the arg is not already a Graphics/Graphics3D expression,
    // try evaluating it (e.g. it could be a variable or function call)
    let evaled;
    let expr_ref = match arg {
      Expr::FunctionCall { name, .. }
        if name == "Graphics" || name == "Graphics3D" =>
      {
        arg
      }
      Expr::Rule { .. } => arg,
      _ => {
        evaled = evaluate_expr_to_expr(arg).unwrap_or_else(|_| arg.clone());
        &evaled
      }
    };

    match expr_ref {
      Expr::FunctionCall { name, args: gargs } if name == "Graphics" => {
        if !gargs.is_empty() {
          merged_primitives.push(gargs[0].clone());
        }
        for opt in gargs.iter().skip(1) {
          merge_option(&mut merged_options, opt);
        }
      }
      Expr::FunctionCall { name, args: gargs }
        if name == "MeshRegion" && gargs.len() == 2 =>
      {
        // Convert MeshRegion to Graphics primitives for Show merging
        if let Some(graphics_prims) =
          mesh_region_to_graphics_prims(&gargs[0], &gargs[1])
        {
          merged_primitives.push(Expr::List(graphics_prims.into()));
        }
      }
      Expr::FunctionCall { name, args: gargs } if name == "Graphics3D" => {
        is_3d = true;
        if !gargs.is_empty() {
          merged_primitives.push(gargs[0].clone());
        }
        for opt in gargs.iter().skip(1) {
          merge_option(&mut merged_options, opt);
        }
      }
      Expr::Graphics {
        is_3d: g_is_3d,
        source,
        ..
      } => {
        is_3d = *g_is_3d;
        if let Some(src) = source {
          plot_sources.push(src.as_ref().clone());
        } else {
          // No source data — collect as opaque pre-rendered graphic
          rendered_graphics.push(expr_ref.clone());
        }
      }
      Expr::Rule { .. } => {
        merge_option(&mut merged_options, expr_ref);
      }
      _ => {}
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
    };

    let svg = crate::functions::plot::render_merged_plot_source(&merged)?;
    return Ok(crate::graphics_result_with_source(svg, merged));
  }

  // Mixed case: plot sources + Graphics primitives.
  // Convert plot source series to Line/Point primitives so they can be
  // merged with the other Graphics primitives via graphics_ast.
  if !plot_sources.is_empty() {
    for ps in &plot_sources {
      let mut series_prims: Vec<Expr> = Vec::new();
      for sd in &ps.series {
        // Color directive
        series_prims.push(Expr::FunctionCall {
          name: "RGBColor".to_string(),
          args: vec![
            Expr::Real(sd.color.0 as f64 / 255.0),
            Expr::Real(sd.color.1 as f64 / 255.0),
            Expr::Real(sd.color.2 as f64 / 255.0),
          ]
          .into(),
        });
        if sd.is_scatter {
          series_prims.push(Expr::FunctionCall {
            name: "PointSize".to_string(),
            args: vec![Expr::Real(0.012)].into(),
          });
          let coords: Vec<Expr> = sd
            .points
            .iter()
            .filter(|(_, y)| y.is_finite())
            .map(|&(x, y)| {
              Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
            })
            .collect();
          if !coords.is_empty() {
            series_prims.push(Expr::FunctionCall {
              name: "Point".to_string(),
              args: vec![Expr::List(coords.into())].into(),
            });
          }
        } else {
          series_prims.push(Expr::FunctionCall {
            name: "AbsoluteThickness".to_string(),
            args: vec![Expr::Real(1.5)].into(),
          });
          let segments =
            crate::functions::plot::split_into_segments(&sd.points);
          for seg in &segments {
            let coords: Vec<Expr> = seg
              .iter()
              .map(|&(x, y)| {
                Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
              })
              .collect();
            if coords.len() >= 2 {
              series_prims.push(Expr::FunctionCall {
                name: "Line".to_string(),
                args: vec![Expr::List(coords.into())].into(),
              });
            }
          }
        }
      }
      merged_primitives.push(Expr::List(series_prims.into()));

      // Deliberately do NOT force a PlotRange from the plot source here: the
      // series are emitted as real Line/Point primitives, so the renderer's
      // automatic range already covers the curve. Forcing the source's tight
      // range would crop any other Graphics primitives that extend beyond it
      // (e.g. a control polygon), whereas Wolfram shows the union of all
      // primitives. Leaving PlotRange unset yields that union.

      // Enable axes
      let axes_rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier("Axes".to_string())),
        replacement: Box::new(Expr::Identifier("True".to_string())),
      };
      merge_option(&mut merged_options, &axes_rule);

      // AspectRatio -> Full for plot-style rendering
      let ar_rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier("AspectRatio".to_string())),
        replacement: Box::new(Expr::Identifier("Full".to_string())),
      };
      merge_option(&mut merged_options, &ar_rule);
    }
  }

  // If we have pre-rendered Graphics but no primitives from Graphics[...],
  // return the rendered result directly. Single-arg Show just passes through.
  if merged_primitives.is_empty() && !rendered_graphics.is_empty() {
    return Ok(rendered_graphics[0].clone());
  }

  if merged_primitives.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "Show".to_string(),
      args: args.to_vec().into(),
    });
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

/// Render a parenthesized grid and return the raw SVG string (for composition).
pub fn grid_svg_with_parens(args: &[Expr]) -> Result<String, InterpreterError> {
  grid_svg_internal(args, &[], true)
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
  let svg = grid_svg_internal(args, group_gaps, parens)?;
  Ok(crate::graphics_result(svg))
}

/// Default style inherited from an outer Style[Grid[...], directives...].
#[derive(Clone, Default)]
pub struct GridStyle {
  pub font_weight: Option<&'static str>,
  pub font_style: Option<&'static str>,
  pub font_size: Option<f64>,
  pub(crate) color: Option<Color>,
}

/// Extract style info from a Style[content, directives...] cell.
/// Returns (content, font_size, font_weight, font_style, color).
fn extract_cell_style(
  cell: &Expr,
) -> (&Expr, Option<f64>, &str, &str, Option<Color>) {
  if let Expr::FunctionCall { name, args } = cell
    && name == "Style"
    && !args.is_empty()
  {
    let content = &args[0];
    let mut fs: Option<f64> = None;
    let mut fw = "normal";
    let mut fst = "normal";
    let mut color: Option<Color> = None;
    for directive in &args[1..] {
      match directive {
        Expr::Identifier(s) if s == "Bold" => fw = "bold",
        Expr::Identifier(s) if s == "Italic" => fst = "italic",
        Expr::Integer(n) => fs = Some(*n as f64),
        Expr::Real(f) => fs = Some(*f),
        Expr::Rule {
          pattern,
          replacement,
        } => {
          if let Expr::Identifier(k) = pattern.as_ref()
            && k == "FontSize"
          {
            match replacement.as_ref() {
              Expr::Integer(n) => fs = Some(*n as f64),
              Expr::Real(f) => fs = Some(*f),
              _ => {}
            }
          }
        }
        _ => {
          if let Some(c) = parse_color(directive) {
            color = Some(c);
          }
        }
      }
    }
    return (content, fs, fw, fst, color);
  }
  (cell, None, "normal", "normal", None)
}

/// Parse Style directives into a GridStyle.
pub fn parse_grid_style(directives: &[Expr]) -> GridStyle {
  let mut gs = GridStyle::default();
  for d in directives {
    match d {
      Expr::Identifier(s) if s == "Bold" => gs.font_weight = Some("bold"),
      Expr::Identifier(s) if s == "Italic" => gs.font_style = Some("italic"),
      Expr::Integer(n) => gs.font_size = Some(*n as f64),
      Expr::Real(f) => gs.font_size = Some(*f),
      Expr::Rule {
        pattern,
        replacement,
      } => {
        if let Expr::Identifier(k) = pattern.as_ref()
          && k == "FontSize"
        {
          match replacement.as_ref() {
            Expr::Integer(n) => gs.font_size = Some(*n as f64),
            Expr::Real(f) => gs.font_size = Some(*f),
            _ => {}
          }
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

/// Check if a cell is or contains an Expr::Image (unwrapping Style).
fn unwrap_to_image(cell: &Expr) -> Option<&Expr> {
  match cell {
    Expr::Image { .. } => Some(cell),
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      unwrap_to_image(&args[0])
    }
    _ => None,
  }
}

/// Convert a WL alignment identifier to SVG text-anchor value.
fn alignment_to_anchor(expr: &Expr) -> &'static str {
  if let Expr::Identifier(val) = expr {
    match val.as_str() {
      "Left" => "start",
      "Right" => "end",
      _ => "middle",
    }
  } else {
    "middle"
  }
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

  // Parse options from remaining args
  let mut frame_outer = false; // Frame -> True: outer border only
  let mut frame_all = false; // Frame -> All: all gridlines
  let mut frame_color: Option<Color> = None; // custom frame color
  let mut row_headings: Vec<Expr> = Vec::new();
  let mut col_headings: Vec<Expr> = Vec::new();
  let mut spacings_h: Option<f64> = None; // horizontal spacing override
  let mut spacings_v: Option<f64> = None; // vertical spacing override
  let mut dividers_col = false; // vertical divider lines between columns
  let mut dividers_row = false; // horizontal divider lines between rows
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
        "TableHeadings" => {
          // TableHeadings -> {{row_h...}, {col_h...}}
          if let Expr::List(lists) = replacement.as_ref() {
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
        }
        _ => {}
      }
    }
  }

  // Inject TableHeadings into the grid data
  if !col_headings.is_empty() {
    // Add column headings as the first row (bold)
    let mut heading_row: Vec<Expr> = col_headings
      .into_iter()
      .map(|h| Expr::FunctionCall {
        name: "Style".to_string(),
        args: vec![h, Expr::Identifier("Bold".to_string())].into(),
      })
      .collect();
    if !row_headings.is_empty() {
      // Insert empty top-left corner cell
      heading_row.insert(0, Expr::Identifier(String::new()));
    }
    rows.insert(0, heading_row);
  }
  if !row_headings.is_empty() {
    // Add row headings as the first column (bold)
    let start = if rows.first().is_some_and(|r| {
      r.first()
        .is_some_and(|c| matches!(c, Expr::Identifier(s) if s.is_empty()))
    }) {
      1 // Skip the heading row (already has corner cell)
    } else {
      0
    };
    for (i, row) in rows.iter_mut().enumerate() {
      if i >= start {
        let idx = i - start;
        if let Some(h) = row_headings.get(idx) {
          row.insert(
            0,
            Expr::FunctionCall {
              name: "Style".to_string(),
              args: vec![h.clone(), Expr::Identifier("Bold".to_string())]
                .into(),
            },
          );
        } else {
          row.insert(0, Expr::Identifier(String::new()));
        }
      }
    }
  }

  // Convert cells to text
  let num_rows = rows.len();
  let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
  if num_cols == 0 {
    return Err(InterpreterError::EvaluationError("Grid: empty data".into()));
  }

  // Compute column widths based on estimated display width
  let char_width: f64 = 8.4; // approximate monospace char width at font-size 14
  let font_size: f64 = 14.0;
  // Apply Spacings option: values are in ems (multiples of char_width / font_size)
  let pad_x: f64 = match spacings_h {
    Some(h) => h * char_width, // Spacings h in ems → pixel padding
    None => 12.0,              // default horizontal padding per cell
  };
  let pad_y: f64 = 2.0; // vertical padding per cell (each side = 1)
  let row_gap: f64 = match spacings_v {
    Some(v) => v * font_size, // Spacings v in ems → pixel gap between rows
    None => 0.0,              // default: no extra row gap
  };
  let group_gap: f64 = 6.0; // extra spacing between groups
  let base_row_height = font_size + pad_y;
  let frac_row_height = font_size + pad_y + 10.0; // taller for stacked fractions

  let mut col_widths: Vec<f64> = vec![0.0; num_cols];
  for row in &rows {
    for (j, cell) in row.iter().enumerate() {
      let w = estimate_display_width(cell) * char_width + pad_x;
      if w > col_widths[j] {
        col_widths[j] = w;
      }
    }
  }

  // Compute per-row heights (taller for fractions or images)
  let row_heights: Vec<f64> = rows
    .iter()
    .map(|row| {
      let mut max_h = if row.iter().any(has_fraction) {
        frac_row_height
      } else {
        base_row_height
      };
      // Check for Image cells — scale to fit column width, compute height
      for (j, cell) in row.iter().enumerate() {
        if let Some(img) = unwrap_to_image(cell)
          && let Expr::Image {
            width: iw,
            height: ih,
            ..
          } = img
        {
          let col_w = if j < col_widths.len() {
            col_widths[j] - pad_x
          } else {
            200.0
          };
          let scale = col_w / (*iw as f64);
          let img_h = (*ih as f64) * scale + pad_y;
          if img_h > max_h {
            max_h = img_h;
          }
        }
      }
      max_h
    })
    .collect();

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
  let has_frame = frame_all || frame_outer || has_per_pos_dividers;
  let frame_pad: f64 = if has_frame { 0.5 } else { 0.0 };
  let svg_w = (total_width + 2.0 * frame_pad).ceil() as u32;
  let svg_h = (total_height + 2.0 * frame_pad).ceil() as u32;
  let mut svg = String::with_capacity(2048);
  if has_frame {
    svg.push_str(&format!(
      "<svg width=\"{}\" height=\"{}\" viewBox=\"-0.5 -0.5 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
      svg_w, svg_h, svg_w, svg_h
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
      svg_w, svg_h, svg_w, svg_h
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
      (prev_bottom + content_y_starts[i]) / 2.0
    };
    let bottom = if i == num_rows - 1 {
      total_height
    } else {
      let this_bottom = content_y_starts[i] + row_heights[i];
      (this_bottom + content_y_starts[i + 1]) / 2.0
    };
    visual_tops.push(top);
    visual_bottoms.push(bottom);
  }

  // Draw cell backgrounds
  for (i, row) in rows.iter().enumerate() {
    let bg_y = visual_tops[i];
    let bg_h = visual_bottoms[i] - visual_tops[i];
    let mut x_offset: f64 = paren_margin;
    for (j, _cell) in row.iter().enumerate() {
      let col_w = col_widths[j];
      let bg = row_backgrounds
        .get(i % row_backgrounds.len().max(1))
        .and_then(|c| c.as_ref())
        .or_else(|| {
          col_backgrounds
            .get(j % col_backgrounds.len().max(1))
            .and_then(|c| c.as_ref())
        })
        .or(background_color.as_ref());
      if let Some(color) = bg {
        svg.push_str(&format!(
          "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"{}/>\n",
          x_offset, bg_y, col_w, bg_h, color.to_svg_rgb(), color.opacity_attr()
        ));
      }
      x_offset += col_w;
    }
  }

  // Draw cell contents — text is centered within visual row bounds
  for (i, row) in rows.iter().enumerate() {
    let mut x_offset: f64 = paren_margin;
    for (j, cell) in row.iter().enumerate() {
      let col_w = col_widths[j];
      let col_align = col_alignments.get(j).copied().unwrap_or(alignment_h);
      let cx = match col_align {
        "start" => x_offset + pad_x / 2.0,
        "end" => x_offset + col_w - pad_x / 2.0,
        _ => x_offset + col_w / 2.0,
      };
      // Shift text down slightly to compensate for ascenders being taller
      // than descenders, which makes mathematical centering look top-heavy.
      let cy = (visual_tops[i] + visual_bottoms[i]) / 2.0 - 1.0;

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
          let avail_w = col_w - pad_x;
          let scale = avail_w / (*iw as f64);
          let draw_w = avail_w;
          let draw_h = (*ih as f64) * scale;
          let ix = x_offset + pad_x / 2.0;
          let vis_h = visual_bottoms[i] - visual_tops[i];
          let iy = visual_tops[i] + (vis_h - draw_h) / 2.0;

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
        let (content, cell_fs, cell_fw, cell_fst, cell_color) =
          extract_cell_style(cell);

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
        let eff_fw = if cell_fw != "normal" {
          cell_fw
        } else {
          default_style.font_weight.unwrap_or("normal")
        };
        let eff_fst = if cell_fst != "normal" {
          cell_fst
        } else {
          default_style.font_style.unwrap_or("normal")
        };
        let fw_attr = if eff_fw != "normal" {
          format!(" font-weight=\"{}\"", eff_fw)
        } else {
          String::new()
        };
        let fst_attr = if eff_fst != "normal" {
          format!(" font-style=\"{}\"", eff_fst)
        } else {
          String::new()
        };
        // Hyperlink cells default to a link-blue fill (overridable via
        // explicit Style[..., color]). Plain cells use the cell/default/theme
        // colors as before.
        let text_fill = if let Some(ref c) = cell_color {
          c.to_svg_rgb()
        } else if link_href.is_some() {
          "#1a73e8".to_string()
        } else if let Some(ref c) = default_style.color {
          c.to_svg_rgb()
        } else {
          theme().text_primary.to_string()
        };
        let text_elem = format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"sans-serif\" font-size=\"{fs}\"{fw_attr}{fst_attr} fill=\"{text_fill}\" text-anchor=\"{col_align}\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(text_content)
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
    .map(|c| c.to_svg_rgb())
    .unwrap_or_else(|| theme().stroke_default.to_string());

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
        } else if is_border && draw_outer {
          (true, default_stroke.clone())
        } else {
          (false, String::new())
        }
      } else if (is_border && draw_outer) || (!is_border && draw_inner_h) {
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
        svg.push_str(&format!(
          "<line x1=\"{paren_margin:.1}\" y1=\"{draw_y:.1}\" x2=\"{:.1}\" y2=\"{draw_y:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n",
          paren_margin + grid_width
        ));
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
        } else if is_border && draw_outer {
          (true, default_stroke.clone())
        } else {
          (false, String::new())
        }
      } else if (is_border && draw_outer) || (!is_border && draw_inner_v) {
        (true, default_stroke.clone())
      } else {
        (false, String::new())
      };
      if should_draw {
        svg.push_str(&format!(
          "<line x1=\"{x_offset:.1}\" y1=\"0\" x2=\"{x_offset:.1}\" y2=\"{total_height:.1}\" stroke=\"{stroke}\" stroke-width=\"1\"/>\n"
        ));
      }
      if j < num_cols {
        x_offset += col_widths[j];
      }
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
  let num_outer_cols = outer_rows.iter().map(|r| r.len()).max().unwrap_or(0);
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
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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

/// Stack multiple SVG strings vertically with spacing, capture the result.
pub fn stack_svgs_vertically(
  svgs: &[String],
) -> Result<Expr, InterpreterError> {
  if svgs.is_empty() {
    return Err(InterpreterError::EvaluationError("No SVGs to stack".into()));
  }

  // Parse each SVG to get its dimensions and content
  let mut parsed: Vec<(f64, f64, String, String)> = Vec::new(); // (w, h, viewBox, innerContent)
  for svg in svgs {
    if let Some(p) = parse_svg_dimensions(svg) {
      let parts: Vec<f64> = p
        .view_box
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
      if parts.len() >= 4 {
        parsed.push((parts[2], parts[3], p.view_box.clone(), p.inner_content));
      }
    }
  }

  if parsed.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Could not parse SVGs".into(),
    ));
  }

  let gap = 8.0_f64;
  let max_width: f64 = parsed.iter().map(|(w, _, _, _)| *w).fold(0.0, f64::max);
  let total_height: f64 = parsed.iter().map(|(_, h, _, _)| *h).sum::<f64>()
    + (parsed.len() as f64 - 1.0) * gap;

  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    max_width.ceil() as u32,
    total_height.ceil() as u32,
    max_width.ceil() as u32,
    total_height.ceil() as u32,
  ));

  let mut y = 0.0_f64;
  for (w, h, vb, content) in &parsed {
    // Center horizontally if narrower than max
    let x = (max_width - w) / 2.0;
    svg.push_str(&format!(
      "<svg x=\"{:.0}\" y=\"{:.0}\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"{}\">\n",
      x, y, w, h, vb
    ));
    svg.push_str(content);
    svg.push_str("</svg>\n");
    y += h + gap;
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
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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
              .map(|(_, v)| v.clone())
              .unwrap_or(Expr::FunctionCall {
                name: "Missing".to_string(),
                args: vec![].into(),
              })
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
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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
fn parse_svg_numeric_attr(svg: &str, attr: &str) -> Option<f64> {
  // Only consider the first `<svg ...>` opening tag to avoid matching
  // attributes on nested cells.
  let tag_end = svg.find('>')?;
  let header = &svg[..tag_end];
  let needle = format!("{attr}=\"");
  let start = header.find(&needle)? + needle.len();
  let rel_end = header[start..].find('"')?;
  let raw = header[start..start + rel_end].trim();
  let numeric_end = raw
    .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
    .unwrap_or(raw.len());
  raw[..numeric_end].parse().ok()
}

/// Parse width, height, and viewBox from an SVG string
fn parse_svg_dimensions(svg: &str) -> Option<ParsedSvg> {
  // Extract viewBox attribute
  let vb_start = svg.find("viewBox=\"")?;
  let vb_value_start = vb_start + "viewBox=\"".len();
  let vb_end = svg[vb_value_start..].find('"')? + vb_value_start;
  let view_box = svg[vb_value_start..vb_end].to_string();

  // Parse viewBox to get dimensions: "x y w h"
  let parts: Vec<f64> = view_box
    .split_whitespace()
    .filter_map(|s| s.parse().ok())
    .collect();
  if parts.len() < 4 {
    return None;
  }
  let (vb_w, vb_h) = (parts[2], parts[3]);

  // Extract inner content (everything between first > and last </svg>)
  let inner_start = svg.find('>')? + 1;
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
        "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
         font-size=\"{font_size:.1}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
         dominant-baseline=\"central\">,</text>\n",
        comma_center,
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
    "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size:.1}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">}}</text>\n",
    close_x,
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
  if rows.iter().all(|r| r.is_empty()) {
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

  if parsed_rows.iter().all(|r| r.is_empty()) {
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
  let max_cols = parsed_rows.iter().map(|r| r.len()).max().unwrap_or(1);
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

  // Per-cell layout: keep natural aspect ratios, vertically center
  // shorter cells within their row.
  let mut row_layouts: Vec<GridRowLayout> = Vec::new();

  for parsed_row in &parsed_rows {
    if parsed_row.is_empty() {
      row_layouts.push(GridRowLayout {
        cells: Vec::new(),
        row_h: 0.0,
      });
      continue;
    }

    // Scaled per-cell dimensions (enforce a minimum so pathological
    // zero-sized inputs don't vanish entirely).
    let cell_dims: Vec<(f64, f64)> = parsed_row
      .iter()
      .map(|(_, _, nw, nh)| ((nw * scale).max(1.0), (nh * scale).max(1.0)))
      .collect();

    let row_h = cell_dims
      .iter()
      .map(|(_, h)| *h)
      .fold(0.0_f64, f64::max)
      .max(10.0);

    // Horizontal gap: Scaled is resolved against the average cell width.
    let avg_cell_w: f64 =
      cell_dims.iter().map(|(w, _)| *w).sum::<f64>() / cell_dims.len() as f64;
    let h_gap = opts.h_spacing.to_px(avg_cell_w);

    let mut x = 0.0_f64;
    let mut cells = Vec::with_capacity(parsed_row.len());
    for ((vb, inner, _, _), (cw, ch)) in parsed_row.iter().zip(cell_dims.iter())
    {
      let y_off = ((row_h - ch) / 2.0).max(0.0);
      cells.push(LayoutCell {
        x,
        y_off,
        w: *cw,
        h: *ch,
        view_box: vb.clone(),
        inner: inner.clone(),
      });
      x += cw + h_gap;
    }

    row_layouts.push(GridRowLayout { cells, row_h });
  }

  // Compute total dimensions.
  let total_width = row_layouts
    .iter()
    .map(|r| r.cells.last().map_or(0.0, |c: &LayoutCell| c.x + c.w))
    .fold(0.0_f64, f64::max);
  let v_gap = if !row_layouts.is_empty() {
    let avg_h = row_layouts.iter().map(|r| r.row_h).sum::<f64>()
      / row_layouts.len().max(1) as f64;
    opts.v_spacing.to_px(avg_h)
  } else {
    0.0
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
    .map(|r| &r.cells[..])
    .unwrap_or(&[]);
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
  let outer_items = match grid_list_ref {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GraphicsGrid expects a list of lists as its first argument".into(),
      ));
    }
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
  if rows.iter().all(|r| r.is_empty()) {
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
              .map(|(_, v)| v.clone())
              .unwrap_or(Expr::FunctionCall {
                name: "Missing".to_string(),
                args: vec![].into(),
              })
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
          items.get(i).cloned().unwrap_or(Expr::FunctionCall {
            name: "Missing".to_string(),
            args: vec![].into(),
          })
        } else if i == 0 {
          v.clone()
        } else {
          Expr::FunctionCall {
            name: "Missing".to_string(),
            args: vec![].into(),
          }
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
  let num_cols = if !col_keys.is_empty() {
    col_keys.len()
  } else {
    grid.iter().map(|r| r.len()).max().unwrap_or(0)
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
  let max_row_digits = format!("{}", num_data_rows).len().max(1) as f64;
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
  let num_header_rows = if show_headers && !col_keys.is_empty() {
    1
  } else {
    0
  };
  let total_height: f64 = if num_header_rows > 0 {
    header_row_height + (num_data_rows as f64) * row_height
  } else {
    (num_data_rows as f64) * row_height
  };

  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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

  // Parse optional alignment from second arg (default: Left)
  let alignment = if args.len() >= 2 {
    match &args[1] {
      Expr::Identifier(s) if s == "Center" => "middle",
      Expr::Identifier(s) if s == "Right" => "end",
      _ => "start", // Left or anything else
    }
  } else {
    "start"
  };

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
  let text_row_height = font_size + pad_y;
  let gap = spacing_ems * font_size;

  // An item is either a pre-rendered SVG (e.g. nested TableForm/Framed/Grid)
  // or a plain expression rendered as a single text line.
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
    .map(|item| match item {
      Expr::Graphics { svg, .. } => {
        let (w, h) = parse_svg_wh(svg);
        Cell::Svg {
          svg: svg.clone(),
          width: w,
          height: h,
        }
      }
      _ => Cell::Text(item.clone()),
    })
    .collect();

  // Compute column width from widest item
  let col_width: f64 = cells
    .iter()
    .map(|c| match c {
      Cell::Svg { width, .. } => *width,
      Cell::Text(e) => estimate_display_width(e) * char_width + pad_x,
    })
    .fold(0.0_f64, f64::max);

  // Per-row heights
  let row_heights: Vec<f64> = cells
    .iter()
    .map(|c| match c {
      Cell::Svg { height, .. } => *height,
      Cell::Text(_) => text_row_height,
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
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
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
        svg.push_str(&format!(
          "<text x=\"{text_x:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"{alignment}\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(expr)
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
        svg.push_str(&format!(
          "<svg x=\"{x_off:.1}\" y=\"{y_cursor:.1}\" width=\"{cw:.1}\" height=\"{ch:.1}\">\n"
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
  let items = match &args[0] {
    Expr::List(items) => items,
    _ => return None,
  };

  if items.is_empty() {
    return None;
  }

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_y: f64 = 8.0;

  // Determine separator: either Spacer[n] (pixel gap) or rendered expression
  enum Separator {
    Gap(f64),          // pixel gap (from Spacer[n])
    Text(String, f64), // rendered text and its width
  }

  let separator = if args.len() >= 2 {
    if let Some(pts) = crate::syntax::spacer_width_pts(&args[1]) {
      Separator::Gap(pts)
    } else {
      let text = expr_to_svg_markup(&args[1]);
      let w = estimate_display_width(&args[1]) * char_width;
      Separator::Text(text, w)
    }
  } else {
    Separator::Gap(0.0) // no separator
  };

  // Compute item widths
  let item_widths: Vec<f64> = items
    .iter()
    .map(|item| estimate_display_width(item) * char_width)
    .collect();

  let sep_width = match &separator {
    Separator::Gap(g) => *g,
    Separator::Text(_, w) => *w,
  };

  let items_width: f64 = item_widths.iter().sum();
  let seps_total = if items.len() > 1 {
    (items.len() - 1) as f64 * sep_width
  } else {
    0.0
  };
  let total_w = items_width + seps_total;
  let total_h = font_size + pad_y;

  let svg_w = total_w.ceil().max(1.0) as u32;
  let svg_h = total_h.ceil() as u32;

  let mut svg = String::with_capacity(1024);
  svg.push_str(&format!(
    "<svg width=\"{svg_w}\" height=\"{svg_h}\" viewBox=\"0 0 {svg_w} {svg_h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  ));

  let mid_y = total_h / 2.0;
  let text_fill = theme().text_primary;

  let mut x: f64 = 0.0;
  for (i, item) in items.iter().enumerate() {
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

    let cx = x + item_widths[i] / 2.0;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(item)
    ));
    x += item_widths[i];
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// Render `Framed[expr]` as an SVG box with a rectangular border around the content.
/// Handles nested Framed by recursively rendering inner content as embedded SVG.
pub fn framed_to_svg(args: &[Expr]) -> Option<String> {
  if args.is_empty() {
    return None;
  }

  let content = &args[0];

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
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"none\" stroke=\"{framed_border}\" stroke-width=\"{stroke_width}\"/>\n",
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
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"none\" stroke=\"{framed_border}\" stroke-width=\"{stroke_width}\"/>\n",
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
    .map(|c| c.to_svg_rgb())
    .unwrap_or_else(|| theme().highlighted_bg.to_string());

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
        "Highlighted" => highlighted_to_svg(inner_args)
          .map(|svg| {
            let (w, h) = parse_svg_wh(&svg);
            (Some(svg), w, h)
          })
          .unwrap_or((None, 0.0, 0.0)),
        "Framed" => framed_to_svg(inner_args)
          .map(|svg| {
            let (w, h) = parse_svg_wh(&svg);
            (Some(svg), w, h)
          })
          .unwrap_or((None, 0.0, 0.0)),
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
  let start = svg.find('>').map(|i| i + 1).unwrap_or(0);
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
      return Ok(Expr::FunctionCall {
        name: "KochCurve".to_string(),
        args: args.to_vec().into(),
      });
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

  Ok(Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(point_exprs.into())].into(),
  })
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
    .map(|i| crate::functions::make_rational_pub(i as i128, denom))
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
        op: crate::syntax::UnaryOperator::Minus,
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
    return Ok(Expr::FunctionCall {
      name: "DropShadowing".to_string(),
      args: args.to_vec().into(),
    });
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
          crate::functions::make_rational_pub(1, 3),
          Expr::FunctionCall {
            name: "ThemeColor".to_string(),
            args: vec![Expr::Identifier("Foreground".to_string())].into(),
          },
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
      Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![Expr::Integer(0)].into(),
      },
      Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![Expr::Integer(1)].into(),
      },
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
        return Ok(Expr::FunctionCall {
          name: "LinearGradientFilling".to_string(),
          args: vec![other.clone(), angle, space].into(),
        });
      }
    }
  };

  // Build: LinearGradientFilling[{stops} -> {colors}, angle, space]
  let rule = Expr::Rule {
    pattern: Box::new(Expr::List(stops.into())),
    replacement: Box::new(Expr::List(colors.into())),
  };

  Ok(Expr::FunctionCall {
    name: "LinearGradientFilling".to_string(),
    args: vec![rule, angle, space].into(),
  })
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

  Ok(Expr::FunctionCall {
    name: "Manipulate".to_string(),
    args: out_args.into(),
  })
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
    // Try to evaluate bounds; if evaluation fails, keep the original so
    // the echoed form still round-trips.
    let evaluated =
      evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone());
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
      // A trailing control option such as `ControlType -> None` is not a
      // range, so it must not be wrapped in Dynamic[…].
      Expr::Rule { .. } | Expr::RuleDelayed { .. } => false,
      _ => true,
    }
    && needs_dynamic
  {
    let range = new_items.pop().unwrap();
    new_items.push(Expr::FunctionCall {
      name: "Dynamic".to_string(),
      args: vec![range].into(),
    });
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
  Ok(Expr::FunctionCall {
    name: "Control".to_string(),
    args: out_args.into(),
  })
}

// ─────────────────────────────────────────────────────────────────
// Interactive Manipulate support (for Woxi Playground / Woxi Studio)
// ─────────────────────────────────────────────────────────────────

/// A styled run of a control label. A label is a sequence of these so the
/// UI can render `Style["t", Italic]` as an italic `t` while leaving the
/// rest upright. Only italic is tracked — the styling Wolfram labels use in
/// practice for slider captions.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelRun {
  pub text: String,
  pub italic: bool,
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
  },
  Discrete {
    name: String,
    /// Each discrete value rendered as InputForm — echoed back into the
    /// variable binding. For a rule-form choice `value -> "label"` this is
    /// the left side (the actual value), never the whole rule.
    values: Vec<String>,
    /// The display label for each choice, parallel to `values`. For a plain
    /// choice this equals the value's InputForm; for a rule-form choice it is
    /// the (unquoted) right side of the rule.
    value_labels: Vec<String>,
    initial_index: usize,
    label: String,
    label_runs: Vec<LabelRun>,
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
}

impl ManipulateControl {
  /// The bound variable name for this control.
  pub fn name(&self) -> &str {
    match self {
      ManipulateControl::Continuous { name, .. }
      | ManipulateControl::Discrete { name, .. }
      | ManipulateControl::Slider2D { name, .. }
      | ManipulateControl::IntervalSlider { name, .. } => name,
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
  /// Whether this spec should auto-play, i.e. it came from `Animate[…]` or
  /// `ListAnimate[…]`. An animated widget advances its first continuous
  /// control on a timer (with a play/pause toggle) instead of sitting still
  /// until the user drags a slider. Plain `Manipulate`/`Control` widgets leave
  /// this `false`.
  pub animated: bool,
}

/// Result of parsing a single list-shaped Manipulate argument.
enum ParsedControl {
  /// A control that renders a UI element (slider or pick list). The second
  /// field is an optional `Enabled` condition (InputForm code) that gates the
  /// widget: when it evaluates to `False` against the live bindings the
  /// control is shown greyed-out and non-interactive.
  Visible(ManipulateControl, Option<String>),
  /// A `Locator` control with no widget. It contributes a fixed `name =
  /// value` binding that is baked directly into the body so the variable is
  /// in scope while the visible controls drive the plot.
  Fixed { name: String, value: String },
  /// A `ControlType -> None` variable: no widget, but a *mutable* binding
  /// passed live in the binding set so an interactive display element (a
  /// `Checkbox`, `Setter`, …) can write back into it.
  State { name: String, value: String },
}

/// Attempt to extract a `ManipulateSpec` from a held `Manipulate[…]` or
/// `Animate[…]` expression. `Animate` shares `Manipulate`'s argument shape
/// (a body followed by `{u, umin, umax}`-style control specs) but auto-plays,
/// so it produces the same spec with `animated` set. Returns `None` if the
/// expression is not a well-formed Manipulate/Animate (e.g. `Manipulate[]`,
/// `Manipulate[expr]`, or a spec that isn't a list). In those cases the caller
/// should fall back to the standard text/graphics output path.
pub fn extract_manipulate_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
  };
  if (name != "Manipulate" && name != "Animate") || args.len() < 2 {
    return None;
  }
  let animated = name == "Animate";

  let body_code = crate::syntax::expr_to_input_form(&args[0]);
  let mut controls = Vec::with_capacity(args.len() - 1);
  // `Locator` bindings are baked into the body (never rewritten by a
  // display); `ControlType -> None` bindings become live mutable state.
  let mut fixed: Vec<(String, String)> = Vec::new();
  let mut state: Vec<(String, String)> = Vec::new();
  let mut displays: Vec<String> = Vec::new();
  let mut initialization: Option<String> = None;
  let mut control_enabled: Vec<(String, String)> = Vec::new();
  for spec in &args[1..] {
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
      continue;
    }
    // Only list-shaped arguments are control specs. Any other trailing
    // argument (e.g. a `Dynamic[Panel[…]]` of checkboxes) is an extra
    // display element: capture it so the frontend can render it live.
    if !matches!(spec, Expr::List(_)) {
      displays.push(crate::syntax::expr_to_input_form(spec));
      continue;
    }
    match parse_manipulate_control(spec)? {
      ParsedControl::Visible(c, enabled) => {
        if let Some(cond) = enabled {
          control_enabled.push((c.name().to_string(), cond));
        }
        controls.push(c);
      }
      ParsedControl::Fixed { name, value } => fixed.push((name, value)),
      ParsedControl::State { name, value } => state.push((name, value)),
    }
  }

  // A Manipulate with no controls or state at all (e.g. `Manipulate[x^2,
  // badspec]`, where `badspec` is neither a spec nor an option) isn't
  // renderable as an interactive widget — fall back to the plain path.
  if controls.is_empty() && fixed.is_empty() && state.is_empty() {
    return None;
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
    animated,
  })
}

/// Attempt to extract a `ManipulateSpec` from a held `ListAnimate[{e1, …, en}]`
/// expression. Each element is one animation frame; the widget cycles through
/// them by binding an integer frame index and displaying the selected element.
/// It renders as an auto-playing single-slider Manipulate whose body is
/// `Part[{e1, …, en}, i]`. Returns `None` when the argument is not a non-empty
/// list literal (e.g. `ListAnimate[expr]` or `ListAnimate[{}]`), so the caller
/// falls back to the plain output path.
pub fn extract_list_animate_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
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
  let body_code = format!("Part[{}, Round[i]]", list_code);
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
    }],
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    animated: true,
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
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
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
    var = s.clone();
    idx = 1;
  }
  let (min, max, step, initial) = match args.get(idx) {
    Some(Expr::List(items)) if !items.is_empty() => {
      let min = crate::functions::math_ast::try_eval_to_f64(&items[0])?;
      let max = items
        .get(1)
        .and_then(crate::functions::math_ast::try_eval_to_f64)
        .unwrap_or(min + 1.0);
      let step = items
        .get(2)
        .and_then(crate::functions::math_ast::try_eval_to_f64);
      (min, max, step, min)
    }
    // A single number is the initial value over the default 0..1 range.
    Some(other) if idx == 0 => {
      let init = crate::functions::math_ast::try_eval_to_f64(other)?;
      (0.0, 1.0, None, init)
    }
    // `Animator[]` / `Animator[Dynamic[v]]`: default 0..1 range.
    None => (0.0, 1.0, None, 0.0),
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
    }],
  };
  Some(ManipulateSpec {
    body_code: var,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    animated: true,
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
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
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
    pt => match list2_f64(pt) {
      Some(p) => ("p".to_string(), Some(p)),
      None => return None,
    },
  };
  let body_code = crate::syntax::expr_to_input_form(&args[1]);
  let ((x_min, y_min), (x_max, y_max)) = pane_range(args.get(2));
  // Start the locator at the given point, else the range centre.
  let (x_initial, y_initial) =
    explicit_init.unwrap_or(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0));
  let control = ManipulateControl::Slider2D {
    name: var.clone(),
    x_min,
    x_max,
    y_min,
    y_max,
    x_initial,
    y_initial,
    label: var,
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    animated: false,
  })
}

/// Attempt to extract a `ManipulateSpec` from a held `ClickPane[…]`.
/// A click pane applies a handler to the coordinates of each click. We model
/// it as a 2D pad whose position feeds the handler: `ClickPane[expr, func]`
/// (and `ClickPane[expr, {{xmin, ymin}, {xmax, ymax}}, func]`) render as a
/// clickable/draggable pad with the live `func[{x, y}]` result shown beside
/// it. Returns `None` if there's no handler to apply.
pub fn extract_click_pane_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
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
  let body_code = format!("({})[pos]", func_code);
  let control = ManipulateControl::Slider2D {
    name: "pos".to_string(),
    x_min,
    x_max,
    y_min,
    y_max,
    x_initial: (x_min + x_max) / 2.0,
    y_initial: (y_min + y_max) / 2.0,
    label: "pos".to_string(),
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled: Vec::new(),
    animated: false,
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
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
  };
  if name != "Control" || args.is_empty() {
    return None;
  }
  // The first argument is the variable specification; any trailing options
  // are ignored for rendering purposes.
  if !matches!(&args[0], Expr::List(items) if !items.is_empty()) {
    return None;
  }
  let (control, enabled) = match parse_manipulate_control(&args[0])? {
    ParsedControl::Visible(c, enabled) => (c, enabled),
    // A hidden control (`ControlType -> None` / Locator) has no widget and
    // nothing to display on its own — fall back to the plain output path.
    ParsedControl::Fixed { .. } | ParsedControl::State { .. } => return None,
  };
  // Display the bound variable so the control's effect is visible.
  let body_code = control.name().to_string();
  let control_enabled = match enabled {
    Some(cond) => vec![(control.name().to_string(), cond)],
    None => Vec::new(),
  };
  Some(ManipulateSpec {
    body_code,
    controls: vec![control],
    state: Vec::new(),
    displays: Vec::new(),
    initialization: None,
    control_enabled,
    animated: false,
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
fn list2_f64(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::List(l) if l.len() == 2 => {
      let a = crate::functions::math_ast::try_eval_to_f64(&l[0])?;
      let b = crate::functions::math_ast::try_eval_to_f64(&l[1])?;
      Some((a, b))
    }
    _ => None,
  }
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
fn manipulate_label_runs(expr: &Expr, italic: bool) -> Vec<LabelRun> {
  let output_run = |italic: bool| {
    let text =
      crate::syntax::format_expr(expr, crate::syntax::ExprForm::Output);
    if text.is_empty() {
      vec![]
    } else {
      vec![LabelRun { text, italic }]
    }
  };
  match expr {
    Expr::FunctionCall { name, args } => match name.as_str() {
      // Style[expr, dir…] — render `expr`, turning italic on if any directive
      // asks for it (bare `Italic` or `FontSlant -> "Italic"`).
      "Style" => {
        let styled = italic || args.iter().skip(1).any(is_italic_directive);
        args
          .first()
          .map(|a| manipulate_label_runs(a, styled))
          .unwrap_or_default()
      }
      // Presentation wrappers whose content may nest styling/subscripts —
      // recurse rather than defer to OutputForm.
      "Text" | "DisplayForm" | "TraditionalForm" => args
        .first()
        .map(|a| manipulate_label_runs(a, italic))
        .unwrap_or_default(),
      // Row[{a, b, …}] — concatenate the (recursively rendered) parts.
      "Row" => match args.first() {
        Some(Expr::List(parts)) => parts
          .iter()
          .flat_map(|p| manipulate_label_runs(p, italic))
          .collect(),
        _ => output_run(italic),
      },
      "Subscript" => script_runs(args, italic, false),
      "Superscript" => script_runs(args, italic, true),
      _ => output_run(italic),
    },
    _ => output_run(italic),
  }
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
    });
  }
  runs
}

/// True when a `Style` directive requests italic: bare `Italic` or
/// `FontSlant -> "Italic" | Italic`.
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

/// Map the characters of `s` to their Unicode sub-/superscript form when a
/// mapping exists, leaving other characters unchanged. Used to render
/// `Subscript`/`Superscript` control labels inline.
fn to_unicode_script(s: &str, superscript: bool) -> String {
  s.chars()
    .map(|c| unicode_script_char(c, superscript).unwrap_or(c))
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
      'n' => '\u{207F}',
      'i' => '\u{2071}',
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
      _ => return None,
    }
  };
  Some(mapped)
}

/// Parse a single variable-spec list into a `ParsedControl`.
fn parse_manipulate_control(spec: &Expr) -> Option<ParsedControl> {
  let items = match spec {
    Expr::List(items) => items,
    _ => return None,
  };
  if items.is_empty() {
    return None;
  }

  // Head can be either a plain symbol `u` or `{u, uinit}` / `{u, uinit, ulbl}`.
  let plain_run = |s: String| {
    vec![LabelRun {
      text: s,
      italic: false,
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
      let lbl = match head_items.get(2) {
        Some(Expr::String(s)) => plain_run(s.clone()),
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

  // Hidden controls: a `Locator` marker (`{{p, init}, pmin, pmax, Locator}`)
  // or `ControlType -> None` (`{{v, init}, ControlType -> None}`). Neither
  // renders a UI element in the static widget; both bind their variable to
  // the (evaluated) initial value so the body can reference it.
  let is_locator = items
    .iter()
    .any(|it| matches!(it, Expr::Identifier(s) if s == "Locator"));
  let is_hidden = items.iter().any(|it| {
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
    let value = manipulate_value_to_input_form(&value_expr);
    // A Locator's initial point list is baked into the body (it is never
    // rewritten by a display); a `ControlType -> None` variable stays a
    // live, mutable binding so an interactive display can rewrite it.
    return Some(if is_locator {
      ParsedControl::Fixed { name, value }
    } else {
      ParsedControl::State { name, value }
    });
  }

  // A `ControlType -> Slider2D` / `ControlType -> IntervalSlider` option
  // selects a compound control. The bounds are the non-option items after
  // the head.
  let control_type = items.iter().find_map(|it| match it {
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
  });
  let bounds: Vec<&Expr> = items[1..]
    .iter()
    .filter(|it| !matches!(it, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .collect();

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
      let mn = bounds
        .first()
        .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e))?;
      let mx = bounds
        .get(1)
        .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e))
        .unwrap_or(mn + 1.0);
      (mn, mx, mn, mx)
    };
    let (x_initial, y_initial) =
      match explicit_initial.as_ref().and_then(list2_f64) {
        Some((a, b)) => (a, b),
        None => (x_min, y_min),
      };
    return Some(ParsedControl::Visible(
      ManipulateControl::Slider2D {
        name,
        x_min,
        x_max,
        y_min,
        y_max,
        x_initial,
        y_initial,
        label,
      },
      enabled,
    ));
  }

  // Interval control: `{u, min, max, ControlType -> IntervalSlider}` binds
  // `u` to a `{low, high}` pair.
  if control_type.as_deref() == Some("IntervalSlider") {
    let min = bounds
      .first()
      .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e))?;
    let max = bounds
      .get(1)
      .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e))
      .unwrap_or(min + 1.0);
    let step = bounds
      .get(2)
      .and_then(|e| crate::functions::math_ast::try_eval_to_f64(e));
    let (low_initial, high_initial) =
      match explicit_initial.as_ref().and_then(list2_f64) {
        Some((a, b)) => (a, b),
        None => (min, max),
      };
    return Some(ParsedControl::Visible(
      ManipulateControl::IntervalSlider {
        name,
        min,
        max,
        step,
        low_initial,
        high_initial,
        label,
      },
      enabled,
    ));
  }

  // Discrete form: `{u, {u1, u2, …}}` or `{{u, uinit, …}, {u1, u2, …}}`.
  // The value list may also be given as an expression that evaluates to a
  // list (e.g. `{g, PolyhedronData[All]}`), so evaluate a non-literal.
  if items.len() == 2 {
    let value_items: Option<Vec<Expr>> = match &items[1] {
      Expr::List(vs) => Some(vs.iter().cloned().collect()),
      other => match crate::evaluator::evaluate_expr_to_expr(other) {
        Ok(Expr::List(ref vs)) => Some(vs.iter().cloned().collect()),
        _ => None,
      },
    };
    if let Some(value_items) = value_items {
      // A choice may be given as a rule `value -> "label"` (e.g. a SetterBar
      // spec `{True -> "Yin-Yang", False -> "alternate image"}`). In that
      // case the left side is the value bound to the variable and the right
      // side is only the display label. Split the two so the binding sees the
      // real value, not the whole rule.
      let (values, value_labels): (Vec<String>, Vec<String>) = value_items
        .iter()
        .map(|item| match discrete_choice_rule(item) {
          Some((value, label)) => (
            crate::syntax::expr_to_input_form(value),
            discrete_choice_label(label),
          ),
          None => (
            crate::syntax::expr_to_input_form(item),
            discrete_choice_label(item),
          ),
        })
        .unzip();
      if values.is_empty() {
        return None;
      }
      let initial_index = match explicit_initial {
        Some(init) => {
          let init_code = crate::syntax::expr_to_input_form(&init);
          values.iter().position(|v| *v == init_code).unwrap_or(0)
        }
        None => 0,
      };
      return Some(ParsedControl::Visible(
        ManipulateControl::Discrete {
          name,
          values,
          value_labels,
          initial_index,
          label,
          label_runs,
        },
        enabled,
      ));
    }
  }

  // Continuous form: {u, umin, umax} or {u, umin, umax, du}
  // (or with labelled head: {{u, uinit, ulbl}, umin, umax, …})
  if items.len() < 3 {
    return None;
  }
  let min = crate::functions::math_ast::try_eval_to_f64(&items[1])?;
  let max = crate::functions::math_ast::try_eval_to_f64(&items[2])?;
  let step = items
    .get(3)
    .and_then(crate::functions::math_ast::try_eval_to_f64);
  let initial = match explicit_initial.as_ref() {
    Some(init) => {
      crate::functions::math_ast::try_eval_to_f64(init).unwrap_or(min)
    }
    None => min,
  };

  Some(ParsedControl::Visible(
    ManipulateControl::Continuous {
      name,
      min,
      max,
      step,
      initial,
      label,
      label_runs,
    },
    enabled,
  ))
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
/// surrounding quotes; anything else falls back to its InputForm.
fn discrete_choice_label(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_input_form(other),
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
    .map(|c| match c {
      ManipulateControl::Continuous { name, initial, .. } => {
        (name.clone(), format_f64_input(*initial))
      }
      ManipulateControl::Discrete {
        name,
        values,
        initial_index,
        ..
      } => (
        name.clone(),
        values
          .get(*initial_index)
          .cloned()
          .unwrap_or_else(|| "Null".to_string()),
      ),
      ManipulateControl::Slider2D {
        name,
        x_initial,
        y_initial,
        ..
      } => (
        name.clone(),
        format!(
          "{{{}, {}}}",
          format_f64_input(*x_initial),
          format_f64_input(*y_initial)
        ),
      ),
      ManipulateControl::IntervalSlider {
        name,
        low_initial,
        high_initial,
        ..
      } => (
        name.clone(),
        format!(
          "{{{}, {}}}",
          format_f64_input(*low_initial),
          format_f64_input(*high_initial)
        ),
      ),
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
    format!("{}", v)
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
    .map(|(name, value)| format!("{} = {}", name, value))
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
    let (key, next) = match parse_json_string(bytes, i) {
      Some(v) => v,
      None => break,
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
      format!(
        r#"{{"text":"{}","italic":{}}}"#,
        json_escape_manipulate(&r.text),
        r.italic,
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
      } => {
        let step_json = match step {
          Some(s) => format!(r#","step":{}"#, s),
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
        initial_index,
        label,
        label_runs,
      } => {
        let value_parts: Vec<String> = values
          .iter()
          .map(|v| format!(r#""{}""#, json_escape_manipulate(v)))
          .collect();
        let label_parts: Vec<String> = value_labels
          .iter()
          .map(|v| format!(r#""{}""#, json_escape_manipulate(v)))
          .collect();
        ctrl_parts.push(format!(
          r#"{{"kind":"discrete","name":"{}","label":"{}","labelRuns":{},"values":[{}],"valueLabels":[{}],"initialIndex":{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          label_runs_to_json(label_runs),
          value_parts.join(","),
          label_parts.join(","),
          initial_index,
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
          Some(s) => format!(r#","step":{}"#, s),
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
    r#","animated":true"#
  } else {
    ""
  };

  format!(
    r#""body":"{}","controls":[{}],"state":{{{}}},"displays":[{}]{}"#,
    json_escape_manipulate(&spec.body_code),
    ctrl_parts.join(","),
    state_parts.join(","),
    display_parts.join(","),
    animated_json,
  )
}

/// A node in a rendered Manipulate extra-display widget tree. Both frontends
/// consume this: the Playground via `render_manipulate_display` (JSON), the
/// Studio via `build_manipulate_display` (this enum directly).
#[derive(Debug, Clone)]
pub enum DisplayNode {
  /// A framed container wrapping a single child (`Panel`, `Framed`, …).
  Panel(Box<DisplayNode>),
  /// A 2D grid of cells (`Grid`).
  Grid(Vec<Vec<DisplayNode>>),
  /// A vertical stack (`Column`, or a bare list).
  Column(Vec<DisplayNode>),
  /// A horizontal stack (`Row`).
  Row(Vec<DisplayNode>),
  /// A checkbox. `target` is the InputForm of the write-back lvalue (e.g.
  /// `data[[3, 5]]`), `None` for a non-interactive checkbox; `checked` is its
  /// current state; `on`/`off` are the InputForm values a toggle writes back.
  Checkbox {
    target: Option<String>,
    checked: bool,
    on: String,
    off: String,
  },
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
  let expr = match crate::interpret_to_expr(display_code) {
    Ok(e) => e,
    Err(_) => {
      return DisplayNode::Static {
        svg: None,
        text: String::new(),
      };
    }
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
      _ => static_leaf_node(expr, bindings),
    },
    // A bare list of display elements stacks vertically, like `Column`.
    Expr::List(_) => {
      DisplayNode::Column(list_children(expr, bindings, probes, ons))
    }
    _ => static_leaf_node(expr, bindings),
  }
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
    DisplayNode::Checkbox { checked, .. } => {
      if let Some(f) = flags.get(*idx) {
        *checked = *f;
      }
      *idx += 1;
    }
    DisplayNode::Static { .. } => {}
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

#[cfg(test)]
mod manipulate_label_tests {
  use super::*;
  use crate::syntax::Expr;

  fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::FunctionCall {
      name: name.to_string(),
      args: args.into(),
    }
  }

  fn runs(expr: &Expr) -> Vec<LabelRun> {
    manipulate_label_runs(expr, false)
  }

  fn run(text: &str, italic: bool) -> LabelRun {
    LabelRun {
      text: text.to_string(),
      italic,
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
    let label = call("Text", vec![subscript]);
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
    let row = call(
      "Row",
      vec![Expr::List(vec![italic_a, Expr::String("b".into())].into())],
    );
    assert_eq!(runs(&row), vec![run("a", true), run("b", false)]);
    assert_eq!(flatten_label_runs(&runs(&row)), "ab");
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
  points: Vec<(i64, i64)>,
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
  Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(point_exprs.into())].into(),
  }
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
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "HilbertCurve".to_string(),
      args: args.to_vec().into(),
    })
  };
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
  Ok(space_filling_curve("HilbertCurve", args, side, points))
}

/// PeanoCurve[n] — one Line through all 9^n cells of the 3^n × 3^n grid in
/// Peano order. Coordinates come from Peano's digit construction: index
/// digits alternate y/x roles (y first), and each digit is complemented
/// (d → 2 - d) when the sum of the preceding other-coordinate digits is
/// odd — which reproduces wolframscript's serpentine orientation.
pub fn peano_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "PeanoCurve".to_string(),
      args: args.to_vec().into(),
    })
  };
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
  Ok(space_filling_curve("PeanoCurve", args, side, points))
}

/// SierpinskiCurve[n] — the closed Sierpiński square curve as a Line,
/// generated by Wirth's classic four-procedure recursion
///   A: A↘B→D↗A   B: B↙C↓A↘B   C: C↖D←B↙C   D: D↗A↑C↖D
/// glued as A↘B↙C↖D↗ (closed), with a fixed half-step of 32 — diagonal
/// moves are (±32, ±32) and axis moves 64 — matching wolframscript's
/// absolute integer coordinates at every order.
pub fn sierpinski_curve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "SierpinskiCurve".to_string(),
      args: args.to_vec().into(),
    })
  };
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
  Ok(Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(point_exprs.into())].into(),
  })
}
