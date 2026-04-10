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

#[derive(Debug, Clone, Copy)]
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
        args: vec![Expr::Real(self.r)],
      }
    } else {
      Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(self.r), Expr::Real(self.g), Expr::Real(self.b)],
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
  font_size: f64,
  font_weight: &'static str,
  font_style: &'static str,
}

#[derive(Debug, Clone)]
struct EdgeForm {
  color: Option<Color>,
  thickness: Option<f64>,
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
      font_size: 14.0,
      font_weight: "normal",
      font_style: "normal",
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
      "Directive" => {
        for a in args {
          apply_directive(a, style);
        }
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
            // Handle Style[expr, Bold], Style[expr, Italic], Style[expr, fontSize]
            match directive {
              Expr::Identifier(s) if s == "Bold" => style.font_weight = "bold",
              Expr::Identifier(s) if s == "Italic" => {
                style.font_style = "italic"
              }
              Expr::Integer(n) => style.font_size = *n as f64,
              Expr::Real(f) => style.font_size = *f,
              _ => {}
            }
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
        "Polygon" if !args.is_empty() => {
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
        "Point" | "Line" | "Polygon" | "Arrow" | "BezierCurve"
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
            args: new_args,
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
    Expr::List(vec![Expr::Real(x), Expr::Real(y)])
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
  prims.push(Primitive::CircleArc {
    cx,
    cy,
    rx,
    ry,
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
    (1.0, 1.0)
  };
  prims.push(Primitive::RectPrim {
    x_min,
    y_min,
    x_max,
    y_max,
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

fn parse_text(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  // Text[str, {x, y}] or Text[Style[str, ...], {x, y}]
  let mut local_style = style.clone();
  let text = match &args[0] {
    Expr::String(s) => s.clone(),
    Expr::FunctionCall { name, args: sargs }
      if name == "Style" && !sargs.is_empty() =>
    {
      // Apply style directives
      for d in &sargs[1..] {
        apply_directive(d, &mut local_style);
        match d {
          Expr::Identifier(s) if s == "Bold" => {
            local_style.font_weight = "bold"
          }
          Expr::Identifier(s) if s == "Italic" => {
            local_style.font_style = "italic"
          }
          Expr::Integer(n) => local_style.font_size = *n as f64,
          Expr::Real(f) => local_style.font_size = *f,
          _ => {}
        }
      }
      match &sargs[0] {
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_string(other),
      }
    }
    other => crate::syntax::expr_to_string(other),
  };

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
fn bspline_basis(i: usize, k: usize, t: f64, knots: &[f64]) -> f64 {
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
  }
  bb
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
      style,
    } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *rx / bb.width() * svg_w;
      let sry = *ry / bb.height() * svg_h;
      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(&style.dashing, bb, svg_w);
      out.push_str(&format!(
        "<ellipse cx=\"{scx:.2}\" cy=\"{scy:.2}\" rx=\"{srx:.2}\" ry=\"{sry:.2}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{sw:.2}\"{}{}/>\n",
        color.to_svg_rgb(),
        color.opacity_attr(),
        dash,
      ));
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
          let head_len = (sw * 6.0).max(9.0);
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

      if text.contains('\n') {
        // Multi-line text with tspan
        let lines: Vec<&str> = text.split('\n').collect();
        out.push_str(&format!(
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\" text-anchor=\"middle\" dominant-baseline=\"central\"{}>",
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
          "<text x=\"{sx:.2}\" y=\"{sy:.2}\" fill=\"{}\" font-size=\"{fs}\" font-weight=\"{}\" font-style=\"{}\" text-anchor=\"middle\" dominant-baseline=\"central\"{}>{}</text>\n",
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
    _ => None,
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

  // Compute margins for axis tick labels when axes are enabled.
  let margin_left: f64 = if axes.1 { 50.0 } else { 0.0 };
  let margin_bottom: f64 = if axes.0 { 25.0 } else { 0.0 };
  let total_width = svg_w + margin_left;
  let total_height = svg_h + margin_bottom;

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

  // Background (covers the full SVG including margins)
  if let Some(bg) = background {
    svg.push_str(&format!(
      "<rect width=\"{total_width:.0}\" height=\"{total_height:.0}\" fill=\"{}\"/>\n",
      bg.to_svg_rgb(),
    ));
  }

  // Offset the drawing area so axes labels fit in the margins
  if margin_left > 0.0 || margin_bottom > 0.0 {
    svg.push_str(&format!(
      "<g transform=\"translate({margin_left:.0},0)\">\n"
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

  render_axes(&mut svg, axes, &bb, svg_w, svg_h);

  // Render primitives
  for prim in &primitives {
    render_primitive(prim, &bb, svg_w, svg_h, &mut svg);
  }

  if margin_left > 0.0 || margin_bottom > 0.0 {
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

impl BoxLayout {
  /// Create a layout for a simple text atom.
  fn text(s: &str, font_size: f64) -> Self {
    let ch = font_size * 0.6; // approximate monospace char width
    let w = s.chars().count() as f64 * ch;
    let ascent = font_size * 0.8; // approximate ascent
    let descent = font_size * 0.25; // approximate descent
    let height = ascent + descent;
    let escaped = svg_escape(s);
    BoxLayout {
      width: w,
      height,
      baseline: ascent,
      elements: format!(
        "<text x=\"0\" y=\"{ascent:.1}\" font-family=\"monospace\" font-size=\"{font_size:.1}\" stroke=\"none\">{escaped}</text>"
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
  let ch = font_size * 0.6;

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
        let children: Vec<BoxLayout> =
          items.iter().map(|e| layout_box(e, font_size)).collect();
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
          let dy = max_baseline - child.baseline;
          elements.push_str(&child.translate(x, dy));
          x += child.width;
          // Add space after bare comma
          if matches!(&items[i], Expr::String(s) if s == ",")
            && i + 1 < items.len()
          {
            x += ch * 0.5;
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

      // SqrtBox: √ symbol + overline (vinculum) above content
      "SqrtBox" if args.len() == 1 => {
        let content = layout_box(&args[0], font_size);
        let radical = BoxLayout::text("\u{221A}", font_size);
        let pad = 2.0; // space above content for the vinculum
        let line_y = pad * 0.5;
        let content_x = radical.width;
        let w = radical.width + content.width;
        let h = content.height + pad;
        let elements = format!(
          "{}\
           <line x1=\"{content_x:.1}\" y1=\"{line_y:.1}\" x2=\"{w:.1}\" y2=\"{line_y:.1}\" stroke=\"currentColor\" stroke-width=\"0.8\"/>\
           {}",
          radical.translate(0.0, pad),
          content.translate(content_x, pad),
        );
        BoxLayout {
          width: w,
          height: h,
          baseline: pad + content.baseline,
          elements,
        }
      }

      // RadicalBox: index + √ symbol + overline above content
      "RadicalBox" if args.len() == 2 => {
        let content = layout_box(&args[0], font_size);
        let index = layout_box(&args[1], font_size * 0.7);
        let radical = BoxLayout::text("\u{221A}", font_size);
        let pad = 2.0;
        let index_dy = 0.0;
        let body_dy = index.height * 0.3 + pad;
        let content_x = index.width + radical.width;
        let w = content_x + content.width;
        let line_y = body_dy - pad * 0.5;
        let elements = format!(
          "{}{}\
           <line x1=\"{content_x:.1}\" y1=\"{line_y:.1}\" x2=\"{w:.1}\" y2=\"{line_y:.1}\" stroke=\"currentColor\" stroke-width=\"0.8\"/>\
           {}",
          index.translate(0.0, index_dy),
          radical.translate(index.width, body_dy),
          content.translate(content_x, body_dy),
        );
        BoxLayout {
          width: w,
          height: body_dy + content.height,
          baseline: body_dy + content.baseline,
          elements,
        }
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
      "InterpretationBox" if args.len() == 2 => layout_box(&args[0], font_size),

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

/// Helper: group digits with thin spaces, returning plain text (not SVG).
fn group_digits_str(s: &str) -> String {
  // Just return the number as-is for simplicity in layout
  s.to_string()
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
                  args: fn_args[1..].to_vec(),
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
      "InterpretationBox" if args.len() == 2 => boxes_to_svg(&args[0]),

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
      "InterpretationBox" if args.len() == 2 => {
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
      Some((lo, hi)) => Expr::List(vec![Expr::Real(lo), Expr::Real(hi)]),
      None => Expr::Identifier("All".to_string()),
    }
  };

  Some(Expr::List(vec![range_to_expr(mx), range_to_expr(my)]))
}

/// Implementation of Show[g1, g2, ..., opts...].
/// Convert MeshRegion vertex/polygon data to Graphics primitives (Polygon with coordinates).
fn mesh_region_to_graphics_prims(
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
    args: vec![Color::gray(0.4).to_expr()],
  });
  result.push(Expr::FunctionCall {
    name: "FaceForm".to_string(),
    args: vec![Color::new(0.626, 0.836, 0.919).to_expr()],
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
                  Some(Expr::List(vec![Expr::Real(x), Expr::Real(y)]))
                } else {
                  None
                }
              })
            })
            .collect();
          if points.len() >= 3 {
            result.push(Expr::FunctionCall {
              name: "Polygon".to_string(),
              args: vec![Expr::List(points)],
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
          merged_primitives.push(Expr::List(graphics_prims));
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
          ],
        });
        if sd.is_scatter {
          series_prims.push(Expr::FunctionCall {
            name: "PointSize".to_string(),
            args: vec![Expr::Real(0.012)],
          });
          let coords: Vec<Expr> = sd
            .points
            .iter()
            .filter(|(_, y)| y.is_finite())
            .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)]))
            .collect();
          if !coords.is_empty() {
            series_prims.push(Expr::FunctionCall {
              name: "Point".to_string(),
              args: vec![Expr::List(coords)],
            });
          }
        } else {
          series_prims.push(Expr::FunctionCall {
            name: "AbsoluteThickness".to_string(),
            args: vec![Expr::Real(1.5)],
          });
          let segments =
            crate::functions::plot::split_into_segments(&sd.points);
          for seg in &segments {
            let coords: Vec<Expr> = seg
              .iter()
              .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)]))
              .collect();
            if coords.len() >= 2 {
              series_prims.push(Expr::FunctionCall {
                name: "Line".to_string(),
                args: vec![Expr::List(coords)],
              });
            }
          }
        }
      }
      merged_primitives.push(Expr::List(series_prims));

      // Merge range as PlotRange option
      let range_rule = Expr::Rule {
        pattern: Box::new(Expr::Identifier("PlotRange".to_string())),
        replacement: Box::new(Expr::List(vec![
          Expr::List(vec![Expr::Real(ps.x_range.0), Expr::Real(ps.x_range.1)]),
          Expr::List(vec![Expr::Real(ps.y_range.0), Expr::Real(ps.y_range.1)]),
        ])),
      };
      merge_option(&mut merged_options, &range_rule);

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
      args: args.to_vec(),
    });
  }

  let content = Expr::List(merged_primitives);
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
              cells.clone()
            } else {
              vec![row.clone()]
            }
          })
          .collect()
      } else {
        // 1D list → single row
        vec![items.clone()]
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
                        col_align_repeating = rep_items
                          .iter()
                          .map(|e| alignment_to_anchor(e))
                          .collect();
                      } else if before_repeat {
                        col_align_explicit_start
                          .push(alignment_to_anchor(spec));
                      }
                      // Note: trailing explicit not common for alignment
                    }
                  } else {
                    col_alignments = col_specs
                      .iter()
                      .map(|e| alignment_to_anchor(e))
                      .collect();
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
              row_headings = rh.clone();
            }
            if lists.len() >= 2
              && let Expr::List(ch) = &lists[1]
            {
              col_headings = ch.clone();
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
        args: vec![h, Expr::Identifier("Bold".to_string())],
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
              args: vec![h.clone(), Expr::Identifier("Bold".to_string())],
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
        let text_fill = if let Some(ref c) = cell_color {
          c.to_svg_rgb()
        } else if let Some(ref c) = default_style.color {
          c.to_svg_rgb()
        } else {
          theme().text_primary.to_string()
        };
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{fs}\"{fw_attr}{fst_attr} fill=\"{text_fill}\" text-anchor=\"{col_align}\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(content)
        ));
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
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
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

/// List of associations: multi-row table with column headers on top.
fn dataset_list_to_svg(items: &[Expr]) -> Option<String> {
  if items.is_empty() {
    return None;
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
    } else {
      return None; // Not a list of associations
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
              .unwrap_or(Expr::Identifier("Missing[]".to_string()))
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

  Some(ParsedSvg {
    view_box,
    inner_content,
  })
}

/// Combine multiple SVG strings arranged as rows of cells into a single SVG.
/// `rows` is a Vec of rows, each row is a Vec of SVG strings.
pub fn combine_graphics_svgs(rows: &[Vec<String>]) -> Option<String> {
  if rows.is_empty() {
    return None;
  }

  // Determine cell size based on grid dimensions
  let num_rows = rows.len();
  let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
  if num_cols == 0 {
    return None;
  }

  let gap = 4.0_f64;
  let cell_size = if num_rows == 1 { 100.0 } else { 80.0 };

  let total_width = num_cols as f64 * cell_size + (num_cols as f64 - 1.0) * gap;
  let total_height =
    num_rows as f64 * cell_size + (num_rows as f64 - 1.0) * gap;

  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    total_width.ceil() as u32,
    total_height.ceil() as u32,
    total_width.ceil() as u32,
    total_height.ceil() as u32,
  ));

  for (i, row) in rows.iter().enumerate() {
    for (j, cell_svg) in row.iter().enumerate() {
      if let Some(parsed) = parse_svg_dimensions(cell_svg) {
        let x = j as f64 * (cell_size + gap);
        let y = i as f64 * (cell_size + gap);
        svg.push_str(&format!(
          "<svg x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" viewBox=\"{}\">\n",
          x.round() as u32,
          y.round() as u32,
          cell_size as u32,
          cell_size as u32,
          parsed.view_box,
        ));
        svg.push_str(&parsed.inner_content);
        svg.push_str("</svg>\n");
      }
    }
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// Render a 1-D list of SVGs as `{ svg₁, svg₂, … }` with brace/comma text
/// interleaved between the nested graphic cells.
pub fn graphics_list_svg(svgs: &[String]) -> Option<String> {
  if svgs.is_empty() {
    return None;
  }

  let cell_size = 100.0_f64;
  let font_size = 18.0_f64;
  let brace_w = 12.0_f64; // width reserved for `{` / `}`
  let comma_w = 20.0_f64; // width reserved for `, `
  let height = cell_size;
  let text_y = height / 2.0;

  // Total width: { + cells with commas + }
  let n = svgs.len() as f64;
  let total_width = brace_w + n * cell_size + (n - 1.0) * comma_w + brace_w;

  let mut out = String::with_capacity(4096);
  out.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n",
    total_width.ceil() as u32,
    height as u32,
    total_width.ceil() as u32,
    height as u32,
  ));

  let mut x = 0.0_f64;

  let text_fill = theme().text_primary;

  // Opening brace
  out.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">{{</text>\n",
    x + brace_w / 2.0,
  ));
  x += brace_w;

  for (i, cell_svg) in svgs.iter().enumerate() {
    if i > 0 {
      // Comma separator
      out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
         font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
         dominant-baseline=\"central\">,</text>\n",
        x + comma_w / 2.0,
      ));
      x += comma_w;
    }

    // Nested graphic cell
    if let Some(parsed) = parse_svg_dimensions(cell_svg) {
      out.push_str(&format!(
        "<svg x=\"{:.0}\" y=\"0\" width=\"{cell_size:.0}\" \
         height=\"{cell_size:.0}\" viewBox=\"{}\">\n",
        x, parsed.view_box,
      ));
      out.push_str(&parsed.inner_content);
      out.push_str("</svg>\n");
    }
    x += cell_size;
  }

  // Closing brace
  out.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">}}</text>\n",
    x + brace_w / 2.0,
  ));

  out.push_str("</svg>");
  Some(out)
}

// ── GraphicsRow / GraphicsColumn / GraphicsGrid ────────────────────────

/// Extract SVG strings from a list of evaluated expressions.
/// Each item must be an Expr::Graphics to be included.
fn extract_svgs_from_list(items: &[Expr]) -> Vec<String> {
  items
    .iter()
    .filter_map(|item| {
      if let Expr::Graphics { svg, .. } = item {
        Some(svg.clone())
      } else {
        None
      }
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

/// Parsed layout options for GraphicsRow/Column/Grid.
struct LayoutOptions {
  h_spacing: SpacingSpec,
  v_spacing: SpacingSpec,
  /// Total width constraint (from ImageSize -> n or ImageSize -> {w, h})
  target_width: Option<f64>,
  /// Total height constraint (only from ImageSize -> {w, h})
  target_height: Option<f64>,
}

/// Parse Spacings and ImageSize options from rule arguments.
fn parse_layout_options(args: &[Expr]) -> LayoutOptions {
  let mut opts = LayoutOptions {
    h_spacing: SpacingSpec::default_val(),
    v_spacing: SpacingSpec::default_val(),
    target_width: None,
    target_height: None,
  };

  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "Spacings" => {
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

/// Get the aspect ratio (width/height) from a viewBox string "x y w h".
fn viewbox_aspect(vb: &str) -> f64 {
  let parts: Vec<f64> = vb
    .split_whitespace()
    .filter_map(|s| s.parse().ok())
    .collect();
  if parts.len() >= 4 && parts[3] > 0.0 {
    parts[2] / parts[3]
  } else {
    1.0 // fallback to square
  }
}

/// Combine SVG strings in a grid layout with configurable spacing and size.
/// Uses aspect-ratio-aware sizing: within each row, all cells share the same
/// height and each cell's width is determined by its native aspect ratio.
fn combine_svgs_grid(
  rows: &[Vec<String>],
  opts: &LayoutOptions,
) -> Option<String> {
  if rows.is_empty() {
    return None;
  }
  if rows.iter().all(|r| r.is_empty()) {
    return None;
  }

  // Parse all SVGs and collect their viewBox + inner content.
  // For each row, collect (viewBox, inner_content, aspect_ratio).
  let parsed_rows: Vec<Vec<(String, String, f64)>> = rows
    .iter()
    .map(|row| {
      row
        .iter()
        .filter_map(|svg| {
          let p = parse_svg_dimensions(svg)?;
          let ar = viewbox_aspect(&p.view_box);
          Some((p.view_box, p.inner_content, ar))
        })
        .collect()
    })
    .collect();

  if parsed_rows.iter().all(|r| r.is_empty()) {
    return None;
  }

  // Default total width when no ImageSize given
  let default_total_w = 360.0_f64;
  let target_w = opts.target_width.unwrap_or(default_total_w);

  // For each row: compute cell widths and row height given target_w.
  // In a row, all cells have the same height h.
  // cell_i width = aspect_i * h
  // sum(aspect_i * h) + (n-1)*h_gap = target_w
  // h = (target_w - (n-1)*h_gap) / sum(aspect_i)
  // But h_gap may depend on h (if Scaled), so we use an initial estimate
  // and iterate once.
  struct RowLayout {
    cells: Vec<(f64, f64, String, String)>, // (x, cell_w, viewBox, inner)
    row_h: f64,
  }

  let mut row_layouts: Vec<RowLayout> = Vec::new();

  for parsed_row in &parsed_rows {
    if parsed_row.is_empty() {
      row_layouts.push(RowLayout {
        cells: Vec::new(),
        row_h: 0.0,
      });
      continue;
    }
    let n = parsed_row.len();
    let sum_aspect: f64 = parsed_row.iter().map(|(_, _, ar)| ar).sum();

    // First estimate of row_h (assume h_gap = 0 initially for Scaled)
    let h_est = if sum_aspect > 0.0 {
      target_w / sum_aspect
    } else {
      target_w / n as f64
    };

    // Resolve h_gap using estimated cell width
    let avg_cell_w = if sum_aspect > 0.0 {
      h_est * sum_aspect / n as f64
    } else {
      h_est
    };
    let h_gap = opts.h_spacing.to_px(avg_cell_w);

    // Now compute actual row height with gaps accounted for
    let available_w = target_w - (n as f64 - 1.0).max(0.0) * h_gap;
    let row_h = if sum_aspect > 0.0 {
      (available_w / sum_aspect).max(10.0)
    } else {
      (available_w / n as f64).max(10.0)
    };

    // Lay out cells left-to-right
    let mut x = 0.0_f64;
    let mut cells = Vec::new();
    for (vb, inner, ar) in parsed_row {
      let cell_w = ar * row_h;
      cells.push((x, cell_w, vb.clone(), inner.clone()));
      x += cell_w + h_gap;
    }

    row_layouts.push(RowLayout { cells, row_h });
  }

  // If explicit height given, scale rows to fit
  if let Some(total_h) = opts.target_height {
    let num_nonempty = row_layouts.iter().filter(|r| r.row_h > 0.0).count();
    if num_nonempty > 0 {
      let current_h: f64 = row_layouts.iter().map(|r| r.row_h).sum();
      // Estimate v_gap from average row height
      let avg_row_h = current_h / num_nonempty as f64;
      let v_gap = opts.v_spacing.to_px(avg_row_h);
      let available_h = total_h - (num_nonempty as f64 - 1.0).max(0.0) * v_gap;
      if current_h > 0.0 && available_h > 0.0 {
        let scale = available_h / current_h;
        for layout in &mut row_layouts {
          layout.row_h *= scale;
          for cell in &mut layout.cells {
            cell.0 *= scale; // x
            cell.1 *= scale; // cell_w
          }
        }
      }
    }
  }

  // Compute total dimensions
  let total_width = row_layouts
    .iter()
    .map(|r| r.cells.last().map_or(0.0, |(x, w, _, _)| x + w))
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

  let mut svg = String::with_capacity(4096);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    total_width.ceil() as u32,
    total_height.ceil() as u32,
    total_width.ceil() as u32,
    total_height.ceil() as u32,
  ));

  let mut y = 0.0_f64;
  for layout in &row_layouts {
    if layout.row_h <= 0.0 {
      continue;
    }
    for (cx, cw, vb, inner) in &layout.cells {
      svg.push_str(&format!(
        "<svg x=\"{:.0}\" y=\"{:.0}\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"{}\">\n",
        cx, y, cw, layout.row_h, vb,
      ));
      svg.push_str(inner);
      svg.push_str("</svg>\n");
    }
    y += layout.row_h + v_gap;
  }

  svg.push_str("</svg>");
  Some(svg)
}

/// GraphicsRow[{g1, g2, ...}] or GraphicsRow[{g1, g2, ...}, opts...]
/// Arranges graphics side-by-side in a single row.
pub fn graphics_row_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Evaluate the first argument (should be a list of graphics)
  let list_expr = evaluate_expr_to_expr(&args[0])?;
  let items = match &list_expr {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GraphicsRow expects a list as its first argument".into(),
      ));
    }
  };

  let svgs = extract_svgs_from_list(&items);
  if svgs.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let opts = parse_layout_options(&args[1..]);
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
  let list_expr = evaluate_expr_to_expr(&args[0])?;
  let items = match &list_expr {
    Expr::List(items) => items.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GraphicsColumn expects a list as its first argument".into(),
      ));
    }
  };

  let svgs = extract_svgs_from_list(&items);
  if svgs.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let opts = parse_layout_options(&args[1..]);
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
  let grid_expr = evaluate_expr_to_expr(&args[0])?;
  let outer_items = match &grid_expr {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "GraphicsGrid expects a list of lists as its first argument".into(),
      ));
    }
  };

  let mut rows: Vec<Vec<String>> = Vec::new();
  for item in outer_items {
    match item {
      Expr::List(row_items) => {
        rows.push(extract_svgs_from_list(row_items));
      }
      _ => {
        // Single item treated as a single-element row
        let svgs = extract_svgs_from_list(&[item.clone()]);
        rows.push(svgs);
      }
    }
  }

  // Check if we have any SVGs at all
  if rows.iter().all(|r| r.is_empty()) {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let opts = parse_layout_options(&args[1..]);
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
              .unwrap_or(Expr::Identifier("Missing[]".to_string()))
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
        items.clone()
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
          items
            .get(i)
            .cloned()
            .unwrap_or(Expr::Identifier("Missing[]".to_string()))
        } else if i == 0 {
          v.clone()
        } else {
          Expr::Identifier("Missing[]".to_string())
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
  let row_height = font_size + pad_y;
  let gap = spacing_ems * font_size;

  // Compute column width from widest item
  let col_width: f64 = items
    .iter()
    .map(|item| estimate_display_width(item) * char_width + pad_x)
    .fold(0.0_f64, f64::max);

  let n = items.len() as f64;
  let total_height = row_height * n + gap * (n - 1.0);
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
  for (i, item) in items.iter().enumerate() {
    let cy = i as f64 * (row_height + gap) + row_height / 2.0;
    svg.push_str(&format!(
      "<text x=\"{text_x:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"{text_fill}\" text-anchor=\"{alignment}\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(item)
    ));
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

/// Render a list that contains Framed elements as a horizontal row SVG.
/// Plain items are rendered as text; Framed items are fully rendered via
/// `framed_to_svg` (which handles arbitrary nesting) and embedded as child SVGs.
/// The result looks like `{x, |a|, ||b||}` with visual brackets and commas.
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
      && name == "Framed"
      && !args.is_empty()
    {
      // Render the entire Framed (with any nesting) as SVG
      if let Some(child_svg) = framed_to_svg(args) {
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
        args: args.to_vec(),
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
    .map(|(x, y)| Expr::List(vec![Expr::Real(*x), Expr::Real(*y)]))
    .collect();

  Ok(Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(point_exprs)],
  })
}

// ─── LinearGradientFilling ──────────────────────────────────────────

/// Check if an expression is a color specification (RGBColor, GrayLevel, Hue, CMYKColor)
fn is_color_expr(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } => matches!(
      name.as_str(),
      "RGBColor" | "GrayLevel" | "Hue" | "CMYKColor"
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
        args: vec![Expr::Integer(0)],
      },
      Expr::FunctionCall {
        name: "GrayLevel".to_string(),
        args: vec![Expr::Integer(1)],
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
          let colors = items.clone();
          (stops, colors, angle, space)
        }
      }
      // Single non-list color arg
      other => {
        return Ok(Expr::FunctionCall {
          name: "LinearGradientFilling".to_string(),
          args: vec![other.clone(), angle, space],
        });
      }
    }
  };

  // Build: LinearGradientFilling[{stops} -> {colors}, angle, space]
  let rule = Expr::Rule {
    pattern: Box::new(Expr::List(stops)),
    replacement: Box::new(Expr::List(colors)),
  };

  Ok(Expr::FunctionCall {
    name: "LinearGradientFilling".to_string(),
    args: vec![rule, angle, space],
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
        // Preserve the head (variable symbol or {u, uinit, ulbl}) as-is;
        // evaluate any trailing bounds/step/discrete-values.
        let mut new_items: Vec<Expr> = Vec::with_capacity(items.len());
        new_items.push(items[0].clone());
        for item in &items[1..] {
          // Try to evaluate bounds; if evaluation fails, keep the
          // original so the echoed form still round-trips.
          let evaluated =
            evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone());
          new_items.push(evaluated);
        }
        out_args.push(Expr::List(new_items));
      }
      Expr::List(_) => {
        // Empty list — echo as-is.
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
    args: out_args,
  })
}

// ─────────────────────────────────────────────────────────────────
// Interactive Manipulate support (for Woxi Playground / Woxi Studio)
// ─────────────────────────────────────────────────────────────────

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
    label: String,
  },
  Discrete {
    name: String,
    /// Each discrete value rendered as InputForm (so the UI can show it
    /// and echo it back into a Block binding).
    values: Vec<String>,
    initial_index: usize,
    label: String,
  },
}

/// A parsed Manipulate expression ready for interactive rendering.
#[derive(Debug, Clone)]
pub struct ManipulateSpec {
  /// The body expression as an InputForm-compatible string, ready to be
  /// substituted into a `Block[{…}, body]` for re-evaluation.
  pub body_code: String,
  pub controls: Vec<ManipulateControl>,
}

/// Attempt to extract a `ManipulateSpec` from a held `Manipulate[…]`
/// expression. Returns `None` if the expression is not a well-formed
/// Manipulate (e.g. `Manipulate[]`, `Manipulate[expr]`, or a spec that
/// isn't a list). In those cases the caller should fall back to the
/// standard text/graphics output path.
pub fn extract_manipulate_spec(expr: &Expr) -> Option<ManipulateSpec> {
  let (name, args) = match expr {
    Expr::FunctionCall { name, args } => (name, args),
    _ => return None,
  };
  if name != "Manipulate" || args.len() < 2 {
    return None;
  }

  let body_code = crate::syntax::expr_to_input_form(&args[0]);
  let mut controls = Vec::with_capacity(args.len() - 1);
  for spec in &args[1..] {
    controls.push(parse_manipulate_control(spec)?);
  }

  Some(ManipulateSpec {
    body_code,
    controls,
  })
}

/// Parse a single variable-spec list into a `ManipulateControl`.
fn parse_manipulate_control(spec: &Expr) -> Option<ManipulateControl> {
  let items = match spec {
    Expr::List(items) => items,
    _ => return None,
  };
  if items.is_empty() {
    return None;
  }

  // Head can be either a plain symbol `u` or `{u, uinit}` / `{u, uinit, ulbl}`.
  let (name, explicit_initial, label) = match &items[0] {
    Expr::Identifier(n) => (n.clone(), None, n.clone()),
    Expr::List(head_items) if !head_items.is_empty() => {
      let n = match &head_items[0] {
        Expr::Identifier(n) => n.clone(),
        _ => return None,
      };
      let init = head_items.get(1).cloned();
      let lbl = match head_items.get(2) {
        Some(Expr::String(s)) => s.clone(),
        Some(other) => crate::syntax::expr_to_string(other),
        None => n.clone(),
      };
      (n, init, lbl)
    }
    _ => return None,
  };

  // Discrete form: {u, {u1, u2, …}} or {{u, uinit, …}, {u1, u2, …}}
  if items.len() == 2
    && let Expr::List(value_items) = &items[1]
  {
    let values: Vec<String> = value_items
      .iter()
      .map(crate::syntax::expr_to_input_form)
      .collect();
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
    return Some(ManipulateControl::Discrete {
      name,
      values,
      initial_index,
      label,
    });
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

  Some(ManipulateControl::Continuous {
    name,
    min,
    max,
    step,
    initial,
    label,
  })
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
    })
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
      } => {
        let step_json = match step {
          Some(s) => format!(r#","step":{}"#, s),
          None => String::new(),
        };
        ctrl_parts.push(format!(
          r#"{{"kind":"continuous","name":"{}","label":"{}","min":{},"max":{},"initial":{}{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          min,
          max,
          initial,
          step_json,
        ));
      }
      ManipulateControl::Discrete {
        name,
        values,
        initial_index,
        label,
      } => {
        let value_parts: Vec<String> = values
          .iter()
          .map(|v| format!(r#""{}""#, json_escape_manipulate(v)))
          .collect();
        ctrl_parts.push(format!(
          r#"{{"kind":"discrete","name":"{}","label":"{}","values":[{}],"initialIndex":{}}}"#,
          json_escape_manipulate(name),
          json_escape_manipulate(label),
          value_parts.join(","),
          initial_index,
        ));
      }
    }
  }

  format!(
    r#""body":"{}","controls":[{}]"#,
    json_escape_manipulate(&spec.body_code),
    ctrl_parts.join(","),
  )
}
