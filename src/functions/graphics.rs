use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::parse_image_size;
use crate::syntax::Expr;

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
    r: f64,
    style: StyleState,
  },
  Disk {
    cx: f64,
    cy: f64,
    r: f64,
    style: StyleState,
  },
  DiskSector {
    cx: f64,
    cy: f64,
    r: f64,
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
}

// ── Parsing helpers ──────────────────────────────────────────────────────

fn expr_to_f64(expr: &Expr) -> Option<f64> {
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
        if let Some(t) = expr_to_f64(&args[0]) {
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
        match &args[0] {
          Expr::List(items) => {
            let dashes: Vec<f64> =
              items.iter().filter_map(expr_to_f64).collect();
            if !dashes.is_empty() {
              style.dashing = Some(dashes);
            }
          }
          _ => {
            if let Some(d) = expr_to_f64(&args[0]) {
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
    _ => false,
  }
}

// ── AST walker ───────────────────────────────────────────────────────────

fn collect_primitives(
  expr: &Expr,
  style: &mut StyleState,
  prims: &mut Vec<Primitive>,
) {
  match expr {
    Expr::List(items) => {
      // Nested list scopes style changes
      let saved = style.clone();
      for item in items {
        collect_primitives(item, style, prims);
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
          collect_primitives(&args[0], style, prims);
          *style = saved;
        }

        // Geometric primitives
        "Point" if !args.is_empty() => {
          parse_point(args, style, prims);
        }
        "Line" if !args.is_empty() => {
          parse_line(args, style, prims);
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
          parse_polygon(args, style, prims);
        }
        "Arrow" if !args.is_empty() => {
          parse_arrow(args, style, prims);
        }
        "Text" if !args.is_empty() => {
          parse_text(args, style, prims);
        }
        "BezierCurve" if !args.is_empty() => {
          parse_bezier(args, style, prims);
        }
        "Inset" if !args.is_empty() => {
          // Inset[text, pos] is similar to Text
          parse_text(args, style, prims);
        }

        _ => {
          // Try as directive first
          if !apply_directive(expr, style) {
            // Not recognized - could be a nested graphics expression
            for a in args {
              collect_primitives(a, style, prims);
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
  let r = if args.len() >= 2 {
    expr_to_f64(&args[1]).unwrap_or(1.0)
  } else {
    1.0
  };
  prims.push(Primitive::CircleArc {
    cx,
    cy,
    r,
    style: style.clone(),
  });
}

fn parse_disk(args: &[Expr], style: &StyleState, prims: &mut Vec<Primitive>) {
  let (cx, cy) = if !args.is_empty() {
    expr_to_point(&args[0]).unwrap_or((0.0, 0.0))
  } else {
    (0.0, 0.0)
  };
  let r = if args.len() >= 2 {
    expr_to_f64(&args[1]).unwrap_or(1.0)
  } else {
    1.0
  };
  // Disk[center, r, {angle1, angle2}] creates a sector
  if args.len() >= 3
    && let Some((a1, a2)) = expr_to_point(&args[2])
  {
    prims.push(Primitive::DiskSector {
      cx,
      cy,
      r,
      angle1: a1,
      angle2: a2,
      style: style.clone(),
    });
    return;
  }
  prims.push(Primitive::Disk {
    cx,
    cy,
    r,
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
    Primitive::CircleArc { cx, cy, r, .. }
    | Primitive::Disk { cx, cy, r, .. } => {
      bb.include_point(cx - r, cy - r);
      bb.include_point(cx + r, cy + r);
    }
    Primitive::DiskSector {
      cx,
      cy,
      r,
      angle1,
      angle2,
      ..
    } => {
      // Include center point (sector always connects to center)
      bb.include_point(*cx, *cy);
      // Include the two endpoint arcs
      bb.include_point(cx + r * angle1.cos(), cy + r * angle1.sin());
      bb.include_point(cx + r * angle2.cos(), cy + r * angle2.sin());
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
          bb.include_point(cx + r * cardinal.cos(), cy + r * cardinal.sin());
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
  const AXIS_STROKE: &str = "#b3b3b3";
  const TICK_LABEL_FILL: &str = "#555555";

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
      "<line x1=\"0.00\" y1=\"{axis_y_px:.2}\" x2=\"{svg_w:.2}\" y2=\"{axis_y_px:.2}\" stroke=\"{AXIS_STROKE}\" stroke-width=\"1\"/>\n"
    ));
    for t in generate_ticks(bb.x_min, bb.x_max, 6) {
      let x = coord_x(t, bb, svg_w);
      if !x.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{x:.2}\" y2=\"{:.2}\" stroke=\"{AXIS_STROKE}\" stroke-width=\"1\"/>\n",
        axis_y_px - 4.0,
        axis_y_px + 4.0
      ));
      let label = format_tick_value(t);
      if axes.1 && label == "0" {
        continue;
      }
      svg.push_str(&format!(
        "<text x=\"{x:.2}\" y=\"{:.2}\" fill=\"{TICK_LABEL_FILL}\" font-size=\"11\" font-family=\"monospace\" text-anchor=\"middle\" dominant-baseline=\"hanging\">{}</text>\n",
        axis_y_px + 6.0,
        svg_escape(&label),
      ));
    }
  }

  if axes.1 {
    svg.push_str(&format!(
      "<line x1=\"{axis_x_px:.2}\" y1=\"0.00\" x2=\"{axis_x_px:.2}\" y2=\"{svg_h:.2}\" stroke=\"{AXIS_STROKE}\" stroke-width=\"1\"/>\n"
    ));
    for t in generate_ticks(bb.y_min, bb.y_max, 6) {
      let y = coord_y(t, bb, svg_h);
      if !y.is_finite() {
        continue;
      }
      svg.push_str(&format!(
        "<line x1=\"{:.2}\" y1=\"{y:.2}\" x2=\"{:.2}\" y2=\"{y:.2}\" stroke=\"{AXIS_STROKE}\" stroke-width=\"1\"/>\n",
        axis_x_px - 4.0,
        axis_x_px + 4.0
      ));
      let label = format_tick_value(t);
      if axes.0 && label == "0" {
        continue;
      }
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{y:.2}\" fill=\"{TICK_LABEL_FILL}\" font-size=\"11\" font-family=\"monospace\" text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>\n",
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
    Primitive::CircleArc { cx, cy, r, style } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *r / bb.width() * svg_w;
      let sry = *r / bb.height() * svg_h;
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
    Primitive::Disk { cx, cy, r, style } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *r / bb.width() * svg_w;
      let sry = *r / bb.height() * svg_h;
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
      r,
      angle1,
      angle2,
      style,
    } => {
      let scx = coord_x(*cx, bb, svg_w);
      let scy = coord_y(*cy, bb, svg_h);
      let srx = *r / bb.width() * svg_w;
      let sry = *r / bb.height() * svg_h;
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
          let head_len = (sw * 4.0).max(6.0);
          let head_half_w = head_len * 0.4;
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
      Primitive::CircleArc { cx, cy, r, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::circle_box(*cx, *cy, *r));
      }
      Primitive::Disk { cx, cy, r, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::disk_box(*cx, *cy, *r));
      }
      Primitive::DiskSector {
        cx,
        cy,
        r,
        angle1,
        angle2,
        style,
      } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::disk_sector_box(*cx, *cy, *r, *angle1, *angle2));
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
          if let Some((w, h, fw)) = parse_image_size(replacement) {
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
        _ => {}
      }
    }
  }

  // Collect primitives
  let mut style = StyleState::default();
  let mut primitives = Vec::new();
  collect_primitives(&content, &mut style, &mut primitives);

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
  let svg_aspect = svg_w / svg_h;
  let data_aspect_wh = bb.width() / bb.height();
  if svg_aspect.is_finite()
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

  let mut svg = String::with_capacity(4096);

  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n",
      svg_width, svg_height
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n",
      svg_width, svg_height, svg_width, svg_height
    ));
  }

  // Background
  if let Some(bg) = background {
    svg.push_str(&format!(
      "<rect width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
      svg_width,
      svg_height,
      bg.to_svg_rgb(),
    ));
  }

  render_axes(&mut svg, axes, &bb, svg_w, svg_h);

  // Render primitives
  for prim in &primitives {
    render_primitive(prim, &bb, svg_w, svg_h, &mut svg);
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

/// Check if an expression is a simple numeric atom (integer, real, etc.).
fn is_numeric_atom(e: &Expr) -> bool {
  matches!(
    e,
    Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_) | Expr::BigFloat(..)
  )
}

/// Determine the separator between two adjacent factors in Times SVG rendering.
/// Returns `""` (no separator) or `" "` (space) — never `"*"`.
fn times_svg_separator(left: &Expr, right: &Expr) -> &'static str {
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
  // Number followed by identifier → no separator (e.g. 10x)
  if is_numeric_atom(left) && matches!(right, Expr::Identifier(_)) {
    return "";
  }
  // Number followed by Power of identifier → no separator (e.g. 10x²)
  if is_numeric_atom(left)
    && let Some((base, _)) = as_power(right)
    && matches!(base, Expr::Identifier(_))
  {
    return "";
  }
  // Default: space (implicit multiplication)
  " "
}

/// Render a stacked fraction (numerator over denominator) as SVG tspan markup.
/// Uses `<tspan>` elements with `dy`/`dx` positioning:
/// numerator shifted up, fraction bar (─ characters), denominator shifted down,
/// all at 70% font-size, then baseline reset.
fn stacked_fraction_svg(
  num_markup: &str,
  den_markup: &str,
  num_w: f64,
  den_w: f64,
) -> String {
  let frac_chars = num_w.max(den_w).ceil() as usize;
  let frac_chars = frac_chars.max(1);

  let char_px = 5.88_f64; // 8.4 * 0.7 (monospace char width at 70% font-size)
  let num_px = num_w * char_px;
  let den_px = den_w * char_px;
  let bar_px = frac_chars as f64 * char_px;

  // Center offsets for numerator and denominator
  let num_center = (bar_px - num_px) / 2.0;
  let den_center = (bar_px - den_px) / 2.0;

  // dx offsets for positioning
  let back_from_num = -(num_center + num_px); // go back to start after numerator
  let dx_den = den_center - bar_px; // center denominator after bar
  let advance = den_center; // advance cursor to fraction end

  // Fraction bar using box-drawing character
  let bar: String = "\u{2500}".repeat(frac_chars);

  format!(
    "<tspan font-size=\"70%\">\
     <tspan dx=\"{num_center:.1}\" dy=\"-4\">{num_markup}</tspan>\
     <tspan dx=\"{back_from_num:.1}\" dy=\"6\">{bar}</tspan>\
     <tspan dx=\"{dx_den:.1}\" dy=\"6\">{den_markup}</tspan>\
     </tspan>\
     <tspan dx=\"{advance:.1}\" dy=\"-8\" font-size=\"1\"> </tspan>"
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
      let parts: Vec<String> =
        args.iter().map(quantity_unit_to_svg_abbrev).collect();
      parts.join("\u{22c5}")
    }
    _ => expr_to_svg_markup(unit),
  }
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
      // Graphical output shows only `prec` significant digits (no backtick notation)
      svg_escape(&truncate_bigfloat_digits(digits, *prec))
    }
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::Constant(_)
    | Expr::Slot(_) => svg_escape(&expr_to_output(expr)),

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
      truncate_bigfloat_digits(digits, *prec).len() as f64
    }
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::Constant(_)
    | Expr::Slot(_) => expr_to_output(expr).len() as f64,

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

    // Expr::Image → width in character units (pixel width / char_width)
    Expr::Image { width, .. } => *width as f64 / 8.4,

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
  if let Expr::Rule { pattern, .. } = opt
    && let Some(opt_name) = option_name(pattern)
  {
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

/// Implementation of Show[g1, g2, ..., opts...].
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
      Expr::FunctionCall { name, args: gargs } if name == "Graphics3D" => {
        is_3d = true;
        if !gargs.is_empty() {
          merged_primitives.push(gargs[0].clone());
        }
        for opt in gargs.iter().skip(1) {
          merge_option(&mut merged_options, opt);
        }
      }
      Expr::Graphics { is_3d: g_is_3d, .. } => {
        // Pre-rendered Graphics from Plot[], Plot3D[], etc.
        is_3d = *g_is_3d;
        rendered_graphics.push(expr_ref.clone());
      }
      Expr::Rule { .. } => {
        merge_option(&mut merged_options, expr_ref);
      }
      _ => {}
    }
  }

  // If we have pre-rendered Graphics but no primitives from Graphics[...],
  // return the rendered result directly. Single-arg Show just passes through.
  // Multi-arg (combining rendered plots) returns the first one since we
  // cannot merge SVGs without re-rendering from source data.
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

/// Extract style info from a Style[content, directives...] cell.
/// Returns (content, font_size, font_weight, font_style).
fn extract_cell_style(cell: &Expr) -> (&Expr, Option<f64>, &str, &str) {
  if let Expr::FunctionCall { name, args } = cell
    && name == "Style"
    && !args.is_empty()
  {
    let content = &args[0];
    let mut fs: Option<f64> = None;
    let mut fw = "normal";
    let mut fst = "normal";
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
        _ => {}
      }
    }
    return (content, fs, fw, fst);
  }
  (cell, None, "normal", "normal")
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

fn grid_svg_internal(
  args: &[Expr],
  group_gaps: &[usize],
  parens: bool,
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
  let mut frame_all = false;
  let mut row_headings: Vec<Expr> = Vec::new();
  let mut col_headings: Vec<Expr> = Vec::new();
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
        "Frame" => {
          if let Expr::Identifier(val) = replacement.as_ref()
            && val == "All"
          {
            frame_all = true;
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
  let pad_x: f64 = 12.0; // horizontal padding per cell (each side = 6)
  let pad_y: f64 = 8.0; // vertical padding per cell (each side = 4)
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
  let total_height: f64 = row_heights.iter().sum::<f64>() + total_gap;

  // When parentheses are enabled, reserve space on left and right
  let paren_margin: f64 = if parens { 12.0 } else { 0.0 };
  let total_width: f64 = grid_width + 2.0 * paren_margin;

  // Build SVG
  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(2048);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
  ));

  // Draw round parentheses if enabled
  if parens {
    let h = total_height;
    let inset = 8.0; // how far the curve bows inward
    let stroke_w = 1.2;
    // Left parenthesis: smooth arc from top to bottom, bowing left
    // Cubic Bézier: start at (margin, 0), control points pull left, end at (margin, h)
    let lx = paren_margin;
    svg.push_str(&format!(
      "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{stroke_w}\"/>\n",
      lx, 0.0,
      lx - inset, h * 0.33,
      lx - inset, h * 0.67,
      lx, h
    ));
    // Right parenthesis: smooth arc from top to bottom, bowing right
    let rx = paren_margin + grid_width;
    svg.push_str(&format!(
      "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{stroke_w}\"/>\n",
      rx, 0.0,
      rx + inset, h * 0.33,
      rx + inset, h * 0.67,
      rx, h
    ));
  }

  // Draw cell contents (shifted right by paren_margin when parens are enabled)
  let mut y_offset: f64 = 0.0;
  for (i, row) in rows.iter().enumerate() {
    // Add group gap before this row if it's a group boundary
    if group_gaps.contains(&i) {
      y_offset += group_gap;
    }
    let rh = row_heights[i];
    let mut x_offset: f64 = paren_margin;
    for (j, cell) in row.iter().enumerate() {
      let col_w = col_widths[j];
      let cx = x_offset + col_w / 2.0;
      let cy = y_offset + rh / 2.0;

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
          let iy = y_offset + (rh - draw_h) / 2.0;

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
        let (content, cell_fs, cell_fw, cell_fst) = extract_cell_style(cell);
        let fs = cell_fs.unwrap_or(font_size);
        let fw_attr = if cell_fw != "normal" {
          format!(" font-weight=\"{}\"", cell_fw)
        } else {
          String::new()
        };
        let fst_attr = if cell_fst != "normal" {
          format!(" font-style=\"{}\"", cell_fst)
        } else {
          String::new()
        };
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{fs}\"{fw_attr}{fst_attr} text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(content)
        ));
      }
      x_offset += col_w;
    }
    y_offset += rh;
  }

  // Draw frame lines if Frame -> All
  if frame_all {
    // Horizontal lines (num_rows + 1 lines)
    let mut y = 0.0_f64;
    for i in 0..=num_rows {
      svg.push_str(&format!(
        "<line x1=\"{paren_margin:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"black\" stroke-width=\"1\"/>\n",
        paren_margin + grid_width
      ));
      if i < num_rows {
        y += row_heights[i];
      }
    }
    // Vertical lines (num_cols + 1 lines)
    let mut x_offset: f64 = paren_margin;
    for j in 0..=num_cols {
      svg.push_str(&format!(
        "<line x1=\"{x_offset:.1}\" y1=\"0\" x2=\"{x_offset:.1}\" y2=\"{total_height:.1}\" stroke=\"black\" stroke-width=\"1\"/>\n"
      ));
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
  svg.push_str(&format!(
    "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{stroke_w}\"/>\n",
    lx, 0.0,
    lx - outer_paren_inset, h * 0.33,
    lx - outer_paren_inset, h * 0.67,
    lx, h
  ));
  let rx = outer_paren_margin + grid_width;
  svg.push_str(&format!(
    "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{stroke_w}\"/>\n",
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
        "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{sub_stroke}\"/>\n",
        sub_lx, sub_top,
        sub_lx - paren_inset, sub_top + sub_h * 0.33,
        sub_lx - paren_inset, sub_top + sub_h * 0.67,
        sub_lx, sub_bot
      ));
      svg.push_str(&format!(
        "<path d=\"M {:.1} {:.1} C {:.1} {:.1}, {:.1} {:.1}, {:.1} {:.1}\" fill=\"none\" stroke=\"black\" stroke-width=\"{sub_stroke}\"/>\n",
        sub_rx, sub_top,
        sub_rx + paren_inset, sub_top + sub_h * 0.33,
        sub_rx + paren_inset, sub_top + sub_h * 0.67,
        sub_rx, sub_bot
      ));

      // Draw sub-items as text, vertically stacked
      for (k, item) in sub_items.iter().enumerate() {
        let cx = x_off + cell_w / 2.0;
        let cy = sub_y_start + k as f64 * row_height + row_height / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
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

  // Key column background
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{key_col_w:.1}\" height=\"{total_height:.1}\" fill=\"#f0f0f0\"/>\n"
  ));

  // Rows
  let mut y_offset: f64 = 0.0;
  for (i, (_, v)) in pairs.iter().enumerate() {
    let cy = y_offset + row_height / 2.0;
    // Key (bold, in left column)
    let kx = key_col_w / 2.0;
    svg.push_str(&format!(
      "<text x=\"{kx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      keys[i]
    ));
    // Value (in right column)
    let vx = key_col_w + val_col_w / 2.0;
    svg.push_str(&format!(
      "<text x=\"{vx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(v)
    ));
    y_offset += row_height;
  }

  // Grid lines
  // Horizontal lines
  let mut y = 0.0_f64;
  for i in 0..=num_rows {
    let stroke_width = if i == 0 || i == num_rows {
      "1.5"
    } else {
      "0.5"
    };
    let color = if i == 0 || i == num_rows {
      "#999"
    } else {
      "#ccc"
    };
    svg.push_str(&format!(
      "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>\n"
    ));
    y += row_height;
  }
  // Vertical lines: outer borders + separator between key and value columns
  svg.push_str(&format!(
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{key_col_w:.1}\" y1=\"0\" x2=\"{key_col_w:.1}\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
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

  // Header background
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{total_width:.1}\" height=\"{header_row_height:.1}\" fill=\"#f0f0f0\"/>\n"
  ));

  // Header text (bold)
  {
    let mut x_offset: f64 = 0.0;
    for (j, header) in headers.iter().enumerate() {
      let col_w = col_widths[j];
      let cx = x_offset + col_w / 2.0;
      let cy = header_row_height / 2.0;
      svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" text-anchor=\"middle\" dominant-baseline=\"central\">{header}</text>\n"
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
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(cell)
        ));
        x_offset += col_w;
      }
    }
    y_offset += row_height;
  }

  // Grid lines
  // Horizontal lines
  let mut y = 0.0_f64;
  for i in 0..=num_total_rows {
    let stroke_width = if i == 0 || i == 1 || i == num_total_rows {
      "1.5"
    } else {
      "0.5"
    };
    let color = if i == 0 || i == 1 || i == num_total_rows {
      "#999"
    } else {
      "#ccc"
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
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
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

  // Opening brace
  out.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
     font-size=\"{font_size}\" text-anchor=\"middle\" \
     dominant-baseline=\"central\">{{</text>\n",
    x + brace_w / 2.0,
  ));
  x += brace_w;

  for (i, cell_svg) in svgs.iter().enumerate() {
    if i > 0 {
      // Comma separator
      out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{text_y:.1}\" font-family=\"monospace\" \
         font-size=\"{font_size}\" text-anchor=\"middle\" \
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
     font-size=\"{font_size}\" text-anchor=\"middle\" \
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
              } else if let Some((w, _, _)) = parse_image_size(other) {
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

  // Row-number column background (light blue-gray)
  svg.push_str(&format!(
    "<rect x=\"0\" y=\"0\" width=\"{row_num_col_w:.1}\" height=\"{total_height:.1}\" fill=\"#eef2f7\"/>\n"
  ));

  // Header row background (if applicable)
  if num_header_rows > 0 {
    svg.push_str(&format!(
      "<rect x=\"{row_num_col_w:.1}\" y=\"0\" width=\"{data_width:.1}\" height=\"{header_row_height:.1}\" fill=\"#f0f0f0\"/>\n"
    ));
    // Also extend the row-number column header background
    svg.push_str(&format!(
      "<rect x=\"0\" y=\"0\" width=\"{row_num_col_w:.1}\" height=\"{header_row_height:.1}\" fill=\"#dde4ed\"/>\n"
    ));
  }

  // Header text (bold)
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
        "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" font-weight=\"bold\" text-anchor=\"middle\" dominant-baseline=\"central\">{header}</text>\n"
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
  let mut y_offset: f64 = y_start;
  for (i, row) in grid.iter().enumerate() {
    // Row number (1-based, in left column)
    let row_num = format!("{}", i + 1);
    let rx = row_num_col_w / 2.0;
    let cy = y_offset + row_height / 2.0;
    svg.push_str(&format!(
      "<text x=\"{rx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" fill=\"#888\" text-anchor=\"middle\" dominant-baseline=\"central\">{row_num}</text>\n"
    ));

    // Data cells
    let mut x_offset: f64 = row_num_col_w;
    for (j, cell) in row.iter().enumerate() {
      if j < num_cols && j < col_widths.len() {
        let col_w = col_widths[j];
        let cx = x_offset + col_w / 2.0;
        svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(cell)
        ));
        x_offset += col_w;
      }
    }
    y_offset += row_height;
  }

  // Grid lines
  // Horizontal lines
  let num_total_rows = num_header_rows + num_data_rows;
  let mut y = 0.0_f64;
  for i in 0..=num_total_rows {
    let is_border =
      i == 0 || i == num_total_rows || (num_header_rows > 0 && i == 1);
    let stroke_width = if is_border { "1.5" } else { "0.5" };
    let color = if is_border { "#999" } else { "#ccc" };
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
    "<line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{row_num_col_w:.1}\" y1=\"0\" x2=\"{row_num_col_w:.1}\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
  ));
  svg.push_str(&format!(
    "<line x1=\"{total_width:.1}\" y1=\"0\" x2=\"{total_width:.1}\" y2=\"{total_height:.1}\" stroke=\"#999\" stroke-width=\"1.5\"/>\n"
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

  let char_width: f64 = 8.4;
  let font_size: f64 = 14.0;
  let pad_x: f64 = 12.0;
  let pad_y: f64 = 8.0;
  let row_height = font_size + pad_y;

  // Compute column width from widest item
  let col_width: f64 = items
    .iter()
    .map(|item| estimate_display_width(item) * char_width + pad_x)
    .fold(0.0_f64, f64::max);

  let total_height = row_height * items.len() as f64;
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

  for (i, item) in items.iter().enumerate() {
    let cy = i as f64 * row_height + row_height / 2.0;
    svg.push_str(&format!(
      "<text x=\"{text_x:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"{alignment}\" dominant-baseline=\"central\">{}</text>\n",
      expr_to_svg_markup(item)
    ));
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
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"none\" stroke=\"rgb(190,190,190)\" stroke-width=\"{stroke_width}\"/>\n",
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
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{rounding:.1}\" fill=\"none\" stroke=\"rgb(190,190,190)\" stroke-width=\"{stroke_width}\"/>\n",
      stroke_width / 2.0, stroke_width / 2.0,
      total_w - stroke_width, total_h - stroke_width,
    ));
    // Text centered inside
    let cx = total_w / 2.0;
    let cy = total_h / 2.0;
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
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

  // Opening brace
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{{</text>\n",
    brace_width / 2.0
  ));

  let mut x = brace_width;
  for (i, cell) in cells.iter().enumerate() {
    if i > 0 {
      // Draw comma separator
      svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">,</text>\n",
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
          "<text x=\"{cx:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
          expr_to_svg_markup(expr)
        ));
      }
    }
    x += cell.width;
  }

  // Closing brace
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{mid_y:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">}}</text>\n",
    x + brace_width / 2.0
  ));

  svg.push_str("</svg>");
  Some(svg)
}
