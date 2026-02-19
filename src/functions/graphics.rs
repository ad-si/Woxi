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
    Color::new(self.r * f, self.g * f, self.b * f).with_alpha(self.a)
  }

  fn lighter(self, amount: f64) -> Self {
    let f = amount;
    Color::new(
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
    Color::new(r, g, bl)
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
    "Red" => Color::new(1.0, 0.0, 0.0),
    "Blue" => Color::new(0.0, 0.0, 1.0),
    "Green" => Color::new(0.0, 0.5, 0.0),
    "Black" => Color::new(0.0, 0.0, 0.0),
    "White" => Color::new(1.0, 1.0, 1.0),
    "Gray" => Color::new(0.5, 0.5, 0.5),
    "LightGray" => Color::new(0.83, 0.83, 0.83),
    "Orange" => Color::new(1.0, 0.5, 0.0),
    "Yellow" => Color::new(1.0, 1.0, 0.0),
    "Purple" => Color::new(0.5, 0.0, 0.5),
    "Cyan" => Color::new(0.0, 1.0, 1.0),
    "Magenta" => Color::new(1.0, 0.0, 1.0),
    "Brown" => Color::new(0.6, 0.3, 0.0),
    "Pink" => Color::new(1.0, 0.75, 0.8),
    "LightRed" => Color::new(1.0, 0.5, 0.5),
    "LightBlue" => Color::new(0.68, 0.85, 0.9),
    "LightGreen" => Color::new(0.56, 0.93, 0.56),
    "LightOrange" => Color::new(1.0, 0.8, 0.5),
    "LightYellow" => Color::new(1.0, 1.0, 0.88),
    "LightPurple" => Color::new(0.8, 0.6, 0.8),
    "LightCyan" => Color::new(0.88, 1.0, 1.0),
    "LightMagenta" => Color::new(1.0, 0.75, 1.0),
    "LightBrown" => Color::new(0.8, 0.6, 0.4),
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

  fn merge(&mut self, other: &BBox) {
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
    BBox {
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
          // RGBColor[gray]
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
  // Arrow[{{x1,y1},{x2,y2},...}]
  if let Some(pts) = expr_to_point_list(&args[0])
    && pts.len() >= 2
  {
    prims.push(Primitive::ArrowPrim {
      points: pts,
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
    Primitive::PolygonPrim { points, .. }
    | Primitive::ArrowPrim { points, .. } => {
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
    Primitive::ArrowPrim { points, style } => {
      let color = style.effective_color();
      let sw = thickness_px(style.thickness, bb, svg_w).max(0.5);
      let dash = dash_attr(&style.dashing, bb, svg_w);

      // Draw the line
      let pts: Vec<String> = points
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
      if points.len() >= 2 {
        let n = points.len();
        let (x1, y1) = points[n - 2];
        let (x2, y2) = points[n - 1];
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
      Primitive::ArrowPrim { points, style } => {
        elements.extend(tracker.emit_style_changes(style));
        elements.push(gbox::arrow_box(points));
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

  // Render primitives
  for prim in &primitives {
    render_primitive(prim, &bb, svg_w, svg_h, &mut svg);
  }

  svg.push_str("</svg>");

  // Store the SVG for capture by Jupyter/Export
  crate::capture_graphics(&svg);

  // Generate and store GraphicsBox expression for .nb export
  let box_elements = primitives_to_box_elements(&primitives);
  let graphicsbox = gbox::graphics_box(&box_elements);
  crate::capture_graphicsbox(&graphicsbox);

  Ok(Expr::Identifier("-Graphics-".to_string()))
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
    return format!(
      "{}<tspan baseline-shift=\"super\" font-size=\"70%\">{}</tspan>",
      base_markup, exp_markup
    );
  }

  match expr {
    // ── Atoms ──
    Expr::String(s) => svg_escape(s),
    Expr::Identifier(s) => svg_escape(s),
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(..)
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
        BinaryOperator::Times => ("*", false),
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
          // Times[-1, x, ...] → -x*...
          if matches!(&args[0], Expr::Integer(-1)) {
            let rest: Vec<String> = args[1..]
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
            return format!("-{}", rest.join("*"));
          }
          // General: a*b*c
          args
            .iter()
            .map(|a| {
              let s = expr_to_svg_markup(a);
              if is_additive_expr(a) {
                format!("({})", s)
              } else {
                s
              }
            })
            .collect::<Vec<_>>()
            .join("*")
        }

        // Rational[n, d] → stacked fraction
        "Rational" if args.len() == 2 => {
          let num_markup = expr_to_svg_markup(&args[0]);
          let den_markup = expr_to_svg_markup(&args[1]);
          let num_w = estimate_display_width(&args[0]);
          let den_w = estimate_display_width(&args[1]);
          stacked_fraction_svg(&num_markup, &den_markup, num_w, den_w)
        }

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
    return estimate_display_width(base) + estimate_display_width(exp) * 0.7;
  }

  match expr {
    // Atoms
    Expr::String(s) => s.len() as f64,
    Expr::Identifier(s) => s.len() as f64,
    Expr::Integer(_)
    | Expr::Real(_)
    | Expr::BigInteger(_)
    | Expr::BigFloat(..)
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
      let op_len: f64 = match op {
        BinaryOperator::Plus | BinaryOperator::Minus => 3.0,
        BinaryOperator::Times | BinaryOperator::Divide => 1.0,
        BinaryOperator::Power => unreachable!(),
        BinaryOperator::And => 4.0,
        BinaryOperator::Or => 4.0,
        BinaryOperator::StringJoin => 2.0,
        BinaryOperator::Alternatives => 3.0,
      };
      estimate_display_width(left) + op_len + estimate_display_width(right)
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
          let rest: f64 = args[1..].iter().map(estimate_display_width).sum();
          let seps = if args.len() > 2 {
            (args.len() - 2) as f64
          } else {
            0.0
          };
          1.0 + rest + seps
        } else {
          let factors: f64 = args.iter().map(estimate_display_width).sum();
          factors + (args.len() - 1) as f64
        }
      }
      "Rational" if args.len() == 2 => stacked_fraction_width(
        estimate_display_width(&args[0]),
        estimate_display_width(&args[1]),
      ),
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

    // Fallback
    _ => expr_to_output(expr).len() as f64,
  }
}

pub fn grid_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Extract rows from args[0]
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows: Vec<Vec<Expr>> = match &data {
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
      return Ok(Expr::FunctionCall {
        name: "Grid".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Parse options from remaining args
  let mut frame_all = false;
  for raw_opt in &args[1..] {
    let opt =
      evaluate_expr_to_expr(raw_opt).unwrap_or_else(|_| raw_opt.clone());
    if let Expr::Rule {
      pattern,
      replacement,
    } = &opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "Frame"
      && let Expr::Identifier(val) = replacement.as_ref()
      && val == "All"
    {
      frame_all = true;
    }
  }

  // Convert cells to text
  let num_rows = rows.len();
  let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
  if num_cols == 0 {
    return Ok(Expr::FunctionCall {
      name: "Grid".to_string(),
      args: args.to_vec(),
    });
  }

  // Compute column widths based on estimated display width
  let char_width: f64 = 8.4; // approximate monospace char width at font-size 14
  let font_size: f64 = 14.0;
  let pad_x: f64 = 12.0; // horizontal padding per cell (each side = 6)
  let pad_y: f64 = 8.0; // vertical padding per cell (each side = 4)
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

  // Compute per-row heights (taller when a row contains a fraction)
  let row_heights: Vec<f64> = rows
    .iter()
    .map(|row| {
      if row.iter().any(has_fraction) {
        frac_row_height
      } else {
        base_row_height
      }
    })
    .collect();

  let total_width: f64 = col_widths.iter().sum();
  let total_height: f64 = row_heights.iter().sum();

  // Build SVG
  let svg_w = total_width.ceil() as u32;
  let svg_h = total_height.ceil() as u32;
  let mut svg = String::with_capacity(2048);
  svg.push_str(&format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
    svg_w, svg_h, svg_w, svg_h
  ));

  // Draw cell text
  let mut y_offset: f64 = 0.0;
  for (i, row) in rows.iter().enumerate() {
    let rh = row_heights[i];
    let mut x_offset: f64 = 0.0;
    for (j, cell) in row.iter().enumerate() {
      let col_w = col_widths[j];
      let cx = x_offset + col_w / 2.0;
      let cy = y_offset + rh / 2.0;
      svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"monospace\" font-size=\"{font_size}\" text-anchor=\"middle\" dominant-baseline=\"central\">{}</text>\n",
        expr_to_svg_markup(cell)
      ));
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
        "<line x1=\"0\" y1=\"{y:.1}\" x2=\"{total_width:.1}\" y2=\"{y:.1}\" stroke=\"black\" stroke-width=\"1\"/>\n"
      ));
      if i < num_rows {
        y += row_heights[i];
      }
    }
    // Vertical lines (num_cols + 1 lines)
    let mut x_offset: f64 = 0.0;
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

  crate::capture_graphics(&svg);

  Ok(Expr::Identifier("-Graphics-".to_string()))
}
