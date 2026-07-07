//! Rendering of `Epilog ->` graphics primitives on top of function plots.
//!
//! The plotters pipeline in `plot.rs` renders the curves; an `Epilog`
//! option supplies extra graphics primitives expressed in *data*
//! coordinates that must be drawn over the finished plot. The primitives
//! are rendered here as an SVG fragment (using the same linear data→pixel
//! transform as the axis labels and dash overlays) and injected just
//! before `</svg>`.

use crate::functions::graphics::{Color, parse_color};
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// The data→pixel mapping for the plotting area of a rendered plot.
/// `x0`/`y0` are the top-left corner of the plotting area in render-space
/// pixels, `w`/`h` its size, and the ranges are the displayed data extents.
pub(crate) struct PlotArea {
  pub x0: f64,
  pub y0: f64,
  pub w: f64,
  pub h: f64,
  pub x_min: f64,
  pub x_max: f64,
  pub y_min: f64,
  pub y_max: f64,
  /// Render-space pixels per display pixel (RESOLUTION_SCALE).
  pub scale: f64,
}

impl PlotArea {
  fn px(&self, x: f64) -> f64 {
    self.x0 + (x - self.x_min) / (self.x_max - self.x_min) * self.w
  }
  fn py(&self, y: f64) -> f64 {
    self.y0 + (self.y_max - y) / (self.y_max - self.y_min) * self.h
  }
  /// A data-space x-distance in render pixels.
  fn rx(&self, r: f64) -> f64 {
    (r / (self.x_max - self.x_min) * self.w).abs()
  }
  /// A data-space y-distance in render pixels.
  fn ry(&self, r: f64) -> f64 {
    (r / (self.y_max - self.y_min) * self.h).abs()
  }
}

/// Graphics-directive state carried while walking an Epilog expression.
/// Nested lists scope their directives (as in `Graphics`), so the walker
/// clones this at each list boundary.
#[derive(Clone)]
struct EpilogStyle {
  color: Color,
  /// Stroke width in *display* pixels (multiplied by `scale` on emit).
  thickness: f64,
  /// Dash pattern as fractions of the plot width; `None` = solid.
  dashing: Option<Vec<f64>>,
  /// Point diameter as a fraction of the plot width.
  point_size: f64,
  opacity: f64,
}

impl EpilogStyle {
  fn new() -> Self {
    // Epilog primitives default to black (theme-adjusted in dark mode),
    // hairline strokes, and Wolfram's default point size.
    let default_color = if crate::is_dark_mode() {
      Color::new(0.88, 0.88, 0.88)
    } else {
      Color::new(0.0, 0.0, 0.0)
    };
    EpilogStyle {
      color: default_color,
      thickness: 1.0,
      dashing: None,
      point_size: 0.012,
      opacity: 1.0,
    }
  }

  fn stroke_attrs(&self, area: &PlotArea) -> String {
    let mut s = format!(
      "stroke=\"{}\" stroke-width=\"{:.1}\"",
      self.color.to_svg_rgb(),
      (self.thickness * area.scale).max(0.5),
    );
    if let Some(dashes) = &self.dashing {
      let parts: Vec<String> = dashes
        .iter()
        .map(|d| format!("{:.1}", (d * area.w).max(0.5)))
        .collect();
      s.push_str(&format!(" stroke-dasharray=\"{}\"", parts.join(",")));
    }
    let alpha = self.color.a * self.opacity;
    if alpha < 1.0 {
      s.push_str(&format!(" opacity=\"{:.3}\"", alpha));
    }
    s
  }

  fn fill_attrs(&self) -> String {
    let mut s = format!("fill=\"{}\"", self.color.to_svg_rgb());
    let alpha = self.color.a * self.opacity;
    if alpha < 1.0 {
      s.push_str(&format!(" fill-opacity=\"{:.3}\"", alpha));
    }
    s
  }
}

/// Render a list of Epilog primitives as an SVG fragment positioned over
/// the plotting area described by `area`.
pub(crate) fn render_epilog_svg(prims: &[Expr], area: &PlotArea) -> String {
  if (area.x_max - area.x_min).abs() < 1e-12
    || (area.y_max - area.y_min).abs() < 1e-12
  {
    return String::new();
  }
  let mut out = String::new();
  let mut style = EpilogStyle::new();
  for prim in prims {
    render_item(prim, &mut style, area, &mut out);
  }
  out
}

/// Try to interpret an expression as a `{x, y}` coordinate pair.
fn point2(expr: &Expr) -> Option<(f64, f64)> {
  if let Expr::List(items) = expr
    && items.len() == 2
  {
    return Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?));
  }
  None
}

/// Collect the polyline(s) described by a `Line`/`Arrow`/`Polygon` argument:
/// either a single list of points or a list of point lists.
fn collect_point_lists(arg: &Expr) -> Vec<Vec<(f64, f64)>> {
  let items = match arg {
    Expr::List(items) if !items.is_empty() => items,
    _ => return Vec::new(),
  };
  if point2(&items[0]).is_some() {
    let pts: Vec<(f64, f64)> = items.iter().filter_map(point2).collect();
    if pts.len() >= 2 {
      vec![pts]
    } else {
      Vec::new()
    }
  } else {
    items
      .iter()
      .flat_map(collect_point_lists)
      .collect::<Vec<_>>()
  }
}

fn polyline_points(pts: &[(f64, f64)], area: &PlotArea) -> String {
  pts
    .iter()
    .map(|&(x, y)| format!("{:.1},{:.1}", area.px(x), area.py(y)))
    .collect::<Vec<_>>()
    .join(" ")
}

/// Apply a single style directive to `style`. Returns `true` when the
/// expression was a directive (and thus produces no visible output).
fn apply_directive(expr: &Expr, style: &mut EpilogStyle) -> bool {
  if let Some(c) = parse_color(expr) {
    style.color = c;
    return true;
  }
  match expr {
    Expr::Identifier(s) => match s.as_str() {
      "Thick" => style.thickness = 2.0,
      "Thin" => style.thickness = 0.5,
      "Dashed" => style.dashing = Some(vec![0.02, 0.02]),
      "Dotted" => style.dashing = Some(vec![0.002, 0.01]),
      "DotDashed" => style.dashing = Some(vec![0.002, 0.01, 0.02, 0.01]),
      _ => return false,
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Directive" => {
        for a in args {
          apply_directive(a, style);
        }
      }
      "Opacity" if !args.is_empty() => {
        if let Some(o) = try_eval_to_f64(&args[0]) {
          style.opacity = o.clamp(0.0, 1.0);
        }
      }
      "Thickness" if args.len() == 1 => match &args[0] {
        Expr::Identifier(s) => match s.as_str() {
          "Large" => style.thickness = 2.0,
          "Medium" => style.thickness = 1.0,
          "Small" => style.thickness = 0.75,
          "Tiny" => style.thickness = 0.5,
          _ => return false,
        },
        other => {
          if let Some(t) = try_eval_to_f64(other) {
            // Fraction of the plot width → display pixels (360px default).
            style.thickness = t * 360.0;
          }
        }
      },
      "AbsoluteThickness" if args.len() == 1 => {
        if let Some(t) = try_eval_to_f64(&args[0]) {
          style.thickness = t;
        }
      }
      "Dashing" if args.len() == 1 => {
        let dashes: Vec<f64> = match &args[0] {
          Expr::List(items) => {
            items.iter().filter_map(try_eval_to_f64).collect()
          }
          other => try_eval_to_f64(other)
            .map(|d| vec![d, d])
            .unwrap_or_default(),
        };
        style.dashing = if dashes.is_empty() {
          None
        } else {
          Some(dashes)
        };
      }
      "PointSize" if args.len() == 1 => {
        if let Some(p) = try_eval_to_f64(&args[0]) {
          style.point_size = p;
        }
      }
      "AbsolutePointSize" if args.len() == 1 => {
        if let Some(p) = try_eval_to_f64(&args[0]) {
          // Absolute size is in display pixels; store as width fraction.
          style.point_size = p / 360.0;
        }
      }
      _ => return false,
    },
    _ => return false,
  }
  true
}

/// Render one Epilog item (directive, primitive, or nested scoped list).
fn render_item(
  expr: &Expr,
  style: &mut EpilogStyle,
  area: &PlotArea,
  out: &mut String,
) {
  if apply_directive(expr, style) {
    return;
  }
  match expr {
    // A nested list scopes its directives, like a Graphics sub-list.
    Expr::List(items) => {
      let mut scoped = style.clone();
      for item in items {
        render_item(item, &mut scoped, area, out);
      }
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Line" if !args.is_empty() => {
        for pts in collect_point_lists(&args[0]) {
          out.push_str(&format!(
            "<polyline fill=\"none\" {} stroke-linecap=\"round\" \
             stroke-linejoin=\"round\" points=\"{}\"/>\n",
            style.stroke_attrs(area),
            polyline_points(&pts, area),
          ));
        }
      }
      "Arrow" if !args.is_empty() => {
        // Arrow[{p1, …, pn}] — a polyline with an arrowhead on the last
        // segment. `Arrow[Line[…]]` unwraps to the same point list.
        let arg = match &args[0] {
          Expr::FunctionCall { name, args: inner }
            if name == "Line" && !inner.is_empty() =>
          {
            &inner[0]
          }
          other => other,
        };
        for pts in collect_point_lists(arg) {
          out.push_str(&format!(
            "<polyline fill=\"none\" {} stroke-linecap=\"round\" \
             stroke-linejoin=\"round\" points=\"{}\"/>\n",
            style.stroke_attrs(area),
            polyline_points(&pts, area),
          ));
          if let [.., prev, last] = pts.as_slice() {
            out.push_str(&arrow_head(*prev, *last, style, area));
          }
        }
      }
      "Circle" | "Disk" => {
        render_circle_like(name == "Disk", args, style, area, out);
      }
      "Point" if !args.is_empty() => {
        let points: Vec<(f64, f64)> = match point2(&args[0]) {
          Some(p) => vec![p],
          None => match &args[0] {
            Expr::List(items) => items.iter().filter_map(point2).collect(),
            _ => Vec::new(),
          },
        };
        let r = (style.point_size * area.w / 2.0).max(0.5);
        for (x, y) in points {
          out.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" {}/>\n",
            area.px(x),
            area.py(y),
            r,
            style.fill_attrs(),
          ));
        }
      }
      "Rectangle" => {
        let (x0, y0) = args.first().and_then(point2).unwrap_or((0.0, 0.0));
        let (x1, y1) =
          args.get(1).and_then(point2).unwrap_or((x0 + 1.0, y0 + 1.0));
        let (px0, px1) = (area.px(x0.min(x1)), area.px(x0.max(x1)));
        let (py0, py1) = (area.py(y0.max(y1)), area.py(y0.min(y1)));
        out.push_str(&format!(
          "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" {}/>\n",
          px0,
          py0,
          px1 - px0,
          py1 - py0,
          style.fill_attrs(),
        ));
      }
      "Polygon" if !args.is_empty() => {
        for pts in collect_point_lists(&args[0]) {
          out.push_str(&format!(
            "<polygon {} points=\"{}\"/>\n",
            style.fill_attrs(),
            polyline_points(&pts, area),
          ));
        }
      }
      "Text" if args.len() >= 2 => {
        if let Some((x, y)) = point2(&args[1]) {
          let label = crate::functions::chart::expr_to_label(&args[0])
            .unwrap_or_else(|| crate::syntax::expr_to_output(&args[0]));
          let font_size = 13.0 * area.scale;
          out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
             dominant-baseline=\"middle\" font-family=\"sans-serif\" \
             font-size=\"{:.0}\" {}>{}</text>\n",
            area.px(x),
            area.py(y),
            font_size,
            style.fill_attrs(),
            svg_escape_text(&label),
          ));
        }
      }
      // `Style[prim, directives…]` styles just its own primitive.
      "Style" if !args.is_empty() => {
        let mut scoped = style.clone();
        for d in &args[1..] {
          apply_directive(d, &mut scoped);
        }
        render_item(&args[0], &mut scoped, area, out);
      }
      _ => {}
    },
    _ => {}
  }
}

/// Render `Circle[…]` (stroked) or `Disk[…]` (filled). Supports the center,
/// scalar or `{rx, ry}` radius, and (for the stroked form) an arc angle
/// range `{θ1, θ2}`.
fn render_circle_like(
  filled: bool,
  args: &[Expr],
  style: &EpilogStyle,
  area: &PlotArea,
  out: &mut String,
) {
  let (cx, cy) = match args.first().and_then(point2) {
    Some(c) => c,
    None if args.is_empty() => (0.0, 0.0),
    None => return,
  };
  let (r_x, r_y) = match args.get(1) {
    None => (1.0, 1.0),
    Some(e) => match point2(e) {
      Some(pair) => pair,
      None => match try_eval_to_f64(e) {
        Some(r) => (r, r),
        None => return,
      },
    },
  };
  let (pcx, pcy) = (area.px(cx), area.py(cy));
  let (prx, pry) = (area.rx(r_x), area.ry(r_y));

  // An angle range `{θ1, θ2}` renders as an elliptical arc (Circle) or a
  // pie wedge (Disk).
  let angles = args.get(2).and_then(|e| {
    if let Expr::List(items) = e
      && items.len() == 2
    {
      Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?))
    } else {
      None
    }
  });

  if let Some((a1, a2)) = angles {
    let (sx, sy) = (pcx + prx * a1.cos(), pcy - pry * a1.sin());
    let (ex, ey) = (pcx + prx * a2.cos(), pcy - pry * a2.sin());
    let large = if (a2 - a1).abs() % (2.0 * std::f64::consts::PI)
      > std::f64::consts::PI
    {
      1
    } else {
      0
    };
    // SVG y grows downward, so a counter-clockwise data-space arc is
    // sweep-flag 0.
    if filled {
      out.push_str(&format!(
        "<path d=\"M {pcx:.1} {pcy:.1} L {sx:.1} {sy:.1} \
         A {prx:.1} {pry:.1} 0 {large} 0 {ex:.1} {ey:.1} Z\" {}/>\n",
        style.fill_attrs(),
      ));
    } else {
      out.push_str(&format!(
        "<path d=\"M {sx:.1} {sy:.1} \
         A {prx:.1} {pry:.1} 0 {large} 0 {ex:.1} {ey:.1}\" fill=\"none\" {}/>\n",
        style.stroke_attrs(area),
      ));
    }
    return;
  }

  if filled {
    out.push_str(&format!(
      "<ellipse cx=\"{pcx:.1}\" cy=\"{pcy:.1}\" rx=\"{prx:.1}\" \
       ry=\"{pry:.1}\" {}/>\n",
      style.fill_attrs(),
    ));
  } else {
    out.push_str(&format!(
      "<ellipse cx=\"{pcx:.1}\" cy=\"{pcy:.1}\" rx=\"{prx:.1}\" \
       ry=\"{pry:.1}\" fill=\"none\" {}/>\n",
      style.stroke_attrs(area),
    ));
  }
}

/// A filled triangular arrowhead at `to`, oriented along `from → to`.
fn arrow_head(
  from: (f64, f64),
  to: (f64, f64),
  style: &EpilogStyle,
  area: &PlotArea,
) -> String {
  let (x1, y1) = (area.px(from.0), area.py(from.1));
  let (x2, y2) = (area.px(to.0), area.py(to.1));
  let (dx, dy) = (x2 - x1, y2 - y1);
  let len = (dx * dx + dy * dy).sqrt();
  if len < 1e-9 {
    return String::new();
  }
  let (ux, uy) = (dx / len, dy / len);
  // Head size scales with stroke width but stays visible for thin strokes.
  let size = (style.thickness * area.scale * 4.0).max(6.0 * area.scale);
  let (bx, by) = (x2 - ux * size, y2 - uy * size);
  let (nx, ny) = (-uy, ux);
  let half = size * 0.45;
  format!(
    "<polygon {} points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\"/>\n",
    style.fill_attrs(),
    x2,
    y2,
    bx + nx * half,
    by + ny * half,
    bx - nx * half,
    by - ny * half,
  )
}

/// Escape text content for inclusion in an SVG text element.
fn svg_escape_text(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
}
