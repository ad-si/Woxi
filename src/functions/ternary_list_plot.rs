use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{PLOT_COLORS, RESOLUTION_SCALE, parse_image_size};
use crate::syntax::Expr;

/// Default side length (in user-space px) of the bounding box for the plot.
const DEFAULT_TLP_SIZE: u32 = 360;

/// A single data point given as a normalized barycentric triple that sums
/// to 1. The three components map to the top, bottom-left and bottom-right
/// corners of the triangle respectively.
type Triple = (f64, f64, f64);

/// TernaryListPlot[{{a1,b1,c1}, ...}] or
/// TernaryListPlot[{data1, data2, ...}] with each `data` a list of triples.
///
/// Each triple `{a, b, c}` is normalized so that `a + b + c == 1` and plotted
/// inside an equilateral triangle. Component 1 pulls the point towards the top
/// corner, component 2 towards the bottom-left corner and component 3 towards
/// the bottom-right corner.
pub fn ternary_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_ternary_data(&args[0])?;

  let mut svg_width = DEFAULT_TLP_SIZE;
  let mut svg_height = DEFAULT_TLP_SIZE;
  let mut full_width = false;
  let mut custom_colors: Vec<String> = Vec::new();

  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, _h, fw)) =
            parse_image_size(replacement, DEFAULT_TLP_SIZE, DEFAULT_TLP_SIZE)
          {
            svg_width = w;
            // Keep the drawing area square regardless of the parsed height.
            svg_height = w;
            full_width = fw;
          }
        }
        "PlotStyle" => {
          custom_colors = crate::functions::plot::parse_plot_style(replacement)
            .iter()
            .filter_map(|s| s.color.as_ref().map(|c| c.to_svg_rgb()))
            .collect();
        }
        _ => {}
      }
    }
  }

  let svg = render_ternary_svg(
    &all_series,
    svg_width,
    svg_height,
    full_width,
    &custom_colors,
  );
  Ok(crate::graphics_result(svg))
}

/// Parse the first argument into one or more series of normalized triples.
fn parse_ternary_data(
  arg: &Expr,
) -> Result<Vec<Vec<Triple>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;

  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TernaryListPlot: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![vec![]]);
  }

  // Multiple datasets: {{{a,b,c}, ...}, {{a,b,c}, ...}} — the first element is
  // itself a list whose first element is a list (i.e. a triple).
  let is_multiple = matches!(&items[0], Expr::List(inner)
    if inner.first().is_some_and(|e| matches!(e, Expr::List(_))));

  if is_multiple {
    let mut all_series = Vec::new();
    for item in items {
      if let Expr::List(series_items) = item {
        all_series.push(parse_triple_series(series_items));
      }
    }
    if !all_series.is_empty() {
      return Ok(all_series);
    }
  }

  // Single dataset: {{a,b,c}, {a,b,c}, ...}
  Ok(vec![parse_triple_series(items)])
}

/// Parse a flat list of `{a, b, c}` triples into normalized barycentric
/// coordinates, dropping any entry that is not a numeric triple with a
/// positive sum.
fn parse_triple_series(series_items: &[Expr]) -> Vec<Triple> {
  let mut points = Vec::with_capacity(series_items.len());
  for item in series_items {
    if let Expr::List(triple) = item
      && triple.len() == 3
    {
      let a = eval_num(&triple[0]);
      let b = eval_num(&triple[1]);
      let c = eval_num(&triple[2]);
      if let (Some(a), Some(b), Some(c)) = (a, b, c) {
        let s = a + b + c;
        if s.is_finite() && s.abs() > f64::EPSILON {
          points.push((a / s, b / s, c / s));
        }
      }
    }
  }
  points
}

/// Evaluate an expression to an `f64` if possible.
fn eval_num(e: &Expr) -> Option<f64> {
  let ev = evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
  try_eval_to_f64(&ev)
}

/// Theme-appropriate colors: (background, triangle stroke, grid, label).
fn theme_colors() -> (&'static str, &'static str, &'static str, &'static str) {
  if crate::is_dark_mode() {
    ("#1a1a1a", "#bbbbbb", "#3a3a3a", "#cccccc")
  } else {
    ("#ffffff", "#555555", "#dddddd", "#333333")
  }
}

/// Color for a series by index, honoring an explicit `PlotStyle` override.
fn series_color(idx: usize, custom_colors: &[String]) -> String {
  if !custom_colors.is_empty() {
    return custom_colors[idx % custom_colors.len()].clone();
  }
  let (r, g, b) = PLOT_COLORS[idx % PLOT_COLORS.len()];
  format!("rgb({},{},{})", r, g, b)
}

/// Render the ternary diagram to a complete SVG document.
fn render_ternary_svg(
  all_series: &[Vec<Triple>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  custom_colors: &[String],
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let (bg_color, stroke_color, grid_color, label_color) = theme_colors();

  // Margins leave room for the corner labels.
  let margin = 24.0 * sf;
  let avail_w = render_width as f64 - 2.0 * margin;
  let avail_h = render_height as f64 - 2.0 * margin;

  // Equilateral triangle: height = side * sqrt(3)/2. Fit the largest triangle
  // that fits inside the available area while staying centered.
  let tri_h_ratio = 3.0_f64.sqrt() / 2.0;
  let side = (avail_w).min(avail_h / tri_h_ratio);
  let tri_h = side * tri_h_ratio;

  let cx = render_width as f64 / 2.0;
  let bottom_y = (render_height as f64 + tri_h) / 2.0;
  let top_y = bottom_y - tri_h;

  // Triangle vertices in pixel space.
  let v_top = (cx, top_y); // component 1
  let v_bl = (cx - side / 2.0, bottom_y); // component 2
  let v_br = (cx + side / 2.0, bottom_y); // component 3

  // Map a normalized triple to pixel coordinates via barycentric blend.
  let to_px = |t: &Triple| -> (f64, f64) {
    let (a, b, c) = *t;
    (
      a * v_top.0 + b * v_bl.0 + c * v_br.0,
      a * v_top.1 + b * v_bl.1 + c * v_br.1,
    )
  };

  let mut svg = String::new();

  // Background.
  svg.push_str(&format!(
    "<rect width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
    render_width, render_height, bg_color
  ));

  // Grid lines parallel to each edge at 0.2 intervals.
  let grid_w = (sf * 0.5).max(1.0);
  for i in 1..5 {
    let f = i as f64 / 5.0;
    // Lines of constant component 1 (parallel to the bottom edge).
    let p1 = to_px(&(f, 1.0 - f, 0.0));
    let p2 = to_px(&(f, 0.0, 1.0 - f));
    push_line(&mut svg, p1, p2, grid_color, grid_w);
    // Lines of constant component 2 (parallel to the right edge).
    let p1 = to_px(&(1.0 - f, f, 0.0));
    let p2 = to_px(&(0.0, f, 1.0 - f));
    push_line(&mut svg, p1, p2, grid_color, grid_w);
    // Lines of constant component 3 (parallel to the left edge).
    let p1 = to_px(&(1.0 - f, 0.0, f));
    let p2 = to_px(&(0.0, 1.0 - f, f));
    push_line(&mut svg, p1, p2, grid_color, grid_w);
  }

  // Triangle outline.
  let stroke_w = (sf * 1.0).max(1.0);
  svg.push_str(&format!(
    "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" \
     fill=\"none\" stroke=\"{}\" stroke-width=\"{:.1}\"/>\n",
    v_top.0, v_top.1, v_bl.0, v_bl.1, v_br.0, v_br.1, stroke_color, stroke_w
  ));

  // Corner labels.
  let font_size = 13.0 * sf;
  push_label(
    &mut svg,
    v_top.0,
    v_top.1 - font_size * 0.5,
    "middle",
    font_size,
    label_color,
    "1",
  );
  push_label(
    &mut svg,
    v_bl.0 - font_size * 0.4,
    v_bl.1 + font_size,
    "end",
    font_size,
    label_color,
    "2",
  );
  push_label(
    &mut svg,
    v_br.0 + font_size * 0.4,
    v_br.1 + font_size,
    "start",
    font_size,
    label_color,
    "3",
  );

  // Data points.
  let radius = 3.0 * sf;
  for (idx, series) in all_series.iter().enumerate() {
    let color = series_color(idx, custom_colors);
    for t in series {
      let (px, py) = to_px(t);
      svg.push_str(&format!(
        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\"/>\n",
        px, py, radius, color
      ));
    }
  }

  let mut buf = format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{}</svg>",
    svg_width, svg_height, render_width, render_height, svg
  );

  if full_width {
    let old = format!("width=\"{}\" height=\"{}\"", svg_width, svg_height);
    buf = buf.replacen(&old, "width=\"100%\"", 1);
  }

  buf
}

/// Append an SVG line segment.
fn push_line(
  svg: &mut String,
  p1: (f64, f64),
  p2: (f64, f64),
  color: &str,
  width: f64,
) {
  svg.push_str(&format!(
    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
     stroke=\"{}\" stroke-width=\"{:.1}\"/>\n",
    p1.0, p1.1, p2.0, p2.1, color, width
  ));
}

/// Append an SVG text label.
fn push_label(
  svg: &mut String,
  x: f64,
  y: f64,
  anchor: &str,
  font_size: f64,
  color: &str,
  text: &str,
) {
  svg.push_str(&format!(
    "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" \
     font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{}\">{}</text>\n",
    x, y, anchor, font_size, color, text
  ));
}
