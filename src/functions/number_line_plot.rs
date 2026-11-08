#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::interval_ast::is_interval;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, PLOT_COLORS, RESOLUTION_SCALE,
  parse_image_size,
};

const DEFAULT_NLP_WIDTH: u32 = 360;
/// WL's layout at the default 360-point width, in points: the axis lies
/// `ROW_PITCH` below the top for each series, the series split the space
/// above it evenly, and the tick labels take `LABEL_ROOM` below it.
const ROW_PITCH: f64 = 22.125;
const LABEL_ROOM: f64 = 15.171875;
/// Width of a series' line and radius of its point markers.
const SERIES_THICKNESS: f64 = 1.6;
const POINT_RADIUS: f64 = 2.25;
/// An open (excluded) endpoint: a ring from `OPEN_INNER` to `OPEN_OUTER`.
const OPEN_INNER: f64 = 2.09;
const OPEN_OUTER: f64 = 2.33;
/// Axis and tick strokes, tick lengths (pointing up) and label metrics.
const AXIS_STROKE: f64 = 0.2;
const MAJOR_TICK: f64 = 2.31;
const MINOR_TICK: f64 = 1.39;
const TICK_FONT: f64 = 10.0;
const TICK_LABEL_BASELINE: f64 = 12.05;

/// A single interval span with inclusivity flags for each endpoint.
struct Span {
  lo: f64,
  hi: f64,
  lo_inclusive: bool,
  hi_inclusive: bool,
}

/// Represents the kinds of data a NumberLinePlot can display.
enum NumberLineData {
  /// A list of discrete point values.
  Points(Vec<f64>),
  /// A list of intervals with endpoint inclusivity.
  Intervals(Vec<Span>),
  /// A predicate over a variable, sampled in [xmin, xmax].
  Predicate {
    body: Expr,
    var: String,
    xmin: f64,
    xmax: f64,
  },
}

/// NumberLinePlot[data] or NumberLinePlot[pred, {x, xmin, xmax}]
pub fn number_line_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut svg_width = DEFAULT_NLP_WIDTH;
  let mut full_width = false;

  // Parse options (Rules at the end)
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ImageSize"
      && let Some((w, _h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    {
      svg_width = w;
      full_width = fw;
    }
  }

  // Determine the data spec(s); `data -> labels` labels the points.
  let (args, point_labels) = split_number_line_labels(args);
  let args = args.as_slice();
  let series = parse_number_line_args(args)?;
  if series.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "NumberLinePlot: no data provided".into(),
    ));
  }

  let num_rows = series.len().max(1);
  let unit = svg_width as f64 / DEFAULT_NLP_WIDTH as f64;
  let svg_height =
    ((ROW_PITCH * num_rows as f64 + LABEL_ROOM) * unit).ceil() as u32;

  // Compute the global x range across all series.
  let (x_min, x_max) = compute_global_x_range(&series);

  // Generate SVG
  let svg = render_number_line_svg(
    &series,
    &point_labels,
    x_min,
    x_max,
    svg_width,
    svg_height,
    full_width,
  );

  Ok(crate::graphics_result(svg))
}

/// Take the labels off `NumberLinePlot[data -> labels, …]` and turn an
/// association into its values (one series per value that is a list, or a
/// single series of scalar values). Returns the arguments with plain data
/// first, plus `(value, label)` for every labeled point of that data.
fn split_number_line_labels(args: &[Expr]) -> (Vec<Expr>, Vec<(f64, String)>) {
  let mut args = args.to_vec();
  let mut labels = Vec::new();
  let first =
    evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
  let first = match &first {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let data =
        evaluate_expr_to_expr(pattern).unwrap_or_else(|_| *pattern.clone());
      let names = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| *replacement.clone());
      if let (Expr::List(values), Expr::List(names)) = (&data, &names) {
        for (v, name) in values.iter().zip(names.iter()) {
          if let Some(x) = try_eval_to_f64(v) {
            let text = match name {
              Expr::String(t) => t.clone(),
              other => crate::syntax::format_expr(
                other,
                crate::syntax::ExprForm::Output,
              ),
            };
            labels.push((x, text));
          }
        }
      }
      data
    }
    _ => first,
  };
  let first = match &first {
    Expr::Association(pairs) => {
      Expr::List(pairs.iter().map(|(_, v)| v.clone()).collect())
    }
    _ => first,
  };
  args[0] = first;
  (args, labels)
}

/// Parse NumberLinePlot arguments into one or more series.
fn parse_number_line_args(
  args: &[Expr],
) -> Result<Vec<NumberLineData>, InterpreterError> {
  let first = evaluate_expr_to_expr(&args[0])?;

  // Case: NumberLinePlot[pred, {x, xmin, xmax}] — predicate form
  if args.len() >= 2 {
    if let Expr::List(spec) = &args[1]
      && spec.len() >= 3
      && let Expr::Identifier(var) = &spec[0]
    {
      let xmin_expr = evaluate_expr_to_expr(&spec[1])?;
      let xmax_expr = evaluate_expr_to_expr(&spec[2])?;
      if let (Some(xmin), Some(xmax)) =
        (try_eval_to_f64(&xmin_expr), try_eval_to_f64(&xmax_expr))
      {
        // `{pred1, pred2, …}` plots one row per predicate.
        let predicates: Vec<Expr> = match &first {
          Expr::List(items) => items.to_vec(),
          other => vec![other.clone()],
        };
        return Ok(
          predicates
            .into_iter()
            .map(|body| NumberLineData::Predicate {
              body,
              var: var.clone(),
              xmin,
              xmax,
            })
            .collect(),
        );
      }
    }
    // Case: NumberLinePlot[pred, x] — predicate with default range
    if let Expr::Identifier(var) = &args[1] {
      return Ok(vec![NumberLineData::Predicate {
        body: first,
        var: var.clone(),
        xmin: -10.0,
        xmax: 10.0,
      }]);
    }
  }

  // Case: Interval[{a, b}, ...] as direct argument
  if let Some(spans) = is_interval(&first) {
    let intervals = parse_interval_spans(&spans);
    if !intervals.is_empty() {
      return Ok(vec![NumberLineData::Intervals(intervals)]);
    }
  }

  // Case: List of specs — could be multiple series or a single list of numbers
  if let Expr::List(items) = &first {
    if items.is_empty() {
      return Ok(vec![NumberLineData::Points(vec![])]);
    }

    // Check if first element is a list or interval (multiple series)
    let first_is_compound =
      matches!(&items[0], Expr::List(_)) || is_interval(&items[0]).is_some();

    if first_is_compound {
      let mut series = Vec::new();
      for item in items {
        let item_eval = evaluate_expr_to_expr(item)?;
        if let Some(spans) = is_interval(&item_eval) {
          let intervals = parse_interval_spans(&spans);
          if !intervals.is_empty() {
            series.push(NumberLineData::Intervals(intervals));
          }
        } else if let Expr::List(inner) = &item_eval {
          let points: Vec<f64> = inner
            .iter()
            .filter_map(|e| {
              let ev = evaluate_expr_to_expr(e).unwrap_or(e.clone());
              try_eval_to_f64(&ev)
            })
            .collect();
          series.push(NumberLineData::Points(points));
        }
      }
      if !series.is_empty() {
        return Ok(series);
      }
    }

    // Single list of numbers
    let points: Vec<f64> = items
      .iter()
      .filter_map(|e| {
        let ev = evaluate_expr_to_expr(e).unwrap_or(e.clone());
        try_eval_to_f64(&ev)
      })
      .collect();
    return Ok(vec![NumberLineData::Points(points)]);
  }

  Err(InterpreterError::EvaluationError(
    "NumberLinePlot: unsupported argument format".into(),
  ))
}

/// Convert interval spans from `is_interval()` into `Span` structs.
/// Interval endpoints are always inclusive (closed intervals).
fn parse_interval_spans(spans: &[(&Expr, &Expr)]) -> Vec<Span> {
  spans
    .iter()
    .filter_map(|(lo, hi)| {
      let lo_f = try_eval_to_f64_or_inf(lo);
      let hi_f = try_eval_to_f64_or_inf(hi);
      match (lo_f, hi_f) {
        (Some(a), Some(b)) => Some(Span {
          lo: a,
          hi: b,
          lo_inclusive: true,
          hi_inclusive: true,
        }),
        _ => None,
      }
    })
    .collect()
}

/// Try to convert an Expr to f64, treating Infinity as f64::INFINITY.
fn try_eval_to_f64_or_inf(expr: &Expr) -> Option<f64> {
  crate::functions::math_ast::try_eval_to_f64_with_infinity(expr)
}

/// Compute the global x range, adding padding.
fn compute_global_x_range(series: &[NumberLineData]) -> (f64, f64) {
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;

  for s in series {
    match s {
      NumberLineData::Points(pts) => {
        for &v in pts {
          if v.is_finite() {
            x_min = x_min.min(v);
            x_max = x_max.max(v);
          }
        }
      }
      NumberLineData::Intervals(intervals) => {
        for span in intervals {
          if span.lo.is_finite() {
            x_min = x_min.min(span.lo);
          }
          if span.hi.is_finite() {
            x_max = x_max.max(span.hi);
          }
          // For semi-infinite intervals, use the finite end +/- some padding
          if span.lo.is_infinite() && span.hi.is_finite() {
            x_min = x_min.min(span.hi - 5.0);
          }
          if span.hi.is_infinite() && span.lo.is_finite() {
            x_max = x_max.max(span.lo + 5.0);
          }
        }
      }
      NumberLineData::Predicate { xmin, xmax, .. } => {
        x_min = x_min.min(*xmin);
        x_max = x_max.max(*xmax);
      }
    }
  }

  if !x_min.is_finite() || !x_max.is_finite() {
    x_min = 0.0;
    x_max = 10.0;
  }

  // WL pads the data by an eighth of its span on each side.
  let range = x_max - x_min;
  let pad = if range.abs() < f64::EPSILON {
    1.0
  } else {
    range * 0.125
  };
  (x_min - pad, x_max + pad)
}

/// Evaluate a predicate at a given x value.
fn eval_predicate(body: &Expr, var: &str, x: f64) -> bool {
  use crate::functions::plot::substitute_var;
  let x_expr = Expr::Real(x);
  let substituted = substitute_var(body, var, &x_expr);
  if let Ok(result) = evaluate_expr_to_expr(&substituted) {
    matches!(result, Expr::Identifier(ref s) if s == "True")
  } else {
    false
  }
}

/// Sample a predicate and return spans where it is true.
/// Uses binary search to refine boundaries and determine inclusivity.
/// Spans that extend to the domain boundary are marked with infinite
/// endpoints so arrows are drawn (indicating the region continues).
fn sample_predicate(body: &Expr, var: &str, xmin: f64, xmax: f64) -> Vec<Span> {
  let n = 500;
  let step = (xmax - xmin) / n as f64;
  let mut intervals = Vec::new();
  let mut in_region = false;
  let mut region_start_idx: usize = 0;

  for i in 0..=n {
    let x = xmin + i as f64 * step;
    let val = eval_predicate(body, var, x);
    if val && !in_region {
      region_start_idx = i;
      in_region = true;
    } else if !val && in_region {
      let region_start = xmin + region_start_idx as f64 * step;
      let region_end = x; // first sample where predicate is false
      let at_domain_start = region_start_idx == 0;

      let (lo, lo_inclusive) = if at_domain_start {
        (f64::NEG_INFINITY, false)
      } else {
        // Refine lo boundary between (region_start - step) and region_start
        refine_boundary(body, var, region_start - step, region_start)
      };
      // Refine hi boundary between (region_end - step) and region_end
      let (hi_boundary, hi_at_boundary) =
        refine_boundary(body, var, region_end, region_end - step);
      let hi_inclusive = hi_at_boundary;

      intervals.push(Span {
        lo,
        hi: hi_boundary,
        lo_inclusive,
        hi_inclusive,
      });
      in_region = false;
    }
  }
  if in_region {
    let region_start = xmin + region_start_idx as f64 * step;
    let at_domain_start = region_start_idx == 0;

    let (lo, lo_inclusive) = if at_domain_start {
      (f64::NEG_INFINITY, false)
    } else {
      refine_boundary(body, var, region_start - step, region_start)
    };

    // Region extends to domain end — use +Infinity so an arrow is drawn
    intervals.push(Span {
      lo,
      hi: f64::INFINITY,
      lo_inclusive,
      hi_inclusive: false,
    });
  }
  intervals
}

/// Binary-search to find the precise transition point between a false sample
/// and a true sample, then determine inclusivity at the snapped boundary.
/// Returns `(boundary, inclusive)`.
fn refine_boundary(
  body: &Expr,
  var: &str,
  x_false: f64,
  x_true: f64,
) -> (f64, bool) {
  let mut lo = x_false;
  let mut hi = x_true;
  for _ in 0..50 {
    let mid = f64::midpoint(lo, hi);
    if eval_predicate(body, var, mid) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  // `hi` is approximately the first value where the predicate is true.
  // Snap to a nearby "nice" number (integer or simple fraction).
  let boundary = snap_to_nice_value(hi);
  let inclusive = eval_predicate(body, var, boundary);
  (boundary, inclusive)
}

/// Snap a floating-point value to a nearby integer or simple fraction
/// if it is extremely close, to avoid floating-point noise at boundaries.
fn snap_to_nice_value(x: f64) -> f64 {
  let rounded = x.round();
  if (x - rounded).abs() < 1e-9 {
    return rounded;
  }
  let doubled = (x * 2.0).round() / 2.0;
  if (x - doubled).abs() < 1e-9 {
    return doubled;
  }
  let tenths = (x * 10.0).round() / 10.0;
  if (x - tenths).abs() < 1e-9 {
    return tenths;
  }
  x
}

/// Render the complete SVG for the NumberLinePlot.
fn render_number_line_svg(
  series: &[NumberLineData],
  point_labels: &[(f64, String)],
  x_min: f64,
  x_max: f64,
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> std::string::String {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;
  // Render pixels per WL point (the layout scales with the width).
  let sf =
    RESOLUTION_SCALE as f64 * svg_width as f64 / DEFAULT_NLP_WIDTH as f64;

  // The axis spans the picture (0.5 points in from the left, 1.5 from
  // the right), and the padded range maps onto it.
  let axis_x0 = 0.5 * sf;
  let axis_x1 = render_width as f64 - 1.5 * sf;
  let plot_width = axis_x1 - axis_x0;
  let x_to_px =
    |x: f64| -> f64 { axis_x0 + (x - x_min) / (x_max - x_min) * plot_width };

  let num_rows = series.len().max(1);
  let axis_y = ROW_PITCH * num_rows as f64 * sf;
  let row_spacing = axis_y / (num_rows as f64 + 1.0);

  let (bg_color, axis_color, _origin_color, label_fill, _title_fill) =
    theme_colors();

  let mut svg = String::new();

  // Background
  svg.push_str(&format!(
    "<rect width=\"{render_width}\" height=\"{render_height}\" fill=\"{bg_color}\"/>\n"
  ));

  // A run of the line, with its endpoint markers or, for an unbounded
  // end, an arrowhead whose tip sits 2% of the axis in from its end.
  let arrow_len = 0.04 * plot_width;
  let arrow_half = 3.77 * sf;
  let line_w = SERIES_THICKNESS * sf;
  let draw_span = |svg: &mut String, span: &Span, y: f64, color: &str| {
    let lo_tip = axis_x0 + 0.02 * plot_width;
    let hi_tip = axis_x1 - 0.02 * plot_width;
    let px_lo = if span.lo.is_infinite() {
      lo_tip + arrow_len
    } else {
      x_to_px(span.lo)
    };
    let px_hi = if span.hi.is_infinite() {
      hi_tip - arrow_len
    } else {
      x_to_px(span.hi)
    };
    svg.push_str(&format!(
      "<line x1=\"{px_lo:.1}\" y1=\"{y:.1}\" x2=\"{px_hi:.1}\" y2=\"{y:.1}\" \
       stroke=\"{color}\" stroke-width=\"{line_w:.1}\"/>\n"
    ));
    if span.lo.is_finite() {
      draw_endpoint(svg, px_lo, y, sf, color, bg_color, span.lo_inclusive);
    } else {
      draw_arrow(svg, lo_tip, px_lo, y, arrow_half, color);
    }
    if span.hi.is_finite() {
      draw_endpoint(svg, px_hi, y, sf, color, bg_color, span.hi_inclusive);
    } else {
      draw_arrow(svg, hi_tip, px_hi, y, arrow_half, color);
    }
  };

  // Draw each series row (first series closest to axis, rest stacked upward).
  for (row_idx, data) in series.iter().enumerate() {
    let color = series_color(row_idx);
    let data_y = axis_y - (row_idx + 1) as f64 * row_spacing;

    match data {
      NumberLineData::Points(pts) => {
        let radius = POINT_RADIUS * sf;
        for &v in pts {
          if v.is_finite() {
            let px = x_to_px(v);
            svg.push_str(&format!(
              "<circle cx=\"{px:.1}\" cy=\"{data_y:.1}\" r=\"{radius:.1}\" fill=\"{color}\"/>\n"
            ));
          }
        }
        // Labels from `data -> labels` sit above their points.
        if row_idx == 0 {
          for (v, text) in point_labels {
            if v.is_finite() {
              svg.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
                 font-family=\"sans-serif\" font-size=\"{:.0}\" \
                 fill=\"{label_fill}\">{}</text>\n",
                x_to_px(*v),
                data_y - radius - 3.0 * sf,
                TICK_FONT * sf,
                crate::functions::graphics::box_string_to_svg(text)
              ));
            }
          }
        }
      }
      NumberLineData::Intervals(intervals) => {
        for span in intervals {
          draw_span(&mut svg, span, data_y, &color);
        }
      }
      NumberLineData::Predicate {
        body,
        var,
        xmin,
        xmax,
      } => {
        for span in &sample_predicate(body, var, *xmin, *xmax) {
          draw_span(&mut svg, span, data_y, &color);
        }
      }
    }
  }

  // The axis line, with tick marks pointing up from it and the labels of
  // the major ones below.
  let axis_w = AXIS_STROKE * sf;
  svg.push_str(&format!(
    "<line x1=\"{axis_x0:.1}\" y1=\"{axis_y:.1}\" x2=\"{axis_x1:.1}\" y2=\"{axis_y:.1}\" \
     stroke=\"{axis_color}\" stroke-width=\"{axis_w:.1}\"/>\n"
  ));
  use crate::functions::plot_axes::{AxisScale, AxisTicks};
  let ticks = AxisTicks::new(
    AxisScale {
      min: x_min,
      max: x_max,
      log: false,
    },
    false,
    None,
  );
  let font_size = TICK_FONT * sf;
  let tick = |svg: &mut String, v: f64, len: f64| {
    let px = x_to_px(v);
    svg.push_str(&format!(
      "<line x1=\"{px:.1}\" y1=\"{axis_y:.1}\" x2=\"{px:.1}\" y2=\"{:.1}\" \
       stroke=\"{axis_color}\" stroke-width=\"{axis_w:.1}\"/>\n",
      axis_y - len * sf,
    ));
  };
  for &v in &ticks.minors {
    tick(&mut svg, v, MINOR_TICK);
  }
  for (v, label) in &ticks.majors {
    tick(&mut svg, *v, MAJOR_TICK);
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
       fill=\"{label_fill}\">{label}</text>\n",
      x_to_px(*v),
      axis_y + TICK_LABEL_BASELINE * sf,
    ));
  }

  // Wrap in SVG element
  let mut buf = format!(
    "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {render_width} {render_height}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{svg}</svg>"
  );

  if full_width {
    // Replace width/height with 100%
    let old = format!("width=\"{svg_width}\" height=\"{svg_height}\"");
    let new = "width=\"100%\"".to_string();
    buf = buf.replacen(&old, &new, 1);
  }

  buf
}

/// An arrowhead with its tip at `tip` and its base at `base`, on the line
/// at height `y`.
fn draw_arrow(
  svg: &mut String,
  tip: f64,
  base: f64,
  y: f64,
  half: f64,
  color: &str,
) {
  svg.push_str(&format!(
    "<polygon points=\"{tip:.1},{y:.1} {base:.1},{:.1} {base:.1},{:.1}\" fill=\"{color}\"/>\n",
    y - half,
    y + half,
  ));
}

/// Draw an endpoint marker: a filled disk for an included endpoint, a
/// ring around the background color for an excluded one. `sf` is render
/// pixels per point.
fn draw_endpoint(
  svg: &mut String,
  cx: f64,
  cy: f64,
  sf: f64,
  color: &str,
  bg_color: &str,
  inclusive: bool,
) {
  if inclusive {
    let r = POINT_RADIUS * sf;
    svg.push_str(&format!(
      "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r:.1}\" fill=\"{color}\"/>\n"
    ));
  } else {
    let r = f64::midpoint(OPEN_INNER, OPEN_OUTER) * sf;
    let stroke_w = (OPEN_OUTER - OPEN_INNER) * sf;
    svg.push_str(&format!(
      "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r:.1}\" \
       fill=\"{bg_color}\" stroke=\"{color}\" stroke-width=\"{stroke_w:.1}\"/>\n"
    ));
  }
}

/// Get theme-appropriate colors.
fn theme_colors() -> (
  &'static str,
  &'static str,
  &'static str,
  &'static str,
  &'static str,
) {
  if crate::is_dark_mode() {
    ("#1a1a1a", "#999999", "#444444", "#999", "#e0e0e0")
  } else {
    ("#ffffff", "#666666", "#cccccc", "#666", "#333")
  }
}

/// Get color for a series by index.
fn series_color(idx: usize) -> String {
  let (r, g, b) = PLOT_COLORS[idx % PLOT_COLORS.len()];
  format!("rgb({r},{g},{b})")
}
