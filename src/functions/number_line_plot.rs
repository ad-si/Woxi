use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::interval_ast::is_interval;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  PLOT_COLORS, RESOLUTION_SCALE, format_tick, is_major_tick, nice_step,
  parse_image_size,
};
use crate::syntax::Expr;

const DEFAULT_NLP_WIDTH: u32 = 360;
const DEFAULT_NLP_HEIGHT: u32 = 60;
/// Height per row when multiple series are displayed.
const ROW_HEIGHT: u32 = 40;
/// Top/bottom padding for the overall plot.
const PADDING_TOP: u32 = 15;
const PADDING_BOTTOM: u32 = 30;

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
      && let Some((w, _h, fw)) = parse_image_size(replacement)
    {
      svg_width = w;
      full_width = fw;
    }
  }

  // Determine the data spec(s).
  let series = parse_number_line_args(args)?;
  if series.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "NumberLinePlot: no data provided".into(),
    ));
  }

  let num_rows = series.len();
  let svg_height = if num_rows <= 1 {
    DEFAULT_NLP_HEIGHT
  } else {
    PADDING_TOP + num_rows as u32 * ROW_HEIGHT + PADDING_BOTTOM
  };

  // Compute the global x range across all series.
  let (x_min, x_max) = compute_global_x_range(&series);

  // Generate SVG
  let svg = render_number_line_svg(
    &series, x_min, x_max, svg_width, svg_height, full_width,
  )?;

  Ok(crate::graphics_result(svg))
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
        return Ok(vec![NumberLineData::Predicate {
          body: first,
          var: var.clone(),
          xmin,
          xmax,
        }]);
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

  let range = x_max - x_min;
  let pad = if range.abs() < f64::EPSILON {
    1.0
  } else {
    range * 0.08
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
/// Endpoints are marked exclusive (open circles) since the boundary
/// is determined by sampling and typically corresponds to strict inequalities.
fn sample_predicate(body: &Expr, var: &str, xmin: f64, xmax: f64) -> Vec<Span> {
  let n = 500;
  let step = (xmax - xmin) / n as f64;
  let mut intervals = Vec::new();
  let mut in_region = false;
  let mut region_start = xmin;

  for i in 0..=n {
    let x = xmin + i as f64 * step;
    let val = eval_predicate(body, var, x);
    if val && !in_region {
      region_start = x;
      in_region = true;
    } else if !val && in_region {
      intervals.push(Span {
        lo: region_start,
        hi: x - step,
        lo_inclusive: false,
        hi_inclusive: false,
      });
      in_region = false;
    }
  }
  if in_region {
    intervals.push(Span {
      lo: region_start,
      hi: xmax,
      lo_inclusive: false,
      hi_inclusive: false,
    });
  }
  intervals
}

/// Render the complete SVG for the NumberLinePlot.
fn render_number_line_svg(
  series: &[NumberLineData],
  x_min: f64,
  x_max: f64,
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let margin_left = 60.0 * sf;
  let margin_right = 20.0 * sf;
  let plot_width = render_width as f64 - margin_left - margin_right;

  let num_rows = series.len();
  let axis_area_top = PADDING_TOP as f64 * sf;
  let axis_area_bottom = PADDING_BOTTOM as f64 * sf;
  let usable_height = render_height as f64 - axis_area_top - axis_area_bottom;
  let row_height = if num_rows <= 1 {
    usable_height
  } else {
    usable_height / num_rows as f64
  };

  // Map x value to pixel coordinate.
  let x_to_px = |x: f64| -> f64 {
    margin_left + (x - x_min) / (x_max - x_min) * plot_width
  };

  let (bg_color, axis_color, _origin_color, label_fill, _title_fill) =
    theme_colors();

  let mut svg = String::new();

  // Background
  svg.push_str(&format!(
    "<rect width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
    render_width, render_height, bg_color
  ));

  // Offset for data above the axis line.
  let data_offset = 10.0 * sf;
  let axis_x0 = margin_left;
  let axis_x1 = margin_left + plot_width;

  // Single axis line at the bottom of the plot area.
  let axis_y = axis_area_top
    + if num_rows <= 1 {
      usable_height / 2.0
    } else {
      (num_rows - 1) as f64 * row_height + row_height / 2.0
    };
  svg.push_str(&format!(
    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
     stroke=\"{}\" stroke-width=\"{:.0}\"/>\n",
    axis_x0, axis_y, axis_x1, axis_y, axis_color, sf
  ));

  // Draw each series row (first series on top, last at the bottom near axis).
  for (row_idx, data) in series.iter().enumerate() {
    let color = series_color(row_idx);
    // First series closest to the axis, subsequent series stacked upward.
    let data_y = axis_y - data_offset - row_idx as f64 * row_height;

    match data {
      NumberLineData::Points(pts) => {
        let radius = 4.0 * sf;
        for &v in pts {
          if v.is_finite() {
            let px = x_to_px(v);
            svg.push_str(&format!(
              "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\"/>\n",
              px, data_y, radius, color
            ));
          }
        }
      }
      NumberLineData::Intervals(intervals) => {
        let bar_half = 2.5 * sf;
        let circle_r = 3.5 * sf;
        for span in intervals {
          let px_lo = if span.lo.is_infinite() {
            axis_x0
          } else {
            x_to_px(span.lo)
          };
          let px_hi = if span.hi.is_infinite() {
            axis_x1
          } else {
            x_to_px(span.hi)
          };
          // Draw interval bar
          svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
             fill=\"{}\"/>\n",
            px_lo,
            data_y - bar_half,
            (px_hi - px_lo).max(0.0),
            bar_half * 2.0,
            color,
          ));
          // Draw endpoint markers
          if span.lo.is_finite() {
            draw_endpoint(
              &mut svg,
              px_lo,
              data_y,
              circle_r,
              &color,
              bg_color,
              span.lo_inclusive,
            );
          } else {
            draw_arrow_left(&mut svg, axis_x0, data_y, bar_half, &color);
          }
          if span.hi.is_finite() {
            draw_endpoint(
              &mut svg,
              px_hi,
              data_y,
              circle_r,
              &color,
              bg_color,
              span.hi_inclusive,
            );
          } else {
            draw_arrow_right(&mut svg, axis_x1, data_y, bar_half, &color);
          }
        }
      }
      NumberLineData::Predicate {
        body,
        var,
        xmin,
        xmax,
      } => {
        let intervals = sample_predicate(body, var, *xmin, *xmax);
        let bar_half = 2.5 * sf;
        let circle_r = 3.5 * sf;
        for span in &intervals {
          let px_lo = x_to_px(span.lo);
          let px_hi = x_to_px(span.hi);
          svg.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
             fill=\"{}\"/>\n",
            px_lo,
            data_y - bar_half,
            (px_hi - px_lo).max(0.0),
            bar_half * 2.0,
            color,
          ));
          draw_endpoint(
            &mut svg,
            px_lo,
            data_y,
            circle_r,
            &color,
            bg_color,
            span.lo_inclusive,
          );
          draw_endpoint(
            &mut svg,
            px_hi,
            data_y,
            circle_r,
            &color,
            bg_color,
            span.hi_inclusive,
          );
        }
      }
    }
  }

  // Draw tick marks and labels on the bottom axis
  let tick_y = axis_y;
  let x_range = x_max - x_min;
  let major_step = nice_step(x_range, 6);
  let minor_step = major_step / 5.0;
  let major_tick_len = 5.0 * sf;
  let minor_tick_len = 3.0 * sf;
  let font_size = 14.0 * sf;

  // Minor ticks
  let first_minor = (x_min / minor_step).ceil() * minor_step;
  let mut tick_val = first_minor;
  while tick_val <= x_max + minor_step * 1e-9 {
    if !is_major_tick(tick_val, major_step) {
      let px = x_to_px(tick_val);
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"{}\" stroke-width=\"{:.0}\"/>\n",
        px,
        tick_y,
        px,
        tick_y + minor_tick_len,
        axis_color,
        sf * 0.5
      ));
    }
    tick_val += minor_step;
  }

  // Major ticks with labels
  let first_major = (x_min / major_step).ceil() * major_step;
  tick_val = first_major;
  while tick_val <= x_max + major_step * 1e-9 {
    if is_major_tick(tick_val, major_step) {
      let px = x_to_px(tick_val);
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"{}\" stroke-width=\"{:.0}\"/>\n",
        px,
        tick_y,
        px,
        tick_y + major_tick_len,
        axis_color,
        sf
      ));
      let label = format_tick(tick_val);
      svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{:.0}\" \
         fill=\"{}\">{}</text>\n",
        px,
        tick_y + major_tick_len + font_size * 1.1,
        font_size,
        label_fill,
        label
      ));
    }
    tick_val += major_step;
  }

  // Wrap in SVG element
  let mut buf = format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{}</svg>",
    svg_width, svg_height, render_width, render_height, svg
  );

  if full_width {
    // Replace width/height with 100%
    let old = format!("width=\"{}\" height=\"{}\"", svg_width, svg_height);
    let new = "width=\"100%\"".to_string();
    buf = buf.replacen(&old, &new, 1);
  }

  Ok(buf)
}

/// Draw a left-pointing arrow for -Infinity.
fn draw_arrow_left(svg: &mut String, x: f64, y: f64, size: f64, color: &str) {
  svg.push_str(&format!(
    "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"/>\n",
    x - size * 2.0,
    y,
    x,
    y - size,
    x,
    y + size,
    color
  ));
}

/// Draw a right-pointing arrow for +Infinity.
fn draw_arrow_right(svg: &mut String, x: f64, y: f64, size: f64, color: &str) {
  svg.push_str(&format!(
    "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"/>\n",
    x + size * 2.0,
    y,
    x,
    y - size,
    x,
    y + size,
    color
  ));
}

/// Draw an endpoint circle: filled for inclusive, empty (stroke-only) for exclusive.
fn draw_endpoint(
  svg: &mut String,
  cx: f64,
  cy: f64,
  r: f64,
  color: &str,
  bg_color: &str,
  inclusive: bool,
) {
  let stroke_w = r * 0.5;
  if inclusive {
    svg.push_str(&format!(
      "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\"/>\n",
      cx, cy, r, color
    ));
  } else {
    svg.push_str(&format!(
      "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" \
       fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.1}\"/>\n",
      cx, cy, r, bg_color, color, stroke_w
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
  format!("rgb({},{},{})", r, g, b)
}
