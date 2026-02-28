use plotters::prelude::*;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{LabelPosition, StyledLabel};
use crate::functions::graphics::{Color as WoxiColor, parse_color};
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

pub(crate) const DEFAULT_WIDTH: u32 = 360;
pub(crate) const DEFAULT_HEIGHT: u32 = 225;
/// Internal rendering resolution multiplier for sub-pixel precision.
/// Plotters maps to integer coordinates, so we render at a higher resolution
/// and scale down via SVG viewBox to get smooth curves.
pub(crate) const RESOLUTION_SCALE: u32 = 10;
pub(crate) const NUM_SAMPLES: usize = 500;

/// Substitute all occurrences of a variable with a value in an expression
pub(crate) fn substitute_var(expr: &Expr, var: &str, value: &Expr) -> Expr {
  match expr {
    Expr::Identifier(name) if name == var => value.clone(),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(|a| substitute_var(a, var, value)).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(substitute_var(left, var, value)),
      right: Box::new(substitute_var(right, var, value)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(substitute_var(operand, var, value)),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| substitute_var(i, var, value))
        .collect(),
    ),
    other => other.clone(),
  }
}

/// Evaluate the function body at a given x value
pub(crate) fn evaluate_at_point(body: &Expr, var: &str, x: f64) -> Option<f64> {
  let substituted = substitute_var(body, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  try_eval_to_f64(&result)
}

/// Split points into contiguous finite segments, breaking at NaN/Infinity
pub(crate) fn split_into_segments(
  points: &[(f64, f64)],
) -> Vec<Vec<(f64, f64)>> {
  let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
  let mut current: Vec<(f64, f64)> = Vec::new();

  for &(x, y) in points {
    if y.is_finite() {
      current.push((x, y));
    } else if current.len() > 1 {
      segments.push(std::mem::take(&mut current));
    } else {
      current.clear();
    }
  }
  if current.len() > 1 {
    segments.push(current);
  }
  segments
}

/// Compute a "nice" major tick step given the axis range and desired label count.
pub(crate) fn nice_step(range: f64, target_labels: usize) -> f64 {
  let raw = range / target_labels as f64;
  let mag = 10_f64.powf(raw.abs().log10().floor());
  let norm = raw / mag;
  let nice = if norm <= 1.0 {
    1.0
  } else if norm <= 2.0 {
    2.0
  } else if norm <= 5.0 {
    5.0
  } else {
    10.0
  };
  nice * mag
}

/// Check whether a tick value falls on a major tick grid.
pub(crate) fn is_major_tick(v: f64, step: f64) -> bool {
  if step == 0.0 {
    return true;
  }
  let remainder = (v / step).round() * step - v;
  remainder.abs() < step * 1e-9
}

/// Format a tick value, dropping the trailing ".0" for integers.
pub(crate) fn format_tick(v: f64) -> String {
  if (v - v.round()).abs() < 1e-9 {
    format!("{}", v.round() as i64)
  } else {
    format!("{v:.1}")
  }
}

/// Filling mode for line plots.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Filling {
  None,
  Axis,
}

/// Options for line-based plots (Plot, ListLinePlot, etc.).
pub(crate) struct PlotOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub filling: Filling,
  pub plot_label: Option<StyledLabel>,
  pub axes_label: Option<(String, String)>,
  pub plot_style: Vec<WoxiColor>,
  /// Per-axis visibility: (x_axis, y_axis). Both true = default.
  pub axes: (bool, bool),
  /// Ticks option: true = show tick marks and labels (default), false = hide
  pub ticks: bool,
  /// Number of sample points for Plot[] (default: NUM_SAMPLES)
  pub plot_points: usize,
}

impl Default for PlotOptions {
  fn default() -> Self {
    Self {
      svg_width: DEFAULT_WIDTH,
      svg_height: DEFAULT_HEIGHT,
      full_width: false,
      filling: Filling::None,
      plot_label: None,
      axes_label: None,
      plot_style: Vec::new(),
      axes: (true, true),
      ticks: true,
      plot_points: NUM_SAMPLES,
    }
  }
}

/// Default Wolfram plot color palette (ColorData[97]).
pub(crate) const PLOT_COLORS: [(u8, u8, u8); 6] = [
  (0x5E, 0x81, 0xB5), // blue
  (0xE0, 0x93, 0x2C), // orange
  (0x8F, 0xB0, 0x32), // green
  (0xD9, 0x51, 0x19), // red
  (0x6B, 0x48, 0x9D), // purple
  (0x8E, 0xB1, 0xCC), // light blue
];

/// Generate SVG for a 2D plot using plotters.
/// When `full_width` is true, the SVG uses `width="100%"` to fill its container.
/// Accepts multiple series of points, each drawn in a different color.
pub(crate) fn generate_svg(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  let opts = PlotOptions {
    svg_width,
    svg_height,
    full_width,
    ..Default::default()
  };
  generate_svg_with_options(all_points, x_range, y_range, &opts)
}

/// Generate SVG for a 2D plot with filling (legacy wrapper for list_plot callers).
pub(crate) fn generate_svg_with_filling(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  generate_svg_with_options(all_points, x_range, y_range, opts)
}

/// Core SVG generation for 2D line plots with full option support.
fn generate_svg_with_options(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let svg_width = opts.svg_width;
  let svg_height = opts.svg_height;
  let full_width = opts.full_width;
  let filling = opts.filling;
  let (show_x_axis, show_y_axis) = opts.axes;
  let show_ticks = opts.ticks;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let sf = RESOLUTION_SCALE as f64;
  let s = RESOLUTION_SCALE as i32;

  // Compute dynamic margins for labels
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  let has_x_axis_label =
    opts.axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_y_axis_label =
    opts.axes_label.as_ref().is_some_and(|(_, y)| !y.is_empty());

  let top_margin = if has_plot_label { 25 * s } else { 10 * s };

  // Label areas and margins computed per-axis.
  // Setting a label area to 0 suppresses that axis line in plotters.
  let bottom_extra = if show_x_axis && show_ticks && has_x_axis_label {
    16.0 * sf
  } else {
    0.0
  };
  let x_label_area: u32 = if !show_x_axis {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    25 * RESOLUTION_SCALE + bottom_extra as u32
  };
  let y_label_area: u32 = if !show_y_axis {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    40 * RESOLUTION_SCALE
  };
  let margin_left: u32 = if show_y_axis {
    10 * s as u32
  } else {
    5 * s as u32
  };
  let margin_right: u32 = 10 * s as u32;
  let margin_bottom: u32 = if show_x_axis {
    10 * s as u32
  } else {
    5 * s as u32
  };

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&WHITE)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let tick = 4 * s;
    let dark_gray = RGBColor(0x66, 0x66, 0x66);
    let light_gray = RGBColor(0xCC, 0xCC, 0xCC);

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(margin_right)
      .margin_bottom(margin_bottom)
      .margin_left(margin_left)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Configure mesh: per-axis tick counts and sizes, unified axis style.
    // When a label area is 0, plotters suppresses that axis border line.
    let x_labels_count;
    let y_labels_count;
    let x_tick_size;
    let y_tick_size;
    let x_major;
    let y_major;
    if show_ticks && (show_x_axis || show_y_axis) {
      // Compute nice major tick step for each visible axis
      let xmaj = nice_step(x_max - x_min, 5);
      let ymaj = nice_step(y_max - y_min, 5);
      x_major = xmaj;
      y_major = ymaj;
      let x_minor = xmaj / 5.0;
      let y_minor = ymaj / 5.0;
      x_labels_count = if show_x_axis {
        ((x_max - x_min) / x_minor).round() as usize + 1
      } else {
        0
      };
      y_labels_count = if show_y_axis {
        ((y_max - y_min) / y_minor).round() as usize + 1
      } else {
        0
      };
      x_tick_size = if show_x_axis { tick } else { 0 };
      y_tick_size = if show_y_axis { tick } else { 0 };
    } else {
      x_major = 1.0;
      y_major = 1.0;
      x_labels_count = 0;
      y_labels_count = 0;
      x_tick_size = 0;
      y_tick_size = 0;
    }
    let any_axis = show_x_axis || show_y_axis;
    let axis_style = if any_axis {
      dark_gray.stroke_width(RESOLUTION_SCALE)
    } else {
      ShapeStyle::from(&WHITE).stroke_width(0)
    };
    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_labels_count)
      .y_labels(y_labels_count)
      .x_label_formatter(&move |v: &f64| {
        if x_labels_count > 0 && is_major_tick(*v, x_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if y_labels_count > 0 && is_major_tick(*v, y_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .axis_style(axis_style)
      .label_style(
        ("sans-serif", RESOLUTION_SCALE as f64 * 11.0)
          .into_font()
          .color(&dark_gray),
      )
      .set_tick_mark_size(LabelAreaPosition::Left, y_tick_size)
      .set_tick_mark_size(LabelAreaPosition::Bottom, x_tick_size)
      .draw()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Draw lighter origin lines through x=0 and y=0 if visible
    let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
    if y_min < 0.0 && y_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x_min, 0.0), (x_max, 0.0)],
          origin_line,
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }
    if x_min < 0.0 && x_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(0.0, y_min), (0.0, y_max)],
          origin_line,
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    for (series_idx, points) in all_points.iter().enumerate() {
      let (r, g, b) = series_color(&opts.plot_style, series_idx);
      let color = RGBColor(r, g, b);
      let segments = split_into_segments(points);

      // Draw filled area before the line so the line renders on top
      if filling == Filling::Axis {
        let fill_style = RGBColor(r, g, b).mix(0.2).filled();
        for segment in &segments {
          if segment.len() < 2 {
            continue;
          }
          let mut poly_points: Vec<(f64, f64)> = segment.clone();
          // Close the polygon along y=0 (the axis)
          poly_points.push((segment.last().unwrap().0, 0.0));
          poly_points.push((segment.first().unwrap().0, 0.0));
          chart
            .draw_series(std::iter::once(Polygon::new(poly_points, fill_style)))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("Plot: {e}"))
            })?;
        }
      }

      for segment in &segments {
        chart
          .draw_series(std::iter::once(PathElement::new(
            segment.clone(),
            color.stroke_width(15), // 1.5px at display size
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      }
    }

    root
      .present()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Inject label SVG elements before </svg>
  if has_plot_label || has_x_axis_label || has_y_axis_label {
    let margin_left_f = margin_left as f64;
    let margin_right_f = margin_right as f64;
    let margin_bottom_f = margin_bottom as f64;
    let margin_top = top_margin as f64;
    let plot_x0 = margin_left_f + y_label_area as f64;
    let plot_w = render_width as f64
      - margin_left_f
      - margin_right_f
      - y_label_area as f64;
    let plot_h =
      render_height as f64 - margin_top - margin_bottom_f - x_label_area as f64;
    let axis_y = margin_top + plot_h;
    let font_size = sf * 11.0;
    let title_font_size = sf * 13.0;

    if let Some(insert_pos) = buf.rfind("</svg>") {
      let mut labels_svg = String::new();

      // AxesLabel
      if let Some((x_label, y_label)) = &opts.axes_label {
        if !x_label.is_empty() {
          let cx = plot_x0 + plot_w / 2.0;
          let base_y = axis_y + font_size * 1.5;
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"#666\">{}</text>\n",
            html_escape(x_label)
          ));
        }
        if !y_label.is_empty() {
          let cy = margin_top + plot_h / 2.0;
          let lx = margin_left_f + font_size * 0.8;
          labels_svg.push_str(&format!(
            "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"#666\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
            html_escape(y_label)
          ));
        }
      }

      // PlotLabel
      if let Some(sl) = &opts.plot_label
        && !sl.text.is_empty()
      {
        let cx = plot_x0 + plot_w / 2.0;
        let ty = margin_top - title_font_size * 0.5;
        let fs = sl.font_size.map(|f| f * sf).unwrap_or(title_font_size);
        let fill = sl
          .color
          .as_ref()
          .map(|c| c.to_svg_rgb())
          .unwrap_or_else(|| "#333".to_string());
        let mut style_attrs = String::new();
        if sl.bold {
          style_attrs.push_str(" font-weight=\"bold\"");
        }
        if sl.italic {
          style_attrs.push_str(" font-style=\"italic\"");
        }
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
             fill=\"{fill}\"{style_attrs}>{}</text>\n",
          html_escape(&sl.text)
        ));
      }

      buf.insert_str(insert_pos, &labels_svg);
    }
  }

  Ok(buf)
}

/// Get the (r, g, b) color for a series, using custom plot_style if available.
fn series_color(plot_style: &[WoxiColor], idx: usize) -> (u8, u8, u8) {
  if plot_style.is_empty() {
    PLOT_COLORS[idx % PLOT_COLORS.len()]
  } else {
    let c = &plot_style[idx % plot_style.len()];
    (
      (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  }
}

/// Generate SVG for a scatter plot with full option support (including PlotStyle).
pub(crate) fn generate_scatter_svg_with_options(
  all_series: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let svg_width = opts.svg_width;
  let svg_height = opts.svg_height;
  let full_width = opts.full_width;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&WHITE)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;
    let dark_gray = RGBColor(0x66, 0x66, 0x66);
    let light_gray = RGBColor(0xCC, 0xCC, 0xCC);

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(25 * RESOLUTION_SCALE)
      .y_label_area_size(40 * RESOLUTION_SCALE)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let x_major = nice_step(x_max - x_min, 5);
    let y_major = nice_step(y_max - y_min, 5);
    let x_minor_step = x_major / 5.0;
    let y_minor_step = y_major / 5.0;
    let x_tick_count = ((x_max - x_min) / x_minor_step).round() as usize + 1;
    let y_tick_count = ((y_max - y_min) / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_tick_count)
      .y_labels(y_tick_count)
      .x_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, x_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
      .label_style(
        ("sans-serif", RESOLUTION_SCALE as f64 * 11.0)
          .into_font()
          .color(&dark_gray),
      )
      .set_tick_mark_size(LabelAreaPosition::Left, tick)
      .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
      .draw()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Origin lines
    let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
    if y_min < 0.0 && y_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x_min, 0.0), (x_max, 0.0)],
          origin_line,
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }
    if x_min < 0.0 && x_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(0.0, y_min), (0.0, y_max)],
          origin_line,
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    // Draw scatter points using plotters Circle markers
    let marker_size = 3 * RESOLUTION_SCALE;
    for (series_idx, points) in all_series.iter().enumerate() {
      let (r, g, b) = series_color(&opts.plot_style, series_idx);
      let color = RGBColor(r, g, b);
      let finite_pts: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect();
      chart
        .draw_series(
          finite_pts
            .iter()
            .map(|&(x, y)| Circle::new((x, y), marker_size, color.filled())),
        )
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    root
      .present()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );
  Ok(buf)
}

/// Generate SVG for a bar chart using plotters.
pub(crate) fn generate_bar_svg(
  values: &[f64],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  chart_labels: &[String],
  chart_label_position: LabelPosition,
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let n = values.len();
  let y_max = values
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max)
    .max(0.0)
    * 1.1;
  let y_max = if y_max <= 0.0 { 1.0 } else { y_max };

  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  // Extra space for labels
  let has_chart_labels = !chart_labels.is_empty();
  let has_x_axis_label =
    axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());

  let top_margin = if has_plot_label { 25 * s } else { 10 * s };
  let bottom_extra = if has_chart_labels { 15.0 * sf } else { 0.0 }
    + if has_x_axis_label { 16.0 * sf } else { 0.0 };
  let x_label_area = 25 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 40 * RESOLUTION_SCALE;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;

    let tick = 4 * s;
    let dark_gray = RGBColor(0x66, 0x66, 0x66);

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(10 * s as u32)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(0.0..(n as f64), 0.0..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BarChart: {e}"))
      })?;

    let y_major = nice_step(y_max, 5);
    let y_minor_step = y_major / 5.0;
    let y_tick_count = (y_max / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(0) // no x ticks for bar chart
      .y_labels(y_tick_count)
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
      .label_style(("sans-serif", sf * 11.0).into_font().color(&dark_gray))
      .set_tick_mark_size(LabelAreaPosition::Left, tick)
      .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
      .draw()
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BarChart: {e}"))
      })?;

    // Draw bars as plotters Rectangle elements
    let gap = 0.1; // gap fraction per bar
    for (i, &val) in values.iter().enumerate() {
      let (br, bg, bb) = if !chart_style.is_empty() {
        let c = &chart_style[i % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else {
        PLOT_COLORS[0]
      };
      let color = RGBColor(br, bg, bb);
      let x0 = i as f64 + gap;
      let x1 = (i + 1) as f64 - gap;
      chart
        .draw_series(std::iter::once(Rectangle::new(
          [(x0, 0.0), (x1, val)],
          color.filled(),
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BarChart: {e}"))
        })?;
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Compute plot area coordinates (same logic as generate_axes_only_opts)
  let margin_left = 10.0 * sf;
  let margin_top = top_margin as f64;
  let margin_right = 10.0 * sf;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = margin_top;
  let plot_w =
    render_width as f64 - margin_left - margin_right - y_label_area as f64;
  let plot_h =
    render_height as f64 - margin_top - 10.0 * sf - x_label_area as f64;
  let axis_y = plot_y0 + plot_h;

  let font_size = sf * 11.0;
  let title_font_size = sf * 13.0;

  // Insert label SVG elements before </svg>
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();

    // ChartLabels: position based on chart_label_position
    if has_chart_labels {
      let slot_w = plot_w / n as f64;
      let map_y_val =
        |v: f64| -> f64 { plot_y0 + (y_max - v) / y_max * plot_h };
      for (i, label) in chart_labels.iter().enumerate().take(n) {
        let cx = plot_x0 + (i as f64 + 0.5) * slot_w;
        let (ly, fill) = match chart_label_position {
          LabelPosition::Above => {
            let bar_top = map_y_val(values[i]);
            (bar_top - font_size * 0.5, "#333")
          }
          LabelPosition::Center => {
            let bar_top = map_y_val(values[i]);
            let bar_center = (bar_top + axis_y) / 2.0 + font_size * 0.4;
            (bar_center, "white")
          }
          LabelPosition::Below => (axis_y + font_size * 1.5, "#666"),
        };
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{fill}\">{}</text>\n",
          html_escape(label)
        ));
      }
    }

    // AxesLabel: x-axis label centered below chart labels, y-axis label rotated
    if let Some((x_label, y_label)) = &axes_label {
      if !x_label.is_empty() {
        let cx = plot_x0 + plot_w / 2.0;
        let base_y = axis_y
          + if has_chart_labels {
            font_size * 1.5 + font_size * 1.3
          } else {
            font_size * 1.5
          };
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"#666\">{}</text>\n",
          html_escape(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"#666\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
          html_escape(y_label)
        ));
      }
    }

    // PlotLabel: centered above the chart
    if let Some(sl) = plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      let ty = margin_top - title_font_size * 0.5;
      let fs = sl.font_size.map(|f| f * sf).unwrap_or(title_font_size);
      let fill = sl
        .color
        .as_ref()
        .map(|c| c.to_svg_rgb())
        .unwrap_or_else(|| "#333".to_string());
      let mut style_attrs = String::new();
      if sl.bold {
        style_attrs.push_str(" font-weight=\"bold\"");
      }
      if sl.italic {
        style_attrs.push_str(" font-style=\"italic\"");
      }
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
           fill=\"{fill}\"{style_attrs}>{}</text>\n",
        html_escape(&sl.text)
      ));
    }

    buf.insert_str(insert_pos, &labels_svg);
  }

  Ok(buf)
}

/// Escape special characters for SVG text content.
fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// Generate SVG for a histogram using plotters.
pub(crate) fn generate_histogram_svg(
  values: &[f64],
  bin_count: Option<usize>,
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  // Use provided bin count or fall back to Sturges' rule
  let n = values.len();
  let num_bins = bin_count
    .unwrap_or_else(|| ((1.0 + (n as f64).log2()).ceil() as usize).max(1));
  let d_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
  let d_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let range = d_max - d_min;
  let bin_width = if range.abs() < f64::EPSILON {
    1.0
  } else {
    range / num_bins as f64
  };

  let mut counts = vec![0usize; num_bins];
  for &v in values {
    let idx = ((v - d_min) / bin_width).floor() as usize;
    counts[idx.min(num_bins - 1)] += 1;
  }

  let max_count = *counts.iter().max().unwrap_or(&1);
  let y_max = max_count as f64 * 1.1;
  let x_lo = d_min;
  let x_hi = d_min + num_bins as f64 * bin_width;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;
    let dark_gray = RGBColor(0x66, 0x66, 0x66);

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(25 * RESOLUTION_SCALE)
      .y_label_area_size(40 * RESOLUTION_SCALE)
      .build_cartesian_2d(x_lo..x_hi, 0.0..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("Histogram: {e}"))
      })?;

    let x_major = nice_step(x_hi - x_lo, 5);
    let y_major = nice_step(y_max, 5);
    let x_minor_step = x_major / 5.0;
    let y_minor_step = y_major / 5.0;
    let x_tick_count = ((x_hi - x_lo) / x_minor_step).round() as usize + 1;
    let y_tick_count = (y_max / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_tick_count)
      .y_labels(y_tick_count)
      .x_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, x_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick(*v)
        } else {
          String::new()
        }
      })
      .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
      .label_style(
        ("sans-serif", RESOLUTION_SCALE as f64 * 11.0)
          .into_font()
          .color(&dark_gray),
      )
      .set_tick_mark_size(LabelAreaPosition::Left, tick)
      .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
      .draw()
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("Histogram: {e}"))
      })?;

    // Draw contiguous histogram bars using plotters Rectangles
    let (r, g, b) = PLOT_COLORS[0];
    let color = RGBColor(r, g, b);
    for (i, &count) in counts.iter().enumerate() {
      let bx0 = x_lo + i as f64 * bin_width;
      let bx1 = bx0 + bin_width;
      chart
        .draw_series(std::iter::once(Rectangle::new(
          [(bx0, 0.0), (bx1, count as f64)],
          color.filled(),
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("Histogram: {e}"))
        })?;
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );
  Ok(buf)
}

/// Generate SVG with plotters axes and a blank chart area, returning the SVG
/// and the coordinate transform needed to overlay custom elements.
///
/// Returns (svg_string, plot_area_info) where plot_area_info contains the
/// pixel coordinates of the chart area for overlaying custom content.
pub(crate) struct PlotArea {
  pub svg: String,
  /// Pixel offset of plot area from SVG origin (at render resolution)
  pub plot_x0: f64,
  pub plot_y0: f64,
  pub plot_w: f64,
  pub plot_h: f64,
  pub render_width: u32,
  pub x_min: f64,
  pub x_max: f64,
  pub y_min: f64,
  pub y_max: f64,
}

/// Optional margin overrides for `generate_axes_only_opts`.
pub(crate) struct MarginOverrides {
  pub top_margin: u32,
  pub x_label_area: u32,
  pub y_label_area: u32,
}

/// Create a plotters chart with axes drawn, returning the SVG and coordinate info.
/// Callers can then append custom SVG elements using the coordinate transform.
pub(crate) fn generate_axes_only(
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<PlotArea, InterpreterError> {
  generate_axes_only_opts(
    x_range, y_range, svg_width, svg_height, full_width, None, None,
  )
}

/// Like `generate_axes_only` but with custom x-axis tick positions (tick marks only, no labels).
/// When `x_tick_positions` is `Some`, only those positions get tick marks on the x-axis.
/// When `margins` is `Some`, overrides the default margins for top, x_label_area, and y_label_area.
pub(crate) fn generate_axes_only_opts(
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  x_tick_positions: Option<&[f64]>,
  margins: Option<&MarginOverrides>,
) -> Result<PlotArea, InterpreterError> {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&WHITE)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;
    let dark_gray = RGBColor(0x66, 0x66, 0x66);

    let top_margin = margins
      .map(|m| m.top_margin)
      .unwrap_or(10 * RESOLUTION_SCALE);
    let x_label_area = margins.map(|m| m.x_label_area).unwrap_or(
      if x_tick_positions.is_some() {
        8 * RESOLUTION_SCALE
      } else {
        25 * RESOLUTION_SCALE
      },
    );
    let y_label_area = margins
      .map(|m| m.y_label_area)
      .unwrap_or(40 * RESOLUTION_SCALE);
    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin)
      .margin_right(10 * s as u32)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let y_major = nice_step(y_max - y_min, 5);
    let y_minor_step = y_major / 5.0;
    let y_tick_count = ((y_max - y_min) / y_minor_step).round() as usize + 1;

    if x_tick_positions.is_some() {
      // Custom tick mode: suppress plotters' x-axis ticks entirely.
      // We'll draw tick marks manually after computing the plot area.
      chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(0)
        .y_labels(y_tick_count)
        .y_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, y_major) {
            format_tick(*v)
          } else {
            String::new()
          }
        })
        .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
        .label_style(
          ("sans-serif", RESOLUTION_SCALE as f64 * 11.0)
            .into_font()
            .color(&dark_gray),
        )
        .set_tick_mark_size(LabelAreaPosition::Left, tick)
        .set_tick_mark_size(LabelAreaPosition::Bottom, 0)
        .draw()
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    } else {
      let x_major = nice_step(x_max - x_min, 5);
      let x_minor_step = x_major / 5.0;
      let x_tick_count = ((x_max - x_min) / x_minor_step).round() as usize + 1;

      chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(x_tick_count)
        .y_labels(y_tick_count)
        .x_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, x_major) {
            format_tick(*v)
          } else {
            String::new()
          }
        })
        .y_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, y_major) {
            format_tick(*v)
          } else {
            String::new()
          }
        })
        .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
        .label_style(
          ("sans-serif", RESOLUTION_SCALE as f64 * 11.0)
            .into_font()
            .color(&dark_gray),
        )
        .set_tick_mark_size(LabelAreaPosition::Left, tick)
        .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
        .draw()
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    root
      .present()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Compute the plot area coordinates.
  // ChartBuilder uses: margin on each side, x_label_area at bottom,
  // y_label_area on left. So the plot area starts at:
  let s = RESOLUTION_SCALE as f64;
  let margin = 10.0 * s;
  let top_margin_f = margins.map(|m| m.top_margin as f64).unwrap_or(margin);
  let y_label_area_f =
    margins.map(|m| m.y_label_area as f64).unwrap_or(40.0 * s);
  let x_label_area_f = margins.map(|m| m.x_label_area as f64).unwrap_or(
    if x_tick_positions.is_some() {
      8.0 * s
    } else {
      25.0 * s
    },
  );
  let plot_x0 = margin + y_label_area_f;
  let plot_y0 = top_margin_f;
  let plot_w = render_width as f64 - 2.0 * margin - y_label_area_f;
  let plot_h = render_height as f64 - top_margin_f - margin - x_label_area_f;

  // Draw custom x-axis tick marks if specified
  if let Some(positions) = x_tick_positions {
    let tick_len = 4.0 * s;
    let axis_y = plot_y0 + plot_h;
    let stroke_w = s;
    if let Some(insert_pos) = buf.rfind("</svg>") {
      let mut ticks_svg = String::new();
      for &pos in positions {
        let x = plot_x0 + (pos - x_min) / (x_max - x_min) * plot_w;
        ticks_svg.push_str(&format!(
          "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#666\" stroke-width=\"{stroke_w:.0}\"/>\n",
          axis_y, axis_y + tick_len
        ));
      }
      buf.insert_str(insert_pos, &ticks_svg);
    }
  }

  Ok(PlotArea {
    svg: buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    render_width,
    x_min,
    x_max,
    y_min,
    y_max,
  })
}

/// Rewrite the SVG header to use viewBox for display scaling.
fn rewrite_svg_header(
  buf: &mut String,
  svg_width: u32,
  svg_height: u32,
  render_width: u32,
  render_height: u32,
  full_width: bool,
) {
  if let Some(pos) = buf.find('>') {
    let new_header = if full_width {
      format!(
        "<svg width=\"100%\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
        render_width, render_height,
      )
    } else {
      format!(
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
        svg_width, svg_height, render_width, render_height,
      )
    };
    buf.replace_range(..pos, &new_header);
  }
}

/// Parse ImageSize option value into (width, height, full_width).
/// Supports: integer, {w, h}, and named sizes (Tiny, Small, Medium, Large, Full).
/// Full uses a 720px render resolution but emits `width="100%"` in SVG.
pub(crate) fn parse_image_size(value: &Expr) -> Option<(u32, u32, bool)> {
  match value {
    Expr::Integer(n) if *n > 0 => {
      let w = *n as u32;
      let h = (w as f64 * DEFAULT_HEIGHT as f64 / DEFAULT_WIDTH as f64).round()
        as u32;
      Some((w, h, false))
    }
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      let w = n.to_u32()?;
      if w == 0 {
        return None;
      }
      let h = (w as f64 * DEFAULT_HEIGHT as f64 / DEFAULT_WIDTH as f64).round()
        as u32;
      Some((w, h, false))
    }
    Expr::Real(f) if *f > 0.0 => {
      let w = f.round() as u32;
      let h = (w as f64 * DEFAULT_HEIGHT as f64 / DEFAULT_WIDTH as f64).round()
        as u32;
      Some((w, h, false))
    }
    Expr::List(items) if items.len() == 2 => {
      let w = match &items[0] {
        Expr::Integer(n) if *n > 0 => *n as u32,
        Expr::BigInteger(n) => {
          use num_traits::ToPrimitive;
          let v = n.to_u32()?;
          if v == 0 {
            return None;
          }
          v
        }
        Expr::Real(f) if *f > 0.0 => f.round() as u32,
        _ => return None,
      };
      let h = match &items[1] {
        Expr::Integer(n) if *n > 0 => *n as u32,
        Expr::BigInteger(n) => {
          use num_traits::ToPrimitive;
          let v = n.to_u32()?;
          if v == 0 {
            return None;
          }
          v
        }
        Expr::Real(f) if *f > 0.0 => f.round() as u32,
        _ => return None,
      };
      Some((w, h, false))
    }
    Expr::Identifier(name) => match name.as_str() {
      "Tiny" => Some((100, 63, false)),
      "Small" => Some((200, 125, false)),
      "Medium" => Some((DEFAULT_WIDTH, DEFAULT_HEIGHT, false)),
      "Large" => Some((480, 300, false)),
      "Full" => Some((720, 450, true)),
      _ => None,
    },
    _ => None,
  }
}

/// Implementation of Plot[f, {x, xmin, xmax}]
pub fn plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Plot requires at least 2 arguments: Plot[f, {x, xmin, xmax}]".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  // Parse options (Rule expressions after the first two arguments)
  let mut plot_opts = PlotOptions::default();
  // PlotRange and AspectRatio are applied after parsing all options
  let mut plot_range_x: Option<(f64, f64)> = None;
  let mut plot_range_y: Option<(f64, f64)> = None;
  let mut aspect_ratio: Option<f64> = None;
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) = parse_image_size(replacement) {
            plot_opts.svg_width = w;
            plot_opts.svg_height = h;
            plot_opts.full_width = fw;
          }
        }
        "PlotLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
            plot_opts.plot_label = Some(sl);
          }
        }
        "AxesLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Expr::List(items) = &val
            && items.len() >= 2
          {
            let x = crate::functions::chart::expr_to_label(&items[0])
              .unwrap_or_default();
            let y = crate::functions::chart::expr_to_label(&items[1])
              .unwrap_or_default();
            plot_opts.axes_label = Some((x, y));
          }
        }
        "PlotStyle" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          match &val {
            Expr::List(items) => {
              for item in items {
                // Support {Thick, RGBColor[...]} inner lists
                if let Expr::List(inner) = item {
                  for sub in inner {
                    if let Some(c) = parse_color(sub) {
                      plot_opts.plot_style.push(c);
                      break;
                    }
                  }
                } else if let Some(c) = parse_color(item) {
                  plot_opts.plot_style.push(c);
                }
              }
            }
            _ => {
              if let Some(c) = parse_color(&val) {
                plot_opts.plot_style.push(c);
              }
            }
          }
        }
        "PlotRange" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          match &val {
            // PlotRange -> All or Automatic: use auto range (default)
            Expr::Identifier(s)
              if s == "All" || s == "Automatic" || s == "Full" =>
            {
              plot_range_x = None;
              plot_range_y = None;
            }
            // PlotRange -> {ymin, ymax}: set y range only
            Expr::List(items) if items.len() == 2 => {
              let a = try_eval_to_f64(
                &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
              );
              let b = try_eval_to_f64(
                &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
              );
              match (a, b) {
                (Some(lo), Some(hi)) => plot_range_y = Some((lo, hi)),
                // {All/Automatic, {ymin, ymax}} or similar list-of-ranges form
                _ => {
                  if let (Expr::List(xa), Expr::List(ya)) =
                    (&items[0], &items[1])
                    && xa.len() == 2
                    && ya.len() == 2
                  {
                    let x0 = try_eval_to_f64(
                      &evaluate_expr_to_expr(&xa[0]).unwrap_or(xa[0].clone()),
                    );
                    let x1 = try_eval_to_f64(
                      &evaluate_expr_to_expr(&xa[1]).unwrap_or(xa[1].clone()),
                    );
                    let y0 = try_eval_to_f64(
                      &evaluate_expr_to_expr(&ya[0]).unwrap_or(ya[0].clone()),
                    );
                    let y1 = try_eval_to_f64(
                      &evaluate_expr_to_expr(&ya[1]).unwrap_or(ya[1].clone()),
                    );
                    if let (Some(x0), Some(x1)) = (x0, x1) {
                      plot_range_x = Some((x0, x1));
                    }
                    if let (Some(y0), Some(y1)) = (y0, y1) {
                      plot_range_y = Some((y0, y1));
                    }
                  }
                }
              }
            }
            _ => {}
          }
        }
        "Axes" => {
          let parse_bool =
            |e: &Expr| matches!(e, Expr::Identifier(s) if s == "True");
          match replacement.as_ref() {
            Expr::Identifier(s) if s == "True" => plot_opts.axes = (true, true),
            Expr::Identifier(s) if s == "False" => {
              plot_opts.axes = (false, false)
            }
            // {xbool, ybool}: independent per-axis control
            Expr::List(items) if items.len() == 2 => {
              plot_opts.axes = (parse_bool(&items[0]), parse_bool(&items[1]));
            }
            _ => {}
          }
        }
        "AspectRatio" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(r) = try_eval_to_f64(&val)
            && r > 0.0
          {
            aspect_ratio = Some(r);
          }
        }
        "PlotPoints" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          match &val {
            Expr::Integer(n) if *n > 0 => plot_opts.plot_points = *n as usize,
            _ => {}
          }
        }
        "Ticks" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "None" => plot_opts.ticks = false,
          Expr::Identifier(s) if s == "Automatic" || s == "All" => {
            plot_opts.ticks = true
          }
          _ => {}
        },
        _ => {}
      }
    }
  }

  // Apply AspectRatio: override svg_height based on width * ratio
  // (AspectRatio is height/width in Wolfram Language)
  if let Some(ar) = aspect_ratio {
    plot_opts.svg_height = (plot_opts.svg_width as f64 * ar).round() as u32;
  }

  // Parse iterator spec: {x, xmin, xmax}
  let (var_name, x_min, x_max) = match iter_spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Plot: iterator variable must be a symbol".into(),
          ));
        }
      };
      // Evaluate xmin and xmax
      let x_min_expr = evaluate_expr_to_expr(&items[1])?;
      let x_max_expr = evaluate_expr_to_expr(&items[2])?;
      let x_min = try_eval_to_f64(&x_min_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Plot: cannot evaluate xmin to a number".into(),
        )
      })?;
      let x_max = try_eval_to_f64(&x_max_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Plot: cannot evaluate xmax to a number".into(),
        )
      })?;
      (var, x_min, x_max)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Plot: second argument must be {x, xmin, xmax}".into(),
      ));
    }
  };

  // Collect function bodies: single function or list of functions
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  // Sample each function using the configured number of sample points
  let n_samples = plot_opts.plot_points.max(2);
  let step = (x_max - x_min) / (n_samples - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let mut points = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
      let x = x_min + i as f64 * step;
      if let Some(y) = evaluate_at_point(func_body, &var_name, x) {
        points.push((x, y));
      } else {
        points.push((x, f64::NAN));
      }
    }
    all_points.push(points);
  }

  // Compute Y range from finite values across all series
  let finite_ys: Vec<f64> = all_points
    .iter()
    .flat_map(|pts| pts.iter())
    .filter(|(_, y)| y.is_finite())
    .map(|(_, y)| *y)
    .collect();

  if finite_ys.is_empty() {
    // Wolfram returns -Graphics- even when no finite values are produced
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let y_data_min = finite_ys.iter().cloned().fold(f64::INFINITY, f64::min);
  let y_data_max = finite_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

  // Add 4% padding to the auto-computed y range
  let y_range = y_data_max - y_data_min;
  let padding = if y_range.abs() < f64::EPSILON {
    1.0
  } else {
    y_range * 0.04
  };
  let y_auto_min = y_data_min - padding;
  let y_auto_max = y_data_max + padding;

  // Apply PlotRange overrides (PlotRange -> {ymin, ymax} or {{xmin,xmax},{ymin,ymax}})
  let (x_display_min, x_display_max) = plot_range_x.unwrap_or((x_min, x_max));
  let (y_display_min, y_display_max) =
    plot_range_y.unwrap_or((y_auto_min, y_auto_max));

  // Generate SVG
  let svg = generate_svg_with_filling(
    &all_points,
    (x_display_min, x_display_max),
    (y_display_min, y_display_max),
    &plot_opts,
  )?;

  // Generate GraphicsBox expression for .nb export
  let rgb_values = [
    "0.24, 0.6, 0.8",
    "0.88, 0.58, 0.17",
    "0.56, 0.69, 0.20",
    "0.85, 0.32, 0.10",
    "0.42, 0.28, 0.61",
    "0.56, 0.69, 0.80",
  ];
  let mut box_elements = Vec::new();
  for (i, points) in all_points.iter().enumerate() {
    let rgb = rgb_values[i % rgb_values.len()];
    box_elements.push(format!("RGBColor[{rgb}]"));
    box_elements.push("AbsoluteThickness[2]".to_string());
    box_elements.push("Opacity[1.]".to_string());
    let segments = split_into_segments(points);
    box_elements.extend(crate::functions::graphicsbox::line_box(&segments));
  }
  let graphicsbox = crate::functions::graphicsbox::graphics_box(&box_elements);
  crate::capture_graphicsbox(&graphicsbox);

  // Return -Graphics- as the text representation
  Ok(crate::graphics_result(svg))
}
