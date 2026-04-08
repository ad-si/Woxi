use plotters::prelude::*;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{ChartLabel, LabelPosition, StyledLabel};
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

/// Return plotters colors adapted to the current light/dark theme.
pub(crate) fn plot_theme()
-> (RGBColor, RGBColor, RGBColor, &'static str, &'static str) {
  if crate::is_dark_mode() {
    // (background, axis_gray, origin_line_gray, label_fill, title_default_fill)
    (
      RGBColor(0x1a, 0x1a, 0x1a), // dark background
      RGBColor(0x99, 0x99, 0x99), // lighter axes for dark bg
      RGBColor(0x44, 0x44, 0x44), // subtle origin lines
      "#999",                     // axis label fill
      "#e0e0e0",                  // default plot label fill
    )
  } else {
    (
      RGBColor(0xFF, 0xFF, 0xFF), // white background
      RGBColor(0x66, 0x66, 0x66), // dark gray axes
      RGBColor(0xCC, 0xCC, 0xCC), // light gray origin lines
      "#666",                     // axis label fill
      "#333",                     // default plot label fill
    )
  }
}

/// Substitute all occurrences of a variable with a value in an expression
pub(crate) fn substitute_var(expr: &Expr, var: &str, value: &Expr) -> Expr {
  let sub = |e: &Expr| substitute_var(e, var, value);
  match expr {
    Expr::Identifier(name) if name == var => value.clone(),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(sub).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(sub(left)),
      right: Box::new(sub(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(sub(operand)),
    },
    Expr::List(items) => Expr::List(items.iter().map(sub).collect()),
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands.iter().map(sub).collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => {
      Expr::CompoundExpr(exprs.iter().map(sub).collect())
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(sub(pattern)),
      replacement: Box::new(sub(replacement)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(sub(pattern)),
      replacement: Box::new(sub(replacement)),
    },
    Expr::ReplaceAll { expr, rules } => Expr::ReplaceAll {
      expr: Box::new(sub(expr)),
      rules: Box::new(sub(rules)),
    },
    Expr::ReplaceRepeated { expr, rules } => Expr::ReplaceRepeated {
      expr: Box::new(sub(expr)),
      rules: Box::new(sub(rules)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(sub(func)),
      arg: Box::new(sub(arg)),
    },
    Expr::Postfix { expr, func } => Expr::Postfix {
      expr: Box::new(sub(expr)),
      func: Box::new(sub(func)),
    },
    Expr::Part { expr, index } => Expr::Part {
      expr: Box::new(sub(expr)),
      index: Box::new(sub(index)),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(sub(func)),
      args: args.iter().map(sub).collect(),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(sub(body)),
    },
    Expr::Association(pairs) => {
      Expr::Association(pairs.iter().map(|(k, v)| (sub(k), sub(v))).collect())
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default.as_ref().map(|d| Box::new(sub(d))),
    },
    other => other.clone(),
  }
}

/// Evaluate the function body at a given x value
pub(crate) fn evaluate_at_point(body: &Expr, var: &str, x: f64) -> Option<f64> {
  let substituted = substitute_var(body, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  try_eval_to_f64(&result)
}

/// Adaptively sample a function, adding more points where the function changes rapidly.
fn adaptive_sample(
  func_body: &Expr,
  var_name: &str,
  x_min: f64,
  x_max: f64,
  initial_n: usize,
  max_total: usize,
) -> Vec<(f64, f64)> {
  // Initial uniform sampling
  let step = (x_max - x_min) / (initial_n - 1) as f64;
  let mut points: Vec<(f64, f64)> = (0..initial_n)
    .map(|i| {
      let x = x_min + i as f64 * step;
      let y = evaluate_at_point(func_body, var_name, x).unwrap_or(f64::NAN);
      (x, y)
    })
    .collect();

  // Adaptive refinement passes
  let max_depth = 6;
  for _ in 0..max_depth {
    if points.len() >= max_total {
      break;
    }
    let mut new_points: Vec<(f64, f64)> = Vec::new();
    let budget = max_total - points.len();

    for i in 0..points.len().saturating_sub(1) {
      if new_points.len() >= budget {
        break;
      }
      let (x0, y0) = points[i];
      let (x1, y1) = points[i + 1];

      // Skip if interval is too small
      if (x1 - x0) < (x_max - x_min) * 1e-10 {
        continue;
      }

      let needs_refine = if !y0.is_finite() || !y1.is_finite() {
        // Refine near discontinuities to find the boundary
        true
      } else if i + 2 < points.len() {
        // Check curvature using three consecutive points
        let (x2, y2) = points[i + 2];
        if y2.is_finite() {
          // Linear interpolation error: how much does the middle point
          // deviate from the line connecting its neighbors?
          let y_interp = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0);
          let y_range = (y2 - y0).abs().max(1e-10);
          let deviation = (y1 - y_interp).abs() / y_range;
          deviation > 0.05
        } else {
          true
        }
      } else {
        false
      };

      if needs_refine {
        let xm = (x0 + x1) / 2.0;
        let ym = evaluate_at_point(func_body, var_name, xm).unwrap_or(f64::NAN);
        new_points.push((xm, ym));
      }
    }

    if new_points.is_empty() {
      break;
    }

    // Merge new points into sorted order
    points.extend(new_points);
    points.sort_by(|a, b| {
      a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
  }

  points
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

/// Format an AbsoluteTime value (seconds since 1900-01-01) as a date string.
pub(crate) fn format_date_tick(seconds: f64) -> String {
  let (year, month, day, _, _, _) =
    crate::functions::datetime_ast::absolute_seconds_to_date(seconds);
  let month_abbr = || match month {
    1 => "Jan",
    2 => "Feb",
    3 => "Mar",
    4 => "Apr",
    5 => "May",
    6 => "Jun",
    7 => "Jul",
    8 => "Aug",
    9 => "Sep",
    10 => "Oct",
    11 => "Nov",
    12 => "Dec",
    _ => "???",
  };
  if day == 1 && month == 1 {
    format!("{year}")
  } else if day == 1 {
    format!("{year} {}", month_abbr())
  } else {
    format!("{year} {} {day}", month_abbr())
  }
}

/// Date tick step specification for nice date axis ticks.
#[derive(Clone, Copy)]
pub(crate) enum DateStep {
  Years(i64),
  Months(i64),
  Days(i64),
}

/// Compute a nice step for date axis ticks based on the range in seconds.
pub(crate) fn nice_date_step_spec(range_seconds: f64) -> DateStep {
  let range_days = range_seconds / 86400.0;
  let range_years = range_days / 365.25;

  if range_years > 100.0 {
    let step = nice_step(range_years, 5) as i64;
    DateStep::Years(step.max(10))
  } else if range_years > 20.0 {
    DateStep::Years(5)
  } else if range_years > 8.0 {
    DateStep::Years(2)
  } else if range_years > 2.0 {
    DateStep::Years(1)
  } else if range_days > 180.0 {
    DateStep::Months(3)
  } else if range_days > 60.0 {
    DateStep::Months(1)
  } else if range_days > 14.0 {
    let step = nice_step(range_days, 5) as i64;
    DateStep::Days(step.max(1))
  } else {
    let step = nice_step(range_days, 5) as i64;
    DateStep::Days(step.max(1))
  }
}

/// Generate nice date tick positions between min and max AbsoluteTime values.
pub(crate) fn generate_date_ticks(x_min: f64, x_max: f64) -> Vec<f64> {
  use crate::functions::datetime_ast::{
    absolute_seconds_to_date, date_to_absolute_seconds,
  };

  let range = x_max - x_min;
  let step = nice_date_step_spec(range);
  let mut ticks = Vec::new();

  let (start_y, start_m, start_d, _, _, _) = absolute_seconds_to_date(x_min);

  match step {
    DateStep::Years(n) => {
      // Round start year down to multiple of n
      let mut y = (start_y / n) * n;
      if y > start_y {
        y -= n;
      }
      loop {
        let t = date_to_absolute_seconds(y, 1, 1, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        y += n;
      }
    }
    DateStep::Months(n) => {
      let mut y = start_y;
      let mut m = ((start_m - 1) / n) * n + 1;
      loop {
        let t = date_to_absolute_seconds(y, m, 1, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        m += n;
        while m > 12 {
          m -= 12;
          y += 1;
        }
      }
    }
    DateStep::Days(n) => {
      // Round start day down to multiple of n
      let mut y = start_y;
      let mut m = start_m;
      let mut d = ((start_d - 1) / n) * n + 1;
      loop {
        let t = date_to_absolute_seconds(y, m, d, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        // Advance by n days and normalize
        let next_t = t + (n as f64) * 86400.0;
        let (ny, nm, nd, _, _, _) = absolute_seconds_to_date(next_t);
        y = ny;
        m = nm;
        d = nd;
      }
    }
  }

  ticks
}

/// Approximate step in seconds for date ticks (used for tick count estimation).
pub(crate) fn nice_date_step(range_seconds: f64) -> f64 {
  match nice_date_step_spec(range_seconds) {
    DateStep::Years(n) => (n as f64) * 365.25 * 86400.0,
    DateStep::Months(n) => (n as f64) * 30.44 * 86400.0,
    DateStep::Days(n) => (n as f64) * 86400.0,
  }
}

/// Orientation info for placing log-axis labels.
enum LogAxisOrientation {
  /// Y-axis: labels placed at fixed x, varying y
  Y { x: f64, plot_top: f64, plot_h: f64 },
  /// X-axis: labels placed at fixed y, varying x
  X { y: f64, plot_left: f64, plot_w: f64 },
}

/// Inject SVG `<text>` elements for log-scale axis labels with proper
/// superscript rendering (e.g. 10 with superscript 6 instead of "1000000").
/// Intelligently selects which powers of 10 to label based on the range.
fn inject_log_axis_labels(
  out: &mut String,
  data_min: f64,
  data_max: f64,
  font_size: f64,
  fill: &str,
  orientation: LogAxisOrientation,
) {
  let log_min = data_min.log10();
  let log_max = data_max.log10();
  let decades = (log_max - log_min).abs();

  // Choose labeling step: label every 1, 2, or 3 powers of 10
  let step = if decades <= 8.0 {
    1
  } else if decades <= 16.0 {
    2
  } else {
    3
  };

  let exp_start = log_min.ceil() as i64;
  let exp_end = log_max.floor() as i64;

  // Align to step grid
  let first = if exp_start % step as i64 == 0 {
    exp_start
  } else {
    exp_start + (step as i64 - exp_start.rem_euclid(step as i64))
  };

  let (anchor, is_y) = match &orientation {
    LogAxisOrientation::Y { .. } => ("end", true),
    LogAxisOrientation::X { .. } => ("middle", false),
  };

  let mut exp = first;
  while exp <= exp_end {
    // Compute pixel position
    let frac = (exp as f64 - log_min) / (log_max - log_min);
    let pos = match &orientation {
      LogAxisOrientation::Y {
        plot_top, plot_h, ..
      } => plot_top + plot_h * (1.0 - frac),
      LogAxisOrientation::X {
        plot_left, plot_w, ..
      } => plot_left + plot_w * frac,
    };
    let (x, y) = match &orientation {
      LogAxisOrientation::Y { x, .. } => (*x, pos),
      LogAxisOrientation::X { y, .. } => (pos, *y),
    };

    let dy = if is_y { " dy=\"0.5ex\"" } else { "" };

    if exp == 0 {
      // 10^0 = 1
      out.push_str(&format!(
        "<text x=\"{x:.1}\" y=\"{y:.1}\"{dy} text-anchor=\"{anchor}\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{fill}\">1</text>\n"
      ));
    } else if exp == 1 {
      // 10^1 = 10
      out.push_str(&format!(
        "<text x=\"{x:.1}\" y=\"{y:.1}\"{dy} text-anchor=\"{anchor}\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{fill}\">10</text>\n"
      ));
    } else {
      // 10^n with superscript
      let sup_size = font_size * 0.7;
      out.push_str(&format!(
        "<text x=\"{x:.1}\" y=\"{y:.1}\"{dy} text-anchor=\"{anchor}\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{fill}\">10<tspan baseline-shift=\"super\" \
         font-size=\"{sup_size:.0}\">{exp}</tspan></text>\n"
      ));
    }

    exp += step as i64;
  }
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
  Bottom,
  Top,
  Value(f64),
}

/// Mesh mode for line plots.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mesh {
  None,
  All,
}

impl Filling {
  /// Compute the y-value to fill to, given the current plot y-range.
  /// Returns `None` for `Filling::None`.
  pub fn reference_y(&self, y_min: f64, y_max: f64) -> Option<f64> {
    match self {
      Filling::None => None,
      Filling::Axis => Some(0.0),
      Filling::Bottom => Some(y_min),
      Filling::Top => Some(y_max),
      Filling::Value(v) => Some(*v),
    }
  }
}

/// Parse a `Filling` option value from an expression.
pub(crate) fn parse_filling(replacement: &Expr) -> Filling {
  match replacement {
    Expr::Identifier(v) if v == "Axis" => Filling::Axis,
    Expr::Identifier(v) if v == "Automatic" => Filling::Axis,
    Expr::Identifier(v) if v == "Bottom" => Filling::Bottom,
    Expr::Identifier(v) if v == "Top" => Filling::Top,
    Expr::Identifier(v) if v == "None" => Filling::None,
    other => {
      let evaled =
        evaluate_expr_to_expr(other).unwrap_or_else(|_| other.clone());
      if let Some(v) = try_eval_to_f64(&evaled) {
        Filling::Value(v)
      } else {
        Filling::None
      }
    }
  }
}

/// Adjust y-range so the fill reference level is included.
/// For `Axis`, ensures y=0 is in range. For `Value(v)`, ensures v is in range.
/// `Bottom`/`Top`/`None` don't need adjustment (they use the range edges).
pub(crate) fn adjust_y_range_for_filling(
  filling: Filling,
  y_range: (f64, f64),
) -> (f64, f64) {
  let (mut y_lo, mut y_hi) = y_range;
  match filling {
    Filling::Axis => {
      if y_lo > 0.0 {
        y_lo = 0.0 - (y_hi - 0.0) * 0.04;
      }
      if y_hi < 0.0 {
        y_hi = 0.0 + (0.0 - y_lo) * 0.04;
      }
    }
    Filling::Value(v) => {
      if y_lo > v {
        y_lo = v - (y_hi - v) * 0.04;
      }
      if y_hi < v {
        y_hi = v + (v - y_lo) * 0.04;
      }
    }
    _ => {}
  }
  (y_lo, y_hi)
}

/// Options for line-based plots (Plot, ListLinePlot, etc.).
pub(crate) struct PlotOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub filling: Filling,
  pub mesh: Mesh,
  pub plot_label: Option<StyledLabel>,
  pub axes_label: Option<(String, String)>,
  pub plot_style: Vec<WoxiColor>,
  /// Per-axis visibility: (x_axis, y_axis). Both true = default.
  pub axes: (bool, bool),
  /// Ticks option: true = show tick marks and labels (default), false = hide
  pub ticks: bool,
  /// Number of sample points for Plot[] (default: NUM_SAMPLES)
  pub plot_points: usize,
  /// Legend labels for each series (empty = no legend)
  pub plot_legends: Vec<String>,
  /// Show horizontal grid lines (dashed)
  pub grid_lines_y: bool,
  /// Use frame (left+bottom border) instead of axes
  pub frame: bool,
  /// Format x-axis labels as dates (AbsoluteTime seconds since 1900-01-01)
  pub date_axis: bool,
  /// Whether x-axis is logarithmic (data is in log10 space)
  pub log_x: bool,
  /// Whether y-axis is logarithmic (data is in log10 space)
  pub log_y: bool,
}

impl Default for PlotOptions {
  fn default() -> Self {
    Self {
      svg_width: DEFAULT_WIDTH,
      svg_height: DEFAULT_HEIGHT,
      full_width: false,
      filling: Filling::None,
      mesh: Mesh::None,
      plot_label: None,
      axes_label: None,
      plot_style: Vec::new(),
      axes: (true, true),
      ticks: true,
      plot_points: NUM_SAMPLES,
      plot_legends: Vec::new(),
      grid_lines_y: false,
      frame: false,
      date_axis: false,
      log_x: false,
      log_y: false,
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

  let top_margin = if has_plot_label { 35 * s } else { 10 * s };

  // Label areas and margins computed per-axis.
  // Setting a label area to 0 suppresses that axis line in plotters.
  let bottom_extra = if show_x_axis && show_ticks && has_x_axis_label {
    24.0 * sf
  } else {
    0.0
  };
  let x_label_area: u32 = if !show_x_axis {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    40 * RESOLUTION_SCALE + bottom_extra as u32
  };
  let y_label_area: u32 = if !show_y_axis {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    65 * RESOLUTION_SCALE
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

  let (bg_color, dark_gray, light_gray, label_fill, title_default_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let tick = 4 * s;

    // Macro to configure mesh and draw series on any chart coordinate type.
    // This avoids duplicating the drawing code for each LogCoord combination.
    macro_rules! draw_chart {
      ($chart:expr) => {{
        let mut chart = $chart;

        // Configure mesh: tick counts, sizes, label formatting, axis style.
        let x_labels_count;
        let y_labels_count;
        let x_tick_size;
        let y_tick_size;
        let x_major;
        let y_major;
        let date_axis = opts.date_axis;
        let log_x = opts.log_x;
        let log_y = opts.log_y;
        if show_ticks && (show_x_axis || show_y_axis) {
          let xmaj = if date_axis {
            nice_date_step(x_max - x_min)
          } else {
            nice_step(x_max - x_min, 5)
          };
          let ymaj = nice_step(y_max - y_min, 5);
          x_major = xmaj;
          y_major = ymaj;
          let x_minor = if date_axis { xmaj } else { xmaj / 5.0 };
          let y_minor = ymaj / 5.0;
          x_labels_count = if !show_x_axis || date_axis {
            0
          } else if log_x {
            // Let LogCoord decide tick placement; ~10 labels for log axes
            10
          } else {
            ((x_max - x_min) / x_minor).round() as usize + 1
          };
          y_labels_count = if !show_y_axis {
            0
          } else if log_y {
            10
          } else {
            ((y_max - y_min) / y_minor).round() as usize + 1
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
        let axis_style = if opts.frame {
          ShapeStyle::from(&bg_color).stroke_width(0)
        } else if any_axis {
          dark_gray.stroke_width(RESOLUTION_SCALE)
        } else {
          ShapeStyle::from(&bg_color).stroke_width(0)
        };
        chart
          .configure_mesh()
          .disable_mesh()
          .x_labels(x_labels_count)
          .y_labels(y_labels_count)
          .x_label_formatter(&move |v: &f64| {
            if x_labels_count == 0 {
              return String::new();
            }
            if date_axis {
              format_date_tick(*v)
            } else if log_x {
              // Suppress plotters labels; we inject custom SVG with superscripts
              String::new()
            } else if is_major_tick(*v, x_major) {
              format_tick(*v)
            } else {
              String::new()
            }
          })
          .y_label_formatter(&move |v: &f64| {
            if y_labels_count == 0 {
              return String::new();
            }
            if log_y {
              String::new()
            } else if is_major_tick(*v, y_major) {
              format_tick(*v)
            } else {
              String::new()
            }
          })
          .axis_style(axis_style)
          .label_style(
            ("sans-serif", sf * 18.0)
              .into_font()
              .color(&dark_gray),
          )
          .set_tick_mark_size(LabelAreaPosition::Left, y_tick_size)
          .set_tick_mark_size(LabelAreaPosition::Bottom, x_tick_size)
          .draw()
          .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

        // Draw horizontal grid lines (dashed) when grid_lines_y is enabled
        // (only for linear y-axis; log axis grid not yet supported)
        if opts.grid_lines_y && !log_y {
          let grid_color = RGBColor(0x66, 0x66, 0x66).mix(0.5);
          let grid_step = nice_step(y_max - y_min, 5);
          let mut gy = (y_min / grid_step).ceil() * grid_step;
          while gy <= y_max {
            let dash_len = (x_max - x_min) * 0.005;
            let gap_len = dash_len * 2.0;
            let mut dx = x_min;
            while dx < x_max {
              let end = (dx + dash_len).min(x_max);
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![(dx, gy), (end, gy)],
                  grid_color.stroke_width(RESOLUTION_SCALE),
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
              dx += dash_len + gap_len;
            }
            gy += grid_step;
          }
        }

        // Draw frame (bottom border only) when frame mode is enabled
        if opts.frame {
          let frame_style = dark_gray.stroke_width(RESOLUTION_SCALE * 2);
          chart
            .draw_series(std::iter::once(PathElement::new(
              vec![(x_min, y_min), (x_max, y_min)],
              frame_style,
            )))
            .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
        }

        // Draw lighter origin lines through x=0 and y=0 if visible
        if !opts.frame {
          let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
          if y_min < 0.0 && y_max > 0.0 {
            chart
              .draw_series(std::iter::once(PathElement::new(
                vec![(x_min, 0.0), (x_max, 0.0)],
                origin_line,
              )))
              .map_err(|e| {
                InterpreterError::EvaluationError(format!("Plot: {e}"))
              })?;
          }
          if x_min < 0.0 && x_max > 0.0 {
            chart
              .draw_series(std::iter::once(PathElement::new(
                vec![(0.0, y_min), (0.0, y_max)],
                origin_line,
              )))
              .map_err(|e| {
                InterpreterError::EvaluationError(format!("Plot: {e}"))
              })?;
          }
        }

        for (series_idx, points) in all_points.iter().enumerate() {
          let (r, g, b) = series_color(&opts.plot_style, series_idx);
          let color = RGBColor(r, g, b);
          let segments = split_into_segments(points);

          // Draw filled area before the line so the line renders on top
          if let Some(ref_y) = filling.reference_y(y_min, y_max) {
            for segment in &segments {
              if segment.len() < 2 {
                continue;
              }
              chart
                .draw_series(AreaSeries::new(
                  segment.iter().copied(),
                  ref_y,
                  RGBColor(r, g, b).mix(0.2),
                ))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }

          for segment in &segments {
            chart
              .draw_series(LineSeries::new(
                segment.iter().copied(),
                color.stroke_width(15), // 1.5px at display size
              ))
              .map_err(|e| {
                InterpreterError::EvaluationError(format!("Plot: {e}"))
              })?;
          }

          // Draw mesh dots at each data point when Mesh -> All
          if opts.mesh == Mesh::All {
            let marker_size = 3 * RESOLUTION_SCALE;
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
              .map_err(|e| {
                InterpreterError::EvaluationError(format!("Plot: {e}"))
              })?;
          }
        }

        root
          .present()
          .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
      }};
    }

    // Build chart with appropriate coordinate types for log/linear axes.
    // LogCoord handles logarithmic tick placement, scaling, and labeling.
    // Each arm creates its own ChartBuilder because it borrows `root`.
    macro_rules! chart_builder {
      () => {
        ChartBuilder::on(&root)
          .margin_top(top_margin as u32)
          .margin_right(margin_right)
          .margin_bottom(margin_bottom)
          .margin_left(margin_left)
          .x_label_area_size(x_label_area)
          .y_label_area_size(y_label_area)
      };
    }
    let err = |e| InterpreterError::EvaluationError(format!("Plot: {e}"));
    match (opts.log_x, opts.log_y) {
      (false, false) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(x_min..x_max, y_min..y_max)
          .map_err(err)?
      ),
      (false, true) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(x_min..x_max, (y_min..y_max).log_scale())
          .map_err(err)?
      ),
      (true, false) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d((x_min..x_max).log_scale(), y_min..y_max)
          .map_err(err)?
      ),
      (true, true) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(
            (x_min..x_max).log_scale(),
            (y_min..y_max).log_scale()
          )
          .map_err(err)?
      ),
    }
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
    let font_size = sf * 18.0;
    let title_font_size = sf * 22.0;

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
             fill=\"{label_fill}\">{}</text>\n",
            html_escape(x_label)
          ));
        }
        if !y_label.is_empty() {
          let cy = margin_top + plot_h / 2.0;
          let lx = margin_left_f + font_size * 0.8;
          labels_svg.push_str(&format!(
            "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
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
          .unwrap_or_else(|| title_default_fill.to_string());
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

  // Inject logarithmic axis labels with superscript formatting
  if (opts.log_y && show_y_axis && show_ticks)
    || (opts.log_x && show_x_axis && show_ticks)
  {
    let margin_left_f = margin_left as f64;
    let margin_right_f = margin_right as f64;
    let margin_bottom_f = margin_bottom as f64;
    let margin_top_f = top_margin as f64;
    let plot_x0 = margin_left_f + y_label_area as f64;
    let plot_w = render_width as f64
      - margin_left_f
      - margin_right_f
      - y_label_area as f64;
    let plot_h = render_height as f64
      - margin_top_f
      - margin_bottom_f
      - x_label_area as f64;
    // Plotters SVG backend divides font size by 1.24
    let font_size = sf * 18.0 / 1.24;
    let label_color = if crate::is_dark_mode() {
      "#999"
    } else {
      "#666"
    };

    if let Some(insert_pos) = buf.rfind("</svg>") {
      let mut log_labels = String::new();

      if opts.log_y && show_y_axis {
        inject_log_axis_labels(
          &mut log_labels,
          y_min,
          y_max,
          font_size,
          label_color,
          LogAxisOrientation::Y {
            x: plot_x0 - font_size * 0.55,
            plot_top: margin_top_f,
            plot_h,
          },
        );
      }

      if opts.log_x && show_x_axis {
        inject_log_axis_labels(
          &mut log_labels,
          x_min,
          x_max,
          font_size,
          label_color,
          LogAxisOrientation::X {
            y: margin_top_f + plot_h + font_size * 1.3,
            plot_left: plot_x0,
            plot_w,
          },
        );
      }

      buf.insert_str(insert_pos, &log_labels);
    }
  }

  inject_legend(&mut buf, opts);

  Ok(buf)
}

/// Build a `PlotSource` from sampled plot data so that `Show` can later
/// merge multiple pre-rendered plots and re-render via plotters.
pub(crate) fn build_plot_source(
  all_points: &[Vec<(f64, f64)>],
  plot_style: &[WoxiColor],
  x_range: (f64, f64),
  y_range: (f64, f64),
  image_size: (u32, u32),
  is_scatter: bool,
) -> crate::syntax::PlotSource {
  let series = all_points
    .iter()
    .enumerate()
    .map(|(i, points)| {
      let color = series_color(plot_style, i);
      crate::syntax::PlotSeriesData {
        points: points.clone(),
        color,
        is_scatter,
      }
    })
    .collect();

  crate::syntax::PlotSource {
    series,
    x_range,
    y_range,
    image_size,
  }
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

  let (bg_color, dark_gray, light_gray, _label_fill, _title_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(40 * RESOLUTION_SCALE)
      .y_label_area_size(65 * RESOLUTION_SCALE)
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
        ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
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

      // Draw stem lines from each point to the fill reference level
      if let Some(ref_y) = opts.filling.reference_y(y_min, y_max) {
        let stem_style =
          RGBColor(r, g, b).mix(0.2).stroke_width(RESOLUTION_SCALE);
        for &(x, y) in &finite_pts {
          chart
            .draw_series(std::iter::once(PathElement::new(
              vec![(x, y), (x, ref_y)],
              stem_style,
            )))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("Plot: {e}"))
            })?;
        }
      }

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
  inject_legend(&mut buf, opts);
  Ok(buf)
}

/// Render a merged `PlotSource` (from `Show`) via plotters.
/// Handles both line and scatter series in one chart.
pub(crate) fn render_merged_plot_source(
  source: &crate::syntax::PlotSource,
) -> Result<String, InterpreterError> {
  let (x_min, x_max) = source.x_range;
  let (y_min, y_max) = source.y_range;
  let (svg_width, svg_height) = source.image_size;

  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let (bg_color, dark_gray, light_gray, _label_fill, _title_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(40 * RESOLUTION_SCALE)
      .y_label_area_size(65 * RESOLUTION_SCALE)
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
        ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
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

    // Draw each series
    let marker_size = 3 * RESOLUTION_SCALE;
    for sd in &source.series {
      let color = RGBColor(sd.color.0, sd.color.1, sd.color.2);

      if sd.is_scatter {
        // Scatter points
        let finite_pts: Vec<(f64, f64)> = sd
          .points
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
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      } else {
        // Line series (split into finite segments)
        let segments = split_into_segments(&sd.points);
        for segment in &segments {
          chart
            .draw_series(LineSeries::new(
              segment.iter().copied(),
              color.stroke_width(15),
            ))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("Plot: {e}"))
            })?;
        }
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
    false,
  );
  Ok(buf)
}

/// Generate SVG for a bar chart using plotters.
pub(crate) fn generate_bar_svg(
  groups: &[Vec<f64>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  chart_labels: &[ChartLabel],
  chart_label_position: LabelPosition,
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let n = groups.len(); // number of groups
  let k = groups.iter().map(|g| g.len()).max().unwrap_or(1); // max bars per group
  let y_max = groups
    .iter()
    .flat_map(|g| g.iter())
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

  let top_margin = if has_plot_label { 35 * s } else { 10 * s };
  let has_rotated_labels = chart_labels.iter().any(|l| l.rotation.abs() > 0.01);
  let label_extra = if has_rotated_labels {
    50.0 * sf // more space for angled labels
  } else if has_chart_labels {
    24.0 * sf
  } else {
    0.0
  };
  let bottom_extra =
    label_extra + if has_x_axis_label { 24.0 * sf } else { 0.0 };
  let x_label_area = 40 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 65 * RESOLUTION_SCALE;

  let (bg_color, dark_gray, _light_gray, label_fill, title_default_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;

    let tick = 4 * s;

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
      .label_style(("sans-serif", sf * 18.0).into_font().color(&dark_gray))
      .set_tick_mark_size(LabelAreaPosition::Left, tick)
      .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
      .draw()
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BarChart: {e}"))
      })?;

    // Draw bars as plotters Rectangle elements
    let gap = 0.1; // gap between groups
    for (gi, group) in groups.iter().enumerate() {
      let group_x0 = gi as f64 + gap;
      let group_x1 = (gi + 1) as f64 - gap;
      let group_w = group_x1 - group_x0;
      let bar_w = group_w / k as f64;

      for (bi, &val) in group.iter().enumerate() {
        let (br, bg, bb) = if !chart_style.is_empty() {
          // For grouped charts, color by bar index within group
          let color_idx = if k > 1 { bi } else { gi };
          let c = &chart_style[color_idx % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        } else if k > 1 {
          // Grouped: color by position within group
          PLOT_COLORS[bi % PLOT_COLORS.len()]
        } else {
          // Flat: single default color
          PLOT_COLORS[0]
        };
        let color = RGBColor(br, bg, bb);
        let x0 = group_x0 + bi as f64 * bar_w;
        let x1 = x0 + bar_w;
        chart
          .draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, val)],
            color.filled(),
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("BarChart: {e}"))
          })?;
      }
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

  let font_size = sf * 18.0;
  let title_font_size = sf * 22.0;

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
        // For Above/Center positioning, use the max value in the group
        let group_max =
          groups[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (ly, fill) = match chart_label_position {
          LabelPosition::Above => {
            let bar_top = map_y_val(group_max);
            (bar_top - font_size * 0.5, title_default_fill)
          }
          LabelPosition::Center => {
            let bar_top = map_y_val(group_max);
            let bar_center = (bar_top + axis_y) / 2.0 + font_size * 0.4;
            (bar_center, "white")
          }
          LabelPosition::Below => (axis_y + font_size * 1.5, label_fill),
        };
        // Mathematica Rotate is counterclockwise-positive; SVG is clockwise-positive
        let svg_rotation_deg = -label.rotation.to_degrees();
        let is_rotated = svg_rotation_deg.abs() > 0.01;
        if is_rotated {
          // With text-anchor=middle and rotation, the left half of the text
          // swings upward. Offset the pivot down so the highest point
          // (pivot_y - half_width * sin(angle)) stays below the axis.
          let char_width_estimate = font_size * 0.6;
          let half_text_w = label.text.len() as f64 * char_width_estimate / 2.0;
          let sin_a = svg_rotation_deg.to_radians().sin().abs();
          let offset = half_text_w * sin_a + font_size * 0.5;
          let ay = axis_y + offset;
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ay:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\" transform=\"rotate({svg_rotation_deg:.1},{cx:.1},{ay:.1})\">{}</text>\n",
            html_escape(&label.text)
          ));
        } else {
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\">{}</text>\n",
            html_escape(&label.text)
          ));
        }
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
           fill=\"{label_fill}\">{}</text>\n",
          html_escape(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
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
        .unwrap_or_else(|| title_default_fill.to_string());
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

/// Inject a legend into an SVG plot. Widens the SVG to make room for the legend
/// on the right side, then draws colored line swatches with text labels.
pub(crate) fn inject_legend(buf: &mut String, opts: &PlotOptions) {
  if opts.plot_legends.is_empty() {
    return;
  }

  let sf = RESOLUTION_SCALE as f64;
  let (_bg_color, _dark_gray, _light_gray, label_fill, _title_fill) =
    plot_theme();

  let font_size = sf * 18.0;
  let line_height = font_size * 1.6;
  let swatch_len = sf * 20.0;
  let swatch_gap = sf * 6.0;
  let legend_padding = sf * 10.0;

  // Measure legend width: swatch + gap + text
  let max_text_width = opts
    .plot_legends
    .iter()
    .map(|s| s.len() as f64 * font_size * 0.55)
    .fold(0.0_f64, f64::max);
  let legend_width = swatch_len + swatch_gap + max_text_width + legend_padding;

  // Parse current viewBox to widen
  let vb_re = regex::Regex::new(
    r#"viewBox="(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)""#,
  )
  .unwrap();

  let (vb_w, vb_h) = if let Some(caps) = vb_re.captures(buf) {
    let w: f64 = caps[3].parse().unwrap_or(0.0);
    let h: f64 = caps[4].parse().unwrap_or(0.0);
    (w, h)
  } else {
    return;
  };

  let new_vb_w = vb_w + legend_width;

  // Update viewBox width
  let old_vb = format!("viewBox=\"0 0 {} {}\"", vb_w as u32, vb_h as u32);
  let new_vb = format!("viewBox=\"0 0 {} {}\"", new_vb_w as u32, vb_h as u32);
  *buf = buf.replacen(&old_vb, &new_vb, 1);

  // Update width attribute if present (non-full-width)
  let w_re = regex::Regex::new(r#"width="(\d+)""#).unwrap();
  if let Some(caps) = w_re.captures(&buf.clone()) {
    let old_w: u32 = caps[1].parse().unwrap_or(0);
    if old_w > 0 {
      let render_w = vb_w as u32;
      let render_new_w = new_vb_w as u32;
      // Scale display width proportionally
      let new_display_w =
        (old_w as f64 * render_new_w as f64 / render_w as f64).round() as u32;
      let old_attr = format!("width=\"{}\"", old_w);
      let new_attr = format!("width=\"{}\"", new_display_w);
      *buf = buf.replacen(&old_attr, &new_attr, 1);

      // Also update height if present to maintain aspect ratio
      let h_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
      if let Some(hcaps) = h_re.captures(&buf.clone()) {
        let old_h: u32 = hcaps[1].parse().unwrap_or(0);
        if old_h > 0 {
          let new_display_h =
            (new_display_w as f64 * vb_h / new_vb_w).round() as u32;
          let old_hattr = format!("height=\"{}\"", old_h);
          let new_hattr = format!("height=\"{}\"", new_display_h);
          *buf = buf.replacen(&old_hattr, &new_hattr, 1);
        }
      }
    }
  }

  // Insert legend elements before </svg>
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut legend_svg = String::new();
    let legend_x = vb_w + legend_padding * 0.5;
    let n = opts.plot_legends.len();
    let legend_total_h = n as f64 * line_height;
    let legend_y0 = (vb_h - legend_total_h) / 2.0;

    for (i, label) in opts.plot_legends.iter().enumerate() {
      let (r, g, b) = series_color(&opts.plot_style, i);
      let y = legend_y0 + i as f64 * line_height + line_height * 0.5;

      // Colored line swatch
      legend_svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
         stroke=\"rgb({},{},{})\" stroke-width=\"{}\"/>\n",
        legend_x,
        y,
        legend_x + swatch_len,
        y,
        r,
        g,
        b,
        (sf * 1.5) as u32,
      ));

      // Text label
      let text_x = legend_x + swatch_len + swatch_gap;
      let text_y = y + font_size * 0.35;
      legend_svg.push_str(&format!(
        "<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\">{}</text>\n",
        html_escape(label),
      ));
    }

    buf.insert_str(insert_pos, &legend_svg);
  }
}

/// Generate SVG for a histogram using plotters.
/// Specifies how histogram bins are determined.
pub enum BinSpec {
  /// A fixed number of equal-width bins.
  Count(usize),
  /// Explicit bin edges (must be sorted, at least 2 elements).
  Edges(Vec<f64>),
}

pub(crate) fn generate_histogram_svg(
  values: &[f64],
  bin_spec: Option<BinSpec>,
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  // Build bin edges and counts based on the bin specification
  let (bin_edges, counts) = match bin_spec {
    Some(BinSpec::Edges(mut edges)) => {
      edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let num_bins = edges.len() - 1;
      let mut cnts = vec![0usize; num_bins];
      for &v in values {
        // Find the bin for this value
        for i in 0..num_bins {
          let in_bin = if i == num_bins - 1 {
            v >= edges[i] && v <= edges[i + 1]
          } else {
            v >= edges[i] && v < edges[i + 1]
          };
          if in_bin {
            cnts[i] += 1;
            break;
          }
        }
      }
      (edges, cnts)
    }
    _ => {
      // Use provided bin count or fall back to Sturges' rule
      let n = values.len();
      let num_bins = match &bin_spec {
        Some(BinSpec::Count(c)) => *c,
        _ => ((1.0 + (n as f64).log2()).ceil() as usize).max(1),
      };
      let d_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
      let d_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
      let range = d_max - d_min;
      let bin_width = if range.abs() < f64::EPSILON {
        1.0
      } else {
        range / num_bins as f64
      };

      let mut cnts = vec![0usize; num_bins];
      for &v in values {
        let idx = ((v - d_min) / bin_width).floor() as usize;
        cnts[idx.min(num_bins - 1)] += 1;
      }

      let edges: Vec<f64> = (0..=num_bins)
        .map(|i| d_min + i as f64 * bin_width)
        .collect();
      (edges, cnts)
    }
  };

  let num_bins = counts.len();
  let max_count = *counts.iter().max().unwrap_or(&1);
  let y_max = max_count as f64 * 1.1;
  let x_lo = bin_edges[0];
  let x_hi = bin_edges[num_bins];

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    let (bg_color, dark_gray, _light_gray, _label_fill, _title_fill) =
      plot_theme();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(40 * RESOLUTION_SCALE)
      .y_label_area_size(65 * RESOLUTION_SCALE)
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
        ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
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
      let bx0 = bin_edges[i];
      let bx1 = bin_edges[i + 1];
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

  let (bg_color, dark_gray, _light_gray, label_fill, _title_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = 4 * s;

    let top_margin = margins
      .map(|m| m.top_margin)
      .unwrap_or(10 * RESOLUTION_SCALE);
    let x_label_area = margins.map(|m| m.x_label_area).unwrap_or(
      if x_tick_positions.is_some() {
        12 * RESOLUTION_SCALE
      } else {
        40 * RESOLUTION_SCALE
      },
    );
    let y_label_area = margins
      .map(|m| m.y_label_area)
      .unwrap_or(65 * RESOLUTION_SCALE);
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
          ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
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
          ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
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
    margins.map(|m| m.y_label_area as f64).unwrap_or(65.0 * s);
  let x_label_area_f = margins.map(|m| m.x_label_area as f64).unwrap_or(
    if x_tick_positions.is_some() {
      12.0 * s
    } else {
      40.0 * s
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
      let tick_color = label_fill;
      for &pos in positions {
        let x = plot_x0 + (pos - x_min) / (x_max - x_min) * plot_w;
        ticks_svg.push_str(&format!(
          "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"{tick_color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
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
pub(crate) fn rewrite_svg_header(
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
///
/// For single-number and named sizes, the height is derived from the width
/// using the caller-provided default aspect ratio (`def_w`, `def_h`).
/// For explicit `{w, h}` lists, the user-specified dimensions are used directly.
pub(crate) fn parse_image_size(
  value: &Expr,
  def_w: u32,
  def_h: u32,
) -> Option<(u32, u32, bool)> {
  let aspect = def_h as f64 / def_w as f64;
  match value {
    Expr::Integer(n) if *n > 0 => {
      let w = *n as u32;
      let h = (w as f64 * aspect).round() as u32;
      Some((w, h, false))
    }
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      let w = n.to_u32()?;
      if w == 0 {
        return None;
      }
      let h = (w as f64 * aspect).round() as u32;
      Some((w, h, false))
    }
    Expr::Real(f) if *f > 0.0 => {
      let w = f.round() as u32;
      let h = (w as f64 * aspect).round() as u32;
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
    Expr::Identifier(name) => {
      let base_w = match name.as_str() {
        "Tiny" => 100,
        "Small" => 200,
        "Medium" => def_w,
        "Large" => 480,
        "Full" => 720,
        _ => return None,
      };
      let h = (base_w as f64 * aspect).round() as u32;
      let fw = name == "Full";
      Some((base_w, h, fw))
    }
    _ => None,
  }
}

/// Parse PlotLegends option value into a list of legend strings.
/// Returns (legends, is_automatic).
pub(crate) fn parse_plot_legends(value: &Expr) -> (Vec<String>, bool) {
  let val = evaluate_expr_to_expr(value).unwrap_or(value.clone());
  match &val {
    Expr::Identifier(s) if s == "Automatic" => (Vec::new(), true),
    Expr::Identifier(s) if s == "None" => (Vec::new(), false),
    Expr::List(items) => {
      let labels = items
        .iter()
        .map(|item| {
          crate::functions::chart::expr_to_label(item)
            .unwrap_or_else(|| crate::syntax::expr_to_string(item))
        })
        .collect();
      (labels, false)
    }
    Expr::String(s) => (vec![s.clone()], false),
    _ => (Vec::new(), false),
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
  let mut legends_automatic = false;
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
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
        "Filling" => {
          plot_opts.filling = parse_filling(replacement);
        }
        "Ticks" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "None" => plot_opts.ticks = false,
          Expr::Identifier(s) if s == "Automatic" || s == "All" => {
            plot_opts.ticks = true
          }
          _ => {}
        },
        "PlotLegends" => {
          let (labels, auto) = parse_plot_legends(replacement);
          if auto {
            legends_automatic = true;
          } else {
            plot_opts.plot_legends = labels;
          }
        }
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

  // Fill automatic legends from expression strings
  if legends_automatic && plot_opts.plot_legends.is_empty() {
    for b in &bodies {
      plot_opts
        .plot_legends
        .push(crate::syntax::expr_to_string(b));
    }
  }

  // Adaptive sampling: start with initial points, then refine where needed
  let initial_samples = plot_opts.plot_points.max(2).min(200);
  let max_total = plot_opts.plot_points.max(500);
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let points = adaptive_sample(
      func_body,
      &var_name,
      x_min,
      x_max,
      initial_samples,
      max_total,
    );
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
  let (y_auto_min, y_auto_max) =
    adjust_y_range_for_filling(plot_opts.filling, (y_auto_min, y_auto_max));

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

  // Build source data for Show merging
  let source = build_plot_source(
    &all_points,
    &plot_opts.plot_style,
    (x_display_min, x_display_max),
    (y_display_min, y_display_max),
    (plot_opts.svg_width, plot_opts.svg_height),
    false,
  );

  // Return -Graphics- as the text representation
  Ok(crate::graphics_result_with_source(svg, source))
}

/// LogLogPlot[f, {x, xmin, xmax}] — plot f with log-scaled x and y axes.
pub fn log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, true, true)
}

/// LogPlot[f, {x, xmin, xmax}] — plot f with log-scaled y axis.
pub fn log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, false, true)
}

/// LogLinearPlot[f, {x, xmin, xmax}] — plot f with log-scaled x axis.
pub fn log_linear_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, true, false)
}

/// Common implementation for LogLogPlot, LogPlot, LogLinearPlot.
/// `log_x`: whether x axis is logarithmic
/// `log_y`: whether y axis is logarithmic
fn log_scale_plot_ast(
  args: &[Expr],
  log_x: bool,
  log_y: bool,
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Plot requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  // Parse options
  let mut plot_opts = PlotOptions::default();
  let mut plot_range_y: Option<(f64, f64)> = None;
  let mut legends_automatic = false;
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
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
          if let Expr::List(items) = &val
            && items.len() == 2
          {
            let a = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
            );
            let b = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
            );
            if let (Some(lo), Some(hi)) = (a, b) {
              plot_range_y = Some((lo, hi));
            }
          }
        }
        "PlotPoints" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Expr::Integer(n) = &val
            && *n > 0
          {
            plot_opts.plot_points = *n as usize;
          }
        }
        "Filling" => {
          plot_opts.filling = parse_filling(replacement);
        }
        "PlotLegends" => {
          let (labels, auto) = parse_plot_legends(replacement);
          if auto {
            legends_automatic = true;
          } else {
            plot_opts.plot_legends = labels;
          }
        }
        _ => {}
      }
    }
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

  // For log x-axis, xmin and xmax must be positive
  if log_x && (x_min <= 0.0 || x_max <= 0.0) {
    return Err(InterpreterError::EvaluationError(
      "LogLogPlot/LogLinearPlot: x range must be positive".into(),
    ));
  }

  // Collect function bodies
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  if legends_automatic && plot_opts.plot_legends.is_empty() {
    for b in &bodies {
      plot_opts
        .plot_legends
        .push(crate::syntax::expr_to_string(b));
    }
  }

  let num_samples = plot_opts.plot_points.max(2).min(2000);
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let mut points = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
      let t = i as f64 / (num_samples - 1) as f64;
      // Sample x: log-spaced if log_x, linear otherwise
      let x = if log_x {
        let log_min = x_min.ln();
        let log_max = x_max.ln();
        (log_min + t * (log_max - log_min)).exp()
      } else {
        x_min + t * (x_max - x_min)
      };
      if let Some(y) = evaluate_at_point(func_body, &var_name, x) {
        // Skip non-positive values on log axes (can't be plotted)
        if (log_x && x <= 0.0) || (log_y && y <= 0.0) {
          continue;
        }
        // Data stays in original space; LogCoord handles scaling
        points.push((x, y));
      }
    }
    all_points.push(points);
  }

  // Compute ranges
  let finite_ys: Vec<f64> = all_points
    .iter()
    .flat_map(|pts| pts.iter())
    .filter(|(_, y)| y.is_finite())
    .map(|(_, y)| *y)
    .collect();

  if finite_ys.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let finite_xs: Vec<f64> = all_points
    .iter()
    .flat_map(|pts| pts.iter())
    .filter(|(x, _)| x.is_finite())
    .map(|(x, _)| *x)
    .collect();

  let x_min_display = finite_xs.iter().cloned().fold(f64::INFINITY, f64::min);
  let x_max_display =
    finite_xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
  let y_data_min = finite_ys.iter().cloned().fold(f64::INFINITY, f64::min);
  let y_data_max = finite_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

  let (y_auto_min, y_auto_max) = if log_y {
    // Multiplicative padding in log space (equivalent to additive 4% in log10)
    let log_range = (y_data_max / y_data_min).ln();
    let factor = (log_range * 0.04).exp();
    (y_data_min / factor, y_data_max * factor)
  } else {
    let y_range = y_data_max - y_data_min;
    let padding = if y_range.abs() < f64::EPSILON {
      1.0
    } else {
      y_range * 0.04
    };
    (y_data_min - padding, y_data_max + padding)
  };
  let y_auto =
    adjust_y_range_for_filling(plot_opts.filling, (y_auto_min, y_auto_max));

  let (y_display_min, y_display_max) = plot_range_y.unwrap_or(y_auto);

  plot_opts.log_x = log_x;
  plot_opts.log_y = log_y;

  let svg = generate_svg_with_filling(
    &all_points,
    (x_min_display, x_max_display),
    (y_display_min, y_display_max),
    &plot_opts,
  )?;

  Ok(crate::graphics_result(svg))
}
