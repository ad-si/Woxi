use plotters::prelude::{Color, *};

#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{
  ChartLabel, ChartOptions, LabelPosition, StyledLabel, parse_label_style,
};
use crate::functions::graphics::{Color as WoxiColor, parse_color};
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::PlotMarker;

/// How many lines below the first a stacked `PlotLabel` (a `Grid`/`Column`
/// title) occupies — the extra top margin every renderer has to reserve for
/// it. Zero for the ordinary single-line title and for no title at all.
fn plot_label_extra_lines(label: Option<&StyledLabel>) -> usize {
  label
    .filter(|sl| !sl.text.is_empty())
    .map_or(0, StyledLabel::extra_line_count)
}

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
/// Parse a plot iterator specification `{var, min, max}`.
/// Whether a plot iterator names the same point twice (`{x, 1, 1}`).
/// Wolfram will not sample such a range: it reports `head::plld` and leaves
/// the call unevaluated, so every plot head refuses it the same way through
/// this one check. A reversed range (`{x, 1, 0}`) is fine and samples
/// backwards.
pub(crate) fn degenerate_iterator(head: &str, spec: &Expr) -> bool {
  let Expr::List(items) = spec else {
    return false;
  };
  if items.len() != 3 {
    return false;
  }
  let Expr::Identifier(var) = &items[0] else {
    return false;
  };
  let (Ok(lo), Ok(hi)) = (
    evaluate_expr_to_expr(&items[1]),
    evaluate_expr_to_expr(&items[2]),
  ) else {
    return false;
  };
  let (Some(a), Some(b)) = (try_eval_to_f64(&lo), try_eval_to_f64(&hi)) else {
    return false;
  };
  if a != b {
    return false;
  }
  crate::emit_message(&format!(
    "{head}::plld: Endpoints for {var} in {{{var}, {}, {}}} must have \
     distinct machine-precision numerical values.",
    crate::syntax::expr_to_input_form(&lo),
    crate::syntax::expr_to_input_form(&hi),
  ));
  true
}

pub(crate) fn parse_iterator(
  spec: &Expr,
  label: &str,
) -> Result<(String, f64, f64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "{label}: iterator variable must be a symbol"
          )));
        }
      };
      let min_expr = evaluate_expr_to_expr(&items[1])?;
      let max_expr = evaluate_expr_to_expr(&items[2])?;
      let min_val = try_eval_to_f64(&min_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{label}: cannot evaluate iterator min to a number"
        ))
      })?;
      let max_val = try_eval_to_f64(&max_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{label}: cannot evaluate iterator max to a number"
        ))
      })?;
      Ok((var, min_val, max_val))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{label}: iterator must be {{var, min, max}}"
    ))),
  }
}

/// Evaluate the function body at given (x, y) values.
pub(crate) fn evaluate_at_xy(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<f64> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  if let Some(v) = try_eval_to_f64(&result) {
    return Some(v);
  }
  // The body may reference a variable (e.g. `lineA` holding `-5 - 3 x + 2 y`)
  // that only resolved to an x/y expression during evaluation — after the
  // substitution above already ran. Substitute into the evaluated form and
  // evaluate once more.
  let sub1 = substitute_var(&result, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  try_eval_to_f64(&result)
}

/// Simple SVG header for plots without plotters axes (ArrayPlot, charts).
pub(crate) fn svg_header(w: u32, h: u32, full_width: bool) -> String {
  let (bg, _, _, _, _) = plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  if full_width {
    format!(
      "<svg width=\"100%\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  } else {
    format!(
      "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  }
}

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
  monitor: Option<&Expr>,
) -> Vec<(f64, f64)> {
  // Evaluate one sample, firing the `EvaluationMonitor :> expr` hook (with
  // the plot variable bound to x) first, e.g. so `Sow[{x, f[x]}]` records
  // the sampled points.
  let sample = |x: f64| -> f64 {
    if let Some(m) = monitor {
      let sub = crate::syntax::substitute_variable(m, var_name, &Expr::Real(x));
      let _ = crate::evaluator::evaluate_expr_to_expr(&sub);
    }
    evaluate_at_point(func_body, var_name, x).unwrap_or(f64::NAN)
  };

  // Initial uniform sampling
  let step = (x_max - x_min) / (initial_n - 1) as f64;
  let mut points: Vec<(f64, f64)> = (0..initial_n)
    .map(|i| {
      let x = x_min + i as f64 * step;
      (x, sample(x))
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
      } else if i >= 1 {
        // This is the rightmost interval in the current point list (no
        // `i + 2` neighbor yet), which is structurally true of the last
        // interval on every refinement pass — without this branch it would
        // never be curvature-checked at all and a fast-oscillating tail
        // (e.g. a high-frequency Sin) would stay a single straight segment
        // no matter how many `MaxRecursion` passes ran. Mirror the
        // look-ahead check using the point *before* x0 instead.
        let (xm1, ym1) = points[i - 1];
        if ym1.is_finite() {
          let y_interp = ym1 + (y1 - ym1) * (x0 - xm1) / (x1 - xm1);
          let y_range = (y1 - ym1).abs().max(1e-10);
          let deviation = (y0 - y_interp).abs() / y_range;
          deviation > 0.05
        } else {
          true
        }
      } else {
        // Fewer than 3 points in the whole domain: no neighbor on either
        // side to assess curvature against, so refine to be safe.
        true
      };

      if needs_refine {
        let xm = f64::midpoint(x0, x1);
        new_points.push((xm, sample(xm)));
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

/// True when a `PlotRange` value asks for the whole y extent: `All` itself, or
/// `All` in the y slot of `{xrange, All}`.
fn plot_range_requests_all_y(value: &Expr) -> bool {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let is_all = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "All");
  match &val {
    e if is_all(e) => true,
    Expr::List(items) if items.len() == 2 => is_all(&items[1]),
    _ => false,
  }
}

/// Compute the y-range from uniformly-spaced x samples — uniform rather than
/// the adaptively refined plot points so a singularity can't bias the
/// distribution.
///
/// `keep_outliers` is `PlotRange -> All`: report the whole extent. Otherwise
/// extreme values are excluded by an IQR fence, which is what keeps a pole
/// from flattening the rest of the curve into the axis.
fn sampled_y_range(
  bodies: &[&Expr],
  var_name: &str,
  x_min: f64,
  x_max: f64,
  keep_outliers: bool,
) -> (f64, f64) {
  // Evaluate at uniformly-spaced x values to get an unbiased y distribution
  let n_uniform = 200;
  let step = (x_max - x_min) / (n_uniform - 1) as f64;
  let mut ys: Vec<f64> = Vec::new();
  for body in bodies {
    for i in 0..n_uniform {
      let x = x_min + i as f64 * step;
      if let Some(y) = evaluate_at_point(body, var_name, x)
        && y.is_finite()
      {
        ys.push(y);
      }
    }
  }

  if ys.is_empty() {
    return (-1.0, 1.0);
  }
  ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  let n = ys.len();
  if n == 1 {
    return (ys[0], ys[0]);
  }

  if keep_outliers {
    return (ys[0], ys[n - 1]);
  }

  let q1 = ys[n / 4];
  let q3 = ys[3 * n / 4];
  let iqr = q3 - q1;

  // If IQR is negligible, no outliers — use full min/max
  if iqr < 1e-10 {
    return (ys[0], ys[n - 1]);
  }

  let fence_lo = q1 - 3.0 * iqr;
  let fence_hi = q3 + 3.0 * iqr;

  let y_min = ys.iter().copied().find(|&y| y >= fence_lo).unwrap_or(ys[0]);
  let y_max = ys
    .iter()
    .rev()
    .copied()
    .find(|&y| y <= fence_hi)
    .unwrap_or(ys[n - 1]);

  (y_min, y_max)
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

/// Clip line segments to a y-range, interpolating at boundaries.
/// Points outside the range are removed and the line is split,
/// with interpolated points added at the boundary crossings.
fn clip_segments_to_y_range(
  segments: Vec<Vec<(f64, f64)>>,
  y_min: f64,
  y_max: f64,
) -> Vec<Vec<(f64, f64)>> {
  let mut result: Vec<Vec<(f64, f64)>> = Vec::new();

  for segment in segments {
    let mut current: Vec<(f64, f64)> = Vec::new();

    for i in 0..segment.len() {
      let (x, y) = segment[i];
      let inside = y >= y_min && y <= y_max;

      if i > 0 {
        let (px, py) = segment[i - 1];
        let prev_inside = py >= y_min && py <= y_max;

        if prev_inside != inside {
          // Line crosses the boundary — interpolate
          let boundary = if !inside {
            if y > y_max { y_max } else { y_min }
          } else if py > y_max {
            y_max
          } else {
            y_min
          };
          let t = (boundary - py) / (y - py);
          let bx = px + t * (x - px);
          current.push((bx, boundary));

          if !inside {
            // Leaving the range — flush segment
            if current.len() > 1 {
              result.push(std::mem::take(&mut current));
            } else {
              current.clear();
            }
          }
        } else if !prev_inside && !inside {
          // Both outside — check if the line passes through the range
          // (e.g. one above y_max and the other below y_min)
          if (py > y_max && y < y_min) || (py < y_min && y > y_max) {
            // Crosses both boundaries
            let t_min = (y_min - py) / (y - py);
            let t_max = (y_max - py) / (y - py);
            let (t1, t2) = if t_min < t_max {
              (t_min, t_max)
            } else {
              (t_max, t_min)
            };
            let x1 = px + t1 * (x - px);
            let y1 = py + t1 * (y - py);
            let x2 = px + t2 * (x - px);
            let y2 = py + t2 * (y - py);
            result.push(vec![(x1, y1), (x2, y2)]);
          }
        }
      }

      if inside {
        current.push((x, y));
      }
    }

    if current.len() > 1 {
      result.push(current);
    }
  }

  result
}

/// Compute a "nice" major tick step given the axis range and desired label
/// count. [`AXIS_TICK_TARGET`] is the count the Wolfram Language divides a
/// plot axis into.
pub(crate) fn nice_step(range: f64, target_labels: usize) -> f64 {
  let raw = range / target_labels as f64;
  let mag = 10_f64.powf(raw.abs().log10().floor());
  let norm = raw / mag;
  // The Wolfram Language's own set of step multipliers — 2.5 included, which
  // is what makes a 0..15 axis step by 2.5 rather than 5.
  let nice = if norm <= 1.0 {
    1.0
  } else if norm <= 2.0 {
    2.0
  } else if norm <= 2.5 {
    2.5
  } else if norm <= 5.0 {
    5.0
  } else {
    10.0
  };
  nice * mag
}

/// Widest major y-axis tick label, in characters. Used to position the rotated
/// y-axis/frame label just left of the tick column instead of at the far edge
/// of the (fixed-width) gutter — so narrow ticks (e.g. single digits) don't
/// leave a large gap.
fn max_y_tick_label_chars(y_min: f64, y_max: f64) -> usize {
  let step = nice_step(y_max - y_min, AXIS_TICK_TARGET);
  if step <= 0.0 || !step.is_finite() {
    return 3;
  }
  let mut max_len = 1usize;
  let mut v = (y_min / step).ceil() * step;
  let mut guard = 0;
  while v <= y_max + step * 1e-9 && guard < 1000 {
    if is_major_tick(v, step) {
      max_len = max_len.max(format_tick_with_step(v, step).chars().count());
    }
    v += step;
    guard += 1;
  }
  max_len
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
fn nice_date_step_spec(range_seconds: f64) -> DateStep {
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
fn nice_date_step(range_seconds: f64) -> f64 {
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
  orientation: &LogAxisOrientation,
) {
  // A log axis needs positive, finite bounds; otherwise no labels can be
  // placed (log10 would yield NaN positions).
  if data_min <= 0.0
    || !data_min.is_finite()
    || !data_max.is_finite()
    || data_max <= data_min
  {
    return;
  }
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
/// The SVG for an `AxesLabel`, placed the way the Wolfram Language does: the
/// x label just past the right end of the x axis and level with it, the y
/// label above the top of the y axis. (A `FrameLabel`, by contrast, is
/// centred outside the plot area — see the callers.)
///
/// `area` is the plotting rectangle `(x0, y0, w, h)` in render units and
/// `range` the data range `(x_min, x_max, y_min, y_max)` it maps, so the
/// labels sit on the axis lines themselves when those fall inside the plot.
pub(crate) fn axes_label_svg(
  axes_label: Option<&(String, String)>,
  (x0, y0, w, h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  (show_x_axis, show_y_axis): (bool, bool),
  font_size: f64,
  label_fill: &str,
) -> String {
  let Some((x_label, y_label)) = axes_label else {
    return String::new();
  };
  let mut svg = String::new();
  // The axis lines: where y = 0 and x = 0 fall, clamped into the plot area
  // (an all-positive range draws its axes along the bottom and left edges).
  let axis_y = if y_max > y_min {
    y0 + h - ((0.0 - y_min) / (y_max - y_min)).clamp(0.0, 1.0) * h
  } else {
    y0 + h
  };
  let axis_x = if x_max > x_min {
    x0 + ((0.0 - x_min) / (x_max - x_min)).clamp(0.0, 1.0) * w
  } else {
    x0
  };
  if show_x_axis && !x_label.is_empty() {
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{axis_y:.1}\" text-anchor=\"start\" \
       dominant-baseline=\"central\" font-family=\"sans-serif\" \
       font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
      x0 + w + font_size * 0.6,
      crate::functions::graphics::box_string_to_svg(x_label)
    ));
  }
  if show_y_axis && !y_label.is_empty() {
    // Centred over the top of the y axis, but never so far left that the
    // label runs off the image — a wide one (`{a, b, c, d}`) over an axis
    // that sits near the left edge slides right until it fits, the way
    // Wolfram keeps an axis label inside the graphic.
    let half_width = crate::functions::graphics::box_string_visible_len(y_label)
      as f64
      * font_size
      * 0.55
      / 2.0;
    let lx = axis_x.max(half_width + font_size * 0.2);
    svg.push_str(&format!(
      "<text x=\"{lx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
       fill=\"{label_fill}\">{}</text>\n",
      y0 - font_size * 0.6,
      crate::functions::graphics::box_string_to_svg(y_label)
    ));
  }
  svg
}

/// Label a tick that belongs to a sequence of `step`-spaced ticks.
///
/// The Wolfram Language gives every label in a tick set the same number of
/// decimals — the number the step itself needs — so an axis stepping by 0.5
/// reads `-1.0, -0.5, 0.0, 0.5, 1.0`, not `-1, -0.5, 0, 0.5, 1`, while one
/// stepping by 2 reads `0, 2, 4`.
pub(crate) fn format_tick_with_step(v: f64, step: f64) -> String {
  let decimals = tick_step_decimals(step);
  if decimals == 0 {
    return format_tick(v);
  }
  let magnitude = v.abs();
  // A tick a hair off zero is the axis zero, not a tiny scientific value:
  // scale the smallness test against the step, not against an absolute
  // threshold, so `0.5`-spaced ticks label their origin `0.0`.
  let effectively_zero = magnitude < step.abs() * 1e-6;
  if magnitude >= 1e6 || (!effectively_zero && magnitude < 1e-5) {
    return format_tick(v);
  }
  let v = if effectively_zero { 0.0 } else { v };
  format!("{v:.decimals$}")
}

/// How many decimals it takes to write `step` — the decimals every label of
/// a `step`-spaced tick set carries. Capped at 6, the precision
/// [`format_tick`] itself works to.
fn tick_step_decimals(step: f64) -> usize {
  let step = step.abs();
  if !step.is_finite() || step <= 0.0 {
    return 0;
  }
  (0..=6)
    .find(|d| {
      let scaled = step * 10f64.powi(*d as i32);
      (scaled - scaled.round()).abs() < 1e-6 * scaled.max(1.0)
    })
    .unwrap_or(6)
}

pub(crate) fn format_tick(v: f64) -> String {
  // Outside `[10^-5, 10^6)` the Wolfram Language labels a tick in
  // scientific notation — a frequency axis running to 6*10^15 is labelled
  // `2x10^15` with real superscript digits, not `2000000000000000`. The
  // exponent uses the Unicode superscripts so it survives as plain text.
  // A value that is an integer within float noise prints as that integer —
  // a tick that lands on `-1.11*10^-16` is the axis zero, not a tiny
  // scientific value.
  if (v - v.round()).abs() < 1e-9 {
    let rounded = v.round() as i64;
    if rounded.abs() < 1_000_000 {
      return format!("{rounded}");
    }
  }
  let magnitude = v.abs();
  if magnitude >= 1e6 || (magnitude > 0.0 && magnitude < 1e-5) {
    let exp = magnitude.log10().floor() as i32;
    let mantissa = v / 10f64.powi(exp);
    let mantissa = if (mantissa - mantissa.round()).abs() < 1e-9 {
      format!("{}", mantissa.round() as i64)
    } else {
      let s = format!("{mantissa:.3}");
      s.trim_end_matches('0').trim_end_matches('.').to_string()
    };
    return format!(
      "{mantissa}\u{d7}10{}",
      crate::functions::graphics::to_unicode_script(&exp.to_string(), true)
    );
  }
  // Enough decimals to tell neighbouring ticks apart — a 0.0002 step used
  // to render every label as `0.0`.
  let s = format!("{v:.6}");
  s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Inject SVG `<line>` elements extending labeled (major) ticks a few pixels
/// further out than the minor ticks drawn by plotters. This is called after
/// plotters renders the chart, using post-render coordinates in the final SVG
/// space.
///
/// `x_axis` and `y_axis` are `Some((min, max, major_step))` if that axis
/// should get extensions, or `None` to skip (e.g. for log/date axes where
/// plotters places ticks itself and linear spacing is wrong).
///
/// The extension is drawn from `minor_len` to `major_len` along the tick
/// direction, so it connects seamlessly with the plotters-drawn tick.
#[allow(clippy::too_many_arguments)]
fn inject_major_tick_extensions(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
  minor_len: f64,
  major_len: f64,
  stroke_w: f64,
  color: &str,
) {
  let extension = major_len - minor_len;
  if extension <= 0.0 {
    return;
  }
  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut svg = String::new();

  // X axis: major ticks extend downward (below the plot area).
  if let Some((x_min, x_max, x_major)) = x_axis
    && x_major > 0.0
    && x_max > x_min
  {
    let axis_y = plot_y0 + plot_h;
    let y1 = axis_y + minor_len;
    let y2 = y1 + extension;
    let eps = x_major * 1e-6;
    let mut v = (x_min / x_major).ceil() * x_major;
    let max_steps = ((x_max - x_min) / x_major).abs() as usize + 4;
    for _ in 0..max_steps {
      if v > x_max + eps {
        break;
      }
      let x = plot_x0 + (v - x_min) / (x_max - x_min) * plot_w;
      svg.push_str(&format!(
        "<line x1=\"{x:.1}\" y1=\"{y1:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += x_major;
    }
  }

  // Y axis: major ticks extend leftward (to the left of the plot area).
  if let Some((y_min, y_max, y_major)) = y_axis
    && y_major > 0.0
    && y_max > y_min
  {
    let axis_x = plot_x0;
    let x1 = axis_x - minor_len;
    let x2 = x1 - extension;
    let eps = y_major * 1e-6;
    let mut v = (y_min / y_major).ceil() * y_major;
    let max_steps = ((y_max - y_min) / y_major).abs() as usize + 4;
    for _ in 0..max_steps {
      if v > y_max + eps {
        break;
      }
      let y = plot_y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h;
      svg.push_str(&format!(
        "<line x1=\"{x1:.1}\" y1=\"{y:.1}\" x2=\"{x2:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += y_major;
    }
  }

  if !svg.is_empty() {
    buf.insert_str(insert_pos, &svg);
  }

  inject_missing_major_tick_labels(
    buf, plot_x0, plot_y0, plot_w, plot_h, x_axis, y_axis,
  );
}

/// Label the major ticks plotters left unlabelled.
///
/// Plotters picks its own key points from a range and a count hint, and on a
/// range with a negative minimum it drops the last one — `PlotRange -> {-1,
/// 1}` gets ticks up to `0.9` and no `1.0` label, where the Wolfram Language
/// labels the whole set. The tick *mark* is already drawn (the major
/// extension pass above walks every major), so only the text is missing; it
/// is emitted with the attributes of the labels plotters did draw, so the
/// injected one is indistinguishable from them.
fn inject_missing_major_tick_labels(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
) {
  let mut svg = String::new();
  for (axis, anchor, horizontal) in
    [(x_axis, "middle", true), (y_axis, "end", false)]
  {
    let Some((min, max, major)) = axis else {
      continue;
    };
    if major <= 0.0 || max <= min {
      continue;
    }
    let Some(template) = tick_label_template(buf, anchor) else {
      continue;
    };
    let eps = major * 1e-6;
    let mut v = (min / major).ceil() * major;
    let steps = ((max - min) / major).abs() as usize + 4;
    for _ in 0..steps {
      if v > max + eps {
        break;
      }
      let frac = (v - min) / (max - min);
      let (x, y) = if horizontal {
        (plot_x0 + frac * plot_w, template.pos)
      } else {
        (template.pos, plot_y0 + plot_h - frac * plot_h)
      };
      let along = if horizontal { x } else { y };
      if !template.occupied.iter().any(|p| (p - along).abs() < 2.0) {
        svg.push_str(&template.render(x, y, &format_tick_with_step(v, major)));
      }
      v += major;
    }
  }
  if !svg.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    buf.insert_str(pos, &svg);
  }
}

/// One of the tick labels plotters drew, as a stencil for the ones it did
/// not: the attributes to copy, the fixed coordinate they all share (the
/// label column or row) and the varying coordinates already taken.
struct TickLabelTemplate {
  attrs: String,
  pos: f64,
  occupied: Vec<f64>,
}

impl TickLabelTemplate {
  fn render(&self, x: f64, y: f64, text: &str) -> String {
    format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" {}>{}</text>\n",
      self.attrs,
      crate::functions::graphics::svg_escape(text)
    )
  }
}

/// Collect the tick labels plotters drew for one axis, identified by their
/// `text-anchor` (`middle` for the x axis, `end` for the y axis).
fn tick_label_template(buf: &str, anchor: &str) -> Option<TickLabelTemplate> {
  let marker = format!("text-anchor=\"{anchor}\"");
  let mut attrs: Option<String> = None;
  let mut fixed: Option<f64> = None;
  let mut occupied = Vec::new();
  for tag in buf.split("<text ").skip(1) {
    let Some(end) = tag.find('>') else { continue };
    let (open, rest) = tag.split_at(end);
    if !open.contains(&marker) {
      continue;
    }
    let Some(x) = svg_attr_f64(open, "x") else {
      continue;
    };
    let Some(y) = svg_attr_f64(open, "y") else {
      continue;
    };
    // Only labels with text pin down where the row/column sits; the empty
    // ones plotters emits for minor ticks would match anywhere.
    let text = rest[1..].split('<').next().unwrap_or("").trim();
    if text.is_empty() {
      continue;
    }
    if attrs.is_none() {
      let after_y = open
        .find("\" y=\"")
        .and_then(|i| open[i + 5..].find('"').map(|j| i + 6 + j))
        .unwrap_or(0);
      attrs = Some(open[after_y..].trim().to_string());
      fixed = Some(if anchor == "middle" { y } else { x });
    }
    occupied.push(if anchor == "middle" { x } else { y });
  }
  Some(TickLabelTemplate {
    attrs: attrs?,
    pos: fixed?,
    occupied,
  })
}

/// Read a numeric attribute out of an SVG opening tag.
fn svg_attr_f64(tag: &str, name: &str) -> Option<f64> {
  let key = format!("{name}=\"");
  let start = tag.find(&key)? + key.len();
  let end = start + tag[start..].find('"')?;
  tag[start..end].parse().ok()
}

/// Draw top and right axis lines closing the plot rectangle, plus minor/major
/// tick marks mirroring the bottom and left axes. Ticks point outward (up from
/// the top axis, right from the right axis), matching the direction of the
/// bottom/left ticks drawn by plotters + `inject_major_tick_extensions`.
///
/// `x_axis` / `y_axis` each supply `(min, max, major_step)`. Minor ticks are
/// drawn at `major_step / 5` intervals. When an axis is `None` the
/// corresponding frame line is still drawn, but without tick marks.
fn inject_top_right_frame(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
  minor_len: f64,
  major_len: f64,
  stroke_w: f64,
  color: &str,
) {
  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut svg = String::new();

  let top_y = plot_y0;
  let right_x = plot_x0 + plot_w;
  let bottom_y = plot_y0 + plot_h;

  // Top frame line.
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.1}\" y1=\"{top_y:.1}\" x2=\"{right_x:.1}\" y2=\"{top_y:.1}\" \
       stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
  ));
  // Right frame line.
  svg.push_str(&format!(
    "<line x1=\"{right_x:.1}\" y1=\"{plot_y0:.1}\" x2=\"{right_x:.1}\" y2=\"{bottom_y:.1}\" \
       stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
  ));

  // Top axis ticks (pointing upward / outward).
  if let Some((x_min, x_max, x_major)) = x_axis
    && x_major > 0.0
    && x_max > x_min
  {
    let x_minor = x_major / 5.0;
    let eps = x_minor.abs() * 1e-6;
    let mut v = (x_min / x_minor).ceil() * x_minor;
    let max_steps = ((x_max - x_min) / x_minor).abs() as usize + 8;
    for _ in 0..max_steps {
      if v > x_max + eps {
        break;
      }
      let x = plot_x0 + (v - x_min) / (x_max - x_min) * plot_w;
      let len = if is_major_tick(v, x_major) {
        major_len
      } else {
        minor_len
      };
      let y2 = top_y - len;
      svg.push_str(&format!(
        "<line x1=\"{x:.1}\" y1=\"{top_y:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" \
           stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += x_minor;
    }
  }

  // Right axis ticks (pointing rightward / outward).
  if let Some((y_min, y_max, y_major)) = y_axis
    && y_major > 0.0
    && y_max > y_min
  {
    let y_minor = y_major / 5.0;
    let eps = y_minor.abs() * 1e-6;
    let mut v = (y_min / y_minor).ceil() * y_minor;
    let max_steps = ((y_max - y_min) / y_minor).abs() as usize + 8;
    for _ in 0..max_steps {
      if v > y_max + eps {
        break;
      }
      let y = plot_y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h;
      let len = if is_major_tick(v, y_major) {
        major_len
      } else {
        minor_len
      };
      let x2 = right_x + len;
      svg.push_str(&format!(
        "<line x1=\"{right_x:.1}\" y1=\"{y:.1}\" x2=\"{x2:.1}\" y2=\"{y:.1}\" \
           stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += y_minor;
    }
  }

  if !svg.is_empty() {
    buf.insert_str(insert_pos, &svg);
  }
}

/// Default minor (unlabeled) tick length in render-space units.
pub(crate) const MINOR_TICK_LEN: i32 = 4;
/// Default major (labeled) tick length in render-space units — slightly longer
/// than minor ticks so labeled ticks stand out visually.
pub(crate) const MAJOR_TICK_LEN: i32 = 7;

/// Filling mode for line plots.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Filling {
  None,
  Axis,
  Bottom,
  Top,
  Value(f64),
}

/// Mesh mode for line plots. `All` marks every point of the drawn curve,
/// `Full` marks the original data points (the two differ when the curve is
/// interpolated, e.g. `InterpolationOrder -> 0` steps).
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mesh {
  None,
  All,
  Full,
}

impl Filling {
  /// Compute the y-value to fill to, given the current plot y-range.
  /// Returns `None` for `Filling::None`.
  pub(crate) fn reference_y(&self, y_min: f64, y_max: f64) -> Option<f64> {
    match self {
      Self::None => None,
      Self::Axis => Some(0.0),
      Self::Bottom => Some(y_min),
      Self::Top => Some(y_max),
      Self::Value(v) => Some(*v),
    }
  }

  /// Convert to the serializable `SeriesFilling` variant stored on
  /// `PlotSeriesData` so `Show` can re-render filled regions.
  pub(crate) fn to_series_filling(self) -> crate::syntax::SeriesFilling {
    match self {
      Self::None => crate::syntax::SeriesFilling::None,
      Self::Axis => crate::syntax::SeriesFilling::Axis,
      Self::Bottom => crate::syntax::SeriesFilling::Bottom,
      Self::Top => crate::syntax::SeriesFilling::Top,
      Self::Value(v) => crate::syntax::SeriesFilling::Value(v),
    }
  }
}

/// Parsed `FillingStyle -> …` value: an optional fill color override
/// (`None` = use the series color) and an optional opacity (`None` = the
/// default 0.2 translucency).
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct FillStyle {
  pub color: Option<(u8, u8, u8)>,
  pub opacity: Option<f64>,
}

/// Parse a `FillingStyle` option value. Supported forms: a color,
/// `Opacity[a]`, `Opacity[a, color]`, `Directive[…]`, and a directive list
/// `{Opacity[a], color}`. Returns `None` for `Automatic`/`None`/unrecognized
/// values (keeping the default fill appearance).
pub(crate) fn parse_filling_style(replacement: &Expr) -> Option<FillStyle> {
  fn walk(e: &Expr, fs: &mut FillStyle) {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Opacity" && !args.is_empty() =>
      {
        if let Some(a) = try_eval_to_f64(&args[0]) {
          fs.opacity = Some(a.clamp(0.0, 1.0));
        }
        if args.len() >= 2
          && let Some(c) = parse_color(&args[1])
        {
          fs.color = Some(color_to_rgb8(c));
        }
      }
      Expr::FunctionCall { name, args } if name == "Directive" => {
        for a in args {
          walk(a, fs);
        }
      }
      Expr::List(items) => {
        for item in items {
          walk(item, fs);
        }
      }
      Expr::Identifier(s) if s == "Automatic" || s == "None" => {}
      other => {
        if let Some(c) = parse_color(other) {
          fs.color = Some(color_to_rgb8(c));
          // An alpha channel on the color (`RGBColor[r, g, b, a]`) doubles
          // as the fill opacity when no explicit Opacity[…] is given.
          if c.a < 1.0 && fs.opacity.is_none() {
            fs.opacity = Some(c.a);
          }
        }
      }
    }
  }
  let val = evaluate_expr_to_expr(replacement).unwrap_or(replacement.clone());
  let mut fs = FillStyle {
    color: None,
    opacity: None,
  };
  walk(&val, &mut fs);
  if fs.color.is_none() && fs.opacity.is_none() {
    None
  } else {
    Some(fs)
  }
}

/// Convert a parsed `Color` (float channels) to 8-bit RGB.
fn color_to_rgb8(c: WoxiColor) -> (u8, u8, u8) {
  (
    (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
    (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
    (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
  )
}

/// Resolve the fill paint for a series: the `FillingStyle` color override
/// (else the series color) at the `FillingStyle` opacity (else 0.2).
fn fill_paint(
  filling_style: Option<FillStyle>,
  series_rgb: (u8, u8, u8),
) -> plotters::style::RGBAColor {
  let fs = filling_style.unwrap_or(FillStyle {
    color: None,
    opacity: None,
  });
  let (r, g, b) = fs.color.unwrap_or(series_rgb);
  RGBColor(r, g, b).mix(fs.opacity.unwrap_or(0.2))
}

impl crate::syntax::SeriesFilling {
  /// Reference y-value for the fill, given the current y-range.
  pub fn reference_y(self, y_min: f64, y_max: f64) -> Option<f64> {
    match self {
      Self::None => None,
      Self::Axis => Some(0.0),
      Self::Bottom => Some(y_min),
      Self::Top => Some(y_max),
      Self::Value(v) => Some(v),
    }
  }
}

/// Fill target for one series, from the rule-list form
/// `Filling -> {i -> spec, …}`.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FillTarget {
  /// Fill to a constant reference level (`Axis`, `Bottom`, `Top`, a value).
  Level(Filling),
  /// Fill to another series (0-based index), i.e. `i -> {j}`.
  Series(usize),
}

/// Parse the rule-list form `Filling -> {i -> spec, …}`, where `spec` is
/// `{j}` (fill between series i and j), `{{j}, style}` (the style is not
/// rendered yet), or a constant level (`Axis`, `Bottom`, `Top`, a number).
/// Returns `None` when `replacement` is not a list of rules keyed by
/// 1-based series indices, so the caller can fall back to `parse_filling`.
///
/// The rules may be grouped in sub-lists — `{{1 -> {2}}, {3 -> 0}}` fills the
/// same way `{1 -> {2}, 3 -> 0}` does — so the outer structure is flattened
/// first. Only the list scaffolding around the rules is flattened; a rule's
/// own right-hand side (`{j}`, the brace form that names a target series)
/// stays as written.
fn parse_filling_rules(replacement: &Expr) -> Option<Vec<(usize, FillTarget)>> {
  fn flatten_rules<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::List(items) => items.iter().for_each(|i| flatten_rules(i, out)),
      other => out.push(other),
    }
  }
  if !matches!(replacement, Expr::List(_)) {
    return None;
  }
  let mut items: Vec<&Expr> = Vec::new();
  flatten_rules(replacement, &mut items);
  let mut rules = Vec::new();
  for item in items {
    let (pattern, rhs) = match item {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    let series_idx = match pattern {
      Expr::Integer(n) if *n >= 1 => *n as usize - 1,
      _ => return None,
    };
    let target = match rhs {
      // Braces mark a series target: `{j}` or `{{j}, style}`.
      Expr::List(spec) => {
        let target_idx = match spec.first()? {
          Expr::Integer(j) if *j >= 1 => *j as usize - 1,
          Expr::List(inner) => match inner.first()? {
            Expr::Integer(j) if *j >= 1 => *j as usize - 1,
            _ => return None,
          },
          _ => return None,
        };
        FillTarget::Series(target_idx)
      }
      other => FillTarget::Level(parse_filling(other)),
    };
    rules.push((series_idx, target));
  }
  Some(rules)
}

/// Apply a `Filling` option value to `opts`: either the per-series rule
/// list `{i -> spec, …}` (stored in `filling_rules`) or a global mode
/// (stored in `filling`).
pub(crate) fn apply_filling_option(replacement: &Expr, opts: &mut PlotOptions) {
  // The value may be computed rather than named — a Demonstration switches
  // its shading with `Filling -> If[b === Axis, Axis, None]` — so read it
  // evaluated. Every spelling it can produce is a symbol, a number or a
  // list of rules, all of which evaluate to themselves.
  let value =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  if let Some(rules) = parse_filling_rules(&value) {
    opts.filling_rules = rules;
  } else {
    opts.filling = parse_filling(&value);
  }
}

/// The `FillingStyle` that applies to series `idx`: its own when `Show`
/// merged several plots, else the one style the plot was given.
fn series_filling_style(opts: &PlotOptions, idx: usize) -> Option<FillStyle> {
  opts
    .filling_styles
    .get(idx)
    .copied()
    .flatten()
    .or(opts.filling_style)
}

/// Effective fill target for series `idx`: its entry in the rule list when
/// `Filling -> {i -> spec, …}` was given (series without a rule stay
/// unfilled), otherwise the global filling mode.
fn series_fill_target(opts: &PlotOptions, idx: usize) -> FillTarget {
  if opts.filling_rules.is_empty() {
    FillTarget::Level(opts.filling)
  } else {
    opts
      .filling_rules
      .iter()
      .find(|(i, _)| *i == idx)
      .map_or(FillTarget::Level(Filling::None), |&(_, t)| t)
  }
}

/// Linearly interpolate a polyline (in data order) at `x`. Returns `None`
/// when `x` lies outside every segment's x-span, so filling between series
/// with different x-domains stops at the overlap.
fn interp_polyline_y(points: &[(f64, f64)], x: f64) -> Option<f64> {
  for w in points.windows(2) {
    let (x0, y0) = w[0];
    let (x1, y1) = w[1];
    if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
      continue;
    }
    let (lo, hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    if x >= lo && x <= hi {
      if (x1 - x0).abs() < f64::EPSILON {
        return Some(y0);
      }
      return Some(y0 + (y1 - y0) * (x - x0) / (x1 - x0));
    }
  }
  None
}

/// Polygon between two polylines over the overlap of their x-domains: the
/// source curve forms the top boundary and the reversed target curve the
/// bottom, both clipped (with interpolated endpoints) to the overlap.
fn fill_between_polygon(
  source: &[(f64, f64)],
  target: &[(f64, f64)],
) -> Option<Vec<(f64, f64)>> {
  let finite = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
    pts
      .iter()
      .copied()
      .filter(|(x, y)| x.is_finite() && y.is_finite())
      .collect()
  };
  let src = finite(source);
  let tgt = finite(target);
  if src.len() < 2 || tgt.len() < 2 {
    return None;
  }
  let x_span = |pts: &[(f64, f64)]| {
    pts
      .iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(x, _)| {
        (lo.min(x), hi.max(x))
      })
  };
  let (s_lo, s_hi) = x_span(&src);
  let (t_lo, t_hi) = x_span(&tgt);
  let lo = s_lo.max(t_lo);
  let hi = s_hi.min(t_hi);
  if lo >= hi {
    return None;
  }
  let clip = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(y) = interp_polyline_y(pts, lo) {
      out.push((lo, y));
    }
    out.extend(pts.iter().copied().filter(|&(x, _)| x > lo && x < hi));
    if let Some(y) = interp_polyline_y(pts, hi) {
      out.push((hi, y));
    }
    out
  };
  let mut polygon = clip(&src);
  let mut bottom = clip(&tgt);
  bottom.reverse();
  polygon.extend(bottom);
  if polygon.len() >= 3 {
    Some(polygon)
  } else {
    None
  }
}

/// Parse a `Filling` option value from an expression.
fn parse_filling(replacement: &Expr) -> Filling {
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
fn adjust_y_range_for_filling(
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

/// Like `adjust_y_range_for_filling`, but also accounts for the per-series
/// `Filling -> {i -> spec, …}` rules: every constant-level target must lie
/// inside the plotted y-range (between-series targets need no adjustment —
/// both bounds are data curves already contained in the range).
pub(crate) fn adjust_y_range_for_filling_opts(
  opts: &PlotOptions,
  mut y_range: (f64, f64),
) -> (f64, f64) {
  y_range = adjust_y_range_for_filling(opts.filling, y_range);
  for (_, target) in &opts.filling_rules {
    if let FillTarget::Level(level) = target {
      y_range = adjust_y_range_for_filling(*level, y_range);
    }
  }
  y_range
}

/// Per-series style: color, line thickness, and dashing pattern.
#[derive(Clone, Debug, Default)]
pub(crate) struct SeriesStyle {
  pub color: Option<WoxiColor>,
  /// Line thickness in display pixels (e.g. 1.5 = default, 2.0 = Thick).
  /// None means use the default (1.5px).
  pub thickness: Option<f64>,
  /// Dash pattern in display pixels. None = solid line.
  pub dashing: Option<Vec<f64>>,
  /// DropShadowing[...] directive: draw the curve with a drop shadow.
  pub shadow: Option<crate::functions::graphics::DropShadow>,
}

/// A single explicit grid line: a position on its axis plus an optional
/// per-line style override (color, thickness, dashing).
#[derive(Clone, Debug)]
pub(crate) struct GridLine {
  pub pos: f64,
  pub style: SeriesStyle,
}

/// One side (x or y) of a `GridLines` specification.
pub(crate) enum GridSide {
  /// No grid lines on this axis.
  None,
  /// Evenly spaced automatic grid lines (default gray, solid).
  Automatic,
  /// Explicit grid lines at the given positions/styles.
  Explicit(Vec<GridLine>),
}

/// Position for plot legends
#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum LegendPosition {
  #[default]
  Right,
  Top,
  Bottom,
}

/// How `Around` uncertainty intervals are drawn (IntervalMarkers option):
/// capped error bars (the default), a filled band spanning the intervals
/// across the series, or nothing.
#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum IntervalMarkers {
  /// Error bars with end caps ("Fences", the default).
  #[default]
  Fences,
  /// One filled band per series covering the y-uncertainty intervals.
  Bands,
  /// `IntervalMarkers -> None`: uncertainties are not drawn.
  None,
}

/// Options for line-based plots (Plot, ListLinePlot, etc.).
// Each flag is an independent Wolfram option (Frame, Joined, Filling, …),
// not a state machine, so they do not collapse into an enum.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub(crate) struct PlotOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub filling: Filling,
  /// Per-series filling from `Filling -> {i -> spec, …}` (0-based series
  /// index → target). When non-empty this replaces the global `filling`:
  /// series without a rule are not filled.
  pub filling_rules: Vec<(usize, FillTarget)>,
  /// `FillingStyle -> …`: color/opacity for the filled regions (applies to
  /// every filled series). `None` keeps the default appearance (series
  /// color at 0.2 opacity).
  pub filling_style: Option<FillStyle>,
  /// Per-series `FillingStyle`, as `Show` collects it from the plots it
  /// merges: each curve keeps the fill its own plot asked for. Empty means
  /// the single `filling_style` applies to every series.
  pub filling_styles: Vec<Option<FillStyle>>,
  pub mesh: Mesh,
  pub plot_label: Option<StyledLabel>,
  pub axes_label: Option<(String, String)>,
  pub plot_style: Vec<SeriesStyle>,
  /// Per-axis visibility: (x_axis, y_axis). Both true = default.
  pub axes: (bool, bool),
  /// Ticks option: true = show tick marks and labels (default), false = hide
  pub ticks: bool,
  /// `Ticks -> {xspec, yspec}` with explicit positions: each entry is a
  /// position and the text drawn at it. `None` = the automatic ticks.
  pub ticks_x: Option<Vec<(f64, String)>>,
  pub ticks_y: Option<Vec<(f64, String)>>,
  /// Number of sample points for Plot[] (default: NUM_SAMPLES)
  pub plot_points: usize,
  /// Legend labels for each series (empty = no legend)
  pub plot_legends: Vec<String>,
  /// Position of the legend (Right, Top, Bottom)
  pub legend_position: LegendPosition,
  /// Show evenly spaced horizontal grid lines (automatic).
  pub grid_lines_y: bool,
  /// Show evenly spaced vertical grid lines (automatic).
  pub grid_lines_x: bool,
  /// Explicit vertical grid lines (positions + per-line style). When
  /// non-empty, these replace the automatic `grid_lines_x` lines.
  pub grid_x_lines: Vec<GridLine>,
  /// Explicit horizontal grid lines. Replace automatic `grid_lines_y` lines.
  pub grid_y_lines: Vec<GridLine>,
  /// `GridLinesStyle -> Directive[…]`: the default style for every grid line
  /// that carries no style of its own. `None` = the built-in gray.
  pub grid_lines_style: Option<SeriesStyle>,
  /// Use frame (left+bottom border) instead of axes
  pub frame: bool,
  /// `Evaluated -> True`: work the body out once, with the plot variable
  /// symbolic, rather than re-evaluating it at every sample point.
  pub evaluated: bool,
  /// Labels on the bottom and left frame edges (`FrameLabel`). These sit
  /// centred outside the plot area, unlike an [`Self::axes_label`], which
  /// Wolfram writes at the far end of its axis.
  pub frame_label_bottom: Option<String>,
  pub frame_label_left: Option<String>,
  /// Label on the top frame edge (FrameLabel 4-element form: top of {bottom,top})
  pub frame_label_top: Option<String>,
  /// Label on the right frame edge (FrameLabel 4-element form: right of {left,right})
  pub frame_label_right: Option<String>,
  /// Format x-axis labels as dates (AbsoluteTime seconds since 1900-01-01)
  pub date_axis: bool,
  /// Whether x-axis is logarithmic (data is in log10 space)
  pub log_x: bool,
  /// Whether y-axis is logarithmic (data is in log10 space)
  pub log_y: bool,
  /// Callout labels for each series (None = no callout for that series)
  pub callout_labels: Vec<Option<String>>,
  /// Per-point labels from `Labeled[y, label]` data entries, parallel to
  /// each series' points (None = no label for that point). Empty when no
  /// point is labeled.
  pub point_labels: Vec<Vec<Option<String>>>,
  /// Stacked mode (StackedListPlot): fill each series down to the previous
  /// series' curve instead of to a constant baseline, producing opaque
  /// stacked bands. The `all_points` passed in are the cumulative curves.
  pub stacked: bool,
  /// `Epilog -> {…}` graphics primitives (already evaluated), drawn on top
  /// of the plotted data in data coordinates.
  pub epilog: Vec<Expr>,
  /// Per-series error bars from `Around` data values, parallel to the
  /// series' points: each entry is ((dx_minus, dx_plus), (dy_minus,
  /// dy_plus)) in data units. Empty when the data has no uncertainties.
  pub error_bars: Vec<Vec<((f64, f64), (f64, f64))>>,
  /// How the `error_bars` uncertainties are rendered.
  pub interval_markers: IntervalMarkers,
  /// The original data points per series when the drawn curve differs from
  /// the data (e.g. `InterpolationOrder` step/spline curves). Anchors
  /// `Mesh -> Full` dots, error bars, and per-point labels. Empty when the
  /// drawn curve is the data itself.
  pub data_points: Vec<Vec<(f64, f64)>>,
  /// `Background -> color`: fill for the whole image, replacing the
  /// theme background. `None` keeps the theme default.
  pub background: Option<RGBColor>,
  /// `ImagePadding -> {{left, right}, {bottom, top}}`: the space, in display
  /// pixels, between the plotting area and the edges of the image. When set
  /// it replaces the automatic margins, so the frame fills whatever the
  /// padding leaves — which is how a plot given a short `ImageSize` keeps a
  /// usable frame instead of losing it to fixed label margins. `None` keeps
  /// the automatic margins.
  pub image_padding: Option<[f64; 4]>,
  /// `AspectRatio -> r`: ratio (height/width) of the plotting *area* (the
  /// data frame, excluding label/tick margins). When set, the total image
  /// height is derived so the frame has this ratio, matching Wolfram. `None`
  /// keeps `svg_height` as given.
  pub aspect_ratio: Option<f64>,
  /// `PlotMarkers -> …`: the glyph each series' points are drawn with,
  /// cycled over the series. Empty = plain round points.
  pub plot_markers: Vec<Option<PlotMarker>>,
  /// `LabelStyle -> …`: font size / color applied to the `FrameLabel`,
  /// `AxesLabel`, and `PlotLabel` text. `None` keeps the theme default.
  pub label_style: Option<LabelStyleSpec>,
}

/// Font size / color parsed from a `LabelStyle -> …` option. Either field
/// may be absent — an unset one keeps the theme default for that property.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LabelStyleSpec {
  pub color: Option<WoxiColor>,
  pub font_size: Option<f64>,
}

impl Default for PlotOptions {
  fn default() -> Self {
    Self {
      svg_width: DEFAULT_WIDTH,
      svg_height: DEFAULT_HEIGHT,
      full_width: false,
      filling: Filling::None,
      filling_rules: Vec::new(),
      filling_style: None,
      filling_styles: Vec::new(),
      mesh: Mesh::None,
      plot_label: None,
      axes_label: None,
      plot_style: Vec::new(),
      axes: (true, true),
      ticks: true,
      plot_points: NUM_SAMPLES,
      plot_legends: Vec::new(),
      legend_position: LegendPosition::default(),
      grid_lines_y: false,
      grid_lines_x: false,
      grid_x_lines: Vec::new(),
      grid_y_lines: Vec::new(),
      grid_lines_style: None,
      frame: false,
      evaluated: false,
      ticks_x: None,
      ticks_y: None,
      frame_label_bottom: None,
      frame_label_left: None,
      frame_label_top: None,
      frame_label_right: None,
      date_axis: false,
      callout_labels: Vec::new(),
      point_labels: Vec::new(),
      log_x: false,
      log_y: false,
      stacked: false,
      epilog: Vec::new(),
      error_bars: Vec::new(),
      interval_markers: IntervalMarkers::default(),
      data_points: Vec::new(),
      background: None,
      image_padding: None,
      aspect_ratio: None,
      plot_markers: Vec::new(),
      label_style: None,
    }
  }
}

/// Draw `Around`-style error bars for one series: a bar spanning each
/// point's uncertainty interval, with short caps at both ends, in the
/// series color. `bars` is parallel to `points` and holds the
/// ((dx_minus, dx_plus), (dy_minus, dy_plus)) half-widths in data units;
/// zero widths draw nothing for that direction.
fn draw_error_bars<
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  points: &[(f64, f64)],
  bars: &[((f64, f64), (f64, f64))],
  color: RGBColor,
  x_span: f64,
  y_span: f64,
) -> Result<(), InterpreterError> {
  // Cap half-lengths: a small fixed fraction of the plotted span.
  let cap_x = x_span * 0.012;
  let cap_y = y_span * 0.012;
  let style = color.stroke_width(RESOLUTION_SCALE);
  let mut segments: Vec<[(f64, f64); 2]> = Vec::new();
  for (&(x, y), &(dx, dy)) in points.iter().zip(bars) {
    if !(x.is_finite() && y.is_finite()) {
      continue;
    }
    if dy.0 > 0.0 || dy.1 > 0.0 {
      let (lo, hi) = (y - dy.0, y + dy.1);
      segments.push([(x, lo), (x, hi)]);
      segments.push([(x - cap_x, lo), (x + cap_x, lo)]);
      segments.push([(x - cap_x, hi), (x + cap_x, hi)]);
    }
    if dx.0 > 0.0 || dx.1 > 0.0 {
      let (lo, hi) = (x - dx.0, x + dx.1);
      segments.push([(lo, y), (hi, y)]);
      segments.push([(lo, y - cap_y), (lo, y + cap_y)]);
      segments.push([(hi, y - cap_y), (hi, y + cap_y)]);
    }
  }
  for seg in segments {
    chart
      .draw_series(std::iter::once(PathElement::new(seg.to_vec(), style)))
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }
  Ok(())
}

/// Draw `IntervalMarkers -> "Bands"` for one series: a translucent filled
/// band spanning each point's y-uncertainty interval, outlined in the series
/// color. `bars` is parallel to `points` (see [`draw_error_bars`]); a point
/// without y-uncertainty pinches the band to the point itself.
fn draw_interval_band<
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  points: &[(f64, f64)],
  bars: &[((f64, f64), (f64, f64))],
  color: RGBColor,
) -> Result<(), InterpreterError> {
  let edges: Vec<((f64, f64), (f64, f64))> = points
    .iter()
    .zip(bars)
    .filter(|((x, y), _)| x.is_finite() && y.is_finite())
    .map(|(&(x, y), &(_dx, dy))| ((x, y - dy.0), (x, y + dy.1)))
    .collect();
  if edges.is_empty() {
    return Ok(());
  }
  // Closed outline: along the upper edge, then back along the lower edge.
  let mut outline: Vec<(f64, f64)> = edges.iter().map(|e| e.1).collect();
  outline.extend(edges.iter().rev().map(|e| e.0));
  if outline.len() >= 3 {
    chart
      .draw_series(std::iter::once(Polygon::new(
        outline.clone(),
        color.mix(0.2),
      )))
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }
  outline.push(edges[0].1);
  chart
    .draw_series(std::iter::once(PathElement::new(
      outline,
      color.stroke_width(RESOLUTION_SCALE),
    )))
    .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  Ok(())
}

/// Draw a dashed/dotted line on a chart.
/// `dash_pattern` contains alternating dash/gap lengths as fractions of the
/// data range (matching Wolfram's Dashing convention where 0.01 ≈ 1% of width).
fn draw_dashed_line<
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  segment: &[(f64, f64)],
  color: RGBColor,
  stroke_w: u32,
  dash_pattern: &[f64],
  x_span: f64,
) -> Result<(), InterpreterError> {
  if segment.len() < 2 || dash_pattern.is_empty() {
    return Ok(());
  }
  // Convert fractional dash lengths to data-space lengths
  let dashes: Vec<f64> = dash_pattern.iter().map(|d| d * x_span).collect();
  let style = color.stroke_width(stroke_w);

  let mut dash_idx = 0; // index into dashes array
  let mut remaining = dashes[0]; // remaining length in current dash/gap
  let mut drawing = true; // true = dash, false = gap
  let mut current_start = segment[0];

  for i in 1..segment.len() {
    let (x0, y0) = current_start;
    let (x1, y1) = segment[i];
    let dx = x1 - x0;
    let dy = y1 - y0;
    let seg_len = (dx * dx + dy * dy).sqrt();
    if seg_len < 1e-12 {
      current_start = segment[i];
      continue;
    }

    let mut consumed = 0.0;
    while consumed < seg_len {
      let available = seg_len - consumed;
      let take = remaining.min(available);
      let t0 = consumed / seg_len;
      let t1 = (consumed + take) / seg_len;
      let p0 = (x0 + dx * t0, y0 + dy * t0);
      let p1 = (x0 + dx * t1, y0 + dy * t1);

      if drawing && take > 1e-12 {
        chart
          .draw_series(std::iter::once(PathElement::new(vec![p0, p1], style)))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      }

      consumed += take;
      remaining -= take;
      if remaining < 1e-12 {
        drawing = !drawing;
        dash_idx = (dash_idx + 1) % dashes.len();
        remaining = dashes[dash_idx];
      }
    }
    current_start = segment[i];
  }
  Ok(())
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

/// A dashed data series deferred for post-render emission as a single
/// `<polyline stroke-dasharray>` instead of one `<polyline>` per dash — which
/// plotters' backend would otherwise produce.
struct DashedOverlay {
  color: (u8, u8, u8),
  stroke_w: u32,
  /// Dash pattern as fractions of the plot width (matching the legend swatch).
  dashes: Vec<f64>,
  /// Polyline vertices in data space.
  points: Vec<(f64, f64)>,
}

/// One dash length, in the coordinate system its `<polyline>` is drawn in.
///
/// A negative length is absolute pixels (a named size — `Dashed` is 4 on / 4
/// off whatever the image size); a positive one is a fraction of `reference`,
/// the width Wolfram measures relative dashes against; zero is the dot
/// Wolfram draws as one pixel. `px_per_unit` converts a pixel length into the
/// target system (render space counts `RESOLUTION_SCALE` units per pixel).
pub(crate) fn dash_len(d: f64, reference: f64, px_per_unit: f64) -> f64 {
  let len = if d < 0.0 {
    -d * px_per_unit
  } else if d == 0.0 {
    px_per_unit
  } else {
    d * reference
  };
  len.max(0.5)
}

/// Render collected dashed lines as single `<polyline stroke-dasharray>`
/// elements, mapping data coordinates to render-space pixels with the same
/// linear transform used for axis labels/ticks. Dash lengths use the
/// plotting-area extent so series dashes match the legend swatches exactly.
#[allow(clippy::too_many_arguments)]
fn render_dash_overlays(
  overlays: &[DashedOverlay],
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  render_w: f64,
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
) -> String {
  let xr = x_max - x_min;
  let yr = y_max - y_min;
  if xr.abs() < 1e-12 || yr.abs() < 1e-12 {
    return String::new();
  }
  let mut svg = String::new();
  for ov in overlays {
    if ov.points.len() < 2 {
      continue;
    }
    let pts: Vec<String> = ov
      .points
      .iter()
      .map(|&(x, y)| {
        let px = plot_x0 + (x - x_min) / xr * plot_w;
        let py = plot_y0 + (y_max - y) / yr * plot_h;
        format!("{px:.1},{py:.1}")
      })
      .collect();
    let dash: Vec<String> = ov
      .dashes
      .iter()
      .map(|d| {
        format!("{:.1}", dash_len(*d, render_w, RESOLUTION_SCALE as f64))
      })
      .collect();
    let (r, g, b) = ov.color;
    svg.push_str(&format!(
      "<polyline fill=\"none\" stroke=\"rgb({},{},{})\" \
       stroke-width=\"{}\" stroke-dasharray=\"{}\" stroke-linecap=\"round\" \
       points=\"{}\"/>\n",
      r,
      g,
      b,
      ov.stroke_w,
      dash.join(","),
      pts.join(" ")
    ));
  }
  svg
}

/// Default color for an unstyled dashed grid line. Approximates the flat-gray
/// look of the solid grid lines (`rgb(102,102,102)` at 50% opacity over a white
/// background), since dash overlays carry no opacity.
const DEFAULT_GRID_DASH_RGB: (u8, u8, u8) = (179, 179, 179);

/// Resolve one axis' grid-line positions into `(position, style)` pairs.
///
/// Explicit positions (with their optional per-line directives) take
/// precedence over the evenly spaced automatic lines that `GridLines ->
/// Automatic` draws at `step` intervals. Lines that carry no style of their
/// own inherit `default_style` (the plot's `GridLinesStyle`).
fn resolve_grid_lines(
  explicit: &[GridLine],
  automatic: bool,
  min: f64,
  max: f64,
  step: f64,
  default_style: Option<&SeriesStyle>,
) -> Vec<(f64, SeriesStyle)> {
  let inherit = |style: &SeriesStyle| -> SeriesStyle {
    match default_style {
      Some(d)
        if style.color.is_none()
          && style.thickness.is_none()
          && style.dashing.is_none() =>
      {
        d.clone()
      }
      _ => style.clone(),
    }
  };
  if !explicit.is_empty() {
    return explicit
      .iter()
      .map(|g| (g.pos, inherit(&g.style)))
      .collect();
  }
  if !automatic || step <= 0.0 || !step.is_finite() {
    return Vec::new();
  }
  let style = inherit(&SeriesStyle::default());
  let mut v = Vec::new();
  let mut pos = (min / step).ceil() * step;
  while pos <= max {
    v.push((pos, style.clone()));
    pos += step;
  }
  v
}

/// Resolve a grid line's style into rendering properties:
/// `(optional rgb, stroke width in render units, optional dash fractions)`.
/// A `None` rgb means "use the default gray grid color".
fn grid_line_props(
  style: &SeriesStyle,
) -> (Option<(u8, u8, u8)>, u32, Option<Vec<f64>>) {
  let rgb = style.color.as_ref().map(|c| {
    (
      (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  });
  let stroke_w = match style.thickness {
    Some(t) => ((t * RESOLUTION_SCALE as f64).round() as u32).max(1),
    None => RESOLUTION_SCALE,
  };
  (rgb, stroke_w, style.dashing.clone())
}

/// Draw the marks and labels of an explicit `Ticks -> {xspec, yspec}`, in
/// the same place the automatic ticks would sit: the x ticks below the
/// bottom axis and the y ticks left of the left one. Returns the empty
/// string when neither axis was given explicit positions.
///
/// Shared by the line and scatter renderers, both of which suppress their
/// automatic ticks once an axis carries explicit ones. The labels arrive
/// ready to draw from [`parse_explicit_ticks`] — escaped, and typeset when
/// the tick was given as an expression (`Pi/2` reads `π/2`).
/// `area` is the plotting rectangle `(x0, y0, w, h)` and `range` the data
/// range `(x_min, x_max, y_min, y_max)` it maps, both in render units.
fn explicit_ticks_svg(
  opts: &PlotOptions,
  (plot_x0, plot_y0, plot_w, plot_h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  sf: f64,
  label_fill: &str,
) -> String {
  let mut out = String::new();
  let axis_y = plot_y0 + plot_h;
  let tick_len = sf * 5.0;
  let tick_font = sf * 13.0;
  if let Some(ticks) = &opts.ticks_x {
    for (pos, label) in ticks {
      if *pos < x_min || *pos > x_max || x_max <= x_min {
        continue;
      }
      let px = plot_x0 + (pos - x_min) / (x_max - x_min) * plot_w;
      out.push_str(&format!(
        "<line x1=\"{px:.1}\" y1=\"{axis_y:.1}\" x2=\"{px:.1}\" y2=\"{:.1}\" stroke=\"{label_fill}\" stroke-width=\"{sf:.1}\"/>\n",
        axis_y - tick_len,
      ));
      out.push_str(&format!(
        "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{tick_font:.0}\" fill=\"{label_fill}\">{label}</text>\n",
        axis_y + tick_font,
      ));
    }
  }
  if let Some(ticks) = &opts.ticks_y {
    for (pos, label) in ticks {
      if *pos < y_min || *pos > y_max || y_max <= y_min {
        continue;
      }
      let py = plot_y0 + plot_h - (pos - y_min) / (y_max - y_min) * plot_h;
      out.push_str(&format!(
        "<line x1=\"{plot_x0:.1}\" y1=\"{py:.1}\" x2=\"{:.1}\" y2=\"{py:.1}\" stroke=\"{label_fill}\" stroke-width=\"{sf:.1}\"/>\n",
        plot_x0 + tick_len,
      ));
      out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{py:.1}\" text-anchor=\"end\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{tick_font:.0}\" fill=\"{label_fill}\">{label}</text>\n",
        plot_x0 - tick_len,
      ));
    }
  }
  out
}

/// Core SVG generation for 2D line plots with full option support.
/// Draw a plot's labels — `FrameLabel` on all four edges, `AxesLabel` at the
/// ends of the axes, and the `PlotLabel` above them — into `labels_svg`.
///
/// Shared by the line and scatter renderers and by the density/contour
/// family in `field_plot`: each draws its own axes and then injects text,
/// and the placement rules are the same for all of them.
/// `area` is the plotting rectangle `(x0, y0, w, h)` and `range` the data
/// range `(x_min, x_max, y_min, y_max)` it maps, both in render units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plot_labels_svg(
  opts: &PlotOptions,
  (plot_x0, margin_top, plot_w, plot_h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  margin_left_f: f64,
  sf: f64,
  label_fill: &str,
  title_default_fill: &str,
) -> String {
  let axis_y = margin_top + plot_h;
  // `LabelStyle` overrides the frame/axes/title text's default size and
  // color; either half may be absent, in which case that property keeps
  // the theme default.
  let label_style_font_size = opts.label_style.and_then(|s| s.font_size);
  let label_style_color = opts.label_style.and_then(|s| s.color);
  let font_size = label_style_font_size.map_or(sf * 14.0, |pt| sf * pt);
  let title_font_size = label_style_font_size.map_or(sf * 17.0, |pt| sf * pt);
  let label_fill_owned =
    label_style_color.map(super::graphics::Color::to_svg_rgb);
  let label_fill = label_fill_owned.as_deref().unwrap_or(label_fill);
  let title_default_fill_owned =
    label_style_color.map(super::graphics::Color::to_svg_rgb);
  let title_default_fill = title_default_fill_owned
    .as_deref()
    .unwrap_or(title_default_fill);
  let has_top_label =
    opts.frame_label_top.as_ref().is_some_and(|t| !t.is_empty());
  let axes_label_y = opts
    .axes_label
    .as_ref()
    .map(|(_, y)| y.as_str())
    .filter(|y| !y.is_empty() && opts.axes.1);
  let mut labels_svg = String::new();
  // Bottom/left FrameLabel: centred outside the plot area.
  if let Some(x_label) = &opts.frame_label_bottom
    && !x_label.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    // Sit clearly below the x tick labels (which occupy ~one tick-font
    // height below the axis) rather than crowding the frame.
    let base_y = axis_y + sf * 13.0 + font_size * 1.4;
    labels_svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(x_label)
    ));
  }
  if let Some(y_label) = &opts.frame_label_left
    && !y_label.is_empty()
  {
    let cy = margin_top + plot_h / 2.0;
    // Place the rotated label just left of the y tick-label column
    // (which right-aligns near the axis) instead of at the far gutter
    // edge — adapting to the actual tick width.
    let tick_w = max_y_tick_label_chars(y_min, y_max) as f64 * sf * 13.0 * 0.6;
    let tick_left = plot_x0 - 8.0 * sf - tick_w;
    let lx = (tick_left - font_size * 0.5 - sf * 5.0)
      .max(margin_left_f + font_size * 0.5);
    labels_svg.push_str(&format!(
      "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(y_label)
    ));
  }

  // AxesLabel: at the far end of each axis, the way Wolfram writes it.
  labels_svg.push_str(&axes_label_svg(
    opts.axes_label.as_ref(),
    (plot_x0, margin_top, plot_w, plot_h),
    (x_min, x_max, y_min, y_max),
    opts.axes,
    font_size,
    label_fill,
  ));

  // Top FrameLabel (sits just above the plot's top edge)
  if let Some(top_label) = &opts.frame_label_top
    && !top_label.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let ty = margin_top - font_size * 0.6;
    labels_svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(top_label)
    ));
  }

  // Right FrameLabel (rotated +90 on the right frame edge)
  if let Some(right_label) = &opts.frame_label_right
    && !right_label.is_empty()
  {
    let cy = margin_top + plot_h / 2.0;
    let rx = plot_x0 + plot_w + font_size * 1.4;
    labels_svg.push_str(&format!(
      "<text x=\"{rx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\" transform=\"rotate(90,{rx:.1},{cy:.1})\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(right_label)
    ));
  }

  // PlotLabel — shifted above the top FrameLabel and/or the y AxesLabel
  // when they share the top margin.
  if let Some(sl) = &opts.plot_label
    && !sl.text.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let stacked = if has_top_label { 1.0 } else { 0.0 }
      + if axes_label_y.is_some() { 1.0 } else { 0.0 };
    let ty = margin_top - title_font_size * 0.5 - stacked * font_size * 1.2;
    let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
    // A stacked title grows upwards: its last line stays where a one-line
    // title would sit, so whatever shares the top margin below it (the y
    // `AxesLabel`) is not written over.
    let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
    let fill = sl
      .color
      .as_ref()
      .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
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
      sl.svg_scaled_stacked(sf, cx, fs * 1.2)
    ));
  }

  labels_svg
}

/// Draw an `Epilog`'s primitives over a finished plot. Both renderers (the
/// line one below and the scatter one) inject them the same way, so a plot
/// keeps its epilog whichever of the two drew it.
///
/// `area` is the plotting area as `(x0, y0, width, height)` in render-space
/// pixels and `ranges` the data extents it displays, as `(x_min, x_max,
/// y_min, y_max)`.
fn inject_epilog(
  buf: &mut String,
  opts: &PlotOptions,
  area: (f64, f64, f64, f64),
  ranges: (f64, f64, f64, f64),
  scale: f64,
) {
  if opts.epilog.is_empty() {
    return;
  }
  let (x0, y0, w, h) = area;
  let (x_min, x_max, y_min, y_max) = ranges;
  let area = crate::functions::plot_epilog::PlotArea {
    x0,
    y0,
    w,
    h,
    x_min,
    x_max,
    y_min,
    y_max,
    scale,
  };
  let epilog_svg =
    crate::functions::plot_epilog::render_epilog_svg(&opts.epilog, &area);
  if let Some(pos) = buf.rfind("</svg>") {
    buf.insert_str(pos, &epilog_svg);
  }
}

fn generate_svg_with_options(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let svg_width = opts.svg_width;
  let mut svg_height = opts.svg_height;
  let full_width = opts.full_width;
  let (show_x_axis, show_y_axis) = opts.axes;
  // A framed plot carries its ticks on the frame edges, so they are labelled
  // whether or not the axes themselves are drawn — `Frame -> True, Axes ->
  // False` is the standard framed look, and it used to come out bare.
  let (tick_axis_x, tick_axis_y) = if opts.frame {
    (true, true)
  } else {
    (show_x_axis, show_y_axis)
  };
  let show_ticks = opts.ticks;
  let render_width = svg_width * RESOLUTION_SCALE;
  let mut render_height = svg_height * RESOLUTION_SCALE;

  let sf = RESOLUTION_SCALE as f64;
  let s = RESOLUTION_SCALE as i32;

  // Compute dynamic margins for labels
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  // A `FrameLabel` sits centred outside the bottom/left edge; an `AxesLabel`
  // sits at the far end of its axis, so it needs room to the right and above
  // instead.
  let has_x_axis_label = opts
    .frame_label_bottom
    .as_ref()
    .is_some_and(|t| !t.is_empty());
  let has_y_axis_label = opts
    .frame_label_left
    .as_ref()
    .is_some_and(|t| !t.is_empty());
  let axes_label_x = opts
    .axes_label
    .as_ref()
    .map(|(x, _)| x.as_str())
    .filter(|x| !x.is_empty() && show_x_axis);
  let axes_label_y = opts
    .axes_label
    .as_ref()
    .map(|(_, y)| y.as_str())
    .filter(|y| !y.is_empty() && show_y_axis);
  let has_top_label =
    opts.frame_label_top.as_ref().is_some_and(|t| !t.is_empty());
  let has_right_label = opts
    .frame_label_right
    .as_ref()
    .is_some_and(|t| !t.is_empty());

  // Reserve top room for a PlotLabel, a top FrameLabel and/or the y AxesLabel
  // (they stack).
  let top_margin = 10 * s
    + if has_plot_label { 25 * s } else { 0 }
    + plot_label_extra_lines(opts.plot_label.as_ref()) as i32 * 20 * s
    + if has_top_label { 25 * s } else { 0 }
    + if axes_label_y.is_some() { 20 * s } else { 0 };

  // Label areas and margins computed per-axis.
  // Setting a label area to 0 suppresses that axis line in plotters.
  let bottom_extra = if tick_axis_x && show_ticks && has_x_axis_label {
    24.0 * sf
  } else {
    0.0
  };
  let x_label_area: u32 = if !tick_axis_x {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    40 * RESOLUTION_SCALE + bottom_extra as u32
  };
  let y_label_area: u32 = if !tick_axis_y {
    0
  } else if !show_ticks {
    5 * RESOLUTION_SCALE
  } else {
    65 * RESOLUTION_SCALE
  };
  let margin_left: u32 = if tick_axis_y {
    10 * s as u32
  } else {
    5 * s as u32
  };
  let margin_right: u32 = if has_right_label {
    40 * s as u32
  } else if let Some(label) = axes_label_x {
    // Enough room for the x AxesLabel, which runs to the right of the axis
    // at `sf * 14` — ~0.62 em per character covers the proportional font.
    (10.0 * sf
      + 8.0 * sf
      + crate::functions::graphics::box_string_visible_len(label) as f64
        * 0.62
        * sf
        * 14.0)
      .round() as u32
  } else {
    10 * s as u32
  };
  let margin_bottom: u32 = if tick_axis_x {
    10 * s as u32
  } else {
    5 * s as u32
  };

  // With `ImagePadding` the padding *is* the margin, so an `AspectRatio`
  // fixes the canvas height directly: the plot area spans the width the
  // padding leaves, and the height follows from the ratio. Deriving it here,
  // before the margins are computed, keeps the frame as wide as the image
  // instead of shrinking it to fit a canvas sized by the default ratio.
  if let Some(ar) = opts.aspect_ratio
    && let Some([pad_left, pad_right, pad_bottom, pad_top]) = opts.image_padding
  {
    let plot_w = render_width as f64 - (pad_left + pad_right) * sf;
    if plot_w > 0.0 && ar > 0.0 {
      let target_render_h = plot_w * ar + (pad_bottom + pad_top) * sf;
      svg_height = ((target_render_h / sf).round() as u32).max(1);
      render_height = svg_height * RESOLUTION_SCALE;
    }
  }

  // `ImagePadding` replaces the automatic margins: the left/bottom padding
  // becomes the tick label areas (the axis labels are drawn inside them) and
  // the right/top padding the corresponding margins.
  let (
    top_margin,
    x_label_area,
    y_label_area,
    margin_left,
    margin_right,
    margin_bottom,
  ) = match opts.image_padding {
    Some(padding) => {
      let m = padded_margins(
        padding,
        (render_width, render_height),
        opts.aspect_ratio,
        sf,
      );
      (m.top as i32, m.bottom, m.left, 0u32, m.right, 0u32)
    }
    None => (
      top_margin,
      x_label_area,
      y_label_area,
      margin_left,
      margin_right,
      margin_bottom,
    ),
  };

  // AspectRatio sizes the plotting *area* (the data frame), not the whole
  // image. Derive the total height so that
  //   plot_area_height = plot_area_width * aspect_ratio
  // where the plot area excludes the label/tick margins. Recomputing here
  // (rather than pre-setting svg_height from svg_width) keeps a short/wide
  // AspectRatio from collapsing the frame when fixed-size label margins would
  // otherwise consume most of the canvas.
  if let Some(ar) = opts.aspect_ratio
    && opts.image_padding.is_none()
  {
    let plot_w = (render_width as f64)
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    if plot_w > 0.0 {
      let plot_h = plot_w * ar;
      let target_render_h =
        plot_h + top_margin as f64 + margin_bottom as f64 + x_label_area as f64;
      // Round to a whole svg unit so svg_height * scale == render_height.
      svg_height = ((target_render_h / sf).round() as u32).max(1);
      render_height = svg_height * RESOLUTION_SCALE;
    }
  }

  let (theme_bg, dark_gray, light_gray, label_fill, title_default_fill) =
    plot_theme();
  // Background -> color replaces the theme background for the whole image.
  let bg_color = opts.background.unwrap_or(theme_bg);

  // Dashed series are collected here and emitted after the plot is drawn as a
  // single `<polyline stroke-dasharray>` each (rather than one element per
  // dash). Skipped for log axes, which keep the per-dash fallback because the
  // coordinate transform is non-linear.
  let mut dashed_overlays: Vec<DashedOverlay> = Vec::new();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let tick = MINOR_TICK_LEN * s;

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
        // An axis given explicit ticks draws them itself, below.
        let has_explicit_x_ticks = opts.ticks_x.is_some();
        let has_explicit_y_ticks = opts.ticks_y.is_some();
        if show_ticks && (tick_axis_x || tick_axis_y) {
          let xmaj = if date_axis {
            nice_date_step(x_max - x_min)
          } else {
            nice_step(x_max - x_min, AXIS_TICK_TARGET)
          };
          let ymaj = nice_step(y_max - y_min, AXIS_TICK_TARGET);
          x_major = xmaj;
          y_major = ymaj;
          let x_minor = if date_axis { xmaj } else { xmaj / 5.0 };
          let y_minor = ymaj / 5.0;
          x_labels_count = if !tick_axis_x || date_axis || has_explicit_x_ticks
          {
            0
          } else if log_x {
            // Let LogCoord decide tick placement; ~10 labels for log axes
            10
          } else {
            ((x_max - x_min) / x_minor).round() as usize + 1
          };
          y_labels_count = if !tick_axis_y || has_explicit_y_ticks {
            0
          } else if log_y {
            10
          } else {
            ((y_max - y_min) / y_minor).round() as usize + 1
          };
          x_tick_size = if tick_axis_x { tick } else { 0 };
          y_tick_size = if tick_axis_y { tick } else { 0 };
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
              format_tick_with_step(*v, x_major)
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
              format_tick_with_step(*v, y_major)
            } else {
              String::new()
            }
          })
          .axis_style(axis_style)
          .label_style(
            ("sans-serif", sf * 13.0)
              .into_font()
              .color(&dark_gray),
          )
          .set_tick_mark_size(LabelAreaPosition::Left, y_tick_size)
          .set_tick_mark_size(LabelAreaPosition::Bottom, x_tick_size)
          .draw()
          .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

        // Draw grid lines before the series so they sit behind the data.
        // Explicit positions (with optional per-line color/thickness/dashing)
        // take precedence over the evenly spaced automatic lines. Solid lines
        // go through plotters; dashed lines are deferred to single
        // stroke-dasharray overlays (which render on top). Linear axes only —
        // log axis grid not yet supported. (WL grid lines are solid by
        // default; dashing needs an explicit style.)
        let grid_color = RGBColor(0x66, 0x66, 0x66).mix(0.5);

        // Horizontal (y) grid lines.
        let y_grid = resolve_grid_lines(
          &opts.grid_y_lines,
          opts.grid_lines_y,
          y_min,
          y_max,
          nice_step(y_max - y_min, AXIS_TICK_TARGET),
          opts.grid_lines_style.as_ref(),
        );
        if !log_y {
          let tol = (y_max - y_min).abs() * 1e-9;
          for (gy, style) in &y_grid {
            if *gy < y_min - tol || *gy > y_max + tol {
              continue;
            }
            let (rgb, stroke_w, dash) = grid_line_props(style);
            if let Some(dashes) = dash {
              dashed_overlays.push(DashedOverlay {
                color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
                stroke_w,
                dashes,
                points: vec![(x_min, *gy), (x_max, *gy)],
              });
            } else {
              let shape = match rgb {
                Some((r, g, b)) => RGBColor(r, g, b).stroke_width(stroke_w),
                None => grid_color.stroke_width(stroke_w),
              };
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![(x_min, *gy), (x_max, *gy)],
                  shape,
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }
        }

        // Vertical (x) grid lines.
        let x_grid = resolve_grid_lines(
          &opts.grid_x_lines,
          opts.grid_lines_x,
          x_min,
          x_max,
          if date_axis {
            nice_date_step(x_max - x_min)
          } else {
            nice_step(x_max - x_min, AXIS_TICK_TARGET)
          },
          opts.grid_lines_style.as_ref(),
        );
        if !log_x {
          let tol = (x_max - x_min).abs() * 1e-9;
          for (gx, style) in &x_grid {
            if *gx < x_min - tol || *gx > x_max + tol {
              continue;
            }
            let (rgb, stroke_w, dash) = grid_line_props(style);
            if let Some(dashes) = dash {
              dashed_overlays.push(DashedOverlay {
                color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
                stroke_w,
                dashes,
                points: vec![(*gx, y_min), (*gx, y_max)],
              });
            } else {
              let shape = match rgb {
                Some((r, g, b)) => RGBColor(r, g, b).stroke_width(stroke_w),
                None => grid_color.stroke_width(stroke_w),
              };
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![(*gx, y_min), (*gx, y_max)],
                  shape,
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }
        }

        // Draw a full frame (all four borders) when frame mode is enabled
        if opts.frame {
          let frame_style = dark_gray.stroke_width(RESOLUTION_SCALE);
          chart
            .draw_series(std::iter::once(PathElement::new(
              vec![
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
                (x_min, y_min),
              ],
              frame_style,
            )))
            .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
        }

        // Draw lighter origin lines through x=0 and y=0 if visible.
        // `Axes -> False` hides them along with the axes themselves.
        if !opts.frame && (show_x_axis || show_y_axis) {
          let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
          if show_x_axis && y_min < 0.0 && y_max > 0.0 {
            chart
              .draw_series(std::iter::once(PathElement::new(
                vec![(x_min, 0.0), (x_max, 0.0)],
                origin_line,
              )))
              .map_err(|e| {
                InterpreterError::EvaluationError(format!("Plot: {e}"))
              })?;
          }
          if show_y_axis && x_min < 0.0 && x_max > 0.0 {
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
          let stroke_w = series_thickness(&opts.plot_style, series_idx);
          let dashing = series_dashing(&opts.plot_style, series_idx);
          let segments = clip_segments_to_y_range(
            split_into_segments(points),
            y_min,
            y_max,
          );

          // Uncertainty bands lie underneath the curve and any filling.
          // Like the bars, they are parallel to the original data points,
          // which may differ from the drawn curve (InterpolationOrder).
          if opts.interval_markers == IntervalMarkers::Bands
            && let Some(bars) = opts.error_bars.get(series_idx)
          {
            let anchor = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            draw_interval_band(&mut chart, anchor, bars, color)?;
          }

          // Draw filled area before the line so the line renders on top.
          // In stacked mode each band is bounded below by the previous
          // cumulative curve (or the axis for the first series) and above by
          // the current one; the bands are disjoint so an opaque polygon fill
          // renders cleanly. Otherwise fill each segment down to the constant
          // reference level given by the Filling option.
          if opts.stacked {
            let baseline: Vec<(f64, f64)> = if series_idx == 0 {
              points.iter().map(|&(x, _)| (x, 0.0)).collect()
            } else {
              all_points[series_idx - 1].clone()
            };
            let mut polygon: Vec<(f64, f64)> = points.to_vec();
            polygon.extend(baseline.iter().rev().copied());
            if polygon.len() >= 3 {
              chart
                .draw_series(std::iter::once(Polygon::new(
                  polygon,
                  RGBColor(r, g, b).mix(0.6),
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          } else {
            let paint =
              fill_paint(series_filling_style(opts, series_idx), (r, g, b));
            match series_fill_target(opts, series_idx) {
              FillTarget::Level(level) => {
                if let Some(ref_y) = level.reference_y(y_min, y_max) {
                  for segment in &segments {
                    if segment.len() < 2 {
                      continue;
                    }
                    chart
                      .draw_series(AreaSeries::new(
                        segment.iter().copied(),
                        ref_y,
                        paint,
                      ))
                      .map_err(|e| {
                        InterpreterError::EvaluationError(format!("Plot: {e}"))
                      })?;
                  }
                }
              }
              // `Filling -> {i -> {j}}`: fill the region between this
              // series and series j over the overlap of their x-domains.
              FillTarget::Series(target_idx) => {
                if target_idx != series_idx
                  && let Some(target) = all_points.get(target_idx)
                  && let Some(polygon) = fill_between_polygon(points, target)
                {
                  chart
                    .draw_series(std::iter::once(Polygon::new(polygon, paint)))
                    .map_err(|e| {
                      InterpreterError::EvaluationError(format!("Plot: {e}"))
                    })?;
                }
              }
            }
          }

          if let Some(ref dash_pattern) = dashing {
            if opts.log_x || opts.log_y {
              // Log axes use a non-linear transform; keep the per-dash
              // fallback that lets plotters map each segment.
              for segment in &segments {
                draw_dashed_line(
                  &mut chart,
                  segment,
                  color,
                  stroke_w,
                  dash_pattern,
                  x_max - x_min,
                )?;
              }
            } else {
              // Defer to a single <polyline stroke-dasharray> per segment,
              // emitted after the plot is drawn.
              for segment in &segments {
                if segment.len() >= 2 {
                  dashed_overlays.push(DashedOverlay {
                    color: (r, g, b),
                    stroke_w,
                    dashes: dash_pattern.clone(),
                    points: segment.clone(),
                  });
                }
              }
            }
          } else {
            for segment in &segments {
              chart
                .draw_series(LineSeries::new(
                  segment.iter().copied(),
                  color.stroke_width(stroke_w),
                ))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }

          // Error bars from Around data values, on top of the line. They
          // are parallel to the original data points, which may differ from
          // the drawn curve (InterpolationOrder).
          if opts.interval_markers == IntervalMarkers::Fences
            && let Some(bars) = opts.error_bars.get(series_idx)
          {
            let anchor = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            draw_error_bars(
              &mut chart,
              anchor,
              bars,
              color,
              x_max - x_min,
              y_max - y_min,
            )?;
          }

          // Draw mesh dots: at every curve point for Mesh -> All, at the
          // original data points for Mesh -> Full.
          if opts.mesh != Mesh::None {
            let mesh_src: &[(f64, f64)] = if opts.mesh == Mesh::Full {
              opts
                .data_points
                .get(series_idx)
                .map(|v| v.as_slice())
                .unwrap_or(points)
            } else {
              points
            };
            let marker_size = 3 * RESOLUTION_SCALE;
            let finite_pts: Vec<(f64, f64)> = mesh_src
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

          // `PlotMarkers` glyphs sit on the data points of a joined plot,
          // over the line, exactly as the scatter renderer draws them.
          if let Some(marker) = series_marker(&opts.plot_markers, series_idx) {
            let marker_src: &[(f64, f64)] = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            let (mr, mg, mb) = marker.color.unwrap_or_else(|| {
              let c = color.to_rgba();
              (c.0, c.1, c.2)
            });
            let text_style = ("sans-serif", marker.size * RESOLUTION_SCALE as f64)
              .into_font()
              .color(&RGBColor(mr, mg, mb))
              .pos(plotters::style::text_anchor::Pos::new(
                plotters::style::text_anchor::HPos::Center,
                plotters::style::text_anchor::VPos::Center,
              ));
            chart
              .draw_series(
                marker_src
                  .iter()
                  .copied()
                  .filter(|(x, y)| x.is_finite() && y.is_finite())
                  .map(|(x, y)| {
                    plotters::element::Text::new(
                      marker.glyph.clone(),
                      (x, y),
                      text_style.clone(),
                    )
                  }),
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

  inject_drop_shadows(&mut buf, &opts.plot_style, all_points.len());

  // Emit deferred dashed series as single stroke-dasharray polylines (on top
  // of the plot content, just before </svg>).
  if !dashed_overlays.is_empty() {
    let plot_x0 = margin_left as f64 + y_label_area as f64;
    let plot_y0 = top_margin as f64;
    let plot_w = render_width as f64
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    let plot_h = render_height as f64
      - top_margin as f64
      - margin_bottom as f64
      - x_label_area as f64;
    let series_svg = render_dash_overlays(
      &dashed_overlays,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      render_width as f64,
      x_min,
      x_max,
      y_min,
      y_max,
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &series_svg);
    }
  }

  // Draw Epilog primitives over the plotted data, using the same
  // data→pixel transform as the dash overlays above.
  inject_epilog(
    &mut buf,
    opts,
    (
      margin_left as f64 + y_label_area as f64,
      top_margin as f64,
      render_width as f64
        - margin_left as f64
        - margin_right as f64
        - y_label_area as f64,
      render_height as f64
        - top_margin as f64
        - margin_bottom as f64
        - x_label_area as f64,
    ),
    (x_min, x_max, y_min, y_max),
    sf,
  );

  // Extend labeled (major) ticks so they appear slightly longer than the
  // unlabeled minor ticks drawn by plotters. Only applies when ticks are
  // enabled, a visible axis style is used, and the axis uses linear
  // (non-log, non-date) spacing — log/date axes have their own tick placement.
  if show_ticks && !opts.frame && (show_x_axis || show_y_axis) {
    let margin_left_f = margin_left as f64;
    let margin_right_f = margin_right as f64;
    let margin_bottom_f = margin_bottom as f64;
    let margin_top_f = top_margin as f64;
    let plot_x0 = margin_left_f + y_label_area as f64;
    let plot_y0 = margin_top_f;
    let plot_w = render_width as f64
      - margin_left_f
      - margin_right_f
      - y_label_area as f64;
    let plot_h = render_height as f64
      - margin_top_f
      - margin_bottom_f
      - x_label_area as f64;
    // An axis given explicit `Ticks` marks exactly the positions it names
    // (below), so the automatic majors must not be extended over it — the
    // leftover stubs would sit between the labels, marking nothing.
    let x_axis_ext = if show_x_axis
      && !opts.log_x
      && !opts.date_axis
      && opts.ticks_x.is_none()
    {
      Some((x_min, x_max, nice_step(x_max - x_min, AXIS_TICK_TARGET)))
    } else {
      None
    };
    let y_axis_ext = if show_y_axis && !opts.log_y && opts.ticks_y.is_none() {
      Some((y_min, y_max, nice_step(y_max - y_min, AXIS_TICK_TARGET)))
    } else {
      None
    };
    inject_major_tick_extensions(
      &mut buf,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      x_axis_ext,
      y_axis_ext,
      MINOR_TICK_LEN as f64 * sf,
      MAJOR_TICK_LEN as f64 * sf,
      sf,
      label_fill,
    );
  }

  // Inject label SVG elements before </svg>
  if has_plot_label
    || has_x_axis_label
    || has_y_axis_label
    || axes_label_x.is_some()
    || axes_label_y.is_some()
    || has_top_label
    || has_right_label
    || opts.ticks_x.is_some()
    || opts.ticks_y.is_some()
  {
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

    if let Some(insert_pos) = buf.rfind("</svg>") {
      let mut labels_svg = explicit_ticks_svg(
        opts,
        (plot_x0, margin_top, plot_w, plot_h),
        (x_min, x_max, y_min, y_max),
        sf,
        label_fill,
      );

      labels_svg.push_str(&plot_labels_svg(
        opts,
        (plot_x0, margin_top, plot_w, plot_h),
        (x_min, x_max, y_min, y_max),
        margin_left_f,
        sf,
        label_fill,
        title_default_fill,
      ));

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
          &LogAxisOrientation::Y {
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
          &LogAxisOrientation::X {
            y: margin_top_f + plot_h + font_size * 1.3,
            plot_left: plot_x0,
            plot_w,
          },
        );
      }

      buf.insert_str(insert_pos, &log_labels);
    }
  }

  // Inject Callout labels: text annotation near each labeled series
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
    inject_callout_labels(
      &mut buf,
      opts,
      all_points,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, margin_top_f, plot_w, plot_h),
    );
    inject_point_labels(
      &mut buf,
      opts,
      all_points,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, margin_top_f, plot_w, plot_h),
    );
  }

  inject_legend(&mut buf, opts);

  Ok(buf)
}

/// Inject Callout/Labeled series labels into a finished SVG buffer as a text
/// annotation near each labeled series. `plot_area` is the drawing region in
/// render-space pixels: (left, top, width, height).
fn inject_callout_labels(
  buf: &mut String,
  opts: &PlotOptions,
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_area: (f64, f64, f64, f64),
) {
  if opts.callout_labels.is_empty()
    || !opts.callout_labels.iter().any(std::option::Option::is_some)
  {
    return;
  }
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  let sf = RESOLUTION_SCALE as f64;
  let callout_font_size = sf * 16.0;

  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut callout_svg = String::new();

  for (series_idx, label) in opts.callout_labels.iter().enumerate() {
    let Some(label_text) = label else { continue };
    if series_idx >= all_points.len() {
      continue;
    }
    let points = &all_points[series_idx];

    // Find a good label point: pick the point closest to 2/3 of x range
    let target_x = x_min + (x_max - x_min) * 2.0 / 3.0;
    let best = points
      .iter()
      .filter(|(x, y)| x.is_finite() && y.is_finite())
      .min_by(|a, b| {
        (a.0 - target_x)
          .abs()
          .partial_cmp(&(b.0 - target_x).abs())
          .unwrap_or(std::cmp::Ordering::Equal)
      });

    let Some(&(data_x, data_y)) = best else {
      continue;
    };

    // Convert data coordinates to SVG pixel coordinates
    let frac_x = (data_x - x_min) / (x_max - x_min);
    let frac_y = (data_y - y_min) / (y_max - y_min);
    let px = plot_x0 + frac_x * plot_w;
    let py = plot_top + plot_h * (1.0 - frac_y);

    // Label offset: place text above the curve point
    let label_px = px + sf * 5.0;
    let label_py = py - sf * 12.0;

    // Draw a small line from the curve point to the label
    let (r, g, b) = series_color(&opts.plot_style, series_idx);
    let color_str = format!("rgb({r},{g},{b})");

    callout_svg.push_str(&format!(
      "<line x1=\"{px:.1}\" y1=\"{py:.1}\" x2=\"{label_px:.1}\" y2=\"{label_py:.1}\" \
       stroke=\"{color_str}\" stroke-width=\"{sw}\" />\n",
      sw = sf * 1.0,
    ));
    callout_svg.push_str(&format!(
      "<text x=\"{label_px:.1}\" y=\"{label_py:.1}\" \
       font-family=\"sans-serif\" font-size=\"{callout_font_size:.0}\" \
       fill=\"{color_str}\" dominant-baseline=\"auto\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(label_text)
    ));
  }

  buf.insert_str(insert_pos, &callout_svg);
}

/// Inject per-point labels (from `Labeled[y, label]` data entries) into a
/// finished SVG buffer as text next to each labeled point. `plot_area` is
/// the drawing region in render-space pixels: (left, top, width, height).
fn inject_point_labels(
  buf: &mut String,
  opts: &PlotOptions,
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_area: (f64, f64, f64, f64),
) {
  if !opts
    .point_labels
    .iter()
    .any(|series| series.iter().any(std::option::Option::is_some))
  {
    return;
  }
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  let sf = RESOLUTION_SCALE as f64;
  let font_size = sf * 16.0;
  let (_, _, _, _, text_fill) = plot_theme();

  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut label_svg = String::new();

  for (series_idx, labels) in opts.point_labels.iter().enumerate() {
    // Labels are parallel to the original data points, which may differ
    // from the drawn curve (InterpolationOrder).
    let Some(points) = opts
      .data_points
      .get(series_idx)
      .or_else(|| all_points.get(series_idx))
    else {
      continue;
    };
    for (&(data_x, data_y), label) in points.iter().zip(labels) {
      let Some(label_text) = label else { continue };
      if !(data_x.is_finite() && data_y.is_finite()) {
        continue;
      }
      let frac_x = (data_x - x_min) / (x_max - x_min);
      let frac_y = (data_y - y_min) / (y_max - y_min);
      let px = plot_x0 + frac_x * plot_w;
      let py = plot_top + plot_h * (1.0 - frac_y);
      // Place the label above-right of the point, flipping to above-left
      // near the right edge so it stays inside the plot area.
      let (label_px, anchor) = if frac_x > 0.92 {
        (px - sf * 4.0, "end")
      } else {
        (px + sf * 4.0, "start")
      };
      // Keep the label inside the image when the point sits near the top.
      let label_py = (py - sf * 8.0).max(font_size);
      label_svg.push_str(&format!(
        "<text x=\"{label_px:.1}\" y=\"{label_py:.1}\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{text_fill}\" text-anchor=\"{anchor}\">{}</text>\n",
        crate::functions::graphics::box_string_to_svg(label_text)
      ));
    }
  }

  buf.insert_str(insert_pos, &label_svg);
}

/// Build a `PlotSource` from sampled plot data so that `Show` can later
/// merge multiple pre-rendered plots and re-render via plotters.
pub(crate) fn build_plot_source(
  all_points: &[Vec<(f64, f64)>],
  plot_style: &[SeriesStyle],
  x_range: (f64, f64),
  y_range: (f64, f64),
  image_size: (u32, u32),
  is_scatter: bool,
  filling: Filling,
  filling_style: Option<FillStyle>,
  options: Vec<Expr>,
) -> crate::syntax::PlotSource {
  let series_filling = filling.to_series_filling();
  let (fill_color, fill_opacity) = match filling_style {
    Some(fs) => (fs.color, fs.opacity),
    None => (None, None),
  };
  let series = all_points
    .iter()
    .enumerate()
    .map(|(i, points)| {
      let color = series_color(plot_style, i);
      let thickness = if plot_style.is_empty() {
        None
      } else {
        plot_style[i % plot_style.len()].thickness
      };
      crate::syntax::PlotSeriesData {
        points: points.clone(),
        color,
        is_scatter,
        filling: series_filling,
        fill_color,
        fill_opacity,
        marker: None,
        thickness,
      }
    })
    .collect();

  crate::syntax::PlotSource {
    series,
    x_range,
    y_range,
    image_size,
    options,
  }
}

/// The option rules a graphics function was called with, verbatim, so
/// `Show` can carry them into a merged graphic.
pub(crate) fn explicit_options(args: &[Expr]) -> Vec<Expr> {
  args
    .iter()
    .filter(|a| matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .cloned()
    .collect()
}

/// With a single series a `PlotStyle` list is one combined style, not a
/// per-series cycle: `ListPlot[pts, PlotStyle -> {PointSize[.02], Red}]`
/// draws one red series, not a `PointSize` series and a `Red` series. Later
/// directives win, so a nested `{PointSize[.02], {PointSize[.04], Red}}`
/// (the shape a `Which` inside `PlotStyle` produces) ends up red.
pub(crate) fn collapse_style_for_single_series(
  styles: &[SeriesStyle],
) -> Vec<SeriesStyle> {
  if styles.len() < 2 {
    return styles.to_vec();
  }
  let mut merged = SeriesStyle::default();
  for style in styles {
    if style.color.is_some() {
      merged.color = style.color;
    }
    if style.thickness.is_some() {
      merged.thickness = style.thickness;
    }
    if style.dashing.is_some() {
      merged.dashing.clone_from(&style.dashing);
    }
    if style.shadow.is_some() {
      merged.shadow.clone_from(&style.shadow);
    }
  }
  vec![merged]
}

/// Get the (r, g, b) color for a series, using custom plot_style if available.
fn series_color(plot_style: &[SeriesStyle], idx: usize) -> (u8, u8, u8) {
  if plot_style.is_empty() {
    PLOT_COLORS[idx % PLOT_COLORS.len()]
  } else {
    let style = &plot_style[idx % plot_style.len()];
    if let Some(c) = &style.color {
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
    } else {
      PLOT_COLORS[idx % PLOT_COLORS.len()]
    }
  }
}

/// Get the line thickness (in render-space units) for a series.
/// Default is 15 (1.5px at display size with RESOLUTION_SCALE=10).
fn series_thickness(plot_style: &[SeriesStyle], idx: usize) -> u32 {
  let default_thickness = 15; // 1.5px * RESOLUTION_SCALE
  if plot_style.is_empty() {
    return default_thickness;
  }
  let style = &plot_style[idx % plot_style.len()];
  if let Some(t) = style.thickness {
    (t * RESOLUTION_SCALE as f64).round() as u32
  } else {
    default_thickness
  }
}

/// Get the dash pattern (in data-space fractions) for a series, if any.
fn series_dashing(plot_style: &[SeriesStyle], idx: usize) -> Option<Vec<f64>> {
  if plot_style.is_empty() {
    return None;
  }
  let style = &plot_style[idx % plot_style.len()];
  style.dashing.clone()
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
  let mut svg_height = opts.svg_height;
  let full_width = opts.full_width;
  let render_width = svg_width * RESOLUTION_SCALE;
  let mut render_height = svg_height * RESOLUTION_SCALE;

  // The scatter layout uses fixed margins — `margin(10*s)` on every side plus
  // the left/bottom label areas — unless `ImagePadding` states them outright.
  let sf = RESOLUTION_SCALE as f64;
  // A `PlotLabel` and a y `AxesLabel` stack above the plot area, and an x
  // `AxesLabel` runs past the right end of its axis — each needs its room
  // reserved, as in the line renderer.
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  let axes_label_x = opts
    .axes_label
    .as_ref()
    .map(|(x, _)| x.as_str())
    .filter(|x| !x.is_empty() && opts.axes.0);
  let has_y_axes_label = opts
    .axes_label
    .as_ref()
    .is_some_and(|(_, y)| !y.is_empty() && opts.axes.1);
  let margin_top = 10.0 * sf
    + if has_plot_label { 25.0 * sf } else { 0.0 }
    + plot_label_extra_lines(opts.plot_label.as_ref()) as f64 * 20.0 * sf
    + if has_y_axes_label { 20.0 * sf } else { 0.0 };
  let margin_right = 10.0 * sf
    + axes_label_x.map_or(0.0, |label| {
      8.0 * sf
        + crate::functions::graphics::box_string_visible_len(label) as f64
          * 0.62
          * sf
          * 14.0
    });
  let (margin_left, margin_bottom) = (10.0 * sf, 10.0 * sf);
  let (y_label_area, x_label_area) = (65.0 * sf, 40.0 * sf);

  // AspectRatio sizes the plotting area (the data frame), not the whole image.
  // Derive the total height so the frame has the requested height/width ratio.
  // With `ImagePadding` the padding replaces the margins, so the area spans
  // the width the padding leaves and the height follows from the ratio.
  if let Some(ar) = opts.aspect_ratio {
    let (plot_w, extra_h) = match opts.image_padding {
      Some([pad_left, pad_right, pad_bottom, pad_top]) => (
        render_width as f64 - (pad_left + pad_right) * sf,
        (pad_bottom + pad_top) * sf,
      ),
      None => (
        render_width as f64 - margin_left - margin_right - y_label_area,
        margin_top + margin_bottom + x_label_area,
      ),
    };
    if plot_w > 0.0 && ar > 0.0 {
      let target_render_h = plot_w * ar + extra_h;
      svg_height = ((target_render_h / sf).round() as u32).max(1);
      render_height = svg_height * RESOLUTION_SCALE;
    }
  }

  let (
    margin_top,
    margin_right,
    margin_left,
    margin_bottom,
    y_label_area,
    x_label_area,
  ) = match opts.image_padding {
    Some(padding) => {
      let m = padded_margins(
        padding,
        (render_width, render_height),
        opts.aspect_ratio,
        sf,
      );
      (
        m.top as f64,
        m.right as f64,
        0.0,
        0.0,
        m.left as f64,
        m.bottom as f64,
      )
    }
    None => (
      margin_top,
      margin_right,
      margin_left,
      margin_bottom,
      y_label_area,
      x_label_area,
    ),
  };

  let (bg_color, dark_gray, light_gray, label_fill, title_default_fill) =
    plot_theme();

  // Dashed grid lines can't be drawn through plotters, so they are collected
  // here and emitted as `stroke-dasharray` polylines once the chart is done.
  let mut dashed_overlays: Vec<DashedOverlay> = Vec::new();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = MINOR_TICK_LEN * s;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(margin_top.round() as u32)
      .margin_right(margin_right.round() as u32)
      .margin_bottom(margin_bottom.round() as u32)
      .margin_left(margin_left.round() as u32)
      .x_label_area_size(x_label_area.round() as u32)
      .y_label_area_size(y_label_area.round() as u32)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let x_major = nice_step(x_max - x_min, AXIS_TICK_TARGET);
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
    let x_minor_step = x_major / 5.0;
    let y_minor_step = y_major / 5.0;
    // `Axes -> False` hides the axis line, its ticks and its labels — as in
    // the line renderer, a framed plot still labels its frame edges.
    let (show_x_axis, show_y_axis) = opts.axes;
    let (tick_axis_x, tick_axis_y) = if opts.frame {
      (true, true)
    } else {
      (show_x_axis, show_y_axis)
    };
    // An axis given explicit `Ticks` draws them itself, after the chart.
    let x_tick_count = if opts.ticks_x.is_some() || !tick_axis_x {
      0
    } else {
      ((x_max - x_min) / x_minor_step).round() as usize + 1
    };
    let y_tick_count = if opts.ticks_y.is_some() || !tick_axis_y {
      0
    } else {
      ((y_max - y_min) / y_minor_step).round() as usize + 1
    };
    let axis_style = if opts.frame || show_x_axis || show_y_axis {
      dark_gray.stroke_width(RESOLUTION_SCALE)
    } else {
      ShapeStyle::from(&bg_color).stroke_width(0)
    };

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_tick_count)
      .y_labels(y_tick_count)
      .x_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, x_major) {
          format_tick_with_step(*v, x_major)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick_with_step(*v, y_major)
        } else {
          String::new()
        }
      })
      .axis_style(axis_style)
      .label_style(
        ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
          .into_font()
          .color(&dark_gray),
      )
      .set_tick_mark_size(
        LabelAreaPosition::Left,
        if tick_axis_y { tick } else { 0 },
      )
      .set_tick_mark_size(
        LabelAreaPosition::Bottom,
        if tick_axis_x { tick } else { 0 },
      )
      .draw()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Grid lines, drawn before the points so they sit behind the data.
    // Same rules as the line renderer: explicit positions beat the evenly
    // spaced automatic ones, dashed lines are deferred to stroke-dasharray
    // overlays emitted at the end.
    {
      let grid_color = RGBColor(0x66, 0x66, 0x66).mix(0.5);
      let y_grid = resolve_grid_lines(
        &opts.grid_y_lines,
        opts.grid_lines_y,
        y_min,
        y_max,
        y_major,
        opts.grid_lines_style.as_ref(),
      );
      let y_tol = (y_max - y_min).abs() * 1e-9;
      for (gy, style) in &y_grid {
        if *gy < y_min - y_tol || *gy > y_max + y_tol {
          continue;
        }
        let (rgb, stroke_w, dash) = grid_line_props(style);
        if let Some(dashes) = dash {
          dashed_overlays.push(DashedOverlay {
            color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
            stroke_w,
            dashes,
            points: vec![(x_min, *gy), (x_max, *gy)],
          });
          continue;
        }
        let shape = match rgb {
          Some((r, g, b)) => RGBColor(r, g, b).stroke_width(stroke_w),
          None => grid_color.stroke_width(stroke_w),
        };
        chart
          .draw_series(std::iter::once(PathElement::new(
            vec![(x_min, *gy), (x_max, *gy)],
            shape,
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      }

      let x_grid = resolve_grid_lines(
        &opts.grid_x_lines,
        opts.grid_lines_x,
        x_min,
        x_max,
        x_major,
        opts.grid_lines_style.as_ref(),
      );
      let x_tol = (x_max - x_min).abs() * 1e-9;
      for (gx, style) in &x_grid {
        if *gx < x_min - x_tol || *gx > x_max + x_tol {
          continue;
        }
        let (rgb, stroke_w, dash) = grid_line_props(style);
        if let Some(dashes) = dash {
          dashed_overlays.push(DashedOverlay {
            color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
            stroke_w,
            dashes,
            points: vec![(*gx, y_min), (*gx, y_max)],
          });
          continue;
        }
        let shape = match rgb {
          Some((r, g, b)) => RGBColor(r, g, b).stroke_width(stroke_w),
          None => grid_color.stroke_width(stroke_w),
        };
        chart
          .draw_series(std::iter::once(PathElement::new(
            vec![(*gx, y_min), (*gx, y_max)],
            shape,
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      }
    }

    // Origin lines — the axes drawn through zero. `Axes -> False` drops
    // them along with the axis ticks and labels.
    let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
    if show_x_axis && y_min < 0.0 && y_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x_min, 0.0), (x_max, 0.0)],
          origin_line,
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }
    if show_y_axis && x_min < 0.0 && x_max > 0.0 {
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

      // Draw stem lines from each point to the fill reference: a constant
      // level for Axis/Bottom/Top/value, or — for `Filling -> {i -> {j}}` —
      // the other series, linearly interpolated at this point's x so
      // irregularly spaced datasets fill correctly.
      let stem_style =
        fill_paint(series_filling_style(opts, series_idx), (r, g, b))
          .stroke_width(RESOLUTION_SCALE);
      let stem_targets: Vec<((f64, f64), f64)> =
        match series_fill_target(opts, series_idx) {
          FillTarget::Level(level) => level
            .reference_y(y_min, y_max)
            .map(|ref_y| finite_pts.iter().map(|&p| (p, ref_y)).collect())
            .unwrap_or_default(),
          FillTarget::Series(target_idx) => {
            if target_idx != series_idx
              && let Some(target) = all_series.get(target_idx)
            {
              finite_pts
                .iter()
                .filter_map(|&(x, y)| {
                  interp_polyline_y(target, x).map(|ty| ((x, y), ty))
                })
                .collect()
            } else {
              Vec::new()
            }
          }
        };
      for ((x, y), ref_y) in stem_targets {
        chart
          .draw_series(std::iter::once(PathElement::new(
            vec![(x, y), (x, ref_y)],
            stem_style,
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Plot: {e}"))
          })?;
      }

      // Uncertainty intervals from Around data values, under the point
      // markers: capped error bars by default, one filled band per series
      // for `IntervalMarkers -> "Bands"`.
      if let Some(bars) = opts.error_bars.get(series_idx) {
        match opts.interval_markers {
          IntervalMarkers::Fences => draw_error_bars(
            &mut chart,
            points,
            bars,
            color,
            x_max - x_min,
            y_max - y_min,
          )?,
          IntervalMarkers::Bands => {
            draw_interval_band(&mut chart, points, bars, color)?;
          }
          IntervalMarkers::None => {}
        }
      }

      // `PlotMarkers` replaces the round dot with its glyph, drawn
      // centred on the point at the size the marker spec asks for.
      match series_marker(&opts.plot_markers, series_idx) {
        Some(marker) => {
          let (mr, mg, mb) = marker.color.unwrap_or((r, g, b));
          let style = ("sans-serif", marker.size * sf)
            .into_font()
            .color(&RGBColor(mr, mg, mb))
            .pos(plotters::style::text_anchor::Pos::new(
              plotters::style::text_anchor::HPos::Center,
              plotters::style::text_anchor::VPos::Center,
            ));
          chart
            .draw_series(finite_pts.iter().map(|&(x, y)| {
              plotters::element::Text::new(
                marker.glyph.clone(),
                (x, y),
                style.clone(),
              )
            }))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("Plot: {e}"))
            })?;
        }
        None => {
          chart
            .draw_series(
              finite_pts.iter().map(|&(x, y)| {
                Circle::new((x, y), marker_size, color.filled())
              }),
            )
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
    full_width,
  );

  // Extend labeled (major) ticks beyond the minor ticks drawn by plotters.
  let plot_x0 = margin_left + y_label_area;
  let plot_y0 = margin_top;
  let plot_w = render_width as f64 - margin_left - margin_right - y_label_area;
  let plot_h = render_height as f64 - margin_top - margin_bottom - x_label_area;

  // Deferred dashed grid lines, as single stroke-dasharray polylines.
  if !dashed_overlays.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    let overlay_svg = render_dash_overlays(
      &dashed_overlays,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      render_width as f64,
      x_min,
      x_max,
      y_min,
      y_max,
    );
    buf.insert_str(pos, &overlay_svg);
  }

  // Epilog primitives sit over the points, as in the line renderer.
  inject_epilog(
    &mut buf,
    opts,
    (plot_x0, plot_y0, plot_w, plot_h),
    (x_min, x_max, y_min, y_max),
    sf,
  );
  {
    // As in the line renderer: an axis carrying explicit `Ticks` marks only
    // the positions it names, so its automatic majors get no extension.
    let x_major = nice_step(x_max - x_min, AXIS_TICK_TARGET);
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
    inject_major_tick_extensions(
      &mut buf,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      opts.ticks_x.is_none().then_some((x_min, x_max, x_major)),
      opts.ticks_y.is_none().then_some((y_min, y_max, y_major)),
      MINOR_TICK_LEN as f64 * sf,
      MAJOR_TICK_LEN as f64 * sf,
      sf,
      label_fill,
    );
  }

  // Callout/Labeled series labels (same plot area as the tick pass above).
  {
    inject_callout_labels(
      &mut buf,
      opts,
      all_series,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, plot_y0, plot_w, plot_h),
    );
    inject_point_labels(
      &mut buf,
      opts,
      all_series,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, plot_y0, plot_w, plot_h),
    );
  }

  inject_legend(&mut buf, opts);

  // FrameLabel / AxesLabel / PlotLabel, placed exactly as the line renderer
  // places them.
  if let Some(pos) = buf.rfind("</svg>") {
    let labels = plot_labels_svg(
      opts,
      (plot_x0, plot_y0, plot_w, plot_h),
      (x_min, x_max, y_min, y_max),
      margin_left,
      sf,
      label_fill,
      title_default_fill,
    );
    if !labels.is_empty() {
      buf.insert_str(pos, &labels);
    }
  }

  // Explicit `Ticks`: mark and label exactly the positions asked for,
  // in the same place the automatic ones would sit.
  if opts.ticks_x.is_some() || opts.ticks_y.is_some() {
    let ticks_svg = explicit_ticks_svg(
      opts,
      (plot_x0, plot_y0, plot_w, plot_h),
      (x_min, x_max, y_min, y_max),
      RESOLUTION_SCALE as f64,
      label_fill,
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &ticks_svg);
    }
  }

  Ok(buf)
}

/// Render a merged `PlotSource` (from `Show`).
///
/// `Show` keeps the options of the graphics it was given, so the merged
/// graphic goes back through the ordinary plot renderers rather than a
/// second implementation that would drift from them: line series render as
/// a line plot, scatter series as points (as the plot itself, when every
/// series is one, otherwise drawn over the lines as epilog primitives).
pub(crate) fn render_merged_plot_source(
  source: &crate::syntax::PlotSource,
) -> Result<String, InterpreterError> {
  let mut opts = PlotOptions {
    svg_width: source.image_size.0,
    svg_height: source.image_size.1,
    ..PlotOptions::default()
  };
  let mut overrides = PlotRangeOverrides::default();
  for opt in &source.options {
    if let Some((name, value)) =
      crate::functions::graphics::option_name_value(opt)
    {
      apply_common_plot_option(name, &value, &mut opts, &mut overrides);
    }
  }
  if let Some(ar) = overrides.aspect_ratio {
    opts.aspect_ratio = Some(ar);
  }
  let x_range = overrides.x.unwrap_or(source.x_range);
  let y_range = overrides.y.unwrap_or(source.y_range);

  let all_scatter =
    !source.series.is_empty() && source.series.iter().all(|s| s.is_scatter);
  // The series that make up the chart itself; scatter series overlaid on
  // lines are drawn as epilog primitives instead.
  let drawn: Vec<&crate::syntax::PlotSeriesData> = source
    .series
    .iter()
    .filter(|s| s.is_scatter == all_scatter)
    .collect();

  // Each series carries the colour its own plot resolved, so `PlotStyle`
  // is rebuilt from the series rather than re-read from the options.
  opts.plot_style = drawn
    .iter()
    .map(|s| SeriesStyle {
      color: Some(WoxiColor::new(
        s.color.0 as f64 / 255.0,
        s.color.1 as f64 / 255.0,
        s.color.2 as f64 / 255.0,
      )),
      thickness: s.thickness,
      ..SeriesStyle::default()
    })
    .collect();
  // Markers travel with the series too, so a `Show` of a marked scatter
  // plot keeps drawing the glyphs its `PlotMarkers` asked for.
  opts.plot_markers = drawn.iter().map(|s| s.marker.clone()).collect();
  if opts.plot_markers.iter().all(Option::is_none) {
    opts.plot_markers.clear();
  }
  opts.filling_rules = drawn
    .iter()
    .enumerate()
    .filter_map(|(i, s)| {
      series_filling_to_filling(s.filling).map(|f| (i, FillTarget::Level(f)))
    })
    .collect();
  // Each merged plot keeps the fill it asked for: three normal curves
  // filled to the axis in three colours stay three colours, rather than all
  // taking the first plot's.
  opts.filling_styles = drawn
    .iter()
    .map(|s| {
      (s.fill_color.is_some() || s.fill_opacity.is_some()).then_some(
        FillStyle {
          color: s.fill_color,
          opacity: s.fill_opacity,
        },
      )
    })
    .collect();
  if opts.filling_styles.iter().all(Option::is_none) {
    opts.filling_styles.clear();
  }
  opts.epilog.extend(
    source
      .series
      .iter()
      .filter(|s| s.is_scatter != all_scatter)
      .flat_map(|s| scatter_overlay_primitives(s, y_range)),
  );

  let points: Vec<Vec<(f64, f64)>> =
    drawn.iter().map(|s| s.points.clone()).collect();
  if all_scatter {
    generate_scatter_svg_with_options(&points, x_range, y_range, &opts)
  } else {
    generate_svg_with_options(&points, x_range, y_range, &opts)
  }
}

/// The `Filling` a merged series was drawn with, or `None` when it is unfilled.
fn series_filling_to_filling(
  filling: crate::syntax::SeriesFilling,
) -> Option<Filling> {
  match filling {
    crate::syntax::SeriesFilling::None => None,
    crate::syntax::SeriesFilling::Axis => Some(Filling::Axis),
    crate::syntax::SeriesFilling::Bottom => Some(Filling::Bottom),
    crate::syntax::SeriesFilling::Top => Some(Filling::Top),
    crate::syntax::SeriesFilling::Value(v) => Some(Filling::Value(v)),
  }
}

/// A scatter series shown on top of line series: its points (and the stems
/// of a filled scatter) as graphics primitives in data coordinates.
fn scatter_overlay_primitives(
  series: &crate::syntax::PlotSeriesData,
  y_range: (f64, f64),
) -> Vec<Expr> {
  let point =
    |(x, y): (f64, f64)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let finite: Vec<(f64, f64)> = series
    .points
    .iter()
    .copied()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .collect();
  if finite.is_empty() {
    return Vec::new();
  }
  let color = Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![
      Expr::Real(series.color.0 as f64 / 255.0),
      Expr::Real(series.color.1 as f64 / 255.0),
      Expr::Real(series.color.2 as f64 / 255.0),
    ]
    .into(),
  };
  let mut prims = vec![color];
  // Stems of a filled scatter series, drawn to the same reference level the
  // standalone scatter plot would use.
  if let Some(ref_y) = series_filling_to_filling(series.filling)
    .and_then(|f| f.reference_y(y_range.0, y_range.1))
  {
    prims.push(Expr::FunctionCall {
      name: "AbsoluteThickness".to_string(),
      args: vec![Expr::Real(1.0)].into(),
    });
    for &(x, y) in &finite {
      prims.push(Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(vec![point((x, y)), point((x, ref_y))].into())]
          .into(),
      });
    }
  }
  // A `PlotMarkers` series draws its glyph at every point instead of a dot.
  if let Some(marker) = &series.marker {
    let mut styled = vec![Expr::String(marker.glyph.clone())];
    if let Some((r, g, b)) = marker.color {
      styled.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![
          Expr::Real(r as f64 / 255.0),
          Expr::Real(g as f64 / 255.0),
          Expr::Real(b as f64 / 255.0),
        ]
        .into(),
      });
    }
    styled.push(Expr::Real(marker.size));
    let content = Expr::FunctionCall {
      name: "Style".to_string(),
      args: styled.into(),
    };
    for &(x, y) in &finite {
      prims.push(Expr::FunctionCall {
        name: "Text".to_string(),
        args: vec![content.clone(), point((x, y))].into(),
      });
    }
    return prims;
  }
  prims.push(Expr::FunctionCall {
    name: "AbsolutePointSize".to_string(),
    args: vec![Expr::Real(6.0)].into(),
  });
  prims.push(Expr::FunctionCall {
    name: "Point".to_string(),
    args: vec![Expr::List(
      finite.into_iter().map(point).collect::<Vec<_>>().into(),
    )]
    .into(),
  });
  prims
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
  frame_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  bar_labels: &[String],
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let n = groups.len(); // number of groups
  let k = groups.iter().map(std::vec::Vec::len).max().unwrap_or(1); // max bars per group

  // y-axis range: explicit PlotRange overrides the auto-computed extent
  // (which adds 10% headroom above the tallest bar and anchors at 0).
  let (y_min, y_max) = if let Some((ymin, ymax)) = plot_range_y {
    (ymin, ymax)
  } else {
    let y_max_auto = groups
      .iter()
      .flat_map(|g| g.iter())
      .copied()
      .fold(f64::NEG_INFINITY, f64::max)
      .max(0.0)
      * 1.1;
    let y_max_auto = if y_max_auto <= 0.0 { 1.0 } else { y_max_auto };
    (0.0, y_max_auto)
  };
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };

  // x-axis range: bars are categorical, living at 0..n. An explicit
  // PlotRange -> {{xmin, xmax}, ...} extends the drawn axis (bars stay
  // at their slots and the excess becomes empty padding), mirroring
  // the way ListLinePlot treats an x-range wider than the data.
  let (x_min, x_max) = plot_range_x.unwrap_or((0.0, n as f64));
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };

  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  // Extra space for labels
  let has_chart_labels = !chart_labels.is_empty();
  // An `AxesLabel` sits at the far end of its axis, so it takes room to the
  // right (x) and above (y), not below and to the left.
  let axes_label_x = axes_label.map(|(x, _)| x).filter(|x| !x.is_empty());
  let axes_label_y = axes_label.map(|(_, y)| y).filter(|y| !y.is_empty());
  let frame_label_x = frame_label.map(|(x, _)| x).filter(|x| !x.is_empty());
  let frame_label_y = frame_label.map(|(_, y)| y).filter(|y| !y.is_empty());
  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());

  let has_value_labels = bar_labels.iter().any(|s| !s.is_empty());
  let top_margin = if has_plot_label { 35 * s } else { 10 * s }
    + plot_label_extra_lines(plot_label) as i32 * 20 * s
    + if axes_label_y.is_some() { 20 * s } else { 0 };
  let has_rotated_labels = chart_labels.iter().any(|l| l.rotation.abs() > 0.01);
  // Only content drawn below the axis consumes bottom margin. Chart labels do
  // so only when positioned Below; Center/Above labels sit on the bars and
  // need no reservation, so the plot fills the canvas instead of leaving a
  // large empty strip.
  let labels_below =
    has_chart_labels && matches!(chart_label_position, LabelPosition::Below);
  let mut bottom_extra = 0.0_f64;
  if labels_below {
    bottom_extra += if has_rotated_labels {
      50.0 * sf
    } else {
      24.0 * sf
    };
  }
  if has_value_labels {
    bottom_extra += 30.0 * sf;
  }
  if frame_label_x.is_some() {
    bottom_extra += 24.0 * sf;
  }
  let x_label_area = 12 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 65 * RESOLUTION_SCALE;

  let (bg_color, dark_gray, _light_gray, label_fill, title_default_fill) =
    plot_theme();

  // Reserve extra right margin for chart legends
  let axes_label_margin_right = axes_label_x.map_or(0.0, |label| {
    // The label is drawn at `font_size`; ~0.62 em per character covers the
    // proportional font with a little slack.
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(label) as f64
        * 0.62
        * sf
        * 18.0
  }) as u32;
  let legend_margin_right = if chart_legends.is_empty() {
    10 * s as u32
  } else {
    let max_label_len = chart_legends
      .iter()
      .map(std::string::String::len)
      .max()
      .unwrap_or(0);
    // swatch width + gap + estimated text width + padding
    (sf * 12.0 + sf * 6.0 + max_label_len as f64 * sf * 10.0 + sf * 16.0) as u32
  };
  let legend_margin_right = legend_margin_right + axes_label_margin_right;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;

    let tick = MINOR_TICK_LEN * s;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(legend_margin_right)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BarChart: {e}"))
      })?;

    let y_span = y_max - y_min;
    let y_major = nice_step(y_span, AXIS_TICK_TARGET);
    let y_minor_step = y_major / 5.0;
    let y_tick_count = (y_span / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(0) // no x ticks for bar chart
      .y_labels(y_tick_count)
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick_with_step(*v, y_major)
        } else {
          String::new()
        }
      })
      .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
      // Tick labels are typeset smaller than in-chart labels, matching
      // wolframscript's proportionally small default axis ticks.
      .label_style(("sans-serif", sf * 13.0).into_font().color(&dark_gray))
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
        } else if !chart_legends.is_empty() {
          // Flat with legends: distinct color per group
          PLOT_COLORS[gi % PLOT_COLORS.len()]
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

  add_bar_borders(&mut buf, RESOLUTION_SCALE);

  // Inject hover tooltips into bar rects
  let bar_values: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().copied()).collect();
  inject_bar_tooltips(&mut buf, &bar_values);

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
  let margin_right = legend_margin_right as f64;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = margin_top;
  let plot_w =
    render_width as f64 - margin_left - margin_right - y_label_area as f64;
  let plot_h =
    render_height as f64 - margin_top - 10.0 * sf - x_label_area as f64;
  let axis_y = plot_y0 + plot_h;

  // Extend labeled (major) y ticks. BarChart has no x ticks.
  inject_major_tick_extensions(
    &mut buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    None,
    Some((y_min, y_max, nice_step(y_max - y_min, AXIS_TICK_TARGET))),
    MINOR_TICK_LEN as f64 * sf,
    MAJOR_TICK_LEN as f64 * sf,
    sf,
    label_fill,
  );

  let font_size = sf * 18.0;
  let title_font_size = sf * 22.0;

  // Insert label SVG elements before </svg>
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();

    // Bars live on the categorical x-axis 0..n, which may be a subset of the
    // displayed x-range when PlotRange extends the axis beyond the data. Map
    // slot centers through the same linear transform used by the cartesian
    // chart above so labels line up with their bars.
    let x_span = x_max - x_min;
    let map_x_val = |v: f64| -> f64 { plot_x0 + (v - x_min) / x_span * plot_w };

    // ChartLabels: position based on chart_label_position
    if has_chart_labels {
      let y_span = y_max - y_min;
      let map_y_val =
        |v: f64| -> f64 { plot_y0 + (y_max - v) / y_span * plot_h };
      for (i, label) in chart_labels.iter().enumerate().take(n) {
        let cx = map_x_val(i as f64 + 0.5);
        // For Above/Center positioning, use the max value in the group
        let group_max =
          groups[i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Mathematica Rotate is counterclockwise-positive; SVG is clockwise-positive
        let svg_rotation_deg = -label.rotation.to_degrees();
        let is_rotated = svg_rotation_deg.abs() > 0.01;
        let (ly, fill) = match chart_label_position {
          LabelPosition::Above => {
            let bar_top = map_y_val(group_max);
            (bar_top - font_size * 0.5, title_default_fill)
          }
          LabelPosition::Center => {
            // wolframscript draws centered labels in the default (dark) text
            // color, not white; `title_default_fill` is near-black in light
            // mode and light in dark mode for readability on the bars.
            let bar_top = map_y_val(group_max);
            let bar_center = f64::midpoint(bar_top, axis_y) + font_size * 0.4;
            (bar_center, title_default_fill)
          }
          LabelPosition::Below => {
            if is_rotated {
              // With text-anchor=middle and rotation, the left half of the
              // text swings upward. Offset the pivot down so the highest point
              // (pivot_y - half_width * sin(angle)) stays below the axis.
              let char_width_estimate = font_size * 0.6;
              let half_text_w =
                crate::functions::graphics::box_string_visible_len(&label.text)
                  as f64
                  * char_width_estimate
                  / 2.0;
              let sin_a = svg_rotation_deg.to_radians().sin().abs();
              let offset = half_text_w * sin_a + font_size * 0.5;
              (axis_y + offset, label_fill)
            } else {
              (axis_y + font_size * 1.5, label_fill)
            }
          }
        };
        if is_rotated {
          // `dominant-baseline=central` centers the glyphs on `y` (not the
          // baseline); with `text-anchor=middle` the text is centered on the
          // pivot in both axes, so rotating about (cx, ly) keeps it centered on
          // the bar horizontally instead of hanging to one side of the baseline.
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" dominant-baseline=\"central\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\" transform=\"rotate({svg_rotation_deg:.1},{cx:.1},{ly:.1})\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        } else {
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        }
      }
    }

    // Value labels from `LabelingFunction` (e.g. `Placed[#, Below] &`): one
    // per bar, drawn just below the bar's base at the axis. `bar_labels` is a
    // flat list across all groups/bars, aligned with the bar draw order.
    if has_value_labels {
      let gap = 0.1;
      let bar_w = (1.0 - 2.0 * gap) / k as f64;
      // Shrink the font so the widest label fits within one bar's
      // center-to-center spacing (`1.0` chart unit between groups, `bar_w`
      // between bars in a group), preventing adjacent labels from overlapping.
      let spacing_chart = if k > 1 { bar_w } else { 1.0 };
      let spacing_px = (map_x_val(spacing_chart) - map_x_val(0.0)).abs();
      let max_len = bar_labels
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| crate::functions::graphics::box_string_visible_len(s))
        .max()
        .unwrap_or(1)
        .max(1);
      let fit_font = (spacing_px * 0.9 / (max_len as f64 * 0.6))
        .min(font_size)
        .max(font_size * 0.35);
      let mut flat = 0usize;
      for (gi, group) in groups.iter().enumerate() {
        let group_x0 = gi as f64 + gap;
        for bi in 0..group.len() {
          let text = bar_labels.get(flat).cloned().unwrap_or_default();
          flat += 1;
          if text.is_empty() {
            continue;
          }
          let cx = map_x_val(group_x0 + (bi as f64 + 0.5) * bar_w);
          let ly = axis_y + fit_font * 1.3;
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{fit_font:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&text)
          ));
        }
      }
    }

    // FrameLabel: centred outside the bottom/left edge.
    if let Some(x_label) = frame_label_x {
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
        crate::functions::graphics::box_string_to_svg(x_label)
      ));
    }
    if let Some(y_label) = frame_label_y {
      let cy = plot_y0 + plot_h / 2.0;
      let lx = margin_left + font_size * 0.8;
      labels_svg.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
        crate::functions::graphics::box_string_to_svg(y_label)
      ));
    }

    // AxesLabel: at the far end of each axis, as in a `Plot`.
    labels_svg.push_str(&axes_label_svg(
      axes_label
        .map(|(x, y)| (x.to_string(), y.to_string()))
        .as_ref(),
      (plot_x0, plot_y0, plot_w, plot_h),
      (0.0, groups.len() as f64, y_min, y_max),
      (true, true),
      font_size,
      label_fill,
    ));

    // PlotLabel: centered above the chart
    if let Some(sl) = plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      // The y AxesLabel shares the top margin, so the title sits above it.
      let ty = margin_top
        - title_font_size * 0.5
        - if axes_label_y.is_some() {
          font_size * 1.2
        } else {
          0.0
        };
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
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
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    // ChartLegends: color swatch + label, positioned to the right of the plot
    if !chart_legends.is_empty() {
      let legend_font = sf * 16.0;
      let swatch_size = sf * 12.0;
      let swatch_gap = sf * 6.0;
      let legend_x = plot_x0 + plot_w + sf * 16.0;
      let legend_y_start = plot_y0 + sf * 8.0;
      let line_height = sf * 22.0;

      for (i, label) in chart_legends.iter().enumerate() {
        let (cr, cg, cb) = if chart_style.is_empty() {
          PLOT_COLORS[i % PLOT_COLORS.len()]
        } else {
          let c = &chart_style[i % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        };
        let ly = legend_y_start + i as f64 * line_height;
        // Color swatch
        labels_svg.push_str(&format!(
          "<rect x=\"{legend_x:.1}\" y=\"{ly:.1}\" width=\"{swatch_size:.0}\" height=\"{swatch_size:.0}\" \
           fill=\"rgb({cr},{cg},{cb})\"/>\n"
        ));
        // Label text
        labels_svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
           fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
          legend_x + swatch_size + swatch_gap,
          ly + swatch_size / 2.0,
          crate::functions::graphics::box_string_to_svg(label)
        ));
      }
    }

    buf.insert_str(insert_pos, &labels_svg);
  }

  Ok(buf)
}

/// Generate SVG for a horizontal BarChart (`BarOrigin -> Left`).
///
/// Categories run down the left edge (first input at the bottom, matching
/// wolframscript), bars grow rightward from zero, and the value axis with its
/// ticks sits along the bottom. `bar_labels` (from `LabelingFunction`) are
/// drawn just past each bar's end. Unlike the vertical renderer this builds
/// the SVG directly rather than via plotters, since the axes are swapped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_horizontal_bar_svg(
  groups: &[Vec<f64>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  chart_labels: &[crate::functions::chart::ChartLabel],
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  plot_range_x: Option<(f64, f64)>,
  bar_labels: &[String],
) -> std::string::String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;
  let rw = render_width as f64;
  let rh = render_height as f64;

  let n = groups.len();
  let k = groups
    .iter()
    .map(std::vec::Vec::len)
    .max()
    .unwrap_or(1)
    .max(1);

  // Value axis (horizontal): anchored at 0 with 10% headroom unless an
  // explicit PlotRange overrides it.
  let (x_min, x_max) = if let Some((a, b)) = plot_range_x {
    (a, b)
  } else {
    let m = groups
      .iter()
      .flat_map(|g| g.iter())
      .copied()
      .fold(f64::NEG_INFINITY, f64::max)
      .max(0.0)
      * 1.1;
    (0.0, if m <= 0.0 { 1.0 } else { m })
  };
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };

  let (bg, axis_gray, _light, label_fill, title_default_fill) = plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  let axis_stroke =
    format!("rgb({},{},{})", axis_gray.0, axis_gray.1, axis_gray.2);

  let font_size = sf * 14.0;
  let title_font_size = sf * 16.0;
  let char_w = font_size * 0.55;

  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());
  let has_chart_labels = chart_labels.iter().any(|l| !l.text.is_empty());
  let has_value_labels = bar_labels.iter().any(|s| !s.is_empty());
  let (x_axis_label, y_axis_label) = match axes_label {
    Some((x, y)) => (x, y),
    None => ("", ""),
  };

  // Margins. An `AxesLabel` sits at the far end of its axis, so it takes
  // room above (the category axis) and to the right (the value axis).
  let top_margin = if has_plot_label { 38.0 * sf } else { 16.0 * sf }
    + plot_label_extra_lines(plot_label) as f64 * 20.0 * sf
    + if x_axis_label.is_empty() {
      0.0
    } else {
      20.0 * sf
    };
  let bottom_area = 44.0 * sf;

  // Left area: widest category label (capped) plus a rotated y-axis label.
  let max_cat_len = chart_labels
    .iter()
    .map(|l| crate::functions::graphics::box_string_visible_len(&l.text))
    .max()
    .unwrap_or(0);
  let cat_label_area = if has_chart_labels {
    (max_cat_len as f64 * char_w + 14.0 * sf).min(rw * 0.45)
  } else {
    14.0 * sf
  };
  let left_area = cat_label_area;

  // Right area: value labels (drawn past each bar) plus any legend block.
  let max_vlabel_len = bar_labels
    .iter()
    .map(|s| crate::functions::graphics::box_string_visible_len(s))
    .max()
    .unwrap_or(0);
  let value_label_area = if has_value_labels {
    max_vlabel_len as f64 * char_w + 14.0 * sf
  } else {
    0.0
  };
  let legend_area = if chart_legends.is_empty() {
    0.0
  } else {
    let maxlen = chart_legends
      .iter()
      .map(|l| crate::functions::graphics::box_string_visible_len(l))
      .max()
      .unwrap_or(0);
    sf * 12.0 + sf * 6.0 + maxlen as f64 * sf * 10.0 + sf * 16.0
  };
  let axes_label_area = if y_axis_label.is_empty() {
    0.0
  } else {
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(y_axis_label) as f64
        * 0.62
        * font_size
  };
  let right_margin =
    14.0 * sf + value_label_area + legend_area + axes_label_area;

  let plot_x0 = left_area;
  let plot_y0 = top_margin;
  let plot_w = (rw - plot_x0 - right_margin).max(1.0);
  let plot_h = (rh - top_margin - bottom_area).max(1.0);
  let axis_bottom = plot_y0 + plot_h;

  let map_x =
    |v: f64| -> f64 { plot_x0 + (v - x_min) / (x_max - x_min) * plot_w };
  let bar_base = map_x(x_min.max(0.0));

  let mut svg = String::new();
  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {render_width} {render_height}\" \
       preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" \
       viewBox=\"0 0 {render_width} {render_height}\" \
       preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  }
  svg.push_str(&format!(
    "<rect width=\"{render_width}\" height=\"{render_height}\" fill=\"{bg_fill}\"/>\n"
  ));

  // Bars. Category index 0 sits at the bottom band.
  let band_h = plot_h / n.max(1) as f64;
  let gap_frac = 0.2;
  for (gi, group) in groups.iter().enumerate() {
    let band_bottom = axis_bottom - gi as f64 * band_h;
    let band_top = band_bottom - band_h;
    let inner_top = band_top + band_h * gap_frac / 2.0;
    let inner_h = band_h * (1.0 - gap_frac);
    let sub_h = inner_h / k as f64;
    for (bi, &val) in group.iter().enumerate() {
      let (r, g, b) = if !chart_style.is_empty() {
        let idx = if k > 1 { bi } else { gi };
        let c = &chart_style[idx % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else if k > 1 {
        PLOT_COLORS[bi % PLOT_COLORS.len()]
      } else if !chart_legends.is_empty() {
        PLOT_COLORS[gi % PLOT_COLORS.len()]
      } else {
        PLOT_COLORS[0]
      };
      let y0 = inner_top + bi as f64 * sub_h;
      let x_end = map_x(val);
      let (rx, rwid) = if x_end >= bar_base {
        (bar_base, x_end - bar_base)
      } else {
        (x_end, bar_base - x_end)
      };
      let tooltip = crate::functions::chart::format_chart_value(val);
      svg.push_str(&format!(
        "<rect x=\"{rx:.2}\" y=\"{y0:.2}\" width=\"{rwid:.2}\" height=\"{sub_h:.2}\" \
         fill=\"rgb({r},{g},{b})\"><title>{tooltip}</title></rect>\n"
      ));
    }
  }

  // Axis lines: left (categories) and bottom (values).
  let stroke_w = sf;
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.2}\" y1=\"{plot_y0:.2}\" x2=\"{plot_x0:.2}\" y2=\"{axis_bottom:.2}\" \
     stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n"
  ));
  let plot_right = plot_x0 + plot_w;
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.2}\" y1=\"{axis_bottom:.2}\" x2=\"{plot_right:.2}\" y2=\"{axis_bottom:.2}\" \
     stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n"
  ));

  // Value-axis ticks along the bottom.
  let x_major = nice_step(x_max - x_min, AXIS_TICK_TARGET);
  if x_major > 0.0 {
    let tick_len = MAJOR_TICK_LEN as f64 * sf;
    let start = (x_min / x_major).ceil() * x_major;
    let mut v = start;
    let mut guard = 0;
    while v <= x_max + x_major * 1e-6 && guard < 1000 {
      guard += 1;
      let tx = map_x(v);
      svg.push_str(&format!(
        "<line x1=\"{tx:.2}\" y1=\"{axis_bottom:.2}\" x2=\"{tx:.2}\" y2=\"{:.2}\" \
         stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n",
        axis_bottom + tick_len
      ));
      svg.push_str(&format!(
        "<text x=\"{tx:.2}\" y=\"{:.2}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
        axis_bottom + tick_len + font_size,
        html_escape(&format_tick_with_step(v, x_major))
      ));
      v += x_major;
    }
  }

  // Category labels: right-aligned just left of each band.
  if has_chart_labels {
    for (i, label) in chart_labels.iter().enumerate().take(n) {
      if label.text.is_empty() {
        continue;
      }
      let band_center = axis_bottom - (i as f64 + 0.5) * band_h;
      let lx = plot_x0 - 8.0 * sf;
      svg.push_str(&format!(
        "<text x=\"{lx:.2}\" y=\"{:.2}\" text-anchor=\"end\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
        band_center + font_size * 0.35,
        crate::functions::graphics::box_string_to_svg(&label.text)
      ));
    }
  }

  // Value labels (LabelingFunction) just past each bar end.
  if has_value_labels {
    let mut flat = 0usize;
    for (gi, group) in groups.iter().enumerate() {
      let band_bottom = axis_bottom - gi as f64 * band_h;
      let band_top = band_bottom - band_h;
      let inner_top = band_top + band_h * gap_frac / 2.0;
      let inner_h = band_h * (1.0 - gap_frac);
      let sub_h = inner_h / k as f64;
      for (bi, &val) in group.iter().enumerate() {
        let text = bar_labels.get(flat).cloned().unwrap_or_default();
        flat += 1;
        if text.is_empty() {
          continue;
        }
        let y0 = inner_top + bi as f64 * sub_h;
        let lx = map_x(val).max(bar_base) + 6.0 * sf;
        svg.push_str(&format!(
          "<text x=\"{lx:.2}\" y=\"{:.2}\" text-anchor=\"start\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
          y0 + sub_h / 2.0 + font_size * 0.35,
          crate::functions::graphics::box_string_to_svg(&text)
        ));
      }
    }
  }

  // AxesLabel: at the far end of each axis, as everywhere else. `BarOrigin ->
  // Left` transposes the chart, so the *first* entry labels the vertical axis
  // and the second the horizontal one — the swap wolframscript makes too.
  svg.push_str(&axes_label_svg(
    Some(&(y_axis_label.to_string(), x_axis_label.to_string())),
    (plot_x0, plot_y0, plot_w, plot_h),
    (x_min, x_max, 0.0, n as f64),
    (true, true),
    font_size,
    label_fill,
  ));

  // Plot label, centered above the plot.
  if let Some(sl) = plot_label
    && !sl.text.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    // Place the title near the top so a clear gap remains above the bars.
    let ty = title_font_size * 1.2;
    let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
    // A stacked title grows upwards: its last line stays where a one-line
    // title would sit, so whatever shares the top margin below it (the y
    // `AxesLabel`) is not written over.
    let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
    let fill = sl
      .color
      .as_ref()
      .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
    let mut style_attrs = String::new();
    if sl.bold {
      style_attrs.push_str(" font-weight=\"bold\"");
    }
    if sl.italic {
      style_attrs.push_str(" font-style=\"italic\"");
    }
    svg.push_str(&format!(
      "<text x=\"{cx:.2}\" y=\"{ty:.2}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{fs:.0}\" fill=\"{fill}\"{style_attrs}>{}</text>\n",
      sl.svg_scaled_stacked(sf, cx, fs * 1.2)
    ));
  }

  // Legends to the right of the plot.
  if !chart_legends.is_empty() {
    let legend_font = sf * 16.0;
    let swatch = sf * 12.0;
    let swatch_gap = sf * 6.0;
    let legend_x = plot_x0 + plot_w + value_label_area + 14.0 * sf;
    let legend_y_start = plot_y0 + sf * 8.0;
    let line_height = sf * 22.0;
    for (i, label) in chart_legends.iter().enumerate() {
      let (cr, cg, cb) = if chart_style.is_empty() {
        PLOT_COLORS[i % PLOT_COLORS.len()]
      } else {
        let c = &chart_style[i % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      };
      let ly = legend_y_start + i as f64 * line_height;
      svg.push_str(&format!(
        "<rect x=\"{legend_x:.2}\" y=\"{ly:.2}\" width=\"{swatch:.0}\" height=\"{swatch:.0}\" \
         fill=\"rgb({cr},{cg},{cb})\"/>\n"
      ));
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
         fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
        legend_x + swatch + swatch_gap,
        ly + swatch / 2.0,
        crate::functions::graphics::box_string_to_svg(label)
      ));
    }
  }

  svg.push_str("</svg>");
  svg
}

/// Wolfram's `"WL12DefaultVectorGradient"` color scheme — the gradient
/// BubbleHistogram uses to color bubbles by bin count. Four LAB control
/// points (from `ColorData["WL12DefaultVectorGradient", "BlendArgument"]`)
/// are interpolated linearly in LAB space (WL `Blend` semantics), then
/// converted D50-LAB → sRGB with out-of-gamut channels clipped. Matches
/// wolframscript's rendered colors to well within 8-bit quantization.
pub(crate) fn wl_default_vector_gradient(t: f64) -> (u8, u8, u8) {
  // LABColor stores L in [0, 1] and a/b divided by 100.
  const LAB_STOPS: [(f64, f64, f64); 4] = [
    (
      0.188_667_522_250_113_1,
      0.522_596_101_276_680_9,
      -0.857_210_761_912_256_8,
    ),
    (
      0.402_329_034_265_632_8,
      0.522_214_589_645_690_3,
      -0.346_571_396_721_582_05,
    ),
    (
      0.644_456_290_545_989,
      0.537_992_026_017_925,
      0.732_111_567_127_367_9,
    ),
    (
      0.816_796_918_973_305_3,
      0.145_001_147_401_695_45,
      0.828_253_206_343_699_9,
    ),
  ];
  let pos = t.clamp(0.0, 1.0) * (LAB_STOPS.len() - 1) as f64;
  let i = (pos.floor() as usize).min(LAB_STOPS.len() - 2);
  let f = pos - i as f64;
  let lerp = |a: f64, b: f64| a + (b - a) * f;
  let l = lerp(LAB_STOPS[i].0, LAB_STOPS[i + 1].0) * 100.0;
  let a = lerp(LAB_STOPS[i].1, LAB_STOPS[i + 1].1) * 100.0;
  let b = lerp(LAB_STOPS[i].2, LAB_STOPS[i + 1].2) * 100.0;

  // CIELAB → XYZ, D50 reference white.
  let fy = (l + 16.0) / 116.0;
  let fx = fy + a / 500.0;
  let fz = fy - b / 200.0;
  let f_inv = |c: f64| {
    let c3 = c * c * c;
    if c3 > 216.0 / 24389.0 {
      c3
    } else {
      (116.0 * c - 16.0) / (24389.0 / 27.0)
    }
  };
  let (x, y, z) = (f_inv(fx) * 0.9642, f_inv(fy), f_inv(fz) * 0.8249);

  // XYZ (D50) → linear sRGB (Bradford-adapted ICC matrix), then gamma
  // encoding; channel clipping reproduces WL's gamut handling.
  const M: [[f64; 3]; 3] = [
    [3.133_856_1, -1.616_866_7, -0.490_614_6],
    [-0.978_768_4, 1.916_141_5, 0.033_454_0],
    [0.071_945_3, -0.228_991_4, 1.405_242_7],
  ];
  let encode = |row: &[f64; 3]| -> u8 {
    let v = (row[0] * x + row[1] * y + row[2] * z).clamp(0.0, 1.0);
    let v = if v <= 0.003_130_8 {
      12.92 * v
    } else {
      1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
  };
  (encode(&M[0]), encode(&M[1]), encode(&M[2]))
}

/// Generate SVG for a BubbleChart — a scatter plot with variable-radius
/// circles drawn over labeled x/y axes. Each input triple is `(x, y, z)`
/// where `z` drives the bubble area (matching Mathematica's convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_bubble_chart_svg(
  groups: &[Vec<(f64, f64, f64)>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  chart_labels: &[crate::functions::chart::ChartLabel],
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  // Data-space bin widths `(dx, dy)` when the bubbles sit on a regular grid
  // (BubbleHistogram). Caps the bubble radius so bubbles in adjacent bins
  // can at most touch, never overlap. `None` for free-form BubbleChart data.
  bin_size: Option<(f64, f64)>,
  // Per-bubble fill colors, indexed by flat position across all groups.
  // When set, the bubbles are drawn without the black edge — matching
  // wolframscript's BubbleHistogram, whose count-gradient bubbles have no
  // border (unlike BubbleChart's).
  bubble_colors: Option<&[(u8, u8, u8)]>,
  // `(min_count, max_count)` when a count-gradient bar legend should be
  // drawn right of the plot (BubbleHistogram's `PlotLegends -> Automatic`,
  // Wolfram's `BarLegend`).
  count_legend: Option<(f64, f64)>,
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;
  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  let has_x_axis_label =
    axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());

  let top_margin = if has_plot_label { 35 * s } else { 10 * s }
    + plot_label_extra_lines(plot_label) as i32 * 20 * s;
  let bottom_extra = if has_x_axis_label { 24.0 * sf } else { 0.0 };
  let x_label_area = 40 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 65 * RESOLUTION_SCALE;

  let (bg_color, dark_gray, light_gray, label_fill, title_default_fill) =
    plot_theme();

  let legend_margin_right = if chart_legends.is_empty() {
    10 * s as u32
  } else {
    let max_label_len =
      chart_legends.iter().map(std::string::String::len).max().unwrap_or(0);
    (sf * 12.0 + sf * 6.0 + max_label_len as f64 * sf * 10.0 + sf * 16.0) as u32
  }
  // The count-gradient bar legend occupies a 53-display-pixel strip right
  // of the plot — the widening wolframscript applies (360 → 413 pt).
  + if count_legend.is_some() { 53 * s as u32 } else { 0 };

  // Plot-area dimensions in render-space pixels — must stay in sync with the
  // `ChartBuilder` margins below so that pixel-aware range padding (computed
  // next) keeps bubbles inside the frame.
  let plot_w_px = (render_width as f64
    - 10.0 * sf
    - legend_margin_right as f64
    - y_label_area as f64)
    .max(1.0);
  let plot_h_px = (render_height as f64
    - top_margin as f64
    - 10.0 * sf
    - x_label_area as f64)
    .max(1.0);

  // Auto-compute x/y ranges. Raw min/max first, then expand so the largest
  // possible bubble (`max_bubble_radius` pixels, plus a small visual gap)
  // fits between every data point and the frame. Only applied when the user
  // hasn't supplied an explicit PlotRange — an explicit range is respected
  // verbatim.
  let raw_range = |vals: &[f64]| -> (f64, f64) {
    let mn = vals
      .iter()
      .copied()
      .filter(|v| v.is_finite())
      .fold(f64::INFINITY, f64::min);
    let mx = vals
      .iter()
      .copied()
      .filter(|v| v.is_finite())
      .fold(f64::NEG_INFINITY, f64::max);
    if !mn.is_finite() || !mx.is_finite() {
      return (0.0, 1.0);
    }
    if (mx - mn).abs() < f64::EPSILON {
      // Single distinct value: start with a unit-wide window so the
      // pixel-pad step below has a well-defined span to expand.
      return (mn - 0.5, mx + 0.5);
    }
    (mn, mx)
  };

  // Expand a raw [min, max] so that `pad_px` render-pixels of padding are
  // reserved on each side of the plot area. Solved for the final span with
  // `span_final = span_raw * plot_px / (plot_px - 2 * pad_px)` — after the
  // expansion, data extremes sit exactly `pad_px` pixels from the frame.
  let pad_for_bubbles =
    |min_0: f64, max_0: f64, plot_px: f64, pad_px: f64| -> (f64, f64) {
      let span_0 = max_0 - min_0;
      // Guard: if the plot is too narrow to even fit the padding, skip the
      // expansion rather than blow up or invert the range.
      if span_0 <= 0.0 || plot_px <= 2.0 * pad_px {
        return (min_0, max_0);
      }
      let extra = pad_px * span_0 / (plot_px - 2.0 * pad_px);
      (min_0 - extra, max_0 + extra)
    };

  let xs: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().map(|t| t.0)).collect();
  let ys: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().map(|t| t.1)).collect();
  let (x_min_raw, x_max_raw) = raw_range(&xs);
  let (y_min_raw, y_max_raw) = raw_range(&ys);

  // Max bubble radius in render-space pixels. 20 display pixels matches the
  // previous (axis-less) implementation and stays readable without occluding
  // neighbors at the default image size. For binned data (`bin_size`), the
  // radius is additionally capped at half the pixel distance between
  // adjacent bin centers so bubbles never overlap.
  let max_bubble_radius = {
    let default_r = 20.0 * sf;
    // Half the border stroke width, so even the outlines of two touching
    // max-size bubbles stay disjoint.
    let margin = 0.5 * sf;
    let cap_axis = |bin: f64,
                    raw_span: f64,
                    plot_px: f64,
                    explicit_span: Option<f64>|
     -> f64 {
      match explicit_span {
        // Explicit PlotRange: the data-to-pixel scale is fixed, adjacent
        // bin centers are `bin / span * plot_px` pixels apart.
        Some(span) => bin * plot_px / span.max(f64::EPSILON) / 2.0 - margin,
        // Auto range: the span is later padded by `radius + 4 px` per side
        // (see `pad_for_bubbles`), so the bin's final pixel size is
        // `bin * (plot_px - 2 * (r + 4 sf)) / raw_span`. Solving
        // `2 * (r + margin) <= that` for `r`:
        None => {
          (bin * (plot_px - 8.0 * sf) - 2.0 * margin * raw_span)
            / (2.0 * (raw_span + bin))
        }
      }
    };
    match bin_size {
      None => default_r,
      Some((bx, by)) => {
        let span_of = |range: Option<(f64, f64)>| {
          range.map(|(lo, hi)| if hi <= lo { 1.0 } else { hi - lo })
        };
        let cap_x =
          cap_axis(bx, x_max_raw - x_min_raw, plot_w_px, span_of(plot_range_x));
        let cap_y =
          cap_axis(by, y_max_raw - y_min_raw, plot_h_px, span_of(plot_range_y));
        // Keep at least one display pixel so dense binnings stay visible.
        default_r.min(cap_x).min(cap_y).max(sf)
      }
    }
  };

  // Extra visual gap beyond the radius so bubbles don't visually kiss the
  // frame line.
  let bubble_pad_px = max_bubble_radius + 4.0 * sf;

  let (x_min_auto, x_max_auto) =
    pad_for_bubbles(x_min_raw, x_max_raw, plot_w_px, bubble_pad_px);
  let (y_min_auto, y_max_auto) =
    pad_for_bubbles(y_min_raw, y_max_raw, plot_h_px, bubble_pad_px);
  let (x_min, x_max) = plot_range_x.unwrap_or((x_min_auto, x_max_auto));
  let (y_min, y_max) = plot_range_y.unwrap_or((y_min_auto, y_max_auto));
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };

  // Max |z| across all groups — used to normalize bubble radii so that
  // bubbles are comparable between datasets.
  let z_max = groups
    .iter()
    .flat_map(|g| g.iter().map(|t| t.2.abs()))
    .fold(0.0_f64, f64::max);

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
    })?;

    let tick = MINOR_TICK_LEN * s;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(legend_margin_right)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
      })?;

    let x_major = nice_step(x_max - x_min, AXIS_TICK_TARGET);
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
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
          format_tick_with_step(*v, x_major)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick_with_step(*v, y_major)
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
        InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
      })?;

    // Origin lines — rendered only when the axis crosses zero.
    let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
    if y_min < 0.0 && y_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x_min, 0.0), (x_max, 0.0)],
          origin_line,
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
        })?;
    }
    if x_min < 0.0 && x_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(0.0, y_min), (0.0, y_max)],
          origin_line,
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
        })?;
    }

    // Draw the bubbles with pixel-space radii so they look the same
    // regardless of the data range. Colors are assigned per group so that
    // multi-dataset BubbleChart input visually distinguishes the datasets —
    // unless explicit per-bubble colors were supplied (BubbleHistogram's
    // count gradient), which take precedence.
    let mut flat_idx = 0usize;
    for (gi, group) in groups.iter().enumerate() {
      let (cr, cg, cb) = if chart_style.is_empty() {
        PLOT_COLORS[gi % PLOT_COLORS.len()]
      } else {
        let c = &chart_style[gi % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      };
      let group_fill = RGBColor(cr, cg, cb);
      // Semi-transparent black so the underlying bubble color bleeds
      // through the border, tinting it slightly toward the fill.
      let border = RGBColor(0, 0, 0).mix(0.4).stroke_width(RESOLUTION_SCALE);
      for &(x, y, z) in group {
        let fill = bubble_colors
          .and_then(|cs| cs.get(flat_idx))
          .map_or(group_fill, |&(r, g, b)| RGBColor(r, g, b));
        flat_idx += 1;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
          continue;
        }
        // Area-proportional: radius ∝ sqrt(z / z_max) * max_radius.
        let radius = if z_max > 0.0 {
          ((z.abs() / z_max).sqrt() * max_bubble_radius)
            .max((2.0 * sf).min(max_bubble_radius))
        } else {
          (5.0 * sf).min(max_bubble_radius)
        };
        chart
          .draw_series(std::iter::once(Circle::new(
            (x, y),
            radius as i32,
            fill.filled(),
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
          })?;
        if bubble_colors.is_none() {
          chart
            .draw_series(std::iter::once(Circle::new(
              (x, y),
              radius as i32,
              border,
            )))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
            })?;
        }
      }
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
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

  // Plot area coordinates (must match the margins/areas above).
  let margin_left = 10.0 * sf;
  let margin_top = top_margin as f64;
  let margin_right = legend_margin_right as f64;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = margin_top;
  let plot_w =
    render_width as f64 - margin_left - margin_right - y_label_area as f64;
  let plot_h =
    render_height as f64 - margin_top - 10.0 * sf - x_label_area as f64;
  let axis_y = plot_y0 + plot_h;

  let x_major_step = nice_step(x_max - x_min, AXIS_TICK_TARGET);
  let y_major_step = nice_step(y_max - y_min, AXIS_TICK_TARGET);

  inject_major_tick_extensions(
    &mut buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    Some((x_min, x_max, x_major_step)),
    Some((y_min, y_max, y_major_step)),
    MINOR_TICK_LEN as f64 * sf,
    MAJOR_TICK_LEN as f64 * sf,
    sf,
    label_fill,
  );

  // BubbleChart closes the plot rectangle with a top and right axis line
  // (each carrying minor + major tick marks) so the plot area reads as a
  // full frame. Matches Mathematica's default BubbleChart framing.
  inject_top_right_frame(
    &mut buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    Some((x_min, x_max, x_major_step)),
    Some((y_min, y_max, y_major_step)),
    MINOR_TICK_LEN as f64 * sf,
    MAJOR_TICK_LEN as f64 * sf,
    sf,
    label_fill,
  );

  let font_size = sf * 18.0;
  let title_font_size = sf * 22.0;

  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();

    if let Some((x_label, y_label)) = &axes_label {
      if !x_label.is_empty() {
        let cx = plot_x0 + plot_w / 2.0;
        let base_y = axis_y + font_size * 2.8;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(y_label)
        ));
      }
    }

    if let Some(sl) = plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      let ty = margin_top - title_font_size * 0.5;
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
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
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    if !chart_legends.is_empty() {
      let legend_font = sf * 16.0;
      let swatch_size = sf * 12.0;
      let swatch_gap = sf * 6.0;
      let legend_x = plot_x0 + plot_w + sf * 16.0;
      let legend_y_start = plot_y0 + sf * 8.0;
      let line_height = sf * 22.0;

      for (i, label) in chart_legends.iter().enumerate() {
        let (cr, cg, cb) = if chart_style.is_empty() {
          PLOT_COLORS[i % PLOT_COLORS.len()]
        } else {
          let c = &chart_style[i % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        };
        let ly = legend_y_start + i as f64 * line_height;
        // Bubbles are drawn fully opaque with a black border — mirror that
        // on the swatch so the legend color visually matches the bubbles.
        labels_svg.push_str(&format!(
          "<rect x=\"{legend_x:.1}\" y=\"{ly:.1}\" width=\"{swatch_size:.0}\" height=\"{swatch_size:.0}\" \
           fill=\"rgb({cr},{cg},{cb})\" stroke=\"#000000\" stroke-width=\"1\"/>\n"
        ));
        labels_svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
           fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
          legend_x + swatch_size + swatch_gap,
          ly + swatch_size / 2.0,
          crate::functions::graphics::box_string_to_svg(label)
        ));
      }
    }

    // Count-gradient bar legend (BubbleHistogram's PlotLegends ->
    // Automatic): a vertical WL12DefaultVectorGradient bar right of the
    // plot, smallest count at the bottom, labels at nice count steps —
    // Wolfram's BarLegend. Layout proportions match the wolframscript
    // rendering (bar ≈ 13 pt wide, ≈ 5/8 of the plot height, vertically
    // centered).
    if let Some((c_lo, c_hi)) = count_legend {
      let bar_w = 13.0 * sf;
      let bar_h = plot_h * 0.625;
      let bar_x = plot_x0 + plot_w + 27.0 * sf;
      let bar_y = plot_y0 + (plot_h - bar_h) / 2.0;
      let slices = 64usize;
      let slice_h = bar_h / slices as f64;
      for i in 0..slices {
        // t = 1 at the top slice, 0 at the bottom.
        let t = 1.0 - (i as f64 + 0.5) / slices as f64;
        let (r, g, b) = wl_default_vector_gradient(t);
        labels_svg.push_str(&format!(
          "<rect x=\"{bar_x:.1}\" y=\"{:.1}\" width=\"{bar_w:.1}\" \
           height=\"{:.1}\" fill=\"rgb({r},{g},{b})\"/>\n",
          bar_y + slice_h * i as f64,
          slice_h + 0.6 * sf
        ));
      }
      labels_svg.push_str(&format!(
        "<rect x=\"{bar_x:.1}\" y=\"{bar_y:.1}\" width=\"{bar_w:.1}\" \
         height=\"{bar_h:.1}\" fill=\"none\" stroke=\"rgb(110,110,110)\" \
         stroke-width=\"{:.1}\"/>\n",
        0.8 * sf
      ));
      let legend_font = sf * 16.0;
      let label_x = bar_x + bar_w + 6.0 * sf;
      let label_at = |labels_svg: &mut String, value: f64, frac: f64| {
        labels_svg.push_str(&format!(
          "<text x=\"{label_x:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
           font-size=\"{legend_font:.0}\" fill=\"{label_fill}\" \
           dominant-baseline=\"central\">{}</text>\n",
          bar_y + bar_h * (1.0 - frac),
          format_tick(value)
        ));
      };
      if c_hi > c_lo {
        // Six target labels reproduces wolframscript's BarLegend ticks
        // (step 2 for counts 1..12, step 5 for 1..25).
        let step = nice_step(c_hi - c_lo, 6);
        let mut v = (c_lo / step).ceil() * step;
        while v <= c_hi + 1e-9 {
          label_at(&mut labels_svg, v, (v - c_lo) / (c_hi - c_lo));
          v += step;
        }
      } else {
        // Degenerate range (all bins hold the same count): label the bar
        // once at its center.
        label_at(&mut labels_svg, c_lo, 0.5);
      }
    }

    // ChartLabels — draw each label centered on its bubble. Labels map to
    // points in input order, flattened across datasets.
    if !chart_labels.is_empty() {
      let bubble_label_font = sf * 16.0;
      let x_span = x_max - x_min;
      let y_span = y_max - y_min;
      let mut idx = 0usize;
      'outer: for group in groups {
        for &(x, y, _z) in group {
          if idx >= chart_labels.len() {
            break 'outer;
          }
          let label = &chart_labels[idx];
          idx += 1;
          if !x.is_finite() || !y.is_finite() || label.text.is_empty() {
            continue;
          }
          let px = plot_x0 + (x - x_min) / x_span * plot_w;
          let py = plot_y0 + (y_max - y) / y_span * plot_h;
          let transform = if label.rotation.abs() > f64::EPSILON {
            // ChartLabel rotation is in radians; SVG rotates clockwise for
            // positive degrees, so negate to match Wolfram's convention
            // (positive radians = counter-clockwise).
            let deg = -label.rotation.to_degrees();
            format!(" transform=\"rotate({deg:.2},{px:.1},{py:.1})\"")
          } else {
            String::new()
          };
          // Bubble labels use full black (not the axis label gray) so
          // they stay legible against the colored fill.
          labels_svg.push_str(&format!(
            "<text x=\"{px:.1}\" y=\"{py:.1}\" text-anchor=\"middle\" \
             dominant-baseline=\"central\" font-family=\"sans-serif\" \
             font-size=\"{bubble_label_font:.0}\" fill=\"#000000\"\
             {transform}>{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        }
      }
    }

    buf.insert_str(insert_pos, &labels_svg);
  }

  Ok(buf)
}

/// Escape special characters for SVG/HTML text content.
pub(crate) fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// Width (in viewBox units) of the plotting area, mirroring the margins used
/// by `generate_svg_with_options`. Used to scale dashed legend swatches so the
/// dash pattern matches the on-chart line (which dashes at `d * plot_w`).
fn legend_plot_area_width(vb_w: f64, opts: &PlotOptions) -> f64 {
  let sf = RESOLUTION_SCALE as f64;
  let show_y_axis = opts.axes.1;
  let y_label_area = if !show_y_axis {
    0.0
  } else if !opts.ticks {
    5.0 * sf
  } else {
    65.0 * sf
  };
  let margin_left = if show_y_axis { 10.0 * sf } else { 5.0 * sf };
  let has_right_label = opts
    .frame_label_right
    .as_ref()
    .is_some_and(|t| !t.is_empty());
  let margin_right = if has_right_label {
    40.0 * sf
  } else {
    10.0 * sf
  };
  (vb_w - margin_left - margin_right - y_label_area).max(vb_w * 0.5)
}

/// Inject a legend into an SVG plot. Depending on `legend_position`, the legend
/// is placed on the right (default), top, or bottom of the plot.
fn inject_legend(buf: &mut String, opts: &PlotOptions) {
  if opts.plot_legends.is_empty() {
    return;
  }

  let sf = RESOLUTION_SCALE as f64;
  let (_bg_color, _dark_gray, _light_gray, label_fill, _title_fill) =
    plot_theme();

  let font_size = sf * 14.0;
  let line_height = font_size * 1.6;
  let swatch_len = sf * 20.0;
  let swatch_gap = sf * 6.0;
  let legend_padding = sf * 10.0;

  // Parse current viewBox
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

  // The on-chart dashed lines are drawn with a dash length of `d * plot_w`
  // (a fraction of the plotting-area width — see `draw_dashed_line`). Scale the
  // legend swatch's stroke-dasharray by the same width so the two match.
  let dash_scale = legend_plot_area_width(vb_w, opts);

  match opts.legend_position {
    LegendPosition::Top | LegendPosition::Bottom => {
      // Horizontal legend: all entries in one row
      let legend_height = line_height + legend_padding;

      // Calculate per-entry widths for horizontal layout
      let entry_widths: Vec<f64> = opts
        .plot_legends
        .iter()
        .map(|s| {
          swatch_len
            + swatch_gap
            + crate::functions::graphics::box_string_visible_len(s) as f64
              * font_size
              * 0.55
        })
        .collect();
      let entry_spacing = legend_padding;

      let new_vb_h = vb_h + legend_height;

      // Update viewBox height
      let old_vb = format!("viewBox=\"0 0 {} {}\"", vb_w as u32, vb_h as u32);
      let new_vb =
        format!("viewBox=\"0 0 {} {}\"", vb_w as u32, new_vb_h as u32);
      *buf = buf.replacen(&old_vb, &new_vb, 1);

      // Update height attribute if present
      let h_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
      if let Some(hcaps) = h_re.captures(&buf.clone()) {
        let old_h: u32 = hcaps[1].parse().unwrap_or(0);
        if old_h > 0 {
          let new_display_h = (old_h as f64 * new_vb_h / vb_h).round() as u32;
          let old_hattr = format!("height=\"{old_h}\"");
          let new_hattr = format!("height=\"{new_display_h}\"");
          *buf = buf.replacen(&old_hattr, &new_hattr, 1);
        }
      }

      // For Top: shift existing content down and draw legend at top
      // For Bottom: draw legend at the bottom
      let legend_y = if opts.legend_position == LegendPosition::Top {
        // Shift existing content down by wrapping in a translate group
        // Find end of opening <svg ...> tag
        if let Some(svg_tag_end) = buf.find('>') {
          let after_tag = svg_tag_end + 1;
          let shift_open =
            format!("<g transform=\"translate(0,{})\">", legend_height as u32);
          buf.insert_str(after_tag, &shift_open);
          // Insert closing </g> before </svg>
          if let Some(close_pos) = buf.rfind("</svg>") {
            buf.insert_str(close_pos, "</g>");
          }
        }
        legend_padding * 0.5 + line_height * 0.5
      } else {
        // Bottom: legend goes after existing content
        vb_h + legend_padding * 0.5 + line_height * 0.5
      };

      // Draw legend entries horizontally
      if let Some(insert_pos) = buf.rfind("</svg>") {
        let mut legend_svg = String::new();
        // Center the legend row within the viewBox width
        let total_w: f64 = entry_widths.iter().sum::<f64>()
          + entry_spacing * (entry_widths.len().max(1) - 1) as f64;
        let mut cursor_x = (vb_w - total_w).max(0.0) / 2.0;

        for (i, label) in opts.plot_legends.iter().enumerate() {
          let (r, g, b) = series_color(&opts.plot_style, i);
          let thickness = series_thickness(&opts.plot_style, i);
          let dashing = series_dashing(&opts.plot_style, i);
          let sw = (thickness as f64 / RESOLUTION_SCALE as f64 * sf).max(sf);

          let mut dash_attr = String::new();
          if let Some(ref pattern) = dashing {
            let dash_vals: Vec<String> = pattern
              .iter()
              .map(|d| format!("{:.1}", dash_len(*d, dash_scale, sf)))
              .collect();
            dash_attr =
              format!(" stroke-dasharray=\"{}\"", dash_vals.join(","));
          }

          // Swatch line
          legend_svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"rgb({},{},{})\" stroke-width=\"{}\"{}/>\n",
            cursor_x,
            legend_y,
            cursor_x + swatch_len,
            legend_y,
            r,
            g,
            b,
            sw as u32,
            dash_attr,
          ));

          // Text label
          let text_x = cursor_x + swatch_len + swatch_gap;
          let text_y = legend_y + font_size * 0.35;
          legend_svg.push_str(&format!(
            "<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(label),
          ));

          cursor_x += entry_widths[i] + entry_spacing;
        }

        buf.insert_str(insert_pos, &legend_svg);
      }
    }
    LegendPosition::Right => {
      // Original right-side legend behavior
      let max_text_width = opts
        .plot_legends
        .iter()
        .map(|s| {
          crate::functions::graphics::box_string_visible_len(s) as f64
            * font_size
            * 0.55
        })
        .fold(0.0_f64, f64::max);
      let legend_width =
        swatch_len + swatch_gap + max_text_width + legend_padding;

      let new_vb_w = vb_w + legend_width;

      // Update viewBox width
      let old_vb = format!("viewBox=\"0 0 {} {}\"", vb_w as u32, vb_h as u32);
      let new_vb =
        format!("viewBox=\"0 0 {} {}\"", new_vb_w as u32, vb_h as u32);
      *buf = buf.replacen(&old_vb, &new_vb, 1);

      // Update width attribute if present (non-full-width)
      let w_re = regex::Regex::new(r#"width="(\d+)""#).unwrap();
      if let Some(caps) = w_re.captures(&buf.clone()) {
        let old_w: u32 = caps[1].parse().unwrap_or(0);
        if old_w > 0 {
          let render_w = vb_w as u32;
          let render_new_w = new_vb_w as u32;
          let new_display_w = (old_w as f64 * render_new_w as f64
            / render_w as f64)
            .round() as u32;
          let old_attr = format!("width=\"{old_w}\"");
          let new_attr = format!("width=\"{new_display_w}\"");
          *buf = buf.replacen(&old_attr, &new_attr, 1);

          let h_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
          if let Some(hcaps) = h_re.captures(&buf.clone()) {
            let old_h: u32 = hcaps[1].parse().unwrap_or(0);
            if old_h > 0 {
              let new_display_h =
                (new_display_w as f64 * vb_h / new_vb_w).round() as u32;
              let old_hattr = format!("height=\"{old_h}\"");
              let new_hattr = format!("height=\"{new_display_h}\"");
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
          let thickness = series_thickness(&opts.plot_style, i);
          let dashing = series_dashing(&opts.plot_style, i);
          let y = legend_y0 + i as f64 * line_height + line_height * 0.5;
          let sw = (thickness as f64 / RESOLUTION_SCALE as f64 * sf).max(sf);

          let mut dash_attr = String::new();
          if let Some(ref pattern) = dashing {
            let dash_vals: Vec<String> = pattern
              .iter()
              .map(|d| format!("{:.1}", dash_len(*d, dash_scale, sf)))
              .collect();
            dash_attr =
              format!(" stroke-dasharray=\"{}\"", dash_vals.join(","));
          }
          legend_svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
             stroke=\"rgb({},{},{})\" stroke-width=\"{}\"{}/>\n",
            legend_x,
            y,
            legend_x + swatch_len,
            y,
            r,
            g,
            b,
            sw as u32,
            dash_attr,
          ));

          let text_x = legend_x + swatch_len + swatch_gap;
          let text_y = y + font_size * 0.35;
          legend_svg.push_str(&format!(
            "<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(label),
          ));
        }

        buf.insert_str(insert_pos, &legend_svg);
      }
    }
  }
}

/// Generate SVG for a histogram using plotters.
/// Specifies how histogram bins are determined.
#[derive(Clone)]
pub enum BinSpec {
  /// A fixed number of equal-width bins.
  Count(usize),
  /// A fixed bin width; bin boundaries are aligned to multiples of the width.
  Width(f64),
  /// Explicit bin edges (must be sorted, at least 2 elements).
  Edges(Vec<f64>),
}

/// Histogram bar-height specification (Wolfram's `hspec`).
#[derive(Clone, Copy, PartialEq)]
pub enum HistogramHeight {
  /// Raw counts per bin (the default).
  Count,
  /// Running total of counts up to and including each bin.
  CumulativeCount,
  /// Fraction of samples in each bin (`c / n`); heights sum to 1.
  Probability,
  /// Probability density (`c / (n * width)`); total area is 1.
  Pdf,
  /// Cumulative fraction of samples (`cumsum / n`); reaches 1 at the last bin.
  Cdf,
}

/// Compute the common bin edges for one or more datasets given a bin spec.
fn histogram_bin_edges(
  all_values: &[f64],
  bin_spec: Option<&BinSpec>,
) -> Vec<f64> {
  if let Some(BinSpec::Edges(edges)) = bin_spec {
    let mut edges = edges.clone();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    return edges;
  }

  let d_min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
  let d_max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let (d_min, d_max) = if d_min.is_finite() && d_max.is_finite() {
    (d_min, d_max)
  } else {
    (0.0, 1.0)
  };

  match bin_spec {
    Some(BinSpec::Width(w)) if *w > 0.0 => {
      let w = *w;
      // Align boundaries to multiples of the width (Wolfram convention),
      // so `{1}` yields integer-aligned unit bins `[k w, (k+1) w)`.
      let start = (d_min / w).floor() * w;
      // Top edge: the first multiple of w strictly greater than d_max, so the
      // largest sample gets its own bin (matching Wolfram — data 0..9 with
      // width 1 yields the 10 bins [0,1),…,[9,10)).
      let mut top = (d_max / w).floor() * w + w;
      if top <= d_max + w * 1e-9 {
        top += w;
      }
      let n_bins = (((top - start) / w).round() as usize).max(1);
      (0..=n_bins).map(|i| start + i as f64 * w).collect()
    }
    _ => {
      // A bare bin count uses wolframscript's `userBinningN` placement,
      // shared with `HistogramList[data, n]`.
      if let Some(BinSpec::Count(c)) = bin_spec
        && *c > 0
        && let Some(bins) =
          crate::functions::list_helpers_ast::wl_user_binning_n(
            all_values, *c as i128,
          )
      {
        return bins.f64s;
      }
      let num_bins = match bin_spec {
        Some(BinSpec::Count(c)) if *c > 0 => *c,
        _ => ((1.0 + (all_values.len().max(1) as f64).log2()).ceil() as usize)
          .max(1),
      };
      let range = d_max - d_min;
      let bin_width = if range.abs() < f64::EPSILON {
        1.0
      } else {
        range / num_bins as f64
      };
      (0..=num_bins)
        .map(|i| d_min + i as f64 * bin_width)
        .collect()
    }
  }
}

/// Count how many values of `values` fall into each bin defined by `edges`.
/// The last bin is closed on the right; all others are half-open `[lo, hi)`.
fn histogram_bin_counts(values: &[f64], edges: &[f64]) -> Vec<usize> {
  let num_bins = edges.len().saturating_sub(1);
  let mut counts = vec![0usize; num_bins];
  if num_bins == 0 {
    return counts;
  }
  for &v in values {
    for i in 0..num_bins {
      let in_bin = if i == num_bins - 1 {
        v >= edges[i] && v <= edges[i + 1]
      } else {
        v >= edges[i] && v < edges[i + 1]
      };
      if in_bin {
        counts[i] += 1;
        break;
      }
    }
  }
  counts
}

/// Convert per-bin counts into bar heights according to a height spec.
fn histogram_bin_heights(
  counts: &[usize],
  edges: &[f64],
  height: HistogramHeight,
) -> Vec<f64> {
  let n: usize = counts.iter().sum();
  let n_f = n.max(1) as f64;
  match height {
    HistogramHeight::Count => counts.iter().map(|&c| c as f64).collect(),
    HistogramHeight::Probability => {
      counts.iter().map(|&c| c as f64 / n_f).collect()
    }
    HistogramHeight::Pdf => counts
      .iter()
      .enumerate()
      .map(|(i, &c)| {
        let w = (edges[i + 1] - edges[i]).abs();
        if w > 0.0 { c as f64 / (n_f * w) } else { 0.0 }
      })
      .collect(),
    HistogramHeight::CumulativeCount => {
      let mut acc = 0.0;
      counts
        .iter()
        .map(|&c| {
          acc += c as f64;
          acc
        })
        .collect()
    }
    HistogramHeight::Cdf => {
      let mut acc = 0.0;
      counts
        .iter()
        .map(|&c| {
          acc += c as f64;
          acc / n_f
        })
        .collect()
    }
  }
}

/// Build a `PlotSource` describing histogram bars as one filled step-outline
/// series per dataset. This lets `Show` overlay a Histogram with plots
/// (`Show[Histogram[...], Plot[...]]`) in a shared coordinate system, exactly
/// as Wolfram does — the bars and any overlaid curves share the same axes.
pub(crate) fn histogram_plot_source(
  datasets: &[Vec<f64>],
  bin_spec: Option<&BinSpec>,
  height: HistogramHeight,
  colors: &[(u8, u8, u8)],
  image_size: (u32, u32),
) -> Option<crate::syntax::PlotSource> {
  let all_values: Vec<f64> =
    datasets.iter().flat_map(|d| d.iter().copied()).collect();
  if all_values.is_empty() {
    return None;
  }
  let bin_edges = histogram_bin_edges(&all_values, bin_spec);
  let num_bins = bin_edges.len().saturating_sub(1);
  if num_bins == 0 {
    return None;
  }

  let mut series = Vec::with_capacity(datasets.len());
  let mut max_h = 0.0_f64;
  for (di, d) in datasets.iter().enumerate() {
    let counts = histogram_bin_counts(d, &bin_edges);
    let heights = histogram_bin_heights(&counts, &bin_edges, height);
    // Trace the bar silhouette: up each bar's left edge, across its top,
    // then back down to the axis at the end. Filling to the axis paints
    // the bars.
    let mut points = Vec::with_capacity(num_bins * 2 + 2);
    points.push((bin_edges[0], 0.0));
    for i in 0..num_bins {
      points.push((bin_edges[i], heights[i]));
      points.push((bin_edges[i + 1], heights[i]));
      max_h = max_h.max(heights[i]);
    }
    points.push((bin_edges[num_bins], 0.0));
    let color = colors
      .get(di)
      .copied()
      .unwrap_or(PLOT_COLORS[di % PLOT_COLORS.len()]);
    series.push(crate::syntax::PlotSeriesData {
      points,
      color,
      is_scatter: false,
      filling: crate::syntax::SeriesFilling::Axis,
      fill_color: None,
      fill_opacity: None,
      marker: None,
      thickness: None,
    });
  }

  let y_hi = if max_h > 0.0 { max_h * 1.05 } else { 1.0 };
  Some(crate::syntax::PlotSource {
    series,
    x_range: (bin_edges[0], bin_edges[num_bins]),
    y_range: (0.0, y_hi),
    image_size,
    options: Vec::new(),
  })
}

pub(crate) fn generate_histogram_svg(
  datasets: &[Vec<f64>],
  bin_spec: Option<&BinSpec>,
  height: HistogramHeight,
  opts: &mut ChartOptions,
) -> Result<String, InterpreterError> {
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  // Shared bin edges spanning every dataset, then per-dataset counts and
  // bar heights (heights depend on the height spec: Count, PDF, …).
  let all_values: Vec<f64> =
    datasets.iter().flat_map(|d| d.iter().copied()).collect();
  let bin_edges = histogram_bin_edges(&all_values, bin_spec);
  let num_bins = bin_edges.len().saturating_sub(1);
  if num_bins == 0 {
    return Ok("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string());
  }
  let dataset_counts: Vec<Vec<usize>> = datasets
    .iter()
    .map(|d| histogram_bin_counts(d, &bin_edges))
    .collect();
  let dataset_heights: Vec<Vec<f64>> = dataset_counts
    .iter()
    .map(|c| histogram_bin_heights(c, &bin_edges, height))
    .collect();

  // A named ChartStyle scheme colors each bin with a distinct sampled color.
  crate::functions::chart::apply_color_scheme(opts, num_bins);
  let max_height = dataset_heights
    .iter()
    .flat_map(|h| h.iter().copied())
    .fold(0.0_f64, f64::max);
  let max_height = if max_height > 0.0 { max_height } else { 1.0 };

  // Explicit PlotRange overrides the auto-computed extents (10% headroom
  // above the tallest bar, bins spanning exactly the edge range).
  let (y_min, y_max) = opts.plot_range_y.unwrap_or((0.0, max_height * 1.1));
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };
  let (x_lo, x_hi) = opts
    .plot_range_x
    .unwrap_or((bin_edges[0], bin_edges[num_bins]));
  let x_hi = if x_hi <= x_lo { x_lo + 1.0 } else { x_hi };

  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  // Extra room for a PlotLabel above and an x-axis label below.
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  // A `FrameLabel` takes room below/left; an `AxesLabel` sits at the far end
  // of its axis, so it takes room above and to the right instead.
  let has_x_frame_label = opts
    .frame_label
    .as_ref()
    .is_some_and(|(x, _)| !x.is_empty());
  let axes_label_x = opts
    .axes_label
    .as_ref()
    .map(|(x, _)| x.as_str())
    .filter(|x| !x.is_empty());
  let has_y_axes_label =
    opts.axes_label.as_ref().is_some_and(|(_, y)| !y.is_empty());
  let top_margin = if has_plot_label { 35 * s } else { 10 * s }
    + plot_label_extra_lines(opts.plot_label.as_ref()) as i32 * 20 * s
    + if has_y_axes_label { 20 * s } else { 0 };
  let bottom_extra = if has_x_frame_label { 24.0 * sf } else { 0.0 };
  let x_label_area = 40 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 65 * RESOLUTION_SCALE;
  let axes_label_margin_right = axes_label_x.map_or(0.0, |label| {
    // The label is drawn at `font_size`; ~0.62 em per character covers the
    // proportional font with a little slack.
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(label) as f64
        * 0.62
        * sf
        * 18.0
  }) as u32;

  let (bg_color, dark_gray, _light_gray, label_fill, title_default_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;

    let tick = MINOR_TICK_LEN * s;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(10 * s as u32 + axes_label_margin_right)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_lo..x_hi, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("Histogram: {e}"))
      })?;

    let x_major = nice_step(x_hi - x_lo, AXIS_TICK_TARGET);
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
    let x_minor_step = x_major / 5.0;
    let y_minor_step = y_major / 5.0;
    let x_tick_count = ((x_hi - x_lo) / x_minor_step).round() as usize + 1;
    let y_tick_count = ((y_max - y_min) / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_tick_count)
      .y_labels(y_tick_count)
      .x_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, x_major) {
          format_tick_with_step(*v, x_major)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick_with_step(*v, y_major)
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

    // Draw contiguous histogram bars using plotters Rectangles.
    // ChartStyle colors cycle per bin (matching BarChart's per-element
    // convention); the default is a single uniform color.
    // With one dataset, ChartStyle colors cycle per bin (matching BarChart);
    // the default is a single uniform color. With several datasets, each is
    // drawn in its own color and overlaid semi-transparently so overlapping
    // bars stay visible.
    let multi = dataset_heights.len() > 1;
    let style_color = |idx: usize| -> (u8, u8, u8) {
      if !opts.chart_style.is_empty() {
        let c = &opts.chart_style[idx % opts.chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else if multi {
        PLOT_COLORS[idx % PLOT_COLORS.len()]
      } else {
        PLOT_COLORS[0]
      }
    };
    for (di, heights) in dataset_heights.iter().enumerate() {
      for (i, &h) in heights.iter().enumerate() {
        let (r, g, b) = if multi {
          style_color(di)
        } else {
          style_color(i)
        };
        let color = RGBColor(r, g, b);
        let bx0 = bin_edges[i];
        let bx1 = bin_edges[i + 1];
        let fill: ShapeStyle = if multi {
          color.mix(0.55).filled()
        } else {
          color.filled()
        };
        chart
          .draw_series(std::iter::once(Rectangle::new(
            [(bx0, 0.0), (bx1, h)],
            fill,
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Histogram: {e}"))
          })?;
      }
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;
  }

  add_bar_borders(&mut buf, RESOLUTION_SCALE);

  // Inject hover tooltips with bin range and count into histogram rects.
  // Order matches the draw order above (dataset-major, then bin).
  let hist_tooltips: Vec<String> = dataset_counts
    .iter()
    .flat_map(|counts| {
      counts.iter().enumerate().map(|(i, &c)| {
        let lo = format_tooltip_value(bin_edges[i]);
        let hi = format_tooltip_value(bin_edges[i + 1]);
        format!("[{lo}, {hi}): {c}")
      })
    })
    .collect();
  inject_bar_tooltips_str(&mut buf, &hist_tooltips);

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Plot area coordinates (kept in sync with the ChartBuilder margins above).
  let margin_left = 10.0 * sf;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = top_margin as f64;
  let plot_w = render_width as f64
    - margin_left
    - 10.0 * sf
    - axes_label_margin_right as f64
    - y_label_area as f64;
  let plot_h =
    render_height as f64 - top_margin as f64 - 10.0 * sf - x_label_area as f64;

  // Extend labeled (major) ticks beyond the minor ticks drawn by plotters.
  {
    let x_major = nice_step(x_hi - x_lo, AXIS_TICK_TARGET);
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
    inject_major_tick_extensions(
      &mut buf,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      Some((x_lo, x_hi, x_major)),
      Some((y_min, y_max, y_major)),
      MINOR_TICK_LEN as f64 * sf,
      MAJOR_TICK_LEN as f64 * sf,
      sf,
      label_fill,
    );
  }

  // AxesLabel / FrameLabel and PlotLabel, inserted before </svg>.
  let font_size = sf * 18.0;
  let title_font_size = sf * 22.0;
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();
    let axis_y = plot_y0 + plot_h;

    // FrameLabel: centred outside the bottom/left edge.
    if let Some((x_label, y_label)) = &opts.frame_label {
      if !x_label.is_empty() {
        let cx = plot_x0 + plot_w / 2.0;
        let base_y = axis_y + font_size * 2.8;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(y_label)
        ));
      }
    }

    // AxesLabel: at the far end of each axis.
    labels_svg.push_str(&axes_label_svg(
      opts.axes_label.as_ref(),
      (plot_x0, plot_y0, plot_w, plot_h),
      (x_lo, x_hi, y_min, y_max),
      (true, true),
      font_size,
      label_fill,
    ));

    // PlotLabel: centered above the chart
    if let Some(sl) = &opts.plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      let ty = top_margin as f64
        - title_font_size * 0.5
        - if has_y_axes_label {
          font_size * 1.2
        } else {
          0.0
        };
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
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
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    if !labels_svg.is_empty() {
      buf.insert_str(insert_pos, &labels_svg);
    }
  }

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
    let tick = MINOR_TICK_LEN * s;

    let top_margin = margins.map_or(10 * RESOLUTION_SCALE, |m| m.top_margin);
    let x_label_area = margins.map_or(
      if x_tick_positions.is_some() {
        12 * RESOLUTION_SCALE
      } else {
        40 * RESOLUTION_SCALE
      },
      |m| m.x_label_area,
    );
    let y_label_area =
      margins.map_or(65 * RESOLUTION_SCALE, |m| m.y_label_area);
    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin)
      .margin_right(10 * s as u32)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
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
            format_tick_with_step(*v, y_major)
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
      let x_major = nice_step(x_max - x_min, AXIS_TICK_TARGET);
      let x_minor_step = x_major / 5.0;
      let x_tick_count = ((x_max - x_min) / x_minor_step).round() as usize + 1;

      chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(x_tick_count)
        .y_labels(y_tick_count)
        .x_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, x_major) {
            format_tick_with_step(*v, x_major)
          } else {
            String::new()
          }
        })
        .y_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, y_major) {
            format_tick_with_step(*v, y_major)
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
  let top_margin_f = margins.map_or(margin, |m| m.top_margin as f64);
  let y_label_area_f = margins.map_or(65.0 * s, |m| m.y_label_area as f64);
  let x_label_area_f = margins.map_or(
    if x_tick_positions.is_some() {
      12.0 * s
    } else {
      40.0 * s
    },
    |m| m.x_label_area as f64,
  );
  let plot_x0 = margin + y_label_area_f;
  let plot_y0 = top_margin_f;
  let plot_w = render_width as f64 - 2.0 * margin - y_label_area_f;
  let plot_h = render_height as f64 - top_margin_f - margin - x_label_area_f;

  // Extend labeled (major) ticks beyond the minor ticks drawn by plotters.
  // In custom-tick mode, the x axis is drawn manually (below) at the major
  // tick length, so we only extend the y axis here.
  {
    let y_major = nice_step(y_max - y_min, AXIS_TICK_TARGET);
    let x_axis_ext = if x_tick_positions.is_none() {
      Some((x_min, x_max, nice_step(x_max - x_min, AXIS_TICK_TARGET)))
    } else {
      None
    };
    inject_major_tick_extensions(
      &mut buf,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      x_axis_ext,
      Some((y_min, y_max, y_major)),
      MINOR_TICK_LEN as f64 * s,
      MAJOR_TICK_LEN as f64 * s,
      s,
      label_fill,
    );
  }

  // Draw custom x-axis tick marks if specified. These are at user-supplied
  // positions and all get the "major" tick length since they're all labeled.
  if let Some(positions) = x_tick_positions {
    let tick_len = MAJOR_TICK_LEN as f64 * s;
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

/// Add a border to filled chart bars in the SVG.
/// Plotters' SVG backend doesn't support stroke-width on rects,
/// so we post-process the SVG to add it.
fn add_bar_borders(buf: &mut String, stroke_width: u32) {
  // The first <rect> is the background, skip it.
  // All subsequent filled rects (with stroke="none") are bars.
  let marker = "stroke=\"none\"/>";
  let color = if crate::is_dark_mode() {
    "#555555"
  } else {
    "#000000"
  };
  let replacement =
    format!("stroke=\"{color}\" stroke-width=\"{stroke_width}\"/>");
  // Skip the first occurrence (background rect)
  if let Some(first) = buf.find(marker) {
    let after_first = first + marker.len();
    let rest = buf[after_first..].replace(marker, &replacement);
    buf.truncate(after_first);
    buf.push_str(&rest);
  }
}

/// Format a numeric value for tooltip display.
/// Integers (or values very close to integers) are shown without decimals.
fn format_tooltip_value(v: f64) -> String {
  if (v - v.round()).abs() < 1e-10 {
    format!("{}", v as i64)
  } else {
    format!("{v}")
  }
}

/// Inject `<title>` tooltip elements into bar `<rect>` elements.
/// Skips the first `<rect` with `stroke="none"` (the background).
/// All subsequent `<rect` elements that end with `/>` are bars.
fn inject_bar_tooltips(buf: &mut String, values: &[f64]) {
  let tooltips: Vec<String> =
    values.iter().map(|&v| format_tooltip_value(v)).collect();
  inject_bar_tooltips_str(buf, &tooltips);
}

/// Inject `<title>` tooltip strings into bar `<rect>` elements.
/// Skips the first `<rect` with `stroke="none"` (the background rect).
fn inject_bar_tooltips_str(buf: &mut String, tooltips: &[String]) {
  // After add_bar_borders, bar rects have a border style while the
  // background rect still has stroke="none". Find all <rect ... /> that
  // do NOT have stroke="none" and inject titles.
  let mut result = String::with_capacity(buf.len() + tooltips.len() * 30);
  let mut remaining = buf.as_str();
  let mut tooltip_idx = 0;

  while let Some(rect_start) = remaining.find("<rect ") {
    // Find the end of this rect element
    let after_rect = &remaining[rect_start..];
    if let Some(close_pos) = after_rect.find("/>") {
      let rect_tag = &after_rect[..close_pos + 2];
      let is_background = rect_tag.contains("stroke=\"none\"");

      // Copy everything up to the rect
      result.push_str(&remaining[..rect_start]);

      if !is_background && tooltip_idx < tooltips.len() {
        // Replace self-closing /> with ><title>...</title></rect>
        let escaped = html_escape(&tooltips[tooltip_idx]);
        result.push_str(&after_rect[..close_pos]);
        result.push_str(&format!("><title>{escaped}</title></rect>"));
        tooltip_idx += 1;
      } else {
        // Keep as-is
        result.push_str(rect_tag);
      }

      remaining = &remaining[rect_start + close_pos + 2..];
    } else {
      break;
    }
  }

  // Append any remaining content
  result.push_str(remaining);
  *buf = result;
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
        "<svg width=\"100%\" viewBox=\"0 0 {render_width} {render_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
      )
    } else {
      format!(
        "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {render_width} {render_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
      )
    };
    buf.replace_range(..pos, &new_header);
  }
}

/// Inject drop-shadow filters for series styled with `DropShadowing[…]`
/// into a finished plotters SVG. plotters has no filter API, so a
/// `<defs>` block is inserted after the `<svg>` header and each curve
/// polyline — identified by the exact stroke color of its series — is
/// tagged with the filter attribute. Shadow parameters are scaled by
/// RESOLUTION_SCALE to match the oversampled render coordinates.
fn inject_drop_shadows(
  buf: &mut String,
  plot_style: &[SeriesStyle],
  n_series: usize,
) {
  if plot_style.is_empty() || !plot_style.iter().any(|s| s.shadow.is_some()) {
    return;
  }
  let mut defs = String::new();
  let mut seen_ids: Vec<String> = Vec::new();
  // (needle with the series' stroke attr, tagged replacement)
  let mut rules: Vec<(String, String)> = Vec::new();
  for idx in 0..n_series {
    let style = &plot_style[idx % plot_style.len()];
    if let Some(ds) = &style.shadow {
      let id = ds.filter_id();
      if !seen_ids.contains(&id) {
        seen_ids.push(id.clone());
        defs.push_str(&ds.filter_def(RESOLUTION_SCALE as f64));
        defs.push('\n');
      }
      let (r, g, b) = series_color(plot_style, idx);
      let needle = format!(
        "<polyline fill=\"none\" opacity=\"1\" stroke=\"#{r:02X}{g:02X}{b:02X}\""
      );
      let replacement = format!(
        "<polyline filter=\"url(#{id})\" fill=\"none\" opacity=\"1\" stroke=\"#{r:02X}{g:02X}{b:02X}\""
      );
      if !rules.iter().any(|(n, _)| n == &needle) {
        rules.push((needle, replacement));
      }
    }
  }
  if let Some(pos) = buf.find('>') {
    buf.insert_str(pos + 1, &format!("\n<defs>\n{defs}</defs>"));
  }
  for (needle, replacement) in &rules {
    *buf = buf.replace(needle, replacement);
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
  // A plot holds its arguments, so an option value that is still an
  // arithmetic expression never reached the evaluator: `ImageSize ->
  // 400 {1, 1}` — the way a Demonstration asks for a square picture —
  // arrives as `Times[400, {1, 1}]`. Evaluate it once so the shapes below
  // see the `{400, 400}` they understand.
  let evaluated = match value {
    Expr::BinaryOp { .. } | Expr::FunctionCall { .. } => {
      evaluate_expr_to_expr(value).ok()
    }
    _ => None,
  };
  let value = evaluated.as_ref().unwrap_or(value);
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
    // `{w, h}` fixes both dimensions; `Automatic` in either slot leaves
    // that one to follow from the other and the default aspect, which is
    // how a Demonstration sizes a row of plots by height alone.
    Expr::List(items) if items.len() == 2 => {
      let dim = |e: &Expr| -> Option<Option<u32>> {
        match e {
          Expr::Identifier(n) if n == "Automatic" => Some(None),
          Expr::Integer(n) if *n > 0 => Some(Some(*n as u32)),
          Expr::BigInteger(n) => {
            use num_traits::ToPrimitive;
            let v = n.to_u32()?;
            (v > 0).then_some(Some(v))
          }
          Expr::Real(f) if *f > 0.0 => Some(Some(f.round() as u32)),
          _ => None,
        }
      };
      match (dim(&items[0])?, dim(&items[1])?) {
        (Some(w), Some(h)) => Some((w, h, false)),
        (Some(w), None) => Some((w, (w as f64 * aspect).round() as u32, false)),
        (None, Some(h)) => {
          Some((((h as f64) / aspect).round() as u32, h, false))
        }
        (None, None) => Some((def_w, def_h, false)),
      }
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

/// Parse a single PlotStyle element into a SeriesStyle.
/// Handles: a color, `Thick`, `Dashed`, `Dotted`, `DotDashed`,
/// `Directive[...]`, `{Red, Thick, Dashed}`, etc.
fn parse_one_series_style(expr: &Expr) -> SeriesStyle {
  let mut style = SeriesStyle::default();
  apply_style_directive(expr, &mut style);
  style
}

/// Apply a style directive expression to a SeriesStyle.
fn apply_style_directive(expr: &Expr, style: &mut SeriesStyle) {
  // Try as a color first
  if let Some(c) = parse_color(expr) {
    style.color = Some(c);
    return;
  }
  match expr {
    Expr::Identifier(s) => match s.as_str() {
      "Thick" => style.thickness = Some(2.0),
      "Thin" => style.thickness = Some(0.5),
      // A named dash length is absolute pixels, not a fraction of the
      // picture: `Dashed` is 4 on / 4 off whatever the image size. Negative
      // marks absolute, as it does for thickness.
      "Dashed" => style.dashing = Some(vec![-4.0, -4.0]),
      "Dotted" => style.dashing = Some(vec![0.0, -4.0]),
      "DotDashed" => style.dashing = Some(vec![0.0, -4.0, -4.0, -4.0]),
      _ => {}
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Directive" => {
        for a in args {
          apply_style_directive(a, style);
        }
      }
      "DropShadowing" => {
        style.shadow = crate::functions::graphics::parse_drop_shadowing(args);
      }
      "Thickness" if args.len() == 1 => {
        if let Expr::Identifier(s) = &args[0] {
          match s.as_str() {
            "Large" => style.thickness = Some(2.0),
            "Tiny" => style.thickness = Some(0.5),
            _ => {
              if let Some(t) = try_eval_to_f64(&args[0]) {
                // Relative thickness: fraction of plot width → display px
                // 360px default width * fraction
                style.thickness = Some(t * 360.0);
              }
            }
          }
        } else if let Some(t) = try_eval_to_f64(&args[0]) {
          style.thickness = Some(t * 360.0);
        }
      }
      "AbsoluteThickness" if args.len() == 1 => {
        if let Some(t) = try_eval_to_f64(&args[0]) {
          style.thickness = Some(t);
        }
      }
      "Dashing" if !args.is_empty() => {
        if let Expr::List(items) = &args[0] {
          let dashes: Vec<f64> = items
            .iter()
            .filter_map(|e| match e {
              Expr::Identifier(s) => match s.as_str() {
                "Tiny" => Some(-2.0),
                "Small" => Some(-4.0),
                "Medium" => Some(-8.0),
                "Large" => Some(-16.0),
                _ => None,
              },
              _ => try_eval_to_f64(e),
            })
            .collect();
          if !dashes.is_empty() {
            style.dashing = Some(dashes);
          }
        } else {
          let d = match &args[0] {
            Expr::Identifier(s) => match s.as_str() {
              "Tiny" => Some(0.005),
              "Small" => Some(0.01),
              "Medium" => Some(0.02),
              "Large" => Some(0.04),
              _ => None,
            },
            _ => try_eval_to_f64(&args[0]),
          };
          if let Some(d) = d {
            style.dashing = Some(vec![d, d]);
          }
        }
      }
      _ => {}
    },
    Expr::List(items) => {
      // {Red, Thick, Dashed} — apply all sub-directives
      for item in items {
        apply_style_directive(item, style);
      }
    }
    _ => {}
  }
}

/// The font size a `PlotMarkers` glyph is drawn at when the marker spec
/// names none — the default graphics font size the Wolfram Language uses.
const DEFAULT_MARKER_SIZE: f64 = 12.0;

/// Read one `PlotMarkers` entry: a glyph, optionally wrapped in `Style`
/// (which may carry a colour and a font size in any order) or paired with
/// its size as `{marker, size}`. Anything else (a `Graphics` marker, say)
/// gives `None`, which leaves that series on plain round points.
fn parse_one_marker(expr: &Expr) -> Option<PlotMarker> {
  let val = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  match &val {
    Expr::String(s) => Some(PlotMarker {
      glyph: s.clone(),
      size: DEFAULT_MARKER_SIZE,
      color: None,
    }),
    // `Style[marker, directives…]` — a colour and/or a font size.
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      let mut marker = parse_one_marker(&args[0])?;
      for directive in &args[1..] {
        if let Some(c) = crate::functions::graphics::parse_color(directive) {
          marker.color = Some((
            (c.r * 255.0).round() as u8,
            (c.g * 255.0).round() as u8,
            (c.b * 255.0).round() as u8,
          ));
        } else if let Some(size) = try_eval_to_f64(directive) {
          marker.size = size;
        } else if let Expr::FunctionCall { name, args } = directive
          && name == "FontSize"
          && args.len() == 1
          && let Some(size) = try_eval_to_f64(&args[0])
        {
          marker.size = size;
        }
      }
      Some(marker)
    }
    // `{marker, size}` — the size is always the second element.
    Expr::List(items) if items.len() == 2 => {
      let size = try_eval_to_f64(&items[1])?;
      let mut marker = parse_one_marker(&items[0])?;
      marker.size = size;
      Some(marker)
    }
    _ => None,
  }
}

/// Parse a `PlotMarkers` option value into one marker per series. A single
/// marker (or `{marker, size}` pair) applies to every series; a list of
/// markers is cycled over the series, as the Wolfram Language does.
pub(crate) fn parse_plot_markers(
  replacement: &Expr,
) -> Vec<Option<PlotMarker>> {
  let val =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  // `{m1, m2, …}` is per-series unless it is the `{marker, size}` pair,
  // which `parse_one_marker` recognises.
  if let Expr::List(items) = &val
    && parse_one_marker(&val).is_none()
  {
    return items.iter().map(parse_one_marker).collect();
  }
  match parse_one_marker(&val) {
    Some(marker) => vec![Some(marker)],
    None => Vec::new(),
  }
}

/// The marker for series `idx`, cycling the list as the Wolfram Language
/// does when there are fewer markers than series.
fn series_marker(
  markers: &[Option<PlotMarker>],
  idx: usize,
) -> Option<&PlotMarker> {
  if markers.is_empty() {
    return None;
  }
  markers[idx % markers.len()].as_ref()
}

/// Carry the parsed `PlotMarkers` into a plot's [`PlotSource`], so a
/// `Show` that merges the plot keeps drawing the same glyphs.
pub(crate) fn apply_markers_to_source(
  source: &mut crate::syntax::PlotSource,
  markers: &[Option<PlotMarker>],
) {
  for (i, series) in source.series.iter_mut().enumerate() {
    series.marker = series_marker(markers, i).cloned();
  }
}

/// Parse a PlotStyle option value into a list of SeriesStyles.
pub(crate) fn parse_plot_style(replacement: &Expr) -> Vec<SeriesStyle> {
  let val = evaluate_expr_to_expr(replacement).unwrap_or(replacement.clone());
  if let Expr::List(items) = &val {
    // Check if this is a list of per-series styles or a single compound style.
    // If any item is itself a Directive or a List, treat as per-series.
    // If all items are simple directives (colors, Thick, etc.), treat as
    // a single compound style applied to all series.
    let has_per_series = items.iter().any(|item| {
      matches!(
        item,
        Expr::FunctionCall { name, .. } if name == "Directive"
      ) || matches!(item, Expr::List(_))
        || matches!(
          item,
          Expr::FunctionCall { name, .. }
            if name == "RGBColor"
              || name == "Hue"
              || name == "GrayLevel"
              || name == "Darker"
              || name == "Lighter"
              || name == "Blend"
        )
        || matches!(item, Expr::Identifier(s) if crate::functions::graphics::named_color(s).is_some())
    });
    if has_per_series {
      items.iter().map(parse_one_series_style).collect()
    } else {
      // Single compound style: {Purple, Thick, Dashed}
      let mut style = SeriesStyle::default();
      for item in items {
        apply_style_directive(item, &mut style);
      }
      vec![style]
    }
  } else {
    let style = parse_one_series_style(&val);
    if style.color.is_some()
      || style.thickness.is_some()
      || style.dashing.is_some()
      || style.shadow.is_some()
    {
      vec![style]
    } else {
      Vec::new()
    }
  }
}

/// Apply a named PlotTheme to PlotOptions.
pub(crate) fn apply_plot_theme(opts: &mut PlotOptions, theme: &str) {
  match theme {
    "Scientific" => {
      opts.frame = true;
      opts.grid_lines_x = true;
      opts.grid_lines_y = true;
    }
    "Business" => {
      opts.frame = true;
      opts.grid_lines_y = true;
    }
    "Detailed" => {
      opts.frame = true;
      opts.grid_lines_x = true;
      opts.grid_lines_y = true;
    }
    "Web" => {
      opts.grid_lines_y = true;
    }
    "Minimal" => {
      opts.axes = (false, false);
      opts.ticks = false;
    }
    "Classic" => {
      // Default Wolfram look: axes, no frame, no grid
    }
    _ => {}
  }
}

/// Parse a `GridLines` option value into per-axis (x, y) specifications.
///
/// Supported forms:
/// - `Automatic` / `All` / `True` → automatic lines on both axes
/// - `None` / `False` → no grid lines
/// - `{xspec, yspec}` → independent specs per axis, where each spec is
///   `Automatic`, `None`, or a list of entries. Each entry is a bare position
///   or a `{position, directive}` pair (e.g. `{Pi, Dashed}`, `{1, Orange}`).
pub(crate) fn parse_grid_lines_spec(expr: &Expr) -> (GridSide, GridSide) {
  match expr {
    Expr::List(items) if items.len() == 2 => {
      (parse_grid_side(&items[0]), parse_grid_side(&items[1]))
    }
    Expr::Identifier(s) if s == "None" || s == "False" => {
      (GridSide::None, GridSide::None)
    }
    Expr::Identifier(s) if s == "Automatic" || s == "All" || s == "True" => {
      (GridSide::Automatic, GridSide::Automatic)
    }
    _ => (GridSide::None, GridSide::None),
  }
}

/// Parse one axis side of a `GridLines` spec.
fn parse_grid_side(expr: &Expr) -> GridSide {
  match expr {
    Expr::Identifier(s) if s == "None" || s == "False" => GridSide::None,
    Expr::Identifier(s) if s == "Automatic" || s == "All" || s == "True" => {
      GridSide::Automatic
    }
    Expr::List(items) => GridSide::Explicit(
      items.iter().filter_map(parse_grid_line_entry).collect(),
    ),
    other => match try_eval_to_f64(other) {
      Some(pos) => GridSide::Explicit(vec![GridLine {
        pos,
        style: SeriesStyle::default(),
      }]),
      None => GridSide::None,
    },
  }
}

/// Parse a single grid-line entry: a bare position, or a `{position, directive}`
/// pair where the directive styles just that line.
fn parse_grid_line_entry(entry: &Expr) -> Option<GridLine> {
  match entry {
    Expr::Identifier(s) if s == "None" => None,
    Expr::List(pair) if pair.len() == 2 => {
      let pos = try_eval_to_f64(&pair[0])?;
      let mut style = SeriesStyle::default();
      apply_style_directive(&pair[1], &mut style);
      Some(GridLine { pos, style })
    }
    other => {
      let pos = try_eval_to_f64(other)?;
      Some(GridLine {
        pos,
        style: SeriesStyle::default(),
      })
    }
  }
}

/// Apply a parsed `GridSide` to the matching PlotOptions fields.
pub(crate) fn apply_grid_side(
  side: GridSide,
  auto_flag: &mut bool,
  explicit: &mut Vec<GridLine>,
) {
  match side {
    GridSide::None => {
      *auto_flag = false;
      explicit.clear();
    }
    GridSide::Automatic => {
      *auto_flag = true;
      explicit.clear();
    }
    GridSide::Explicit(v) => {
      *auto_flag = false;
      *explicit = v;
    }
  }
}

/// Parse an `Axes` option value into `(show_x, show_y)`. Accepts `True` /
/// `False` and the per-axis `{xbool, ybool}` form; anything else (e.g.
/// `Automatic`) leaves the current setting alone.
pub(crate) fn parse_axes_option(value: &Expr) -> Option<(bool, bool)> {
  let is_true = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "True");
  match value {
    Expr::Identifier(s) if s == "True" => Some((true, true)),
    Expr::Identifier(s) if s == "False" => Some((false, false)),
    Expr::List(items) if items.len() == 2 => {
      Some((is_true(&items[0]), is_true(&items[1])))
    }
    _ => None,
  }
}

/// Parse a style option's value — `Directive[Red, Dashed]`, a bare color,
/// `{Thick, Blue}`, … — into the style it describes. `Automatic` / `None`,
/// and any value that sets nothing, come back as `None` so the caller keeps
/// its built-in look. Shared by `GridLinesStyle`, `ContourStyle` and the
/// other single-style options.
pub(crate) fn parse_style_directives(value: &Expr) -> Option<SeriesStyle> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  if matches!(&val, Expr::Identifier(s) if s == "Automatic" || s == "None") {
    return None;
  }
  let mut style = SeriesStyle::default();
  match &val {
    Expr::List(items) => {
      for item in items {
        apply_style_directive(item, &mut style);
      }
    }
    other => apply_style_directive(other, &mut style),
  }
  (style.color.is_some()
    || style.thickness.is_some()
    || style.dashing.is_some())
  .then_some(style)
}

/// Parse a PlotRange option value into (x_range, y_range) overrides.
///
/// Supported forms:
/// - `All` / `Automatic` / `Full` → (None, None)
/// - `{ymin, ymax}` → (None, Some((ymin, ymax)))
/// - `{{xmin, xmax}, {ymin, ymax}}` → (Some(x), Some(y))
/// - `{All, {ymin, ymax}}` / `{{xmin,xmax}, All}` → only the specified axis
#[allow(clippy::type_complexity)]
pub(crate) fn parse_plot_range(
  value: &Expr,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());

  // Automatic / All / Full → no override
  if matches!(&val, Expr::Identifier(s) if s == "All" || s == "Automatic" || s == "Full")
  {
    return (None, None);
  }

  let parse_pair = |e: &Expr| -> Option<(f64, f64)> {
    if let Expr::List(items) = e
      && items.len() == 2
    {
      let a = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[0]).unwrap_or_else(|_| items[0].clone()),
      )?;
      let b = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[1]).unwrap_or_else(|_| items[1].clone()),
      )?;
      // Wolfram normalizes a reversed range: `PlotRange -> {3, -3}` plots
      // exactly like `{-3, 3}` (verified against a FrontEnd-saved raster
      // of the oscilloscope Demonstration, which uses the reversed form).
      Some(if a <= b { (a, b) } else { (b, a) })
    } else {
      None
    }
  };

  if let Expr::List(items) = &val
    && items.len() == 2
  {
    // {{xmin,xmax}, {ymin,ymax}} (optionally with All/Automatic as a placeholder)
    if matches!(&items[0], Expr::List(_))
      || matches!(&items[1], Expr::List(_))
      || matches!(&items[0], Expr::Identifier(s) if s == "All" || s == "Automatic" || s == "Full")
    {
      let x_range = parse_pair(&items[0]);
      let y_range = parse_pair(&items[1]);
      // If neither inner is a pair, fall through to {ymin, ymax} handling.
      if x_range.is_some() || y_range.is_some() {
        return (x_range, y_range);
      }
    }

    // {ymin, ymax}: y range only
    if let Some(y) = parse_pair(&val) {
      return (None, Some(y));
    }
  }

  (None, None)
}

/// Parsed `FrameLabel` edges. Empty strings mean "no label on that edge".
#[derive(Default)]
pub(crate) struct FrameLabels {
  pub bottom: String,
  pub left: String,
  pub top: String,
  pub right: String,
}

/// Convert a single FrameLabel entry to a label string. `None` (the symbol)
/// and unrenderable expressions become an empty string.
fn frame_label_entry(e: &Expr) -> String {
  if matches!(e, Expr::Identifier(s) if s == "None") {
    return String::new();
  }
  crate::functions::chart::expr_to_label(e).unwrap_or_default()
}

/// One side of `Ticks -> {xspec, yspec}`: an explicit list of positions,
/// each optionally carrying the text to draw at it (`{pos, label}`).
/// `None` for `Automatic`/`None`/anything that is not a list.
///
/// Each label comes back as the SVG `<text>` content to draw — already
/// escaped, and typeset where the tick asked for something more than plain
/// text — so [`explicit_ticks_svg`] can emit it verbatim.
pub(crate) fn parse_explicit_ticks(value: &Expr) -> Option<Vec<(f64, String)>> {
  // A spec written out as a list keeps its entries as written: a tick is
  // labelled with the expression standing at it, and `3 Pi/2` set as one
  // fraction is not the same reading as the `(3/2) Pi` it evaluates to.
  // Anything else (a symbol, a `Table[…]`) has to be worked out first.
  let val = match value {
    Expr::List(_) => value.clone(),
    _ => evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone()),
  };
  let Expr::List(entries) = &val else {
    return None;
  };
  let ticks: Vec<(f64, String)> = entries
    .iter()
    .filter_map(|entry| match entry {
      Expr::List(pair) if pair.len() >= 2 => {
        let pos = try_eval_to_f64(&pair[0])?;
        // A label given as text may still embed box notation (a notebook
        // writes a superscript that way), so it renders like every other
        // label a plot draws rather than being escaped raw.
        let label = crate::functions::chart::expr_to_label(&pair[1])
          .map_or_else(
            || crate::functions::graphics::svg_escape(&format_tick(pos)),
            |text| crate::functions::graphics::box_string_to_svg(&text),
          );
        Some((pos, label))
      }
      other => {
        let pos = try_eval_to_f64(other)?;
        Some((pos, bare_tick_label(other, pos)))
      }
    })
    .collect();
  (!ticks.is_empty()).then_some(ticks)
}

/// The `<text>` content drawn at a tick given as a bare position rather than
/// as a `{pos, label}` pair. Wolfram labels such a tick with the position
/// *expression*, typeset — `Ticks -> {{0, Pi/2, Pi, 3 Pi/2, 2 Pi}, …}` reads
/// "0", "π/2", "π", "3π/2", "2π" — so only a literal number is labelled by
/// its value; anything symbolic is set the way the expression is written.
pub(crate) fn bare_tick_label(e: &Expr, pos: f64) -> String {
  fn is_literal_number(e: &Expr) -> bool {
    match e {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..) => true,
      // A negative literal may arrive either folded into the number or as
      // a minus applied to it, depending on how the list was written.
      Expr::UnaryOp {
        op: crate::syntax::UnaryOperator::Minus,
        operand,
      } => is_literal_number(operand),
      _ => false,
    }
  }
  if is_literal_number(e) {
    return crate::functions::graphics::svg_escape(&format_tick(pos));
  }
  crate::functions::graphics::expr_to_svg_markup(e)
}

/// Apply a `FrameLabel` value to a plot's options. Bottom and left reuse
/// the axes-label render path; top and right get their own frame edges.
/// The space, in render units, each side of the plotting area gets under an
/// `ImagePadding` specification.
pub(crate) struct PaddedMargins {
  pub left: u32,
  pub right: u32,
  pub bottom: u32,
  pub top: u32,
}

/// Wolfram's default plot `AspectRatio`: 1/GoldenRatio, the height/width of
/// the plotting area when nothing else fixes it.
pub(crate) const DEFAULT_ASPECT_RATIO: f64 = 0.618_033_988_749_895;

/// How many intervals the Wolfram Language aims to divide a plot axis into
/// when placing automatic ticks. Verified against `Charting`ScaledTicks` over
/// a range of axis spans: with this target and [`nice_step`]'s multipliers,
/// Woxi picks the same step Wolfram does.
pub(crate) const AXIS_TICK_TARGET: usize = 6;

/// Lay out the plotting area inside an image with `ImagePadding`.
///
/// Wolfram treats the padding as a *minimum* per side: the plotting area is
/// then as large as the padding leaves while keeping its `AspectRatio`, and
/// whatever space is left over is split evenly between the two sides of each
/// axis. So `ImagePadding -> 20` on a wide-but-short image leaves far more
/// than 20 px at the left — which is what keeps the tick labels of e.g.
/// `ListPlot[…, ImageSize -> {400, 200}, ImagePadding -> 20]` from being
/// clipped.
fn padded_margins(
  [left, right, bottom, top]: [f64; 4],
  (render_width, render_height): (u32, u32),
  aspect_ratio: Option<f64>,
  sf: f64,
) -> PaddedMargins {
  let (left, right, bottom, top) =
    (left * sf, right * sf, bottom * sf, top * sf);
  let avail_w = (render_width as f64 - left - right).max(0.0);
  let avail_h = (render_height as f64 - bottom - top).max(0.0);
  let ar = aspect_ratio.unwrap_or(DEFAULT_ASPECT_RATIO);
  let (plot_w, plot_h) = if ar <= 0.0 || avail_w * ar <= avail_h {
    (avail_w, avail_w * ar)
  } else {
    (avail_h / ar, avail_h)
  };
  let (slack_w, slack_h) = ((avail_w - plot_w) / 2.0, (avail_h - plot_h) / 2.0);
  PaddedMargins {
    left: (left + slack_w).round() as u32,
    right: (right + slack_w).round() as u32,
    bottom: (bottom + slack_h).round() as u32,
    top: (top + slack_h).round() as u32,
  }
}

/// Parse an `ImagePadding` value into `[left, right, bottom, top]` display
/// pixels. Accepted forms mirror Wolfram: a single number (all four sides),
/// `{{left, right}, {bottom, top}}`, `None`/`0` (no padding) and
/// `Automatic`/`All` (which keep the automatic margins, i.e. `None` here).
pub(crate) fn parse_image_padding(value: &Expr) -> Option<[f64; 4]> {
  fn side(e: &Expr) -> Option<f64> {
    match e {
      Expr::Identifier(v) if v == "None" => Some(0.0),
      _ => crate::functions::graphics::expr_to_f64(e),
    }
  }
  match value {
    Expr::Identifier(v) if v == "Automatic" || v == "All" => None,
    Expr::Identifier(v) if v == "None" => Some([0.0; 4]),
    Expr::List(pairs) if pairs.len() == 2 => {
      let (Expr::List(lr), Expr::List(bt)) = (&pairs[0], &pairs[1]) else {
        // `{h, v}` is not a Wolfram form, so a two-element list of numbers
        // is only meaningful as the nested one.
        return None;
      };
      if lr.len() != 2 || bt.len() != 2 {
        return None;
      }
      Some([side(&lr[0])?, side(&lr[1])?, side(&bt[0])?, side(&bt[1])?])
    }
    _ => side(value).map(|p| [p; 4]),
  }
}

/// `Frame -> True | All` draws the boxed frame; anything else (`False`,
/// `Automatic`, a per-edge list) leaves the plain axes.
pub(crate) fn parse_frame_option(value: &Expr) -> bool {
  matches!(value, Expr::Identifier(v) if v == "True" || v == "All")
}

pub(crate) fn apply_frame_label_option(value: &Expr, opts: &mut PlotOptions) {
  let fl = parse_frame_label(value);
  if !fl.bottom.is_empty() {
    opts.frame_label_bottom = Some(fl.bottom);
  }
  if !fl.left.is_empty() {
    opts.frame_label_left = Some(fl.left);
  }
  if !fl.top.is_empty() {
    opts.frame_label_top = Some(fl.top);
  }
  if !fl.right.is_empty() {
    opts.frame_label_right = Some(fl.right);
  }
}

/// Parse a `FrameLabel` option value, supporting both forms:
///   `{bottom, left}` and `{{left, right}, {bottom, top}}`.
/// A bare label applies to the bottom edge.
pub(crate) fn parse_frame_label(value: &Expr) -> FrameLabels {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let mut out = FrameLabels::default();
  match &val {
    Expr::List(items) => {
      // 4-element nested form: both entries are themselves lists.
      if items.len() == 2
        && let (Expr::List(lr), Expr::List(bt)) = (&items[0], &items[1])
      {
        if let Some(e) = lr.first() {
          out.left = frame_label_entry(e);
        }
        if let Some(e) = lr.get(1) {
          out.right = frame_label_entry(e);
        }
        if let Some(e) = bt.first() {
          out.bottom = frame_label_entry(e);
        }
        if let Some(e) = bt.get(1) {
          out.top = frame_label_entry(e);
        }
        return out;
      }
      // 2-element form `{bottom, left}`.
      if let Some(e) = items.first() {
        out.bottom = frame_label_entry(e);
      }
      if let Some(e) = items.get(1) {
        out.left = frame_label_entry(e);
      }
    }
    // A bare label labels the bottom edge.
    _ => out.bottom = frame_label_entry(&val),
  }
  out
}

/// Parse PlotLegends option value into a list of legend strings.
/// Returns (legends, is_automatic, is_expressions, legend_position).
pub(crate) fn parse_plot_legends(
  value: &Expr,
) -> (Vec<String>, bool, bool, LegendPosition) {
  let val = evaluate_expr_to_expr(value).unwrap_or(value.clone());

  // Check for Placed[content, position] wrapper
  if let Expr::FunctionCall { name, args } = &val
    && name == "Placed"
    && args.len() == 2
  {
    let pos = match &args[1] {
      Expr::Identifier(s) => match s.as_str() {
        "Top" | "Above" => LegendPosition::Top,
        "Bottom" | "Below" => LegendPosition::Bottom,
        _ => LegendPosition::Right,
      },
      _ => LegendPosition::Right,
    };
    let (labels, auto, expressions, _) = parse_plot_legends(&args[0]);
    return (labels, auto, expressions, pos);
  }

  match &val {
    Expr::Identifier(s) if s == "Automatic" => {
      (Vec::new(), true, false, LegendPosition::Right)
    }
    Expr::Identifier(s) if s == "None" => {
      (Vec::new(), false, false, LegendPosition::Right)
    }
    Expr::String(s) if s == "Expressions" => {
      (Vec::new(), false, true, LegendPosition::Right)
    }
    Expr::List(items) => {
      let labels = items
        .iter()
        .map(|item| {
          crate::functions::chart::expr_to_label(item)
            .unwrap_or_else(|| crate::syntax::expr_to_string(item))
        })
        .collect();
      (labels, false, false, LegendPosition::Right)
    }
    Expr::String(s) => (vec![s.clone()], false, false, LegendPosition::Right),
    _ => (Vec::new(), false, false, LegendPosition::Right),
  }
}

/// Parse a `Background -> color` option value into a plotters fill color.
/// Returns `None` for `Background -> None` and unrecognized values, which
/// keep the theme default.
pub(crate) fn parse_background_option(expr: &Expr) -> Option<RGBColor> {
  let val = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  crate::functions::graphics::parse_color(&val).map(|c| {
    RGBColor(
      (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  })
}

/// Range/aspect overrides parsed from options and applied after all options
/// (and the data) are known.
#[derive(Default)]
pub(crate) struct PlotRangeOverrides {
  pub x: Option<(f64, f64)>,
  pub y: Option<(f64, f64)>,
  /// `PlotRange -> All` on the y axis: show every sampled value instead of
  /// the automatic range, which drops extreme outliers.
  pub y_all: bool,
  pub aspect_ratio: Option<f64>,
}

/// Apply one plot option shared by the function plotters (Plot,
/// ParametricPlot, PolarPlot, …) to `plot_opts` / `overrides`. Returns
/// `true` when `name` names an option this helper understands.
pub(crate) fn apply_common_plot_option(
  name: &str,
  replacement: &Expr,
  plot_opts: &mut PlotOptions,
  overrides: &mut PlotRangeOverrides,
) -> bool {
  match name {
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
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
        plot_opts.plot_label = Some(sl);
      }
    }
    "AxesLabel" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Expr::List(items) = &val
        && items.len() >= 2
      {
        let x =
          crate::functions::chart::expr_to_label(&items[0]).unwrap_or_default();
        let y =
          crate::functions::chart::expr_to_label(&items[1]).unwrap_or_default();
        plot_opts.axes_label = Some((x, y));
      }
    }
    "PlotStyle" => {
      plot_opts.plot_style = parse_plot_style(replacement);
    }
    "PlotTheme" => {
      if let Expr::String(theme) = replacement {
        apply_plot_theme(plot_opts, theme);
      }
    }
    "GridLines" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      let (sx, sy) = parse_grid_lines_spec(&val);
      apply_grid_side(
        sx,
        &mut plot_opts.grid_lines_x,
        &mut plot_opts.grid_x_lines,
      );
      apply_grid_side(
        sy,
        &mut plot_opts.grid_lines_y,
        &mut plot_opts.grid_y_lines,
      );
    }
    "GridLinesStyle" => {
      plot_opts.grid_lines_style = parse_style_directives(replacement);
    }
    "PlotRange" => {
      let (rx, ry) = parse_plot_range(replacement);
      overrides.x = rx;
      overrides.y = ry;
      overrides.y_all = plot_range_requests_all_y(replacement);
    }
    "Axes" => {
      if let Some(axes) = parse_axes_option(replacement) {
        plot_opts.axes = axes;
      }
    }
    "AspectRatio" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Some(r) = try_eval_to_f64(&val)
        && r > 0.0
      {
        overrides.aspect_ratio = Some(r);
      }
    }
    "PlotPoints" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      match &val {
        Expr::Integer(n) if *n > 0 => plot_opts.plot_points = *n as usize,
        _ => {}
      }
    }
    "Filling" => {
      apply_filling_option(replacement, plot_opts);
    }
    "FillingStyle" => {
      plot_opts.filling_style = parse_filling_style(replacement);
    }
    "Background" => {
      plot_opts.background = parse_background_option(replacement);
    }
    "Frame" => plot_opts.frame = parse_frame_option(replacement),
    "Evaluated" => {
      plot_opts.evaluated =
        matches!(replacement, Expr::Identifier(v) if v == "True");
    }
    "ImagePadding" => {
      plot_opts.image_padding = parse_image_padding(replacement);
    }
    "FrameLabel" => apply_frame_label_option(replacement, plot_opts),
    "LabelStyle" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      let (color, font_size) = parse_label_style(&val);
      if color.is_some() || font_size.is_some() {
        plot_opts.label_style = Some(LabelStyleSpec { color, font_size });
      }
    }
    "Ticks" => match replacement {
      Expr::Identifier(s) if s == "None" => plot_opts.ticks = false,
      Expr::Identifier(s) if s == "Automatic" || s == "All" => {
        plot_opts.ticks = true;
      }
      // `Ticks -> {xspec, yspec}`: each side is None, Automatic, or an
      // explicit list of positions (each optionally `{pos, label}`).
      // The y side may be left off — `Ticks -> {xspec}` states the x ticks
      // and leaves the y axis to its default.
      Expr::List(items) if (1..=2).contains(&items.len()) => {
        plot_opts.ticks_x = parse_explicit_ticks(&items[0]);
        if let Some(y) = items.get(1) {
          plot_opts.ticks_y = parse_explicit_ticks(y);
        }
      }
      _ => {}
    },
    "Epilog" => {
      // Evaluate now so primitives inside (e.g. `Line[{ReIm[…], …}]`)
      // resolve to numeric coordinates while the surrounding variable
      // bindings (Block scope) are still live.
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      plot_opts.epilog = match val {
        Expr::List(ref items) => items.to_vec(),
        Expr::Identifier(ref s) if s == "None" => Vec::new(),
        other => vec![other],
      };
    }
    _ => return false,
  }
  true
}

/// Build the compactifying substitution `Tan[Pi*inner/2]`, the bijection
/// used to fold an infinite plot range into a finite display coordinate.
fn tan_compactify(inner: Expr) -> Expr {
  Expr::FunctionCall {
    name: "Tan".to_string(),
    args: vec![Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Times,
        left: Box::new(Expr::Identifier("Pi".to_string())),
        right: Box::new(inner),
      }),
      right: Box::new(Expr::Integer(2)),
    }]
    .into(),
  }
}

/// Given the (possibly infinite) raw endpoints of a `Plot` range, return the
/// finite display range plus, when an endpoint is infinite, a substitution
/// `var -> g(var)` mapping the finite display coordinate back onto the real
/// line. This mirrors Wolfram, which plots e.g. `{x, -Infinity, Infinity}`
/// over the compact display range `{-1, 1}`.
///
/// - `{-Infinity, Infinity}` -> `[-1, 1]`,   `x = Tan[Pi u/2]`
/// - `{a, Infinity}`         -> `[a, a+1]`,  `x = a + Tan[Pi (u-a)/2]`
/// - `{-Infinity, b}`        -> `[b-1, b]`,  `x = b - Tan[Pi (b-u)/2]`
fn compactify_plot_range(
  var: &str,
  raw_min: f64,
  raw_max: f64,
) -> Result<(f64, f64, Option<Expr>), InterpreterError> {
  let min_inf = raw_min == f64::NEG_INFINITY;
  let max_inf = raw_max == f64::INFINITY;
  let u = || Expr::Identifier(var.to_string());
  match (min_inf, max_inf) {
    (false, false) => {
      if !raw_min.is_finite() || !raw_max.is_finite() {
        return Err(InterpreterError::EvaluationError(
          "Plot: range endpoints must be finite or ±Infinity".into(),
        ));
      }
      Ok((raw_min, raw_max, None))
    }
    // {-Infinity, Infinity}: x = Tan[Pi u/2] over u in [-1, 1]
    (true, true) => Ok((-1.0, 1.0, Some(tan_compactify(u())))),
    // {a, Infinity}: x = a + Tan[Pi (u-a)/2] over u in [a, a+1]
    (false, true) => {
      let a = raw_min;
      let inner = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(u()),
        right: Box::new(Expr::Real(a)),
      };
      let transform = Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Real(a)),
        right: Box::new(tan_compactify(inner)),
      };
      Ok((a, a + 1.0, Some(transform)))
    }
    // {-Infinity, b}: x = b - Tan[Pi (b-u)/2] over u in [b-1, b]
    (true, false) => {
      let b = raw_max;
      let inner = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Real(b)),
        right: Box::new(u()),
      };
      let transform = Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Real(b)),
        right: Box::new(tan_compactify(inner)),
      };
      Ok((b - 1.0, b, Some(transform)))
    }
  }
}

/// Returns true if `expr` textually references the identifier `var`. Used by
/// the plotting heads to decide whether a held body (which may be a bare
/// assigned symbol) still needs to be evaluated to surface the plot variable.
pub(crate) fn expr_mentions_var(expr: &Expr, var: &str) -> bool {
  // Substituting the variable with a sentinel changes the tree iff the
  // variable actually occurs; comparing structurally avoids hand-writing a
  // walker over every Expr variant.
  let sentinel = Expr::Identifier("$WoxiPlotVarProbe$".to_string());
  let replaced = crate::syntax::substitute_variable(expr, var, &sentinel);
  !crate::evaluator::pattern_matching::expr_equal(&replaced, expr)
}

/// Peel presentation wrappers (`Highlighted`, `Tooltip`, `Legended`,
/// `Labeled`) off a plot body, returning the inner plottable expression. These
/// heads annotate a curve — a highlight marker, a hover tooltip, a legend
/// entry, an inline label — but the function actually sampled is always their
/// first argument. The wrappers may enclose either a single function or a whole
/// list of them (e.g. `Highlighted[{Sin[x], Cos[x]}, …]`), so peeling happens
/// both before and after the body is split into individual curves. Callout is
/// intentionally left for the dedicated per-curve handling that records its
/// label.
fn peel_plot_wrappers(mut e: &Expr) -> &Expr {
  while let Expr::FunctionCall { name, args } = e {
    if !args.is_empty()
      && matches!(
        name.as_str(),
        "Highlighted" | "Tooltip" | "Legended" | "Labeled"
      )
    {
      e = &args[0];
    } else {
      break;
    }
  }
  e
}

/// Evaluate a plot body once with `var` kept symbolic, so assigned symbols
/// (e.g. `f = Sin[x]`) expand to their definitions without the plot variable
/// being replaced by any global value it might carry.
fn eval_body_var_symbolic(body: &Expr, var: &str) -> Expr {
  eval_body_vars_symbolic(body, &[var])
}

/// Expand a held generator such as `Table[f[i, x], {i, …}]` into the list of
/// curves it produces. The plot body is held, so such a generator only takes
/// list shape once evaluated; Wolfram plots one curve per element of the
/// result. Returns `None` when the body is already a list or does not
/// generate one, leaving plain `Plot[f[x], …]` untouched.
fn expand_generated_bodies(body: &Expr, var: &str) -> Option<Expr> {
  if matches!(body, Expr::List(_)) {
    return None;
  }
  match eval_body_var_symbolic(body, var) {
    evaluated @ Expr::List(_) => Some(evaluated),
    _ => None,
  }
}

/// As [`eval_body_var_symbolic`], for a body in several plot variables.
fn eval_body_vars_symbolic(body: &Expr, vars: &[&str]) -> Expr {
  let saved: Vec<(&str, Option<crate::StoredValue>)> = vars
    .iter()
    .map(|v| (*v, crate::ENV.with(|e| e.borrow_mut().remove(*v))))
    .collect();
  let result = evaluate_expr_to_expr(body).unwrap_or_else(|_| body.clone());
  for (var, value) in saved {
    if let Some(v) = value {
      crate::ENV.with(|e| {
        e.borrow_mut().insert(var.to_string(), v);
      });
    }
  }
  result
}

/// Resolve a plot body that names part of itself indirectly — `p = a x^2 +
/// b x; RegionPlot[y > p, …]`. The plotting heads hold their arguments, so
/// `p` arrives unevaluated and substituting the plot variables into it finds
/// nothing to replace; the body has to be evaluated once, with those
/// variables cleared, before it can be sampled.
///
/// Returns `None` — leaving the body untouched, so ordinary plots are
/// unaffected — unless evaluating it brings in a plot variable that was not
/// there before.
pub(crate) fn resolve_indirect_plot_body(
  body: &Expr,
  vars: &[&str],
) -> Option<Expr> {
  let mentioned =
    |e: &Expr| vars.iter().filter(|v| expr_mentions_var(e, v)).count();
  let before = mentioned(body);
  if before == vars.len() {
    return None;
  }
  let evaluated = eval_body_vars_symbolic(body, vars);
  (mentioned(&evaluated) > before).then_some(evaluated)
}

/// Implementation of Plot[f, {x, xmin, xmax}]
pub fn plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Plot requires at least 2 arguments: Plot[f, {x, xmin, xmax}]".into(),
    ));
  }

  let iter_spec = &args[1];

  // Parse options (Rule expressions after the first two arguments).
  // Matching Wolfram Language, the first occurrence of a repeated option
  // wins (e.g. `PlotPoints -> 200, …, PlotPoints -> 10` uses 200).
  let mut plot_opts = PlotOptions::default();
  // PlotRange and AspectRatio are applied after parsing all options
  let mut overrides = PlotRangeOverrides::default();
  let mut legends_automatic = false;
  let mut legends_expressions = false;
  // `EvaluationMonitor :> expr` — evaluated (with the plot variable bound)
  // at every sampled point, e.g. to `Sow` the sample locations.
  let mut monitor: Option<Expr> = None;
  let mut seen: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "EvaluationMonitor")
    {
      if monitor.is_none() {
        monitor = Some(replacement.as_ref().clone());
      }
      continue;
    }
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      if !seen.insert(name.to_string()) {
        continue;
      }
      if apply_common_plot_option(
        name,
        replacement,
        &mut plot_opts,
        &mut overrides,
      ) {
        continue;
      }
      if name == "PlotLegends" {
        let (labels, auto, expressions, position) =
          parse_plot_legends(replacement);
        plot_opts.legend_position = position;
        if auto {
          legends_automatic = true;
        } else if expressions {
          legends_expressions = true;
        } else {
          plot_opts.plot_legends = labels;
        }
      }
    }
  }
  let (plot_range_x, plot_range_y) = (overrides.x, overrides.y);
  let plot_range_y_all = overrides.y_all;

  // Apply AspectRatio to the plotting area (not the whole image); the total
  // height is derived in generate_svg_with_options once margins are known.
  // (AspectRatio is height/width in Wolfram Language.)
  if let Some(ar) = overrides.aspect_ratio {
    plot_opts.aspect_ratio = Some(ar);
  }

  // Parse iterator spec: {x, xmin, xmax}. Endpoints may be ±Infinity, in
  // which case the range is compactified onto a finite display coordinate
  // and `x_transform` folds that coordinate back onto the real line.
  let (var_name, x_min, x_max, x_transform) = match iter_spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Plot: iterator variable must be a symbol".into(),
          ));
        }
      };
      // Evaluate xmin and xmax (accepting ±Infinity)
      let x_min_expr = evaluate_expr_to_expr(&items[1])?;
      let x_max_expr = evaluate_expr_to_expr(&items[2])?;
      let raw_min =
        crate::functions::math_ast::try_eval_to_f64_with_infinity(&x_min_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Plot: cannot evaluate xmin to a number".into(),
            )
          })?;
      let raw_max =
        crate::functions::math_ast::try_eval_to_f64_with_infinity(&x_max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Plot: cannot evaluate xmax to a number".into(),
            )
          })?;
      let (x_min, x_max, transform) =
        compactify_plot_range(&var, raw_min, raw_max)?;
      (var, x_min, x_max, transform)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Plot: second argument must be {x, xmin, xmax}".into(),
      ));
    }
  };

  // When the range was infinite, substitute the compactifying map into the
  // function body so sampling the finite display coordinate covers the whole
  // real line. `body_storage` owns the rewritten body for the borrow below.
  let body_storage;
  let body: &Expr = if let Some(t) = &x_transform {
    body_storage = crate::syntax::substitute_variable(&args[0], &var_name, t);
    &body_storage
  } else {
    &args[0]
  };

  // Plot is HoldAll, so a body passed as a bare symbol (e.g.
  // `f = Sin[x]; Plot[f, {x, 0, 10}]`) reaches here unevaluated. Sampling
  // substitutes the plot variable into the body and then evaluates, but a
  // held `f` contains no `x` to substitute, so the sampler would evaluate the
  // symbol `f` back to `Sin[x]` with `x` still symbolic — never a number —
  // yielding an all-white plot. When the body does not textually reference
  // the plot variable, evaluate it once with that variable kept symbolic so
  // assigned symbols expand to their definitions (and reveal the variable)
  // before sampling. Bodies that already mention the variable are left
  // untouched, so ordinary plots are unaffected.
  // `Evaluated -> True` asks for the body to be worked out once, with the
  // plot variable still symbolic, instead of at every sample point. It is
  // how a body that only makes sense symbolically gets plotted at all —
  // `Plot[Normal[NonlinearModelFit[data, …, y]], {y, 0, 1}]` would
  // otherwise re-fit with a number in place of `y` at every point.
  let evaluated_storage;
  let body: &Expr = if plot_opts.evaluated {
    evaluated_storage = eval_body_var_symbolic(body, &var_name);
    &evaluated_storage
  } else {
    body
  };

  let resolved_storage;
  let body: &Expr = if expr_mentions_var(body, &var_name) {
    body
  } else {
    let evaluated = eval_body_var_symbolic(body, &var_name);
    if expr_mentions_var(&evaluated, &var_name) {
      resolved_storage = evaluated;
      &resolved_storage
    } else {
      body
    }
  };

  // Peel a presentation wrapper that encloses the whole plot spec so that a
  // list inside it (e.g. `Highlighted[{Sin[x], Cos[x]}, …]`) is split into
  // individual curves below rather than sampled as one non-numeric expression.
  let body = peel_plot_wrappers(body);

  // `Plot[Table[f[i, x], {i, …}], …]` names its curves only after the
  // generator runs; expand it so each element becomes its own curve
  // instead of being sampled as one expression yielding a list.
  let generated_storage;
  let body: &Expr = match expand_generated_bodies(body, &var_name) {
    Some(expanded) => {
      generated_storage = expanded;
      &generated_storage
    }
    None => body,
  };

  // Collect function bodies: a single function or a (possibly nested) list of
  // functions. Wolfram flattens nested lists into individual curves, so
  // `Plot[{{f}, {g}}, …]` draws two curves. Nesting arises naturally from
  // idioms like `expr /. C[1] -> Range[…]`, where threading a replacement over
  // several expressions yields a list of lists.
  fn flatten_plot_bodies<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::List(items) => {
        for it in items {
          flatten_plot_bodies(it, out);
        }
      }
      _ => out.push(e),
    }
  }
  let mut raw_bodies: Vec<&Expr> = Vec::new();
  flatten_plot_bodies(body, &mut raw_bodies);

  // Unwrap Callout[expr, label] wrappers, storing labels. Other presentation
  // wrappers (Highlighted/Tooltip/Legended/Labeled) around a single curve are
  // peeled first so their inner function is what gets sampled.
  let mut bodies: Vec<&Expr> = Vec::with_capacity(raw_bodies.len());
  for b in &raw_bodies {
    let b = peel_plot_wrappers(b);
    if let Expr::FunctionCall { name, args: cargs } = b
      && name == "Callout"
      && cargs.len() >= 2
    {
      bodies.push(&cargs[0]);
      let label = match &cargs[1] {
        Expr::String(s) => s.clone(),
        other => crate::syntax::expr_to_output(other),
      };
      plot_opts.callout_labels.push(Some(label));
    } else {
      bodies.push(b);
      plot_opts.callout_labels.push(None);
    }
  }

  // Fill automatic legends from expression strings
  if (legends_automatic || legends_expressions)
    && plot_opts.plot_legends.is_empty()
  {
    for b in &bodies {
      plot_opts
        .plot_legends
        .push(crate::syntax::expr_to_output(b));
    }
  }

  // Adaptive sampling: start with initial points, then refine where needed
  let initial_samples = plot_opts.plot_points.clamp(2, 200);
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
      monitor.as_ref(),
    );
    all_points.push(points);
  }

  // A `PlotStyle` list applied to a single curve is one combined directive
  // set, not a per-curve cycle: `Plot[f, …, PlotStyle -> {Thick, Green}]`
  // draws one thick green curve, not a `Thick` curve and a `Green` one.
  if all_points.len() == 1 {
    plot_opts.plot_style =
      collapse_style_for_single_series(&plot_opts.plot_style);
  }

  // Compute Y range using robust outlier exclusion on uniform samples —
  // unless `PlotRange -> All` asked for every sampled value to be shown.
  let (y_data_min, y_data_max) =
    sampled_y_range(&bodies, &var_name, x_min, x_max, plot_range_y_all);

  // Check if we have any plottable data
  let has_finite = all_points
    .iter()
    .any(|pts| pts.iter().any(|(_, y)| y.is_finite()));
  if !has_finite {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

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
    adjust_y_range_for_filling_opts(&plot_opts, (y_auto_min, y_auto_max));

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
    plot_opts.filling,
    plot_opts.filling_style,
    crate::functions::plot::explicit_options(args),
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
  let mut legends_expressions = false;
  for opt in &args[2..] {
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      match name {
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
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
            plot_opts.plot_label = Some(sl);
          }
        }
        "AxesLabel" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
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
          plot_opts.plot_style = parse_plot_style(replacement);
        }
        "PlotTheme" => {
          if let Expr::String(theme) = replacement {
            apply_plot_theme(&mut plot_opts, theme);
          }
        }
        "GridLines" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          let (sx, sy) = parse_grid_lines_spec(&val);
          apply_grid_side(
            sx,
            &mut plot_opts.grid_lines_x,
            &mut plot_opts.grid_x_lines,
          );
          apply_grid_side(
            sy,
            &mut plot_opts.grid_lines_y,
            &mut plot_opts.grid_y_lines,
          );
        }
        "GridLinesStyle" => {
          plot_opts.grid_lines_style = parse_style_directives(replacement);
        }
        "PlotRange" => {
          let (_rx, ry) = parse_plot_range(replacement);
          if ry.is_some() {
            plot_range_y = ry;
          }
        }
        "PlotPoints" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          if let Expr::Integer(n) = &val
            && *n > 0
          {
            plot_opts.plot_points = *n as usize;
          }
        }
        "Filling" => {
          apply_filling_option(replacement, &mut plot_opts);
        }
        "FillingStyle" => {
          plot_opts.filling_style = parse_filling_style(replacement);
        }
        "Background" => {
          plot_opts.background = parse_background_option(replacement);
        }
        "PlotLegends" => {
          let (labels, auto, expressions, position) =
            parse_plot_legends(replacement);
          plot_opts.legend_position = position;
          if auto {
            legends_automatic = true;
          } else if expressions {
            legends_expressions = true;
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

  // A held generator such as `Table[f[i, x], {i, …}]` only takes list
  // shape once evaluated; Wolfram plots one curve per element of the
  // resulting list, so evaluate a non-list body (the plot variable stays
  // symbolic) and use the list when there is one. Bodies that stay
  // scalar keep their held form, so plain `Plot[f[x], …]` is unaffected.
  let expanded_body = expand_generated_bodies(body, &var_name);

  // Collect function bodies
  let bodies: Vec<&Expr> = match expanded_body.as_ref().unwrap_or(body) {
    Expr::List(items) => items.iter().collect(),
    scalar => vec![scalar],
  };

  if (legends_automatic || legends_expressions)
    && plot_opts.plot_legends.is_empty()
  {
    for b in &bodies {
      plot_opts
        .plot_legends
        .push(crate::syntax::expr_to_output(b));
    }
  }

  let num_samples = plot_opts.plot_points.clamp(2, 2000);
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

  if all_points.len() == 1 {
    plot_opts.plot_style =
      collapse_style_for_single_series(&plot_opts.plot_style);
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

  let x_min_display = finite_xs.iter().copied().fold(f64::INFINITY, f64::min);
  let x_max_display =
    finite_xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let y_data_min = finite_ys.iter().copied().fold(f64::INFINITY, f64::min);
  let y_data_max = finite_ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);

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
    adjust_y_range_for_filling_opts(&plot_opts, (y_auto_min, y_auto_max));

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
