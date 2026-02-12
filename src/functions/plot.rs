use plotters::prelude::*;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

const DEFAULT_WIDTH: u32 = 360;
const DEFAULT_HEIGHT: u32 = 225;
/// Internal rendering resolution multiplier for sub-pixel precision.
/// Plotters maps to integer coordinates, so we render at a higher resolution
/// and scale down via SVG viewBox to get smooth curves.
const RESOLUTION_SCALE: u32 = 10;
const NUM_SAMPLES: usize = 500;

/// Substitute all occurrences of a variable with a value in an expression
fn substitute_var(expr: &Expr, var: &str, value: &Expr) -> Expr {
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
fn evaluate_at_point(body: &Expr, var: &str, x: f64) -> Option<f64> {
  let substituted = substitute_var(body, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  try_eval_to_f64(&result)
}

/// Split points into contiguous finite segments, breaking at NaN/Infinity
fn split_into_segments(points: &[(f64, f64)]) -> Vec<Vec<(f64, f64)>> {
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
fn nice_step(range: f64, target_labels: usize) -> f64 {
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
fn is_major_tick(v: f64, step: f64) -> bool {
  if step == 0.0 {
    return true;
  }
  let remainder = (v / step).round() * step - v;
  remainder.abs() < step * 1e-9
}

/// Format a tick value, dropping the trailing ".0" for integers.
fn format_tick(v: f64) -> String {
  if (v - v.round()).abs() < 1e-9 {
    format!("{}", v.round() as i64)
  } else {
    format!("{v:.1}")
  }
}

/// Generate SVG for a 2D plot using plotters.
/// When `full_width` is true, the SVG uses `width="100%"` to fill its container.
fn generate_svg(
  points: &[(f64, f64)],
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
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
    let light_gray = RGBColor(0xCC, 0xCC, 0xCC);

    let mut chart = ChartBuilder::on(&root)
      .margin(10 * s)
      .x_label_area_size(25 * RESOLUTION_SCALE)
      .y_label_area_size(40 * RESOLUTION_SCALE)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Compute nice major tick step (~5 labels), then request enough ticks
    // for ~5 minor subdivisions between each major tick.
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

    let wolfram_blue = RGBColor(0x5E, 0x81, 0xB5);
    let segments = split_into_segments(points);

    for segment in &segments {
      chart
        .draw_series(std::iter::once(PathElement::new(
          segment.clone(),
          wolfram_blue.stroke_width(15), // 1.5px at display size
        )))
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    root
      .present()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  // Rewrite the opening <svg> tag to use viewBox so the high-resolution
  // internal coordinates are scaled down to the intended display size.
  if let Some(pos) = buf.find('>') {
    let new_header = if full_width {
      format!(
        "<svg width=\"100%\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\"",
        render_width, render_height,
      )
    } else {
      format!(
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\"",
        svg_width, svg_height, render_width, render_height,
      )
    };
    // Replace everything from <svg up to (but not including) the first >
    buf.replace_range(..pos, &new_header);
  }

  Ok(buf)
}

/// Parse ImageSize option value into (width, height, full_width).
/// Supports: integer, {w, h}, and named sizes (Tiny, Small, Medium, Large, Full).
/// Full uses a 720px render resolution but emits `width="100%"` in SVG.
fn parse_image_size(value: &Expr) -> Option<(u32, u32, bool)> {
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
  let mut svg_width = DEFAULT_WIDTH;
  let mut svg_height = DEFAULT_HEIGHT;
  let mut full_width = false;
  for opt in &args[2..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "ImageSize")
      && let Some((w, h, fw)) = parse_image_size(replacement)
    {
      svg_width = w;
      svg_height = h;
      full_width = fw;
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

  // Sample the function
  let mut points = Vec::with_capacity(NUM_SAMPLES);
  let step = (x_max - x_min) / (NUM_SAMPLES - 1) as f64;

  for i in 0..NUM_SAMPLES {
    let x = x_min + i as f64 * step;
    if let Some(y) = evaluate_at_point(body, &var_name, x) {
      points.push((x, y));
    } else {
      points.push((x, f64::NAN));
    }
  }

  // Compute Y range from finite values
  let finite_ys: Vec<f64> = points
    .iter()
    .filter(|(_, y)| y.is_finite())
    .map(|(_, y)| *y)
    .collect();

  if finite_ys.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Plot: function produced no finite values in the given range".into(),
    ));
  }

  let y_data_min = finite_ys.iter().cloned().fold(f64::INFINITY, f64::min);
  let y_data_max = finite_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

  // Add 4% padding
  let y_range = y_data_max - y_data_min;
  let padding = if y_range.abs() < f64::EPSILON {
    1.0
  } else {
    y_range * 0.04
  };
  let y_min = y_data_min - padding;
  let y_max = y_data_max + padding;

  // Generate SVG
  let svg = generate_svg(
    &points,
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  // Store the SVG for capture by the Jupyter kernel
  crate::capture_graphics(&svg);

  // Return -Graphics- as the text representation
  Ok(Expr::Identifier("-Graphics-".to_string()))
}
