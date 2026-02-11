use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;
use crate::InterpreterError;

/// Plot dimensions
const SVG_WIDTH: f64 = 360.0;
const SVG_HEIGHT: f64 = 225.0;
const MARGIN_LEFT: f64 = 55.0;
const MARGIN_RIGHT: f64 = 10.0;
const MARGIN_TOP: f64 = 10.0;
const MARGIN_BOTTOM: f64 = 35.0;
const PLOT_WIDTH: f64 = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const PLOT_HEIGHT: f64 = SVG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
const NUM_SAMPLES: usize = 500;

/// Substitute all occurrences of a variable with a value in an expression
fn substitute_var(expr: &Expr, var: &str, value: &Expr) -> Expr {
  match expr {
    Expr::Identifier(name) if name == var => value.clone(),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| substitute_var(a, var, value))
        .collect(),
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

/// Generate nice tick values for an axis
fn nice_ticks(min: f64, max: f64, target_count: usize) -> Vec<f64> {
  if (max - min).abs() < f64::EPSILON {
    return vec![min];
  }
  let range = max - min;
  let rough_step = range / target_count as f64;
  let mag = 10.0_f64.powf(rough_step.log10().floor());
  let normalized = rough_step / mag;
  let nice_step = if normalized < 1.5 {
    1.0
  } else if normalized < 3.0 {
    2.0
  } else if normalized < 7.0 {
    5.0
  } else {
    10.0
  };
  let step = nice_step * mag;
  let start = (min / step).ceil() * step;
  let mut ticks = vec![];
  let mut tick = start;
  while tick <= max + step * 0.001 {
    ticks.push(tick);
    tick += step;
  }
  ticks
}

/// Format a tick label value
fn format_tick_label(val: f64) -> String {
  if (val - val.round()).abs() < 1e-10 {
    format!("{}", val.round() as i64)
  } else if (val * 10.0 - (val * 10.0).round()).abs() < 1e-9 {
    format!("{:.1}", val)
  } else {
    format!("{:.2}", val)
  }
}

/// Generate SVG for a 2D plot
fn generate_svg(
  points: &[(f64, f64)],
  x_range: (f64, f64),
  y_range: (f64, f64),
) -> String {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;

  let to_svg_x = |x: f64| -> f64 {
    MARGIN_LEFT + (x - x_min) / (x_max - x_min) * PLOT_WIDTH
  };
  let to_svg_y = |y: f64| -> f64 {
    MARGIN_TOP + PLOT_HEIGHT - (y - y_min) / (y_max - y_min) * PLOT_HEIGHT
  };

  let mut svg = String::new();
  svg.push_str(&format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" \
     width=\"{}\" height=\"{}\" \
     viewBox=\"0 0 {} {}\">",
    SVG_WIDTH as i32, SVG_HEIGHT as i32, SVG_WIDTH as i32, SVG_HEIGHT as i32,
  ));
  svg.push('\n');

  // White background for plot area
  svg.push_str(&format!(
    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"white\"/>",
    MARGIN_LEFT, MARGIN_TOP, PLOT_WIDTH, PLOT_HEIGHT,
  ));
  svg.push('\n');

  // Frame
  svg.push_str(&format!(
    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" \
     fill=\"none\" stroke=\"#333\" stroke-width=\"0.5\"/>",
    MARGIN_LEFT, MARGIN_TOP, PLOT_WIDTH, PLOT_HEIGHT,
  ));
  svg.push('\n');

  // X axis ticks and labels
  let x_ticks = nice_ticks(x_min, x_max, 5);
  for &tick in &x_ticks {
    let sx = to_svg_x(tick);
    let tick_bottom = MARGIN_TOP + PLOT_HEIGHT;
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
       stroke=\"#333\" stroke-width=\"0.5\"/>",
      sx,
      tick_bottom,
      sx,
      tick_bottom + 5.0,
    ));
    svg.push('\n');
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"Arial,sans-serif\" font-size=\"11\" \
       fill=\"#333\">{}</text>",
      sx,
      tick_bottom + 18.0,
      format_tick_label(tick),
    ));
    svg.push('\n');
  }

  // Y axis ticks and labels
  let y_ticks = nice_ticks(y_min, y_max, 4);
  for &tick in &y_ticks {
    let sy = to_svg_y(tick);
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
       stroke=\"#333\" stroke-width=\"0.5\"/>",
      MARGIN_LEFT - 5.0,
      sy,
      MARGIN_LEFT,
      sy,
    ));
    svg.push('\n');
    svg.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" \
       font-family=\"Arial,sans-serif\" font-size=\"11\" \
       fill=\"#333\">{}</text>",
      MARGIN_LEFT - 8.0,
      sy + 4.0,
      format_tick_label(tick),
    ));
    svg.push('\n');
  }

  // Build path segments, breaking on NaN/Infinity
  let mut path_parts: Vec<Vec<(f64, f64)>> = Vec::new();
  let mut current_segment: Vec<(f64, f64)> = Vec::new();

  for &(x, y) in points {
    if y.is_finite() {
      let sx = to_svg_x(x);
      let sy = to_svg_y(y);
      // Clamp to plot area
      let sy_clamped = sy.max(MARGIN_TOP).min(MARGIN_TOP + PLOT_HEIGHT);
      current_segment.push((sx, sy_clamped));
    } else {
      if current_segment.len() > 1 {
        path_parts.push(std::mem::take(&mut current_segment));
      } else {
        current_segment.clear();
      }
    }
  }
  if current_segment.len() > 1 {
    path_parts.push(current_segment);
  }

  // Draw each segment as an SVG path
  for segment in &path_parts {
    let mut d = String::new();
    for (i, &(sx, sy)) in segment.iter().enumerate() {
      if i == 0 {
        d.push_str(&format!("M{:.2},{:.2}", sx, sy));
      } else {
        d.push_str(&format!("L{:.2},{:.2}", sx, sy));
      }
    }
    svg.push_str(&format!(
      "<path d=\"{}\" fill=\"none\" stroke=\"#5E81B5\" \
       stroke-width=\"1.5\" stroke-linejoin=\"round\" \
       stroke-linecap=\"round\"/>",
      d,
    ));
    svg.push('\n');
  }

  svg.push_str("</svg>");
  svg
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
  let y_data_max = finite_ys
    .iter()
    .cloned()
    .fold(f64::NEG_INFINITY, f64::max);

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
  let svg = generate_svg(&points, (x_min, x_max), (y_min, y_max));

  // Store the SVG for capture by the Jupyter kernel
  crate::capture_graphics(&svg);

  // Return -Graphics- as the text representation
  Ok(Expr::Identifier("-Graphics-".to_string()))
}
