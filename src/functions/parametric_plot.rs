use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, NUM_SAMPLES, evaluate_at_point, generate_svg,
  parse_image_size,
};
use crate::syntax::Expr;

/// ParametricPlot[{fx[t], fy[t]}, {t, tmin, tmax}]
pub fn parametric_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  // Parse options
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

  // Parse iterator: {t, tmin, tmax}
  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "ParametricPlot")?;

  // Collect curve bodies: either {fx, fy} or {{fx1,fy1}, {fx2,fy2}, ...}
  let curves: Vec<(&Expr, &Expr)> = match body {
    Expr::List(items) if items.len() == 2 => {
      // Check if it's {fx, fy} (single curve) or {{fx1,fy1}, {fx2,fy2}}
      if matches!(&items[0], Expr::List(inner) if inner.len() == 2) {
        // Multiple curves: {{fx1,fy1}, {fx2,fy2}, ...}
        items
          .iter()
          .filter_map(|item| {
            if let Expr::List(pair) = item
              && pair.len() == 2
            {
              return Some((&pair[0], &pair[1]));
            }
            None
          })
          .collect()
      } else {
        // Single curve: {fx, fy}
        vec![(&items[0], &items[1])]
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ParametricPlot: first argument must be {fx, fy}".into(),
      ));
    }
  };

  let step = (t_max - t_min) / (NUM_SAMPLES - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(curves.len());

  for (fx, fy) in &curves {
    let mut points = Vec::with_capacity(NUM_SAMPLES);
    for i in 0..NUM_SAMPLES {
      let t = t_min + i as f64 * step;
      if let (Some(x), Some(y)) = (
        evaluate_at_point(fx, &var_name, t),
        evaluate_at_point(fy, &var_name, t),
      ) {
        points.push((x, y));
      } else {
        points.push((f64::NAN, f64::NAN));
      }
    }
    all_points.push(points);
  }

  // Compute ranges
  let (x_range, y_range) = compute_data_ranges(&all_points);

  // Adjust aspect ratio to match data (so circles render round)
  let data_w = x_range.1 - x_range.0;
  let data_h = y_range.1 - y_range.0;
  if data_w > 0.0 && data_h > 0.0 {
    let data_aspect = data_h / data_w;
    if data_aspect.is_finite() {
      svg_height = (svg_width as f64 * data_aspect).round() as u32;
    }
  }

  let svg = generate_svg(
    &all_points,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;

  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// PolarPlot[r[theta], {theta, tmin, tmax}]
pub fn polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "PolarPlot requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  // Parse options
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

  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "PolarPlot")?;

  // Collect function bodies
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  let step = (t_max - t_min) / (NUM_SAMPLES - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let mut points = Vec::with_capacity(NUM_SAMPLES);
    for i in 0..NUM_SAMPLES {
      let theta = t_min + i as f64 * step;
      if let Some(r) = evaluate_at_point(func_body, &var_name, theta) {
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push((x, y));
      } else {
        points.push((f64::NAN, f64::NAN));
      }
    }
    all_points.push(points);
  }

  let (x_range, y_range) = compute_data_ranges(&all_points);

  // Adjust aspect ratio to match data (so circles render round)
  let data_w = x_range.1 - x_range.0;
  let data_h = y_range.1 - y_range.0;
  if data_w > 0.0 && data_h > 0.0 {
    let data_aspect = data_h / data_w;
    if data_aspect.is_finite() {
      svg_height = (svg_width as f64 * data_aspect).round() as u32;
    }
  }

  let svg = generate_svg(
    &all_points,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;

  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

fn parse_iterator(
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

/// Compute x/y ranges from point data with 4% padding.
fn compute_data_ranges(
  all_points: &[Vec<(f64, f64)>],
) -> ((f64, f64), (f64, f64)) {
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;

  for series in all_points {
    for &(x, y) in series {
      if x.is_finite() {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
      }
      if y.is_finite() {
        y_min = y_min.min(y);
        y_max = y_max.max(y);
      }
    }
  }

  if !x_min.is_finite() {
    x_min = -1.0;
    x_max = 1.0;
  }
  if !y_min.is_finite() {
    y_min = -1.0;
    y_max = 1.0;
  }

  let xr = x_max - x_min;
  let yr = y_max - y_min;
  let xp = if xr.abs() < f64::EPSILON {
    1.0
  } else {
    xr * 0.04
  };
  let yp = if yr.abs() < f64::EPSILON {
    1.0
  } else {
    yr * 0.04
  };

  ((x_min - xp, x_max + xp), (y_min - yp, y_max + yp))
}
