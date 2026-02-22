use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, Filling, generate_scatter_svg, generate_svg,
  generate_svg_with_filling, parse_image_size,
};
use crate::syntax::Expr;

/// Parse list data from the first argument.
/// Returns a vector of series, each series being a vector of (x, y) points.
///
/// Supported formats:
/// - `{y1, y2, y3}` → single series with x = 1,2,3,...
/// - `{{x1,y1}, {x2,y2}}` → single series with explicit coordinates
/// - `{{y1, y2}, {y3, y4}}` → multiple series (if inner lists don't look like points)
fn parse_list_data(
  arg: &Expr,
) -> Result<Vec<Vec<(f64, f64)>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListPlot: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![vec![]]);
  }

  // Check if it's a list of {x, y} pairs
  if let Expr::List(first_inner) = &items[0] {
    if first_inner.len() == 2
      && try_eval_to_f64(
        &evaluate_expr_to_expr(&first_inner[0])
          .unwrap_or(first_inner[0].clone()),
      )
      .is_some()
      && try_eval_to_f64(
        &evaluate_expr_to_expr(&first_inner[1])
          .unwrap_or(first_inner[1].clone()),
      )
      .is_some()
    {
      // {{x1,y1}, {x2,y2}, ...} → single series with explicit coords
      let mut points = Vec::with_capacity(items.len());
      for item in items {
        if let Expr::List(pair) = item
          && pair.len() == 2
        {
          let x_expr =
            evaluate_expr_to_expr(&pair[0]).unwrap_or(pair[0].clone());
          let y_expr =
            evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone());
          if let (Some(x), Some(y)) =
            (try_eval_to_f64(&x_expr), try_eval_to_f64(&y_expr))
          {
            points.push((x, y));
          }
        }
      }
      return Ok(vec![points]);
    }

    // Check if it's multiple series: {{y1, y2, ...}, {y3, y4, ...}}
    if first_inner.len() != 2
      || items
        .iter()
        .all(|item| matches!(item, Expr::List(l) if l.len() > 2))
    {
      let mut all_series = Vec::new();
      for item in items {
        if let Expr::List(series_items) = item {
          let mut points = Vec::with_capacity(series_items.len());
          for (i, val) in series_items.iter().enumerate() {
            let v_expr = evaluate_expr_to_expr(val).unwrap_or(val.clone());
            if let Some(y) = try_eval_to_f64(&v_expr) {
              points.push(((i + 1) as f64, y));
            }
          }
          all_series.push(points);
        }
      }
      if !all_series.is_empty() {
        return Ok(all_series);
      }
    }
  }

  // Simple list: {y1, y2, y3} → single series with x = 1,2,3,...
  let mut points = Vec::with_capacity(items.len());
  for (i, item) in items.iter().enumerate() {
    let v_expr = evaluate_expr_to_expr(item).unwrap_or(item.clone());
    if let Some(y) = try_eval_to_f64(&v_expr) {
      points.push(((i + 1) as f64, y));
    }
  }
  Ok(vec![points])
}

/// Parse common plot options from args[1..].
fn parse_plot_options(args: &[Expr]) -> (u32, u32, bool, bool, Filling) {
  let mut svg_width = DEFAULT_WIDTH;
  let mut svg_height = DEFAULT_HEIGHT;
  let mut full_width = false;
  let mut joined = false;
  let mut filling = Filling::None;
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) = parse_image_size(replacement) {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        "Joined" => {
          if matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
          {
            joined = true;
          }
        }
        "Filling" => {
          if matches!(replacement.as_ref(), Expr::Identifier(v) if v == "Axis")
          {
            filling = Filling::Axis;
          }
        }
        _ => {}
      }
    }
  }
  (svg_width, svg_height, full_width, joined, filling)
}

/// Compute x/y ranges from data with 4% padding.
fn compute_ranges(all_series: &[Vec<(f64, f64)>]) -> ((f64, f64), (f64, f64)) {
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;

  for series in all_series {
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

  if !x_min.is_finite() || !x_max.is_finite() {
    x_min = 0.0;
    x_max = 1.0;
  }
  if !y_min.is_finite() || !y_max.is_finite() {
    y_min = 0.0;
    y_max = 1.0;
  }

  let x_range = x_max - x_min;
  let y_range = y_max - y_min;
  let x_pad = if x_range.abs() < f64::EPSILON {
    1.0
  } else {
    x_range * 0.04
  };
  let y_pad = if y_range.abs() < f64::EPSILON {
    1.0
  } else {
    y_range * 0.04
  };

  (
    (x_min - x_pad, x_max + x_pad),
    (y_min - y_pad, y_max + y_pad),
  )
}

/// ListPlot[{y1, y2, ...}] or ListPlot[{{x1,y1}, ...}]
pub fn list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, joined, _filling) =
    parse_plot_options(args);
  let (x_range, y_range) = compute_ranges(&all_series);

  let svg = if joined {
    generate_svg(
      &all_series,
      x_range,
      y_range,
      svg_width,
      svg_height,
      full_width,
    )?
  } else {
    generate_scatter_svg(
      &all_series,
      x_range,
      y_range,
      svg_width,
      svg_height,
      full_width,
    )?
  };

  Ok(crate::graphics_result(svg))
}

/// ListLinePlot[{y1, y2, ...}]
pub fn list_line_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, filling) =
    parse_plot_options(args);
  let (x_range, mut y_range) = compute_ranges(&all_series);

  // When filling to axis, ensure y=0 is included in the range
  if filling == Filling::Axis {
    if y_range.0 > 0.0 {
      y_range.0 = 0.0 - (y_range.1 - 0.0) * 0.04;
    }
    if y_range.1 < 0.0 {
      y_range.1 = 0.0 + (0.0 - y_range.0) * 0.04;
    }
  }

  let svg = generate_svg_with_filling(
    &all_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
    filling,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ListStepPlot[{y1, y2, ...}]
pub fn list_step_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, _) = parse_plot_options(args);

  // Transform each series into staircase coordinates
  let step_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      let mut steps = Vec::with_capacity(series.len() * 2);
      for (i, &(x, y)) in series.iter().enumerate() {
        if i > 0 {
          steps.push((x, series[i - 1].1)); // horizontal to new x
        }
        steps.push((x, y)); // vertical to new y
      }
      steps
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&step_series);
  let svg = generate_svg(
    &step_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ListLogPlot: y-axis is log10 scale
pub fn list_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, _) = parse_plot_options(args);

  let log_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      series
        .iter()
        .filter_map(|&(x, y)| if y > 0.0 { Some((x, y.log10())) } else { None })
        .collect()
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&log_series);
  let svg = generate_svg(
    &log_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ListLogLogPlot: both axes log10 scale
pub fn list_log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, _) = parse_plot_options(args);

  let log_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      series
        .iter()
        .filter_map(|&(x, y)| {
          if x > 0.0 && y > 0.0 {
            Some((x.log10(), y.log10()))
          } else {
            None
          }
        })
        .collect()
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&log_series);
  let svg = generate_svg(
    &log_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ListLogLinearPlot: x-axis is log10 scale
pub fn list_log_linear_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, _) = parse_plot_options(args);

  let log_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      series
        .iter()
        .filter_map(|&(x, y)| if x > 0.0 { Some((x.log10(), y)) } else { None })
        .collect()
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&log_series);
  let svg = generate_svg(
    &log_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;
  Ok(crate::graphics_result(svg))
}

/// ListPolarPlot[{r1, r2, ...}]: plot data in polar coordinates
pub fn list_polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let (svg_width, svg_height, full_width, _, _) = parse_plot_options(args);

  let polar_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      let n = series.len();
      series
        .iter()
        .enumerate()
        .map(|(i, &(_x, r))| {
          let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
          (r * theta.cos(), r * theta.sin())
        })
        .collect()
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&polar_series);
  let svg = generate_svg(
    &polar_series,
    x_range,
    y_range,
    svg_width,
    svg_height,
    full_width,
  )?;
  Ok(crate::graphics_result(svg))
}
