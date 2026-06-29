use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, NUM_SAMPLES, PlotOptions,
  adjust_y_range_for_filling, evaluate_at_point, generate_svg_with_filling,
  parse_filling, parse_image_size, substitute_var,
};
use crate::syntax::Expr;

/// A single parametric curve, sampled either component-wise (`{fx, fy}` where
/// both components are explicit expressions) or as a whole expression that
/// only yields a coordinate pair once the parameter is numeric (e.g.
/// `BSplineFunction[…][t]` or any `f[t]` returning `{x, y}`).
enum CurveSrc<'a> {
  Pair(&'a Expr, &'a Expr),
  Whole(&'a Expr),
}

/// Sample a whole-expression curve at parameter `t`, expecting the evaluated
/// body to be a two-element coordinate list.
fn sample_whole_pair(body: &Expr, var: &str, t: f64) -> Option<(f64, f64)> {
  let substituted = substitute_var(body, var, &Expr::Real(t));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  if let Expr::List(items) = &result
    && items.len() == 2
  {
    return Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?));
  }
  None
}

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
  let mut plot_opts = PlotOptions::default();
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
        "Filling" => {
          plot_opts.filling = parse_filling(replacement);
        }
        _ => {}
      }
    }
  }

  // Parse iterator: {t, tmin, tmax}
  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "ParametricPlot")?;

  // Collect curve bodies. Wolfram accepts either a single curve
  // `{fx, fy}` or any number of curves `{{fx1,fy1}, {fx2,fy2}, …}`.
  // The previous form-check required `items.len() == 2`, which
  // rejected 3+ curves; relax it to inspect the *inner* shape so
  // arbitrarily many curves work (regression for mathics doc-044
  // `ParametricPlot[{{Sin[u], Cos[u]}, {0.6 Sin[u], 0.6 Cos[u]},
  // {0.2 Sin[u], 0.2 Cos[u]}}, {u, 0, 2 Pi}, …]`).
  let curves: Vec<CurveSrc> = match body {
    Expr::List(items) if !items.is_empty() => {
      // If every item is a 2-element list, treat as multi-curve.
      let all_pairs = items
        .iter()
        .all(|i| matches!(i, Expr::List(inner) if inner.len() == 2));
      if all_pairs && items.len() != 2 {
        // Clearly multi-curve (3+ curves).
        items
          .iter()
          .filter_map(|item| {
            if let Expr::List(pair) = item
              && pair.len() == 2
            {
              return Some(CurveSrc::Pair(&pair[0], &pair[1]));
            }
            None
          })
          .collect()
      } else if items.len() == 2 {
        if matches!(&items[0], Expr::List(inner) if inner.len() == 2)
          && matches!(&items[1], Expr::List(inner) if inner.len() == 2)
        {
          // Ambiguous-looking `{{a,b}, {c,d}}` resolves to two curves.
          items
            .iter()
            .filter_map(|item| {
              if let Expr::List(pair) = item
                && pair.len() == 2
              {
                return Some(CurveSrc::Pair(&pair[0], &pair[1]));
              }
              None
            })
            .collect()
        } else {
          // Single curve: {fx, fy}
          vec![CurveSrc::Pair(&items[0], &items[1])]
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "ParametricPlot: first argument must be {fx, fy}".into(),
        ));
      }
    }
    // A non-list body (e.g. `f[t]` or `BSplineFunction[…][t]`) is a curve
    // whose coordinate pair only materialises once `t` is numeric; sample
    // the whole expression instead of decomposing it into components.
    _ => vec![CurveSrc::Whole(body)],
  };

  let step = (t_max - t_min) / (NUM_SAMPLES - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(curves.len());

  for curve in &curves {
    let mut points = Vec::with_capacity(NUM_SAMPLES);
    for i in 0..NUM_SAMPLES {
      let t = t_min + i as f64 * step;
      let pt = match curve {
        CurveSrc::Pair(fx, fy) => match (
          evaluate_at_point(fx, &var_name, t),
          evaluate_at_point(fy, &var_name, t),
        ) {
          (Some(x), Some(y)) => Some((x, y)),
          _ => None,
        },
        CurveSrc::Whole(b) => sample_whole_pair(b, &var_name, t),
      };
      points.push(pt.unwrap_or((f64::NAN, f64::NAN)));
    }
    all_points.push(points);
  }

  // Compute ranges
  let (x_range, y_range) = compute_data_ranges(&all_points);
  let y_range = adjust_y_range_for_filling(plot_opts.filling, y_range);

  // Adjust aspect ratio to match data (so circles render round)
  let data_w = x_range.1 - x_range.0;
  let data_h = y_range.1 - y_range.0;
  if data_w > 0.0 && data_h > 0.0 {
    let data_aspect = data_h / data_w;
    if data_aspect.is_finite() {
      plot_opts.svg_height =
        (plot_opts.svg_width as f64 * data_aspect).round() as u32;
    }
  }

  let svg =
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?;

  Ok(crate::graphics_result(svg))
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
  let mut plot_opts = PlotOptions::default();
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
        "Filling" => {
          plot_opts.filling = parse_filling(replacement);
        }
        _ => {}
      }
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
  let y_range = adjust_y_range_for_filling(plot_opts.filling, y_range);

  // Adjust aspect ratio to match data (so circles render round)
  let data_w = x_range.1 - x_range.0;
  let data_h = y_range.1 - y_range.0;
  if data_w > 0.0 && data_h > 0.0 {
    let data_aspect = data_h / data_w;
    if data_aspect.is_finite() {
      plot_opts.svg_height =
        (plot_opts.svg_width as f64 * data_aspect).round() as u32;
    }
  }

  let svg =
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?;

  Ok(crate::graphics_result(svg))
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
