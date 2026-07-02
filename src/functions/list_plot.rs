use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, Mesh, PlotOptions, adjust_y_range_for_filling,
  apply_plot_theme, build_plot_source, generate_scatter_svg_with_options,
  generate_svg_with_filling, parse_filling, parse_image_size,
  parse_plot_legends, parse_plot_style,
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

  // A TimeSeries / multi-path TemporalData becomes one (x, y) series per path.
  if let Some(paths) = crate::functions::timeseries_ast::temporal_paths(&data) {
    let series = paths
      .into_iter()
      .map(|path| {
        path
          .into_iter()
          .filter_map(|(t, v)| {
            let te = evaluate_expr_to_expr(&t).unwrap_or(t);
            let ve = evaluate_expr_to_expr(&v).unwrap_or(v);
            Some((try_eval_to_f64(&te)?, try_eval_to_f64(&ve)?))
          })
          .collect()
      })
      .collect();
    return Ok(series);
  }

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
    // or multiple series of explicit coords: {{{x1,y1}, ...}, {{x2,y2}, ...}}
    if first_inner.len() != 2
      || items
        .iter()
        .all(|item| matches!(item, Expr::List(l) if l.len() > 2))
    {
      let mut all_series = Vec::new();
      for item in items {
        if let Expr::List(series_items) = item {
          let points = parse_single_series(series_items)?;
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

/// Parse a single series: either plain y-values or {x, y} pairs.
fn parse_single_series(
  series_items: &[Expr],
) -> Result<Vec<(f64, f64)>, InterpreterError> {
  // Check if items are {x, y} pairs
  let is_xy_pairs = !series_items.is_empty()
    && series_items
      .iter()
      .all(|item| matches!(item, Expr::List(pair) if pair.len() == 2));

  let mut points = Vec::with_capacity(series_items.len());
  if is_xy_pairs {
    for item in series_items {
      if let Expr::List(pair) = item
        && pair.len() == 2
      {
        let x_expr = evaluate_expr_to_expr(&pair[0]).unwrap_or(pair[0].clone());
        let y_expr = evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone());
        if let (Some(x), Some(y)) =
          (try_eval_to_f64(&x_expr), try_eval_to_f64(&y_expr))
        {
          points.push((x, y));
        }
      }
    }
  } else {
    for (i, val) in series_items.iter().enumerate() {
      let v_expr = evaluate_expr_to_expr(val).unwrap_or(val.clone());
      if let Some(y) = try_eval_to_f64(&v_expr) {
        points.push(((i + 1) as f64, y));
      }
    }
  }
  Ok(points)
}

/// Parsed list-plot options, including explicit PlotRange overrides.
#[derive(Default)]
struct ParsedOptions {
  opts: PlotOptions,
  joined: bool,
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
}

/// Parse common plot options from args[1..].
fn parse_plot_options(args: &[Expr]) -> ParsedOptions {
  let mut out = ParsedOptions::default();
  let opts = &mut out.opts;
  for opt in &args[1..] {
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
            opts.svg_width = w;
            opts.svg_height = h;
            opts.full_width = fw;
          }
        }
        "Joined" => {
          if matches!(replacement.as_ref(), Expr::Identifier(v) if v == "True")
          {
            out.joined = true;
          }
        }
        "PlotRange" => {
          let (rx, ry) = crate::functions::plot::parse_plot_range(replacement);
          out.plot_range_x = rx;
          out.plot_range_y = ry;
        }
        "Filling" => {
          opts.filling = parse_filling(replacement);
        }
        "PlotStyle" => {
          opts.plot_style = parse_plot_style(replacement);
        }
        "PlotLegends" => {
          let (labels, _auto, _expressions, _pos) =
            parse_plot_legends(replacement);
          opts.plot_legends = labels;
        }
        "PlotLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
            opts.plot_label = Some(sl);
          }
        }
        // AxesLabel is always the 2-element `{x, y}` (bottom, left) form.
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
            opts.axes_label = Some((x, y));
          }
        }
        // FrameLabel supports both `{bottom, left}` and the nested
        // `{{left, right}, {bottom, top}}` form. Bottom/left reuse the
        // axes-label render path; top/right get their own frame edges.
        "FrameLabel" => {
          let fl = crate::functions::plot::parse_frame_label(replacement);
          opts.axes_label = Some((fl.bottom, fl.left));
          if !fl.top.is_empty() {
            opts.frame_label_top = Some(fl.top);
          }
          if !fl.right.is_empty() {
            opts.frame_label_right = Some(fl.right);
          }
        }
        "Frame" => {
          if matches!(replacement.as_ref(),
            Expr::Identifier(v) if v == "True" || v == "All")
          {
            opts.frame = true;
          }
        }
        "Mesh" => {
          if matches!(replacement.as_ref(), Expr::Identifier(v) if v == "All") {
            opts.mesh = Mesh::All;
          }
        }
        "PlotTheme" => {
          if let Expr::String(theme) = replacement.as_ref() {
            apply_plot_theme(opts, theme);
          }
        }
        "GridLines" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          let (sx, sy) = crate::functions::plot::parse_grid_lines_spec(&val);
          crate::functions::plot::apply_grid_side(
            sx,
            &mut opts.grid_lines_x,
            &mut opts.grid_x_lines,
          );
          crate::functions::plot::apply_grid_side(
            sy,
            &mut opts.grid_lines_y,
            &mut opts.grid_y_lines,
          );
        }
        _ => {}
      }
    }
  }
  out
}

/// Apply explicit PlotRange overrides from parsed options on top of the
/// auto-computed (x_range, y_range) pair. Axes that were not specified
/// (None) keep their auto-computed values.
fn apply_plot_range_override(
  parsed: &ParsedOptions,
  x_range: (f64, f64),
  y_range: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
  (
    parsed.plot_range_x.unwrap_or(x_range),
    parsed.plot_range_y.unwrap_or(y_range),
  )
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
  let parsed = parse_plot_options(args);
  let (x_range, y_range) = compute_ranges(&all_series);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let joined = parsed.joined;
  let opts = &parsed.opts;

  let svg = if joined {
    generate_svg_with_filling(&all_series, x_range, y_range, opts)?
  } else {
    generate_scatter_svg_with_options(&all_series, x_range, y_range, opts)?
  };

  let source = build_plot_source(
    &all_series,
    &opts.plot_style,
    x_range,
    y_range,
    (opts.svg_width, opts.svg_height),
    !joined,
    opts.filling,
  );
  Ok(crate::graphics_result_with_source(svg, source))
}

/// ListLinePlot[{y1, y2, ...}]
pub fn list_line_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let parsed = parse_plot_options(args);
  let (x_range, mut y_range) = compute_ranges(&all_series);

  y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);

  let svg =
    generate_svg_with_filling(&all_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// StackedListPlot[{list1, list2, ...}]: plot several datasets with their
/// y-values accumulated, so each successive dataset is stacked on top of the
/// running total of the preceding ones. The regions between consecutive
/// cumulative curves are filled by default.
pub fn stacked_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let mut parsed = parse_plot_options(args);
  parsed.opts.stacked = true;
  // Default to filled bands unless the user explicitly overrode Filling.
  if parsed.opts.filling == crate::functions::plot::Filling::None {
    parsed.opts.filling = crate::functions::plot::Filling::Axis;
  }

  // Accumulate y-values by index position across the series so that each
  // curve is the running total up to and including that dataset.
  let mut cumulative: Vec<Vec<(f64, f64)>> =
    Vec::with_capacity(all_series.len());
  let mut running: Vec<f64> = Vec::new();
  for series in &all_series {
    let mut curve = Vec::with_capacity(series.len());
    for (j, &(x, y)) in series.iter().enumerate() {
      if j >= running.len() {
        running.push(0.0);
      }
      running[j] += y;
      curve.push((x, running[j]));
    }
    cumulative.push(curve);
  }

  let (x_range, mut y_range) = compute_ranges(&cumulative);
  y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);

  let svg =
    generate_svg_with_filling(&cumulative, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListStepPlot[{y1, y2, ...}]
pub fn list_step_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let parsed = parse_plot_options(args);

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
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&step_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListLogPlot: y-axis is log10 scale
pub fn list_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let mut parsed = parse_plot_options(args);
  parsed.opts.log_y = true;

  // Filter non-positive y values; data stays in original space (LogCoord handles scaling)
  let filtered: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| series.iter().filter(|&&(_, y)| y > 0.0).copied().collect())
    .collect();

  let (x_range, y_range) = compute_ranges(&filtered);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&filtered, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListLogLogPlot: both axes log10 scale
pub fn list_log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let mut parsed = parse_plot_options(args);
  parsed.opts.log_x = true;
  parsed.opts.log_y = true;

  // Filter non-positive values; data stays in original space (LogCoord handles scaling)
  let filtered: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| {
      series
        .iter()
        .filter(|&&(x, y)| x > 0.0 && y > 0.0)
        .copied()
        .collect()
    })
    .collect();

  let (x_range, y_range) = compute_ranges(&filtered);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&filtered, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListLogLinearPlot: x-axis is log10 scale
pub fn list_log_linear_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let mut parsed = parse_plot_options(args);
  parsed.opts.log_x = true;

  // Filter non-positive x values; data stays in original space (LogCoord handles scaling)
  let filtered: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| series.iter().filter(|&&(x, _)| x > 0.0).copied().collect())
    .collect();

  let (x_range, y_range) = compute_ranges(&filtered);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&filtered, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListPolarPlot[{r1, r2, ...}]: plot data in polar coordinates
pub fn list_polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_list_data(&args[0])?;
  let parsed = parse_plot_options(args);

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
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&polar_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// DiscretePlot[expr, {n, nmin, nmax}] or DiscretePlot[expr, {n, nmin, nmax, step}]
pub fn discrete_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use crate::functions::plot::substitute_var;

  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot expects at least 2 arguments".into(),
    ));
  }

  let expr = &args[0];

  // Parse iteration spec: {var, min, max} or {var, min, max, step}
  let iter_spec = match &args[1] {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscretePlot: second argument must be an iteration specification {var, min, max}".into(),
      ));
    }
  };

  if iter_spec.len() < 2 || iter_spec.len() > 4 {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot: iteration spec must be {var, max}, {var, min, max}, or {var, min, max, step}".into(),
    ));
  }

  let var_name = match &iter_spec[0] {
    Expr::Identifier(name) => name.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DiscretePlot: first element of iteration spec must be a variable"
          .into(),
      ));
    }
  };

  // {var, max} → {var, 1, max}
  let (n_min, n_max) = if iter_spec.len() == 2 {
    let n_max_expr = evaluate_expr_to_expr(&iter_spec[1])?;
    let n_max = try_eval_to_f64(&n_max_expr).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "DiscretePlot: max must be numeric".into(),
      )
    })?;
    (1.0, n_max)
  } else {
    let n_min_expr = evaluate_expr_to_expr(&iter_spec[1])?;
    let n_max_expr = evaluate_expr_to_expr(&iter_spec[2])?;
    let n_min = try_eval_to_f64(&n_min_expr).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "DiscretePlot: min must be numeric".into(),
      )
    })?;
    let n_max = try_eval_to_f64(&n_max_expr).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "DiscretePlot: max must be numeric".into(),
      )
    })?;
    (n_min, n_max)
  };

  let step = if iter_spec.len() == 4 {
    let step_expr = evaluate_expr_to_expr(&iter_spec[3])?;
    try_eval_to_f64(&step_expr).ok_or_else(|| {
      InterpreterError::EvaluationError(
        "DiscretePlot: step must be numeric".into(),
      )
    })?
  } else {
    1.0
  };

  if step <= 0.0 {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot: step must be positive".into(),
    ));
  }

  // Generate data points by evaluating expr at each discrete value
  let mut points: Vec<(f64, f64)> = Vec::new();
  let mut n = n_min;
  let max_points = 10000;
  while n <= n_max + step * 0.5e-10 && points.len() < max_points {
    let n_expr = if n == n.floor() && n.abs() < 1e15 {
      Expr::Integer(n as i128)
    } else {
      Expr::Real(n)
    };
    let substituted = substitute_var(expr, &var_name, &n_expr);
    if let Ok(result) = evaluate_expr_to_expr(&substituted)
      && let Some(y) = try_eval_to_f64(&result)
      && y.is_finite()
    {
      points.push((n, y));
    }
    n += step;
  }

  let all_series = vec![points];
  let parsed = parse_plot_options(args);
  let (x_range, y_range) = compute_ranges(&all_series);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);

  let svg = generate_scatter_svg_with_options(
    &all_series,
    x_range,
    y_range,
    &parsed.opts,
  )?;
  Ok(crate::graphics_result(svg))
}

/// Sample rate wolframscript assumes for `Audio[{samples…}]` built from raw
/// sample data (Hz).
const AUDIO_SAMPLE_RATE: f64 = 44100.0;

/// Cap on rendered points per channel. Longer audio is reduced to a min/max
/// envelope per bucket so the SVG stays small while the waveform shape is
/// preserved (wolframscript likewise draws an envelope for long audio).
const AUDIO_MAX_POINTS: usize = 2000;

/// Extract the sample rate from an `Audio[data, opts…]` argument list
/// (`SampleRate -> r`), falling back to the 44100 Hz default.
fn audio_sample_rate(args: &[Expr]) -> f64 {
  for opt in args.iter().skip(1) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "SampleRate")
    {
      let val =
        evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
      if let Some(r) = try_eval_to_f64(&val)
        && r > 0.0
      {
        return r;
      }
    }
  }
  AUDIO_SAMPLE_RATE
}

/// Turn one audio-like object into per-channel (time, amplitude) series:
/// `Audio[{s1, s2, …}]` (one channel), `Audio[{{ch1…}, {ch2…}}]` (one series
/// per channel), or a `Sound`/`Play` expression sampled via the synthesizer.
/// Returns `None` when the expression is not a samplable audio object.
fn audio_channels(expr: &Expr) -> Option<Vec<Vec<(f64, f64)>>> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Audio" && !args.is_empty() =>
    {
      let data = evaluate_expr_to_expr(&args[0]).unwrap_or(args[0].clone());
      let Expr::List(items) = &data else {
        return None;
      };
      let rate = audio_sample_rate(args);
      let channels: Vec<Vec<Expr>> = if !items.is_empty()
        && items.iter().all(|i| matches!(i, Expr::List(_)))
      {
        items
          .iter()
          .map(|i| match i {
            Expr::List(l) => l.to_vec(),
            _ => unreachable!(),
          })
          .collect()
      } else {
        vec![items.to_vec()]
      };
      let mut out = Vec::with_capacity(channels.len());
      for channel in channels {
        let mut points = Vec::with_capacity(channel.len());
        for (i, sample) in channel.iter().enumerate() {
          let val =
            evaluate_expr_to_expr(sample).unwrap_or_else(|_| sample.clone());
          let y = try_eval_to_f64(&val)?;
          points.push((i as f64 / rate, y));
        }
        out.push(points);
      }
      Some(out)
    }
    Expr::FunctionCall { name, .. } if name == "Sound" || name == "Play" => {
      let (samples, rate) = crate::functions::sound::sound_to_samples(expr)?;
      Some(vec![
        samples
          .iter()
          .enumerate()
          .map(|(i, &y)| (i as f64 / rate as f64, y))
          .collect(),
      ])
    }
    _ => None,
  }
}

/// Collect the channel series of an audio object or of a list of audio
/// objects (`AudioPlot[{audio1, audio2}]` overlays them as separate series).
fn collect_audio_series(expr: &Expr) -> Option<Vec<Vec<(f64, f64)>>> {
  if let Some(channels) = audio_channels(expr) {
    return Some(channels);
  }
  if let Expr::List(items) = expr
    && !items.is_empty()
  {
    let mut out = Vec::new();
    for item in items.iter() {
      out.extend(audio_channels(item)?);
    }
    return Some(out);
  }
  None
}

/// Reduce a long sample series to a min/max envelope: each bucket contributes
/// its extreme points in chronological order, preserving the waveform shape
/// with a bounded point count.
fn envelope_downsample(
  points: Vec<(f64, f64)>,
  max_points: usize,
) -> Vec<(f64, f64)> {
  if points.len() <= max_points {
    return points;
  }
  let buckets = max_points / 2;
  let n = points.len();
  let mut out = Vec::with_capacity(buckets * 2);
  for b in 0..buckets {
    let start = b * n / buckets;
    let end = (((b + 1) * n / buckets).max(start + 1)).min(n);
    let slice = &points[start..end];
    let mut lo = slice[0];
    let mut hi = slice[0];
    for &p in slice {
      if p.1 < lo.1 {
        lo = p;
      }
      if p.1 > hi.1 {
        hi = p;
      }
    }
    let (first, second) = if lo.0 <= hi.0 { (lo, hi) } else { (hi, lo) };
    out.push(first);
    if second != first {
      out.push(second);
    }
  }
  out
}

/// AudioPlot[audio] — plot the waveform of an `Audio` or `Sound` object (or
/// a list of them) as amplitude over time in seconds. Non-audio arguments
/// leave the expression unevaluated, matching wolframscript.
pub fn audio_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data =
    evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
  let Some(all_series) = collect_audio_series(&data) else {
    return Ok(Expr::FunctionCall {
      name: "AudioPlot".to_string(),
      args: args.to_vec().into(),
    });
  };
  let all_series: Vec<Vec<(f64, f64)>> = all_series
    .into_iter()
    .map(|s| envelope_downsample(s, AUDIO_MAX_POINTS))
    .collect();

  let mut parsed = parse_plot_options(args);
  // AudioPlot fills the waveform to the zero axis unless Filling is given.
  let has_filling_opt = args[1..].iter().any(|opt| {
    matches!(opt, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Filling"))
  });
  if !has_filling_opt {
    parsed.opts.filling = crate::functions::plot::Filling::Axis;
  }

  let (x_range, y_range) = compute_ranges(&all_series);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&all_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}
