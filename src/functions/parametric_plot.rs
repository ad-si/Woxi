#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  NUM_SAMPLES, PlotOptions, PlotRangeOverrides,
  adjust_y_range_for_filling_opts, apply_common_plot_option, build_plot_source,
  evaluate_at_point, generate_svg_with_filling, parse_iterator, substitute_var,
};

/// Parse the trailing option rules of a ParametricPlot/PolarPlot call.
/// Matching Wolfram Language, the first occurrence of a repeated option
/// wins (e.g. `PlotPoints -> 200, …, PlotPoints -> 10` uses 200).
fn parse_options(opts: &[Expr]) -> (PlotOptions, PlotRangeOverrides) {
  let mut plot_opts = PlotOptions::default();
  let mut overrides = PlotRangeOverrides::default();
  let mut seen: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  for opt in opts {
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      if !seen.insert(name.to_string()) {
        continue;
      }
      apply_common_plot_option(
        name,
        replacement,
        &mut plot_opts,
        &mut overrides,
      );
      // In ParametricPlot both coordinates are dependent values, so a
      // scalar-pair `PlotRange -> {min, max}` fixes *both* axes (Plot's
      // parser maps it to y only, which fits plots of one dependent
      // variable but would leave the x range jumping between animation
      // frames here).
      if name == "PlotRange"
        && overrides.x.is_none()
        && let Expr::List(items) = replacement
        && items.len() == 2
        && !matches!(&items[0], Expr::List(_))
        && !matches!(&items[1], Expr::List(_))
      {
        overrides.x = overrides.y;
      }
    }
  }
  (plot_opts, overrides)
}

/// Pick the displayed x/y ranges and the SVG height for a parametric-style
/// plot: explicit PlotRange components override the data extents, and the
/// image aspect follows the displayed data (so circles render round) unless
/// an explicit AspectRatio was given.
fn apply_ranges_and_aspect(
  plot_opts: &mut PlotOptions,
  overrides: &PlotRangeOverrides,
  data_x: (f64, f64),
  data_y: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
  let x_range = overrides.x.unwrap_or(data_x);
  let y_range = overrides.y.unwrap_or(data_y);

  if let Some(ar) = overrides.aspect_ratio {
    // Explicit AspectRatio sizes the plotting area; the total height is
    // derived in generate_svg_with_options once margins are known.
    plot_opts.aspect_ratio = Some(ar);
  } else {
    // Adjust aspect ratio to match the displayed data (circles stay round).
    let data_w = x_range.1 - x_range.0;
    let data_h = y_range.1 - y_range.0;
    if data_w > 0.0 && data_h > 0.0 {
      let data_aspect = data_h / data_w;
      if data_aspect.is_finite() {
        plot_opts.svg_height =
          (plot_opts.svg_width as f64 * data_aspect).round() as u32;
      }
    }
  }
  (x_range, y_range)
}

/// A single parametric curve, sampled either component-wise (`{fx, fy}` where
/// both components are explicit expressions) or as a whole expression that
/// only yields a coordinate pair once the parameter is numeric (e.g.
/// `BSplineFunction[…][t]` or any `f[t]` returning `{x, y}`).
enum CurveSrc<'a> {
  Pair(&'a Expr, &'a Expr),
  Whole(&'a Expr),
}

/// Sample a whole-expression curve at parameter `t`. The evaluated body may
/// be a single coordinate pair (one curve) or a list of coordinate pairs
/// (one sample per curve, e.g. `ReIm[{c1, c2, c3}]`); either way one row of
/// per-curve samples is returned.
fn sample_whole_rows(
  body: &Expr,
  var: &str,
  t: f64,
) -> Option<Vec<(f64, f64)>> {
  let substituted = substitute_var(body, var, &Expr::Real(t));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  let items = match &result {
    Expr::List(items) if !items.is_empty() => items,
    _ => return None,
  };
  // A flat `{x, y}` pair is a single curve.
  if items.len() == 2
    && let (Some(x), Some(y)) =
      (try_eval_to_f64(&items[0]), try_eval_to_f64(&items[1]))
  {
    return Some(vec![(x, y)]);
  }
  // `{{x1, y1}, …, {xn, yn}}` is one sample for each of n curves.
  let rows: Option<Vec<(f64, f64)>> = items
    .iter()
    .map(|item| {
      if let Expr::List(pair) = item
        && pair.len() == 2
      {
        Some((try_eval_to_f64(&pair[0])?, try_eval_to_f64(&pair[1])?))
      } else {
        None
      }
    })
    .collect();
  rows
}

/// Whether an argument is an iterator (`{v, vmin, vmax}`) rather than an
/// option rule — what tells a second parameter apart from the trailing
/// options of a one-parameter plot.
fn is_iterator_spec(arg: &Expr) -> bool {
  matches!(arg, Expr::List(items)
    if items.len() == 3 && matches!(items[0], Expr::Identifier(_)))
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

  // Two iterators make the plot a *region*: the image of the parameter
  // rectangle under `{fx, fy}`, not a curve.
  if args.len() > 2 && is_iterator_spec(&args[2]) {
    return parametric_region_ast(body, iter_spec, &args[2], &args[3..]);
  }

  let (mut plot_opts, overrides) = parse_options(&args[2..]);

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

  // Sample at least NUM_SAMPLES points; an explicit larger PlotPoints
  // raises the resolution. (Wolfram refines a coarse PlotPoints setting
  // adaptively, so a smaller explicit value must not reduce smoothness.)
  let num_samples = plot_opts.plot_points.max(NUM_SAMPLES);
  let step = (t_max - t_min) / (num_samples - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(curves.len());

  for curve in &curves {
    match curve {
      CurveSrc::Pair(fx, fy) => {
        let mut points = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
          let t = t_min + i as f64 * step;
          let pt = match (
            evaluate_at_point(fx, &var_name, t),
            evaluate_at_point(fy, &var_name, t),
          ) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
          };
          points.push(pt.unwrap_or((f64::NAN, f64::NAN)));
        }
        all_points.push(points);
      }
      CurveSrc::Whole(b) => {
        // The whole body is evaluated once per sample; each sample may
        // yield one coordinate pair or one pair per curve (e.g.
        // `ReIm[{c1, c2, c3}]`). The curve count comes from the first
        // sample that evaluates successfully.
        let rows: Vec<Option<Vec<(f64, f64)>>> = (0..num_samples)
          .map(|i| sample_whole_rows(b, &var_name, t_min + i as f64 * step))
          .collect();
        let n_curves = rows
          .iter()
          .find_map(|r| r.as_ref().map(|v| v.len()))
          .unwrap_or(1);
        let mut pts = vec![Vec::with_capacity(num_samples); n_curves];
        for row in rows {
          match row {
            Some(r) if r.len() == n_curves => {
              for (k, p) in r.into_iter().enumerate() {
                pts[k].push(p);
              }
            }
            _ => {
              for series in pts.iter_mut() {
                series.push((f64::NAN, f64::NAN));
              }
            }
          }
        }
        all_points.extend(pts);
      }
    }
  }

  // Compute ranges (explicit PlotRange components override the data extents)
  let (data_x, data_y) = compute_data_ranges(&all_points);
  // A `PlotStyle` list applied to a single curve is one combined directive
  // set, not a per-curve cycle: `ParametricPlot[c, …, PlotStyle -> {Thick,
  // Red}]` draws one thick red curve.
  if all_points.len() == 1 {
    plot_opts.plot_style =
      crate::functions::plot::collapse_style_for_single_series(
        &plot_opts.plot_style,
      );
  }
  let data_y = adjust_y_range_for_filling_opts(&plot_opts, data_y);
  let (x_range, y_range) =
    apply_ranges_and_aspect(&mut plot_opts, &overrides, data_x, data_y);

  let svg =
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?;

  // Attach the sampled curves as a PlotSource so `Show` can merge this plot
  // with other graphics (re-rendering the curves as Line primitives). The
  // style goes with them: a curve merged by `Show` keeps the colour and
  // weight `PlotStyle` gave it, as it does when drawn on its own.
  let source = build_plot_source(
    &all_points,
    &plot_opts.plot_style,
    x_range,
    y_range,
    (plot_opts.svg_width, plot_opts.svg_height),
    false,
    plot_opts.filling,
    plot_opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  Ok(crate::graphics_result_with_source(svg, source))
}

/// Samples per parameter direction for a two-parameter (region) plot.
/// Wolfram starts from a coarse grid and refines it adaptively; sampling a
/// finer uniform grid gets to the same smooth boundary without the
/// refinement machinery.
const REGION_GRID: usize = 33;

/// Evaluate a `{fx, fy}` body at one point of the parameter rectangle.
fn evaluate_at_uv(
  body: &Expr,
  uvar: &str,
  u: f64,
  vvar: &str,
  v: f64,
) -> Option<(f64, f64)> {
  let sub = substitute_var(body, uvar, &Expr::Real(u));
  let sub = substitute_var(&sub, vvar, &Expr::Real(v));
  match evaluate_expr_to_expr(&sub).ok()? {
    Expr::List(ref items) if items.len() == 2 => {
      Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?))
    }
    _ => None,
  }
}

/// `ParametricPlot[{fx, fy}, {u, umin, umax}, {v, vmin, vmax}]` — the image
/// of the parameter rectangle, drawn the way Wolfram draws it: the quads of
/// the sampled grid filled at 30% opacity in the plot colour, with the image
/// of the rectangle's boundary stroked on top.
///
/// The picture is assembled as a `Graphics[…]` of primitives rather than
/// rendered directly, so it carries a structure that `Show` can merge with
/// the other layers of a composite figure.
fn parametric_region_ast(
  body: &Expr,
  u_spec: &Expr,
  v_spec: &Expr,
  opt_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (uvar, u_min, u_max) = parse_iterator(u_spec, "ParametricPlot")?;
  let (vvar, v_min, v_max) = parse_iterator(v_spec, "ParametricPlot")?;
  let (plot_opts, _) = parse_options(opt_args);

  // An explicit `PlotPoints` sets the grid; the option's plot default is a
  // curve-sample count, so it cannot be used as the grid size directly.
  let grid = opt_args
    .iter()
    .find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      match (name, &*value) {
        ("PlotPoints", Expr::Integer(n)) if *n >= 2 => Some(*n as usize),
        _ => None,
      }
    })
    .unwrap_or(REGION_GRID);

  let du = (u_max - u_min) / (grid - 1) as f64;
  let dv = (v_max - v_min) / (grid - 1) as f64;
  let mut points: Vec<Vec<Option<(f64, f64)>>> = Vec::with_capacity(grid);
  for i in 0..grid {
    let u = u_min + i as f64 * du;
    let mut row = Vec::with_capacity(grid);
    for j in 0..grid {
      let v = v_min + j as f64 * dv;
      row.push(evaluate_at_uv(body, &uvar, u, &vvar, v));
    }
    points.push(row);
  }

  let pt_expr =
    |(x, y): (f64, f64)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());

  // One quad per grid cell; a cell with a point the body could not be
  // evaluated at (a singularity, a complex value) is left out.
  let mut quads = Vec::new();
  for i in 0..grid - 1 {
    for j in 0..grid - 1 {
      let corners = [
        points[i][j],
        points[i + 1][j],
        points[i + 1][j + 1],
        points[i][j + 1],
      ];
      if corners
        .iter()
        .any(|c| c.is_none_or(|(x, y)| !x.is_finite() || !y.is_finite()))
      {
        continue;
      }
      quads.push(Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(
          corners.iter().map(|c| pt_expr(c.unwrap())).collect(),
        )]
        .into(),
      });
    }
  }
  if quads.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot: the body evaluates to no points over that region".into(),
    ));
  }

  // The boundary of the picture is the image of the parameter rectangle's
  // boundary, walked once counter-clockwise.
  let last = grid - 1;
  let mut border_idx: Vec<(usize, usize)> = Vec::with_capacity(4 * grid);
  border_idx.extend((0..grid).map(|j| (0, j)));
  border_idx.extend((1..grid).map(|i| (i, last)));
  border_idx.extend((0..last).rev().map(|j| (last, j)));
  border_idx.extend((0..last).rev().map(|i| (i, 0)));
  let border: Vec<Expr> = border_idx
    .iter()
    .filter_map(|&(i, j)| points[i][j])
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .map(pt_expr)
    .collect();

  let (r, g, b) = plot_opts
    .plot_style
    .first()
    .and_then(|s| s.color.as_ref())
    .map(|c| (c.r, c.g, c.b))
    .unwrap_or((0.24, 0.6, 0.8));
  let color = Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
  };
  let opacity = Expr::FunctionCall {
    name: "Opacity".to_string(),
    args: vec![Expr::Real(0.3)].into(),
  };
  let no_edges = Expr::FunctionCall {
    name: "EdgeForm".to_string(),
    args: vec![].into(),
  };

  let mut face_group = vec![no_edges, color.clone(), opacity];
  face_group.append(&mut quads);
  let mut items = vec![Expr::List(face_group.into())];
  if border.len() > 1 {
    items.push(Expr::List(
      vec![
        color,
        Expr::FunctionCall {
          name: "AbsoluteThickness".to_string(),
          args: vec![Expr::Real(2.0)].into(),
        },
        Expr::FunctionCall {
          name: "Line".to_string(),
          args: vec![Expr::List(border.into())].into(),
        },
      ]
      .into(),
    ));
  }

  // Wolfram draws a ParametricPlot with axes and equally scaled ones; the
  // caller's own options come last so they win.
  let mut graphics_args = vec![Expr::List(items.into())];
  graphics_args.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("Axes".to_string())),
    replacement: Box::new(Expr::Identifier("True".to_string())),
  });
  graphics_args.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("AspectRatio".to_string())),
    replacement: Box::new(Expr::Identifier("Automatic".to_string())),
  });
  graphics_args.extend(opt_args.iter().cloned());
  let mut result = crate::functions::graphics::graphics_ast(&graphics_args)?;
  // Keep the primitives on the rendering so `Show` can draw the region
  // together with the other layers of a composite figure instead of
  // dropping it for having no plot data.
  if let Expr::Graphics { structure, .. } = &mut result {
    *structure = Some(Box::new(Expr::FunctionCall {
      name: "Graphics".to_string(),
      args: graphics_args.into(),
    }));
  }
  Ok(result)
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

  let (mut plot_opts, overrides) = parse_options(&args[2..]);

  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "PolarPlot")?;

  // Collect function bodies
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  let num_samples = plot_opts.plot_points.max(NUM_SAMPLES);
  let step = (t_max - t_min) / (num_samples - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let mut points = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
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

  let (data_x, data_y) = compute_data_ranges(&all_points);
  let data_y = adjust_y_range_for_filling_opts(&plot_opts, data_y);
  let (x_range, y_range) =
    apply_ranges_and_aspect(&mut plot_opts, &overrides, data_x, data_y);

  let svg =
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?;

  // Attach the sampled curves as a PlotSource so `Show` can merge this plot
  // with other graphics (re-rendering the curves as Line primitives). The
  // style goes with them: a curve merged by `Show` keeps the colour and
  // weight `PlotStyle` gave it, as it does when drawn on its own.
  let source = build_plot_source(
    &all_points,
    &plot_opts.plot_style,
    x_range,
    y_range,
    (plot_opts.svg_width, plot_opts.svg_height),
    false,
    plot_opts.filling,
    plot_opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  Ok(crate::graphics_result_with_source(svg, source))
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
