#[allow(unused_imports)]
use super::*;
use std::cmp::Ordering;

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

/// Whether a list node can be read as a curve or as a grouping of curves.
/// Wolfram lets the curve specification be nested to any depth — the
/// grouping only steers styling — so `{{fx, fy}, …}` and
/// `{{{fx, fy}, …}}` both denote the same set of curves.
fn is_curve_or_group(expr: &Expr) -> bool {
  match expr {
    // A pair is either a curve `{fx, fy}` or a group of two curves;
    // `collect_curves` decides which when it descends.
    Expr::List(items) if items.len() == 2 => true,
    Expr::List(items) if !items.is_empty() => {
      items.iter().all(is_curve_or_group)
    }
    _ => false,
  }
}

/// Flatten a (possibly nested) curve specification into individual curves,
/// appending them to `out`. Returns whether the shape was understood.
fn collect_curves<'a>(expr: &'a Expr, out: &mut Vec<CurveSrc<'a>>) -> bool {
  let Expr::List(items) = expr else {
    return false;
  };
  if items.is_empty() {
    return false;
  }
  // Every element being curve-shaped makes this a grouping level — which
  // is how the ambiguous `{{a, b}, {c, d}}` resolves to two curves.
  if items.iter().all(is_curve_or_group) {
    return items.iter().all(|item| collect_curves(item, out));
  }
  if items.len() == 2 {
    out.push(CurveSrc::Pair(&items[0], &items[1]));
    return true;
  }
  false
}

/// Bisection passes used to smooth out an under-resolved curve. Each pass
/// halves the flagged intervals, so six passes shrink a step by 64×.
const MAX_REFINEMENT_DEPTH: usize = 6;

/// One sampled parameter value together with the coordinate pair of every
/// curve there (`None` when the body did not evaluate to coordinates).
type Sample = (f64, Option<Vec<(f64, f64)>>);

/// Whether the middle of a sampled triple is far enough off the straight
/// line between its neighbours that the interval needs more points. The
/// measure is the deviation relative to the neighbours' separation, so it
/// is independent of the coordinate scale.
fn needs_refinement(a: &Sample, b: &Sample, c: &Sample) -> bool {
  let (Some(ra), Some(rb), Some(rc)) = (&a.1, &b.1, &c.1) else {
    // A gap in the samples is a discontinuity (or a domain boundary);
    // refining pins down where it actually starts.
    return true;
  };
  if ra.len() != rb.len() || rb.len() != rc.len() {
    return true;
  }
  let span = c.0 - a.0;
  if span == 0.0 {
    return false;
  }
  let u = (b.0 - a.0) / span;
  ra.iter().zip(rb).zip(rc).any(|((p0, p1), p2)| {
    if !p0.0.is_finite()
      || !p0.1.is_finite()
      || !p1.0.is_finite()
      || !p1.1.is_finite()
      || !p2.0.is_finite()
      || !p2.1.is_finite()
    {
      return true;
    }
    let interp = (p0.0 + (p2.0 - p0.0) * u, p0.1 + (p2.1 - p0.1) * u);
    let deviation = (p1.0 - interp.0).hypot(p1.1 - interp.1);
    let separation = (p2.0 - p0.0).hypot(p2.1 - p0.1).max(1e-10);
    deviation / separation > 0.05
  })
}

/// Adaptively sample a parametric curve: lay down a uniform grid, then
/// bisect the intervals where the polyline visibly departs from the curve.
/// A uniform grid alone leaves a curve polygonal whenever the part that
/// bends is a small slice of the parameter range (e.g. `{t, -1000, 1000}`
/// for a curve shaped near `t == 0`), which is why Wolfram refines instead
/// of sampling a fixed grid.
fn adaptive_samples<F>(
  sample: F,
  t_min: f64,
  t_max: f64,
  initial_n: usize,
  max_total: usize,
) -> Vec<Sample>
where
  F: Fn(f64) -> Option<Vec<(f64, f64)>>,
{
  let initial_n = initial_n.max(2);
  let step = (t_max - t_min) / (initial_n - 1) as f64;
  let mut samples: Vec<Sample> = (0..initial_n)
    .map(|i| {
      let t = t_min + i as f64 * step;
      (t, sample(t))
    })
    .collect();

  let min_span = (t_max - t_min).abs() * 1e-10;
  for _ in 0..MAX_REFINEMENT_DEPTH {
    if samples.len() >= max_total {
      break;
    }
    let budget = max_total - samples.len();
    let mut added: Vec<Sample> = Vec::new();
    for triple in samples.windows(3) {
      if added.len() >= budget {
        break;
      }
      if (triple[2].0 - triple[0].0).abs() < min_span {
        continue;
      }
      if !needs_refinement(&triple[0], &triple[1], &triple[2]) {
        continue;
      }
      for (left, right) in [(&triple[0], &triple[1]), (&triple[1], &triple[2])]
      {
        let mid = f64::midpoint(left.0, right.0);
        // Neighbouring triples share an interval, so the same midpoint
        // comes up twice; skipping it saves evaluating the body again.
        if mid > left.0
          && mid < right.0
          && added.last().is_none_or(|last| last.0 != mid)
        {
          added.push((mid, sample(mid)));
        }
      }
    }
    if added.is_empty() {
      break;
    }
    samples.extend(added);
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    samples.dedup_by(|a, b| a.0 == b.0);
  }

  samples
}

/// Split the per-parameter samples into one point series per curve, keeping
/// a gap (as a non-finite point) wherever the body did not evaluate.
fn samples_to_series(samples: Vec<Sample>) -> Vec<Vec<(f64, f64)>> {
  // The curve count comes from the first sample that evaluated.
  let n_curves = samples
    .iter()
    .find_map(|(_, row)| row.as_ref().map(std::vec::Vec::len))
    .unwrap_or(1);
  let mut series = vec![Vec::with_capacity(samples.len()); n_curves];
  for (_, row) in samples {
    match row {
      Some(r) if r.len() == n_curves => {
        for (curve, point) in series.iter_mut().zip(r) {
          curve.push(point);
        }
      }
      _ => {
        for curve in &mut series {
          curve.push((f64::NAN, f64::NAN));
        }
      }
    }
  }
  series
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

  // Collect curve bodies. Wolfram accepts a single curve `{fx, fy}` or
  // any number of curves nested to any depth, e.g.
  // `{{fx1,fy1}, {fx2,fy2}, …}` and `{Table[{fx, fy}, {i, …}]}` — the
  // extra grouping levels only matter for styling, so they are flattened
  // away here.
  let mut syntactic = Vec::new();
  let is_list_body = matches!(body, Expr::List(_));
  let syntactic_ok = is_list_body && collect_curves(body, &mut syntactic);

  // The first argument is held, so a generated specification such as
  // `{Table[{fx, fy}, {i, …}]}` only takes curve shape once evaluated.
  // Evaluate it (the plot variable stays symbolic) and retry, matching
  // Wolfram, which samples the held body rather than requiring
  // `Evaluate[…]` around it.
  let evaluated_body: Option<Expr> = if is_list_body && !syntactic_ok {
    evaluate_expr_to_expr(body).ok()
  } else {
    None
  };

  let curves: Vec<CurveSrc> = if syntactic_ok {
    syntactic
  } else if is_list_body {
    let mut collected = Vec::new();
    let ok = evaluated_body
      .as_ref()
      .is_some_and(|ev| collect_curves(ev, &mut collected));
    if !ok {
      return Err(InterpreterError::EvaluationError(
        "ParametricPlot: first argument must be {fx, fy}".into(),
      ));
    }
    collected
  } else {
    // A non-list body (e.g. `f[t]` or `BSplineFunction[…][t]`) is a curve
    // whose coordinate pair only materialises once `t` is numeric; sample
    // the whole expression instead of decomposing it into components.
    vec![CurveSrc::Whole(body)]
  };

  // Sample at least NUM_SAMPLES points; an explicit larger PlotPoints
  // raises the resolution. (Wolfram refines a coarse PlotPoints setting
  // adaptively, so a smaller explicit value must not reduce smoothness.)
  // The uniform grid is the starting point; intervals where it would
  // render the curve as a polygon get bisected on top of it.
  let num_samples = plot_opts.plot_points.max(NUM_SAMPLES);
  let max_total = num_samples.saturating_mul(2);
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(curves.len());

  for curve in &curves {
    // Each sample yields one coordinate pair, or one pair per curve when
    // the whole body is sampled (e.g. `ReIm[{c1, c2, c3}]`).
    let samples = match curve {
      CurveSrc::Pair(fx, fy) => adaptive_samples(
        |t| match (
          evaluate_at_point(fx, &var_name, t),
          evaluate_at_point(fy, &var_name, t),
        ) {
          (Some(x), Some(y)) => Some(vec![(x, y)]),
          _ => None,
        },
        t_min,
        t_max,
        num_samples,
        max_total,
      ),
      CurveSrc::Whole(b) => adaptive_samples(
        |t| sample_whole_rows(b, &var_name, t),
        t_min,
        t_max,
        num_samples,
        max_total,
      ),
    };
    all_points.extend(samples_to_series(samples));
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
    .map_or((0.24, 0.6, 0.8), |c| (c.r, c.g, c.b));
  let color = call(
    "RGBColor",
    vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
  );
  let opacity = call1("Opacity", Expr::Real(0.3));
  let no_edges = call("EdgeForm", vec![]);

  let mut face_group = vec![no_edges, color.clone(), opacity];
  face_group.append(&mut quads);
  let mut items = vec![Expr::List(face_group.into())];
  if border.len() > 1 {
    items.push(Expr::List(
      vec![
        color,
        call1("AbsoluteThickness", Expr::Real(2.0)),
        call1("Line", Expr::List(border.into())),
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
    *structure = Some(Box::new(call("Graphics", graphics_args)));
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
