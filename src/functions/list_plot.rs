#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, FillTarget, Filling, IntervalMarkers,
  PlotOptions, SeriesStyle, adjust_y_range_for_filling_opts,
  apply_filling_option, build_plot_source, expand_plot_theme,
  generate_scatter_svg_with_options, generate_svg_with_filling,
  parse_image_size, parse_plot_style,
};
use crate::functions::plot_scaling::Scaling;
use crate::functions::sound::audio_sample_rate;

/// One parsed data point: coordinates plus asymmetric x/y uncertainties.
/// The uncertainties are (minus, plus) half-widths taken from
/// `Around[value, err]` / `Around[value, {minus, plus}]` entries and are
/// `(0, 0)` for exact values.
#[derive(Clone, Copy)]
struct ErrPoint {
  x: f64,
  y: f64,
  dx: (f64, f64),
  dy: (f64, f64),
  /// The x-coordinate is the point's position in a plain list of values
  /// (1, 2, …) rather than given by the data.
  implicit_x: bool,
}

impl ErrPoint {
  fn from_y(x: f64, (y, dy): (f64, (f64, f64))) -> Self {
    Self {
      x,
      y,
      dx: (0.0, 0.0),
      dy,
      implicit_x: true,
    }
  }

  fn from_xy((x, dx): (f64, (f64, f64)), (y, dy): (f64, (f64, f64))) -> Self {
    Self {
      x,
      y,
      dx,
      dy,
      implicit_x: false,
    }
  }
}

/// Whether every point of the parsed data sits at its list position, so the
/// x axis of the plot starts at 0 (`ListPlot[{y1, y2, …}]`).
fn all_implicit_x(all_series: &[Vec<ErrPoint>]) -> bool {
  let mut points = all_series.iter().flatten().peekable();
  points.peek().is_some() && points.all(|p| p.implicit_x)
}

/// List plots use a coarse index grid, refining its beginning when that
/// grid leaves fewer than the requested number of points. Selection is
/// independent of coordinates, and always preserves both endpoints.
fn plot_sample_indices(n: usize, limit: usize) -> Vec<usize> {
  if n <= limit || limit < 2 {
    return (0..n).collect();
  }
  let stride = n / limit + 1;
  let mut selected = vec![false; n];
  selected[0] = true;
  selected[n - 1] = true;
  let mut count = 2;
  for i in (stride..n).step_by(stride).take(limit - 2) {
    selected[i - 1] = true;
    count += 1;
  }
  let refinement = stride.div_ceil(2);
  let mut candidate = refinement - 1;
  while count < limit {
    while selected[candidate] {
      candidate += 1;
    }
    selected[candidate] = true;
    count += 1;
    candidate += refinement;
  }
  selected
    .into_iter()
    .enumerate()
    .filter_map(|(i, keep)| keep.then_some(i))
    .collect()
}

fn sample_plot_data(
  args: &[Expr],
  series: &mut [Vec<ErrPoint>],
  labels: &mut [Vec<Option<PointLabel>>],
) {
  fn limit(expr: &Expr) -> Option<usize> {
    match expr {
      Expr::Integer(n) if *n >= 2 => usize::try_from(*n).ok(),
      Expr::List(items) if items.len() <= 2 => {
        items.iter().rev().find_map(limit)
      }
      _ => None,
    }
  }
  let Some(option) = option_value(args, "MaxPlotPoints") else {
    return;
  };
  let option = evaluate_expr_to_expr(option).unwrap_or_else(|_| option.clone());
  let Some(limit) = limit(&option) else {
    return;
  };
  for (i, points) in series.iter_mut().enumerate() {
    let indices = plot_sample_indices(points.len(), limit);
    *points = indices.iter().map(|&j| points[j]).collect();
    if let Some(labels) = labels.get_mut(i) {
      *labels = indices
        .iter()
        .map(|&j| labels.get(j).cloned().flatten())
        .collect();
    }
  }
}

/// Strips a `Quantity[magnitude, unit]` wrapper down to its magnitude, like
/// wolframscript's plot functions do when handed dimensioned data (the axis
/// then just carries the bare number). Any other expression passes through
/// unchanged.
fn strip_quantity(expr: &Expr) -> &Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      &args[0]
    }
    _ => expr,
  }
}

/// Evaluate an expression to a plottable value with uncertainty: a plain
/// number carries zero uncertainty, `Around[v, u]` / `Around[v, {m, p}]`
/// carry (minus, plus) error-bar half-widths.
fn eval_to_value_err(expr: &Expr) -> Option<(f64, (f64, f64))> {
  let e = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  let e = strip_quantity(&e).clone();
  if let Expr::FunctionCall { name, args } = &e
    && name == "Around"
    && args.len() == 2
  {
    let v = try_eval_to_f64(&args[0])?;
    let err = match &args[1] {
      Expr::List(items) if items.len() == 2 => (
        try_eval_to_f64(&items[0])?.abs(),
        try_eval_to_f64(&items[1])?.abs(),
      ),
      u => {
        let u = try_eval_to_f64(u)?.abs();
        (u, u)
      }
    };
    return Some((v, err));
  }
  // `Interval[{lo, hi}]` plots like an uncertainty around its midpoint
  // (drawn without end caps, see `interval_only_data`).
  if let Expr::FunctionCall { name, args } = &e
    && name == "Interval"
    && let [Expr::List(bounds)] = args.as_slice()
    && let [lo, hi] = bounds.as_slice()
  {
    let (lo, hi) = (try_eval_to_f64(lo)?, try_eval_to_f64(hi)?);
    let half = (hi - lo).abs() / 2.0;
    return Some((f64::midpoint(lo, hi), (half, half)));
  }
  try_eval_to_f64(&e).map(|v| (v, (0.0, 0.0)))
}

/// Whether the plot data holds `Interval` values but no `Around`: WL draws
/// an interval as a bar without the end caps an `Around` gets.
fn interval_only_data(data: &Expr) -> bool {
  fn scan(e: &Expr, interval: &mut bool, around: &mut bool) {
    match e {
      Expr::FunctionCall { name, args } => {
        match name.as_str() {
          "Interval" => *interval = true,
          "Around" => *around = true,
          _ => {}
        }
        for a in args {
          scan(a, interval, around);
        }
      }
      Expr::List(items) => {
        for a in items {
          scan(a, interval, around);
        }
      }
      _ => {}
    }
  }
  let (mut interval, mut around) = (false, false);
  scan(data, &mut interval, &mut around);
  interval && !around
}

/// Strip the uncertainties from parsed series, leaving plain (x, y) points.
fn strip_errors(all_series: &[Vec<ErrPoint>]) -> Vec<Vec<(f64, f64)>> {
  all_series
    .iter()
    .map(|series| series.iter().map(|p| (p.x, p.y)).collect())
    .collect()
}

/// Per-series error bars parallel to the series' points, in the
/// `PlotOptions::error_bars` layout. Empty when no point has an uncertainty.
fn collect_error_bars(
  all_series: &[Vec<ErrPoint>],
) -> Vec<Vec<((f64, f64), (f64, f64))>> {
  let has_any = all_series
    .iter()
    .flatten()
    .any(|p| p.dx.0 > 0.0 || p.dx.1 > 0.0 || p.dy.0 > 0.0 || p.dy.1 > 0.0);
  if !has_any {
    return Vec::new();
  }
  all_series
    .iter()
    .map(|series| series.iter().map(|p| (p.dx, p.dy)).collect())
    .collect()
}

/// Expand each point to the extremes of its error bars so that range
/// computation covers the full uncertainty intervals.
fn error_extremes(all_series: &[Vec<ErrPoint>]) -> Vec<Vec<(f64, f64)>> {
  all_series
    .iter()
    .map(|series| {
      series
        .iter()
        .flat_map(|p| {
          [(p.x - p.dx.0, p.y - p.dy.0), (p.x + p.dx.1, p.y + p.dy.1)]
        })
        .collect()
    })
    .collect()
}

/// Render an expression as a short display label, leaving strings unquoted.
fn label_string(e: &Expr) -> String {
  match e {
    Expr::String(s) => s.clone(),
    other => expr_to_output(other),
  }
}

/// Convert an association into plottable list data. Keys that are all numeric
/// become x-coordinates (`<|k -> v|>` → `{{k, v}, …}`); with non-numeric keys
/// the values are plotted at sequential x and, when `keys_label_points` is
/// set (point plots like ListPlot, but not line plots), each key labels its
/// point (`{Labeled[v1, k1], …}`). When every value is itself a list, each
/// value is a separate series and its key becomes the series label.
fn association_to_data(
  pairs: &[(Expr, Expr)],
  keys_label_points: bool,
) -> Expr {
  let all_values_lists =
    !pairs.is_empty() && pairs.iter().all(|(_, v)| matches!(v, Expr::List(_)));
  if all_values_lists {
    return Expr::List(
      pairs
        .iter()
        .map(|(k, v)| {
          call(
            "Labeled",
            vec![canonicalize_element(v), Expr::String(label_string(k))],
          )
        })
        .collect(),
    );
  }
  let all_keys_numeric =
    pairs.iter().all(|(k, _)| eval_to_value_err(k).is_some());
  if all_keys_numeric {
    return Expr::List(
      pairs
        .iter()
        .map(|(k, v)| Expr::List(vec![k.clone(), v.clone()].into()))
        .collect(),
    );
  }
  // Non-numeric keys label their values: `<|"a" -> 2, …|>` plots the values
  // at sequential x with each key drawn as the point's label.
  if keys_label_points {
    return Expr::List(
      pairs
        .iter()
        .map(|(k, v)| {
          call(
            "Labeled",
            vec![canonicalize_element(v), Expr::String(label_string(k))],
          )
        })
        .collect(),
    );
  }
  Expr::List(pairs.iter().map(|(_, v)| v.clone()).collect())
}

/// Head names that ListPlot-family functions (and ordinary `Plot`) treat as
/// transparent display wrappers around a data argument: `Tooltip[data,
/// label]`, `Style[data, ...]`, etc. draw the underlying data with the
/// wrapper's presentation but are otherwise invisible to data validation.
pub const PLOT_DATA_DISPLAY_WRAPPER_HEADS: [&str; 8] = [
  "Tooltip",
  "Style",
  "Legended",
  "Annotation",
  "PopupWindow",
  "Mouseover",
  "StatusArea",
  "Button",
];

/// `ListPlot[Highlighted[data, Placed["XSlice", {{i}, …}]], opts]`: the
/// data is plotted as is, and each highlighted point gets what
/// wolframscript draws for an x slice — a gray vertical line through it, the
/// point enlarged on a white halo and its value beside it — as `Epilog`.
/// `None` when the data is not `Highlighted`.
pub fn expand_highlighted_data(args: &[Expr]) -> Option<Vec<Expr>> {
  let Expr::FunctionCall { name, args: hargs } = &args[0] else {
    return None;
  };
  if name != "Highlighted" || hargs.is_empty() {
    return None;
  }
  let data = crate::evaluator::evaluate_expr_to_expr(&hargs[0]).ok()?;
  let mut extras: Vec<Expr> = Vec::new();
  if let Some(Expr::FunctionCall {
    name: placed,
    args: pargs,
  }) = hargs.get(1)
    && placed == "Placed"
    && pargs.len() == 2
    && matches!(&pargs[0], Expr::String(k) if k == "XSlice")
    && let Expr::List(positions) = &pargs[1]
    && let Expr::List(items) = &data
  {
    for pos in positions {
      let Expr::List(index) = pos else { continue };
      let Some(Expr::Integer(i)) = index.first() else {
        continue;
      };
      let Some(item) = usize::try_from(*i)
        .ok()
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| items.get(i))
      else {
        continue;
      };
      let (x, y) = match item {
        Expr::List(xy) if xy.len() == 2 => (xy[0].clone(), xy[1].clone()),
        y => (Expr::Integer(*i), y.clone()),
      };
      let point = Expr::List(vec![x, y.clone()].into());
      extras.push(Expr::List(
        vec![
          call("GrayLevel", vec![Expr::Real(0.7)]),
          call("AbsoluteThickness", vec![Expr::Integer(1)]),
          // wolframscript draws the slice as the ray up from the point.
          call(
            "ConicHullRegion",
            vec![
              crate::evaluator::evaluate_expr_to_expr(&call(
                "N",
                vec![point.clone()],
              ))
              .unwrap_or_else(|_| point.clone()),
              Expr::List(
                vec![Expr::List(
                  vec![Expr::Integer(0), Expr::Integer(1)].into(),
                )]
                .into(),
              ),
            ],
          ),
          call("PointSize", vec![Expr::Real(0.028875)]),
          call("GrayLevel", vec![Expr::Integer(1)]),
          call("Point", vec![point.clone()]),
          call(
            "RGBColor",
            vec![Expr::Real(0.24), Expr::Real(0.6), Expr::Real(0.8)],
          ),
          call("PointSize", vec![Expr::Real(0.01925)]),
          call("Point", vec![point.clone()]),
          call("GrayLevel", vec![Expr::Integer(0)]),
          call(
            "Text",
            vec![
              y,
              point,
              Expr::List(vec![Expr::Integer(-2), Expr::Integer(-2)].into()),
            ],
          ),
        ]
        .into(),
      ));
    }
  }
  let mut rewritten = vec![data];
  let mut epilog_done = false;
  for opt in &args[1..] {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Epilog") => {
        let mut combined = vec![replacement.as_ref().clone()];
        combined.extend(extras.iter().cloned());
        rewritten.push(crate::helpers::rule_expr(
          Expr::Identifier("Epilog".into()),
          Expr::List(combined.into()),
        ));
        epilog_done = true;
      }
      other => rewritten.push(other.clone()),
    }
  }
  if !epilog_done && !extras.is_empty() {
    rewritten.push(crate::helpers::rule_expr(
      Expr::Identifier("Epilog".into()),
      Expr::List(extras.into()),
    ));
  }
  Some(rewritten)
}

/// Peel a chain of [`PLOT_DATA_DISPLAY_WRAPPER_HEADS`] wrappers to expose
/// the data expression underneath, e.g. `Tooltip[{1, 2, 3}, "cost"]` ->
/// `{1, 2, 3}`. Used both to validate a data argument (is it a wrapper
/// around a list?) and, via [`canonicalize_plot_data`]/[`canonicalize_element`],
/// to actually strip it before parsing.
pub fn strip_plot_data_display_wrapper(expr: &Expr) -> Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if !args.is_empty()
        && PLOT_DATA_DISPLAY_WRAPPER_HEADS.contains(&name.as_str()) =>
    {
      strip_plot_data_display_wrapper(&args[0])
    }
    _ => expr.clone(),
  }
}

/// Strip display wrappers from a single list entry (a scalar, an `{x, y}`
/// pair, or a nested series), recursing into inner lists so that
/// `{1, Style[2, Red], 3}` keeps its numbers. `Callout` is kept as a
/// `Labeled` so its label still draws; an entry `datum -> label` is the
/// same as `Callout[datum, label]`.
fn canonicalize_element(e: &Expr) -> Expr {
  let e = strip_plot_data_display_wrapper(e);
  match &e {
    Expr::Rule {
      pattern,
      replacement,
    } => call(
      "Labeled",
      vec![canonicalize_element(pattern), replacement.as_ref().clone()],
    ),
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      match name.as_str() {
        "Callout" if args.len() >= 2 => call(
          "Labeled",
          std::iter::once(canonicalize_element(&args[0]))
            .chain(args[1..].iter().cloned())
            .collect(),
        ),
        "Callout" => canonicalize_element(&args[0]),
        _ => e.clone(),
      }
    }
    Expr::List(inner) => {
      Expr::List(inner.iter().map(canonicalize_element).collect())
    }
    _ => e,
  }
}

/// Extract the raw data points from a `WeightedData` object. The evaluated
/// canonical form is `WeightedData[Automatic, {{data…}, {weights…}}]`; the
/// user-facing form is `WeightedData[data, weights]`. Either way the data is
/// the list of values (weights only affect statistics, not point positions).
pub(crate) fn weighted_data_values(args: &[Expr]) -> Expr {
  // Canonical `WeightedData[_, {{data}, {weights}}]`.
  if let Some(Expr::List(inner)) = args.get(1)
    && let Some(data @ Expr::List(_)) = inner.first()
  {
    return data.clone();
  }
  // User form `WeightedData[data, …]`.
  match args.first() {
    Some(data @ Expr::List(_)) => data.clone(),
    _ => Expr::List(Vec::new().into()),
  }
}

/// Unwrap the container / display wrappers that ListPlot accepts around its
/// data into a plain list the parser understands: `Tooltip`, `Style`,
/// `Legended`, `Annotation`, `Callout` (kept as a `Labeled` so its label still
/// draws), `SparseArray` (densified), `WeightedData` (points only) and
/// `Association`s. The `data -> labels` rule form is handled by the caller.
fn canonicalize_plot_data(expr: &Expr, keys_label_points: bool) -> Expr {
  let e = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  let e = strip_plot_data_display_wrapper(&e);
  match &e {
    Expr::Association(pairs) => association_to_data(pairs, keys_label_points),
    Expr::List(items) => {
      Expr::List(items.iter().map(canonicalize_element).collect())
    }
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      match name.as_str() {
        "Callout" if args.len() >= 2 => call(
          "Labeled",
          std::iter::once(canonicalize_plot_data(&args[0], keys_label_points))
            .chain(args[1..].iter().cloned())
            .collect(),
        ),
        "Callout" => canonicalize_plot_data(&args[0], keys_label_points),
        "WeightedData" => {
          canonicalize_plot_data(&weighted_data_values(args), keys_label_points)
        }
        "SparseArray" => evaluate_expr_to_expr(&call1("Normal", e.clone()))
          .unwrap_or_else(|_| e.clone()),
        _ => e.clone(),
      }
    }
    _ => e,
  }
}

/// Parse list data from the first argument.
/// Returns a vector of series, each series being a vector of (x, y) points.
///
/// Supported formats:
/// - `{y1, y2, y3}` → single series with x = 1,2,3,...
/// - `{{x1,y1}, {x2,y2}}` → single series with explicit coordinates
/// - `{{y1, y2}, {y3, y4}}` → multiple series (if inner lists don't look like points)
///
/// Any value (y or x/y pair element) may be an `Around[...]`, whose central
/// value becomes the coordinate and whose uncertainty an error bar.
fn parse_list_data_err(
  arg: &Expr,
  keys_label_points: bool,
) -> Result<Vec<Vec<ErrPoint>>, InterpreterError> {
  let data = canonicalize_plot_data(arg, keys_label_points);

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
            // Date-list / DateObject stamps become AbsoluteTime seconds.
            let x = crate::functions::timeseries_ast::to_time(&te)?;
            Some(ErrPoint::from_xy((x, (0.0, 0.0)), eval_to_value_err(&ve)?))
          })
          .collect()
      })
      .collect();
    return Ok(series);
  }

  let Expr::List(items) = &data else {
    return Err(InterpreterError::EvaluationError(
      "ListPlot: first argument must be a list".into(),
    ));
  };

  if items.is_empty() {
    return Ok(vec![vec![]]);
  }

  // Check if it's a list of {x, y} pairs
  if let Expr::List(first_inner) = &items[0] {
    // A leading pair with a non-numeric coordinate (`{0., Indeterminate}`
    // from a `Table` that starts at a singularity) still marks a list of
    // (x, y) pairs as long as every row is a scalar pair and some pair is
    // plottable; the non-numeric rows are skipped below.
    let is_numeric_pair = |item: &Expr| {
      matches!(item, Expr::List(pair) if pair.len() == 2
        && eval_to_value_err(&pair[0]).is_some()
        && eval_to_value_err(&pair[1]).is_some())
    };
    if first_inner.len() == 2
      && (is_numeric_pair(&items[0])
        || (items.iter().all(|item| {
          matches!(item, Expr::List(pair) if pair.len() == 2
            && !pair.iter().any(|e| matches!(e, Expr::List(_))))
        }) && items.iter().any(is_numeric_pair)))
    {
      // {{x1,y1}, {x2,y2}, ...} → single series with explicit coords
      let mut points = Vec::with_capacity(items.len());
      for item in items {
        if let Expr::List(pair) = item
          && pair.len() == 2
          && let (Some(x), Some(y)) =
            (eval_to_value_err(&pair[0]), eval_to_value_err(&pair[1]))
        {
          points.push(ErrPoint::from_xy(x, y));
        }
      }
      return Ok(vec![points]);
    }

    // Check if it's multiple series: {{y1, y2, ...}, {y3, y4, ...}}
    // or multiple series of explicit coords: {{{x1,y1}, ...}, {{x2,y2}, ...}}
    // (a 2-element first row is only ambiguous when its entries are scalars;
    // rows made of {x, y} pairs are always series of explicit coords)
    if first_inner.len() != 2
      || items
        .iter()
        .all(|item| matches!(item, Expr::List(l) if l.len() > 2))
      || (!first_inner.is_empty()
        && first_inner.iter().all(|e| matches!(e, Expr::List(_))))
    {
      let mut all_series = Vec::new();
      for item in items {
        if let Expr::List(series_items) = item {
          let points = parse_single_series(series_items);
          all_series.push(points);
        }
      }
      if !all_series.is_empty() {
        return Ok(all_series);
      }
    }
  }

  // Simple list: {y1, y2, y3} → single series with x = 1,2,3,...
  // A `Labeled[y, …]` entry still contributes its point here (the label is
  // only drawn by the labeled parser).
  let mut points = Vec::with_capacity(items.len());
  for (i, item) in items.iter().enumerate() {
    let value = match unwrap_labeled(item) {
      Some((content, _)) => content,
      None => item.clone(),
    };
    if let Some(y) = eval_to_value_err(&value) {
      points.push(ErrPoint::from_y((i + 1) as f64, y));
    }
  }
  Ok(vec![points])
}

/// Parse a single series: either plain y-values or {x, y} pairs.
fn parse_single_series(
  series_items: &[Expr],
) -> std::vec::Vec<crate::functions::list_plot::ErrPoint> {
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
        && let (Some(x), Some(y)) =
          (eval_to_value_err(&pair[0]), eval_to_value_err(&pair[1]))
      {
        points.push(ErrPoint::from_xy(x, y));
      }
    }
  } else {
    for (i, val) in series_items.iter().enumerate() {
      if let Some(y) = eval_to_value_err(val) {
        points.push(ErrPoint::from_y((i + 1) as f64, y));
      }
    }
  }
  points
}

/// Reinterpret list data according to an explicit `DataRange` option, the
/// way Wolfram does: `DataRange -> All` forces value rows to be read as
/// separate datasets of y-values rather than as (x, y) coordinates. So
/// `ListPlot[{{1, 1}, {2, 2}, {3, 3}}, DataRange -> All]` plots three
/// 2-point datasets, not the three points (1,1), (2,2), (3,3). `All` keeps
/// x = 1, 2, …; `{xmin, xmax}` spaces the x-values evenly over the span,
/// but leaves a list of (x, y) pairs untouched — pairs keep their explicit
/// x-coordinates, so the option has no effect there.
///
/// Returns the rewritten data (y-values paired with explicit x-coordinates)
/// or `None` when the option doesn't change the interpretation (data with
/// explicit coordinates at the innermost level, or shapes we don't handle).
fn apply_data_range(data: &Expr, spec: &DataRangeSpec) -> Option<Expr> {
  let data = evaluate_expr_to_expr(data).unwrap_or_else(|_| data.clone());
  let Expr::List(items) = &data else {
    return None;
  };
  if items.is_empty() {
    return None;
  }

  let x_at = |i: usize, n: usize| -> Expr {
    match spec {
      DataRangeSpec::All => Expr::Integer((i + 1) as i128),
      DataRangeSpec::Span(lo, hi) => {
        if n <= 1 {
          Expr::Real(*lo)
        } else {
          Expr::Real(lo + (hi - lo) * i as f64 / (n - 1) as f64)
        }
      }
    }
  };
  let pair_up = |row: &[Expr]| -> Expr {
    let n = row.len();
    Expr::List(
      row
        .iter()
        .enumerate()
        .map(|(i, y)| Expr::List(vec![x_at(i, n), y.clone()].into()))
        .collect(),
    )
  };

  // Flat list of y-values: only an explicit span changes the x-coordinates.
  if items.iter().all(|it| !matches!(it, Expr::List(_))) {
    if matches!(spec, DataRangeSpec::All) {
      return None;
    }
    return Some(pair_up(items));
  }

  // Matrix of scalar values: every row becomes its own y-value dataset.
  if items.iter().all(|it| {
    matches!(it, Expr::List(row)
      if row.iter().all(|e| !matches!(e, Expr::List(_))))
  }) {
    // Rows of exactly two scalars are (x, y) pairs. They keep their
    // explicit x-coordinates under `DataRange -> {xmin, xmax}`; only
    // `DataRange -> All` forces the separate-dataset reading.
    if matches!(spec, DataRangeSpec::Span(..))
      && items
        .iter()
        .all(|it| matches!(it, Expr::List(row) if row.len() == 2))
    {
      return None;
    }
    return Some(Expr::List(
      items
        .iter()
        .map(|row| match row {
          Expr::List(row) => pair_up(row),
          _ => unreachable!("checked above"),
        })
        .collect(),
    ));
  }

  // Several datasets of mixed shapes (`{values, {{x, y}, …}}`): an explicit
  // span lays out each dataset of plain values, and the datasets of points
  // keep their own coordinates.
  let is_points = |row: &[Expr]| {
    row
      .iter()
      .all(|e| matches!(e, Expr::List(p) if p.len() == 2))
  };
  if matches!(spec, DataRangeSpec::Span(..))
    && items.iter().all(|it| matches!(it, Expr::List(_)))
    && items
      .iter()
      .any(|it| matches!(it, Expr::List(row) if is_points(row)))
  {
    let mut changed = false;
    let out: Vec<Expr> = items
      .iter()
      .map(|it| match it {
        Expr::List(row)
          if !row.is_empty()
            && row.iter().all(|e| !matches!(e, Expr::List(_))) =>
        {
          changed = true;
          pair_up(row)
        }
        other => other.clone(),
      })
      .collect();
    if changed {
      return Some(Expr::List(out.into()));
    }
  }

  None
}

/// The plot's first argument with any explicit `DataRange` option applied.
fn data_arg(args: &[Expr], parsed: &ParsedOptions) -> Expr {
  parsed
    .data_range
    .as_ref()
    .and_then(|spec| apply_data_range(&args[0], spec))
    .unwrap_or_else(|| args[0].clone())
}

/// The value of option `name` among a plot's arguments, if given.
fn option_value<'a>(args: &'a [Expr], name: &str) -> Option<&'a Expr> {
  args[1..].iter().find_map(|a| match a {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name) => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

/// `TargetUnits -> unit`: every `Quantity` of the data converted to `unit`
/// (a quantity of another dimension is kept as it is).
fn convert_to_target_units(data: &Expr, unit: &Expr) -> Expr {
  match data {
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      match crate::evaluator::evaluate_function_call_ast(
        "UnitConvert",
        &[data.clone(), unit.clone()],
      ) {
        Ok(q @ Expr::FunctionCall { .. }) if matches!(&q, Expr::FunctionCall { name, .. } if name == "Quantity") => {
          q
        }
        _ => data.clone(),
      }
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| convert_to_target_units(e, unit))
        .collect(),
    ),
    _ => data.clone(),
  }
}

/// The units of the x and y values of quantity-valued list-plot data
/// (`{q1, q2, …}`, `{{x1, y1}, …}` or a list of such series): those of its
/// first value or point.
fn data_units(data: &Expr) -> (Option<String>, Option<String>) {
  let unit = |e: &Expr| match e {
    Expr::FunctionCall { name, args }
      if name == "Quantity" && args.len() == 2 =>
    {
      match &args[1] {
        Expr::String(u) => Some(u.clone()),
        other => Some(crate::syntax::expr_to_string(other)),
      }
    }
    _ => None,
  };
  let Expr::List(items) = data else {
    return (None, None);
  };
  match items.first() {
    Some(Expr::List(pair))
      if pair.len() == 2
        && !matches!(pair[0], Expr::List(_))
        && pair.iter().any(|e| unit(e).is_some()) =>
    {
      (unit(&pair[0]), unit(&pair[1]))
    }
    Some(Expr::List(_)) => data_units(&items[0]),
    Some(first) => (None, unit(first)),
    None => (None, None),
  }
}

/// `AxesLabel -> Automatic` on quantity data: each axis labelled with the
/// abbreviation of its unit (`QuantityForm[unit, "PlotLabel"]`).
fn automatic_unit_axes_label(data: &Expr) -> Option<(String, String)> {
  let (x, y) = data_units(data);
  if x.is_none() && y.is_none() {
    return None;
  }
  let label = |u: Option<String>| {
    u.map(|u| {
      crate::functions::quantity_ast::unit_abbreviation(&u)
        .map(str::to_string)
        .unwrap_or(u)
    })
    .unwrap_or_default()
  };
  Some((label(x), label(y)))
}

/// `PlotLayout -> "Stacked"` / `"Percentile"`: each series drawn on top of
/// the ones before it — its i-th value the sum of the i-th values of it and
/// all earlier series — and for `"Percentile"` that sum as a percentage of
/// the total of all series.
fn apply_stacked_layout(args: &[Expr], all_series: &mut [Vec<ErrPoint>]) {
  let percentile = match option_value(args, "PlotLayout") {
    Some(Expr::String(s)) if s == "Stacked" => false,
    Some(Expr::String(s)) if s == "Percentile" => true,
    _ => return,
  };
  let len = all_series.iter().map(Vec::len).max().unwrap_or(0);
  for i in 0..len {
    let mut sum = 0.0;
    for series in all_series.iter_mut() {
      if let Some(p) = series.get_mut(i) {
        sum += p.y;
        p.y = sum;
      }
    }
    if percentile && sum != 0.0 {
      for series in all_series.iter_mut() {
        if let Some(p) = series.get_mut(i) {
          p.y = p.y / sum * 100.0;
        }
      }
    }
  }
}

/// One y axis of `MultiaxisArrangement`: the series it scales and the
/// frame edge it is drawn on (1 = left, 3 = right).
struct MultiAxis {
  edge: usize,
  series: Vec<usize>,
}

/// Parse `MultiaxisArrangement -> All | {spec, …} | {Left -> specs,
/// Right -> specs}` for `n` series. A spec is a series index (an axis of
/// its own) or `{i, j, …}` (one axis shared by those series); a side given
/// a flat list `{i, j}` gets one axis per index, listed left to right as
/// they are drawn. `All` gives every series its own axis, the first half
/// (rounded up) on the left; without sides, the last axis goes to the
/// right and the others to the left. The axes come out left to right.
fn parse_multiaxis(args: &[Expr], n: usize) -> Option<Vec<MultiAxis>> {
  let value = option_value(args, "MultiaxisArrangement")?;
  let index = |e: &Expr| match e {
    Expr::Integer(i) if *i >= 1 && (*i as usize) <= n => Some(*i as usize - 1),
    _ => None,
  };
  let group = |e: &Expr| -> Option<Vec<usize>> {
    match e {
      Expr::List(items) => items.iter().map(index).collect(),
      other => index(other).map(|i| vec![i]),
    }
  };
  let mut axes: Vec<(Option<usize>, Vec<usize>)> = Vec::new();
  match value {
    Expr::Identifier(a) if a == "All" => {
      let left = n.div_ceil(2);
      axes
        .extend((0..n).map(|i| (Some(if i < left { 1 } else { 3 }), vec![i])));
    }
    Expr::List(items) => {
      for item in items {
        match item {
          Expr::Rule {
            pattern,
            replacement,
          } => {
            let edge = match pattern.as_ref() {
              Expr::Identifier(s) if s == "Left" => 1,
              Expr::Identifier(s) if s == "Right" => 3,
              _ => return None,
            };
            match replacement.as_ref() {
              // `{{i, j}}`: one shared axis; `{i, j}`: one axis each.
              Expr::List(specs) => {
                for spec in specs {
                  axes.push((Some(edge), group(spec)?));
                }
              }
              other => axes.push((Some(edge), group(other)?)),
            }
          }
          other => axes.push((None, group(other)?)),
        }
      }
    }
    _ => return None,
  }
  if axes.len() < 2 {
    return None;
  }
  let last = axes.len() - 1;
  let mut out: Vec<MultiAxis> = axes
    .into_iter()
    .enumerate()
    .map(|(k, (edge, series))| MultiAxis {
      edge: edge.unwrap_or(if k == last { 3 } else { 1 }),
      series,
    })
    .collect();
  // Left to right: the left axes (in their order), then the right ones.
  out.sort_by_key(|a| a.edge);
  Some(out)
}

/// Draw the series on the y axes of `MultiaxisArrangement`: every axis
/// gets the automatic range of its own series, each series is mapped from
/// its axis's range onto the first (left) one's, and the frame's left and
/// right edges carry the ticks of their axis, coloured like its series.
fn apply_multiaxis(
  args: &[Expr],
  err_series: &mut [Vec<ErrPoint>],
  opts: &mut PlotOptions,
) {
  let Some(axes) = parse_multiaxis(args, err_series.len()) else {
    return;
  };
  let ranges: Vec<(f64, f64)> = axes
    .iter()
    .map(|axis| {
      let (lo, hi) = axis
        .series
        .iter()
        .flat_map(|&i| err_series[i].iter())
        .filter(|p| p.y.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
          (lo.min(p.y - p.dy.0), hi.max(p.y + p.dy.1))
        });
      let extent = list_axis_extent(lo, hi, ListAxis::Y, false, false, false);
      pad_list_axis(extent, ListAxis::Y, false)
    })
    .collect();
  // The axis of each side drawn on the frame edge is the one next to the
  // plot (the last left one, the first right one); the others stand
  // outside it, in order outward. The frame's left axis is the reference
  // every series is mapped onto.
  let left_frame = axes.iter().rposition(|a| a.edge == 1);
  let right_frame = axes.iter().position(|a| a.edge == 3);
  let reference = left_frame.or(right_frame).unwrap_or(0);
  let (ref_lo, ref_hi) = ranges[reference];
  let palette = &crate::functions::plot::PLOT_COLORS_RGB;
  let series_color = |i: usize| {
    opts
      .plot_style
      .get(i)
      .and_then(|s| s.color)
      .unwrap_or_else(|| {
        let (r, g, b) = palette[i % palette.len()];
        crate::functions::graphics::Color::new(r, g, b)
      })
  };
  opts.frame = true;
  // Outward rank of each axis among those of its side (0 = the frame
  // edge itself).
  let rank = |k: usize| -> usize {
    match axes[k].edge {
      1 => left_frame.map_or(0, |f| f - k),
      _ => right_frame.map_or(0, |f| k - f),
    }
  };
  let mut extra: Vec<(usize, usize, crate::functions::plot::ExtraFrameAxis)> =
    Vec::new();
  for (k, (axis, &(lo, hi))) in axes.iter().zip(&ranges).enumerate() {
    let scale = (ref_hi - ref_lo) / (hi - lo);
    let map = |y: f64| ref_lo + (y - lo) * scale;
    for &i in &axis.series {
      for p in &mut err_series[i] {
        p.y = map(p.y);
        p.dy = (p.dy.0 * scale, p.dy.1 * scale);
      }
    }
    let ticks = crate::functions::plot_axes::AxisTicks::new(
      crate::functions::plot_axes::AxisScale {
        min: lo,
        max: hi,
        log: false,
      },
      false,
      None,
    );
    let mut explicit: Vec<crate::functions::plot::ExplicitTick> = ticks
      .majors
      .iter()
      .map(|(pos, label)| crate::functions::plot::ExplicitTick {
        pos: map(*pos),
        label: label.clone(),
        len: None,
        style: None,
      })
      .collect();
    explicit.extend(ticks.minors.iter().map(|pos| {
      crate::functions::plot::ExplicitTick {
        pos: map(*pos),
        label: String::new(),
        len: Some((0.00375, 0.0)),
        style: None,
      }
    }));
    // A shared axis is split evenly among its series' colours, the first
    // at the top.
    let shares = axis.series.len() as f64;
    let segments: Vec<(f64, f64, crate::functions::graphics::Color)> = axis
      .series
      .iter()
      .enumerate()
      .map(|(j, &i)| {
        let top = 1.0 - j as f64 / shares;
        (top - 1.0 / shares, top, series_color(i))
      })
      .collect();
    if rank(k) == 0 {
      opts.frame_tick_edges[axis.edge] =
        crate::functions::plot::FrameEdgeTicks::Explicit(explicit);
      for (from, to, color) in segments {
        opts.frame_edge_segments.push((axis.edge, from, to, color));
      }
    } else {
      extra.push((
        axis.edge,
        rank(k),
        crate::functions::plot::ExtraFrameAxis {
          edge: axis.edge,
          ticks: explicit,
          segments,
        },
      ));
    }
  }
  extra.sort_by_key(|(edge, rank, _)| (*edge, *rank));
  opts.extra_frame_axes = extra.into_iter().map(|(_, _, a)| a).collect();
}

/// Whether the point labels of the data argument are callouts — pulled
/// away from their points along leader lines — rather than plain labels
/// beside them: `Callout[pt, label]` entries and the `data -> labels` form
/// are; `Labeled[pt, label]` entries are not.
fn labels_are_callouts(data: &Expr) -> bool {
  fn has_callout(e: &Expr, depth: usize) -> bool {
    match e {
      Expr::FunctionCall { name, .. } if name == "Callout" => true,
      // An entry `datum -> label` is a callout.
      Expr::Rule { .. } if depth < 2 => true,
      Expr::List(items) if depth > 0 => {
        items.iter().any(|item| has_callout(item, depth - 1))
      }
      _ => false,
    }
  }
  matches!(data, Expr::Rule { .. }) || has_callout(data, 2)
}

/// If `expr` is `Labeled[content, label, ...]`, return the content and the
/// label rendered as display text.
fn unwrap_labeled(expr: &Expr) -> Option<(Expr, String)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Labeled"
    && args.len() >= 2
  {
    Some((
      args[0].clone(),
      crate::functions::graphics::traditional_label(&args[1]),
    ))
  } else {
    None
  }
}

/// One parsed series per dataset plus one optional dataset label per series.
type LabeledSeries = (Vec<Vec<(f64, f64)>>, Vec<Option<String>>);

/// A point's label and, from `Labeled[pt, label, pos]`, the side of the
/// point it is placed on (`Above`, `Below`, `Left`, `Right`, …).
type PointLabel = (String, Option<String>);

/// Like [`LabeledSeries`], but keeping the `Around` uncertainties plus
/// per-point labels (parallel to each series' points) from `Labeled`
/// wrappers around scalar data entries.
type LabeledErrSeries = (
  Vec<Vec<ErrPoint>>,
  Vec<Option<String>>,
  Vec<Vec<Option<PointLabel>>>,
);

/// The side a `Labeled[pt, label, pos]` places its label on, in the names
/// the point-label renderer reads (`Before` / `After` are `Left` /
/// `Right`). Other positions (a number, `{…}`) give `None`.
fn labeled_point_position(expr: &Expr) -> Option<String> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  if name != "Labeled" {
    return None;
  }
  match args.get(2)? {
    Expr::Identifier(pos) => Some(
      match pos.as_str() {
        "Above" | "Top" => "Above",
        "Below" | "Bottom" => "Below",
        "Before" | "Left" => "Left",
        "After" | "Right" => "Right",
        "Center" => "Center",
        _ => return None,
      }
      .to_string(),
    ),
    _ => None,
  }
}

/// Like [`parse_list_data_err_labeled`], with the uncertainties and
/// per-point labels stripped.
fn parse_list_data_labeled(
  arg: &Expr,
) -> Result<LabeledSeries, InterpreterError> {
  let (series, labels, _) = parse_list_data_err_labeled(arg, false)?;
  Ok((strip_errors(&series), labels))
}

/// Parse `ListPlot[ts -> "key"]` / `ListPlot[ts -> {"k1", …}]`: select the
/// named components of a component-keyed TimeSeries, one series per key.
fn parse_temporal_components(ts: &Expr, selector: &Expr) -> LabeledErrSeries {
  let pairs =
    crate::functions::timeseries_ast::time_series_pairs(ts).unwrap_or_default();
  let sel =
    evaluate_expr_to_expr(selector).unwrap_or_else(|_| selector.clone());
  let keys: Vec<String> = match &sel {
    Expr::List(items) => items.iter().map(label_string).collect(),
    other => vec![label_string(other)],
  };
  let series: Vec<Vec<ErrPoint>> = keys
    .iter()
    .map(|key| {
      pairs
        .iter()
        .filter_map(|(t, v)| {
          let x = crate::functions::timeseries_ast::to_time(t)?;
          let comp = match v {
            Expr::Association(kv) => kv
              .iter()
              .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
              .map(|(_, val)| val.clone()),
            _ => None,
          }?;
          Some(ErrPoint::from_xy(
            (x, (0.0, 0.0)),
            eval_to_value_err(&comp)?,
          ))
        })
        .collect()
    })
    .collect();
  (series, Vec::new(), Vec::new())
}

/// Parse the `ListPlot[data -> labels]` form: the left side is the data
/// (points), the right side a parallel list of per-point labels attached to
/// the first series.
fn parse_rule_labeled(
  lhs: &Expr,
  rhs: &Expr,
) -> Result<LabeledErrSeries, InterpreterError> {
  let series = parse_list_data_err(lhs, false)?;
  let rhs = evaluate_expr_to_expr(rhs).unwrap_or_else(|_| rhs.clone());
  let labels: Vec<Option<String>> = match &rhs {
    Expr::List(items) => {
      items.iter().map(|it| Some(label_string(it))).collect()
    }
    other => vec![Some(label_string(other))],
  };
  let point_labels: Vec<Vec<Option<PointLabel>>> = series
    .iter()
    .enumerate()
    .map(|(i, s)| {
      if i == 0 {
        (0..s.len())
          .map(|j| labels.get(j).cloned().flatten().map(|l| (l, None)))
          .collect()
      } else {
        vec![None; s.len()]
      }
    })
    .collect();
  Ok((series, Vec::new(), point_labels))
}

/// Parse list data that may carry `Labeled[data, label]` wrappers around the
/// whole argument or around individual datasets. Returns one series per
/// dataset plus one optional label per series (empty when no wrapper was
/// present) plus per-point labels (empty when no point is labeled).
///
/// A Labeled wrapper on a list entry of the outer list marks that list as a
/// list of datasets, so `{Labeled[{1, 2}, "a"], Labeled[{3, 4}, "b"]}`
/// parses as two 2-point series rather than as a list of x/y pairs. When
/// every entry is a scalar, the Labeled wrappers instead label individual
/// points of a single series: `{Labeled[2, "a"], 3}` plots at x = 1, 2.
///
/// `keys_label_points` controls whether the non-numeric keys of an
/// association become per-point labels (ListPlot draws them, line plots
/// don't).
fn parse_list_data_err_labeled(
  arg: &Expr,
  keys_label_points: bool,
) -> Result<LabeledErrSeries, InterpreterError> {
  let raw = evaluate_expr_to_expr(arg)?;

  // `ts -> component(s)` selects named components of a TimeSeries; every other
  // `data -> labels` attaches the right side as per-point labels.
  if let Expr::Rule {
    pattern,
    replacement,
  } = &raw
  {
    let lhs =
      evaluate_expr_to_expr(pattern).unwrap_or_else(|_| *pattern.clone());
    if crate::functions::timeseries_ast::time_series_pairs(&lhs).is_some() {
      return Ok(parse_temporal_components(&lhs, replacement));
    }
    return parse_rule_labeled(pattern, replacement);
  }

  // Normalize container / display wrappers (Tooltip, Style, Callout,
  // SparseArray, WeightedData, associations, …) into plain list data.
  let data = canonicalize_plot_data(&raw, keys_label_points);

  // Whole argument wrapped: the label applies to every series inside.
  if let Some((content, label)) = unwrap_labeled(&data) {
    let series = parse_list_data_err(&content, keys_label_points)?;
    let labels = vec![Some(label); series.len()];
    return Ok((series, labels, Vec::new()));
  }

  if let Expr::List(items) = &data
    && items.iter().any(|item| unwrap_labeled(item).is_some())
  {
    let unwrapped: Vec<(Expr, Option<PointLabel>)> = items
      .iter()
      .map(|item| {
        let (content, label) = match unwrap_labeled(item) {
          Some((content, label)) => {
            (content, Some((label, labeled_point_position(item))))
          }
          None => (item.clone(), None),
        };
        let content = evaluate_expr_to_expr(&content).unwrap_or(content);
        (content, label)
      })
      .collect();

    // All entries scalar: one series with sequential x values and the
    // labels attached to the individual points.
    if unwrapped.iter().all(|(c, _)| !matches!(c, Expr::List(_))) {
      let mut points = Vec::with_capacity(unwrapped.len());
      let mut point_labels = Vec::with_capacity(unwrapped.len());
      for (i, (content, label)) in unwrapped.iter().enumerate() {
        if let Some(y) = eval_to_value_err(content) {
          points.push(ErrPoint::from_y((i + 1) as f64, y));
          point_labels.push(label.clone());
        }
      }
      return Ok((vec![points], Vec::new(), vec![point_labels]));
    }

    // All entries {x, y} pairs (e.g. `Callout[{x, y}, label]` /
    // `Labeled[{x, y}, label]` around individual coordinates): the wrappers
    // label individual points of a single series, not separate datasets.
    let is_numeric_pair = |c: &Expr| {
      matches!(c, Expr::List(pair) if pair.len() == 2
        && eval_to_value_err(&pair[0]).is_some()
        && eval_to_value_err(&pair[1]).is_some())
    };
    if unwrapped.iter().all(|(c, _)| is_numeric_pair(c)) {
      let mut points = Vec::with_capacity(unwrapped.len());
      let mut point_labels = Vec::with_capacity(unwrapped.len());
      for (content, label) in &unwrapped {
        if let Expr::List(pair) = content
          && let (Some(x), Some(y)) =
            (eval_to_value_err(&pair[0]), eval_to_value_err(&pair[1]))
        {
          points.push(ErrPoint::from_xy(x, y));
          point_labels.push(label.clone());
        }
      }
      return Ok((vec![points], Vec::new(), vec![point_labels]));
    }

    let mut all_series = Vec::with_capacity(unwrapped.len());
    let mut labels = Vec::with_capacity(unwrapped.len());
    for (content, label) in unwrapped {
      match &content {
        Expr::List(series_items) => {
          all_series.push(parse_single_series(series_items));
        }
        // A labeled scalar among datasets contributes a one-point series.
        other => match eval_to_value_err(other) {
          Some(y) => all_series.push(vec![ErrPoint::from_y(1.0, y)]),
          None => all_series.push(Vec::new()),
        },
      }
      // A dataset's label keeps no point position (its own placement is
      // read from the data by `apply_dataset_labels`).
      labels.push(label.map(|(text, _)| text));
    }
    return Ok((all_series, labels, Vec::new()));
  }

  Ok((
    parse_list_data_err(&data, keys_label_points)?,
    Vec::new(),
    Vec::new(),
  ))
}

/// An explicit `DataRange` option value.
enum DataRangeSpec {
  /// `DataRange -> All`: rows of a matrix are separate y-value datasets
  /// plotted at x = 1, 2, ….
  All,
  /// `DataRange -> {xmin, xmax}`: y-values are spaced evenly over the span.
  Span(f64, f64),
}

/// Parsed list-plot options, including explicit PlotRange overrides.
#[derive(Default)]
struct ParsedOptions {
  opts: PlotOptions,
  joined: bool,
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  /// `PlotRange -> All` (or `All` in the slot of an axis): the axis shows
  /// every point, outliers included.
  plot_range_all: (bool, bool),
  /// `LabelingFunction -> None` (or `Tooltip`): no static point labels drawn.
  hide_point_labels: bool,
  /// `LabelingFunction -> f`: every point labelled `f[{x, y}, {i, j}]` (its
  /// coordinates, and its dataset's and its own index).
  labeling_function: Option<Expr>,
  /// `LabelingSize -> w`: a label wider than `w` points is left to a
  /// tooltip rather than drawn.
  labeling_size: Option<f64>,
  /// `InterpolationOrder -> n` for joined curves: 0 draws piecewise-constant
  /// steps, 1 straight lines, >= 2 a smooth interpolated curve. `None`
  /// (option absent or `-> None`) keeps the default straight lines.
  interpolation_order: Option<i128>,
  /// `DataRange -> All` / `{xmin, xmax}` (None for the Automatic default).
  data_range: Option<DataRangeSpec>,
  /// `ScalingFunctions -> {sx, sy}` (the first one given).
  scaling: Option<(Option<Scaling>, Option<Scaling>)>,
}

thread_local! {
  /// Set while `DateListPlot` draws through the list-plot machinery: the
  /// x values are AbsoluteTime seconds, labelled as dates.
  static DATE_AXIS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `f` with the list plots it draws labelling their x axis with dates.
pub(crate) fn with_date_axis<T>(f: impl FnOnce() -> T) -> T {
  struct Reset(bool);
  impl Drop for Reset {
    fn drop(&mut self) {
      DATE_AXIS.with(|d| d.set(self.0));
    }
  }
  let _reset = Reset(DATE_AXIS.with(|d| d.replace(true)));
  f()
}

/// Whether list-plot data is time-stamped by dates: a `TimeSeries` /
/// `TemporalData` (or `ts -> components`) whose first stamp is a date list
/// or `DateObject` rather than a plain number.
fn has_date_stamps(data: &Expr) -> bool {
  let data = match data {
    Expr::Rule { pattern, .. } => {
      evaluate_expr_to_expr(pattern).unwrap_or_else(|_| *pattern.clone())
    }
    other => other.clone(),
  };
  let Some(paths) = crate::functions::timeseries_ast::temporal_paths(&data)
  else {
    return false;
  };
  paths.iter().flatten().next().is_some_and(|(t, _)| {
    let t = evaluate_expr_to_expr(t).unwrap_or_else(|_| t.clone());
    !matches!(t, Expr::Integer(_) | Expr::Real(_) | Expr::BigInteger(_))
  })
}

/// Parse common plot options from args[1..].
fn parse_plot_options(args: &[Expr]) -> ParsedOptions {
  let mut out = ParsedOptions::default();
  out.opts.date_axis = DATE_AXIS.with(std::cell::Cell::get);
  let (options, simple_ticks) = expand_plot_theme(&args[1..], true);
  let opts = &mut out.opts;
  opts.simple_ticks = simple_ticks;
  // Joined data points are joined as given, also across a jump.
  opts.join_across_jumps = true;
  for opt in &options {
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      match name {
        "ScalingFunctions" if out.scaling.is_none() => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          out.scaling = Scaling::parse_option(&value);
        }
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
          {
            opts.svg_width = w;
            opts.svg_height = h;
            opts.full_width = fw;
            opts.image_size_value = Some(replacement.clone());
          }
        }
        "Joined" => match replacement {
          Expr::Identifier(v) if v == "True" => out.joined = true,
          // `Joined -> {b1, b2, ...}`: join series `i` only when `bi` is
          // True, leaving the rest as scattered points (ListPlot only).
          Expr::List(items) => {
            let flags: Vec<bool> = items
              .iter()
              .map(|e| matches!(e, Expr::Identifier(v) if v == "True"))
              .collect();
            out.joined = flags.iter().any(|&b| b);
            opts.joined_per_series = Some(flags);
          }
          _ => {}
        },
        "PlotRange" => {
          let (rx, ry) =
            crate::functions::plot::parse_series_plot_range(replacement);
          out.plot_range_x = rx;
          out.plot_range_y = ry;
          out.plot_range_all =
            crate::functions::plot::plot_range_all_axes(replacement);
        }
        "DataRange" => {
          out.data_range = match replacement {
            Expr::Identifier(v) if v == "All" => Some(DataRangeSpec::All),
            Expr::List(pair) if pair.len() == 2 => {
              match (try_eval_to_f64(&pair[0]), try_eval_to_f64(&pair[1])) {
                (Some(lo), Some(hi)) => Some(DataRangeSpec::Span(lo, hi)),
                _ => None,
              }
            }
            // A bare number `xmax` spans the data over {0, xmax}.
            other => try_eval_to_f64(other)
              .filter(|hi| hi.is_finite())
              .map(|hi| DataRangeSpec::Span(0.0, hi)),
          };
        }
        "PlotLayout" => {
          opts.plot_layout =
            crate::functions::plot_layout::parse_plot_layout(replacement);
        }
        "AspectRatio" => {
          // Ratio (height/width) of the plotting area; the total image height
          // is derived once margins are known (see generate_svg_with_options).
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          if let Some(r) = try_eval_to_f64(&val)
            && r > 0.0
          {
            opts.aspect_ratio = Some(r);
          } else if matches!(&val, Expr::Identifier(s) if s == "Automatic") {
            opts.aspect_automatic = true;
          }
        }
        "Filling" => {
          apply_filling_option(replacement, opts);
        }
        "FillingStyle" => {
          opts.filling_style =
            crate::functions::plot::parse_filling_style(replacement);
        }
        "Background" => {
          opts.background =
            crate::functions::plot::parse_background_option(replacement);
        }
        "PlotStyle" => {
          opts.plot_style = parse_plot_style(replacement);
        }
        // `PlotMarkers -> marker` draws that glyph at every data point
        // instead of the default round dot, one marker per series.
        "PlotMarkers" => {
          opts.plot_markers =
            crate::functions::plot::parse_plot_markers(replacement);
        }
        // Epilog/Prolog primitives and the axes origin are read by the shared
        // plot-option reader, so the list plots and `Plot` cannot drift on
        // what they accept.
        "Epilog"
        | "Prolog"
        | "AxesOrigin"
        | "ColorFunction"
        | "ColorFunctionScaling"
        | "AxesStyle"
        | "TicksStyle"
        | "FrameStyle"
        | "FrameTicks"
        | "FrameTicksStyle"
        | "LabelStyle"
        | "Mesh"
        | "MeshFunctions"
        | "MeshShading"
        | "MeshStyle"
        | "PlotLabels"
        | "PlotLegends" => {
          let mut overrides =
            crate::functions::plot::PlotRangeOverrides::default();
          crate::functions::plot::apply_common_plot_option(
            name,
            replacement,
            opts,
            &mut overrides,
          );
        }
        // Labels that only appear on hover (Tooltip) or are switched off
        // (None) are not drawn on a static SVG.
        "LabelingFunction" => {
          if matches!(replacement,
            Expr::Identifier(v) if v == "None" || v == "Tooltip")
          {
            out.hide_point_labels = true;
          } else if let Expr::Identifier(pos) = replacement
            && matches!(
              pos.as_str(),
              "Above"
                | "Below"
                | "Left"
                | "Right"
                | "Center"
                | "Top"
                | "Bottom"
            )
          {
            // A bare position places every point label on that side.
            opts.point_label_position = Some(pos.clone());
          } else if matches!(
            replacement,
            Expr::Function { .. } | Expr::NamedFunction { .. }
          ) || matches!(replacement, Expr::FunctionCall { name, .. } if name == "Function")
          {
            out.labeling_function = Some(replacement.clone());
          }
        }
        "LabelingSize" => {
          out.labeling_size = crate::functions::graphics::expr_to_f64(
            &evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.clone()),
          );
        }
        "PlotLabel" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
            opts.plot_label = Some(sl);
          }
        }
        // AxesLabel is always the 2-element `{x, y}` (bottom, left) form.
        "AxesLabel" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
          if let Expr::List(items) = &val
            && !items.is_empty()
          {
            let x = crate::functions::graphics::styled_axis_label(&items[0])
              .unwrap_or_default();
            // `{xlabel}` names the x axis only.
            let y = items
              .get(1)
              .and_then(crate::functions::graphics::styled_axis_label)
              .unwrap_or_default();
            opts.axes_label = Some((x, y));
          }
        }
        // FrameLabel supports both `{bottom, left}` and the nested
        // `{{left, right}, {bottom, top}}` form. Bottom/left reuse the
        // axes-label render path; top/right get their own frame edges.
        "FrameLabel" => {
          crate::functions::plot::apply_frame_label_option(replacement, opts);
        }
        // `Ticks -> {xspec, yspec}` marks exactly the positions named,
        // each optionally with its own `{pos, label}` text.
        "Ticks" => match replacement {
          Expr::Identifier(v) if v == "None" => opts.ticks = false,
          Expr::Identifier(v) if v == "Automatic" || v == "All" => {
            opts.ticks = true;
          }
          Expr::List(items) if (1..=2).contains(&items.len()) => {
            opts.ticks_x =
              crate::functions::plot::parse_explicit_ticks(&items[0]);
            if let Some(y) = items.get(1) {
              opts.ticks_y = crate::functions::plot::parse_explicit_ticks(y);
            }
          }
          _ => {}
        },
        "Frame" => {
          crate::functions::plot::apply_frame_option(replacement, opts);
        }
        "Axes" => {
          if let Some(axes) =
            crate::functions::plot::parse_axes_option(replacement)
          {
            opts.axes = axes;
          }
        }
        "ImagePadding" => {
          opts.image_padding =
            crate::functions::plot::parse_image_padding(replacement);
        }
        "IntervalMarkersStyle" => {
          opts.interval_markers_color =
            crate::functions::plot::parse_style_directives(replacement)
              .and_then(|st| st.color)
              .map(|c| {
                let (r, g, b) = crate::functions::plot::color_to_rgb8(c);
                plotters::style::RGBColor(r, g, b)
              });
        }
        "ClippingStyle" => {
          opts.clipping_style =
            crate::functions::plot::parse_clipping_style(replacement);
        }
        // "Fences" (capped error bars) is the default; "Bars" drops the
        // caps. None hides the uncertainty intervals.
        "IntervalMarkers" => match replacement {
          Expr::String(s) if s == "Bands" => {
            opts.interval_markers = IntervalMarkers::Bands;
          }
          Expr::String(s) if s == "Bars" => {
            opts.interval_markers = IntervalMarkers::Bars;
          }
          Expr::Identifier(v) if v == "None" => {
            opts.interval_markers = IntervalMarkers::None;
          }
          _ => {}
        },
        // `None` (the default) means straight lines, like order 1.
        "InterpolationOrder" => {
          if let Some(n) = crate::functions::math_ast::expr_to_i128(replacement)
            && n >= 0
          {
            out.interpolation_order = Some(n);
          }
        }
        "GridLines" => {
          let val = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone());
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
        "GridLinesStyle" => {
          opts.grid_lines_style =
            crate::functions::plot::parse_style_directives(replacement);
        }
        _ => {}
      }
    }
  }
  out
}

/// What `PlotRange` asks of the automatic range of each axis.
#[derive(Clone, Copy)]
struct RangeRequest {
  /// The axis range is given explicitly.
  explicit: (bool, bool),
  /// `All`: the whole extent of the points, outliers included.
  keep_outliers: (bool, bool),
}

impl ParsedOptions {
  /// What `PlotRange` fixes of the x and the y range.
  fn explicit_range(&self) -> RangeRequest {
    RangeRequest {
      explicit: (self.plot_range_x.is_some(), self.plot_range_y.is_some()),
      keep_outliers: self.plot_range_all,
    }
  }
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
    log_safe_range(parsed.plot_range_x, x_range),
    log_safe_range(parsed.plot_range_y, y_range),
  )
}

/// An explicit `PlotRange` side, with the ends a scaled axis cannot reach
/// (`PlotRange -> {0, 1}` on a log axis) left at the automatic ones, as
/// the Wolfram Language does.
fn log_safe_range(
  explicit: Option<(f64, f64)>,
  auto: (f64, f64),
) -> (f64, f64) {
  crate::functions::plot::fill_range(explicit, auto).unwrap_or(auto)
}

/// Where `ListStepPlot` puts each value's step relative to its data point.
#[derive(Clone, Copy, PartialEq, Debug)]
enum StepPosition {
  /// The step starts at its point (the default).
  Right,
  /// The step ends at its point.
  Left,
  /// The step is centred on its point.
  Center,
}

impl StepPosition {
  fn parse(e: &Expr) -> Option<Self> {
    match e {
      Expr::Identifier(n) => match n.as_str() {
        "Right" => Some(Self::Right),
        "Left" => Some(Self::Left),
        "Center" => Some(Self::Center),
        _ => None,
      },
      Expr::List(items) if items.len() == 1 => Self::parse(&items[0]),
      _ => None,
    }
  }
}

/// The staircase `ListStepPlot` draws through one series: each value is
/// held over its own x interval, delimited by the neighbouring points
/// (`Right`: up to the next point, `Left`: from the previous one,
/// `Center`: between the midpoints). The open end steps reuse the width
/// of the adjacent gap (half of it for `Center`); a lone point gets a
/// unit step that lies on the opposite side, as in the Wolfram Language.
fn positioned_step_curve(
  series: &[(f64, f64)],
  position: StepPosition,
) -> Vec<(f64, f64)> {
  let n = series.len();
  if n == 0 {
    return Vec::new();
  }
  let (x_first, x_last) = (series[0].0, series[n - 1].0);
  let (gap_first, gap_last) = if n >= 2 {
    (series[1].0 - x_first, x_last - series[n - 2].0)
  } else {
    (1.0, 1.0)
  };
  // The x interval of step `i`.
  let interval = |i: usize| -> (f64, f64) {
    let x = series[i].0;
    let prev = if i > 0 {
      series[i - 1].0
    } else {
      x - gap_first
    };
    let next = if i + 1 < n {
      series[i + 1].0
    } else {
      x + gap_last
    };
    match (position, n) {
      (StepPosition::Right, 1) => (x - 1.0, x),
      (StepPosition::Left, 1) => (x, x + 1.0),
      (StepPosition::Right, _) => (x, next),
      (StepPosition::Left, _) => (prev, x),
      (StepPosition::Center, _) => {
        (f64::midpoint(prev, x), f64::midpoint(x, next))
      }
    }
  };
  let mut steps = Vec::with_capacity(2 * n);
  for (i, &(_, y)) in series.iter().enumerate() {
    let (a, b) = interval(i);
    steps.push((a, y));
    steps.push((b, y));
  }
  steps
}

/// Turn one series into the staircase that zero-order interpolation draws:
/// each value is held until the next data point, where the curve jumps
/// vertically (ListLinePlot's `InterpolationOrder -> 0`).
fn step_curve(series: &[(f64, f64)]) -> Vec<(f64, f64)> {
  let mut steps = Vec::with_capacity(series.len() * 2);
  for (i, &(x, y)) in series.iter().enumerate() {
    if i > 0 {
      steps.push((x, series[i - 1].1));
    }
    steps.push((x, y));
  }
  steps
}

/// The cardinal B-spline of degree `degree` on the knots `0, 1, …,
/// degree + 1`, at `x` (Cox–de Boor recursion on uniform knots).
fn cardinal_bspline(x: f64, degree: usize) -> f64 {
  if x < 0.0 || x >= (degree + 1) as f64 {
    return 0.0;
  }
  if degree == 0 {
    return 1.0;
  }
  let k = degree as f64;
  (x * cardinal_bspline(x, degree - 1)
    + (k + 1.0 - x) * cardinal_bspline(x - 1.0, degree - 1))
    / k
}

/// Solve the banded system `matrix · c = rhs` (half-bandwidth `band`,
/// rows stored densely) by Gaussian elimination without pivoting — the
/// B-spline collocation matrices it is used on are totally positive.
fn solve_banded(matrix: &mut [Vec<f64>], rhs: &mut [(f64, f64)], band: usize) {
  let n = rhs.len();
  for col in 0..n {
    let pivot = matrix[col][col];
    if pivot.abs() < f64::MIN_POSITIVE {
      continue;
    }
    let last = (col + band).min(n - 1);
    for row in col + 1..=last {
      let factor = matrix[row][col] / pivot;
      if factor == 0.0 {
        continue;
      }
      for k in col..=(col + 2 * band).min(n - 1) {
        matrix[row][k] -= factor * matrix[col][k];
      }
      rhs[row].0 -= factor * rhs[col].0;
      rhs[row].1 -= factor * rhs[col].1;
    }
  }
  for col in (0..n).rev() {
    let (mut sx, mut sy) = rhs[col];
    for k in col + 1..=(col + 2 * band).min(n - 1) {
      sx -= matrix[col][k] * rhs[k].0;
      sy -= matrix[col][k] * rhs[k].1;
    }
    let pivot = matrix[col][col];
    rhs[col] = (sx / pivot, sy / pivot);
  }
}

/// Sample the smooth order-`order` curve a joined list plot draws through
/// `series`, as the Wolfram Language does: a uniform parametric B-spline of
/// degree `order` in the point index, its knots at the indices (odd
/// orders) or halfway between them (even orders), with the `order / 2`
/// control points beyond each end equal to the end control point, and the
/// control points solved so that the curve passes through every point at
/// its index. Each index interval is sampled `2 ⌊1200 / (n - 1)⌋` times,
/// clamped to `6..=26`.
fn spline_curve(series: &[(f64, f64)], order: usize) -> Vec<(f64, f64)> {
  let n = series.len();
  let degree = order.min(n.saturating_sub(1));
  if degree <= 1 {
    return series.to_vec();
  }
  let pad = degree / 2;
  let controls = n + 2 * pad;
  // Basis `j` (control `j - pad`) starts at knot `first_knot + j`.
  let first_knot = -(pad as f64) - (degree + 1) as f64 / 2.0;
  let basis =
    |j: usize, t: f64| cardinal_bspline(t - first_knot - j as f64, degree);
  // Control `j` stands for unknown `clamp(j - pad, 0, n - 1)`.
  let unknown =
    |j: usize| (j as i64 - pad as i64).clamp(0, n as i64 - 1) as usize;
  let mut matrix = vec![vec![0.0; n]; n];
  for (i, row) in matrix.iter_mut().enumerate() {
    for j in i..=(i + 2 * pad).min(controls - 1) {
      row[unknown(j)] += basis(j, i as f64);
    }
  }
  let mut solved = series.to_vec();
  solve_banded(&mut matrix, &mut solved, 2 * pad);
  let eval = |t: f64| {
    let (mut x, mut y) = (0.0, 0.0);
    let lo = (t - first_knot).floor() as i64 - degree as i64;
    for j in lo.max(0)..=(lo + degree as i64).min(controls as i64 - 1) {
      let w = basis(j as usize, t);
      let (cx, cy) = solved[unknown(j as usize)];
      x += w * cx;
      y += w * cy;
    }
    (x, y)
  };
  let per = (2 * (1200 / (n - 1))).clamp(6, 26);
  let mut curve = Vec::with_capacity((n - 1) * per + 1);
  for idx in 0..n - 1 {
    curve.push(series[idx]);
    for k in 1..per {
      curve.push(eval(idx as f64 + k as f64 / per as f64));
    }
  }
  curve.push(series[n - 1]);
  curve
}

/// Replace each series by the curve `InterpolationOrder -> order` draws
/// through its points. Order 1 (the straight-line default) returns the data
/// unchanged.
fn interpolate_series(
  all_series: &[Vec<(f64, f64)>],
  order: i128,
) -> Vec<Vec<(f64, f64)>> {
  all_series
    .iter()
    .map(|series| match order {
      0 => step_curve(series),
      o if o >= 2 => spline_curve(series, o.min(16) as usize),
      _ => series.clone(),
    })
    .collect()
}

/// Make the automatic plot range cover a numeric `AxesOrigin`, by adding
/// the origin's coordinates as range-only points to every series.
fn include_axes_origin(
  range_series: &mut [Vec<(f64, f64)>],
  origin: (Option<f64>, Option<f64>),
) {
  for series in range_series.iter_mut() {
    let Some(&(x0, y0)) =
      series.iter().find(|(x, y)| x.is_finite() && y.is_finite())
    else {
      continue;
    };
    if let Some(ox) = origin.0 {
      series.push((ox, y0));
    }
    if let Some(oy) = origin.1 {
      series.push((x0, oy));
    }
  }
}

/// The axis of a list plot an automatic range is computed for: the two
/// differ in how far from 0 the data may lie and still have the range
/// reach out to it, and in their padding.
#[derive(Clone, Copy, PartialEq)]
enum ListAxis {
  X,
  Y,
}

/// The automatic plot range (before padding) of one axis of a WL 15 list
/// plot whose data spans `min..max`, as `Options[ListPlot[…], PlotRange]`
/// reports it. Data close enough to 0 — its nearer end at most 1/6 (x) or
/// 1/2 (y) of the farther one — is extended to 0; positive data farther
/// away gets its lower end lowered by 1/48 (x) or 1/18 (y) of its span,
/// which is where the axis then crosses; negative data keeps its range.
/// `from_zero` (the positions of a plain value list) always starts at 0. A
/// single value `v` spans `0..2 v` (`-1..1` for 0). `whole` (`PlotRange ->
/// All` on the x axis) reaches out to 0 neither way:
/// `ListPlot[{1, …, 10}, PlotRange -> All]` runs from `1 - 9/48`.
fn list_axis_extent(
  min: f64,
  max: f64,
  axis: ListAxis,
  from_zero: bool,
  no_zero: bool,
  whole: bool,
) -> (f64, f64) {
  if !(min.is_finite() && max.is_finite()) {
    return (0.0, 1.0);
  }
  // A scaled axis without a place for 0 (log, reciprocal) is lowered by
  // its padding twice instead: by `Scaled[0.05]` (x: `0.02`) of the data
  // span, then by `Scaled[0.02]` of the result under the
  // `{0.02, 0.05}` padding it then gets.
  if no_zero {
    if min == max {
      return (min - 1.0, max);
    }
    let (first, second) = match axis {
      ListAxis::X => (1.0 / 48.0, 1.0 / 48.0),
      ListAxis::Y => (1.0 / 18.0, 2.0 / 93.0),
    };
    let lo = min - (max - min) * first;
    return (lo - (max - lo) * second, max);
  }
  let from_zero = from_zero && !whole;
  let (min, max) = if from_zero {
    (min.min(0.0), max.max(0.0))
  } else {
    (min, max)
  };
  if min == max {
    return match min {
      0.0 => (-1.0, 1.0),
      v if v > 0.0 => (0.0, 2.0 * v),
      v => (2.0 * v, 0.0),
    };
  }
  let (ratio, reach) = match axis {
    ListAxis::X if whole => (0.0, 1.0 / 48.0),
    ListAxis::X => (1.0 / 6.0, 1.0 / 48.0),
    ListAxis::Y => (0.5, 1.0 / 18.0),
  };
  if min > 0.0 {
    if min <= ratio * max {
      (0.0, max)
    } else {
      (min - (max - min) * reach, max)
    }
  } else if max < 0.0 && -max <= ratio * -min {
    (min, 0.0)
  } else {
    (min, max)
  }
}

/// Pad an automatic list-plot range by its `PlotRangePadding`: `Scaled[0.02]`
/// on both ends of the x axis; on the y axis `Scaled[0.02]` at an end that is
/// 0 and `Scaled[0.05]` elsewhere. `Scaled[s]` is that fraction of the
/// *padded* span, so `{0, 3}` pads to `{-1/16, 3 + 1/16}`.
fn pad_list_axis(
  (lo, hi): (f64, f64),
  axis: ListAxis,
  no_zero: bool,
) -> (f64, f64) {
  let span = hi - lo;
  let side = |v: f64| {
    if axis == ListAxis::X || v == 0.0 || (no_zero && v == lo) {
      0.02
    } else {
      0.05
    }
  };
  let (s_lo, s_hi) = (side(lo), side(hi));
  let total = span / (1.0 - s_lo - s_hi);
  (lo - total * s_lo, hi + total * s_hi)
}

/// The automatic, padded x/y ranges of a linear `ListPlot` /
/// `ListLinePlot` of the points `all_series`, following the WL 15 rules of
/// [`list_axis_extent`]. Also records the unpadded lower ends as the place
/// the automatic axes cross (`AxesOrigin`) for the axes without an explicit
/// `PlotRange`.
fn list_plot_ranges(
  all_series: &[Vec<(f64, f64)>],
  implicit_x: bool,
  request: RangeRequest,
  scaling: &(Option<Scaling>, Option<Scaling>),
  opts: &mut PlotOptions,
) -> ((f64, f64), (f64, f64)) {
  let (explicit_x, explicit_y) = request.explicit;
  // The points' extent, outliers clipped off an axis whose range is
  // automatic (WL does not on a logarithmic one).
  // A numeric `AxesOrigin` is part of the range, but not of the
  // distribution the outliers are told apart by.
  let bounds = |pick: fn(&(f64, f64)) -> f64, keep_outliers: bool, origin| {
    let values = all_series
      .iter()
      .flatten()
      .filter(|p| p.0.is_finite() && p.1.is_finite())
      .map(pick);
    let (lo, hi) =
      crate::functions::plot::list_value_extent(values, keep_outliers)
        .unwrap_or((f64::INFINITY, f64::NEG_INFINITY));
    match origin {
      Some(o) if lo <= hi => (lo.min(o), hi.max(o)),
      _ => (lo, hi),
    }
  };
  // A scaled axis without a place for 0 (log, reciprocal) keeps its
  // outliers, as WL's log list plots do.
  let no_zero =
    |s: &Option<Scaling>| s.as_ref().is_some_and(|s| s.zero().is_none());
  let (nx, ny) = (no_zero(&scaling.0), no_zero(&scaling.1));
  let (x_min, x_max) =
    bounds(|p| p.0, request.keep_outliers.0 || nx, opts.axes_origin.0);
  let (y_min, y_max) =
    bounds(|p| p.1, request.keep_outliers.1 || ny, opts.axes_origin.1);
  // A date axis spans just the dates (WL's DateListPlot and a ListPlot of a
  // TimeSeries).
  let x = if opts.date_axis && x_min < x_max {
    (x_min, x_max)
  } else {
    list_axis_extent(
      x_min,
      x_max,
      ListAxis::X,
      implicit_x && !nx,
      nx,
      request.keep_outliers.0,
    )
  };
  let y = list_axis_extent(y_min, y_max, ListAxis::Y, false, ny, false);
  if !explicit_x {
    opts.axes_origin_low.0 = Some(x.0);
  }
  if !explicit_y {
    opts.axes_origin_low.1 = Some(y.0);
  }
  opts.grid_interior = (!explicit_x && !explicit_y).then_some((x, y));
  let (x, y_padded) = (
    pad_list_axis(x, ListAxis::X, nx),
    pad_list_axis(y, ListAxis::Y, ny),
  );
  // The range does not reach out to a filling's level: `Filling -> Axis`
  // fills to wherever the horizontal axis crosses, which is not 0 when the
  // range leaves 0 out.
  if !explicit_y && opts.axes_origin.1.is_none() && !(y.0 <= 0.0 && 0.0 <= y.1)
  {
    let axis_y = if y.0 > 0.0 { y.0 } else { y_padded.1 };
    let at_axis = |f: &mut Filling| {
      if *f == Filling::Axis {
        *f = Filling::Value(axis_y);
      }
    };
    at_axis(&mut opts.filling);
    for (_, target) in &mut opts.filling_rules {
      if let FillTarget::Level(level) = target {
        at_axis(level);
      }
    }
  }
  if !explicit_y {
    opts.curve_clip_y =
      Some(crate::functions::plot::include_filling_levels(opts, y));
  }
  // Labels placed above or below their datasets get room inside the range.
  let y_padded = if explicit_y {
    y_padded
  } else {
    crate::functions::plot_curve_labels::widen_y_range(
      &opts.curve_label_specs,
      all_series,
      x,
      y_padded,
      crate::functions::plot::estimated_plot_area(opts),
    )
  };
  (x, y_padded)
}

/// Widen the automatic ranges so the point labels fit at their first-choice
/// places, as WL makes room for them. `data` are the labelled points (the
/// labels run parallel to them).
fn widen_for_point_labels(
  opts: &PlotOptions,
  data: &[Vec<(f64, f64)>],
  (x, y): ((f64, f64), (f64, f64)),
) -> ((f64, f64), (f64, f64)) {
  crate::functions::plot_curve_labels::widen_for_boxes(
    &crate::functions::plot::point_label_boxes(opts, data),
    x,
    y,
    crate::functions::plot::estimated_plot_area(opts),
  )
}

/// The automatic size (in printer's points at a 360-point-wide image) of a
/// list plot's dots, as WL's `Charting`adaptivePointSize` picks it from the
/// distinct points of all series: 7 for fewer than 50 points, 6, 5 and 4
/// from 50, 100 and 250 on, 1 above 15000. In between it depends also on
/// how clustered the points are: their data range is cut into 10 × 10
/// bins, and the fullest bin's share over the mean share of the non-empty
/// bins classes the data as even (below 2), clustered (above 10) or in
/// between.
pub(crate) fn point_size_steps(all_series: &[Vec<(f64, f64)>]) -> f64 {
  let mut seen = std::collections::HashSet::new();
  let points: Vec<(f64, f64)> = all_series
    .iter()
    .flatten()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .filter(|(x, y)| seen.insert((x.to_bits(), y.to_bits())))
    .copied()
    .collect();
  let n = points.len();
  match n {
    0..50 => return 7.0,
    50..100 => return 6.0,
    100..250 => return 5.0,
    250..500 => return 4.0,
    15001.. => return 1.0,
    _ => {}
  }
  let span = |sel: fn(&(f64, f64)) -> f64| {
    let (lo, hi) = points
      .iter()
      .map(sel)
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
      });
    if lo == hi {
      (lo - 1.0, hi + 1.0)
    } else {
      (lo, hi)
    }
  };
  let (x1, x2) = span(|p| p.0);
  let (y1, y2) = span(|p| p.1);
  let width = |lo: f64, hi: f64| (hi - lo + 0.0001 * (hi - lo)) / 10.0;
  let (wx, wy) = (width(x1, x2), width(y1, y2));
  let mut bins = [[0usize; 10]; 10];
  for &(x, y) in &points {
    let i = ((x - x1) / wx).floor();
    let j = ((y - y1) / wy).floor();
    if (0.0..10.0).contains(&i) && (0.0..10.0).contains(&j) {
      bins[i as usize][j as usize] += 1;
    }
  }
  let filled: Vec<usize> =
    bins.iter().flatten().copied().filter(|&c| c > 0).collect();
  let max = filled.iter().copied().max().unwrap_or(0) as f64;
  let mean = filled.iter().sum::<usize>() as f64 / filled.len().max(1) as f64;
  let density = max / mean;
  let class = if density < 2.0 {
    0
  } else if density > 10.0 {
    1
  } else {
    2
  };
  let table: [f64; 3] = match n {
    ..1000 => [4.0, 4.0, 3.0],
    1000..2500 => [3.0, 3.0, 2.5],
    2500..5000 => [2.5, 2.5, 2.0],
    5000..10000 => [2.0, 2.0, 1.0],
    _ => [2.0, 1.0, 1.0],
  };
  table[class]
}

/// The automatic `PointSize` WL 15 gives the dots of a `ListPlot`: 0.66 of
/// the adaptive size, but at least 1, over the 360-point image width.
fn list_plot_point_size(all_series: &[Vec<(f64, f64)>]) -> f64 {
  (0.66 * point_size_steps(all_series)).max(1.0) / 360.0
}

/// The automatic `PointSize` of a `ListLinePlot`'s mesh points: 1/360 of
/// the image width per step (`PointSize[7/360]` for 8 points, `1/90` for
/// 300).
fn list_line_plot_point_size(all_series: &[Vec<(f64, f64)>]) -> f64 {
  point_size_steps(all_series) / 360.0
}

/// Compute x/y ranges from data with 4% padding.
fn compute_ranges(all_series: &[Vec<(f64, f64)>]) -> ((f64, f64), (f64, f64)) {
  compute_ranges_scaled(all_series, false, false)
}

/// Compute x/y ranges from data with 4% padding. Log axes are padded
/// multiplicatively in log space (4% of the log range, matching LogPlot),
/// so the padded range stays positive.
fn compute_ranges_scaled(
  all_series: &[Vec<(f64, f64)>],
  log_x: bool,
  log_y: bool,
) -> ((f64, f64), (f64, f64)) {
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
    (x_min, x_max) = if log_x { (1.0, 10.0) } else { (0.0, 1.0) };
  }
  if !y_min.is_finite() || !y_max.is_finite() {
    (y_min, y_max) = if log_y { (1.0, 10.0) } else { (0.0, 1.0) };
  }

  let pad_linear = |min: f64, max: f64| {
    let range = max - min;
    let pad = if range.abs() < f64::EPSILON {
      1.0
    } else {
      range * 0.04
    };
    (min - pad, max + pad)
  };
  let pad_log = |min: f64, max: f64| {
    let log_range = (max / min).ln();
    let factor = if log_range.abs() < f64::EPSILON {
      std::f64::consts::E
    } else {
      (log_range * 0.04).exp()
    };
    (min / factor, max * factor)
  };

  (
    if log_x {
      pad_log(x_min, x_max)
    } else {
      pad_linear(x_min, x_max)
    },
    if log_y {
      pad_log(y_min, y_max)
    } else {
      pad_linear(y_min, y_max)
    },
  )
}

/// Render one panel per series for `PlotLayout -> "Row" / "Column" / …`
/// (see [`crate::functions::plot_layout`]). Every panel shows the plot range
/// of all series together, and its dataset label (from `Labeled`) if one
/// was given. `range_series` supplies the per-series points used for range
/// computation (the error-bar extremes when a series carries `Around`
/// uncertainties; otherwise `all_series`).
fn render_panel_layout(
  all_series: &[Vec<(f64, f64)>],
  range_series: &[Vec<(f64, f64)>],
  labels: &[Option<String>],
  parsed: &ParsedOptions,
  scatter: bool,
  implicit_x: bool,
) -> Result<Expr, InterpreterError> {
  let Some(layout) = parsed.opts.plot_layout else {
    return Err(InterpreterError::EvaluationError(
      "ListPlot: no PlotLayout to render".into(),
    ));
  };
  let mut base = parsed.opts.clone();
  let (x_range, y_range) = if base.log_x || base.log_y {
    let mut range_series = range_series.to_vec();
    include_axes_origin(&mut range_series, base.axes_origin);
    let (x_range, y_range) =
      compute_ranges_scaled(&range_series, base.log_x, base.log_y);
    (x_range, adjust_y_range_for_filling_opts(&base, y_range))
  } else {
    list_plot_ranges(
      range_series,
      implicit_x,
      parsed.explicit_range(),
      &(None, None),
      &mut base,
    )
  };
  let (x_range, y_range) = apply_plot_range_override(parsed, x_range, y_range);
  let svg = crate::functions::plot_layout::render_plot_layout(
    layout,
    all_series.len(),
    &base,
    x_range,
    y_range,
    |idx, mut opts| {
      opts.curve_label_specs = crate::functions::plot_curve_labels::panel_specs(
        &parsed.opts.curve_label_specs,
        idx,
      );
      // A panel shows its series' label (`PlotLabels` or a `Labeled`
      // dataset) as a title centred inside its top edge, its baseline
      // 11.6 pt below the frame as in wolframscript, rather than at the
      // curve's end.
      if !opts.plot_labels.is_empty() {
        let label = parsed
          .opts
          .plot_labels
          .get(idx)
          .cloned()
          .flatten()
          .or_else(|| labels.get(idx).cloned().flatten());
        opts.plot_labels = Vec::new();
        opts.plot_label_leaders = Vec::new();
        if let Some(label) = label {
          opts.epilog.push(call(
            "Text",
            vec![
              call(
                "Style",
                vec![Expr::String(label), Expr::Integer(10.into())],
              ),
              call(
                "Offset",
                vec![
                  Expr::List(
                    vec![Expr::Integer(0.into()), Expr::Integer((-9).into())]
                      .into(),
                  ),
                  call1(
                    "Scaled",
                    Expr::List(
                      vec![Expr::Real(0.5), Expr::Integer(1.into())].into(),
                    ),
                  ),
                ],
              ),
            ],
          ));
        }
      }
      // Keep only this panel's error bars and data-point anchors (they are
      // indexed by series).
      opts.error_bars = match parsed.opts.error_bars.get(idx) {
        Some(bars) => vec![bars.clone()],
        None => Vec::new(),
      };
      opts.data_points = match parsed.opts.data_points.get(idx) {
        Some(data) => vec![data.clone()],
        None => Vec::new(),
      };
      // A per-series `Joined -> {..., False, ...}` still picks the panel's
      // renderer, cycling a shorter flags list the same way every other
      // reader of `joined_per_series` does (see `resolve_series_joined`).
      // The panel holds its series at index 0, so its own flags list
      // shrinks to that one resolved state.
      let scatter = if let Some(flags) = &parsed.opts.joined_per_series {
        let joined =
          crate::functions::plot::resolve_series_joined(Some(flags), idx);
        opts.joined_per_series = Some(vec![joined]);
        !joined
      } else {
        scatter
      };
      let single = std::slice::from_ref(&all_series[idx]);
      if scatter {
        generate_scatter_svg_with_options(single, x_range, y_range, &opts)
      } else {
        generate_svg_with_filling(single, x_range, y_range, &opts)
      }
    },
  )?;
  Ok(crate::graphics_result(svg))
}

/// `ListPlot[{a, Legended[b, label], …}]`: a dataset wrapped in `Legended`
/// gets a legend entry (`Placed[label, pos]` places the legend), unless
/// `PlotLegends` already gave one.
/// `Style[data, dirs…]` around the whole data styles every dataset, and
/// around one dataset of a list (`{Style[{1, 2, 3}, Green], {4, 5, 6}}`)
/// that dataset: its directives are laid over the dataset's `PlotStyle`
/// (or default) style, the way WL lets the wrapper win.
fn apply_styled_datasets(data: &Expr, opts: &mut PlotOptions) {
  let style_of = |args: &[Expr], base: SeriesStyle| -> SeriesStyle {
    let mut style = base;
    for d in &args[1..] {
      crate::functions::plot::apply_style_directive(d, &mut style);
    }
    style
  };
  match data {
    Expr::FunctionCall { name, args } if name == "Style" && args.len() >= 2 => {
      let base = opts.plot_style.first().cloned().unwrap_or_default();
      let style = style_of(args, base);
      if opts.plot_style.is_empty() {
        opts.plot_style = vec![style];
      } else {
        for s in &mut opts.plot_style {
          *s = style_of(args, s.clone());
        }
      }
      // A dataset's own `Style` inside still wins over the outer one.
      apply_styled_datasets(&args[0], opts);
    }
    Expr::List(items) => {
      let is_styled_dataset = |e: &Expr| {
        matches!(e, Expr::FunctionCall { name, args }
          if name == "Style" && args.len() >= 2
            && matches!(&args[0], Expr::List(ds)
              // A dataset, not a styled `{x, y}` point.
              if ds.len() != 2 || ds.iter().any(|v| matches!(v, Expr::List(_)))))
      };
      if !items.iter().any(is_styled_dataset) {
        return;
      }
      let base = opts.plot_style.clone();
      opts.plot_style = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
          let mut style = if base.is_empty() {
            SeriesStyle::default()
          } else {
            base[i % base.len()].clone()
          };
          // An unstyled entry keeps the default palette colour of its slot.
          if style.color.is_none() {
            let (r, g, b) = crate::functions::plot::PLOT_COLORS
              [i % crate::functions::plot::PLOT_COLORS.len()];
            style.color = Some(crate::functions::graphics::Color::new(
              r as f64 / 255.0,
              g as f64 / 255.0,
              b as f64 / 255.0,
            ));
          }
          style.from_list = true;
          match item {
            Expr::FunctionCall { args, .. } if is_styled_dataset(item) => {
              style_of(args, style)
            }
            _ => style,
          }
        })
        .collect();
    }
    _ => {}
  }
}

/// The points of a list plot's data written `Style[point, dirs…]` (`{1,
/// Style[2, Red], 3}`, `Style[{x, y}, Hue[t]]`): per dataset, the parsed
/// coordinates of each such point and the colour it is drawn in. The
/// innermost `Style` wins, so a point's own colour beats its dataset's.
fn styled_point_colors(
  data: &Expr,
  series: &[Vec<(f64, f64)>],
) -> Vec<Vec<((f64, f64), (u8, u8, u8))>> {
  fn unstyle(e: &Expr) -> &Expr {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Style" && !args.is_empty() =>
      {
        unstyle(&args[0])
      }
      _ => e,
    }
  }
  let style_color = |e: &Expr| -> Option<(u8, u8, u8)> {
    let Expr::FunctionCall { name, args } = e else {
      return None;
    };
    if name != "Style" || args.len() < 2 {
      return None;
    }
    let mut style = SeriesStyle::default();
    for d in &args[1..] {
      crate::functions::plot::apply_style_directive(d, &mut style);
    }
    let c = style.color?;
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Some((ch(c.r), ch(c.g), ch(c.b)))
  };
  let is_dataset = |e: &Expr| {
    matches!(unstyle(e), Expr::List(items)
      if items.len() != 2 || items.iter().any(|v| matches!(v, Expr::List(_))))
  };
  let Expr::List(items) = unstyle(data) else {
    return Vec::new();
  };
  // Several datasets, or one dataset of points.
  let datasets: Vec<&Expr> = if series.len() > 1
    && series.len() == items.len()
    && items.iter().all(is_dataset)
  {
    items.iter().collect()
  } else {
    vec![unstyle(data)]
  };
  let mut out = Vec::new();
  let mut any = false;
  for (i, ds) in datasets.iter().enumerate() {
    let mut colors = Vec::new();
    if let (Expr::List(points), Some(parsed)) = (unstyle(ds), series.get(i))
      && points.len() == parsed.len()
    {
      for (point, &xy) in points.iter().zip(parsed) {
        if let Some(rgb) = style_color(point) {
          colors.push((xy, rgb));
          any = true;
        }
      }
    }
    out.push(colors);
  }
  if any { out } else { Vec::new() }
}

fn apply_legended_datasets(data: &Expr, opts: &mut PlotOptions) {
  let Expr::List(items) = data else {
    return;
  };
  if !opts.plot_legends.is_empty()
    || !items
      .iter()
      .all(|i| matches!(i, Expr::List(_) | Expr::FunctionCall { .. }))
  {
    return;
  }
  let mut labels = Vec::new();
  let mut series = Vec::new();
  for (i, item) in items.iter().enumerate() {
    if let Expr::FunctionCall { name, args } = item
      && name == "Legended"
      && args.len() == 2
    {
      let (mut entry, _, _, position) =
        crate::functions::plot::parse_plot_legends(&match &args[1] {
          Expr::FunctionCall { name, args: p }
            if name == "Placed" && p.len() == 2 =>
          {
            Expr::FunctionCall {
              name: "Placed".into(),
              args: vec![Expr::List(vec![p[0].clone()].into()), p[1].clone()]
                .into(),
            }
          }
          label => Expr::List(vec![label.clone()].into()),
        });
      if let Some(label) = entry.pop() {
        labels.push(label);
        series.push(i);
        opts.legend_position = position;
      }
    }
  }
  if !labels.is_empty() {
    opts.plot_legends = labels;
    opts.legend_series = series;
  }
}

/// The position and kind of a dataset's own label: `Labeled[data, l,
/// pos]` / `Callout[data, l, pos]` (`pos` absent at the automatic place,
/// the dataset's end) and whether it is a callout.
fn dataset_label_kind(e: &Expr) -> Option<(Option<Expr>, bool)> {
  match e {
    Expr::FunctionCall { name, args }
      if (name == "Labeled" || name == "Callout") && args.len() >= 2 =>
    {
      Some((args.get(2).cloned(), name == "Callout"))
    }
    _ => None,
  }
}

/// Attach the dataset labels of `Labeled` / `Callout` wrappers in the
/// (evaluated) data `data`: those at their automatic place go to the
/// series' end (`plot_labels`), a callout's off a leader line, the others
/// where their position puts them. Explicit `PlotLabels` win.
fn apply_dataset_labels(
  opts: &mut PlotOptions,
  labels: &[Option<String>],
  data: &Expr,
) {
  if !labels.iter().any(std::option::Option::is_some)
    || !opts.plot_labels.is_empty()
    || !opts.curve_label_specs.is_empty()
  {
    return;
  }
  let data = evaluate_expr_to_expr(data).unwrap_or_else(|_| data.clone());
  let kinds: Vec<Option<(Option<Expr>, bool)>> = match &data {
    Expr::List(items) if items.len() == labels.len() => {
      items.iter().map(dataset_label_kind).collect()
    }
    whole => vec![dataset_label_kind(whole); labels.len()],
  };
  let entries: Vec<_> = labels
    .iter()
    .zip(kinds)
    .map(|(label, kind)| {
      let (pos, callout) = kind.unwrap_or((None, false));
      label.clone().map(|l| (l, pos, callout))
    })
    .collect();
  let end_leaders = opts.plot_labels_automatic.then_some(true);
  crate::functions::plot::apply_curve_label_entries(
    opts,
    &entries,
    end_leaders,
  );
}

/// The point labels drawn: none with `LabelingFunction -> None`, those a
/// `LabelingFunction -> f` makes (as callouts), and none wider than a
/// `LabelingSize`.
fn resolve_point_labels(
  parsed: &mut ParsedOptions,
  point_labels: Vec<Vec<Option<PointLabel>>>,
  all_series: &[Vec<(f64, f64)>],
) -> Vec<Vec<Option<String>>> {
  // A finite LabelingSize produces FEPrivate`TruncateStringToWidth boxes.
  // Wolfram's static exporter leaves those front-end boxes unpainted.
  if parsed.hide_point_labels
    || parsed.labeling_size.is_some_and(f64::is_finite)
  {
    return Vec::new();
  }
  // `Labeled[pt, label, pos]` sides, parallel to the labels.
  parsed.opts.point_label_positions = point_labels
    .iter()
    .map(|series| {
      series
        .iter()
        .map(|l| l.as_ref().and_then(|(_, pos)| pos.clone()))
        .collect()
    })
    .collect();
  let point_labels: Vec<Vec<Option<String>>> = point_labels
    .into_iter()
    .map(|series| {
      series
        .into_iter()
        .map(|l| l.map(|(text, _)| text))
        .collect()
    })
    .collect();
  let mut labels = match &parsed.labeling_function {
    Some(f) => {
      parsed.opts.point_label_callouts = true;
      all_series
        .iter()
        .enumerate()
        .map(|(i, series)| {
          series
            .iter()
            .enumerate()
            .map(|(j, &(x, y))| {
              let call = Expr::CurriedCall {
                func: Box::new(f.clone()),
                args: vec![
                  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()),
                  Expr::List(
                    vec![
                      Expr::Integer(i as i128 + 1),
                      Expr::Integer(j as i128 + 1),
                    ]
                    .into(),
                  ),
                ],
              };
              match &evaluate_expr_to_expr(&call).ok()? {
                Expr::Identifier(n) if n == "None" || n == "Null" => None,
                label => Some(label_string(label)),
              }
            })
            .collect()
        })
        .collect()
    }
    None => point_labels,
  };
  if let Some(max_w) = parsed.labeling_size {
    for label in labels.iter_mut().flatten() {
      if label.as_ref().is_some_and(|l| {
        crate::functions::plot_padding::text_width(l, 10.0) > max_w
      }) {
        *label = None;
      }
    }
  }
  labels
}

/// ListPlot[{y1, y2, ...}] or ListPlot[{{x1,y1}, ...}]
/// Put the data and the coordinate options of a list plot into the scaled
/// coordinates of its `ScalingFunctions`; points outside a scaling's domain
/// are dropped.
fn scale_list_data(
  scaling: &(Option<Scaling>, Option<Scaling>),
  draw_series: &mut [Vec<(f64, f64)>],
  range_series: &mut [Vec<(f64, f64)>],
  parsed: &mut ParsedOptions,
) {
  if scaling.0.is_none() && scaling.1.is_none() {
    return;
  }
  let map = |series: &mut [Vec<(f64, f64)>]| {
    for s in series.iter_mut() {
      *s = s
        .iter()
        .filter_map(|&(x, y)| {
          let x = match &scaling.0 {
            Some(sx) => sx.forward(x)?,
            None => x,
          };
          let y = match &scaling.1 {
            Some(sy) => sy.forward(y)?,
            None => y,
          };
          Some((x, y))
        })
        .collect();
    }
  };
  map(draw_series);
  map(range_series);
  parsed.plot_range_x = crate::functions::plot::scale_range(
    parsed.plot_range_x,
    scaling.0.as_ref(),
  );
  parsed.plot_range_y = crate::functions::plot::scale_range(
    parsed.plot_range_y,
    scaling.1.as_ref(),
  );
  crate::functions::plot::scale_axis_options(&mut parsed.opts, scaling);
}

/// Finish the scaled axes of a list plot (see
/// [`crate::functions::plot::finish_scaled_axis`]). The other axis crosses a
/// log axis where its automatic range starts, inside the padding.
fn finish_list_scaling(
  scaling: &(Option<Scaling>, Option<Scaling>),
  opts: &mut PlotOptions,
  series: &mut [Vec<(f64, f64)>],
  (x_range, y_range): ((f64, f64), (f64, f64)),
) -> ((f64, f64), (f64, f64)) {
  if scaling.0.is_none() && scaling.1.is_none() {
    return (x_range, y_range);
  }
  let low = opts.axes_origin_low;
  let x = crate::functions::plot::finish_scaled_axis(
    0,
    scaling.0.as_ref(),
    opts,
    series,
    x_range,
    true,
  );
  let y = crate::functions::plot::finish_scaled_axis(
    1,
    scaling.1.as_ref(),
    opts,
    series,
    y_range,
    true,
  );
  if opts.log_x {
    opts.axes_origin_low.0 = low.0.map(f64::exp);
  }
  if opts.log_y {
    opts.axes_origin_low.1 = low.1.map(f64::exp);
  }
  if opts.log_x || opts.log_y {
    opts.grid_interior = None;
  }
  (x, y)
}

pub fn list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut parsed = parse_plot_options(args);
  let mut data = evaluate_expr_to_expr(&data_arg(args, &parsed))?;
  if let Some(unit) = option_value(args, "TargetUnits") {
    data = convert_to_target_units(&data, unit);
  }
  if matches!(option_value(args, "AxesLabel"), Some(Expr::Identifier(a)) if a == "Automatic")
  {
    parsed.opts.axes_label = automatic_unit_axes_label(&data);
  }
  if has_date_stamps(&data) {
    parsed.opts.date_axis = true;
  }
  apply_legended_datasets(&data, &mut parsed.opts);
  apply_styled_datasets(&data, &mut parsed.opts);
  parsed.opts.point_label_callouts = labels_are_callouts(&data);
  let (mut err_series, labels, mut point_labels) =
    parse_list_data_err_labeled(&data, true)?;
  apply_stacked_layout(args, &mut err_series);
  apply_multiaxis(args, &mut err_series, &mut parsed.opts);
  let source_point_size = list_plot_point_size(&strip_errors(&err_series));
  let mut point_colors = styled_point_colors(&data, &strip_errors(&err_series));
  sample_plot_data(args, &mut err_series, &mut point_labels);
  let all_series = strip_errors(&err_series);
  for (colors, points) in point_colors.iter_mut().zip(&all_series) {
    colors.retain(|(xy, _)| points.contains(xy));
  }
  // `DataRange -> All` puts the values at their positions 1, 2, …, an x
  // axis that starts at 0 like a plain value list's.
  let implicit_x = all_implicit_x(&err_series)
    || matches!(parsed.data_range, Some(DataRangeSpec::All));
  apply_dataset_labels(&mut parsed.opts, &labels, &data);
  parsed.opts.point_labels =
    resolve_point_labels(&mut parsed, point_labels, &all_series);
  parsed.opts.error_bars = collect_error_bars(&err_series);
  if parsed.opts.interval_markers == IntervalMarkers::Fences
    && interval_only_data(&data)
  {
    parsed.opts.interval_markers = IntervalMarkers::Bars;
  }
  parsed.opts.point_colors = point_colors;

  // InterpolationOrder reshapes only a joined series into a curve (0 =
  // steps, >= 2 = smooth spline); a series marked `Joined -> False` keeps
  // its raw data points, since it draws discrete circles rather than a
  // curve. Mesh dots, error bars, and point labels stay anchored to the
  // original data points.
  let interpolating = matches!(parsed.interpolation_order, Some(o) if o != 1);
  let series_joined = |i: usize| match &parsed.opts.joined_per_series {
    Some(flags) => {
      crate::functions::plot::resolve_series_joined(Some(flags), i)
    }
    None => parsed.joined,
  };
  let curve_transformed =
    interpolating && (0..all_series.len()).any(series_joined);
  let draw_series = if curve_transformed {
    parsed.opts.data_points.clone_from(&all_series);
    let curves =
      interpolate_series(&all_series, parsed.interpolation_order.unwrap());
    all_series
      .iter()
      .zip(curves)
      .enumerate()
      .map(
        |(i, (raw, curve))| {
          if series_joined(i) { curve } else { raw.clone() }
        },
      )
      .collect()
  } else {
    all_series
  };

  // The plot range must cover the full extent of any error bars, plus any
  // overshoot of an interpolated curve beyond the data (kept parallel to
  // the series so the panel layout can index it per panel).
  let mut range_series = error_extremes(&err_series);
  if curve_transformed {
    for (rs, cs) in range_series.iter_mut().zip(&draw_series) {
      rs.extend(cs.iter().copied());
    }
  }

  // The data decide the dot size, not an interpolated curve through them.
  parsed.opts.default_point_size = Some(source_point_size);
  if parsed.opts.plot_layout.is_some() && draw_series.len() > 1 {
    return render_panel_layout(
      &draw_series,
      &range_series,
      &labels,
      &parsed,
      !parsed.joined,
      implicit_x,
    );
  }
  let scaling = parsed.scaling.clone().unwrap_or((None, None));
  let mut draw_series = draw_series;
  scale_list_data(&scaling, &mut draw_series, &mut range_series, &mut parsed);
  let (x_range, y_range) = list_plot_ranges(
    &range_series,
    implicit_x,
    parsed.explicit_range(),
    &scaling,
    &mut parsed.opts,
  );
  let (x_range, y_range) =
    widen_for_point_labels(&parsed.opts, &draw_series, (x_range, y_range));
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let (x_range, y_range) = finish_list_scaling(
    &scaling,
    &mut parsed.opts,
    &mut draw_series,
    (x_range, y_range),
  );
  let joined = parsed.joined;
  let mut parsed = parsed;
  if draw_series.len() == 1 {
    parsed.opts.plot_style =
      crate::functions::plot::collapse_style_for_single_series(
        &parsed.opts.plot_style,
      );
  }
  let opts = &parsed.opts;

  let svg = if joined {
    generate_svg_with_filling(&draw_series, x_range, y_range, opts)?
  } else {
    generate_scatter_svg_with_options(&draw_series, x_range, y_range, opts)?
  };

  // Resolved per series rather than copied from `opts.joined_per_series`
  // directly: a flags list shorter than `draw_series` must cycle exactly
  // like `generate_svg_with_options` cycles it when actually drawing, or
  // `PlotSource.is_scatter` (which `Show` merges and interactive
  // re-renders read) would disagree with the SVG this same call produced.
  let is_scatter: Vec<bool> = if opts.joined_per_series.is_some() {
    (0..draw_series.len())
      .map(|i| {
        !crate::functions::plot::resolve_series_joined(
          opts.joined_per_series.as_deref(),
          i,
        )
      })
      .collect()
  } else {
    vec![!joined]
  };
  let mut source = build_plot_source(
    &draw_series,
    &opts.plot_style,
    x_range,
    y_range,
    (opts.svg_width, opts.svg_height),
    &is_scatter,
    opts.filling,
    opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  source.plot_range = opts.grid_interior;
  source.log_x = opts.log_x;
  source.log_y = opts.log_y;
  crate::functions::plot::apply_markers_to_source(
    &mut source,
    &opts.plot_markers,
  );
  for series in &mut source.series {
    series.point_size = series.point_size.or(opts.default_point_size);
  }
  for (series, labels) in source.series.iter_mut().zip(&opts.point_labels) {
    if labels.len() == series.points.len() {
      series.labels.clone_from(labels);
    }
  }
  Ok(crate::graphics_result_with_source(svg, source))
}

/// Replace every number of `ComplexListPlot` data by its `{Re[z], Im[z]}`
/// point, looking through the data forms `ListPlot` accepts: lists,
/// association values, the data side of `data -> labels`, `SparseArray`s and
/// the content of display wrappers (`Labeled`, `Callout`, `Tooltip`, …).
/// Anything that is no number is kept as is, for `ListPlot` to judge.
fn complex_data_to_points(e: &Expr) -> Expr {
  let e = evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
  if let Some((re, im)) =
    crate::functions::list_helpers_ast::expr_to_complex_parts(&e)
  {
    return Expr::List(vec![Expr::Real(re), Expr::Real(im)].into());
  }
  // An exact numeric value (`Exp[2 Pi I / 7]`, `(1 + I)^3 / 4`) is plotted
  // at its numerical value.
  if !matches!(e, Expr::List(_) | Expr::Association(_) | Expr::Rule { .. })
    && let Ok(numeric) = evaluate_expr_to_expr(&Expr::FunctionCall {
      name: "N".into(),
      args: vec![e.clone()].into(),
    })
    && let Some((re, im)) =
      crate::functions::list_helpers_ast::expr_to_complex_parts(&numeric)
  {
    return Expr::List(vec![Expr::Real(re), Expr::Real(im)].into());
  }
  match &e {
    Expr::List(items) => {
      Expr::List(items.iter().map(complex_data_to_points).collect())
    }
    Expr::Association(pairs) => Expr::Association(
      pairs
        .iter()
        .map(|(k, v)| (k.clone(), complex_data_to_points(v)))
        .collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(complex_data_to_points(pattern)),
      replacement: replacement.clone(),
    },
    Expr::FunctionCall { name, .. } if name == "SparseArray" => {
      let normal = evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "Normal".into(),
        args: vec![e.clone()].into(),
      })
      .unwrap_or_else(|_| e.clone());
      if matches!(normal, Expr::List(_)) {
        complex_data_to_points(&normal)
      } else {
        e.clone()
      }
    }
    Expr::FunctionCall { name, args }
      if !args.is_empty()
        && matches!(
          name.as_str(),
          "Labeled"
            | "Callout"
            | "Tooltip"
            | "Style"
            | "Legended"
            | "Annotation"
        ) =>
    {
      let mut args = args.to_vec();
      args[0] = complex_data_to_points(&args[0]);
      Expr::FunctionCall {
        name: name.clone(),
        args: args.into(),
      }
    }
    _ => e,
  }
}

/// ComplexListPlot[{z1, z2, ...}] plots complex numbers as points at
/// (Re[z], Im[z]) in the complex plane: it is `ListPlot` of those points,
/// so every data form and option of `ListPlot` carries over.
pub fn complex_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut new_args = args.to_vec();
  new_args[0] = complex_data_to_points(&args[0]);
  // The complex plane is drawn to scale: ComplexListPlot's AspectRatio
  // defaults to Automatic, not ListPlot's 1/GoldenRatio.
  let has_aspect = args[1..].iter().any(|a| {
    matches!(a, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "AspectRatio"))
  });
  if !has_aspect {
    new_args.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("AspectRatio".into())),
      replacement: Box::new(Expr::Identifier("Automatic".into())),
    });
  }
  list_plot_ast(&new_args)
}

/// ListLinePlot[{y1, y2, ...}]
pub fn list_line_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut parsed = parse_plot_options(args);
  // Association keys don't label the points here: wolframscript draws key
  // labels for ListPlot but not for ListLinePlot.
  let data = evaluate_expr_to_expr(&data_arg(args, &parsed))?;
  if has_date_stamps(&data) {
    parsed.opts.date_axis = true;
  }
  apply_legended_datasets(&data, &mut parsed.opts);
  apply_styled_datasets(&data, &mut parsed.opts);
  let (mut err_series, labels, mut point_labels) =
    parse_list_data_err_labeled(&data, false)?;
  apply_stacked_layout(args, &mut err_series);
  parsed.opts.mesh_point_size =
    Some(list_line_plot_point_size(&strip_errors(&err_series)));
  sample_plot_data(args, &mut err_series, &mut point_labels);
  let all_series = strip_errors(&err_series);
  // `DataRange -> All` puts the values at their positions 1, 2, …, an x
  // axis that starts at 0 like a plain value list's.
  let implicit_x = all_implicit_x(&err_series)
    || matches!(parsed.data_range, Some(DataRangeSpec::All));
  apply_dataset_labels(&mut parsed.opts, &labels, &data);
  parsed.opts.point_labels =
    resolve_point_labels(&mut parsed, point_labels, &all_series);
  parsed.opts.error_bars = collect_error_bars(&err_series);
  if parsed.opts.interval_markers == IntervalMarkers::Fences
    && interval_only_data(&data)
  {
    parsed.opts.interval_markers = IntervalMarkers::Bars;
  }

  // ListLinePlot always joins, so InterpolationOrder always reshapes the
  // curve (0 = steps, >= 2 = smooth spline).
  let curve_transformed =
    matches!(parsed.interpolation_order, Some(o) if o != 1);
  let draw_series = if curve_transformed {
    parsed.opts.data_points.clone_from(&all_series);
    interpolate_series(&all_series, parsed.interpolation_order.unwrap())
  } else {
    all_series
  };
  let mut range_series = error_extremes(&err_series);
  if curve_transformed {
    for (rs, cs) in range_series.iter_mut().zip(&draw_series) {
      rs.extend(cs.iter().copied());
    }
  }

  if parsed.opts.plot_layout.is_some() && draw_series.len() > 1 {
    return render_panel_layout(
      &draw_series,
      &range_series,
      &labels,
      &parsed,
      false,
      implicit_x,
    );
  }
  let scaling = parsed.scaling.clone().unwrap_or((None, None));
  let mut draw_series = draw_series;
  scale_list_data(&scaling, &mut draw_series, &mut range_series, &mut parsed);
  let (x_range, y_range) = list_plot_ranges(
    &range_series,
    implicit_x,
    parsed.explicit_range(),
    &scaling,
    &mut parsed.opts,
  );
  let (x_range, y_range) =
    widen_for_point_labels(&parsed.opts, &draw_series, (x_range, y_range));
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let (x_range, y_range) = finish_list_scaling(
    &scaling,
    &mut parsed.opts,
    &mut draw_series,
    (x_range, y_range),
  );

  // A single curve takes a `PlotStyle` list as one combined style, as
  // `ListPlot` and `Plot` do: `{Thick, Orange}` is one thick orange line.
  if draw_series.len() == 1 {
    parsed.opts.plot_style =
      crate::functions::plot::collapse_style_for_single_series(
        &parsed.opts.plot_style,
      );
  }

  let svg =
    generate_svg_with_filling(&draw_series, x_range, y_range, &parsed.opts)?;
  // Carry the series so `Show` can merge this plot with others instead of
  // treating it as an opaque rendering.
  let mut source = build_plot_source(
    &draw_series,
    &parsed.opts.plot_style,
    x_range,
    y_range,
    (parsed.opts.svg_width, parsed.opts.svg_height),
    &[false],
    parsed.opts.filling,
    parsed.opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  source.plot_range = parsed.opts.grid_interior;
  source.log_x = parsed.opts.log_x;
  source.log_y = parsed.opts.log_y;
  crate::functions::plot::apply_markers_to_source(
    &mut source,
    &parsed.opts.plot_markers,
  );
  Ok(crate::graphics_result_with_source(svg, source))
}

/// StackedListPlot[{list1, list2, ...}]: plot several datasets with their
/// y-values accumulated, so each successive dataset is stacked on top of the
/// running total of the preceding ones. The regions between consecutive
/// cumulative curves are filled by default.
pub fn stacked_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut parsed = parse_plot_options(args);
  let data = data_arg(args, &parsed);
  let (all_series, labels) = parse_list_data_labeled(&data)?;
  apply_dataset_labels(&mut parsed.opts, &labels, &data);
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

  // The automatic range of a WL 15 list plot, over the stacked curves.
  let (x_range, y_range) = list_plot_ranges(
    &cumulative,
    false,
    parsed.explicit_range(),
    &(None, None),
    &mut parsed.opts,
  );
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);

  let svg =
    generate_svg_with_filling(&cumulative, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListStepPlot[{y1, y2, ...}]
pub fn list_step_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut parsed = parse_plot_options(args);
  let data = data_arg(args, &parsed);
  let (all_series, labels) = parse_list_data_labeled(&data)?;
  apply_dataset_labels(&mut parsed.opts, &labels, &data);

  // `ListStepPlot[data, Left | Right | Center]` places the steps.
  let position = args
    .get(1)
    .and_then(StepPosition::parse)
    .unwrap_or(StepPosition::Right);
  // Transform each series into staircase coordinates
  let step_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| positioned_step_curve(series, position))
    .collect();
  // Markers and labels stay on the data points, not the step ends.
  parsed.opts.data_points.clone_from(&all_series);

  if parsed.opts.plot_layout.is_some() && step_series.len() > 1 {
    return render_panel_layout(
      &step_series,
      &step_series,
      &labels,
      &parsed,
      false,
      false,
    );
  }
  let (x_range, y_range) = list_plot_ranges(
    &step_series,
    false,
    parsed.explicit_range(),
    &(None, None),
    &mut parsed.opts,
  );
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&step_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// `ListLogPlot`: `ListPlot` with `ScalingFunctions -> "Log"` on
/// its log axes, as in the Wolfram Language.
pub fn list_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scaled_list_plot(
    args,
    (
      Expr::Identifier("None".to_string()),
      Expr::String("Log".to_string()),
    ),
  )
}

/// `ListLogLogPlot`: `ListPlot` with `ScalingFunctions -> "Log"` on
/// its log axes, as in the Wolfram Language.
pub fn list_log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scaled_list_plot(
    args,
    (
      Expr::String("Log".to_string()),
      Expr::String("Log".to_string()),
    ),
  )
}

/// `ListLogLinearPlot`: `ListPlot` with `ScalingFunctions -> "Log"` on
/// its log axes, as in the Wolfram Language.
pub fn list_log_linear_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  log_scaled_list_plot(
    args,
    (
      Expr::String("Log".to_string()),
      Expr::Identifier("None".to_string()),
    ),
  )
}

/// `ListPlot` with the `ScalingFunctions -> {x, y}` of a log list plot,
/// ahead of any `ScalingFunctions` the call gives.
fn log_scaled_list_plot(
  args: &[Expr],
  (x, y): (Expr, Expr),
) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return list_plot_ast(args);
  }
  let mut scaled = vec![args[0].clone()];
  scaled.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("ScalingFunctions".to_string())),
    replacement: Box::new(Expr::List(vec![x, y].into())),
  });
  scaled.extend_from_slice(&args[1..]);
  list_plot_ast(&scaled)
}

/// ListPolarPlot[{r1, r2, ...}]: plot data in polar coordinates
pub fn list_polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  // A polar plot has no series end to the right: its dataset labels stay
  // callouts (an empty data list matches no dataset).
  apply_dataset_labels(
    &mut parsed.opts,
    &labels,
    &Expr::List(Vec::new().into()),
  );

  // `{{θ1, r1}, …}` pairs give each point its angle; plain radii
  // `{r1, r2, …}` are spread evenly over the circle, starting at angle 0,
  // or over the `DataRange`.
  let is_pair = |e: &Expr| matches!(e, Expr::List(p) if p.len() == 2 && !matches!(p[0], Expr::List(_)));
  let pairs_per_series: Vec<bool> = match &args[0] {
    Expr::List(items) if items.first().is_some_and(is_pair) => vec![true],
    Expr::List(items) if items.iter().all(|s| matches!(s, Expr::List(_))) => {
      items
        .iter()
        .map(
          |s| matches!(s, Expr::List(pts) if pts.first().is_some_and(is_pair)),
        )
        .collect()
    }
    _ => vec![false],
  };
  let polar_series: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .enumerate()
    .map(|(s, series)| {
      let n = series.len();
      let pairs = pairs_per_series.get(s).copied().unwrap_or(false);
      series
        .iter()
        .enumerate()
        .map(|(i, &(x, r))| {
          // `DataRange -> {a, b}` runs the angles from `a` to `b`, both
          // ends included.
          let theta = match &parsed.data_range {
            _ if pairs => x,
            Some(DataRangeSpec::Span(a, b)) if n > 1 => {
              a + (b - a) * i as f64 / (n - 1) as f64
            }
            _ => 2.0 * std::f64::consts::PI * i as f64 / n as f64,
          };
          (r * theta.cos(), r * theta.sin())
        })
        .collect()
    })
    .collect();

  // A polar plot keeps one scale on both axes (`AspectRatio -> Automatic`)
  // unless an explicit ratio is given.
  if parsed.opts.aspect_ratio.is_none() {
    parsed.opts.aspect_automatic = true;
  }
  let (x_range, y_range) = compute_ranges(&polar_series);
  let y_range = adjust_y_range_for_filling_opts(&parsed.opts, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  // Like ListPlot, the points stay unconnected unless `Joined -> True`,
  // with the same automatic dot size.
  parsed.opts.default_point_size = Some(list_plot_point_size(&polar_series));
  let svg = if parsed.joined {
    generate_svg_with_filling(&polar_series, x_range, y_range, &parsed.opts)?
  } else {
    generate_scatter_svg_with_options(
      &polar_series,
      x_range,
      y_range,
      &parsed.opts,
    )?
  };
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
  let Expr::List(iter_spec) = &args[1] else {
    return Err(InterpreterError::EvaluationError(
        "DiscretePlot: second argument must be an iteration specification {var, min, max}".into(),
      ));
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

  // A single function or a (possibly nested) list of functions: Wolfram
  // flattens `DiscretePlot[{f, g}, …]` into two separately-colored series,
  // just like `Plot`.
  let mut bodies: Vec<&Expr> = Vec::new();
  crate::functions::plot::flatten_plot_bodies(expr, &mut bodies);

  // Generate data points by evaluating each body at each discrete value.
  let max_points = 10000;
  let mut all_series: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());
  for body in &bodies {
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut n = n_min;
    while n <= n_max + step * 0.5e-10 && points.len() < max_points {
      let n_expr = if n == n.floor() && n.abs() < 1e15 {
        Expr::Integer(n as i128)
      } else {
        Expr::Real(n)
      };
      let substituted = substitute_var(body, &var_name, &n_expr);
      if let Ok(result) = evaluate_expr_to_expr(&substituted)
        && let Some(y) = try_eval_to_f64(&result)
        && y.is_finite()
      {
        points.push((n, y));
      }
      n += step;
    }
    all_series.push(points);
  }

  let option = |key: &str| {
    args[2..].iter().rev().find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      (name == key).then(|| value.into_owned())
    })
  };
  let rule = |name: &str, value: Expr| Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.to_string())),
    replacement: Box::new(value),
  };
  let given = |key: &str| option(key).is_some();
  // The user's options, without the ones DiscretePlot itself resolves.
  let passed_options = |skip: &[&str]| -> Vec<Expr> {
    args[2..]
      .iter()
      .filter(|opt| {
        !crate::functions::graphics::option_name_value(opt)
          .is_some_and(|(name, _)| skip.contains(&name))
      })
      .cloned()
      .collect()
  };
  let point =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  // `Filling -> Axis` is DiscretePlot's default: a stem from every point
  // down to the axis.
  let filling = !matches!(
    option("Filling"),
    Some(Expr::Identifier(ref s)) if s == "None" || s == "False"
  );

  // `ExtentSize -> Right | Left | Full` draws each value as a flat step over
  // that part of its cell, filled down to the axis, instead of a point.
  let extent = match &option("ExtentSize") {
    Some(Expr::Identifier(s)) if s == "Right" => Some((0.0, step)),
    Some(Expr::Identifier(s)) if s == "Left" => Some((-step, 0.0)),
    Some(Expr::Identifier(s)) if s == "Full" => Some((-step / 2.0, step / 2.0)),
    Some(e) => try_eval_to_f64(e)
      .filter(|w| *w > 0.0)
      .map(|w| (-w * step / 2.0, w * step / 2.0)),
    None => None,
  };
  if let Some((left, right)) = extent {
    let palette = crate::functions::plot::PLOT_COLORS_RGB;
    let mut items: Vec<Expr> = Vec::new();
    let (mut y_lo, mut y_hi) = (0.0_f64, 0.0_f64);
    for (k, points) in all_series.iter().enumerate() {
      let (r, g, b) = palette[k % palette.len()];
      let color = Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
      };
      let style = |extra: Vec<Expr>| {
        let mut directives = extra;
        directives.push(color.clone());
        directives.push(Expr::FunctionCall {
          name: "AbsoluteThickness".to_string(),
          args: vec![Expr::Integer(2)].into(),
        });
        Expr::FunctionCall {
          name: "Directive".to_string(),
          args: directives.into(),
        }
      };
      if filling {
        let mut group = vec![style(vec![Expr::FunctionCall {
          name: "Opacity".to_string(),
          args: vec![Expr::Real(0.2)].into(),
        }])];
        group.extend(points.iter().map(|&(n, y)| Expr::FunctionCall {
          name: "Rectangle".to_string(),
          args: vec![point(n + left, 0.0), point(n + right, y)].into(),
        }));
        items.push(Expr::List(group.into()));
      }
      let steps: Vec<Expr> = points
        .iter()
        .map(|&(n, y)| {
          y_lo = y_lo.min(y);
          y_hi = y_hi.max(y);
          Expr::List(vec![point(n + left, y), point(n + right, y)].into())
        })
        .collect();
      items.push(Expr::List(
        vec![
          style(Vec::new()),
          Expr::FunctionCall {
            name: "Line".to_string(),
            args: vec![Expr::List(steps.into())].into(),
          },
        ]
        .into(),
      ));
    }
    let scaled = |f: f64| Expr::FunctionCall {
      name: "Scaled".to_string(),
      args: vec![Expr::Real(f)].into(),
    };
    let pair = |a: Expr, b: Expr| Expr::List(vec![a, b].into());
    let mut graphics_args = vec![Expr::List(items.into())];
    graphics_args.extend(passed_options(&[
      "ExtentSize",
      "Filling",
      "PlotLegends",
      "PlotStyle",
    ]));
    let defaults = [
      ("Axes", Expr::Identifier("True".to_string())),
      (
        "AspectRatio",
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::Identifier("GoldenRatio".to_string()),
            Expr::Integer(-1),
          ]
          .into(),
        },
      ),
      ("AxesOrigin", point(0.0, 0.0)),
      (
        "PlotRange",
        pair(point(n_min + left, n_max + right), point(y_lo, y_hi)),
      ),
      (
        "PlotRangePadding",
        pair(
          pair(scaled(0.02), scaled(0.02)),
          pair(scaled(0.05), scaled(0.05)),
        ),
      ),
    ];
    for (name, value) in defaults {
      if !given(name) {
        graphics_args.push(rule(name, value));
      }
    }
    return crate::functions::graphics::graphics_ast(&graphics_args);
  }

  // Otherwise the plot is `ListPlot` of the sampled points, with the stems
  // of `Filling -> Axis` unless the caller chose another filling.
  let data: Vec<Expr> = all_series
    .iter()
    .map(|points| {
      Expr::List(points.iter().map(|&(n, y)| point(n, y)).collect())
    })
    .collect();
  let data = if data.len() == 1 {
    data.into_iter().next().unwrap()
  } else {
    Expr::List(data.into())
  };
  let mut list_args = vec![data];
  list_args.extend(passed_options(&["PlotLegends"]));
  if !given("Filling") {
    list_args.push(rule("Filling", Expr::Identifier("Axis".to_string())));
  }
  // `PlotLegends -> "Expressions"` labels each series with its function,
  // typeset in TraditionalForm as Plot's are.
  match &option("PlotLegends") {
    Some(Expr::String(s)) if s == "Expressions" => {
      let labels: Vec<Expr> = bodies
        .iter()
        .map(|b| Expr::FunctionCall {
          name: "TraditionalForm".to_string(),
          args: vec![(*b).clone()].into(),
        })
        .collect();
      list_args.push(rule("PlotLegends", Expr::List(labels.into())));
    }
    Some(other) => list_args.push(rule("PlotLegends", other.clone())),
    None => {}
  }
  list_plot_ast(&list_args)
}

/// Cap on rendered points per channel. Longer audio is reduced to a min/max
/// envelope per bucket so the SVG stays small while the waveform shape is
/// preserved (wolframscript likewise draws an envelope for long audio).
const AUDIO_MAX_POINTS: usize = 2000;

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
    for item in items {
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
    return Ok(unevaluated("AudioPlot", args));
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
  let y_range = adjust_y_range_for_filling_opts(&parsed.opts, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&all_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListFitPlot[data, model, opts] — the data points together with the curve
/// of the model fitted to them. Delegates to `Fit`, `ListPlot` and `Plot`.
///
/// Supported models: "Linear" (default), "Quadratic", "Cubic", "Quartic",
/// "Quintic", "Exponential", "Kernel" (Gaussian kernel smoothing), or an
/// explicit list of basis functions in `x`.
pub fn list_fit_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let is_option = |e: &Expr| {
    matches!(e, Expr::Rule { .. })
      || matches!(e, Expr::FunctionCall { name, .. }
        if name == "Rule" || name == "RuleDelayed")
  };
  let model = args.get(1).filter(|e| !is_option(e));
  let opts: Vec<String> = args
    .iter()
    .skip(if model.is_some() { 2 } else { 1 })
    .map(expr_to_string)
    .collect();
  let opts = if opts.is_empty() {
    String::new()
  } else {
    format!(", {}", opts.join(", "))
  };

  let poly = |n: usize| {
    let basis: Vec<String> = (0..=n).map(|k| format!("x$lfp^{k}")).collect();
    Some(format!("Fit[pts$lfp, {{{}}}, x$lfp]", basis.join(", ")))
  };
  let fit = match model {
    None => poly(1),
    Some(Expr::String(s)) => match s.as_str() {
      "Linear" => poly(1),
      "Quadratic" => poly(2),
      "Cubic" => poly(3),
      "Quartic" => poly(4),
      "Quintic" => poly(5),
      "Exponential" => Some(
        "Exp[Fit[{#[[1]], Log[#[[2]]]} & /@ pts$lfp, {1, x$lfp}, x$lfp]]"
          .to_string(),
      ),
      "Kernel" => Some(
        "With[{h$lfp = (Max[pts$lfp[[All, 1]]] - Min[pts$lfp[[All, 1]]])/ \
         Length[pts$lfp]^(1/5)/2}, Total[Exp[-(x$lfp - pts$lfp[[All, 1]])^2/ \
         (2 h$lfp^2)]*pts$lfp[[All, 2]]]/ \
         Total[Exp[-(x$lfp - pts$lfp[[All, 1]])^2/(2 h$lfp^2)]]]"
          .to_string(),
      ),
      _ => None,
    },
    Some(basis @ Expr::List(_)) => Some(format!(
      "Fit[pts$lfp, {}, x] /. x -> x$lfp",
      expr_to_string(basis)
    )),
    Some(_) => None,
  };
  let Some(fit) = fit else {
    return Ok(unevaluated("ListFitPlot", args));
  };

  // `data -> {lbl1, …}` fits `data`; the labels go to the plotted points.
  let data = evaluate_expr_to_expr(&args[0])?;
  let fit_data = match &data {
    Expr::Rule { pattern, .. } => pattern.as_ref(),
    other => other,
  };
  let code = format!(
    "Module[{{d$lfp = {fit_data}, pts$lfp, f$lfp}}, \
     pts$lfp = If[MatrixQ[d$lfp] && Dimensions[d$lfp][[2]] == 2, d$lfp, \
     Transpose[{{Range[Length[d$lfp]], d$lfp}}]]; \
     f$lfp = {fit}; \
     Show[ListPlot[{data}{opts}], Plot[Evaluate[f$lfp], {{x$lfp, \
     Min[pts$lfp[[All, 1]]], Max[pts$lfp[[All, 1]]]}}]]]",
    fit_data = expr_to_string(fit_data),
    data = expr_to_string(&data),
  );
  evaluate_expr_to_expr(&crate::syntax::string_to_expr(&code)?)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn max_plot_points_matches_wolfram_indices() {
    let one_based = |n, m| {
      plot_sample_indices(n, m)
        .into_iter()
        .map(|i| i + 1)
        .collect::<Vec<_>>()
    };
    assert_eq!(one_based(10, 5), vec![1, 3, 6, 9, 10]);
    assert_eq!(one_based(10, 6), vec![1, 2, 4, 6, 8, 10]);
    assert_eq!(one_based(15, 8), vec![1, 2, 4, 6, 8, 10, 12, 15]);
    assert_eq!(
      one_based(100, 30),
      vec![
        1, 2, 4, 5, 7, 8, 9, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56,
        60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100,
      ]
    );
    assert_eq!(
      one_based(100, 50),
      vec![
        1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27,
        28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 48, 51, 54, 57, 60, 63,
        66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 100,
      ]
    );
    for n in 2..120 {
      for m in 2..=n {
        let indices = plot_sample_indices(n, m);
        assert_eq!(indices.len(), m);
        assert_eq!(indices.first(), Some(&0));
        assert_eq!(indices.last(), Some(&(n - 1)));
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
      }
    }
  }

  #[test]
  fn max_plot_points_samples_all_list_plot_series() {
    for plot in [list_plot_ast, list_line_plot_ast] {
      let data = Expr::List(
        (1..=10)
          .map(|n| Expr::Integer(if n == 5 { 100 } else { n }))
          .collect(),
      );
      let args = vec![
        Expr::List(vec![data.clone(), data].into()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("MaxPlotPoints".into())),
          replacement: Box::new(Expr::Integer(3)),
        },
      ];
      let Expr::Graphics {
        source: Some(ref source),
        ..
      } = plot(&args).unwrap()
      else {
        panic!("expected plot source");
      };
      assert_eq!(source.series.len(), 2);
      assert!(source.y_range.1 < 20.0, "omitted outlier set plot range");
      for series in &source.series {
        assert_eq!(series.points, vec![(1.0, 1.0), (4.0, 4.0), (10.0, 10.0)]);
      }
    }
  }

  #[test]
  fn max_plot_points_keeps_labels_and_uncertainties_aligned() {
    let mut series = vec![
      (1..=10)
        .map(|n| ErrPoint::from_y(n as f64, (n as f64, (0.0, n as f64))))
        .collect(),
    ];
    let mut labels =
      vec![(1..=10).map(|n| Some((n.to_string(), None))).collect()];
    sample_plot_data(
      &[
        Expr::List(Vec::new().into()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("MaxPlotPoints".into())),
          replacement: Box::new(Expr::Integer(5)),
        },
      ],
      &mut series,
      &mut labels,
    );
    assert_eq!(
      series[0]
        .iter()
        .map(|p| p.dy.1 as usize)
        .collect::<Vec<_>>(),
      vec![1, 3, 6, 9, 10]
    );
    assert_eq!(
      labels[0]
        .iter()
        .map(|label| label.as_ref().unwrap().0.as_str())
        .collect::<Vec<_>>(),
      vec!["1", "3", "6", "9", "10"]
    );
  }

  #[test]
  fn max_plot_points_preserves_original_point_size() {
    let data = Expr::List((1..=100).map(Expr::Integer).collect());
    let Expr::Graphics {
      source: Some(ref full),
      ..
    } = list_plot_ast(&[data.clone()]).unwrap()
    else {
      panic!("expected plot source");
    };
    let Expr::Graphics {
      source: Some(ref sampled),
      ..
    } = list_plot_ast(&[
      data,
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("MaxPlotPoints".into())),
        replacement: Box::new(Expr::Integer(3)),
      },
    ])
    .unwrap()
    else {
      panic!("expected plot source");
    };
    assert_eq!(sampled.series[0].points.len(), 3);
    assert_eq!(sampled.series[0].point_size, full.series[0].point_size);
  }

  fn values(ys: &[f64]) -> Vec<(f64, f64)> {
    ys.iter()
      .enumerate()
      .map(|(i, &y)| ((i + 1) as f64, y))
      .collect()
  }

  fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12
  }

  fn step_xs(series: &[(f64, f64)], position: StepPosition) -> Vec<f64> {
    positioned_step_curve(series, position)
      .iter()
      .map(|p| p.0)
      .collect()
  }

  // Step intervals as wolframscript's ListStepPlot Line points give them.
  #[test]
  fn step_curve_positions_match_wolfram() {
    let data = values(&[1.0, 3.0, 2.0]);
    assert_eq!(
      step_xs(&data, StepPosition::Right),
      vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    );
    assert_eq!(
      step_xs(&data, StepPosition::Left),
      vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    );
    assert_eq!(
      step_xs(&data, StepPosition::Center),
      vec![0.5, 1.5, 1.5, 2.5, 2.5, 3.5]
    );
    // Irregular x: the open end steps reuse the adjacent gap.
    let uneven = vec![(1.0, 1.0), (2.0, 3.0), (4.0, 2.0), (7.0, 5.0)];
    assert_eq!(
      step_xs(&uneven, StepPosition::Right),
      vec![1.0, 2.0, 2.0, 4.0, 4.0, 7.0, 7.0, 10.0]
    );
    assert_eq!(
      step_xs(&uneven, StepPosition::Left),
      vec![0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 7.0]
    );
    assert_eq!(
      step_xs(&uneven, StepPosition::Center),
      vec![0.5, 1.5, 1.5, 3.0, 3.0, 5.5, 5.5, 8.5]
    );
    // A lone point: a unit step on the side opposite the position.
    let one = vec![(1.0, 5.0)];
    assert_eq!(step_xs(&one, StepPosition::Right), vec![0.0, 1.0]);
    assert_eq!(step_xs(&one, StepPosition::Left), vec![1.0, 2.0]);
    assert_eq!(step_xs(&one, StepPosition::Center), vec![0.5, 1.5]);
    let ys: Vec<f64> = positioned_step_curve(&data, StepPosition::Right)
      .iter()
      .map(|p| p.1)
      .collect();
    assert_eq!(ys, vec![1.0, 1.0, 3.0, 3.0, 2.0, 2.0]);
  }

  #[test]
  fn step_position_parses_symbol_and_list() {
    let id = |s: &str| Expr::Identifier(s.to_string());
    assert_eq!(StepPosition::parse(&id("Left")), Some(StepPosition::Left));
    assert_eq!(
      StepPosition::parse(&Expr::List(vec![id("Center")].into())),
      Some(StepPosition::Center)
    );
    assert_eq!(StepPosition::parse(&id("Top")), None);
  }

  // The joined InterpolationOrder curve is WL's uniform parametric
  // B-spline (values from wolframscript's ListPlot Line points).
  #[test]
  fn spline_curve_matches_wolfram_points() {
    let data = values(&[1.0, 4.0, 2.0, 3.0, 10.0, 8.0, 2.0]);
    let cubic = spline_curve(&data, 3);
    assert_eq!(cubic.len(), 157);
    assert!(close(cubic[1], (1.0252993839230191, 1.1112786798837362)));
    let top = cubic.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    assert!((top - 10.509765002305535).abs() < 1e-12, "{top}");
    let quadratic = spline_curve(&data, 2);
    assert!(close(quadratic[1], (1.023396301156198, 1.09145323398689)));
    let top = quadratic.iter().map(|p| p.1).fold(f64::MIN, f64::max);
    assert!((top - 10.442350579300358).abs() < 1e-12, "{top}");
    let quintic = spline_curve(&data, 5);
    assert!(close(quintic[1], (1.0281310425380483, 1.1313604537547608)));
    assert!(close(quintic[89], (4.433552453390919, 6.128563635638771)));
    // The curve passes through every data point.
    for (i, p) in data.iter().enumerate() {
      assert!(close(cubic[26 * i], *p));
    }
  }

  // Each index interval is sampled 2 ⌊1200 / (n - 1)⌋ times, in 6..=26.
  #[test]
  fn spline_curve_sample_count() {
    for (n, per) in [(5, 26), (93, 26), (95, 24), (150, 16), (500, 6)] {
      let ys: Vec<f64> = (0..n).map(|i| (i as f64 / 3.0).sin()).collect();
      let data = values(&ys);
      assert_eq!(spline_curve(&data, 3).len(), (n - 1) * per + 1, "n = {n}");
    }
  }

  // WL's Charting`adaptivePointSize: steps by count, then by clustering.
  #[test]
  fn point_size_steps_follow_density() {
    let golden = |n: usize, cube: bool| {
      vec![
        (1..=n)
          .map(|i| {
            let y = (i as f64 * 0.414214).fract();
            (
              (i as f64 * 0.618034).fract(),
              if cube { y.powi(3) } else { y },
            )
          })
          .collect::<Vec<_>>(),
      ]
    };
    assert_eq!(point_size_steps(&golden(300, false)), 4.0);
    assert_eq!(point_size_steps(&golden(600, false)), 4.0);
    assert_eq!(point_size_steps(&golden(600, true)), 3.0);
    assert_eq!(point_size_steps(&golden(1200, true)), 2.5);
    assert_eq!(point_size_steps(&golden(3000, true)), 2.0);
    assert_eq!(point_size_steps(&golden(6000, true)), 1.0);
    assert_eq!(point_size_steps(&golden(6000, false)), 2.0);
    assert_eq!(point_size_steps(&golden(20000, false)), 1.0);
    // Duplicate points count once.
    let a = golden(300, false).remove(0);
    assert_eq!(point_size_steps(&[a.clone(), a.clone(), a]), 4.0);
    // Below 1 the size stays 1/360 of the image width.
    assert_eq!(list_plot_point_size(&golden(20000, false)), 1.0 / 360.0);
  }
}
