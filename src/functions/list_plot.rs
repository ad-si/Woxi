use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, Mesh, PLOT_COLORS, PlotOptions, SeriesStyle,
  adjust_y_range_for_filling, apply_plot_theme, build_plot_source,
  generate_scatter_svg_with_options, generate_svg_with_filling, parse_filling,
  parse_image_size, parse_plot_legends, parse_plot_style,
};
use crate::functions::sound::audio_sample_rate;
use crate::syntax::{Expr, unevaluated};

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
}

impl ErrPoint {
  fn from_y(x: f64, (y, dy): (f64, (f64, f64))) -> Self {
    ErrPoint {
      x,
      y,
      dx: (0.0, 0.0),
      dy,
    }
  }

  fn from_xy((x, dx): (f64, (f64, f64)), (y, dy): (f64, (f64, f64))) -> Self {
    ErrPoint { x, y, dx, dy }
  }
}

/// Evaluate an expression to a plottable value with uncertainty: a plain
/// number carries zero uncertainty, `Around[v, u]` / `Around[v, {m, p}]`
/// carry (minus, plus) error-bar half-widths.
fn eval_to_value_err(expr: &Expr) -> Option<(f64, (f64, f64))> {
  let e = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
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
  try_eval_to_f64(&e).map(|v| (v, (0.0, 0.0)))
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
    other => crate::syntax::expr_to_output(other),
  }
}

/// Convert an association into plottable list data. Keys that are all numeric
/// become x-coordinates (`<|k -> v|>` → `{{k, v}, …}`); with non-numeric keys
/// the values are plotted at sequential x. When every value is itself a list,
/// each value is a separate series and its key becomes the series label
/// (`{Labeled[v1, k1], …}`).
fn association_to_data(pairs: &[(Expr, Expr)]) -> Expr {
  let all_values_lists =
    !pairs.is_empty() && pairs.iter().all(|(_, v)| matches!(v, Expr::List(_)));
  if all_values_lists {
    return Expr::List(
      pairs
        .iter()
        .map(|(k, v)| Expr::FunctionCall {
          name: "Labeled".into(),
          args: vec![canonicalize_element(v), Expr::String(label_string(k))]
            .into(),
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
  Expr::List(pairs.iter().map(|(_, v)| v.clone()).collect())
}

/// Strip display wrappers from a single list entry (a scalar, an `{x, y}`
/// pair, or a nested series), recursing into inner lists so that
/// `{1, Style[2, Red], 3}` keeps its numbers. `Callout` is kept as a
/// `Labeled` so its label still draws.
fn canonicalize_element(e: &Expr) -> Expr {
  match e {
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      match name.as_str() {
        "Tooltip" | "Style" | "Legended" | "Annotation" | "PopupWindow"
        | "Mouseover" | "StatusArea" | "Button" => {
          canonicalize_element(&args[0])
        }
        "Callout" if args.len() >= 2 => Expr::FunctionCall {
          name: "Labeled".into(),
          args: vec![canonicalize_element(&args[0]), args[1].clone()].into(),
        },
        "Callout" => canonicalize_element(&args[0]),
        _ => e.clone(),
      }
    }
    Expr::List(inner) => {
      Expr::List(inner.iter().map(canonicalize_element).collect())
    }
    _ => e.clone(),
  }
}

/// Extract the raw data points from a `WeightedData` object. The evaluated
/// canonical form is `WeightedData[Automatic, {{data…}, {weights…}}]`; the
/// user-facing form is `WeightedData[data, weights]`. Either way the data is
/// the list of values (weights only affect statistics, not point positions).
fn weighted_data_values(args: &[Expr]) -> Expr {
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
fn canonicalize_plot_data(expr: &Expr) -> Expr {
  let e = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  match &e {
    Expr::Association(pairs) => association_to_data(pairs),
    Expr::List(items) => {
      Expr::List(items.iter().map(canonicalize_element).collect())
    }
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      match name.as_str() {
        "Tooltip" | "Style" | "Legended" | "Annotation" | "PopupWindow"
        | "Mouseover" | "StatusArea" | "Button" => {
          canonicalize_plot_data(&args[0])
        }
        "Callout" if args.len() >= 2 => Expr::FunctionCall {
          name: "Labeled".into(),
          args: vec![canonicalize_plot_data(&args[0]), args[1].clone()].into(),
        },
        "Callout" => canonicalize_plot_data(&args[0]),
        "WeightedData" => canonicalize_plot_data(&weighted_data_values(args)),
        "SparseArray" => evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Normal".into(),
          args: vec![e.clone()].into(),
        })
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
) -> Result<Vec<Vec<ErrPoint>>, InterpreterError> {
  let data = canonicalize_plot_data(arg);

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
      && eval_to_value_err(&first_inner[0]).is_some()
      && eval_to_value_err(&first_inner[1]).is_some()
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
    if let Some(y) = eval_to_value_err(item) {
      points.push(ErrPoint::from_y((i + 1) as f64, y));
    }
  }
  Ok(vec![points])
}

/// Parse a single series: either plain y-values or {x, y} pairs.
fn parse_single_series(
  series_items: &[Expr],
) -> Result<Vec<ErrPoint>, InterpreterError> {
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
  Ok(points)
}

/// If `expr` is `Labeled[content, label, ...]`, return the content and the
/// label rendered as display text.
fn unwrap_labeled(expr: &Expr) -> Option<(Expr, String)> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Labeled"
    && args.len() >= 2
  {
    let label = match &args[1] {
      Expr::String(s) => s.clone(),
      other => crate::syntax::expr_to_output(other),
    };
    Some((args[0].clone(), label))
  } else {
    None
  }
}

/// One parsed series per dataset plus one optional dataset label per series.
type LabeledSeries = (Vec<Vec<(f64, f64)>>, Vec<Option<String>>);

/// Like [`LabeledSeries`], but keeping the `Around` uncertainties plus
/// per-point labels (parallel to each series' points) from `Labeled`
/// wrappers around scalar data entries.
type LabeledErrSeries = (
  Vec<Vec<ErrPoint>>,
  Vec<Option<String>>,
  Vec<Vec<Option<String>>>,
);

/// Like [`parse_list_data_err_labeled`], with the uncertainties and
/// per-point labels stripped.
fn parse_list_data_labeled(
  arg: &Expr,
) -> Result<LabeledSeries, InterpreterError> {
  let (series, labels, _) = parse_list_data_err_labeled(arg)?;
  Ok((strip_errors(&series), labels))
}

/// Parse `ListPlot[ts -> "key"]` / `ListPlot[ts -> {"k1", …}]`: select the
/// named components of a component-keyed TimeSeries, one series per key.
fn parse_temporal_components(
  ts: &Expr,
  selector: &Expr,
) -> Result<LabeledErrSeries, InterpreterError> {
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
  Ok((series, Vec::new(), Vec::new()))
}

/// Parse the `ListPlot[data -> labels]` form: the left side is the data
/// (points), the right side a parallel list of per-point labels attached to
/// the first series.
fn parse_rule_labeled(
  lhs: &Expr,
  rhs: &Expr,
) -> Result<LabeledErrSeries, InterpreterError> {
  let series = parse_list_data_err(lhs)?;
  let rhs = evaluate_expr_to_expr(rhs).unwrap_or_else(|_| rhs.clone());
  let labels: Vec<Option<String>> = match &rhs {
    Expr::List(items) => {
      items.iter().map(|it| Some(label_string(it))).collect()
    }
    other => vec![Some(label_string(other))],
  };
  let point_labels: Vec<Vec<Option<String>>> = series
    .iter()
    .enumerate()
    .map(|(i, s)| {
      if i == 0 {
        (0..s.len())
          .map(|j| labels.get(j).cloned().flatten())
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
fn parse_list_data_err_labeled(
  arg: &Expr,
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
      return parse_temporal_components(&lhs, replacement);
    }
    return parse_rule_labeled(pattern, replacement);
  }

  // Normalize container / display wrappers (Tooltip, Style, Callout,
  // SparseArray, WeightedData, associations, …) into plain list data.
  let data = canonicalize_plot_data(&raw);

  // Whole argument wrapped: the label applies to every series inside.
  if let Some((content, label)) = unwrap_labeled(&data) {
    let series = parse_list_data_err(&content)?;
    let labels = vec![Some(label); series.len()];
    return Ok((series, labels, Vec::new()));
  }

  if let Expr::List(items) = &data
    && items.iter().any(|item| unwrap_labeled(item).is_some())
  {
    let unwrapped: Vec<(Expr, Option<String>)> = items
      .iter()
      .map(|item| {
        let (content, label) = match unwrap_labeled(item) {
          Some((content, label)) => (content, Some(label)),
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
          all_series.push(parse_single_series(series_items)?);
        }
        // A labeled scalar among datasets contributes a one-point series.
        other => match eval_to_value_err(other) {
          Some(y) => all_series.push(vec![ErrPoint::from_y(1.0, y)]),
          None => all_series.push(Vec::new()),
        },
      }
      labels.push(label);
    }
    return Ok((all_series, labels, Vec::new()));
  }

  Ok((parse_list_data_err(&data)?, Vec::new(), Vec::new()))
}

/// Panel arrangement from the PlotLayout option.
#[derive(Default, Clone, Copy, PartialEq)]
enum PanelLayout {
  /// All datasets share one panel (the default).
  #[default]
  Overlaid,
  /// One panel per dataset, side by side.
  Row,
  /// One panel per dataset, stacked vertically.
  Column,
}

/// Parsed list-plot options, including explicit PlotRange overrides.
#[derive(Default)]
struct ParsedOptions {
  opts: PlotOptions,
  joined: bool,
  layout: PanelLayout,
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  /// `LabelingFunction -> None` (or `Tooltip`): no static point labels drawn.
  hide_point_labels: bool,
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
        "PlotLayout" => {
          if let Expr::String(layout) = replacement.as_ref() {
            match layout.as_str() {
              "Row" => out.layout = PanelLayout::Row,
              "Column" => out.layout = PanelLayout::Column,
              _ => {}
            }
          }
        }
        "AspectRatio" => {
          // Ratio (height/width) of the plotting area; the total image height
          // is derived once margins are known (see generate_svg_with_options).
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(r) = try_eval_to_f64(&val)
            && r > 0.0
          {
            opts.aspect_ratio = Some(r);
          }
        }
        "Filling" => {
          opts.filling = parse_filling(replacement);
        }
        "Background" => {
          opts.background =
            crate::functions::plot::parse_background_option(replacement);
        }
        "PlotStyle" => {
          opts.plot_style = parse_plot_style(replacement);
        }
        // Labels that only appear on hover (Tooltip) or are switched off
        // (None) are not drawn on a static SVG.
        "LabelingFunction" => {
          if matches!(replacement.as_ref(),
            Expr::Identifier(v) if v == "None" || v == "Tooltip")
          {
            out.hide_point_labels = true;
          }
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

/// Style for the panel showing series `idx`: the user-supplied style for
/// that series if any, falling back to the default palette color the series
/// would have had in the overlaid layout.
fn panel_style(user_styles: &[SeriesStyle], idx: usize) -> SeriesStyle {
  let mut style = if user_styles.is_empty() {
    SeriesStyle::default()
  } else {
    user_styles[idx % user_styles.len()].clone()
  };
  if style.color.is_none() {
    let (r, g, b) = PLOT_COLORS[idx % PLOT_COLORS.len()];
    style.color = Some(crate::functions::graphics::Color::new(
      r as f64 / 255.0,
      g as f64 / 255.0,
      b as f64 / 255.0,
    ));
  }
  style
}

/// Render one panel per series for `PlotLayout -> "Row" / "Column"` and
/// combine the panels into a single graphic. Each panel keeps the color its
/// series would have had overlaid, gets its own axis ranges, and shows its
/// dataset label (from `Labeled`) if one was given. `range_series` supplies
/// the per-series points used for range computation (the error-bar extremes
/// when a series carries `Around` uncertainties; otherwise `all_series`).
fn render_panel_layout(
  all_series: &[Vec<(f64, f64)>],
  range_series: &[Vec<(f64, f64)>],
  labels: &[Option<String>],
  parsed: &ParsedOptions,
  scatter: bool,
) -> Result<Expr, InterpreterError> {
  let mut svgs = Vec::with_capacity(all_series.len());
  for (idx, series) in all_series.iter().enumerate() {
    let mut opts = parsed.opts.clone();
    opts.plot_style = vec![panel_style(&parsed.opts.plot_style, idx)];
    opts.callout_labels = vec![labels.get(idx).cloned().flatten()];
    // A shared legend across separate panels would repeat per panel.
    opts.plot_legends = Vec::new();
    // Keep only this panel's error bars (they are indexed by series).
    opts.error_bars = match parsed.opts.error_bars.get(idx) {
      Some(bars) => vec![bars.clone()],
      None => Vec::new(),
    };

    let single = std::slice::from_ref(series);
    let range_single = range_series
      .get(idx)
      .map(std::slice::from_ref)
      .unwrap_or(single);
    let (x_range, y_range) =
      compute_ranges_scaled(range_single, opts.log_x, opts.log_y);
    let y_range = adjust_y_range_for_filling(opts.filling, y_range);
    let (x_range, y_range) =
      apply_plot_range_override(parsed, x_range, y_range);
    let svg = if scatter {
      generate_scatter_svg_with_options(single, x_range, y_range, &opts)?
    } else {
      generate_svg_with_filling(single, x_range, y_range, &opts)?
    };
    svgs.push(svg);
  }

  let rows: Vec<Vec<String>> = match parsed.layout {
    PanelLayout::Column => svgs.into_iter().map(|svg| vec![svg]).collect(),
    _ => vec![svgs],
  };
  let combined = crate::functions::graphics::combine_graphics_svgs(&rows)
    .unwrap_or_else(|| {
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string()
    });
  Ok(crate::graphics_result(combined))
}

/// Attach dataset labels from `Labeled` wrappers as callout labels, unless
/// explicit Callout labels are already present.
fn apply_dataset_labels(opts: &mut PlotOptions, labels: &[Option<String>]) {
  if labels.iter().any(|l| l.is_some()) && opts.callout_labels.is_empty() {
    opts.callout_labels = labels.to_vec();
  }
}

/// ListPlot[{y1, y2, ...}] or ListPlot[{{x1,y1}, ...}]
pub fn list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (err_series, labels, point_labels) =
    parse_list_data_err_labeled(&args[0])?;
  let all_series = strip_errors(&err_series);
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
  parsed.opts.point_labels = if parsed.hide_point_labels {
    Vec::new()
  } else {
    point_labels
  };
  parsed.opts.error_bars = collect_error_bars(&err_series);
  if parsed.layout != PanelLayout::Overlaid && all_series.len() > 1 {
    return render_panel_layout(
      &all_series,
      &error_extremes(&err_series),
      &labels,
      &parsed,
      !parsed.joined,
    );
  }
  // The plot range must cover the full extent of any error bars.
  let (x_range, y_range) = compute_ranges(&error_extremes(&err_series));
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

/// Convert a flat list of complex numbers into (Re, Im) points.
/// Entries that don't evaluate to a (possibly complex) number are skipped.
fn complex_series_points(items: &[Expr]) -> Vec<(f64, f64)> {
  items
    .iter()
    .filter_map(|z| {
      let e = evaluate_expr_to_expr(z).unwrap_or_else(|_| z.clone());
      crate::functions::list_helpers_ast::expr_to_complex_parts(&e)
    })
    .filter(|(re, im)| re.is_finite() && im.is_finite())
    .collect()
}

/// Parse complex list data from the first argument of ComplexListPlot.
///
/// Supported formats:
/// - `{z1, z2, ...}` → single series of (Re, Im) points
/// - `{{z11, z12, ...}, {z21, z22, ...}}` → multiple series
fn parse_complex_list_data(
  arg: &Expr,
) -> Result<Vec<Vec<(f64, f64)>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ComplexListPlot: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![vec![]]);
  }

  // Multiple datasets: {{z11, ...}, {z21, ...}}
  if items.iter().all(|item| matches!(item, Expr::List(_))) {
    let mut all_series = Vec::new();
    for item in items {
      if let Expr::List(zs) = item {
        all_series.push(complex_series_points(zs));
      }
    }
    return Ok(all_series);
  }

  Ok(vec![complex_series_points(items)])
}

/// ComplexListPlot[{z1, z2, ...}] plots complex numbers as points at
/// (Re[z], Im[z]) in the complex plane, like ListPlot[ReIm[data]].
pub fn complex_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let all_series = parse_complex_list_data(&args[0])?;
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
  let (err_series, labels, point_labels) =
    parse_list_data_err_labeled(&args[0])?;
  let all_series = strip_errors(&err_series);
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
  parsed.opts.point_labels = if parsed.hide_point_labels {
    Vec::new()
  } else {
    point_labels
  };
  parsed.opts.error_bars = collect_error_bars(&err_series);
  if parsed.layout != PanelLayout::Overlaid && all_series.len() > 1 {
    return render_panel_layout(
      &all_series,
      &error_extremes(&err_series),
      &labels,
      &parsed,
      false,
    );
  }
  let (x_range, mut y_range) = compute_ranges(&error_extremes(&err_series));

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
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
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
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);

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

  if parsed.layout != PanelLayout::Overlaid && step_series.len() > 1 {
    return render_panel_layout(
      &step_series,
      &step_series,
      &labels,
      &parsed,
      false,
    );
  }
  let (x_range, y_range) = compute_ranges(&step_series);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&step_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListLogPlot: y-axis is log10 scale
pub fn list_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
  parsed.opts.log_y = true;

  // Filter non-positive y values; data stays in original space (LogCoord handles scaling)
  let filtered: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| series.iter().filter(|&&(_, y)| y > 0.0).copied().collect())
    .collect();

  if parsed.layout != PanelLayout::Overlaid && filtered.len() > 1 {
    return render_panel_layout(&filtered, &filtered, &labels, &parsed, false);
  }
  let (x_range, y_range) = compute_ranges_scaled(&filtered, false, true);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&filtered, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListLogLogPlot: both axes log10 scale
pub fn list_log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
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

  if parsed.layout != PanelLayout::Overlaid && filtered.len() > 1 {
    return render_panel_layout(&filtered, &filtered, &labels, &parsed, false);
  }
  let (x_range, y_range) = compute_ranges_scaled(&filtered, true, true);
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
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);
  parsed.opts.log_x = true;

  // Filter non-positive x values; data stays in original space (LogCoord handles scaling)
  let filtered: Vec<Vec<(f64, f64)>> = all_series
    .iter()
    .map(|series| series.iter().filter(|&&(x, _)| x > 0.0).copied().collect())
    .collect();

  if parsed.layout != PanelLayout::Overlaid && filtered.len() > 1 {
    return render_panel_layout(&filtered, &filtered, &labels, &parsed, false);
  }
  let (x_range, y_range) = compute_ranges_scaled(&filtered, true, false);
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&filtered, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}

/// ListPolarPlot[{r1, r2, ...}]: plot data in polar coordinates
pub fn list_polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (all_series, labels) = parse_list_data_labeled(&args[0])?;
  let mut parsed = parse_plot_options(args);
  apply_dataset_labels(&mut parsed.opts, &labels);

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
  let y_range = adjust_y_range_for_filling(parsed.opts.filling, y_range);
  let (x_range, y_range) = apply_plot_range_override(&parsed, x_range, y_range);
  let svg =
    generate_svg_with_filling(&all_series, x_range, y_range, &parsed.opts)?;
  Ok(crate::graphics_result(svg))
}
