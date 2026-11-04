//! `BoxWhiskerChart[{d1, d2, …}, bwspec]`: one box per dataset (quartiles,
//! median line, whiskers with caps), drawn the way Wolfram builds it — a
//! symbolic framed `Graphics` with a tick under each box. The `bwspec`
//! presets `"Basic"`, `"Median"`, `"Mean"`, `"Notched"` and `"Outliers"`
//! (and the `{{"Outliers"}, {"MeanMarker"}, …}` element list) change the
//! marks; `BarOrigin -> Left | Right | Top` turns or flips the chart.
//!
//! The box geometry (width from `BarSpacing`, the slot-padding invisible
//! `Point`) is shared with `DistributionChart`'s rectangular elements.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{invalid_dataset, parse_datasets};
use crate::functions::distribution_chart::{chart_legended, chart_styles};
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(key: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(key.to_string())),
    replacement: Box::new(value),
  }
}

fn id(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

fn list(items: Vec<Expr>) -> Expr {
  Expr::List(items.into())
}

fn gray(level: f64) -> Expr {
  call("GrayLevel", vec![Expr::Real(level)])
}

/// The automatic `BarSpacing` of `n` side-by-side boxes, as a fraction of
/// the box width (`BarSpacing -> s` makes each box `1/(1 + s)` of its
/// unit slot). Fitted to WL 15's choice, which narrows the gaps as the
/// count grows: 1.55 for one box, 1.01 for three, 0.48 for ten.
fn automatic_spacing(n: usize) -> f64 {
  1.0 / (0.645 + 0.1453 * (n.max(1) - 1) as f64)
}

/// Geometry of `n` boxes centred on 1, 2, …, n.
#[derive(Clone, Copy)]
pub(crate) struct BoxSlots {
  /// Half the width of a box.
  pub(crate) half: f64,
  /// How far the plot range reaches beyond the outer box edges.
  pub(crate) outer: f64,
}

impl BoxSlots {
  /// `spacing` is an explicit `BarSpacing` value, `None` for `Automatic`;
  /// `whisker_caps` widens the outer reach the way `BoxWhiskerChart`
  /// leaves more room than `DistributionChart`.
  pub(crate) fn new(
    n: usize,
    spacing: Option<f64>,
    whisker_caps: bool,
  ) -> Self {
    let s = spacing
      .filter(|s| s.is_finite() && *s >= 0.0)
      .unwrap_or_else(|| automatic_spacing(n));
    let half = 0.5 / (1.0 + s);
    let gap = 0.5 - half;
    // A lone box gets a whole slot either side; more boxes settle towards
    // half a gap (fitted to WL 15).
    let mut outer = gap + (1.0 - gap) * (-1.829 * (n.max(1) - 1) as f64).exp();
    if !whisker_caps {
      outer -= 0.3 * gap;
    }
    Self { half, outer }
  }

  /// The horizontal (category) extent of the plot range.
  pub(crate) fn range(&self, n: usize) -> (f64, f64) {
    (
      1.0 - self.half - self.outer,
      n as f64 + self.half + self.outer,
    )
  }
}

/// `BarSpacing -> s` (a number), `None` for `Automatic` or anything else.
pub(crate) fn bar_spacing_value(value: &Expr) -> Option<f64> {
  let v = match value {
    Expr::List(items) => items.first()?,
    other => other,
  };
  crate::functions::math_ast::try_eval_to_f64(v)
}

/// Category ticks (one per box, labelled by `labels`) for the frame edge
/// facing the category axis, and the unlabelled ticks of the edge opposite.
pub(crate) fn category_frame_ticks(n: usize, labels: &[Expr]) -> (Expr, Expr) {
  let len = || list(vec![Expr::Real(0.008), Expr::Integer(0)]);
  let labelled = (0..n)
    .map(|i| {
      let label = labels
        .get(i)
        .cloned()
        .unwrap_or_else(|| Expr::String(String::new()));
      list(vec![Expr::Real((i + 1) as f64), label, len()])
    })
    .collect();
  let plain = (0..n)
    .map(|i| {
      list(vec![
        Expr::Real((i + 1) as f64),
        Expr::String(String::new()),
        len(),
      ])
    })
    .collect();
  (list(labelled), list(plain))
}

/// Where the boxes grow from.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Origin {
  Bottom,
  Top,
  Left,
  Right,
}

impl Origin {
  pub(crate) fn parse(value: &Expr) -> Option<Self> {
    match value {
      Expr::Identifier(s) => match s.as_str() {
        "Bottom" => Some(Self::Bottom),
        "Top" => Some(Self::Top),
        "Left" => Some(Self::Left),
        "Right" => Some(Self::Right),
        _ => None,
      },
      _ => None,
    }
  }

  fn horizontal(self) -> bool {
    matches!(self, Self::Left | Self::Right)
  }

  /// `Top` and `Right` mirror the value axis (WL negates the values and
  /// labels the axis with their magnitudes).
  fn mirrored(self) -> bool {
    matches!(self, Self::Top | Self::Right)
  }

  /// The plot coordinates of category coordinate `c` and value `v`.
  pub(crate) fn coords(self, c: f64, v: f64) -> (f64, f64) {
    let v = if self.mirrored() { -v } else { v };
    if self.horizontal() { (v, c) } else { (c, v) }
  }

  /// The point at category coordinate `c` and value `v`.
  pub(crate) fn point(self, c: f64, v: f64) -> Expr {
    let (x, y) = self.coords(c, v);
    list(vec![Expr::Real(x), Expr::Real(y)])
  }
}

/// The frame ticks of a mirrored value axis: automatic positions over
/// `lo..hi` (the unmirrored values), placed at `-v` and labelled `v`.
fn mirrored_value_ticks(lo: f64, hi: f64, labelled: bool) -> Expr {
  use crate::functions::plot::{
    axis_tick_step, format_tick_with_step, minor_tick_step,
  };
  let step = axis_tick_step(hi - lo);
  let minor = minor_tick_step(step);
  let first = (lo / minor).ceil() as i64;
  let last = (hi / minor).floor() as i64;
  let per_major = (step / minor).round() as i64;
  let mut ticks = Vec::new();
  for k in first..=last {
    let v = k as f64 * minor;
    let major = per_major > 0 && k.rem_euclid(per_major) == 0;
    let label = if major && labelled {
      let v = if v.abs() < minor * 1e-9 { 0.0 } else { v };
      Expr::String(format_tick_with_step(v, step))
    } else {
      Expr::String(String::new())
    };
    let len = if major { 0.00625 } else { 0.00375 };
    ticks.push(list(vec![
      Expr::Real(-v),
      label,
      list(vec![Expr::Real(len), Expr::Integer(0)]),
    ]));
  }
  list(ticks)
}

/// The `FrameTicks` of a chart of `n` categories: per-box ticks on the
/// category edges, automatic (or mirrored) value ticks on the others.
pub(crate) fn chart_frame_ticks(
  origin: Origin,
  n: usize,
  labels: &[Expr],
  value_range: (f64, f64),
) -> Expr {
  let (cat, cat_opposite) = category_frame_ticks(n, labels);
  let value = |labelled: bool| {
    if origin.mirrored() {
      mirrored_value_ticks(value_range.0, value_range.1, labelled)
    } else {
      id("Automatic")
    }
  };
  let (value_side, value_opposite) = (value(true), value(false));
  let value_pair = list(vec![value_side, value_opposite]);
  let cat_pair = list(vec![cat, cat_opposite]);
  if origin.horizontal() {
    list(vec![cat_pair, value_pair])
  } else {
    list(vec![value_pair, cat_pair])
  }
}

/// The invisible `Point` that fixes a chart's plot range: the category
/// extent and the value range widened by 5% each side.
pub(crate) fn range_point(
  origin: Origin,
  cat: (f64, f64),
  values: (f64, f64),
) -> Expr {
  let (lo, hi) = values;
  // A constant value range stays as it is (the renderer pads it).
  let pad = if hi > lo { 0.05 * (hi - lo) } else { 0.0 };
  let (lo, hi) = (lo - pad, hi + pad);
  list(vec![
    call("Opacity", vec![Expr::Integer(0)]),
    call(
      "Point",
      vec![list(vec![origin.point(cat.0, lo), origin.point(cat.1, hi)])],
    ),
  ])
}

/// The common `Graphics` options of a framed chart, after the caller's own
/// (`user`) options so those take precedence.
pub(crate) fn framed_chart_options(
  user: Vec<Expr>,
  frame_ticks: Expr,
) -> Vec<Expr> {
  let has = |k: &str| {
    user.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == k))
    })
  };
  let defaults = vec![
    rule(
      "AspectRatio",
      call("Power", vec![id("GoldenRatio"), Expr::Integer(-1)]),
    ),
    rule("Axes", list(vec![id("False"), id("False")])),
    rule("Frame", id("True")),
    rule("FrameTicks", frame_ticks),
    rule("PlotRangePadding", call("Scaled", vec![Expr::Real(0.05)])),
  ];
  let keep: Vec<Expr> = defaults
    .into_iter()
    .filter(|d| match d {
      Expr::Rule { pattern, .. } => match pattern.as_ref() {
        Expr::Identifier(k) => !has(k),
        _ => true,
      },
      _ => true,
    })
    .collect();
  let mut options = user;
  options.extend(keep);
  options
}

/// Which marks a `bwspec` asks for.
#[derive(Default)]
#[allow(clippy::struct_excessive_bools)]
struct Marks {
  basic: bool,
  mean: bool,
  notched: bool,
  outliers: bool,
}

fn bwspec_message(spec: &Expr) {
  crate::emit_message(&format!(
    "BoxWhiskerChart::bwspec: {} cannot be used as a box-and-whisker symbol specification.",
    crate::syntax::format_expr(spec, crate::syntax::ExprForm::Output)
  ));
}

fn parse_bwspec(spec: &Expr) -> Marks {
  let mut marks = Marks::default();
  match spec {
    Expr::String(s) => match s.as_str() {
      "Median" | "Diamond" => {}
      "Basic" => marks.basic = true,
      "Mean" => marks.mean = true,
      "Notched" => marks.notched = true,
      "Outliers" => marks.outliers = true,
      _ => bwspec_message(spec),
    },
    // `{{"Outliers", …}, {"MeanMarker", …}, …}`: one entry per element.
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(|i| matches!(i, Expr::List(_))) =>
    {
      for item in items {
        let Expr::List(parts) = item else { continue };
        match parts.first() {
          Some(Expr::String(s)) if s == "Outliers" || s == "FarOutliers" => {
            marks.outliers = true;
          }
          Some(Expr::String(s)) if s == "MeanMarker" => marks.mean = true,
          Some(Expr::String(s))
            if matches!(
              s.as_str(),
              "MedianMarker" | "Whiskers" | "Fences" | "Notch"
            ) => {}
          _ => bwspec_message(item),
        }
      }
    }
    _ => bwspec_message(spec),
  }
  marks
}

/// Quartiles the way `Quartiles` interpolates them: positions (n+2)/4,
/// (n+1)/2 and (3n+2)/4 of the sorted data (1-indexed).
fn quartiles(sorted: &[f64]) -> (f64, f64, f64) {
  let n = sorted.len() as f64;
  let interp = |pos: f64| -> f64 {
    let idx = (pos - 1.0).clamp(0.0, n - 1.0);
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
  };
  (
    interp((n + 2.0) / 4.0),
    interp(f64::midpoint(n, 1.0)),
    interp((3.0 * n + 2.0) / 4.0),
  )
}

pub fn box_whisker_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Association / temporal data chart their values (keys name the boxes
  // under `ChartLabels -> Automatic`).
  if let Some(new_args) = crate::functions::chart::chart_data_args(args, true) {
    return box_whisker_chart_ast(&new_args);
  }
  let Ok(mut datasets) = parse_datasets(&args[0]) else {
    return Ok(invalid_dataset("BoxWhiskerChart", args));
  };
  if datasets.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }
  let mut marks = Marks::default();
  let mut style = None;
  let mut legends = None;
  let mut labels = None;
  let mut origin = Origin::Bottom;
  let mut spacing = None;
  let mut graphics_opts = Vec::new();
  for arg in &args[1..] {
    match arg {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => {
        let value = || {
          evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone())
        };
        match pattern.as_ref() {
          Expr::Identifier(k) if k == "ChartStyle" => style = Some(value()),
          Expr::Identifier(k) if k == "ChartLegends" => legends = Some(value()),
          Expr::Identifier(k) if k == "ChartLabels" => labels = Some(value()),
          Expr::Identifier(k) if k == "BarOrigin" => {
            if let Some(o) = Origin::parse(&value()) {
              origin = o;
            }
          }
          Expr::Identifier(k) if k == "BarSpacing" => {
            spacing = bar_spacing_value(&value());
          }
          Expr::Identifier(k)
            if matches!(
              k.as_str(),
              "ChartBaseStyle"
                | "ChartElementFunction"
                | "ChartElements"
                | "ChartLayout"
                | "LabelingFunction"
                | "LegendAppearance"
                | "PerformanceGoal"
            ) => {}
          _ => graphics_opts.push(arg.clone()),
        }
      }
      spec => {
        let spec = evaluate_expr_to_expr(spec).unwrap_or_else(|_| spec.clone());
        marks = parse_bwspec(&spec);
      }
    }
  }

  let n = datasets.len();
  let fills = chart_styles(style.as_ref(), n);
  let slots = BoxSlots::new(n, spacing, true);
  let half = slots.half;
  let cap = half / 2.0;
  let p = |c: f64, v: f64| origin.point(c, v);
  let seg = |a: Expr, b: Expr| list(vec![a, b]);

  let (mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY);
  let mut boxes = Vec::with_capacity(n);
  for (i, data) in datasets.iter_mut().enumerate() {
    data.sort_by(f64::total_cmp);
    let c = (i + 1) as f64;
    let (min, max) = (data[0], data[data.len() - 1]);
    v_lo = v_lo.min(min);
    v_hi = v_hi.max(max);
    let (q1, median, q3) = quartiles(data);
    let iqr = q3 - q1;
    // With outliers marked, whiskers stop at the last datum inside the
    // 1.5 IQR fences; the rest are drawn as dots (far ones, beyond 3 IQR,
    // in gray).
    let (w_lo, w_hi) = if marks.outliers {
      let (f_lo, f_hi) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
      let inside = data.iter().copied().filter(|&v| v >= f_lo && v <= f_hi);
      let lo = inside.clone().fold(f64::INFINITY, f64::min);
      let hi = inside.fold(f64::NEG_INFINITY, f64::max);
      (lo.min(q1), hi.max(q3))
    } else {
      (min, max)
    };

    let (l, r) = (c - half, c + half);
    let box_polygon = if marks.notched {
      // McGill's notch, median ± 1.7 · 1.25 IQR / (1.35 √n), clipped to
      // the box.
      let depth = 1.7 * 1.25 * iqr / (1.35 * (data.len() as f64).sqrt());
      let (n_lo, n_hi) = ((median - depth).max(q1), (median + depth).min(q3));
      vec![
        p(l, q3),
        p(r, q3),
        p(r, n_hi),
        p(c + cap, median),
        p(r, n_lo),
        p(r, q1),
        p(l, q1),
        p(l, n_lo),
        p(c - cap, median),
        p(l, n_hi),
      ]
    } else {
      vec![p(l, q3), p(r, q3), p(r, q1), p(l, q1)]
    };
    let mut parts = vec![
      call(
        "Directive",
        vec![call("EdgeForm", vec![id("None")]), fills[i].clone()],
      ),
      call("Polygon", vec![list(box_polygon)]),
    ];
    if marks.outliers {
      let far = |v: f64| v < q1 - 3.0 * iqr || v > q3 + 3.0 * iqr;
      for (is_far, level) in [(false, 0.0), (true, 0.5)] {
        let dots: Vec<Expr> = data
          .iter()
          .copied()
          .filter(|&v| (v < w_lo || v > w_hi) && far(v) == is_far)
          .map(|v| p(c, v))
          .collect();
        if !dots.is_empty() {
          parts.push(list(vec![
            gray(level),
            call("AbsolutePointSize", vec![Expr::Integer(4)]),
            call("Point", vec![list(dots)]),
          ]));
        }
      }
    }
    let median_half = if marks.notched { cap } else { half };
    parts.push(list(vec![
      gray(1.0),
      call(
        "Line",
        vec![seg(p(c + median_half, median), p(c - median_half, median))],
      ),
    ]));
    if marks.mean {
      let mean = data.iter().sum::<f64>() / data.len() as f64;
      parts.push(list(vec![
        gray(0.0),
        call("Line", vec![seg(p(c + cap, mean), p(c - cap, mean))]),
      ]));
    }
    parts.push(list(vec![
      call("CapForm", vec![Expr::String("Butt".to_string())]),
      gray(0.5),
      call(
        "Line",
        vec![list(vec![
          seg(p(c, q1), p(c, w_lo)),
          seg(p(c, q3), p(c, w_hi)),
        ])],
      ),
    ]));
    if !marks.basic {
      parts.push(list(vec![
        gray(0.5),
        call(
          "Line",
          vec![list(vec![
            seg(p(c + cap, w_lo), p(c - cap, w_lo)),
            seg(p(c + cap, w_hi), p(c - cap, w_hi)),
          ])],
        ),
      ]));
    }
    boxes.push(list(parts));
  }

  let content = vec![
    range_point(origin, slots.range(n), (v_lo, v_hi)),
    list(vec![call("CapForm", vec![id("None")]), list(boxes)]),
  ];
  let tick_labels: Vec<Expr> = match &labels {
    Some(Expr::List(items)) => items.to_vec(),
    _ => Vec::new(),
  };
  let span = (v_hi - v_lo).max(0.0);
  let frame_ticks = chart_frame_ticks(
    origin,
    n,
    &tick_labels,
    (v_lo - 0.11 * span, v_hi + 0.11 * span),
  );
  let mut graphics_args = vec![list(content)];
  graphics_args.extend(framed_chart_options(graphics_opts, frame_ticks));
  let chart = evaluate_expr_to_expr(&call("Graphics", graphics_args))?;
  Ok(chart_legended(chart, fills, legends.as_ref(), n))
}
