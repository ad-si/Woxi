//! `BubbleChart[{{x, y, z}, …}]`, a list of such datasets, or an
//! association of them.
//!
//! The chart is built as a symbolic `Graphics` of `Disk` primitives, the way
//! wolframscript builds it, so `ImageSize`, `PlotLabel`, `Frame…` and the
//! other graphics options pass straight through to the shared renderer — and
//! the chart scales like any other picture when a layout such as
//! `GraphicsRow` redraws it at a smaller size.
//!
//! Geometry, as decoded from wolframscript's output:
//!
//! - every bubble is an ellipse in data coordinates whose semi-axes are a
//!   fixed fraction of the data span along each axis. With the default
//!   `BubbleSizes -> Automatic` the largest bubble's radius is a tenth of
//!   the span and a bubble's area is proportional to its `z`
//!   (`r = 0.1 span Sqrt[z / zmax]`); `BubbleSizes -> {smin, smax}` sets
//!   the smallest and largest diameters as fractions of the span instead,
//!   interpolating linearly in `Sqrt[z]`. An axis with no spread (all x
//!   equal) uses `2 Abs[x]` as its span, or 2 at x = 0;
//! - bubbles with `z <= 0` are not drawn, but still count towards the span;
//! - the plot range just encloses the bubbles, padded by `Scaled[0.05]` on
//!   every side, and the plot is square (`AspectRatio -> 1`), so every
//!   ellipse comes out as a circle;
//! - the bubbles are drawn largest first, so small ones stay visible on top;
//! - default colors run by dataset (a single dataset is drawn in the first
//!   color); a `ChartStyle` list cycles over the points of a single dataset
//!   or over the datasets of several;
//! - `ChartLabels` sit centered on their bubbles.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{
  chart_option_alias, expr_to_label, invalid_dataset, named_color_scheme,
  sample_gradient,
};
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::rectangle_chart::chart_style_for;
use crate::syntax::Expr;

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.into(),
    args: args.into(),
  }
}

fn rule(name: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(name.into())),
    replacement: Box::new(value),
  }
}

fn pair(a: f64, b: f64) -> Expr {
  Expr::List(vec![Expr::Real(a), Expr::Real(b)].into())
}

/// One parsed bubble: its coordinates, size and flat input position (the
/// index `ChartLabels` are matched against).
#[derive(Clone, Copy)]
struct Bubble {
  x: f64,
  y: f64,
  z: f64,
  pos: usize,
}

fn parse_triple(item: &Expr) -> Option<(f64, f64, f64)> {
  let Expr::List(inner) = item else {
    return None;
  };
  if inner.len() < 3 {
    return None;
  }
  let num = |e: &Expr| {
    try_eval_to_f64(&evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone()))
  };
  Some((num(&inner[0])?, num(&inner[1])?, num(&inner[2])?))
}

/// Parse a list of triples, numbering the entries from `*cursor` on. Entries
/// that are no triple (`Missing[]`, a complex number, …) are skipped but
/// still take up a position.
fn parse_set(items: &[Expr], cursor: &mut usize) -> Vec<Bubble> {
  let mut out = Vec::with_capacity(items.len());
  for item in items {
    let pos = *cursor;
    *cursor += 1;
    if let Some((x, y, z)) = parse_triple(item) {
      out.push(Bubble { x, y, z, pos });
    }
  }
  out
}

/// Whether `e` is a list of triples (a dataset) rather than a single triple.
fn is_dataset(e: &Expr) -> bool {
  matches!(e, Expr::List(inner) if inner.first().is_some_and(|f| matches!(f, Expr::List(_))))
}

/// The datasets of the chart, whether it has several, and the association
/// keys (one per dataset when `multi`, else one per point).
type Parsed = (Vec<Vec<Bubble>>, bool, Option<Vec<Expr>>);

fn parse_data(data: &Expr) -> Option<Parsed> {
  match data {
    // `<|k -> {x, y, z}, …|>` is one dataset whose points are named by the
    // keys; `<|k -> {{x, y, z}, …}, …|>` is one dataset per key.
    Expr::Association(pairs) => {
      let keys: Vec<Expr> = pairs
        .iter()
        .map(|(k, _)| evaluate_expr_to_expr(k).unwrap_or_else(|_| k.clone()))
        .collect();
      let values: Vec<Expr> = pairs
        .iter()
        .map(|(_, v)| evaluate_expr_to_expr(v).unwrap_or_else(|_| v.clone()))
        .collect();
      let mut cursor = 0;
      if values.iter().any(is_dataset) {
        let sets = values
          .iter()
          .map(|v| match v {
            Expr::List(items) if is_dataset(v) => parse_set(items, &mut cursor),
            single => parse_set(std::slice::from_ref(single), &mut cursor),
          })
          .collect();
        Some((sets, true, Some(keys)))
      } else {
        Some((vec![parse_set(&values, &mut cursor)], false, Some(keys)))
      }
    }
    Expr::List(items) => {
      let mut cursor = 0;
      if !items.is_empty() && items.iter().all(is_dataset) {
        let sets = items
          .iter()
          .map(|set| match set {
            Expr::List(inner) => parse_set(inner, &mut cursor),
            _ => Vec::new(),
          })
          .collect();
        Some((sets, true, None))
      } else {
        Some((vec![parse_set(items, &mut cursor)], false, None))
      }
    }
    _ => None,
  }
}

/// The span the bubble radii along one axis are measured in: the spread of
/// the values, or — when they all coincide — twice their magnitude (2 at 0).
fn effective_span(values: impl Iterator<Item = f64>) -> f64 {
  let (lo, hi) = values
    .filter(|v| v.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
      (lo.min(v), hi.max(v))
    });
  if !lo.is_finite() {
    return 2.0;
  }
  let span = hi - lo;
  if span > 0.0 {
    span
  } else if lo != 0.0 {
    2.0 * lo.abs()
  } else {
    2.0
  }
}

/// A bubble's radius as a fraction of the axis span. `sizes` is the
/// `BubbleSizes -> {smin, smax}` diameter range, `None` for `Automatic`.
fn radius_fraction(
  z: f64,
  z_range: (f64, f64),
  sizes: Option<(f64, f64)>,
) -> f64 {
  let (z_min, z_max) = z_range;
  match sizes {
    None => 0.1 * (z / z_max).sqrt(),
    Some((s_min, s_max)) => {
      let (r_min, r_max) = (s_min / 2.0, s_max / 2.0);
      let (lo, hi) = (z_min.sqrt(), z_max.sqrt());
      if hi - lo <= 0.0 {
        r_max
      } else {
        r_min + (r_max - r_min) * (z.sqrt() - lo) / (hi - lo)
      }
    }
  }
}

pub fn bubble_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  const HEAD: &str = "BubbleChart";
  let data = evaluate_expr_to_expr(&args[0])?;
  let Some((sets, multi, keys)) = parse_data(&data) else {
    return Ok(invalid_dataset(HEAD, args));
  };

  let mut chart_style = None;
  let mut base_style = None;
  let mut labels = None;
  let mut legends = None;
  let mut sizes = None;
  let mut passthrough = Vec::new();
  let mut has = std::collections::HashSet::new();
  for opt in &args[1..] {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = opt
    else {
      continue;
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    let value = replacement.as_ref();
    match chart_option_alias(name) {
      "ChartStyle" => chart_style = Some(value.clone()),
      "ChartBaseStyle" => base_style = Some(value.clone()),
      "ChartLabels" => labels = Some(value.clone()),
      "ChartLegends" => legends = Some(value.clone()),
      "BubbleSizes" => {
        sizes = match value {
          Expr::List(xs) if xs.len() == 2 => {
            match (try_eval_to_f64(&xs[0]), try_eval_to_f64(&xs[1])) {
              (Some(a), Some(b)) => Some((a, b)),
              _ => None,
            }
          }
          _ => None,
        }
      }
      // Chart-only options with no graphics counterpart.
      "BubbleScale"
      | "ChartElementFunction"
      | "ChartElements"
      | "ChartLayout"
      | "LabelingFunction"
      | "LegendAppearance"
      | "PerformanceGoal" => {}
      _ => {
        has.insert(name.clone());
        passthrough.push(opt.clone());
      }
    }
  }

  // `Automatic` labels and legends come from the association keys: a
  // dataset per key is named in the legend, a point per key on its bubble.
  let is_automatic = |v: &Option<Expr>| matches!(v, Some(Expr::Identifier(a)) if a == "Automatic");
  if is_automatic(&legends) {
    legends = keys
      .clone()
      .filter(|_| multi)
      .map(|ks| Expr::List(ks.into()));
  }
  if is_automatic(&labels) {
    labels = keys.filter(|_| !multi).map(|ks| Expr::List(ks.into()));
  }
  // `Placed[labels, pos]` keeps the labels on their bubbles.
  if let Some(Expr::FunctionCall { name, args: wargs }) = &labels
    && name == "Placed"
    && let Some(inner) = wargs.first()
  {
    labels = Some(inner.clone());
  }

  // A named color scheme is sampled evenly, one color per styled element:
  // per point of a single dataset, per dataset of several.
  if let Some(Expr::String(scheme)) = &chart_style
    && let Some(controls) = named_color_scheme(scheme)
  {
    let count = if multi {
      sets.len()
    } else {
      sets.iter().map(Vec::len).sum()
    };
    let colors: Vec<Expr> = sample_gradient(controls, count.max(1))
      .into_iter()
      .map(|c| {
        call(
          "RGBColor",
          vec![Expr::Real(c.r), Expr::Real(c.g), Expr::Real(c.b)],
        )
      })
      .collect();
    chart_style = Some(Expr::List(colors.into()));
  }

  let edge = call(
    "EdgeForm",
    vec![call(
      "Directive",
      vec![
        call("GrayLevel", vec![Expr::Integer(0)]),
        call("Opacity", vec![Expr::Real(0.3)]),
      ],
    )],
  );
  // The style of the point at `index` within dataset `set`.
  let style_of = |set: usize, index: usize| {
    let color = if multi {
      chart_style_for(chart_style.as_ref(), set, true)
    } else {
      chart_style_for(chart_style.as_ref(), index, false)
    };
    call("Directive", vec![edge.clone(), color])
  };

  let all = || sets.iter().flatten();
  let span_x = effective_span(all().map(|b| b.x));
  let span_y = effective_span(all().map(|b| b.y));
  let drawn: Vec<(usize, usize, Bubble)> = sets
    .iter()
    .enumerate()
    .flat_map(|(s, set)| set.iter().enumerate().map(move |(i, b)| (s, i, *b)))
    .filter(|(_, _, b)| {
      b.x.is_finite() && b.y.is_finite() && b.z.is_finite() && b.z > 0.0
    })
    .collect();
  let z_range = drawn
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (_, _, b)| {
      (lo.min(b.z), hi.max(b.z))
    });

  let mut order: Vec<usize> = (0..drawn.len()).collect();
  order.sort_by(|&a, &b| drawn[b].2.z.total_cmp(&drawn[a].2.z));
  let mut bubbles = Vec::with_capacity(drawn.len());
  let mut x_bounds = (f64::INFINITY, f64::NEG_INFINITY);
  let mut y_bounds = (f64::INFINITY, f64::NEG_INFINITY);
  for &k in &order {
    let (s, i, b) = drawn[k];
    let f = radius_fraction(b.z, z_range, sizes);
    let (rx, ry) = (f * span_x, f * span_y);
    x_bounds = (x_bounds.0.min(b.x - rx), x_bounds.1.max(b.x + rx));
    y_bounds = (y_bounds.0.min(b.y - ry), y_bounds.1.max(b.y + ry));
    bubbles.push(Expr::List(
      vec![
        style_of(s, i),
        call("Disk", vec![pair(b.x, b.y), pair(rx, ry)]),
      ]
      .into(),
    ));
  }

  let mut content = Vec::new();
  if let Some(b) = base_style {
    content.push(b);
  }
  content.extend(bubbles);
  // Labels match the input positions, so an entry that failed to parse
  // drops its label along with its bubble. They are set in the chart's
  // own face (wolframscript's "PlotGraphics" style: Arial at the tick-label
  // size), not the serif default of a bare `Text`.
  if let Some(Expr::List(ls)) = &labels {
    let texts: Vec<Expr> = drawn
      .iter()
      .filter_map(|(_, _, b)| {
        let label = ls.get(b.pos)?;
        expr_to_label(label)?;
        Some(call("Text", vec![label.clone(), pair(b.x, b.y)]))
      })
      .collect();
    if !texts.is_empty() {
      content.push(call(
        "Style",
        vec![
          Expr::List(texts.into()),
          Expr::Real(crate::functions::plot_padding::TICK_FONT),
          rule("FontFamily", Expr::String("Arial".into())),
        ],
      ));
    }
  }

  let mut opts = passthrough;
  let mut default = |name: &str, value: Expr| {
    if !has.contains(name) {
      opts.push(rule(name, value));
    }
  };
  default("AspectRatio", Expr::Integer(1));
  default("Frame", Expr::Identifier("True".into()));
  if x_bounds.0.is_finite() && y_bounds.0.is_finite() {
    default(
      "PlotRange",
      Expr::List(
        vec![pair(x_bounds.0, x_bounds.1), pair(y_bounds.0, y_bounds.1)].into(),
      ),
    );
  }
  default("PlotRangeClipping", Expr::Identifier("True".into()));
  let scaled = || call("Scaled", vec![Expr::Real(0.05)]);
  default(
    "PlotRangePadding",
    Expr::List(
      vec![
        Expr::List(vec![scaled(), scaled()].into()),
        Expr::List(vec![scaled(), scaled()].into()),
      ]
      .into(),
    ),
  );

  let mut gargs = vec![Expr::List(content.into())];
  gargs.extend(opts);
  let graphics = call("Graphics", gargs);
  // A legend names the datasets of a multi-dataset chart, or the points of
  // a single one.
  let graphics = match &legends {
    Some(Expr::List(names)) if !names.is_empty() => {
      let swatches: Vec<Expr> = (0..names.len())
        .map(|i| {
          if multi {
            style_of(i, 0)
          } else {
            style_of(0, i)
          }
        })
        .collect();
      call(
        "Legended",
        vec![
          graphics,
          call(
            "SwatchLegend",
            vec![Expr::List(swatches.into()), Expr::List(names.clone())],
          ),
        ],
      )
    }
    _ => graphics,
  };
  evaluate_expr_to_expr(&graphics)
}
