//! `RectangleChart[{{w1, h1}, …}]` and `RectangleChart3D[{{w1, d1, h1}, …}]`.
//!
//! The chart is built as a symbolic `Graphics` (`Graphics3D`) expression of
//! `Rectangle` (`Cuboid`) primitives, so `ChartStyle`/`ChartBaseStyle`
//! directives and the ordinary graphics options pass straight through to the
//! shared renderer. Layout follows wolframscript: rectangles stand on the axis
//! from x = 0, separated by a gap of about 0.0975 (`BarSpacing -> s` sets it,
//! `None` removes it); with several datasets each starts 0.1 in and datasets
//! are 0.2 apart, and the default colours run by position within a dataset
//! (a single dataset is drawn in the first colour).

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::unevaluated;
use crate::syntax::Expr;

/// wolframscript's default chart colours, by position within a dataset.
const PALETTE: [(f64, f64, f64); 6] = [
  (1., 0.835, 0.25),
  (0., 0.665, 0.95),
  (0.375, 0.75, 0.),
  (1., 0.37, 0.1),
  (0.725, 0.45, 1.),
  (0.85, 0.51, 0.),
];

const DEFAULT_GAP: f64 = 0.0975;
const GROUP_GAP: f64 = 0.2;
const GROUP_INSET: f64 = 0.1;

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

/// One chart element: its sizes and any per-element `Style` directives.
struct Item {
  sizes: Vec<f64>,
  style: Vec<Expr>,
}

fn parse_item(e: &Expr, arity: usize) -> Option<Item> {
  match e {
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      let mut item = parse_item(&args[0], arity)?;
      item.style.extend(args[1..].iter().cloned());
      Some(item)
    }
    Expr::List(xs) if xs.len() == arity => Some(Item {
      sizes: xs.iter().map(try_eval_to_f64).collect::<Option<_>>()?,
      style: Vec::new(),
    }),
    _ => None,
  }
}

/// A single dataset `{{w, h}, …}` or a list of datasets.
/// A single dataset `{{w, h}, …}` or a list of datasets; entries that are no
/// rectangle are skipped, as in wolframscript. `None` when nothing usable is
/// left.
fn parse_datasets(data: &Expr, arity: usize) -> Option<Vec<Vec<Item>>> {
  let Expr::List(items) = data else {
    return None;
  };
  let set: Vec<Item> =
    items.iter().filter_map(|e| parse_item(e, arity)).collect();
  if !set.is_empty() {
    return Some(vec![set]);
  }
  let sets: Vec<Vec<Item>> = items
    .iter()
    .filter_map(|set| match set {
      Expr::List(xs) => {
        let set: Vec<Item> =
          xs.iter().filter_map(|e| parse_item(e, arity)).collect();
        (!set.is_empty()).then_some(set)
      }
      _ => None,
    })
    .collect();
  (!sets.is_empty()).then_some(sets)
}

fn palette_color(i: usize) -> Expr {
  let (r, g, b) = PALETTE[i % PALETTE.len()];
  call(
    "RGBColor",
    vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
  )
}

/// The directive for the element at `pos` within its dataset, from a
/// `ChartStyle` value: a single directive applies to all elements, a list
/// cycles over positions, and a list entry that is itself a list combines
/// its directives.
fn chart_style_for(style: Option<&Expr>, pos: usize, multi: bool) -> Expr {
  let as_directive = |e: &Expr| match e {
    Expr::List(ds) => call("Directive", ds.to_vec()),
    other => other.clone(),
  };
  match style {
    // The default colours only vary across datasets, by position; a single
    // dataset is drawn in the first colour throughout.
    None | Some(Expr::Identifier(_) | Expr::String(_)) => {
      palette_color(if multi { pos } else { 0 })
    }
    Some(Expr::List(xs)) if !xs.is_empty() => as_directive(&xs[pos % xs.len()]),
    Some(other) => other.clone(),
  }
}

pub fn rectangle_chart_ast(
  args: &[Expr],
  three_d: bool,
) -> Result<Expr, InterpreterError> {
  let head = if three_d {
    "RectangleChart3D"
  } else {
    "RectangleChart"
  };
  let arity = if three_d { 3 } else { 2 };
  let Some(data) = args.first() else {
    return Ok(unevaluated(head, args));
  };
  let Some(sets) = parse_datasets(data, arity) else {
    return Ok(crate::functions::chart::invalid_dataset(head, args));
  };

  let mut chart_style = None;
  let mut base_style = None;
  let mut labels = None;
  let mut legends = None;
  let mut gap = DEFAULT_GAP;
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
      return Ok(unevaluated(head, args));
    };
    let Expr::Identifier(name) = pattern.as_ref() else {
      continue;
    };
    let value = replacement.as_ref();
    match crate::functions::chart::chart_option_alias(name) {
      "ChartStyle" => chart_style = Some(value.clone()),
      "ChartBaseStyle" => base_style = Some(value.clone()),
      "ChartLabels" => labels = Some(value.clone()),
      "ChartLegends" => legends = Some(value.clone()),
      "BarSpacing" => {
        gap = match value {
          Expr::Identifier(s) if s == "None" => 0.0,
          Expr::List(xs) => xs.first().and_then(try_eval_to_f64).unwrap_or(gap),
          v => try_eval_to_f64(v).unwrap_or(gap),
        }
      }
      // Chart-only options with no graphics counterpart.
      "ChartElementFunction"
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

  // `Placed[labels, pos]` / `Callout[labels, pos]` wrap the label list;
  // `Above` / `Top` put the labels over the bars instead of under them.
  let mut labels_above = false;
  if let Some(Expr::FunctionCall { name, args: wargs }) = &labels
    && (name == "Placed" || name == "Callout")
    && let Some(inner) = wargs.first()
  {
    labels_above = name == "Callout"
      || matches!(wargs.get(1), Some(Expr::Identifier(p)) if p == "Above" || p == "Top");
    labels = Some(inner.clone());
  }
  let n_sets = sets.len();
  let multi = n_sets > 1;
  let mut x = if multi { GROUP_INSET } else { 0.0 };
  let mut prims = Vec::new();
  let mut label_prims = Vec::new();
  let mut flat_index = 0usize;
  let mut swatches = Vec::new();
  for (s, set) in sets.iter().enumerate() {
    if s > 0 {
      x += GROUP_GAP - gap;
    }
    for (pos, item) in set.iter().enumerate() {
      let w = item.sizes[0];
      let style = chart_style_for(chart_style.as_ref(), pos, multi);
      if s == 0 || !multi {
        swatches.push(style.clone());
      }
      let shape = if three_d {
        let (d, h) = (item.sizes[1], item.sizes[2]);
        call(
          "Cuboid",
          vec![
            Expr::List(
              vec![Expr::Real(x), Expr::Real(0.0), Expr::Real(0.0)].into(),
            ),
            Expr::List(
              vec![Expr::Real(x + w), Expr::Real(d), Expr::Real(h)].into(),
            ),
          ],
        )
      } else {
        let h = item.sizes[1];
        call(
          "Rectangle",
          vec![
            Expr::List(vec![Expr::Real(x), Expr::Real(0.0)].into()),
            Expr::List(vec![Expr::Real(x + w), Expr::Real(h)].into()),
          ],
        )
      };
      let mut group = vec![style];
      group.extend(item.style.iter().cloned());
      group.push(shape);
      prims.push(Expr::List(group.into()));
      if !three_d
        && let Some(Expr::List(ls)) = &labels
        && let Some(label) = ls.get(flat_index)
      {
        let (y, offset) = if labels_above {
          (item.sizes[1], -1.5)
        } else {
          (0.0, 1.5)
        };
        label_prims.push(call(
          "Text",
          vec![
            label.clone(),
            Expr::List(vec![Expr::Real(x + w / 2.0), Expr::Real(y)].into()),
            Expr::List(vec![Expr::Integer(0), Expr::Real(offset)].into()),
          ],
        ));
      }
      flat_index += 1;
      x += w + gap;
    }
  }

  let edge = call(
    "EdgeForm",
    vec![call(
      "Directive",
      vec![
        call("Thickness", vec![Expr::Identifier("Small".into())]),
        call("Opacity", vec![Expr::Real(0.686)]),
      ],
    )],
  );
  let mut content = vec![edge];
  if let Some(b) = base_style {
    content.push(b);
  }
  content.extend(prims);
  if !label_prims.is_empty() {
    content.push(Expr::List(
      std::iter::once(call("GrayLevel", vec![Expr::Integer(0)]))
        .chain(label_prims)
        .collect(),
    ));
  }

  let mut opts = passthrough.clone();
  if three_d {
    if !has.contains("Axes") {
      opts.push(rule("Axes", Expr::Identifier("True".into())));
    }
    if !has.contains("Boxed") {
      opts.push(rule("Boxed", Expr::Identifier("False".into())));
    }
  } else {
    // A frame replaces the value axis, and its bottom edge is labelled by
    // the `ChartLabels` rather than by numeric ticks.
    let framed = passthrough.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Frame")
          && !matches!(replacement.as_ref(),
            Expr::Identifier(v) if v == "False" || v == "None"))
    });
    if !has.contains("Axes") {
      let flag =
        |on: bool| Expr::Identifier(if on { "True" } else { "False" }.into());
      opts.push(rule(
        "Axes",
        if framed {
          flag(false)
        } else {
          Expr::List(vec![flag(false), flag(true)].into())
        },
      ));
    }
    if framed && !has.contains("FrameTicks") {
      let ident = |s: &str| Expr::Identifier(s.into());
      opts.push(rule(
        "FrameTicks",
        Expr::List(
          vec![
            Expr::List(vec![ident("Automatic"), ident("Automatic")].into()),
            Expr::List(vec![ident("None"), ident("None")].into()),
          ]
          .into(),
        ),
      ));
    }
    // wolframscript pads the range and puts the value axis just left of the
    // first rectangle (2% of the chart width).
    if !has.contains("PlotRangePadding") {
      let scaled = |v: f64| call("Scaled", vec![Expr::Real(v)]);
      opts.push(rule(
        "PlotRangePadding",
        Expr::List(
          vec![
            Expr::List(vec![scaled(0.02), scaled(0.02)].into()),
            Expr::List(vec![scaled(0.02), scaled(0.05)].into()),
          ]
          .into(),
        ),
      ));
    }
    if !has.contains("AxesOrigin") {
      let width = x - gap;
      opts.push(rule(
        "AxesOrigin",
        Expr::List(vec![Expr::Real(-0.02 * width), Expr::Real(0.0)].into()),
      ));
    }
    if !has.contains("AspectRatio") {
      opts.push(rule(
        "AspectRatio",
        call(
          "Power",
          vec![Expr::Identifier("GoldenRatio".into()), Expr::Integer(-1)],
        ),
      ));
    }
  }
  let mut gargs = vec![Expr::List(content.into())];
  gargs.extend(opts);
  let graphics = call(if three_d { "Graphics3D" } else { "Graphics" }, gargs);
  let graphics = match &legends {
    Some(Expr::List(names)) if !names.is_empty() => call(
      "Legended",
      vec![
        graphics,
        call(
          "SwatchLegend",
          vec![
            Expr::List(swatches.into_iter().take(names.len()).collect()),
            Expr::List(names.clone()),
          ],
        ),
      ],
    ),
    _ => graphics,
  };
  evaluate_expr_to_expr(&graphics)
}

/// `` BarCharts`GeneralizedBarChart[{{x1, h1, w1}, …}] `` — the legacy
/// package chart: a bar of height `h` and width `w` centred at `x`, drawn
/// as plain `Rectangle` primitives in a `Graphics` with axes. `` BarCharts`BarStyle ``
/// holds the directive applied to every bar; other options pass straight
/// through to `Graphics`.
pub fn generalized_bar_chart_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  const HEAD: &str = "BarCharts`GeneralizedBarChart";
  let Some(Expr::List(bars)) = args.first() else {
    return Ok(unevaluated(HEAD, args));
  };
  let mut rects = Vec::new();
  for bar in bars {
    let Some(Item { sizes, .. }) = parse_item(bar, 3) else {
      return Ok(unevaluated(HEAD, args));
    };
    let (x, h, w) = (sizes[0], sizes[1], sizes[2]);
    let pt =
      |a: f64, b: f64| Expr::List(vec![Expr::Real(a), Expr::Real(b)].into());
    rects.push(call(
      "Rectangle",
      vec![pt(x - w / 2.0, 0.0), pt(x + w / 2.0, h)],
    ));
  }
  let mut style = None;
  let mut options = Vec::new();
  let mut has_axes = false;
  for opt in &args[1..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return Ok(unevaluated(HEAD, args));
    };
    match pattern.as_ref() {
      Expr::Identifier(n) if n == "BarCharts`BarStyle" || n == "BarStyle" => {
        style = Some(replacement.as_ref().clone());
      }
      Expr::Identifier(n) => {
        has_axes |= n == "Axes";
        options.push(opt.clone());
      }
      _ => options.push(opt.clone()),
    }
  }
  if !has_axes {
    options.push(rule("Axes", Expr::Identifier("True".into())));
  }
  let mut prims = Vec::new();
  prims.extend(style);
  prims.extend(rects);
  let mut graphics_args = vec![Expr::List(prims.into())];
  graphics_args.extend(options);
  evaluate_expr_to_expr(&call("Graphics", graphics_args))
}
