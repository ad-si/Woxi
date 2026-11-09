//! `RectangleChart[{{w1, h1}, …}]` and `RectangleChart3D[{{w1, d1, h1}, …}]`.
//!
//! The chart is built as a symbolic `Graphics` (`Graphics3D`) expression of
//! `Rectangle` (`Cuboid`) primitives, so `ChartStyle`/`ChartBaseStyle`
//! directives and the ordinary graphics options pass straight through to the
//! shared renderer. Layout follows wolframscript: rectangles stand on the axis
//! from x = 0, separated by a gap of 0.1 - 0.001 n for n rectangles in all
//! (`BarSpacing -> s` sets it, `None` removes it); with several datasets each
//! starts 0.1 in and datasets are 0.2 apart, and the default colors run by
//! position within a dataset (a single dataset is drawn in the first color).
//! Each 2D bar is wrapped the way wolframscript's chart holds it:
//! `{style, Tooltip[StatusArea[Charting`DelayedMouseEffect[Rectangle[…,
//! "RoundingRadius" -> 0], effect], {w, h}], Style[{w, h}, {}]]}`.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::unevaluated;
use crate::syntax::Expr;

/// wolframscript's default chart colors, by position within a dataset.
const PALETTE: [(f64, f64, f64); 15] =
  crate::functions::chart::CHART_COLORS_RGB;

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

/// A bar size as the chart's tooltip writes it: the number given.
fn sizes_expr(v: f64) -> Expr {
  if v.fract() == 0.0 && v.abs() < 1e15 {
    Expr::Integer(v as i128)
  } else {
    Expr::Real(v)
  }
}

pub(crate) fn palette_color(i: usize) -> Expr {
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
pub(crate) fn chart_style_for(
  style: Option<&Expr>,
  pos: usize,
  multi: bool,
) -> Expr {
  let as_directive = |e: &Expr| match e {
    Expr::List(ds) => call("Directive", ds.to_vec()),
    other => other.clone(),
  };
  match style {
    // The default colors only vary across datasets, by position; a single
    // dataset is drawn in the first color throughout.
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
  let n_bars: usize = sets.iter().map(Vec::len).sum();
  let mut gap = 0.1 - 0.001 * n_bars as f64;
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
  let mut callout = false;
  if let Some(Expr::FunctionCall { name, args: wargs }) = &labels
    && (name == "Placed" || name == "Callout")
    && let Some(inner) = wargs.first()
  {
    callout = name == "Callout";
    labels_above = matches!(wargs.get(1), Some(Expr::Identifier(p)) if p == "Above" || p == "Top")
      || (callout && wargs.len() == 1);
    labels = Some(inner.clone());
  }
  let n_sets = sets.len();
  let multi = n_sets > 1;
  // A named color scheme (`ChartStyle -> "Rainbow"`) is sampled evenly,
  // one color per element of a dataset — the list of them then cycles
  // like any explicit `ChartStyle` list.
  if let Some(Expr::String(scheme)) = &chart_style
    && let Some(controls) = crate::functions::chart::named_color_scheme(scheme)
  {
    let count = sets.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let colors: Vec<Expr> =
      crate::functions::chart::sample_gradient(controls, count)
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
  // The bars' edges fade a little with every bar.
  let edge = call(
    "EdgeForm",
    vec![call(
      "Directive",
      vec![
        call("Thickness", vec![Expr::Identifier("Small".into())]),
        call("Opacity", vec![Expr::Real(0.7 - 0.007 * n_bars as f64)]),
      ],
    )],
  );
  let max_depth = sets
    .iter()
    .flatten()
    .map(|item| item.sizes.get(1).copied().unwrap_or(0.0))
    .fold(0.0_f64, f64::max);
  let max_height = sets
    .iter()
    .flatten()
    .map(|item| item.sizes.get(2).copied().unwrap_or(0.0))
    .fold(0.0_f64, f64::max);
  let mut x = if multi { GROUP_INSET } else { 0.0 };
  let mut end = x;
  let mut prims = Vec::new();
  let mut label_prims = Vec::new();
  let mut flat_index = 0usize;
  let mut swatches = Vec::new();
  // Where each block starts and ends along x.
  let (mut starts, mut ends) = (Vec::new(), Vec::new());
  let mut group_ends = Vec::new();
  for (s, set) in sets.iter().enumerate() {
    // A dataset starts 0.2 past the end of the one before.
    if s > 0 {
      x = end + GROUP_GAP;
    }
    group_ends.push(x);
    for (pos, item) in set.iter().enumerate() {
      let w = item.sizes[0];
      let style = chart_style_for(chart_style.as_ref(), pos, multi);
      if s == 0 || !multi {
        swatches.push(style.clone());
      }
      let shape = if three_d {
        // A block flush with the back of the deepest one, shaded as
        // wolframscript shades a 3D chart's bars.
        let (d, h) = (item.sizes[1], item.sizes[2]);
        let color = match chart_style.as_ref() {
          None | Some(Expr::Identifier(_) | Expr::String(_)) => {
            crate::functions::chart3d::chart3d_color(if multi {
              pos
            } else {
              0
            })
          }
          Some(_) => crate::functions::graphics::parse_color(&style)
            .map_or_else(
              || crate::functions::chart3d::chart3d_color(0),
              |c| (c.r, c.g, c.b),
            ),
        };
        crate::functions::chart3d::chart3d_block(
          [x, max_depth - d, 0.0],
          [x + w, max_depth, h],
          color,
        )
      } else {
        let h = item.sizes[1];
        call(
          "Rectangle",
          vec![
            Expr::List(vec![Expr::Real(x), Expr::Real(0.0)].into()),
            Expr::List(vec![Expr::Real(x + w), Expr::Real(h)].into()),
            Expr::Rule {
              pattern: Box::new(Expr::String("RoundingRadius".into())),
              replacement: Box::new(Expr::Integer(0)),
            },
          ],
        )
      };
      let mut group = if three_d { vec![] } else { vec![style] };
      group.extend(item.style.iter().cloned());
      if three_d {
        group.push(shape);
      } else {
        group = vec![call("Directive", {
          let mut d = vec![edge.clone()];
          d.extend(group);
          d
        })];
        let size = Expr::List(
          item
            .sizes
            .iter()
            .map(|v| sizes_expr(*v))
            .collect::<Vec<_>>()
            .into(),
        );
        group.push(call(
          "Tooltip",
          vec![
            call(
              "StatusArea",
              vec![
                call(
                  "Charting`DelayedMouseEffect",
                  vec![
                    shape,
                    Expr::List(
                      vec![
                        Expr::String("Style".into()),
                        call(
                          "EdgeForm",
                          vec![Expr::List(
                            vec![
                              call("GrayLevel", vec![Expr::Real(0.5)]),
                              call("AbsoluteThickness", vec![Expr::Real(1.5)]),
                              call("Opacity", vec![Expr::Real(0.66)]),
                            ]
                            .into(),
                          )],
                        ),
                      ]
                      .into(),
                    ),
                  ],
                ),
                size.clone(),
              ],
            ),
            call("Style", vec![size, Expr::List(vec![].into())]),
          ],
        ));
      }
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
        if callout {
          label_prims.extend(callout_label(
            label,
            (x + (x + w)) / 2.0,
            y,
            labels_above,
          ));
          flat_index += 1;
          end = x + w;
          x = x + w + gap;
          continue;
        }
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
      starts.push(x);
      ends.push(x + w);
      end = x + w;
      x = x + w + gap;
    }
    group_ends.push(end);
  }

  let mut content = if three_d { vec![edge.clone()] } else { vec![] };
  if three_d {
    // The invisible base plate that sets the scene's extent.
    let (x1, z1) = (end, max_height);
    content.push(Expr::List(
      vec![
        call("Opacity", vec![Expr::Integer(0)]),
        call("EdgeForm", vec![]),
        call(
          "Cuboid",
          vec![
            Expr::List(
              vec![
                Expr::Real(-0.05 * x1),
                Expr::Real(-0.25 * max_depth),
                Expr::Integer(0),
              ]
              .into(),
            ),
            Expr::List(
              vec![
                Expr::Real(x1 + 0.05 * x1),
                Expr::Real(max_depth + 0.25 * max_depth),
                Expr::Real(0.008 * z1),
              ]
              .into(),
            ),
          ],
        ),
      ]
      .into(),
    ));
  }
  if let Some(b) = base_style {
    content.push(b);
  }
  let framed = passthrough.iter().any(|o| {
    matches!(o, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Frame")
        && !matches!(replacement.as_ref(), Expr::Identifier(v) if v == "False" || v == "None"))
  });
  let axes_disabled = passthrough.iter().any(|o| {
    let off = |v: &Expr| matches!(v, Expr::Identifier(s) if s == "False" || s == "None");
    matches!(o, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Axes")
        && (off(replacement) || matches!(replacement.as_ref(), Expr::List(items) if items.iter().all(off))))
  });
  if !three_d && !framed && !axes_disabled {
    // The category baseline is a chart primitive, even though Axes enables
    // only the value axis. Its end marks are six printer points long.
    let point =
      |x: f64| Expr::List(vec![Expr::Real(x), Expr::Integer(0)].into());
    let start = group_ends[0];
    let width = end - start;
    let mut baseline = vec![
      call("GrayLevel", vec![Expr::Real(0.5)]),
      call("AbsoluteThickness", vec![Expr::Real(0.5)]),
      call(
        "Line",
        vec![Expr::List(
          vec![point(start - 0.02 * width), point(end + 0.0204 * width)].into(),
        )],
      ),
    ];
    for x in group_ends {
      baseline.push(call(
        "Line",
        vec![Expr::List(
          vec![
            point(x),
            call(
              "Offset",
              vec![
                Expr::List(vec![Expr::Integer(0), Expr::Integer(-6)].into()),
                point(x),
              ],
            ),
          ]
          .into(),
        )],
      ));
    }
    content.push(Expr::List(baseline.into()));
  }
  content.extend(prims);
  if !label_prims.is_empty() {
    content.push(call(
      "Style",
      vec![
        Expr::List(label_prims.into()),
        Expr::Real(crate::functions::plot_padding::TICK_FONT),
        call("GrayLevel", vec![Expr::Real(1.0 / 3.0)]),
      ],
    ));
  }

  let mut opts = passthrough.clone();
  if three_d {
    // Wolfram's 3D chart frame: seen almost from the front, the value
    // axis on the right, the blocks' extents as unlabeled ticks below.
    let group_of: Vec<usize> = sets
      .iter()
      .enumerate()
      .flat_map(|(s, set)| std::iter::repeat_n(s, set.len()))
      .collect();
    let mut ticks = Vec::new();
    if let (Some(&first), Some(&last)) = (starts.first(), ends.last()) {
      ticks.push((first, 0.005));
      ticks.push((last, 0.005));
      for i in 1..starts.len() {
        if group_of[i] == group_of[i - 1] {
          ticks.push((f64::midpoint(ends[i - 1], starts[i]), 0.005));
        }
      }
      ticks.push((first, 0.01));
      ticks.push((last, 0.01));
    }
    for o in crate::functions::chart3d::bar_chart3d_options(
      &ticks, end, max_depth, true, true,
    ) {
      let named = match &o {
        Expr::Rule { pattern, .. } => match pattern.as_ref() {
          Expr::Identifier(n) => n.clone(),
          _ => String::new(),
        },
        _ => String::new(),
      };
      if !has.contains(&named) {
        opts.push(o);
      }
    }
  } else {
    // A frame replaces the value axis, and its bottom edge is labeled by
    // the `ChartLabels` rather than by numeric ticks.
    if !has.contains("Axes") {
      let flag =
        |on: bool| Expr::Identifier(if on { "True" } else { "False" }.into());
      opts.push(rule(
        "Axes",
        Expr::List(vec![flag(false), flag(!framed)].into()),
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
/// package chart, once the package is loaded: a bar of height `h` and width
/// `w` centered at `x`, each drawn as `{style, EdgeForm[Opacity[0.5]],
/// Rectangle[…]}` in the bars' own arithmetic, in a `Graphics` with axes.
/// `` BarCharts`BarStyle `` gives the style (the package's blue
/// `Hue[0.67, 0.45, 0.65]` by default); other options pass through to
/// `Graphics`.
pub fn generalized_bar_chart_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  const HEAD: &str = "BarCharts`GeneralizedBarChart";
  if !crate::packages_list().iter().any(|p| p == "BarCharts`") {
    return Ok(unevaluated(HEAD, args));
  }
  let Some(Expr::List(bars)) = args.first() else {
    return Ok(unevaluated(HEAD, args));
  };
  let mut style = call(
    "Hue",
    vec![Expr::Real(0.67), Expr::Real(0.45), Expr::Real(0.65)],
  );
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
        style = replacement.as_ref().clone();
      }
      Expr::Identifier(n) => {
        has_axes |= n == "Axes";
        options.push(opt.clone());
      }
      _ => options.push(opt.clone()),
    }
  }
  let mut prims = Vec::new();
  for bar in bars {
    let Expr::List(xhw) = bar else {
      return Ok(unevaluated(HEAD, args));
    };
    let [x, h, w] = xhw.as_slice() else {
      return Ok(unevaluated(HEAD, args));
    };
    if [x, h, w]
      .iter()
      .any(|v| crate::functions::math_ast::try_eval_to_f64(v).is_none())
    {
      return Ok(unevaluated(HEAD, args));
    }
    let half = crate::helpers::times2(
      w.clone(),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    );
    let left =
      evaluate_expr_to_expr(&crate::helpers::minus2(x.clone(), half.clone()))?;
    let right = evaluate_expr_to_expr(&crate::helpers::plus2(x.clone(), half))?;
    let pt = |a: Expr, b: Expr| Expr::List(vec![a, b].into());
    prims.push(Expr::List(
      vec![
        style.clone(),
        call("EdgeForm", vec![call("Opacity", vec![Expr::Real(0.5)])]),
        call(
          "Rectangle",
          vec![pt(left, Expr::Integer(0)), pt(right, h.clone())],
        ),
      ]
      .into(),
    ));
  }
  if !has_axes {
    options.push(rule("Axes", Expr::Identifier("True".into())));
  }
  let mut graphics_args = vec![Expr::List(prims.into())];
  graphics_args.extend(options);
  evaluate_expr_to_expr(&call("Graphics", graphics_args))
}

/// A `Callout` label as wolframscript places it: the label, unrotated and
/// opaque, a fixed number of printer points past the bar's end (19, or 20
/// when the text rises above the x-height), with a leader curve from the
/// bar to the label.
fn callout_label(label: &Expr, xc: f64, y: f64, above: bool) -> Vec<Expr> {
  let angle = if above {
    std::f64::consts::FRAC_PI_2
  } else {
    3.0 * std::f64::consts::FRAC_PI_2
  };
  let sign = if above { 1.0 } else { -1.0 };
  let tall = match label {
    Expr::String(t) => t.chars().any(|c| {
      c.is_ascii_uppercase() || c.is_ascii_digit() || "bdfhklt".contains(c)
    }),
    _ => true,
  };
  let distance = if tall { 20.0 } else { 19.0 };
  let anchor = Expr::List(vec![Expr::Real(xc), Expr::Real(y)].into());
  let offset = |dx: f64, dy: f64| {
    call(
      "Offset",
      vec![
        Expr::List(vec![Expr::Real(dx), Expr::Real(dy)].into()),
        anchor.clone(),
      ],
    )
  };
  let text = call(
    "Text",
    vec![
      call(
        "Style",
        vec![
          call("Rotate", vec![label.clone(), Expr::Integer(0)]),
          Expr::List(
            vec![
              call("Directive", vec![]),
              call("Opacity", vec![Expr::Integer(1)]),
            ]
            .into(),
          ),
        ],
      ),
      offset(angle.cos() * 10.0, sign * distance),
      Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
    ],
  );
  let leader = call(
    "BSplineCurve",
    vec![Expr::List(
      vec![
        offset(angle.cos() * 3.0, sign * 3.0),
        offset(angle.cos() * 10.0, sign * 10.0),
        offset(angle.cos() * 10.0, sign * 12.0),
        offset(angle.cos() * 10.0, sign * 14.0),
        offset(angle.cos() * 10.0, sign * 14.0),
      ]
      .into(),
    )],
  );
  vec![leader, text]
}
