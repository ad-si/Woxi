//! The legacy `PlotLegends`` package's `ShowLegend`, once loaded.
//!
//! `ShowLegend[graphic, legend1, legend2, …]` places one or more boxed
//! legends over (or beside) a graphic. The package lays everything out in a
//! frame where the graphic is centered at the origin and its longer side
//! runs from -1 to 1; `LegendPosition` names a legend's lower-left corner
//! and `LegendSize` its width and height in that same frame, so a legend
//! may sit inside the graphic or stick out past its edge (the picture then
//! grows to hold it).
//!
//! Each legend is either a list of `{icon, label}` items (an icon is any
//! graphic, drawn in a key box to the left of its label) or a color
//! gradient `{colorFunction, n, minLabel, maxLabel}`, both followed by
//! options: `LegendPosition` (default `{-1, -1}`), `LegendSize` (`0.8`),
//! `LegendShadow` (`{0.05, -0.05}`, or `None`), `LegendBorder` (a thin
//! black frame, or `False`/`None`/a color), `LegendBackground` (white),
//! `LegendLabel` (a title above the rows, in a strip `LegendLabelSpace`
//! rows tall) and `LegendTextSpace` (the width of the labels as a multiple
//! of the key width).
//!
//! A gradient legend lists the function's values from 0 at the top to 1 at
//! the bottom, in a column of touching swatches: `minLabel` names the top
//! swatch and `maxLabel` the bottom one.

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;

/// The name of an option rule's left-hand side without its context, so
/// `PlotLegends`LegendSize` and a bare `LegendSize` read the same.
fn option_short_name(pattern: &Expr) -> Option<&str> {
  match pattern {
    Expr::Identifier(name) => Some(name.rsplit('`').next().unwrap_or(name)),
    _ => None,
  }
}

fn real(x: f64) -> Expr {
  Expr::Real(x)
}

fn point(x: f64, y: f64) -> Expr {
  Expr::List(vec![real(x), real(y)].into())
}

fn pair(e: &Expr) -> Option<(f64, f64)> {
  match e {
    Expr::List(items) if items.len() == 2 => {
      Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?))
    }
    _ => None,
  }
}

fn is_none_like(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "None" || s == "False")
}

/// One legend's settings, in the package's frame.
struct LegendSpec {
  /// `{icon, label}` rows, top to bottom.
  items: Vec<(Expr, Expr)>,
  position: (f64, f64),
  size: (f64, f64),
  shadow: Option<(f64, f64)>,
  /// The border's directive, `None` for no border.
  border: Option<Expr>,
  background: Expr,
  /// The rows are color swatches of a gradient, drawn touching.
  gradient: bool,
  /// `LegendLabel`: a title above the rows.
  label: Option<Expr>,
  /// `LegendLabelSpace`: the height of the title strip, in rows.
  label_space: Option<f64>,
  /// `LegendTextSpace`: the width of the labels over that of the keys.
  text_space: Option<f64>,
}

impl LegendSpec {
  /// The legend's corners, its shadow included.
  fn bounds(&self) -> ((f64, f64), (f64, f64)) {
    let (x0, y0) = self.position;
    let (x1, y1) = (x0 + self.size.0, y0 + self.size.1);
    let (dx, dy) = self.shadow.unwrap_or((0.0, 0.0));
    (
      (x0.min(x0 + dx), y0.min(y0 + dy)),
      (x1.max(x1 + dx), y1.max(y1 + dy)),
    )
  }

  /// The legend's primitives: shadow, background box, border, and each
  /// row's icon (fitted into a key box) followed by its label.
  fn primitives(&self) -> Vec<Expr> {
    let (x0, y0) = self.position;
    let (w, h) = self.size;
    let rect = |ax: f64, ay: f64| {
      call("Rectangle", vec![point(ax, ay), point(ax + w, ay + h)])
    };
    let mut out = Vec::new();
    if let Some((dx, dy)) = self.shadow {
      out.push(Expr::List(
        vec![call("GrayLevel", vec![real(0.0)]), rect(x0 + dx, y0 + dy)].into(),
      ));
    }
    let edge = match &self.border {
      Some(directive) => call("EdgeForm", vec![directive.clone()]),
      None => call("EdgeForm", vec![Expr::Identifier("None".into())]),
    };
    out.push(Expr::List(
      vec![self.background.clone(), edge, rect(x0, y0)].into(),
    ));
    let n = self.items.len().max(1) as f64;
    // The title's strip takes `label_space` rows (one by default) above
    // the items.
    let label_rows = if self.label.is_some() {
      self.label_space.unwrap_or(1.0).max(0.0)
    } else {
      0.0
    };
    // A gradient's column keeps a margin of half a row all around.
    let units = if self.gradient {
      n + label_rows + 1.0
    } else {
      n + label_rows
    };
    let row_h = h / units;
    if let Some(label) = &self.label {
      let strip_top = if self.gradient { 0.5 * row_h } else { 0.0 };
      out.push(call(
        "Text",
        vec![
          label.clone(),
          point(x0 + w / 2.0, y0 + h - strip_top - label_rows * row_h / 2.0),
        ],
      ));
    }
    let rows_top = y0 + h - label_rows * row_h;
    // A square key box filling most of its row, never wider than a third
    // of the legend so the labels keep their room; `LegendTextSpace -> t`
    // makes the labels `t` key widths wide instead.
    let (key_w, key_h, margin, rows_top) = if self.gradient {
      let margin = 0.5 * row_h;
      let key_w = match self.text_space {
        Some(t) if t >= 0.0 => (w - 2.0 * margin) / (1.0 + t),
        _ => row_h.min(w / 3.0),
      };
      (key_w, row_h, margin, rows_top - margin)
    } else {
      let key = match self.text_space {
        Some(t) if t >= 0.0 => (0.8 * row_h).min(w / (1.0 + t)),
        _ => (0.8 * row_h).min(w / 3.0),
      };
      (key, key, (row_h - key) / 2.0, rows_top)
    };
    for (i, (icon, label)) in self.items.iter().enumerate() {
      let cy = rows_top - (i as f64 + 0.5) * row_h;
      let kx = x0 + margin + key_w / 2.0;
      out.push(call(
        "Inset",
        vec![
          icon.clone(),
          point(kx, cy),
          Expr::Identifier("Center".into()),
          point(key_w, key_h),
        ],
      ));
      let is_empty_label = matches!(label, Expr::String(s) if s.is_empty());
      if !is_empty_label {
        out.push(call(
          "Text",
          vec![
            label.clone(),
            point(x0 + margin + key_w + (0.25 * margin).max(0.02 * w), cy),
            point(-1.0, 0.0),
          ],
        ));
      }
    }
    out
  }
}

/// Read one legend argument: `{{{icon, label}, …}, opts…}` or
/// `{colorFunction, n, minLabel, maxLabel, opts…}`.
fn parse_legend(arg: &Expr) -> Option<LegendSpec> {
  let Expr::List(parts) = arg else {
    return None;
  };
  let is_option =
    |e: &Expr| matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. });
  let content: Vec<&Expr> =
    parts.iter().take_while(|e| !is_option(e)).collect();
  let options = &parts[content.len()..];

  let items: Vec<(Expr, Expr)> = match content.as_slice() {
    [Expr::List(rows)] => rows
      .iter()
      .map(|row| match row {
        Expr::List(cells) if cells.len() == 2 => {
          Some((cells[0].clone(), cells[1].clone()))
        }
        _ => None,
      })
      .collect::<Option<_>>()?,
    [function, n, min_label, max_label] => {
      let n = match evaluate_expr_to_expr(n).ok()? {
        Expr::Integer(n) if n >= 2 => n as usize,
        _ => return None,
      };
      // The top row shows the function's value at 0, the bottom row at 1.
      (0..n)
        .map(|k| {
          let t = k as f64 / (n - 1) as f64;
          let color =
            evaluate_expr_to_expr(&call_value(function, real(t))).ok()?;
          let swatch = evaluate_expr_to_expr(&call(
            "Graphics",
            vec![
              Expr::List(vec![color, call("Rectangle", vec![])].into()),
              Expr::Rule {
                pattern: Box::new(Expr::Identifier("AspectRatio".into())),
                replacement: Box::new(Expr::Identifier("Full".into())),
              },
              Expr::Rule {
                pattern: Box::new(Expr::Identifier("PlotRangePadding".into())),
                replacement: Box::new(Expr::Identifier("None".into())),
              },
            ],
          ))
          .ok()?;
          let label = if k == 0 {
            (*min_label).clone()
          } else if k == n - 1 {
            (*max_label).clone()
          } else {
            Expr::String(String::new())
          };
          Some((swatch, label))
        })
        .collect::<Option<_>>()?
    }
    _ => return None,
  };
  if items.is_empty() {
    return None;
  }
  let gradient = !matches!(content.as_slice(), [Expr::List(_)]);

  let mut spec = LegendSpec {
    items,
    position: (-1.0, -1.0),
    size: (0.8, 0.8),
    shadow: Some((0.05, -0.05)),
    border: Some(call("GrayLevel", vec![real(0.0)])),
    background: call("GrayLevel", vec![real(1.0)]),
    gradient,
    label: None,
    label_space: None,
    text_space: None,
  };
  for option in options {
    let (Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    }) = option
    else {
      continue;
    };
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| (**replacement).clone());
    match option_short_name(pattern) {
      Some("LegendPosition") => {
        if let Some(p) = pair(&value) {
          spec.position = p;
        }
      }
      Some("LegendSize") => {
        if let Some(s) = pair(&value) {
          spec.size = s;
        } else if let Some(s) = try_eval_to_f64(&value) {
          spec.size = (s, s);
        }
      }
      Some("LegendShadow") => {
        spec.shadow = if is_none_like(&value) {
          None
        } else {
          pair(&value).or(spec.shadow)
        };
      }
      Some("LegendBorder") => {
        spec.border = if is_none_like(&value) {
          None
        } else if matches!(&value, Expr::Identifier(s) if s == "True" || s == "Automatic")
        {
          spec.border
        } else {
          Some(value)
        };
      }
      Some("LegendLabel") => {
        spec.label = (!is_none_like(&value)).then_some(value);
      }
      Some("LegendLabelSpace") => {
        spec.label_space = try_eval_to_f64(&value);
      }
      Some("LegendTextSpace") => {
        spec.text_space = try_eval_to_f64(&value);
      }
      Some("LegendBackground") => {
        spec.background = if is_none_like(&value) {
          call("Opacity", vec![real(0.0)])
        } else {
          value
        };
      }
      _ => {}
    }
  }
  Some(spec)
}

/// `f[t]` for a function given as a symbol or a pure function.
fn call_value(function: &Expr, t: Expr) -> Expr {
  match function {
    Expr::Identifier(name) => call(name, vec![t]),
    other => Expr::CurriedCall {
      func: Box::new(other.clone()),
      args: vec![t],
    },
  }
}

/// `PlotLegends`ShowLegend[graphic, legend…]`.
pub fn show_legend_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("PlotLegends`ShowLegend", args);
  let Some((graphic, legend_args)) = args.split_first() else {
    return Ok(original());
  };
  // A plot arrives already rendered; a plain `Graphics[…]` stays symbolic
  // until it is shown, so render it to learn its size.
  let rendered = match graphic {
    Expr::FunctionCall { name, args }
      if name == "Graphics" && !args.is_empty() =>
    {
      crate::functions::graphics::graphics_ast(args)?
    }
    other => other.clone(),
  };
  let Expr::Graphics {
    svg, is_3d: false, ..
  } = &rendered
  else {
    return Ok(original());
  };
  let Some(dims) = crate::functions::graphics::parse_svg_dimensions(svg) else {
    return Ok(original());
  };
  let (nat_w, nat_h) = (dims.nat_w, dims.nat_h);
  let Some(legends) = legend_args
    .iter()
    .map(parse_legend)
    .collect::<Option<Vec<_>>>()
  else {
    return Ok(original());
  };
  if legends.is_empty() || nat_w <= 0.0 || nat_h <= 0.0 {
    return Ok(original());
  }

  // The graphic's longer side spans -1..1, in the shape of its picture.
  let ratio = nat_h / nat_w;
  let (pw, ph) = if ratio >= 1.0 {
    (2.0 / ratio, 2.0)
  } else {
    (2.0, 2.0 * ratio)
  };
  let (mut lo, mut hi) = ((-pw / 2.0, -ph / 2.0), (pw / 2.0, ph / 2.0));
  let mut primitives = vec![call(
    "Inset",
    vec![
      graphic.clone(),
      point(0.0, 0.0),
      Expr::Identifier("Center".into()),
      point(pw, ph),
    ],
  )];
  for legend in &legends {
    let ((ax, ay), (bx, by)) = legend.bounds();
    lo = (lo.0.min(ax), lo.1.min(ay));
    hi = (hi.0.max(bx), hi.1.max(by));
    primitives.extend(legend.primitives());
  }
  // The graphic keeps its own width; the picture grows around it.
  let image_width = (nat_w * (hi.0 - lo.0) / pw).round();
  let shown = call(
    "Graphics",
    vec![
      Expr::List(primitives.into()),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("PlotRange".into())),
        replacement: Box::new(Expr::List(
          vec![point(lo.0, hi.0), point(lo.1, hi.1)].into(),
        )),
      },
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("ImageSize".into())),
        replacement: Box::new(real(image_width)),
      },
    ],
  );
  evaluate_expr_to_expr(&shown)
}

/// The plots the package teaches its `PlotLegend` option.
const LEGEND_PLOTS: &[&str] = &[
  "Plot",
  "ListPlot",
  "ListLinePlot",
  "ParametricPlot",
  "PolarPlot",
  "LogPlot",
  "LogLinearPlot",
  "LogLogPlot",
  "ListLogPlot",
  "ListLogLinearPlot",
  "ListLogLogPlot",
];

/// The `Legend` options a plot with `PlotLegend` hands on to its legend.
const LEGEND_OPTIONS: &[&str] = &[
  "LegendBackground",
  "LegendBorder",
  "LegendLabel",
  "LegendLabelSpace",
  "LegendPosition",
  "LegendShadow",
  "LegendSize",
  "LegendTextSpace",
];

/// The name `name` stands for, if it is the package's symbol `short`: its
/// qualified name, or the bare one a read resolves to once the package is
/// on `$ContextPath`. (`LegendLabel` is a System symbol, which the package
/// reads as well.)
fn is_package_symbol(pattern: &Expr, short: &str) -> bool {
  match pattern {
    Expr::Identifier(name) => {
      name == short
        || name
          .strip_prefix("PlotLegends`")
          .is_some_and(|rest| rest == short)
    }
    _ => false,
  }
}

fn rule_parts(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((pattern, replacement)),
    _ => None,
  }
}

/// The color of the `k`-th curve in the plots the package was written
/// for: hues stepping by `Sqrt[5] - 2` from `0.67`, at saturation and
/// brightness `0.6`. The legend keys keep these colors whatever the plot
/// itself draws today.
fn legacy_curve_style(k: usize) -> Expr {
  let hue = (0.67 + k as f64 * (5f64.sqrt() - 2.0)).fract();
  call("Hue", vec![real(hue), real(0.6), real(0.6)])
}

/// The `k`-th entry of a per-dataset setting: a list is cycled through,
/// anything else applies to every dataset.
fn nth_setting(value: &Expr, k: usize) -> Expr {
  match value {
    Expr::List(items) if !items.is_empty() => items[k % items.len()].clone(),
    other => other.clone(),
  }
}

/// `plot[…, PlotLegend -> {label1, …}, legendOpts…]` once the package is
/// loaded: the plot without those options, under `ShowLegend` with one row
/// per label. Each row's key is a short line in the curve's style (a point
/// for a `ListPlot` dataset that is not joined); the `Legend` options given
/// to the plot style the legend, and an automatic size gives each row 0.3
/// of the frame's height in a box 0.8 wide, shrunk to fit a 0.8 square.
pub fn plot_with_legend(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if !LEGEND_PLOTS.contains(&name) {
    return None;
  }
  let legend_at = args.iter().position(|a| {
    rule_parts(a).is_some_and(|(p, _)| is_package_symbol(p, "PlotLegend"))
  })?;
  let mut plot_args = Vec::new();
  let mut legend_options = Vec::new();
  for (i, arg) in args.iter().enumerate() {
    match rule_parts(arg) {
      Some(_) if i == legend_at => {}
      // A second `PlotLegend` is shadowed by the first.
      Some((p, _)) if is_package_symbol(p, "PlotLegend") => {}
      Some((p, _))
        if LEGEND_OPTIONS.iter().any(|o| is_package_symbol(p, o)) =>
      {
        legend_options.push(arg.clone());
      }
      _ => plot_args.push(arg.clone()),
    }
  }
  Some(build_plot_legend(
    name,
    args,
    legend_at,
    plot_args,
    legend_options,
  ))
}

fn build_plot_legend(
  name: &str,
  args: &[Expr],
  legend_at: usize,
  plot_args: Vec<Expr>,
  mut legend_options: Vec<Expr>,
) -> Result<Expr, InterpreterError> {
  let (_, labels) = rule_parts(&args[legend_at]).unwrap();
  let labels = match evaluate_expr_to_expr(labels)? {
    Expr::List(ref items) => items.to_vec(),
    single => vec![single],
  };
  let plot = evaluate_expr_to_expr(&call(name, plot_args.clone()))?;
  if labels.is_empty() {
    return Ok(plot);
  }
  let setting = |option: &str| -> Option<Expr> {
    plot_args.iter().rev().find_map(|a| {
      let (p, r) = rule_parts(a)?;
      matches!(p, Expr::Identifier(n) if n == option)
        .then(|| evaluate_expr_to_expr(r).unwrap_or_else(|_| r.clone()))
    })
  };
  let styles = setting("PlotStyle")
    .filter(|s| !matches!(s, Expr::Identifier(n) if n == "Automatic"));
  let joined = setting("Joined");
  let points_by_default = name.starts_with("List") && name != "ListLinePlot";
  let rows: Vec<Expr> = labels
    .into_iter()
    .enumerate()
    .map(|(k, label)| {
      let style = match &styles {
        Some(s) => nth_setting(s, k),
        None => legacy_curve_style(k),
      };
      let as_point = points_by_default
        && !matches!(
          joined.as_ref().map(|j| nth_setting(j, k)),
          Some(Expr::Identifier(ref b)) if b == "True"
        );
      let mark = if as_point {
        call("Point", vec![point(0.5, 0.0)])
      } else {
        call(
          "Line",
          vec![Expr::List(vec![point(0.0, 0.0), point(1.0, 0.0)].into())],
        )
      };
      // The mark takes the middle half of its key box.
      let key = call(
        "Graphics",
        vec![
          Expr::List(vec![style, mark].into()),
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("AspectRatio".into())),
            replacement: Box::new(Expr::Identifier("Full".into())),
          },
          Expr::Rule {
            pattern: Box::new(Expr::Identifier("PlotRange".into())),
            replacement: Box::new(Expr::List(
              vec![point(-0.5, 1.5), point(-1.0, 1.0)].into(),
            )),
          },
        ],
      );
      Expr::List(vec![key, label].into())
    })
    .collect();
  let has_size = legend_options.iter().any(|o| {
    rule_parts(o).is_some_and(|(p, _)| is_package_symbol(p, "LegendSize"))
  });
  if !has_size {
    let (w, h): (f64, f64) = (0.8, 0.3 * rows.len() as f64);
    let fit = (0.8 / w.max(h)).min(1.0);
    legend_options.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("PlotLegends`LegendSize".into())),
      replacement: Box::new(point(w * fit, h * fit)),
    });
  }
  let mut legend = vec![Expr::List(rows.into())];
  legend.extend(legend_options);
  show_legend_ast(&[plot, Expr::List(legend.into())])
}

/// The legacy packages that announce on loading that they are obsolete.
pub const OBSOLETE_PACKAGES: &[&str] = &["PlotLegends`", "VectorFieldPlots`"];

/// `General::obspkg`, printed each time an obsolete package is read.
pub fn announce_obsolete_package(ctx: &str) {
  if OBSOLETE_PACKAGES.contains(&ctx) {
    crate::emit_message_to_stdout(&format!(
      "General::obspkg: {ctx} is now obsolete. The legacy version being \
       loaded may conflict with current functionality. See the Compatibility \
       Guide for updating information."
    ));
  }
}
