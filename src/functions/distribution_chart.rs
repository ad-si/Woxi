//! `DistributionChart[{d1, d2, …}]`: one "violin" per dataset, the
//! dataset's Gaussian kernel density estimate (Silverman bandwidth)
//! mirrored about the dataset's position on the horizontal axis. The
//! result is a symbolic `Graphics` (wrapped in `Legended` for
//! `ChartLegends`).

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{invalid_dataset, parse_datasets};
use crate::functions::field_plot::{
  apply_named_color_function, color_function_scheme_name,
};
use crate::functions::math_ast::silverman_bandwidth;
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

fn pair(a: f64, b: f64) -> Expr {
  Expr::List(vec![Expr::Real(a), Expr::Real(b)].into())
}

/// Wolfram's default `DistributionChart` fill.
const DEFAULT_STYLE: (f64, f64, f64) = (1.0, 0.835, 0.25);

/// The fill of each of `n` datasets from a `ChartStyle` value: a named
/// gradient sampled across the datasets, one colour, or a list cycled.
fn chart_styles(spec: Option<&Expr>, n: usize) -> Vec<Expr> {
  let default = || {
    let (r, g, b) = DEFAULT_STYLE;
    call(
      "RGBColor",
      vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
    )
  };
  let Some(spec) = spec else {
    return vec![default(); n];
  };
  if let Expr::List(items) = spec {
    if items.is_empty() {
      return vec![default(); n];
    }
    return (0..n).map(|i| items[i % items.len()].clone()).collect();
  }
  if let Some(name) = color_function_scheme_name(spec) {
    return (0..n)
      .map(|i| {
        let t = if n > 1 {
          i as f64 / (n - 1) as f64
        } else {
          0.0
        };
        let (r, g, b) = apply_named_color_function(&name, t);
        call(
          "RGBColor",
          vec![
            Expr::Real(r as f64 / 255.0),
            Expr::Real(g as f64 / 255.0),
            Expr::Real(b as f64 / 255.0),
          ],
        )
      })
      .collect();
  }
  vec![spec.clone(); n]
}

/// The violin of one dataset centred at `x`: its density estimate at
/// `SAMPLES` heights across `[lo, hi]`, scaled to a half-width of at most
/// `half_width`, traced up the right side and back down the left.
fn violin(data: &[f64], x: f64, half_width: f64, smooth: bool) -> Vec<Expr> {
  const SAMPLES: usize = 64;
  let h = silverman_bandwidth(data);
  let h = if h.is_finite() && h > 0.0 { h } else { 1.0 };
  let (min, max) = data
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
      (a.min(v), b.max(v))
    });
  // "SmoothDensity" keeps to the data's range; the default density
  // trails off over three bandwidths either side.
  let (lo, hi) = if smooth {
    (min, max)
  } else {
    (min - 3.0 * h, max + 3.0 * h)
  };
  let (lo, hi) = if hi > lo { (lo, hi) } else { (lo - h, hi + h) };
  let density = |y: f64| {
    data
      .iter()
      .map(|v| (-0.5 * ((y - v) / h).powi(2)).exp())
      .sum::<f64>()
  };
  let ys: Vec<f64> = (0..SAMPLES)
    .map(|k| lo + (hi - lo) * k as f64 / (SAMPLES - 1) as f64)
    .collect();
  let ds: Vec<f64> = ys.iter().map(|&y| density(y)).collect();
  let peak = ds.iter().copied().fold(0.0, f64::max);
  let scale = if peak > 0.0 { half_width / peak } else { 0.0 };
  let mut points = Vec::with_capacity(2 * SAMPLES);
  for (y, d) in ys.iter().zip(&ds) {
    points.push(pair(x + d * scale, *y));
  }
  for (y, d) in ys.iter().zip(&ds).rev() {
    points.push(pair(x - d * scale, *y));
  }
  points
}

pub fn distribution_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Ok(datasets) = parse_datasets(&args[0]) else {
    return Ok(invalid_dataset("DistributionChart", args));
  };
  let mut style = None;
  let mut element = None;
  let mut legends = None;
  let mut labels = None;
  let mut graphics_opts = Vec::new();
  for opt in &args[1..] {
    match opt {
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
          Expr::Identifier(k) if k == "ChartElementFunction" => {
            element = Some(value());
          }
          Expr::Identifier(k) if k == "ChartLegends" => legends = Some(value()),
          Expr::Identifier(k) if k == "ChartLabels" => labels = Some(value()),
          Expr::Identifier(k)
            if matches!(
              k.as_str(),
              "BarOrigin"
                | "BarSpacing"
                | "ChartBaseStyle"
                | "ChartLayout"
                | "LabelingFunction"
                | "LegendAppearance"
                | "PerformanceGoal"
            ) => {}
          _ => graphics_opts.push(opt.clone()),
        }
      }
      _ => return Ok(crate::helpers::unevaluated("DistributionChart", args)),
    }
  }
  let n = datasets.len();
  let fills = chart_styles(style.as_ref(), n);
  let smooth =
    matches!(&element, Some(Expr::String(s)) if s == "SmoothDensity");
  let edge = call(
    "EdgeForm",
    vec![call(
      "Directive",
      vec![
        call("Opacity", vec![Expr::Real(0.5)]),
        call("AbsoluteThickness", vec![Expr::Integer(1)]),
      ],
    )],
  );
  let mut content = Vec::with_capacity(n);
  for (i, data) in datasets.iter().enumerate() {
    let x = (i + 1) as f64;
    content.push(Expr::List(
      vec![
        call("Directive", vec![edge.clone(), fills[i].clone()]),
        call(
          "Polygon",
          vec![Expr::List(violin(data, x, 0.4, smooth).into())],
        ),
      ]
      .into(),
    ));
  }
  // A tick (labelled by `ChartLabels`) under each violin, none on the top.
  let tick_labels: Vec<Expr> = match &labels {
    Some(Expr::List(items)) => items.to_vec(),
    _ => Vec::new(),
  };
  let bottom = Expr::List(
    (0..n)
      .map(|i| {
        let label = tick_labels
          .get(i)
          .cloned()
          .unwrap_or_else(|| Expr::String(String::new()));
        Expr::List(vec![Expr::Integer(i as i128 + 1), label].into())
      })
      .collect(),
  );
  let mut options = graphics_opts;
  let has = |opts: &[Expr], k: &str| {
    opts.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == k))
    })
  };
  let mut defaults = vec![
    rule(
      "AspectRatio",
      call("Power", vec![id("GoldenRatio"), Expr::Integer(-1)]),
    ),
    rule("Axes", Expr::List(vec![id("False"), id("False")].into())),
    rule("Frame", id("True")),
    rule(
      "FrameTicks",
      Expr::List(
        vec![
          Expr::List(vec![id("Automatic"), id("Automatic")].into()),
          Expr::List(vec![bottom, id("None")].into()),
        ]
        .into(),
      ),
    ),
    rule("PlotRangePadding", call("Scaled", vec![Expr::Real(0.05)])),
  ];
  defaults.retain(|d| match d {
    Expr::Rule { pattern, .. } => match pattern.as_ref() {
      Expr::Identifier(k) => !has(&options, k),
      _ => true,
    },
    _ => true,
  });
  options.extend(defaults);
  let mut graphics_args = vec![Expr::List(content.into())];
  graphics_args.extend(options);
  let chart = evaluate_expr_to_expr(&call("Graphics", graphics_args))?;
  match &legends {
    Some(Expr::List(names)) if !names.is_empty() => Ok(call(
      "Legended",
      vec![
        chart,
        call(
          "Placed",
          vec![
            call(
              "SwatchLegend",
              vec![
                Expr::List(fills.into()),
                Expr::List(names.iter().take(n).cloned().collect()),
              ],
            ),
            id("After"),
          ],
        ),
      ],
    )),
    _ => Ok(chart),
  }
}
