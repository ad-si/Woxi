//! `QuantilePlot`, `ProbabilityPlot` and `ProbabilityScalePlot`: the
//! sorted data of each dataset against a reference distribution (a normal
//! fitted to the data unless one is given) or a second dataset, as a framed
//! `ListPlot` with a dotted reference line in the series' color.
//!
//! The three share their layout, decoded from wolframscript: plotting
//! positions `(i - 3/10)/(n + 2/5)` (Benard's median ranks), a fitted
//! normal with the maximum-likelihood spread, a reference line through the
//! quartile points (`Quantile[…, {1/4, 3/4}, {{1/2, 0}, {0, 1}}]`) and two
//! invisible points on that line that make the plot range reach the
//! line's ends.

use crate::InterpreterError;
use crate::evaluator::evaluate_function_call_ast;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, call1, rule_expr, unevaluated};
use crate::syntax::Expr;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
  /// Quantiles of the data against the reference quantiles.
  Quantile,
  /// The data's empirical CDF against the reference CDF.
  Probability,
  /// The data on a normal probability scale.
  ProbabilityScale,
}

impl Kind {
  fn head(self) -> &'static str {
    match self {
      Kind::Quantile => "QuantilePlot",
      Kind::Probability => "ProbabilityPlot",
      Kind::ProbabilityScale => "ProbabilityScalePlot",
    }
  }
}

/// What a dataset is compared against.
enum Reference {
  /// A normal distribution fitted to each dataset.
  Fitted,
  /// An explicit distribution.
  Distribution(Expr),
  /// A second dataset (sorted).
  Data(Vec<f64>),
}

/// QuantilePlot[data], QuantilePlot[data, dist], QuantilePlot[d1, d2].
pub fn quantile_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  quantile_family_ast(Kind::Quantile, args)
}

/// ProbabilityPlot[data], ProbabilityPlot[data, dist],
/// ProbabilityPlot[d1, d2].
pub fn probability_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  quantile_family_ast(Kind::Probability, args)
}

/// ProbabilityScalePlot[data] and ProbabilityScalePlot[data, "Normal"].
pub fn probability_scale_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  quantile_family_ast(Kind::ProbabilityScale, args)
}

fn is_option(e: &Expr) -> bool {
  matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })
}

fn numbers(e: &Expr) -> Option<Vec<f64>> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      items.iter().map(try_eval_to_f64).collect()
    }
    _ => None,
  }
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
  v.sort_by(f64::total_cmp);
  v
}

/// `Quantile[sorted, q, {{1/2, 0}, {0, 1}}]`: linear interpolation between
/// the order statistics around position `n q + 1/2`.
fn quartile(sorted: &[f64], q: f64) -> f64 {
  let n = sorted.len();
  let pos = (n as f64 * q + 0.5).clamp(1.0, n as f64);
  let lo = pos.floor() as usize;
  let frac = pos - lo as f64;
  let a = sorted[lo - 1];
  if frac == 0.0 || lo >= n {
    a
  } else {
    a + frac * (sorted[lo] - a)
  }
}

/// `Quantile[sorted, p]` with the default parameters: the order statistic
/// at position `⌈n p⌉`.
fn empirical_quantile(sorted: &[f64], p: f64) -> f64 {
  let n = sorted.len();
  let k = ((n as f64 * p).ceil() as usize).clamp(1, n);
  sorted[k - 1]
}

fn apply(head: &str, dist: &Expr, v: f64) -> Option<f64> {
  let r =
    evaluate_function_call_ast(head, &[dist.clone(), Expr::Real(v)]).ok()?;
  try_eval_to_f64(&r)
}

/// The normal distribution fitted to `data` by maximum likelihood.
fn fitted_normal(data: &[f64]) -> Expr {
  let n = data.len() as f64;
  let mean = data.iter().sum::<f64>() / n;
  let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
  call(
    "NormalDistribution",
    vec![Expr::Real(mean), Expr::Real(var.sqrt())],
  )
}

fn point(x: f64, y: f64) -> Expr {
  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
}

/// The x span the invisible points pin the plot range to: the data's
/// `lo..hi` reaching 0 when its nearer end is within 1/6 of the farther,
/// lowered by 1/48 of that span and padded by 2% of the result on both
/// sides.
fn line_span(lo: f64, hi: f64) -> (f64, f64) {
  let lo = if lo > 0.0 && lo <= hi / 6.0 { 0.0 } else { lo };
  let drop = (hi - lo) / 48.0;
  let width = (hi - lo + drop) / 0.96;
  (lo - drop - 0.02 * width, hi + 0.02 * width)
}

fn quantile_family_ast(
  kind: Kind,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let head = kind.head();
  let weibull = kind == Kind::ProbabilityScale
    && matches!(args.get(1), Some(Expr::String(s)) if s == "Weibull");
  let unchanged = || Ok(unevaluated(head, args));
  let data_count = args.iter().take_while(|e| !is_option(e)).count();
  if data_count == 0 || data_count > 2 {
    return unchanged();
  }
  let options: Vec<&Expr> = args[data_count..].iter().collect();
  if !options.iter().all(|o| is_option(o)) {
    return unchanged();
  }
  let sets: Vec<Vec<f64>> = if let Some(v) = numbers(&args[0]) {
    vec![sorted(v)]
  } else if let Expr::List(items) = &args[0]
    && !items.is_empty()
    && let Some(sets) = items.iter().map(numbers).collect::<Option<Vec<_>>>()
  {
    sets.into_iter().map(sorted).collect()
  } else {
    return unchanged();
  };
  let reference = match args.get(1).filter(|_| data_count == 2) {
    None => Reference::Fitted,
    Some(Expr::String(s)) if kind == Kind::ProbabilityScale => {
      if s == "Normal" || s == "Weibull" {
        Reference::Fitted
      } else {
        return unchanged();
      }
    }
    Some(_) if kind == Kind::ProbabilityScale => return unchanged(),
    Some(e) => match numbers(e) {
      Some(v) if sets.len() == 1 => Reference::Data(sorted(v)),
      Some(_) => return unchanged(),
      None if apply("InverseCDF", e, 0.5).is_some() => {
        Reference::Distribution(e.clone())
      }
      None => return unchanged(),
    },
  };

  // The points of each series.
  let mut series: Vec<Vec<(f64, f64)>> = Vec::new();
  for s in &sets {
    let n = s.len();
    let ps: Vec<f64> = (1..=n)
      .map(|i| (i as f64 - 0.3) / (n as f64 + 0.4))
      .collect();
    let dist = match &reference {
      Reference::Fitted => Some(fitted_normal(s)),
      Reference::Distribution(d) => Some(d.clone()),
      Reference::Data(_) => None,
    };
    let mut pts = Vec::with_capacity(n);
    for (i, &y) in s.iter().enumerate() {
      let pt = match (kind, &reference, &dist) {
        (Kind::Quantile, Reference::Data(other), _) => {
          Some((empirical_quantile(other, ps[i]), y))
        }
        (Kind::Quantile, _, Some(d)) => {
          apply("InverseCDF", d, ps[i]).map(|x| (x, y))
        }
        (Kind::Probability, Reference::Data(other), _) => {
          let below = other.iter().filter(|&&u| u <= y).count();
          Some((below as f64 / other.len() as f64, (i + 1) as f64 / n as f64))
        }
        (Kind::Probability, _, Some(d)) => {
          apply("CDF", d, y).map(|x| (x, (i + 1) as f64 / n as f64))
        }
        (Kind::ProbabilityScale, _, _) => {
          if weibull {
            if y <= 0.0 {
              continue;
            }
            pts.push((y.ln(), (-(1.0 - ps[i]).ln()).ln()));
            continue;
          }
          let normal = call(
            "NormalDistribution",
            vec![Expr::Integer(0), Expr::Integer(1)],
          );
          apply("InverseCDF", &normal, ps[i]).map(|z| (y, 1.0 + z))
        }
        _ => None,
      };
      match pt {
        Some((x, y)) if x.is_finite() && y.is_finite() => pts.push((x, y)),
        _ => return unchanged(),
      }
    }
    if pts.is_empty() {
      return unchanged();
    }
    series.push(pts);
  }

  // The reference line of each series as a function of x, and its ends.
  let all_x: Vec<f64> =
    series.iter().flatten().map(|p| p.0).collect::<Vec<_>>();
  let x_lo = all_x.iter().copied().fold(f64::INFINITY, f64::min);
  let x_hi = all_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let mut lines: Vec<Option<((f64, f64), (f64, f64))>> = Vec::new();
  let mut hidden: Vec<Expr> = Vec::new();
  for pts in &series {
    if kind == Kind::Probability {
      let eps = 1.0 / 49_000_000.0;
      lines.push(Some(((eps, eps), (1.0 - eps, 1.0 - eps))));
      if hidden.is_empty() {
        hidden.push(point(eps, eps));
        hidden.push(point(1.0 - eps, 1.0 - eps));
      }
      continue;
    }
    let xs = sorted(pts.iter().map(|p| p.0).collect());
    let ys = sorted(pts.iter().map(|p| p.1).collect());
    let (qx1, qx3) = (quartile(&xs, 0.25), quartile(&xs, 0.75));
    let (qy1, qy3) = (quartile(&ys, 0.25), quartile(&ys, 0.75));
    if qx3 == qx1 || !(qx1.is_finite() && qx3.is_finite()) {
      lines.push(None);
      continue;
    }
    let slope = (qy3 - qy1) / (qx3 - qx1);
    let at = |x: f64| qy1 + slope * (x - qx1);
    let ends = match kind {
      Kind::ProbabilityScale if slope != 0.0 => {
        let x_at = |y: f64| qx1 + (y - qy1) / slope;
        let (y0, y1) = (pts[0].1, pts[pts.len() - 1].1);
        ((x_at(y0), y0), (x_at(y1), y1))
      }
      _ => {
        let (a, b) = (x_lo - 2.0 * x_lo.abs(), x_hi + 2.0 * x_hi.abs());
        ((a, at(a)), (b, at(b)))
      }
    };
    lines.push(Some(ends));
    let (l, r) = line_span(x_lo, x_hi);
    hidden.push(point(l, at(l)));
    hidden.push(point(r, at(r)));
  }

  // Series styles: the user's PlotStyle, else the default plot colors.
  let user_style = options.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle") => {
      Some(match replacement.as_ref() {
        Expr::List(items) if !items.is_empty() => items.to_vec(),
        other => vec![other.clone()],
      })
    }
    _ => None,
  });
  let style_of = |i: usize| {
    if let Some(styles) = &user_style {
      styles[i % styles.len()].clone()
    } else {
      let (r, g, b) = crate::functions::plot::PLOT_COLORS_RGB[i % 15];
      call(
        "RGBColor",
        vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
      )
    }
  };
  let epilog: Vec<Expr> = lines
    .iter()
    .enumerate()
    .filter_map(|(i, l)| {
      let (a, b) = (*l)?;
      Some(call(
        "Style",
        vec![
          call1(
            "Line",
            Expr::List(vec![point(a.0, a.1), point(b.0, b.1)].into()),
          ),
          call(
            "Directive",
            vec![
              style_of(i),
              call1("AbsoluteThickness", Expr::Integer(2)),
              call1(
                "Dashing",
                Expr::List(
                  vec![Expr::Integer(0), Expr::Identifier("Small".into())]
                    .into(),
                ),
              ),
            ],
          ),
        ],
      ))
    })
    .collect();

  let id = |s: &str| Expr::Identifier(s.into());
  let mut plot_args = vec![Expr::List(
    series
      .iter()
      .map(|pts| Expr::List(pts.iter().map(|&(x, y)| point(x, y)).collect()))
      .collect(),
  )];
  let scale = kind == Kind::ProbabilityScale;
  plot_args.extend(
    options
      .iter()
      .filter(|o| {
        !(scale
          && matches!(o, Expr::Rule { pattern, .. }
            if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle")))
      })
      .map(|o| (*o).clone()),
  );
  plot_args.push(rule_expr(id("Frame"), id("True")));
  plot_args.push(rule_expr(id("Axes"), id("False")));
  plot_args.push(rule_expr(id("Epilog"), Expr::List(epilog.into())));
  if scale {
    // The dots are a little larger than a ListPlot's, and the vertical
    // axis is labeled in percent.
    let size = crate::functions::math_ast::make_rational(
      crate::functions::list_plot::point_size_steps(&series) as i128,
      360,
    );
    let styles = (0..series.len())
      .map(|i| {
        call(
          "Directive",
          vec![style_of(i), call1("PointSize", size.clone())],
        )
      })
      .collect();
    plot_args.push(rule_expr(id("PlotStyle"), Expr::List(styles)));
    let scale_name = if weibull { "WeibullY" } else { "NormalY" };
    let scaling = Expr::List(
      vec![
        Expr::String(scale_name.into()),
        Expr::String(format!("{scale_name}Inverse")),
      ]
      .into(),
    );
    let left = call("Charting`ScaledTicks", vec![scaling.clone()]);
    let right = call("Charting`ScaledFrameTicks", vec![scaling]);
    plot_args.push(rule_expr(
      id("FrameTicks"),
      Expr::List(
        vec![
          Expr::List(vec![left, right].into()),
          if weibull {
            let log = Expr::List(vec![id("Log"), id("Exp")].into());
            Expr::List(
              vec![
                call("Charting`ScaledTicks", vec![log.clone()]),
                call("Charting`ScaledFrameTicks", vec![log]),
              ]
              .into(),
            )
          } else {
            Expr::List(vec![id("Automatic"), id("Automatic")].into())
          },
        ]
        .into(),
      ),
    ));
  }
  let plot = evaluate_function_call_ast("ListPlot", &plot_args)?;
  if hidden.is_empty() {
    return Ok(plot);
  }
  let invisible = call(
    "Graphics",
    vec![Expr::List(
      vec![
        call1("Opacity", Expr::Integer(0)),
        call1("Point", Expr::List(hidden.into())),
      ]
      .into(),
    )],
  );
  evaluate_function_call_ast("Show", &[plot, invisible])
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn quartiles_interpolate_order_statistics() {
    let s = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    assert_eq!(quartile(&s, 0.25), 2.0);
    assert_eq!(quartile(&s, 0.75), 5.0);
    let s = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(quartile(&s, 0.25), 1.75);
    assert_eq!(quartile(&s, 0.75), 4.25);
  }

  // wolframscript's invisible points of QuantilePlot[{1, …, 6}] and
  // QuantilePlot[{1, 2, 3, 4, 5, 10}].
  #[test]
  fn line_span_matches_wolfram() {
    let (l, r) = line_span(1.3996163075372414, 5.600383692462758);
    assert!((l - 1.222761083432304).abs() < 1e-12, "{l}");
    assert!((r - 5.689722929381747).abs() < 1e-12, "{r}");
    let (l, r) = line_span(0.5869066920755728, 7.746426641257761);
    assert!((l + 0.3261299410598971).abs() < 1e-12, "{l}");
    assert!((r - 7.911172693958123).abs() < 1e-12, "{r}");
  }
}
