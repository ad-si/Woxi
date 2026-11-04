//! `DensityHistogram[{{x1, y1}, …}, bspec, hspec]`: the 2D histogram of
//! the points as a grid of rectangles shaded by their heights.
//!
//! Each axis is binned by `HistogramList` (the canonical 1D binning) on
//! that coordinate, then the points are counted into the cells. Cells are
//! grouped by height, each group in the colour of its height scaled to
//! `0..1` over the heights that occur, as wolframscript draws them.

use crate::InterpreterError;
use crate::evaluator::evaluate_function_call_ast;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, call1, rule_expr, unevaluated};
use crate::syntax::Expr;

/// The default colour gradient, decoded from wolframscript: piecewise
/// linear in LAB between these four stops at 0, 1/3, 2/3 and 1.
const GRADIENT: [[f64; 3]; 4] = [
  [0.2374577454874495, 0.4502326479674301, -0.29344758919909575],
  [0.48288891610929785, 0.7019509501984089, 0.1585008503354124],
  [0.7226680127694958, 0.3783642915277907, 0.4813038070805067],
  [
    0.9735241229755421,
    -0.05249018370661607,
    0.31994142972937545,
  ],
];

fn gradient(t: f64) -> Expr {
  let t = t.clamp(0.0, 1.0) * 3.0;
  let k = (t.floor() as usize).min(2);
  let f = t - k as f64;
  let (a, b) = (GRADIENT[k], GRADIENT[k + 1]);
  call(
    "LABColor",
    (0..3)
      .map(|i| Expr::Real(a[i] + f * (b[i] - a[i])))
      .collect(),
  )
}

fn is_option(e: &Expr) -> bool {
  matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })
}

/// The bin edges `HistogramList` picks for `values` under `spec`.
fn edges(values: &[f64], spec: Option<&Expr>) -> Option<Vec<f64>> {
  let data = Expr::List(values.iter().map(|&v| Expr::Real(v)).collect());
  let mut list_args = vec![data];
  list_args.extend(spec.cloned());
  let result = evaluate_function_call_ast("HistogramList", &list_args).ok()?;
  let Expr::List(parts) = &result else {
    return None;
  };
  let Some(Expr::List(edges)) = parts.first() else {
    return None;
  };
  let edges: Vec<f64> =
    edges.iter().map(try_eval_to_f64).collect::<Option<_>>()?;
  (edges.len() >= 2).then_some(edges)
}

/// The cell of `edges` that holds `v` (bins are closed on the left).
fn bin_of(edges: &[f64], v: f64) -> Option<usize> {
  if v < edges[0] || v > edges[edges.len() - 1] {
    return None;
  }
  let k = edges.partition_point(|&e| e <= v);
  Some(k.saturating_sub(1).min(edges.len() - 2))
}

pub fn density_histogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unchanged = || Ok(unevaluated("DensityHistogram", args));
  let data_count = args.iter().take_while(|e| !is_option(e)).count();
  if data_count == 0 || data_count > 3 {
    return unchanged();
  }
  let options: Vec<&Expr> = args[data_count..].iter().collect();
  let Expr::List(items) = &args[0] else {
    return unchanged();
  };
  let mut points = Vec::with_capacity(items.len());
  for item in items {
    match item {
      Expr::List(xy) if xy.len() == 2 => {
        match (try_eval_to_f64(&xy[0]), try_eval_to_f64(&xy[1])) {
          (Some(x), Some(y)) => points.push((x, y)),
          _ => return unchanged(),
        }
      }
      _ => return unchanged(),
    }
  }
  if points.is_empty() {
    return unchanged();
  }
  // The bin specification of each axis: `{dx}` or `n` (or Automatic) for
  // both, `{xspec, yspec}` for each.
  let spec = args.get(1).filter(|_| data_count >= 2);
  let (x_spec, y_spec) = match spec {
    None => (None, None),
    Some(Expr::Identifier(a)) if a == "Automatic" => (None, None),
    Some(Expr::List(parts))
      if parts.len() == 2
        && parts.iter().all(|p| {
          matches!(p, Expr::List(_) | Expr::Integer(_) | Expr::Identifier(_))
        }) =>
    {
      let axis = |p: &Expr| match p {
        Expr::Identifier(a) if a == "Automatic" => None,
        other => Some(other.clone()),
      };
      (axis(&parts[0]), axis(&parts[1]))
    }
    Some(other) => (Some(other.clone()), Some(other.clone())),
  };
  let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
  let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
  let (Some(x_edges), Some(y_edges)) =
    (edges(&xs, x_spec.as_ref()), edges(&ys, y_spec.as_ref()))
  else {
    return unchanged();
  };
  let (nx, ny) = (x_edges.len() - 1, y_edges.len() - 1);
  let mut counts = vec![vec![0usize; ny]; nx];
  for &(x, y) in &points {
    if let (Some(i), Some(j)) = (bin_of(&x_edges, x), bin_of(&y_edges, y)) {
      counts[i][j] += 1;
    }
  }
  let total = points.len() as f64;
  let height_spec = match args.get(2).filter(|_| data_count == 3) {
    None => "Count",
    Some(Expr::String(s))
      if matches!(s.as_str(), "Count" | "Probability" | "PDF") =>
    {
      s.as_str()
    }
    Some(_) => return unchanged(),
  };
  let height = |i: usize, j: usize| {
    let c = counts[i][j] as f64;
    match height_spec {
      "Probability" => c / total,
      "PDF" => {
        c / (total
          * (x_edges[i + 1] - x_edges[i])
          * (y_edges[j + 1] - y_edges[j]))
      }
      _ => c,
    }
  };
  // Non-empty cells grouped by height, lowest first.
  let mut cells: Vec<(f64, usize, usize, usize)> = Vec::new();
  for i in 0..nx {
    for j in 0..ny {
      if counts[i][j] > 0 {
        cells.push((height(i, j), counts[i][j], i, j));
      }
    }
  }
  cells.sort_by(|a, b| {
    a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)).then(a.3.cmp(&b.3))
  });
  let lo = cells.first().map_or(0.0, |c| c.0);
  let hi = cells.last().map_or(1.0, |c| c.0);
  let color_function = options.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ColorFunction")
      && !matches!(replacement.as_ref(), Expr::Identifier(a) if a == "Automatic") =>
    {
      Some(replacement.as_ref().clone())
    }
    _ => None,
  });
  let color = |h: f64| -> Expr {
    let t = if hi > lo { (h - lo) / (hi - lo) } else { 1.0 };
    match &color_function {
      Some(f) => crate::evaluator::evaluate_expr_to_expr(&Expr::CurriedCall {
        func: Box::new(f.clone()),
        args: vec![Expr::Real(t)],
      })
      .unwrap_or_else(|_| gradient(t)),
      None => gradient(t),
    }
  };
  let number = |h: f64, c: usize| {
    if height_spec == "Count" {
      Expr::Integer(c as i128)
    } else {
      Expr::Real(h)
    }
  };
  let point =
    |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let mut groups = Vec::new();
  let mut k = 0;
  while k < cells.len() {
    let h = cells[k].0;
    let count = cells[k].1;
    let mut rects = Vec::new();
    while k < cells.len() && cells[k].0 == h {
      let (_, _, i, j) = cells[k];
      rects.push(call(
        "Rectangle",
        vec![
          point(x_edges[i], y_edges[j]),
          point(x_edges[i + 1], y_edges[j + 1]),
        ],
      ));
      k += 1;
    }
    groups.push(Expr::List(
      vec![
        call(
          "Directive",
          vec![call1("EdgeForm", Expr::Identifier("None".into())), color(h)],
        ),
        call("Tooltip", vec![Expr::List(rects.into()), number(h, count)]),
      ]
      .into(),
    ));
  }
  let id = |s: &str| Expr::Identifier(s.into());
  let range = |e: &[f64]| {
    Expr::List(vec![Expr::Real(e[0]), Expr::Real(e[e.len() - 1])].into())
  };
  let mut graphics_args = vec![Expr::List(groups.into())];
  graphics_args.extend(
    options
      .iter()
      .filter(|o| {
        !matches!(o, Expr::Rule { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ColorFunction"))
      })
      .map(|o| (*o).clone()),
  );
  graphics_args.extend([
    rule_expr(id("AspectRatio"), Expr::Integer(1)),
    rule_expr(id("Frame"), id("True")),
    rule_expr(
      id("PlotRange"),
      Expr::List(vec![range(&x_edges), range(&y_edges)].into()),
    ),
    rule_expr(id("PlotRangePadding"), call1("Scaled", Expr::Real(0.05))),
  ]);
  evaluate_function_call_ast("Graphics", &graphics_args)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bins_are_closed_on_the_left() {
    let e = [0.0, 1.0, 2.0];
    assert_eq!(bin_of(&e, 0.0), Some(0));
    assert_eq!(bin_of(&e, 1.0), Some(1));
    assert_eq!(bin_of(&e, 2.0), Some(1));
    assert_eq!(bin_of(&e, 2.5), None);
  }
}
