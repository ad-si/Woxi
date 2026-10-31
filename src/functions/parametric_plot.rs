use super::*;
use std::cmp::Ordering;

use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  NUM_SAMPLES, PlotOptions, PlotRangeOverrides, SeriesStyle,
  adjust_y_range_for_filling_opts, apply_common_plot_option, build_plot_source,
  evaluate_at_point, generate_svg_with_filling, parse_iterator, substitute_var,
};

/// Parse the trailing option rules of a ParametricPlot/PolarPlot call.
/// Matching Wolfram Language, the first occurrence of a repeated option
/// wins (e.g. `PlotPoints -> 200, …, PlotPoints -> 10` uses 200).
fn parse_options(opts: &[Expr]) -> (PlotOptions, PlotRangeOverrides) {
  let mut plot_opts = PlotOptions::default();
  let mut overrides = PlotRangeOverrides::default();
  let mut seen: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  for opt in opts {
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      if !seen.insert(name.to_string()) {
        continue;
      }
      apply_common_plot_option(
        name,
        replacement,
        &mut plot_opts,
        &mut overrides,
      );
      // In ParametricPlot both coordinates are dependent values, so a
      // scalar-pair `PlotRange -> {min, max}` fixes *both* axes (Plot's
      // parser maps it to y only, which fits plots of one dependent
      // variable but would leave the x range jumping between animation
      // frames here).
      // A bare `PlotRange -> r` is likewise the square `{{-r, r}, {-r, r}}`.
      if name == "PlotRange" && overrides.x.is_none() {
        let scalar_pair = matches!(replacement, Expr::List(items)
          if items.len() == 2
            && !matches!(&items[0], Expr::List(_))
            && !matches!(&items[1], Expr::List(_)));
        let scalar = try_eval_to_f64(
          &evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.clone()),
        )
        .is_some_and(|r| r > 0.0);
        if scalar_pair || scalar {
          overrides.x = overrides.y;
        }
      }
    }
  }
  (plot_opts, overrides)
}

/// Pick the displayed x/y ranges and the SVG height for a parametric-style
/// plot: explicit PlotRange components override the data extents, and the
/// image aspect follows the displayed data (so circles render round) unless
/// an explicit AspectRatio was given.
fn apply_ranges_and_aspect(
  plot_opts: &mut PlotOptions,
  overrides: &PlotRangeOverrides,
  data_x: (f64, f64),
  data_y: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
  let x_range = overrides.x.unwrap_or(data_x);
  let y_range = overrides.y.unwrap_or(data_y);

  if let Some(ar) = overrides.aspect_ratio {
    // Explicit AspectRatio sizes the plotting area; the total height is
    // derived in generate_svg_with_options once margins are known.
    plot_opts.aspect_ratio = Some(ar);
  } else {
    // Adjust aspect ratio to match the displayed data (circles stay round).
    let data_w = x_range.1 - x_range.0;
    let data_h = y_range.1 - y_range.0;
    if data_w > 0.0 && data_h > 0.0 {
      let data_aspect = data_h / data_w;
      if data_aspect.is_finite() {
        plot_opts.svg_height =
          (plot_opts.svg_width as f64 * data_aspect).round() as u32;
        plot_opts.svg_height = plot_opts
          .svg_height
          .clamp(1, crate::functions::plot::MAX_SVG_DIMENSION);
      }
    }
  }
  (x_range, y_range)
}

/// A single parametric curve, sampled either component-wise (`{fx, fy}` where
/// both components are explicit expressions) or as a whole expression that
/// only yields a coordinate pair once the parameter is numeric (e.g.
/// `BSplineFunction[…][t]` or any `f[t]` returning `{x, y}`).
enum CurveSrc<'a> {
  Pair(&'a Expr, &'a Expr),
  Whole(&'a Expr),
}

/// Whether a list node can be read as a curve or as a grouping of curves.
/// Wolfram lets the curve specification be nested to any depth — the
/// grouping only steers styling — so `{{fx, fy}, …}` and
/// `{{{fx, fy}, …}}` both denote the same set of curves.
fn is_curve_or_group(expr: &Expr) -> bool {
  match expr {
    // `{}` is a curve slot that draws nothing — how a conditionally hidden
    // curve is written (`If[cond, {fx, fy}, {}]`) once `cond` resolves to
    // `False`.
    Expr::List(items) if items.is_empty() => true,
    // `If[cond, {fx, fy}]` without an else branch evaluates to `Null` when
    // `cond` is `False`, which hides the curve the same way.
    Expr::Identifier(name) if name == "Null" => true,
    // A pair is either a curve `{fx, fy}` or a group of two curves;
    // `collect_curves` decides which when it descends.
    Expr::List(items) if items.len() == 2 => true,
    Expr::List(items) => items.iter().all(is_curve_or_group),
    _ => false,
  }
}

/// Whether `expr` is unambiguously a (possibly nested) group of curves —
/// every element is itself curve-shaped, including an empty `{}` slot. This
/// is the same test `collect_curves` uses to pick its grouping branch,
/// exposed separately so a caller can tell that branch apart from the
/// ambiguous `{fx, fy}` single-curve fallback *before* descending.
fn is_curve_group(expr: &Expr) -> bool {
  matches!(expr, Expr::List(items) if !items.is_empty() && items.iter().all(is_curve_or_group))
}

/// Flatten a (possibly nested) curve specification into individual curves,
/// appending them to `out`. Returns whether the shape was understood.
fn collect_curves<'a>(expr: &'a Expr, out: &mut Vec<CurveSrc<'a>>) -> bool {
  if matches!(expr, Expr::Identifier(name) if name == "Null") {
    return true;
  }
  let Expr::List(items) = expr else {
    return false;
  };
  // An empty curve slot (`{}`) contributes no curves but is a valid,
  // understood shape — not a parse failure.
  if items.is_empty() {
    return true;
  }
  // Every element being curve-shaped makes this a grouping level — which
  // is how the ambiguous `{{a, b}, {c, d}}` resolves to two curves.
  if items.iter().all(is_curve_or_group) {
    return items.iter().all(|item| collect_curves(item, out));
  }
  // A coordinate is never itself a list, so a pair of lists is a group of
  // curves whose slots only resolve once the parameter is numeric.
  if items.len() == 2 && !items.iter().any(|i| matches!(i, Expr::List(_))) {
    out.push(CurveSrc::Pair(&items[0], &items[1]));
    return true;
  }
  false
}

/// Bisection passes used to smooth out an under-resolved curve. Each pass
/// halves the flagged intervals, so six passes shrink a step by 64×.
const MAX_REFINEMENT_DEPTH: usize = 6;

/// One sampled parameter value together with the coordinate pair of every
/// curve there (`None` when the body did not evaluate to coordinates).
type Sample = (f64, Option<Vec<(f64, f64)>>);

/// Whether the middle of a sampled triple is far enough off the straight
/// line between its neighbours that the interval needs more points. The
/// measure is the deviation relative to the neighbours' separation, so it
/// is independent of the coordinate scale.
fn needs_refinement(a: &Sample, b: &Sample, c: &Sample) -> bool {
  let (Some(ra), Some(rb), Some(rc)) = (&a.1, &b.1, &c.1) else {
    // A gap in the samples is a discontinuity (or a domain boundary);
    // refining pins down where it actually starts.
    return true;
  };
  if ra.len() != rb.len() || rb.len() != rc.len() {
    return true;
  }
  let span = c.0 - a.0;
  if span == 0.0 {
    return false;
  }
  let u = (b.0 - a.0) / span;
  ra.iter().zip(rb).zip(rc).any(|((p0, p1), p2)| {
    if !p0.0.is_finite()
      || !p0.1.is_finite()
      || !p1.0.is_finite()
      || !p1.1.is_finite()
      || !p2.0.is_finite()
      || !p2.1.is_finite()
    {
      return true;
    }
    let interp = (p0.0 + (p2.0 - p0.0) * u, p0.1 + (p2.1 - p0.1) * u);
    let deviation = (p1.0 - interp.0).hypot(p1.1 - interp.1);
    let separation = (p2.0 - p0.0).hypot(p2.1 - p0.1).max(1e-10);
    deviation / separation > 0.05
  })
}

/// Adaptively sample a parametric curve: lay down a uniform grid, then
/// bisect the intervals where the polyline visibly departs from the curve.
/// A uniform grid alone leaves a curve polygonal whenever the part that
/// bends is a small slice of the parameter range (e.g. `{t, -1000, 1000}`
/// for a curve shaped near `t == 0`), which is why Wolfram refines instead
/// of sampling a fixed grid.
fn adaptive_samples<F>(
  sample: F,
  t_min: f64,
  t_max: f64,
  initial_n: usize,
  max_total: usize,
) -> Vec<Sample>
where
  F: Fn(f64) -> Option<Vec<(f64, f64)>>,
{
  let initial_n = initial_n.max(2);
  let step = (t_max - t_min) / (initial_n - 1) as f64;
  let mut samples: Vec<Sample> = (0..initial_n)
    .map(|i| {
      let t = t_min + i as f64 * step;
      (t, sample(t))
    })
    .collect();

  let min_span = (t_max - t_min).abs() * 1e-10;
  for _ in 0..MAX_REFINEMENT_DEPTH {
    if samples.len() >= max_total {
      break;
    }
    let budget = max_total - samples.len();
    let mut added: Vec<Sample> = Vec::new();
    for triple in samples.windows(3) {
      if added.len() >= budget {
        break;
      }
      if (triple[2].0 - triple[0].0).abs() < min_span {
        continue;
      }
      if !needs_refinement(&triple[0], &triple[1], &triple[2]) {
        continue;
      }
      for (left, right) in [(&triple[0], &triple[1]), (&triple[1], &triple[2])]
      {
        let mid = f64::midpoint(left.0, right.0);
        // Neighbouring triples share an interval, so the same midpoint
        // comes up twice; skipping it saves evaluating the body again.
        if mid > left.0
          && mid < right.0
          && added.last().is_none_or(|last| last.0 != mid)
        {
          added.push((mid, sample(mid)));
        }
      }
    }
    if added.is_empty() {
      break;
    }
    samples.extend(added);
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    samples.dedup_by(|a, b| a.0 == b.0);
  }

  samples
}

/// Split the per-parameter samples into one point series per curve, keeping
/// a gap (as a non-finite point) wherever the body did not evaluate.
fn samples_to_series(samples: Vec<Sample>) -> (Vec<Vec<(f64, f64)>>, Vec<f64>) {
  // The curve count comes from the first sample that evaluated.
  let n_curves = samples
    .iter()
    .find_map(|(_, row)| row.as_ref().map(std::vec::Vec::len))
    .unwrap_or(1);
  let mut series = vec![Vec::with_capacity(samples.len()); n_curves];
  let params: Vec<f64> = samples.iter().map(|(t, _)| *t).collect();
  for (_, row) in samples {
    match row {
      Some(r) if r.len() == n_curves => {
        for (curve, point) in series.iter_mut().zip(r) {
          curve.push(point);
        }
      }
      _ => {
        for curve in &mut series {
          curve.push((f64::NAN, f64::NAN));
        }
      }
    }
  }
  (series, params)
}

/// Split each sampled curve into runs of constant `ColorFunction` color.
/// The function receives `{x, y, t}` taken at the midpoint of every step;
/// with `ColorFunctionScaling` on, `x` and `y` are rescaled over the plot
/// range and `t` over the parameter range first. A named gradient colors by
/// the scaled `x`. Consecutive runs share their boundary point so the curve
/// stays connected. Returns the run polylines and the color of each.
fn color_function_series(
  curves: &[Vec<(f64, f64)>],
  params: &[Vec<f64>],
  cf: &Expr,
  scaling: bool,
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  (t_min, t_max): (f64, f64),
) -> (Vec<Vec<(f64, f64)>>, Vec<(u8, u8, u8)>, Vec<usize>) {
  let scale = |v: f64, lo: f64, hi: f64| {
    if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }
  };
  let scheme = crate::functions::field_plot::color_function_scheme_name(cf);
  let color_at = |x: f64, y: f64, t: f64| -> (u8, u8, u8) {
    let (sx, sy, st) = (
      scale(x, x_min, x_max),
      scale(y, y_min, y_max),
      scale(t, t_min, t_max),
    );
    if let Some(name) = &scheme {
      return crate::functions::field_plot::apply_named_color_function(
        name, sx,
      );
    }
    let args = if scaling {
      vec![Expr::Real(sx), Expr::Real(sy), Expr::Real(st)]
    } else {
      vec![Expr::Real(x), Expr::Real(y), Expr::Real(t)]
    };
    let call = Expr::CurriedCall {
      func: Box::new(cf.clone()),
      args,
    };
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    evaluate_expr_to_expr(&call)
      .ok()
      .and_then(|r| crate::functions::graphics::parse_color(&r))
      .map_or((0, 0, 0), |c| (ch(c.r), ch(c.g), ch(c.b)))
  };
  let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
  let mut colors: Vec<(u8, u8, u8)> = Vec::new();
  let mut owners: Vec<usize> = Vec::new();
  for (ci, (curve, ts)) in curves.iter().zip(params).enumerate() {
    // Index of the run still being extended (None after a gap).
    let mut open: Option<usize> = None;
    for (pair, tp) in curve.windows(2).zip(ts.windows(2)) {
      let (a, b) = (pair[0], pair[1]);
      if ![a.0, a.1, b.0, b.1].iter().all(|v| v.is_finite()) {
        open = None;
        continue;
      }
      let c = color_at(
        f64::midpoint(a.0, b.0),
        f64::midpoint(a.1, b.1),
        f64::midpoint(tp[0], tp[1]),
      );
      match open {
        Some(i) if colors[i] == c => runs[i].push(b),
        _ => {
          runs.push(vec![a, b]);
          colors.push(c);
          owners.push(ci);
          open = Some(runs.len() - 1);
        }
      }
    }
  }
  (runs, colors, owners)
}

/// Sample a whole-expression curve at parameter `t`. The evaluated body may
/// be a single coordinate pair (one curve) or a list of coordinate pairs
/// (one sample per curve, e.g. `ReIm[{c1, c2, c3}]`); either way one row of
/// per-curve samples is returned.
fn sample_whole_rows(
  body: &Expr,
  var: &str,
  t: f64,
) -> Option<Vec<(f64, f64)>> {
  let substituted = substitute_var(body, var, &Expr::Real(t));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  let items = match &result {
    Expr::List(items) if !items.is_empty() => items,
    _ => return None,
  };
  // A flat `{x, y}` pair is a single curve.
  if items.len() == 2
    && let (Some(x), Some(y)) =
      (try_eval_to_f64(&items[0]), try_eval_to_f64(&items[1]))
  {
    return Some(vec![(x, y)]);
  }
  // `{{x1, y1}, …, {xn, yn}}` is one sample for each of n curves. Extra
  // grouping levels (`{{{x1, y1}}, {{x2, y2}}}`, as produced by
  // `{{fx, fy} /. rule}` curve slots) only steer styling and are flattened.
  let mut rows = Vec::new();
  if collect_pairs(&result, &mut rows) && !rows.is_empty() {
    Some(rows)
  } else {
    None
  }
}

/// Flatten a nested list of numeric `{x, y}` pairs into `out`. Returns
/// whether every leaf was a numeric pair.
fn collect_pairs(expr: &Expr, out: &mut Vec<(f64, f64)>) -> bool {
  let Expr::List(items) = expr else {
    return false;
  };
  if items.len() == 2
    && let (Some(x), Some(y)) =
      (try_eval_to_f64(&items[0]), try_eval_to_f64(&items[1]))
  {
    out.push((x, y));
    return true;
  }
  items.iter().all(|item| collect_pairs(item, out))
}

/// Whether an argument is an iterator (`{v, vmin, vmax}`) rather than an
/// option rule — what tells a second parameter apart from the trailing
/// options of a one-parameter plot.
fn is_iterator_spec(arg: &Expr) -> bool {
  matches!(arg, Expr::List(items)
    if items.len() == 3 && matches!(items[0], Expr::Identifier(_)))
}

/// Parameter values at which an `Exclusions` equation (`lhs == rhs`) holds,
/// or that `Exclusions` lists as plain numbers.
/// Wolfram removes these points from the curve, so the plotted line breaks
/// there instead of being drawn straight across the gap. Roots are located
/// as sign changes of `lhs - rhs` on a fine grid and pinned down by
/// bisection; sign changes through a pole (where the difference does not
/// actually vanish) are not exclusions and are left to the discontinuity
/// handling of the sampler.
pub(crate) fn exclusion_roots(
  options: &[Expr],
  var: &str,
  t_min: f64,
  t_max: f64,
) -> Vec<f64> {
  // `option_name_value` evaluates a delayed value, so only look at the
  // `Exclusions` rule itself — not at e.g. an `EvaluationMonitor :> …`.
  let Some(spec) = options.iter().find_map(|opt| {
    let names_exclusions = matches!(opt,
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Exclusions"));
    if !names_exclusions {
      return None;
    }
    match crate::functions::graphics::option_name_value(opt) {
      Some(("Exclusions", value)) => Some(value.into_owned()),
      _ => None,
    }
  }) else {
    return Vec::new();
  };
  let conditions: Vec<Expr> = match spec {
    Expr::List(ref items) => items.iter().cloned().collect(),
    other => vec![other],
  };
  let mut roots = Vec::new();
  for condition in &conditions {
    let (lhs, rhs) = match condition {
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && matches!(operators[0], crate::syntax::ComparisonOp::Equal) =>
      {
        (&operands[0], &operands[1])
      }
      Expr::FunctionCall { name, args }
        if name == "Equal" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      // A bare number `c` is shorthand for `var == c`.
      other => {
        if let Some(c) = try_eval_to_f64(other)
          && c >= t_min
          && c <= t_max
        {
          roots.push(c);
        }
        continue;
      }
    };
    let diff = |t: f64| -> Option<f64> {
      Some(evaluate_at_point(lhs, var, t)? - evaluate_at_point(rhs, var, t)?)
    };
    const GRID: usize = 2000;
    let step = (t_max - t_min) / GRID as f64;
    let mut prev = (t_min, diff(t_min));
    for i in 1..=GRID {
      let t = t_min + i as f64 * step;
      let cur = (t, diff(t));
      if let (Some(a), Some(b)) = (prev.1, cur.1) {
        if a == 0.0 {
          roots.push(prev.0);
        } else if a.signum() != b.signum() && b != 0.0 {
          let (mut lo, mut hi, mut flo) = (prev.0, cur.0, a);
          for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            match diff(mid) {
              Some(f) if f.signum() == flo.signum() => {
                lo = mid;
                flo = f;
              }
              Some(_) => hi = mid,
              None => break,
            }
          }
          let root = f64::midpoint(lo, hi);
          if diff(root)
            .is_some_and(|f| f.abs() < 1e-6 * (1.0 + a.abs().max(b.abs())))
          {
            roots.push(root);
          }
        }
      }
      prev = cur;
    }
  }
  roots
}

/// Break the sampled curves at the excluded parameter values by inserting an
/// empty sample at each one.
fn insert_exclusion_gaps(samples: &mut Vec<Sample>, roots: &[f64]) {
  if roots.is_empty() {
    return;
  }
  samples.extend(roots.iter().map(|&t| (t, None)));
  samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
}

/// ParametricPlot[{fx[t], fy[t]}, {t, tmin, tmax}]
/// Whether `body` is a two-slot list whose slots each evaluate, at the
/// parameter value `t`, to a numeric coordinate pair — i.e. it denotes two
/// curves (`{c1[t], c2[t]}`) rather than the components of one.
fn slots_yield_points(body: &Expr, var: &str, t: f64) -> bool {
  let Expr::List(items) = body else {
    return false;
  };
  items.len() == 2
    && items.iter().all(|item| {
      let sub = substitute_var(item, var, &Expr::Real(t));
      matches!(
        evaluate_expr_to_expr(&sub),
        Ok(Expr::List(ref pt)) if pt.len() == 2
          && pt.iter().all(|c| try_eval_to_f64(c).is_some())
      )
    })
}

pub fn parametric_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  // Two iterators make the plot a *region*: the image of the parameter
  // rectangle under `{fx, fy}`, not a curve.
  if args.len() > 2 && is_iterator_spec(&args[2]) {
    return parametric_region_ast(body, iter_spec, &args[2], &args[3..]);
  }

  let (mut plot_opts, overrides) = parse_options(&args[2..]);

  // Parse iterator: {t, tmin, tmax}
  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "ParametricPlot")?;

  // Collect curve bodies. Wolfram accepts a single curve `{fx, fy}` or
  // any number of curves nested to any depth, e.g.
  // `{{fx1,fy1}, {fx2,fy2}, …}` and `{Table[{fx, fy}, {i, …}]}` — the
  // extra grouping levels only matter for styling, so they are flattened
  // away here.
  let mut syntactic = Vec::new();
  let is_list_body = matches!(body, Expr::List(_));
  let syntactic_ok = is_list_body && collect_curves(body, &mut syntactic);
  // Whether the syntactic body is *unambiguously* a group of curves (every
  // element itself curve-shaped) rather than a `{fx, fy}` pair resolved
  // only through `collect_curves`'s ambiguous 2-item fallback. A body like
  // `{If[cond1, {fx1, fy1}, {}], If[cond2, {fx2, fy2}, {}]}` — the common
  // "conditionally hidden curve" idiom — has two non-list elements at the
  // syntactic level (the `If[…]` calls), so it fails this test and falls
  // through the same fallback as a genuine `{fx, fy}` pair; only evaluating
  // it (below) reveals the two `If`s actually resolve to a curve and an
  // empty slot.
  let body_is_group = is_curve_group(body);

  // The first argument is held, so a generated or conditionally-selected
  // curve list (`{Table[{fx, fy}, {i, …}]}`, `{If[cond, {fx, fy}, {}], …}`)
  // only reveals its true shape once evaluated. Evaluate it (the plot
  // variable stays symbolic) and retry, matching Wolfram, which samples the
  // held body rather than requiring `Evaluate[…]` around it. Attempted
  // whenever the syntactic body didn't already resolve unambiguously, since
  // an ambiguous `{fx, fy}`-shaped syntactic parse may really be a group of
  // curves that only evaluation can reveal.
  let evaluated_body: Option<Expr> = if is_list_body && !body_is_group {
    Some(crate::functions::plot::eval_body_var_symbolic(
      body, &var_name,
    ))
  } else {
    None
  };
  // Prefer the evaluated structure once it turns out to unambiguously be a
  // curve group — the syntactic Pair fallback above cannot distinguish a
  // real `{fx, fy}` pair from two conditionally-hidden curves, but a group
  // that only appears after evaluation is exactly that ambiguous case
  // resolving.
  let mut evaluated_group = Vec::new();
  let evaluated_group_ok = evaluated_body.as_ref().is_some_and(|ev| {
    is_curve_group(ev) && collect_curves(ev, &mut evaluated_group)
  });

  // A non-list body (`{fx, fy} /. rules`, `Evaluate[…]`, a function call…)
  // may still evaluate — with the plot variable symbolic — to a curve list,
  // and its held symbols (`xp` standing for an expression in `t`) only
  // reveal the variable once evaluated. Evaluating first matches Wolfram; a
  // body that stays opaque (`f[t]`) is sampled whole below.
  let evaluated_nonlist: Option<Expr> = if is_list_body {
    None
  } else {
    Some(crate::functions::plot::eval_body_var_symbolic(
      body, &var_name,
    ))
    .filter(|ev| matches!(ev, Expr::List(_)))
  };
  let mut nonlist_curves = Vec::new();
  let nonlist_ok = evaluated_nonlist
    .as_ref()
    .is_some_and(|ev| collect_curves(ev, &mut nonlist_curves));

  // A body that is a group of curves keeps its per-slot `PlotStyle` cycle
  // even when only one slot ended up drawn (`{c[t], If[cond, d[t]]}` with
  // the condition false): the visible curve still takes the first style.
  // A `{p1[t], p2[t]}` body whose two slots are opaque calls (or sums with
  // them) that return a *point* per sample is two curves, not an `{fx, fy}`
  // pair; only a numeric probe can tell.
  let vector_pair = is_list_body
    && !nonlist_ok
    && !body_is_group
    && !evaluated_group_ok
    && slots_yield_points(body, &var_name, f64::midpoint(t_min, t_max));
  let from_group = body_is_group || evaluated_group_ok || vector_pair;

  let curves: Vec<CurveSrc> = if nonlist_ok {
    nonlist_curves
  } else if vector_pair {
    vec![CurveSrc::Whole(body)]
  } else if evaluated_group_ok {
    evaluated_group
  } else if syntactic_ok {
    syntactic
  } else if is_list_body {
    let mut collected = Vec::new();
    let ok = evaluated_body
      .as_ref()
      .is_some_and(|ev| collect_curves(ev, &mut collected));
    // Only compound slots can resolve to coordinates later (a list, or a
    // call/product such as `{k*c[t], A . c[t]}` returning a point per
    // sample); a body of bare atoms (`{1, 2, 3}`) is not a curve
    // specification at all.
    let has_slots = matches!(body, Expr::List(items)
      if items.iter().any(|i| !matches!(i,
        Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_)
        | Expr::BigFloat(..) | Expr::String(_) | Expr::Identifier(_))));
    if ok {
      collected
    } else if !has_slots {
      return Err(InterpreterError::EvaluationError(
        "ParametricPlot: first argument must be {fx, fy}".into(),
      ));
    } else {
      // Curve slots such as `{{fx, fy} /. FindRoot[…]}` only turn into
      // coordinates once the parameter is numeric; sample the whole body.
      vec![CurveSrc::Whole(body)]
    }
  } else {
    // A non-list body (e.g. `f[t]` or `BSplineFunction[…][t]`) is a curve
    // whose coordinate pair only materialises once `t` is numeric; sample
    // the whole expression instead of decomposing it into components.
    vec![CurveSrc::Whole(body)]
  };

  // Sample at least NUM_SAMPLES points; an explicit larger PlotPoints
  // raises the resolution. (Wolfram refines a coarse PlotPoints setting
  // adaptively, so a smaller explicit value must not reduce smoothness.)
  // The uniform grid is the starting point; intervals where it would
  // render the curve as a polygon get bisected on top of it.
  let num_samples = plot_opts.plot_points.max(NUM_SAMPLES);
  let max_total = num_samples.saturating_mul(2);
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(curves.len());
  let mut all_params: Vec<Vec<f64>> = Vec::with_capacity(curves.len());
  let excluded = exclusion_roots(&args[2..], &var_name, t_min, t_max);

  for curve in &curves {
    // Each sample yields one coordinate pair, or one pair per curve when
    // the whole body is sampled (e.g. `ReIm[{c1, c2, c3}]`).
    let mut samples = match curve {
      CurveSrc::Pair(fx, fy) => {
        // ParametricPlot is HoldFirst-like, so a curve component passed as
        // a bare symbol or a `Part` extraction (e.g. `s[[1, 1, 2]]` off a
        // `DSolve` result) reaches here unevaluated and textually free of
        // the plot variable. Sampling substitutes the variable into the
        // component and evaluates, but a held reference like that contains
        // no occurrence of the variable to substitute — evaluating it just
        // looks the reference up again with the variable still symbolic,
        // never producing a number, so every sample is dropped and the
        // curve renders empty. When a component does not textually mention
        // the plot variable, evaluate it once with that variable kept
        // symbolic first, so an assigned symbol or a part extraction
        // expands to its definition (and reveals the variable) before
        // per-sample substitution — mirroring `Plot`'s own fix for the
        // same shape of body.
        let fx_storage;
        let fx: &Expr =
          if crate::functions::plot::expr_mentions_var(fx, &var_name) {
            fx
          } else {
            fx_storage =
              crate::functions::plot::eval_body_var_symbolic(fx, &var_name);
            &fx_storage
          };
        let fy_storage;
        let fy: &Expr =
          if crate::functions::plot::expr_mentions_var(fy, &var_name) {
            fy
          } else {
            fy_storage =
              crate::functions::plot::eval_body_var_symbolic(fy, &var_name);
            &fy_storage
          };
        adaptive_samples(
          |t| match (
            evaluate_at_point(fx, &var_name, t),
            evaluate_at_point(fy, &var_name, t),
          ) {
            (Some(x), Some(y)) => Some(vec![(x, y)]),
            _ => None,
          },
          t_min,
          t_max,
          num_samples,
          max_total,
        )
      }
      CurveSrc::Whole(b) => adaptive_samples(
        |t| sample_whole_rows(b, &var_name, t),
        t_min,
        t_max,
        num_samples,
        max_total,
      ),
    };
    insert_exclusion_gaps(&mut samples, &excluded);
    let (series, params) = samples_to_series(samples);
    all_params.extend(std::iter::repeat_n(params, series.len()));
    all_points.extend(series);
  }

  // Compute ranges (explicit PlotRange components override the data extents)
  let (data_x, data_y) = compute_data_ranges(&all_points);
  // A `PlotStyle` list applied to a single curve is one combined directive
  // set, not a per-curve cycle: `ParametricPlot[c, …, PlotStyle -> {Thick,
  // Red}]` draws one thick red curve.
  if all_points.len() == 1 && !from_group {
    plot_opts.plot_style =
      crate::functions::plot::collapse_style_for_single_series(
        &plot_opts.plot_style,
      );
  }
  let data_y = adjust_y_range_for_filling_opts(&plot_opts, data_y);
  let (x_range, y_range) =
    apply_ranges_and_aspect(&mut plot_opts, &overrides, data_x, data_y);

  // `ColorFunction -> f` colors the curve along its length by `f[x, y, t]`:
  // draw it as runs of constant color, one series per run. A `Show` merge
  // gets the coloured runs, since the parameter `f` needs is gone by then.
  let mut colored_runs: Option<(Vec<Vec<(f64, f64)>>, Vec<SeriesStyle>)> = None;
  let svg = if let Some(cf) = plot_opts.color_function.take() {
    let (runs, colors, owners) = color_function_series(
      &all_points,
      &all_params,
      &cf,
      plot_opts.color_function_scaling,
      x_range,
      y_range,
      (t_min, t_max),
    );
    let base = plot_opts.plot_style.clone();
    plot_opts.plot_style = colors
      .iter()
      .zip(&owners)
      .map(|(&(r, g, b), &curve)| {
        // Runs keep the weight and opacity of the curve they came from.
        let mut st = if base.is_empty() {
          SeriesStyle::default()
        } else {
          base[curve % base.len()].clone()
        };
        st.color = Some(crate::functions::graphics::Color::new(
          f64::from(r) / 255.0,
          f64::from(g) / 255.0,
          f64::from(b) / 255.0,
        ));
        st
      })
      .collect();
    let svg = generate_svg_with_filling(&runs, x_range, y_range, &plot_opts);
    colored_runs =
      Some((runs, std::mem::replace(&mut plot_opts.plot_style, base)));
    plot_opts.color_function = Some(cf);
    svg?
  } else {
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?
  };

  // Attach the sampled curves as a PlotSource so `Show` can merge this plot
  // with other graphics (re-rendering the curves as Line primitives). The
  // style goes with them: a curve merged by `Show` keeps the colour and
  // weight `PlotStyle` gave it, as it does when drawn on its own.
  let mut options = crate::functions::plot::explicit_options(args);
  let (points, styles) = match &colored_runs {
    Some((runs, styles)) => {
      options.retain(|o| {
        !matches!(o, Expr::Rule { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n)
            if n == "ColorFunction" || n == "ColorFunctionScaling"))
      });
      (runs, styles)
    }
    None => (&all_points, &plot_opts.plot_style),
  };
  let source = build_plot_source(
    points,
    styles,
    x_range,
    y_range,
    (plot_opts.svg_width, plot_opts.svg_height),
    &[false],
    plot_opts.filling,
    plot_opts.filling_style,
    options,
  );
  Ok(crate::graphics_result_with_source(svg, source))
}

/// Samples per parameter direction for a two-parameter (region) plot.
/// Wolfram starts from a coarse grid and refines it adaptively; sampling a
/// finer uniform grid gets to the same smooth boundary without the
/// refinement machinery.
const REGION_GRID: usize = 33;

/// How many times a bent grid cell of a region plot may split by default.
const REGION_MAX_RECURSION: usize = 2;

/// Evaluate a `{fx, fy}` body at one point of the parameter rectangle.
fn evaluate_at_uv(
  body: &Expr,
  uvar: &str,
  u: f64,
  vvar: &str,
  v: f64,
) -> Option<(f64, f64)> {
  let sub = substitute_var(body, uvar, &Expr::Real(u));
  let sub = substitute_var(&sub, vvar, &Expr::Real(v));
  match evaluate_expr_to_expr(&sub).ok()? {
    Expr::List(ref items) if items.len() == 2 => {
      Some((try_eval_to_f64(&items[0])?, try_eval_to_f64(&items[1])?))
    }
    _ => None,
  }
}

/// `ParametricPlot[{fx, fy}, {u, umin, umax}, {v, vmin, vmax}]` — the image
/// of the parameter rectangle, drawn the way Wolfram draws it: the quads of
/// the sampled grid filled at 30% opacity in the plot colour, with the image
/// of the rectangle's boundary stroked on top.
///
/// The picture is assembled as a `Graphics[…]` of primitives rather than
/// rendered directly, so it carries a structure that `Show` can merge with
/// the other layers of a composite figure.
fn parametric_region_ast(
  body: &Expr,
  u_spec: &Expr,
  v_spec: &Expr,
  opt_args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (plot_opts, _) = parse_options(opt_args);
  // A body naming a symbol whose value depends on `u`/`v` is read through
  // that value first (see `resolve_indirect_plot_body`).
  let var_of = |spec: &Expr| match spec {
    Expr::List(items) => match items.first() {
      Some(Expr::Identifier(name)) => Some(name.clone()),
      _ => None,
    },
    _ => None,
  };
  let resolved_body;
  let body = match (var_of(u_spec), var_of(v_spec)) {
    (Some(u), Some(v)) => {
      let vars = [u.as_str(), v.as_str()];
      let resolve = |e: &Expr| match e {
        Expr::Identifier(_) => {
          crate::functions::plot::resolve_indirect_plot_body(e, &vars)
            .unwrap_or_else(|| e.clone())
        }
        other => other.clone(),
      };
      resolved_body = match body {
        Expr::List(items) => Expr::List(
          items
            .iter()
            .map(|it| match it {
              Expr::List(pair) => {
                Expr::List(pair.iter().map(resolve).collect())
              }
              other => resolve(other),
            })
            .collect(),
        ),
        other => resolve(other),
      };
      &resolved_body
    }
    _ => body,
  };
  // The body is held, so a generator such as `Table[{fx, fy}, {j, n}]` only
  // takes the shape of a list of regions once evaluated (with `u` and `v`
  // staying symbolic).
  let generated;
  let body = match (body, var_of(u_spec), var_of(v_spec)) {
    (Expr::FunctionCall { .. }, Some(u), Some(v)) => {
      match crate::functions::plot::eval_body_vars_symbolic(
        body,
        &[u.as_str(), v.as_str()],
      ) {
        evaluated @ Expr::List(_) => {
          generated = evaluated;
          &generated
        }
        _ => body,
      }
    }
    _ => body,
  };
  // `{{fx1, fy1}, {fx2, fy2}, …}` is several regions drawn in one plot, each
  // in the next colour of the cycle.
  // A one-element list `{{fx, fy}}` is a single region.
  let bodies: Vec<&Expr> = match body {
    Expr::List(items)
      if !items.is_empty()
        && items
          .iter()
          .all(|b| matches!(b, Expr::List(p) if p.len() == 2)) =>
    {
      items.iter().collect()
    }
    _ => vec![body],
  };
  let mut items = Vec::with_capacity(bodies.len());
  for (i, b) in bodies.iter().enumerate() {
    items.push(parametric_region_layer(
      b, u_spec, v_spec, opt_args, &plot_opts, i,
    )?);
  }

  // Wolfram draws a ParametricPlot with axes and equally scaled ones; the
  // caller's own options come last so they win.
  let mut graphics_args = vec![Expr::List(items.into())];
  graphics_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
  graphics_args.push(rule_expr(id_expr("AspectRatio"), id_expr("Automatic")));
  graphics_args.extend(opt_args.iter().cloned());
  let mut result = crate::functions::graphics::graphics_ast(&graphics_args)?;
  // Keep the primitives on the rendering so `Show` can draw the region
  // together with the other layers of a composite figure instead of
  // dropping it for having no plot data.
  if let Expr::Graphics { structure, .. } = &mut result {
    *structure = Some(Box::new(crate::plot_option_list(call(
      "Graphics",
      graphics_args,
    ))));
  }
  Ok(result)
}

/// `Mesh -> {m, n}`, `MeshStyle` and `MeshShading` of a two-parameter
/// `ParametricPlot`: how many mesh lines cut each parameter range, how they
/// are drawn, and the colours of the cells between them.
struct RegionMesh {
  lines_u: usize,
  lines_v: usize,
  style_u: Expr,
  style_v: Expr,
  /// `MeshShading` colours, rows following `u` and columns `v`.
  shading: Vec<Vec<Option<Expr>>>,
}

impl RegionMesh {
  fn shading_at(&self, i: usize, j: usize) -> Option<Expr> {
    let row = self.shading.get(i % self.shading.len().max(1))?;
    row.get(j % row.len().max(1))?.clone()
  }
}

fn region_mesh_spec(opt_args: &[Expr]) -> Option<RegionMesh> {
  let option = |wanted: &str| {
    opt_args.iter().find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      (name == wanted).then(|| {
        let value = value.into_owned();
        evaluate_expr_to_expr(&value).unwrap_or(value)
      })
    })
  };
  let count = |e: &Expr| match e {
    Expr::Integer(n) if *n >= 0 => Some(*n as usize),
    _ => None,
  };
  let (mut lines_u, mut lines_v) = (None, None);
  match option("Mesh").as_ref() {
    Some(Expr::List(items)) if items.len() == 2 => {
      lines_u = count(&items[0]);
      lines_v = count(&items[1]);
    }
    Some(e) => {
      lines_u = count(e);
      lines_v = lines_u;
    }
    None => {}
  }
  let color_of = |e: &Expr| {
    let c = crate::functions::graphics::parse_color(e)?;
    Some(call(
      "RGBColor",
      vec![Expr::Real(c.r), Expr::Real(c.g), Expr::Real(c.b)],
    ))
  };
  // A list of rows is a 2D array of colours; a flat list of colours is a
  // single row.
  let shading: Vec<Vec<Option<Expr>>> = match option("MeshShading").as_ref() {
    Some(Expr::List(rows))
      if rows.iter().all(|r| matches!(r, Expr::List(_))) =>
    {
      rows
        .iter()
        .map(|r| match r {
          Expr::List(cells) => cells.iter().map(&color_of).collect(),
          _ => Vec::new(),
        })
        .collect()
    }
    Some(Expr::List(cells)) => vec![cells.iter().map(&color_of).collect()],
    _ => Vec::new(),
  };
  if lines_u.is_none() && lines_v.is_none() && shading.is_empty() {
    return None;
  }
  // Without a `Mesh` count, shading alone still splits the picture along
  // the shading array.
  let lines_u = lines_u.unwrap_or_else(|| shading.len().saturating_sub(1));
  let lines_v = lines_v.unwrap_or_else(|| {
    shading.first().map_or(0, |r| r.len().saturating_sub(1))
  });
  let style = |k: usize| match option("MeshStyle").as_ref() {
    Some(Expr::List(items)) if !items.is_empty() => {
      let item = &items[k % items.len()];
      vec![item.clone()]
    }
    Some(e) => vec![e.clone()],
    None => Vec::new(),
  };
  let group = |k: usize| {
    let mut directives = style(k);
    if directives.is_empty() {
      directives.push(call1("GrayLevel", Expr::Real(0.4)));
    }
    Expr::List(directives.into())
  };
  Some(RegionMesh {
    lines_u,
    lines_v,
    style_u: group(0),
    style_v: group(1),
    shading,
  })
}

/// The colour cycle `Plot`-family functions assign to successive curves.
const REGION_COLORS: [(f64, f64, f64); 6] = [
  (0.24, 0.6, 0.8),
  (0.88, 0.58, 0.17),
  (0.56, 0.69, 0.20),
  (0.85, 0.32, 0.10),
  (0.42, 0.28, 0.61),
  (0.56, 0.69, 0.80),
];

/// Whether `region` (a `RegionFunction -> f` value, called as
/// `f[x, y, u, v]`) accepts the point.
fn parametric_region_allows(
  region: &Expr,
  (x, y): (f64, f64),
  u: f64,
  v: f64,
) -> bool {
  let call = Expr::CurriedCall {
    func: Box::new(region.clone()),
    args: vec![Expr::Real(x), Expr::Real(y), Expr::Real(u), Expr::Real(v)],
  };
  matches!(
    evaluate_expr_to_expr(&call),
    Ok(Expr::Identifier(ref s)) if s == "True"
  )
}

/// Emit the quad with `corners` (images of the parameter cell starting at
/// `origin` with size `step`, in the order `(0,0) (1,0) (1,1) (0,1)`), first
/// splitting it into four while the image of its centre or an edge midpoint
/// strays more than `tolerance` from the bilinear interpolation of the
/// corners and `depth` allows.
fn refine_region_cell(
  corners: [(f64, f64); 4],
  origin: (f64, f64),
  step: (f64, f64),
  depth: usize,
  tolerance: f64,
  sample: &dyn Fn(f64, f64) -> Option<(f64, f64)>,
  emit: &mut dyn FnMut([(f64, f64); 4], (f64, f64)),
) {
  if depth > 0 {
    let (u0, v0) = origin;
    let (hu, hv) = (step.0 / 2.0, step.1 / 2.0);
    let mid = |a: (f64, f64), b: (f64, f64)| {
      (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1))
    };
    // (parameter point, bilinear estimate of its image)
    let probes = [
      ((u0 + hu, v0), mid(corners[0], corners[1])),
      ((u0 + step.0, v0 + hv), mid(corners[1], corners[2])),
      ((u0 + hu, v0 + step.1), mid(corners[3], corners[2])),
      ((u0, v0 + hv), mid(corners[0], corners[3])),
      (
        (u0 + hu, v0 + hv),
        mid(mid(corners[0], corners[2]), mid(corners[1], corners[3])),
      ),
    ];
    let mut images = [(0.0, 0.0); 5];
    let mut bent = false;
    let mut holed = false;
    for (k, ((u, v), estimate)) in probes.iter().enumerate() {
      if let Some(p) = sample(*u, *v) {
        images[k] = p;
        bent |= (p.0 - estimate.0).hypot(p.1 - estimate.1) > tolerance;
      } else {
        // A hole inside the cell: keep the cell as the grid has it.
        holed = true;
        break;
      }
    }
    if bent && !holed {
      let [bottom, right, top, left, centre] = images;
      let sub = [
        ([corners[0], bottom, centre, left], (u0, v0)),
        ([bottom, corners[1], right, centre], (u0 + hu, v0)),
        ([centre, right, corners[2], top], (u0 + hu, v0 + hv)),
        ([left, centre, top, corners[3]], (u0, v0 + hv)),
      ];
      for (quad, o) in sub {
        refine_region_cell(
          quad,
          o,
          (hu, hv),
          depth - 1,
          tolerance,
          sample,
          emit,
        );
      }
      return;
    }
  }
  emit(corners, (origin.0 + step.0 / 2.0, origin.1 + step.1 / 2.0));
}

/// The primitives (filled quads plus stroked boundary) of one region body.
/// One filled cell of a two-parameter `ParametricPlot`, with the image
/// (`xy`) and parameter (`uv`) coordinates of its centre for `ColorFunction`.
struct RegionQuad {
  polygon: Expr,
  xy: (f64, f64),
  uv: (f64, f64),
}

fn parametric_region_layer(
  body: &Expr,
  u_spec: &Expr,
  v_spec: &Expr,
  opt_args: &[Expr],
  plot_opts: &PlotOptions,
  layer: usize,
) -> Result<Expr, InterpreterError> {
  let (uvar, u_min, u_max) = parse_iterator(u_spec, "ParametricPlot")?;
  let (vvar, v_min, v_max) = parse_iterator(v_spec, "ParametricPlot")?;
  let region_function = opt_args.iter().find_map(|opt| {
    let (name, value) = crate::functions::graphics::option_name_value(opt)?;
    (name == "RegionFunction").then(|| value.into_owned())
  });

  // An explicit `PlotPoints` sets the grid; the option's plot default is a
  // curve-sample count, so it cannot be used as the grid size directly.
  let grid = opt_args
    .iter()
    .find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      // `plp*{10, 10}` and the like: the option value may still need
      // evaluating to an integer or a list of them.
      let value = if name == "PlotPoints" {
        evaluate_expr_to_expr(&value).unwrap_or_else(|_| value.into_owned())
      } else {
        value.into_owned()
      };
      match (name, &value) {
        ("PlotPoints", Expr::Integer(n)) if *n >= 2 => Some(*n as usize),
        // `{nu, nv}`: one square grid with the finer of the two counts.
        ("PlotPoints", Expr::List(ns)) => ns
          .iter()
          .filter_map(|n| match n {
            Expr::Integer(n) if *n >= 2 => Some(*n as usize),
            _ => None,
          })
          .max(),
        _ => None,
      }
    })
    .unwrap_or(REGION_GRID);

  let du = (u_max - u_min) / (grid - 1) as f64;
  let dv = (v_max - v_min) / (grid - 1) as f64;
  let mut points: Vec<Vec<Option<(f64, f64)>>> = Vec::with_capacity(grid);
  for i in 0..grid {
    let u = u_min + i as f64 * du;
    let mut row = Vec::with_capacity(grid);
    for j in 0..grid {
      let v = v_min + j as f64 * dv;
      row.push(evaluate_at_uv(body, &uvar, u, &vvar, v).filter(|&pt| {
        region_function
          .as_ref()
          .is_none_or(|rf| parametric_region_allows(rf, pt, u, v))
      }));
    }
    points.push(row);
  }

  let pt_expr =
    |(x, y): (f64, f64)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());

  // `MaxRecursion -> n` lets a grid cell whose image bends away from its
  // bilinear interpolation split into four, up to `n` times — a
  // parametrization such as `u^2` is far too curved for the coarse grid.
  let max_recursion = opt_args
    .iter()
    .find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      match (name, &*value) {
        ("MaxRecursion", Expr::Integer(n)) if *n >= 0 => Some(*n as usize),
        _ => None,
      }
    })
    .unwrap_or(REGION_MAX_RECURSION);
  let finite =
    |p: Option<(f64, f64)>| p.filter(|(x, y)| x.is_finite() && y.is_finite());
  let sample = |u: f64, v: f64| {
    finite(evaluate_at_uv(body, &uvar, u, &vvar, v).filter(|&pt| {
      region_function
        .as_ref()
        .is_none_or(|rf| parametric_region_allows(rf, pt, u, v))
    }))
  };
  let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
  for pt in points.iter().flatten().filter_map(|p| finite(*p)) {
    lo = (lo.0.min(pt.0), lo.1.min(pt.1));
    hi = (hi.0.max(pt.0), hi.1.max(pt.1));
  }
  let tolerance = 0.005 * ((hi.0 - lo.0).hypot(hi.1 - lo.1));

  // One quad per (possibly refined) cell of the `g`-point grid over
  // `[ua, ub] x [va, vb]`; a cell with a point the body could not be
  // evaluated at (a singularity, a complex value) is left out.
  let quads_over =
    |(ua, ub): (f64, f64), (va, vb): (f64, f64), g: usize| -> Vec<RegionQuad> {
      let (dg_u, dg_v) =
        ((ub - ua) / (g - 1) as f64, (vb - va) / (g - 1) as f64);
      let grid_points: Vec<Vec<Option<(f64, f64)>>> = (0..g)
        .map(|i| {
          (0..g)
            .map(|j| finite(sample(ua + i as f64 * dg_u, va + j as f64 * dg_v)))
            .collect()
        })
        .collect();
      let mut quads = Vec::new();
      let mut emit = |corners: [(f64, f64); 4], uv: (f64, f64)| {
        let n = corners.len() as f64;
        let xy = (
          corners.iter().map(|c| c.0).sum::<f64>() / n,
          corners.iter().map(|c| c.1).sum::<f64>() / n,
        );
        quads.push(RegionQuad {
          polygon: call1(
            "Polygon",
            Expr::List(corners.iter().map(|c| pt_expr(*c)).collect()),
          ),
          xy,
          uv,
        });
      };
      for i in 0..g - 1 {
        for j in 0..g - 1 {
          let corners = [
            grid_points[i][j],
            grid_points[i + 1][j],
            grid_points[i + 1][j + 1],
            grid_points[i][j + 1],
          ];
          let Some(c) =
            corners.map(finite).into_iter().collect::<Option<Vec<_>>>()
          else {
            continue;
          };
          refine_region_cell(
            [c[0], c[1], c[2], c[3]],
            (ua + i as f64 * dg_u, va + j as f64 * dg_v),
            (dg_u, dg_v),
            max_recursion,
            tolerance,
            &sample,
            &mut emit,
          );
        }
      }
      quads
    };
  let mesh = region_mesh_spec(opt_args);
  let mut quads = Vec::new();
  let mut mesh_cells: Vec<Expr> = Vec::new();
  let mut mesh_lines: Vec<Expr> = Vec::new();
  if let Some(mesh) = &mesh {
    // Every cell between consecutive mesh lines is filled on its own, in the
    // colour `MeshShading` assigns it (cycling through the array), and the
    // mesh lines themselves are the images of the iso-`u` / iso-`v` lines.
    let (cells_u, cells_v) = (mesh.lines_u + 1, mesh.lines_v + 1);
    let cell_grid = (grid / cells_u.max(cells_v)).max(4);
    let edge = |k: usize, n: usize, lo: f64, hi: f64| {
      lo + (hi - lo) * k as f64 / n as f64
    };
    for i in 0..cells_u {
      for j in 0..cells_v {
        let polys = quads_over(
          (
            edge(i, cells_u, u_min, u_max),
            edge(i + 1, cells_u, u_min, u_max),
          ),
          (
            edge(j, cells_v, v_min, v_max),
            edge(j + 1, cells_v, v_min, v_max),
          ),
          cell_grid,
        );
        if polys.is_empty() {
          continue;
        }
        let polys: Vec<Expr> = polys.into_iter().map(|q| q.polygon).collect();
        let mut group = vec![call0("EdgeForm")];
        if let Some(shade) = mesh.shading_at(i, j) {
          group.push(shade);
          group.push(call1("Opacity", Expr::Real(1.0)));
        } else {
          let (r, g, b) = REGION_COLORS[layer % REGION_COLORS.len()];
          group.push(call(
            "RGBColor",
            vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
          ));
          group.push(call1("Opacity", Expr::Real(0.3)));
        }
        group.extend(polys);
        mesh_cells.push(Expr::List(group.into()));
      }
    }
    let per_line = grid * 4;
    for k in 1..cells_u {
      let u = edge(k, cells_u, u_min, u_max);
      let line: Vec<Expr> = (0..=per_line)
        .filter_map(|m| {
          finite(sample(u, edge(m, per_line, v_min, v_max))).map(pt_expr)
        })
        .collect();
      if line.len() > 1 {
        mesh_lines.push(Expr::List(
          vec![mesh.style_u.clone(), call1("Line", Expr::List(line.into()))]
            .into(),
        ));
      }
    }
    for k in 1..cells_v {
      let v = edge(k, cells_v, v_min, v_max);
      let line: Vec<Expr> = (0..=per_line)
        .filter_map(|m| {
          finite(sample(edge(m, per_line, u_min, u_max), v)).map(pt_expr)
        })
        .collect();
      if line.len() > 1 {
        mesh_lines.push(Expr::List(
          vec![mesh.style_v.clone(), call1("Line", Expr::List(line.into()))]
            .into(),
        ));
      }
    }
  } else {
    quads = quads_over((u_min, u_max), (v_min, v_max), grid);
  }
  if quads.is_empty() && mesh_cells.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot: the body evaluates to no points over that region".into(),
    ));
  }

  // The boundary of the picture is the image of the parameter rectangle's
  // boundary, walked once counter-clockwise, sampled as finely as the
  // refined cells so it follows them.
  let per_edge = (grid - 1) << max_recursion.min(4);
  let along =
    |t0: f64, t1: f64, k: usize| t0 + (t1 - t0) * k as f64 / per_edge as f64;
  let mut border_uv: Vec<(f64, f64)> = Vec::with_capacity(4 * per_edge + 1);
  border_uv.extend((0..=per_edge).map(|k| (u_min, along(v_min, v_max, k))));
  border_uv.extend((1..=per_edge).map(|k| (along(u_min, u_max, k), v_max)));
  border_uv
    .extend((0..per_edge).rev().map(|k| (u_max, along(v_min, v_max, k))));
  border_uv
    .extend((0..per_edge).rev().map(|k| (along(u_min, u_max, k), v_min)));
  let border: Vec<Expr> = border_uv
    .iter()
    .filter_map(|&(u, v)| sample(u, v))
    .map(pt_expr)
    .collect();

  let (r, g, b) = plot_opts
    .plot_style
    .get(layer)
    .and_then(|s| s.color.as_ref())
    .map_or(REGION_COLORS[layer % REGION_COLORS.len()], |c| {
      (c.r, c.g, c.b)
    });
  let color = call(
    "RGBColor",
    vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
  );
  // An `Opacity[a]` among the `PlotStyle` directives replaces the default
  // 30% fill.
  let fill_opacity = opt_args
    .iter()
    .find_map(|opt| {
      let (name, value) = crate::functions::graphics::option_name_value(opt)?;
      (name == "PlotStyle")
        .then(|| crate::functions::plot::parse_filling_style(&value))
        .flatten()
        .and_then(|fs| fs.opacity)
    })
    .unwrap_or(0.3);
  let opacity = call1("Opacity", Expr::Real(fill_opacity));
  let no_edges = call0("EdgeForm");

  let mut items = if !mesh_cells.is_empty() {
    mesh_cells
  } else if let Some(cf) = &plot_opts.color_function {
    // `ColorFunction -> f` paints every cell in the (opaque) color `f`
    // gives at the cell's centre; cells of one color share a group.
    let scale = |v: f64, lo: f64, hi: f64| {
      if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }
    };
    let scheme = crate::functions::field_plot::color_function_scheme_name(cf);
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let color_at = |q: &RegionQuad| -> (u8, u8, u8) {
      let (x, y) = q.xy;
      let (u, v) = q.uv;
      let (sx, sy) = (scale(x, lo.0, hi.0), scale(y, lo.1, hi.1));
      if let Some(name) = &scheme {
        return crate::functions::field_plot::apply_named_color_function(
          name, sx,
        );
      }
      let args = if plot_opts.color_function_scaling {
        vec![
          Expr::Real(sx),
          Expr::Real(sy),
          Expr::Real(scale(u, u_min, u_max)),
          Expr::Real(scale(v, v_min, v_max)),
        ]
      } else {
        vec![Expr::Real(x), Expr::Real(y), Expr::Real(u), Expr::Real(v)]
      };
      let applied = Expr::CurriedCall {
        func: Box::new(cf.clone()),
        args,
      };
      evaluate_expr_to_expr(&applied)
        .ok()
        .and_then(|r| crate::functions::graphics::parse_color(&r))
        .map_or((0, 0, 0), |c| (ch(c.r), ch(c.g), ch(c.b)))
    };
    let mut groups: Vec<((u8, u8, u8), Vec<Expr>)> = Vec::new();
    for q in quads {
      let c = color_at(&q);
      match groups.iter_mut().find(|(gc, _)| *gc == c) {
        Some((_, polys)) => polys.push(q.polygon),
        None => groups.push((c, vec![q.polygon])),
      }
    }
    groups
      .into_iter()
      .map(|((r, g, b), polys)| {
        let mut group = vec![
          call0("EdgeForm"),
          call(
            "RGBColor",
            vec![
              Expr::Real(f64::from(r) / 255.0),
              Expr::Real(f64::from(g) / 255.0),
              Expr::Real(f64::from(b) / 255.0),
            ],
          ),
          call1("Opacity", Expr::Real(1.0)),
        ];
        group.extend(polys);
        Expr::List(group.into())
      })
      .collect()
  } else {
    let mut face_group = vec![no_edges, color.clone(), opacity];
    face_group.extend(quads.into_iter().map(|q| q.polygon));
    vec![Expr::List(face_group.into())]
  };
  items.append(&mut mesh_lines);
  if border.len() > 1 {
    items.push(Expr::List(
      vec![
        color,
        call1("AbsoluteThickness", Expr::Real(2.0)),
        call1("Line", Expr::List(border.into())),
      ]
      .into(),
    ));
  }

  Ok(Expr::List(items.into()))
}

/// What `PolarAxes`, `PolarTicks` and `PolarGridLines` ask a `PolarPlot` to
/// draw: a circular angular axis with an optional radial axis, instead of the
/// Cartesian axes. Both two-part options are `{angular, radial}`.
#[allow(clippy::struct_excessive_bools)]
struct PolarAxesSpec {
  angular_grid: bool,
  radial_grid: bool,
  degree_labels: bool,
  radial_labels: bool,
}

impl PolarAxesSpec {
  /// `None` unless `PolarAxes` turns the polar axes on.
  fn from_options(opts: &[Expr]) -> Option<Self> {
    let value_of = |wanted: &str| {
      opts.iter().find_map(|opt| {
        let (name, value) = crate::functions::graphics::option_name_value(opt)?;
        (name == wanted).then(|| value.into_owned())
      })
    };
    let is_true = |e: &Expr| matches!(e, Expr::Identifier(n) if n == "True");
    let is_off = |e: &Expr| matches!(e, Expr::Identifier(n) if n == "None" || n == "False");
    let on = match &value_of("PolarAxes")? {
      Expr::List(items) => items.iter().any(is_true),
      other => is_true(other),
    };
    if !on {
      return None;
    }
    // The two parts of a `{angular, radial}` option; a lone value applies to
    // both, and an absent option leaves both at `default`.
    let pair = |name: &str, default: Expr| match &value_of(name) {
      Some(Expr::List(items)) if items.len() == 2 => {
        (items[0].clone(), items[1].clone())
      }
      Some(other) => (other.clone(), other.clone()),
      None => (default.clone(), default),
    };
    let auto = Expr::Identifier("Automatic".to_string());
    let none = Expr::Identifier("None".to_string());
    let (grid_angular, grid_radial) = pair("PolarGridLines", none);
    let (ticks_angular, ticks_radial) = pair("PolarTicks", auto);
    Some(Self {
      angular_grid: !is_off(&grid_angular),
      radial_grid: !is_off(&grid_radial),
      degree_labels: matches!(
        &ticks_angular,
        Expr::String(s) if s == "Degrees"
      ),
      radial_labels: !is_off(&ticks_radial),
    })
  }

  /// The primitives drawn under the curve, and the half-width of the square
  /// plot range that holds the axes and their labels. `r_max` is the largest
  /// radius the curve reaches.
  fn prolog(&self, r_max: f64) -> (Vec<Expr>, f64) {
    let step = crate::functions::plot::nice_step(r_max.max(1e-9), 4);
    let steps = (r_max / step - 1e-9).ceil().max(1.0) as usize;
    let radius = step * steps as f64;
    let point =
      |x: f64, y: f64| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
    let gray = |level: f64| call1("GrayLevel", Expr::Real(level));
    let line = |a: (f64, f64), b: (f64, f64)| {
      call1(
        "Line",
        Expr::List(vec![point(a.0, a.1), point(b.0, b.1)].into()),
      )
    };
    let mut items: Vec<Expr> = Vec::new();

    let mut grid = vec![gray(0.8)];
    if self.radial_grid {
      for k in 1..steps {
        grid.push(call(
          "Circle",
          vec![point(0.0, 0.0), Expr::Real(step * k as f64)],
        ));
      }
    }
    if self.angular_grid {
      for k in 0..12 {
        let a = std::f64::consts::PI / 6.0 * k as f64;
        grid.push(line((0.0, 0.0), (radius * a.cos(), radius * a.sin())));
      }
    }
    items.push(Expr::List(grid.into()));

    // The axes: the angular circle and the radial axis along angle zero.
    let mut axes = vec![gray(0.4)];
    axes.push(call("Circle", vec![point(0.0, 0.0), Expr::Real(radius)]));
    axes.push(line((0.0, 0.0), (radius, 0.0)));
    for k in 0..12 {
      let a = std::f64::consts::PI / 6.0 * k as f64;
      let (c, s) = (a.cos(), a.sin());
      axes.push(line(
        (radius * c, radius * s),
        (radius * 1.03 * c, radius * 1.03 * s),
      ));
    }
    items.push(Expr::List(axes.into()));

    let label = |text: String, at: (f64, f64), offset: (f64, f64)| {
      call(
        "Text",
        vec![
          Expr::String(text),
          point(at.0, at.1),
          point(offset.0, offset.1),
        ],
      )
    };
    let mut labels = vec![gray(0.4)];
    if self.degree_labels {
      for k in 0..12 {
        let a = std::f64::consts::PI / 6.0 * k as f64;
        labels.push(label(
          format!("{}°", 30 * k),
          (radius * 1.12 * a.cos(), radius * 1.12 * a.sin()),
          (0.0, 0.0),
        ));
      }
    }
    if self.radial_labels {
      for k in 1..=steps {
        let r = step * k as f64;
        let text = format!("{r:.6}");
        let text = text.trim_end_matches('0').trim_end_matches('.').to_string();
        labels.push(label(text, (r, 0.0), (0.0, 1.5)));
      }
    }
    items.push(Expr::List(labels.into()));
    (items, radius * 1.3)
  }
}

/// PolarPlot[r[theta], {theta, tmin, tmax}]
pub fn polar_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "PolarPlot requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];
  let iter_spec = &args[1];

  let (mut plot_opts, overrides) = parse_options(&args[2..]);

  let (var_name, t_min, t_max) = parse_iterator(iter_spec, "PolarPlot")?;

  // Collect function bodies
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  let num_samples = plot_opts.plot_points.max(NUM_SAMPLES);
  let step = (t_max - t_min) / (num_samples - 1) as f64;
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());

  for func_body in &bodies {
    let mut points = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
      let theta = t_min + i as f64 * step;
      if let Some(r) = evaluate_at_point(func_body, &var_name, theta) {
        let x = r * theta.cos();
        let y = r * theta.sin();
        points.push((x, y));
      } else {
        points.push((f64::NAN, f64::NAN));
      }
    }
    all_points.push(points);
  }

  let (mut data_x, mut data_y) = compute_data_ranges(&all_points);
  if let Some(axes) = PolarAxesSpec::from_options(&args[2..]) {
    let r_max = all_points
      .iter()
      .flatten()
      .map(|&(x, y)| x.hypot(y))
      .filter(|r| r.is_finite())
      .fold(0.0_f64, f64::max);
    let (prolog, extent) = axes.prolog(r_max);
    plot_opts.prolog.extend(prolog);
    plot_opts.axes = (false, false);
    plot_opts.clip_overlays = false;
    data_x = (-extent, extent);
    data_y = (-extent, extent);
  }
  // A `PlotStyle` list applied to a single curve is one combined directive
  // set, not a per-curve cycle: `PolarPlot[r, …, PlotStyle -> {Thick, Red}]`
  // draws one thick red curve.
  if all_points.len() == 1 {
    plot_opts.plot_style =
      crate::functions::plot::collapse_style_for_single_series(
        &plot_opts.plot_style,
      );
  }
  let data_y = adjust_y_range_for_filling_opts(&plot_opts, data_y);
  let (x_range, y_range) =
    apply_ranges_and_aspect(&mut plot_opts, &overrides, data_x, data_y);

  let svg =
    generate_svg_with_filling(&all_points, x_range, y_range, &plot_opts)?;

  // Attach the sampled curves as a PlotSource so `Show` can merge this plot
  // with other graphics (re-rendering the curves as Line primitives). The
  // style goes with them: a curve merged by `Show` keeps the colour and
  // weight `PlotStyle` gave it, as it does when drawn on its own.
  let source = build_plot_source(
    &all_points,
    &plot_opts.plot_style,
    x_range,
    y_range,
    (plot_opts.svg_width, plot_opts.svg_height),
    &[false],
    plot_opts.filling,
    plot_opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  Ok(crate::graphics_result_with_source(svg, source))
}

/// Compute x/y ranges from point data with 4% padding.
fn compute_data_ranges(
  all_points: &[Vec<(f64, f64)>],
) -> ((f64, f64), (f64, f64)) {
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;

  for series in all_points {
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

  if !x_min.is_finite() {
    x_min = -1.0;
    x_max = 1.0;
  }
  if !y_min.is_finite() {
    y_min = -1.0;
    y_max = 1.0;
  }

  let xr = x_max - x_min;
  let yr = y_max - y_min;
  let xp = if xr.abs() < f64::EPSILON {
    1.0
  } else {
    xr * 0.04
  };
  let yp = if yr.abs() < f64::EPSILON {
    1.0
  } else {
    yr * 0.04
  };

  ((x_min - xp, x_max + xp), (y_min - yp, y_max + yp))
}
