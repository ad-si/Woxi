//! `ScalingFunctions`: plots drawn on a transformed axis.
//!
//! A scaled axis plots `f[v]` instead of `v` and labels its ticks with the
//! original values `v`, exactly as the Wolfram Language does it: `Plot[g,
//! …, ScalingFunctions -> {fx, fy}]` is `Plot[fy[g /. x -> fx⁻¹[u]], {u,
//! …}]` drawn with ticks from `Charting`ScaledTicks[{f, f⁻¹}]`.
//!
//! The tick positions are a port of the Wolfram Language's own tick finder
//! for scaled axes (`Visualization`Divisions` with a `ScalingFunctions`
//! option, called by `Charting`ScaledTicks`), and the labels of its
//! `Charting`SimplePadding`. The log axes of `LogPlot`, `ListLogPlot`, … use
//! the same tick finder through [`log_axis_ticks`].

use crate::functions::plot_axes::AxisTicks;
use crate::syntax::Expr;

/// The transform of one scaled axis.
#[derive(Clone, Debug)]
pub(crate) enum Scaling {
  /// `"Log"`: the natural logarithm (drawn on the native log axes).
  Log,
  /// `"Reverse"`: `-v`.
  Reverse,
  /// `"Reciprocal"`: `1/v`.
  Reciprocal,
  /// `{f, f⁻¹}`: any function with its inverse.
  Custom { f: Expr, inverse: Expr },
}

/// Apply a function expression to a number.
fn apply_numeric(f: &Expr, v: f64) -> Option<f64> {
  let call = Expr::CurriedCall {
    func: Box::new(f.clone()),
    args: vec![Expr::Real(v)],
  };
  let r = crate::evaluator::evaluate_expr_to_expr(&call).ok()?;
  crate::functions::math_ast::try_eval_to_f64(&r).filter(|v| v.is_finite())
}

impl Scaling {
  /// One axis' `ScalingFunctions` entry: `Some(None)` for an unscaled axis
  /// (`None`, `Identity`, `"Linear"`), `None` when it is no scaling at all.
  fn parse_axis(e: &Expr) -> Option<Option<Scaling>> {
    match e {
      Expr::Identifier(s) if s == "None" || s == "Identity" => Some(None),
      Expr::String(s) => match s.as_str() {
        "Linear" => Some(None),
        "Log" | "Log10" | "Log2" => Some(Some(Scaling::Log)),
        "Reverse" => Some(Some(Scaling::Reverse)),
        "Reciprocal" => Some(Some(Scaling::Reciprocal)),
        _ => None,
      },
      Expr::List(items)
        if items.len() == 2 && items.iter().all(is_function) =>
      {
        Some(Some(Scaling::Custom {
          f: items[0].clone(),
          inverse: items[1].clone(),
        }))
      }
      _ => None,
    }
  }

  /// The `(x, y)` scalings of a `ScalingFunctions -> spec` option, or `None`
  /// when `spec` is not one. A single scale (`"Log"`, `{f, f⁻¹}`) is the
  /// y axis'.
  pub(crate) fn parse_option(
    spec: &Expr,
  ) -> Option<(Option<Scaling>, Option<Scaling>)> {
    if let Some(y) = Self::parse_axis(spec) {
      return Some((None, y));
    }
    match spec {
      Expr::List(items) if items.len() == 2 => {
        Some((Self::parse_axis(&items[0])?, Self::parse_axis(&items[1])?))
      }
      _ => None,
    }
  }

  pub(crate) fn is_log(&self) -> bool {
    matches!(self, Scaling::Log)
  }

  /// `f[v]`, when it is a real number.
  pub(crate) fn forward(&self, v: f64) -> Option<f64> {
    let r = match self {
      Scaling::Log => v.ln(),
      Scaling::Reverse => -v,
      Scaling::Reciprocal => 1.0 / v,
      Scaling::Custom { f, .. } => return apply_numeric(f, v),
    };
    r.is_finite().then_some(r)
  }

  /// `f⁻¹[v]`, when it is a real number.
  pub(crate) fn inverse(&self, v: f64) -> Option<f64> {
    let r = match self {
      Scaling::Log => v.exp(),
      Scaling::Reverse => -v,
      Scaling::Reciprocal => 1.0 / v,
      Scaling::Custom { inverse, .. } => return apply_numeric(inverse, v),
    };
    r.is_finite().then_some(r)
  }

  /// The function `f` as an expression, to apply to a plotted function.
  pub(crate) fn forward_expr(&self) -> Expr {
    let parse = |s: &str| {
      crate::syntax::string_to_expr(s).expect("scaling function parses")
    };
    match self {
      Scaling::Log => Expr::Identifier("Log".to_string()),
      Scaling::Reverse => parse("-#&"),
      Scaling::Reciprocal => parse("1/#&"),
      Scaling::Custom { f, .. } => f.clone(),
    }
  }

  /// The function `f⁻¹` as an expression.
  pub(crate) fn inverse_expr(&self) -> Expr {
    match self {
      Scaling::Log => Expr::Identifier("Exp".to_string()),
      Scaling::Custom { inverse, .. } => inverse.clone(),
      other => other.forward_expr(),
    }
  }

  /// Whether the scaled axis has a place for the original 0: an automatic
  /// list-plot range reaches out to it, and the other axis crosses there.
  /// A log or reciprocal axis has none, so its range is padded instead and
  /// the other axis runs along its low end.
  pub(crate) fn zero(&self) -> Option<f64> {
    self.forward(0.0)
  }

  /// The ticks of an axis scaled by `self` over the scaled range
  /// `min..max`, at scaled positions with original-value labels.
  pub(crate) fn ticks(&self, min: f64, max: f64) -> AxisTicks {
    let (min, max) = (min.min(max), min.max(max));
    if !(min.is_finite() && max.is_finite()) || max <= min {
      return AxisTicks::default();
    }
    match self {
      // `Charting`ScaledTicks[{-#&, -#&}]`: the plain ticks of the
      // negated range, negated back.
      Scaling::Reverse => {
        let t = scaled_ticks(&Identity, -max, -min, 6);
        AxisTicks {
          majors: t.majors.into_iter().map(|(p, l)| (-p, l)).collect(),
          minors: t.minors.into_iter().map(|p| -p).collect(),
        }
      }
      Scaling::Reciprocal => reciprocal_ticks(min, max),
      other => scaled_ticks(other, min, max, 6),
    }
  }
}

fn is_function(e: &Expr) -> bool {
  match e {
    Expr::Function { .. } | Expr::NamedFunction { .. } => true,
    Expr::FunctionCall { name, .. } => name == "Function",
    // A built-in function given by name (`{Log, Exp}`).
    Expr::Identifier(s) => s.chars().next().is_some_and(char::is_uppercase),
    _ => false,
  }
}

/// The ticks of a native log axis over the original range `min..max`
/// (`min > 0`): the scaled ticks of `{Log, Exp}`, at original positions.
pub(crate) fn log_axis_ticks(min: f64, max: f64) -> AxisTicks {
  if !(min > 0.0 && max > min && max.is_finite()) {
    return AxisTicks::default();
  }
  let t = scaled_ticks(&Scaling::Log, min.ln(), max.ln(), 6);
  AxisTicks {
    majors: t.majors.into_iter().map(|(p, l)| (p.exp(), l)).collect(),
    minors: t.minors.into_iter().map(f64::exp).collect(),
  }
}

// ---------------------------------------------------------------------------
// The scaling-function pair the tick finder works with.

trait ScalePair {
  fn f(&self, v: f64) -> Option<f64>;
  fn finv(&self, v: f64) -> Option<f64>;
  /// `{Identity, Identity}` (or `None`): plain linear ticks.
  fn identity(&self) -> bool {
    false
  }
}

struct Identity;

impl ScalePair for Identity {
  fn f(&self, v: f64) -> Option<f64> {
    Some(v)
  }
  fn finv(&self, v: f64) -> Option<f64> {
    Some(v)
  }
  fn identity(&self) -> bool {
    true
  }
}

impl ScalePair for Scaling {
  fn f(&self, v: f64) -> Option<f64> {
    self.forward(v)
  }
  fn finv(&self, v: f64) -> Option<f64> {
    self.inverse(v)
  }
}

/// `1/v` both ways: the pair `"Reciprocal"` hands to the tick finder.
struct ReciprocalPair;

impl ScalePair for ReciprocalPair {
  fn f(&self, v: f64) -> Option<f64> {
    Some(1.0 / v).filter(|r| r.is_finite())
  }
  fn finv(&self, v: f64) -> Option<f64> {
    self.f(v)
  }
}

// ---------------------------------------------------------------------------
// Numeric helpers, mirroring the WL built-ins the tick finder uses.

const EPS: f64 = 1e-10;

/// `MantissaExponent[v]` (base 10): `v = m 10^e` with `1/10 <= |m| < 1`;
/// `{0, 0}` for 0.
fn mantissa_exponent(v: f64) -> (f64, i32) {
  if v == 0.0 || !v.is_finite() {
    return (0.0, 0);
  }
  let mut e = v.abs().log10().floor() as i32 + 1;
  let mut m = v / 10f64.powi(e);
  if m.abs() >= 1.0 - 1e-12 {
    m /= 10.0;
    e += 1;
  } else if m.abs() < 0.1 - 1e-13 {
    m *= 10.0;
    e -= 1;
  }
  (m, e)
}

fn exponent(v: f64) -> i32 {
  mantissa_exponent(v).1
}

/// `orderME`: how many decades apart the exponents of `a` and `b` are.
fn order_me(a: f64, b: f64) -> i32 {
  (exponent(a) - exponent(b)).abs()
}

/// `distanceME`: the exponent of the span.
fn distance_me(min: f64, max: f64) -> i32 {
  exponent(max - min)
}

// `Floor[x, d]` and `Ceiling[x, d]` without any tolerance, as WL rounds
// machine numbers: `Ceiling[30.000000000000004, 10]` is 40, and the tick
// finder's choices follow from exactly that.
fn floor_to(x: f64, d: f64) -> f64 {
  d * (x / d).floor()
}

fn ceil_to(x: f64, d: f64) -> f64 {
  d * (x / d).ceil()
}

fn round_to(x: f64, d: f64) -> f64 {
  d * (x / d).round()
}

/// `Range[a, b, step]`.
fn range(a: f64, b: f64, step: f64) -> Vec<f64> {
  if !(step > 0.0 && a.is_finite() && b.is_finite()) {
    return Vec::new();
  }
  let count = ((b - a) / step + EPS).floor();
  if !(0.0..10_000.0).contains(&count) {
    return Vec::new();
  }
  (0..=count as usize).map(|k| a + k as f64 * step).collect()
}

/// `Union`: sorted, without (numerically) repeated values.
fn union(mut v: Vec<f64>) -> Vec<f64> {
  v.retain(|x| x.is_finite());
  v.sort_by(f64::total_cmp);
  v.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0));
  v
}

fn in_range(v: f64, min: f64, max: f64) -> bool {
  // Each end with its own relative tolerance: a range over 20 decades
  // must not let the tolerance of its top swallow the values at its bottom.
  v >= min - 1e-12 * min.abs() - 1e-15 && v <= max + 1e-12 * max.abs() + 1e-15
}

/// `linearDistanceQ` of the tick finder: whether `min..max` is close enough
/// together for linearly spaced ticks.
fn linear_distance_q(min: f64, max: f64) -> bool {
  if min * max < 0.0 {
    return linear_distance_q(min, 0.0) && linear_distance_q(0.0, max);
  }
  if max <= 0.0 {
    return linear_distance_q(max.abs(), min.abs());
  }
  let order = order_me(min, max);
  if order == 1 {
    distance_me(min, max) <= exponent(min) || (min != 0.0 && max / min <= 8.0)
  } else {
    order < 1
  }
}

/// `linearDistanceQ` of `Charting`ScaledTicks`, which decides whether the
/// labels are padded to a common number of decimals.
fn linear_distance_q_labels(min: f64, max: f64) -> bool {
  let (min, max) = if min < 0.0 {
    (min.abs().min(max.abs()), min.abs().max(max.abs()))
  } else {
    (min, max)
  };
  let order = order_me(min, max);
  if order == 1 {
    distance_me(min, max) <= exponent(min) || (min != 0.0 && max / min <= 8.0)
  } else {
    order <= 1 || order >= 18
  }
}

/// `FindDivisions[{a, b}, n]`.
fn find_divisions(a: f64, b: f64, n: usize) -> Vec<f64> {
  let call = Expr::FunctionCall {
    name: "FindDivisions".to_string(),
    args: vec![
      Expr::List(vec![Expr::Real(a), Expr::Real(b)].into()),
      Expr::Integer(n as i128),
    ]
    .into(),
  };
  match &crate::evaluator::evaluate_expr_to_expr(&call) {
    Ok(Expr::List(items)) => items
      .iter()
      .filter_map(crate::functions::math_ast::try_eval_to_f64)
      .collect(),
    _ => Vec::new(),
  }
}

// ---------------------------------------------------------------------------
// The tick finder (`Visualization`DivisionsDump`).

/// `tickPosition`: evenly spaced ticks `{1, 2, 2.5, 5} 10^k` apart, about
/// `num` of them.
fn tick_position(x0: f64, x1: f64, num: f64) -> Vec<f64> {
  let mut num = num;
  loop {
    let mut dist = (x1 - x0) / num;
    if !(dist > 0.0 && dist.is_finite()) {
      return vec![x0];
    }
    let mut scale = 10f64.powf((dist.log10() + 1e-12).floor());
    dist /= scale;
    if dist < 1.0 {
      dist *= 10.0;
      scale /= 10.0;
    }
    if dist >= 10.0 {
      dist /= 10.0;
      scale *= 10.0;
    }
    let m = [1.0, 2.0, 2.5, 5.0, 10.0]
      .into_iter()
      .find(|&s| s >= dist - 1e-9)
      .unwrap_or(10.0);
    let delta = m * scale;
    let res: Vec<f64> = range(floor_to(x0, delta), ceil_to(x1, delta), delta)
      .into_iter()
      .map(|x| round_to(x, delta))
      .collect();
    let inside = res.iter().filter(|&&x| in_range(x, x0, x1)).count();
    if inside as f64 <= num / 2.0 && num < 200.0 {
      num += 1.0;
      continue;
    }
    return res;
  }
}

/// `niceN[10, n]`: `n` equal steps of 0..10, 0 counted as 1.
fn nice_n(n: f64) -> Vec<f64> {
  if n == 4.0 {
    return vec![1.0, 2.0, 5.0, 10.0];
  }
  let mut v = range(0.0, 10.0, 10.0 / n);
  v.push(10.0);
  union(
    v.into_iter()
      .map(|x| if x == 0.0 { 1.0 } else { x })
      .collect(),
  )
}

/// `smoothingFunction[v, 10]`: snap `width` to the nearest of the nice
/// values `v` at the order of magnitude `n` (linear) or of `width` itself.
fn smooth(v: &[f64], width: f64, n: i32, flag: bool) -> f64 {
  let sign = if width > 0.0 {
    1.0
  } else if width < 0.0 {
    -1.0
  } else {
    0.0
  };
  let refs: Vec<f64> = if flag || width.abs() < EPS {
    let p = 10f64.powi(n - 1);
    std::iter::once(0.0)
      .chain(v.iter().copied())
      .map(|vi| sign * vi * p + round_to(width, p))
      .collect()
  } else {
    let p = 10f64.powf(width.abs().log10().floor());
    v.iter().map(|vi| sign * vi * p).collect()
  };
  let diff = |r: f64| (r - width).abs();
  let best = refs.iter().copied().map(diff).fold(f64::INFINITY, f64::min);
  let closest = refs
    .iter()
    .copied()
    .filter(|&r| diff(r) <= best * (1.0 + 1e-12))
    .map(f64::abs)
    .fold(f64::INFINITY, f64::min);
  sign * closest
}

/// `smoothValue`: snap every value to its nice neighbour, keeping signs.
fn smooth_value(linear: bool, v: &[f64], values: &[f64]) -> Vec<f64> {
  if values.is_empty() {
    return Vec::new();
  }
  let min = values.iter().copied().fold(f64::INFINITY, f64::min);
  let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let negatives = || -> Vec<f64> {
    let mut neg: Vec<f64> =
      values.iter().filter(|&&x| x < 0.0).map(|x| -x).collect();
    neg.reverse();
    let mut r: Vec<f64> = smooth_value(linear, v, &neg)
      .into_iter()
      .map(|x| -x)
      .collect();
    r.reverse();
    r
  };
  if max <= 0.0 {
    return negatives();
  }
  if min < 0.0 {
    let pos: Vec<f64> = values.iter().copied().filter(|&x| x > 0.0).collect();
    let mut r = negatives();
    r.push(0.0);
    r.extend(smooth_value(linear, v, &pos));
    return r;
  }
  let order = exponent((max - min) / 3.0);
  union(
    values
      .iter()
      .map(|&x| smooth(v, x, order, linear))
      .collect(),
  )
}

/// `getN`: how many divisions the scaled sampling starts from.
fn get_n(min: f64, max: f64) -> f64 {
  let p = 10f64.powi(distance_me(min, max) - 1);
  let (m, _) = mantissa_exponent(ceil_to(max, p) - floor_to(min, p));
  let tenth = (m - 0.1).abs() < 1e-9;
  let order = order_me(min, max) as f64;
  if !linear_distance_q(min, max) {
    if tenth {
      if order < 2.0 { 10.0 } else { order }
    } else {
      order.max(10.0 * m)
    }
  } else if tenth {
    2.0
  } else {
    10.0 * m
  }
}

#[derive(Clone, Copy)]
enum ExtraRange {
  Linear,
  Scaled,
}

/// `findExtraRange`: a little more than `lmin..lmax` in `n` steps.
fn find_extra_range(
  lmin: f64,
  lmax: f64,
  n: f64,
  kind: ExtraRange,
) -> Vec<f64> {
  let step = (lmax - lmin) / n;
  if step.is_nan() || step <= 0.0 {
    return Vec::new();
  }
  match kind {
    ExtraRange::Linear => {
      range((lmin - step).max(0.0), lmax + 2.0 * step, step)
    }
    ExtraRange::Scaled if lmin >= 0.0 => {
      range((lmin - 4.0 * step).max(0.0), lmax + 4.0 * step, step)
    }
    ExtraRange::Scaled if lmax <= 0.0 => {
      range(lmin - 4.0 * step, (lmax + 4.0 * step).min(0.0), step)
    }
    ExtraRange::Scaled => range(lmin - step, lmax + 4.0 * step, step),
  }
}

/// `scaledTickPosition`: candidate ticks spaced evenly in the scaled
/// coordinate (or linearly, over a short range), snapped to nice values.
/// (WL passes it a tick count too, which it ignores: `getN` decides.)
fn scaled_tick_position(min: f64, max: f64, s: &dyn ScalePair) -> Vec<f64> {
  if max <= 0.0 {
    let mut r: Vec<f64> = scaled_tick_position(-max, -min, s)
      .into_iter()
      .map(|x| -x)
      .collect();
    r.reverse();
    return r;
  }
  let (Some(a), Some(b)) = (s.f(min), s.f(max)) else {
    return Vec::new();
  };
  let (scmin, scmax) = (a.min(b), a.max(b));
  let (linear, smoother, scvalues): (bool, Vec<f64>, Vec<f64>) =
    if min < 0.0 && max > 0.0 {
      let mmax = min.abs().max(max);
      let lo = mmax / 1e4;
      let n = get_n(lo, mmax);
      let o = distance_me(lo, mmax) - 1;
      if linear_distance_q(lo, mmax) {
        let p = 10f64.powi(o);
        let vals = range(0.0, floor_to(max, p), floor_to(mmax, p) / n)
          .into_iter()
          .filter_map(|v| s.f(v))
          .collect();
        (true, range(0.0, 10.0, 1.0), vals)
      } else {
        let Some(fm) = s.f(mmax) else {
          return Vec::new();
        };
        let step = fm / (3.0 * n);
        (false, nice_n(4.0), range(0.0, fm + 5.0 * step, step))
      }
    } else {
      let n = get_n(min, max);
      let o = distance_me(min, max) - 1;
      if linear_distance_q(min, max) {
        let p = 10f64.powi(o);
        let vals = find_extra_range(
          floor_to(min, p),
          floor_to(max, p),
          n,
          ExtraRange::Linear,
        )
        .into_iter()
        .filter_map(|v| s.f(v))
        .collect();
        (true, range(0.0, 10.0, 1.0), vals)
      } else {
        (
          false,
          nice_n(2.0 * n),
          find_extra_range(scmin, scmax, 4.0 * n, ExtraRange::Scaled),
        )
      }
    };
  let values: Vec<f64> =
    scvalues.into_iter().filter_map(|v| s.finv(v)).collect();
  let mut out = smooth_value(linear, &smoother, &values);
  if min < 0.0 && max > 0.0 {
    let mut neg: Vec<f64> = out.iter().map(|x| -x).collect();
    neg.reverse();
    neg.extend(out);
    out = neg;
  }
  out.retain(|v| v.is_finite());
  out
}

/// `restraint`: snap the major ticks to the nice values their spread of
/// magnitudes allows.
fn restraint(ticks: &[f64], min: f64, max: f64) -> Vec<f64> {
  let d: Vec<f64> = ticks
    .iter()
    .filter(|&&t| in_range(t, min, max))
    .map(|t| t.abs())
    .collect();
  let order = if d.is_empty() {
    i32::MAX
  } else {
    let lo = d.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    order_me(lo, hi)
  };
  let v: Vec<f64> = match order {
    0 => range(0.0, 10.0, 1.0),
    1 => nice_n(4.0),
    2 | 3 => nice_n(2.0),
    _ => nice_n(1.0),
  };
  smooth_value(linear_distance_q(min, max), &v, ticks)
}

/// `nearestN[list, v, k]`: the `k` elements nearest to `v`, sorted.
fn nearest_n(list: &[f64], v: f64, k: usize) -> Vec<f64> {
  let mut l = list.to_vec();
  l.sort_by(|a, b| (a - v).abs().total_cmp(&(b - v).abs()));
  l.truncate(k);
  l.sort_by(f64::total_cmp);
  l
}

/// `niceRange`: about `n` nice values in `min..max`.
fn nice_range(min: f64, max: f64, n: usize) -> Vec<f64> {
  let order = exponent(min).min(exponent((max - min) / 2.0)) - 1;
  let p = 10f64.powi(order);
  let mut mi = floor_to(min, p);
  let ma = ceil_to(max, p);
  let d = find_divisions(mi, ma, n);
  let mut step = if d.len() >= 2 {
    round_to((d[1] - d[0]).abs(), p)
  } else {
    0.0
  };
  if step == 0.0 {
    step = 10f64.powi(order - 1);
  }
  if (ma - mi).rem_euclid(step) > step / 2.0 {
    mi += (step / 2.0).floor();
  }
  range(mi, ma, step)
    .into_iter()
    .filter(|&x| in_range(x, min, max))
    .collect()
}

/// `getLargeOrderTicks`: decade ticks over a range of more than 18 orders
/// of magnitude, `min..max` given as exponents.
fn large_order_ticks(min: i32, max: i32) -> (Vec<f64>, Vec<Vec<f64>>) {
  let list = nice_range(min as f64, max as f64, 18);
  if list.len() < 2 {
    let p: Vec<f64> = list.iter().map(|e| 10f64.powf(*e)).collect();
    return (p, Vec::new());
  }
  let idx: Vec<usize> = nice_range(1.0, list.len() as f64, 6)
    .into_iter()
    .map(|i| i.round() as usize)
    .filter(|&i| (1..=list.len()).contains(&i))
    .collect();
  let mut major: Vec<f64> = idx.iter().map(|&i| list[i - 1]).collect();
  let top = major.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  if top < max as f64 {
    major.clone_from(&list);
    major.push(2.0 * list[list.len() - 1] - list[list.len() - 2]);
  }
  let mut bounds = vec![1];
  bounds.extend(&idx);
  bounds.push(list.len());
  let minor = bounds
    .windows(2)
    .map(|w| {
      list[w[0] - 1..w[1].max(w[0])]
        .iter()
        .map(|e| 10f64.powf(*e))
        .collect()
    })
    .collect();
  (major.iter().map(|e| 10f64.powf(*e)).collect(), minor)
}

/// `geNicerRange`: replace majors that leave too much of `min..max` bare.
fn ge_nicer_range(list: &[f64], n: f64, min: f64, max: f64) -> Vec<f64> {
  if list.is_empty() {
    return Vec::new();
  }
  let lmin = list.iter().copied().fold(f64::INFINITY, f64::min);
  let lmax = list.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let mirrored = |sub: Vec<f64>| -> Vec<f64> {
    let mut neg: Vec<f64> = sub.iter().map(|x| -x).collect();
    neg.reverse();
    let (a, b) = (
      neg.iter().copied().fold(f64::INFINITY, f64::min),
      neg.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let mut r: Vec<f64> = ge_nicer_range(&neg, n, a, b)
      .into_iter()
      .map(|x| -x)
      .collect();
    r.reverse();
    r
  };
  if lmax < 0.0 {
    return mirrored(list.to_vec());
  }
  if lmin < 0.0 {
    let mut r = mirrored(list.iter().copied().filter(|&x| x <= 0.0).collect());
    let pos: Vec<f64> = list.iter().copied().filter(|&x| x >= 0.0).collect();
    let (a, b) = (
      pos.iter().copied().fold(f64::INFINITY, f64::min),
      pos.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    r.extend(ge_nicer_range(&pos, n, a, b));
    return r;
  }
  let Some(pos_min) =
    list.iter().copied().filter(|&x| x > 0.0).reduce(f64::min)
  else {
    return list.to_vec();
  };
  let maxo = exponent(lmax);
  let mino = exponent(pos_min) - 1;
  let order = maxo - mino - 1;
  let inside = list.iter().filter(|&&x| in_range(x, min, max)).count() as f64;
  if list.len() == 1 {
    list.to_vec()
  } else if order > 18 {
    large_order_ticks(mino, maxo).0
  } else if inside < n / 3.0 || (inside < n / 2.0 && order < 1) {
    let lo = nearest_n(list, min, 3)[0];
    let hi = *nearest_n(list, max, 3).last().unwrap();
    nice_range(lo, hi, 8)
  } else {
    list.to_vec()
  }
}

fn too_many_ticks(l: usize, n: usize) -> bool {
  match (l.is_multiple_of(2), n.is_multiple_of(2)) {
    (true, true) | (false, false) => l > n,
    _ => l > n + 1,
  }
}

/// `extendEnds`: one more tick past each end, a step (or ratio) away.
fn extend_ends(linear: bool, l: Vec<f64>) -> Vec<f64> {
  if l.len() < 2 {
    return l;
  }
  let (a, b) = (l[l.len() - 2], l[l.len() - 1]);
  let mut out = Vec::with_capacity(l.len() + 2);
  if linear {
    let step = b - a;
    out.push(l[0] - step);
    out.extend(&l);
    out.push(b + step);
  } else {
    let step = b / a;
    out.push(l[0] / step);
    out.extend(&l);
    out.push(b * step);
  }
  let mut seen: Vec<f64> = Vec::new();
  for v in out {
    if !seen
      .iter()
      .any(|s| (s - v).abs() <= 1e-12 * v.abs().max(1.0))
    {
      seen.push(v);
    }
  }
  seen
}

/// `findBestDivider`.
fn find_best_divider(n: usize, lenmin: usize, lenmax: usize) -> usize {
  let failed = (n / lenmin.max(1)).max(1);
  let steps: Vec<usize> = (lenmin..=lenmax)
    .map(|c| n.div_ceil(c.max(1)))
    .filter(|&s| s != 0)
    .collect();
  steps
    .iter()
    .copied()
    .find(|&s| n % s == 1)
    .unwrap_or(failed)
}

/// `downSampleN` (unscaled): keep every `step`-th element.
fn down_sample_plain(list: &[f64], lenmin: usize, lenmax: usize) -> Vec<f64> {
  let n = list.len();
  if n == 0 {
    return Vec::new();
  }
  let step = find_best_divider(n, lenmin, lenmax).max(1);
  let offset = if n % step == 1 {
    1
  } else {
    (n % step) / 2 + step / 2 + 1
  };
  let res: Vec<f64> = (offset..=n).step_by(step).map(|i| list[i - 1]).collect();
  let Some(&last) = res.last() else {
    return list.to_vec();
  };
  let pos = list
    .iter()
    .position(|&x| (x - last).abs() <= 1e-12 * x.abs().max(1.0))
    .map_or(n, |p| p + 1);
  if n - pos > step / 3 && lenmin != lenmax {
    down_sample_plain(list, lenmin + 1, lenmax)
  } else {
    res
  }
}

/// `downSampleN[list, {lenmin, lenmax}, {f, f⁻¹}]`: thin the ticks evenly
/// in the scaled coordinate.
fn down_sample(
  list: &[f64],
  lenmin: usize,
  lenmax: usize,
  s: &dyn ScalePair,
) -> Vec<f64> {
  let scaled: Vec<f64> = list.iter().filter_map(|&v| s.f(v)).collect();
  let nl = down_sample_plain(&scaled, lenmin, lenmax);
  let lo = nl.iter().copied().fold(f64::INFINITY, f64::min);
  let hi = nl.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let shift = if lo < 0.0 && hi > 0.0 {
    nl.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0)
  } else {
    0.0
  };
  nl.into_iter().filter_map(|v| s.finv(v - shift)).collect()
}

/// `trimMajorTicks`: thin out too many majors, and add one past each end.
#[allow(clippy::too_many_arguments)]
fn trim_major_ticks(
  list: &[f64],
  n: usize,
  min: f64,
  max: f64,
  s: &dyn ScalePair,
  linear: bool,
  extend: bool,
) -> Vec<f64> {
  if list.is_empty() {
    return Vec::new();
  }
  let lmin = list.iter().copied().fold(f64::INFINITY, f64::min);
  let lmax = list.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let negate = |v: Vec<f64>| -> Vec<f64> {
    let mut r: Vec<f64> = v.into_iter().map(|x| -x).collect();
    r.reverse();
    r
  };
  if lmax < 0.0 {
    return negate(trim_major_ticks(
      &negate(list.to_vec()),
      n,
      min,
      max,
      s,
      linear,
      extend,
    ));
  }
  if lmin < 0.0 {
    let m = (n as f64 / 1.5).ceil() as usize;
    let reach = min.abs().max(max.abs());
    let neg: Vec<f64> =
      negate(list.iter().copied().filter(|&x| x < 0.0).collect());
    let pos: Vec<f64> = list.iter().copied().filter(|&x| x > 0.0).collect();
    let mut r = negate(trim_major_ticks(&neg, m, 0.0, reach, s, linear, true));
    r.push(0.0);
    r.extend(trim_major_ticks(&pos, m, 0.0, reach, s, linear, false));
    return r;
  }
  let inside: Vec<f64> = list
    .iter()
    .copied()
    .filter(|&x| in_range(x, min, max))
    .collect();
  let thinned = if too_many_ticks(inside.len(), n) {
    down_sample(&inside, n.saturating_sub(2).max(2), n + 1, s)
  } else {
    list.to_vec()
  };
  if extend {
    extend_ends(linear, thinned)
  } else {
    thinned
  }
}

fn median(v: &[f64]) -> f64 {
  let mut s = v.to_vec();
  s.sort_by(f64::total_cmp);
  let k = s.len();
  if k % 2 == 1 {
    s[k / 2]
  } else {
    f64::midpoint(s[k / 2 - 1], s[k / 2])
  }
}

/// `sampleTicksWithinRange`: the minor ticks between the majors `min` and
/// `max`.
fn sample_ticks_within_range(
  ticks: &[Vec<f64>],
  min: f64,
  max: f64,
) -> Vec<f64> {
  let mut tt: Vec<&Vec<f64>> = ticks
    .iter()
    .filter(|t| !t.is_empty() && in_range(median(t), min, max))
    .collect();
  // `Last[SortBy[tt, Abs]]`: the longest, then the largest.
  tt.sort_by(|a, b| {
    a.len().cmp(&b.len()).then_with(|| {
      a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.abs().total_cmp(&y.abs()))
        .find(|o| o.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
    })
  });
  let Some(best) = tt.last() else {
    return Vec::new();
  };
  let picked: Vec<f64> = best
    .iter()
    .copied()
    .filter(|&x| in_range(x, min, max))
    .collect();
  let res = if order_me(min, max) > 1 {
    let mut v = vec![min, max];
    v.extend(picked);
    union(v)
  } else {
    picked
  };
  if res.is_empty() {
    return res;
  }
  smooth_value(linear_distance_q(min, max), &[0.0, 1.0, 10.0], &res)
    .into_iter()
    .filter(|&x| in_range(x, min, max))
    .collect()
}

/// `Visualization`Divisions[{min, max}, {n, n}, 10, ScalingFunctions ->
/// s]`: the major ticks and, per major interval, the minor ones.
fn divisions(
  min: f64,
  max: f64,
  n: usize,
  s: &dyn ScalePair,
) -> (Vec<f64>, Vec<Vec<f64>>) {
  let (min, mut max) = (min.min(max), min.max(max));
  if min == max {
    max = max * (1.0 + f64::EPSILON) + f64::EPSILON;
  }
  let identity = s.identity();
  let linear = linear_distance_q(min, max);
  let (maxo, mino) = (exponent(max), exponent(min));
  if linear && maxo - mino > 18 && !identity {
    return large_order_ticks(mino, maxo);
  }
  let nf = n as f64;
  let positions = |a: f64, b: f64| {
    if linear || identity {
      tick_position(a, b, nf)
    } else {
      scaled_tick_position(a, b, s)
    }
  };
  let mut major = positions(min, max);
  if !identity {
    major = restraint(&major, min, max);
    major = ge_nicer_range(&major, nf, min, max);
    major = trim_major_ticks(&major, n, min, max, s, linear, true);
  }
  let mut sorted: Vec<f64> =
    major.iter().copied().filter(|v| v.is_finite()).collect();
  sorted.sort_by(f64::total_cmp);
  let mut minor: Vec<Vec<f64>> = sorted
    .windows(2)
    .map(|w| positions(w[0], w[1]))
    .filter(|v| !v.is_empty())
    .collect();
  if !identity {
    minor = major
      .windows(2)
      .map(|w| sample_ticks_within_range(&minor, w[0], w[1]))
      .filter(|v| !v.is_empty())
      .collect();
  }
  let mut kept: Vec<f64> = Vec::new();
  for v in major.into_iter().filter(|&v| in_range(v, min, max)) {
    if !kept
      .iter()
      .any(|k| (k - v).abs() <= 1e-12 * v.abs().max(1.0))
    {
      kept.push(v);
    }
  }
  (kept, minor)
}

/// `Charting`ScaledTicks[{f, f⁻¹}]` (`iScaledTicks`) over the scaled range
/// `min..max`, with `n` divisions per level.
fn scaled_ticks(s: &dyn ScalePair, min: f64, max: f64, n: usize) -> AxisTicks {
  let (Some(a), Some(b)) = (s.finv(min), s.finv(max)) else {
    return AxisTicks::default();
  };
  if s.f(a).is_none() || s.f(b).is_none() {
    return AxisTicks::default();
  }
  let (lo, hi) = (a.min(b), a.max(b));
  let (mut major, minor) = divisions(lo, hi, n, s);
  major.sort_by(f64::total_cmp);
  let minor: Vec<f64> = union(minor.into_iter().flatten().collect())
    .into_iter()
    .filter(|m| {
      !major
        .iter()
        .any(|v| (v - m).abs() <= 1e-12 * m.abs().max(1.0))
    })
    .collect();
  let identity = s.identity();
  let labels = simple_padding(
    &major,
    identity || linear_distance_q_labels(lo, hi),
    if identity { 6 } else { 4 },
  );
  AxisTicks {
    majors: major
      .iter()
      .zip(labels)
      .filter_map(|(&v, l)| Some((s.f(v)?, l)))
      .collect(),
    minors: minor.iter().filter_map(|&v| s.f(v)).collect(),
  }
}

/// `Charting`ScaledTicks[Reciprocal, …]`: ticks of a `1/v` axis.
fn reciprocal_ticks(min: f64, max: f64) -> AxisTicks {
  if min < 0.0 && max > 0.0 {
    let span = max - min;
    let d = find_divisions(min + span / 10.0, max - span / 10.0, 6);
    let lo = d.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut t = reciprocal_ticks(0.0, hi);
    if lo < 0.0 {
      let neg = reciprocal_ticks(0.0, lo.abs());
      t.majors
        .extend(neg.majors.into_iter().map(|(p, l)| (-p, negate_label(&l))));
      t.minors.extend(neg.minors.into_iter().map(|p| -p));
    }
    return t;
  }
  if min < 0.0 {
    return AxisTicks::default();
  }
  let newmin = min.max(f64::EPSILON);
  let base = scaled_ticks(&ReciprocalPair, newmin, max, 7);
  let mut inside: Vec<f64> = vec![0.0];
  inside.extend(base.majors.iter().map(|(p, _)| 1.0 / p));
  if inside.len() <= 1 {
    return base;
  }
  let n1 = 5usize;
  let mut groups: Vec<Vec<f64>> = Vec::new();
  for (i, w) in inside.windows(2).enumerate() {
    let i = i + 1;
    let count = n1 + 1 - n1.min(i * i);
    if count > 0 {
      groups.push(find_divisions(w[0], w[1], count));
    }
  }
  let mut major: Vec<f64> = Vec::new();
  let mut prev_max = f64::NEG_INFINITY;
  for (k, g) in groups.iter().enumerate() {
    let gmax = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if k == 0 {
      major.extend(g);
    } else {
      major.extend(g.iter().copied().filter(|&v| v > prev_max));
    }
    prev_max = gmax;
  }
  major.sort_by(f64::total_cmp);
  let minor: Vec<f64> = major
    .windows(2)
    .flat_map(|w| find_divisions(w[0], w[1], 6))
    .collect();
  let mut major = union(major);
  major.retain(|&v| v != 0.0);
  major.pop();
  if max / newmin < 2.0 {
    return base;
  }
  let labels = simple_padding(&major, true, 6);
  let mut minor = union(minor);
  minor.retain(|&v| v != 0.0);
  AxisTicks {
    majors: major
      .iter()
      .zip(labels)
      .filter_map(|(&v, l)| Some((ReciprocalPair.f(v)?, l)))
      .collect(),
    minors: minor.iter().filter_map(|&v| ReciprocalPair.f(v)).collect(),
  }
}

fn negate_label(l: &str) -> String {
  if l == "0" {
    l.to_string()
  } else {
    format!("-{l}")
  }
}

// ---------------------------------------------------------------------------
// Tick labels (`Charting`SimplePadding`), as SVG text markup.

/// `fixMEandTranspose`: mantissas in `1..10` rather than `0.1..1`.
fn fixed_me(v: f64) -> (f64, i32) {
  let (m, e) = mantissa_exponent(v);
  if (m.abs() - 1.0).abs() < 1e-12 || (m == 0.0 && e == 0) {
    (m, e)
  } else {
    (10.0 * m, e - 1)
  }
}

fn is_integer(v: f64) -> bool {
  (v - v.round()).abs() <= 1e-9 * v.abs().max(1.0)
}

/// `mantissaDigits`: minus the number of decimals of `a`.
fn mantissa_digits(a: f64) -> i32 {
  let mut a = a;
  let mut d = 0;
  while !is_integer(a) && d > -15 {
    a *= 10.0;
    d -= 1;
  }
  d
}

fn superscript(e: i32) -> String {
  format!("10<tspan baseline-shift=\"super\" font-size=\"70%\">{e}</tspan>")
}

fn int_label(v: f64) -> String {
  let r = v.round();
  if r == 0.0 {
    "0".to_string()
  } else {
    format!("{r:.0}")
  }
}

/// The labels of the medium-sized values: integers, or a common number of
/// decimals.
fn medium_labels(list: &[f64], pad: bool) -> Vec<String> {
  let mut sorted = list.to_vec();
  sorted.sort_by(f64::total_cmp);
  let chop = |v: f64| if v.abs() < 1e-10 { 0.0 } else { v };
  let diffs: Vec<f64> = if list.len() == 2 && chop(list[0] + list[1]) == 0.0 {
    vec![list[0].abs()]
  } else {
    let mut d: Vec<f64> =
      sorted.windows(2).map(|w| chop(w[1] - w[0])).collect();
    let dmin = d.iter().copied().fold(f64::INFINITY, f64::min);
    if dmin > sorted[0] {
      d.insert(0, sorted[0]);
    }
    d
  };
  let base_digits = list
    .iter()
    .map(|&v| {
      let (m, e) = fixed_me(v);
      e + mantissa_digits(m)
    })
    .min()
    .unwrap_or(0);
  let diff_me: Vec<(f64, i32)> = diffs.iter().map(|&d| fixed_me(d)).collect();
  let diff_digits_min = diff_me
    .iter()
    .map(|&(m, e)| e + mantissa_digits(m))
    .min()
    .unwrap_or(0);
  let mut digits = 0.min(base_digits).min(diff_digits_min);
  let min_e = diff_me.iter().map(|&(_, e)| e).min().unwrap_or(0);
  if min_e - digits > 3 {
    digits = min_e - 3;
  }
  list
    .iter()
    .map(|&x| {
      if digits < 0 && (pad || !is_integer(x)) {
        format!("{:.*}", (-digits) as usize, x)
      } else {
        int_label(x)
      }
    })
    .collect()
}

/// The decimals a medium label was given (`NumberForm[x, {∞, d}]`), if any.
fn label_decimals(l: &str) -> Option<usize> {
  l.split_once('.').map(|(_, d)| d.len())
}

/// `m×10^e` / `10^e` labels of values too large or too small to write out.
fn power_labels(list: &[f64], big: bool) -> Vec<String> {
  let me: Vec<(f64, i32)> = list.iter().map(|&v| fixed_me(v)).collect();
  if me.iter().all(|(m, _)| (m.abs() - 1.0).abs() < 1e-9) {
    return me
      .iter()
      .map(|&(m, e)| {
        if big && m < 0.0 {
          format!("-{}", superscript(e))
        } else {
          superscript(e)
        }
      })
      .collect();
  }
  let mantissas: Vec<f64> = me.iter().map(|(m, _)| *m).collect();
  let mut labels = medium_labels(&mantissas, true);
  if big {
    let mut abs: Vec<f64> = list.iter().map(|v| v.abs()).collect();
    abs.sort_by(f64::total_cmp);
    let diffs: Vec<f64> = abs
      .windows(2)
      .map(|w| w[1] - w[0])
      .filter(|&d| d > 0.0)
      .collect();
    let dd = diffs
      .iter()
      .map(|d| d.log10())
      .fold(f64::INFINITY, f64::min);
    let digits_diff = if diffs.is_empty() {
      0
    } else if is_integer(dd) {
      dd.round() as i32 + 1
    } else {
      dd.floor() as i32
    };
    let digits_max = list
      .iter()
      .map(|v| v.abs().log10().ceil() as i32)
      .max()
      .unwrap_or(0);
    let decimals = digits_max - digits_diff;
    for (l, &m) in labels.iter_mut().zip(&mantissas) {
      if let Some(d) = label_decimals(l) {
        if decimals > 0 {
          *l = format!("{:.*}", d.min(decimals as usize), m);
        }
      } else if is_integer(m) {
        *l = int_label(m);
      }
    }
  }
  labels
    .into_iter()
    .zip(&me)
    .map(|(l, &(_, e))| format!("{l}&#215;{}", superscript(e)))
    .collect()
}

/// `Charting`SimplePadding[list, FullPadding -> pad, CutoffExponent ->
/// cutoff]`: the labels of the sorted major tick values `list`.
fn simple_padding(list: &[f64], pad: bool, cutoff: i32) -> Vec<String> {
  let big_at = 10f64.powi(cutoff);
  let small_at = 10f64.powi(-cutoff + 1) - f64::EPSILON;
  let class = |v: f64| {
    let a = v.abs();
    if a == 0.0 {
      0
    } else if a >= big_at {
      3
    } else if a > small_at {
      2
    } else {
      1
    }
  };
  let pick = |c: u8| -> Vec<f64> {
    list.iter().copied().filter(|&v| class(v) == c).collect()
  };
  let (small, medium, big) = (pick(1), pick(2), pick(3));
  let mut labels: Vec<(f64, String)> = Vec::new();
  if !big.is_empty() {
    labels.extend(big.iter().copied().zip(power_labels(&big, true)));
  }
  if !medium.is_empty() {
    labels.extend(medium.iter().copied().zip(medium_labels(&medium, pad)));
  }
  if !small.is_empty() {
    labels.extend(small.iter().copied().zip(power_labels(&small, false)));
  }
  if list.contains(&0.0) {
    labels.push((0.0, "0".to_string()));
  }
  labels.sort_by(|a, b| a.0.total_cmp(&b.0));
  labels.into_iter().map(|(_, l)| l).collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn labels(t: &AxisTicks) -> Vec<String> {
    t.majors.iter().map(|(_, l)| l.clone()).collect()
  }

  #[test]
  fn log_ticks_match_scaled_ticks() {
    // `Charting`ScaledTicks[{Log, Exp}]` over the padded range of
    // `Plot[Abs[Gamma[z]], {z, -4, 8}, ScalingFunctions -> "Log"]`.
    let t = scaled_ticks(&Scaling::Log, -2.1532045830675375, 12.79161155, 6);
    let l = labels(&t);
    assert_eq!(l[..4], ["1", "10", "100", "1000"]);
    assert_eq!(l.len(), 6);
    assert_eq!(t.minors.len(), 76);
    // ListLogPlot of the binomial coefficients: every other decade.
    let t = scaled_ticks(&Scaling::Log, -1.5687525534369071, 16.3606994756, 6);
    let pos: Vec<f64> = t
      .majors
      .iter()
      .map(|(p, _)| p.exp().log10().round())
      .collect();
    assert_eq!(pos, [1.0, 3.0, 5.0, 7.0]);
    assert_eq!(t.minors.len(), 7);
    // 1, 5, 10, 50, 100 over 0.6..187.
    let t = scaled_ticks(&Scaling::Log, -0.501299821304124, 5.2281130670258, 6);
    assert_eq!(labels(&t), ["1", "5", "10", "50", "100"]);
    assert_eq!(t.minors.len(), 32);
    // 1, 10^6, 10^12, 10^18 over 21 decades.
    let t =
      scaled_ticks(&Scaling::Log, -4.280645161290322, 44.7516129032258, 6);
    assert_eq!(t.majors.len(), 4);
    assert_eq!(t.majors[0].1, "1");
    assert!(t.majors[3].1.contains(">18<"));
    assert_eq!(t.minors.len(), 27);
  }

  #[test]
  fn decimal_labels_share_their_decimals() {
    let t = scaled_ticks(&Scaling::Log, -3.6838576868278725, 5.04143476790, 6);
    assert_eq!(labels(&t), ["0.1", "1", "10", "100"]);
  }

  #[test]
  fn reverse_ticks_negate_the_plain_ones() {
    let t = Scaling::Reverse.ticks(-105.0, 5.0);
    let l = labels(&t);
    assert_eq!(l, ["0", "20", "40", "60", "80", "100"]);
    assert_eq!(t.majors[5].0, -100.0);
  }
}
