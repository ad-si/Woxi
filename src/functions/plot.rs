use plotters::prelude::{Color, *};

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::chart::{
  ChartLabel, ChartOptions, LabelPosition, StyledLabel,
};
use crate::functions::graphics::{Color as WoxiColor, parse_color};
use crate::functions::math_ast::{try_eval_to_f64, try_eval_to_f64_lenient};
use crate::functions::plot_axes::{
  AxisScale, AxisTicks, OriginAxes, origin_axes_svg,
};
use crate::functions::plot_scaling::Scaling;
use crate::syntax::PlotMarker;

/// How many lines below the first a stacked `PlotLabel` (a `Grid`/`Column`
/// title) occupies — the extra top margin every renderer has to reserve for
/// it. Zero for the ordinary single-line title and for no title at all.
pub(crate) fn plot_label_extra_lines(label: Option<&StyledLabel>) -> usize {
  label
    .filter(|sl| !sl.text.is_empty())
    .map_or(0, StyledLabel::extra_line_count)
}

/// Extra top room, in render pixels, a stacked (multi-line) `PlotLabel`
/// needs on top of an explicit `ImagePadding`: the title grows upwards from
/// the padding, so the padding alone would push its extra lines off the
/// canvas.
fn stacked_title_room(label: Option<&StyledLabel>, sf: f64) -> i32 {
  let font = label.and_then(|sl| sl.font_size).unwrap_or(17.0);
  (plot_label_extra_lines(label) as f64 * font * 1.2 * sf).ceil() as i32
}

/// Extra top room, in render pixels, a `PlotLabel` needs when a top
/// `FrameLabel` or a y `AxesLabel` shares the top margin with it. The title
/// is drawn above whatever it stacks on, so with a tight explicit
/// `ImagePadding` its baseline would land above the canvas; this is the
/// shortfall between the padding and the room title and label need.
fn stacked_label_shortfall(
  label: Option<&StyledLabel>,
  stacked_below: usize,
  top_padding: i32,
  sf: f64,
) -> i32 {
  if stacked_below == 0 || plot_label_extra_lines(label) > 0 {
    return 0;
  }
  if label.is_none_or(|sl| sl.text.is_empty()) {
    return 0;
  }
  let needed = (17.0 * 1.4 + stacked_below as f64 * 14.0 * 1.2) * sf;
  (needed.ceil() as i32 - top_padding).max(0)
}

pub(crate) const DEFAULT_WIDTH: u32 = 360;
pub(crate) const DEFAULT_HEIGHT: u32 = 225;
/// Internal rendering resolution multiplier for sub-pixel precision.
/// Plotters maps to integer coordinates, so we render at a higher resolution
/// and scale down via SVG viewBox to get smooth curves.
pub(crate) const RESOLUTION_SCALE: u32 = 10;
pub(crate) const NUM_SAMPLES: usize = 500;

/// Largest logical (pre-`RESOLUTION_SCALE`) SVG dimension a plot is allowed
/// to compute for itself, whether from `ImageSize`, an explicit
/// `AspectRatio`, or one derived from the data range to keep round curves
/// round. A curve that is nearly vertical or horizontal over its sampled
/// range (a near-degenerate x- or y-extent, e.g. a `ParametricPlot` passing
/// close to a pole) makes that derived aspect ratio — and so the computed
/// dimension — arbitrarily large; well past this bound it is not a
/// meaningful image size, just a value that overflows once multiplied by
/// `RESOLUTION_SCALE`. Chosen far above any dimension a real plot would
/// ever legitimately request (thousands of pixels at most) so clamping
/// never changes an intentional `ImageSize`/`AspectRatio`.
pub(crate) const MAX_SVG_DIMENSION: u32 = 100_000;

/// Return plotters colors adapted to the current light/dark theme.
pub(crate) fn plot_theme()
-> (RGBColor, RGBColor, RGBColor, &'static str, &'static str) {
  if crate::is_dark_mode() {
    // (background, axis_gray, origin_line_gray, label_fill, title_default_fill)
    (
      RGBColor(0x1a, 0x1a, 0x1a), // dark background
      RGBColor(0x99, 0x99, 0x99), // lighter axes for dark bg
      RGBColor(0x44, 0x44, 0x44), // subtle origin lines
      "#999",                     // axis label fill
      "#e0e0e0",                  // default plot label fill
    )
  } else {
    (
      RGBColor(0xFF, 0xFF, 0xFF), // white background
      RGBColor(0x66, 0x66, 0x66), // dark gray axes
      RGBColor(0xCC, 0xCC, 0xCC), // light gray origin lines
      "#666",                     // axis label fill
      "#333",                     // default plot label fill
    )
  }
}

/// Substitute all occurrences of a variable with a value in an expression
pub(crate) fn substitute_var(expr: &Expr, var: &str, value: &Expr) -> Expr {
  let sub = |e: &Expr| substitute_var(e, var, value);
  match expr {
    Expr::Identifier(name) if name == var => value.clone(),
    // Numeric data with nothing to substitute (see `is_structured_bspline`).
    Expr::FunctionCall { name, args }
      if crate::syntax::is_structured_bspline(name, args) =>
    {
      expr.clone()
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(sub).collect(),
    },
    Expr::BinaryOp { op, left, right } => binop(*op, sub(left), sub(right)),
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(sub(operand)),
    },
    Expr::List(items) => Expr::List(items.iter().map(sub).collect()),
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands.iter().map(sub).collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(exprs) => {
      Expr::CompoundExpr(exprs.iter().map(sub).collect())
    }
    Expr::Rule {
      pattern,
      replacement,
    } => rule_expr(sub(pattern), sub(replacement)),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => rule_delayed_expr(sub(pattern), sub(replacement)),
    Expr::ReplaceAll { expr, rules } => Expr::ReplaceAll {
      expr: Box::new(sub(expr)),
      rules: Box::new(sub(rules)),
    },
    Expr::ReplaceRepeated { expr, rules } => Expr::ReplaceRepeated {
      expr: Box::new(sub(expr)),
      rules: Box::new(sub(rules)),
    },
    Expr::Map { func, list } => Expr::Map {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: Box::new(sub(func)),
      list: Box::new(sub(list)),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: Box::new(sub(func)),
      arg: Box::new(sub(arg)),
    },
    Expr::Postfix { expr, func } => Expr::Postfix {
      expr: Box::new(sub(expr)),
      func: Box::new(sub(func)),
    },
    Expr::Part { expr, index } => Expr::Part {
      expr: Box::new(sub(expr)),
      index: Box::new(sub(index)),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: Box::new(sub(func)),
      args: args.iter().map(sub).collect(),
    },
    Expr::Function { body } => Expr::Function {
      body: Box::new(sub(body)),
    },
    Expr::Association(pairs) => {
      Expr::Association(pairs.iter().map(|(k, v)| (sub(k), sub(v))).collect())
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: name.clone(),
      head: head.clone(),
      default: default.as_ref().map(|d| Box::new(sub(d))),
    },
    other => other.clone(),
  }
}

/// Evaluate the function body at a given x value
/// Parse a plot iterator specification `{var, min, max}`.
/// Whether a plot iterator names the same point twice (`{x, 1, 1}`).
/// Wolfram will not sample such a range: it reports `head::plld` and leaves
/// the call unevaluated, so every plot head refuses it the same way through
/// this one check. A reversed range (`{x, 1, 0}`) is fine and samples
/// backwards.
/// `as_reals` writes the endpoints as machine reals (`{x, 0., 0.}`), the way
/// `RegionPlot3D` reports them.
pub(crate) fn degenerate_iterator(
  head: &str,
  spec: &Expr,
  as_reals: bool,
) -> bool {
  let Expr::List(items) = spec else {
    return false;
  };
  if items.len() != 3 {
    return false;
  }
  let Expr::Identifier(var) = &items[0] else {
    return false;
  };
  let (Ok(lo), Ok(hi)) = (
    evaluate_expr_to_expr(&items[1]),
    evaluate_expr_to_expr(&items[2]),
  ) else {
    return false;
  };
  let (Some(a), Some(b)) = (try_eval_to_f64(&lo), try_eval_to_f64(&hi)) else {
    return false;
  };
  if a != b {
    return false;
  }
  let (lo, hi) = if as_reals {
    (Expr::Real(a), Expr::Real(b))
  } else {
    (lo, hi)
  };
  crate::emit_message(&format!(
    "{head}::plld: Endpoints for {var} in {{{var}, {}, {}}} must have \
     distinct machine-precision numerical values.",
    crate::syntax::expr_to_input_form(&lo),
    crate::syntax::expr_to_input_form(&hi),
  ));
  true
}

/// Whether a complex plot's `{z, zmin, zmax}` corners share a real or an
/// imaginary part, so they span no rectangle; reports `head::plld` if so.
pub(crate) fn degenerate_complex_corners(head: &str, spec: &Expr) -> bool {
  let Expr::List(items) = spec else {
    return false;
  };
  if items.len() != 3 {
    return false;
  }
  let Expr::Identifier(var) = &items[0] else {
    return false;
  };
  let (Ok(lo), Ok(hi)) = (
    evaluate_expr_to_expr(&items[1]),
    evaluate_expr_to_expr(&items[2]),
  ) else {
    return false;
  };
  let parts = |e: &Expr| {
    crate::functions::math_ast::try_extract_complex_float(e)
      .or_else(|| try_eval_to_f64(e).map(|re| (re, 0.0)))
  };
  let (Some((ar, ai)), Some((br, bi))) = (parts(&lo), parts(&hi)) else {
    return false;
  };
  if ar != br && ai != bi {
    return false;
  }
  crate::emit_message(&format!(
    "{head}::plld: Corners for {var} in {{{var}, {}, {}}} must have \
     distinct machine-precision real and imaginary parts.",
    crate::syntax::expr_to_input_form(&lo),
    crate::syntax::expr_to_input_form(&hi),
  ));
  true
}

pub(crate) fn parse_iterator(
  spec: &Expr,
  label: &str,
) -> Result<(String, f64, f64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "{label}: iterator variable must be a symbol"
          )));
        }
      };
      let min_expr = evaluate_expr_to_expr(&items[1])?;
      let max_expr = evaluate_expr_to_expr(&items[2])?;
      let min_val = try_eval_to_f64(&min_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{label}: cannot evaluate iterator min to a number"
        ))
      })?;
      let max_val = try_eval_to_f64(&max_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{label}: cannot evaluate iterator max to a number"
        ))
      })?;
      Ok((var, min_val, max_val))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{label}: iterator must be {{var, min, max}}"
    ))),
  }
}

/// Evaluate the function body at given (x, y) values.
/// A plot body in two variables, compiled once and sampled many times;
/// where the compiled value is not a finite real the point is evaluated
/// symbolically the way [`evaluate_at_xy`] does.
pub(crate) struct PlotBody2<'a> {
  body: &'a Expr,
  xvar: &'a str,
  yvar: &'a str,
  compiled: Option<crate::functions::ode_ast::CompiledReal>,
}

impl<'a> PlotBody2<'a> {
  pub(crate) fn new(body: &'a Expr, xvar: &'a str, yvar: &'a str) -> Self {
    let compiled = crate::functions::ode_ast::CompiledReal::new(
      body,
      &[xvar.to_string(), yvar.to_string()],
    );
    Self {
      body,
      xvar,
      yvar,
      compiled,
    }
  }

  pub(crate) fn at(&self, x: f64, y: f64) -> Option<f64> {
    if let Some(c) = &self.compiled {
      let v = c.eval(&[x, y]);
      if v.is_finite() {
        return Some(v);
      }
    }
    evaluate_symbolically_at_xy(self.body, self.xvar, self.yvar, x, y)
  }
}

pub(crate) fn evaluate_at_xy(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<f64> {
  // A one-element list is a single surface (see below), so its element is
  // what gets compiled.
  let single = match body {
    Expr::List(items) if items.len() == 1 => &items[0],
    other => other,
  };
  if let Some(c) = compiled_body(single, &[xvar.to_string(), yvar.to_string()])
  {
    let v = c.eval(&[xval, yval]);
    if v.is_finite() {
      return Some(v);
    }
  }
  evaluate_symbolically_at_xy(body, xvar, yvar, xval, yval)
}

fn evaluate_symbolically_at_xy(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<f64> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  // A body that evaluates to a one-element list (e.g. a function returning
  // `{value}`) is a single surface, like `Plot3D[{f}, ...]`.
  let result = match &result {
    Expr::List(items) if items.len() == 1 => items[0].clone(),
    other => other.clone(),
  };
  if let Some(v) = try_eval_to_f64_lenient(&result) {
    return Some(v);
  }
  // The body may reference a variable (e.g. `lineA` holding `-5 - 3 x + 2 y`)
  // that only resolved to an x/y expression during evaluation — after the
  // substitution above already ran. Substitute into the evaluated form and
  // evaluate once more.
  let sub1 = substitute_var(&result, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  try_eval_to_f64_lenient(&result)
}

/// Simple SVG header for plots without plotters axes (ArrayPlot, charts).
pub(crate) fn svg_header(w: u32, h: u32, full_width: bool) -> String {
  let (bg, _, _, _, _) = plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  if full_width {
    format!(
      "<svg width=\"100%\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  } else {
    format!(
      "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"{bg_fill}\"/>\n"
    )
  }
}

/// Whether a data-point marker should be drawn: finite and within one plot
/// span of the visible range. Markers outside the range are not shown, and
/// one far outside it (a 1e300 value) overflows plotters' pixel arithmetic.
fn marker_in_view(x: f64, y: f64, xr: (f64, f64), yr: (f64, f64)) -> bool {
  let near = |v: f64, (lo, hi): (f64, f64)| {
    let span = (hi - lo).abs().max(f64::MIN_POSITIVE);
    v >= lo.min(hi) - span && v <= lo.max(hi) + span
  };
  x.is_finite() && y.is_finite() && near(x, xr) && near(y, yr)
}

thread_local! {
  /// Plot bodies compiled for numeric sampling, keyed by the body's address
  /// (checked structurally, as the address may be reused) and variables.
  static COMPILED_BODIES: std::cell::RefCell<
    Vec<(usize, Expr, Vec<String>, Option<std::rc::Rc<crate::functions::ode_ast::CompiledReal>>)>,
  > = const { std::cell::RefCell::new(Vec::new()) };
}

/// The compiled form of `body` over `vars`, compiled on first use.
fn compiled_body(
  body: &Expr,
  vars: &[String],
) -> Option<std::rc::Rc<crate::functions::ode_ast::CompiledReal>> {
  let key = std::ptr::from_ref::<Expr>(body) as usize;
  if let Some(hit) = COMPILED_BODIES.with(|cache| {
    cache
      .borrow()
      .iter()
      .find(|(k, e, v, _)| {
        *k == key
          && v.as_slice() == vars
          && crate::evaluator::pattern_matching::expr_equal(e, body)
      })
      .map(|(_, _, _, c)| c.clone())
  }) {
    return hit;
  }
  let compiled = crate::functions::ode_ast::CompiledReal::new(body, vars)
    .map(std::rc::Rc::new);
  COMPILED_BODIES.with(|cache| {
    let mut cache = cache.borrow_mut();
    if cache.len() >= 32 {
      cache.remove(0);
    }
    cache.push((key, body.clone(), vars.to_vec(), compiled.clone()));
  });
  compiled
}

pub(crate) fn evaluate_at_point(body: &Expr, var: &str, x: f64) -> Option<f64> {
  if let Some(c) = compiled_body(body, &[var.to_string()]) {
    let v = c.eval(&[x]);
    if v.is_finite() {
      return Some(v);
    }
  }
  let substituted = substitute_var(body, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  if let Some(v) = try_eval_to_f64_lenient(&result) {
    return Some(v);
  }
  // See evaluate_at_xy: body may be a bound variable that only resolved to
  // a function of var during evaluation.
  let substituted = substitute_var(&result, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  if let Some(v) = try_eval_to_f64_lenient(&result) {
    return Some(v);
  }
  // A body can reach the plot variable through a symbol that carries its own
  // stored value — a `Module` local left behind in a partly evaluated
  // `Which[…]`, say, holds `z$1 = x/a` — and textual substitution never
  // touches that stored `x`. Binding the variable as a value, the way
  // `Block[{x = …}, body]` does, lets those symbols see the sample too.
  if let Some(v) = evaluate_with_bound_var(body, var, x) {
    return Some(v);
  }
  // Plot evaluates its body once with the variable still symbolic and only
  // then puts the sample in, so a rule that matches on the variable
  // (`f[t] /. sol`, with `sol` an NDSolve result) still sees `t`. Putting
  // the number in first leaves `f[0.5]`, which no such rule matches.
  // Only for a body that substitutes into the variable itself (`/.`):
  // other bodies, like a fit, stay unplottable without `Evaluated -> True`.
  if !crate::syntax::expr_to_string(body).contains("/.") {
    return None;
  }
  let symbolic = evaluate_expr_to_expr(body).ok()?;
  let substituted = substitute_var(&symbolic, var, &Expr::Real(x));
  let result = evaluate_expr_to_expr(&substituted).ok()?;
  try_eval_to_f64_lenient(&result)
}

/// Evaluate `body` with `var` bound to `x` as a value (dynamic scoping, as in
/// `Block[{var = x}, body]`), restoring the previous binding afterwards.
fn evaluate_with_bound_var(body: &Expr, var: &str, x: f64) -> Option<f64> {
  let saved = crate::ENV.with(|e| {
    e.borrow_mut()
      .insert(var.to_string(), crate::StoredValue::ExprVal(Expr::Real(x)))
  });
  let result = evaluate_expr_to_expr(body)
    .ok()
    .and_then(|r| try_eval_to_f64_lenient(&r));
  crate::ENV.with(|e| match saved {
    Some(v) => {
      e.borrow_mut().insert(var.to_string(), v);
    }
    None => {
      e.borrow_mut().remove(var);
    }
  });
  result
}

/// Adaptively sample a function, adding more points where the function changes rapidly.
fn adaptive_sample(
  func_body: &Expr,
  var_name: &str,
  x_min: f64,
  x_max: f64,
  initial_n: usize,
  max_total: usize,
  max_depth: usize,
  monitor: Option<&Expr>,
) -> Vec<(f64, f64)> {
  // Evaluate one sample, firing the `EvaluationMonitor :> expr` hook (with
  // the plot variable bound to x) first, e.g. so `Sow[{x, f[x]}]` records
  // the sampled points.
  let sample = |x: f64| -> f64 {
    if let Some(m) = monitor {
      let sub = crate::syntax::substitute_variable(m, var_name, &Expr::Real(x));
      let _ = crate::evaluator::evaluate_expr_to_expr(&sub);
    }
    evaluate_at_point(func_body, var_name, x).unwrap_or(f64::NAN)
  };

  // Initial uniform sampling
  let step = (x_max - x_min) / (initial_n - 1) as f64;
  let mut points: Vec<(f64, f64)> = (0..initial_n)
    .map(|i| {
      // The last sample sits exactly on `x_max`, not a rounding error away.
      let x = if i + 1 == initial_n {
        x_max
      } else {
        x_min + i as f64 * step
      };
      (x, sample(x))
    })
    .collect();

  // A bend smaller than ~1/1000 of the curve's height is invisible: the
  // relative test below alone keeps halving the near-flat stretches around
  // every extremum. The height is the 5th..95th percentile spread of the
  // initial samples, so a pole does not flatten the rest of the curve.
  let visible_bend = {
    let mut ys: Vec<f64> = points
      .iter()
      .map(|p| p.1)
      .filter(|y| y.is_finite())
      .collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if ys.len() >= 2 {
      let at = |q: f64| ys[((ys.len() - 1) as f64 * q).round() as usize];
      (at(0.95) - at(0.05)) * 1e-3
    } else {
      0.0
    }
  };

  // Adaptive refinement passes, one level of `MaxRecursion` each.
  for _ in 0..max_depth {
    if points.len() >= max_total {
      break;
    }
    let mut new_points: Vec<(f64, f64)> = Vec::new();
    let budget = max_total - points.len();

    for i in 0..points.len().saturating_sub(1) {
      if new_points.len() >= budget {
        break;
      }
      let (x0, y0) = points[i];
      let (x1, y1) = points[i + 1];

      // Skip if interval is too small
      if (x1 - x0) < (x_max - x_min) * 1e-10 {
        continue;
      }

      let needs_refine = if !y0.is_finite() || !y1.is_finite() {
        // Refine where the curve starts or stops existing to find the
        // boundary; a stretch with no values at all has nothing to find.
        y0.is_finite() || y1.is_finite()
      } else if i + 2 < points.len() {
        // Check curvature using three consecutive points
        let (x2, y2) = points[i + 2];
        if y2.is_finite() {
          // Linear interpolation error: how much does the middle point
          // deviate from the line connecting its neighbors?
          let y_interp = y0 + (y2 - y0) * (x1 - x0) / (x2 - x0);
          let y_range = (y2 - y0).abs().max(1e-10);
          let bend = (y1 - y_interp).abs();
          bend / y_range > 0.05 && bend > visible_bend
        } else {
          true
        }
      } else if i >= 1 {
        // This is the rightmost interval in the current point list (no
        // `i + 2` neighbor yet), which is structurally true of the last
        // interval on every refinement pass — without this branch it would
        // never be curvature-checked at all and a fast-oscillating tail
        // (e.g. a high-frequency Sin) would stay a single straight segment
        // no matter how many `MaxRecursion` passes ran. Mirror the
        // look-ahead check using the point *before* x0 instead.
        let (xm1, ym1) = points[i - 1];
        if ym1.is_finite() {
          let y_interp = ym1 + (y1 - ym1) * (x0 - xm1) / (x1 - xm1);
          let y_range = (y1 - ym1).abs().max(1e-10);
          let bend = (y0 - y_interp).abs();
          bend / y_range > 0.05 && bend > visible_bend
        } else {
          true
        }
      } else {
        // Fewer than 3 points in the whole domain: no neighbor on either
        // side to assess curvature against, so refine to be safe.
        true
      };

      if needs_refine {
        let xm = f64::midpoint(x0, x1);
        new_points.push((xm, sample(xm)));
      }
    }

    if new_points.is_empty() {
      break;
    }

    // Merge new points into sorted order
    points.extend(new_points);
    points.sort_by(|a, b| {
      a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
  }

  points
}

/// Break a sampled curve at the `Exclusions` points: the curve runs up to
/// each excluded x from both sides, with an empty sample at x itself, so no
/// line is drawn across the gap.
fn break_at_exclusions(
  mut points: Vec<(f64, f64)>,
  func_body: &Expr,
  var_name: &str,
  excluded: &[f64],
  span: f64,
) -> Vec<(f64, f64)> {
  if excluded.is_empty() {
    return points;
  }
  let eps = span * 1e-9;
  for &x in excluded {
    // Drop samples that sit on the excluded point itself.
    points.retain(|&(px, _)| (px - x).abs() > eps);
    for dx in [-eps, 0.0, eps] {
      let y = if dx == 0.0 {
        f64::NAN
      } else {
        evaluate_at_point(func_body, var_name, x + dx).unwrap_or(f64::NAN)
      };
      points.push((x + dx, y));
    }
  }
  points
    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
  points
}

/// Whether `RegionFunction -> region` accepts the curve point `(x, y)`:
/// `region[x, y]` evaluates to `True`.
fn region_function_allows(region: &Expr, x: f64, y: f64) -> bool {
  let call = Expr::CurriedCall {
    func: Box::new(region.clone()),
    args: vec![Expr::Real(x), Expr::Real(y)],
  };
  matches!(
    evaluate_expr_to_expr(&call),
    Ok(Expr::Identifier(ref s)) if s == "True"
  )
}

/// Keep only the stretches of a sampled curve that `RegionFunction ->
/// region` accepts. Rejected samples become gaps (`NaN`), and each boundary
/// between an accepted and a rejected sample is located by bisection so the
/// drawn pieces end where the region does, not at the last sample inside.
fn clip_to_region_function(
  points: &[(f64, f64)],
  func_body: &Expr,
  var_name: &str,
  region: &Expr,
) -> Vec<(f64, f64)> {
  let inside: Vec<bool> = points
    .iter()
    .map(|&(x, y)| y.is_finite() && region_function_allows(region, x, y))
    .collect();
  let mut out = Vec::with_capacity(points.len() + 8);
  for (i, &(x, y)) in points.iter().enumerate() {
    if i > 0 && inside[i] != inside[i - 1] {
      let (x_prev, y_prev) = points[i - 1];
      if y_prev.is_finite() && y.is_finite() {
        // Bisect towards the boundary between the two samples.
        let (mut a, mut b) = (x_prev, x);
        let a_inside = inside[i - 1];
        for _ in 0..30 {
          let m = f64::midpoint(a, b);
          let ym =
            evaluate_at_point(func_body, var_name, m).unwrap_or(f64::NAN);
          let m_inside =
            ym.is_finite() && region_function_allows(region, m, ym);
          if m_inside == a_inside {
            a = m;
          } else {
            b = m;
          }
        }
        let edge = if a_inside { a } else { b };
        let y_edge =
          evaluate_at_point(func_body, var_name, edge).unwrap_or(f64::NAN);
        if a_inside {
          out.push((edge, y_edge));
          out.push((f64::midpoint(a, b), f64::NAN));
        } else {
          out.push((f64::midpoint(a, b), f64::NAN));
          out.push((edge, y_edge));
        }
      } else if !inside[i] {
        out.push((x, f64::NAN));
      }
    }
    out.push(if inside[i] { (x, y) } else { (x, f64::NAN) });
  }
  out
}

/// The automatic extent WL gives one coordinate of a plot's points
/// (`Charting`CommonDump`icomputeSceneRange`): with `lo`/`hi` the 17.5%
/// and 82.5% quantiles of the values and `min`/`max` their extremes, an
/// extreme further than 2.5 quantile spans beyond its quantile is an
/// outlier, and the range stops 1.5 spans beyond the quantile instead. An
/// extreme of exactly 0 is always kept.
pub(crate) fn outlier_clipped_extent(
  (lo, hi): (f64, f64),
  (min, max): (f64, f64),
) -> (f64, f64) {
  let (lo, hi) = if lo == hi { (min, max) } else { (lo, hi) };
  let span = hi - lo;
  let min = if min >= lo - 2.5 * span || min == 0.0 {
    min
  } else {
    lo - 1.5 * span
  };
  let max = if max <= hi + 2.5 * span || max == 0.0 {
    max
  } else {
    hi + 1.5 * span
  };
  (min, max)
}

/// `Quantile[values, q]` with WL's default parameters (the
/// `Ceiling[n q]`-th smallest) of the ascending `sorted` values.
pub(crate) fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
  let n = sorted.len();
  let k = ((n as f64) * q).ceil().clamp(1.0, n as f64) as usize;
  sorted[k - 1]
}

/// The automatic extent of a list of values: the whole extent when
/// `keep_outliers` (`PlotRange -> All`), else [`outlier_clipped_extent`]
/// of their quantiles. `None` when there is no finite value.
pub(crate) fn list_value_extent(
  values: impl IntoIterator<Item = f64>,
  keep_outliers: bool,
) -> Option<(f64, f64)> {
  let mut sorted: Vec<f64> =
    values.into_iter().filter(|v| v.is_finite()).collect();
  if sorted.is_empty() {
    return None;
  }
  sorted.sort_by(f64::total_cmp);
  let extent = (sorted[0], sorted[sorted.len() - 1]);
  if keep_outliers {
    return Some(extent);
  }
  let quantiles = (
    sorted_quantile(&sorted, 0.175),
    sorted_quantile(&sorted, 0.825),
  );
  Some(outlier_clipped_extent(quantiles, extent))
}

/// The automatic y extent of the drawn curves of a `Plot`: the y extent of
/// every stretch of two or more finite points (an isolated point, such as
/// a step function's value at a jump on the end of the domain, is not
/// drawn), with outliers clipped as [`outlier_clipped_extent`] does unless
/// `keep_outliers`. The quantiles weigh each sample by the stretch of x it
/// stands for, so the refinement that piles samples up at a pole does not
/// skew them: they are the quantiles of the curve over its domain.
fn drawn_y_range(
  curves: &[Vec<(f64, f64)>],
  keep_outliers: bool,
) -> (f64, f64) {
  let mut weighted: Vec<(f64, f64)> = Vec::new();
  for curve in curves {
    for segment in split_into_segments(curve) {
      let n = segment.len();
      for i in 0..n {
        let left = segment[i.saturating_sub(1)].0;
        let right = segment[(i + 1).min(n - 1)].0;
        weighted.push((segment[i].1, ((right - left) / 2.0).abs()));
      }
    }
  }
  if weighted.is_empty() {
    return (-1.0, 1.0);
  }
  weighted.sort_by(|a, b| a.0.total_cmp(&b.0));
  let extent = (weighted[0].0, weighted[weighted.len() - 1].0);
  if keep_outliers {
    return extent;
  }
  let total: f64 = weighted.iter().map(|&(_, w)| w).sum();
  if !(total > 0.0) {
    return extent;
  }
  let quantile = |q: f64| {
    let mut acc = 0.0;
    for &(y, w) in &weighted {
      acc += w;
      if acc >= q * total {
        return y;
      }
    }
    extent.1
  };
  outlier_clipped_extent((quantile(0.175), quantile(0.825)), extent)
}

/// Whether `Exclusions` leaves the automatic exclusions on: the option is
/// absent, `Automatic`, or a list that includes `Automatic`.
fn automatic_exclusions_enabled(options: &[Expr]) -> bool {
  let value = options.iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Exclusions") => {
      Some(replacement.as_ref())
    }
    _ => None,
  });
  let is_auto = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "Automatic");
  match value {
    None => true,
    Some(Expr::List(items)) => items.iter().any(is_auto),
    Some(v) => is_auto(v),
  }
}

/// Whether the step from `a` to `b` between neighbouring samples is a pole
/// rather than a steep stretch of curve, for the y range `[lo, hi]`: both
/// lie on opposite sides, each more than a range height beyond its edge. A
/// steep but continuous curve stays within a range height of the edges.
fn is_pole_jump(a: f64, b: f64, lo: f64, hi: f64) -> bool {
  let height = hi - lo;
  (a > hi + height && b < lo - height) || (a < lo - height && b > hi + height)
}

/// Whether `body` takes a function with a branch cut (`Sqrt`, `Log`, a
/// fractional power, …) of something complex, where its value can jump as
/// the argument crosses the cut.
fn has_branch_cut(body: &Expr) -> bool {
  const BRANCH_HEADS: [&str; 9] = [
    "Sqrt", "Log", "Power", "ArcSin", "ArcCos", "ArcTan", "ArcCosh", "ArcTanh",
    "Arg",
  ];
  fn is_complex(e: &Expr) -> bool {
    matches!(e, Expr::Identifier(n) if n == "I")
      || matches!(e, Expr::FunctionCall { name, .. } if name == "Complex")
      || crate::syntax::expr_children(e).into_iter().any(is_complex)
  }
  fn branch(e: &Expr) -> bool {
    if let Expr::FunctionCall { name, args } = e
      && BRANCH_HEADS.contains(&name.as_str())
      && args.iter().any(is_complex)
    {
      return true;
    }
    crate::syntax::expr_children(e).into_iter().any(branch)
  }
  branch(body)
}

/// Whether `body` contains a function that jumps between constant (or
/// separately continuous) pieces, whose discontinuities `Exclusions ->
/// Automatic` breaks the curve at.
fn has_step_function(body: &Expr) -> bool {
  const STEP_HEADS: [&str; 13] = [
    "Floor",
    "Ceiling",
    "Round",
    "IntegerPart",
    "FractionalPart",
    "Mod",
    "Quotient",
    "Sign",
    "UnitStep",
    "HeavisideTheta",
    "Boole",
    "SquareWave",
    "SawtoothWave",
  ];
  if matches!(body, Expr::FunctionCall { name, .. } if STEP_HEADS.contains(&name.as_str()))
  {
    return true;
  }
  crate::syntax::expr_children(body)
    .into_iter()
    .any(has_step_function)
}

/// The x positions where the sampled curve `points` of `body` jumps: each
/// pair of neighbouring samples with different values is bisected towards
/// the larger change, and a gap that stays open once the interval has
/// shrunk to nothing is a discontinuity (a continuous stretch closes up).
fn jump_points(points: &[(f64, f64)], body: &Expr, var: &str) -> Vec<f64> {
  let finite: Vec<f64> = points
    .iter()
    .map(|&(_, y)| y)
    .filter(|y| y.is_finite())
    .collect();
  let span = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    - finite.iter().copied().fold(f64::INFINITY, f64::min);
  let tolerance = 1e-6 * span.abs().max(1.0);
  let mut jumps: Vec<f64> = Vec::new();
  for pair in points.windows(2) {
    let ((mut a, mut ya), (mut b, mut yb)) = (pair[0], pair[1]);
    if !(ya.is_finite() && yb.is_finite()) || (yb - ya).abs() <= tolerance {
      continue;
    }
    for _ in 0..60 {
      let m = f64::midpoint(a, b);
      if m <= a || m >= b {
        break;
      }
      let Some(ym) = evaluate_at_point(body, var, m).filter(|v| v.is_finite())
      else {
        break;
      };
      if (yb - ya).abs() <= tolerance {
        break;
      }
      if (ym - ya).abs() >= (yb - ym).abs() {
        b = m;
        yb = ym;
      } else {
        a = m;
        ya = ym;
      }
    }
    if (yb - ya).abs() > tolerance {
      let x = f64::midpoint(a, b);
      if jumps
        .last()
        .is_none_or(|&last| (x - last).abs() > 1e-9 * (b.abs().max(1.0)))
      {
        jumps.push(x);
      }
    }
  }
  jumps
}

/// True when a `PlotRange` value asks for the whole y extent: `All` itself, or
/// `All` in the y slot of `{xrange, All}`.
fn plot_range_requests_all_y(value: &Expr) -> bool {
  plot_range_all_axes(value).1
}

/// Which axes a `PlotRange` value asks to show the whole extent of, outliers
/// included: both for `All`, one for `All` in its slot of `{xrange, yrange}`.
pub(crate) fn plot_range_all_axes(value: &Expr) -> (bool, bool) {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let is_all = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "All");
  match &val {
    e if is_all(e) => (true, true),
    Expr::List(items) if items.len() == 2 => {
      (is_all(&items[0]), is_all(&items[1]))
    }
    _ => (false, false),
  }
}

/// Split points into contiguous finite segments, breaking at NaN/Infinity
/// in either coordinate (a `ParametricPlot` curve can diverge in x too).
pub(crate) fn split_into_segments(
  points: &[(f64, f64)],
) -> Vec<Vec<(f64, f64)>> {
  let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
  let mut current: Vec<(f64, f64)> = Vec::new();

  for &(x, y) in points {
    if x.is_finite() && y.is_finite() {
      current.push((x, y));
    } else if current.len() > 1 {
      segments.push(std::mem::take(&mut current));
    } else {
      current.clear();
    }
  }
  if current.len() > 1 {
    segments.push(current);
  }
  segments
}

/// Finite segments of `points` with every y clamped into `[y_min, y_max]`.
/// A filled region keeps covering the plot area where the curve itself has
/// left the plot range, so it follows the clamped curve instead of the
/// clipped one.
pub(crate) fn clamp_segments_to_y_range(
  points: &[(f64, f64)],
  y_min: f64,
  y_max: f64,
) -> Vec<Vec<(f64, f64)>> {
  split_into_segments(points)
    .into_iter()
    .map(|seg| {
      seg
        .into_iter()
        .map(|(x, y)| (x, y.clamp(y_min, y_max)))
        .collect()
    })
    .collect()
}

/// Clip line segments to a y-range, interpolating at boundaries.
/// Points outside the range are removed and the line is split,
/// with interpolated points added at the boundary crossings.
fn clip_segments_to_y_range(
  segments: Vec<Vec<(f64, f64)>>,
  y_min: f64,
  y_max: f64,
  join_across_jumps: bool,
) -> Vec<Vec<(f64, f64)>> {
  let mut result: Vec<Vec<(f64, f64)>> = Vec::new();

  for segment in segments {
    let mut current: Vec<(f64, f64)> = Vec::new();

    for i in 0..segment.len() {
      let (x, y) = segment[i];
      let inside = y >= y_min && y <= y_max;

      if i > 0 {
        let (px, py) = segment[i - 1];
        let prev_inside = py >= y_min && py <= y_max;

        if prev_inside != inside {
          // Line crosses the boundary — interpolate
          let boundary = if !inside {
            if y > y_max { y_max } else { y_min }
          } else if py > y_max {
            y_max
          } else {
            y_min
          };
          let t = (boundary - py) / (y - py);
          let bx = px + t * (x - px);
          current.push((bx, boundary));

          if !inside {
            // Leaving the range — flush segment
            if current.len() > 1 {
              result.push(std::mem::take(&mut current));
            } else {
              current.clear();
            }
          }
        } else if !prev_inside && !inside {
          // Both outside — check if the line passes through the range
          // (e.g. one above y_max and the other below y_min)
          // A pole (e.g. `1/x` at 0) also jumps from one side to the other
          // between neighbouring samples, but both samples lie far outside
          // the range (a full range height or more); that gap must not be
          // bridged by a vertical line. A steep but continuous curve stays
          // within a range height of the boundaries.
          let far_apart = is_pole_jump(py, y, y_min, y_max);
          if (join_across_jumps || !far_apart)
            && ((py > y_max && y < y_min) || (py < y_min && y > y_max))
          {
            // Crosses both boundaries
            let t_min = (y_min - py) / (y - py);
            let t_max = (y_max - py) / (y - py);
            let (t1, t2) = if t_min < t_max {
              (t_min, t_max)
            } else {
              (t_max, t_min)
            };
            let x1 = px + t1 * (x - px);
            let y1 = py + t1 * (y - py);
            let x2 = px + t2 * (x - px);
            let y2 = py + t2 * (y - py);
            result.push(vec![(x1, y1), (x2, y2)]);
          }
        }
      }

      if inside {
        current.push((x, y));
      }
    }

    if current.len() > 1 {
      result.push(current);
    }
  }

  result
}

/// Clip line segments to an x-range (the y-range clipper with the axes
/// swapped). Curves such as `ParametricPlot` can leave the plot area
/// horizontally by many orders of magnitude.
fn clip_segments_to_x_range(
  segments: Vec<Vec<(f64, f64)>>,
  x_min: f64,
  x_max: f64,
) -> Vec<Vec<(f64, f64)>> {
  let swap = |segs: Vec<Vec<(f64, f64)>>| -> Vec<Vec<(f64, f64)>> {
    segs
      .into_iter()
      .map(|seg| seg.into_iter().map(|(a, b)| (b, a)).collect())
      .collect()
  };
  swap(clip_segments_to_y_range(
    swap(segments),
    x_min,
    x_max,
    false,
  ))
}

/// A curve broken at `x` by an exclusion, with its values just left and
/// right of the break.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExclusionMark {
  pub x: f64,
  pub y_left: f64,
  pub y_right: f64,
}

/// Parse an `ExclusionsStyle` value into the styles of the joining line and
/// of the end points: a single style is the line's, `{line, points}` gives
/// both, and `None` leaves a part out.
pub(crate) fn parse_exclusions_style(value: &Expr) -> [Option<SeriesStyle>; 2] {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let part = |e: &Expr| -> Option<SeriesStyle> {
    match e {
      Expr::Identifier(s) if s == "None" => None,
      // A style that sets nothing (`Automatic`) draws in the default one.
      other => Some(parse_style_directives(other).unwrap_or_default()),
    }
  };
  match &val {
    Expr::List(items) if items.len() == 2 => [part(&items[0]), part(&items[1])],
    other => [part(other), None],
  }
}

/// The marks of `ExclusionsStyle` as SVG, in the render space of the plot
/// area at (`plot_x0`, `plot_y0`) of size `plot_w` × `plot_h` showing
/// `x_range` × `y_range`: each break's ends joined by a line, cut to the
/// y range `clip` (black, 1 pt wide unless styled; a break the curve does
/// not jump at draws as a square dot), and its two ends as points.
#[allow(clippy::too_many_arguments)]
fn render_exclusion_marks(
  opts: &PlotOptions,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  render_w: f64,
  (x_min, x_max): (f64, f64),
  (y_min, y_max): (f64, f64),
  (clip_min, clip_max): (f64, f64),
) -> String {
  let (xr, yr) = (x_max - x_min, y_max - y_min);
  if !(xr.abs() > 1e-300 && yr.abs() > 1e-300) {
    return String::new();
  }
  let px = |x: f64| plot_x0 + (x - x_min) / xr * plot_w;
  let py = |y: f64| plot_y0 + (y_max - y) / yr * plot_h;
  let scale = RESOLUTION_SCALE as f64;
  let rgb = |style: &SeriesStyle| {
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    style
      .color
      .as_ref()
      .map_or((0, 0, 0), |c| (ch(c.r), ch(c.g), ch(c.b)))
  };
  let opacity = |style: &SeriesStyle| match style.opacity {
    Some(a) if a < 1.0 => {
      format!(" stroke-opacity=\"{a}\" fill-opacity=\"{a}\"")
    }
    _ => String::new(),
  };
  let mut svg = String::new();
  for mark in &opts.exclusion_marks {
    if !(mark.x > x_min && mark.x < x_max) {
      continue;
    }
    if let Some(style) = &opts.exclusions_style[0] {
      let finite = |v: f64| if v.is_nan() { None } else { Some(v) };
      if let (Some(a), Some(b)) = (finite(mark.y_left), finite(mark.y_right)) {
        let (lo, hi) = (a.min(b).max(clip_min), a.max(b).min(clip_max));
        if lo <= hi {
          let (r, g, b) = rgb(style);
          let width = style.thickness.unwrap_or(1.0) * scale;
          let dash = style
            .dashing
            .as_ref()
            .map(|d| {
              let lens: Vec<String> = d
                .iter()
                .map(|&l| format!("{:.1}", dash_len(l, render_w, scale)))
                .collect();
              format!(" stroke-dasharray=\"{}\"", lens.join(","))
            })
            .unwrap_or_default();
          if (py(hi) - py(lo)).abs() < 0.5 {
            // The zero-length line WL strokes with square caps: a square
            // as wide as the line.
            svg.push_str(&format!(
              "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{width:.1}\" \
               height=\"{width:.1}\" fill=\"rgb({r},{g},{b})\"{}/>\n",
              px(mark.x) - width / 2.0,
              py(hi) - width / 2.0,
              opacity(style),
            ));
          } else {
            svg.push_str(&format!(
              "<line x1=\"{x:.1}\" y1=\"{y1:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" \
               stroke=\"rgb({r},{g},{b})\" stroke-width=\"{width:.1}\"{dash}{op}/>\n",
              x = px(mark.x),
              y1 = py(hi),
              y2 = py(lo),
              op = opacity(style),
            ));
          }
        }
      }
    }
    if let Some(style) = &opts.exclusions_style[1] {
      let (r, g, b) = rgb(style);
      let radius = match style.point_size {
        Some(f) if f > 0.0 => f * render_w * 0.5,
        Some(p) => -p * 0.5 * scale,
        None => 1.5 * scale,
      };
      for y in [mark.y_left, mark.y_right] {
        if y.is_finite() && y >= clip_min && y <= clip_max {
          svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{radius:.1}\" \
             fill=\"rgb({r},{g},{b})\"{}/>\n",
            px(mark.x),
            py(y),
            opacity(style),
          ));
        }
      }
    }
  }
  svg
}

/// How one edge of the plot range draws the stretches where a curve is cut
/// off there (`ClippingStyle`).
#[derive(Clone, Debug, Default)]
pub(crate) struct ClipEdgeStyle {
  /// Directives overriding the curve's own style.
  pub style: SeriesStyle,
  /// `Automatic`: the curve's style, dotted.
  pub dotted: bool,
}

/// Parse a `ClippingStyle` value into its bottom and top edge styles:
/// `None` draws nothing, `Automatic` a dotted line in the curve's style, a
/// style (`Red`, `Directive[…]`) both edges in it, and `{bottom, top}` each
/// edge in its own.
pub(crate) fn parse_clipping_style(value: &Expr) -> [Option<ClipEdgeStyle>; 2] {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let edge = |e: &Expr| -> Option<ClipEdgeStyle> {
    match e {
      Expr::Identifier(s) if s == "None" || s == "False" => None,
      Expr::Identifier(s) if s == "Automatic" || s == "True" => {
        Some(ClipEdgeStyle {
          style: SeriesStyle::default(),
          dotted: true,
        })
      }
      other => parse_style_directives(other).map(|style| ClipEdgeStyle {
        style,
        dotted: false,
      }),
    }
  };
  match &val {
    Expr::List(items) if items.len() == 2 => [edge(&items[0]), edge(&items[1])],
    other => {
      let e = edge(other);
      [e.clone(), e]
    }
  }
}

/// The stretches where the finite curve `segment` lies below `y_min`
/// (`false`) or above `y_max` (`true`), as the x interval along that edge
/// from where the curve leaves the range to where it comes back (or ends).
fn clipped_edge_runs(
  segment: &[(f64, f64)],
  y_min: f64,
  y_max: f64,
) -> Vec<(bool, f64, f64)> {
  // -1 below, 0 inside, 1 above.
  let side = |y: f64| {
    if y > y_max {
      1
    } else if y < y_min {
      -1
    } else {
      0
    }
  };
  let crossing = |(px, py): (f64, f64), (x, y): (f64, f64), level: f64| {
    if y == py {
      x
    } else {
      px + (level - py) / (y - py) * (x - px)
    }
  };
  let mut runs = Vec::new();
  let mut open: Option<(bool, f64)> = None;
  for (i, &(x, y)) in segment.iter().enumerate() {
    let here = side(y);
    if i == 0 {
      if here != 0 {
        open = Some((here > 0, x));
      }
      continue;
    }
    let prev = segment[i - 1];
    if side(prev.1) == here {
      continue;
    }
    if let Some((top, start)) = open.take() {
      let level = if top { y_max } else { y_min };
      runs.push((top, start, crossing(prev, (x, y), level)));
    }
    if here != 0 {
      let level = if here > 0 { y_max } else { y_min };
      open = Some((here > 0, crossing(prev, (x, y), level)));
    }
  }
  if let (Some((top, start)), Some(&(x, _))) = (open, segment.last()) {
    runs.push((top, start, x));
  }
  runs
}

/// Compute a "nice" major tick step given the axis range and desired label
/// count. The automatic ticks of a plot axis use [`axis_tick_step`].
pub(crate) fn nice_step(range: f64, target_labels: usize) -> f64 {
  let raw = range / target_labels as f64;
  let mag = 10_f64.powf(raw.abs().log10().floor());
  let norm = raw / mag;
  // The Wolfram Language's own set of step multipliers — 2.5 included, which
  // is what makes a 0..15 axis step by 2.5 rather than 5.
  let nice = if norm <= 1.0 {
    1.0
  } else if norm <= 2.0 {
    2.0
  } else if norm <= 2.5 {
    2.5
  } else if norm <= 5.0 {
    5.0
  } else {
    10.0
  };
  nice * mag
}

/// The major tick step of an automatic axis spanning `range` (the plotted,
/// padded span): the smallest of 1, 2 and 5 × 10^k that cuts it into fewer
/// than 8 intervals, as the WL 15 front end picks them for `Ticks ->
/// Automatic` (`AbsoluteOptions[g, Ticks]` shows them; independent of the
/// image size). So 0..7.5 steps by 1, 0..8 by 2 and 0..16 by 5.
pub(crate) fn axis_tick_step(range: f64) -> f64 {
  let range = range.abs();
  if !(range.is_finite() && range > 0.0) {
    return 1.0;
  }
  let mut mag = 10_f64.powf((range / 8.0).log10().floor());
  loop {
    for m in [1.0, 2.0, 5.0] {
      let step = m * mag;
      if range / step < 8.0 * (1.0 - 1e-12) {
        return step;
      }
    }
    mag *= 10.0;
  }
}

/// How many minor intervals split the interval between two automatic major
/// ticks `major` apart: 4 for a step of 2 × 10^k, 5 otherwise (WL 15).
pub(crate) fn minor_tick_divisions(major: f64) -> usize {
  let mantissa = major / 10_f64.powf(major.abs().log10().floor());
  if (mantissa - 2.0).abs() < 1e-6 { 4 } else { 5 }
}

/// The minor tick spacing between automatic majors `major` apart.
pub(crate) fn minor_tick_step(major: f64) -> f64 {
  major / minor_tick_divisions(major) as f64
}

/// Check whether a tick value falls on a major tick grid.
pub(crate) fn is_major_tick(v: f64, step: f64) -> bool {
  if step == 0.0 {
    return true;
  }
  let remainder = (v / step).round() * step - v;
  remainder.abs() < step * 1e-9
}

/// Format an AbsoluteTime value (seconds since 1900-01-01) as a date string.
pub(crate) fn format_date_tick(seconds: f64) -> String {
  let (year, month, day, _, _, _) =
    crate::functions::datetime_ast::absolute_seconds_to_date(seconds);
  let month_abbr = || match month {
    1 => "Jan",
    2 => "Feb",
    3 => "Mar",
    4 => "Apr",
    5 => "May",
    6 => "Jun",
    7 => "Jul",
    8 => "Aug",
    9 => "Sep",
    10 => "Oct",
    11 => "Nov",
    12 => "Dec",
    _ => "???",
  };
  if day == 1 && month == 1 {
    format!("{year}")
  } else if day == 1 {
    format!("{year} {}", month_abbr())
  } else {
    format!("{year} {} {day}", month_abbr())
  }
}

/// Date tick step specification for nice date axis ticks.
#[derive(Clone, Copy)]
pub(crate) enum DateStep {
  Years(i64),
  Months(i64),
  Days(i64),
}

/// Compute a nice step for date axis ticks based on the range in seconds.
fn nice_date_step_spec(range_seconds: f64) -> DateStep {
  let range_days = range_seconds / 86400.0;
  let range_years = range_days / 365.25;

  if range_years > 100.0 {
    let step = nice_step(range_years, 5) as i64;
    DateStep::Years(step.max(10))
  } else if range_years > 20.0 {
    DateStep::Years(5)
  } else if range_years > 8.0 {
    DateStep::Years(2)
  } else if range_years > 2.0 {
    DateStep::Years(1)
  } else if range_days > 180.0 {
    DateStep::Months(3)
  } else if range_days > 60.0 {
    DateStep::Months(1)
  } else {
    let step = nice_step(range_days, 5) as i64;
    DateStep::Days(step.max(1))
  }
}

/// Generate nice date tick positions between min and max AbsoluteTime values.
pub(crate) fn generate_date_ticks(x_min: f64, x_max: f64) -> Vec<f64> {
  use crate::functions::datetime_ast::{
    absolute_seconds_to_date, date_to_absolute_seconds,
  };

  let range = x_max - x_min;
  let step = nice_date_step_spec(range);
  let mut ticks = Vec::new();

  let (start_y, start_m, start_d, _, _, _) = absolute_seconds_to_date(x_min);

  match step {
    DateStep::Years(n) => {
      // Round start year down to multiple of n
      let mut y = (start_y / n) * n;
      if y > start_y {
        y -= n;
      }
      loop {
        let t = date_to_absolute_seconds(y, 1, 1, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        y += n;
      }
    }
    DateStep::Months(n) => {
      let mut y = start_y;
      let mut m = ((start_m - 1) / n) * n + 1;
      loop {
        let t = date_to_absolute_seconds(y, m, 1, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        m += n;
        while m > 12 {
          m -= 12;
          y += 1;
        }
      }
    }
    DateStep::Days(n) => {
      // Round start day down to multiple of n
      let mut y = start_y;
      let mut m = start_m;
      let mut d = ((start_d - 1) / n) * n + 1;
      loop {
        let t = date_to_absolute_seconds(y, m, d, 0, 0, 0.0);
        if t > x_max {
          break;
        }
        if t >= x_min {
          ticks.push(t);
        }
        // Advance by n days and normalize
        let next_t = t + (n as f64) * 86400.0;
        let (ny, nm, nd, _, _, _) = absolute_seconds_to_date(next_t);
        y = ny;
        m = nm;
        d = nd;
      }
    }
  }

  ticks
}

/// Approximate step in seconds for date ticks (used for tick count estimation).
fn nice_date_step(range_seconds: f64) -> f64 {
  match nice_date_step_spec(range_seconds) {
    DateStep::Years(n) => (n as f64) * 365.25 * 86400.0,
    DateStep::Months(n) => (n as f64) * 30.44 * 86400.0,
    DateStep::Days(n) => (n as f64) * 86400.0,
  }
}

/// Format a tick value, dropping the trailing ".0" for integers.
/// The SVG for an `AxesLabel`, placed the way the Wolfram Language does: the
/// x label just past the right end of the x axis and level with it, the y
/// label above the top of the y axis. (A `FrameLabel`, by contrast, is
/// centred outside the plot area — see the callers.)
///
/// `area` is the plotting rectangle `(x0, y0, w, h)` in render units and
/// `range` the data range `(x_min, x_max, y_min, y_max)` it maps, so the
/// labels sit on the axis lines themselves when those fall inside the plot.
pub(crate) fn axes_label_svg(
  axes_label: Option<&(String, String)>,
  (x0, y0, w, h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  axes: (bool, bool),
  font_size: f64,
  label_fill: &str,
) -> String {
  // The axis lines: where y = 0 and x = 0 fall, clamped into the plot area
  // (an all-positive range draws its axes along the bottom and left edges).
  let axis_y = if y_max > y_min {
    y0 + h - ((0.0 - y_min) / (y_max - y_min)).clamp(0.0, 1.0) * h
  } else {
    y0 + h
  };
  let axis_x = if x_max > x_min {
    x0 + ((0.0 - x_min) / (x_max - x_min)).clamp(0.0, 1.0) * w
  } else {
    x0
  };
  axes_label_svg_at(
    axes_label,
    (x0, y0, w, h),
    (axis_x, axis_y),
    axes,
    font_size,
    label_fill,
    None,
  )
}

/// [`axes_label_svg`] for axes that cross at the render-space point
/// `(axis_x, axis_y)` — the vertical axis's x and the horizontal one's y.
pub(crate) fn axes_label_svg_at(
  axes_label: Option<&(String, String)>,
  (x0, y0, w, _h): (f64, f64, f64, f64),
  (axis_x, axis_y): (f64, f64),
  (show_x_axis, show_y_axis): (bool, bool),
  font_size: f64,
  label_fill: &str,
  canvas_width: Option<f64>,
) -> String {
  let Some((x_label, y_label)) = axes_label else {
    return String::new();
  };
  let mut svg = String::new();
  if show_x_axis && !x_label.is_empty() {
    // 8 pt past the end of the axis, vertically centred on it.
    let k = font_size / plot_padding::LABEL_FONT;
    let label_x = x0 + w + plot_padding::AXES_LABEL_GAP_X * k;
    // A label wider than the room left of the canvas edge (an
    // `ImagePadding` can state less than it needs) shrinks to fit rather
    // than being clipped.
    let label_font = match canvas_width {
      Some(cw) => {
        let width = crate::functions::graphics::box_string_visible_len(x_label)
          as f64
          * font_size
          * 0.55;
        let room = cw - label_x;
        if width > room && room > 0.0 {
          (font_size * room / width).max(font_size * 0.6)
        } else {
          font_size
        }
      }
      None => font_size,
    };
    svg.push_str(&format!(
      "<text x=\"{label_x:.1}\" y=\"{:.1}\" text-anchor=\"start\" \
       font-family=\"sans-serif\" \
       font-size=\"{label_font:.0}\" fill=\"{label_fill}\">{}</text>\n",
      axis_y + plot_padding::ascent(label_font)
        - plot_padding::line_height(label_font) / 2.0,
      crate::functions::graphics::box_string_to_svg_scaled(
        x_label,
        RESOLUTION_SCALE as f64
      )
    ));
  }
  if show_y_axis && !y_label.is_empty() {
    // Centred over the top of the y axis, but never so far left that the
    // label runs off the image — a wide one (`{a, b, c, d}`) over an axis
    // that sits near the left edge slides right until it fits, the way
    // Wolfram keeps an axis label inside the graphic.
    let half_width = crate::functions::graphics::box_string_visible_len(y_label)
      as f64
      * font_size
      * 0.55
      / 2.0;
    let lx = axis_x.max(half_width + font_size * 0.2);
    svg.push_str(&format!(
      "<text x=\"{lx:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
       fill=\"{label_fill}\">{}</text>\n",
      // 6 pt above the top of the axis.
      y0 - plot_padding::AXES_LABEL_GAP_Y * font_size
        / plot_padding::LABEL_FONT
        - plot_padding::descent(font_size),
      crate::functions::graphics::box_string_to_svg_scaled(
        y_label,
        RESOLUTION_SCALE as f64
      )
    ));
  }
  svg
}

/// Label a tick that belongs to a sequence of `step`-spaced ticks.
///
/// The Wolfram Language gives every label in a tick set the same number of
/// decimals — the number the step itself needs — so an axis stepping by 0.5
/// reads `-1.0, -0.5, 0.0, 0.5, 1.0`, not `-1, -0.5, 0, 0.5, 1`, while one
/// stepping by 2 reads `0, 2, 4`.
pub(crate) fn format_tick_with_step(v: f64, step: f64) -> String {
  // A tick set spaced finer than `10^-5` lives entirely in scientific
  // notation: its labels are not integers however close to one they sit
  // (`2*10^-36` is not the axis zero), only a tick a hair off zero
  // relative to the step is.
  if step.is_finite() && step != 0.0 && step.abs() < 1e-5 {
    if v.abs() < step.abs() * 1e-6 {
      return "0".to_string();
    }
    return format_scientific_tick(v);
  }
  let decimals = tick_step_decimals(step);
  if decimals == 0 {
    return format_tick(v);
  }
  let magnitude = v.abs();
  // A tick a hair off zero is the axis zero, not a tiny scientific value:
  // scale the smallness test against the step, not against an absolute
  // threshold, so `0.5`-spaced ticks label their origin `0.0`.
  let effectively_zero = magnitude < step.abs() * 1e-6;
  if magnitude >= 1e6 || (!effectively_zero && magnitude < 1e-5) {
    return format_tick(v);
  }
  let v = if effectively_zero { 0.0 } else { v };
  format!("{v:.decimals$}")
}

/// How many decimals it takes to write `step` — the decimals every label of
/// a `step`-spaced tick set carries. Capped at 6, the precision
/// [`format_tick`] itself works to.
fn tick_step_decimals(step: f64) -> usize {
  let step = step.abs();
  if !step.is_finite() || step <= 0.0 {
    return 0;
  }
  (0..=6)
    .find(|d| {
      let scaled = step * 10f64.powi(*d as i32);
      (scaled - scaled.round()).abs() < 1e-6 * scaled.max(1.0)
    })
    .unwrap_or(6)
}

/// `v` as `m×10^e` with Unicode superscript exponent digits.
fn format_scientific_tick(v: f64) -> String {
  let exp = v.abs().log10().floor() as i32;
  let mantissa = v / 10f64.powi(exp);
  let mantissa = if (mantissa - mantissa.round()).abs() < 1e-9 {
    format!("{}", mantissa.round() as i64)
  } else {
    let s = format!("{mantissa:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
  };
  format!(
    "{mantissa}\u{d7}10{}",
    crate::functions::graphics::to_unicode_script(&exp.to_string(), true)
  )
}

pub(crate) fn format_tick(v: f64) -> String {
  // Outside `[10^-5, 10^6)` the Wolfram Language labels a tick in
  // scientific notation — a frequency axis running to 6*10^15 is labelled
  // `2x10^15` with real superscript digits, not `2000000000000000`. The
  // exponent uses the Unicode superscripts so it survives as plain text.
  // A value that is an integer within float noise prints as that integer —
  // a tick that lands on `-1.11*10^-16` is the axis zero, not a tiny
  // scientific value.
  let magnitude = v.abs();
  // Checked before the cast: a value past the i64 range would saturate to
  // `i64::MIN`, whose `abs` overflows.
  if magnitude < 1e6 && (v - v.round()).abs() < 1e-9 {
    return format!("{}", v.round() as i64);
  }
  if magnitude >= 1e6 || (magnitude > 0.0 && magnitude < 1e-5) {
    return format_scientific_tick(v);
  }
  // Enough decimals to tell neighbouring ticks apart — a 0.0002 step used
  // to render every label as `0.0`.
  let s = format!("{v:.6}");
  s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Inject SVG `<line>` elements extending labeled (major) ticks a few pixels
/// further out than the minor ticks drawn by plotters. This is called after
/// plotters renders the chart, using post-render coordinates in the final SVG
/// space.
///
/// `x_axis` and `y_axis` are `Some((min, max, major_step))` if that axis
/// should get extensions, or `None` to skip (e.g. for log/date axes where
/// plotters places ticks itself and linear spacing is wrong).
///
/// The extension is drawn from `minor_len` to `major_len` along the tick
/// direction, so it connects seamlessly with the plotters-drawn tick.
#[allow(clippy::too_many_arguments)]
fn inject_major_tick_extensions(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
  minor_len: f64,
  major_len: f64,
  stroke_w: f64,
  color: &str,
) {
  let extension = major_len - minor_len;
  if extension <= 0.0 {
    return;
  }
  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut svg = String::new();

  // X axis: major ticks extend downward (below the plot area).
  if let Some((x_min, x_max, x_major)) = x_axis
    && x_major > 0.0
    && x_max > x_min
  {
    let axis_y = plot_y0 + plot_h;
    let y1 = axis_y + minor_len;
    let y2 = y1 + extension;
    let eps = x_major * 1e-6;
    let mut v = (x_min / x_major).ceil() * x_major;
    let max_steps = ((x_max - x_min) / x_major).abs() as usize + 4;
    for _ in 0..max_steps {
      if v > x_max + eps {
        break;
      }
      let x = plot_x0 + (v - x_min) / (x_max - x_min) * plot_w;
      svg.push_str(&format!(
        "<line x1=\"{x:.1}\" y1=\"{y1:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += x_major;
    }
  }

  // Y axis: major ticks extend leftward (to the left of the plot area).
  if let Some((y_min, y_max, y_major)) = y_axis
    && y_major > 0.0
    && y_max > y_min
  {
    let axis_x = plot_x0;
    let x1 = axis_x - minor_len;
    let x2 = x1 - extension;
    let eps = y_major * 1e-6;
    let mut v = (y_min / y_major).ceil() * y_major;
    let max_steps = ((y_max - y_min) / y_major).abs() as usize + 4;
    for _ in 0..max_steps {
      if v > y_max + eps {
        break;
      }
      let y = plot_y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h;
      svg.push_str(&format!(
        "<line x1=\"{x1:.1}\" y1=\"{y:.1}\" x2=\"{x2:.1}\" y2=\"{y:.1}\" stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += y_major;
    }
  }

  if !svg.is_empty() {
    buf.insert_str(insert_pos, &svg);
  }

  inject_missing_major_tick_labels(
    buf, plot_x0, plot_y0, plot_w, plot_h, x_axis, y_axis,
  );
}

/// Label the major ticks plotters left unlabelled.
///
/// Plotters picks its own key points from a range and a count hint, and on a
/// range with a negative minimum it drops the last one — `PlotRange -> {-1,
/// 1}` gets ticks up to `0.9` and no `1.0` label, where the Wolfram Language
/// labels the whole set. The tick *mark* is already drawn (the major
/// extension pass above walks every major), so only the text is missing; it
/// is emitted with the attributes of the labels plotters did draw, so the
/// injected one is indistinguishable from them.
fn inject_missing_major_tick_labels(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
) {
  let mut svg = String::new();
  for (axis, anchor, horizontal) in
    [(x_axis, "middle", true), (y_axis, "end", false)]
  {
    let Some((min, max, major)) = axis else {
      continue;
    };
    if major <= 0.0 || max <= min {
      continue;
    }
    let Some(template) = tick_label_template(buf, anchor) else {
      continue;
    };
    let eps = major * 1e-6;
    let mut v = (min / major).ceil() * major;
    let steps = ((max - min) / major).abs() as usize + 4;
    for _ in 0..steps {
      if v > max + eps {
        break;
      }
      let frac = (v - min) / (max - min);
      let (x, y) = if horizontal {
        (plot_x0 + frac * plot_w, template.pos)
      } else {
        (template.pos, plot_y0 + plot_h - frac * plot_h)
      };
      let along = if horizontal { x } else { y };
      if !template.occupied.iter().any(|p| (p - along).abs() < 2.0) {
        svg.push_str(&template.render(x, y, &format_tick_with_step(v, major)));
      }
      v += major;
    }
  }
  if !svg.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    buf.insert_str(pos, &svg);
  }
}

/// One of the tick labels plotters drew, as a stencil for the ones it did
/// not: the attributes to copy, the fixed coordinate they all share (the
/// label column or row) and the varying coordinates already taken.
struct TickLabelTemplate {
  attrs: String,
  pos: f64,
  occupied: Vec<f64>,
}

impl TickLabelTemplate {
  fn render(&self, x: f64, y: f64, text: &str) -> String {
    format!(
      "<text x=\"{x:.1}\" y=\"{y:.1}\" {}>{}</text>\n",
      self.attrs,
      crate::functions::graphics::svg_escape(text)
    )
  }
}

/// Collect the tick labels plotters drew for one axis, identified by their
/// `text-anchor` (`middle` for the x axis, `end` for the y axis).
fn tick_label_template(buf: &str, anchor: &str) -> Option<TickLabelTemplate> {
  let marker = format!("text-anchor=\"{anchor}\"");
  let mut attrs: Option<String> = None;
  let mut fixed: Option<f64> = None;
  let mut occupied = Vec::new();
  for tag in buf.split("<text ").skip(1) {
    let Some(end) = tag.find('>') else { continue };
    let (open, rest) = tag.split_at(end);
    if !open.contains(&marker) {
      continue;
    }
    let Some(x) = svg_attr_f64(open, "x") else {
      continue;
    };
    let Some(y) = svg_attr_f64(open, "y") else {
      continue;
    };
    // Only labels with text pin down where the row/column sits; the empty
    // ones plotters emits for minor ticks would match anywhere.
    let text = rest[1..].split('<').next().unwrap_or("").trim();
    if text.is_empty() {
      continue;
    }
    if attrs.is_none() {
      let after_y = open
        .find("\" y=\"")
        .and_then(|i| open[i + 5..].find('"').map(|j| i + 6 + j))
        .unwrap_or(0);
      attrs = Some(open[after_y..].trim().to_string());
      fixed = Some(if anchor == "middle" { y } else { x });
    }
    occupied.push(if anchor == "middle" { x } else { y });
  }
  Some(TickLabelTemplate {
    attrs: attrs?,
    pos: fixed?,
    occupied,
  })
}

/// Read a numeric attribute out of an SVG opening tag.
fn svg_attr_f64(tag: &str, name: &str) -> Option<f64> {
  let key = format!("{name}=\"");
  let start = tag.find(&key)? + key.len();
  let end = start + tag[start..].find('"')?;
  tag[start..end].parse().ok()
}

/// Draw top and right axis lines closing the plot rectangle, plus minor/major
/// tick marks mirroring the bottom and left axes. Ticks point outward (up from
/// the top axis, right from the right axis), matching the direction of the
/// bottom/left ticks drawn by plotters + `inject_major_tick_extensions`.
///
/// `x_axis` / `y_axis` each supply `(min, max, major_step)`. Minor ticks are
/// drawn at `major_step / 5` intervals. When an axis is `None` the
/// corresponding frame line is still drawn, but without tick marks.
fn inject_top_right_frame(
  buf: &mut String,
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  x_axis: Option<(f64, f64, f64)>,
  y_axis: Option<(f64, f64, f64)>,
  minor_len: f64,
  major_len: f64,
  stroke_w: f64,
  color: &str,
) {
  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut svg = String::new();

  let top_y = plot_y0;
  let right_x = plot_x0 + plot_w;
  let bottom_y = plot_y0 + plot_h;

  // Top frame line.
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.1}\" y1=\"{top_y:.1}\" x2=\"{right_x:.1}\" y2=\"{top_y:.1}\" \
       stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
  ));
  // Right frame line.
  svg.push_str(&format!(
    "<line x1=\"{right_x:.1}\" y1=\"{plot_y0:.1}\" x2=\"{right_x:.1}\" y2=\"{bottom_y:.1}\" \
       stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
  ));

  // Top axis ticks (pointing upward / outward).
  if let Some((x_min, x_max, x_major)) = x_axis
    && x_major > 0.0
    && x_max > x_min
  {
    let x_minor = minor_tick_step(x_major);
    let eps = x_minor.abs() * 1e-6;
    let mut v = (x_min / x_minor).ceil() * x_minor;
    let max_steps = ((x_max - x_min) / x_minor).abs() as usize + 8;
    for _ in 0..max_steps {
      if v > x_max + eps {
        break;
      }
      let x = plot_x0 + (v - x_min) / (x_max - x_min) * plot_w;
      let len = if is_major_tick(v, x_major) {
        major_len
      } else {
        minor_len
      };
      let y2 = top_y - len;
      svg.push_str(&format!(
        "<line x1=\"{x:.1}\" y1=\"{top_y:.1}\" x2=\"{x:.1}\" y2=\"{y2:.1}\" \
           stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += x_minor;
    }
  }

  // Right axis ticks (pointing rightward / outward).
  if let Some((y_min, y_max, y_major)) = y_axis
    && y_major > 0.0
    && y_max > y_min
  {
    let y_minor = minor_tick_step(y_major);
    let eps = y_minor.abs() * 1e-6;
    let mut v = (y_min / y_minor).ceil() * y_minor;
    let max_steps = ((y_max - y_min) / y_minor).abs() as usize + 8;
    for _ in 0..max_steps {
      if v > y_max + eps {
        break;
      }
      let y = plot_y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h;
      let len = if is_major_tick(v, y_major) {
        major_len
      } else {
        minor_len
      };
      let x2 = right_x + len;
      svg.push_str(&format!(
        "<line x1=\"{right_x:.1}\" y1=\"{y:.1}\" x2=\"{x2:.1}\" y2=\"{y:.1}\" \
           stroke=\"{color}\" stroke-width=\"{stroke_w:.0}\"/>\n"
      ));
      v += y_minor;
    }
  }

  if !svg.is_empty() {
    buf.insert_str(insert_pos, &svg);
  }
}

/// What [`frame_ticks_svg`] draws the tick marks and labels of a framed
/// plot from.
pub(crate) struct FrameTicksLayout<'a> {
  /// The plotting rectangle `(x0, y0, w, h)`, in render units.
  pub area: (f64, f64, f64, f64),
  /// The ranges the area maps, in axis space (natural logs on a log axis).
  pub ranges: (f64, f64, f64, f64),
  /// Whether the x and the y axis are logarithmic.
  pub log: (bool, bool),
  /// The automatic ticks of the x and the y axis, as `(min, max, major
  /// step)`; `None` where the axis places its own (log and date axes).
  pub auto: [Option<(f64, f64, f64)>; 2],
  /// The automatic labels of the x and the y axis, in axis space, for a far
  /// edge set to `All`.
  pub auto_labels: [&'a [(f64, String)]; 2],
  /// What each edge (bottom, left, top, right) carries.
  pub edges: &'a [FrameEdgeTicks; 4],
  /// The edges drawn at all, and the ones whose marks are drawn.
  pub present: FrameEdges,
  pub marked: FrameEdges,
  pub paints: &'a [FrameEdgePaint; 4],
  pub sf: f64,
  /// Tick label font size, in points, and whether it is bold.
  pub font: f64,
  pub bold: bool,
}

/// The automatic major tick length of a frame `w` × `h` points large, in
/// points: Wolfram scales it with the frame's size, between 2 and 4 pt. The
/// minor ticks are 0.6 of it.
pub(crate) fn frame_major_tick_len(w: f64, h: f64) -> f64 {
  (0.015 * (w * h).max(0.0).sqrt()).clamp(2.0, 4.0)
}

/// The tick marks of every edge of a framed plot — pointing into the plot,
/// an explicit `{plen, mlen}` one also out of it — and the tick labels the
/// plotters mesh does not draw: those of the top and right edges and of
/// explicit tick lists. A label sits 3 pt beyond its edge's outermost mark.
pub(crate) fn frame_ticks_svg(l: &FrameTicksLayout) -> String {
  use plot_padding::{TICK_LABEL_GAP, ascent, line_height, line_pitch};
  let (x0, y0, w, h) = l.area;
  let (x_min, x_max, y_min, y_max) = l.ranges;
  let sf = l.sf;
  let major_len = frame_major_tick_len(w / sf, h / sf) * sf;
  let minor_len = 0.6 * major_len;
  let rgb = |c: RGBColor| format!("rgb({},{},{})", c.0, c.1, c.2);
  let font = l.font * sf;
  let weight = if l.bold { " font-weight=\"bold\"" } else { "" };
  // The positions along one axis, as (fraction of the span, is major).
  let auto_ticks = |(min, max, major): (f64, f64, f64)| -> Vec<(f64, bool)> {
    if !(major > 0.0 && max > min) {
      return Vec::new();
    }
    let divisions = minor_tick_divisions(major) as i64;
    let minor = major / divisions as f64;
    let eps = 1e-6;
    let first = (min / minor - eps).ceil() as i64;
    let last = (max / minor + eps).floor() as i64;
    (first..=last)
      .map(|k| {
        (
          (k as f64 * minor - min) / (max - min),
          k.rem_euclid(divisions) == 0,
        )
      })
      .collect()
  };
  let present = [
    l.present.bottom,
    l.present.left,
    l.present.top,
    l.present.right,
  ];
  let marked = [l.marked.bottom, l.marked.left, l.marked.top, l.marked.right];
  let mut svg = String::new();
  for edge in 0..4 {
    if !present[edge] {
      continue;
    }
    let horizontal = edge % 2 == 0;
    let (lo, hi, log) = if horizontal {
      (x_min, x_max, l.log.0)
    } else {
      (y_min, y_max, l.log.1)
    };
    // Where along the edge an axis-space value falls, as a fraction.
    let frac_of = |v: f64| (v - lo) / (hi - lo);
    let in_range =
      |f: f64| hi > lo && f.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&f);
    let paint = &l.paints[edge];
    // (fraction, inward length, outward length, colour, width, label)
    let mut ticks: Vec<(f64, f64, f64, RGBColor, f64, Option<&str>)> =
      Vec::new();
    match &l.edges[edge] {
      FrameEdgeTicks::None => continue,
      spec @ (FrameEdgeTicks::Marks | FrameEdgeTicks::Labeled) => {
        let axis = if horizontal { l.auto[0] } else { l.auto[1] };
        for (frac, major) in axis.map(auto_ticks).unwrap_or_default() {
          let len = if major { major_len } else { minor_len };
          ticks.push((frac, len, 0.0, paint.mark, paint.mark_width, None));
        }
        // The bottom and left labels come from the chart mesh.
        if matches!(spec, FrameEdgeTicks::Labeled) && edge >= 2 {
          let labels = if horizontal {
            l.auto_labels[0]
          } else {
            l.auto_labels[1]
          };
          for (v, label) in labels {
            let frac = frac_of(*v);
            if in_range(frac) {
              ticks.push((frac, 0.0, 0.0, paint.mark, 0.0, Some(label)));
            }
          }
        }
      }
      FrameEdgeTicks::Explicit(list) => {
        for t in list {
          if log && t.pos <= 0.0 {
            continue;
          }
          let frac = frac_of(if log { t.pos.ln() } else { t.pos });
          if !in_range(frac) {
            continue;
          }
          let (inward, outward) =
            t.len.map_or((major_len, 0.0), |(i, o)| (i * w, o * w));
          let style = t.style.as_ref();
          let color = style.and_then(|s| s.color).map_or(paint.mark, |c| {
            let (r, g, b) = color_to_rgb8(c);
            RGBColor(r, g, b)
          });
          let width = style
            .and_then(|s| s.thickness)
            .map_or(paint.mark_width, |t| t * RESOLUTION_SCALE as f64);
          let label = (!t.label.is_empty()).then_some(t.label.as_str());
          ticks.push((frac, inward, outward, color, width, label));
        }
      }
    }
    // The labels clear the marks reaching furthest out.
    let out = ticks.iter().map(|t| t.2).fold(0.0, f64::max);
    let gap = TICK_LABEL_GAP * sf + out;
    let fill = rgb(paint.label);
    for &(frac, inward, outward, color, width, label) in &ticks {
      // The edge's point, and the unit vector pointing into the plot.
      let (px, py, nx, ny) = match edge {
        0 => (x0 + frac * w, y0 + h, 0.0, -1.0),
        1 => (x0, y0 + h - frac * h, 1.0, 0.0),
        2 => (x0 + frac * w, y0, 0.0, 1.0),
        _ => (x0 + w, y0 + h - frac * h, -1.0, 0.0),
      };
      if marked[edge] && inward + outward > 0.0 {
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"{width:.0}\"/>\n",
          px - nx * outward,
          py - ny * outward,
          px + nx * inward,
          py + ny * inward,
          rgb(color),
        ));
      }
      let Some(label) = label else { continue };
      let lines = label.split('\n').count();
      let pitch = line_pitch(l.font) * sf;
      let text = match edge {
        0 => format!(
          "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{font:.0}\"{weight} fill=\"{fill}\">{}</text>\n",
          py + gap + ascent(l.font) * sf,
          stacked_lines_svg(label, px, pitch),
        ),
        2 => format!(
          "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{font:.0}\"{weight} fill=\"{fill}\">{}</text>\n",
          py - gap
            - (line_height(l.font) - ascent(l.font)) * sf
            - (lines - 1) as f64 * pitch,
          stacked_lines_svg(label, px, pitch),
        ),
        1 => format!(
          "<text x=\"{:.1}\" y=\"{py:.1}\" text-anchor=\"end\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{font:.0}\"{weight} fill=\"{fill}\">{label}</text>\n",
          px - gap,
        ),
        _ => format!(
          "<text x=\"{:.1}\" y=\"{py:.1}\" text-anchor=\"start\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{font:.0}\"{weight} fill=\"{fill}\">{label}</text>\n",
          px + gap,
        ),
      };
      svg.push_str(&text);
    }
  }
  svg
}

/// Default minor (unlabeled) tick length in render-space units.
pub(crate) const MINOR_TICK_LEN: i32 = 4;
/// Default major (labeled) tick length in render-space units — slightly longer
/// than minor ticks so labeled ticks stand out visually.
pub(crate) const MAJOR_TICK_LEN: i32 = 7;

/// Filling mode for line plots.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Filling {
  None,
  Axis,
  Bottom,
  Top,
  Value(f64),
}

pub(crate) use crate::functions::plot_mesh::Mesh;

impl Filling {
  /// Compute the y-value to fill to, given the current plot y-range.
  /// Returns `None` for `Filling::None`.
  pub(crate) fn reference_y(&self, y_min: f64, y_max: f64) -> Option<f64> {
    match self {
      Self::None => None,
      Self::Axis => Some(0.0),
      Self::Bottom => Some(y_min),
      Self::Top => Some(y_max),
      Self::Value(v) => Some(*v),
    }
  }

  /// Convert to the serializable `SeriesFilling` variant stored on
  /// `PlotSeriesData` so `Show` can re-render filled regions.
  pub(crate) fn to_series_filling(self) -> crate::syntax::SeriesFilling {
    match self {
      Self::None => crate::syntax::SeriesFilling::None,
      Self::Axis => crate::syntax::SeriesFilling::Axis,
      Self::Bottom => crate::syntax::SeriesFilling::Bottom,
      Self::Top => crate::syntax::SeriesFilling::Top,
      Self::Value(v) => crate::syntax::SeriesFilling::Value(v),
    }
  }
}

/// Parsed `FillingStyle -> …` value: an optional fill color override
/// (`None` = use the series color) and an optional opacity (`None` = the
/// default: 0.2 translucency, or fully opaque when a color is given).
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct FillStyle {
  pub color: Option<(u8, u8, u8)>,
  pub opacity: Option<f64>,
  /// Second color of a two-color style `{above, below}`: used where the
  /// filled curve lies below the curve it is filled to.
  pub below_color: Option<(u8, u8, u8)>,
  /// `Dashed`/`Dashing[…]`: the dash and gap length, in points, of a
  /// list plot's filling stems.
  pub dashing: Option<(f64, f64)>,
}

impl FillStyle {
  /// The opacity the fill is drawn at: the explicit one, else fully opaque
  /// when a color was given, else `None` (the default 0.2 translucency).
  pub(crate) fn effective_opacity(&self) -> Option<f64> {
    self.opacity.or(self.color.map(|_| 1.0))
  }
}

/// Parse a `FillingStyle` option value. Supported forms: a color,
/// `Opacity[a]`, `Opacity[a, color]`, `Directive[…]`, and a directive list
/// `{Opacity[a], color}`. Returns `None` for `Automatic`/`None`/unrecognized
/// values (keeping the default fill appearance).
/// The first dash and gap, in points, of a dashing directive.
fn dash_pair(e: &Expr) -> Option<(f64, f64)> {
  let dashes = parse_style_directives(e)?.dashing?;
  let dash = *dashes.first()?;
  Some((dash, dashes.get(1).copied().unwrap_or(dash)))
}

pub(crate) fn parse_filling_style(replacement: &Expr) -> Option<FillStyle> {
  fn walk(e: &Expr, fs: &mut FillStyle) {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Opacity" && !args.is_empty() =>
      {
        if let Some(a) = try_eval_to_f64(&args[0]) {
          fs.opacity = Some(a.clamp(0.0, 1.0));
        }
        if args.len() >= 2
          && let Some(c) = parse_color(&args[1])
        {
          fs.color = Some(color_to_rgb8(c));
        }
      }
      Expr::FunctionCall { name, args } if name == "Directive" => {
        for a in args {
          walk(a, fs);
        }
      }
      Expr::List(items) => {
        // `{below, above}`: two plain colors fill the regions where the
        // curve is below / above its target in different colors
        // (`FillingStyle -> {Red, Blue}` paints the negative lobes red).
        if let [a, b] = items.as_slice()
          && !matches!(a, Expr::FunctionCall { name, .. } if name == "Opacity")
          && !matches!(b, Expr::FunctionCall { name, .. } if name == "Opacity")
          && let (Some(ca), Some(cb)) = (parse_color(a), parse_color(b))
        {
          fs.below_color = Some(color_to_rgb8(ca));
          fs.color = Some(color_to_rgb8(cb));
          return;
        }
        for item in items {
          walk(item, fs);
        }
      }
      Expr::Identifier(s) if s == "Automatic" || s == "None" => {}
      Expr::Identifier(s)
        if matches!(s.as_str(), "Dashed" | "Dotted" | "DotDashed") =>
      {
        fs.dashing = dash_pair(e);
      }
      Expr::FunctionCall { name, .. }
        if name == "Dashing" || name == "AbsoluteDashing" =>
      {
        fs.dashing = dash_pair(e);
      }
      other => {
        if let Some(c) = parse_color(other) {
          fs.color = Some(color_to_rgb8(c));
          // An alpha channel on the color (`RGBColor[r, g, b, a]`) doubles
          // as the fill opacity when no explicit Opacity[…] is given.
          if c.a < 1.0 && fs.opacity.is_none() {
            fs.opacity = Some(c.a);
          }
        }
      }
    }
  }
  let val = evaluate_expr_to_expr(replacement).unwrap_or(replacement.clone());
  let mut fs = FillStyle {
    color: None,
    opacity: None,
    below_color: None,
    dashing: None,
  };
  walk(&val, &mut fs);
  if fs.color.is_none() && fs.opacity.is_none() && fs.dashing.is_none() {
    None
  } else {
    Some(fs)
  }
}

/// Convert a parsed `Color` (float channels) to 8-bit RGB.
pub(crate) fn color_to_rgb8(c: WoxiColor) -> (u8, u8, u8) {
  (
    (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
    (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
    (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
  )
}

/// Resolve the fill paint for a series: the `FillingStyle` color override
/// (else the series color) at the `FillingStyle` opacity. Without one the fill
/// is translucent (0.2) by default, but opaque once a color is given.
fn fill_paint(
  filling_style: Option<FillStyle>,
  series_rgb: (u8, u8, u8),
) -> plotters::style::RGBAColor {
  let fs = filling_style.unwrap_or(FillStyle {
    color: None,
    opacity: None,
    below_color: None,
    dashing: None,
  });
  let (r, g, b) = fs.color.unwrap_or(series_rgb);
  RGBColor(r, g, b).mix(fs.effective_opacity().unwrap_or(0.2))
}

impl crate::syntax::SeriesFilling {
  /// Reference y-value for the fill, given the current y-range.
  pub fn reference_y(self, y_min: f64, y_max: f64) -> Option<f64> {
    match self {
      Self::None => None,
      Self::Axis => Some(0.0),
      Self::Bottom => Some(y_min),
      Self::Top => Some(y_max),
      Self::Value(v) => Some(v),
      // Bounded by another curve, not a constant level.
      Self::Series(_) => None,
    }
  }
}

/// Fill target for one series, from the rule-list form
/// `Filling -> {i -> spec, …}`.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FillTarget {
  /// Fill to a constant reference level (`Axis`, `Bottom`, `Top`, a value).
  Level(Filling),
  /// Fill to another series (0-based index), i.e. `i -> {j}`.
  Series(usize),
}

/// Parse the rule-list form `Filling -> {i -> spec, …}`, where `spec` is
/// `{j}` (fill between series i and j), `{{j}, style}` (a series target with
/// a style), a constant level (`Axis`, `Bottom`, `Top`, a number), or
/// `{level, style}` (a level with a style, e.g. `{0, Opacity[.4, Red]}` to
/// fill down to the x-axis in translucent red). A series target is only a
/// bare integer when it is the sole element of its list (`{j}`) or nested
/// one level deeper alongside a style (`{{j}, style}`); a two-or-more-element
/// list whose first element is a plain number is a level, not a series
/// index, so `{0, style}` doesn't get mistaken for "fill to series 0"
/// (which isn't even a valid 1-based series index) and silently drop the
/// whole option. Each rule's style, when given, is returned alongside its
/// target — the caller stores it by series index for `series_filling_style`
/// to pick up.
/// Returns `None` when `replacement` is not a list of rules keyed by
/// 1-based series indices, so the caller can fall back to `parse_filling`.
///
/// The rules may be grouped in sub-lists — `{{1 -> {2}}, {3 -> 0}}` fills the
/// same way `{1 -> {2}, 3 -> 0}` does — so the outer structure is flattened
/// first. Only the list scaffolding around the rules is flattened; a rule's
/// own right-hand side (`{j}`, the brace form that names a target series)
/// stays as written.
fn parse_filling_rules(
  replacement: &Expr,
) -> Option<Vec<(usize, FillTarget, Option<FillStyle>)>> {
  fn flatten_rules<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
      Expr::List(items) => items.iter().for_each(|i| flatten_rules(i, out)),
      other => out.push(other),
    }
  }
  if !matches!(replacement, Expr::List(_)) {
    return None;
  }
  let mut items: Vec<&Expr> = Vec::new();
  flatten_rules(replacement, &mut items);
  let mut rules = Vec::new();
  for item in items {
    // A bare series number `i` directly in the list (`Filling -> {1, 3}`)
    // is shorthand for `i -> Axis`.
    if let Expr::Integer(n) = item
      && *n >= 1
      && matches!(replacement, Expr::List(top) if top.iter().any(|t| std::ptr::eq(t, item)))
    {
      rules.push((*n as usize - 1, FillTarget::Level(Filling::Axis), None));
      continue;
    }
    let (pattern, rhs) = match item {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => return None,
    };
    let series_idx = match pattern {
      Expr::Integer(n) if *n >= 1 => *n as usize - 1,
      _ => return None,
    };
    let (target, style) = match rhs {
      // A lone `{j}` is a bare series target, with no style.
      Expr::List(spec) if spec.len() == 1 => (
        match &spec[0] {
          Expr::Integer(j) if *j >= 1 => FillTarget::Series(*j as usize - 1),
          other => FillTarget::Level(parse_filling(other)),
        },
        None,
      ),
      // `{target, style, …}`: a series target is nested one level deeper
      // (`{{j}, style}`) so it isn't confused with a level that happens to
      // be a positive integer (`{2, style}` fills to level 2, not series 2).
      Expr::List(spec) => {
        let target = match spec.first()? {
          Expr::List(inner) => {
            let target_idx = match inner.first()? {
              Expr::Integer(j) if *j >= 1 => *j as usize - 1,
              _ => return None,
            };
            FillTarget::Series(target_idx)
          }
          level => FillTarget::Level(parse_filling(level)),
        };
        let style = spec.get(1).and_then(parse_filling_style);
        (target, style)
      }
      other => (FillTarget::Level(parse_filling(other)), None),
    };
    rules.push((series_idx, target, style));
  }
  Some(rules)
}

/// Apply a `Filling` option value to `opts`: either the per-series rule
/// list `{i -> spec, …}` (stored in `filling_rules`, with any per-rule
/// style stored in the corresponding slot of `filling_styles`) or a global
/// mode (stored in `filling`).
pub(crate) fn apply_filling_option(replacement: &Expr, opts: &mut PlotOptions) {
  // The value may be computed rather than named — a Demonstration switches
  // its shading with `Filling -> If[b === Axis, Axis, None]` — so read it
  // evaluated. Every spelling it can produce is a symbol, a number or a
  // list of rules, all of which evaluate to themselves.
  let value =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  if let Some(rules) = parse_filling_rules(&value) {
    if let Some(max_idx) = rules.iter().map(|(i, ..)| *i).max() {
      let mut styles = vec![None; max_idx + 1];
      for (i, _, style) in &rules {
        if style.is_some() {
          styles[*i] = *style;
        }
      }
      opts.filling_styles = styles;
    }
    opts.filling_rules = rules
      .into_iter()
      .map(|(i, target, _)| (i, target))
      .collect();
  } else {
    opts.filling = parse_filling(&value);
  }
}

/// The `FillingStyle` that applies to series `idx`: its own when `Show`
/// merged several plots, else the one style the plot was given.
fn series_filling_style(opts: &PlotOptions, idx: usize) -> Option<FillStyle> {
  opts
    .filling_styles
    .get(idx)
    .copied()
    .flatten()
    .or(opts.filling_style)
}

/// Effective fill target for series `idx`: its entry in the rule list when
/// `Filling -> {i -> spec, …}` was given (series without a rule stay
/// unfilled), otherwise the global filling mode.
fn series_fill_target(opts: &PlotOptions, idx: usize) -> FillTarget {
  if opts.filling_rules.is_empty() {
    FillTarget::Level(opts.filling)
  } else {
    opts
      .filling_rules
      .iter()
      .find(|(i, _)| *i == idx)
      .map_or(FillTarget::Level(Filling::None), |&(_, t)| t)
  }
}

/// Linearly interpolate a polyline (in data order) at `x`. Returns `None`
/// when `x` lies outside every segment's x-span, so filling between series
/// with different x-domains stops at the overlap.
fn interp_polyline_y(points: &[(f64, f64)], x: f64) -> Option<f64> {
  for w in points.windows(2) {
    let (x0, y0) = w[0];
    let (x1, y1) = w[1];
    if !(x0.is_finite() && x1.is_finite() && y0.is_finite() && y1.is_finite()) {
      continue;
    }
    let (lo, hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    if x >= lo && x <= hi {
      if (x1 - x0).abs() < f64::EPSILON {
        return Some(y0);
      }
      return Some(y0 + (y1 - y0) * (x - x0) / (x1 - x0));
    }
  }
  None
}

/// Polygon between two polylines over the overlap of their x-domains: the
/// source curve forms the top boundary and the reversed target curve the
/// bottom, both clipped (with interpolated endpoints) to the overlap.
pub(crate) fn fill_between_polygon(
  source: &[(f64, f64)],
  target: &[(f64, f64)],
) -> Option<Vec<(f64, f64)>> {
  let finite = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
    pts
      .iter()
      .copied()
      .filter(|(x, y)| x.is_finite() && y.is_finite())
      .collect()
  };
  let src = finite(source);
  let tgt = finite(target);
  if src.len() < 2 || tgt.len() < 2 {
    return None;
  }
  let x_span = |pts: &[(f64, f64)]| {
    pts
      .iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(x, _)| {
        (lo.min(x), hi.max(x))
      })
  };
  let (s_lo, s_hi) = x_span(&src);
  let (t_lo, t_hi) = x_span(&tgt);
  let lo = s_lo.max(t_lo);
  let hi = s_hi.min(t_hi);
  if lo >= hi {
    return None;
  }
  let clip = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(y) = interp_polyline_y(pts, lo) {
      out.push((lo, y));
    }
    out.extend(pts.iter().copied().filter(|&(x, _)| x > lo && x < hi));
    if let Some(y) = interp_polyline_y(pts, hi) {
      out.push((hi, y));
    }
    out
  };
  let mut polygon = clip(&src);
  let mut bottom = clip(&tgt);
  bottom.reverse();
  polygon.extend(bottom);
  if polygon.len() >= 3 {
    Some(polygon)
  } else {
    None
  }
}

/// The region between two polylines split where they cross: each piece is
/// tagged `true` when the source curve lies above the target there. Pieces
/// share the interpolated crossing points, so they tile the full
/// `fill_between_polygon` region.
fn fill_between_polygons_by_side(
  source: &[(f64, f64)],
  target: &[(f64, f64)],
) -> Vec<(bool, Vec<(f64, f64)>)> {
  let finite = |pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
    pts
      .iter()
      .copied()
      .filter(|(x, y)| x.is_finite() && y.is_finite())
      .collect()
  };
  let (src, tgt) = (finite(source), finite(target));
  if src.len() < 2 || tgt.len() < 2 {
    return Vec::new();
  }
  let span = |pts: &[(f64, f64)]| {
    pts
      .iter()
      .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(x, _)| {
        (lo.min(x), hi.max(x))
      })
  };
  let (s, t) = (span(&src), span(&tgt));
  let (lo, hi) = (s.0.max(t.0), s.1.min(t.1));
  if lo >= hi {
    return Vec::new();
  }
  let mut xs: Vec<f64> = vec![lo, hi];
  xs.extend(
    src
      .iter()
      .chain(tgt.iter())
      .map(|p| p.0)
      .filter(|&x| x > lo && x < hi),
  );
  xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
  xs.dedup();
  // (x, source y, target y) on the merged grid.
  let rows: Vec<(f64, f64, f64)> = xs
    .iter()
    .filter_map(|&x| {
      Some((x, interp_polyline_y(&src, x)?, interp_polyline_y(&tgt, x)?))
    })
    .collect();
  let mut out: Vec<(bool, Vec<(f64, f64)>)> = Vec::new();
  let mut top: Vec<(f64, f64)> = Vec::new();
  let mut bottom: Vec<(f64, f64)> = Vec::new();
  let mut above = true;
  let flush = |above: bool,
               top: &mut Vec<(f64, f64)>,
               bottom: &mut Vec<(f64, f64)>,
               out: &mut Vec<(bool, Vec<(f64, f64)>)>| {
    let mut poly = std::mem::take(top);
    poly.extend(std::mem::take(bottom).into_iter().rev());
    if poly.len() >= 3 {
      out.push((above, poly));
    }
  };
  for (i, &(x, sy, ty)) in rows.iter().enumerate() {
    if i > 0 {
      let (px, psy, pty) = rows[i - 1];
      let (d0, d1) = (psy - pty, sy - ty);
      if d0 * d1 < 0.0 {
        let f = d0 / (d0 - d1);
        let cx = px + (x - px) * f;
        let cy = psy + (sy - psy) * f;
        top.push((cx, cy));
        bottom.push((cx, cy));
        flush(above, &mut top, &mut bottom, &mut out);
        top.push((cx, cy));
        bottom.push((cx, cy));
      }
    }
    let d = sy - ty;
    top.push((x, sy));
    bottom.push((x, ty));
    if d == 0.0 {
      // The curves meet exactly on a sample: close the piece here so the
      // regions on either side keep their own color.
      if i > 0 && i + 1 < rows.len() {
        flush(above, &mut top, &mut bottom, &mut out);
        top.push((x, sy));
        bottom.push((x, ty));
      }
    } else {
      above = d > 0.0;
    }
  }
  flush(above, &mut top, &mut bottom, &mut out);
  out
}

/// Parse a `Filling` option value from an expression.
fn parse_filling(replacement: &Expr) -> Filling {
  match replacement {
    Expr::Identifier(v) if v == "Axis" => Filling::Axis,
    Expr::Identifier(v) if v == "Automatic" => Filling::Axis,
    // `i -> True` is how a Demonstration switches a curve's shading on.
    Expr::Identifier(v) if v == "True" => Filling::Axis,
    Expr::Identifier(v) if v == "Bottom" => Filling::Bottom,
    Expr::Identifier(v) if v == "Top" => Filling::Top,
    Expr::Identifier(v) if v == "None" => Filling::None,
    other => {
      let evaled =
        evaluate_expr_to_expr(other).unwrap_or_else(|_| other.clone());
      if let Some(v) = try_eval_to_f64(&evaled) {
        Filling::Value(v)
      } else {
        Filling::None
      }
    }
  }
}

/// Adjust y-range so the fill reference level is included.
/// For `Axis`, ensures y=0 is in range. For `Value(v)`, ensures v is in range.
/// `Bottom`/`Top`/`None` don't need adjustment (they use the range edges).
fn adjust_y_range_for_filling(
  filling: Filling,
  y_range: (f64, f64),
) -> (f64, f64) {
  let (mut y_lo, mut y_hi) = y_range;
  match filling {
    Filling::Axis => {
      if y_lo > 0.0 {
        y_lo = 0.0 - (y_hi - 0.0) * 0.04;
      }
      if y_hi < 0.0 {
        y_hi = 0.0 + (0.0 - y_lo) * 0.04;
      }
    }
    Filling::Value(v) => {
      if y_lo > v {
        y_lo = v - (y_hi - v) * 0.04;
      }
      if y_hi < v {
        y_hi = v + (v - y_lo) * 0.04;
      }
    }
    _ => {}
  }
  (y_lo, y_hi)
}

/// Like `adjust_y_range_for_filling`, but also accounts for the per-series
/// `Filling -> {i -> spec, …}` rules: every constant-level target must lie
/// inside the plotted y-range (between-series targets need no adjustment —
/// both bounds are data curves already contained in the range).
pub(crate) fn adjust_y_range_for_filling_opts(
  opts: &PlotOptions,
  mut y_range: (f64, f64),
) -> (f64, f64) {
  y_range = adjust_y_range_for_filling(opts.filling, y_range);
  for (_, target) in &opts.filling_rules {
    if let FillTarget::Level(level) = target {
      y_range = adjust_y_range_for_filling(*level, y_range);
    }
  }
  y_range
}

/// `y_range` stretched to the constant levels the curves are filled to, so
/// that a fill does not stop short of them.
pub(crate) fn include_filling_levels(
  opts: &PlotOptions,
  mut y_range: (f64, f64),
) -> (f64, f64) {
  let levels =
    std::iter::once(opts.filling).chain(opts.filling_rules.iter().filter_map(
      |(_, target)| match target {
        FillTarget::Level(level) => Some(*level),
        FillTarget::Series(_) => None,
      },
    ));
  for level in levels {
    let v = match level {
      Filling::Axis => 0.0,
      Filling::Value(v) => v,
      _ => continue,
    };
    y_range = (y_range.0.min(v), y_range.1.max(v));
  }
  y_range
}

/// Per-series style: color, line thickness, and dashing pattern.
#[derive(Clone, Debug, Default)]
pub(crate) struct SeriesStyle {
  pub color: Option<WoxiColor>,
  /// `color` as written (evaluated), kept for the plot expression; see
  /// `PlotSeriesData::color_expr`.
  pub color_expr: Option<Expr>,
  /// Line thickness in display pixels (e.g. 1.5 = default, 2.0 = Thick).
  /// None means use the default (1.5px).
  pub thickness: Option<f64>,
  /// Scatter dot size: a fraction of the image width for `PointSize[…]`,
  /// stored negative for the absolute printer's points of
  /// `AbsolutePointSize[…]`. None means the default dot.
  pub point_size: Option<f64>,
  /// Dash pattern in display pixels. None = solid line.
  pub dashing: Option<Vec<f64>>,
  /// DropShadowing[...] directive: draw the curve with a drop shadow.
  pub shadow: Option<crate::functions::graphics::DropShadow>,
  /// `Opacity[a]` directive: the curve's alpha. None = fully opaque.
  pub opacity: Option<f64>,
  /// This style came from its own `{…}` entry of a `PlotStyle` list, i.e. it
  /// is one dataset's style rather than a bare directive of a combined one.
  pub from_list: bool,
}

/// A single explicit grid line: a position on its axis plus an optional
/// per-line style override (color, thickness, dashing).
#[derive(Clone, Debug)]
pub(crate) struct GridLine {
  pub pos: f64,
  pub style: SeriesStyle,
}

/// One side (x or y) of a `GridLines` specification.
pub(crate) enum GridSide {
  /// No grid lines on this axis.
  None,
  /// Evenly spaced automatic grid lines (default gray, solid).
  Automatic,
  /// Explicit grid lines at the given positions/styles.
  Explicit(Vec<GridLine>),
}

/// Position for plot legends
/// The marker shape a legend draws beside each label: whatever the series
/// itself looks like (`Automatic` — the `PlotLegends -> {labels}` form), or
/// the one an explicit `LineLegend` / `SwatchLegend` / `PointLegend` asks
/// for.
#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum LegendMarkerKind {
  #[default]
  Automatic,
  Line,
  Swatch,
  Point,
}

/// Position for plot legends
#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum LegendPosition {
  #[default]
  Right,
  Top,
  Bottom,
}

/// How `Around` uncertainty intervals are drawn (IntervalMarkers option):
/// capped error bars (the default), a filled band spanning the intervals
/// across the series, or nothing.
#[derive(Clone, Copy, PartialEq, Default)]
pub(crate) enum IntervalMarkers {
  /// Error bars with end caps ("Fences", the default).
  #[default]
  Fences,
  /// `"Bars"`: the bars without their caps.
  Bars,
  /// One filled band per series covering the y-uncertainty intervals.
  Bands,
  /// `IntervalMarkers -> None`: uncertainties are not drawn.
  None,
}

/// Options for line-based plots (Plot, ListLinePlot, etc.).
// Each flag is an independent Wolfram option (Frame, Joined, Filling, …),
// not a state machine, so they do not collapse into an enum.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone)]
pub(crate) struct PlotOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub filling: Filling,
  /// Per-series filling from `Filling -> {i -> spec, …}` (0-based series
  /// index → target). When non-empty this replaces the global `filling`:
  /// series without a rule are not filled.
  pub filling_rules: Vec<(usize, FillTarget)>,
  /// `FillingStyle -> …`: color/opacity for the filled regions (applies to
  /// every filled series). `None` keeps the default appearance (series
  /// color at 0.2 opacity).
  pub filling_style: Option<FillStyle>,
  /// Per-series `FillingStyle`, as `Show` collects it from the plots it
  /// merges: each curve keeps the fill its own plot asked for. Empty means
  /// the single `filling_style` applies to every series.
  pub filling_styles: Vec<Option<FillStyle>>,
  pub mesh: Mesh,
  pub plot_label: Option<StyledLabel>,
  pub axes_label: Option<(String, String)>,
  pub plot_style: Vec<SeriesStyle>,
  /// `ColorFunction -> f`: colors the curve along its length by `f[x]` (or
  /// `f[x, y]` for a two-parameter function) instead of by `PlotStyle`.
  pub color_function: Option<Expr>,
  /// `ColorFunctionScaling`, per argument (x, y): whether `f` receives the
  /// coordinate rescaled to 0..1 over the plot range (the default) or the
  /// raw value. `{True, False}` scales only x.
  pub color_function_scaling: (bool, bool),
  /// Per-axis visibility: (x_axis, y_axis). Both true = default.
  pub axes: (bool, bool),
  /// `AxesOrigin -> {x, y}`: where the axes cross. `None` coordinates
  /// follow the automatic rule (0 when in range, else the nearest end).
  pub axes_origin: (Option<f64>, Option<f64>),
  /// Per axis, the low end of the plot range before the automatic padding
  /// was added, when the renderer is handed a padded one — the automatic
  /// `AxesOrigin` of an all-positive range sits there rather than at the
  /// padded edge.
  pub axes_origin_low: (Option<f64>, Option<f64>),
  /// Ticks option: true = show tick marks and labels (default), false = hide
  pub ticks: bool,
  /// `Ticks -> {xspec, yspec}` with explicit positions: each entry is a
  /// position and the text drawn at it. `None` = the automatic ticks.
  pub ticks_x: Option<Vec<(f64, String)>>,
  pub ticks_y: Option<Vec<(f64, String)>>,
  /// `FrameTicks`: what each frame edge (bottom, left, top, right) carries.
  /// The labels of an explicit bottom / left list are also in `ticks_x` /
  /// `ticks_y`.
  pub frame_tick_edges: [FrameEdgeTicks; 4],
  /// The plot theme labels the axes without explicit ticks with
  /// `Charting`SimpleTicks` (majors only, about six) over the displayed
  /// range instead of the automatic ticks.
  pub simple_ticks: bool,
  /// Number of sample points for Plot[] (default: NUM_SAMPLES)
  pub plot_points: usize,
  /// `MaxRecursion -> n`: how many times the adaptive sampler may halve an
  /// interval of the initial `PlotPoints` grid (WL's default is 6).
  pub max_recursion: usize,
  /// Legend labels for each series (empty = no legend)
  pub plot_legends: Vec<String>,
  /// The series each `plot_legends` entry stands for, when that is not
  /// simply the entry's own index (`ListPlot[{a, Legended[b, "l"], c}]`
  /// has one entry, for the second series).
  pub legend_series: Vec<usize>,
  /// Position of the legend (Right, Top, Bottom)
  pub legend_position: LegendPosition,
  /// The marker shape of an explicit `LineLegend` / `SwatchLegend` /
  /// `PointLegend` legend.
  pub legend_marker_kind: LegendMarkerKind,
  /// The per-entry colors an explicit `XLegend[{styles}, {labels}]` gives,
  /// which replace the series colors (`None` = the series color).
  pub legend_colors: Vec<Option<(u8, u8, u8)>>,
  /// Show evenly spaced horizontal grid lines (automatic).
  pub grid_lines_y: bool,
  /// Show evenly spaced vertical grid lines (automatic).
  pub grid_lines_x: bool,
  /// Explicit vertical grid lines (positions + per-line style). When
  /// non-empty, these replace the automatic `grid_lines_x` lines.
  pub grid_x_lines: Vec<GridLine>,
  /// Explicit horizontal grid lines. Replace automatic `grid_lines_y` lines.
  pub grid_y_lines: Vec<GridLine>,
  /// `GridLinesStyle -> Directive[…]`: the default style for every grid line
  /// that carries no style of its own. `None` = the built-in gray.
  pub grid_lines_style: Option<SeriesStyle>,
  /// Use frame (left+bottom border) instead of axes
  pub frame: bool,
  /// The unpadded (x, y) plot range of a list plot: its automatic grid
  /// lines are drawn strictly inside it, not at its ends. `None` = draw them
  /// at every major tick of the displayed range.
  pub grid_interior: Option<((f64, f64), (f64, f64))>,
  /// Which edges of a frame are drawn — line and tick marks — as
  /// (left, right, bottom, top): `Frame -> {{True, False}, {True, False}}`
  /// draws only the left and bottom ones.
  pub frame_edges: FrameEdges,
  /// Frame edges whose line and tick marks a transparent `FrameStyle`
  /// leaves undrawn while their tick labels stay (the `"Business"` theme's
  /// left edge). They still count as part of the frame for its padding.
  pub frame_hidden_edges: FrameEdges,
  /// `FrameTicks -> None | False`: a framed plot keeps its border but loses
  /// the tick marks and labels on it.
  pub frame_ticks: bool,
  /// `Evaluated -> True`: work the body out once, with the plot variable
  /// symbolic, rather than re-evaluating it at every sample point.
  pub evaluated: bool,
  /// Labels on the bottom and left frame edges (`FrameLabel`). These sit
  /// centred outside the plot area, unlike an [`Self::axes_label`], which
  /// Wolfram writes at the far end of its axis.
  pub frame_label_bottom: Option<String>,
  pub frame_label_left: Option<String>,
  /// Label on the top frame edge (FrameLabel 4-element form: top of {bottom,top})
  pub frame_label_top: Option<String>,
  /// Label on the right frame edge (FrameLabel 4-element form: right of {left,right})
  pub frame_label_right: Option<String>,
  /// Format x-axis labels as dates (AbsoluteTime seconds since 1900-01-01)
  pub date_axis: bool,
  /// Whether x-axis is logarithmic (data is in log10 space)
  pub log_x: bool,
  /// Whether y-axis is logarithmic (data is in log10 space)
  pub log_y: bool,
  /// The ticks of an axis drawn in the coordinates of its
  /// `ScalingFunctions` (positions scaled, labels original), which replace
  /// its automatic ticks: `(x, y)`.
  pub scaled_ticks: (Option<AxisTicks>, Option<AxisTicks>),
  /// Callout labels for each series (None = no callout for that series)
  pub callout_labels: Vec<Option<String>>,
  /// `PlotLabels -> {l1, l2, …}`: a label drawn after the right-hand end of
  /// each curve (None = that curve is unlabeled).
  pub plot_labels: Vec<Option<String>>,
  /// Per series: its `plot_labels` entry hangs off a horizontal leader line
  /// (a `Callout[data, label]` at its automatic place, the series' end)
  /// rather than sitting right after the last point (`PlotLabels`,
  /// `Labeled`).
  pub plot_label_leaders: Vec<bool>,
  /// Per-point labels from `Labeled[y, label]` data entries, parallel to
  /// each series' points (None = no label for that point). Empty when no
  /// point is labeled.
  pub point_labels: Vec<Vec<Option<String>>>,
  /// Where `point_labels` sit relative to their point, from a position given
  /// as `LabelingFunction` (`Above`, `Below`, `Left`, `Right`, `Center`).
  /// `None` keeps the automatic above-right placement.
  pub point_label_position: Option<String>,
  /// The point labels are callouts (`Callout[pt, label]`, `data -> labels`):
  /// pulled away from their points along leader lines, and dropped where
  /// they find no free place.
  pub point_label_callouts: bool,
  /// Stacked mode (StackedListPlot): fill each series down to the previous
  /// series' curve instead of to a constant baseline, producing opaque
  /// stacked bands. The `all_points` passed in are the cumulative curves.
  pub stacked: bool,
  /// `Epilog -> {…}` graphics primitives (already evaluated), drawn on top
  /// of the plotted data in data coordinates.
  pub epilog: Vec<Expr>,
  /// `Prolog -> {…}` graphics primitives (already evaluated), drawn
  /// underneath the plotted data in data coordinates — the mirror of
  /// `epilog`, painted first instead of last.
  pub prolog: Vec<Expr>,
  /// `PlotRangeClipping -> False` lets Epilog/Prolog primitives spill past
  /// the plotting frame instead of being cut off at its edge.
  pub clip_overlays: bool,
  /// Per-series error bars from `Around` data values, parallel to the
  /// series' points: each entry is ((dx_minus, dx_plus), (dy_minus,
  /// dy_plus)) in data units. Empty when the data has no uncertainties.
  pub error_bars: Vec<Vec<((f64, f64), (f64, f64))>>,
  /// How the `error_bars` uncertainties are rendered.
  pub interval_markers: IntervalMarkers,
  /// `IntervalMarkersStyle -> colour`: the colour of the uncertainty bars
  /// (else the series colour).
  pub interval_markers_color: Option<RGBColor>,
  /// The original data points per series when the drawn curve differs from
  /// the data (e.g. `InterpolationOrder` step/spline curves). Anchors
  /// `Mesh -> Full` dots, error bars, and per-point labels. Empty when the
  /// drawn curve is the data itself.
  pub data_points: Vec<Vec<(f64, f64)>>,
  /// `Background -> color`: fill for the whole image, replacing the
  /// theme background. `None` keeps the theme default.
  pub background: Option<RGBColor>,
  /// `ImagePadding -> {{left, right}, {bottom, top}}`: the space, in display
  /// pixels, between the plotting area and the edges of the image. When set
  /// it replaces the automatic margins, so the frame fills whatever the
  /// padding leaves — which is how a plot given a short `ImageSize` keeps a
  /// usable frame instead of losing it to fixed label margins. `None` keeps
  /// the automatic margins.
  pub image_padding: Option<[f64; 4]>,
  /// `AspectRatio -> r`: ratio (height/width) of the plotting *area* (the
  /// data frame, excluding label/tick margins). When set, the total image
  /// height is derived so the frame has this ratio, matching Wolfram. `None`
  /// keeps `svg_height` as given.
  pub aspect_ratio: Option<f64>,
  /// `AspectRatio -> Automatic`: derive the ratio from the plotted ranges
  /// (`(ymax - ymin) / (xmax - xmin)`) once they are known, so one unit
  /// spans the same length on both axes.
  pub aspect_automatic: bool,
  /// `AspectRatio -> Full`: stretch the plotting area to fill whatever
  /// `ImageSize` gives instead of keeping the default 1/GoldenRatio.
  pub aspect_full: bool,
  /// Whether `ImageSize` fixed the image width / height explicitly
  /// (`{w, h}`, `{Automatic, h}`, a bare width, …). A fixed height makes the
  /// plot keep its `AspectRatio` inside the image instead of resizing it.
  pub image_size_explicit: (bool, bool),
  /// `PlotLayout -> "Row" | "Column" | …`: one panel per curve
  /// (see [`crate::functions::plot_layout`]); `None` overlays the curves.
  pub plot_layout: Option<crate::functions::plot_layout::PlotLayoutSpec>,
  /// The raw `ImageSize` value, which sizes a `PlotLayout` grid.
  pub image_size_value: Option<Expr>,
  /// The plot range before its automatic padding (`Plot`); the axes of a
  /// framed plot are drawn only where the origin lies inside it. List
  /// plots keep theirs in `grid_interior`.
  pub unpadded_range: Option<((f64, f64), (f64, f64))>,
  /// The y range proper of the plot, inside its padding: a joined curve is
  /// cut off where it leaves it, and `Filling -> Bottom | Top` fills to its
  /// edges. `None` uses the displayed range.
  pub curve_clip_y: Option<(f64, f64)>,
  /// `ClippingStyle`: how the stretches where a curve is cut off at the
  /// bottom / top of the plot range are drawn along that edge (`None`:
  /// left out).
  pub clipping_style: [Option<ClipEdgeStyle>; 2],
  /// A joined list of data points is drawn through a jump right across the
  /// plot range, which for a sampled function is a pole left out.
  pub join_across_jumps: bool,
  /// `ExclusionsStyle`: the style of the line joining the two ends of a
  /// curve broken at an exclusion, and of the points marking those ends
  /// (`None`: not drawn).
  pub exclusions_style: [Option<SeriesStyle>; 2],
  /// Where the curves are broken at an exclusion, for `ExclusionsStyle`.
  pub exclusion_marks: Vec<ExclusionMark>,
  /// `PlotMarkers -> …`: the glyph each series' points are drawn with,
  /// cycled over the series. Empty = plain round points.
  pub plot_markers: Vec<Option<PlotMarker>>,
  /// The `PointSize` (a fraction of the image width) of scatter dots whose
  /// style sets none; `None` = the fixed 3 px radius dot.
  pub default_point_size: Option<f64>,
  /// Points of a list plot written `Style[point, colour]`: per series, each
  /// such point's coordinates and the colour its dot takes (over the
  /// series' own).
  pub point_colors: Vec<Vec<((f64, f64), (u8, u8, u8))>>,
  /// `LabelStyle -> …`: font size / color applied to the `FrameLabel`,
  /// `AxesLabel`, and `PlotLabel` text. `None` keeps the theme default.
  pub label_style: Option<LabelStyleSpec>,
  /// `FrameTicksStyle -> …`: font size of the tick labels.
  pub tick_label_style: Option<LabelStyleSpec>,
  /// `FrameTicksStyle -> …` per frame edge (bottom, left, top, right): the
  /// colour and thickness of its tick marks, and the colour of its labels.
  pub frame_ticks_styles: [Option<SeriesStyle>; 4],
  /// Set when the bottom `FrameLabel` is a `Grid[…]`/`TableForm[…]` — a
  /// picture to embed under the frame instead of the plain-text
  /// `frame_label_bottom` (which stays empty in that case).
  pub frame_label_bottom_graphic: Option<GraphicLabel>,
  /// Same, for the top edge.
  pub frame_label_top_graphic: Option<GraphicLabel>,
  /// Per-series `Joined` from `ListPlot[…, Joined -> {b1, b2, …}]`: series
  /// `i` is drawn as a connected line when `joined_per_series[i]` is true,
  /// as scattered points otherwise. `None` means every series follows the
  /// plot's single `Joined -> True`/`False` setting instead.
  pub joined_per_series: Option<Vec<bool>>,
  /// `AxesStyle -> s | {xs, ys}`: the style of each axis line, which its
  /// tick marks and tick labels inherit unless `TicksStyle` overrides them.
  pub axes_style: [Option<SeriesStyle>; 2],
  /// `TicksStyle -> s | {xs, ys}`: the style of each axis's tick marks and
  /// tick labels.
  pub ticks_style: [Option<SeriesStyle>; 2],
  /// `FrameStyle` per frame edge (bottom, left, top, right): the style of
  /// the edge, its tick marks and its tick labels.
  pub frame_styles: [Option<SeriesStyle>; 4],
  /// `MeshStyle -> s | {s1, s2, …}`: the style of the mesh points of each
  /// mesh function (cycled). `None` entries draw them in the curve's style.
  pub mesh_styles: Vec<Option<SeriesStyle>>,
  /// `MeshFunctions -> {f1, …}`; empty for the default `{#1 &}`.
  pub mesh_functions: Vec<Expr>,
  /// `MeshShading -> {s1, s2, …}`: the styles of the stretches of a curve
  /// between its mesh points.
  pub mesh_shading: Option<crate::functions::plot_mesh::MeshShade>,
  /// `Plot`'s initial sampling grid per curve, the points of `Mesh -> Full`.
  pub mesh_full_points: Vec<Vec<(f64, f64)>>,
  /// The default mesh function is the curve parameter (`ParametricPlot`),
  /// taken as the position along the point list, instead of x.
  pub mesh_by_parameter: bool,
  /// Mesh values met exactly at the ends of a curve are not marked
  /// (`Plot`, which samples just inside its domain).
  pub mesh_open_ends: bool,
  /// The automatic `PointSize` of the mesh points when it differs from
  /// `default_point_size` (`ListLinePlot`); `None` falls back to that.
  pub mesh_point_size: Option<f64>,
}

/// A `FrameLabel` entry that renders as a picture (`Grid[…]`/`TableForm[…]`)
/// rather than plain text, captured once at option-parsing time.
#[derive(Clone)]
pub(crate) struct GraphicLabel {
  /// The rendered `<svg>…</svg>` markup, at its natural (1×) size.
  pub svg: String,
  /// Natural width/height in nominal pixels, as `svg_natural_size` reports.
  pub width: f64,
  pub height: f64,
}

/// Scale a [`GraphicLabel`]'s natural render-space size (`width`/`height` ×
/// `sf`) down to fit within `max_w` render-space pixels, preserving its
/// aspect ratio. Returns `(width, height)` in render-space pixels.
pub(crate) fn scaled_graphic_label_size(
  g: &GraphicLabel,
  max_w: f64,
  sf: f64,
) -> (f64, f64) {
  let natural_w = g.width * sf;
  let natural_h = g.height * sf;
  if natural_w > max_w && natural_w > 0.0 {
    let scale = max_w / natural_w;
    (max_w, natural_h * scale)
  } else {
    (natural_w, natural_h)
  }
}

/// Font size / color parsed from a `LabelStyle -> …` option. Either field
/// may be absent — an unset one keeps the theme default for that property.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LabelStyleSpec {
  pub color: Option<WoxiColor>,
  pub font_size: Option<f64>,
  pub bold: bool,
}

impl LabelStyleSpec {
  /// Read a `LabelStyle` / `FrameTicksStyle` value; `None` when it sets
  /// none of the colour, size and weight.
  pub(crate) fn parse(value: &Expr) -> Option<Self> {
    let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
    let items: Vec<Expr> = match &val {
      Expr::List(items) => items.to_vec(),
      Expr::FunctionCall { name, args } if name == "Directive" => args.to_vec(),
      other => vec![other.clone()],
    };
    let (bold, _, color, font_size) =
      crate::functions::chart::parse_style_directive_args(&items);
    (color.is_some() || font_size.is_some() || bold).then_some(Self {
      color,
      font_size,
      bold,
    })
  }
}

impl PlotOptions {
  /// Whether tick marks and their labels are drawn at all: `Ticks -> None`
  /// removes them everywhere, `FrameTicks -> None` only on a frame.
  pub(crate) fn ticks_visible(&self) -> bool {
    self.ticks && (self.frame_ticks || !self.frame)
  }

  /// Where the axes cross on the axes `x` and `y`, in data coordinates.
  pub(crate) fn axes_origin_point(
    &self,
    x: AxisScale,
    y: AxisScale,
  ) -> (f64, f64) {
    (
      x.origin(self.axes_origin.0, self.axes_origin_low.0),
      y.origin(self.axes_origin.1, self.axes_origin_low.1),
    )
  }
}

impl Default for PlotOptions {
  fn default() -> Self {
    Self {
      svg_width: DEFAULT_WIDTH,
      svg_height: DEFAULT_HEIGHT,
      full_width: false,
      filling: Filling::None,
      filling_rules: Vec::new(),
      filling_style: None,
      filling_styles: Vec::new(),
      mesh: Mesh::None,
      plot_label: None,
      axes_label: None,
      plot_style: Vec::new(),
      color_function: None,
      color_function_scaling: (true, true),
      axes: (true, true),
      ticks: true,
      plot_points: NUM_SAMPLES,
      max_recursion: 6,
      plot_legends: Vec::new(),
      legend_series: Vec::new(),
      legend_position: LegendPosition::default(),
      legend_marker_kind: LegendMarkerKind::default(),
      legend_colors: Vec::new(),
      grid_lines_y: false,
      grid_lines_x: false,
      grid_x_lines: Vec::new(),
      grid_y_lines: Vec::new(),
      grid_lines_style: None,
      frame: false,
      frame_edges: FrameEdges::ALL,
      frame_hidden_edges: FrameEdges::NONE,
      grid_interior: None,
      frame_ticks: true,
      evaluated: false,
      ticks_x: None,
      ticks_y: None,
      frame_tick_edges: FrameEdgeTicks::defaults(),
      simple_ticks: false,
      frame_label_bottom: None,
      frame_label_left: None,
      frame_label_top: None,
      frame_label_right: None,
      frame_label_bottom_graphic: None,
      frame_label_top_graphic: None,
      date_axis: false,
      callout_labels: Vec::new(),
      plot_labels: Vec::new(),
      plot_label_leaders: Vec::new(),
      point_labels: Vec::new(),
      point_label_position: None,
      point_label_callouts: false,
      log_x: false,
      log_y: false,
      scaled_ticks: (None, None),
      stacked: false,
      epilog: Vec::new(),
      prolog: Vec::new(),
      clip_overlays: true,
      error_bars: Vec::new(),
      interval_markers: IntervalMarkers::default(),
      interval_markers_color: None,
      data_points: Vec::new(),
      background: None,
      image_padding: None,
      aspect_ratio: None,
      aspect_automatic: false,
      aspect_full: false,
      image_size_explicit: (false, false),
      plot_layout: None,
      image_size_value: None,
      unpadded_range: None,
      curve_clip_y: None,
      clipping_style: [None, None],
      join_across_jumps: false,
      exclusions_style: [None, None],
      exclusion_marks: Vec::new(),
      plot_markers: Vec::new(),
      default_point_size: None,
      point_colors: Vec::new(),
      label_style: None,
      tick_label_style: None,
      frame_ticks_styles: [None, None, None, None],
      joined_per_series: None,
      axes_style: [None, None],
      ticks_style: [None, None],
      frame_styles: [None, None, None, None],
      mesh_styles: Vec::new(),
      mesh_functions: Vec::new(),
      mesh_shading: None,
      mesh_full_points: Vec::new(),
      mesh_by_parameter: false,
      mesh_open_ends: false,
      mesh_point_size: None,
      axes_origin: (None, None),
      axes_origin_low: (None, None),
    }
  }
}

/// Draw `Around`-style error bars for one series: a bar spanning each
/// point's uncertainty interval, with short caps at both ends, in the
/// series color. `bars` is parallel to `points` and holds the
/// ((dx_minus, dx_plus), (dy_minus, dy_plus)) half-widths in data units;
/// zero widths draw nothing for that direction.
fn draw_error_bars<
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  points: &[(f64, f64)],
  bars: &[((f64, f64), (f64, f64))],
  color: RGBColor,
  caps: bool,
) -> Result<(), InterpreterError> {
  // WL 15 caps each end with a mark 3 pt to either side of the bar.
  let cap = (3.0 * RESOLUTION_SCALE as f64).round() as i32;
  let style = color.stroke_width(RESOLUTION_SCALE);
  let err = |e: plotters::drawing::DrawingAreaErrorKind<DB::ErrorType>| {
    InterpreterError::EvaluationError(format!("Plot: {e}"))
  };
  for (&(x, y), &(dx, dy)) in points.iter().zip(bars) {
    if !(x.is_finite() && y.is_finite()) {
      continue;
    }
    if dy.0 > 0.0 || dy.1 > 0.0 {
      let (lo, hi) = (y - dy.0, y + dy.1);
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x, lo), (x, hi)],
          style,
        )))
        .map_err(err)?;
      if caps {
        chart
          .draw_series([lo, hi].into_iter().map(|end| {
            EmptyElement::at((x, end))
              + PathElement::new(vec![(-cap, 0), (cap, 0)], style)
          }))
          .map_err(err)?;
      }
    }
    if dx.0 > 0.0 || dx.1 > 0.0 {
      let (lo, hi) = (x - dx.0, x + dx.1);
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(lo, y), (hi, y)],
          style,
        )))
        .map_err(err)?;
      if caps {
        chart
          .draw_series([lo, hi].into_iter().map(|end| {
            EmptyElement::at((end, y))
              + PathElement::new(vec![(0, -cap), (0, cap)], style)
          }))
          .map_err(err)?;
      }
    }
  }
  Ok(())
}

/// Draw `IntervalMarkers -> "Bands"` for one series: a translucent filled
/// band spanning each point's y-uncertainty interval, outlined in the series
/// color. `bars` is parallel to `points` (see [`draw_error_bars`]); a point
/// without y-uncertainty pinches the band to the point itself.
fn draw_interval_band<
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  points: &[(f64, f64)],
  bars: &[((f64, f64), (f64, f64))],
  color: RGBColor,
) -> Result<(), InterpreterError> {
  let edges: Vec<((f64, f64), (f64, f64))> = points
    .iter()
    .zip(bars)
    .filter(|((x, y), _)| x.is_finite() && y.is_finite())
    .map(|(&(x, y), &(_dx, dy))| ((x, y - dy.0), (x, y + dy.1)))
    .collect();
  if edges.is_empty() {
    return Ok(());
  }
  // Closed outline: along the upper edge, then back along the lower edge.
  let mut outline: Vec<(f64, f64)> = edges.iter().map(|e| e.1).collect();
  outline.extend(edges.iter().rev().map(|e| e.0));
  if outline.len() >= 3 {
    chart
      .draw_series(std::iter::once(Polygon::new(
        outline.clone(),
        color.mix(0.2),
      )))
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }
  outline.push(edges[0].1);
  chart
    .draw_series(std::iter::once(PathElement::new(
      outline,
      color.stroke_width(RESOLUTION_SCALE),
    )))
    .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  Ok(())
}

/// Default plot style colors of WL 15 (the `DefaultPlotStyle` of the
/// Automatic plot theme), exactly as WL writes them into the plot's
/// `Directive`s. The series cycle through all 15.
pub(crate) const PLOT_COLORS_RGB: [(f64, f64, f64); 15] = [
  (0.24, 0.6, 0.8),
  (0.95, 0.627, 0.1425),
  (0.455, 0.7, 0.21),
  (0.922526, 0.385626, 0.209179),
  (0.578, 0.51, 0.85),
  (0.772079, 0.431554, 0.102387),
  (0., 0.6, 0.6),
  (0.9, 0.36, 0.684),
  (0.56, 0.7, 0.),
  (0.9, 0.54, 0.),
  (0.585, 0.5976, 0.9),
  (0.9, 0.45, 0.45),
  (0.7632186717367254, 0.8174402144034898, 0.),
  (0.8554632469481959, 0.24885384718327969, 0.8334053873948551),
  (0.032972823413648836, 0.7396719749139476, 0.534700104525218),
];

/// Font size of chart labels (`ChartLabels`, axes and frame labels) in the
/// bar, bubble and histogram renderers,
/// in display pixels (WL 15: the tick-label size).
const CHART_LABEL_FONT: f64 = 10.0;
/// Font size of a chart's `PlotLabel`, in display pixels (WL 15's
/// `PlotLabel` size).
const CHART_TITLE_FONT: f64 = 12.0;

/// `PLOT_COLORS_RGB` quantized to 8-bit channels for the SVG renderers.
pub(crate) const PLOT_COLORS: [(u8, u8, u8); 15] = {
  let mut out = [(0u8, 0u8, 0u8); 15];
  let mut i = 0;
  while i < 15 {
    let (r, g, b) = PLOT_COLORS_RGB[i];
    out[i] = (
      (r * 255.0 + 0.5) as u8,
      (g * 255.0 + 0.5) as u8,
      (b * 255.0 + 0.5) as u8,
    );
    i += 1;
  }
  out
};

/// The exact WL color of palette entry `idx` (cycling), as an `RGBColor`.
pub(crate) fn plot_color_expr(idx: usize) -> Expr {
  let (r, g, b) = PLOT_COLORS_RGB[idx % PLOT_COLORS_RGB.len()];
  Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)].into(),
  }
}

/// Map an 8-bit palette color back to its exact WL color expression, or
/// `None` when it is not a palette entry (e.g. a user PlotStyle).
pub(crate) fn palette_color_expr(rgb: (u8, u8, u8)) -> Option<Expr> {
  PLOT_COLORS
    .iter()
    .position(|c| *c == rgb)
    .map(plot_color_expr)
}

/// An 8-bit color as an `RGBColor`, written with the exact WL channel values
/// when it is a default palette entry.
pub(crate) fn rgb_u8_expr(rgb: (u8, u8, u8)) -> Expr {
  palette_color_expr(rgb).unwrap_or_else(|| Expr::FunctionCall {
    name: "RGBColor".to_string(),
    args: vec![
      Expr::Real(f64::from(rgb.0) / 255.0),
      Expr::Real(f64::from(rgb.1) / 255.0),
      Expr::Real(f64::from(rgb.2) / 255.0),
    ]
    .into(),
  })
}

/// Generate SVG for a 2D plot with filling (legacy wrapper for list_plot callers).
pub(crate) fn generate_svg_with_filling(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  generate_svg_with_options(all_points, x_range, y_range, opts)
}

/// A dashed data series deferred for post-render emission as a single
/// `<polyline stroke-dasharray>` instead of one `<polyline>` per dash — which
/// plotters' backend would otherwise produce.
struct DashedOverlay {
  color: (u8, u8, u8),
  /// Stroke alpha (1.0 = opaque).
  opacity: f64,
  stroke_w: u32,
  /// Dash pattern as fractions of the plot width (matching the legend swatch).
  dashes: Vec<f64>,
  /// Polyline vertices in data space.
  points: Vec<(f64, f64)>,
}

/// One dash length, in the coordinate system its `<polyline>` is drawn in.
///
/// A negative length is absolute pixels (a named size — `Dashed` is 4 on / 4
/// off whatever the image size); a positive one is a fraction of `reference`,
/// the width Wolfram measures relative dashes against; zero is the dot
/// Wolfram draws as one pixel. `px_per_unit` converts a pixel length into the
/// target system (render space counts `RESOLUTION_SCALE` units per pixel).
pub(crate) fn dash_len(d: f64, reference: f64, px_per_unit: f64) -> f64 {
  let len = if d < 0.0 {
    -d * px_per_unit
  } else if d == 0.0 {
    px_per_unit
  } else {
    d * reference
  };
  len.max(0.5)
}

/// Render collected dashed lines as single `<polyline stroke-dasharray>`
/// elements, mapping data coordinates to render-space pixels with the same
/// linear transform used for axis labels/ticks. Dash lengths use the
/// plotting-area extent so series dashes match the legend swatches exactly.
#[allow(clippy::too_many_arguments)]
fn render_dash_overlays(
  overlays: &[DashedOverlay],
  plot_x0: f64,
  plot_y0: f64,
  plot_w: f64,
  plot_h: f64,
  render_w: f64,
  x_min: f64,
  x_max: f64,
  y_min: f64,
  y_max: f64,
  log_x: bool,
  log_y: bool,
) -> String {
  // Log axes place data at the logarithm of its value, so map through `ln`
  // before the linear transform to the plotting area.
  let to_axis = |v: f64, log: bool| if log { v.ln() } else { v };
  let (x_min, x_max) = (to_axis(x_min, log_x), to_axis(x_max, log_x));
  let (y_min, y_max) = (to_axis(y_min, log_y), to_axis(y_max, log_y));
  let xr = x_max - x_min;
  let yr = y_max - y_min;
  if !xr.is_finite() || !yr.is_finite() || xr.abs() < 1e-12 || yr.abs() < 1e-12
  {
    return String::new();
  }
  let mut svg = String::new();
  for ov in overlays {
    if ov.points.len() < 2 {
      continue;
    }
    let pts: Vec<String> = ov
      .points
      .iter()
      .map(|&(x, y)| {
        let px = plot_x0 + (to_axis(x, log_x) - x_min) / xr * plot_w;
        let py = plot_y0 + (y_max - to_axis(y, log_y)) / yr * plot_h;
        format!("{px:.1},{py:.1}")
      })
      .collect();
    let dash: Vec<String> = ov
      .dashes
      .iter()
      .map(|d| {
        format!("{:.1}", dash_len(*d, render_w, RESOLUTION_SCALE as f64))
      })
      .collect();
    let (r, g, b) = ov.color;
    svg.push_str(&format!(
      "<polyline fill=\"none\" stroke=\"rgb({},{},{})\" \
       {}stroke-width=\"{}\" stroke-dasharray=\"{}\" stroke-linecap=\"round\" \
       points=\"{}\"/>\n",
      r,
      g,
      b,
      if ov.opacity < 1.0 {
        format!("stroke-opacity=\"{}\" ", ov.opacity)
      } else {
        String::new()
      },
      ov.stroke_w,
      dash.join(","),
      pts.join(" ")
    ));
  }
  svg
}

/// Default color for an unstyled dashed grid line. Approximates the flat-gray
/// look of the solid grid lines (`rgb(102,102,102)` at 50% opacity over a white
/// background), since dash overlays carry no opacity.
const DEFAULT_GRID_DASH_RGB: (u8, u8, u8) = (179, 179, 179);
/// Width of a grid line without an explicit thickness, in points.
const DEFAULT_GRID_THICKNESS: f64 = 0.2;

/// Resolve one axis' grid-line positions into `(position, style)` pairs.
///
/// Explicit positions (with their optional per-line directives) take
/// precedence over the evenly spaced automatic lines that `GridLines ->
/// Automatic` draws at `step` intervals. Lines that carry no style of their
/// own inherit `default_style` (the plot's `GridLinesStyle`).
fn resolve_grid_lines(
  explicit: &[GridLine],
  automatic: bool,
  min: f64,
  max: f64,
  step: f64,
  default_style: Option<&SeriesStyle>,
) -> Vec<(f64, SeriesStyle)> {
  let inherit = |style: &SeriesStyle| -> SeriesStyle {
    match default_style {
      Some(d)
        if style.color.is_none()
          && style.thickness.is_none()
          && style.dashing.is_none() =>
      {
        d.clone()
      }
      _ => style.clone(),
    }
  };
  if !explicit.is_empty() {
    return explicit
      .iter()
      .map(|g| (g.pos, inherit(&g.style)))
      .collect();
  }
  if !automatic || step <= 0.0 || !step.is_finite() {
    return Vec::new();
  }
  let style = inherit(&SeriesStyle::default());
  let mut v = Vec::new();
  let mut pos = (min / step).ceil() * step;
  while pos <= max {
    v.push((pos, style.clone()));
    pos += step;
  }
  v
}

/// The grid lines of one axis spanning `min..max`: [`resolve_grid_lines`]
/// on a linear axis; on a log axis the explicit lines, or the automatic
/// ones at the labelled log ticks.
fn resolve_axis_grid_lines(
  explicit: &[GridLine],
  automatic: bool,
  (min, max): (f64, f64),
  log: bool,
  step: f64,
  default_style: Option<&SeriesStyle>,
) -> Vec<(f64, SeriesStyle)> {
  if !log || !explicit.is_empty() {
    return resolve_grid_lines(
      explicit,
      automatic,
      min,
      max,
      step,
      default_style,
    );
  }
  if !automatic {
    return Vec::new();
  }
  let style = default_style.cloned().unwrap_or_default();
  crate::functions::plot_scaling::log_axis_ticks(min, max)
    .majors
    .into_iter()
    .map(|(p, _)| (p, style.clone()))
    .collect()
}

/// Resolve a grid line's style into rendering properties:
/// `(optional rgb, stroke width in render units, optional dash fractions)`.
/// A `None` rgb means "use the default gray grid color".
fn grid_line_props(
  style: &SeriesStyle,
) -> (Option<(u8, u8, u8)>, u32, Option<Vec<f64>>) {
  let rgb = style.color.as_ref().map(|c| {
    (
      (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  });
  let stroke_w = match style.thickness {
    Some(t) => ((t * RESOLUTION_SCALE as f64).round() as u32).max(1),
    // WL 15 strokes grid lines 0.2 pt wide unless a thickness is given.
    None => {
      ((DEFAULT_GRID_THICKNESS * RESOLUTION_SCALE as f64).round() as u32).max(1)
    }
  };
  (rgb, stroke_w, style.dashing.clone())
}

/// A label that spans several lines (a `Column[…]` tick label) as one
/// `<tspan>` per line, each re-anchored at `x` and `line_height` below the
/// previous one. A single-line label is returned unchanged.
fn stacked_lines_svg(label: &str, x: f64, line_height: f64) -> String {
  if !label.contains('\n') {
    return label.to_string();
  }
  let mut out = String::new();
  for (i, line) in label.split('\n').enumerate() {
    let dy = if i == 0 { 0.0 } else { line_height };
    out.push_str(&format!(
      "<tspan x=\"{x:.1}\" dy=\"{dy:.1}\">{line}</tspan>"
    ));
  }
  out
}

/// Draw the marks and labels of an explicit `Ticks -> {xspec, yspec}`, in
/// the same place the automatic ticks would sit: the x ticks below the
/// bottom axis and the y ticks left of the left one. Returns the empty
/// string when neither axis was given explicit positions.
///
/// Shared by the line and scatter renderers, both of which suppress their
/// automatic ticks once an axis carries explicit ones. The labels arrive
/// ready to draw from [`parse_explicit_ticks`] — escaped, and typeset when
/// the tick was given as an expression (`Pi/2` reads `π/2`).
/// `area` is the plotting rectangle `(x0, y0, w, h)` and `range` the data
/// range `(x_min, x_max, y_min, y_max)` it maps, both in render units.
///
/// Takes the explicit tick lists directly (rather than a whole options
/// struct) so the plotters-based chart renderers (`Histogram`, `BarChart`, …)
/// can share this one drawing routine instead of duplicating it.
pub(crate) fn explicit_ticks_svg(
  ticks_x: Option<&Vec<(f64, String)>>,
  ticks_y: Option<&Vec<(f64, String)>>,
  (plot_x0, plot_y0, plot_w, plot_h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  sf: f64,
  label_fill: &str,
  hidden_edges: FrameEdges,
  tick_font_pt: f64,
) -> String {
  let mut out = String::new();
  let axis_y = plot_y0 + plot_h;
  let tick_len = sf * 5.0;
  let tick_font = sf * tick_font_pt;
  if let Some(ticks) = ticks_x {
    for (pos, label) in ticks {
      if *pos < x_min || *pos > x_max || x_max <= x_min {
        continue;
      }
      let px = plot_x0 + (pos - x_min) / (x_max - x_min) * plot_w;
      if !hidden_edges.bottom {
        out.push_str(&format!(
          "<line x1=\"{px:.1}\" y1=\"{axis_y:.1}\" x2=\"{px:.1}\" y2=\"{:.1}\" stroke=\"{label_fill}\" stroke-width=\"{sf:.1}\"/>\n",
          axis_y - tick_len,
        ));
      }
      out.push_str(&format!(
        "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{tick_font:.0}\" fill=\"{label_fill}\">{}</text>\n",
        axis_y + tick_font,
        stacked_lines_svg(label, px, tick_font * 1.2),
      ));
    }
  }
  if let Some(ticks) = ticks_y {
    for (pos, label) in ticks {
      if *pos < y_min || *pos > y_max || y_max <= y_min {
        continue;
      }
      let py = plot_y0 + plot_h - (pos - y_min) / (y_max - y_min) * plot_h;
      if !hidden_edges.left {
        out.push_str(&format!(
          "<line x1=\"{plot_x0:.1}\" y1=\"{py:.1}\" x2=\"{:.1}\" y2=\"{py:.1}\" stroke=\"{label_fill}\" stroke-width=\"{sf:.1}\"/>\n",
          plot_x0 + tick_len,
        ));
      }
      out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{py:.1}\" text-anchor=\"end\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{tick_font:.0}\" fill=\"{label_fill}\">{label}</text>\n",
        plot_x0 - tick_len,
      ));
    }
  }
  out
}

/// Core SVG generation for 2D line plots with full option support.
/// Draw a plot's labels — `FrameLabel` on all four edges, `AxesLabel` at the
/// ends of the axes, and the `PlotLabel` above them — into `labels_svg`.
///
/// Shared by the line and scatter renderers and by the density/contour
/// family in `field_plot`: each draws its own axes and then injects text,
/// and the placement rules are the same for all of them.
/// `area` is the plotting rectangle `(x0, y0, w, h)` and `range` the data
/// range `(x_min, x_max, y_min, y_max)` it maps, both in render units.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plot_labels_svg(
  opts: &PlotOptions,
  (plot_x0, margin_top, plot_w, plot_h): (f64, f64, f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
  layout: &plot_padding::LabelLayout,
  sf: f64,
  label_fill: &str,
  title_default_fill: &str,
  canvas_width: f64,
) -> String {
  use plot_padding::{
    FRAME_LABEL_GAP, ascent, block_height, line_height, line_pitch,
  };
  let axis_y = margin_top + plot_h;
  // `LabelStyle` overrides the frame/axes/title text's default size and
  // color; either half may be absent, in which case that property keeps
  // the theme default.
  let label_style_color = opts.label_style.and_then(|s| s.color);
  let label_pt = plot_padding::plot_label_font(opts);
  let title_pt = plot_padding::plot_title_font(opts);
  let font_size = label_pt * sf;
  let pitch = line_pitch(label_pt) * sf;
  let lines_of = |t: &str| t.split('\n').count();
  let label_fill_owned =
    label_style_color.map(super::graphics::Color::to_svg_rgb);
  let label_fill = label_fill_owned.as_deref().unwrap_or(label_fill);
  let label_weight = if opts.label_style.is_some_and(|s| s.bold) {
    " font-weight=\"bold\""
  } else {
    ""
  };
  let title_default_fill_owned =
    label_style_color.map(super::graphics::Color::to_svg_rgb);
  let title_default_fill = title_default_fill_owned
    .as_deref()
    .unwrap_or(title_default_fill);
  let [room_bottom, room_left, room_top, room_right] =
    layout.tick_room.map(|r| r * sf);
  let gap = FRAME_LABEL_GAP * sf;
  let mut labels_svg = String::new();
  let text_svg = |label: &str, x: f64| {
    stacked_lines_svg(
      &crate::functions::graphics::box_string_to_svg_scaled(
        label,
        RESOLUTION_SCALE as f64,
      ),
      x,
      pitch,
    )
  };
  // Bottom/left FrameLabel: centred outside the plot area, 6 pt beyond
  // the tick labels of their edge.
  let bottom_label_top = axis_y + room_bottom + gap;
  let mut bottom_text_h = 0.0;
  if let Some(x_label) = &opts.frame_label_bottom
    && !x_label.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let base_y = bottom_label_top + ascent(label_pt) * sf;
    bottom_text_h = block_height(lines_of(x_label), label_pt) * sf;
    labels_svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\"{label_weight} \
         fill=\"{label_fill}\">{}</text>\n",
      text_svg(x_label, cx)
    ));
  }
  // A `FrameLabel` that is a `Grid[…]`/`TableForm[…]` is embedded as a
  // picture, scaled to the frame's width, instead of drawn as text.
  if let Some(graphic) = &opts.frame_label_bottom_graphic {
    let (gw, gh) = scaled_graphic_label_size(graphic, plot_w, sf);
    let cx = plot_x0 + plot_w / 2.0;
    // Stack below the plain-text bottom label when both are present.
    let top_y = bottom_label_top
      + if bottom_text_h > 0.0 {
        bottom_text_h + gap
      } else {
        0.0
      };
    let cy = top_y + gh / 2.0;
    labels_svg.push_str(&crate::functions::graphics::embed_svg_centered(
      &graphic.svg,
      cx,
      cy,
      gw,
      gh,
    ));
  }
  if let Some(y_label) = &opts.frame_label_left
    && !y_label.is_empty()
  {
    let cy = margin_top + plot_h / 2.0;
    // Rotated a quarter turn counterclockwise, the glyphs rise to the
    // left of the baseline; the first line is the outermost one.
    let box_left = plot_x0
      - room_left
      - gap
      - block_height(lines_of(y_label), label_pt) * sf;
    let lx = box_left + ascent(label_pt) * sf;
    labels_svg.push_str(&format!(
      "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\"{label_weight} \
         fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
      text_svg(y_label, lx)
    ));
  }

  // AxesLabel: at the far end of each axis, the way Wolfram writes it.
  let x_scale = AxisScale {
    min: x_min,
    max: x_max,
    log: opts.log_x,
  };
  let y_scale = AxisScale {
    min: y_min,
    max: y_max,
    log: opts.log_y,
  };
  let area = (plot_x0, margin_top, plot_w, plot_h);
  labels_svg.push_str(&axes_label_svg_at(
    opts.axes_label.as_ref(),
    area,
    crate::functions::plot_axes::axes_crossing_px(
      area,
      x_scale,
      y_scale,
      opts.axes_origin_point(x_scale, y_scale),
    ),
    opts.axes,
    font_size,
    label_fill,
    Some(canvas_width),
  ));

  // Top FrameLabel: 6 pt above the frame (and its tick labels).
  if let Some(top_label) = &opts.frame_label_top
    && !top_label.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let top = margin_top
      - room_top
      - gap
      - block_height(lines_of(top_label), label_pt) * sf;
    // One element per line, so no renderer collapses the breaks.
    for (i, line) in top_label.split('\n').enumerate() {
      let ty = top + ascent(label_pt) * sf + i as f64 * pitch;
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\"{label_weight} \
           fill=\"{label_fill}\">{}</text>\n",
        text_svg(line, cx)
      ));
    }
  }

  // A top `FrameLabel` drawn as a picture sits just above the frame.
  if let Some(graphic) = &opts.frame_label_top_graphic {
    let (gw, gh) = scaled_graphic_label_size(graphic, plot_w, sf);
    let cx = plot_x0 + plot_w / 2.0;
    let cy = margin_top - room_top - gap - gh / 2.0;
    labels_svg.push_str(&crate::functions::graphics::embed_svg_centered(
      &graphic.svg,
      cx,
      cy,
      gw,
      gh,
    ));
  }

  // Right FrameLabel: 6 pt right of the frame, reading upwards like the
  // left one (the glyphs rise to the left of the baseline).
  if let Some(right_label) = &opts.frame_label_right
    && !right_label.is_empty()
  {
    let cy = margin_top + plot_h / 2.0;
    let rx = plot_x0 + plot_w + room_right + gap + ascent(label_pt) * sf;
    labels_svg.push_str(&format!(
      "<text x=\"{rx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\"{label_weight} \
         fill=\"{label_fill}\" transform=\"rotate(-90,{rx:.1},{cy:.1})\">{}</text>\n",
      text_svg(right_label, rx)
    ));
  }

  // PlotLabel: centred over the plotting area in its own band at the top.
  if let Some(sl) = &opts.plot_label
    && !sl.text.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    let fs = title_pt * sf;
    // A title wider than the picture (a long `Row` of text) shrinks to fit
    // instead of being clipped at both edges.
    let widest = sl.max_line_chars() as f64;
    let fs = if widest > 0.0 && !sl.framed {
      fs.min((canvas_width - 8.0 * sf) / (widest * 0.52))
    } else {
      fs
    };
    let ty = layout.title_top + ascent(fs / sf) * sf;
    // A box around the title must stay inside the image, even when a tight
    // `ImagePadding` leaves less room above the plot than the box needs.
    let ty = if sl.framed {
      let top = ty - fs * 0.9 - 6.0 * sf;
      ty + (sf - top).max(0.0)
    } else {
      ty
    };
    let fill = sl
      .color
      .as_ref()
      .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
    let mut style_attrs = String::new();
    if sl.bold {
      style_attrs.push_str(" font-weight=\"bold\"");
    }
    if sl.italic {
      style_attrs.push_str(" font-style=\"italic\"");
    }
    let title_pitch = line_pitch(fs / sf) * sf;
    if sl.framed {
      // Box the title: the widest line plus Framed's 6pt margin per side.
      let margin = 6.0 * sf;
      let w = sl.max_line_chars() as f64 * fs * 0.47 + 2.0 * margin;
      let lines = 1.0 + sl.extra_line_count() as f64;
      let h =
        line_height(fs / sf) * sf + (lines - 1.0) * title_pitch + 2.0 * margin;
      let top = ty - ascent(fs / sf) * sf - margin;
      labels_svg.push_str(&format!(
        "<rect x=\"{:.1}\" y=\"{top:.1}\" width=\"{w:.1}\" \
           height=\"{h:.1}\" rx=\"{:.1}\" fill=\"none\" \
           stroke=\"{}\" stroke-width=\"{:.1}\"/>\n",
        cx - w / 2.0,
        3.0 * sf,
        crate::functions::graphics::theme().framed_border,
        sf,
      ));
    }
    // A plain string with line breaks stacks its lines too.
    let body = if sl.markup.is_none()
      && sl.extra_line_count() == 0
      && sl.text.contains('\n')
    {
      stacked_lines_svg(
        &crate::functions::graphics::box_string_to_svg_scaled(
          &sl.text,
          RESOLUTION_SCALE as f64,
        ),
        cx,
        title_pitch,
      )
    } else {
      sl.svg_scaled_stacked(sf, cx, title_pitch)
    };
    labels_svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
         fill=\"{fill}\"{style_attrs}>{body}</text>\n",
    ));
  }

  labels_svg
}

/// Overlay primitives on a log axis use that axis's natural-log coordinates
/// (`Point[{Log[x], y}]` in a `LogLinearPlot`), so express the displayed
/// range in them.
fn log_overlay_ranges(
  opts: &PlotOptions,
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
) -> (f64, f64, f64, f64) {
  let (x_min, x_max) = if opts.log_x && x_min > 0.0 && x_max > 0.0 {
    (x_min.ln(), x_max.ln())
  } else {
    (x_min, x_max)
  };
  let (y_min, y_max) = if opts.log_y && y_min > 0.0 && y_max > 0.0 {
    (y_min.ln(), y_max.ln())
  } else {
    (y_min, y_max)
  };
  (x_min, x_max, y_min, y_max)
}

/// Explicit ticks of an axis, expressed in the natural-log coordinates the
/// axis is drawn in when it is logarithmic (ticks at non-positive positions
/// cannot be placed and are dropped).
fn explicit_ticks_in_axis_space(
  ticks: Option<&Vec<(f64, String)>>,
  log: bool,
) -> Option<Vec<(f64, String)>> {
  let ticks = ticks?;
  if !log {
    return Some(ticks.clone());
  }
  Some(
    ticks
      .iter()
      .filter(|(pos, _)| *pos > 0.0)
      .map(|(pos, label)| (pos.ln(), label.clone()))
      .collect(),
  )
}

/// Draw an `Epilog`'s primitives over a finished plot. Both renderers (the
/// line one below and the scatter one) inject them the same way, so a plot
/// keeps its epilog whichever of the two drew it.
///
/// `area` is the plotting area as `(x0, y0, width, height)` in render-space
/// pixels and `ranges` the data extents it displays, as `(x_min, x_max,
/// y_min, y_max)`.
fn inject_epilog(
  buf: &mut String,
  opts: &PlotOptions,
  area: (f64, f64, f64, f64),
  ranges: (f64, f64, f64, f64),
  scale: f64,
) {
  if opts.epilog.is_empty() {
    return;
  }
  let (x0, y0, w, h) = area;
  let (x_min, x_max, y_min, y_max) = log_overlay_ranges(opts, ranges);
  let area = crate::functions::plot_epilog::PlotArea {
    x0,
    y0,
    w,
    h,
    x_min,
    x_max,
    y_min,
    y_max,
    scale,
  };
  let epilog_svg = crate::functions::plot_epilog::render_overlay_svg(
    &opts.epilog,
    &area,
    "epilog",
    opts.clip_overlays,
  );
  if let Some(pos) = buf.rfind("</svg>") {
    buf.insert_str(pos, &epilog_svg);
  }
}

/// Draw a `Prolog`'s primitives under a finished plot — the mirror of
/// [`inject_epilog`]: same primitive-to-SVG rendering and the same `area`/
/// `ranges`/`scale` meaning, but spliced in right after the plot's own
/// background rect instead of before `</svg>`, so later-painted content
/// (axes, the curve itself, any epilog) draws on top of it rather than the
/// reverse.
fn inject_prolog(
  buf: &mut String,
  opts: &PlotOptions,
  area: (f64, f64, f64, f64),
  ranges: (f64, f64, f64, f64),
  scale: f64,
) {
  if opts.prolog.is_empty() {
    return;
  }
  let (x0, y0, w, h) = area;
  let (x_min, x_max, y_min, y_max) = log_overlay_ranges(opts, ranges);
  let area = crate::functions::plot_epilog::PlotArea {
    x0,
    y0,
    w,
    h,
    x_min,
    x_max,
    y_min,
    y_max,
    scale,
  };
  let prolog_svg = crate::functions::plot_epilog::render_overlay_svg(
    &opts.prolog,
    &area,
    "prolog",
    opts.clip_overlays,
  );
  splice_after_background(buf, &prolog_svg);
}

/// Splice `overlay` into a finished SVG right after the canvas background
/// rect, so everything painted later (axes, curves) draws on top of it.
pub(crate) fn splice_after_background(buf: &mut String, overlay: &str) {
  // `root.fill(&bg_color)` (plotters) draws an opaque rect spanning the
  // whole canvas as the very first thing after the opening `<svg …>` tag
  // (and any embedded `<defs>` for fonts, which never paints anything
  // itself). Splicing prolog in right after `<svg …>` would place it
  // *before* that rect in document order, so the rect — opaque, painted
  // later — would cover it entirely. Insert after the background rect's
  // `/>` instead, so prolog paints on the blank canvas but still under
  // the axes and curve that follow.
  let Some(tag_start) = buf.find("<svg") else {
    return;
  };
  let Some(tag_end) = buf[tag_start..].find('>') else {
    return;
  };
  let after_svg_tag = tag_start + tag_end + 1;
  let after_defs = match buf[after_svg_tag..].find("</defs>") {
    Some(p) => after_svg_tag + p + "</defs>".len(),
    None => after_svg_tag,
  };
  let insert_at = match buf[after_defs..].find("/>") {
    Some(p) => after_defs + p + 2,
    None => after_defs,
  };
  buf.insert_str(insert_at, overlay);
}

/// Resolve `AspectRatio -> Automatic` against the final plotted ranges: the
/// plotting area takes the data's own proportions, so equal x and y distances
/// are drawn equally long. Borrows `opts` unchanged for any other aspect.
fn with_automatic_aspect(
  opts: &PlotOptions,
  x_range: (f64, f64),
  y_range: (f64, f64),
) -> std::borrow::Cow<'_, PlotOptions> {
  let (dx, dy) = (x_range.1 - x_range.0, y_range.1 - y_range.0);
  if opts.aspect_automatic
    && opts.aspect_ratio.is_none()
    && dx.is_finite()
    && dy.is_finite()
    && dx > 0.0
    && dy > 0.0
  {
    let mut resolved = opts.clone();
    resolved.aspect_ratio = Some(dy / dx);
    std::borrow::Cow::Owned(resolved)
  } else {
    std::borrow::Cow::Borrowed(opts)
  }
}

/// `opts` with the `Charting`SimpleTicks` of its plot theme filled in on
/// the axes without explicit ticks, over the displayed ranges.
fn with_simple_ticks(
  opts: &PlotOptions,
  x_range: (f64, f64),
  y_range: (f64, f64),
) -> std::borrow::Cow<'_, PlotOptions> {
  if !opts.simple_ticks || (opts.ticks_x.is_some() && opts.ticks_y.is_some()) {
    return std::borrow::Cow::Borrowed(opts);
  }
  let mut resolved = opts.clone();
  resolved
    .ticks_x
    .get_or_insert_with(|| simple_ticks(x_range.0, x_range.1));
  resolved
    .ticks_y
    .get_or_insert_with(|| simple_ticks(y_range.0, y_range.1));
  std::borrow::Cow::Owned(resolved)
}

/// A linear plotters axis whose tick positions are given explicitly (see
/// the frame ticks in [`generate_svg_with_options`]); without points it
/// falls back to plotters' own choice.
struct FixedKeyPoints {
  inner: plotters::coord::types::RangedCoordf64,
  points: Vec<f64>,
}

impl FixedKeyPoints {
  fn new(range: std::ops::Range<f64>, points: Vec<f64>) -> Self {
    Self {
      inner: range.into(),
      points,
    }
  }
}

impl plotters::coord::ranged1d::Ranged for FixedKeyPoints {
  type FormatOption = plotters::coord::ranged1d::DefaultFormatting;
  type ValueType = f64;
  fn map(&self, value: &f64, limit: (i32, i32)) -> i32 {
    self.inner.map(value, limit)
  }
  fn key_points<Hint: plotters::coord::ranged1d::KeyPointHint>(
    &self,
    hint: Hint,
  ) -> Vec<f64> {
    if self.points.is_empty() {
      self.inner.key_points(hint)
    } else {
      self.points.clone()
    }
  }
  fn range(&self) -> std::ops::Range<f64> {
    self.inner.range()
  }
}

fn generate_svg_with_options(
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  let opts = &*with_automatic_aspect(opts, x_range, y_range);
  let opts = &*with_simple_ticks(opts, x_range, y_range);
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  // `ColorFunctionScaling` rescales over the sampled data, not the padded
  // plot range: the highest point of `Plot[Sin[x], …, ColorFunction ->
  // Function[{x, y}, Hue[y]]]` gets `Hue[1.]`.
  let color_scale_ranges =
    data_extents(all_points).unwrap_or((x_min, x_max, y_min, y_max));
  let mut svg_width = opts.svg_width;
  let mut svg_height = opts.svg_height;
  let full_width = opts.full_width;
  let (show_x_axis, show_y_axis) = opts.axes;
  // A framed plot carries its ticks on the frame edges, so they are labelled
  // whether or not the axes themselves are drawn — `Frame -> True, Axes ->
  // False` is the standard framed look, and it used to come out bare.
  let (tick_axis_x, tick_axis_y) = if opts.frame {
    (true, true)
  } else {
    (show_x_axis, show_y_axis)
  };
  let show_ticks = opts.ticks_visible();
  // Which frame edges (bottom, left, top, right) carry their automatic tick
  // labels; an unframed plot labels its axes.
  let frame_edge_labeled: [bool; 4] = std::array::from_fn(|edge| {
    if opts.frame {
      matches!(opts.frame_tick_edges[edge], FrameEdgeTicks::Labeled)
    } else {
      edge < 2
    }
  });
  let mut render_width = svg_width * RESOLUTION_SCALE;
  let mut render_height = svg_height * RESOLUTION_SCALE;

  let sf = RESOLUTION_SCALE as f64;

  // Compute dynamic margins for labels
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  // A `FrameLabel` sits centred outside the bottom/left edge; an `AxesLabel`
  // sits at the far end of its axis, so it needs room to the right and above
  // instead.
  let has_x_axis_label = opts
    .frame_label_bottom
    .as_ref()
    .is_some_and(|t| !t.is_empty())
    || opts.frame_label_bottom_graphic.is_some();
  let has_y_axis_label = opts
    .frame_label_left
    .as_ref()
    .is_some_and(|t| !t.is_empty());
  let axes_label_x = opts
    .axes_label
    .as_ref()
    .map(|(x, _)| x.as_str())
    .filter(|x| !x.is_empty() && show_x_axis);
  let axes_label_y = opts
    .axes_label
    .as_ref()
    .map(|(_, y)| y.as_str())
    .filter(|y| !y.is_empty() && show_y_axis);
  let has_top_label =
    opts.frame_label_top.as_ref().is_some_and(|t| !t.is_empty())
      || opts.frame_label_top_graphic.is_some();
  let has_right_label = opts
    .frame_label_right
    .as_ref()
    .is_some_and(|t| !t.is_empty());

  // The axes' scales and ticks, and where the axes cross.
  let x_scale = AxisScale {
    min: x_min,
    max: x_max,
    log: opts.log_x,
  };
  let y_scale = AxisScale {
    min: y_min,
    max: y_max,
    log: opts.log_y,
  };
  let x_ticks = opts.scaled_ticks.0.clone().unwrap_or_else(|| {
    AxisTicks::new(x_scale, opts.date_axis, opts.ticks_x.as_ref())
  });
  let y_ticks = opts
    .scaled_ticks
    .1
    .clone()
    .unwrap_or_else(|| AxisTicks::new(y_scale, false, opts.ticks_y.as_ref()));
  let origin = opts.axes_origin_point(x_scale, y_scale);
  // An unframed plot draws its axes itself, crossing at the `AxesOrigin`
  // (see `plot_axes`); plotters only maps the data.
  let origin_axes = !opts.frame && (show_x_axis || show_y_axis);

  // Without `ImagePadding` the room around the plotting area follows the
  // Wolfram front end's rules for what the frame, the axes and their labels
  // need (see `plot_padding`). A framed plot hands the left and bottom room
  // to plotters as label areas, which draws the frame's tick labels in
  // them; unframed axes draw their own labels, so it is plain margin.
  let bottom_graphic_h =
    opts.frame_label_bottom_graphic.as_ref().map_or(0.0, |g| {
      scaled_graphic_label_size(g, render_width as f64 * 0.85, sf).1 / sf
    });
  let decorations = plot_padding::plot_decorations(
    opts,
    &x_ticks,
    &y_ticks,
    (x_scale, y_scale),
    origin,
    bottom_graphic_h,
  );
  let auto_padding = opts.image_padding.is_none().then(|| {
    let ar = opts.aspect_ratio.unwrap_or(DEFAULT_ASPECT_RATIO);
    let fixed_h = (opts.image_size_explicit.1 || opts.aspect_full)
      .then_some(svg_height as f64);
    decorations.solve(svg_width as f64, |w| match fixed_h {
      Some(h) => {
        (h - 16.0).min(if opts.aspect_full { f64::MAX } else { w * ar })
      }
      None => w * ar,
    })
  });
  let to_render = |v: f64| (v * sf).round().max(0.0) as u32;
  let pad = auto_padding.unwrap_or_default();
  let top_margin = to_render(pad.above()) as i32;
  let margin_right = to_render(pad.right);
  let (margin_left, y_label_area) = if opts.frame {
    (0, to_render(pad.left))
  } else {
    (to_render(pad.left), 0)
  };
  let (margin_bottom, x_label_area) = if opts.frame {
    (0, to_render(pad.bottom))
  } else {
    (to_render(pad.bottom), 0)
  };
  // With `ImagePadding` the padding *is* the margin, so an `AspectRatio`
  // fixes the canvas height directly: the plot area spans the width the
  // padding leaves, and the height follows from the ratio. Deriving it here,
  // before the margins are computed, keeps the frame as wide as the image
  // instead of shrinking it to fit a canvas sized by the default ratio.
  if let Some(ar) = opts.aspect_ratio
    && let Some([pad_left, pad_right, pad_bottom, pad_top]) = opts.image_padding
  {
    let plot_w = render_width as f64 - (pad_left + pad_right) * sf;
    if plot_w > 0.0 && ar > 0.0 {
      let target_render_h = plot_w * ar + (pad_bottom + pad_top) * sf;
      svg_height =
        ((target_render_h / sf).round() as u32).clamp(1, MAX_SVG_DIMENSION);
      render_height = svg_height * RESOLUTION_SCALE;
    }
  }

  // `ImagePadding` replaces the automatic margins: the left/bottom padding
  // becomes the tick label areas (the axis labels are drawn inside them) and
  // the right/top padding the corresponding margins.
  let (
    mut top_margin,
    mut x_label_area,
    mut y_label_area,
    mut margin_left,
    mut margin_right,
    mut margin_bottom,
  ) = match opts.image_padding {
    Some(padding) => {
      let m = padded_margins(
        padding,
        (render_width, render_height),
        opts.aspect_ratio,
        sf,
      );
      (
        m.top as i32
          + stacked_title_room(opts.plot_label.as_ref(), sf)
          + stacked_label_shortfall(
            opts.plot_label.as_ref(),
            usize::from(has_top_label) + usize::from(axes_label_y.is_some()),
            m.top as i32,
            sf,
          ),
        m.bottom,
        m.left,
        0u32,
        m.right,
        0u32,
      )
    }
    None => (
      top_margin,
      x_label_area,
      y_label_area,
      margin_left,
      margin_right,
      margin_bottom,
    ),
  };

  // The automatic margins above are fixed pixel sizes tuned for a
  // normal-sized plot; a small explicit `ImageSize` (an 80x80 panel in a
  // Manipulate's GraphicsGrid, say) can ask for less canvas than they add up
  // to, leaving zero or negative room for the plot area. `plotters` then
  // collapses every point and tick onto the same pixel instead of drawing
  // anything. Scale every margin on an axis down together so the drawing
  // area always keeps at least half the canvas along that axis. This only
  // applies to the automatic margins: an explicit `ImagePadding` is what the
  // user asked for and is left exactly as computed above, however little
  // room it leaves.
  let h_budget = margin_left as f64 + margin_right as f64 + y_label_area as f64;
  let max_h_budget = render_width as f64 * 0.5;
  if opts.image_padding.is_none() && h_budget > 0.0 && h_budget > max_h_budget {
    let f = max_h_budget / h_budget;
    margin_left = (margin_left as f64 * f).round() as u32;
    margin_right = (margin_right as f64 * f).round() as u32;
    y_label_area = (y_label_area as f64 * f).round() as u32;
  }
  let v_budget = top_margin as f64 + margin_bottom as f64 + x_label_area as f64;
  let max_v_budget = render_height as f64 * 0.5;
  if opts.image_padding.is_none()
    && opts.image_size_explicit.1
    && v_budget > 0.0
    && v_budget > max_v_budget
  {
    let f = max_v_budget / v_budget;
    top_margin = (top_margin as f64 * f).round() as i32;
    margin_bottom = (margin_bottom as f64 * f).round() as u32;
    x_label_area = (x_label_area as f64 * f).round() as u32;
  }

  // AspectRatio sizes the plotting *area* (the data frame), not the whole
  // image. Derive the total height so that
  //   plot_area_height = plot_area_width * aspect_ratio
  // where the plot area excludes the label/tick margins. Recomputing here
  // (rather than pre-setting svg_height from svg_width) keeps a short/wide
  // AspectRatio from collapsing the frame when fixed-size label margins would
  // otherwise consume most of the canvas.
  //
  // An explicit image height (`ImageSize -> {w, h}` or `{Automatic, h}`)
  // is kept instead: the plotting area keeps its `AspectRatio` (the default
  // 1/GoldenRatio when unset) inside the fixed image, centred along the
  // axis with room to spare, and a free width follows from the height.
  // `AspectRatio -> Full` stretches the area over the whole image.
  let h_margins =
    margin_left as f64 + margin_right as f64 + y_label_area as f64;
  let v_margins =
    top_margin as f64 + margin_bottom as f64 + x_label_area as f64;
  if opts.image_size_explicit.1
    && !opts.aspect_full
    && !full_width
    && opts.image_padding.is_none()
  {
    let ar = opts.aspect_ratio.unwrap_or(DEFAULT_ASPECT_RATIO);
    let avail_h = (render_height as f64 - v_margins).max(1.0);
    if opts.image_size_explicit.0 {
      let avail_w = (render_width as f64 - h_margins).max(1.0);
      if avail_w * ar <= avail_h {
        let slack = ((avail_h - avail_w * ar) / 2.0).round();
        top_margin += slack as i32;
        margin_bottom += slack as u32;
      } else {
        let slack = ((avail_w - avail_h / ar) / 2.0).round() as u32;
        margin_left += slack;
        margin_right += slack;
      }
    } else {
      let target_render_w = avail_h / ar + h_margins;
      svg_width =
        ((target_render_w / sf).round() as u32).clamp(1, MAX_SVG_DIMENSION);
      render_width = svg_width * RESOLUTION_SCALE;
    }
  } else if let Some(ar) = opts
    .aspect_ratio
    .or((!opts.aspect_full).then_some(DEFAULT_ASPECT_RATIO))
    && opts.image_padding.is_none()
  {
    let plot_w = (render_width as f64)
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    if plot_w > 0.0 {
      let plot_h = plot_w * ar;
      // Wolfram rounds the image height up to a whole point and leaves the
      // spare fraction below the picture.
      svg_height = (plot_padding::image_height(plot_h / sf, &pad) as u32)
        .clamp(1, MAX_SVG_DIMENSION);
      render_height = svg_height * RESOLUTION_SCALE;
      let bottom = (render_height as f64 - top_margin as f64 - plot_h)
        .round()
        .max(0.0) as u32;
      if opts.frame {
        x_label_area = bottom.saturating_sub(margin_bottom);
      } else {
        margin_bottom = bottom.saturating_sub(x_label_area);
      }
    }
  }

  let (theme_bg, theme_gray, _, label_fill, title_default_fill) = plot_theme();
  // `FrameStyle` and `FrameTicksStyle` paint each frame edge, its tick
  // marks and its tick labels.
  let frame_paints = frame_edge_paints(opts, theme_gray);
  let dark_gray = theme_gray;
  let label_weight = if opts.label_style.is_some_and(|s| s.bold) {
    FontStyle::Bold
  } else {
    FontStyle::Normal
  };
  // Background -> color replaces the theme background for the whole image.
  let bg_color = opts.background.unwrap_or(theme_bg);

  // Dashed series are collected here and emitted after the plot is drawn as a
  // single `<polyline stroke-dasharray>` each (rather than one element per
  // dash). Skipped for log axes, which keep the per-dash fallback because the
  // coordinate transform is non-linear.
  let mut dashed_overlays: Vec<DashedOverlay> = Vec::new();
  let mut cf_gradients: Vec<ColorFunctionGradient> = Vec::new();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    // Plotters sets a tick label twice the tick mark's length away from the
    // frame (the marks themselves are invisible: the frame draws its own).
    // Wolfram puts the y labels 3 pt left of the frame and the top of the
    // x labels 3 pt below it; plotters hangs the x labels' baseline 0.76 em
    // below their top, Arial's ascent 0.905 em.
    let frame_tick_font = plot_padding::plot_tick_font(opts);
    let x_tick_mark = ((plot_padding::TICK_LABEL_GAP
      + (plot_padding::ascent(frame_tick_font) - 0.76 * frame_tick_font))
      * sf
      / 2.0)
      .round() as i32;
    let y_tick_mark = (plot_padding::TICK_LABEL_GAP * sf / 2.0).round() as i32;

    // Macro to configure mesh and draw series on any chart coordinate type.
    // This avoids duplicating the drawing code for each LogCoord combination.
    macro_rules! draw_chart {
      ($chart:expr) => {{
        let mut chart = $chart;

        // Configure mesh: tick counts, sizes, label formatting, axis style.
        let x_labels_count;
        let y_labels_count;
        let x_tick_size;
        let y_tick_size;
        let x_major;
        let y_major;
        let date_axis = opts.date_axis;
        let log_x = opts.log_x;
        let log_y = opts.log_y;
        // An axis given explicit ticks draws them itself, below.
        let has_explicit_x_ticks = opts.ticks_x.is_some();
        let has_explicit_y_ticks = opts.ticks_y.is_some();
        if show_ticks && (tick_axis_x || tick_axis_y) {
          let xmaj = if date_axis {
            nice_date_step(x_max - x_min)
          } else {
            axis_tick_step(x_max - x_min)
          };
          let ymaj = axis_tick_step(y_max - y_min);
          x_major = xmaj;
          y_major = ymaj;
          let x_minor = if date_axis { xmaj } else { minor_tick_step(xmaj) };
          let y_minor = minor_tick_step(ymaj);
          x_labels_count = if !tick_axis_x
            || date_axis
            || has_explicit_x_ticks
            || !frame_edge_labeled[0]
          {
            0
          } else if log_x {
            // Let LogCoord decide tick placement; ~10 labels for log axes
            10
          } else {
            ((x_max - x_min) / x_minor).round() as usize + 1
          };
          y_labels_count = if !tick_axis_y
            || has_explicit_y_ticks
            || !frame_edge_labeled[1]
          {
            0
          } else if log_y {
            10
          } else {
            ((y_max - y_min) / y_minor).round() as usize + 1
          };
          x_tick_size = if tick_axis_x { x_tick_mark } else { 0 };
          y_tick_size = if tick_axis_y { y_tick_mark } else { 0 };
        } else {
          x_major = 1.0;
          y_major = 1.0;
          x_labels_count = 0;
          y_labels_count = 0;
          x_tick_size = 0;
          y_tick_size = 0;
        }
        let any_axis = show_x_axis || show_y_axis;
        let axis_style = if opts.frame {
          ShapeStyle::from(&bg_color).stroke_width(0)
        } else if any_axis {
          dark_gray.stroke_width(RESOLUTION_SCALE)
        } else {
          ShapeStyle::from(&bg_color).stroke_width(0)
        };
        // Unframed axes are drawn by `origin_axes_svg` once the chart is done.
        if !origin_axes {
          chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(x_labels_count)
            .y_labels(y_labels_count)
            .x_label_formatter(&move |v: &f64| {
              if x_labels_count == 0 {
                return String::new();
              }
              if date_axis {
                format_date_tick(*v)
              } else if log_x {
                // Suppress plotters labels; we inject custom SVG with superscripts
                String::new()
              } else if is_major_tick(*v, x_major) {
                format_tick_with_step(*v, x_major)
              } else {
                String::new()
              }
            })
            .y_label_formatter(&move |v: &f64| {
              if y_labels_count == 0 {
                return String::new();
              }
              if log_y {
                String::new()
              } else if is_major_tick(*v, y_major) {
                format_tick_with_step(*v, y_major)
              } else {
                String::new()
              }
            })
            .axis_style(axis_style)
            .x_label_style(
              (
                "sans-serif",
                // Plotters' SVG backend writes 1/1.24 of the size it is given.
                frame_tick_font * sf * 1.24,
                label_weight,
              )
                .into_font()
                .color(&frame_paints[0].label),
            )
            .y_label_style(
              ("sans-serif", frame_tick_font * sf * 1.24, label_weight)
                .into_font()
                .color(&frame_paints[1].label),
            )
            .set_tick_mark_size(LabelAreaPosition::Left, y_tick_size)
            .set_tick_mark_size(LabelAreaPosition::Bottom, x_tick_size)
            .draw()
            .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
        }

        // Draw grid lines before the series so they sit behind the data.
        // Explicit positions (with optional per-line color/thickness/dashing)
        // take precedence over the evenly spaced automatic lines. Solid lines
        // go through plotters; dashed lines are deferred to single
        // stroke-dasharray overlays (which render on top). (WL grid lines
        // are solid by default; dashing needs an explicit style.)
        // WL 15: `GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]]`.
        let grid_color = RGBColor(0x80, 0x80, 0x80).mix(0.4);
        // Automatic grid lines follow the ticks drawn on the axes.

        // Horizontal (y) grid lines.
        let y_grid = resolve_axis_grid_lines(
          &opts.grid_y_lines,
          opts.grid_lines_y,
          (y_min, y_max),
          log_y,
          axis_tick_step(y_max - y_min),
          opts.grid_lines_style.as_ref(),
        );
        {
          let tol = (y_max - y_min).abs() * 1e-9;
          // Automatic lines of a list plot keep off the ends of its
          // unpadded range (`GridLines -> Automatic` in WL 15).
          let interior = opts
            .grid_interior
            .filter(|_| opts.grid_y_lines.is_empty())
            .map(|r| r.1);
          for (gy, style) in &y_grid {
            if *gy < y_min - tol || *gy > y_max + tol {
              continue;
            }
            if let Some((a, b)) = interior {
              let eps = (b - a).abs() * 1e-9;
              if *gy <= a + eps || *gy >= b - eps {
                continue;
              }
            }
            let (rgb, stroke_w, dash) = grid_line_props(style);
            let alpha = style.opacity.unwrap_or(1.0);
            if let Some(dashes) = dash {
              dashed_overlays.push(DashedOverlay {
                color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
                opacity: alpha,
                stroke_w,
                dashes,
                points: vec![(x_min, *gy), (x_max, *gy)],
              });
            } else {
              let shape = match rgb {
                Some((r, g, b)) => RGBColor(r, g, b).mix(alpha).stroke_width(stroke_w),
                None => grid_color.stroke_width(stroke_w),
              };
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![(x_min, *gy), (x_max, *gy)],
                  shape,
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }
        }

        // Vertical (x) grid lines.
        let x_grid = resolve_axis_grid_lines(
          &opts.grid_x_lines,
          opts.grid_lines_x,
          (x_min, x_max),
          log_x,
          if date_axis {
            nice_date_step(x_max - x_min)
          } else {
            axis_tick_step(x_max - x_min)
          },
          opts.grid_lines_style.as_ref(),
        );
        {
          let tol = (x_max - x_min).abs() * 1e-9;
          // Automatic lines of a list plot keep off the ends of its
          // unpadded range (`GridLines -> Automatic` in WL 15).
          let interior = opts
            .grid_interior
            .filter(|_| opts.grid_x_lines.is_empty())
            .map(|r| r.0);
          for (gx, style) in &x_grid {
            if *gx < x_min - tol || *gx > x_max + tol {
              continue;
            }
            if let Some((a, b)) = interior {
              let eps = (b - a).abs() * 1e-9;
              if *gx <= a + eps || *gx >= b - eps {
                continue;
              }
            }
            let (rgb, stroke_w, dash) = grid_line_props(style);
            let alpha = style.opacity.unwrap_or(1.0);
            if let Some(dashes) = dash {
              dashed_overlays.push(DashedOverlay {
                color: rgb.unwrap_or(DEFAULT_GRID_DASH_RGB),
                opacity: alpha,
                stroke_w,
                dashes,
                points: vec![(*gx, y_min), (*gx, y_max)],
              });
            } else {
              let shape = match rgb {
                Some((r, g, b)) => RGBColor(r, g, b).mix(alpha).stroke_width(stroke_w),
                None => grid_color.stroke_width(stroke_w),
              };
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![(*gx, y_min), (*gx, y_max)],
                  shape,
                )))
                .map_err(|e| {
                  InterpreterError::EvaluationError(format!("Plot: {e}"))
                })?;
            }
          }
        }

        // WL 15 keeps the axes of a framed plot: a hairline (0.2 pt) through
        // the origin, drawn where the origin lies strictly inside the
        // unpadded plot range.
        if opts.frame && !log_x && !log_y {
          let (ux, uy) = opts
            .unpadded_range
            .or(opts.grid_interior)
            .unwrap_or(((x_min, x_max), (y_min, y_max)));
          let inside = |v: f64, (a, b): (f64, f64)| {
            let eps = (b - a).abs() * 1e-9;
            v > a.min(b) + eps && v < a.max(b) - eps
          };
          let axis_style = dark_gray.stroke_width(2);
          let ox = opts.axes_origin.0.unwrap_or(0.0);
          let oy = opts.axes_origin.1.unwrap_or(0.0);
          let mut lines = Vec::new();
          if show_x_axis && inside(oy, uy) {
            lines.push(vec![(x_min, oy), (x_max, oy)]);
          }
          if show_y_axis && inside(ox, ux) {
            lines.push(vec![(ox, y_min), (ox, y_max)]);
          }
          for line in lines {
            chart
              .draw_series(std::iter::once(PathElement::new(line, axis_style)))
              .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
          }
        }

        // Draw the frame's borders (all four unless `Frame` names edges),
        // each in its own `FrameStyle`.
        if opts.frame {
          let style_of = |p: &FrameEdgePaint| p.line.stroke_width(p.line_width);
          let edges = opts.frame_edges.without(opts.frame_hidden_edges);
          let uniform = frame_paints
            .iter()
            .all(|p| (p.line, p.line_width) == (frame_paints[0].line, frame_paints[0].line_width));
          let closed = edges == FrameEdges::ALL && uniform;
          if closed {
            chart
              .draw_series(std::iter::once(PathElement::new(
                vec![
                  (x_min, y_min),
                  (x_max, y_min),
                  (x_max, y_max),
                  (x_min, y_max),
                  (x_min, y_min),
                ],
                style_of(&frame_paints[0]),
              )))
              .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
          }
          for (on, paint, from, to) in [
            (edges.bottom, &frame_paints[0], (x_min, y_min), (x_max, y_min)),
            (edges.right, &frame_paints[3], (x_max, y_min), (x_max, y_max)),
            (edges.top, &frame_paints[2], (x_max, y_max), (x_min, y_max)),
            (edges.left, &frame_paints[1], (x_min, y_max), (x_min, y_min)),
          ] {
            if on && !closed {
              chart
                .draw_series(std::iter::once(PathElement::new(
                  vec![from, to],
                  style_of(paint),
                )))
                .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
            }
          }
        }

        // Fills lie beneath every curve, whichever series they belong to:
        // a `Filling -> {2 -> Bottom}` area must not paint over the lines
        // of the series drawn before it.
        if !opts.stacked {
          for (series_idx, points) in all_points.iter().enumerate() {
            if !resolve_series_joined(opts.joined_per_series.as_deref(), series_idx) {
              continue;
            }
            let (r, g, b) = series_color(&opts.plot_style, series_idx);
            let paint =
              fill_paint(series_filling_style(opts, series_idx), (r, g, b));
            match series_fill_target(opts, series_idx) {
              FillTarget::Level(level) => {
                // Fills stop at the plot range, not in its padding.
                let (fill_min, fill_max) =
                  opts.curve_clip_y.unwrap_or((y_min, y_max));
                if let Some(ref_y) = level.reference_y(fill_min, fill_max) {
                  let fill_segments = clip_segments_to_x_range(
                    clamp_segments_to_y_range(points, fill_min, fill_max),
                    x_min,
                    x_max,
                  );
                  for segment in &fill_segments {
                    if segment.len() < 2 {
                      continue;
                    }
                    // With a `ColorFunction` and no explicit `FillingStyle`
                    // (unset or `Automatic`), the fill takes the curve's
                    // colors run by run at full strength (WL paints the
                    // fill polygons with opaque `VertexColors`). Any given
                    // `FillingStyle`, even a bare `Opacity[…]`, replaces
                    // the color function with the plain series color.
                    let tinted = opts
                      .color_function
                      .as_ref()
                      .filter(|_| series_filling_style(opts, series_idx).is_none());
                    let pieces: Vec<(plotters::style::RGBAColor, Vec<(f64, f64)>)> =
                      match tinted {
                        Some(cf) if let Some(stops) = color_function_gradient_stops(
                          segment,
                          cf,
                          opts.color_function_scaling,
                          color_scale_ranges,
                          opts.log_x,
                        ) => {
                          let sentinel =
                            color_function_gradient_sentinel(cf_gradients.len());
                          cf_gradients.push(ColorFunctionGradient { sentinel, stops });
                          let (sr, sg, sb) = sentinel;
                          vec![(RGBColor(sr, sg, sb).mix(1.0), segment.clone())]
                        }
                        Some(cf) => {
                          let runs = color_function_runs(
                            segment,
                            cf,
                            opts.color_function_scaling,
                            color_scale_ranges,
                          );
                          // Each opaque piece reaches one step into the
                          // next, which paints over it: abutting pieces
                          // would leave anti-aliasing seams between them.
                          let overlaps: Vec<Option<(f64, f64)>> = runs
                            .iter()
                            .skip(1)
                            .map(|(_, next)| next.get(1).copied())
                            .chain(std::iter::once(None))
                            .collect();
                          runs
                            .into_iter()
                            .zip(overlaps)
                            .map(|(((cr, cg, cb), mut run), extra)| {
                              run.extend(extra);
                              (RGBColor(cr, cg, cb).mix(1.0), run)
                            })
                            .collect()
                        }
                        None => vec![(paint, segment.clone())],
                      };
                    // `FillingStyle -> {below, above}`: split the area at
                    // the reference level and paint each side in its color.
                    let below_paint = series_filling_style(opts, series_idx)
                      .filter(|_| tinted.is_none())
                      .and_then(|fs| fs.below_color.map(|c| (fs, c)))
                      .map(|(fs, (br, bg, bb))| {
                        RGBColor(br, bg, bb)
                          .mix(fs.effective_opacity().unwrap_or(0.2))
                      });
                    if let Some(bp) = below_paint {
                      let (xa, xb) = (segment[0].0, segment[segment.len() - 1].0);
                      let level = [(xa.min(xb), ref_y), (xa.max(xb), ref_y)];
                      for (is_above, piece) in
                        fill_between_polygons_by_side(segment, &level)
                      {
                        chart
                          .draw_series(std::iter::once(Polygon::new(
                            piece,
                            if is_above { paint } else { bp },
                          )))
                          .map_err(|e| {
                            InterpreterError::EvaluationError(format!(
                              "Plot: {e}"
                            ))
                          })?;
                      }
                      continue;
                    }
                    for (piece_paint, piece) in pieces {
                      if piece.len() < 2 {
                        continue;
                      }
                      chart
                        .draw_series(AreaSeries::new(
                          piece.into_iter(),
                          ref_y,
                          piece_paint,
                        ))
                        .map_err(|e| {
                          InterpreterError::EvaluationError(format!(
                            "Plot: {e}"
                          ))
                        })?;
                    }
                  }
                }
              }
              // `Filling -> {i -> {j}}`: fill the region between this
              // series and series j over the overlap of their x-domains.
              FillTarget::Series(target_idx) => {
                if target_idx != series_idx
                  && let Some(target) = all_points.get(target_idx)
                  && let Some(polygon) = fill_between_polygon(points, target)
                {
                  let below_paint = series_filling_style(opts, series_idx)
                    .and_then(|fs| fs.below_color.map(|c| (fs, c)))
                    .map(|(fs, (br, bg, bb))| {
                      RGBColor(br, bg, bb).mix(fs.effective_opacity().unwrap_or(0.2))
                    });
                  let pieces = match below_paint {
                    Some(_) => fill_between_polygons_by_side(points, target),
                    None => vec![(true, polygon)],
                  };
                  for (is_above, piece) in pieces {
                    let piece_paint = match (is_above, below_paint) {
                      (false, Some(bp)) => bp,
                      _ => paint,
                    };
                    chart
                      .draw_series(std::iter::once(Polygon::new(
                        piece,
                        piece_paint,
                      )))
                      .map_err(|e| {
                        InterpreterError::EvaluationError(format!(
                          "Plot: {e}"
                        ))
                      })?;
                  }
                }
              }
            }
          }
        }

        // The mesh values, from the range the mesh functions take over all
        // the drawn (clipped) curves.
        let mesh_levels = crate::functions::plot_mesh::mesh_levels(
          &all_points
            .iter()
            .map(|points| {
              let (clip_min, clip_max) =
                opts.curve_clip_y.unwrap_or((y_min, y_max));
              clip_segments_to_x_range(
                clip_segments_to_y_range(
                  split_into_segments(points),
                  clip_min,
                  clip_max,
                  opts.join_across_jumps,
                ),
                x_min,
                x_max,
              )
            })
            .collect::<Vec<_>>(),
          &opts.mesh,
          &opts.mesh_functions,
          opts.mesh_by_parameter,
        );
        // (position, radius, colour, opacity) of every curve's mesh points.
        let mut pending_mesh_dots: Vec<((f64, f64), u32, (u8, u8, u8), f64)> =
          Vec::new();
        for (series_idx, points) in all_points.iter().enumerate() {
          let (r, g, b) = series_color(&opts.plot_style, series_idx);
          let color = RGBColor(r, g, b);
          let stroke_w = series_thickness(&opts.plot_style, series_idx);
          let dashing = series_dashing(&opts.plot_style, series_idx);
          let (clip_min, clip_max) = opts.curve_clip_y.unwrap_or((y_min, y_max));
          let segments = clip_segments_to_x_range(
            clip_segments_to_y_range(
              split_into_segments(points),
              clip_min,
              clip_max,
              opts.join_across_jumps,
            ),
            x_min,
            x_max,
          );

          // `ClippingStyle`: where the curve is cut off, a line along the
          // edge of the plot range stands in for it.
          if opts.clipping_style.iter().any(Option::is_some)
            && resolve_series_joined(opts.joined_per_series.as_deref(), series_idx)
          {
            for segment in split_into_segments(points) {
              for (top, from, to) in clipped_edge_runs(&segment, clip_min, clip_max) {
                let Some(edge) = &opts.clipping_style[usize::from(top)] else {
                  continue;
                };
                let (from, to) = (from.clamp(x_min, x_max), to.clamp(x_min, x_max));
                if to <= from {
                  continue;
                }
                let level = if top { clip_max } else { clip_min };
                let (er, eg, eb) = edge
                  .style
                  .color
                  .as_ref()
                  .map(|c| {
                    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                    (ch(c.r), ch(c.g), ch(c.b))
                  })
                  .unwrap_or((r, g, b));
                let width = edge
                  .style
                  .thickness
                  .map(|t| (t * RESOLUTION_SCALE as f64).round() as u32)
                  .unwrap_or(stroke_w);
                // WL dots the line 2.5 pt on, 4.5 pt off with flat ends; the
                // overlays have round caps, which add the line width to
                // every dash.
                let dashes = if edge.dotted {
                  let w = width as f64 / RESOLUTION_SCALE as f64;
                  let on = (2.5 - w).max(0.5);
                  Some(vec![-on, -(7.0 - on)])
                } else {
                  edge.style.dashing.clone()
                };
                let opacity = edge
                  .style
                  .opacity
                  .unwrap_or_else(|| series_opacity(&opts.plot_style, series_idx));
                let line = vec![(from, level), (to, level)];
                match dashes {
                  Some(dashes) => dashed_overlays.push(DashedOverlay {
                    color: (er, eg, eb),
                    opacity,
                    stroke_w: width,
                    dashes,
                    points: line,
                  }),
                  None => {
                    chart
                      .draw_series(LineSeries::new(
                        line,
                        RGBColor(er, eg, eb).mix(opacity).stroke_width(width),
                      ))
                      .map_err(|e| {
                        InterpreterError::EvaluationError(format!("Plot: {e}"))
                      })?;
                  }
                }
              }
            }
          }

          // Uncertainty bands lie underneath the curve and any filling.
          // Like the bars, they are parallel to the original data points,
          // which may differ from the drawn curve (InterpolationOrder).
          if opts.interval_markers == IntervalMarkers::Bands
            && let Some(bars) = opts.error_bars.get(series_idx)
          {
            let anchor = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            draw_interval_band(&mut chart, anchor, bars, color)?;
          }

          // `ListPlot[…, Joined -> {..., False, ...}]`: this series stays a
          // scatter of points instead of a connected curve, while sibling
          // series in the same chart still join.
          let series_joined =
            resolve_series_joined(opts.joined_per_series.as_deref(), series_idx);
          // The mesh points (and `MeshShading` stretches) of the drawn,
          // clipped curve: a curve leaving the plot range ends there.
          let curve_mesh = mesh_levels.as_ref().filter(|_| series_joined).map(|levels| {
            crate::functions::plot_mesh::curve_mesh(
              &segments,
              levels,
              &opts.mesh_functions,
              opts.mesh_shading.as_ref(),
              opts.mesh_by_parameter,
              opts.mesh_open_ends,
            )
          });

          if !series_joined {
            // Capped error bars sit under the points of a scatter series.
            if matches!(
              opts.interval_markers,
              IntervalMarkers::Fences | IntervalMarkers::Bars
            ) && let Some(bars) = opts.error_bars.get(series_idx)
            {
              draw_error_bars(
                &mut chart,
                points,
                bars,
                opts.interval_markers_color.unwrap_or(color),
                opts.interval_markers == IntervalMarkers::Fences,
              )?;
            }
            let marker_size =
              series_point_radius(opts, series_idx, render_width);
            draw_scatter_series(
              &mut chart,
              all_points,
              series_idx,
              opts,
              (r, g, b),
              marker_size,
              (x_min, x_max),
              y_min,
              y_max,
            )?;
          } else {
            // Draw filled area before the line so the line renders on top.
            // In stacked mode each band is bounded below by the previous
            // cumulative curve (or the axis for the first series) and above by
            // the current one; the bands are disjoint so an opaque polygon fill
            // renders cleanly. Otherwise fill each segment down to the constant
            // reference level given by the Filling option.
            if opts.stacked {
              let baseline: Vec<(f64, f64)> = if series_idx == 0 {
                points.iter().map(|&(x, _)| (x, 0.0)).collect()
              } else {
                all_points[series_idx - 1].clone()
              };
              let mut polygon: Vec<(f64, f64)> = points.to_vec();
              polygon.extend(baseline.iter().rev().copied());
              if polygon.len() >= 3 {
                chart
                  .draw_series(std::iter::once(Polygon::new(
                    polygon,
                    RGBColor(r, g, b).mix(0.6),
                  )))
                  .map_err(|e| {
                    InterpreterError::EvaluationError(format!("Plot: {e}"))
                  })?;
              }
            }

            // `MeshShading` cuts the curve at its mesh points into
            // stretches styled in turn; `Automatic` stretches keep the
            // curve's own style (and `ColorFunction`).
            let shaded_runs: Vec<(Option<&SeriesStyle>, &[Vec<(f64, f64)>])> =
              match curve_mesh.as_ref().and_then(|m| m.pieces.as_ref()) {
                Some(pieces) => pieces
                  .iter()
                  .map(|(shade, piece)| {
                    let style = match shade {
                      crate::functions::plot_mesh::MeshShade::Style(st) => Some(st.as_ref()),
                      _ => None,
                    };
                    (style, std::slice::from_ref(piece))
                  })
                  .collect(),
                None => vec![(None, segments.as_slice())],
              };
            let base_color = color;
            let base_stroke_w = stroke_w;
            let base_dashing = dashing.clone();
            for &(shade, segments) in &shaded_runs {
              let shade_color = shade.and_then(|st| st.color).map(color_to_rgb8);
              let (r, g, b) = shade_color.unwrap_or((r, g, b));
              let color = shade_color
                .map(|(r, g, b)| RGBColor(r, g, b))
                .unwrap_or(base_color);
              let stroke_w = match shade {
                Some(st) if st.thickness.is_some() => {
                  series_thickness(std::slice::from_ref(st), 0)
                }
                _ => base_stroke_w,
              };
              let dashing = shade
                .and_then(|st| st.dashing.clone())
                .or_else(|| base_dashing.clone());
              let line_opacity = shade
                .and_then(|st| st.opacity)
                .unwrap_or_else(|| series_opacity(&opts.plot_style, series_idx));
              let color_function =
                opts.color_function.as_ref().filter(|_| shade_color.is_none());
              if let Some(ref dash_pattern) = dashing {
                // Defer to a single <polyline stroke-dasharray> per segment,
                // emitted after the plot is drawn.
                for segment in segments {
                  if segment.len() >= 2 {
                    dashed_overlays.push(DashedOverlay {
                      color: (r, g, b),
                      opacity: line_opacity,
                      stroke_w,
                      dashes: dash_pattern.clone(),
                      points: segment.clone(),
                    });
                  }
                }
              } else if let Some(cf) = color_function {
                for segment in segments {
                  if let Some(stops) = color_function_gradient_stops(
                    segment,
                    cf,
                    opts.color_function_scaling,
                    color_scale_ranges,
                    opts.log_x,
                  ) {
                    let sentinel =
                      color_function_gradient_sentinel(cf_gradients.len());
                    cf_gradients.push(ColorFunctionGradient { sentinel, stops });
                    let (sr, sg, sb) = sentinel;
                    chart
                      .draw_series(LineSeries::new(
                        segment.iter().copied(),
                        RGBColor(sr, sg, sb).stroke_width(stroke_w),
                      ))
                      .map_err(|e| {
                        InterpreterError::EvaluationError(format!("Plot: {e}"))
                      })?;
                    continue;
                  }
                  for (run_color, run) in color_function_runs(
                    segment,
                    cf,
                    opts.color_function_scaling,
                    color_scale_ranges,
                  ) {
                    chart
                      .draw_series(LineSeries::new(
                        run,
                        RGBColor(run_color.0, run_color.1, run_color.2)
                          .stroke_width(stroke_w),
                      ))
                      .map_err(|e| {
                        InterpreterError::EvaluationError(format!("Plot: {e}"))
                      })?;
                  }
                }
              } else {
                let line_style = color
                  .mix(line_opacity)
                  .stroke_width(stroke_w);
                for segment in segments {
                  chart
                    .draw_series(LineSeries::new(segment.iter().copied(), line_style))
                    .map_err(|e| {
                      InterpreterError::EvaluationError(format!("Plot: {e}"))
                    })?;
                }
              }
            }
          }

          // Error bars from Around data values, on top of the line. They
          // are parallel to the original data points, which may differ from
          // the drawn curve (InterpolationOrder).
          if series_joined
            && matches!(
              opts.interval_markers,
              IntervalMarkers::Fences | IntervalMarkers::Bars
            )
            && let Some(bars) = opts.error_bars.get(series_idx)
          {
            let anchor = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            draw_error_bars(
              &mut chart,
              anchor,
              bars,
              opts.interval_markers_color.unwrap_or(color),
              opts.interval_markers == IntervalMarkers::Fences,
            )?;
          }

          // Mesh points: every curve point for `Mesh -> All`, the regular
          // points for `Mesh -> Full`, else where the mesh functions take
          // their mesh values. They are drawn over every curve.
          if series_joined && !opts.mesh.is_none() {
            let in_view = |&(x, y): &(f64, f64)| {
              marker_in_view(x, y, (x_min, x_max), (y_min, y_max))
            };
            let dots: Vec<crate::functions::plot_mesh::MeshDot> = match &opts.mesh {
              Mesh::All | Mesh::Full => {
                let src: &[(f64, f64)] = match opts.mesh {
                  Mesh::Full => opts
                    .mesh_full_points
                    .get(series_idx)
                    .or_else(|| opts.data_points.get(series_idx))
                    .map(|v| v.as_slice())
                    .unwrap_or(points),
                  _ => points,
                };
                src
                  .iter()
                  .filter(|p| in_view(p))
                  .map(|&pos| crate::functions::plot_mesh::MeshDot {
                    pos,
                    func: 0,
                    style: None,
                  })
                  .collect()
              }
              _ => curve_mesh
                .as_ref()
                .map(|m| m.dots.iter().filter(|d| in_view(&d.pos)).cloned().collect())
                .unwrap_or_default(),
            };
            let series_style = opts
              .plot_style
              .get(series_idx % opts.plot_style.len().max(1));
            for dot in dots {
              let mesh_style = opts
                .mesh_styles
                .get(dot.func % opts.mesh_styles.len().max(1))
                .and_then(|s| s.as_ref());
              let layers = [dot.style.as_ref(), mesh_style];
              let pick = |f: &dyn Fn(&SeriesStyle) -> Option<f64>| {
                layers.iter().flatten().find_map(|s| f(s))
              };
              // The curve's colour (its `ColorFunction` colour there),
              // unless `MeshStyle` or the mesh value's style sets one.
              let rgb = layers
                .iter()
                .flatten()
                .find_map(|s| s.color)
                .map(color_to_rgb8)
                .or_else(|| {
                  opts.color_function.as_ref().and_then(|cf| {
                    color_function_rgb(
                      cf,
                      opts.color_function_scaling,
                      dot.pos,
                      color_scale_ranges,
                    )
                  })
                })
                .unwrap_or((r, g, b));
              let opacity = pick(&|s| s.opacity)
                .unwrap_or_else(|| series_opacity(&opts.plot_style, series_idx));
              // WL 15 draws a curve's mesh points 3 pt across, a list
              // plot's at its automatic `PointSize`.
              let size = pick(&|s| s.point_size)
                .or(series_style.and_then(|s| s.point_size))
                .or(opts.mesh_point_size)
                .or(opts.default_point_size);
              let radius = match size {
                Some(f) if f > 0.0 => {
                  (f * render_width as f64 * 0.5).round() as u32
                }
                Some(p) => (-p * 0.5 * RESOLUTION_SCALE as f64).round() as u32,
                None => 15 * RESOLUTION_SCALE / 10,
              };
              pending_mesh_dots.push((dot.pos, radius, rgb, opacity));
            }
          }

          // `PlotMarkers` glyphs sit on the data points of a joined plot,
          // over the line (a scatter series drew them as its points).
          if series_joined
            && let Some(marker) = series_marker(&opts.plot_markers, series_idx)
          {
            let marker_src: &[(f64, f64)] = opts
              .data_points
              .get(series_idx)
              .map(|v| v.as_slice())
              .unwrap_or(points);
            let c = color.to_rgba();
            let in_view: Vec<(f64, f64)> = marker_src
              .iter()
              .copied()
              .filter(|&(x, y)| {
                marker_in_view(x, y, (x_min, x_max), (y_min, y_max))
              })
              .collect();
            draw_marker_glyphs(&mut chart, &in_view, marker, (c.0, c.1, c.2))?;
          }
        }

        // The mesh points of every curve lie over all the curves, as in
        // Wolfram's graphics (one point group after the lines).
        chart
          .draw_series(pending_mesh_dots.iter().map(|&(pos, radius, (r, g, b), a)| {
            Circle::new(pos, radius, RGBColor(r, g, b).mix(a).filled())
          }))
          .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

        root
          .present()
          .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
      }};
    }

    // Build chart with appropriate coordinate types for log/linear axes.
    // LogCoord handles logarithmic tick placement, scaling, and labeling.
    // Each arm creates its own ChartBuilder because it borrows `root`.
    macro_rules! chart_builder {
      () => {
        ChartBuilder::on(&root)
          .margin_top(top_margin as u32)
          .margin_right(margin_right)
          .margin_bottom(margin_bottom)
          .margin_left(margin_left)
          .x_label_area_size(x_label_area)
          .y_label_area_size(y_label_area)
      };
    }
    let err = |e| InterpreterError::EvaluationError(format!("Plot: {e}"));
    // The frame's tick marks and labels sit on every multiple of the minor
    // step inside the range. Plotters' own key-point search accumulates
    // rounding error and drops the last one (`Plot[Sin[x], …, Frame ->
    // True]` lost its `1.0` label), so the points are handed over
    // explicitly.
    let frame_key_points = |lo: f64, hi: f64, labelled: bool| -> Vec<f64> {
      if !labelled || hi <= lo || !(hi - lo).is_finite() {
        return Vec::new();
      }
      let minor = minor_tick_step(axis_tick_step(hi - lo));
      let first = (lo / minor - 1e-9).ceil() as i64;
      let last = (hi / minor + 1e-9).floor() as i64;
      (first..=last)
        .map(|k| {
          let v = k as f64 * minor;
          if k == 0 { 0.0 } else { v }
        })
        .collect()
    };
    let frame_ticks = show_ticks && !origin_axes;
    let x_points = frame_key_points(
      x_min,
      x_max,
      frame_ticks && tick_axis_x && !opts.date_axis && opts.ticks_x.is_none(),
    );
    let y_points = frame_key_points(
      y_min,
      y_max,
      frame_ticks && tick_axis_y && opts.ticks_y.is_none(),
    );
    match (opts.log_x, opts.log_y) {
      (false, false) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(
            FixedKeyPoints::new(x_min..x_max, x_points),
            FixedKeyPoints::new(y_min..y_max, y_points)
          )
          .map_err(err)?
      ),
      (false, true) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(x_min..x_max, (y_min..y_max).log_scale())
          .map_err(err)?
      ),
      (true, false) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d((x_min..x_max).log_scale(), y_min..y_max)
          .map_err(err)?
      ),
      (true, true) => draw_chart!(
        chart_builder!()
          .build_cartesian_2d(
            (x_min..x_max).log_scale(),
            (y_min..y_max).log_scale()
          )
          .map_err(err)?
      ),
    }
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  inject_drop_shadows(&mut buf, &opts.plot_style, all_points.len());
  apply_color_function_gradients(&mut buf, &cf_gradients);

  // Emit deferred dashed series as single stroke-dasharray polylines (on top
  // of the plot content, just before </svg>).
  if !dashed_overlays.is_empty() {
    let plot_x0 = margin_left as f64 + y_label_area as f64;
    let plot_y0 = top_margin as f64;
    let plot_w = render_width as f64
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    let plot_h = render_height as f64
      - top_margin as f64
      - margin_bottom as f64
      - x_label_area as f64;
    let series_svg = render_dash_overlays(
      &dashed_overlays,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      render_width as f64,
      x_min,
      x_max,
      y_min,
      y_max,
      opts.log_x,
      opts.log_y,
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &series_svg);
    }
  }

  if !opts.exclusion_marks.is_empty()
    && opts.exclusions_style.iter().any(Option::is_some)
    && !opts.log_x
    && !opts.log_y
  {
    let plot_x0 = margin_left as f64 + y_label_area as f64;
    let plot_y0 = top_margin as f64;
    let plot_w = render_width as f64
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    let plot_h = render_height as f64
      - top_margin as f64
      - margin_bottom as f64
      - x_label_area as f64;
    let marks_svg = render_exclusion_marks(
      opts,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      render_width as f64,
      (x_min, x_max),
      (y_min, y_max),
      opts.curve_clip_y.unwrap_or((y_min, y_max)),
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &marks_svg);
    }
  }

  // A frame carries its tick marks on all four edges, pointing inward; the
  // majors line up with the labels the chart mesh drew. Log and date axes
  // place their own automatic ticks; an edge given explicit ticks (by
  // `FrameTicks`, or the bottom and left ones by `Ticks`) marks and labels
  // only the positions it names.
  if opts.frame && show_ticks {
    let area = (
      margin_left as f64 + y_label_area as f64,
      top_margin as f64,
      render_width as f64
        - margin_left as f64
        - margin_right as f64
        - y_label_area as f64,
      render_height as f64
        - top_margin as f64
        - margin_bottom as f64
        - x_label_area as f64,
    );
    let mut edges = opts.frame_tick_edges.clone();
    for (edge, explicit) in [(0, &opts.ticks_x), (1, &opts.ticks_y)] {
      if let (FrameEdgeTicks::Labeled, Some(ticks)) = (&edges[edge], explicit) {
        edges[edge] = FrameEdgeTicks::Explicit(
          ticks
            .iter()
            .map(|(pos, label)| ExplicitTick {
              pos: *pos,
              label: label.clone(),
              len: None,
              style: None,
            })
            .collect(),
        );
      }
    }
    let auto_x = !opts.log_x && !opts.date_axis;
    let auto_y = !opts.log_y;
    let far_labels = |scale: AxisScale, date: bool| {
      explicit_ticks_in_axis_space(
        Some(&AxisTicks::new(scale, date, None).majors),
        scale.log,
      )
      .unwrap_or_default()
    };
    let top_labels = far_labels(x_scale, opts.date_axis);
    let right_labels = far_labels(y_scale, false);
    let ticks_svg = frame_ticks_svg(&FrameTicksLayout {
      area,
      ranges: log_overlay_ranges(opts, (x_min, x_max, y_min, y_max)),
      log: (opts.log_x, opts.log_y),
      auto: [
        auto_x.then(|| (x_min, x_max, axis_tick_step(x_max - x_min))),
        auto_y.then(|| (y_min, y_max, axis_tick_step(y_max - y_min))),
      ],
      auto_labels: [&top_labels, &right_labels],
      edges: &edges,
      present: opts.frame_edges,
      marked: opts.frame_edges.without(opts.frame_hidden_edges),
      paints: &frame_paints,
      sf,
      font: plot_padding::plot_tick_font(opts),
      bold: opts.label_style.is_some_and(|s| s.bold),
    });
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &ticks_svg);
    }
  }

  if origin_axes {
    let area = (
      margin_left as f64 + y_label_area as f64,
      top_margin as f64,
      render_width as f64
        - margin_left as f64
        - margin_right as f64
        - y_label_area as f64,
      render_height as f64
        - top_margin as f64
        - margin_bottom as f64
        - x_label_area as f64,
    );
    let label_style_fill = opts
      .label_style
      .and_then(|s| s.color)
      .map(super::graphics::Color::to_svg_rgb);
    let axes_svg = origin_axes_svg(&OriginAxes {
      area,
      x: x_scale,
      y: y_scale,
      origin,
      show: (show_x_axis, show_y_axis),
      ticks: show_ticks.then_some((&x_ticks, &y_ticks)),
      sf,
      axis_color: label_fill,
      label_fill,
      axes_style: &opts.axes_style,
      ticks_style: &opts.ticks_style,
      keep_y_label_on_x_axis: false,
      label_color: label_style_fill.as_deref(),
    });
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &axes_svg);
    }
  }

  // Draw Prolog primitives under the plotted data, and Epilog primitives
  // over it, using the same data→pixel transform as the dash overlays
  // above.
  let plot_area = (
    margin_left as f64 + y_label_area as f64,
    top_margin as f64,
    render_width as f64
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64,
    render_height as f64
      - top_margin as f64
      - margin_bottom as f64
      - x_label_area as f64,
  );
  inject_prolog(&mut buf, opts, plot_area, (x_min, x_max, y_min, y_max), sf);
  inject_epilog(&mut buf, opts, plot_area, (x_min, x_max, y_min, y_max), sf);

  // Inject label SVG elements before </svg>
  if has_plot_label
    || has_x_axis_label
    || has_y_axis_label
    || axes_label_x.is_some()
    || axes_label_y.is_some()
    || has_top_label
    || has_right_label
    || opts.ticks_x.is_some()
    || opts.ticks_y.is_some()
  {
    let margin_left_f = margin_left as f64;
    let margin_right_f = margin_right as f64;
    let margin_bottom_f = margin_bottom as f64;
    let margin_top = top_margin as f64;
    let plot_x0 = margin_left_f + y_label_area as f64;
    let plot_w = render_width as f64
      - margin_left_f
      - margin_right_f
      - y_label_area as f64;
    let plot_h =
      render_height as f64 - margin_top - margin_bottom_f - x_label_area as f64;

    if let Some(insert_pos) = buf.rfind("</svg>") {
      // The origin axes and the frame already carry any explicit ticks.
      let unframed = !origin_axes && !opts.frame;
      let mut labels_svg = explicit_ticks_svg(
        explicit_ticks_in_axis_space(
          opts.ticks_x.as_ref().filter(|_| unframed),
          opts.log_x,
        )
        .as_ref(),
        explicit_ticks_in_axis_space(
          opts.ticks_y.as_ref().filter(|_| unframed),
          opts.log_y,
        )
        .as_ref(),
        (plot_x0, margin_top, plot_w, plot_h),
        log_overlay_ranges(opts, (x_min, x_max, y_min, y_max)),
        sf,
        label_fill,
        opts.frame_hidden_edges,
        crate::functions::plot_padding::plot_tick_font(opts),
      );

      let top_pad = match opts.image_padding {
        Some([_, _, _, top]) => top,
        None => pad.top,
      };
      let layout = plot_padding::LabelLayout {
        tick_room: decorations.tick_rooms(plot_w / sf),
        title_top: (margin_top - (top_pad + decorations.title) * sf).max(0.0),
      };
      labels_svg.push_str(&plot_labels_svg(
        opts,
        (plot_x0, margin_top, plot_w, plot_h),
        (x_min, x_max, y_min, y_max),
        &layout,
        sf,
        label_fill,
        title_default_fill,
        render_width as f64,
      ));

      buf.insert_str(insert_pos, &labels_svg);
    }
  }

  // Label the log axes of a frame with the same `Charting`ScaledTicks`
  // majors the unframed log axes carry (the origin axes label their own);
  // the frame mesh draws the tick marks.
  if !origin_axes
    && ((opts.log_y && tick_axis_y && show_ticks && opts.ticks_y.is_none())
      || (opts.log_x && tick_axis_x && show_ticks && opts.ticks_x.is_none()))
  {
    let plot_x0 = margin_left as f64 + y_label_area as f64;
    let plot_w = render_width as f64
      - margin_left as f64
      - margin_right as f64
      - y_label_area as f64;
    let plot_h = render_height as f64
      - top_margin as f64
      - margin_bottom as f64
      - x_label_area as f64;
    let log_majors = |log: bool, given: bool, lo: f64, hi: f64| {
      (log && !given)
        .then(|| crate::functions::plot_scaling::log_axis_ticks(lo, hi).majors)
    };
    let ticks_x = log_majors(
      opts.log_x && tick_axis_x,
      opts.ticks_x.is_some(),
      x_min,
      x_max,
    );
    let ticks_y = log_majors(
      opts.log_y && tick_axis_y,
      opts.ticks_y.is_some(),
      y_min,
      y_max,
    );
    let labels = explicit_ticks_svg(
      explicit_ticks_in_axis_space(ticks_x.as_ref(), true).as_ref(),
      explicit_ticks_in_axis_space(ticks_y.as_ref(), true).as_ref(),
      (plot_x0, top_margin as f64, plot_w, plot_h),
      log_overlay_ranges(opts, (x_min, x_max, y_min, y_max)),
      sf,
      label_fill,
      FrameEdges::ALL,
      crate::functions::plot_padding::plot_tick_font(opts),
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &labels);
    }
  }

  // Inject Callout labels: text annotation near each labeled series
  {
    let margin_left_f = margin_left as f64;
    let margin_right_f = margin_right as f64;
    let margin_bottom_f = margin_bottom as f64;
    let margin_top_f = top_margin as f64;
    let plot_x0 = margin_left_f + y_label_area as f64;
    let plot_w = render_width as f64
      - margin_left_f
      - margin_right_f
      - y_label_area as f64;
    let plot_h = render_height as f64
      - margin_top_f
      - margin_bottom_f
      - x_label_area as f64;
    inject_callout_labels(
      &mut buf,
      opts,
      all_points,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, margin_top_f, plot_w, plot_h),
    );
    inject_plot_labels(
      &mut buf,
      opts,
      all_points,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, margin_top_f, plot_w, plot_h),
    );
    inject_point_labels(
      &mut buf,
      opts,
      all_points,
      (x_min, x_max),
      (y_min, y_max),
      (plot_x0, margin_top_f, plot_w, plot_h),
    );
  }

  inject_legend(&mut buf, opts);

  Ok(buf)
}

/// Inject Callout/Labeled series labels into a finished SVG buffer as a text
/// annotation near each labeled series. `plot_area` is the drawing region in
/// render-space pixels: (left, top, width, height).
fn inject_callout_labels(
  buf: &mut String,
  opts: &PlotOptions,
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_area: (f64, f64, f64, f64),
) {
  if opts.callout_labels.is_empty()
    || !opts.callout_labels.iter().any(std::option::Option::is_some)
  {
    return;
  }
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  let sf = RESOLUTION_SCALE as f64;
  let callout_font_size = sf * 16.0;

  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let mut callout_svg = String::new();

  for (series_idx, label) in opts.callout_labels.iter().enumerate() {
    let Some(label_text) = label else { continue };
    if series_idx >= all_points.len() {
      continue;
    }
    let points = &all_points[series_idx];

    // Find a good label point: pick the point closest to 2/3 of x range
    let target_x = x_min + (x_max - x_min) * 2.0 / 3.0;
    let best = points
      .iter()
      .filter(|(x, y)| x.is_finite() && y.is_finite())
      .min_by(|a, b| {
        (a.0 - target_x)
          .abs()
          .partial_cmp(&(b.0 - target_x).abs())
          .unwrap_or(std::cmp::Ordering::Equal)
      });

    let Some(&(data_x, data_y)) = best else {
      continue;
    };

    // Convert data coordinates to SVG pixel coordinates
    let frac_x = (data_x - x_min) / (x_max - x_min);
    let frac_y = (data_y - y_min) / (y_max - y_min);
    let px = plot_x0 + frac_x * plot_w;
    let py = plot_top + plot_h * (1.0 - frac_y);

    // Label offset: place text above the curve point
    let label_px = px + sf * 5.0;
    let label_py = py - sf * 12.0;

    // Draw a small line from the curve point to the label
    let (r, g, b) = series_color(&opts.plot_style, series_idx);
    let color_str = format!("rgb({r},{g},{b})");

    callout_svg.push_str(&format!(
      "<line x1=\"{px:.1}\" y1=\"{py:.1}\" x2=\"{label_px:.1}\" y2=\"{label_py:.1}\" \
       stroke=\"{color_str}\" stroke-width=\"{sw}\" />\n",
      sw = sf * 1.0,
    ));
    callout_svg.push_str(&format!(
      "<text x=\"{label_px:.1}\" y=\"{label_py:.1}\" \
       font-family=\"sans-serif\" font-size=\"{callout_font_size:.0}\" \
       fill=\"{color_str}\" dominant-baseline=\"auto\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(label_text)
    ));
  }

  buf.insert_str(insert_pos, &callout_svg);
}

/// Inject `PlotLabels` into a finished SVG buffer: each label sits just past
/// the right-hand end of its curve. `plot_area` is the drawing region in
/// render-space pixels: (left, top, width, height).
fn inject_plot_labels(
  buf: &mut String,
  opts: &PlotOptions,
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_area: (f64, f64, f64, f64),
) {
  if !opts.plot_labels.iter().any(std::option::Option::is_some) {
    return;
  }
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  let sf = RESOLUTION_SCALE as f64;
  let font_size = sf * plot_padding::LABEL_FONT;
  let (_, _, _, _, text_fill) = plot_theme();

  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  // Each label's series, its curve end (px, py) and its own text height.
  let mut placed: Vec<(usize, &String, f64, f64, f64)> = Vec::new();
  for (series_idx, label) in opts.plot_labels.iter().enumerate() {
    let Some(label_text) = label else { continue };
    let Some(points) = all_points.get(series_idx) else {
      continue;
    };
    // The curve's right-hand end: the last point inside the plotted range.
    let Some(&(data_x, data_y)) = points.iter().rev().find(|(x, y)| {
      x.is_finite() && y.is_finite() && *x <= x_max && *x >= x_min
    }) else {
      continue;
    };
    let frac_x = (data_x - x_min) / (x_max - x_min);
    let frac_y = ((data_y - y_min) / (y_max - y_min)).clamp(0.0, 1.0);
    let px = plot_x0 + frac_x * plot_w;
    let py = plot_top + plot_h * (1.0 - frac_y);
    placed.push((series_idx, label_text, px, py, py));
  }
  // Labels whose curves end close together are spread apart vertically
  // (top to bottom, one line height between them); a callout's leader
  // then bends from its curve end to its label.
  let line_h = font_size * 1.6;
  placed
    .sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
  for i in 1..placed.len() {
    let min_y = placed[i - 1].4 + line_h;
    if placed[i].4 < min_y {
      placed[i].4 = min_y;
    }
  }
  // Re-centre a spread group on its curve ends.
  if let (Some(first), Some(last)) = (placed.first(), placed.last()) {
    let shift = ((first.3 + last.3) - (first.4 + last.4)) / 2.0;
    if shift < 0.0 {
      for p in &mut placed {
        p.4 += shift;
      }
    }
  }

  let mut label_svg = String::new();
  for (series_idx, label_text, px, py, ly) in placed {
    // A callout's leader: a gray 1.25 pt line on a translucent white halo,
    // from 3.5 pt past the point, as WL 15 draws it.
    let mut text_x = px + sf * 6.0;
    if opts.plot_label_leaders.get(series_idx) == Some(&true) {
      let (x1, x2) = (px + 3.5 * sf, px + (3.5 + PLOT_LABEL_LEADER) * sf);
      let points = if (ly - py).abs() < 0.5 * sf {
        format!("{x1:.1},{py:.1} {x2:.1},{py:.1}")
      } else {
        format!(
          "{x1:.1},{py:.1} {:.1},{py:.1} {:.1},{ly:.1} {x2:.1},{ly:.1}",
          x1 + 8.0 * sf,
          x1 + 16.0 * sf,
        )
      };
      label_svg.push_str(&format!(
        "<polyline points=\"{points}\" fill=\"none\" stroke=\"#FFFFFF\" \
         stroke-opacity=\"0.667\" stroke-width=\"{:.1}\"/>\n\
         <polyline points=\"{points}\" fill=\"none\" stroke=\"#AAAAAA\" \
         stroke-width=\"{:.1}\"/>\n",
        4.0 * sf,
        1.25 * sf,
      ));
      text_x = x2 + 2.0 * sf;
    }
    label_svg.push_str(&format!(
      "<text x=\"{text_x:.1}\" y=\"{ly:.1}\" font-family=\"sans-serif\" \
       font-size=\"{font_size:.0}\" fill=\"{text_fill}\" \
       dominant-baseline=\"central\">{}</text>\n",
      crate::functions::graphics::box_string_to_svg(label_text)
    ));
  }

  buf.insert_str(insert_pos, &label_svg);
}

/// Length of a series-end callout's leader line, in points.
const PLOT_LABEL_LEADER: f64 = 24.5;

/// Font size of a point label (`Labeled` / `Callout` data entries), in
/// display pixels.
const POINT_LABEL_FONT: f64 = 10.5;

/// An axis-aligned box in render space: (left, top, right, bottom).
type LabelBox = (f64, f64, f64, f64);

fn boxes_overlap(a: LabelBox, b: LabelBox) -> bool {
  a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
}

/// Where a point label goes: its text box, and for a callout the leader
/// line (render-space points) from the point to the box.
struct LabelPlacement {
  bbox: LabelBox,
  leader: Vec<(f64, f64)>,
}

/// The candidate placements of a `w` × `h` label for the point `(px, py)`,
/// in the order they are tried, all in render space (`sf` units per
/// printer's point, y down; `dot` is the radius of the point's dot). A
/// plain label sits just beside its dot, first above right; a callout is pulled away along a leader that leaves the
/// point at an angle, bends and ends in a short horizontal run at the label
/// — the geometry WL 15's `Callout` draws: the leader starts 3 pt out, bends
/// 8 pt out, runs 2 pt on and the label box (with 2.5 pt / 1 pt margins)
/// follows.
fn label_candidates(
  (px, py): (f64, f64),
  (w, h): (f64, f64),
  (sf, dot): (f64, f64),
  callout: bool,
) -> Vec<LabelPlacement> {
  if !callout {
    let dx = w / 2.0 + dot + 1.0 * sf;
    let dy = h * 0.55;
    let up = h / 2.0 + dot + 1.0 * sf;
    return [
      (dx, -dy),
      (-dx, -dy),
      (dx, dy),
      (-dx, dy),
      (0.0, -up),
      (0.0, up),
    ]
    .into_iter()
    .map(|(cx, cy)| LabelPlacement {
      bbox: (
        px + cx - w / 2.0,
        py + cy - h / 2.0,
        px + cx + w / 2.0,
        py + cy + h / 2.0,
      ),
      leader: Vec::new(),
    })
    .collect();
  }
  let (bw, bh) = (w + 5.0 * sf, h + 2.0 * sf);
  [45.0f64, 135.0, -45.0, -135.0, 0.0, 180.0, 90.0, -90.0]
    .into_iter()
    .map(|deg| {
      let (sin, cos) = deg.to_radians().sin_cos();
      let at = |r: f64| (px + r * sf * cos, py - r * sf * sin);
      let start = at(3.0);
      let elbow = at(8.0);
      let side = if cos.abs() < 1e-9 { 0.0 } else { cos.signum() };
      let (leader, bbox) = if side == 0.0 {
        // Straight up or down: the box sits centred on the leader's end.
        let top = if sin > 0.0 { elbow.1 - bh } else { elbow.1 };
        (
          vec![start, elbow],
          (elbow.0 - bw / 2.0, top, elbow.0 + bw / 2.0, top + bh),
        )
      } else {
        let end = (elbow.0 + side * 2.0 * sf, elbow.1);
        let left = if side > 0.0 { end.0 } else { end.0 - bw };
        (
          vec![start, elbow, end],
          (left, end.1 - bh / 2.0, left + bw, end.1 + bh / 2.0),
        )
      };
      LabelPlacement { bbox, leader }
    })
    .collect()
}

/// Inject per-point labels (from `Labeled[y, label]` / `Callout[pt, label]`
/// data entries) into a finished SVG buffer. `plot_area` is the drawing
/// region in render-space pixels: (left, top, width, height).
///
/// Labels are placed one after another where they cover neither an
/// already placed label nor another data point and stay inside the plot area,
/// trying the positions of [`label_candidates`] in turn. A callout that
/// finds no free place is dropped, as WL drops the callouts of crowded
/// points; a plain label then keeps its first position.
fn inject_point_labels(
  buf: &mut String,
  opts: &PlotOptions,
  all_points: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_area: (f64, f64, f64, f64),
) {
  if !opts
    .point_labels
    .iter()
    .any(|series| series.iter().any(std::option::Option::is_some))
  {
    return;
  }
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  let sf = RESOLUTION_SCALE as f64;
  let font_size = sf * POINT_LABEL_FONT;
  let (_, _, _, _, text_fill) = plot_theme();
  let render_width = opts.svg_width * RESOLUTION_SCALE;

  let Some(insert_pos) = buf.rfind("</svg>") else {
    return;
  };
  let to_px = |(x, y): (f64, f64)| {
    (
      plot_x0 + (x - x_min) / (x_max - x_min) * plot_w,
      plot_top + plot_h * (1.0 - (y - y_min) / (y_max - y_min)),
    )
  };
  // Every drawn data point, as a box a label must not cover.
  let point_boxes: Vec<LabelBox> = (0..all_points.len())
    .flat_map(|i| {
      let r = series_point_radius(opts, i, render_width) as f64 + sf;
      opts
        .data_points
        .get(i)
        .or_else(|| all_points.get(i))
        .into_iter()
        .flatten()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|&p| {
          let (px, py) = to_px(p);
          (px - r, py - r, px + r, py + r)
        })
        .collect::<Vec<_>>()
    })
    .collect();
  let mut placed: Vec<LabelBox> = Vec::new();
  let mut label_svg = String::new();

  for (series_idx, labels) in opts.point_labels.iter().enumerate() {
    // Labels are parallel to the original data points, which may differ
    // from the drawn curve (InterpolationOrder).
    let Some(points) = opts
      .data_points
      .get(series_idx)
      .or_else(|| all_points.get(series_idx))
    else {
      continue;
    };
    for (&(data_x, data_y), label) in points.iter().zip(labels) {
      let Some(label_text) = label else { continue };
      if !(data_x.is_finite() && data_y.is_finite()) {
        continue;
      }
      let (px, py) = to_px((data_x, data_y));
      let text = crate::functions::graphics::box_string_to_svg(label_text);
      // A named position puts the label on that side of the point.
      if let Some(pos) = opts.point_label_position.as_deref() {
        let gap = sf * 6.0;
        let (label_px, label_py, anchor) = match pos {
          "Above" | "Top" => (px, py - gap, "middle"),
          "Below" | "Bottom" => (px, py + gap + font_size * 0.75, "middle"),
          "Left" => (px - gap, py + font_size * 0.35, "end"),
          "Right" => (px + gap, py + font_size * 0.35, "start"),
          _ => (px, py + font_size * 0.35, "middle"),
        };
        label_svg.push_str(&format!(
          "<text x=\"{label_px:.1}\" y=\"{label_py:.1}\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{text_fill}\" text-anchor=\"{anchor}\">{text}</text>\n"
        ));
        continue;
      }
      let w = label_text.chars().count() as f64 * font_size * 0.6;
      let h = font_size;
      let own = (px - sf, py - sf, px + sf, py + sf);
      let free = |b: LabelBox| {
        b.0 >= plot_x0
          && b.1 >= plot_top
          && b.2 <= plot_x0 + plot_w
          && b.3 <= plot_top + plot_h
          && !placed.iter().any(|&p| boxes_overlap(p, b))
          && !point_boxes
            .iter()
            .any(|&p| !boxes_overlap(p, own) && boxes_overlap(p, b))
      };
      let dot = series_point_radius(opts, series_idx, render_width) as f64;
      let candidates = label_candidates(
        (px, py),
        (w, h),
        (sf, dot),
        opts.point_label_callouts,
      );
      let choice = match candidates.iter().position(|c| free(c.bbox)) {
        Some(i) => &candidates[i],
        None if opts.point_label_callouts => continue,
        None => &candidates[0],
      };
      placed.push(choice.bbox);
      let (cx, cy) = (
        f64::midpoint(choice.bbox.0, choice.bbox.2),
        f64::midpoint(choice.bbox.1, choice.bbox.3),
      );
      if !choice.leader.is_empty() {
        let path: String = choice
          .leader
          .iter()
          .map(|(x, y)| format!("{x:.1},{y:.1}"))
          .collect::<Vec<_>>()
          .join(" ");
        let (b0, b1, b2, b3) = choice.bbox;
        label_svg.push_str(&format!(
          "<polyline points=\"{path}\" fill=\"none\" stroke=\"#FFFFFF\" \
           stroke-opacity=\"0.667\" stroke-width=\"{:.0}\" \
           stroke-linejoin=\"round\"/>\n\
           <polyline points=\"{path}\" fill=\"none\" stroke=\"#AAAAAA\" \
           stroke-width=\"{:.1}\" stroke-linejoin=\"round\"/>\n\
           <rect x=\"{b0:.1}\" y=\"{b1:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
           fill=\"#FFFFFF\" fill-opacity=\"0.667\"/>\n",
          4.0 * sf,
          1.25 * sf,
          b2 - b0,
          b3 - b1,
        ));
      }
      label_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
         font-size=\"{font_size:.0}\" fill=\"{text_fill}\" \
         text-anchor=\"middle\">{text}</text>\n",
        cy + font_size * 0.35,
      ));
    }
  }

  buf.insert_str(insert_pos, &label_svg);
}

/// Build a `PlotSource` from sampled plot data so that `Show` can later
/// merge multiple pre-rendered plots and re-render via plotters.
pub(crate) fn build_plot_source(
  all_points: &[Vec<(f64, f64)>],
  plot_style: &[SeriesStyle],
  x_range: (f64, f64),
  y_range: (f64, f64),
  image_size: (u32, u32),
  is_scatter: &[bool],
  filling: Filling,
  filling_style: Option<FillStyle>,
  options: Vec<Expr>,
) -> crate::syntax::PlotSource {
  let series_filling = filling.to_series_filling();
  let (fill_color, fill_opacity) = match filling_style {
    Some(fs) => (fs.color, fs.effective_opacity()),
    None => (None, None),
  };
  let series = all_points
    .iter()
    .enumerate()
    .map(|(i, points)| {
      let color = series_color(plot_style, i);
      let (thickness, point_size, dashing) = if plot_style.is_empty() {
        (None, None, None)
      } else {
        let style = &plot_style[i % plot_style.len()];
        (style.thickness, style.point_size, style.dashing.clone())
      };
      let opacity = plot_style
        .get(i % plot_style.len().max(1))
        .and_then(|s| s.opacity);
      let color_expr = plot_style
        .get(i % plot_style.len().max(1))
        .and_then(|s| s.color_expr.clone());
      let scatter = if is_scatter.is_empty() {
        false
      } else {
        is_scatter[i % is_scatter.len()]
      };
      crate::syntax::PlotSeriesData {
        points: points.clone(),
        color,
        color_expr,
        is_scatter: scatter,
        filling: series_filling,
        fill_color,
        fill_opacity,
        marker: None,
        thickness,
        point_size,
        dashing,
        opacity,
        labels: Vec::new(),
        mesh_full_points: Vec::new(),
      }
    })
    .collect();

  crate::syntax::PlotSource {
    series,
    x_range,
    y_range,
    image_size,
    options,
    log_x: false,
    log_y: false,
    plot_range: None,
  }
}

/// The option rules a graphics function was called with, verbatim, so
/// `Show` can carry them into a merged graphic.
pub(crate) fn explicit_options(args: &[Expr]) -> Vec<Expr> {
  // The plot remembers its options for a later `Show`, which may run after
  // the scope that gave them their values is gone (`Table[Plot[…,
  // PlotStyle -> c[[i]]], {i, 2}]`), so an immediate rule keeps the value
  // it has now. A delayed rule stays delayed.
  args
    .iter()
    .filter_map(|a| match a {
      Expr::Rule {
        pattern,
        replacement,
      } => Some(Expr::Rule {
        pattern: pattern.clone(),
        replacement: Box::new(
          crate::evaluator::evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone()),
        ),
      }),
      Expr::RuleDelayed { .. } => Some(a.clone()),
      _ => None,
    })
    .collect()
}

/// With a single series a `PlotStyle` list is one combined style, not a
/// per-series cycle: `ListPlot[pts, PlotStyle -> {PointSize[.02], Red}]`
/// draws one red series, not a `PointSize` series and a `Red` series. A list
/// made only of nested lists is per-dataset instead, so the first one wins. Later
/// directives win, so a nested `{PointSize[.02], {PointSize[.04], Red}}`
/// (the shape a `Which` inside `PlotStyle` produces) ends up red.
pub(crate) fn collapse_style_for_single_series(
  styles: &[SeriesStyle],
) -> Vec<SeriesStyle> {
  if styles.len() < 2 {
    return styles.to_vec();
  }
  // `PlotStyle -> {{…}, {…}}` is one style list per dataset, and a lone
  // dataset takes the first.
  if styles.iter().all(|s| s.from_list) {
    return vec![styles[0].clone()];
  }
  let mut merged = SeriesStyle::default();
  for style in styles {
    if style.color.is_some() {
      merged.color = style.color;
    }
    if style.thickness.is_some() {
      merged.thickness = style.thickness;
    }
    if style.point_size.is_some() {
      merged.point_size = style.point_size;
    }
    if style.dashing.is_some() {
      merged.dashing.clone_from(&style.dashing);
    }
    if style.shadow.is_some() {
      merged.shadow.clone_from(&style.shadow);
    }
    if style.opacity.is_some() {
      merged.opacity = style.opacity;
    }
  }
  vec![merged]
}

/// Get the (r, g, b) color for a series, using custom plot_style if available.
/// The color `ColorFunction -> cf` assigns at `(x, y)`, or `None` when `cf`
/// does not evaluate to a color. A two-parameter function receives `x` and
/// `y`; anything else just `x`. With scaling on, both are first rescaled to
/// 0..1 over the plot range.
pub(crate) fn color_function_rgb(
  cf: &Expr,
  scaling: (bool, bool),
  (x, y): (f64, f64),
  (x_min, x_max, y_min, y_max): (f64, f64, f64, f64),
) -> Option<(u8, u8, u8)> {
  let scale = |v: f64, lo: f64, hi: f64| {
    if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }
  };
  let ax = if scaling.0 { scale(x, x_min, x_max) } else { x };
  let ay = if scaling.1 { scale(y, y_min, y_max) } else { y };
  if let Some(name) =
    crate::functions::field_plot::color_function_scheme_name(cf)
  {
    // A named gradient colors a curve or point by its height (WL:
    // `ListLinePlot[{1, 3, 2, 5}, ColorFunction -> "Thermometer"]` gives
    // its vertices the scheme at the scaled y values 0, 1/2, 1/4, 1). A
    // caller without a y extent (a chart passes its bar values as x) is
    // colored by x.
    let t = if y_max > y_min {
      scale(y, y_min, y_max)
    } else {
      scale(x, x_min, x_max)
    };
    return Some(crate::functions::field_plot::apply_named_color_function(
      &name, t,
    ));
  }
  let mut args = vec![Expr::Real(ax)];
  let two_params = match cf {
    Expr::NamedFunction { params, .. } => params.len() >= 2,
    Expr::FunctionCall { name, args } if name == "Function" => {
      matches!(args.first(), Some(Expr::List(ps)) if ps.len() >= 2)
    }
    Expr::Function { body } => {
      crate::evaluator::function_application::max_slot_index(body) >= 2
    }
    _ => false,
  };
  if two_params {
    args.push(Expr::Real(ay));
  }
  let call = Expr::CurriedCall {
    func: Box::new(cf.clone()),
    args,
  };
  let c = evaluate_expr_to_expr(&call)
    .ok()
    .and_then(|r| parse_color(&r))?;
  let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  Some((ch(c.r), ch(c.g), ch(c.b)))
}

/// Per-point colors for a scatter series drawn with `ColorFunction -> f` in
/// `options` (the plot's own option rules), or `None` when it has none.
/// `x` is rescaled over the series' own extent, as the plot renderer does.
pub(crate) fn scatter_point_colors(
  options: &[Expr],
  points: &[(f64, f64)],
  y_range: (f64, f64),
) -> Option<Vec<(u8, u8, u8)>> {
  let mut opts = PlotOptions::default();
  let mut overrides = PlotRangeOverrides::default();
  for opt in options {
    if let Some((name, value)) =
      crate::functions::graphics::option_name_value(opt)
      && matches!(name, "ColorFunction" | "ColorFunctionScaling")
    {
      apply_common_plot_option(name, &value, &mut opts, &mut overrides);
    }
  }
  let cf = opts.color_function.as_ref()?;
  let (x_min, x_max) = points
    .iter()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(x, _)| {
      (lo.min(x), hi.max(x))
    });
  points
    .iter()
    .map(|&p| {
      color_function_rgb(
        cf,
        opts.color_function_scaling,
        p,
        (x_min, x_max, y_range.0, y_range.1),
      )
    })
    .collect()
}

/// The `(x_min, x_max, y_min, y_max)` extent of the finite points of
/// `series`, or `None` when there are none.
fn data_extents(series: &[Vec<(f64, f64)>]) -> Option<(f64, f64, f64, f64)> {
  let mut pts = series
    .iter()
    .flatten()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .peekable();
  pts.peek()?;
  Some(pts.fold(
    (
      f64::INFINITY,
      f64::NEG_INFINITY,
      f64::INFINITY,
      f64::NEG_INFINITY,
    ),
    |(x0, x1, y0, y1), &(x, y)| (x0.min(x), x1.max(x), y0.min(y), y1.max(y)),
  ))
}

/// Split a curve segment into maximal runs of consecutive points whose
/// `ColorFunction` color agrees. Each run is colored by the function at the
/// midpoint of every step, and consecutive runs share their boundary point
/// so the curve stays connected.
fn color_function_runs(
  segment: &[(f64, f64)],
  cf: &Expr,
  scaling: (bool, bool),
  ranges: (f64, f64, f64, f64),
) -> Vec<((u8, u8, u8), Vec<(f64, f64)>)> {
  let mut runs: Vec<((u8, u8, u8), Vec<(f64, f64)>)> = Vec::new();
  for pair in segment.windows(2) {
    let mid = (
      f64::midpoint(pair[0].0, pair[1].0),
      f64::midpoint(pair[0].1, pair[1].1),
    );
    let c = color_function_rgb(cf, scaling, mid, ranges).unwrap_or((0, 0, 0));
    match runs.last_mut() {
      Some((last, pts)) if *last == c => pts.push(pair[1]),
      _ => runs.push((c, vec![pair[0], pair[1]])),
    }
  }
  runs
}

/// A `ColorFunction`-colored curve or fill area of a function plot. It is
/// drawn as one element in a placeholder color, which
/// [`apply_color_function_gradients`] then turns into a horizontal
/// `linearGradient` through the colors at the curve's vertices — Wolfram's
/// `VertexColors` shading. Drawing it as one piece per color run instead
/// banded the fill and left anti-aliasing seams and notches where the
/// pieces (and the curve's butt-capped strokes) met.
struct ColorFunctionGradient {
  sentinel: (u8, u8, u8),
  /// `(offset, color)` stops, offsets 0..1 along the segment's x extent.
  stops: Vec<(f64, (u8, u8, u8))>,
}

/// The gradient stops of a `ColorFunction` segment, colored at its
/// vertices. `None` unless x strictly increases along the segment (a
/// function plot), where a horizontal gradient is exact.
fn color_function_gradient_stops(
  segment: &[(f64, f64)],
  cf: &Expr,
  scaling: (bool, bool),
  ranges: (f64, f64, f64, f64),
  log_x: bool,
) -> Option<Vec<(f64, (u8, u8, u8))>> {
  if segment.len() < 2 || segment.windows(2).any(|w| w[1].0 <= w[0].0) {
    return None;
  }
  let t = |x: f64| if log_x { x.log10() } else { x };
  let (t0, t1) = (t(segment[0].0), t(segment[segment.len() - 1].0));
  if t1.partial_cmp(&t0) != Some(std::cmp::Ordering::Greater) {
    return None;
  }
  Some(
    segment
      .iter()
      .map(|&p| {
        let c = color_function_rgb(cf, scaling, p, ranges).unwrap_or((0, 0, 0));
        ((t(p.0) - t0) / (t1 - t0), c)
      })
      .collect(),
  )
}

/// The placeholder color of the `k`-th gradient element of a plot.
fn color_function_gradient_sentinel(k: usize) -> (u8, u8, u8) {
  (0x01, ((k >> 8) & 0xff) as u8, (k & 0xff) as u8)
}

/// Replace each placeholder-colored element of `buf` by the same element
/// painted with its horizontal gradient (see [`ColorFunctionGradient`]).
fn apply_color_function_gradients(
  buf: &mut String,
  gradients: &[ColorFunctionGradient],
) {
  use std::hash::{Hash, Hasher};
  for g in gradients {
    let (r, gg, b) = g.sentinel;
    let needle = format!("=\"#{r:02X}{gg:02X}{b:02X}\"");
    let mut search_from = 0;
    while let Some(rel) = buf[search_from..].find(&needle) {
      let at = search_from + rel;
      let Some(start) = buf[..at].rfind('<') else {
        break;
      };
      let Some(end_rel) = buf[at..].find('>') else {
        break;
      };
      let end = at + end_rel + 1;
      let element = buf[start..end].to_string();
      let xs: Vec<f64> = element
        .split("points=\"")
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .map(|pts| {
          pts
            .split_whitespace()
            .filter_map(|xy| xy.split(',').next()?.parse::<f64>().ok())
            .collect()
        })
        .unwrap_or_default();
      let (x1, x2) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
          (a.min(x), b.max(x))
        });
      if x2.partial_cmp(&x1) != Some(std::cmp::Ordering::Greater) {
        search_from = end;
        continue;
      }
      let mut hasher = std::collections::hash_map::DefaultHasher::new();
      element.hash(&mut hasher);
      for (o, c) in &g.stops {
        o.to_bits().hash(&mut hasher);
        c.hash(&mut hasher);
      }
      let id = format!("cfgrad{:016x}", hasher.finish());
      let stops: String = g.stops.iter().fold(
        String::new(),
        |mut acc, (o, (cr, cg, cb))| {
          use std::fmt::Write as _;
          let _ = write!(
            acc,
            "<stop offset=\"{o:.6}\" stop-color=\"#{cr:02X}{cg:02X}{cb:02X}\"/>"
          );
          acc
        },
      );
      let replaced = format!(
        "<defs><linearGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" \
         x1=\"{x1}\" y1=\"0\" x2=\"{x2}\" y2=\"0\">{stops}</linearGradient></defs>{}",
        element.replacen(&needle, &format!("=\"url(#{id})\""), 1)
      );
      buf.replace_range(start..end, &replaced);
      search_from = start + replaced.len();
    }
  }
}

fn series_color(plot_style: &[SeriesStyle], idx: usize) -> (u8, u8, u8) {
  if plot_style.is_empty() {
    PLOT_COLORS[idx % PLOT_COLORS.len()]
  } else {
    let style = &plot_style[idx % plot_style.len()];
    if let Some(c) = &style.color {
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
    } else {
      PLOT_COLORS[idx % PLOT_COLORS.len()]
    }
  }
}

/// Get the line opacity for a series (1.0 = opaque).
fn series_opacity(plot_style: &[SeriesStyle], idx: usize) -> f64 {
  if plot_style.is_empty() {
    1.0
  } else {
    plot_style[idx % plot_style.len()].opacity.unwrap_or(1.0)
  }
}

/// Get the line thickness (in render-space units) for a series.
/// Default is 20 (2px at display size with RESOLUTION_SCALE=10).
fn series_thickness(plot_style: &[SeriesStyle], idx: usize) -> u32 {
  // WL 15 draws plot curves with `AbsoluteThickness[2]`.
  let default_thickness = 20; // 2px * RESOLUTION_SCALE
  if plot_style.is_empty() {
    return default_thickness;
  }
  let style = &plot_style[idx % plot_style.len()];
  if let Some(t) = style.thickness {
    (t * RESOLUTION_SCALE as f64).round() as u32
  } else {
    default_thickness
  }
}

/// The radius, in render-space units, of the dots of a scatter series.
/// `PointSize[f]` gives the dot a diameter of the fraction `f` of the image
/// width; `AbsolutePointSize[p]` (stored negative) is p printer's points
/// whatever the image size. Without either, the plot's default
/// `PlotOptions::default_point_size`, else the 3px default dot.
fn series_point_radius(
  opts: &PlotOptions,
  idx: usize,
  render_width: u32,
) -> u32 {
  let plot_style = &opts.plot_style;
  let size = plot_style
    .get(idx % plot_style.len().max(1))
    .and_then(|s| s.point_size)
    .or(opts.default_point_size);
  match size {
    Some(f) if f > 0.0 => (f * render_width as f64 * 0.5).round() as u32,
    Some(p) => (-p * 0.5 * RESOLUTION_SCALE as f64).round() as u32,
    None => 3 * RESOLUTION_SCALE,
  }
}

/// Whether series `idx` draws as a connected line rather than scattered
/// points, per `ListPlot[…, Joined -> {b1, b2, …}]`. A series past the end
/// of a shorter flags list cycles back into it — the same rule `PlotStyle`,
/// `PlotMarkers`, and every other per-series option already follow — rather
/// than picking an arbitrary default; every place that reads
/// `PlotOptions::joined_per_series` calls this one function so a plot's
/// initial render, a `Show` merge, and an interactive re-render never
/// disagree about which series is which.
pub(crate) fn resolve_series_joined(
  joined_per_series: Option<&[bool]>,
  idx: usize,
) -> bool {
  match joined_per_series {
    Some(flags) if !flags.is_empty() => flags[idx % flags.len()],
    _ => true,
  }
}

/// Get the dash pattern (in data-space fractions) for a series, if any.
fn series_dashing(plot_style: &[SeriesStyle], idx: usize) -> Option<Vec<f64>> {
  if plot_style.is_empty() {
    return None;
  }
  let style = &plot_style[idx % plot_style.len()];
  style.dashing.clone()
}

/// Generate SVG for a scatter plot: the line renderer with every series
/// drawn as unjoined points, so axes, frames, ticks and overlays of a
/// scatter plot never drift from those of a line plot.
pub(crate) fn generate_scatter_svg_with_options(
  all_series: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  opts: &PlotOptions,
) -> Result<String, InterpreterError> {
  let mut opts = opts.clone();
  opts.joined_per_series = Some(vec![false]);
  generate_svg_with_filling(all_series, x_range, y_range, &opts)
}

/// Render a merged `PlotSource` (from `Show`).
///
/// `Show` keeps the options of the graphics it was given, so the merged
/// graphic goes back through the ordinary plot renderers rather than a
/// second implementation that would drift from them: line series render as
/// a line plot, scatter series as points (as the plot itself, when every
/// series is one, otherwise drawn over the lines as epilog primitives).
pub(crate) fn render_merged_plot_source(
  source: &crate::syntax::PlotSource,
) -> Result<String, InterpreterError> {
  let mut opts = PlotOptions {
    svg_width: source.image_size.0,
    svg_height: source.image_size.1,
    ..PlotOptions::default()
  };
  let mut overrides = PlotRangeOverrides::default();
  let (options, simple_ticks) = expand_plot_theme(&source.options, false);
  opts.simple_ticks = simple_ticks;
  for opt in &options {
    if let Some((name, value)) =
      crate::functions::graphics::option_name_value(opt)
      && !apply_common_plot_option(name, &value, &mut opts, &mut overrides)
      && name == "PlotLegends"
    {
      // The legend of the first plot survives the merge, like the other
      // options inherited from it. Explicit labels only: automatic
      // legends are keyed by the per-plot expressions, gone by now.
      let (labels, _, _, position) = parse_plot_legends(&value);
      opts.legend_position = position;
      if !labels.is_empty() {
        opts.plot_legends = labels;
      }
    }
  }
  if let Some(ar) = overrides.aspect_ratio {
    opts.aspect_ratio = Some(ar);
  }
  opts.aspect_automatic = overrides.aspect_automatic;
  opts.log_x = source.log_x;
  opts.log_y = source.log_y;
  let x_range = overrides.x.unwrap_or(source.x_range);
  let y_range = overrides.y.unwrap_or(source.y_range);

  let all_scatter =
    !source.series.is_empty() && source.series.iter().all(|s| s.is_scatter);
  // The series that make up the chart itself; scatter series overlaid on
  // lines are drawn as epilog primitives instead.
  let drawn: Vec<&crate::syntax::PlotSeriesData> = source
    .series
    .iter()
    .filter(|s| s.is_scatter == all_scatter)
    .collect();

  // Each series carries the colour its own plot resolved, so `PlotStyle`
  // is rebuilt from the series rather than re-read from the options.
  opts.plot_style = drawn
    .iter()
    .map(|s| SeriesStyle {
      color: Some(WoxiColor::new(
        s.color.0 as f64 / 255.0,
        s.color.1 as f64 / 255.0,
        s.color.2 as f64 / 255.0,
      )),
      color_expr: s.color_expr.clone(),
      thickness: s.thickness,
      point_size: s.point_size,
      dashing: s.dashing.clone(),
      opacity: s.opacity,
      ..SeriesStyle::default()
    })
    .collect();
  // Markers travel with the series too, so a `Show` of a marked scatter
  // plot keeps drawing the glyphs its `PlotMarkers` asked for.
  opts.plot_markers = drawn.iter().map(|s| s.marker.clone()).collect();
  if opts.plot_markers.iter().all(Option::is_none) {
    opts.plot_markers.clear();
  }
  if drawn.iter().any(|s| !s.labels.is_empty()) {
    opts.point_labels = drawn.iter().map(|s| s.labels.clone()).collect();
  }
  // `Mesh -> Full` marks each curve's initial sampling grid, not its
  // refined points.
  if drawn.iter().any(|s| !s.mesh_full_points.is_empty()) {
    opts.mesh_full_points =
      drawn.iter().map(|s| s.mesh_full_points.clone()).collect();
  }
  opts.filling_rules = drawn
    .iter()
    .enumerate()
    .filter_map(|(i, s)| match s.filling {
      crate::syntax::SeriesFilling::Series(j) => {
        Some((i, FillTarget::Series(j)))
      }
      other => {
        series_filling_to_filling(other).map(|f| (i, FillTarget::Level(f)))
      }
    })
    .collect();
  // Each merged plot keeps the fill it asked for: three normal curves
  // filled to the axis in three colours stay three colours, rather than all
  // taking the first plot's.
  opts.filling_styles = drawn
    .iter()
    .map(|s| {
      (s.fill_color.is_some() || s.fill_opacity.is_some()).then_some(
        FillStyle {
          color: s.fill_color,
          opacity: s.fill_opacity,
          below_color: None,
          dashing: None,
        },
      )
    })
    .collect();
  if opts.filling_styles.iter().all(Option::is_none) {
    opts.filling_styles.clear();
  }
  opts.epilog.extend(
    source
      .series
      .iter()
      .filter(|s| s.is_scatter != all_scatter)
      .flat_map(|s| scatter_overlay_primitives(s, y_range)),
  );

  let points: Vec<Vec<(f64, f64)>> =
    drawn.iter().map(|s| s.points.clone()).collect();
  if all_scatter {
    generate_scatter_svg_with_options(&points, x_range, y_range, &opts)
  } else {
    generate_svg_with_options(&points, x_range, y_range, &opts)
  }
}

/// The `Filling` a merged series was drawn with, or `None` when it is unfilled.
fn series_filling_to_filling(
  filling: crate::syntax::SeriesFilling,
) -> Option<Filling> {
  match filling {
    crate::syntax::SeriesFilling::None
    | crate::syntax::SeriesFilling::Series(_) => None,
    crate::syntax::SeriesFilling::Axis => Some(Filling::Axis),
    crate::syntax::SeriesFilling::Bottom => Some(Filling::Bottom),
    crate::syntax::SeriesFilling::Top => Some(Filling::Top),
    crate::syntax::SeriesFilling::Value(v) => Some(Filling::Value(v)),
  }
}

/// A scatter series shown on top of line series: its points (and the stems
/// of a filled scatter) as graphics primitives in data coordinates.
fn scatter_overlay_primitives(
  series: &crate::syntax::PlotSeriesData,
  y_range: (f64, f64),
) -> Vec<Expr> {
  let point =
    |(x, y): (f64, f64)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into());
  let finite: Vec<(f64, f64)> = series
    .points
    .iter()
    .copied()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .collect();
  if finite.is_empty() {
    return Vec::new();
  }
  let color = series_color_expr(series);
  let mut prims = vec![color];
  // Stems of a filled scatter series, drawn to the same reference level the
  // standalone scatter plot would use.
  if let Some(ref_y) = series_filling_to_filling(series.filling)
    .and_then(|f| f.reference_y(y_range.0, y_range.1))
  {
    prims.push(call1("AbsoluteThickness", Expr::Real(1.0)));
    for &(x, y) in &finite {
      prims.push(call1(
        "Line",
        Expr::List(vec![point((x, y)), point((x, ref_y))].into()),
      ));
    }
  }
  // A `PlotMarkers` series draws its glyph at every point instead of a dot.
  if let Some(marker) = &series.marker {
    let mut styled = vec![Expr::String(marker.glyph.clone())];
    if let Some((r, g, b)) = marker.color {
      styled.push(call(
        "RGBColor",
        vec![
          Expr::Real(r as f64 / 255.0),
          Expr::Real(g as f64 / 255.0),
          Expr::Real(b as f64 / 255.0),
        ],
      ));
    }
    styled.push(Expr::Real(marker.size));
    let content = call("Style", styled);
    for &(x, y) in &finite {
      prims.push(call("Text", vec![content.clone(), point((x, y))]));
    }
    return prims;
  }
  prims.push(match series.point_size {
    Some(f) if f > 0.0 => call1("PointSize", Expr::Real(f)),
    Some(p) => call1("AbsolutePointSize", Expr::Real(-p)),
    None => call1("AbsolutePointSize", Expr::Real(6.0)),
  });
  prims.push(call1(
    "Point",
    Expr::List(finite.into_iter().map(point).collect::<Vec<_>>().into()),
  ));
  // Point labels sit above and right of their points, in the text colour;
  // near the right end of the data they flip to the left so they stay
  // inside the plot, as in a standalone `ListPlot`.
  let (x_lo, x_hi) = series
    .points
    .iter()
    .filter(|(x, _)| x.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(x, _)| {
      (lo.min(x), hi.max(x))
    });
  let labeled: Vec<Expr> = series
    .points
    .iter()
    .zip(&series.labels)
    .filter(|((x, y), _)| x.is_finite() && y.is_finite())
    .filter_map(|(&pt, label)| {
      let label = label.as_ref()?;
      Some(call(
        "Text",
        vec![
          Expr::String(label.clone()),
          point(pt),
          Expr::List(
            vec![
              Expr::Real(if pt.0 > x_lo + 0.92 * (x_hi - x_lo) {
                1.2
              } else {
                -1.2
              }),
              Expr::Real(-1.2),
            ]
            .into(),
          ),
        ],
      ))
    })
    .collect();
  if !labeled.is_empty() {
    prims.push(call1("GrayLevel", Expr::Real(0.4)));
    prims.extend(labeled);
  }
  prims
}

/// Where a chart's legend goes relative to the plot.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LegendLayout {
  /// A column beside the plot (the default).
  #[default]
  Column,
  /// A single row below the plot (`LegendAppearance -> "Row"`).
  Row,
}

/// Where a vertical bar chart draws the labels produced by `LabelingFunction`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BarLabelPlacement {
  Below,
  Above,
}

/// Generate SVG for a bar chart using plotters.
/// The appearance options of a vertical `BarChart`.
pub(crate) struct BarChartLook {
  /// `Axes -> False` hides the axis lines.
  pub axes: bool,
  /// `AxesStyle`: the colour and stroke width (render units) of the axes
  /// and their tick labels.
  pub axes_color: Option<RGBColor>,
  pub axes_width: Option<u32>,
  /// `Frame -> True` draws a border round the plotting area, in the
  /// `FrameStyle` colour and width.
  pub frame: bool,
  pub frame_color: Option<RGBColor>,
  pub frame_width: Option<u32>,
  /// `BarSpacing -> s`: the space between adjacent groups (single bars of a
  /// flat list), as a fraction of a group's width.
  pub bar_spacing: Option<f64>,
  /// `AspectRatio -> r`: height / width of the plotting area.
  pub aspect_ratio: Option<f64>,
}

impl Default for BarChartLook {
  fn default() -> Self {
    Self {
      axes: true,
      axes_color: None,
      axes_width: None,
      frame: false,
      frame_color: None,
      frame_width: None,
      bar_spacing: None,
      aspect_ratio: None,
    }
  }
}

/// The empty margin on each side of a bar group, in units of the group's
/// slot (one per group): 0.1 by default, else the half of the
/// `BarSpacing -> s` space a group of width `1 / (1 + s)` leaves.
pub(crate) fn bar_group_gap(bar_spacing: Option<f64>) -> f64 {
  match bar_spacing {
    Some(sp) if sp.is_finite() && sp >= 0.0 => sp / (1.0 + sp) / 2.0,
    _ => 0.1,
  }
}

/// The bare margin left of and below a vertical bar chart's label areas,
/// in display points (WL 15's automatic `ImagePadding` hugs the labels).
const BAR_SIDE_MARGIN: f64 = 2.0;

pub(crate) fn generate_bar_svg(
  groups: &[Vec<f64>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  chart_labels: &[ChartLabel],
  chart_label_position: LabelPosition,
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  frame_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  legend_layout: LegendLayout,
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  bar_labels: &[String],
  bar_label_placement: BarLabelPlacement,
  stacked: bool,
  show_ticks: bool,
  epilog: &[Expr],
  look: &BarChartLook,
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let mut render_height = svg_height * RESOLUTION_SCALE;
  let mut svg_height = svg_height;

  let n = groups.len(); // number of groups
  // Bars per group across the x slot: stacked groups share a single column.
  let k = if stacked {
    1
  } else {
    groups.iter().map(std::vec::Vec::len).max().unwrap_or(1)
  };
  // Palette index is by position within the group even when stacked.
  let k_colors = groups.iter().map(std::vec::Vec::len).max().unwrap_or(1);
  // Highest drawn point of a group: its tallest bar, or the sum of the
  // positive entries when stacked.
  let group_top = |g: &[f64]| -> f64 {
    if stacked {
      g.iter().filter(|v| **v > 0.0).sum::<f64>()
    } else {
      g.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }
  };

  // y-axis range: explicit PlotRange overrides the auto-computed extent,
  // which anchors at 0 and leaves WL's `Scaled[0.05]` headroom above the
  // tallest bar (5% of the padded span, whose lower `Scaled[0.02]` the
  // axis at 0 takes up).
  let (y_min, y_max) = if let Some((ymin, ymax)) = plot_range_y {
    (ymin, ymax)
  } else {
    let y_max_auto = groups
      .iter()
      .map(|g| group_top(g))
      .fold(f64::NEG_INFINITY, f64::max)
      .max(0.0)
      * (1.0 + 0.05 / 0.93);
    let y_max_auto = if y_max_auto <= 0.0 { 1.0 } else { y_max_auto };
    (0.0, y_max_auto)
  };
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };

  // x-axis range: bars are categorical, living at 0..n. An explicit
  // PlotRange -> {{xmin, xmax}, ...} extends the drawn axis (bars stay
  // at their slots and the excess becomes empty padding), mirroring
  // the way ListLinePlot treats an x-range wider than the data.
  let (x_min, x_max) = plot_range_x.unwrap_or((0.0, n as f64));
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };

  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  // Extra space for labels
  let has_chart_labels = !chart_labels.is_empty();
  // An `AxesLabel` sits at the far end of its axis, so it takes room to the
  // right (x) and above (y), not below and to the left.
  let axes_label_x = axes_label.map(|(x, _)| x).filter(|x| !x.is_empty());
  let axes_label_y = axes_label.map(|(_, y)| y).filter(|y| !y.is_empty());
  let frame_label_x = frame_label.map(|(x, _)| x).filter(|x| !x.is_empty());
  let frame_label_y = frame_label.map(|(_, y)| y).filter(|y| !y.is_empty());
  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());

  let has_value_labels = bar_labels.iter().any(|s| !s.is_empty());
  // WL 15 leaves only the room the labels take: half a point above the
  // plotting area, a strip for a `PlotLabel`.
  // Half a point above the plotting area (whose own `Scaled[0.05]`
  // headroom tops the tallest bar), more for value labels above the bars.
  let top_margin = if has_plot_label {
    14 * s
  } else if has_value_labels {
    4 * s
  } else {
    s / 2
  } + plot_label_extra_lines(plot_label) as i32 * 20 * s
    + if axes_label_y.is_some() { 20 * s } else { 0 };
  let has_rotated_labels = chart_labels.iter().any(|l| l.rotation.abs() > 0.01);
  // Only content drawn below the axis consumes bottom margin. Chart labels do
  // so only when positioned Below; Center/Above labels sit on the bars and
  // need no reservation, so the plot fills the canvas instead of leaving a
  // large empty strip.
  let labels_below =
    has_chart_labels && matches!(chart_label_position, LabelPosition::Below);
  let mut bottom_extra = 0.0_f64;
  if labels_below {
    bottom_extra += if has_rotated_labels {
      50.0 * sf
    } else {
      10.0 * sf
    };
  }
  if has_value_labels {
    bottom_extra += 30.0 * sf;
  }
  // A `LegendAppearance -> "Row"` legend runs along the bottom edge.
  let row_legend =
    legend_layout == LegendLayout::Row && !chart_legends.is_empty();
  if row_legend {
    bottom_extra += 30.0 * sf;
  }
  if frame_label_x.is_some() {
    bottom_extra += 24.0 * sf;
  }
  // The bars' group ticks hang 6 points below the axis.
  let x_label_area = (5.5 * sf) as u32 + bottom_extra as u32;
  // The y axis and its ticks, drawn the way an unframed `Plot` draws them
  // (`plot_axes::origin_axes_svg`): marks pointing into the chart, labels
  // right-aligned 3 pt left of the axis.
  let y_scale = crate::functions::plot_axes::AxisScale {
    min: y_min,
    max: y_max,
    log: false,
  };
  let y_ticks =
    crate::functions::plot_axes::AxisTicks::new(y_scale, false, None);
  // The y tick labels need their own width plus the gap to the axis.
  let y_label_area = if show_ticks {
    let widest = y_ticks
      .majors
      .iter()
      .map(|(_, l)| plot_padding::tick_label_width(l, plot_padding::TICK_FONT))
      .fold(0.0, f64::max);
    (((widest + plot_padding::TICK_LABEL_GAP) * sf - BAR_SIDE_MARGIN * sf)
      .max(2.0 * sf)) as u32
  } else {
    4 * RESOLUTION_SCALE
  };

  let (bg_color, dark_gray, _light_gray, label_fill, title_default_fill) =
    plot_theme();

  // Reserve extra right margin for chart legends
  let axes_label_margin_right = axes_label_x.map_or(0.0, |label| {
    // The label is drawn at `font_size`; ~0.62 em per character covers the
    // proportional font with a little slack.
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(label) as f64
        * 0.62
        * sf
        * 18.0
  }) as u32;
  // WL runs the x axis 2% of the plotting width past the last bar slot
  // and keeps 1.5 pt beyond that: r = 1.5 + 0.02 (W - left - r).
  let axis_overrun = |left: f64| {
    let r = (1.5 * sf + 0.02 * (render_width as f64 - left)) / 1.02;
    r - 1.5 * sf
  };
  let x_axis_overrun = axis_overrun(BAR_SIDE_MARGIN * sf + y_label_area as f64);
  let legend_margin_right = if chart_legends.is_empty() || row_legend {
    (x_axis_overrun + 1.5 * sf) as u32
  } else {
    chart_legend_column_width(chart_legends, sf) as u32
  };
  let legend_margin_right = legend_margin_right + axes_label_margin_right;

  // `AspectRatio -> r` makes the plotting area (not the whole image) r times
  // as tall as it is wide; the image grows or shrinks around it.
  if let Some(r) = look.aspect_ratio {
    let area_w = render_width as f64
      - BAR_SIDE_MARGIN * sf
      - legend_margin_right as f64
      - y_label_area as f64;
    let margins_v =
      top_margin as f64 + BAR_SIDE_MARGIN * sf + x_label_area as f64;
    let target = area_w.max(1.0) * r + margins_v;
    svg_height = ((target / sf).round() as u32).clamp(1, MAX_SVG_DIMENSION);
    render_height = svg_height * RESOLUTION_SCALE;
  }
  // `AxesStyle` recolours the axes; `Axes -> False` hides them.
  let axis_color = look.axes_color.unwrap_or(dark_gray);

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(legend_margin_right)
      .margin_bottom((BAR_SIDE_MARGIN * sf) as u32)
      .margin_left((BAR_SIDE_MARGIN * sf) as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BarChart: {e}"))
      })?;

    // The axes are drawn after the bars (see `origin_axes_svg` below).
    // Draw bars as plotters Rectangle elements
    let gap = bar_group_gap(look.bar_spacing);
    for (gi, group) in groups.iter().enumerate() {
      let group_x0 = gi as f64 + gap;
      let group_x1 = (gi + 1) as f64 - gap;
      let group_w = group_x1 - group_x0;
      let bar_w = group_w / k as f64;
      // Running ends of the stack: positives pile up, negatives pile down.
      let (mut stack_up, mut stack_down) = (0.0_f64, 0.0_f64);

      for (bi, &val) in group.iter().enumerate() {
        let (br, bg, bb) = if !chart_style.is_empty() {
          // For grouped charts, color by bar index within group
          let color_idx = if k_colors > 1 { bi } else { gi };
          let c = &chart_style[color_idx % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        } else if k_colors > 1 {
          // Grouped: color by position within group
          crate::functions::chart::CHART_COLORS
            [bi % crate::functions::chart::CHART_COLORS.len()]
        } else if !chart_legends.is_empty() {
          // Flat with legends: distinct color per group
          crate::functions::chart::CHART_COLORS
            [gi % crate::functions::chart::CHART_COLORS.len()]
        } else {
          // Flat: single default color
          crate::functions::chart::CHART_COLORS[0]
        };
        let color = RGBColor(br, bg, bb);
        let (x0, y0, y1) = if stacked {
          let base = if val >= 0.0 {
            &mut stack_up
          } else {
            &mut stack_down
          };
          let y0 = *base;
          *base += val;
          (group_x0, y0, *base)
        } else {
          (group_x0 + bi as f64 * bar_w, 0.0, val)
        };
        let x1 = x0 + bar_w;
        chart
          .draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            color.filled(),
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("BarChart: {e}"))
          })?;
      }
    }

    // `Frame -> True` closes the plotting area on all four sides.
    if look.frame {
      let color = look.frame_color.unwrap_or(dark_gray);
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
            (x_min, y_min),
          ],
          color.stroke_width(look.frame_width.unwrap_or(RESOLUTION_SCALE)),
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BarChart: {e}"))
        })?;
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("BarChart: {e}"))
    })?;
  }

  add_bar_borders(&mut buf, RESOLUTION_SCALE);

  // Inject hover tooltips into bar rects
  let bar_values: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().copied()).collect();
  inject_bar_tooltips(&mut buf, &bar_values);

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Compute plot area coordinates (same logic as generate_axes_only_opts)
  let margin_left = BAR_SIDE_MARGIN * sf;
  let margin_top = top_margin as f64;
  let margin_right = legend_margin_right as f64;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = margin_top;
  let plot_w =
    render_width as f64 - margin_left - margin_right - y_label_area as f64;
  let plot_h = render_height as f64
    - margin_top
    - BAR_SIDE_MARGIN * sf
    - x_label_area as f64;
  let axis_y = plot_y0 + plot_h;

  if look.axes {
    let axis_hex = format!(
      "#{:02X}{:02X}{:02X}",
      axis_color.0, axis_color.1, axis_color.2
    );
    let axis_style = SeriesStyle {
      thickness: look.axes_width.map(|w| w as f64 / sf),
      ..Default::default()
    };
    let axes_style = [Some(axis_style.clone()), Some(axis_style)];
    let x_scale = crate::functions::plot_axes::AxisScale {
      min: x_min,
      max: x_max,
      log: false,
    };
    let no_x_ticks = crate::functions::plot_axes::AxisTicks::default();
    let x_axis_end = if chart_legends.is_empty() || row_legend {
      plot_x0 + plot_w + x_axis_overrun
    } else {
      plot_x0 + plot_w
    };
    let origin = (x_min, y_scale.origin(None, None));
    let mut axes_svg = crate::functions::plot_axes::origin_axes_svg(
      &crate::functions::plot_axes::OriginAxes {
        area: (plot_x0, plot_y0, plot_w, plot_h),
        x: x_scale,
        y: y_scale,
        origin,
        show: (true, true),
        ticks: show_ticks.then_some((&no_x_ticks, &y_ticks)),
        sf,
        axis_color: &axis_hex,
        label_fill: if look.axes_color.is_some() {
          &axis_hex
        } else {
          label_fill
        },
        axes_style: &axes_style,
        ticks_style: &[None, None],
        keep_y_label_on_x_axis: true,
        label_color: None,
      },
    );
    let axis_y = plot_y0 + plot_h - y_scale.frac(origin.1) * plot_h;
    let stroke = (look.axes_width.unwrap_or(RESOLUTION_SCALE / 2)) as f64;
    if x_axis_end > plot_x0 + plot_w {
      axes_svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{axis_y:.1}\" x2=\"{x_axis_end:.1}\" \
         y2=\"{axis_y:.1}\" stroke=\"{axis_hex}\" stroke-width=\"{stroke:.0}\"/>\n",
        plot_x0 + plot_w
      ));
    }
    // The x axis carries no labels, only marks hanging below it at the
    // outer edges of the bars (6 pt) and, between labelled bars, at both
    // edges of each gap (4 pt).
    if show_ticks && n > 0 {
      let gap = bar_group_gap(look.bar_spacing);
      let px = |v: f64| plot_x0 + x_scale.frac(v) * plot_w;
      let mut mark = |x: f64, len: f64| {
        axes_svg.push_str(&format!(
          "<line x1=\"{x:.1}\" y1=\"{axis_y:.1}\" x2=\"{x:.1}\" \
           y2=\"{:.1}\" stroke=\"{axis_hex}\" stroke-width=\"{stroke:.0}\"/>\n",
          axis_y + len * sf
        ));
      };
      mark(px(gap), 6.0);
      mark(px(n as f64 - gap), 6.0);
      if has_chart_labels {
        for gi in 1..n {
          mark(px(gi as f64 - gap), 4.0);
          mark(px(gi as f64 + gap), 4.0);
        }
      }
    }
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &axes_svg);
    }
  }

  // WL 15 sets chart labels at the tick-label size and the PlotLabel a
  // little larger.
  let font_size = sf * CHART_LABEL_FONT;
  let title_font_size = sf * CHART_TITLE_FONT;

  // `Epilog` primitives are written in chart coordinates, where the first bar
  // group is centred at `x = 1`; the axis here puts it at `0.5`.
  if !epilog.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    let area = crate::functions::plot_epilog::PlotArea {
      x0: plot_x0,
      y0: plot_y0,
      w: plot_w,
      h: plot_h,
      x_min: x_min + 0.5,
      x_max: x_max + 0.5,
      y_min,
      y_max,
      scale: sf,
    };
    let overlay = crate::functions::plot_epilog::render_overlay_svg(
      epilog, &area, "epilog", false,
    );
    buf.insert_str(pos, &overlay);
  }

  // Insert label SVG elements before </svg>
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();

    // Bars live on the categorical x-axis 0..n, which may be a subset of the
    // displayed x-range when PlotRange extends the axis beyond the data. Map
    // slot centers through the same linear transform used by the cartesian
    // chart above so labels line up with their bars.
    let x_span = x_max - x_min;
    let map_x_val = |v: f64| -> f64 { plot_x0 + (v - x_min) / x_span * plot_w };

    // ChartLabels: position based on chart_label_position
    if has_chart_labels {
      let y_span = y_max - y_min;
      let map_y_val =
        |v: f64| -> f64 { plot_y0 + (y_max - v) / y_span * plot_h };
      for (i, label) in chart_labels.iter().enumerate().take(n) {
        let cx = map_x_val(i as f64 + 0.5);
        // For Above/Center positioning, use the max value in the group
        let group_max = group_top(&groups[i]);
        // Mathematica Rotate is counterclockwise-positive; SVG is clockwise-positive
        let svg_rotation_deg = -label.rotation.to_degrees();
        let is_rotated = svg_rotation_deg.abs() > 0.01;
        let (ly, fill) = match chart_label_position {
          LabelPosition::Above => {
            let bar_top = map_y_val(group_max);
            (bar_top - font_size * 0.5, title_default_fill)
          }
          LabelPosition::Center => {
            // wolframscript draws centered labels in the default (dark) text
            // color, not white; `title_default_fill` is near-black in light
            // mode and light in dark mode for readability on the bars.
            let bar_top = map_y_val(group_max);
            let bar_center = f64::midpoint(bar_top, axis_y) + font_size * 0.4;
            (bar_center, title_default_fill)
          }
          LabelPosition::Below => {
            if is_rotated {
              // With text-anchor=middle and rotation, the left half of the
              // text swings upward. Offset the pivot down so the highest point
              // (pivot_y - half_width * sin(angle)) stays below the axis.
              let char_width_estimate = font_size * 0.6;
              let half_text_w =
                crate::functions::graphics::box_string_visible_len(&label.text)
                  as f64
                  * char_width_estimate
                  / 2.0;
              let sin_a = svg_rotation_deg.to_radians().sin().abs();
              let offset = half_text_w * sin_a + font_size * 0.5;
              (axis_y + offset, label_fill)
            } else {
              (axis_y + font_size * 1.5, label_fill)
            }
          }
        };
        if is_rotated {
          // `dominant-baseline=central` centers the glyphs on `y` (not the
          // baseline); with `text-anchor=middle` the text is centered on the
          // pivot in both axes, so rotating about (cx, ly) keeps it centered on
          // the bar horizontally instead of hanging to one side of the baseline.
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" dominant-baseline=\"central\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\" transform=\"rotate({svg_rotation_deg:.1},{cx:.1},{ly:.1})\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        } else {
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        }
      }
    }

    // Value labels from `LabelingFunction` (e.g. `Placed[#, Below] &`): one
    // per bar, drawn just below the bar's base at the axis (or over its top
    // for `Above`). `bar_labels` is a
    // flat list across all groups/bars, aligned with the bar draw order.
    if has_value_labels {
      let gap = 0.1;
      let bar_w = (1.0 - 2.0 * gap) / k as f64;
      let bar_x_offset = |bi: usize| if stacked { 0 } else { bi };
      // Shrink the font so the widest label fits within one bar's
      // center-to-center spacing (`1.0` chart unit between groups, `bar_w`
      // between bars in a group), preventing adjacent labels from overlapping.
      let spacing_chart = if k > 1 { bar_w } else { 1.0 };
      let spacing_px = (map_x_val(spacing_chart) - map_x_val(0.0)).abs();
      let max_len = bar_labels
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| crate::functions::graphics::box_string_visible_len(s))
        .max()
        .unwrap_or(1)
        .max(1);
      let fit_font = (spacing_px * 0.9 / (max_len as f64 * 0.6))
        .min(font_size)
        .max(font_size * 0.35);
      let mut flat = 0usize;
      for (gi, group) in groups.iter().enumerate() {
        let group_x0 = gi as f64 + gap;
        for bi in 0..group.len() {
          let text = bar_labels.get(flat).cloned().unwrap_or_default();
          flat += 1;
          if text.is_empty() {
            continue;
          }
          let cx =
            map_x_val(group_x0 + (bar_x_offset(bi) as f64 + 0.5) * bar_w);
          // `Placed[…, Above]` puts the label over the bar's top instead.
          let ly =
            if bar_label_placement == BarLabelPlacement::Above && !stacked {
              plot_y0 + (y_max - group[bi]) / (y_max - y_min) * plot_h
                - fit_font * 0.4
            } else {
              axis_y + fit_font * 1.3
            };
          labels_svg.push_str(&format!(
            "<text x=\"{cx:.1}\" y=\"{ly:.1}\" text-anchor=\"middle\" \
             font-family=\"sans-serif\" font-size=\"{fit_font:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&text)
          ));
        }
      }
    }

    // FrameLabel: centred outside the bottom/left edge.
    if let Some(x_label) = frame_label_x {
      let cx = plot_x0 + plot_w / 2.0;
      let base_y = axis_y
        + if has_chart_labels {
          font_size * 1.5 + font_size * 1.3
        } else {
          font_size * 1.5
        };
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\">{}</text>\n",
        crate::functions::graphics::box_string_to_svg(x_label)
      ));
    }
    if let Some(y_label) = frame_label_y {
      let cy = plot_y0 + plot_h / 2.0;
      let lx = margin_left + font_size * 0.8;
      labels_svg.push_str(&format!(
        "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
         fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
        crate::functions::graphics::box_string_to_svg(y_label)
      ));
    }

    // AxesLabel: at the far end of each axis, as in a `Plot`.
    labels_svg.push_str(&axes_label_svg(
      axes_label
        .map(|(x, y)| (x.to_string(), y.to_string()))
        .as_ref(),
      (plot_x0, plot_y0, plot_w, plot_h),
      (0.0, groups.len() as f64, y_min, y_max),
      (true, true),
      font_size,
      label_fill,
    ));

    // PlotLabel: centered above the chart
    if let Some(sl) = plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      // The y AxesLabel shares the top margin, so the title sits above it.
      let ty = margin_top
        - title_font_size * 0.5
        - if axes_label_y.is_some() {
          font_size * 1.2
        } else {
          0.0
        };
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
      let mut style_attrs = String::new();
      if sl.bold {
        style_attrs.push_str(" font-weight=\"bold\"");
      }
      if sl.italic {
        style_attrs.push_str(" font-style=\"italic\"");
      }
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
           fill=\"{fill}\"{style_attrs}>{}</text>\n",
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    // ChartLegends: color swatch + label, positioned to the right of the
    // plot (or in a single row below it for `LegendAppearance -> "Row"`)
    if !chart_legends.is_empty() {
      let swatch_size = sf * 12.0;
      let swatch_gap = sf * 6.0;
      let entry_gap = sf * 14.0;
      let entry_width = |label: &String| {
        swatch_size
          + swatch_gap
          + crate::functions::graphics::box_string_visible_len(label) as f64
            * sf
            * 8.5
      };
      let row_width = chart_legends.iter().map(entry_width).sum::<f64>()
        + entry_gap * (chart_legends.len() - 1) as f64;
      let mut row_x = plot_x0 + (plot_w - row_width) / 2.0;
      let row_y = render_height as f64 - sf * 24.0;
      let legend_x = plot_x0 + plot_w + sf * 16.0;
      let legend_y_start = plot_y0 + sf * 8.0;
      let line_height = sf * 22.0;

      for (i, label) in chart_legends.iter().enumerate() {
        let (cr, cg, cb) = if chart_style.is_empty() {
          crate::functions::chart::CHART_COLORS
            [i % crate::functions::chart::CHART_COLORS.len()]
        } else {
          let c = &chart_style[i % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        };
        let (legend_x, ly) = if row_legend {
          let x = row_x;
          row_x += entry_width(label) + entry_gap;
          (x, row_y)
        } else {
          (legend_x, legend_y_start + i as f64 * line_height)
        };
        labels_svg.push_str(&chart_legend_entry_svg(
          label,
          (cr, cg, cb),
          (legend_x, ly),
          sf,
          label_fill,
        ));
      }
    }

    buf.insert_str(insert_pos, &labels_svg);
  }

  Ok(buf)
}

/// The room a column of chart legend entries takes to the right of a
/// chart: swatch, gap, the estimated text width and padding.
pub(crate) fn chart_legend_column_width(labels: &[String], sf: f64) -> f64 {
  let max_label_len = labels
    .iter()
    .map(|l| crate::functions::graphics::box_string_visible_len(l))
    .max()
    .unwrap_or(0);
  sf * 12.0 + sf * 6.0 + max_label_len as f64 * sf * 10.0 + sf * 16.0
}

/// One chart legend entry (`ChartLegends` / `PlotLegends`): a colour swatch
/// with its top-left corner at `(x, y)` and the label beside it.
pub(crate) fn chart_legend_entry_svg(
  label: &str,
  (cr, cg, cb): (u8, u8, u8),
  (x, y): (f64, f64),
  sf: f64,
  label_fill: &str,
) -> String {
  let legend_font = sf * 16.0;
  let swatch_size = sf * 12.0;
  let swatch_gap = sf * 6.0;
  format!(
    "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{swatch_size:.0}\" height=\"{swatch_size:.0}\" \
     fill=\"rgb({cr},{cg},{cb})\"/>\n\
     <text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
     fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
    x + swatch_size + swatch_gap,
    y + swatch_size / 2.0,
    crate::functions::graphics::box_string_to_svg(label)
  )
}

/// Generate SVG for a horizontal BarChart (`BarOrigin -> Left`).
///
/// Categories run down the left edge (first input at the bottom, matching
/// wolframscript), bars grow rightward from zero, and the value axis with its
/// ticks sits along the bottom. `bar_labels` (from `LabelingFunction`) are
/// drawn just past each bar's end. Unlike the vertical renderer this builds
/// the SVG directly rather than via plotters, since the axes are swapped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_horizontal_bar_svg(
  groups: &[Vec<f64>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  chart_labels: &[crate::functions::chart::ChartLabel],
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  plot_range_x: Option<(f64, f64)>,
  bar_labels: &[String],
) -> std::string::String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;
  let rw = render_width as f64;
  let rh = render_height as f64;

  let n = groups.len();
  let k = groups
    .iter()
    .map(std::vec::Vec::len)
    .max()
    .unwrap_or(1)
    .max(1);

  // Value axis (horizontal): anchored at 0 with 10% headroom unless an
  // explicit PlotRange overrides it.
  let (x_min, x_max) = if let Some((a, b)) = plot_range_x {
    (a, b)
  } else {
    let m = groups
      .iter()
      .flat_map(|g| g.iter())
      .copied()
      .fold(f64::NEG_INFINITY, f64::max)
      .max(0.0)
      * 1.1;
    (0.0, if m <= 0.0 { 1.0 } else { m })
  };
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };

  let (bg, axis_gray, _light, label_fill, title_default_fill) = plot_theme();
  let bg_fill = format!("rgb({},{},{})", bg.0, bg.1, bg.2);
  let axis_stroke =
    format!("rgb({},{},{})", axis_gray.0, axis_gray.1, axis_gray.2);

  let font_size = sf * CHART_LABEL_FONT;
  let title_font_size = sf * CHART_TITLE_FONT;
  let char_w = font_size * 0.55;

  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());
  let has_chart_labels = chart_labels.iter().any(|l| !l.text.is_empty());
  let has_value_labels = bar_labels.iter().any(|s| !s.is_empty());
  let (x_axis_label, y_axis_label) = match axes_label {
    Some((x, y)) => (x, y),
    None => ("", ""),
  };

  // Margins. An `AxesLabel` sits at the far end of its axis, so it takes
  // room above (the category axis) and to the right (the value axis).
  let top_margin = if has_plot_label { 38.0 * sf } else { 16.0 * sf }
    + plot_label_extra_lines(plot_label) as f64 * 20.0 * sf
    + if x_axis_label.is_empty() {
      0.0
    } else {
      20.0 * sf
    };
  let bottom_area = 44.0 * sf;

  // Left area: widest category label (capped) plus a rotated y-axis label.
  let max_cat_len = chart_labels
    .iter()
    .map(|l| crate::functions::graphics::box_string_visible_len(&l.text))
    .max()
    .unwrap_or(0);
  let cat_label_area = if has_chart_labels {
    (max_cat_len as f64 * char_w + 14.0 * sf).min(rw * 0.45)
  } else {
    14.0 * sf
  };
  let left_area = cat_label_area;

  // Right area: value labels (drawn past each bar) plus any legend block.
  let max_vlabel_len = bar_labels
    .iter()
    .map(|s| crate::functions::graphics::box_string_visible_len(s))
    .max()
    .unwrap_or(0);
  let value_label_area = if has_value_labels {
    max_vlabel_len as f64 * char_w + 14.0 * sf
  } else {
    0.0
  };
  let legend_area = if chart_legends.is_empty() {
    0.0
  } else {
    let maxlen = chart_legends
      .iter()
      .map(|l| crate::functions::graphics::box_string_visible_len(l))
      .max()
      .unwrap_or(0);
    sf * 12.0 + sf * 6.0 + maxlen as f64 * sf * 10.0 + sf * 16.0
  };
  let axes_label_area = if y_axis_label.is_empty() {
    0.0
  } else {
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(y_axis_label) as f64
        * 0.62
        * font_size
  };
  let right_margin =
    14.0 * sf + value_label_area + legend_area + axes_label_area;

  let plot_x0 = left_area;
  let plot_y0 = top_margin;
  let plot_w = (rw - plot_x0 - right_margin).max(1.0);
  let plot_h = (rh - top_margin - bottom_area).max(1.0);
  let axis_bottom = plot_y0 + plot_h;

  let map_x =
    |v: f64| -> f64 { plot_x0 + (v - x_min) / (x_max - x_min) * plot_w };
  let bar_base = map_x(x_min.max(0.0));

  let mut svg = String::new();
  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {render_width} {render_height}\" \
       preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" \
       viewBox=\"0 0 {render_width} {render_height}\" \
       preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  }
  svg.push_str(&format!(
    "<rect width=\"{render_width}\" height=\"{render_height}\" fill=\"{bg_fill}\"/>\n"
  ));

  // Bars. Category index 0 sits at the bottom band.
  let band_h = plot_h / n.max(1) as f64;
  let gap_frac = 0.2;
  for (gi, group) in groups.iter().enumerate() {
    let band_bottom = axis_bottom - gi as f64 * band_h;
    let band_top = band_bottom - band_h;
    let inner_top = band_top + band_h * gap_frac / 2.0;
    let inner_h = band_h * (1.0 - gap_frac);
    let sub_h = inner_h / k as f64;
    for (bi, &val) in group.iter().enumerate() {
      let (r, g, b) = if !chart_style.is_empty() {
        let idx = if k > 1 { bi } else { gi };
        let c = &chart_style[idx % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else if k > 1 {
        crate::functions::chart::CHART_COLORS
          [bi % crate::functions::chart::CHART_COLORS.len()]
      } else if !chart_legends.is_empty() {
        crate::functions::chart::CHART_COLORS
          [gi % crate::functions::chart::CHART_COLORS.len()]
      } else {
        crate::functions::chart::CHART_COLORS[0]
      };
      let y0 = inner_top + bi as f64 * sub_h;
      let x_end = map_x(val);
      let (rx, rwid) = if x_end >= bar_base {
        (bar_base, x_end - bar_base)
      } else {
        (x_end, bar_base - x_end)
      };
      let tooltip = crate::functions::chart::format_chart_value(val);
      svg.push_str(&format!(
        "<rect x=\"{rx:.2}\" y=\"{y0:.2}\" width=\"{rwid:.2}\" height=\"{sub_h:.2}\" \
         fill=\"rgb({r},{g},{b})\"><title>{tooltip}</title></rect>\n"
      ));
    }
  }

  // Axis lines: left (categories) and bottom (values).
  let stroke_w = sf;
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.2}\" y1=\"{plot_y0:.2}\" x2=\"{plot_x0:.2}\" y2=\"{axis_bottom:.2}\" \
     stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n"
  ));
  let plot_right = plot_x0 + plot_w;
  svg.push_str(&format!(
    "<line x1=\"{plot_x0:.2}\" y1=\"{axis_bottom:.2}\" x2=\"{plot_right:.2}\" y2=\"{axis_bottom:.2}\" \
     stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n"
  ));

  // Value-axis ticks along the bottom.
  let x_major = axis_tick_step(x_max - x_min);
  if x_major > 0.0 {
    let tick_len = MAJOR_TICK_LEN as f64 * sf;
    let start = (x_min / x_major).ceil() * x_major;
    let mut v = start;
    let mut guard = 0;
    while v <= x_max + x_major * 1e-6 && guard < 1000 {
      guard += 1;
      let tx = map_x(v);
      svg.push_str(&format!(
        "<line x1=\"{tx:.2}\" y1=\"{axis_bottom:.2}\" x2=\"{tx:.2}\" y2=\"{:.2}\" \
         stroke=\"{axis_stroke}\" stroke-width=\"{stroke_w:.2}\"/>\n",
        axis_bottom + tick_len
      ));
      svg.push_str(&format!(
        "<text x=\"{tx:.2}\" y=\"{:.2}\" text-anchor=\"middle\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
        axis_bottom + tick_len + font_size,
        html_escape(&format_tick_with_step(v, x_major))
      ));
      v += x_major;
    }
  }

  // Category labels: right-aligned just left of each band.
  if has_chart_labels {
    for (i, label) in chart_labels.iter().enumerate().take(n) {
      if label.text.is_empty() {
        continue;
      }
      let band_center = axis_bottom - (i as f64 + 0.5) * band_h;
      let lx = plot_x0 - 8.0 * sf;
      svg.push_str(&format!(
        "<text x=\"{lx:.2}\" y=\"{:.2}\" text-anchor=\"end\" \
         font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
        band_center + font_size * 0.35,
        crate::functions::graphics::box_string_to_svg(&label.text)
      ));
    }
  }

  // Value labels (LabelingFunction) just past each bar end.
  if has_value_labels {
    let mut flat = 0usize;
    for (gi, group) in groups.iter().enumerate() {
      let band_bottom = axis_bottom - gi as f64 * band_h;
      let band_top = band_bottom - band_h;
      let inner_top = band_top + band_h * gap_frac / 2.0;
      let inner_h = band_h * (1.0 - gap_frac);
      let sub_h = inner_h / k as f64;
      for (bi, &val) in group.iter().enumerate() {
        let text = bar_labels.get(flat).cloned().unwrap_or_default();
        flat += 1;
        if text.is_empty() {
          continue;
        }
        let y0 = inner_top + bi as f64 * sub_h;
        let lx = map_x(val).max(bar_base) + 6.0 * sf;
        svg.push_str(&format!(
          "<text x=\"{lx:.2}\" y=\"{:.2}\" text-anchor=\"start\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{label_fill}\">{}</text>\n",
          y0 + sub_h / 2.0 + font_size * 0.35,
          crate::functions::graphics::box_string_to_svg(&text)
        ));
      }
    }
  }

  // AxesLabel: at the far end of each axis, as everywhere else. `BarOrigin ->
  // Left` transposes the chart, so the *first* entry labels the vertical axis
  // and the second the horizontal one — the swap wolframscript makes too.
  svg.push_str(&axes_label_svg(
    Some(&(y_axis_label.to_string(), x_axis_label.to_string())),
    (plot_x0, plot_y0, plot_w, plot_h),
    (x_min, x_max, 0.0, n as f64),
    (true, true),
    font_size,
    label_fill,
  ));

  // Plot label, centered above the plot.
  if let Some(sl) = plot_label
    && !sl.text.is_empty()
  {
    let cx = plot_x0 + plot_w / 2.0;
    // Place the title near the top so a clear gap remains above the bars.
    let ty = title_font_size * 1.2;
    let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
    // A stacked title keeps its first line at the top and grows downwards
    // into the extra top margin reserved for it.
    let fill = sl
      .color
      .as_ref()
      .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
    let mut style_attrs = String::new();
    if sl.bold {
      style_attrs.push_str(" font-weight=\"bold\"");
    }
    if sl.italic {
      style_attrs.push_str(" font-style=\"italic\"");
    }
    svg.push_str(&format!(
      "<text x=\"{cx:.2}\" y=\"{ty:.2}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{fs:.0}\" fill=\"{fill}\"{style_attrs}>{}</text>\n",
      sl.svg_scaled_stacked(sf, cx, fs * 1.2)
    ));
  }

  // Legends to the right of the plot.
  if !chart_legends.is_empty() {
    let legend_font = sf * 16.0;
    let swatch = sf * 12.0;
    let swatch_gap = sf * 6.0;
    let legend_x = plot_x0 + plot_w + value_label_area + 14.0 * sf;
    let legend_y_start = plot_y0 + sf * 8.0;
    let line_height = sf * 22.0;
    for (i, label) in chart_legends.iter().enumerate() {
      let (cr, cg, cb) = if chart_style.is_empty() {
        crate::functions::chart::CHART_COLORS
          [i % crate::functions::chart::CHART_COLORS.len()]
      } else {
        let c = &chart_style[i % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      };
      let ly = legend_y_start + i as f64 * line_height;
      svg.push_str(&format!(
        "<rect x=\"{legend_x:.2}\" y=\"{ly:.2}\" width=\"{swatch:.0}\" height=\"{swatch:.0}\" \
         fill=\"rgb({cr},{cg},{cb})\"/>\n"
      ));
      svg.push_str(&format!(
        "<text x=\"{:.2}\" y=\"{:.2}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
         fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
        legend_x + swatch + swatch_gap,
        ly + swatch / 2.0,
        crate::functions::graphics::box_string_to_svg(label)
      ));
    }
  }

  svg.push_str("</svg>");
  svg
}

/// Wolfram's `"WL12DefaultVectorGradient"` color scheme — the gradient
/// BubbleHistogram uses to color bubbles by bin count. Four LAB control
/// points (from `ColorData["WL12DefaultVectorGradient", "BlendArgument"]`)
/// are interpolated linearly in LAB space (WL `Blend` semantics), then
/// converted D50-LAB → sRGB with out-of-gamut channels clipped. Matches
/// wolframscript's rendered colors to well within 8-bit quantization.
pub(crate) fn wl_default_vector_gradient(t: f64) -> (u8, u8, u8) {
  // LABColor stores L in [0, 1] and a/b divided by 100.
  const LAB_STOPS: [(f64, f64, f64); 4] = [
    (
      0.188_667_522_250_113_1,
      0.522_596_101_276_680_9,
      -0.857_210_761_912_256_8,
    ),
    (
      0.402_329_034_265_632_8,
      0.522_214_589_645_690_3,
      -0.346_571_396_721_582_05,
    ),
    (
      0.644_456_290_545_989,
      0.537_992_026_017_925,
      0.732_111_567_127_367_9,
    ),
    (
      0.816_796_918_973_305_3,
      0.145_001_147_401_695_45,
      0.828_253_206_343_699_9,
    ),
  ];
  lab_blend(&LAB_STOPS, t)
}

/// Wolfram's `"WL12DefaultVectorDensityGradient"` — the orange-to-yellow
/// background of `StreamDensityPlot`/`VectorDensityPlot`, blended like
/// `wl_default_vector_gradient` (stops from its `"BlendArgument"`).
pub(crate) fn wl_default_vector_density_gradient(t: f64) -> (u8, u8, u8) {
  const LAB_STOPS: [(f64, f64, f64); 2] = [
    (
      0.551_687_138_381_769_2,
      0.385_527_990_328_619_06,
      0.516_444_795_495_070_9,
    ),
    (
      0.935_087_164_500_243_4,
      -0.037_347_741_482_955_39,
      0.546_878_782_824_649_7,
    ),
  ];
  lab_blend(&LAB_STOPS, t)
}

/// Wolfram's `"WL14DefaultDensityGradient"` — the default `ColorFunction`
/// of `DensityPlot`, `ContourPlot` and their list forms (the theme's
/// `"DefaultColorFunction"`): dark purple through pink and orange to cream,
/// blended like `wl_default_vector_gradient` (stops from its
/// `"BlendArgument"`).
pub(crate) fn wl14_default_density_gradient(t: f64) -> (u8, u8, u8) {
  lab_blend(&WL14_DENSITY_LAB_STOPS, t)
}

/// The `LABColor` stops of `"WL14DefaultDensityGradient"`, as
/// `ColorData["WL14DefaultDensityGradient", "BlendArgument"]` gives them.
pub(crate) const WL14_DENSITY_LAB_STOPS: [(f64, f64, f64); 4] = [
  (
    0.237_457_745_487_449_5,
    0.450_232_647_967_430_1,
    -0.293_447_589_199_095_75,
  ),
  (
    0.482_888_916_109_297_85,
    0.701_950_950_198_408_9,
    0.158_500_850_335_412_4,
  ),
  (
    0.722_668_012_769_495_8,
    0.378_364_291_527_790_7,
    0.481_303_807_080_506_7,
  ),
  (
    0.973_524_122_975_542_1,
    -0.052_490_183_706_616_07,
    0.319_941_429_729_375_45,
  ),
];

/// `MatrixPlot`'s default color function (the `"DefaultColorFunction"` of
/// `Charting`ResolvePlotTheme[Automatic, MatrixPlot]`): a LAB blend from
/// blue through near-black at 1/2 to orange.
pub(crate) fn matrix_plot_gradient(t: f64) -> (u8, u8, u8) {
  const LAB_STOPS: [(f64, f64, f64); 9] = [
    (
      0.602_089_947_828_292_8,
      -0.020_256_766_996_109_565,
      -0.594_100_976_643_204_9,
    ),
    (
      0.500_950_689_292_482_5,
      -0.018_349_557_401_535_765,
      -0.513_287_458_202_765_8,
    ),
    (
      0.401_784_938_836_845_3,
      -0.017_877_998_121_300_642,
      -0.430_407_370_784_194_66,
    ),
    (
      0.304_564_431_479_7,
      -0.018_489_247_900_803_64,
      -0.345_981_686_525_546_3,
    ),
    (0.152_794_501_653_630_93, 0.0, 0.0),
    (
      0.353_915_906_341_993_66,
      0.209_580_751_191_310_3,
      0.398_660_281_492_549,
    ),
    (
      0.459_671_728_950_685_4,
      0.283_947_224_061_996_7,
      0.515_293_805_830_302_8,
    ),
    (
      0.563_382_709_272_066_1,
      0.364_302_603_362_989_25,
      0.627_665_078_529_221_5,
    ),
    (
      0.665_829_747_848_021_6,
      0.450_331_088_205_992_44,
      0.737_624_583_210_847_3,
    ),
  ];
  lab_blend(&LAB_STOPS, t)
}

/// `Blend` of `LABColor` stops (L in [0, 1], a/b divided by 100) at `t`:
/// linear in LAB, then converted D50-LAB → sRGB with out-of-gamut channels
/// clipped, as Wolfram renders it.
fn lab_blend(stops: &[(f64, f64, f64)], t: f64) -> (u8, u8, u8) {
  let pos = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
  let i = (pos.floor() as usize).min(stops.len() - 2);
  let f = pos - i as f64;
  let lerp = |a: f64, b: f64| a + (b - a) * f;
  let l = lerp(stops[i].0, stops[i + 1].0) * 100.0;
  let a = lerp(stops[i].1, stops[i + 1].1) * 100.0;
  let b = lerp(stops[i].2, stops[i + 1].2) * 100.0;

  // CIELAB → XYZ, D50 reference white.
  let fy = (l + 16.0) / 116.0;
  let fx = fy + a / 500.0;
  let fz = fy - b / 200.0;
  let f_inv = |c: f64| {
    let c3 = c * c * c;
    if c3 > 216.0 / 24389.0 {
      c3
    } else {
      (116.0 * c - 16.0) / (24389.0 / 27.0)
    }
  };
  let (x, y, z) = (f_inv(fx) * 0.9642, f_inv(fy), f_inv(fz) * 0.8249);

  // XYZ (D50) → linear sRGB (Bradford-adapted ICC matrix), then gamma
  // encoding; channel clipping reproduces WL's gamut handling.
  const M: [[f64; 3]; 3] = [
    [3.133_856_1, -1.616_866_7, -0.490_614_6],
    [-0.978_768_4, 1.916_141_5, 0.033_454_0],
    [0.071_945_3, -0.228_991_4, 1.405_242_7],
  ];
  let encode = |row: &[f64; 3]| -> u8 {
    let v = (row[0] * x + row[1] * y + row[2] * z).clamp(0.0, 1.0);
    let v = if v <= 0.003_130_8 {
      12.92 * v
    } else {
      1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
  };
  (encode(&M[0]), encode(&M[1]), encode(&M[2]))
}

/// Generate SVG for a BubbleChart — a scatter plot with variable-radius
/// circles drawn over labeled x/y axes. Each input triple is `(x, y, z)`
/// where `z` drives the bubble area (matching Mathematica's convention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_bubble_chart_svg(
  groups: &[Vec<(f64, f64, f64)>],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  plot_label: Option<&StyledLabel>,
  axes_label: Option<(&str, &str)>,
  chart_style: &[WoxiColor],
  chart_legends: &[String],
  chart_labels: &[crate::functions::chart::ChartLabel],
  plot_range_x: Option<(f64, f64)>,
  plot_range_y: Option<(f64, f64)>,
  // Data-space bin widths `(dx, dy)` when the bubbles sit on a regular grid
  // (BubbleHistogram). Caps the bubble radius so bubbles in adjacent bins
  // can at most touch, never overlap. `None` for free-form BubbleChart data.
  bin_size: Option<(f64, f64)>,
  // Per-bubble fill colors, indexed by flat position across all groups.
  // When set, the bubbles are drawn without the black edge — matching
  // wolframscript's BubbleHistogram, whose count-gradient bubbles have no
  // border (unlike BubbleChart's).
  bubble_colors: Option<&[(u8, u8, u8)]>,
  // `(min_count, max_count)` when a count-gradient bar legend should be
  // drawn right of the plot (BubbleHistogram's `PlotLegends -> Automatic`,
  // Wolfram's `BarLegend`).
  count_legend: Option<(f64, f64)>,
) -> Result<String, InterpreterError> {
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;
  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  let has_x_axis_label =
    axes_label.as_ref().is_some_and(|(x, _)| !x.is_empty());
  let has_plot_label = plot_label.is_some_and(|sl| !sl.text.is_empty());

  let top_margin = if has_plot_label { 35 * s } else { 10 * s }
    + plot_label_extra_lines(plot_label) as i32 * 20 * s;
  let bottom_extra = if has_x_axis_label { 24.0 * sf } else { 0.0 };
  let x_label_area = 40 * RESOLUTION_SCALE + bottom_extra as u32;
  let y_label_area = 65 * RESOLUTION_SCALE;

  let (bg_color, dark_gray, light_gray, label_fill, title_default_fill) =
    plot_theme();

  let legend_margin_right = if chart_legends.is_empty() {
    10 * s as u32
  } else {
    let max_label_len =
      chart_legends.iter().map(std::string::String::len).max().unwrap_or(0);
    (sf * 12.0 + sf * 6.0 + max_label_len as f64 * sf * 10.0 + sf * 16.0) as u32
  }
  // The count-gradient bar legend occupies a 53-display-pixel strip right
  // of the plot — the widening wolframscript applies (360 → 413 pt).
  + if count_legend.is_some() { 53 * s as u32 } else { 0 };

  // Plot-area dimensions in render-space pixels — must stay in sync with the
  // `ChartBuilder` margins below so that pixel-aware range padding (computed
  // next) keeps bubbles inside the frame.
  let plot_w_px = (render_width as f64
    - 10.0 * sf
    - legend_margin_right as f64
    - y_label_area as f64)
    .max(1.0);
  let plot_h_px = (render_height as f64
    - top_margin as f64
    - 10.0 * sf
    - x_label_area as f64)
    .max(1.0);

  // Auto-compute x/y ranges. Raw min/max first, then expand so the largest
  // possible bubble (`max_bubble_radius` pixels, plus a small visual gap)
  // fits between every data point and the frame. Only applied when the user
  // hasn't supplied an explicit PlotRange — an explicit range is respected
  // verbatim.
  let raw_range = |vals: &[f64]| -> (f64, f64) {
    let mn = vals
      .iter()
      .copied()
      .filter(|v| v.is_finite())
      .fold(f64::INFINITY, f64::min);
    let mx = vals
      .iter()
      .copied()
      .filter(|v| v.is_finite())
      .fold(f64::NEG_INFINITY, f64::max);
    if !mn.is_finite() || !mx.is_finite() {
      return (0.0, 1.0);
    }
    if (mx - mn).abs() < f64::EPSILON {
      // Single distinct value: start with a unit-wide window so the
      // pixel-pad step below has a well-defined span to expand.
      return (mn - 0.5, mx + 0.5);
    }
    (mn, mx)
  };

  // Expand a raw [min, max] so that `pad_px` render-pixels of padding are
  // reserved on each side of the plot area. Solved for the final span with
  // `span_final = span_raw * plot_px / (plot_px - 2 * pad_px)` — after the
  // expansion, data extremes sit exactly `pad_px` pixels from the frame.
  let pad_for_bubbles =
    |min_0: f64, max_0: f64, plot_px: f64, pad_px: f64| -> (f64, f64) {
      let span_0 = max_0 - min_0;
      // Guard: if the plot is too narrow to even fit the padding, skip the
      // expansion rather than blow up or invert the range.
      if span_0 <= 0.0 || plot_px <= 2.0 * pad_px {
        return (min_0, max_0);
      }
      let extra = pad_px * span_0 / (plot_px - 2.0 * pad_px);
      (min_0 - extra, max_0 + extra)
    };

  let xs: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().map(|t| t.0)).collect();
  let ys: Vec<f64> =
    groups.iter().flat_map(|g| g.iter().map(|t| t.1)).collect();
  let (x_min_raw, x_max_raw) = raw_range(&xs);
  let (y_min_raw, y_max_raw) = raw_range(&ys);

  // Max bubble radius in render-space pixels. 20 display pixels matches the
  // previous (axis-less) implementation and stays readable without occluding
  // neighbors at the default image size. For binned data (`bin_size`), the
  // radius is additionally capped at half the pixel distance between
  // adjacent bin centers so bubbles never overlap.
  let max_bubble_radius = {
    let default_r = 20.0 * sf;
    // Half the border stroke width, so even the outlines of two touching
    // max-size bubbles stay disjoint.
    let margin = 0.5 * sf;
    let cap_axis = |bin: f64,
                    raw_span: f64,
                    plot_px: f64,
                    explicit_span: Option<f64>|
     -> f64 {
      match explicit_span {
        // Explicit PlotRange: the data-to-pixel scale is fixed, adjacent
        // bin centers are `bin / span * plot_px` pixels apart.
        Some(span) => bin * plot_px / span.max(f64::EPSILON) / 2.0 - margin,
        // Auto range: the span is later padded by `radius + 4 px` per side
        // (see `pad_for_bubbles`), so the bin's final pixel size is
        // `bin * (plot_px - 2 * (r + 4 sf)) / raw_span`. Solving
        // `2 * (r + margin) <= that` for `r`:
        None => {
          (bin * (plot_px - 8.0 * sf) - 2.0 * margin * raw_span)
            / (2.0 * (raw_span + bin))
        }
      }
    };
    match bin_size {
      None => default_r,
      Some((bx, by)) => {
        let span_of = |range: Option<(f64, f64)>| {
          range.map(|(lo, hi)| if hi <= lo { 1.0 } else { hi - lo })
        };
        let cap_x =
          cap_axis(bx, x_max_raw - x_min_raw, plot_w_px, span_of(plot_range_x));
        let cap_y =
          cap_axis(by, y_max_raw - y_min_raw, plot_h_px, span_of(plot_range_y));
        // Keep at least one display pixel so dense binnings stay visible.
        default_r.min(cap_x).min(cap_y).max(sf)
      }
    }
  };

  // Extra visual gap beyond the radius so bubbles don't visually kiss the
  // frame line.
  let bubble_pad_px = max_bubble_radius + 4.0 * sf;

  let (x_min_auto, x_max_auto) =
    pad_for_bubbles(x_min_raw, x_max_raw, plot_w_px, bubble_pad_px);
  let (y_min_auto, y_max_auto) =
    pad_for_bubbles(y_min_raw, y_max_raw, plot_h_px, bubble_pad_px);
  let (x_min, x_max) = plot_range_x.unwrap_or((x_min_auto, x_max_auto));
  let (y_min, y_max) = plot_range_y.unwrap_or((y_min_auto, y_max_auto));
  let x_max = if x_max <= x_min { x_min + 1.0 } else { x_max };
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };

  // Max |z| across all groups — used to normalize bubble radii so that
  // bubbles are comparable between datasets.
  let z_max = groups
    .iter()
    .flat_map(|g| g.iter().map(|t| t.2.abs()))
    .fold(0.0_f64, f64::max);

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
    })?;

    let tick = MINOR_TICK_LEN * s;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(legend_margin_right)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
      })?;

    let x_major = axis_tick_step(x_max - x_min);
    let y_major = axis_tick_step(y_max - y_min);
    let x_minor_step = minor_tick_step(x_major);
    let y_minor_step = minor_tick_step(y_major);
    let x_tick_count = ((x_max - x_min) / x_minor_step).round() as usize + 1;
    let y_tick_count = ((y_max - y_min) / y_minor_step).round() as usize + 1;

    chart
      .configure_mesh()
      .disable_mesh()
      .x_labels(x_tick_count)
      .y_labels(y_tick_count)
      .x_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, x_major) {
          format_tick_with_step(*v, x_major)
        } else {
          String::new()
        }
      })
      .y_label_formatter(&move |v: &f64| {
        if is_major_tick(*v, y_major) {
          format_tick_with_step(*v, y_major)
        } else {
          String::new()
        }
      })
      .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
      .label_style(("sans-serif", sf * 18.0).into_font().color(&dark_gray))
      .set_tick_mark_size(LabelAreaPosition::Left, tick)
      .set_tick_mark_size(LabelAreaPosition::Bottom, tick)
      .draw()
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
      })?;

    // Origin lines — rendered only when the axis crosses zero.
    let origin_line = light_gray.stroke_width(RESOLUTION_SCALE);
    if y_min < 0.0 && y_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(x_min, 0.0), (x_max, 0.0)],
          origin_line,
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
        })?;
    }
    if x_min < 0.0 && x_max > 0.0 {
      chart
        .draw_series(std::iter::once(PathElement::new(
          vec![(0.0, y_min), (0.0, y_max)],
          origin_line,
        )))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
        })?;
    }

    // Draw the bubbles with pixel-space radii so they look the same
    // regardless of the data range. Colors are assigned per group so that
    // multi-dataset BubbleChart input visually distinguishes the datasets —
    // unless explicit per-bubble colors were supplied (BubbleHistogram's
    // count gradient), which take precedence.
    let mut flat_idx = 0usize;
    for (gi, group) in groups.iter().enumerate() {
      let (cr, cg, cb) = if chart_style.is_empty() {
        crate::functions::chart::CHART_COLORS
          [gi % crate::functions::chart::CHART_COLORS.len()]
      } else {
        let c = &chart_style[gi % chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      };
      let group_fill = RGBColor(cr, cg, cb);
      // Semi-transparent black so the underlying bubble color bleeds
      // through the border, tinting it slightly toward the fill.
      let border = RGBColor(0, 0, 0).mix(0.4).stroke_width(RESOLUTION_SCALE);
      for &(x, y, z) in group {
        let fill = bubble_colors
          .and_then(|cs| cs.get(flat_idx))
          .map_or(group_fill, |&(r, g, b)| RGBColor(r, g, b));
        flat_idx += 1;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
          continue;
        }
        // Area-proportional: radius ∝ sqrt(z / z_max) * max_radius.
        let radius = if z_max > 0.0 {
          ((z.abs() / z_max).sqrt() * max_bubble_radius)
            .max((2.0 * sf).min(max_bubble_radius))
        } else {
          (5.0 * sf).min(max_bubble_radius)
        };
        chart
          .draw_series(std::iter::once(Circle::new(
            (x, y),
            radius as i32,
            fill.filled(),
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
          })?;
        if bubble_colors.is_none() {
          chart
            .draw_series(std::iter::once(Circle::new(
              (x, y),
              radius as i32,
              border,
            )))
            .map_err(|e| {
              InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
            })?;
        }
      }
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("BubbleChart: {e}"))
    })?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Plot area coordinates (must match the margins/areas above).
  let margin_left = 10.0 * sf;
  let margin_top = top_margin as f64;
  let margin_right = legend_margin_right as f64;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = margin_top;
  let plot_w =
    render_width as f64 - margin_left - margin_right - y_label_area as f64;
  let plot_h =
    render_height as f64 - margin_top - 10.0 * sf - x_label_area as f64;
  let axis_y = plot_y0 + plot_h;

  let x_major_step = axis_tick_step(x_max - x_min);
  let y_major_step = axis_tick_step(y_max - y_min);

  inject_major_tick_extensions(
    &mut buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    Some((x_min, x_max, x_major_step)),
    Some((y_min, y_max, y_major_step)),
    MINOR_TICK_LEN as f64 * sf,
    MAJOR_TICK_LEN as f64 * sf,
    sf,
    label_fill,
  );

  // BubbleChart closes the plot rectangle with a top and right axis line
  // (each carrying minor + major tick marks) so the plot area reads as a
  // full frame. Matches Mathematica's default BubbleChart framing.
  inject_top_right_frame(
    &mut buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    Some((x_min, x_max, x_major_step)),
    Some((y_min, y_max, y_major_step)),
    MINOR_TICK_LEN as f64 * sf,
    MAJOR_TICK_LEN as f64 * sf,
    sf,
    label_fill,
  );

  // WL 15 sets chart labels at the tick-label size and the PlotLabel a
  // little larger.
  let font_size = sf * CHART_LABEL_FONT;
  let title_font_size = sf * CHART_TITLE_FONT;

  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();

    if let Some((x_label, y_label)) = &axes_label {
      if !x_label.is_empty() {
        let cx = plot_x0 + plot_w / 2.0;
        let base_y = axis_y + font_size * 2.8;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(y_label)
        ));
      }
    }

    if let Some(sl) = plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      let ty = margin_top - title_font_size * 0.5;
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
      let mut style_attrs = String::new();
      if sl.bold {
        style_attrs.push_str(" font-weight=\"bold\"");
      }
      if sl.italic {
        style_attrs.push_str(" font-style=\"italic\"");
      }
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
           fill=\"{fill}\"{style_attrs}>{}</text>\n",
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    if !chart_legends.is_empty() {
      let legend_font = sf * 16.0;
      let swatch_size = sf * 12.0;
      let swatch_gap = sf * 6.0;
      let legend_x = plot_x0 + plot_w + sf * 16.0;
      let legend_y_start = plot_y0 + sf * 8.0;
      let line_height = sf * 22.0;

      for (i, label) in chart_legends.iter().enumerate() {
        let (cr, cg, cb) = if chart_style.is_empty() {
          crate::functions::chart::CHART_COLORS
            [i % crate::functions::chart::CHART_COLORS.len()]
        } else {
          let c = &chart_style[i % chart_style.len()];
          (
            (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          )
        };
        let ly = legend_y_start + i as f64 * line_height;
        // Bubbles are drawn fully opaque with a black border — mirror that
        // on the swatch so the legend color visually matches the bubbles.
        labels_svg.push_str(&format!(
          "<rect x=\"{legend_x:.1}\" y=\"{ly:.1}\" width=\"{swatch_size:.0}\" height=\"{swatch_size:.0}\" \
           fill=\"rgb({cr},{cg},{cb})\" stroke=\"#000000\" stroke-width=\"1\"/>\n"
        ));
        labels_svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"{legend_font:.0}\" \
           fill=\"{label_fill}\" dominant-baseline=\"central\">{}</text>\n",
          legend_x + swatch_size + swatch_gap,
          ly + swatch_size / 2.0,
          crate::functions::graphics::box_string_to_svg(label)
        ));
      }
    }

    // Count-gradient bar legend (BubbleHistogram's PlotLegends ->
    // Automatic): a vertical WL12DefaultVectorGradient bar right of the
    // plot, smallest count at the bottom, labels at nice count steps —
    // Wolfram's BarLegend. Layout proportions match the wolframscript
    // rendering (bar ≈ 13 pt wide, ≈ 5/8 of the plot height, vertically
    // centered).
    if let Some((c_lo, c_hi)) = count_legend {
      let bar_w = 13.0 * sf;
      let bar_h = plot_h * 0.625;
      let bar_x = plot_x0 + plot_w + 27.0 * sf;
      let bar_y = plot_y0 + (plot_h - bar_h) / 2.0;
      let slices = 64usize;
      let slice_h = bar_h / slices as f64;
      for i in 0..slices {
        // t = 1 at the top slice, 0 at the bottom.
        let t = 1.0 - (i as f64 + 0.5) / slices as f64;
        let (r, g, b) = wl_default_vector_gradient(t);
        labels_svg.push_str(&format!(
          "<rect x=\"{bar_x:.1}\" y=\"{:.1}\" width=\"{bar_w:.1}\" \
           height=\"{:.1}\" fill=\"rgb({r},{g},{b})\"/>\n",
          bar_y + slice_h * i as f64,
          slice_h + 0.6 * sf
        ));
      }
      labels_svg.push_str(&format!(
        "<rect x=\"{bar_x:.1}\" y=\"{bar_y:.1}\" width=\"{bar_w:.1}\" \
         height=\"{bar_h:.1}\" fill=\"none\" stroke=\"rgb(110,110,110)\" \
         stroke-width=\"{:.1}\"/>\n",
        0.8 * sf
      ));
      let legend_font = sf * 16.0;
      let label_x = bar_x + bar_w + 6.0 * sf;
      let label_at = |labels_svg: &mut String, value: f64, frac: f64| {
        labels_svg.push_str(&format!(
          "<text x=\"{label_x:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
           font-size=\"{legend_font:.0}\" fill=\"{label_fill}\" \
           dominant-baseline=\"central\">{}</text>\n",
          bar_y + bar_h * (1.0 - frac),
          format_tick(value)
        ));
      };
      if c_hi > c_lo {
        // Six target labels reproduces wolframscript's BarLegend ticks
        // (step 2 for counts 1..12, step 5 for 1..25).
        let step = nice_step(c_hi - c_lo, 6);
        let mut v = (c_lo / step).ceil() * step;
        while v <= c_hi + 1e-9 {
          label_at(&mut labels_svg, v, (v - c_lo) / (c_hi - c_lo));
          v += step;
        }
      } else {
        // Degenerate range (all bins hold the same count): label the bar
        // once at its center.
        label_at(&mut labels_svg, c_lo, 0.5);
      }
    }

    // ChartLabels — draw each label centered on its bubble. Labels map to
    // points in input order, flattened across datasets.
    if !chart_labels.is_empty() {
      let bubble_label_font = sf * 16.0;
      let x_span = x_max - x_min;
      let y_span = y_max - y_min;
      let mut idx = 0usize;
      'outer: for group in groups {
        for &(x, y, _z) in group {
          if idx >= chart_labels.len() {
            break 'outer;
          }
          let label = &chart_labels[idx];
          idx += 1;
          if !x.is_finite() || !y.is_finite() || label.text.is_empty() {
            continue;
          }
          let px = plot_x0 + (x - x_min) / x_span * plot_w;
          let py = plot_y0 + (y_max - y) / y_span * plot_h;
          let transform = if label.rotation.abs() > f64::EPSILON {
            // ChartLabel rotation is in radians; SVG rotates clockwise for
            // positive degrees, so negate to match Wolfram's convention
            // (positive radians = counter-clockwise).
            let deg = -label.rotation.to_degrees();
            format!(" transform=\"rotate({deg:.2},{px:.1},{py:.1})\"")
          } else {
            String::new()
          };
          // Bubble labels use full black (not the axis label gray) so
          // they stay legible against the colored fill.
          labels_svg.push_str(&format!(
            "<text x=\"{px:.1}\" y=\"{py:.1}\" text-anchor=\"middle\" \
             dominant-baseline=\"central\" font-family=\"sans-serif\" \
             font-size=\"{bubble_label_font:.0}\" fill=\"#000000\"\
             {transform}>{}</text>\n",
            crate::functions::graphics::box_string_to_svg(&label.text)
          ));
        }
      }
    }

    buf.insert_str(insert_pos, &labels_svg);
  }

  Ok(buf)
}

/// Escape special characters for SVG/HTML text content.
pub(crate) fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// Width (in viewBox units) of the plotting area, mirroring the margins used
/// by `generate_svg_with_options`. Used to scale dashed legend swatches so the
/// dash pattern matches the on-chart line (which dashes at `d * plot_w`).
fn legend_plot_area_width(vb_w: f64, opts: &PlotOptions) -> f64 {
  let sf = RESOLUTION_SCALE as f64;
  let show_y_axis = opts.axes.1;
  let y_label_area = if !show_y_axis {
    0.0
  } else if !opts.ticks_visible() {
    5.0 * sf
  } else {
    65.0 * sf
  };
  let margin_left = if show_y_axis { 10.0 * sf } else { 5.0 * sf };
  let has_right_label = opts
    .frame_label_right
    .as_ref()
    .is_some_and(|t| !t.is_empty());
  let margin_right = if has_right_label {
    40.0 * sf
  } else {
    10.0 * sf
  };
  (vb_w - margin_left - margin_right - y_label_area).max(vb_w * 0.5)
}

/// The legend swatch of series `si` centred vertically on `y` and starting
/// at `x`: a short line for a joined series, the point marker for one
/// drawn as points (a list plot's `PointLegend`).
fn legend_swatch_svg(
  opts: &PlotOptions,
  entry: usize,
  si: usize,
  x: f64,
  y: f64,
  swatch_len: f64,
  dash_scale: f64,
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let (r, g, b) = opts
    .legend_colors
    .get(entry)
    .copied()
    .flatten()
    .unwrap_or_else(|| series_color(&opts.plot_style, si));
  let as_points = match opts.legend_marker_kind {
    LegendMarkerKind::Swatch => {
      // A filled square, its right edge where a line swatch would end.
      let side = sf * 10.0;
      return format!(
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{side:.1}\" \
         height=\"{side:.1}\" fill=\"rgb({r},{g},{b})\" \
         stroke=\"rgb({},{},{})\" stroke-width=\"{:.1}\"/>\n",
        x + swatch_len - side,
        y - side / 2.0,
        (r as f64 * 0.67) as u8,
        (g as f64 * 0.67) as u8,
        (b as f64 * 0.67) as u8,
        sf * 0.5,
      );
    }
    LegendMarkerKind::Point => true,
    LegendMarkerKind::Line => false,
    LegendMarkerKind::Automatic => {
      !resolve_series_joined(opts.joined_per_series.as_deref(), si)
    }
  };
  if as_points {
    return format!(
      "<circle cx=\"{:.1}\" cy=\"{y:.1}\" r=\"{:.1}\" fill=\"rgb({r},{g},{b})\"/>\n",
      x + swatch_len / 2.0,
      sf * 2.5,
    );
  }
  let thickness = series_thickness(&opts.plot_style, si);
  let dashing = series_dashing(&opts.plot_style, si);
  let sw = (thickness as f64 / RESOLUTION_SCALE as f64 * sf).max(sf);
  let mut dash_attr = String::new();
  if let Some(ref pattern) = dashing {
    let dash_vals: Vec<String> = pattern
      .iter()
      .map(|d| format!("{:.1}", dash_len(*d, dash_scale, sf)))
      .collect();
    dash_attr = format!(" stroke-dasharray=\"{}\"", dash_vals.join(","));
  }
  format!(
    "<line x1=\"{x:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" \
     stroke=\"rgb({r},{g},{b})\" stroke-width=\"{}\"{dash_attr}/>\n",
    x + swatch_len,
    sw as u32,
  )
}

/// Inject a legend into an SVG plot. Depending on `legend_position`, the legend
/// is placed on the right (default), top, or bottom of the plot.
fn inject_legend(buf: &mut String, opts: &PlotOptions) {
  if opts.plot_legends.is_empty() {
    return;
  }

  let sf = RESOLUTION_SCALE as f64;
  let (_bg_color, _dark_gray, _light_gray, label_fill, _title_fill) =
    plot_theme();

  let font_size = sf * 14.0;
  let line_height = font_size * 1.6;
  let swatch_len = sf * 20.0;
  let swatch_gap = sf * 6.0;
  let legend_padding = sf * 10.0;

  // Parse current viewBox
  let vb_re = regex::Regex::new(
    r#"viewBox="(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)""#,
  )
  .unwrap();

  let (vb_w, vb_h) = if let Some(caps) = vb_re.captures(buf) {
    let w: f64 = caps[3].parse().unwrap_or(0.0);
    let h: f64 = caps[4].parse().unwrap_or(0.0);
    (w, h)
  } else {
    return;
  };

  // The on-chart dashed lines are drawn with a dash length of `d * plot_w`
  // (a fraction of the plotting-area width — see `draw_dashed_line`). Scale the
  // legend swatch's stroke-dasharray by the same width so the two match.
  let dash_scale = legend_plot_area_width(vb_w, opts);

  match opts.legend_position {
    LegendPosition::Top | LegendPosition::Bottom => {
      // Horizontal legend: entries share a row and wrap onto further rows
      // once they no longer fit the plot width.
      // Calculate per-entry widths for horizontal layout
      let entry_widths: Vec<f64> = opts
        .plot_legends
        .iter()
        .map(|s| {
          swatch_len
            + swatch_gap
            + crate::functions::graphics::box_string_visible_len(s) as f64
              * font_size
              * 0.55
        })
        .collect();
      let entry_spacing = legend_padding;

      let mut rows: Vec<Vec<usize>> = Vec::new();
      let mut row_w = 0.0;
      for (i, w) in entry_widths.iter().enumerate() {
        let needed = if row_w > 0.0 { entry_spacing + w } else { *w };
        if row_w > 0.0 && row_w + needed > vb_w {
          rows.push(Vec::new());
          row_w = 0.0;
        }
        if rows.is_empty() {
          rows.push(Vec::new());
        }
        row_w += if row_w > 0.0 { entry_spacing + w } else { *w };
        rows.last_mut().unwrap().push(i);
      }
      let legend_height = rows.len() as f64 * line_height + legend_padding;

      let new_vb_h = vb_h + legend_height;

      // Update viewBox height
      let old_vb = format!("viewBox=\"0 0 {} {}\"", vb_w as u32, vb_h as u32);
      let new_vb =
        format!("viewBox=\"0 0 {} {}\"", vb_w as u32, new_vb_h as u32);
      *buf = buf.replacen(&old_vb, &new_vb, 1);

      // Update height attribute if present
      let h_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
      if let Some(hcaps) = h_re.captures(&buf.clone()) {
        let old_h: u32 = hcaps[1].parse().unwrap_or(0);
        if old_h > 0 {
          let new_display_h = (old_h as f64 * new_vb_h / vb_h).round() as u32;
          let old_hattr = format!("height=\"{old_h}\"");
          let new_hattr = format!("height=\"{new_display_h}\"");
          *buf = buf.replacen(&old_hattr, &new_hattr, 1);
        }
      }

      // For Top: shift existing content down and draw legend at top
      // For Bottom: draw legend at the bottom
      let legend_y = if opts.legend_position == LegendPosition::Top {
        // Shift existing content down by wrapping in a translate group
        // Find end of opening <svg ...> tag
        if let Some(svg_tag_end) = buf.find('>') {
          let after_tag = svg_tag_end + 1;
          let shift_open =
            format!("<g transform=\"translate(0,{})\">", legend_height as u32);
          buf.insert_str(after_tag, &shift_open);
          // Insert closing </g> before </svg>
          if let Some(close_pos) = buf.rfind("</svg>") {
            buf.insert_str(close_pos, "</g>");
          }
        }
        legend_padding * 0.5 + line_height * 0.5
      } else {
        // Bottom: legend goes after existing content
        vb_h + legend_padding * 0.5 + line_height * 0.5
      };

      // Draw legend entries horizontally
      if let Some(insert_pos) = buf.rfind("</svg>") {
        let mut legend_svg = String::new();
        let first_row_y = legend_y;

        for (row_idx, row) in rows.iter().enumerate() {
          let legend_y = first_row_y + row_idx as f64 * line_height;
          // Center each legend row within the viewBox width
          let total_w: f64 = row.iter().map(|&i| entry_widths[i]).sum::<f64>()
            + entry_spacing * (row.len().max(1) - 1) as f64;
          let mut cursor_x = (vb_w - total_w).max(0.0) / 2.0;

          for &i in row {
            let label = &opts.plot_legends[i];
            let si = opts.legend_series.get(i).copied().unwrap_or(i);
            legend_svg.push_str(&legend_swatch_svg(
              opts, i, si, cursor_x, legend_y, swatch_len, dash_scale,
            ));

            // Text label
            let text_x = cursor_x + swatch_len + swatch_gap;
            let text_y = legend_y + font_size * 0.35;
            legend_svg.push_str(&format!(
              "<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
              crate::functions::graphics::box_string_to_svg_scaled(
                label,
                RESOLUTION_SCALE as f64
              ),
            ));

            cursor_x += entry_widths[i] + entry_spacing;
          }
        }

        buf.insert_str(insert_pos, &legend_svg);
      }
    }
    LegendPosition::Right => {
      // Original right-side legend behavior
      let max_text_width = opts
        .plot_legends
        .iter()
        .map(|s| {
          crate::functions::graphics::box_string_visible_len(s) as f64
            * font_size
            * 0.55
        })
        .fold(0.0_f64, f64::max);
      let legend_width =
        swatch_len + swatch_gap + max_text_width + legend_padding;

      let new_vb_w = vb_w + legend_width;

      // Update viewBox width
      let old_vb = format!("viewBox=\"0 0 {} {}\"", vb_w as u32, vb_h as u32);
      let new_vb =
        format!("viewBox=\"0 0 {} {}\"", new_vb_w as u32, vb_h as u32);
      *buf = buf.replacen(&old_vb, &new_vb, 1);

      // Update width attribute if present (non-full-width)
      let w_re = regex::Regex::new(r#"width="(\d+)""#).unwrap();
      if let Some(caps) = w_re.captures(&buf.clone()) {
        let old_w: u32 = caps[1].parse().unwrap_or(0);
        if old_w > 0 {
          let render_w = vb_w as u32;
          let render_new_w = new_vb_w as u32;
          let new_display_w = (old_w as f64 * render_new_w as f64
            / render_w as f64)
            .round() as u32;
          let old_attr = format!("width=\"{old_w}\"");
          let new_attr = format!("width=\"{new_display_w}\"");
          *buf = buf.replacen(&old_attr, &new_attr, 1);

          let h_re = regex::Regex::new(r#"height="(\d+)""#).unwrap();
          if let Some(hcaps) = h_re.captures(&buf.clone()) {
            let old_h: u32 = hcaps[1].parse().unwrap_or(0);
            if old_h > 0 {
              let new_display_h =
                (new_display_w as f64 * vb_h / new_vb_w).round() as u32;
              let old_hattr = format!("height=\"{old_h}\"");
              let new_hattr = format!("height=\"{new_display_h}\"");
              *buf = buf.replacen(&old_hattr, &new_hattr, 1);
            }
          }
        }
      }

      // Insert legend elements before </svg>
      if let Some(insert_pos) = buf.rfind("</svg>") {
        let mut legend_svg = String::new();
        let legend_x = vb_w + legend_padding * 0.5;
        let n = opts.plot_legends.len();
        let legend_total_h = n as f64 * line_height;
        let legend_y0 = (vb_h - legend_total_h) / 2.0;

        for (i, label) in opts.plot_legends.iter().enumerate() {
          let si = opts.legend_series.get(i).copied().unwrap_or(i);
          let y = legend_y0 + i as f64 * line_height + line_height * 0.5;
          legend_svg.push_str(&legend_swatch_svg(
            opts, i, si, legend_x, y, swatch_len, dash_scale,
          ));

          let text_x = legend_x + swatch_len + swatch_gap;
          let text_y = y + font_size * 0.35;
          legend_svg.push_str(&format!(
            "<text x=\"{text_x:.1}\" y=\"{text_y:.1}\" \
             font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
             fill=\"{label_fill}\">{}</text>\n",
            crate::functions::graphics::box_string_to_svg_scaled(
              label,
              RESOLUTION_SCALE as f64
            ),
          ));
        }

        buf.insert_str(insert_pos, &legend_svg);
      }
    }
  }
}

/// Generate SVG for a histogram using plotters.
/// Specifies how histogram bins are determined.
#[derive(Clone)]
pub enum BinSpec {
  /// A fixed number of equal-width bins.
  Count(usize),
  /// A fixed bin width; bin boundaries are aligned to multiples of the width.
  Width(f64),
  /// Explicit bin edges (must be sorted, at least 2 elements).
  Edges(Vec<f64>),
}

/// Histogram bar-height specification (Wolfram's `hspec`).
#[derive(Clone, Copy, PartialEq)]
pub enum HistogramHeight {
  /// Raw counts per bin (the default).
  Count,
  /// Running total of counts up to and including each bin.
  CumulativeCount,
  /// Fraction of samples in each bin (`c / n`); heights sum to 1.
  Probability,
  /// Probability density (`c / (n * width)`); total area is 1.
  Pdf,
  /// Cumulative fraction of samples (`cumsum / n`); reaches 1 at the last bin.
  Cdf,
}

/// Compute the common bin edges for one or more datasets given a bin spec.
pub(crate) fn histogram_bin_edges(
  all_values: &[f64],
  bin_spec: Option<&BinSpec>,
) -> Vec<f64> {
  if let Some(BinSpec::Edges(edges)) = bin_spec {
    let mut edges = edges.clone();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    return edges;
  }

  let d_min = all_values.iter().copied().fold(f64::INFINITY, f64::min);
  let d_max = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let (d_min, d_max) = if d_min.is_finite() && d_max.is_finite() {
    (d_min, d_max)
  } else {
    (0.0, 1.0)
  };

  // A bare bin count uses wolframscript's `userBinningN` placement,
  // shared with `HistogramList[data, n]`.
  if let Some(BinSpec::Count(c)) = bin_spec
    && *c > 0
  {
    if let Some(bins) = crate::functions::list_helpers_ast::wl_user_binning_n(
      all_values, *c as i128,
    ) {
      return bins.f64s;
    }
    let bin_width = if (d_max - d_min).abs() < f64::EPSILON {
      1.0
    } else {
      (d_max - d_min) / *c as f64
    };
    return (0..=*c).map(|i| d_min + i as f64 * bin_width).collect();
  }
  if all_values.is_empty() {
    return vec![0.0, 1.0];
  }
  // Automatic bins and a bin width `{w}` are placed exactly as
  // `HistogramList` places them.
  let width = match bin_spec {
    Some(BinSpec::Width(w)) if *w > 0.0 => Some(*w),
    _ => None,
  };
  let (lo, hi, dx, _) =
    crate::functions::list_helpers_ast::wl_bin_spec(all_values, width);
  let n_bins = (((hi - lo) / dx + 1e-9).floor() as usize).max(1);
  (0..=n_bins).map(|i| lo + i as f64 * dx).collect()
}

/// Count how many values of `values` fall into each bin defined by `edges`.
/// The last bin is closed on the right; all others are half-open `[lo, hi)`.
fn histogram_bin_counts(values: &[f64], edges: &[f64]) -> Vec<usize> {
  let num_bins = edges.len().saturating_sub(1);
  let mut counts = vec![0usize; num_bins];
  if num_bins == 0 {
    return counts;
  }
  for &v in values {
    for i in 0..num_bins {
      let in_bin = if i == num_bins - 1 {
        v >= edges[i] && v <= edges[i + 1]
      } else {
        v >= edges[i] && v < edges[i + 1]
      };
      if in_bin {
        counts[i] += 1;
        break;
      }
    }
  }
  counts
}

/// Convert per-bin counts into bar heights according to a height spec.
fn histogram_bin_heights(
  counts: &[usize],
  edges: &[f64],
  height: HistogramHeight,
) -> Vec<f64> {
  let n: usize = counts.iter().sum();
  let n_f = n.max(1) as f64;
  match height {
    HistogramHeight::Count => counts.iter().map(|&c| c as f64).collect(),
    HistogramHeight::Probability => {
      counts.iter().map(|&c| c as f64 / n_f).collect()
    }
    HistogramHeight::Pdf => counts
      .iter()
      .enumerate()
      .map(|(i, &c)| {
        let w = (edges[i + 1] - edges[i]).abs();
        if w > 0.0 { c as f64 / (n_f * w) } else { 0.0 }
      })
      .collect(),
    HistogramHeight::CumulativeCount => {
      let mut acc = 0.0;
      counts
        .iter()
        .map(|&c| {
          acc += c as f64;
          acc
        })
        .collect()
    }
    HistogramHeight::Cdf => {
      let mut acc = 0.0;
      counts
        .iter()
        .map(|&c| {
          acc += c as f64;
          acc / n_f
        })
        .collect()
    }
  }
}

/// Build a `PlotSource` describing histogram bars as one filled step-outline
/// series per dataset. This lets `Show` overlay a Histogram with plots
/// (`Show[Histogram[...], Plot[...]]`) in a shared coordinate system, exactly
/// as Wolfram does — the bars and any overlaid curves share the same axes.
pub(crate) fn histogram_plot_source(
  datasets: &[Vec<f64>],
  bin_spec: Option<&BinSpec>,
  height: HistogramHeight,
  colors: &[(u8, u8, u8)],
  image_size: (u32, u32),
) -> Option<crate::syntax::PlotSource> {
  let all_values: Vec<f64> =
    datasets.iter().flat_map(|d| d.iter().copied()).collect();
  if all_values.is_empty() {
    return None;
  }
  let bin_edges = histogram_bin_edges(&all_values, bin_spec);
  let num_bins = bin_edges.len().saturating_sub(1);
  if num_bins == 0 {
    return None;
  }

  let mut series = Vec::with_capacity(datasets.len());
  let mut max_h = 0.0_f64;
  for (di, d) in datasets.iter().enumerate() {
    let counts = histogram_bin_counts(d, &bin_edges);
    let heights = histogram_bin_heights(&counts, &bin_edges, height);
    // Trace every bar — up its left edge, across its top, down its right
    // edge — so the outline also separates neighbouring bars. Filling to
    // the axis paints the bars.
    let mut points = Vec::with_capacity(num_bins * 4);
    for i in 0..num_bins {
      points.push((bin_edges[i], 0.0));
      points.push((bin_edges[i], heights[i]));
      points.push((bin_edges[i + 1], heights[i]));
      points.push((bin_edges[i + 1], 0.0));
      max_h = max_h.max(heights[i]);
    }
    let color = colors.get(di).copied().unwrap_or(
      crate::functions::chart::CHART_COLORS
        [di % crate::functions::chart::CHART_COLORS.len()],
    );
    // WL 15 bars: the chart color as the face (translucent when datasets
    // overlap) under a thin, translucent gray edge.
    series.push(crate::syntax::PlotSeriesData {
      points,
      color: (128, 128, 128),
      color_expr: None,
      is_scatter: false,
      filling: crate::syntax::SeriesFilling::Axis,
      fill_color: Some(color),
      fill_opacity: Some(if datasets.len() > 1 { 0.7 } else { 1.0 }),
      marker: None,
      thickness: Some(0.5),
      point_size: None,
      dashing: None,
      opacity: Some(0.66),
      labels: Vec::new(),
      mesh_full_points: Vec::new(),
    });
  }

  let y_hi = if max_h > 0.0 { max_h * 1.05 } else { 1.0 };
  Some(crate::syntax::PlotSource {
    series,
    x_range: (bin_edges[0], bin_edges[num_bins]),
    y_range: (0.0, y_hi),
    image_size,
    options: Vec::new(),
    log_x: false,
    log_y: false,
    plot_range: None,
  })
}

pub(crate) fn generate_histogram_svg(
  datasets: &[Vec<f64>],
  bin_spec: Option<&BinSpec>,
  height: HistogramHeight,
  opts: &mut ChartOptions,
) -> Result<String, InterpreterError> {
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  // Shared bin edges spanning every dataset, then per-dataset counts and
  // bar heights (heights depend on the height spec: Count, PDF, …).
  let all_values: Vec<f64> =
    datasets.iter().flat_map(|d| d.iter().copied()).collect();
  let bin_edges = histogram_bin_edges(&all_values, bin_spec);
  let num_bins = bin_edges.len().saturating_sub(1);
  if num_bins == 0 {
    return Ok("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string());
  }
  let dataset_counts: Vec<Vec<usize>> = datasets
    .iter()
    .map(|d| histogram_bin_counts(d, &bin_edges))
    .collect();
  let dataset_heights: Vec<Vec<f64>> = dataset_counts
    .iter()
    .map(|c| histogram_bin_heights(c, &bin_edges, height))
    .collect();

  // A named ChartStyle scheme colors each bin with a distinct sampled color.
  if let [heights] = dataset_heights.as_slice() {
    crate::functions::chart::apply_color_function(opts, heights);
  }
  crate::functions::chart::apply_color_scheme(opts, num_bins);
  let max_height = dataset_heights
    .iter()
    .flat_map(|h| h.iter().copied())
    .fold(0.0_f64, f64::max);
  let max_height = if max_height > 0.0 { max_height } else { 1.0 };

  // Explicit PlotRange overrides the auto-computed extents: the bins and
  // 0..tallest bar, padded as WL 15's
  // `PlotRangePadding -> {{Scaled[0.02], Scaled[0.02]}, {Scaled[0.02],
  // Scaled[0.05]}}` pads them (fractions of the padded span).
  let scaled_pad = |lo: f64, hi: f64, below: f64, above: f64| {
    let span = (hi - lo) / (1.0 - below - above);
    (lo - below * span, hi + above * span)
  };
  let (y_min, y_max) = opts
    .plot_range_y
    .unwrap_or_else(|| scaled_pad(0.0, max_height, 0.02, 0.05));
  let y_max = if y_max <= y_min { y_min + 1.0 } else { y_max };
  let (x_lo, x_hi) = opts.plot_range_x.unwrap_or_else(|| {
    scaled_pad(bin_edges[0], bin_edges[num_bins], 0.02, 0.02)
  });
  let x_hi = if x_hi <= x_lo { x_lo + 1.0 } else { x_hi };

  let s = RESOLUTION_SCALE as i32;
  let sf = RESOLUTION_SCALE as f64;

  // Extra room for a PlotLabel above and an x-axis label below.
  let has_plot_label = opts
    .plot_label
    .as_ref()
    .is_some_and(|sl| !sl.text.is_empty());
  // A `FrameLabel` takes room below/left; an `AxesLabel` sits at the far end
  // of its axis, so it takes room above and to the right instead.
  let has_x_frame_label = opts
    .frame_label
    .as_ref()
    .is_some_and(|(x, _)| !x.is_empty());
  let axes_label_x = opts
    .axes_label
    .as_ref()
    .map(|(x, _)| x.as_str())
    .filter(|x| !x.is_empty());
  let has_y_axes_label =
    opts.axes_label.as_ref().is_some_and(|(_, y)| !y.is_empty());
  let top_margin = if has_plot_label { 35 * s } else { 6 * s }
    + plot_label_extra_lines(opts.plot_label.as_ref()) as i32 * 20 * s
    + if has_y_axes_label { 20 * s } else { 0 };
  let bottom_extra = if has_x_frame_label { 24.0 * sf } else { 0.0 };
  let has_y_frame_label = opts
    .frame_label
    .as_ref()
    .is_some_and(|(_, y)| !y.is_empty());
  // The axes are drawn the way an unframed `Plot` draws them
  // (`plot_axes::origin_axes_svg`): marks pointing into the chart, the x
  // labels 3 pt below the axis and the y labels 3 pt left of it.
  let x_scale = crate::functions::plot_axes::AxisScale {
    min: x_lo,
    max: x_hi,
    log: false,
  };
  let y_scale = crate::functions::plot_axes::AxisScale {
    min: y_min,
    max: y_max,
    log: false,
  };
  let x_ticks = crate::functions::plot_axes::AxisTicks::new(
    x_scale,
    false,
    opts.ticks_x.as_ref(),
  );
  let y_ticks = crate::functions::plot_axes::AxisTicks::new(
    y_scale,
    false,
    opts.ticks_y.as_ref(),
  );
  let tick_room = |ticks: &crate::functions::plot_axes::AxisTicks| {
    ticks
      .majors
      .iter()
      .map(|(_, l)| plot_padding::tick_label_width(l, plot_padding::TICK_FONT))
      .fold(0.0, f64::max)
  };
  let x_label_area = if opts.ticks {
    ((plot_padding::TICK_LABEL_GAP
      + plot_padding::line_height(plot_padding::TICK_FONT))
      * sf) as u32
  } else {
    0
  } + bottom_extra as u32;
  let y_label_area = if opts.ticks {
    ((tick_room(&y_ticks) + plot_padding::TICK_LABEL_GAP) * sf) as u32
  } else {
    0
  } + if has_y_frame_label {
    (24.0 * sf) as u32
  } else {
    0
  };
  // The last x label may hang past the right end of the axis.
  let x_overhang = x_ticks
    .majors
    .last()
    .filter(|_| opts.ticks)
    .map_or(0.0, |(_, l)| {
      plot_padding::tick_label_width(l, plot_padding::TICK_FONT) / 2.0
    });
  let axes_label_margin_right = axes_label_x.map_or(0.0, |label| {
    // The label is drawn at `font_size`; ~0.62 em per character covers the
    // proportional font with a little slack.
    8.0 * sf
      + crate::functions::graphics::box_string_visible_len(label) as f64
        * 0.62
        * sf
        * 18.0
  }) as u32;

  // `ChartLegends` / `PlotLegends`: a column of swatches to the right.
  let legend_margin_right = if opts.chart_legends.is_empty() {
    0
  } else {
    chart_legend_column_width(&opts.chart_legends, sf) as u32
  };
  let (bg_color, _dark_gray, _light_gray, label_fill, title_default_fill) =
    plot_theme();

  let side_margin = 2.0 * sf;
  let right_margin = ((x_overhang + 2.0) * sf).max(side_margin) as u32;

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;

    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin as u32)
      .margin_right(
        right_margin + axes_label_margin_right + legend_margin_right,
      )
      .margin_bottom(side_margin as u32)
      .margin_left(side_margin as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_lo..x_hi, y_min..y_max)
      .map_err(|e| {
        InterpreterError::EvaluationError(format!("Histogram: {e}"))
      })?;

    // Draw contiguous histogram bars using plotters Rectangles.
    // ChartStyle colors cycle per bin (matching BarChart's per-element
    // convention); the default is a single uniform color.
    // With one dataset, ChartStyle colors cycle per bin (matching BarChart);
    // the default is a single uniform color. With several datasets, each is
    // drawn in its own color and overlaid semi-transparently so overlapping
    // bars stay visible.
    let multi = dataset_heights.len() > 1;
    let style_color = |idx: usize| -> (u8, u8, u8) {
      if !opts.chart_style.is_empty() {
        let c = &opts.chart_style[idx % opts.chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      } else if multi {
        crate::functions::chart::CHART_COLORS
          [idx % crate::functions::chart::CHART_COLORS.len()]
      } else {
        crate::functions::chart::CHART_COLORS[0]
      }
    };
    for (di, heights) in dataset_heights.iter().enumerate() {
      for (i, &h) in heights.iter().enumerate() {
        let (r, g, b) = if multi {
          style_color(di)
        } else {
          style_color(i)
        };
        let color = RGBColor(r, g, b);
        let bx0 = bin_edges[i];
        let bx1 = bin_edges[i + 1];
        // Overlaid datasets are translucent: WL 15's first chart style is
        // `RGBColor[1, 0.8, 0]` at opacity 0.8, the others their color at
        // opacity 0.7.
        let fill: ShapeStyle = if !multi {
          color.filled()
        } else if di == 0 && opts.chart_style.is_empty() {
          RGBColor(255, 204, 0).mix(0.8).filled()
        } else {
          color.mix(0.7).filled()
        };
        chart
          .draw_series(std::iter::once(Rectangle::new(
            [(bx0, 0.0), (bx1, h)],
            fill,
          )))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("Histogram: {e}"))
          })?;
      }
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("Histogram: {e}"))
    })?;
  }

  add_bar_borders(&mut buf, RESOLUTION_SCALE);

  // Inject hover tooltips with bin range and count into histogram rects.
  // Order matches the draw order above (dataset-major, then bin).
  let hist_tooltips: Vec<String> = dataset_counts
    .iter()
    .flat_map(|counts| {
      counts.iter().enumerate().map(|(i, &c)| {
        let lo = format_tooltip_value(bin_edges[i]);
        let hi = format_tooltip_value(bin_edges[i + 1]);
        format!("[{lo}, {hi}): {c}")
      })
    })
    .collect();
  inject_bar_tooltips_str(&mut buf, &hist_tooltips);

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Plot area coordinates (kept in sync with the ChartBuilder margins above).
  let margin_left = side_margin;
  let plot_x0 = margin_left + y_label_area as f64;
  let plot_y0 = top_margin as f64;
  let plot_w = render_width as f64
    - margin_left
    - right_margin as f64
    - axes_label_margin_right as f64
    - legend_margin_right as f64
    - y_label_area as f64;
  let plot_h = render_height as f64
    - top_margin as f64
    - side_margin
    - x_label_area as f64;

  {
    let axes_svg = crate::functions::plot_axes::origin_axes_svg(
      &crate::functions::plot_axes::OriginAxes {
        area: (plot_x0, plot_y0, plot_w, plot_h),
        x: x_scale,
        y: y_scale,
        origin: (x_lo, y_scale.origin(None, None)),
        show: (true, true),
        ticks: opts.ticks.then_some((&x_ticks, &y_ticks)),
        sf,
        axis_color: label_fill,
        label_fill,
        axes_style: &[None, None],
        ticks_style: &[None, None],
        keep_y_label_on_x_axis: true,
        label_color: None,
      },
    );
    if let Some(pos) = buf.rfind("</svg>") {
      buf.insert_str(pos, &axes_svg);
    }
  }

  // `Epilog` primitives are written in the histogram's data coordinates.
  if !opts.epilog.is_empty()
    && let Some(pos) = buf.rfind("</svg>")
  {
    let area = crate::functions::plot_epilog::PlotArea {
      x0: plot_x0,
      y0: plot_y0,
      w: plot_w,
      h: plot_h,
      x_min: x_lo,
      x_max: x_hi,
      y_min,
      y_max,
      scale: sf,
    };
    let overlay = crate::functions::plot_epilog::render_overlay_svg(
      &opts.epilog,
      &area,
      "epilog",
      false,
    );
    buf.insert_str(pos, &overlay);
  }

  // AxesLabel / FrameLabel and PlotLabel, inserted before </svg>.
  // WL 15 sets chart labels at the tick-label size and the PlotLabel a
  // little larger.
  let font_size = sf * CHART_LABEL_FONT;
  let title_font_size = sf * CHART_TITLE_FONT;
  if let Some(insert_pos) = buf.rfind("</svg>") {
    let mut labels_svg = String::new();
    let axis_y = plot_y0 + plot_h;

    // FrameLabel: centred outside the bottom/left edge.
    if let Some((x_label, y_label)) = &opts.frame_label {
      if !x_label.is_empty() {
        let cx = plot_x0 + plot_w / 2.0;
        let base_y = axis_y + font_size * 2.8;
        labels_svg.push_str(&format!(
          "<text x=\"{cx:.1}\" y=\"{base_y:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(x_label)
        ));
      }
      if !y_label.is_empty() {
        let cy = plot_y0 + plot_h / 2.0;
        let lx = margin_left + font_size * 0.8;
        labels_svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font_size:.0}\" \
           fill=\"{label_fill}\" transform=\"rotate(-90,{lx:.1},{cy:.1})\">{}</text>\n",
          crate::functions::graphics::box_string_to_svg(y_label)
        ));
      }
    }

    // AxesLabel: at the far end of each axis.
    labels_svg.push_str(&axes_label_svg(
      opts.axes_label.as_ref(),
      (plot_x0, plot_y0, plot_w, plot_h),
      (x_lo, x_hi, y_min, y_max),
      (true, true),
      font_size,
      label_fill,
    ));

    // Legend entries, one per dataset, in the datasets' colours.
    for (i, label) in opts.chart_legends.iter().enumerate() {
      let color = if opts.chart_style.is_empty() {
        crate::functions::chart::CHART_COLORS
          [i % crate::functions::chart::CHART_COLORS.len()]
      } else {
        let c = &opts.chart_style[i % opts.chart_style.len()];
        (
          (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
        )
      };
      labels_svg.push_str(&chart_legend_entry_svg(
        label,
        color,
        (
          plot_x0 + plot_w + sf * 16.0,
          plot_y0 + sf * 8.0 + i as f64 * sf * 22.0,
        ),
        sf,
        label_fill,
      ));
    }

    // PlotLabel: centered above the chart
    if let Some(sl) = &opts.plot_label
      && !sl.text.is_empty()
    {
      let cx = plot_x0 + plot_w / 2.0;
      let ty = top_margin as f64
        - title_font_size * 0.5
        - if has_y_axes_label {
          font_size * 1.2
        } else {
          0.0
        };
      let fs = sl.font_size.map_or(title_font_size, |f| f * sf);
      // A stacked title grows upwards: its last line stays where a one-line
      // title would sit, so whatever shares the top margin below it (the y
      // `AxesLabel`) is not written over.
      let ty = ty - sl.extra_line_count() as f64 * fs * 1.2;
      let fill = sl
        .color
        .as_ref()
        .map_or_else(|| title_default_fill.to_string(), |c| c.to_svg_rgb());
      let mut style_attrs = String::new();
      if sl.bold {
        style_attrs.push_str(" font-weight=\"bold\"");
      }
      if sl.italic {
        style_attrs.push_str(" font-style=\"italic\"");
      }
      labels_svg.push_str(&format!(
        "<text x=\"{cx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{fs:.0}\" \
           fill=\"{fill}\"{style_attrs}>{}</text>\n",
        sl.svg_scaled_stacked(sf, cx, fs * 1.2)
      ));
    }

    if !labels_svg.is_empty() {
      buf.insert_str(insert_pos, &labels_svg);
    }
  }

  Ok(buf)
}

/// Generate SVG with plotters axes and a blank chart area, returning the SVG
/// and the coordinate transform needed to overlay custom elements.
///
/// Returns (svg_string, plot_area_info) where plot_area_info contains the
/// pixel coordinates of the chart area for overlaying custom content.
pub(crate) struct PlotArea {
  pub svg: String,
  /// Pixel offset of plot area from SVG origin (at render resolution)
  pub plot_x0: f64,
  pub plot_y0: f64,
  pub plot_w: f64,
  pub plot_h: f64,
  pub render_width: u32,
  pub x_min: f64,
  pub x_max: f64,
  pub y_min: f64,
  pub y_max: f64,
}

/// Optional margin overrides for `generate_axes_only_opts`.
pub(crate) struct MarginOverrides {
  pub top_margin: u32,
  pub x_label_area: u32,
  pub y_label_area: u32,
  /// `FrameTicks -> {None, …}` / `{…, None}`: draw no tick marks or labels
  /// along the x axis (bottom edge) and the y axis (left edge) respectively.
  pub hide_x_ticks: bool,
  pub hide_y_ticks: bool,
}

/// Which axes a `FrameTicks` setting leaves ticks on, as `(x, y)` — the
/// bottom and the left edge. Accepts `None | False | All | Automatic`,
/// `{xspec, yspec}`, `{b, l, t, r}` and `{{l, r}, {b, t}}`; an edge pair
/// keeps the axis unless both of its sides are `None`/`False`.
pub(crate) fn frame_tick_axes_visible(value: &Expr) -> (bool, bool) {
  let hidden =
    |e: &Expr| matches!(e, Expr::Identifier(v) if v == "None" || v == "False");
  let is_edge_pair = |e: &Expr| match e {
    Expr::List(p) if p.len() == 2 => p.iter().all(|x| {
      !matches!(x, Expr::Integer(_) | Expr::Real(_) | Expr::String(_))
    }),
    _ => false,
  };
  let pair_visible = |e: &Expr| match e {
    Expr::List(p) if is_edge_pair(e) => !(hidden(&p[0]) && hidden(&p[1])),
    other => !hidden(other),
  };
  match value {
    Expr::List(items) if items.len() == 4 => (
      !(hidden(&items[0]) && hidden(&items[2])),
      !(hidden(&items[1]) && hidden(&items[3])),
    ),
    Expr::List(items) if items.len() == 2 && items.iter().any(is_edge_pair) => {
      (pair_visible(&items[1]), pair_visible(&items[0]))
    }
    Expr::List(items) if items.len() == 2 => {
      (!hidden(&items[0]), !hidden(&items[1]))
    }
    other => (!hidden(other), !hidden(other)),
  }
}

/// What a `FrameTicks` setting draws on one frame edge.
#[derive(Clone, Debug)]
pub(crate) enum FrameEdgeTicks {
  /// No tick marks and no labels.
  None,
  /// The automatic tick marks, unlabelled (`Automatic` on the top and the
  /// right edge).
  Marks,
  /// The automatic tick marks with their labels.
  Labeled,
  /// Exactly the ticks listed.
  Explicit(Vec<ExplicitTick>),
}

impl FrameEdgeTicks {
  /// `FrameTicks -> Automatic`: the bottom and left edges labelled, the top
  /// and right ones marked only (bottom, left, top, right).
  pub(crate) fn defaults() -> [Self; 4] {
    [Self::Labeled, Self::Labeled, Self::Marks, Self::Marks]
  }

  pub(crate) fn is_none(&self) -> bool {
    matches!(self, Self::None)
  }

  /// The ticks this edge labels with an explicit list, as `(pos, label)`.
  pub(crate) fn explicit_labels(&self) -> Option<Vec<(f64, String)>> {
    match self {
      Self::Explicit(ticks) => {
        Some(ticks.iter().map(|t| (t.pos, t.label.clone())).collect())
      }
      _ => None,
    }
  }
}

/// The `(bottom, left, top, right)` edges of a `FrameTicks` value written
/// as `None | All | Automatic`, `{{l, r}, {b, t}}`, `{b, l, t, r}` or
/// `{xspec, yspec}` — the last gives the bottom and top edge `xspec` and
/// the left and right one `yspec`. `Automatic` labels the bottom and left
/// edge and only marks the top and right one; `All` labels any edge.
pub(crate) fn parse_frame_ticks(value: &Expr) -> [FrameEdgeTicks; 4] {
  let one = |e: &Expr, labeled: bool| match e {
    Expr::Identifier(v) if v == "None" || v == "False" => FrameEdgeTicks::None,
    Expr::Identifier(v) if v == "All" => FrameEdgeTicks::Labeled,
    Expr::List(_) | Expr::FunctionCall { .. } | Expr::CurriedCall { .. } => {
      // A spec that still has to be worked out (`Table[{j, …}, {j, 0, 1,
      // .2}]`) is read once evaluated; one that names no ticks keeps the
      // automatic ones.
      match parse_explicit_tick_entries(e) {
        Some(ticks) => FrameEdgeTicks::Explicit(ticks),
        None if labeled => FrameEdgeTicks::Labeled,
        None => FrameEdgeTicks::Marks,
      }
    }
    _ if labeled => FrameEdgeTicks::Labeled,
    _ => FrameEdgeTicks::Marks,
  };
  // An edge pair holds specs, never the bare numbers / strings of a tick.
  let is_pair = |e: &Expr| match e {
    Expr::List(p) if p.len() == 2 => p.iter().all(|x| {
      !matches!(x, Expr::Integer(_) | Expr::Real(_) | Expr::String(_))
    }),
    _ => false,
  };
  match value {
    Expr::List(items) if items.len() == 4 => [
      one(&items[0], true),
      one(&items[1], true),
      one(&items[2], false),
      one(&items[3], false),
    ],
    Expr::List(items) if items.len() == 2 && items.iter().all(is_pair) => {
      match (&items[0], &items[1]) {
        (Expr::List(lr), Expr::List(bt)) => [
          one(&bt[0], true),
          one(&lr[0], true),
          one(&bt[1], false),
          one(&lr[1], false),
        ],
        _ => FrameEdgeTicks::defaults(),
      }
    }
    // `{xspec, yspec}`: an explicit list labels its own edge only.
    Expr::List(items) if items.len() == 2 => {
      let far = |e: &Expr| match one(e, false) {
        FrameEdgeTicks::Explicit(_) => FrameEdgeTicks::Marks,
        other => other,
      };
      [
        one(&items[0], true),
        one(&items[1], true),
        far(&items[0]),
        far(&items[1]),
      ]
    }
    other => [
      one(other, true),
      one(other, true),
      one(other, false),
      one(other, false),
    ],
  }
}

/// Per frame edge `(bottom, left, top, right)`, the style a `FrameStyle` or
/// `FrameTicksStyle` value gives it: `s` styles every edge, `{lr, bt}` the
/// left and right edges with `lr` and the bottom and top ones with `bt`,
/// and `{{l, r}, {b, t}}` each edge on its own.
pub(crate) fn parse_frame_edge_styles(
  value: &Expr,
) -> [Option<SeriesStyle>; 4] {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let pair = |e: &Expr| match e {
    Expr::List(p) if p.len() == 2 => {
      (parse_style_directives(&p[0]), parse_style_directives(&p[1]))
    }
    other => {
      let s = parse_style_directives(other);
      (s.clone(), s)
    }
  };
  match &val {
    Expr::List(items) if items.len() == 2 => {
      let (l, r) = pair(&items[0]);
      let (b, t) = pair(&items[1]);
      [b, l, t, r]
    }
    other => {
      let s = parse_style_directives(other);
      [s.clone(), s.clone(), s.clone(), s]
    }
  }
}
/// Create a plotters chart with axes drawn, returning the SVG and coordinate info.
/// Callers can then append custom SVG elements using the coordinate transform.
pub(crate) fn generate_axes_only(
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<PlotArea, InterpreterError> {
  generate_axes_only_opts(
    x_range, y_range, svg_width, svg_height, full_width, None, None,
  )
}

/// Like `generate_axes_only` but with custom x-axis tick positions (tick marks only, no labels).
/// When `x_tick_positions` is `Some`, only those positions get tick marks on the x-axis.
/// When `margins` is `Some`, overrides the default margins for top, x_label_area, and y_label_area.
pub(crate) fn generate_axes_only_opts(
  x_range: (f64, f64),
  y_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  x_tick_positions: Option<&[f64]>,
  margins: Option<&MarginOverrides>,
) -> Result<PlotArea, InterpreterError> {
  let (x_min, x_max) = x_range;
  let (y_min, y_max) = y_range;
  let render_width = svg_width * RESOLUTION_SCALE;
  let render_height = svg_height * RESOLUTION_SCALE;

  let (bg_color, dark_gray, _light_gray, label_fill, _title_fill) =
    plot_theme();

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root
      .fill(&bg_color)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let s = RESOLUTION_SCALE as i32;
    let tick = MINOR_TICK_LEN * s;

    let top_margin = margins.map_or(10 * RESOLUTION_SCALE, |m| m.top_margin);
    let x_label_area = margins.map_or(
      if x_tick_positions.is_some() {
        12 * RESOLUTION_SCALE
      } else {
        40 * RESOLUTION_SCALE
      },
      |m| m.x_label_area,
    );
    let y_label_area =
      margins.map_or(65 * RESOLUTION_SCALE, |m| m.y_label_area);
    let mut chart = ChartBuilder::on(&root)
      .margin_top(top_margin)
      .margin_right(10 * s as u32)
      .margin_bottom(10 * s as u32)
      .margin_left(10 * s as u32)
      .x_label_area_size(x_label_area)
      .y_label_area_size(y_label_area)
      .build_cartesian_2d(x_min..x_max, y_min..y_max)
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;

    let y_major = axis_tick_step(y_max - y_min);
    let y_minor_step = minor_tick_step(y_major);
    let y_tick_count = ((y_max - y_min) / y_minor_step).round() as usize + 1;

    let hide_x = margins.is_some_and(|m| m.hide_x_ticks);
    let hide_y = margins.is_some_and(|m| m.hide_y_ticks);
    if hide_x && hide_y {
      // Nothing to draw: the caller paints the frame itself.
    } else {
      // Custom x ticks are drawn manually further down, so plotters' own
      // x-axis ticks are suppressed in that mode as well.
      let no_x = hide_x || x_tick_positions.is_some();
      let x_major = axis_tick_step(x_max - x_min);
      let x_minor_step = minor_tick_step(x_major);
      let x_tick_count = if no_x {
        0
      } else {
        ((x_max - x_min) / x_minor_step).round() as usize + 1
      };
      let y_tick_count = if hide_y { 0 } else { y_tick_count };

      chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(x_tick_count)
        .y_labels(y_tick_count)
        .x_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, x_major) {
            format_tick_with_step(*v, x_major)
          } else {
            String::new()
          }
        })
        .y_label_formatter(&move |v: &f64| {
          if is_major_tick(*v, y_major) {
            format_tick_with_step(*v, y_major)
          } else {
            String::new()
          }
        })
        .axis_style(dark_gray.stroke_width(RESOLUTION_SCALE))
        .label_style(
          ("sans-serif", RESOLUTION_SCALE as f64 * 18.0)
            .into_font()
            .color(&dark_gray),
        )
        .set_tick_mark_size(
          LabelAreaPosition::Left,
          if hide_y { 0 } else { tick },
        )
        .set_tick_mark_size(
          LabelAreaPosition::Bottom,
          if no_x { 0 } else { tick },
        )
        .draw()
        .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
    }

    root
      .present()
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  rewrite_svg_header(
    &mut buf,
    svg_width,
    svg_height,
    render_width,
    render_height,
    full_width,
  );

  // Compute the plot area coordinates.
  // ChartBuilder uses: margin on each side, x_label_area at bottom,
  // y_label_area on left. So the plot area starts at:
  let s = RESOLUTION_SCALE as f64;
  let margin = 10.0 * s;
  let top_margin_f = margins.map_or(margin, |m| m.top_margin as f64);
  let y_label_area_f = margins.map_or(65.0 * s, |m| m.y_label_area as f64);
  let x_label_area_f = margins.map_or(
    if x_tick_positions.is_some() {
      12.0 * s
    } else {
      40.0 * s
    },
    |m| m.x_label_area as f64,
  );
  let plot_x0 = margin + y_label_area_f;
  let plot_y0 = top_margin_f;
  let plot_w = render_width as f64 - 2.0 * margin - y_label_area_f;
  let plot_h = render_height as f64 - top_margin_f - margin - x_label_area_f;

  // Extend labeled (major) ticks beyond the minor ticks drawn by plotters.
  // In custom-tick mode, the x axis is drawn manually (below) at the major
  // tick length, so we only extend the y axis here.
  let hide_x = margins.is_some_and(|m| m.hide_x_ticks);
  let hide_y = margins.is_some_and(|m| m.hide_y_ticks);
  if !(hide_x && hide_y) {
    let y_major = axis_tick_step(y_max - y_min);
    let x_axis_ext = if x_tick_positions.is_none() && !hide_x {
      Some((x_min, x_max, axis_tick_step(x_max - x_min)))
    } else {
      None
    };
    let y_axis_ext = (!hide_y).then_some((y_min, y_max, y_major));
    inject_major_tick_extensions(
      &mut buf,
      plot_x0,
      plot_y0,
      plot_w,
      plot_h,
      x_axis_ext,
      y_axis_ext,
      MINOR_TICK_LEN as f64 * s,
      MAJOR_TICK_LEN as f64 * s,
      s,
      label_fill,
    );
  }

  // Draw custom x-axis tick marks if specified. These are at user-supplied
  // positions and all get the "major" tick length since they're all labeled.
  if let Some(positions) = x_tick_positions {
    let tick_len = MAJOR_TICK_LEN as f64 * s;
    let axis_y = plot_y0 + plot_h;
    let stroke_w = s;
    if let Some(insert_pos) = buf.rfind("</svg>") {
      let mut ticks_svg = String::new();
      let tick_color = label_fill;
      for &pos in positions {
        let x = plot_x0 + (pos - x_min) / (x_max - x_min) * plot_w;
        ticks_svg.push_str(&format!(
          "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"{tick_color}\" stroke-width=\"{stroke_w:.0}\"/>\n",
          axis_y, axis_y + tick_len
        ));
      }
      buf.insert_str(insert_pos, &ticks_svg);
    }
  }

  Ok(PlotArea {
    svg: buf,
    plot_x0,
    plot_y0,
    plot_w,
    plot_h,
    render_width,
    x_min,
    x_max,
    y_min,
    y_max,
  })
}

/// Add a border to filled chart bars in the SVG.
/// Plotters' SVG backend doesn't support stroke-width on rects,
/// so we post-process the SVG to add it.
fn add_bar_borders(buf: &mut String, stroke_width: u32) {
  // The first <rect> is the background, skip it.
  // All subsequent filled rects (with stroke="none") are bars.
  // WL 15 charts outline bars with `EdgeForm[Directive[Thickness[Small],
  // Opacity[0.66]]]`: a thin, translucent dark edge.
  let marker = "stroke=\"none\"/>";
  let color = if crate::is_dark_mode() {
    "#555555"
  } else {
    "#000000"
  };
  let width = 0.75 * stroke_width as f64;
  let replacement = format!(
    "stroke=\"{color}\" stroke-opacity=\"0.66\" stroke-width=\"{width}\"/>"
  );
  // Skip the first occurrence (background rect)
  if let Some(first) = buf.find(marker) {
    let after_first = first + marker.len();
    let rest = buf[after_first..].replace(marker, &replacement);
    buf.truncate(after_first);
    buf.push_str(&rest);
  }
}

/// Format a numeric value for tooltip display.
/// Integers (or values very close to integers) are shown without decimals.
fn format_tooltip_value(v: f64) -> String {
  if (v - v.round()).abs() < 1e-10 {
    format!("{}", v as i64)
  } else {
    format!("{v}")
  }
}

/// Inject `<title>` tooltip elements into bar `<rect>` elements.
/// Skips the first `<rect` with `stroke="none"` (the background).
/// All subsequent `<rect` elements that end with `/>` are bars.
fn inject_bar_tooltips(buf: &mut String, values: &[f64]) {
  let tooltips: Vec<String> =
    values.iter().map(|&v| format_tooltip_value(v)).collect();
  inject_bar_tooltips_str(buf, &tooltips);
}

/// Inject `<title>` tooltip strings into bar `<rect>` elements.
/// Skips the first `<rect` with `stroke="none"` (the background rect).
fn inject_bar_tooltips_str(buf: &mut String, tooltips: &[String]) {
  // After add_bar_borders, bar rects have a border style while the
  // background rect still has stroke="none". Find all <rect ... /> that
  // do NOT have stroke="none" and inject titles.
  let mut result = String::with_capacity(buf.len() + tooltips.len() * 30);
  let mut remaining = buf.as_str();
  let mut tooltip_idx = 0;

  while let Some(rect_start) = remaining.find("<rect ") {
    // Find the end of this rect element
    let after_rect = &remaining[rect_start..];
    if let Some(close_pos) = after_rect.find("/>") {
      let rect_tag = &after_rect[..close_pos + 2];
      let is_background = rect_tag.contains("stroke=\"none\"");

      // Copy everything up to the rect
      result.push_str(&remaining[..rect_start]);

      if !is_background && tooltip_idx < tooltips.len() {
        // Replace self-closing /> with ><title>...</title></rect>
        let escaped = html_escape(&tooltips[tooltip_idx]);
        result.push_str(&after_rect[..close_pos]);
        result.push_str(&format!("><title>{escaped}</title></rect>"));
        tooltip_idx += 1;
      } else {
        // Keep as-is
        result.push_str(rect_tag);
      }

      remaining = &remaining[rect_start + close_pos + 2..];
    } else {
      break;
    }
  }

  // Append any remaining content
  result.push_str(remaining);
  *buf = result;
}

/// Rewrite the SVG header to use viewBox for display scaling.
pub(crate) fn rewrite_svg_header(
  buf: &mut String,
  svg_width: u32,
  svg_height: u32,
  render_width: u32,
  render_height: u32,
  full_width: bool,
) {
  if let Some(pos) = buf.find('>') {
    let new_header = if full_width {
      format!(
        "<svg width=\"100%\" viewBox=\"0 0 {render_width} {render_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
      )
    } else {
      format!(
        "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {render_width} {render_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\"",
      )
    };
    buf.replace_range(..pos, &new_header);
  }
}

/// Inject drop-shadow filters for series styled with `DropShadowing[…]`
/// into a finished plotters SVG. plotters has no filter API, so a
/// `<defs>` block is inserted after the `<svg>` header and each curve
/// polyline — identified by the exact stroke color of its series — is
/// tagged with the filter attribute. Shadow parameters are scaled by
/// RESOLUTION_SCALE to match the oversampled render coordinates.
fn inject_drop_shadows(
  buf: &mut String,
  plot_style: &[SeriesStyle],
  n_series: usize,
) {
  if plot_style.is_empty() || !plot_style.iter().any(|s| s.shadow.is_some()) {
    return;
  }
  let mut defs = String::new();
  let mut seen_ids: Vec<String> = Vec::new();
  // (needle with the series' stroke attr, tagged replacement)
  let mut rules: Vec<(String, String)> = Vec::new();
  for idx in 0..n_series {
    let style = &plot_style[idx % plot_style.len()];
    if let Some(ds) = &style.shadow {
      let id = ds.filter_id();
      if !seen_ids.contains(&id) {
        seen_ids.push(id.clone());
        defs.push_str(&ds.filter_def(RESOLUTION_SCALE as f64));
        defs.push('\n');
      }
      let (r, g, b) = series_color(plot_style, idx);
      let needle = format!(
        "<polyline fill=\"none\" opacity=\"1\" stroke=\"#{r:02X}{g:02X}{b:02X}\""
      );
      let replacement = format!(
        "<polyline filter=\"url(#{id})\" fill=\"none\" opacity=\"1\" stroke=\"#{r:02X}{g:02X}{b:02X}\""
      );
      if !rules.iter().any(|(n, _)| n == &needle) {
        rules.push((needle, replacement));
      }
    }
  }
  if let Some(pos) = buf.find('>') {
    buf.insert_str(pos + 1, &format!("\n<defs>\n{defs}</defs>"));
  }
  for (needle, replacement) in &rules {
    *buf = buf.replace(needle, replacement);
  }
}

/// Parse ImageSize option value into (width, height, full_width).
/// Supports: integer, {w, h}, and named sizes (Tiny, Small, Medium, Large, Full).
/// Full uses a 720px render resolution but emits `width="100%"` in SVG.
///
/// For single-number and named sizes, the height is derived from the width
/// using the caller-provided default aspect ratio (`def_w`, `def_h`).
/// For explicit `{w, h}` lists, the user-specified dimensions are used directly.
pub(crate) fn parse_image_size(
  value: &Expr,
  def_w: u32,
  def_h: u32,
) -> Option<(u32, u32, bool)> {
  // A plot holds its arguments, so an option value that is still an
  // arithmetic expression never reached the evaluator: `ImageSize ->
  // 400 {1, 1}` — the way a Demonstration asks for a square picture —
  // arrives as `Times[400, {1, 1}]`. Evaluate it once so the shapes below
  // see the `{400, 400}` they understand.
  let evaluated = match value {
    Expr::BinaryOp { .. } | Expr::FunctionCall { .. } => {
      evaluate_expr_to_expr(value).ok()
    }
    _ => None,
  };
  let value = evaluated.as_ref().unwrap_or(value);
  let aspect = def_h as f64 / def_w as f64;
  match value {
    Expr::Integer(n) if *n > 0 => {
      let w = *n as u32;
      let h = (w as f64 * aspect).round() as u32;
      Some((w, h, false))
    }
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      let w = n.to_u32()?;
      if w == 0 {
        return None;
      }
      let h = (w as f64 * aspect).round() as u32;
      Some((w, h, false))
    }
    Expr::Real(f) if *f > 0.0 => {
      let w = f.round() as u32;
      let h = (w as f64 * aspect).round() as u32;
      Some((w, h, false))
    }
    // `{w}` is the same as `w`.
    Expr::List(items) if items.len() == 1 => {
      parse_image_size(&items[0], def_w, def_h)
    }
    // `{{wmin, wmax}, {hmin, hmax}}`: the default size within those
    // ranges, keeping the aspect ratio where the ranges allow.
    Expr::List(items)
      if items.len() == 2
        && items
          .iter()
          .all(|e| matches!(e, Expr::List(r) if r.len() == 2)) =>
    {
      let range = |e: &Expr| -> Option<(f64, f64)> {
        let Expr::List(r) = e else { return None };
        let lo = crate::functions::math_ast::try_eval_to_f64(&r[0])?;
        let hi = crate::functions::math_ast::try_eval_to_f64(&r[1])?;
        (lo > 0.0 && hi >= lo).then_some((lo, hi))
      };
      let (wr, hr) = (range(&items[0])?, range(&items[1])?);
      let mut w = (def_w as f64).clamp(wr.0, wr.1);
      let h = (w * aspect).clamp(hr.0, hr.1);
      if (h - w * aspect).abs() > 0.5 {
        w = (h / aspect).clamp(wr.0, wr.1);
      }
      Some((w.round() as u32, h.round() as u32, false))
    }
    // `{w, h}` fixes both dimensions; `Automatic` in either slot leaves
    // that one to follow from the other and the default aspect, which is
    // how a Demonstration sizes a row of plots by height alone.
    Expr::List(items) if items.len() == 2 => {
      let dim = |e: &Expr| -> Option<Option<u32>> {
        match e {
          Expr::Identifier(n) if n == "Automatic" => Some(None),
          Expr::Integer(n) if *n > 0 => Some(Some(*n as u32)),
          Expr::BigInteger(n) => {
            use num_traits::ToPrimitive;
            let v = n.to_u32()?;
            (v > 0).then_some(Some(v))
          }
          Expr::Real(f) if *f > 0.0 => Some(Some(f.round() as u32)),
          _ => None,
        }
      };
      match (dim(&items[0])?, dim(&items[1])?) {
        (Some(w), Some(h)) => Some((w, h, false)),
        (Some(w), None) => Some((w, (w as f64 * aspect).round() as u32, false)),
        (None, Some(h)) => {
          Some((((h as f64) / aspect).round() as u32, h, false))
        }
        (None, None) => Some((def_w, def_h, false)),
      }
    }
    Expr::Identifier(name) => {
      // WL 15's named widths (`ImageSize -> Small` is 180, `Large` 576).
      let base_w = match name.as_str() {
        "Tiny" => 100,
        "Small" => 180,
        "Medium" => def_w,
        "Large" => 576,
        "Full" => 720,
        _ => return None,
      };
      let h = (base_w as f64 * aspect).round() as u32;
      let fw = name == "Full";
      Some((base_w, h, fw))
    }
    _ => None,
  }
}

/// Which dimensions an `ImageSize` value fixes: `(width, height)`. A bare
/// width or a named size fixes the width only, `{w, h}` both, and
/// `Automatic` in either slot of the pair leaves that one free.
fn image_size_explicit_dims(value: &Expr) -> (bool, bool) {
  let value = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let fixed = |e: &Expr| {
    !matches!(e, Expr::Identifier(n) if n == "Automatic")
      && !matches!(e, Expr::List(_))
  };
  match &value {
    Expr::List(items) if items.len() == 2 => {
      if items.iter().all(|e| matches!(e, Expr::List(_))) {
        (false, false)
      } else {
        (fixed(&items[0]), fixed(&items[1]))
      }
    }
    Expr::List(items) if items.len() == 1 => {
      image_size_explicit_dims(&items[0])
    }
    _ => (true, false),
  }
}

/// Parse a single PlotStyle element into a SeriesStyle.
/// Handles: a color, `Thick`, `Dashed`, `Dotted`, `DotDashed`,
/// `Directive[...]`, `{Red, Thick, Dashed}`, etc.
fn parse_one_series_style(expr: &Expr) -> SeriesStyle {
  let mut style = SeriesStyle::default();
  apply_style_directive(expr, &mut style);
  // A `{…}` or `Directive[…]` entry is one dataset's whole style: a lone
  // dataset given `{Directive[Red, Thick], Directive[Blue]}` takes the
  // first, where the bare directives of `{Red, Thick}` combine.
  style.from_list = matches!(expr, Expr::List(_))
    || matches!(expr, Expr::FunctionCall { name, .. } if name == "Directive");
  style
}

/// A color directive as the plot expression writes it: evaluated (`Red`
/// becomes `RGBColor[1, 0, 0]`), with the alpha of a four-argument
/// `RGBColor` dropped, since the series carries it as its opacity.
fn opaque_color_expr(expr: &Expr) -> Option<Expr> {
  let value = crate::evaluator::evaluate_expr_to_expr(expr).ok()?;
  match value {
    Expr::FunctionCall { ref name, ref args }
      if name == "RGBColor" && args.len() == 4 =>
    {
      Some(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: args[..3].to_vec().into(),
      })
    }
    Expr::FunctionCall { ref name, .. }
      if matches!(
        name.as_str(),
        "RGBColor"
          | "Hue"
          | "GrayLevel"
          | "CMYKColor"
          | "LABColor"
          | "XYZColor"
          | "LCHColor"
          | "LUVColor"
      ) =>
    {
      Some(value)
    }
    _ => None,
  }
}

/// The color directive of a plot series in the plot expression: the color
/// as `PlotStyle` wrote it, else the exact palette color.
pub(crate) fn series_color_expr(
  series: &crate::syntax::PlotSeriesData,
) -> Expr {
  series
    .color_expr
    .clone()
    .unwrap_or_else(|| rgb_u8_expr(series.color))
}

/// Apply a style directive expression to a SeriesStyle.
pub(crate) fn apply_style_directive(expr: &Expr, style: &mut SeriesStyle) {
  // Try as a color first
  if let Some(c) = parse_color(expr) {
    // A translucent colour (`Transparent`, `RGBColor[r, g, b, a]`) fades the
    // curve by its alpha.
    if c.a < 1.0 {
      style.opacity = Some(c.a);
    }
    style.color = Some(c);
    style.color_expr = opaque_color_expr(expr);
    return;
  }
  match expr {
    Expr::Identifier(s) => match s.as_str() {
      "Thick" => style.thickness = Some(2.0),
      "Thin" => style.thickness = Some(0.5),
      // `None` hides the curve itself; a `Filling` toward it still shows.
      "None" => style.opacity = Some(0.0),
      // A named dash length is absolute pixels, not a fraction of the
      // picture: `Dashed` is 4 on / 4 off whatever the image size. Negative
      // marks absolute, as it does for thickness.
      "Dashed" => style.dashing = Some(vec![-4.0, -4.0]),
      "Dotted" => style.dashing = Some(vec![0.0, -4.0]),
      "DotDashed" => style.dashing = Some(vec![0.0, -4.0, -4.0, -4.0]),
      _ => {}
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Directive" => {
        for a in args {
          apply_style_directive(a, style);
        }
      }
      "DropShadowing" => {
        style.shadow = crate::functions::graphics::parse_drop_shadowing(args);
      }
      // `Opacity[a]` fades the curve; `Opacity[a, color]` also sets its color.
      "Opacity" if !args.is_empty() => {
        if let Some(a) = try_eval_to_f64(&args[0]) {
          style.opacity = Some(a.clamp(0.0, 1.0));
        }
        if let Some(c) = args.get(1).and_then(parse_color) {
          style.color = Some(c);
          style.color_expr = opaque_color_expr(&args[1]);
        }
      }
      "Thickness" if args.len() == 1 => {
        if let Expr::Identifier(s) = &args[0] {
          match s.as_str() {
            "Large" => style.thickness = Some(2.0),
            "Tiny" => style.thickness = Some(0.5),
            _ => {
              if let Some(t) = try_eval_to_f64(&args[0]) {
                // Relative thickness: fraction of plot width → display px
                // 360px default width * fraction
                style.thickness = Some(t * 360.0);
              }
            }
          }
        } else if let Some(t) = try_eval_to_f64(&args[0]) {
          style.thickness = Some(t * 360.0);
        }
      }
      "AbsoluteThickness" if args.len() == 1 => {
        if let Some(t) = try_eval_to_f64(&args[0]) {
          style.thickness = Some(t);
        }
      }
      "PointSize" if args.len() == 1 => {
        if let Some(s) =
          crate::functions::graphics::symbolic_point_size(&args[0])
        {
          style.point_size = Some(s);
        } else if let Some(s) = try_eval_to_f64(&args[0]) {
          style.point_size = Some(s);
        }
      }
      "AbsolutePointSize" if args.len() == 1 => {
        if let Some(s) =
          crate::functions::graphics::symbolic_point_size(&args[0])
        {
          style.point_size = Some(s);
        } else if let Some(s) = try_eval_to_f64(&args[0]) {
          style.point_size = Some(-s);
        }
      }
      // `AbsoluteDashing[{d1, d2, …}]`: dash lengths in printer's points,
      // stored negative like the named (absolute) lengths of `Dashing`.
      "AbsoluteDashing" if args.len() == 1 => {
        let dashes: Vec<f64> = match &args[0] {
          Expr::List(items) => items
            .iter()
            .filter_map(try_eval_to_f64)
            .map(|d| -d.abs().max(0.1))
            .collect(),
          other => try_eval_to_f64(other)
            .map(|d| vec![-d.abs().max(0.1); 2])
            .unwrap_or_default(),
        };
        if !dashes.is_empty() {
          style.dashing = Some(dashes);
        }
      }
      "Dashing" if !args.is_empty() => {
        if let Expr::List(items) = &args[0] {
          let dashes: Vec<f64> = items
            .iter()
            .filter_map(|e| match e {
              Expr::Identifier(s) => match s.as_str() {
                "Tiny" => Some(-2.0),
                "Small" => Some(-4.0),
                "Medium" => Some(-8.0),
                "Large" => Some(-16.0),
                _ => None,
              },
              _ => try_eval_to_f64(e),
            })
            .collect();
          if !dashes.is_empty() {
            style.dashing = Some(dashes);
          }
        } else {
          let d = match &args[0] {
            Expr::Identifier(s) => match s.as_str() {
              "Tiny" => Some(0.005),
              "Small" => Some(0.01),
              "Medium" => Some(0.02),
              "Large" => Some(0.04),
              _ => None,
            },
            _ => try_eval_to_f64(&args[0]),
          };
          if let Some(d) = d {
            style.dashing = Some(vec![d, d]);
          }
        }
      }
      _ => {}
    },
    Expr::List(items) => {
      // {Red, Thick, Dashed} — apply all sub-directives
      for item in items {
        apply_style_directive(item, style);
      }
    }
    _ => {}
  }
}

/// The font size a `PlotMarkers` glyph is drawn at when the marker spec
/// names none — the default graphics font size the Wolfram Language uses.
const DEFAULT_MARKER_SIZE: f64 = 12.0;

/// The shapes `PlotMarkers -> "OpenMarkers"` cycles over the series (in WL
/// 15's order): white-filled outlines rather than the default filled dot,
/// so overlapping series stay distinguishable by outline as well as colour.
/// Each is named by its glyph and drawn by [`open_marker_outline`].
const OPEN_MARKER_GLYPHS: [&str; 5] = ["○", "△", "◇", "□", "▽"];

/// The outline of the open marker `glyph` as offsets from the data point in
/// printer's points (y up), the polygons WL 15's `"OpenMarkers"` insets
/// draw; `None` for the circle (radius 3) and for any other glyph.
fn open_marker_outline(glyph: &str) -> Option<Vec<(f64, f64)>> {
  let r3 = 3f64.sqrt();
  Some(match glyph {
    "△" => vec![(0.0, 4.0), (-2.0 * r3, -2.0), (2.0 * r3, -2.0)],
    "▽" => vec![(0.0, -4.0), (-2.0 * r3, 2.0), (2.0 * r3, 2.0)],
    "◇" => vec![(0.0, 3.75), (3.75, 0.0), (0.0, -3.75), (-3.75, 0.0)],
    "□" => vec![(-2.75, -2.75), (2.75, -2.75), (2.75, 2.75), (-2.75, 2.75)],
    _ => return None,
  })
}

/// The shapes `PlotMarkers -> Automatic` cycles over the series (WL 15's
/// order): five filled, then the same five outlined.
const AUTOMATIC_MARKER_GLYPHS: [&str; 10] =
  ["●", "■", "◆", "▲", "▼", "○", "□", "◇", "△", "▽"];

/// The marker size a named `PlotMarkers -> {Automatic, size}` stands for,
/// in points.
fn named_marker_size(name: &str) -> Option<f64> {
  Some(match name {
    "Tiny" => 4.0,
    "Small" => 7.0,
    "Medium" => 10.0,
    "Large" => 20.0,
    _ => return None,
  })
}

/// An automatic marker's shape around the data point, in points (y up):
/// a circle of the given radius or a polygon, and whether it is filled
/// (else outlined 2 pt wide). These are the sizes WL 15 draws at its
/// default marker size; an explicit size `s` scales a disk to radius
/// 0.35 (s - 1) and the rest in proportion.
enum MarkerShape {
  Circle(f64),
  Polygon(Vec<(f64, f64)>),
}

fn automatic_marker_shape(
  glyph: &str,
  size: f64,
) -> Option<(MarkerShape, bool)> {
  let k = if size > 0.0 {
    0.35 * (size - 1.0) / 3.49
  } else {
    1.0
  };
  let poly = |pts: &[(f64, f64)]| {
    MarkerShape::Polygon(pts.iter().map(|&(x, y)| (x * k, y * k)).collect())
  };
  Some(match glyph {
    "●" => (MarkerShape::Circle(3.49 * k), true),
    "■" => (
      poly(&[(-3.32, -3.32), (3.32, -3.32), (3.32, 3.32), (-3.32, 3.32)]),
      true,
    ),
    "◆" => (
      poly(&[(0.0, 4.41), (4.41, 0.0), (0.0, -4.41), (-4.41, 0.0)]),
      true,
    ),
    "▲" => (poly(&[(0.0, 4.33), (-4.99, -4.32), (4.99, -4.32)]), true),
    "▼" => (poly(&[(0.0, -4.37), (-5.05, 4.37), (5.05, 4.37)]), true),
    "○" => (MarkerShape::Circle(3.41 * k), false),
    "□" => (
      poly(&[(-3.41, -3.41), (3.41, -3.41), (3.41, 3.41), (-3.41, 3.41)]),
      false,
    ),
    "◇" => (
      poly(&[(0.0, 4.22), (4.22, 0.0), (0.0, -4.22), (-4.22, 0.0)]),
      false,
    ),
    "△" => (poly(&[(0.0, 3.46), (-3.95, -3.39), (3.95, -3.39)]), false),
    "▽" => (poly(&[(0.0, -3.46), (-3.95, 3.39), (3.95, 3.39)]), false),
    _ => return None,
  })
}

/// Draw the `PlotMarkers` glyph `marker` at each of `points`, centred on
/// the point: text at the marker's size, or for an `"OpenMarkers"` shape a
/// white-filled outline stroked 1.5 pt wide in `rgb` (the series colour).
fn draw_marker_glyphs<DB, CT>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  points: &[(f64, f64)],
  marker: &PlotMarker,
  rgb: (u8, u8, u8),
) -> Result<(), InterpreterError>
where
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
{
  let err = |e: plotters::drawing::DrawingAreaErrorKind<DB::ErrorType>| {
    InterpreterError::EvaluationError(format!("Plot: {e}"))
  };
  let (mr, mg, mb) = marker.color.unwrap_or(rgb);
  let color = RGBColor(mr, mg, mb);
  if marker.automatic
    && let Some((shape, filled)) =
      automatic_marker_shape(&marker.glyph, marker.size)
  {
    let scale = RESOLUTION_SCALE as f64;
    let stroke = color.stroke_width((2.0 * scale).round() as u32);
    match shape {
      MarkerShape::Circle(r) => {
        let r = (r * scale).round() as i32;
        let style = if filled { color.filled() } else { stroke };
        chart
          .draw_series(
            points
              .iter()
              .map(|&p| EmptyElement::at(p) + Circle::new((0, 0), r, style)),
          )
          .map_err(err)?;
      }
      MarkerShape::Polygon(outline) => {
        let px: Vec<(i32, i32)> = outline
          .iter()
          .map(|&(dx, dy)| {
            ((dx * scale).round() as i32, (-dy * scale).round() as i32)
          })
          .collect();
        if filled {
          chart
            .draw_series(points.iter().map(|&p| {
              EmptyElement::at(p) + Polygon::new(px.clone(), color.filled())
            }))
            .map_err(err)?;
        } else {
          let mut closed = px.clone();
          closed.push(px[0]);
          chart
            .draw_series(points.iter().map(|&p| {
              EmptyElement::at(p) + PathElement::new(closed.clone(), stroke)
            }))
            .map_err(err)?;
        }
      }
    }
    return Ok(());
  }
  if marker.open {
    let scale = RESOLUTION_SCALE as f64;
    let stroke = color.stroke_width((1.5 * scale).round() as u32);
    if let Some(outline) = open_marker_outline(&marker.glyph) {
      let px: Vec<(i32, i32)> = outline
        .iter()
        .map(|&(dx, dy)| {
          ((dx * scale).round() as i32, (-dy * scale).round() as i32)
        })
        .collect();
      let mut closed = px.clone();
      closed.push(px[0]);
      chart
        .draw_series(points.iter().map(|&p| {
          EmptyElement::at(p)
            + Polygon::new(px.clone(), WHITE.filled())
            + PathElement::new(closed.clone(), stroke)
        }))
        .map_err(err)?;
    } else {
      let r = (3.0 * scale).round() as i32;
      chart
        .draw_series(points.iter().map(|&p| {
          EmptyElement::at(p)
            + Circle::new((0, 0), r, WHITE.filled())
            + Circle::new((0, 0), r, stroke)
        }))
        .map_err(err)?;
    }
    return Ok(());
  }
  let style = ("sans-serif", marker.size * RESOLUTION_SCALE as f64)
    .into_font()
    .color(&color)
    .pos(plotters::style::text_anchor::Pos::new(
      plotters::style::text_anchor::HPos::Center,
      plotters::style::text_anchor::VPos::Center,
    ));
  chart
    .draw_series(points.iter().map(|&p| {
      plotters::element::Text::new(marker.glyph.clone(), p, style.clone())
    }))
    .map_err(err)?;
  Ok(())
}

/// Read one `PlotMarkers` entry: a glyph, optionally wrapped in `Style`
/// (which may carry a colour and a font size in any order) or paired with
/// its size as `{marker, size}`. Anything else (a `Graphics` marker, say)
/// gives `None`, which leaves that series on plain round points.
fn parse_one_marker(expr: &Expr) -> Option<PlotMarker> {
  let val = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  match &val {
    Expr::String(s) => Some(PlotMarker {
      glyph: s.clone(),
      size: DEFAULT_MARKER_SIZE,
      color: None,
      open: false,
      automatic: false,
    }),
    // `Row[{a, b, …}]` — the pieces laid side by side as one text glyph,
    // e.g. a number followed by its unit.
    Expr::FunctionCall { name, args }
      if name == "Row"
        && args.len() == 1
        && matches!(&args[0], Expr::List(_)) =>
    {
      let Expr::List(items) = &args[0] else {
        return None;
      };
      let glyph: String = items
        .iter()
        .map(|item| match item {
          Expr::String(s) => s.clone(),
          other => crate::syntax::expr_to_string(other),
        })
        .collect();
      Some(PlotMarker {
        glyph,
        size: DEFAULT_MARKER_SIZE,
        color: None,
        open: false,
        automatic: false,
      })
    }
    // `Style[marker, directives…]` — a colour and/or a font size.
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      let mut marker = parse_one_marker(&args[0])?;
      for directive in &args[1..] {
        if let Some(c) = crate::functions::graphics::parse_color(directive) {
          marker.color = Some((
            (c.r * 255.0).round() as u8,
            (c.g * 255.0).round() as u8,
            (c.b * 255.0).round() as u8,
          ));
        } else if let Some(size) = try_eval_to_f64(directive) {
          marker.size = size;
        } else if let Expr::FunctionCall { name, args } = directive
          && name == "FontSize"
          && args.len() == 1
          && let Some(size) = try_eval_to_f64(&args[0])
        {
          marker.size = size;
        }
      }
      Some(marker)
    }
    // `{marker, size}` — the size is always the second element.
    Expr::List(items) if items.len() == 2 => {
      let size = try_eval_to_f64(&items[1])?;
      let mut marker = parse_one_marker(&items[0])?;
      marker.size = size;
      Some(marker)
    }
    _ => None,
  }
}

/// Parse a `PlotMarkers` option value into one marker per series. A single
/// marker (or `{marker, size}` pair) applies to every series; a list of
/// markers is cycled over the series, as the Wolfram Language does.
pub(crate) fn parse_plot_markers(
  replacement: &Expr,
) -> Vec<Option<PlotMarker>> {
  let val =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  // `"OpenMarkers"` is a named marker set, not a literal glyph: it cycles
  // the open shapes above over the series instead of drawing the word.
  if matches!(&val, Expr::String(s) if s == "OpenMarkers") {
    return OPEN_MARKER_GLYPHS
      .iter()
      .map(|glyph| {
        Some(PlotMarker {
          glyph: (*glyph).to_string(),
          size: DEFAULT_MARKER_SIZE,
          color: None,
          open: true,
          automatic: false,
        })
      })
      .collect();
  }
  // `Automatic` (or `{Automatic, size}`) cycles WL's filled and outlined
  // shapes over the series.
  let automatic_size = match &val {
    Expr::Identifier(s) if s == "Automatic" => Some(0.0),
    Expr::List(items)
      if items.len() == 2
        && matches!(&items[0], Expr::Identifier(s) if s == "Automatic") =>
    {
      match &items[1] {
        Expr::Identifier(name) => named_marker_size(name),
        other => try_eval_to_f64(other),
      }
    }
    _ => None,
  };
  if let Some(size) = automatic_size {
    return AUTOMATIC_MARKER_GLYPHS
      .iter()
      .map(|glyph| {
        Some(PlotMarker {
          glyph: (*glyph).to_string(),
          size,
          color: None,
          open: false,
          automatic: true,
        })
      })
      .collect();
  }
  // `{m1, m2, …}` is per-series unless it is the `{marker, size}` pair,
  // which `parse_one_marker` recognises.
  if let Expr::List(items) = &val
    && parse_one_marker(&val).is_none()
  {
    return items.iter().map(parse_one_marker).collect();
  }
  match parse_one_marker(&val) {
    Some(marker) => vec![Some(marker)],
    None => Vec::new(),
  }
}

/// The marker for series `idx`, cycling the list as the Wolfram Language
/// does when there are fewer markers than series.
fn series_marker(
  markers: &[Option<PlotMarker>],
  idx: usize,
) -> Option<&PlotMarker> {
  if markers.is_empty() {
    return None;
  }
  markers[idx % markers.len()].as_ref()
}

/// Draw one series of scattered points onto an already-built chart:
/// fill-target stem lines (mirroring the fill a joined curve would get from
/// `Filling`), then either the series' `PlotMarkers` glyph or, lacking one,
/// the default round dot — the two are mutually exclusive. The line
/// renderer (`generate_svg_with_options`) draws every unjoined series with
/// it, an all-points `ListPlot` (`generate_scatter_svg_with_options`) as well
/// as a `Joined -> {…, False, …}` series among joined ones.
#[allow(clippy::too_many_arguments)]
fn draw_scatter_series<DB, CT>(
  chart: &mut plotters::prelude::ChartContext<DB, CT>,
  all_series: &[Vec<(f64, f64)>],
  series_idx: usize,
  opts: &PlotOptions,
  rgb: (u8, u8, u8),
  marker_size: u32,
  (x_min, x_max): (f64, f64),
  y_min: f64,
  y_max: f64,
) -> Result<(), InterpreterError>
where
  DB: plotters::prelude::DrawingBackend,
  CT: plotters::prelude::CoordTranslate<From = (f64, f64)>,
{
  let (r, g, b) = rgb;
  let color = RGBColor(r, g, b);
  let points = &all_series[series_idx];
  // A point far outside the plot range is never visible, and mapping it to
  // a pixel overflows the backend's integer coordinates (an extrapolated
  // 10^300 against PlotRange -> {-0.05, 1.05} panicked).
  // A dot beyond the plot range is not drawn at all (Wolfram clips it
  // rather than pinning it to the frame edge); the tolerance keeps dots
  // sitting exactly on the boundary.
  let near = |v: f64, lo: f64, hi: f64| {
    let span = (hi - lo).abs().max(f64::MIN_POSITIVE);
    v >= lo - 1e-9 * span && v <= hi + 1e-9 * span
  };
  let finite_pts: Vec<(f64, f64)> = points
    .iter()
    .copied()
    .filter(|(x, y)| {
      x.is_finite()
        && y.is_finite()
        && near(*x, x_min, x_max)
        && near(*y, y_min, y_max)
    })
    .collect();

  // Draw stem lines from each point to the fill reference: a constant
  // level for Axis/Bottom/Top/value, or — for `Filling -> {i -> {j}}` —
  // the other series, linearly interpolated at this point's x so
  // irregularly spaced datasets fill correctly.
  // WL 15 draws the stems of a filled scatter at Opacity[0.3], a little
  // stronger than the 0.2 of a filled curve's area.
  let fill_style = series_filling_style(opts, series_idx);
  let stem_style = match fill_style {
    None => RGBColor(r, g, b).mix(0.3),
    fs => fill_paint(fs, (r, g, b)),
  }
  .stroke_width(RESOLUTION_SCALE);
  // A two-colour `{below, above}` style paints the stems of points under
  // their reference in the first colour.
  let below_style =
    fill_style
      .and_then(|fs| fs.below_color)
      .map(|(br, bg, bb)| {
        RGBColor(br, bg, bb)
          .mix(
            fill_style
              .and_then(|fs| fs.effective_opacity())
              .unwrap_or(1.0),
          )
          .stroke_width(RESOLUTION_SCALE)
      });
  let stem_dashes = fill_style.and_then(|fs| fs.dashing);
  let stem_targets: Vec<((f64, f64), f64)> =
    match series_fill_target(opts, series_idx) {
      FillTarget::Level(level) => level
        .reference_y(y_min, y_max)
        .map(|ref_y| finite_pts.iter().map(|&p| (p, ref_y)).collect())
        .unwrap_or_default(),
      FillTarget::Series(target_idx) => {
        if target_idx != series_idx
          && let Some(target) = all_series.get(target_idx)
        {
          finite_pts
            .iter()
            .copied()
            .filter_map(|(x, y)| {
              interp_polyline_y(target, x).map(|ty| ((x, y), ty))
            })
            .collect()
        } else {
          Vec::new()
        }
      }
    };
  for ((x, y), ref_y) in stem_targets {
    let style = match below_style {
      Some(below) if y < ref_y => below,
      _ => stem_style,
    };
    let stem = vec![(x, y), (x, ref_y)];
    let drawn = match stem_dashes {
      Some((dash, gap)) => {
        let scale = RESOLUTION_SCALE as f64;
        chart.draw_series(std::iter::once(
          plotters::element::DashedPathElement::new(
            stem,
            (dash * scale).round().max(1.0) as u32,
            (gap * scale).round().max(1.0) as u32,
            style,
          ),
        ))
      }
      None => chart.draw_series(std::iter::once(PathElement::new(stem, style))),
    };
    drawn
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }

  // `PlotMarkers` replaces the round dot with its glyph, drawn
  // centred on the point at the size the marker spec asks for.
  if let Some(marker) = series_marker(&opts.plot_markers, series_idx) {
    draw_marker_glyphs(chart, &finite_pts, marker, (r, g, b))?;
  } else {
    // `ColorFunction -> f` colors each dot by `f` at its own position
    // (rescaled over the plotted data when `ColorFunctionScaling` is on).
    let cf_ranges =
      data_extents(all_series).unwrap_or((x_min, x_max, y_min, y_max));
    let styled = opts.point_colors.get(series_idx);
    let dot_color = |p: (f64, f64)| {
      if let Some(&(_, (r, g, b))) =
        styled.and_then(|pts| pts.iter().find(|(q, _)| *q == p))
      {
        return RGBColor(r, g, b);
      }
      match &opts.color_function {
        Some(cf) => {
          color_function_rgb(cf, opts.color_function_scaling, p, cf_ranges)
            .map_or(color, |(r, g, b)| RGBColor(r, g, b))
        }
        None => color,
      }
    };
    chart
      .draw_series(finite_pts.iter().map(|&(x, y)| {
        Circle::new((x, y), marker_size, dot_color((x, y)).filled())
      }))
      .map_err(|e| InterpreterError::EvaluationError(format!("Plot: {e}")))?;
  }
  Ok(())
}

/// Carry the parsed `PlotMarkers` into a plot's [`PlotSource`], so a
/// `Show` that merges the plot keeps drawing the same glyphs.
pub(crate) fn apply_markers_to_source(
  source: &mut crate::syntax::PlotSource,
  markers: &[Option<PlotMarker>],
) {
  for (i, series) in source.series.iter_mut().enumerate() {
    series.marker = series_marker(markers, i).cloned();
  }
}

/// Parse a PlotStyle option value into a list of SeriesStyles.
pub(crate) fn parse_plot_style(replacement: &Expr) -> Vec<SeriesStyle> {
  let val = evaluate_expr_to_expr(replacement).unwrap_or(replacement.clone());
  if let Expr::List(items) = &val {
    // A list is one style per series (`{Thin, Thick}` strokes the first
    // curve thin and the second thick). Callers drawing a single series
    // merge the entries into one compound style (`{Purple, Thick, Dashed}`).
    items.iter().map(parse_one_series_style).collect()
  } else {
    let style = parse_one_series_style(&val);
    if style.color.is_some()
      || style.thickness.is_some()
      || style.point_size.is_some()
      || style.dashing.is_some()
      || style.shadow.is_some()
      || style.opacity.is_some()
    {
      vec![style]
    } else {
      Vec::new()
    }
  }
}

/// The WL 15 palette of `PlotTheme -> "Marketing"`.
const MARKETING_COLORS: [&str; 15] = [
  "RGBColor[1., 0.4, 0.]",
  "RGBColor[0.655728, 0.8, 0.]",
  "RGBColor[0., 0.742291, 0.873126]",
  "RGBColor[1., 0.656408, 0.]",
  "RGBColor[0.893126, 0.4, 0.767184]",
  "RGBColor[0.295048, 0.8, 0.286932]",
  "RGBColor[0.238758, 0.610466, 1.]",
  "RGBColor[1., 0.325204, 0.406504]",
  "RGBColor[0., 0.786874, 0.739379]",
  "RGBColor[1., 0.520437, 0.]",
  "RGBColor[0.7529330319872088, 0.4176501130047967, 1.]",
  "RGBColor[0.5572809000084149, 0.8, 0]",
  "RGBColor[1., 0.06811595600706821, 0.0851449450088353]",
  "RGBColor[0, 0.7226017980018511, 0.9321946059944466]",
  "RGBColor[1., 0.7154761789941944, 0]",
];

/// The WL 15 palette of `PlotTheme -> "Business"`.
const BUSINESS_COLORS: [&str; 15] = [
  "RGBColor[0.34398, 0.49112, 0.89936]",
  "RGBColor[0.97, 0.606, 0.081]",
  "RGBColor[0.91, 0.318, 0.243]",
  "RGBColor[0.448, 0.69232, 0.1538]",
  "RGBColor[0.62168, 0.2798, 0.6914]",
  "RGBColor[0.09096, 0.6296, 0.85532]",
  "RGBColor[0.46056, 0.40064, 0.81392]",
  "RGBColor[0.94, 0.462, 0.162]",
  "RGBColor[0., 0.7, 0.7]",
  "RGBColor[0.827051, 0.418034, 0.0243459]",
  "RGBColor[0.5511749434976025, 0.32014794962639853, 0.8720626412559938]",
  "RGBColor[0.72694101250947, 0.7196601125010522, 0.]",
  "RGBColor[0.8680706456216862, 0.2563858708756628, 0.30321559063052295]",
  "RGBColor[0.2418693812442152, 0.5065044950046278, 0.9902432574930582]",
  "RGBColor[0.9573908706237908, 0.5369543531189542, 0.11504464931576472]",
];

/// The WL 15 palette of `PlotTheme -> "Web"`.
const WEB_COLORS: [&str; 15] = [
  "RGBColor[0.790588, 0.201176, 0.]",
  "RGBColor[0.192157, 0.388235, 0.807843]",
  "RGBColor[1., 0.607843, 0.]",
  "RGBColor[0., 0.596078, 0.109804]",
  "RGBColor[0.567426, 0.32317, 0.729831]",
  "RGBColor[0., 0.588235, 0.705882]",
  "RGBColor[0.8505, 0.4275, 0.13185]",
  "RGBColor[0.499929, 0.285875, 0.775177]",
  "RGBColor[0.12490296143062507, 0.63, 0.47103259454284074]",
  "RGBColor[0.823948950768196, 0.29474475384097976, 0.19291741323314934]",
  "RGBColor[0.42126358105951733, 0.33224185136428963, 0.9]",
  "RGBColor[0.7239916650994997, 0.6554435183443889, 0.]",
  "RGBColor[0.6746366424582266, 0.252, 0.45055901160272827]",
  "RGBColor[0.12582006271512805, 0.5293439498278976, 0.7809840752581536]",
  "RGBColor[0.8604677779867332, 0.4773388899336664, 0.10673699943582014]",
];

/// The WL 15 palette of `PlotTheme -> "Scientific"`.
const SCIENTIFIC_COLORS: [&str; 15] = [
  "RGBColor[0.9, 0.36, 0.054]",
  "RGBColor[0.365248, 0.427802, 0.758297]",
  "RGBColor[0.945109, 0.593901, 0.]",
  "RGBColor[0.645957, 0.253192, 0.685109]",
  "RGBColor[0.285821, 0.56, 0.450773]",
  "RGBColor[0.7, 0.336, 0.]",
  "RGBColor[0.491486, 0.345109, 0.8]",
  "RGBColor[0.71788, 0.568653, 0.]",
  "RGBColor[0.70743, 0.224, 0.542415]",
  "RGBColor[0.287228, 0.490217, 0.664674]",
  "RGBColor[0.982289285128704, 0.5771321368979874, 0.011542503255145636]",
  "RGBColor[0.5876740325800278, 0.2877284499870081, 0.7500695697462922]",
  "RGBColor[0.4262088601796793, 0.5581552810007578, 0.2777996730417023]",
  "RGBColor[0.9431487543762861, 0.414555896337833, 0.07140829055870854]",
  "RGBColor[0.41497437140121635, 0.393632147507352, 0.7842993779115092]",
];

/// The options a `PlotTheme` stands for, as
/// `Charting`ResolvePlotTheme[theme, Plot]` (or `ListPlot` for
/// `list_plot`) resolves them in WL 15, with `DefaultPlotStyle` becoming
/// `PlotStyle`, in input form; `None` for a theme without such a table.
/// The second value is whether the theme labels the plot with
/// `Charting`SimpleTicks` (`PlotOptions::simple_ticks`).
fn plot_theme_options(
  theme: &str,
  list_plot: bool,
) -> Option<(Vec<String>, bool)> {
  let palette = |colors: &[&str], thickness: &str| {
    let styles: Vec<String> = colors
      .iter()
      .map(|c| {
        format!(
          "Directive[{c}, CapForm[\"Butt\"], AbsoluteThickness[{thickness}]]"
        )
      })
      .collect();
    format!("PlotStyle -> {{{}}}", styles.join(", "))
  };
  let dotted = "GridLinesStyle -> Directive[GrayLevel[0.4, 0.5], \
                AbsoluteThickness[1], AbsoluteDashing[{1, 2}]]";
  let left_bottom = "Frame -> {{True, False}, {True, False}}";
  let hidden_left = "FrameStyle -> {{StrokeForm[Opacity[0]], \
                     StrokeForm[Opacity[0]]}, {Automatic, None}}";
  // Only the list plots of these themes mark their points.
  let open_markers =
    || list_plot.then(|| "PlotMarkers -> \"OpenMarkers\"".to_string());
  let opts: (Vec<String>, bool) = match theme {
    "Marketing" => (
      [
        Some("Axes -> False".into()),
        Some("FillingStyle -> Opacity[0.3]".into()),
        Some(left_bottom.into()),
        Some("GridLines -> Automatic".into()),
        Some(
          "GridLinesStyle -> Directive[AbsoluteThickness[1], GrayLevel[0, 0.2]]"
            .into(),
        ),
        open_markers(),
        Some(
          "Prolog -> {GrayLevel[0, 0.7], Rectangle[Scaled[{0, 0}], \
           Scaled[{1, 1}]]}"
            .into(),
        ),
        Some(palette(&MARKETING_COLORS, "3")),
      ]
      .into_iter()
      .flatten()
      .collect(),
      true,
    ),
    "Business" => (
      [
        Some("Axes -> False".into()),
        Some("FillingStyle -> Opacity[0.3]".into()),
        Some(left_bottom.into()),
        Some(hidden_left.into()),
        Some("GridLines -> {None, Automatic}".into()),
        Some(dotted.into()),
        open_markers(),
        Some(palette(&BUSINESS_COLORS, "3")),
      ]
      .into_iter()
      .flatten()
      .collect(),
      true,
    ),
    "Web" => (
      vec![
        "Axes -> False".into(),
        "FillingStyle -> Opacity[0.3]".into(),
        left_bottom.into(),
        hidden_left.into(),
        "GridLines -> None".into(),
        palette(&WEB_COLORS, "3"),
      ],
      true,
    ),
    "Detailed" => (
      vec![
        "Axes -> False".into(),
        "Frame -> True".into(),
        "GridLines -> Automatic".into(),
        dotted.into(),
      ],
      false,
    ),
    "Scientific" => (
      vec![
        "Axes -> False".into(),
        "FillingStyle -> Opacity[0.3]".into(),
        "Frame -> True".into(),
        "GridLines -> {{0}, {0}}".into(),
        "GridLinesStyle -> Directive[GrayLevel[0.5, 0.4]]".into(),
        palette(&SCIENTIFIC_COLORS, "1.6"),
      ],
      false,
    ),
    "Minimal" => (
      vec![
        "Axes -> True".into(),
        "Frame -> False".into(),
        "Ticks -> None".into(),
      ],
      false,
    ),
    "Grid" => (vec!["GridLines -> Automatic".into(), dotted.into()], false),
    "Frame" => (
      vec![
        "Axes -> False".into(),
        "Frame -> True".into(),
        "GridLines -> None".into(),
      ],
      false,
    ),
    _ => return None,
  };
  Some(opts)
}

/// The option name of a `name -> value` or `name :> value` rule, read
/// without evaluating the value (a held `EvaluationMonitor` must not fire).
fn rule_option_name(opt: &Expr) -> Option<&str> {
  match opt {
    Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
      match pattern.as_ref() {
        Expr::Identifier(name) => Some(name.as_str()),
        Expr::String(name) => Some(name.as_str()),
        _ => None,
      }
    }
    _ => None,
  }
}

/// The options of a plot call with its `PlotTheme` expanded: the explicit
/// options, then those of the theme (see [`plot_theme_options`]) that no
/// explicit option sets. No name is given twice by the theme part, so the
/// result reads the same to a parser that keeps the first setting of an
/// option as to one that keeps the last. A theme without a table is left
/// in place. The flag is whether the theme asks for `SimpleTicks`.
pub(crate) fn expand_plot_theme(
  options: &[Expr],
  list_plot: bool,
) -> (Vec<Expr>, bool) {
  let mut theme_opts = None;
  let mut explicit = Vec::with_capacity(options.len());
  for opt in options {
    if rule_option_name(opt) == Some("PlotTheme")
      && let Some((_, value)) =
        crate::functions::graphics::option_name_value(opt)
      && let Expr::String(theme) = &*value
      && let Some(table) = plot_theme_options(theme, list_plot)
    {
      // The first `PlotTheme` wins, like any other repeated option.
      theme_opts.get_or_insert(table);
      continue;
    }
    explicit.push(opt.clone());
  }
  let Some((table, simple_ticks)) = theme_opts else {
    return (explicit, false);
  };
  let given: std::collections::HashSet<&str> =
    options.iter().filter_map(rule_option_name).collect();
  let themed: Vec<Expr> = table
    .iter()
    .filter_map(|o| crate::syntax::string_to_expr(o).ok())
    .filter(|o| rule_option_name(o).is_none_or(|n| !given.contains(n)))
    .collect();
  explicit.extend(themed);
  (explicit, simple_ticks)
}

/// Parse a `GridLines` option value into per-axis (x, y) specifications.
///
/// Supported forms:
/// - `Automatic` / `All` / `True` → automatic lines on both axes
/// - `None` / `False` → no grid lines
/// - `{xspec, yspec}` → independent specs per axis, where each spec is
///   `Automatic`, `None`, or a list of entries. Each entry is a bare position
///   or a `{position, directive}` pair (e.g. `{Pi, Dashed}`, `{1, Orange}`).
pub(crate) fn parse_grid_lines_spec(expr: &Expr) -> (GridSide, GridSide) {
  match expr {
    Expr::List(items) if items.len() == 2 => {
      (parse_grid_side(&items[0]), parse_grid_side(&items[1]))
    }
    Expr::Identifier(s) if s == "None" || s == "False" => {
      (GridSide::None, GridSide::None)
    }
    Expr::Identifier(s) if s == "Automatic" || s == "All" || s == "True" => {
      (GridSide::Automatic, GridSide::Automatic)
    }
    _ => (GridSide::None, GridSide::None),
  }
}

/// Parse one axis side of a `GridLines` spec.
fn parse_grid_side(expr: &Expr) -> GridSide {
  match expr {
    Expr::Identifier(s) if s == "None" || s == "False" => GridSide::None,
    Expr::Identifier(s) if s == "Automatic" || s == "All" || s == "True" => {
      GridSide::Automatic
    }
    Expr::List(items) => GridSide::Explicit(
      items.iter().filter_map(parse_grid_line_entry).collect(),
    ),
    other => match try_eval_to_f64(other) {
      Some(pos) => GridSide::Explicit(vec![GridLine {
        pos,
        style: SeriesStyle::default(),
      }]),
      None => GridSide::None,
    },
  }
}

/// Parse a single grid-line entry: a bare position, or a `{position, directive}`
/// pair where the directive styles just that line.
fn parse_grid_line_entry(entry: &Expr) -> Option<GridLine> {
  match entry {
    Expr::Identifier(s) if s == "None" => None,
    Expr::List(pair) if pair.len() == 2 => {
      let pos = try_eval_to_f64(&pair[0])?;
      let mut style = SeriesStyle::default();
      apply_style_directive(&pair[1], &mut style);
      Some(GridLine { pos, style })
    }
    other => {
      let pos = try_eval_to_f64(other)?;
      Some(GridLine {
        pos,
        style: SeriesStyle::default(),
      })
    }
  }
}

/// Apply a parsed `GridSide` to the matching PlotOptions fields.
pub(crate) fn apply_grid_side(
  side: GridSide,
  auto_flag: &mut bool,
  explicit: &mut Vec<GridLine>,
) {
  match side {
    GridSide::None => {
      *auto_flag = false;
      explicit.clear();
    }
    GridSide::Automatic => {
      *auto_flag = true;
      explicit.clear();
    }
    GridSide::Explicit(v) => {
      *auto_flag = false;
      *explicit = v;
    }
  }
}

/// Parse an `Axes` option value into `(show_x, show_y)`. Accepts `True` /
/// `False` and the per-axis `{xbool, ybool}` form; anything else (e.g.
/// `Automatic`) leaves the current setting alone.
/// Parse `AxesOrigin -> {x, y}` into the point the axes cross at. A
/// coordinate that is not numeric (`Automatic`) is left to the automatic
/// rule, see [`crate::functions::plot_axes::AxisScale::origin`].
pub(crate) fn parse_axes_origin(value: &Expr) -> (Option<f64>, Option<f64>) {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items) if items.len() == 2 => {
      (try_eval_to_f64(&items[0]), try_eval_to_f64(&items[1]))
    }
    _ => (None, None),
  }
}

/// Widen an automatic data range so it contains an explicit `AxesOrigin`
/// coordinate, as Wolfram does; `None` (an automatic origin) leaves it alone.
pub(crate) fn include_in_range(
  (min, max): (f64, f64),
  origin: Option<f64>,
) -> (f64, f64) {
  match origin {
    Some(o) if o.is_finite() => (min.min(o), max.max(o)),
    _ => (min, max),
  }
}

pub(crate) fn parse_axes_option(value: &Expr) -> Option<(bool, bool)> {
  let is_true = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "True");
  match value {
    Expr::Identifier(s) if s == "True" => Some((true, true)),
    Expr::Identifier(s) if s == "False" || s == "None" => Some((false, false)),
    Expr::List(items) if items.len() == 2 => {
      Some((is_true(&items[0]), is_true(&items[1])))
    }
    _ => None,
  }
}

/// Parse a style option's value — `Directive[Red, Dashed]`, a bare color,
/// `{Thick, Blue}`, … — into the style it describes. `Automatic` / `None`,
/// and any value that sets nothing, come back as `None` so the caller keeps
/// its built-in look. Shared by `GridLinesStyle`, `ContourStyle` and the
/// other single-style options.
pub(crate) fn parse_style_directives(value: &Expr) -> Option<SeriesStyle> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  if matches!(&val, Expr::Identifier(s) if s == "Automatic" || s == "None") {
    return None;
  }
  let mut style = SeriesStyle::default();
  match &val {
    Expr::List(items) => {
      for item in items {
        apply_style_directive(item, &mut style);
      }
    }
    other => apply_style_directive(other, &mut style),
  }
  (style.color.is_some()
    || style.thickness.is_some()
    || style.dashing.is_some())
  .then_some(style)
}

/// How one edge of a plot's frame is painted: its line, its tick marks and
/// its tick labels.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct FrameEdgePaint {
  pub line: RGBColor,
  /// Stroke width of the line, in render units.
  pub line_width: u32,
  pub mark: RGBColor,
  /// Stroke width of the tick marks, in render units.
  pub mark_width: f64,
  pub label: RGBColor,
}

/// The paint of each frame edge (bottom, left, top, right). The line takes
/// its `FrameStyle` (else the theme's `gray` at one display pixel); the
/// tick marks their `FrameTicksStyle` over the `FrameStyle`; the labels the
/// colour of the `FrameTicksStyle`, else of the `FrameStyle`, else of the
/// `LabelStyle`.
pub(crate) fn frame_edge_paints(
  opts: &PlotOptions,
  gray: RGBColor,
) -> [FrameEdgePaint; 4] {
  let rgb = |c: WoxiColor| {
    let (r, g, b) = color_to_rgb8(c);
    RGBColor(r, g, b)
  };
  let width = |t: f64| ((t * RESOLUTION_SCALE as f64).round() as u32).max(1);
  std::array::from_fn(|edge| {
    let frame = opts.frame_styles[edge].as_ref();
    let ticks = opts.frame_ticks_styles[edge].as_ref();
    let frame_color = frame.and_then(|s| s.color);
    let tick_color = ticks.and_then(|s| s.color).or(frame_color);
    let line_width = frame
      .and_then(|s| s.thickness)
      .map_or(RESOLUTION_SCALE, width);
    FrameEdgePaint {
      line: frame_color.map_or(gray, rgb),
      line_width,
      mark: tick_color.map_or(gray, rgb),
      mark_width: ticks
        .and_then(|s| s.thickness)
        .or(frame.and_then(|s| s.thickness))
        .map_or(RESOLUTION_SCALE, width) as f64,
      label: tick_color
        .or(opts.label_style.and_then(|s| s.color))
        .map_or(gray, rgb),
    }
  })
}

/// Parse an axis style option (`AxesStyle`, `TicksStyle`) into the style of
/// the x and the y axis: `{xstyle, ystyle}` styles each axis on its own, any
/// other value styles both alike.
pub(crate) fn parse_axis_pair_styles(value: &Expr) -> [Option<SeriesStyle>; 2] {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items) if items.len() == 2 => [
      parse_style_directives(&items[0]),
      parse_style_directives(&items[1]),
    ],
    other => {
      let style = parse_style_directives(other);
      [style.clone(), style]
    }
  }
}

/// The SVG stroke colour and width (in render units, `sf` per display pixel)
/// a style gives a line, falling back to `color` / `width` for whatever it
/// leaves unset.
pub(crate) fn style_stroke(
  style: Option<&SeriesStyle>,
  color: &str,
  width: f64,
  sf: f64,
) -> (String, f64) {
  let stroke = style.and_then(|s| s.color).map_or_else(
    || color.to_string(),
    |c| {
      let (r, g, b) = color_to_rgb8(c);
      format!("rgb({r},{g},{b})")
    },
  );
  let w = style.and_then(|s| s.thickness).map_or(width, |t| t * sf);
  (stroke, w)
}

/// Parse a PlotRange option value into (x_range, y_range) overrides.
///
/// Supported forms:
/// - `All` / `Automatic` / `Full` → (None, None)
/// - `r` (a bare number) → (Some((-r, r)), Some((-r, r))) — "r in all
///   directions from the origin", equivalent to `{{-r, r}, {-r, r}}`
/// - `{ymin, ymax}` → (None, Some((ymin, ymax)))
/// - `{{xmin, xmax}, {ymin, ymax}}` → (Some(x), Some(y))
/// - `{All, {ymin, ymax}}` / `{{xmin,xmax}, All}` → only the specified axis
#[allow(clippy::type_complexity)]
pub(crate) fn parse_plot_range(
  value: &Expr,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());

  // Automatic / All / Full → no override
  if matches!(&val, Expr::Identifier(s) if s == "All" || s == "Automatic" || s == "Full")
  {
    return (None, None);
  }

  // A bare positive number `r` sets a square window centered at the
  // origin on both axes.
  if let Some(r) = try_eval_to_f64(&val)
    && r > 0.0
  {
    return (Some((-r, r)), Some((-r, r)));
  }

  let parse_pair = |e: &Expr| -> Option<(f64, f64)> {
    if let Expr::List(items) = e
      && items.len() == 2
    {
      let a = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[0]).unwrap_or_else(|_| items[0].clone()),
      )?;
      let b = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[1]).unwrap_or_else(|_| items[1].clone()),
      )?;
      // Wolfram normalizes a reversed range: `PlotRange -> {3, -3}` plots
      // exactly like `{-3, 3}` (verified against a FrontEnd-saved raster
      // of the oscilloscope Demonstration, which uses the reversed form).
      Some(if a <= b { (a, b) } else { (b, a) })
    } else {
      None
    }
  };

  if let Expr::List(items) = &val
    && items.len() == 2
  {
    // {{xmin,xmax}, {ymin,ymax}} (optionally with All/Automatic as a placeholder)
    if matches!(&items[0], Expr::List(_))
      || matches!(&items[1], Expr::List(_))
      || matches!(&items[0], Expr::Identifier(s) if s == "All" || s == "Automatic" || s == "Full")
    {
      let x_range = parse_pair(&items[0]);
      let y_range = parse_pair(&items[1]);
      // If neither inner is a pair, fall through to {ymin, ymax} handling.
      if x_range.is_some() || y_range.is_some() {
        return (x_range, y_range);
      }
    }

    // {ymin, ymax}: y range only
    if let Some(y) = parse_pair(&val) {
      return (None, Some(y));
    }
  }

  (None, None)
}

/// `PlotRange` for the function/list plots (`Plot`, `ListPlot`, charts, ...).
/// Unlike graphics primitives, where a bare number `r` is a square window,
/// the sampled x domain of a plot stays as is and `r` bounds only the
/// vertical axis: `PlotRange -> r` is `PlotRange -> {-r, r}` on y.
pub(crate) fn parse_series_plot_range(
  value: &Expr,
) -> (Option<(f64, f64)>, Option<(f64, f64)>) {
  let (x, y) = parse_plot_range(value);
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  if try_eval_to_f64(&val).is_some_and(|r| r > 0.0) {
    return (None, y);
  }
  (x, y)
}

/// Parsed `FrameLabel` edges. Empty strings mean "no label on that edge".
#[derive(Default)]
pub(crate) struct FrameLabels {
  pub bottom: String,
  pub left: String,
  pub top: String,
  pub right: String,
  /// Set when `bottom`'s source expression is a `Grid[…]`/`TableForm[…]`
  /// that renders as a picture — `bottom` itself is left empty in that
  /// case.
  pub bottom_graphic: Option<GraphicLabel>,
  /// Likewise for the top edge, e.g. a `Row[…]` legend whose entries
  /// embed small `Graphics`.
  pub top_graphic: Option<GraphicLabel>,
}

/// Convert a single FrameLabel entry to a label string. `None` (the symbol)
/// and unrenderable expressions become an empty string.
fn frame_label_entry(e: &Expr) -> String {
  if matches!(e, Expr::Identifier(s) if s == "None") {
    return String::new();
  }
  crate::functions::graphics::styled_axis_label(e).unwrap_or_default()
}

/// A `FrameLabel` entry that is a `Grid[…]`/`TableForm[…]` — a table meant
/// to be laid out as a picture, not read off as text (`expr_to_label`
/// silently drops it since `Grid` is a known builtin, not an unrenderable
/// applied form). Renders it through the same pipeline a standalone
/// `Grid[…]` uses so the frame label shows the table instead of nothing.
fn frame_label_graphic(e: &Expr) -> Option<GraphicLabel> {
  let val = evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
  // `Style[label, …]` is looked through to the label it styles.
  let mut val = &val;
  while let Expr::FunctionCall { name, args } = val
    && name == "Style"
    && !args.is_empty()
  {
    val = &args[0];
  }
  let Expr::FunctionCall { name, args } = val else {
    return None;
  };
  let svg = match name.as_str() {
    "Grid" | "TableForm" if !args.is_empty() => {
      crate::functions::graphics::grid_svg_with_gaps(args, &[]).ok()?
    }
    // A `Row` holding pictures (a legend swatch drawn as `Graphics`) can
    // only be shown as a picture, not read off as text.
    "Row" if row_holds_graphics(args) => {
      crate::functions::graphics::row_to_svg(args)?
    }
    _ => return None,
  };
  let (width, height) = crate::functions::graphics::svg_natural_size(&svg)?;
  Some(GraphicLabel { svg, width, height })
}

/// Whether the items of a `Row[{…}]` include a `Graphics`, directly or in a
/// nested `Row`/`Style`.
fn row_holds_graphics(args: &[Expr]) -> bool {
  fn has_graphics(e: &Expr) -> bool {
    match e {
      Expr::Graphics { .. } => true,
      Expr::List(items) => items.iter().any(has_graphics),
      Expr::FunctionCall { name, .. } if name == "Graphics" => true,
      Expr::FunctionCall { name, args } if name == "Row" || name == "Style" => {
        args.first().is_some_and(has_graphics)
      }
      _ => false,
    }
  }
  args.first().is_some_and(has_graphics)
}

/// One side of `Ticks -> {xspec, yspec}`: an explicit list of positions,
/// each optionally carrying the text to draw at it (`{pos, label}`).
/// `None` for `Automatic`/`None`/anything that is not a list.
///
/// Each label comes back as the SVG `<text>` content to draw — already
/// escaped, and typeset where the tick asked for something more than plain
/// text — so [`explicit_ticks_svg`] can emit it verbatim.
pub(crate) fn parse_explicit_ticks(value: &Expr) -> Option<Vec<(f64, String)>> {
  parse_explicit_tick_entries(value)
    .map(|ticks| ticks.into_iter().map(|t| (t.pos, t.label)).collect())
}

/// One entry of an explicit tick list: `pos`, `{pos, label}`, `{pos, label,
/// len}`, `{pos, label, {plen, mlen}}` or `{pos, label, len, style}`.
#[derive(Clone, Debug)]
pub(crate) struct ExplicitTick {
  pub pos: f64,
  /// The SVG `<text>` content drawn at the tick (see
  /// [`parse_explicit_ticks`]).
  pub label: String,
  /// How far the mark reaches into and out of the plot, as fractions of the
  /// plot width; `None` gives it the automatic major tick length, inward.
  pub len: Option<(f64, f64)>,
  /// The mark's own style (`{pos, label, len, Red}`); it never colours the
  /// label.
  pub style: Option<SeriesStyle>,
}

/// The entries of an explicit tick list with their lengths and styles; see
/// [`parse_explicit_ticks`] for how positions and labels are read.
pub(crate) fn parse_explicit_tick_entries(
  value: &Expr,
) -> Option<Vec<ExplicitTick>> {
  // A spec written out as a list keeps its entries as written: a tick is
  // labelled with the expression standing at it, and `3 Pi/2` set as one
  // fraction is not the same reading as the `(3/2) Pi` it evaluates to.
  // Anything else (a symbol, a `Table[…]`) has to be worked out first.
  let val = match value {
    Expr::List(_) => value.clone(),
    _ => evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone()),
  };
  let Expr::List(entries) = &val else {
    return None;
  };
  // `len` reaches inward only (whatever its sign); `{plen, mlen}` reaches
  // `plen` in and `mlen` out.
  let tick_len = |e: &Expr| match e {
    Expr::List(pm) if pm.len() == 2 => {
      Some((try_eval_to_f64(&pm[0])?, try_eval_to_f64(&pm[1])?))
    }
    other => try_eval_to_f64(other).map(|l| (l.abs(), 0.0)),
  };
  let ticks: Vec<ExplicitTick> = entries
    .iter()
    .filter_map(|entry| match entry {
      Expr::List(pair) if pair.len() >= 2 => {
        let pos = tick_position(&pair[0])?;
        // A label given as text may still embed box notation (a notebook
        // writes a superscript that way), so it renders like every other
        // label a plot draws rather than being escaped raw.
        // The label is kept as written, so a call inside it — the
        // `ToString[a]` of `Row[{"a = ", ToString[a]}]` — is worked out
        // before it is read as text.
        let label_expr =
          evaluate_expr_to_expr(&pair[1]).unwrap_or_else(|_| pair[1].clone());
        let label = crate::functions::chart::expr_to_label(&label_expr)
          .map_or_else(
            || crate::functions::graphics::svg_escape(&format_tick(pos)),
            |text| crate::functions::graphics::box_string_to_svg(&text),
          );
        Some(ExplicitTick {
          pos,
          label,
          len: pair.get(2).and_then(tick_len),
          style: pair.get(3).and_then(parse_style_directives),
        })
      }
      other => {
        let other = &numeric_tick_value(other);
        let pos = tick_position(other)?;
        Some(ExplicitTick {
          pos,
          label: bare_tick_label(other, pos),
          len: None,
          style: None,
        })
      }
    })
    .collect();
  // An empty list (`Ticks -> {{}, {}}`) is an explicit request for no ticks
  // on that axis, not a fall-back to the automatic ones.
  (!ticks.is_empty() || entries.is_empty()).then_some(ticks)
}

/// A bare tick position that evaluates to a plain number (`rate - vac` with
/// both set, `N[f[x], 4]`) is that number, as plot options are evaluated
/// before the plot reads them. A symbolic result (`3 Pi/2`) keeps the form it
/// was written in so it is labelled as typed.
fn numeric_tick_value(e: &Expr) -> Expr {
  match evaluate_expr_to_expr(e) {
    Ok(
      v @ (Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..)),
    ) => v,
    _ => e.clone(),
  }
}

/// The numeric position of a tick. The spec is kept as written, so a position
/// that depends on variables (`a/4` with `a` set elsewhere) is evaluated here
/// before being read as a number.
fn tick_position(e: &Expr) -> Option<f64> {
  try_eval_to_f64(e).or_else(|| {
    evaluate_expr_to_expr(e)
      .ok()
      .and_then(|v| try_eval_to_f64(&v))
  })
}

/// The `<text>` content drawn at a tick given as a bare position rather than
/// as a `{pos, label}` pair. Wolfram labels such a tick with the position
/// *expression*, typeset — `Ticks -> {{0, Pi/2, Pi, 3 Pi/2, 2 Pi}, …}` reads
/// "0", "π/2", "π", "3π/2", "2π" — so only a literal number is labelled by
/// its value; anything symbolic is set the way the expression is written.
pub(crate) fn bare_tick_label(e: &Expr, pos: f64) -> String {
  fn is_literal_number(e: &Expr) -> bool {
    match e {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..) => true,
      // A negative literal may arrive either folded into the number or as
      // a minus applied to it, depending on how the list was written.
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => is_literal_number(operand),
      _ => false,
    }
  }
  if is_literal_number(e) {
    return crate::functions::graphics::svg_escape(&format_tick(pos));
  }
  crate::functions::graphics::expr_to_svg_markup(e)
}

/// Apply a `FrameLabel` value to a plot's options. Bottom and left reuse
/// the axes-label render path; top and right get their own frame edges.
/// The space, in render units, each side of the plotting area gets under an
/// `ImagePadding` specification.
pub(crate) struct PaddedMargins {
  pub left: u32,
  pub right: u32,
  pub bottom: u32,
  pub top: u32,
}

/// Wolfram's default plot `AspectRatio`: 1/GoldenRatio, the height/width of
/// the plotting area when nothing else fixes it.
pub(crate) const DEFAULT_ASPECT_RATIO: f64 = 0.618_033_988_749_895;

/// Lay out the plotting area inside an image with `ImagePadding`.
///
/// Wolfram treats the padding as a *minimum* per side: the plotting area is
/// then as large as the padding leaves while keeping its `AspectRatio`, and
/// whatever space is left over is split evenly between the two sides of each
/// axis. So `ImagePadding -> 20` on a wide-but-short image leaves far more
/// than 20 px at the left — which is what keeps the tick labels of e.g.
/// `ListPlot[…, ImageSize -> {400, 200}, ImagePadding -> 20]` from being
/// clipped.
fn padded_margins(
  [left, right, bottom, top]: [f64; 4],
  (render_width, render_height): (u32, u32),
  aspect_ratio: Option<f64>,
  sf: f64,
) -> PaddedMargins {
  let (left, right, bottom, top) =
    (left * sf, right * sf, bottom * sf, top * sf);
  let avail_w = (render_width as f64 - left - right).max(0.0);
  let avail_h = (render_height as f64 - bottom - top).max(0.0);
  let ar = aspect_ratio.unwrap_or(DEFAULT_ASPECT_RATIO);
  let (plot_w, plot_h) = if ar <= 0.0 || avail_w * ar <= avail_h {
    (avail_w, avail_w * ar)
  } else {
    (avail_h / ar, avail_h)
  };
  let (slack_w, slack_h) = ((avail_w - plot_w) / 2.0, (avail_h - plot_h) / 2.0);
  PaddedMargins {
    left: (left + slack_w).round() as u32,
    right: (right + slack_w).round() as u32,
    bottom: (bottom + slack_h).round() as u32,
    top: (top + slack_h).round() as u32,
  }
}

/// Parse an `ImagePadding` value into `[left, right, bottom, top]` display
/// pixels. Accepted forms mirror Wolfram: a single number (all four sides),
/// `{{left, right}, {bottom, top}}`, `None`/`0` (no padding) and
/// `Automatic`/`All` (which keep the automatic margins, i.e. `None` here).
pub(crate) fn parse_image_padding(value: &Expr) -> Option<[f64; 4]> {
  fn side(e: &Expr) -> Option<f64> {
    match e {
      Expr::Identifier(v) if v == "None" => Some(0.0),
      _ => crate::functions::graphics::expr_to_f64(e),
    }
  }
  match value {
    Expr::Identifier(v) if v == "Automatic" || v == "All" => None,
    Expr::Identifier(v) if v == "None" => Some([0.0; 4]),
    Expr::List(pairs) if pairs.len() == 2 => {
      let (Expr::List(lr), Expr::List(bt)) = (&pairs[0], &pairs[1]) else {
        // `{h, v}` is not a Wolfram form, so a two-element list of numbers
        // is only meaningful as the nested one.
        return None;
      };
      if lr.len() != 2 || bt.len() != 2 {
        return None;
      }
      Some([side(&lr[0])?, side(&lr[1])?, side(&bt[0])?, side(&bt[1])?])
    }
    _ => side(value).map(|p| [p; 4]),
  }
}

/// The edges of a plot frame that are drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct FrameEdges {
  pub left: bool,
  pub right: bool,
  pub bottom: bool,
  pub top: bool,
}

impl FrameEdges {
  pub(crate) const ALL: Self = Self {
    left: true,
    right: true,
    bottom: true,
    top: true,
  };
  pub(crate) const NONE: Self = Self {
    left: false,
    right: false,
    bottom: false,
    top: false,
  };

  /// These edges minus the ones in `hidden`.
  pub(crate) fn without(self, hidden: Self) -> Self {
    Self {
      left: self.left && !hidden.left,
      right: self.right && !hidden.right,
      bottom: self.bottom && !hidden.bottom,
      top: self.top && !hidden.top,
    }
  }
}

/// The edges a `Frame` value draws: all of them for `True | All`, the
/// named ones for `{{left, right}, {bottom, top}}` and for the older
/// `{bottom, left, top, right}`.
pub(crate) fn parse_frame_edges(value: &Expr) -> FrameEdges {
  let on = |e: &Expr| parse_frame_option(e);
  match value {
    Expr::List(items) if items.len() == 2 => match (&items[0], &items[1]) {
      (Expr::List(lr), Expr::List(bt)) if lr.len() == 2 && bt.len() == 2 => {
        FrameEdges {
          left: on(&lr[0]),
          right: on(&lr[1]),
          bottom: on(&bt[0]),
          top: on(&bt[1]),
        }
      }
      _ => FrameEdges::ALL,
    },
    Expr::List(items) if items.len() == 4 => FrameEdges {
      bottom: on(&items[0]),
      left: on(&items[1]),
      top: on(&items[2]),
      right: on(&items[3]),
    },
    _ => FrameEdges::ALL,
  }
}

/// `FrameStyle -> {{left, right}, {bottom, top}}` with an edge styled
/// `StrokeForm[Opacity[0]]` (or `None`) leaves that edge's line and tick
/// marks undrawn — its tick labels stay — as the `"Business"` and `"Web"`
/// themes draw their left edge.
pub(crate) fn hide_transparent_frame_edges(
  value: &Expr,
  opts: &mut PlotOptions,
) {
  fn hidden(e: &Expr) -> bool {
    match e {
      Expr::Identifier(n) => n == "None",
      Expr::FunctionCall { name, args }
        if name == "StrokeForm" && args.len() == 1 =>
      {
        matches!(&args[0], Expr::FunctionCall { name, args }
          if name == "Opacity"
            && args.first().and_then(try_eval_to_f64) == Some(0.0))
      }
      Expr::FunctionCall { name, args } if name == "Directive" => {
        args.iter().any(hidden)
      }
      _ => false,
    }
  }
  if let Expr::List(items) = value
    && let [Expr::List(lr), Expr::List(bt)] = items.as_slice()
    && lr.len() == 2
    && bt.len() == 2
  {
    // The edge stays part of the frame — its tick labels keep their room —
    // only its line and tick marks go.
    opts.frame_hidden_edges = FrameEdges {
      left: hidden(&lr[0]),
      right: hidden(&lr[1]),
      bottom: hidden(&bt[0]),
      top: hidden(&bt[1]),
    };
  }
}

/// The `Charting`SimpleTicks` WL 15 plot themes label a frame with: majors
/// only, stepped by the smallest of 1, 2, 5 × 10^k that leaves at most six
/// of them in `min..max`, with 0 written as `0`.
pub(crate) fn simple_ticks(min: f64, max: f64) -> Vec<(f64, String)> {
  let span = max - min;
  if !(span.is_finite() && span > 0.0) {
    return Vec::new();
  }
  let count =
    |step: f64| ((max / step).floor() - (min / step).ceil()) as i64 + 1;
  let mut mag = 10_f64.powf((span / 6.0).log10().floor() - 1.0);
  let step = 'search: loop {
    for m in [1.0, 2.0, 5.0] {
      if count(m * mag) <= 6 {
        break 'search m * mag;
      }
    }
    mag *= 10.0;
  };
  let first = (min / step - 1e-9).ceil() as i64;
  let last = (max / step + 1e-9).floor() as i64;
  (first..=last)
    .map(|k| {
      let v = k as f64 * step;
      let label = if k == 0 {
        "0".to_string()
      } else {
        format_tick_with_step(v, step)
      };
      (if k == 0 { 0.0 } else { v }, label)
    })
    .collect()
}

/// Apply a `Frame` option: whether there is a frame and which of its edges
/// are drawn.
pub(crate) fn apply_frame_option(value: &Expr, opts: &mut PlotOptions) {
  opts.frame = parse_frame_option(value);
  opts.frame_edges = parse_frame_edges(value);
}

/// `Frame -> True | All` draws the boxed frame; anything else (`False`,
/// `Automatic`, a per-edge list) leaves the plain axes.
/// `Frame -> True | All`, or the per-edge forms `{l, r, b, t}` and
/// `{{l, r}, {b, t}}`, the frame is drawn when any edge asks for it (which
/// edges: [`parse_frame_edges`]).
pub(crate) fn parse_frame_option(value: &Expr) -> bool {
  match value {
    Expr::Identifier(v) => v == "True" || v == "All",
    Expr::List(items) => items.iter().any(parse_frame_option),
    _ => false,
  }
}

pub(crate) fn apply_frame_label_option(value: &Expr, opts: &mut PlotOptions) {
  let fl = parse_frame_label(value);
  if !fl.bottom.is_empty() {
    opts.frame_label_bottom = Some(fl.bottom);
  }
  if fl.bottom_graphic.is_some() {
    opts.frame_label_bottom_graphic = fl.bottom_graphic;
  }
  if !fl.left.is_empty() {
    opts.frame_label_left = Some(fl.left);
  }
  if !fl.top.is_empty() {
    opts.frame_label_top = Some(fl.top);
  }
  if fl.top_graphic.is_some() {
    opts.frame_label_top_graphic = fl.top_graphic;
  }
  if !fl.right.is_empty() {
    opts.frame_label_right = Some(fl.right);
  }
}

/// Parse a `FrameLabel` option value, supporting both forms:
///   `{bottom, left}` and `{{left, right}, {bottom, top}}`.
/// A bare label applies to the bottom edge.
pub(crate) fn parse_frame_label(value: &Expr) -> FrameLabels {
  let mut out = parse_frame_label_edges(value);
  // An edge drawn as a picture has no text to draw as well.
  if out.bottom_graphic.is_some() {
    out.bottom.clear();
  }
  if out.top_graphic.is_some() {
    out.top.clear();
  }
  out
}

fn parse_frame_label_edges(value: &Expr) -> FrameLabels {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let mut out = FrameLabels::default();
  if let Expr::List(items) = &val {
    // 4-element nested form: both entries are themselves lists.
    if items.len() == 2
      && let (Expr::List(lr), Expr::List(bt)) = (&items[0], &items[1])
    {
      if let Some(e) = lr.first() {
        out.left = frame_label_entry(e);
      }
      if let Some(e) = lr.get(1) {
        out.right = frame_label_entry(e);
      }
      if let Some(e) = bt.first() {
        out.bottom = frame_label_entry(e);
        out.bottom_graphic = frame_label_graphic(e);
      }
      if let Some(e) = bt.get(1) {
        out.top = frame_label_entry(e);
        out.top_graphic = frame_label_graphic(e);
      }
      return out;
    }
    // 2-element form `{bottom, left}`.
    if let Some(e) = items.first() {
      out.bottom = frame_label_entry(e);
      out.bottom_graphic = frame_label_graphic(e);
    }
    if let Some(e) = items.get(1) {
      out.left = frame_label_entry(e);
    }
    // Flat 4-element form `{bottom, left, top, right}`.
    if let Some(e) = items.get(2) {
      out.top = frame_label_entry(e);
      out.top_graphic = frame_label_graphic(e);
    }
    if let Some(e) = items.get(3) {
      out.right = frame_label_entry(e);
    }
  } else {
    // A bare label labels the bottom edge.
    out.bottom = frame_label_entry(&val);
    out.bottom_graphic = frame_label_graphic(&val);
  }
  out
}

/// Parse PlotLegends option value into a list of legend strings.
/// Returns (legends, is_automatic, is_expressions, legend_position).
pub(crate) fn parse_plot_legends(
  value: &Expr,
) -> (Vec<String>, bool, bool, LegendPosition) {
  let val = evaluate_expr_to_expr(value).unwrap_or(value.clone());

  // Check for Placed[content, position] wrapper
  if let Expr::FunctionCall { name, args } = &val
    && name == "Placed"
    && args.len() == 2
  {
    let pos = match &args[1] {
      Expr::Identifier(s) => match s.as_str() {
        "Top" | "Above" => LegendPosition::Top,
        "Bottom" | "Below" => LegendPosition::Bottom,
        _ => LegendPosition::Right,
      },
      _ => LegendPosition::Right,
    };
    let (labels, auto, expressions, _) = parse_plot_legends(&args[0]);
    return (labels, auto, expressions, pos);
  }

  // `LineLegend[{labels}, opts]` / `SwatchLegend[{labels}, opts]` /
  // `PointLegend[{labels}, opts]` name the series just like a plain list of
  // labels; the explicit `{styles}, {labels}` form carries them second.
  if let Expr::FunctionCall { name, args } = &val
    && matches!(name.as_str(), "LineLegend" | "SwatchLegend" | "PointLegend")
    && let Some(first) = args.first()
  {
    let labels = match args.get(1) {
      Some(second @ Expr::List(_)) => second,
      _ => first,
    };
    if matches!(labels, Expr::List(_)) {
      return parse_plot_legends(labels);
    }
  }

  match &val {
    Expr::Identifier(s) if s == "Automatic" => {
      (Vec::new(), true, false, LegendPosition::Right)
    }
    Expr::Identifier(s) if s == "None" => {
      (Vec::new(), false, false, LegendPosition::Right)
    }
    Expr::String(s) if s == "Expressions" => {
      (Vec::new(), false, true, LegendPosition::Right)
    }
    Expr::List(items) => {
      let labels = items
        .iter()
        .map(|item| {
          crate::functions::graphics::styled_axis_label(item)
            .unwrap_or_else(|| expr_to_string(item))
        })
        .collect();
      (labels, false, false, LegendPosition::Right)
    }
    Expr::String(s) => (vec![s.clone()], false, false, LegendPosition::Right),
    _ => (Vec::new(), false, false, LegendPosition::Right),
  }
}

/// The marker shape and explicit per-entry colors of a `PlotLegends`
/// value: `SwatchLegend[{Blue, Red}, {"a", "b"}]` draws blue and red squares
/// whatever the series look like, `SwatchLegend[{"a", "b"}]` squares in the
/// series colors, and a plain label list markers matching the series.
pub(crate) fn parse_legend_markers(
  value: &Expr,
) -> (LegendMarkerKind, Vec<Option<(u8, u8, u8)>>) {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  let val = match &val {
    Expr::FunctionCall { name, args }
      if name == "Placed" && !args.is_empty() =>
    {
      args[0].clone()
    }
    _ => val,
  };
  let Expr::FunctionCall { name, args } = &val else {
    return (LegendMarkerKind::Automatic, Vec::new());
  };
  let kind = match name.as_str() {
    "LineLegend" => LegendMarkerKind::Line,
    "SwatchLegend" => LegendMarkerKind::Swatch,
    "PointLegend" => LegendMarkerKind::Point,
    _ => return (LegendMarkerKind::Automatic, Vec::new()),
  };
  // Only the two-list `XLegend[{styles}, {labels}]` form carries styles.
  let colors = match (args.first(), args.get(1)) {
    (Some(Expr::List(styles)), Some(Expr::List(_))) => {
      styles.iter().map(legend_style_color).collect()
    }
    _ => Vec::new(),
  };
  (kind, colors)
}

/// The color a legend style (`Red`, `Directive[Thick, Red]`, `{Red}`)
/// draws its marker in.
fn legend_style_color(style: &Expr) -> Option<(u8, u8, u8)> {
  if let Some(c) = crate::functions::graphics::parse_color(style) {
    let ch = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    return Some((ch(c.r), ch(c.g), ch(c.b)));
  }
  match style {
    Expr::FunctionCall { name, args } if name == "Directive" => {
      args.iter().rev().find_map(legend_style_color)
    }
    Expr::List(items) => items.iter().rev().find_map(legend_style_color),
    _ => None,
  }
}

/// Whether `Callout[data, label, pos…]` sits at its automatic place, the
/// end of its curve or dataset (no position, or `Automatic`).
pub(crate) fn callout_at_series_end(callout_args: &[Expr]) -> bool {
  match callout_args.get(2) {
    None => true,
    Some(Expr::Identifier(p)) => p == "Automatic",
    Some(_) => false,
  }
}

/// Parse a `PlotLabels -> {l1, l2, …}` value into one optional label per
/// curve; `None` and `Automatic` entries leave their curve unlabeled.
fn parse_plot_labels(value: &Expr) -> Vec<Option<String>> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  fn label(item: &Expr) -> Option<String> {
    match item {
      Expr::Identifier(s) if s == "None" || s == "Automatic" => None,
      // `Callout[l, pos…]` / `Placed[l, pos]` label the curve with `l`.
      Expr::FunctionCall { name, args }
        if (name == "Callout" || name == "Placed") && !args.is_empty() =>
      {
        label(&args[0])
      }
      other => Some(crate::functions::graphics::traditional_label(other)),
    }
  }
  match &val {
    Expr::List(items) => items.iter().map(label).collect(),
    // `Callout[{l1, l2, …}, pos…]` / `Placed[{l1, l2, …}, pos]` label every
    // curve with its own entry.
    Expr::FunctionCall { name, args }
      if (name == "Callout" || name == "Placed")
        && matches!(args.first(), Some(Expr::List(_))) =>
    {
      parse_plot_labels(&args[0])
    }
    // Filled in from the curves' expressions by the plot itself.
    Expr::Identifier(s) if s == "None" || s == "Automatic" => Vec::new(),
    Expr::String(s) if s == "Expressions" => Vec::new(),
    other => vec![label(other)],
  }
}

/// Whether a `PlotLabels` value labels each curve with its own expression:
/// `Automatic` and `"Expressions"`.
fn plot_labels_from_expressions(value: &Expr) -> bool {
  matches!(value, Expr::Identifier(s) if s == "Automatic")
    || matches!(value, Expr::String(s) if s == "Expressions")
}

/// Parse a `Background -> color` option value into a plotters fill color.
/// Returns `None` for `Background -> None` and unrecognized values, which
/// keep the theme default.
pub(crate) fn parse_background_option(expr: &Expr) -> Option<RGBColor> {
  let val = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  crate::functions::graphics::parse_color(&val).map(|c| {
    RGBColor(
      (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
      (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
    )
  })
}

/// Range/aspect overrides parsed from options and applied after all options
/// (and the data) are known.
#[derive(Default)]
pub(crate) struct PlotRangeOverrides {
  pub x: Option<(f64, f64)>,
  pub y: Option<(f64, f64)>,
  /// `PlotRange -> All` on the y axis: show every sampled value instead of
  /// the automatic range, which drops extreme outliers.
  pub y_all: bool,
  pub aspect_ratio: Option<f64>,
  /// `AspectRatio -> Automatic`.
  pub aspect_automatic: bool,
}

/// Apply one plot option shared by the function plotters (Plot,
/// ParametricPlot, PolarPlot, …) to `plot_opts` / `overrides`. Returns
/// `true` when `name` names an option this helper understands.
pub(crate) fn apply_common_plot_option(
  name: &str,
  replacement: &Expr,
  plot_opts: &mut PlotOptions,
  overrides: &mut PlotRangeOverrides,
) -> bool {
  match name {
    "ImageSize" => {
      if let Some((w, h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
      {
        plot_opts.svg_width = w;
        plot_opts.svg_height = h;
        plot_opts.full_width = fw;
        plot_opts.image_size_explicit = image_size_explicit_dims(replacement);
        plot_opts.image_size_value = Some(replacement.clone());
      }
    }
    "ClippingStyle" => {
      plot_opts.clipping_style = parse_clipping_style(replacement);
    }
    "ExclusionsStyle" => {
      plot_opts.exclusions_style = parse_exclusions_style(replacement);
    }
    "PlotLabel" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Some(sl) = crate::functions::chart::parse_styled_label(&val) {
        plot_opts.plot_label = Some(sl);
      }
    }
    "AxesLabel" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Expr::List(items) = &val
        && !items.is_empty()
      {
        let x = crate::functions::graphics::styled_axis_label(&items[0])
          .unwrap_or_default();
        // `{xlabel}` names the x axis only.
        let y = items
          .get(1)
          .and_then(crate::functions::graphics::styled_axis_label)
          .unwrap_or_default();
        plot_opts.axes_label = Some((x, y));
      }
    }
    "PlotStyle" => {
      plot_opts.plot_style = parse_plot_style(replacement);
    }
    "ColorFunction" => {
      plot_opts.color_function = match replacement {
        Expr::Identifier(v) if v == "Automatic" || v == "None" => None,
        other => Some(other.clone()),
      };
    }
    "ColorFunctionScaling" => {
      // Evaluated: `Table[Plot[…, ColorFunctionScaling -> cf], {cf, …}]`.
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      let on = |e: &Expr| !matches!(e, Expr::Identifier(v) if v == "False");
      plot_opts.color_function_scaling = match &val {
        Expr::List(items) if !items.is_empty() => {
          (on(&items[0]), items.get(1).is_none_or(on))
        }
        other => (on(other), on(other)),
      };
    }
    // Expanded into the options it stands for by `expand_plot_theme`;
    // a theme without a table changes nothing.
    "PlotTheme" => {}
    "GridLines" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      let (sx, sy) = parse_grid_lines_spec(&val);
      apply_grid_side(
        sx,
        &mut plot_opts.grid_lines_x,
        &mut plot_opts.grid_x_lines,
      );
      apply_grid_side(
        sy,
        &mut plot_opts.grid_lines_y,
        &mut plot_opts.grid_y_lines,
      );
    }
    "GridLinesStyle" => {
      plot_opts.grid_lines_style = parse_style_directives(replacement);
    }
    "PlotRange" => {
      let (rx, ry) = parse_series_plot_range(replacement);
      overrides.x = rx;
      overrides.y = ry;
      overrides.y_all = plot_range_requests_all_y(replacement);
    }
    "Axes" => {
      if let Some(axes) = parse_axes_option(replacement) {
        plot_opts.axes = axes;
      }
    }
    "AxesOrigin" => {
      plot_opts.axes_origin = parse_axes_origin(replacement);
    }
    "AspectRatio" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Some(r) = try_eval_to_f64(&val)
        && r > 0.0
      {
        overrides.aspect_ratio = Some(r);
      } else if matches!(&val, Expr::Identifier(s) if s == "Automatic") {
        overrides.aspect_automatic = true;
      } else if matches!(&val, Expr::Identifier(s) if s == "Full") {
        plot_opts.aspect_full = true;
      }
    }
    "MaxRecursion" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      if let Expr::Integer(n) = val
        && n >= 0
      {
        plot_opts.max_recursion = (n as usize).min(15);
      }
    }
    "PlotPoints" => {
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      match &val {
        Expr::Integer(n) if *n > 0 => plot_opts.plot_points = *n as usize,
        _ => {}
      }
    }
    "Filling" => {
      apply_filling_option(replacement, plot_opts);
    }
    "FillingStyle" => {
      plot_opts.filling_style = parse_filling_style(replacement);
    }
    "Background" => {
      plot_opts.background = parse_background_option(replacement);
    }
    // WL 15 hangs `PlotLabels` off the curve ends on callout leaders.
    "PlotLabels" => {
      plot_opts.plot_labels = parse_plot_labels(replacement);
      plot_opts.plot_label_leaders = vec![true; plot_opts.plot_labels.len()];
    }
    "PlotLegends" => {
      let (labels, _, _, position) = parse_plot_legends(replacement);
      plot_opts.legend_position = position;
      plot_opts.plot_legends = labels;
      let (kind, colors) = parse_legend_markers(replacement);
      plot_opts.legend_marker_kind = kind;
      plot_opts.legend_colors = colors;
    }
    "AxesStyle" => plot_opts.axes_style = parse_axis_pair_styles(replacement),
    "TicksStyle" => {
      plot_opts.ticks_style = parse_axis_pair_styles(replacement);
    }
    "FrameStyle" => {
      plot_opts.frame_styles = parse_frame_edge_styles(replacement);
      hide_transparent_frame_edges(replacement, plot_opts);
    }
    "MeshStyle" => {
      plot_opts.mesh_styles =
        crate::functions::plot_mesh::parse_mesh_styles(replacement);
    }
    "Mesh" => {
      plot_opts.mesh = crate::functions::plot_mesh::parse_mesh(replacement);
    }
    "MeshFunctions" => {
      plot_opts.mesh_functions =
        crate::functions::plot_mesh::parse_mesh_functions(replacement);
    }
    "MeshShading" => {
      plot_opts.mesh_shading =
        crate::functions::plot_mesh::parse_mesh_shading(replacement);
    }
    "Frame" => apply_frame_option(replacement, plot_opts),
    "FrameTicks" => {
      let edges = parse_frame_ticks(replacement);
      plot_opts.frame_ticks = edges.iter().any(|e| !e.is_none());
      if let Some(t) = edges[0].explicit_labels() {
        plot_opts.ticks_x = Some(t);
      }
      if let Some(t) = edges[1].explicit_labels() {
        plot_opts.ticks_y = Some(t);
      }
      plot_opts.frame_tick_edges = edges;
    }
    "Evaluated" => {
      plot_opts.evaluated =
        matches!(replacement, Expr::Identifier(v) if v == "True");
    }
    "ImagePadding" => {
      plot_opts.image_padding = parse_image_padding(replacement);
    }
    "PlotLayout" => {
      plot_opts.plot_layout =
        crate::functions::plot_layout::parse_plot_layout(replacement);
    }
    "FrameLabel" => apply_frame_label_option(replacement, plot_opts),
    "FrameTicksStyle" => {
      plot_opts.tick_label_style = LabelStyleSpec::parse(replacement);
      plot_opts.frame_ticks_styles = parse_frame_edge_styles(replacement);
    }
    "LabelStyle" => {
      plot_opts.label_style = LabelStyleSpec::parse(replacement);
    }
    "Ticks" => match replacement {
      Expr::Identifier(s) if s == "None" || s == "False" => {
        plot_opts.ticks = false;
      }
      Expr::Identifier(s) if s == "Automatic" || s == "All" => {
        plot_opts.ticks = true;
      }
      // `None` and `False` on both sides is `Ticks -> None`.
      Expr::List(items)
        if items.len() == 2
          && items.iter().all(
            |i| matches!(i, Expr::Identifier(s) if s == "None" || s == "False"),
          ) =>
      {
        plot_opts.ticks = false;
      }
      // `Ticks -> {xspec, yspec}`: each side is None, Automatic, or an
      // explicit list of positions (each optionally `{pos, label}`).
      // The y side may be left off — `Ticks -> {xspec}` states the x ticks
      // and leaves the y axis to its default.
      Expr::List(items) if (1..=2).contains(&items.len()) => {
        plot_opts.ticks_x = parse_explicit_ticks(&items[0]);
        if let Some(y) = items.get(1) {
          plot_opts.ticks_y = parse_explicit_ticks(y);
        }
      }
      _ => {}
    },
    "Epilog" => {
      // Evaluate now so primitives inside (e.g. `Line[{ReIm[…], …}]`)
      // resolve to numeric coordinates while the surrounding variable
      // bindings (Block scope) are still live.
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      plot_opts.epilog = match val {
        Expr::List(ref items) => items.to_vec(),
        Expr::Identifier(ref s) if s == "None" => Vec::new(),
        other => vec![other],
      };
    }
    "PlotRangeClipping" => {
      plot_opts.clip_overlays =
        !matches!(replacement, Expr::Identifier(s) if s == "False");
    }
    "Prolog" => {
      // Same evaluate-now treatment as `Epilog` — see above.
      let val = evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| replacement.clone());
      plot_opts.prolog = match val {
        Expr::List(ref items) => items.to_vec(),
        Expr::Identifier(ref s) if s == "None" => Vec::new(),
        other => vec![other],
      };
    }
    _ => return false,
  }
  true
}

/// Build the compactifying substitution `Tan[Pi*inner/2]`, the bijection
/// used to fold an infinite plot range into a finite display coordinate.
fn tan_compactify(inner: Expr) -> Expr {
  call1("Tan", div2(times2(id_expr("Pi"), inner), Expr::Integer(2)))
}

/// Given the (possibly infinite) raw endpoints of a `Plot` range, return the
/// finite display range plus, when an endpoint is infinite, a substitution
/// `var -> g(var)` mapping the finite display coordinate back onto the real
/// line. This mirrors Wolfram, which plots e.g. `{x, -Infinity, Infinity}`
/// over the compact display range `{-1, 1}`.
///
/// - `{-Infinity, Infinity}` -> `[-1, 1]`,   `x = Tan[Pi u/2]`
/// - `{a, Infinity}`         -> `[a, a+1]`,  `x = a + Tan[Pi (u-a)/2]`
/// - `{-Infinity, b}`        -> `[b-1, b]`,  `x = b - Tan[Pi (b-u)/2]`
pub(crate) fn compactify_plot_range(
  var: &str,
  raw_min: f64,
  raw_max: f64,
) -> Result<(f64, f64, Option<Expr>), InterpreterError> {
  let min_inf = raw_min == f64::NEG_INFINITY;
  let max_inf = raw_max == f64::INFINITY;
  let u = || id_expr(var);
  match (min_inf, max_inf) {
    (false, false) => {
      if !raw_min.is_finite() || !raw_max.is_finite() {
        return Err(InterpreterError::EvaluationError(
          "Plot: range endpoints must be finite or ±Infinity".into(),
        ));
      }
      Ok((raw_min, raw_max, None))
    }
    // {-Infinity, Infinity}: x = Tan[Pi u/2] over u in [-1, 1]
    (true, true) => Ok((-1.0, 1.0, Some(tan_compactify(u())))),
    // {a, Infinity}: x = a + Tan[Pi (u-a)/2] over u in [a, a+1]
    (false, true) => {
      let a = raw_min;
      let inner = minus2(u(), Expr::Real(a));
      let transform = plus2(Expr::Real(a), tan_compactify(inner));
      Ok((a, a + 1.0, Some(transform)))
    }
    // {-Infinity, b}: x = b - Tan[Pi (b-u)/2] over u in [b-1, b]
    (true, false) => {
      let b = raw_max;
      let inner = minus2(Expr::Real(b), u());
      let transform = minus2(Expr::Real(b), tan_compactify(inner));
      Ok((b - 1.0, b, Some(transform)))
    }
  }
}

/// Returns true if `expr` textually references the identifier `var`. Used by
/// the plotting heads to decide whether a held body (which may be a bare
/// assigned symbol) still needs to be evaluated to surface the plot variable.
pub(crate) fn expr_mentions_var(expr: &Expr, var: &str) -> bool {
  // Substituting the variable with a sentinel changes the tree iff the
  // variable actually occurs; comparing structurally avoids hand-writing a
  // walker over every Expr variant.
  let sentinel = id_expr("$WoxiPlotVarProbe$");
  let replaced = crate::syntax::substitute_variable(expr, var, &sentinel);
  !crate::evaluator::pattern_matching::expr_equal(&replaced, expr)
}

/// Peel presentation wrappers (`Highlighted`, `Tooltip`, `Legended`,
/// `Labeled`) off a plot body, returning the inner plottable expression. These
/// heads annotate a curve — a highlight marker, a hover tooltip, a legend
/// entry, an inline label — but the function actually sampled is always their
/// first argument. The wrappers may enclose either a single function or a whole
/// list of them (e.g. `Highlighted[{Sin[x], Cos[x]}, …]`), so peeling happens
/// both before and after the body is split into individual curves. Callout is
/// intentionally left for the dedicated per-curve handling that records its
/// label.
fn peel_plot_wrappers(mut e: &Expr) -> &Expr {
  while let Expr::FunctionCall { name, args } = e {
    if !args.is_empty()
      && matches!(
        name.as_str(),
        "Highlighted" | "Tooltip" | "Legended" | "Labeled"
      )
    {
      e = &args[0];
    } else {
      break;
    }
  }
  e
}

/// Evaluate a plot body once with `var` kept symbolic, so assigned symbols
/// (e.g. `f = Sin[x]`) expand to their definitions without the plot variable
/// being replaced by any global value it might carry.
pub(crate) fn eval_body_var_symbolic(body: &Expr, var: &str) -> Expr {
  eval_body_vars_symbolic(body, &[var])
}

/// Expand a held generator such as `Table[f[i, x], {i, …}]` into the list of
/// curves it produces. The plot body is held, so such a generator only takes
/// list shape once evaluated; Wolfram plots one curve per element of the
/// result. Returns `None` when the body is already a list or does not
/// generate one, leaving plain `Plot[f[x], …]` untouched.
fn expand_generated_bodies(body: &Expr, var: &str) -> Option<Expr> {
  if matches!(body, Expr::List(_)) {
    return None;
  }
  match eval_body_var_symbolic(body, var) {
    evaluated @ Expr::List(_) => Some(evaluated),
    _ => None,
  }
}

/// A body that stays unevaluated while the plot variable is symbolic — a
/// user function wrapping `If[x == 1., 0., f[x, {a, b}]]` — can still yield a
/// list once the variable is a number. Wolfram then draws one curve per
/// element, so probe the body at `x` and split it into `Part[body, i]`
/// curves. Returns `None` for a body that samples to a plain number.
fn expand_list_valued_body(body: &Expr, var: &str, x: f64) -> Option<Expr> {
  if !expr_mentions_var(body, var) {
    return None;
  }
  let probe =
    evaluate_expr_to_expr(&substitute_var(body, var, &Expr::Real(x))).ok()?;
  match &probe {
    Expr::List(items) if !items.is_empty() => Some(Expr::List(
      (1..=items.len())
        .map(|i| Expr::FunctionCall {
          name: "Part".to_string(),
          args: vec![body.clone(), Expr::Integer(i as i128)].into(),
        })
        .collect(),
    )),
    _ => None,
  }
}

/// A body that applies replacement rules to the plot variable — `y[t] /.
/// sol` — has to be replaced while the variable is still symbolic: once
/// sampling puts a number in `t`, the rule `y[t] -> …` no longer matches
/// `y[0.3]`. Returns the replaced body when it still depends on `var`.
fn resolve_replaced_body(body: &Expr, var: &str) -> Option<Expr> {
  let is_replacement = match body {
    Expr::ReplaceAll { .. } => true,
    Expr::FunctionCall { name, args } => {
      name == "ReplaceAll" && args.len() == 2
    }
    _ => false,
  };
  if !is_replacement || !expr_mentions_var(body, var) {
    return None;
  }
  let evaluated = eval_body_var_symbolic(body, var);
  expr_mentions_var(&evaluated, var).then_some(evaluated)
}

/// As [`eval_body_var_symbolic`], for a body in several plot variables.
pub(crate) fn eval_body_vars_symbolic(body: &Expr, vars: &[&str]) -> Expr {
  let saved: Vec<(&str, Option<crate::StoredValue>)> = vars
    .iter()
    .map(|v| (*v, crate::ENV.with(|e| e.borrow_mut().remove(*v))))
    .collect();
  crate::evaluator::function_application::SYMBOLIC_PLOT_PROBE
    .with(|p| p.set(p.get() + 1));
  let result = evaluate_expr_to_expr(body).unwrap_or_else(|_| body.clone());
  crate::evaluator::function_application::SYMBOLIC_PLOT_PROBE
    .with(|p| p.set(p.get() - 1));
  for (var, value) in saved {
    if let Some(v) = value {
      crate::ENV.with(|e| {
        e.borrow_mut().insert(var.to_string(), v);
      });
    }
  }
  result
}

/// Resolve a plot body that names part of itself indirectly — `p = a x^2 +
/// b x; RegionPlot[y > p, …]`. The plotting heads hold their arguments, so
/// `p` arrives unevaluated and substituting the plot variables into it finds
/// nothing to replace; the body has to be evaluated once, with those
/// variables cleared, before it can be sampled.
///
/// Returns `None` — leaving the body untouched, so ordinary plots are
/// unaffected — unless evaluating it brings in a plot variable that was not
/// there before.
pub(crate) fn resolve_indirect_plot_body(
  body: &Expr,
  vars: &[&str],
) -> Option<Expr> {
  let mentioned =
    |e: &Expr| vars.iter().filter(|v| expr_mentions_var(e, v)).count();
  let before = mentioned(body);
  let evaluated = eval_body_vars_symbolic(body, vars);
  let after = mentioned(&evaluated);
  // A body that already names every plot variable can still hide a helper
  // symbol whose value depends on them (`Module[{y1, …}, y1 = x1/…; {x1 - y1,
  // …}]`); it counts as resolved once evaluation changes it without dropping
  // any plot variable.
  let changed_in_full = before == vars.len()
    && after == vars.len()
    && !crate::syntax::expr_to_string(&evaluated)
      .eq(&crate::syntax::expr_to_string(body));
  (after > before || changed_in_full).then_some(evaluated)
}

/// Flatten a single function or a (possibly nested) list of functions into
/// individual curve bodies. Wolfram flattens nested lists into individual
/// curves, so `Plot[{{f}, {g}}, …]` draws two curves. Nesting arises
/// naturally from idioms like `expr /. C[1] -> Range[…]`, where threading a
/// replacement over several expressions yields a list of lists.
pub(crate) fn flatten_plot_bodies<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
  match e {
    Expr::List(items) => {
      for it in items {
        flatten_plot_bodies(it, out);
      }
    }
    _ => out.push(e),
  }
}

/// Implementation of Plot[f, {x, xmin, xmax}]
/// The scaled counterpart of an explicit `PlotRange` side given in original
/// coordinates. An end outside the scaling's domain (`0` on a log axis)
/// comes back as NaN: that end stays automatic, as in the Wolfram Language.
pub(crate) fn scale_range(
  range: Option<(f64, f64)>,
  scale: Option<&Scaling>,
) -> Option<(f64, f64)> {
  let (Some((a, b)), Some(s)) = (range, scale) else {
    return range;
  };
  let increasing = match (s.forward(1.0), s.forward(2.0)) {
    (Some(u), Some(v)) => u < v,
    _ => true,
  };
  let (fa, fb) = (
    s.forward(a.min(b)).unwrap_or(f64::NAN),
    s.forward(a.max(b)).unwrap_or(f64::NAN),
  );
  Some(if increasing { (fa, fb) } else { (fb, fa) })
}

/// An explicit plot range side whose NaN ends (see [`scale_range`]) are
/// filled in from the automatic range `auto`.
pub(crate) fn fill_range(
  explicit: Option<(f64, f64)>,
  auto: (f64, f64),
) -> Option<(f64, f64)> {
  explicit.map(|(lo, hi)| {
    (
      if lo.is_nan() { auto.0 } else { lo },
      if hi.is_nan() { auto.1 } else { hi },
    )
  })
}

/// Map the positions an axis' options name — its `AxesOrigin` coordinate,
/// explicit `GridLines` and `Ticks` — through `f` (axis 0 is x, 1 is y),
/// dropping those `f` cannot map.
fn map_axis_options(
  opts: &mut PlotOptions,
  axis: usize,
  f: &dyn Fn(f64) -> Option<f64>,
) {
  let map_ticks = |t: &mut Option<Vec<(f64, String)>>| {
    if let Some(ticks) = t {
      *ticks = ticks
        .drain(..)
        .filter_map(|(p, l)| Some((f(p)?, l)))
        .collect();
    }
  };
  let map_grid = |g: &mut Vec<GridLine>| {
    g.retain_mut(|line| match f(line.pos) {
      Some(p) => {
        line.pos = p;
        true
      }
      None => false,
    });
  };
  // Explicit frame ticks of the edges along this axis (bottom/top for x,
  // left/right for y).
  let map_edge = |e: &mut FrameEdgeTicks| {
    if let FrameEdgeTicks::Explicit(ticks) = e {
      ticks.retain_mut(|t| match f(t.pos) {
        Some(p) => {
          t.pos = p;
          true
        }
        None => false,
      });
    }
  };
  if axis == 0 {
    opts.axes_origin.0 = opts.axes_origin.0.and_then(f);
    map_ticks(&mut opts.ticks_x);
    map_edge(&mut opts.frame_tick_edges[0]);
    map_edge(&mut opts.frame_tick_edges[2]);
    map_grid(&mut opts.grid_x_lines);
  } else {
    opts.axes_origin.1 = opts.axes_origin.1.and_then(f);
    map_ticks(&mut opts.ticks_y);
    map_edge(&mut opts.frame_tick_edges[1]);
    map_edge(&mut opts.frame_tick_edges[3]);
    map_grid(&mut opts.grid_y_lines);
  }
}

/// Put the options a plot was given in original coordinates into the
/// scaled coordinates of its `ScalingFunctions`.
pub(crate) fn scale_axis_options(
  opts: &mut PlotOptions,
  scaling: &(Option<Scaling>, Option<Scaling>),
) {
  if let Some(s) = &scaling.0 {
    map_axis_options(opts, 0, &|v| s.forward(v));
  }
  if let Some(s) = &scaling.1 {
    map_axis_options(opts, 1, &|v| s.forward(v));
  }
}

/// `sy[f[sx⁻¹[x]]]`: what a scaled plot samples for the curve `f`.
fn scaled_plot_body(
  body: &Expr,
  var: &str,
  (sx, sy): &(Option<Scaling>, Option<Scaling>),
) -> Expr {
  let mut e = body.clone();
  if let Some(sx) = sx {
    let original = Expr::CurriedCall {
      func: Box::new(sx.inverse_expr()),
      args: vec![Expr::Identifier(var.to_string())],
    };
    e = crate::syntax::substitute_variable(&e, var, &original);
  }
  if let Some(sy) = sy {
    e = Expr::CurriedCall {
      func: Box::new(sy.forward_expr()),
      args: vec![e],
    };
  }
  e
}

/// Finish axis `axis` (0 is x, 1 is y) of a plot laid out in the scaled
/// coordinates of `scale`, whose displayed range is `range`; returns the
/// range to draw. A log axis goes back to original coordinates and is
/// drawn as a native log axis, crossed by the other axis at its low end.
/// Any other scaled axis stays scaled and is labelled with
/// `Charting`ScaledTicks`; without a place for the original 0 it is
/// crossed at its low end too.
pub(crate) fn finish_scaled_axis(
  axis: usize,
  scale: Option<&Scaling>,
  opts: &mut PlotOptions,
  points: &mut [Vec<(f64, f64)>],
  range: (f64, f64),
  list_plot: bool,
) -> (f64, f64) {
  let Some(s) = scale else {
    return range;
  };
  // Without a place for the original 0 the other axis crosses at the low
  // end: where the unpadded range starts when that is at or above 0, at the
  // padded edge otherwise (a list plot's range starts inside its padding).
  let low = if axis == 0 {
    opts.axes_origin_low.0
  } else {
    opts.axes_origin_low.1
  };
  let crossing = match low {
    Some(l) if list_plot || l >= 0.0 => l,
    _ => range.0,
  };
  if s.is_log() {
    for p in points.iter_mut().flatten() {
      if axis == 0 {
        p.0 = p.0.exp();
      } else {
        p.1 = p.1.exp();
      }
    }
    map_axis_options(opts, axis, &|v| Some(v.exp()));
    if let Some((ux, uy)) = &mut opts.unpadded_range {
      let u = if axis == 0 { ux } else { uy };
      *u = (u.0.exp(), u.1.exp());
    }
    let crossing = Some(crossing.exp());
    if axis == 0 {
      for m in &mut opts.exclusion_marks {
        m.x = m.x.exp();
      }
    } else {
      opts.curve_clip_y = opts.curve_clip_y.map(|(a, b)| (a.exp(), b.exp()));
      for m in &mut opts.exclusion_marks {
        m.y_left = m.y_left.exp();
        m.y_right = m.y_right.exp();
      }
    }
    if axis == 0 {
      opts.log_x = true;
      opts.axes_origin_low.0 = crossing;
    } else {
      opts.log_y = true;
      opts.axes_origin_low.1 = crossing;
    }
    return (range.0.exp(), range.1.exp());
  }
  let (origin, explicit) = if axis == 0 {
    (&mut opts.axes_origin.0, opts.ticks_x.is_some())
  } else {
    (&mut opts.axes_origin.1, opts.ticks_y.is_some())
  };
  if origin.is_none() && s.zero().is_none() {
    *origin = Some(crossing);
  }
  if !explicit {
    let mut ticks = s.ticks(range.0, range.1);
    // The tick finder also places ticks past the ends of the range.
    let (lo, hi) = (range.0.min(range.1), range.0.max(range.1));
    let eps = (hi - lo) * 1e-9;
    let inside = |p: f64| p >= lo - eps && p <= hi + eps;
    ticks.majors.retain(|(p, _)| inside(*p));
    ticks.minors.retain(|p| inside(*p));
    if opts.frame {
      let majors = Some(ticks.majors.clone());
      if axis == 0 {
        opts.ticks_x = majors;
      } else {
        opts.ticks_y = majors;
      }
    }
    if axis == 0 {
      opts.scaled_ticks.0 = Some(ticks);
    } else {
      opts.scaled_ticks.1 = Some(ticks);
    }
  }
  range
}

pub fn plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Plot requires at least 2 arguments: Plot[f, {x, xmin, xmax}]".into(),
    ));
  }

  let iter_spec = &args[1];

  // Parse options (Rule expressions after the first two arguments).
  // Matching Wolfram Language, the first occurrence of a repeated option
  // wins (e.g. `PlotPoints -> 200, …, PlotPoints -> 10` uses 200).
  let mut plot_opts = PlotOptions::default();
  // PlotRange and AspectRatio are applied after parsing all options
  let mut overrides = PlotRangeOverrides::default();
  let mut legends_automatic = false;
  let mut legends_expressions = false;
  // `PlotLabels -> Automatic | "Expressions"`: each curve is labelled with
  // its expression.
  let mut labels_expressions = false;
  // `EvaluationMonitor :> expr` — evaluated (with the plot variable bound)
  // at every sampled point, e.g. to `Sow` the sample locations.
  let mut monitor: Option<Expr> = None;
  // `RegionFunction -> f`: only the stretches of the curve where
  // `f[x, y]` is True are drawn.
  let mut region_function: Option<Expr> = None;
  // `ScalingFunctions -> {sx, sy}`: the plot is drawn in the scaled
  // coordinates of each scaled axis.
  let mut scaling: (Option<Scaling>, Option<Scaling>) = (None, None);
  let mut seen: std::collections::HashSet<String> =
    std::collections::HashSet::new();
  let (options, simple_ticks) = expand_plot_theme(&args[2..], false);
  plot_opts.simple_ticks = simple_ticks;
  // A function plot samples just inside its domain: a mesh value taken at
  // an endpoint is not marked.
  plot_opts.mesh_open_ends = true;
  for opt in &options {
    if let Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "EvaluationMonitor")
    {
      if monitor.is_none() {
        monitor = Some(replacement.as_ref().clone());
      }
      continue;
    }
    if let Some((name, replacement)) =
      crate::functions::graphics::option_name_value(opt)
    {
      let replacement = &*replacement;
      if !seen.insert(name.to_string()) {
        continue;
      }
      // The shared reader takes explicit legend labels; `Automatic` and
      // `"Expressions"` are filled in from the curves here.
      if name == "PlotLegends" {
        let (_, auto, expressions, _) = parse_plot_legends(replacement);
        legends_automatic |= auto;
        legends_expressions |= expressions;
      }
      if name == "PlotLabels" {
        labels_expressions = plot_labels_from_expressions(replacement);
      }
      if name == "ScalingFunctions" {
        let value = evaluate_expr_to_expr(replacement)
          .unwrap_or_else(|_| replacement.clone());
        scaling = Scaling::parse_option(&value).unwrap_or((None, None));
        continue;
      }
      if name == "RegionFunction" {
        // `RegionFunction -> (True &)` is the default: every point drawn.
        let always = matches!(replacement,
          Expr::Function { body } if matches!(body.as_ref(), Expr::Identifier(t) if t == "True"));
        if !always {
          // Evaluated once, so `Function[{x, y}, Evaluate[D[f, x] > 0]]`
          // is called with its derivative already worked out.
          region_function = Some(
            evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.clone()),
          );
        }
        continue;
      }
      apply_common_plot_option(
        name,
        replacement,
        &mut plot_opts,
        &mut overrides,
      );
    }
  }
  let (plot_range_x, plot_range_y) = (overrides.x, overrides.y);
  let plot_range_y_all = overrides.y_all;

  // Apply AspectRatio to the plotting area (not the whole image); the total
  // height is derived in generate_svg_with_options once margins are known.
  // (AspectRatio is height/width in Wolfram Language.)
  if let Some(ar) = overrides.aspect_ratio {
    plot_opts.aspect_ratio = Some(ar);
  }
  plot_opts.aspect_automatic = overrides.aspect_automatic;

  // Parse iterator spec: {x, xmin, xmax}. Endpoints may be ±Infinity, in
  // which case the range is compactified onto a finite display coordinate
  // and `x_transform` folds that coordinate back onto the real line.
  let (var_name, x_min, x_max, x_transform) = match iter_spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Plot: iterator variable must be a symbol".into(),
          ));
        }
      };
      // Evaluate xmin and xmax (accepting ±Infinity)
      let x_min_expr = evaluate_expr_to_expr(&items[1])?;
      let x_max_expr = evaluate_expr_to_expr(&items[2])?;
      let raw_min =
        crate::functions::math_ast::try_eval_to_f64_with_infinity(&x_min_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Plot: cannot evaluate xmin to a number".into(),
            )
          })?;
      let raw_max =
        crate::functions::math_ast::try_eval_to_f64_with_infinity(&x_max_expr)
          .ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Plot: cannot evaluate xmax to a number".into(),
            )
          })?;
      let (x_min, x_max, transform) =
        compactify_plot_range(&var, raw_min, raw_max)?;
      (var, x_min, x_max, transform)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Plot: second argument must be {x, xmin, xmax}".into(),
      ));
    }
  };

  // A scaled x axis is sampled evenly in its scaled coordinate.
  if x_transform.is_some() {
    scaling.0 = None;
  }
  let (x_min, x_max) = match &scaling.0 {
    Some(sx) => match (sx.forward(x_min), sx.forward(x_max)) {
      (Some(a), Some(b)) => (a.min(b), a.max(b)),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Plot: the plot range is outside the domain of the x scaling".into(),
        ));
      }
    },
    None => (x_min, x_max),
  };
  scale_axis_options(&mut plot_opts, &scaling);
  let (plot_range_x, plot_range_y) = (
    scale_range(plot_range_x, scaling.0.as_ref()),
    scale_range(plot_range_y, scaling.1.as_ref()),
  );

  // When the range was infinite, substitute the compactifying map into the
  // function body so sampling the finite display coordinate covers the whole
  // real line. `body_storage` owns the rewritten body for the borrow below.
  let body_storage;
  let body: &Expr = if let Some(t) = &x_transform {
    body_storage = crate::syntax::substitute_variable(&args[0], &var_name, t);
    &body_storage
  } else {
    &args[0]
  };

  // Plot is HoldAll, so a body passed as a bare symbol (e.g.
  // `f = Sin[x]; Plot[f, {x, 0, 10}]`) reaches here unevaluated. Sampling
  // substitutes the plot variable into the body and then evaluates, but a
  // held `f` contains no `x` to substitute, so the sampler would evaluate the
  // symbol `f` back to `Sin[x]` with `x` still symbolic — never a number —
  // yielding an all-white plot. When the body does not textually reference
  // the plot variable, evaluate it once with that variable kept symbolic so
  // assigned symbols expand to their definitions (and reveal the variable)
  // before sampling. Bodies that already mention the variable are left
  // untouched, so ordinary plots are unaffected.
  // `Evaluated -> True` asks for the body to be worked out once, with the
  // plot variable still symbolic, instead of at every sample point. It is
  // how a body that only makes sense symbolically gets plotted at all —
  // `Plot[Normal[NonlinearModelFit[data, …, y]], {y, 0, 1}]` would
  // otherwise re-fit with a number in place of `y` at every point.
  let evaluated_storage;
  let body: &Expr = if plot_opts.evaluated {
    evaluated_storage = eval_body_var_symbolic(body, &var_name);
    &evaluated_storage
  } else {
    body
  };

  let resolved_storage;
  let body: &Expr = if expr_mentions_var(body, &var_name) {
    body
  } else {
    let evaluated = eval_body_var_symbolic(body, &var_name);
    if expr_mentions_var(&evaluated, &var_name) {
      resolved_storage = evaluated;
      &resolved_storage
    } else {
      body
    }
  };

  // Peel a presentation wrapper that encloses the whole plot spec so that a
  // list inside it (e.g. `Highlighted[{Sin[x], Cos[x]}, …]`) is split into
  // individual curves below rather than sampled as one non-numeric expression.
  let body = peel_plot_wrappers(body);

  // `Plot[Table[f[i, x], {i, …}], …]` names its curves only after the
  // generator runs; expand it so each element becomes its own curve
  // instead of being sampled as one expression yielding a list. This has
  // to recurse into an explicit `{f, g, …}` list too — `NDSolve` returns a
  // list of solution sets even when there is only one, so an element like
  // `y[t] /. sol` (or `Evaluate[y[t] /. sol]`, the same thing one Hold
  // level up) evaluates to a one-element list, not a bare number, exactly
  // like the bare `Table[…]` case below. Resolving it here — before the
  // list is split into per-curve bodies — turns that one element into its
  // own singleton list, which the flattening step below unwraps same as
  // any other nested list; left unresolved, every sample point comes back
  // as an unconvertible list and the curve silently plots as nothing.
  fn resolve_generated_bodies_deep(
    body: &Expr,
    var: &str,
    probe_x: f64,
  ) -> Expr {
    match body {
      Expr::List(items) => Expr::List(
        items
          .iter()
          .map(|it| resolve_generated_bodies_deep(it, var, probe_x))
          .collect(),
      ),
      _ => expand_generated_bodies(body, var)
        .or_else(|| resolve_replaced_body(body, var))
        .or_else(|| expand_list_valued_body(body, var, probe_x))
        .unwrap_or_else(|| body.clone()),
    }
  }
  let generated_storage = resolve_generated_bodies_deep(
    body,
    &var_name,
    x_min + 0.5 * (x_max - x_min),
  );
  let body: &Expr = &generated_storage;

  // Collect function bodies: a single function or a (possibly nested) list of
  // functions. Wolfram flattens nested lists into individual curves, so
  // `Plot[{{f}, {g}}, …]` draws two curves. Nesting arises naturally from
  // idioms like `expr /. C[1] -> Range[…]`, where threading a replacement over
  // several expressions yields a list of lists.
  let mut raw_bodies: Vec<&Expr> = Vec::new();
  flatten_plot_bodies(body, &mut raw_bodies);

  // Unwrap Callout[expr, label] wrappers, storing labels. Other presentation
  // wrappers (Highlighted/Tooltip/Legended/Labeled) around a single curve are
  // peeled first so their inner function is what gets sampled.
  let mut bodies: Vec<&Expr> = Vec::with_capacity(raw_bodies.len());
  let mut end_labels: Vec<Option<String>> = Vec::new();
  for b in &raw_bodies {
    let b = peel_plot_wrappers(b);
    if let Expr::FunctionCall { name, args: cargs } = b
      && name == "Callout"
      && cargs.len() >= 2
    {
      bodies.push(&cargs[0]);
      let label = crate::functions::graphics::traditional_label(&cargs[1]);
      // At its automatic place a curve's callout hangs off the curve's end.
      if callout_at_series_end(cargs) {
        end_labels.push(Some(label));
        plot_opts.callout_labels.push(None);
      } else {
        end_labels.push(None);
        plot_opts.callout_labels.push(Some(label));
      }
    } else {
      bodies.push(b);
      end_labels.push(None);
      plot_opts.callout_labels.push(None);
    }
  }
  if plot_opts.plot_labels.is_empty() && end_labels.iter().any(Option::is_some)
  {
    plot_opts.plot_label_leaders =
      end_labels.iter().map(Option::is_some).collect();
    plot_opts.plot_labels = end_labels;
  }

  // Fill automatic legends from expression strings; `"Expressions"` sets
  // them in TraditionalForm, as WL does.
  if (legends_automatic || legends_expressions)
    && plot_opts.plot_legends.is_empty()
  {
    for b in &bodies {
      plot_opts.plot_legends.push(if legends_expressions {
        crate::functions::graphics::traditional_label(b)
      } else {
        expr_to_output(b)
      });
    }
  }
  if labels_expressions && plot_opts.plot_labels.is_empty() {
    plot_opts.plot_labels = bodies
      .iter()
      .map(|b| Some(crate::functions::graphics::traditional_label(b)))
      .collect();
    plot_opts.plot_label_leaders = vec![true; bodies.len()];
  }

  // The scaled plot is the plot of `sy[f[sx⁻¹[x]]]`.
  let scaled_bodies: Vec<Expr>;
  let bodies: Vec<&Expr> = if scaling.0.is_some() || scaling.1.is_some() {
    scaled_bodies = bodies
      .iter()
      .map(|b| scaled_plot_body(b, &var_name, &scaling))
      .collect();
    scaled_bodies.iter().collect()
  } else {
    bodies
  };

  // Adaptive sampling: start with initial points, then refine where needed
  // WL starts from `PlotPoints -> 50` when none is given.
  // (Only the option name is read: evaluating the values here would fire
  // an `EvaluationMonitor :> Sow[…]` with the variable still symbolic.)
  let plot_points_given = args[2..].iter().any(|o| {
    matches!(
      o,
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotPoints")
    )
  });
  let initial_samples = if plot_points_given {
    plot_opts.plot_points.clamp(2, 1000)
  } else {
    50
  };
  // Each recursion level may at most double the samples; WL's default
  // (50 points, MaxRecursion 6) ends with up to ~2500 on a rapidly
  // oscillating curve such as `Sin[30 Sin[x]]`.
  let max_depth = plot_opts.max_recursion;
  let max_total = initial_samples << max_depth.min(6);
  let mut all_points: Vec<Vec<(f64, f64)>> = Vec::with_capacity(bodies.len());
  let excluded = crate::functions::parametric_plot::exclusion_roots(
    &args[2..],
    &var_name,
    x_min,
    x_max,
  );
  let auto_exclusions = automatic_exclusions_enabled(&args[2..]);

  for func_body in &bodies {
    // `Mesh -> Full` marks the initial sampling grid, before refinement —
    // all but its first point, as in WL 15.
    if matches!(plot_opts.mesh, Mesh::Full) {
      let step = (x_max - x_min) / (initial_samples - 1) as f64;
      plot_opts.mesh_full_points.push(
        (1..initial_samples)
          .map(|i| {
            let x = if i + 1 == initial_samples {
              x_max
            } else {
              x_min + i as f64 * step
            };
            (
              x,
              evaluate_at_point(func_body, &var_name, x).unwrap_or(f64::NAN),
            )
          })
          .collect(),
      );
    }
    let points = adaptive_sample(
      func_body,
      &var_name,
      x_min,
      x_max,
      initial_samples,
      max_total,
      max_depth,
      monitor.as_ref(),
    );
    // `Exclusions -> Automatic` breaks the curve where a step function in
    // it jumps (`Floor[x]` draws separate flat segments, not a staircase).
    // So does a branch cut a complex-valued part of it is taken across.
    let mut excluded_here = excluded.clone();
    if auto_exclusions
      && (has_step_function(func_body) || has_branch_cut(func_body))
    {
      excluded_here.extend(jump_points(&points, func_body, &var_name));
    }
    if plot_opts.exclusions_style.iter().any(Option::is_some) {
      let eps = (x_max - x_min) * 1e-9;
      let at =
        |x: f64| evaluate_at_point(func_body, &var_name, x).unwrap_or(f64::NAN);
      // A break at an end of the domain only drops the value there.
      plot_opts.exclusion_marks.extend(
        excluded_here
          .iter()
          .filter(|&&x| x > x_min + eps && x < x_max - eps)
          .map(|&x| ExclusionMark {
            x,
            y_left: at(x - eps),
            y_right: at(x + eps),
          }),
      );
    }
    let points = break_at_exclusions(
      points,
      func_body,
      &var_name,
      &excluded_here,
      x_max - x_min,
    );
    all_points.push(match &region_function {
      Some(region) => {
        clip_to_region_function(&points, func_body, &var_name, region)
      }
      None => points,
    });
  }

  // A `PlotStyle` list applied to a single curve is one combined directive
  // set, not a per-curve cycle: `Plot[f, …, PlotStyle -> {Thick, Green}]`
  // draws one thick green curve, not a `Thick` curve and a `Green` one.
  if all_points.len() == 1 {
    plot_opts.plot_style =
      collapse_style_for_single_series(&plot_opts.plot_style);
  }

  // The automatic y range clips outliers off the drawn curves — unless
  // `PlotRange -> All` asked for every sampled value to be shown.
  let (y_data_min, y_data_max) = drawn_y_range(&all_points, plot_range_y_all);

  // Check if we have any plottable data
  let has_finite = all_points
    .iter()
    .any(|pts| pts.iter().any(|(_, y)| y.is_finite()));
  // Nothing plottable (every function is `Null` — a Manipulate whose
  // checkboxes switch all curves off — or non-numeric): Wolfram still draws
  // the empty plot, with its axes and `PlotLabel`, over the requested
  // `PlotRange` (unit height when none is given).
  if !has_finite {
    let x_display = plot_range_x.unwrap_or((x_min, x_max));
    let y_display = plot_range_y.unwrap_or((0.0, 1.0));
    let svg = generate_svg_with_filling(&[], x_display, y_display, &plot_opts)?;
    // Keep the (empty) plot's source, so `Show` can tell it from an opaque
    // picture and merge it as the layer without curves that it is.
    let source = build_plot_source(
      &[],
      &plot_opts.plot_style,
      x_display,
      y_display,
      (plot_opts.svg_width, plot_opts.svg_height),
      &[false],
      plot_opts.filling,
      plot_opts.filling_style,
      crate::functions::plot::explicit_options(args),
    );
    return Ok(crate::graphics_result_with_source(svg, source));
  }

  // A numeric `AxesOrigin` y is part of the automatic plot range.
  let (y_data_min, y_data_max) =
    include_in_range((y_data_min, y_data_max), plot_opts.axes_origin.1);

  // WL 15 pads an automatic range by `PlotRangePadding -> {{Scaled[0.02],
  // …}, {Scaled[0.05], …}}`: that fraction of the padded span, i.e. 1/48
  // of the x span and 1/18 of the y span on each side. An explicit
  // `PlotRange` is not padded.
  let y_span = y_data_max - y_data_min;
  let padding = if y_span.abs() < f64::EPSILON {
    1.0
  } else {
    y_span / 18.0
  };
  let y_auto_min = y_data_min - padding;
  let y_auto_max = y_data_max + padding;
  let (y_auto_min, y_auto_max) =
    adjust_y_range_for_filling_opts(&plot_opts, (y_auto_min, y_auto_max));
  let x_padding = (x_max - x_min) / 48.0;

  // Apply PlotRange overrides (PlotRange -> {ymin, ymax} or {{xmin,xmax},{ymin,ymax}})
  let plot_range_x =
    fill_range(plot_range_x, (x_min - x_padding, x_max + x_padding));
  let plot_range_y = fill_range(plot_range_y, (y_auto_min, y_auto_max));
  let (x_display_min, x_display_max) =
    plot_range_x.unwrap_or((x_min - x_padding, x_max + x_padding));
  let (y_display_min, y_display_max) =
    plot_range_y.unwrap_or((y_auto_min, y_auto_max));
  // The axes of an all-positive curve touch its lowest point, inside the
  // padding, as in Wolfram.
  if plot_range_x.is_none() {
    plot_opts.axes_origin_low.0 = Some(x_min);
  }
  if plot_range_y.is_none() {
    plot_opts.axes_origin_low.1 = Some(y_data_min);
  }
  plot_opts.unpadded_range = Some((
    plot_range_x.unwrap_or((x_min, x_max)),
    plot_range_y.unwrap_or((y_data_min, y_data_max)),
  ));
  plot_opts.curve_clip_y = Some(plot_range_y.unwrap_or_else(|| {
    include_filling_levels(&plot_opts, (y_data_min, y_data_max))
  }));
  // The poles `Exclusions -> Automatic` leaves out: the curve jumps from
  // far beyond one edge of the plot range to far beyond the other.
  if auto_exclusions && plot_opts.exclusions_style.iter().any(Option::is_some) {
    let (lo, hi) = plot_opts.curve_clip_y.unwrap_or((y_data_min, y_data_max));
    for curve in &all_points {
      for pair in curve.windows(2) {
        let ((px, py), (x, y)) = (pair[0], pair[1]);
        if px.is_finite() && x.is_finite() && is_pole_jump(py, y, lo, hi) {
          plot_opts.exclusion_marks.push(ExclusionMark {
            x: f64::midpoint(px, x),
            y_left: py,
            y_right: y,
          });
        }
      }
    }
  }
  // Back from the scaled coordinates: a log axis is drawn by the native log
  // axes, any other scaled axis keeps them and gets scaled ticks.
  let (x_display_min, x_display_max) = finish_scaled_axis(
    0,
    scaling.0.as_ref(),
    &mut plot_opts,
    &mut all_points,
    (x_display_min, x_display_max),
    false,
  );
  let (y_display_min, y_display_max) = finish_scaled_axis(
    1,
    scaling.1.as_ref(),
    &mut plot_opts,
    &mut all_points,
    (y_display_min, y_display_max),
    false,
  );

  // `PlotLayout -> "Row" | "Column" | …`: one panel per curve, all over
  // the same plot range.
  if let Some(layout) = plot_opts.plot_layout
    && all_points.len() > 1
  {
    let svg = crate::functions::plot_layout::render_plot_layout(
      layout,
      all_points.len(),
      &plot_opts,
      (x_display_min, x_display_max),
      (y_display_min, y_display_max),
      |i, mut opts| {
        opts.callout_labels =
          vec![plot_opts.callout_labels.get(i).cloned().flatten()];
        generate_svg_with_filling(
          std::slice::from_ref(&all_points[i]),
          (x_display_min, x_display_max),
          (y_display_min, y_display_max),
          &opts,
        )
      },
    )?;
    return Ok(crate::graphics_result(svg));
  }

  // Generate SVG
  let svg = generate_svg_with_filling(
    &all_points,
    (x_display_min, x_display_max),
    (y_display_min, y_display_max),
    &plot_opts,
  )?;

  // Generate GraphicsBox expression for .nb export
  let mut box_elements = Vec::new();
  for (i, points) in all_points.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS_RGB[i % PLOT_COLORS_RGB.len()];
    box_elements.push(format!("RGBColor[{r}, {g}, {b}]"));
    box_elements.push("AbsoluteThickness[2]".to_string());
    box_elements.push("Opacity[1.]".to_string());
    let segments = split_into_segments(points);
    box_elements.extend(crate::functions::graphicsbox::line_box(&segments));
  }
  let graphicsbox = crate::functions::graphicsbox::graphics_box(&box_elements);
  crate::capture_graphicsbox(&graphicsbox);

  // Build source data for Show merging
  let mut source = build_plot_source(
    &all_points,
    &plot_opts.plot_style,
    (x_display_min, x_display_max),
    (y_display_min, y_display_max),
    (plot_opts.svg_width, plot_opts.svg_height),
    &[false],
    plot_opts.filling,
    plot_opts.filling_style,
    crate::functions::plot::explicit_options(args),
  );
  source.plot_range = plot_opts.unpadded_range;
  source.log_x = plot_opts.log_x;
  source.log_y = plot_opts.log_y;
  // A redraw marks the `Mesh -> Full` points the plot sampled first.
  for (s, grid) in source.series.iter_mut().zip(&plot_opts.mesh_full_points) {
    s.mesh_full_points = grid.clone();
  }
  // `build_plot_source` only knows the single global `Filling` mode; a
  // per-series `Filling -> {i -> spec, …}` rule list is resolved here
  // instead, so `source.series[i].filling` (which `Part`/`First` reads to
  // reconstitute the plot's primitives, and which `Show` reads to re-render
  // it) reflects each series' own target rather than staying unfilled.
  // A curve-to-curve target (`{i -> {j}}`) is kept as `SeriesFilling::Series`.
  if !plot_opts.filling_rules.is_empty() {
    for (i, s) in source.series.iter_mut().enumerate() {
      let target = series_fill_target(&plot_opts, i);
      s.filling = match target {
        FillTarget::Level(level) => level.to_series_filling(),
        FillTarget::Series(j) => crate::syntax::SeriesFilling::Series(j),
      };
      if let Some(style) = series_filling_style(&plot_opts, i) {
        s.fill_color = style.color;
        s.fill_opacity = style.effective_opacity();
      }
    }
  }

  // Return -Graphics- as the text representation
  Ok(crate::graphics_result_with_source(svg, source))
}

/// LogLogPlot[f, {x, xmin, xmax}] — plot f with log-scaled x and y axes.
pub fn log_log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, true, true)
}

/// LogPlot[f, {x, xmin, xmax}] — plot f with log-scaled y axis.
pub fn log_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, false, true)
}

/// LogLinearPlot[f, {x, xmin, xmax}] — plot f with log-scaled x axis.
pub fn log_linear_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  log_scale_plot_ast(args, true, false)
}

/// LogLogPlot, LogPlot and LogLinearPlot are `Plot` with `ScalingFunctions`
/// set to `"Log"` on their log axes, as in the Wolfram Language. The
/// scaling goes right after the iterator, where it takes precedence over
/// any later `ScalingFunctions` option.
fn log_scale_plot_ast(
  args: &[Expr],
  log_x: bool,
  log_y: bool,
) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "Plot requires at least 2 arguments".into(),
    ));
  }
  let axis = |log: bool| {
    if log {
      Expr::String("Log".to_string())
    } else {
      Expr::Identifier("None".to_string())
    }
  };
  let mut scaled = args[..2].to_vec();
  scaled.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("ScalingFunctions".to_string())),
    replacement: Box::new(Expr::List(vec![axis(log_x), axis(log_y)].into())),
  });
  scaled.extend_from_slice(&args[2..]);
  plot_ast(&scaled)
}
