//! `AngularGauge`, `ClockGauge`, `HorizontalGauge`, `VerticalGauge` and
//! `ThermometerGauge`: the `Graphics` wolframscript draws for each — a gray
//! bezel or scale line, major and minor tick marks with `FindDivisions`
//! labels, and a red needle, pointer or mercury column at the value.
//!
//! The pieces are the ones WL's gauges are made of (tick marks as
//! `Scaled` offsets from the scale, labels as `Offset` texts, the needle as
//! a transformed polygon), with WL's internal `Style` nesting collapsed to
//! the directives that take effect.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::{Expr, expr_to_source_form};

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn eval(e: Expr) -> Expr {
  evaluate_expr_to_expr(&e).unwrap_or(e)
}

/// A machine number as WL source (tiny round-off is dropped to `0.`).
fn num(x: f64) -> String {
  let x = if x.abs() < 1e-14 { 0.0 } else { x };
  let s = format!("{x:?}");
  if s.contains('e') {
    format!("{x:.17}")
  } else {
    s
  }
}

fn pt(x: f64, y: f64) -> String {
  format!("{{{}, {}}}", num(x), num(y))
}

const GRAY_25: &str =
  "RGBColor[0.25005721986295587, 0.25005721986295587, 0.25005721986295587]";
const LINEAR_FONT: &str = "FontSize -> 10, FontFamily -> \"Helvetica\"";
const DIAL_FONT: &str = "FontSize -> Scaled[0.07], FontFamily -> \"Helvetica\"";

/// The color of the `i`-th marker (0-based): hues `9/10 φ⁻¹` apart
/// (modulo 0.9) from `0.9/60`, at saturation and brightness 0.73.
pub(crate) fn series_color(i: usize) -> Expr {
  let phi = f64::midpoint(1.0, 5f64.sqrt());
  let step = 0.9 * (phi - 1.0);
  let mut h = 0.9 / 60.0;
  for _ in 0..i {
    h += step;
    if h >= 0.9 {
      h -= 0.9;
    }
  }
  let (s, v) = (0.73_f64, 0.73_f64);
  let x = h * 6.0;
  let k = x.floor();
  let f = x - k;
  let p = v * (1.0 - s);
  let q = v * (1.0 - s * f);
  let t = v * (1.0 - s * (1.0 - f));
  let (r, g, b) = match k as i64 % 6 {
    0 => (v, t, p),
    1 => (q, v, p),
    2 => (p, v, t),
    3 => (p, q, v),
    4 => (t, p, v),
    _ => (v, p, q),
  };
  call(
    "RGBColor",
    vec![Expr::Real(r), Expr::Real(g), Expr::Real(b)],
  )
}

/// Decimal places of a step such as `2.5` or `0.25` (at most 6).
fn decimals(step: f64) -> usize {
  (0..=6)
    .find(|&d| {
      let scaled = step * 10f64.powi(d as i32);
      (scaled - scaled.round()).abs() < 1e-9
    })
    .unwrap_or(6)
}

/// A gauge's scale over `{lo, hi}`: the major ticks with their labels and
/// the minor ticks, each as a fraction `0..1` of the range.
struct Scale {
  major: Vec<(f64, String)>,
  minor: Vec<f64>,
}

fn find_divisions(lo: &Expr, hi: &Expr, n: i128) -> Vec<Expr> {
  match &eval(call(
    "FindDivisions",
    vec![
      Expr::List(vec![lo.clone(), hi.clone()].into()),
      Expr::Integer(n),
    ],
  )) {
    Expr::List(items) => items.to_vec(),
    _ => Vec::new(),
  }
}

/// The major ticks are `FindDivisions[{lo, hi}, 5]` within the range; the
/// minor ones the finer `FindDivisions` with four times as many steps,
/// less the majors. Labels are integers when every major tick is one, and
/// otherwise numbers to the step's decimals (`0` stays exact).
fn scale(lo: &Expr, hi: &Expr, divisions: Option<(i128, i128)>) -> Scale {
  let (lo_f, hi_f) = (
    try_eval_to_f64(lo).unwrap_or(0.0),
    try_eval_to_f64(hi).unwrap_or(1.0),
  );
  let span = hi_f - lo_f;
  let inside =
    |v: f64| v >= lo_f - 1e-9 * span.abs() && v <= hi_f + 1e-9 * span.abs();
  let (major_n, minor_per) = divisions.unwrap_or((5, 4));
  let all_major = find_divisions(lo, hi, major_n);
  let major: Vec<Expr> = all_major
    .iter()
    .filter(|t| try_eval_to_f64(t).is_some_and(inside))
    .cloned()
    .collect();
  let major_f: Vec<f64> = major.iter().filter_map(try_eval_to_f64).collect();
  let gaps = all_major.len().saturating_sub(1).max(1) as i128;
  let minor: Vec<f64> = find_divisions(lo, hi, gaps * minor_per)
    .iter()
    .filter_map(try_eval_to_f64)
    .filter(|&v| inside(v))
    .filter(|v| major_f.iter().all(|m| (m - v).abs() > 1e-9 * span.abs()))
    .map(|v| (v - lo_f) / span)
    .collect();
  let integral = major.iter().all(|t| matches!(t, Expr::Integer(_)));
  let step = major_f
    .windows(2)
    .map(|w| w[1] - w[0])
    .next()
    .unwrap_or(1.0);
  let d = decimals(step).max(1);
  let major = major
    .iter()
    .zip(&major_f)
    .map(|(t, &v)| {
      let label = if integral || v == 0.0 {
        expr_to_source_form(t)
      } else {
        format!("NumberForm[{}, {{Infinity, {d}}}]", num(v))
      };
      ((v - lo_f) / span, label)
    })
    .collect();
  Scale { major, minor }
}

/// The values of a gauge's first argument: one or a list.
fn values_of(e: &Expr) -> Vec<Option<f64>> {
  match e {
    Expr::List(items) => items.iter().map(try_eval_to_f64).collect(),
    other => vec![try_eval_to_f64(other)],
  }
}

/// The `{min, max}` argument (or `{0, 1}`) as given.
fn range_of(args: &[Expr]) -> Option<(Expr, Expr)> {
  match args.get(1) {
    None | Some(Expr::Rule { .. } | Expr::RuleDelayed { .. }) => {
      Some((Expr::Integer(0), Expr::Integer(1)))
    }
    Some(Expr::List(items))
      if items.len() == 2
        && try_eval_to_f64(&items[0]).is_some()
        && try_eval_to_f64(&items[1]).is_some() =>
    {
      Some((items[0].clone(), items[1].clone()))
    }
    _ => None,
  }
}

/// The value of option `key`, evaluated.
fn option(args: &[Expr], key: &str) -> Option<Expr> {
  args.iter().skip(1).find_map(|a| match a {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } if matches!(&**pattern, Expr::Identifier(n) if n == key) => {
      Some(eval((**replacement).clone()))
    }
    _ => None,
  })
}

/// `ScaleDivisions -> n` / `{major, minor}`; `None` (the option value)
/// draws no ticks at all.
fn scale_divisions(args: &[Expr]) -> Result<Option<(i128, i128)>, ()> {
  match &option(args, "ScaleDivisions") {
    Some(Expr::Integer(n)) if *n > 0 => Ok(Some((*n, 4))),
    Some(Expr::List(items)) => match items.as_slice() {
      [Expr::Integer(a), Expr::Integer(b)] if *a > 0 && *b > 0 => {
        Ok(Some((*a, *b)))
      }
      _ => Ok(None),
    },
    Some(Expr::Identifier(n)) if n == "None" => Err(()),
    _ => Ok(None),
  }
}

/// The colors of the markers: `GaugeStyle` (one or a list), else the
/// default series colors.
fn marker_colors(args: &[Expr], n: usize) -> Vec<String> {
  let styles: Vec<Expr> = match &option(args, "GaugeStyle") {
    Some(Expr::List(items)) => items.to_vec(),
    Some(Expr::Identifier(a)) if a == "Automatic" => Vec::new(),
    Some(e) => vec![e.clone()],
    None => Vec::new(),
  };
  (0..n)
    .map(|i| {
      styles
        .get(i % styles.len().max(1))
        .cloned()
        .unwrap_or_else(|| series_color(i))
    })
    .map(|c| expr_to_source_form(&c))
    .collect()
}

/// The user's options, which override the gauge's own.
fn user_options(args: &[Expr]) -> Vec<String> {
  const GAUGE_ONLY: [&str; 8] = [
    "GaugeStyle",
    "GaugeMarkers",
    "GaugeLabels",
    "GaugeFaceStyle",
    "GaugeFrameStyle",
    "ScaleDivisions",
    "ScaleRanges",
    "ScaleOrigin",
  ];
  args
    .iter()
    .skip(1)
    .filter(|a| match a {
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
        !matches!(&**pattern, Expr::Identifier(n) if GAUGE_ONLY.contains(&n.as_str()))
      }
      _ => false,
    })
    .map(expr_to_source_form)
    .collect()
}

fn graphics(
  primitives: &str,
  args: &[Expr],
  clipping: bool,
  label_style: &str,
) -> Result<Expr, InterpreterError> {
  let mut opts = user_options(args);
  opts.extend([
    "AspectRatio -> Automatic".to_string(),
    "Axes -> False".to_string(),
    "Frame -> False".to_string(),
    "ImageMargins -> 0.".to_string(),
    "ImagePadding -> All".to_string(),
    "ImageSize -> Small".to_string(),
    format!("LabelStyle -> {label_style}"),
    "PlotRange -> All".to_string(),
    format!(
      "PlotRangeClipping -> {}",
      if clipping { "True" } else { "False" }
    ),
    "PlotRangePadding -> Automatic".to_string(),
  ]);
  let source = format!("Graphics[{{{primitives}}}, {{{}}}]", opts.join(", "));
  let expr = crate::syntax::string_to_expr(&source)?;
  evaluate_expr_to_expr(&expr)
}

/// `n + 1` points around the circle of radius `r` from angle `start`
/// counterclockwise, closing where they began.
fn circle_points(r: f64, start: f64, n: usize) -> String {
  let pts: Vec<String> = (0..=n)
    .map(|k| {
      let a = start + std::f64::consts::TAU * k as f64 / n as f64;
      pt(r * a.cos(), r * a.sin())
    })
    .collect();
  format!("{{{}}}", pts.join(", "))
}

/// The dial's white face and gray bezel ring, starting at angle `start`.
fn dial_face(start: f64) -> String {
  let inner = circle_points(1.0, start, 100);
  let outer = circle_points(1.07, start, 102);
  format!(
    "{{Directive[EdgeForm[None], LightDarkSwitched[GrayLevel[1], GrayLevel[1/3]]], FilledCurve[{{Line[{inner}]}}]}}, \
     {{Directive[EdgeForm[{{Thickness[0.01], GrayLevel[0.6]}}], GrayLevel[0.8]], FilledCurve[{{{{Line[{inner}]}}, {{Line[{outer}]}}}}]}}"
  )
}

/// Tick marks pointing inwards from radius `r` at the angles given, each
/// `len` of the plot range long (`ky` stretching the vertical share).
fn dial_ticks(angles: &[f64], r: f64, len: f64, ky: f64) -> String {
  let ticks: Vec<String> = angles
    .iter()
    .map(|&a| {
      let base = pt(r * a.cos(), r * a.sin());
      format!(
        "{{{base}, Scaled[{}, {base}]}}",
        pt(-len * a.cos(), -len * ky * a.sin())
      )
    })
    .collect();
  format!("Line[{{{}}}]", ticks.join(", "))
}

/// Labels just inside the major ticks: 3 points further in, aligned on the
/// side facing the rim.
fn dial_labels(labels: &[(f64, String)], r: f64, len: f64, ky: f64) -> String {
  let texts: Vec<String> = labels
    .iter()
    .map(|(a, label)| {
      let (c, s) = (a.cos(), a.sin());
      let base = pt(r * c, r * s);
      let m = c.abs().max(s.abs());
      format!(
        "Text[{label}, Offset[{}, Scaled[{}, {base}]], {}, {{1, 0}}]",
        pt(-3.0 * c, -3.0 * s),
        pt(-len * c, -len * ky * s),
        pt(c / m, s / m)
      )
    })
    .collect();
  texts.join(", ")
}

/// `AngularGauge[value, {min, max}, opts]`: a dial sweeping 300° clockwise
/// from the lower left, with a red needle at the value.
pub fn angular_gauge_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((lo, hi)) = range_of(args) else {
    return Ok(None);
  };
  let (lo_f, hi_f) =
    (try_eval_to_f64(&lo).unwrap(), try_eval_to_f64(&hi).unwrap());
  if hi_f == lo_f {
    return Ok(None);
  }
  // 0 of the scale sits at 240°, the end at -60°.
  let start = 240f64.to_radians();
  let sweep = 300f64.to_radians();
  let angle = |t: f64| start - sweep * t;
  // The arc spans 2 across but only 1 + sin 60° up, so a vertical share of
  // the plot range is that much shorter.
  let ky = 2.0 / (1.0 + 60f64.to_radians().sin());
  let mut parts = vec![dial_face(start)];
  if let Ok(divisions) = scale_divisions(args) {
    let sc = scale(&lo, &hi, divisions);
    let major: Vec<f64> = sc.major.iter().map(|(t, _)| angle(*t)).collect();
    let minor: Vec<f64> = sc.minor.iter().map(|&t| angle(t)).collect();
    let labels: Vec<(f64, String)> = sc
      .major
      .iter()
      .map(|(t, l)| (angle(*t), l.clone()))
      .collect();
    parts.push(format!(
      "{{Directive[CapForm[\"Butt\"], Thickness[0.004], {GRAY_25}], {}, {}}}, \
       Style[{{{}}}, {GRAY_25}, {DIAL_FONT}]",
      dial_ticks(&major, 1.0, 0.06, ky),
      dial_ticks(&minor, 1.0, 0.04, ky),
      dial_labels(&labels, 1.0, 0.06, ky)
    ));
  }
  let values = values_of(&args[0]);
  let colors = marker_colors(args, values.len());
  for (v, color) in values.iter().zip(&colors) {
    let Some(v) = v else { continue };
    let t = ((v - lo_f) / (hi_f - lo_f)).clamp(0.0, 1.0);
    let a = angle(t);
    let (c, s) = (0.9 * a.cos(), 0.9 * a.sin());
    parts.push(format!(
      "GeometricTransformation[Tooltip[{{{{{color}, Polygon[{{{{0., -0.0547}}, {{1, -0.025}}, {{1, 0.025}}, {{0., 0.0547}}}}]}}, \
       {{{color}, Disk[{{0, 0}}, 0.07]}}, {{GrayLevel[1], Disk[{{0, 0}}, 0.035]}}}}, {}], {{{{{}, {}}}, {{0., 0.}}}}]",
      num(*v),
      pt(c, -s),
      pt(s, c)
    ));
  }
  Ok(Some(graphics(&parts.join(", "), args, false, GRAY_25)?))
}

/// The time of day of a `ClockGauge` argument as `(h, m, s)`.
fn clock_time(
  args: &[Expr],
) -> Result<Option<(f64, f64, f64)>, InterpreterError> {
  let positional: Vec<&Expr> = args
    .iter()
    .filter(|a| !matches!(a, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
    .collect();
  let time_of = |list: &Expr| -> Option<(f64, f64, f64)> {
    match list {
      Expr::List(d) if d.len() >= 6 => Some((
        try_eval_to_f64(&d[3])?,
        try_eval_to_f64(&d[4])?,
        try_eval_to_f64(&d[5])?,
      )),
      _ => None,
    }
  };
  Ok(match positional.first() {
    None => time_of(&crate::evaluator::evaluate_function_call_ast(
      "DateList",
      &[],
    )?),
    Some(Expr::List(t)) if (1..=3).contains(&t.len()) => {
      let get = |i: usize| t.get(i).map_or(Some(0.0), try_eval_to_f64);
      match (get(0), get(1), get(2)) {
        (Some(h), Some(m), Some(sec)) => Some((h, m, sec)),
        _ => None,
      }
    }
    Some(date @ Expr::FunctionCall { name: d, .. }) if d == "DateObject" => {
      time_of(&crate::evaluator::evaluate_function_call_ast(
        "DateList",
        &[(*date).clone()],
      )?)
    }
    Some(e) => try_eval_to_f64(e).map(|secs| {
      let secs = secs.rem_euclid(86400.0);
      (
        (secs / 3600.0).floor(),
        ((secs % 3600.0) / 60.0).floor(),
        secs % 60.0,
      )
    }),
  })
}

/// `ClockGauge[time]`: a clock face with an hour tick and label per hour, a
/// minute tick between, red hour and minute hands and a gray second hand.
pub fn clock_gauge_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((h, m, sec)) = clock_time(args)? else {
    return Ok(None);
  };
  let quarter = std::f64::consts::FRAC_PI_2;
  let at = |turns: f64| quarter - turns * std::f64::consts::TAU;
  // The scale runs from 0 to 12 o'clock, both at the top: 0 is labeled
  // with an empty string under the 12.
  let major: Vec<f64> = (0..=12).map(|k| at(f64::from(k) / 12.0)).collect();
  let minor: Vec<f64> = (0..60)
    .filter(|k| k % 5 != 0)
    .map(|k| at(f64::from(k) / 60.0))
    .collect();
  let labels: Vec<(f64, String)> = (0..=12)
    .map(|k| {
      let label = if k == 0 {
        "\"\"".to_string()
      } else {
        k.to_string()
      };
      (at(f64::from(k) / 12.0), label)
    })
    .collect();
  let mut parts = vec![dial_face(quarter)];
  parts.push(format!(
    "{{Directive[CapForm[\"Butt\"], Thickness[0.006], {GRAY_25}], {}, {}}}, \
     Style[{{{}}}, {GRAY_25}, {DIAL_FONT}]",
    dial_ticks(&major, 0.93, 0.03, 1.0),
    dial_ticks(&minor, 0.93, 0.02, 1.0),
    dial_labels(&labels, 0.93, 0.03, 1.0)
  ));
  let total = sec + 60.0 * m + 3600.0 * h;
  let hand = |turns: f64, len: f64, color: &str, shape: &str, hub: f64| {
    let a = at(turns.rem_euclid(1.0));
    let (c, s) = (len * a.cos(), len * a.sin());
    format!(
      "GeometricTransformation[Inset[Graphics[{{Directive[{color}], {{{{Polygon[{shape}], Disk[{{0, 0}}, {hub}]}}, {{GrayLevel[1], Disk[{{0, 0}}, 0.03]}}}}}}], {{0, 0}}, {{0, 0}}, {{1, 1}}], {{{{{}, {}}}, {{0., 0.}}}}]",
      pt(c, -s),
      pt(s, c)
    )
  };
  parts.push(hand(
    total / 43200.0,
    0.6225,
    "Hue[0, 1, 0.8]",
    "{{0, -0.08}, {1, -0.04}, {1, 0.04}, {0, 0.08}}",
    0.09,
  ));
  parts.push(hand(
    total / 3600.0,
    0.996,
    "Hue[0, 1, 0.8]",
    "{{0, -0.05}, {1, -0.02}, {1, 0.02}, {0, 0.05}}",
    0.09,
  ));
  parts.push(hand(
    total / 60.0,
    1.079,
    "GrayLevel[0.6]",
    "{{-0.2, -0.03}, {1, -0.01}, {1, 0.01}, {-0.2, 0.03}}",
    0.07,
  ));
  Ok(Some(graphics(&parts.join(", "), args, true, GRAY_25)?))
}

/// The ticks and labels of a linear scale along `0..1`: `place(along,
/// across)` turns scale coordinates into picture ones, `tick` is the
/// direction ticks point (as a `Scaled` share per unit length), and the
/// labels sit `label_offset` points from the scale, aligned by `align`.
struct LinearScale<'a> {
  place: &'a dyn Fn(f64, f64) -> (f64, f64),
  tick: (f64, f64),
  major_len: f64,
  minor_len: f64,
  /// Labels hang off the tick's far end rather than the scale line.
  label_at_tick_end: bool,
  label_offset: (f64, f64),
  align: (f64, f64),
}

impl LinearScale<'_> {
  fn ticks(&self, at: &[f64], len: f64) -> String {
    let ticks: Vec<String> = at
      .iter()
      .map(|&t| {
        let (x, y) = (self.place)(t, 0.0);
        let base = pt(x, y);
        format!(
          "{{{base}, Scaled[{}, {base}]}}",
          pt(self.tick.0 * len, self.tick.1 * len)
        )
      })
      .collect();
    format!("Line[{{{}}}]", ticks.join(", "))
  }

  fn render(&self, sc: &Scale, line: bool) -> String {
    let major: Vec<f64> = sc.major.iter().map(|(t, _)| *t).collect();
    let texts: Vec<String> = sc
      .major
      .iter()
      .map(|(t, label)| {
        let (x, y) = (self.place)(*t, 0.0);
        let anchor = if self.label_at_tick_end {
          format!(
            "Scaled[{}, {}]",
            pt(self.tick.0 * self.major_len, self.tick.1 * self.major_len),
            pt(x, y)
          )
        } else {
          pt(x, y)
        };
        format!(
          "Text[{label}, Offset[{}, {anchor}], {}, {{1, 0}}]",
          pt(self.label_offset.0, self.label_offset.1),
          pt(self.align.0, self.align.1)
        )
      })
      .collect();
    let axis = if line {
      let (x0, y0) = (self.place)(0.0, 0.0);
      let (x1, y1) = (self.place)(1.0, 0.0);
      format!(
        "{{AbsoluteThickness[1], {GRAY_25}, Line[{{{}, {}}}]}}, ",
        pt(x0, y0),
        pt(x1, y1)
      )
    } else {
      String::new()
    };
    format!(
      "{axis}{{Directive[CapForm[\"Butt\"], AbsoluteThickness[0.5], GrayLevel[0.5]], {}, {}}}, \
       Style[{{{}}}, {GRAY_25}, {LINEAR_FONT}]",
      self.ticks(&major, self.major_len),
      self.ticks(&sc.minor, self.minor_len),
      texts.join(", ")
    )
  }
}

/// `HorizontalGauge[value(s), {min, max}, opts]` / `VerticalGauge[…]`: a
/// scale line with ticks and labels and a triangular pointer per value.
pub fn linear_gauge_ast(
  args: &[Expr],
  vertical: bool,
) -> Result<Option<Expr>, InterpreterError> {
  let Some((lo, hi)) = range_of(args) else {
    return Ok(None);
  };
  let (lo_f, hi_f) =
    (try_eval_to_f64(&lo).unwrap(), try_eval_to_f64(&hi).unwrap());
  if hi_f == lo_f {
    return Ok(None);
  }
  let place = move |along: f64, across: f64| {
    if vertical {
      (across, along)
    } else {
      (along, across)
    }
  };
  let (p0, p1) = (place(-0.1, -0.15), place(1.1, 0.15));
  let mut parts = vec![format!(
    "{{Opacity[0], Point[{{{}, {}}}]}}",
    pt(p0.0, p0.1),
    pt(p1.0, p1.1)
  )];
  let scale_spec = LinearScale {
    place: &place,
    tick: if vertical { (-1.0, 0.0) } else { (0.0, 1.0) },
    major_len: 0.06,
    minor_len: 0.04,
    label_at_tick_end: false,
    label_offset: if vertical { (3.0, 0.0) } else { (0.0, -3.0) },
    align: if vertical { (-1.0, 0.0) } else { (0.0, 1.0) },
  };
  match scale_divisions(args) {
    Ok(divisions) => {
      parts.push(scale_spec.render(&scale(&lo, &hi, divisions), true));
    }
    Err(()) => parts.push(scale_spec.render(
      &Scale {
        major: Vec::new(),
        minor: Vec::new(),
      },
      true,
    )),
  }
  let values = values_of(&args[0]);
  let colors = marker_colors(args, values.len());
  for (v, color) in values.iter().zip(&colors) {
    let Some(v) = v else { continue };
    let t = ((v - lo_f) / (hi_f - lo_f)).clamp(0.0, 1.0);
    // The triangle points at the scale from the side the ticks are on.
    let (matrix, shift) = if vertical {
      ("{{0., -0.075}, {0.075, 0.}}", pt(-0.0375, t))
    } else {
      ("{{0.075, 0.}, {0., 0.075}}", pt(t, 0.0375))
    };
    parts.push(format!(
      "{{Directive[{color}], GeometricTransformation[Tooltip[Polygon[{{{{0.5, 0.5}}, {{-0.5, 0.5}}, {{0., -0.5}}}}], {}], {{{matrix}, {shift}}}]}}",
      num(*v)
    ));
  }
  Ok(Some(graphics(&parts.join(", "), args, false, GRAY_25)?))
}

/// The red, green, blue of a color expression, if it is one.
fn rgb_of(color: &Expr) -> Option<(f64, f64, f64)> {
  let rgb = eval(call(
    "ColorConvert",
    vec![color.clone(), Expr::String("RGB".into())],
  ));
  match &rgb {
    Expr::FunctionCall { name, args }
      if name == "RGBColor" && args.len() >= 3 =>
    {
      Some((
        try_eval_to_f64(&args[0])?,
        try_eval_to_f64(&args[1])?,
        try_eval_to_f64(&args[2])?,
      ))
    }
    _ => None,
  }
}

/// Hue and saturation of an RGB color.
fn hue_saturation((r, g, b): (f64, f64, f64)) -> (f64, f64) {
  let max = r.max(g).max(b);
  let min = r.min(g).min(b);
  let d = max - min;
  if d <= 0.0 || max <= 0.0 {
    return (0.0, 0.0);
  }
  let h = if max == r {
    ((g - b) / d).rem_euclid(6.0)
  } else if max == g {
    (b - r) / d + 2.0
  } else {
    (r - g) / d + 4.0
  };
  (h / 6.0, d / max)
}

/// `ThermometerGauge[value, {min, max}, opts]`: a scale on the left of a
/// glass tube whose red column rises from the bulb to the value.
pub fn thermometer_gauge_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((lo, hi)) = range_of(args) else {
    return Ok(None);
  };
  let (lo_f, hi_f) =
    (try_eval_to_f64(&lo).unwrap(), try_eval_to_f64(&hi).unwrap());
  if hi_f == lo_f {
    return Ok(None);
  }
  let place = |along: f64, across: f64| (across, along);
  let mut parts = vec![format!(
    "{{Opacity[0], Point[{{{}, {}}}]}}",
    pt(-0.22, -0.15),
    pt(0.13, 1.1)
  )];
  let scale_spec = LinearScale {
    place: &place,
    tick: (-1.0, 0.0),
    major_len: 0.1,
    minor_len: 0.065,
    label_at_tick_end: true,
    label_offset: (-3.0, 0.0),
    align: (1.0, 0.0),
  };
  if let Ok(divisions) = scale_divisions(args) {
    parts.push(scale_spec.render(&scale(&lo, &hi, divisions), false));
  }
  // The column's hue and saturation follow `GaugeStyle`.
  let (hue, sat) = option(args, "GaugeStyle")
    .as_ref()
    .and_then(rgb_of)
    .map_or((0.015, 0.73), hue_saturation);
  let (x0, x1, cx) = (0.0046875, 0.0703125, 0.0375);
  let bulb_r = 0.04921875;
  let bottom = -bulb_r;
  // Four overlapping strips shade a tube: the full width, its left part,
  // the middle and a highlight.
  let strips = [
    (x0, x1, 0.8),
    (x0, 0.06375, 0.6),
    (0.01453125, 0.050625, 0.7),
    (0.02109375, cx, 1.0),
  ];
  let tube = |y0: f64, y1: f64, h: f64, s: f64| -> String {
    strips
      .iter()
      .map(|&(a, b, v)| {
        format!(
          "{{EdgeForm[None], Hue[{}, {}, {}], Polygon[{{{}, {}, {}, {}}}]}}",
          num(h),
          num(s),
          num(v),
          pt(a, y0),
          pt(b, y0),
          pt(b, y1),
          pt(a, y1)
        )
      })
      .collect::<Vec<_>>()
      .join(", ")
  };
  let value = try_eval_to_f64(&args[0]);
  let t = value.map_or(0.0, |v| ((v - lo_f) / (hi_f - lo_f)).clamp(0.0, 1.0));
  let tooltip = value.map_or_else(|| "Null".to_string(), num);
  parts.push(format!(
    "Tooltip[{{{}}}, {tooltip}]",
    tube(bottom, t, hue, sat)
  ));
  parts.push(tube(t, 1.0, 0.0, 0.0));
  parts.push(format!(
    "{{AbsoluteThickness[2], CapForm[\"Butt\"], Hue[{}, {}, 1], Line[{{{}, {}}}]}}",
    num(hue),
    num(sat),
    pt(x0, t),
    pt(x1, t)
  ));
  // The bulb: a fan from a highlight above and left of its center, its rim
  // shaded from the column's color at the top to 0.3 brightness at the
  // bottom.
  let n = 50;
  let mut bulb_pts = vec![pt(cx - bulb_r / 4.0, bottom + bulb_r / 2.0)];
  let mut bulb_colors = vec![format!("Hue[{}, {}, 1]", num(hue), num(sat))];
  for k in 0..=n {
    let a = std::f64::consts::FRAC_PI_2
      - std::f64::consts::TAU * f64::from(k) / f64::from(n);
    bulb_pts.push(pt(cx + bulb_r * a.cos(), bottom + bulb_r * a.sin()));
    let v = 0.65 + 0.35 * a.sin();
    bulb_colors.push(format!("Hue[{}, {}, {}]", num(hue), num(sat), num(v)));
  }
  // A solid disk under the shading keeps the fan's seams from showing.
  parts.push(format!(
    "Tooltip[{{EdgeForm[None], Hue[{h}, {s}, 0.7], Disk[{}, {}], Polygon[{{{}}}, VertexColors -> {{{}}}]}}, {tooltip}]",
    pt(cx, bottom),
    num(bulb_r),
    bulb_pts.join(", "),
    bulb_colors.join(", "),
    h = num(hue),
    s = num(sat),
  ));
  Ok(Some(graphics(&parts.join(", "), args, false, "{}")?))
}
