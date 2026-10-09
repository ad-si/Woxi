//! The axes of an unframed 2D plot, drawn where the Wolfram Language draws
//! them: crossing at the `AxesOrigin` rather than along the edges of the
//! plotting area.
//!
//! Shared by the line and scatter renderers in [`super::plot`], which draw
//! their data through plotters but leave the axes (lines, tick marks and
//! tick labels) to [`origin_axes_svg`].

use crate::functions::plot::{
  AXIS_TICK_TARGET, format_date_tick, format_tick_with_step,
  generate_date_ticks, nice_step,
};

/// Tick-label font size, in display pixels.
pub(crate) const AXIS_TICK_FONT: f64 = 10.5;
/// Length of a labelled tick mark, in display pixels (Wolfram draws 4).
const AXIS_MAJOR_TICK_LEN: f64 = 4.0;
/// Length of an unlabelled tick mark, in display pixels (Wolfram draws 2.4).
const AXIS_MINOR_TICK_LEN: f64 = 2.4;

/// How one axis maps data values onto its length.
#[derive(Clone, Copy)]
pub(crate) struct AxisScale {
  pub min: f64,
  pub max: f64,
  pub log: bool,
}

impl AxisScale {
  /// Where `v` falls along the axis, as a fraction of its length.
  pub(crate) fn frac(&self, v: f64) -> f64 {
    if self.log {
      (v.ln() - self.min.ln()) / (self.max.ln() - self.min.ln())
    } else {
      (v - self.min) / (self.max - self.min)
    }
  }

  fn contains(&self, v: f64) -> bool {
    let eps = (self.max - self.min).abs() * 1e-9;
    v >= self.min - eps && v <= self.max + eps
  }

  /// The value an axis sits at on the *other* axis: an explicit `AxesOrigin`
  /// coordinate, or the natural origin (0, or 1 on a log scale) when it is
  /// in range, else the end of the range nearest to it — so an all-positive
  /// range carries its crossing axis at its low end and an all-negative one
  /// at its high end, as Wolfram's `AxesOrigin -> Automatic` does.
  ///
  /// `low`, when given, is the low end of the plot range *before* its
  /// padding: Wolfram puts the axis of an all-positive `Plot` at the lowest
  /// point of the curve rather than at the padded edge. (An all-negative
  /// one goes to the padded edge.)
  pub(crate) fn origin(&self, explicit: Option<f64>, low: Option<f64>) -> f64 {
    let (lo, hi) = (self.min.min(self.max), self.min.max(self.max));
    if !(lo.is_finite() && hi.is_finite()) {
      return explicit.unwrap_or(0.0);
    }
    if let Some(o) = explicit.filter(|v| v.is_finite()) {
      return o.clamp(lo, hi);
    }
    let natural = if self.log { 1.0 } else { 0.0 };
    if natural < lo {
      low.filter(|l| l.is_finite()).unwrap_or(lo).clamp(lo, hi)
    } else {
      natural.min(hi)
    }
  }
}

/// The render-space point where the axes cross — the x of the vertical axis
/// and the y of the horizontal one — for the plotting rectangle `area` and
/// the crossing point `origin` in data coordinates.
pub(crate) fn axes_crossing_px(
  (x0, y0, w, h): (f64, f64, f64, f64),
  x: AxisScale,
  y: AxisScale,
  origin: (f64, f64),
) -> (f64, f64) {
  (x0 + x.frac(origin.0) * w, y0 + h - y.frac(origin.1) * h)
}

/// The tick marks of one axis: the labelled ones with their SVG label
/// markup, and the unlabelled ones between them.
#[derive(Default)]
pub(crate) struct AxisTicks {
  pub majors: Vec<(f64, String)>,
  pub minors: Vec<f64>,
}

impl AxisTicks {
  /// The automatic ticks of an axis: `step`-spaced majors with a fifth of
  /// that between them on a linear axis, decades on a log axis and calendar
  /// steps on a date axis. `explicit` (from `Ticks -> {…}`) replaces them
  /// with exactly the positions it names.
  pub(crate) fn new(
    scale: AxisScale,
    date: bool,
    explicit: Option<&Vec<(f64, String)>>,
  ) -> Self {
    let AxisScale { min, max, log } = scale;
    if let Some(ticks) = explicit {
      return Self {
        majors: ticks
          .iter()
          .filter(|(p, _)| scale.contains(*p))
          .cloned()
          .collect(),
        minors: Vec::new(),
      };
    }
    if !(min.is_finite() && max.is_finite()) || max <= min {
      return Self::default();
    }
    if log {
      return Self::log(min, max);
    }
    if date {
      return Self {
        majors: generate_date_ticks(min, max)
          .into_iter()
          .filter(|t| scale.contains(*t))
          .map(|t| {
            (
              t,
              crate::functions::graphics::svg_escape(&format_date_tick(t)),
            )
          })
          .collect(),
        minors: Vec::new(),
      };
    }
    let major = nice_step(max - min, AXIS_TICK_TARGET);
    if !(major.is_finite() && major > 0.0) {
      return Self::default();
    }
    let minor = major / 5.0;
    let mut ticks = Self::default();
    let first = (min / minor - 1e-9).ceil() as i64;
    let last = (max / minor + 1e-9).floor() as i64;
    if last - first > 10_000 {
      return ticks;
    }
    for i in first..=last {
      let v = i as f64 * minor;
      if i.rem_euclid(5) == 0 {
        let v = if v.abs() < minor * 1e-6 { 0.0 } else { v };
        ticks.majors.push((
          v,
          crate::functions::graphics::svg_escape(&format_tick_with_step(
            v, major,
          )),
        ));
      } else {
        ticks.minors.push(v);
      }
    }
    ticks
  }

  /// Decades `10^k` as majors — every one, or every second/third over a
  /// wide range — written with a superscript exponent, and the multiples
  /// `2…9 × 10^k` between them as minors.
  fn log(min: f64, max: f64) -> Self {
    let mut ticks = Self::default();
    if min <= 0.0 {
      return ticks;
    }
    let (log_min, log_max) = (min.log10(), max.log10());
    let decades = log_max - log_min;
    let step = if decades <= 8.0 {
      1
    } else if decades <= 16.0 {
      2
    } else {
      3
    };
    let lo = log_min.floor() as i64;
    let hi = log_max.ceil() as i64;
    let contains = |v: f64| v >= min * (1.0 - 1e-9) && v <= max * (1.0 + 1e-9);
    for exp in lo..=hi {
      let decade = 10f64.powi(exp as i32);
      if exp.rem_euclid(step) == 0 && contains(decade) {
        let label = match exp {
          0 => "1".to_string(),
          1 => "10".to_string(),
          _ => format!(
            "10<tspan baseline-shift=\"super\" font-size=\"70%\">{exp}</tspan>"
          ),
        };
        ticks.majors.push((decade, label));
      } else if contains(decade) {
        ticks.minors.push(decade);
      }
      if step == 1 {
        for m in 2..=9 {
          let v = m as f64 * decade;
          if contains(v) {
            ticks.minors.push(v);
          }
        }
      }
    }
    ticks
  }

  /// Widest label, in estimated display pixels at the tick font.
  pub(crate) fn max_label_width(&self) -> f64 {
    self
      .majors
      .iter()
      .map(|(_, l)| label_width(l))
      .fold(0.0, f64::max)
  }
}

/// A rough width for a tick label's markup: its visible characters at
/// ~0.6 em each.
fn label_width(markup: &str) -> f64 {
  let mut visible = 0usize;
  let mut in_tag = false;
  let mut in_entity = false;
  for c in markup.chars() {
    match c {
      '<' => in_tag = true,
      '>' => in_tag = false,
      '&' if !in_tag => {
        in_entity = true;
        visible += 1;
      }
      ';' if in_entity => in_entity = false,
      _ if in_tag || in_entity => {}
      _ => visible += 1,
    }
  }
  visible as f64 * AXIS_TICK_FONT * 0.6
}

/// Room, in display pixels, the tick labels of an axis need on the far side
/// of it: the y labels to the left of the vertical axis, the x labels
/// below the horizontal one.
pub(crate) fn y_tick_label_room(ticks: &AxisTicks) -> f64 {
  if ticks.majors.is_empty() {
    0.0
  } else {
    ticks.max_label_width() + AXIS_TICK_FONT * 0.45 + 2.0
  }
}

pub(crate) fn x_tick_label_room(ticks: &AxisTicks) -> f64 {
  if ticks.majors.is_empty() {
    0.0
  } else {
    AXIS_TICK_FONT * 1.45
  }
}

/// The gutter an axis's labels need outside the plotting area when the axis
/// sits `frac` of the way along the other axis's `length` (all in render
/// units): whatever part of `room` the distance to the edge does not cover.
pub(crate) fn label_gutter(room: f64, frac: f64, length: f64) -> f64 {
  (room - frac.clamp(0.0, 1.0) * length).max(0.0)
}

/// Everything [`origin_axes_svg`] draws from.
pub(crate) struct OriginAxes<'a> {
  /// The plotting rectangle `(x0, y0, w, h)` in render units.
  pub area: (f64, f64, f64, f64),
  pub x: AxisScale,
  pub y: AxisScale,
  /// The point the axes cross at, in data coordinates.
  pub origin: (f64, f64),
  /// Which axes are drawn.
  pub show: (bool, bool),
  /// The ticks of each axis; `None` for `Ticks -> None`.
  pub ticks: Option<(&'a AxisTicks, &'a AxisTicks)>,
  /// Render units per display pixel.
  pub sf: f64,
  pub axis_color: &'a str,
  pub label_fill: &'a str,
}

/// The axis lines, tick marks and tick labels of an unframed plot.
///
/// The x axis runs the full width of the plotting area at `origin.1` and
/// the y axis its full height at `origin.0`. Tick marks point into the
/// plot (up from the x axis, right from the y axis); labels sit below the x
/// axis and left of the y axis. A label that would collide with the other
/// axis is dropped, which is how the shared `0` at a centred origin
/// disappears from both.
pub(crate) fn origin_axes_svg(a: &OriginAxes) -> String {
  let (x0, y0, w, h) = a.area;
  let sf = a.sf;
  let mut svg = String::new();
  if !(w > 0.0 && h > 0.0) {
    return svg;
  }
  let px = |v: f64| x0 + a.x.frac(v) * w;
  let py = |v: f64| y0 + h - a.y.frac(v) * h;
  let axis_y = py(a.origin.1);
  let axis_x = px(a.origin.0);
  if !(axis_x.is_finite() && axis_y.is_finite()) {
    return svg;
  }
  let color = a.axis_color;
  let fill = a.label_fill;
  let font = AXIS_TICK_FONT * sf;
  let stroke = sf;
  let line = |svg: &mut String, x1: f64, y1: f64, x2: f64, y2: f64| {
    svg.push_str(&format!(
      "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" \
       stroke=\"{color}\" stroke-width=\"{stroke:.0}\"/>\n"
    ));
  };
  let major_len = AXIS_MAJOR_TICK_LEN * sf;
  let minor_len = AXIS_MINOR_TICK_LEN * sf;

  if a.show.0 {
    line(&mut svg, x0, axis_y, x0 + w, axis_y);
    if let Some((ticks, _)) = a.ticks {
      for &v in &ticks.minors {
        let x = px(v);
        if x.is_finite() {
          line(&mut svg, x, axis_y, x, axis_y - minor_len);
        }
      }
      for (v, label) in &ticks.majors {
        let x = px(*v);
        if !x.is_finite() {
          continue;
        }
        line(&mut svg, x, axis_y, x, axis_y - major_len);
        // The label would straddle the vertical axis.
        if a.show.1
          && (x - axis_x).abs() < (label_width(label) / 2.0 + 2.0) * sf
        {
          continue;
        }
        svg.push_str(&format!(
          "<text x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
           font-family=\"sans-serif\" font-size=\"{font:.0}\" \
           fill=\"{fill}\">{label}</text>\n",
          axis_y + font * 1.15
        ));
      }
    }
  }

  if a.show.1 {
    line(&mut svg, axis_x, y0, axis_x, y0 + h);
    if let Some((_, ticks)) = a.ticks {
      for &v in &ticks.minors {
        let y = py(v);
        if y.is_finite() {
          line(&mut svg, axis_x, y, axis_x + minor_len, y);
        }
      }
      for (v, label) in &ticks.majors {
        let y = py(*v);
        if !y.is_finite() {
          continue;
        }
        line(&mut svg, axis_x, y, axis_x + major_len, y);
        // The label would sit on the horizontal axis.
        if a.show.0 && (y - axis_y).abs() < font * 0.6 {
          continue;
        }
        svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" \
           font-family=\"sans-serif\" font-size=\"{font:.0}\" \
           fill=\"{fill}\">{label}</text>\n",
          axis_x - font * 0.45,
          y + font * 0.35
        ));
      }
    }
  }
  svg
}

#[cfg(test)]
mod tests {
  use super::*;

  fn lin(min: f64, max: f64) -> AxisScale {
    AxisScale {
      min,
      max,
      log: false,
    }
  }

  #[test]
  fn automatic_origin_is_zero_or_nearest_end() {
    assert_eq!(lin(-2.0, 5.0).origin(None, None), 0.0);
    assert_eq!(lin(2.0, 10.0).origin(None, None), 2.0);
    assert_eq!(lin(-10.0, -2.0).origin(None, None), -2.0);
    assert_eq!(lin(-2.0, 5.0).origin(Some(3.0), None), 3.0);
    let log = AxisScale {
      min: 0.5,
      max: 100.0,
      log: true,
    };
    assert_eq!(log.origin(None, None), 1.0);
    // The low end of the unpadded range, not the padded axis — unless the
    // origin is explicit, or 0 is within the padding.
    assert_eq!(lin(1.6, 10.4).origin(None, Some(2.0)), 2.0);
    assert_eq!(lin(1.6, 10.4).origin(Some(1.8), Some(2.0)), 1.8);
    assert_eq!(lin(-0.1, 2.0).origin(None, Some(0.05)), 0.0);
  }

  #[test]
  fn linear_ticks_split_majors_and_minors() {
    let t = AxisTicks::new(lin(-1.0, 1.0), false, None);
    let labels: Vec<&str> = t.majors.iter().map(|(_, l)| l.as_str()).collect();
    assert_eq!(labels, ["-1.0", "-0.5", "0.0", "0.5", "1.0"]);
    assert_eq!(t.minors.len(), 16);
  }

  #[test]
  fn log_ticks_label_decades() {
    let t = AxisTicks::log(1.0, 1000.0);
    let pos: Vec<f64> = t.majors.iter().map(|(p, _)| *p).collect();
    assert_eq!(pos, [1.0, 10.0, 100.0, 1000.0]);
    assert!(t.majors[2].1.contains("baseline-shift"));
  }
}
