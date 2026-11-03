//! Curve and dataset labels placed somewhere along their curve:
//! `Labeled[f, label, pos]` and `Callout[f, label, pos]` in `Plot` and
//! `ListPlot`, and the `Placed[…, pos]` / `Callout[…, pos]` forms of
//! `PlotLabels`.
//!
//! WL 15 resolves these into primitives anchored at a data point and laid
//! out in printer's points around it (`Offset[{dx, dy}, {x, y}]`): a
//! translucent white box with the label in it, and for a callout a gray
//! leader from the curve to the box. This module computes the same layout —
//! anchor in data coordinates, everything else in points — once to widen
//! the plot range so the labels fit, and once more at render time with the
//! exact scale of the drawn plot.

use crate::functions::plot_padding;
use crate::syntax::Expr;

/// Where along its curve a label goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LabelAt {
  /// The curve's highest point (`Above`).
  Max,
  /// The curve's lowest point (`Below`).
  Min,
  /// Left of the curve's first point (`Before`).
  Start,
  /// Right of the curve's last point (`After`).
  Finish,
  /// The curve point at this x (`Labeled[f, l, 3]`).
  X(f64),
  /// The curve point this fraction along its x extent (`Scaled[s]`).
  Scaled(f64),
}

/// Which side of its anchor point a label is put on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LabelSide {
  Above,
  Below,
  Left,
  Right,
}

/// One label of a curve, as given.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CurveLabelSpec {
  /// The series (curve / dataset) the label belongs to.
  pub series: usize,
  /// The label text (box markup, as from `traditional_label`).
  pub text: String,
  pub at: LabelAt,
  /// The side the position asks for; `None` lets the layout pick a free
  /// one.
  pub side: Option<LabelSide>,
  /// A callout: hung off a leader line instead of sitting at the curve.
  pub callout: bool,
}

/// Parse a label position (the third argument of `Labeled` / `Callout`, or
/// the second of `Placed`) into where along the curve and on which side the
/// label goes. `None` for the automatic place, the end of the curve.
pub(crate) fn parse_label_position(
  pos: &Expr,
) -> Option<(LabelAt, Option<LabelSide>)> {
  let named = |s: &str| match s {
    "Above" | "Top" => Some((LabelAt::Max, Some(LabelSide::Above))),
    "Below" | "Bottom" => Some((LabelAt::Min, Some(LabelSide::Below))),
    "Before" | "Left" => Some((LabelAt::Start, Some(LabelSide::Left))),
    "After" | "Right" => Some((LabelAt::Finish, Some(LabelSide::Right))),
    _ => None,
  };
  let side_of = |e: &Expr| match e {
    Expr::Identifier(s) => match s.as_str() {
      "Above" | "Top" => Some(LabelSide::Above),
      "Below" | "Bottom" => Some(LabelSide::Below),
      "Before" | "Left" => Some(LabelSide::Left),
      "After" | "Right" => Some(LabelSide::Right),
      _ => None,
    },
    _ => None,
  };
  let at_of = |e: &Expr| -> Option<LabelAt> {
    match e {
      Expr::FunctionCall { name, args }
        if name == "Scaled" && args.len() == 1 =>
      {
        crate::functions::graphics::expr_to_f64(&args[0]).map(LabelAt::Scaled)
      }
      Expr::Identifier(s) => named(s).map(|(at, _)| at),
      other => crate::functions::graphics::expr_to_f64(other).map(LabelAt::X),
    }
  };
  match pos {
    Expr::Identifier(s) if s == "Automatic" => None,
    Expr::Identifier(s) => named(s),
    Expr::List(items) if items.len() == 2 => {
      let at = at_of(&items[0])?;
      Some((at, side_of(&items[1])))
    }
    other => at_of(other).map(|at| (at, None)),
  }
}

/// A label laid out: its box and leader in points around `anchor`.
#[derive(Clone, Debug)]
pub(crate) struct PlacedLabel {
  pub text: String,
  /// The anchor, in data coordinates.
  pub anchor: (f64, f64),
  /// The label box relative to the anchor, in points, y up:
  /// (left, bottom, right, top).
  pub bbox: (f64, f64, f64, f64),
  /// A callout's leader as points relative to the anchor (y up): its start
  /// near the curve, the bend and its end at the box.
  pub leader: Vec<(f64, f64)>,
}

/// How data coordinates map to points: points per data unit on each axis
/// and the displayed range.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LabelScale {
  pub kx: f64,
  pub ky: f64,
  pub x_range: (f64, f64),
  pub y_range: (f64, f64),
}

/// Font size of curve labels, in points.
const FONT: f64 = plot_padding::LABEL_FONT;

/// The box of a label: the text plus 3 pt either side; 14 pt tall for a
/// string (12 for a callout), 4 pt more for a typeset expression.
fn box_size(text: &str, callout: bool) -> (f64, f64) {
  let markup = crate::functions::graphics::box_string_to_svg(text);
  let w = plot_padding::text_width(&markup, FONT) + 6.0;
  let typeset =
    text.starts_with(crate::functions::graphics::STYLED_LABEL_MARKER);
  let h = if callout { 12.0 } else { 14.0 } + if typeset { 4.0 } else { 0.0 };
  (w, h)
}

/// The curve's value at `x`, linearly interpolated between its samples.
fn curve_at(points: &[(f64, f64)], x: f64) -> Option<(f64, f64)> {
  let finite: Vec<(f64, f64)> = points
    .iter()
    .copied()
    .filter(|(a, b)| a.is_finite() && b.is_finite())
    .collect();
  for w in finite.windows(2) {
    let ((x0, y0), (x1, y1)) = (w[0], w[1]);
    if (x0 <= x && x <= x1) || (x1 <= x && x <= x0) {
      let t = if (x1 - x0).abs() < f64::EPSILON {
        0.0
      } else {
        (x - x0) / (x1 - x0)
      };
      return Some((x, y0 + t * (y1 - y0)));
    }
  }
  // Outside the samples: the nearest one.
  finite.into_iter().min_by(|a, b| {
    (a.0 - x)
      .abs()
      .partial_cmp(&(b.0 - x).abs())
      .unwrap_or(std::cmp::Ordering::Equal)
  })
}

/// The point a label is attached to.
fn anchor_point(points: &[(f64, f64)], at: LabelAt) -> Option<(f64, f64)> {
  let finite = || {
    points
      .iter()
      .copied()
      .filter(|(a, b)| a.is_finite() && b.is_finite())
  };
  let by_y = |a: &(f64, f64), b: &(f64, f64)| {
    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
  };
  // The first of the (numerically) highest / lowest points.
  let extreme = |best: Option<(f64, f64)>| {
    let (_, y) = best?;
    let (lo, hi) = finite().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| {
      (a.0.min(p.1), a.1.max(p.1))
    });
    let tol = 1e-6 * (hi - lo).abs().max(f64::MIN_POSITIVE);
    finite().find(|p| (p.1 - y).abs() <= tol)
  };
  match at {
    LabelAt::Max => extreme(finite().max_by(by_y)),
    LabelAt::Min => extreme(finite().min_by(by_y)),
    LabelAt::Start => finite().next(),
    LabelAt::Finish => finite().next_back(),
    LabelAt::X(x) => curve_at(points, x),
    LabelAt::Scaled(s) => {
      let (lo, hi) = finite()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, p| {
          (acc.0.min(p.0), acc.1.max(p.0))
        });
      curve_at(points, lo + s * (hi - lo))
    }
  }
}

/// Lay out the labels of `specs` along `curves` at the given scale.
pub(crate) fn layout_curve_labels(
  specs: &[CurveLabelSpec],
  curves: &[Vec<(f64, f64)>],
  scale: LabelScale,
) -> Vec<PlacedLabel> {
  let LabelScale {
    kx,
    ky,
    x_range,
    y_range,
  } = scale;
  if !(kx.is_finite() && ky.is_finite() && kx > 0.0 && ky > 0.0) {
    return Vec::new();
  }
  let mut out = Vec::new();
  for spec in specs {
    let Some(points) = curves.get(spec.series) else {
      continue;
    };
    let Some(p) = anchor_point(points, spec.at) else {
      continue;
    };
    let (w, h) = box_size(&spec.text, spec.callout);
    // Whether a label box stays inside the range.
    let inside = |anchor: (f64, f64), bbox: (f64, f64, f64, f64)| {
      anchor.0 + bbox.0 / kx >= x_range.0
        && anchor.0 + bbox.2 / kx <= x_range.1
        && anchor.1 + bbox.1 / ky >= y_range.0
        && anchor.1 + bbox.3 / ky <= y_range.1
    };
    // Gaps WL leaves: 3% of the y span between the curve and a label above
    // or below it, the x padding (1/48 of the span) before / after it.
    let y_gap = 0.027 * (y_range.1 - y_range.0);
    let x_gap = (x_range.1 - x_range.0) / 50.0;
    let above = (-w / 2.0, 0.0, w / 2.0, h);
    let below = (-w / 2.0, -h, w / 2.0, 0.0);
    let placed = if spec.callout {
      callout_layout(spec, points, p, (w, h), (kx, ky))
    } else {
      let (anchor, bbox) = match spec.side {
        Some(LabelSide::Above) => ((p.0, p.1 + y_gap), above),
        Some(LabelSide::Below) => ((p.0, p.1 - y_gap), below),
        Some(LabelSide::Left) => (
          (x_range.0.min(p.0 - x_gap), p.1),
          (-w, -h / 2.0, 0.0, h / 2.0),
        ),
        Some(LabelSide::Right) => (
          (x_range.1.max(p.0 + x_gap), p.1),
          (0.0, -h / 2.0, w, h / 2.0),
        ),
        None => {
          // Over a flat stretch the label sits above the point (below
          // when above leaves the range); on a slope it moves a step
          // along the curve to the open side: up the slope's high end is
          // behind it, so above-right of a falling curve and above-left of
          // a rising one, or the mirror image below.
          let step = 0.0248 * (x_range.1 - x_range.0);
          let slope = match (
            curve_at(points, p.0 - step),
            curve_at(points, p.0 + step),
          ) {
            (Some(a), Some(b)) if b.0 > a.0 => {
              (b.1 - a.1) * ky / ((b.0 - a.0) * kx)
            }
            _ => 0.0,
          };
          let at = |dx: f64| curve_at(points, p.0 + dx).unwrap_or(p);
          let (up, down) = if slope.abs() < 0.5 {
            (p, p)
          } else if slope < 0.0 {
            (at(step), at(-step))
          } else {
            (at(-step), at(step))
          };
          [
            ((up.0, up.1 + y_gap), above),
            ((down.0, down.1 - y_gap), below),
          ]
          .into_iter()
          .find(|&(anchor, bbox)| inside(anchor, bbox))
          .unwrap_or(((up.0, up.1 + y_gap), above))
        }
      };
      PlacedLabel {
        text: spec.text.clone(),
        anchor,
        bbox,
        leader: Vec::new(),
      }
    };
    out.push(keep_in_x_range(placed, kx, x_range, spec.side));
  }
  out
}

/// A callout: the leader leaves the curve along its normal (3 pt out), bends
/// 8 pt out — the anchor — and runs 10 pt on to the box.
fn callout_layout(
  spec: &CurveLabelSpec,
  points: &[(f64, f64)],
  p: (f64, f64),
  (w, h): (f64, f64),
  (kx, ky): (f64, f64),
) -> PlacedLabel {
  let side = spec.side.unwrap_or(match spec.at {
    LabelAt::Min => LabelSide::Below,
    LabelAt::Start => LabelSide::Left,
    LabelAt::Finish => LabelSide::Right,
    _ => LabelSide::Above,
  });
  // The curve's direction at the point (over half a point either way).
  let dx = 0.5 / kx;
  let tangent = match (curve_at(points, p.0 - dx), curve_at(points, p.0 + dx)) {
    (Some(a), Some(b)) => ((b.0 - a.0) * kx, (b.1 - a.1) * ky),
    _ => (1.0, 0.0),
  };
  let len = tangent.0.hypot(tangent.1);
  let (tx, ty) = if len > 0.0 {
    (tangent.0 / len, tangent.1 / len)
  } else {
    (1.0, 0.0)
  };
  // The normal on the label's side.
  let (mut nx, mut ny) = (-ty, tx);
  let wanted = match side {
    LabelSide::Above => (0.0, 1.0),
    LabelSide::Below => (0.0, -1.0),
    LabelSide::Left => (-1.0, 0.0),
    LabelSide::Right => (1.0, 0.0),
  };
  if nx * wanted.0 + ny * wanted.1 < 0.0 {
    nx = -nx;
    ny = -ny;
  }
  // The anchor is 8 pt out along the normal; the leader starts 5 pt short
  // of it.
  let anchor = (p.0 + 8.0 * nx / kx, p.1 + 8.0 * ny / ky);
  let start = (-5.0 * nx, -5.0 * ny);
  let run = 10.0;
  let (end, bbox) = match side {
    LabelSide::Above => ((0.0, run), (-w / 2.0, run, w / 2.0, run + h)),
    LabelSide::Below => ((0.0, -run), (-w / 2.0, -run - h, w / 2.0, -run)),
    LabelSide::Left => ((-run, 0.0), (-run - w, -h / 2.0, -run, h / 2.0)),
    LabelSide::Right => ((run, 0.0), (run, -h / 2.0, run + w, h / 2.0)),
  };
  PlacedLabel {
    text: spec.text.clone(),
    anchor,
    bbox,
    leader: vec![start, (0.0, 0.0), end],
  }
}

/// Shift a label box that sticks out past the left or right end of the x
/// range back inside (a label `Before` / `After` the curve stays outside).
fn keep_in_x_range(
  mut label: PlacedLabel,
  kx: f64,
  x_range: (f64, f64),
  side: Option<LabelSide>,
) -> PlacedLabel {
  if matches!(side, Some(LabelSide::Left | LabelSide::Right)) {
    return label;
  }
  let left = label.anchor.0 + label.bbox.0 / kx;
  let right = label.anchor.0 + label.bbox.2 / kx;
  let shift = if left < x_range.0 {
    (x_range.0 - left) * kx
  } else if right > x_range.1 {
    (x_range.1 - right) * kx
  } else {
    0.0
  };
  label.bbox.0 += shift;
  label.bbox.2 += shift;
  label
}

/// The labels of series `series` alone, renumbered as the only series (one
/// panel of a `PlotLayout`).
pub(crate) fn panel_specs(
  specs: &[CurveLabelSpec],
  series: usize,
) -> Vec<CurveLabelSpec> {
  specs
    .iter()
    .filter(|s| s.series == series)
    .map(|s| CurveLabelSpec {
      series: 0,
      ..s.clone()
    })
    .collect()
}

/// Points of room the labels `Before` / `After` their curves need to the
/// left and right of the plotting area.
pub(crate) fn side_room(specs: &[CurveLabelSpec]) -> (f64, f64) {
  let mut room = (0.0f64, 0.0f64);
  for spec in specs {
    let (w, _) = box_size(&spec.text, spec.callout);
    let extra = if spec.callout { 18.0 } else { 0.0 };
    match spec.side {
      Some(LabelSide::Left) => room.0 = room.0.max(w + extra + 24.0),
      Some(LabelSide::Right) => room.1 = room.1.max(w + extra + 24.0),
      _ => {}
    }
  }
  room
}

/// Widen the y range so the labels laid out at a plot area of `plot_pt`
/// (width, height in points) fit inside it, as WL raises the
/// `PlotRangePadding` of a plot with labels above or below its curves.
pub(crate) fn widen_y_range(
  specs: &[CurveLabelSpec],
  curves: &[Vec<(f64, f64)>],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_pt: (f64, f64),
) -> (f64, f64) {
  if specs.is_empty() {
    return y_range;
  }
  let (mut lo, mut hi) = y_range;
  let margin = 0.08 * plot_pt.1;
  for _ in 0..4 {
    let scale = LabelScale {
      kx: plot_pt.0 / (x_range.1 - x_range.0),
      ky: plot_pt.1 / (hi - lo),
      x_range,
      y_range: (lo, hi),
    };
    let placed = layout_curve_labels(specs, curves, scale);
    let (mut top, mut bottom) = (0.0f64, 0.0f64);
    for l in &placed {
      // How far (in points) the box reaches past each end, keeping WL's
      // `Scaled[0.05]` padding beyond it.
      let t = (l.anchor.1 - hi) * scale.ky + l.bbox.3 + margin;
      let b = (lo - l.anchor.1) * scale.ky - l.bbox.1 + margin;
      top = top.max(t);
      bottom = bottom.max(b);
    }
    if top <= 1e-6 && bottom <= 1e-6 {
      break;
    }
    // Grow the range by that much at the current scale.
    hi += top.max(0.0) / scale.ky;
    lo -= bottom.max(0.0) / scale.ky;
  }
  (lo, hi)
}

/// Widen the x and y ranges so the boxes (in points, y up, around their
/// data points) fit inside them at a plot area of `plot_pt` points.
pub(crate) fn widen_for_boxes(
  boxes: &[((f64, f64), (f64, f64, f64, f64))],
  x_range: (f64, f64),
  y_range: (f64, f64),
  plot_pt: (f64, f64),
) -> ((f64, f64), (f64, f64)) {
  if boxes.is_empty() {
    return (x_range, y_range);
  }
  let ((mut x0, mut x1), (mut y0, mut y1)) = (x_range, y_range);
  for _ in 0..4 {
    let kx = plot_pt.0 / (x1 - x0);
    let ky = plot_pt.1 / (y1 - y0);
    if !(kx.is_finite() && ky.is_finite()) {
      break;
    }
    let mut reach = [0.0f64; 4];
    for &((x, y), (l, b, r, t)) in boxes {
      reach[0] = reach[0].max((x0 - x) * kx - l);
      reach[1] = reach[1].max((x - x1) * kx + r);
      reach[2] = reach[2].max((y0 - y) * ky - b);
      reach[3] = reach[3].max((y - y1) * ky + t);
    }
    if reach.iter().all(|&v| v <= 1e-6) {
      break;
    }
    x0 -= reach[0].max(0.0) / kx;
    x1 += reach[1].max(0.0) / kx;
    y0 -= reach[2].max(0.0) / ky;
    y1 += reach[3].max(0.0) / ky;
  }
  ((x0, x1), (y0, y1))
}

/// SVG for laid-out labels. `to_px` maps data coordinates to render pixels
/// and `sf` is render pixels per point.
pub(crate) fn curve_labels_svg(
  labels: &[PlacedLabel],
  to_px: impl Fn((f64, f64)) -> (f64, f64),
  sf: f64,
  text_fill: &str,
) -> String {
  let mut svg = String::new();
  for l in labels {
    let (ax, ay) = to_px(l.anchor);
    let at = |(dx, dy): (f64, f64)| (ax + dx * sf, ay - dy * sf);
    if l.leader.len() == 3 {
      let (s, m, e) = (at(l.leader[0]), at(l.leader[1]), at(l.leader[2]));
      let d = format!(
        "M{:.1},{:.1} Q{:.1},{:.1} {:.1},{:.1}",
        s.0, s.1, m.0, m.1, e.0, e.1
      );
      svg.push_str(&format!(
        "<path d=\"{d}\" fill=\"none\" stroke=\"#FFFFFF\" \
         stroke-opacity=\"0.667\" stroke-width=\"{:.1}\" \
         stroke-linejoin=\"round\"/>\n\
         <path d=\"{d}\" fill=\"none\" stroke=\"#AAAAAA\" \
         stroke-width=\"{:.1}\"/>\n",
        4.0 * sf,
        1.25 * sf,
      ));
    }
    let (l0, t0) = at((l.bbox.0, l.bbox.3));
    let (r0, b0) = at((l.bbox.2, l.bbox.1));
    svg.push_str(&format!(
      "<rect x=\"{l0:.1}\" y=\"{t0:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
       fill=\"#FFFFFF\" fill-opacity=\"0.667\"/>\n",
      r0 - l0,
      b0 - t0,
    ));
    let (cx, cy) = (f64::midpoint(l0, r0), f64::midpoint(t0, b0));
    svg.push_str(&format!(
      "<text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"sans-serif\" \
       font-size=\"{:.0}\" fill=\"{text_fill}\" text-anchor=\"middle\" \
       dominant-baseline=\"central\">{}</text>\n",
      FONT * sf,
      crate::functions::graphics::box_string_to_svg(&l.text),
    ));
  }
  svg
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn positions_parse() {
    let id = |s: &str| Expr::Identifier(s.into());
    assert_eq!(
      parse_label_position(&id("Above")),
      Some((LabelAt::Max, Some(LabelSide::Above)))
    );
    assert_eq!(parse_label_position(&id("Automatic")), None);
    assert_eq!(
      parse_label_position(&Expr::Integer(3)),
      Some((LabelAt::X(3.0), None))
    );
  }
}
