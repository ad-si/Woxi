//! Static highlights of a curve: `Highlighted[f, Placed[effect, pos]]` with
//! the effects `"Ball"`, `"Dropline"`, `"XSlice"` and `"YSlice"`.
//!
//! WL 15 draws what the effect shows on hover, fixed at the given place:
//! the curve point enlarged on a white halo, the gray drop lines or slice
//! line, and a speech-bubble callout with the value, in a pale tint of the
//! curve's colour, its edge in the colour itself.

use crate::syntax::Expr;

/// A highlight effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HighlightKind {
  Ball,
  Dropline,
  XSlice,
  YSlice,
}

/// One `Highlighted[f, Placed[effect, pos]]`, placed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlacedHighlight {
  pub kind: HighlightKind,
  /// The x positions (y for a `"YSlice"`) it is drawn at.
  pub at: Vec<f64>,
  /// The curve it belongs to; `None` for all of them (a `Highlighted`
  /// around the whole list of curves).
  pub series: Option<usize>,
}

/// Parse a `Highlighted` spec `Placed[effect, pos]`: `pos` a coordinate or
/// a list `{{x1}, {x2}, …}` of them.
pub(crate) fn parse_placed_highlight(
  spec: &Expr,
  series: Option<usize>,
) -> Option<PlacedHighlight> {
  let Expr::FunctionCall { name, args } = spec else {
    return None;
  };
  if name != "Placed" || args.len() != 2 {
    return None;
  }
  let kind = match &args[0] {
    Expr::String(s) => match s.as_str() {
      "Ball" => HighlightKind::Ball,
      "Dropline" => HighlightKind::Dropline,
      "XSlice" => HighlightKind::XSlice,
      "YSlice" => HighlightKind::YSlice,
      _ => return None,
    },
    _ => return None,
  };
  let pos = crate::evaluator::evaluate_expr_to_expr(&args[1])
    .unwrap_or_else(|_| args[1].clone());
  fn coords(e: &Expr, out: &mut Vec<f64>) {
    match e {
      Expr::List(items) => items.iter().for_each(|i| coords(i, out)),
      other => out.extend(crate::functions::graphics::expr_to_f64(other)),
    }
  }
  let mut at = Vec::new();
  coords(&pos, &mut at);
  (!at.is_empty()).then_some(PlacedHighlight { kind, at, series })
}

/// The curve's value at `x` (`y` for `by_y`: the first crossing of the
/// level `y`), interpolated between its samples.
fn curve_point(
  points: &[(f64, f64)],
  v: f64,
  by_y: bool,
) -> Option<(f64, f64)> {
  let key = |p: &(f64, f64)| if by_y { p.1 } else { p.0 };
  points
    .windows(2)
    .filter(|w| w.iter().all(|p| p.0.is_finite() && p.1.is_finite()))
    .find(|w| {
      let (a, b) = (key(&w[0]), key(&w[1]));
      (a <= v && v <= b) || (b <= v && v <= a)
    })
    .map(|w| {
      let (a, b) = (key(&w[0]), key(&w[1]));
      let t = if (b - a).abs() < f64::EPSILON {
        0.0
      } else {
        (v - a) / (b - a)
      };
      (
        w[0].0 + t * (w[1].0 - w[0].0),
        w[0].1 + t * (w[1].1 - w[0].1),
      )
    })
}

/// A value as the highlight callout prints it: rounded to
/// `3 - ⌈log10 span⌉` decimals (at least one) of its axis span, trailing
/// zeros dropped.
fn format_value(v: f64, span: f64) -> String {
  let decimals = (3 - span.abs().log10().ceil() as i32).max(1) as usize;
  // Halves round away from zero, as `Round` does on the printed digits.
  let scale = 10f64.powi(decimals as i32);
  let v = (v * scale).round() / scale;
  let s = format!("{v:.decimals$}");
  let s = s.trim_end_matches('0').trim_end_matches('.');
  if s == "-0" { "0".into() } else { s.into() }
}

/// SVG for the highlights. `curves` are the drawn curves, `colors` their
/// colours, `to_px` maps data coordinates to render pixels, `sf` is render
/// pixels per point, `origin` where the axes cross and `ranges` the
/// displayed x and y ranges.
pub(crate) fn highlights_svg(
  highlights: &[PlacedHighlight],
  curves: &[Vec<(f64, f64)>],
  colors: impl Fn(usize) -> (u8, u8, u8),
  to_px: impl Fn((f64, f64)) -> (f64, f64),
  sf: f64,
  origin: (f64, f64),
  (x_range, y_range): ((f64, f64), (f64, f64)),
  plot_area: (f64, f64, f64, f64),
) -> String {
  let mut lines = String::new();
  let mut balls = String::new();
  let mut bubbles = String::new();
  let gray = "#B3B3B3";
  let (plot_x0, plot_top, plot_w, plot_h) = plot_area;
  for h in highlights {
    let series: Vec<usize> = match h.series {
      Some(s) => vec![s],
      None => (0..curves.len()).collect(),
    };
    let by_y = h.kind == HighlightKind::YSlice;
    for &v in &h.at {
      // Each curve's point, lowest (leftmost for a y slice) first: that
      // one's callout goes to the left of a slice.
      let mut hits: Vec<(usize, (f64, f64))> = series
        .iter()
        .filter_map(|&s| Some((s, curve_point(curves.get(s)?, v, by_y)?)))
        .collect();
      hits.sort_by(|a, b| {
        let key = |p: &(f64, f64)| if by_y { p.0 } else { p.1 };
        key(&a.1)
          .partial_cmp(&key(&b.1))
          .unwrap_or(std::cmp::Ordering::Equal)
      });
      match h.kind {
        HighlightKind::XSlice => {
          let (px, _) = to_px((v, 0.0));
          lines.push_str(&format!(
            "<line x1=\"{px:.1}\" y1=\"{plot_top:.1}\" x2=\"{px:.1}\" \
             y2=\"{:.1}\" stroke=\"{gray}\" stroke-width=\"{sf:.1}\"/>\n",
            plot_top + plot_h,
          ));
        }
        HighlightKind::YSlice => {
          let (_, py) = to_px((0.0, v));
          lines.push_str(&format!(
            "<line x1=\"{plot_x0:.1}\" y1=\"{py:.1}\" x2=\"{:.1}\" \
             y2=\"{py:.1}\" stroke=\"{gray}\" stroke-width=\"{sf:.1}\"/>\n",
            plot_x0 + plot_w,
          ));
        }
        HighlightKind::Dropline => {
          for &(_, p) in &hits {
            let (px, py) = to_px(p);
            let (ox, _) = to_px((origin.0, p.1));
            let (_, oy) = to_px((p.0, origin.1));
            lines.push_str(&format!(
              "<line x1=\"{px:.1}\" y1=\"{py:.1}\" x2=\"{px:.1}\" \
               y2=\"{oy:.1}\" stroke=\"{gray}\" stroke-width=\"{sf:.1}\"/>\n\
               <line x1=\"{px:.1}\" y1=\"{py:.1}\" x2=\"{ox:.1}\" \
               y2=\"{py:.1}\" stroke=\"{gray}\" stroke-width=\"{sf:.1}\"/>\n",
            ));
          }
        }
        HighlightKind::Ball => {}
      }
      let slice =
        matches!(h.kind, HighlightKind::XSlice | HighlightKind::YSlice);
      for (rank, &(s, p)) in hits.iter().enumerate() {
        let (r, g, b) = colors(s);
        let (px, py) = to_px(p);
        balls.push_str(&format!(
          "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"{:.1}\" fill=\"#FFFFFF\"/>\n\
           <circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"{:.1}\" \
           fill=\"rgb({r},{g},{b})\"/>\n",
          4.5 * sf,
          3.0 * sf,
        ));
        // The value shown: a slice's coordinate across it; for a ball the
        // x value, which WL 15 sets rotated by the y value in radians
        // (`Rotate[x, y, 0]`).
        let x_text = format_value(p.0, x_range.1 - x_range.0);
        let y_text = format_value(p.1, y_range.1 - y_range.0);
        let (text, rotated_by) = match h.kind {
          HighlightKind::XSlice => (y_text, None),
          HighlightKind::YSlice => (x_text, None),
          _ => (x_text, Some(y_text)),
        };
        let left = slice && hits.len() > 1 && rank == 0;
        bubbles.push_str(&bubble_svg(
          (px, py),
          &text,
          rotated_by.as_deref(),
          left,
          (r, g, b),
          sf,
        ));
      }
    }
  }
  format!("{lines}{balls}{bubbles}")
}

/// A speech-bubble callout for the point `(px, py)` (render pixels): its
/// tail starts 6 pt out at 45°, the box 8 pt out, 18 pt tall and 8.5 pt
/// wider than the text, to the right of the point (or mirrored to the
/// left).
fn bubble_svg(
  (px, py): (f64, f64),
  text: &str,
  rotated_by: Option<&str>,
  left: bool,
  (r, g, b): (u8, u8, u8),
  sf: f64,
) -> String {
  let font = crate::functions::plot_padding::LABEL_FONT;
  let width = |t: &str| crate::functions::plot_padding::text_width(t, font);
  // WL sizes the box of a rotated `Rotate[x, y, 0]` label as if it held
  // both numbers.
  let w = width(text)
    + match rotated_by {
      Some(y) => width(y) + 11.0,
      None => 8.5,
    };
  let d = std::f64::consts::FRAC_1_SQRT_2 * 3.0;
  let a = 4.0 * std::f64::consts::SQRT_2;
  let (tip, top, mid) = (2.0 * d, a + 18.0, a + 9.0);
  let sx = if left { -1.0 } else { 1.0 };
  // Outline relative to the point, in points, y up.
  let outline = [
    (tip, tip),
    (a + 6.0 + d, a + d),
    (a + d + w, a + d),
    (a + d + w, top + d),
    (a + d, top + d),
    (a + d, mid + d),
    (a + d, a + 6.0 + d),
  ];
  let pts: Vec<String> = outline
    .iter()
    .map(|&(x, y)| format!("{:.1},{:.1}", px + sx * x * sf, py - y * sf))
    .collect();
  let fill = |c: u8| (c as f64 * 0.15 + 255.0 * 0.85).round() as u8;
  let (cx, cy) = (px + sx * (a + d + w / 2.0) * sf, py - (mid + d) * sf);
  let angle = rotated_by
    .and_then(|y| y.parse::<f64>().ok())
    .unwrap_or(0.0);
  let rotate = if angle == 0.0 {
    String::new()
  } else {
    format!(
      " transform=\"rotate({:.2} {cx:.1} {cy:.1})\"",
      -angle.to_degrees()
    )
  };
  format!(
    "<polygon points=\"{}\" fill=\"rgb({},{},{})\" stroke=\"rgb({r},{g},{b})\" \
     stroke-width=\"{sf:.1}\"/>\n\
     <text x=\"{cx:.1}\" y=\"{cy:.1}\" font-family=\"sans-serif\" \
     font-size=\"{:.0}\" text-anchor=\"middle\" dominant-baseline=\"central\"\
     {rotate}>{text}</text>\n",
    pts.join(" "),
    fill(r),
    fill(g),
    fill(b),
    font * sf,
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn values_round_to_the_span() {
    assert_eq!(format_value(0.454_542_8, 1.22), "0.45");
    assert_eq!(format_value(7.654_136, 9.46), "7.65");
    assert_eq!(format_value(2.0, 10.0), "2");
    assert_eq!(format_value(123.25, 200.0), "123.3");
    assert_eq!(format_value(0.123_25, 0.2), "0.123");
  }
}
