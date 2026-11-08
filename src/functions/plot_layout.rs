//! `PlotLayout -> "Row" | "Column" | {"Row", n} | {"Column", UpTo[n]}`:
//! one framed panel per curve instead of all curves in one plot.
//!
//! Shared by `Plot` and the list plots. Every panel shows the same plot
//! range; frame ticks and their labels appear only along the outer edges
//! (the left column and the bottom row), and the panels
//! sit 5 pt apart, as in WL 15. Each panel is drawn by the caller's own
//! renderer as a bare plot (no axes, no frame) over exactly the panel's
//! rectangle; the frames and ticks are drawn here.

use crate::InterpreterError;
use crate::functions::plot::{
  DEFAULT_ASPECT_RATIO, PlotOptions, collapse_style_for_single_series,
  format_tick_with_step, parse_image_size, plot_theme,
};
use crate::functions::plot_axes::{AXIS_TICK_FONT, AxisScale, AxisTicks};
use crate::syntax::Expr;

/// Space between neighboring panels, in points.
const PANEL_GAP: f64 = 5.0;
/// Length of a frame tick mark (drawn inwards), in points.
const TICK_LEN: f64 = 2.9;

/// Which way the panels run, and how many rows (`Row`) or columns
/// (`Column`) they fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlotLayoutSpec {
  /// `true` for `"Column"`: the panels fill columns top to bottom.
  column: bool,
  /// `{"Row", n}` / `{"Column", n}`: exactly `n` rows / columns.
  /// `UpTo[n]`: at most `n`, balanced. `None`: a single row / column.
  count: Option<(usize, bool)>,
}

/// Read a `PlotLayout` value. `None` for the overlaid default (and for
/// layouts that are not panel arrangements, such as `"Stacked"`).
pub(crate) fn parse_plot_layout(value: &Expr) -> Option<PlotLayoutSpec> {
  let kind = |e: &Expr| match e {
    Expr::String(s) if s == "Row" => Some(false),
    Expr::String(s) if s == "Column" => Some(true),
    _ => None,
  };
  match value {
    Expr::List(items) if items.len() == 2 => {
      let column = kind(&items[0])?;
      let count = match &items[1] {
        Expr::Integer(n) if *n > 0 => (*n as usize, false),
        Expr::FunctionCall { name, args }
          if name == "UpTo" && args.len() == 1 =>
        {
          match &args[0] {
            Expr::Integer(n) if *n > 0 => (*n as usize, true),
            _ => return None,
          }
        }
        _ => return None,
      };
      Some(PlotLayoutSpec {
        column,
        count: Some(count),
      })
    }
    other => kind(other).map(|column| PlotLayoutSpec {
      column,
      count: None,
    }),
  }
}

/// `k` panels split into `m` groups as evenly as possible, the first groups
/// taking the extra ones.
fn distribute(k: usize, m: usize) -> Vec<usize> {
  let m = m.clamp(1, k.max(1));
  (0..m).map(|i| k / m + usize::from(i < k % m)).collect()
}

/// The `(row, column)` of every panel, plus the grid's row and column count.
fn panel_cells(
  spec: PlotLayoutSpec,
  k: usize,
) -> (Vec<(usize, usize)>, usize, usize) {
  // Number of groups (rows for "Row", columns for "Column").
  let groups = match spec.count {
    None => 1,
    Some((n, false)) => n.min(k),
    Some((n, true)) => {
      let per = k.div_ceil(n.max(1));
      k.div_ceil(per.max(1))
    }
  };
  let sizes = distribute(k, groups);
  let mut cells = Vec::with_capacity(k);
  for (g, &size) in sizes.iter().enumerate() {
    for i in 0..size {
      cells.push(if spec.column { (i, g) } else { (g, i) });
    }
  }
  let along = sizes.iter().copied().max().unwrap_or(1);
  let (rows, cols) = if spec.column {
    (along, sizes.len())
  } else {
    (sizes.len(), along)
  };
  (cells, rows, cols)
}

/// The box the whole layout must fit in: its width and, when bounded, its
/// height. By default a layout with several columns may be 600 wide (360
/// for a single column) and one plot height (223) per row; a named size
/// keeps its width and caps the height at 1.2 times that, as WL does for
/// any graphic; `{w, h}` bounds both.
fn layout_box(
  image_size: Option<&Expr>,
  rows: usize,
  cols: usize,
) -> (f64, f64) {
  let default = (if cols > 1 { 600.0 } else { 360.0 }, 223.0 * rows as f64);
  let Some(value) = image_size else {
    return default;
  };
  match value {
    Expr::Identifier(name) => match parse_image_size(value, 360, 223) {
      Some((w, _, _)) if name != "Automatic" => (w as f64, 1.2 * w as f64),
      _ => default,
    },
    Expr::List(items)
      if items.len() == 2
        && !items.iter().any(|e| {
          matches!(e, Expr::List(_))
            || matches!(e, Expr::Identifier(n) if n == "Automatic")
        }) =>
    {
      match parse_image_size(value, 360, 223) {
        Some((w, h, _)) => (w as f64, h as f64),
        None => default,
      }
    }
    _ => match parse_image_size(value, 360, 223) {
      Some((w, _, _)) => (w as f64, f64::INFINITY),
      None => default,
    },
  }
}

/// The labeled ticks of a panel axis over `range`: the smallest step of
/// 1, 2, 2.5 or 5 × 10^k that leaves at most five ticks in the range (WL's
/// panel tick function labels 0, 2.5, …, 10 where a lone plot has 0, 2, …,
/// 10). Explicit, logarithmic and date ticks keep the plot's usual ones.
fn panel_ticks(
  scale: AxisScale,
  date: bool,
  explicit: Option<&Vec<(f64, String)>>,
) -> Vec<(f64, String)> {
  let (lo, hi) = (scale.min, scale.max);
  if explicit.is_some()
    || date
    || scale.log
    || !(lo.is_finite() && hi.is_finite())
    || hi <= lo
  {
    return AxisTicks::new(scale, date, explicit).majors;
  }
  let span = hi - lo;
  let first_exp = (span / 5.0).log10().floor() as i32 - 1;
  for exp in (first_exp..).take(8) {
    let decade = 10f64.powi(exp);
    for m in [1.0, 2.0, 2.5, 5.0] {
      let step = m * decade;
      let first = (lo / step - 1e-9).ceil() as i64;
      let last = (hi / step + 1e-9).floor() as i64;
      if last - first < 5 {
        return (first..=last)
          .map(|i| {
            let v = i as f64 * step;
            let v = if v.abs() < step * 1e-6 { 0.0 } else { v };
            (
              v,
              crate::functions::graphics::svg_escape(&format_tick_with_step(
                v, step,
              )),
            )
          })
          .collect();
      }
    }
  }
  AxisTicks::new(scale, date, explicit).majors
}

/// Estimated width of a tick label's markup at the tick font.
fn label_width(markup: &str) -> f64 {
  let visible = crate::functions::graphics::box_string_visible_len(
    &markup.replace("&#8722;", "-").replace("&amp;", "&"),
  );
  visible as f64 * AXIS_TICK_FONT * 0.6
}

/// Render `n` panels laid out by `spec`. `render_panel(i, opts)` draws
/// panel `i` (the `i`-th curve or dataset) with the given options, which
/// already describe a bare plot of the panel's size over `x_range` ×
/// `y_range`; the caller only adds what belongs to that one dataset.
pub(crate) fn render_plot_layout(
  spec: PlotLayoutSpec,
  n: usize,
  base: &PlotOptions,
  x_range: (f64, f64),
  y_range: (f64, f64),
  mut render_panel: impl FnMut(
    usize,
    PlotOptions,
  ) -> Result<String, InterpreterError>,
) -> Result<String, InterpreterError> {
  let (cells, rows, cols) = panel_cells(spec, n);
  let x_scale = AxisScale {
    min: x_range.0,
    max: x_range.1,
    log: base.log_x,
  };
  let y_scale = AxisScale {
    min: y_range.0,
    max: y_range.1,
    log: base.log_y,
  };
  let x_ticks = panel_ticks(x_scale, base.date_axis, base.ticks_x.as_ref());
  let y_ticks = panel_ticks(y_scale, false, base.ticks_y.as_ref());
  let show_ticks = base.ticks || base.frame_ticks;

  // Gutters around the grid: tick labels left and below, a little room
  // on the other two sides for the frame and the overhang of the last x
  // label. WL keeps a wider top margin over a single row of panels.
  let left = if show_ticks {
    y_ticks
      .iter()
      .map(|(_, l)| label_width(l))
      .fold(0.0, f64::max)
      + 6.0
  } else {
    2.0
  };
  let bottom = if show_ticks {
    AXIS_TICK_FONT * 1.5
  } else {
    2.0
  };
  let right = 7.8;
  let top = if cols == 1 {
    2.0
  } else {
    (11.2 - PANEL_GAP * (rows as f64 - 1.0)).max(2.0)
  };
  let ar = base.aspect_ratio.unwrap_or(DEFAULT_ASPECT_RATIO);
  let (max_w, max_h) = layout_box(base.image_size_value.as_ref(), rows, cols);
  let gaps_w = PANEL_GAP * (cols as f64 - 1.0);
  let gaps_h = PANEL_GAP * (rows as f64 - 1.0);
  let mut pw = ((max_w - left - right - gaps_w) / cols as f64).max(10.0);
  if top + bottom + gaps_h + rows as f64 * pw * ar > max_h {
    pw = ((max_h - top - bottom - gaps_h) / (rows as f64 * ar)).max(10.0);
  }
  let ph = pw * ar;
  let width = left + right + gaps_w + cols as f64 * pw;
  let height = top + bottom + gaps_h + rows as f64 * ph;

  // The lowest y label of the left column, which an x label at the left
  // end of the bottom-left panel must not run into.
  let lowest_y_label = y_ticks
    .iter()
    .map(|(v, _)| y_scale.frac(*v) * ph)
    .fold(f64::INFINITY, f64::min);

  let (_, theme_gray, _, label_fill, _) = plot_theme();
  let frame_color = format!(
    "#{:02X}{:02X}{:02X}",
    theme_gray.0, theme_gray.1, theme_gray.2
  );
  let panel_style = collapse_style_for_single_series(&base.plot_style);
  let mut body = String::new();
  let mut overlay = String::new();
  for (idx, &(r, c)) in cells.iter().enumerate() {
    let x0 = left + c as f64 * (pw + PANEL_GAP);
    let y0 = top + r as f64 * (ph + PANEL_GAP);

    let mut opts = base.clone();
    opts.plot_layout = None;
    opts.plot_style.clone_from(&panel_style);
    opts.frame = false;
    opts.axes = (false, false);
    opts.ticks = false;
    opts.frame_ticks = false;
    opts.grid_lines_x = base.grid_lines_x;
    opts.plot_label = None;
    opts.axes_label = None;
    opts.frame_label_bottom = None;
    opts.frame_label_left = None;
    opts.frame_label_top = None;
    opts.frame_label_right = None;
    opts.frame_label_bottom_graphic = None;
    opts.frame_label_top_graphic = None;
    opts.plot_legends = Vec::new();
    opts.image_padding = Some([0.0; 4]);
    opts.aspect_ratio = Some(ar);
    opts.aspect_automatic = false;
    opts.aspect_full = false;
    opts.image_size_explicit = (true, false);
    opts.full_width = false;
    opts.svg_width = pw.round().max(1.0) as u32;
    opts.svg_height = ph.round().max(1.0) as u32;
    let svg = render_panel(idx, opts)?;
    body.push_str(&crate::functions::graphics::embed_svg_in_rect(
      &svg, x0, y0, pw, ph,
    ));

    overlay.push_str(&format!(
      "<rect x=\"{x0:.2}\" y=\"{y0:.2}\" width=\"{pw:.2}\" height=\"{ph:.2}\" fill=\"none\" stroke=\"{frame_color}\" stroke-width=\"0.5\"/>\n"
    ));
    if !show_ticks {
      continue;
    }
    if c == 0 {
      for (v, label) in &y_ticks {
        let py = y0 + ph - y_scale.frac(*v) * ph;
        overlay.push_str(&format!(
          "<line x1=\"{x0:.2}\" y1=\"{py:.2}\" x2=\"{:.2}\" y2=\"{py:.2}\" stroke=\"{frame_color}\" stroke-width=\"0.5\"/>\n",
          x0 + TICK_LEN
        ));
        overlay.push_str(&format!(
          "<text x=\"{:.2}\" y=\"{py:.2}\" text-anchor=\"end\" dominant-baseline=\"central\" font-family=\"sans-serif\" font-size=\"{AXIS_TICK_FONT}\" fill=\"{label_fill}\">{label}</text>\n",
          x0 - 3.0
        ));
      }
    }
    // Only the bottom row carries x ticks (a shorter column leaves its
    // lowest panel without any).
    if r + 1 == rows {
      let by = y0 + ph;
      for (v, label) in &x_ticks {
        let px = x0 + x_scale.frac(*v) * pw;
        overlay.push_str(&format!(
          "<line x1=\"{px:.2}\" y1=\"{by:.2}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"{frame_color}\" stroke-width=\"0.5\"/>\n",
          by - TICK_LEN
        ));
        // A label that would reach past the left end of the panel is
        // dropped when it runs into something there: the previous panel's
        // last label (WL's `"Interior"` tick functions) or, in the first
        // column, the lowest y label.
        let overhangs = px - x0 < label_width(label) / 2.0;
        if overhangs && (c > 0 || lowest_y_label < AXIS_TICK_FONT * 0.4) {
          continue;
        }
        overlay.push_str(&format!(
          "<text x=\"{px:.2}\" y=\"{:.2}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{AXIS_TICK_FONT}\" fill=\"{label_fill}\">{label}</text>\n",
          by + AXIS_TICK_FONT * 1.1
        ));
      }
    }
  }

  let (w, h) = ((width - 1e-6).ceil(), (height - 1e-6).ceil());
  let mut svg = format!(
    "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
  );
  if let Some(bg) = &base.background {
    svg.push_str(&format!(
      "<rect width=\"100%\" height=\"100%\" fill=\"rgb({},{},{})\"/>\n",
      bg.0, bg.1, bg.2
    ));
  }
  svg.push_str(&body);
  svg.push_str(&overlay);
  svg.push_str("</svg>");
  Ok(svg)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn spec(column: bool, count: Option<(usize, bool)>) -> PlotLayoutSpec {
    PlotLayoutSpec { column, count }
  }

  #[test]
  fn row_and_column_cells() {
    let (cells, rows, cols) = panel_cells(spec(false, None), 3);
    assert_eq!((rows, cols), (1, 3));
    assert_eq!(cells, vec![(0, 0), (0, 1), (0, 2)]);
    let (cells, rows, cols) = panel_cells(spec(true, None), 2);
    assert_eq!((rows, cols), (2, 1));
    assert_eq!(cells, vec![(0, 0), (1, 0)]);
  }

  #[test]
  fn counted_layouts_distribute_the_extra_panels_first() {
    // {"Column", 4} with 6 panels: columns of 2, 2, 1, 1, filled top down.
    let (cells, rows, cols) = panel_cells(spec(true, Some((4, false))), 6);
    assert_eq!((rows, cols), (2, 4));
    assert_eq!(cells, vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (0, 3)]);
    // {"Row", 2}: two rows of three.
    let (_, rows, cols) = panel_cells(spec(false, Some((2, false))), 6);
    assert_eq!((rows, cols), (2, 3));
    // {"Column", UpTo[4]}: balanced, three columns of two.
    let (_, rows, cols) = panel_cells(spec(true, Some((4, true))), 6);
    assert_eq!((rows, cols), (2, 3));
    // {"Row", UpTo[4]}: three rows of two.
    let (_, rows, cols) = panel_cells(spec(false, Some((4, true))), 6);
    assert_eq!((rows, cols), (3, 2));
  }
}
