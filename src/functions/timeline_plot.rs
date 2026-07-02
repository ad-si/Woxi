use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::interval_ast::is_interval;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, PLOT_COLORS, RESOLUTION_SCALE,
  format_date_tick, generate_date_ticks, parse_image_size,
};
use crate::syntax::Expr;

/// Default display width of a TimelinePlot in pixels.
const DEFAULT_TLP_WIDTH: u32 = 360;
/// Height allotted to each stacked event row (display pixels).
const ROW_HEIGHT: u32 = 26;
/// Padding above the first row.
const PADDING_TOP: u32 = 15;
/// Padding below the axis (space for the date tick labels).
const PADDING_BOTTOM: u32 = 30;
/// Extra vertical space between the lowest row and the axis line.
const AXIS_GAP: u32 = 12;

/// A single timeline event: either an instant (a point) or a span (interval).
struct TimelineEvent {
  /// Start of the event in AbsoluteTime seconds (since 1900-01-01).
  start: f64,
  /// End of the event for intervals; `None` for point events.
  end: Option<f64>,
  /// Text label rendered next to the event marker.
  label: String,
}

/// `TimelinePlot[{date, ...}]` — render events on a horizontal timeline.
pub fn timeline_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut svg_width = DEFAULT_TLP_WIDTH;
  let mut full_width = false;

  // Parse options (Rules at the end), e.g. ImageSize.
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ImageSize"
      && let Some((w, _h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    {
      svg_width = w;
      full_width = fw;
    }
  }

  let events = parse_events(&args[0]);
  if events.is_empty() {
    // Mirror Wolfram: return unevaluated for data we cannot interpret.
    return Ok(Expr::FunctionCall {
      name: "TimelinePlot".to_string(),
      args: args.to_vec().into(),
    });
  }

  let svg = render_timeline_svg(&events, svg_width, full_width);
  Ok(crate::graphics_result(svg))
}

/// Parse the first argument into a flat list of timeline events.
fn parse_events(arg: &Expr) -> Vec<TimelineEvent> {
  let evaluated = evaluate_expr_to_expr(arg).unwrap_or_else(|_| arg.clone());

  let mut events = Vec::new();
  match &evaluated {
    // Association: each key -> value pair is a labelled event.
    Expr::Association(pairs) => {
      for (key, val) in pairs {
        if let Some(mut ev) = event_from_date(val) {
          ev.label = expr_to_label(key);
          events.push(ev);
        }
      }
    }
    Expr::List(items) => {
      for item in items {
        if let Some(ev) = parse_one_event(item) {
          events.push(ev);
        }
      }
    }
    // A single date given directly.
    other => {
      if let Some(ev) = parse_one_event(other) {
        events.push(ev);
      }
    }
  }
  events
}

/// Parse a single event specification (a list element).
fn parse_one_event(item: &Expr) -> Option<TimelineEvent> {
  let evaluated = evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone());

  match &evaluated {
    // Labeled[date, label] — explicit label wrapper.
    Expr::FunctionCall { name, args }
      if name == "Labeled" && args.len() == 2 =>
    {
      let mut ev = event_from_date(&args[0])?;
      ev.label = expr_to_label(&args[1]);
      Some(ev)
    }
    // Style[date, ...] — ignore styling, keep the underlying date.
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      event_from_date(&args[0])
    }
    // Tooltip[date, label] — treat the tooltip text as the label.
    Expr::FunctionCall { name, args }
      if name == "Tooltip" && args.len() == 2 =>
    {
      let mut ev = event_from_date(&args[0])?;
      ev.label = expr_to_label(&args[1]);
      Some(ev)
    }
    // label -> date (an association entry written as a Rule).
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let mut ev = event_from_date(replacement)?;
      ev.label = expr_to_label(pattern);
      Some(ev)
    }
    _ => event_from_date(&evaluated),
  }
}

/// Build an event from a date-like expression, defaulting the label to the
/// formatted date (or date range for intervals).
fn event_from_date(expr: &Expr) -> Option<TimelineEvent> {
  let evaluated = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());

  // DateInterval[{d1, d2}] — an explicit time span.
  if let Expr::FunctionCall { name, args } = &evaluated
    && name == "DateInterval"
    && args.len() == 1
    && let Expr::List(pair) = &args[0]
    && pair.len() == 2
    && let (Some(lo), Some(hi)) =
      (date_to_absolute_time(&pair[0]), date_to_absolute_time(&pair[1]))
  {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    return Some(TimelineEvent {
      start: lo,
      end: Some(hi),
      label: format!("{} – {}", format_date_tick(lo), format_date_tick(hi)),
    });
  }

  // Interval[{d1, d2}] — also treated as a time span.
  if let Some(spans) = is_interval(&evaluated)
    && let Some((lo_e, hi_e)) = spans.first()
    && let (Some(lo), Some(hi)) =
      (date_to_absolute_time(lo_e), date_to_absolute_time(hi_e))
  {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    return Some(TimelineEvent {
      start: lo,
      end: Some(hi),
      label: format!("{} – {}", format_date_tick(lo), format_date_tick(hi)),
    });
  }

  // A single instant.
  let t = date_to_absolute_time(&evaluated)?;
  Some(TimelineEvent {
    start: t,
    end: None,
    label: format_date_tick(t),
  })
}

/// Convert a date expression to AbsoluteTime seconds (since 1900-01-01).
/// Handles raw numbers (already AbsoluteTime), date lists `{y,m,d,...}` and
/// `DateObject[...]`.
fn date_to_absolute_time(expr: &Expr) -> Option<f64> {
  let evaluated = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());

  if let Some(t) = try_eval_to_f64(&evaluated) {
    return Some(t);
  }

  let components =
    crate::functions::datetime_ast::extract_date_components(&evaluated)?;
  if components.is_empty() {
    return None;
  }
  let year = components[0] as i64;
  let month = components.get(1).map(|v| *v as i64).unwrap_or(1);
  let day = components.get(2).map(|v| *v as i64).unwrap_or(1);
  let hour = components.get(3).map(|v| *v as i64).unwrap_or(0);
  let minute = components.get(4).map(|v| *v as i64).unwrap_or(0);
  let second = components.get(5).copied().unwrap_or(0.0);
  Some(crate::functions::datetime_ast::date_to_absolute_seconds(
    year, month, day, hour, minute, second,
  ))
}

/// Render a label expression to display text (strings without their quotes).
fn expr_to_label(expr: &Expr) -> String {
  let evaluated = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  match &evaluated {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_output(other),
  }
}

/// Greedily pack events into as few rows as possible so their markers and
/// labels do not overlap horizontally. Returns the row index per event.
fn pack_rows(
  events: &[TimelineEvent],
  x_to_px: impl Fn(f64) -> f64,
  label_px_width: impl Fn(&str) -> f64,
  marker_gap: f64,
  row_gap: f64,
) -> Vec<usize> {
  // Process events left-to-right by start position for tidy packing.
  let mut order: Vec<usize> = (0..events.len()).collect();
  order.sort_by(|&a, &b| {
    events[a]
      .start
      .partial_cmp(&events[b].start)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let mut row_end: Vec<f64> = Vec::new();
  let mut assignment = vec![0usize; events.len()];

  for &idx in &order {
    let ev = &events[idx];
    let left = x_to_px(ev.start);
    let marker_right = match ev.end {
      Some(end) => x_to_px(end),
      None => left,
    };
    let occupied_end =
      marker_right + marker_gap + label_px_width(&ev.label) + row_gap;

    // Find the first row whose content ends before this event starts.
    let mut placed = None;
    for (row, end) in row_end.iter_mut().enumerate() {
      if left >= *end {
        *end = occupied_end;
        placed = Some(row);
        break;
      }
    }
    let row = placed.unwrap_or_else(|| {
      row_end.push(occupied_end);
      row_end.len() - 1
    });
    assignment[idx] = row;
  }

  assignment
}

/// Estimate the pixel width of a text label at the given font size.
fn est_text_width(label: &str, font_size: f64) -> f64 {
  label.chars().count() as f64 * font_size * 0.6
}

/// Render the complete TimelinePlot SVG.
fn render_timeline_svg(
  events: &[TimelineEvent],
  svg_width: u32,
  full_width: bool,
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;

  let (bg_color, axis_color, label_fill, event_color) = theme_colors();

  let font_size = 13.0 * sf;
  let marker_gap = 6.0 * sf;
  let row_gap = 10.0 * sf;

  // Compute the x range across all events, with a little padding.
  let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
  for ev in events {
    x_min = x_min.min(ev.start);
    x_max = x_max.max(ev.end.unwrap_or(ev.start));
  }
  if !x_min.is_finite() || !x_max.is_finite() {
    x_min = 0.0;
    x_max = 86400.0;
  }
  let range = x_max - x_min;
  let pad = if range.abs() < f64::EPSILON {
    // All events at the same instant: default to a one-year window.
    365.25 * 86400.0
  } else {
    range * 0.08
  };
  x_min -= pad;
  x_max += pad;

  // Reserve right margin large enough for the longest label so nothing clips.
  let longest_label = events
    .iter()
    .map(|e| est_text_width(&e.label, font_size))
    .fold(0.0_f64, f64::max);
  let margin_left = 10.0 * sf;
  let margin_right = (longest_label + marker_gap + 10.0 * sf)
    .min(render_width as f64 * 0.5)
    .max(20.0 * sf);
  let plot_width = (render_width as f64 - margin_left - margin_right).max(1.0);

  let x_to_px =
    |x: f64| -> f64 { margin_left + (x - x_min) / (x_max - x_min) * plot_width };

  // Pack events into rows.
  let rows = pack_rows(
    events,
    |x| x_to_px(x),
    |lbl| est_text_width(lbl, font_size),
    marker_gap,
    row_gap,
  );
  let num_rows = rows.iter().copied().max().map(|m| m + 1).unwrap_or(1);

  let svg_height = PADDING_TOP
    + num_rows as u32 * ROW_HEIGHT
    + AXIS_GAP
    + PADDING_BOTTOM;
  let render_height = svg_height * RESOLUTION_SCALE;

  let top = PADDING_TOP as f64 * sf;
  let axis_y = top + num_rows as f64 * ROW_HEIGHT as f64 * sf + AXIS_GAP as f64 * sf;
  let row_y = |row: usize| -> f64 {
    top + (row as f64 + 0.5) * ROW_HEIGHT as f64 * sf
  };

  let mut body = String::new();

  // Background.
  body.push_str(&format!(
    "<rect width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
    render_width, render_height, bg_color
  ));

  let axis_x0 = margin_left;
  let axis_x1 = margin_left + plot_width;

  // Draw events (in original order for stable colors/z-order).
  let marker_r = 4.0 * sf;
  let bar_half = 3.0 * sf;
  for (idx, ev) in events.iter().enumerate() {
    let y = row_y(rows[idx]);
    let x_start = x_to_px(ev.start);

    // Light vertical guide from the marker down to the axis.
    body.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
       stroke=\"{}\" stroke-width=\"{:.1}\" stroke-dasharray=\"{:.0},{:.0}\"/>\n",
      x_start,
      y,
      x_start,
      axis_y,
      axis_color,
      sf * 0.5,
      sf * 2.0,
      sf * 2.0,
    ));

    let label_x = match ev.end {
      Some(end) => {
        // Interval: draw a rounded bar from start to end.
        let x_end = x_to_px(end);
        body.push_str(&format!(
          "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
           rx=\"{:.1}\" fill=\"{}\"/>\n",
          x_start,
          y - bar_half,
          (x_end - x_start).max(sf),
          bar_half * 2.0,
          bar_half,
          event_color,
        ));
        x_end + marker_gap
      }
      None => {
        // Point: draw a filled circle marker.
        body.push_str(&format!(
          "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{}\"/>\n",
          x_start, y, marker_r, event_color
        ));
        x_start + marker_r + marker_gap
      }
    };

    // Event label to the right of the marker, vertically centered.
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" dy=\"0.32em\" text-anchor=\"start\" \
       font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{}\">{}</text>\n",
      label_x,
      y,
      font_size,
      label_fill,
      html_escape(&ev.label),
    ));
  }

  // Bottom axis line.
  body.push_str(&format!(
    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
     stroke=\"{}\" stroke-width=\"{:.0}\"/>\n",
    axis_x0, axis_y, axis_x1, axis_y, axis_color, sf
  ));

  // Date ticks and labels on the axis.
  let ticks = generate_date_ticks(x_min, x_max);
  let tick_len = 5.0 * sf;
  let tick_font = 12.0 * sf;
  for &t in &ticks {
    if t < x_min || t > x_max {
      continue;
    }
    let px = x_to_px(t);
    body.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
       stroke=\"{}\" stroke-width=\"{:.0}\"/>\n",
      px,
      axis_y,
      px,
      axis_y + tick_len,
      axis_color,
      sf
    ));
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
       font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{}\">{}</text>\n",
      px,
      axis_y + tick_len + tick_font * 1.1,
      tick_font,
      label_fill,
      html_escape(&format_date_tick(t)),
    ));
  }

  let mut buf = format!(
    "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{}</svg>",
    svg_width, svg_height, render_width, render_height, body
  );

  if full_width {
    let old = format!("width=\"{}\" height=\"{}\"", svg_width, svg_height);
    buf = buf.replacen(&old, "width=\"100%\"", 1);
  }

  buf
}

/// Escape special HTML characters in text content.
fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// Theme colors: `(background, axis, label, event)`.
fn theme_colors() -> (&'static str, &'static str, String, String) {
  let (r, g, b) = PLOT_COLORS[0];
  let event = format!("rgb({},{},{})", r, g, b);
  if crate::is_dark_mode() {
    ("#1a1a1a", "#999999", "#999".to_string(), event)
  } else {
    ("#ffffff", "#666666", "#666".to_string(), event)
  }
}
