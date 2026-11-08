#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::interval_ast::is_interval;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, PLOT_COLORS, RESOLUTION_SCALE,
  format_date_tick, generate_date_ticks, html_escape, parse_image_size,
};

/// Default display width of a TimelinePlot (Wolfram's `ImageSize -> 600`).
const DEFAULT_TLP_WIDTH: u32 = 600;

/// A single timeline event: either an instant (a point) or a span (interval).
struct TimelineEvent {
  /// Start of the event in AbsoluteTime seconds (since 1900-01-01).
  start: f64,
  /// End of the event for intervals; `None` for point events.
  end: Option<f64>,
  /// The label drawn in a callout above the event; Wolfram labels only
  /// events given a label (an association key, `Labeled`, `label -> date`).
  label: Option<String>,
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
    return Ok(unevaluated("TimelinePlot", args));
  }

  let svg = render_timeline_svg(&events, svg_width, full_width);
  Ok(crate::graphics_result(svg))
}

/// Parse the first argument into a flat list of timeline events.
fn parse_events(arg: &Expr) -> Vec<TimelineEvent> {
  let evaluated = evaluate_expr_to_expr(arg).unwrap_or_else(|_| arg.clone());

  let mut events = Vec::new();
  match &evaluated {
    // Association: each key -> value pair is a labeled event.
    Expr::Association(pairs) => {
      for (key, val) in pairs {
        if let Some(mut ev) = event_from_date(val) {
          ev.label = Some(expr_to_label(key));
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
      ev.label = Some(expr_to_label(&args[1]));
      Some(ev)
    }
    // Style[date, ...] — ignore styling, keep the underlying date.
    Expr::FunctionCall { name, args }
      if name == "Style" && !args.is_empty() =>
    {
      event_from_date(&args[0])
    }
    // Tooltip[date, label] — the tooltip only shows on hover.
    Expr::FunctionCall { name, args }
      if name == "Tooltip" && args.len() == 2 =>
    {
      event_from_date(&args[0])
    }
    // label -> date (an association entry written as a Rule).
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let mut ev = event_from_date(replacement)?;
      ev.label = Some(expr_to_label(pattern));
      Some(ev)
    }
    _ => event_from_date(&evaluated),
  }
}

/// Build an unlabeled event from a date-like expression.
fn event_from_date(expr: &Expr) -> Option<TimelineEvent> {
  let evaluated = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());

  // DateInterval[{d1, d2}] — an explicit time span. Also matches the
  // canonical form DateInterval[{{d1, d2}}, granularity, calendar, timezone]
  // that evaluating DateInterval produces.
  if let Expr::FunctionCall { name, args } = &evaluated
    && name == "DateInterval"
    && !args.is_empty()
    && let Expr::List(spec) = &args[0]
    && let Some(pair) = match spec.as_slice() {
      [d1, d2] => Some([d1, d2]),
      [Expr::List(span)] if span.len() == 2 => Some([&span[0], &span[1]]),
      _ => None,
    }
    && let (Some(lo), Some(hi)) = (
      date_to_absolute_time(pair[0]),
      date_to_absolute_time(pair[1]),
    )
  {
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    return Some(TimelineEvent {
      start: lo,
      end: Some(hi),
      label: None,
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
      label: None,
    });
  }

  // A single instant.
  let t = date_to_absolute_time(&evaluated)?;
  Some(TimelineEvent {
    start: t,
    end: None,
    label: None,
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
  let month = components.get(1).map_or(1, |v| *v as i64);
  let day = components.get(2).map_or(1, |v| *v as i64);
  let hour = components.get(3).map_or(0, |v| *v as i64);
  let minute = components.get(4).map_or(0, |v| *v as i64);
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
    other => expr_to_output(other),
  }
}

/// Estimate the width, in points, of a label at `font_size` points.
fn est_text_width(label: &str, font_size: f64) -> f64 {
  label
    .chars()
    .map(|c| match c {
      'i' | 'j' | 'l' | 't' | 'f' | 'r' | ' ' | '.' | ',' => 0.3,
      'm' | 'w' | 'M' | 'W' => 0.85,
      c if c.is_uppercase() => 0.68,
      _ => 0.55,
    })
    .sum::<f64>()
    * font_size
}

/// `TimelinePlot`'s date ticks, as `(time, label)` for the labeled ones
/// plus the unlabeled minor ones: month ticks (`"Mar 2020"`) a step of
/// 1, 2, 3 or 6 months apart with a tick every month between, or year
/// ticks over long spans; under two months the generic date ticks.
fn timeline_ticks(x_min: f64, x_max: f64) -> (Vec<(f64, String)>, Vec<f64>) {
  use crate::functions::datetime_ast::{
    absolute_seconds_to_date, date_to_absolute_seconds,
  };
  const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
  ];
  let months = (x_max - x_min) / (30.44 * 86400.0);
  if months < 2.0 {
    let ticks = generate_date_ticks(x_min, x_max);
    return (
      ticks.iter().map(|&t| (t, format_date_tick(t))).collect(),
      Vec::new(),
    );
  }
  let (y0, m0, ..) = absolute_seconds_to_date(x_min);
  let (y1, ..) = absolute_seconds_to_date(x_max);
  if months > 36.0 {
    // Years, a step of 1, 2, 5, 10, … apart.
    let span = (y1 - y0 + 1) as f64;
    let step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
      .into_iter()
      .find(|s| span / *s as f64 <= 5.0)
      .unwrap_or(1000);
    let mut majors = Vec::new();
    let mut minors = Vec::new();
    for y in y0..=y1 + 1 {
      let t = date_to_absolute_seconds(y, 1, 1, 0, 0, 0.0);
      if t < x_min || t > x_max {
        continue;
      }
      if y.rem_euclid(step) == 0 {
        majors.push((t, y.to_string()));
      } else if step <= 10 {
        minors.push(t);
      }
    }
    return (majors, minors);
  }
  let step = [1, 2, 3, 6]
    .into_iter()
    .find(|s| months / *s as f64 <= 5.0)
    .unwrap_or(12);
  let mut majors = Vec::new();
  let mut minors = Vec::new();
  let (mut y, mut m) = (y0, m0);
  loop {
    let t = date_to_absolute_seconds(y, m, 1, 0, 0, 0.0);
    if t > x_max {
      break;
    }
    if t >= x_min {
      if (m - 1).rem_euclid(step) == 0 {
        majors.push((t, format!("{} {y}", MONTHS[(m - 1) as usize])));
      }
      minors.push(t);
    }
    m += 1;
    if m > 12 {
      m = 1;
      y += 1;
    }
  }
  (majors, minors)
}

/// Render the complete TimelinePlot SVG, laid out as Wolfram does: each
/// event on the lowest level where it does not run into another (a point
/// as a dot, an interval as a thin line between open dots), a date axis
/// along the bottom, and every label in a rounded callout box above the
/// plot with a stem down to its event.
fn render_timeline_svg(
  events: &[TimelineEvent],
  svg_width: u32,
  full_width: bool,
) -> String {
  let sf = RESOLUTION_SCALE as f64;
  let render_width = svg_width * RESOLUTION_SCALE;
  // Points per WL point at the default 600-point width.
  let w = svg_width as f64;

  let (bg_color, axis_color, label_fill, event_color) = theme_colors();
  let halo = if crate::is_dark_mode() {
    "#1a1a1a"
  } else {
    "#ffffff"
  };

  // The axis spans the picture; the events take its middle 88%.
  let axis_x0 = 0.5;
  let axis_x1 = w - 1.5;
  let (mut t_min, mut t_max) = (f64::INFINITY, f64::NEG_INFINITY);
  for ev in events {
    t_min = t_min.min(ev.start);
    t_max = t_max.max(ev.end.unwrap_or(ev.start));
  }
  if !t_min.is_finite() || !t_max.is_finite() {
    t_min = 0.0;
    t_max = 86400.0;
  }
  if (t_max - t_min).abs() < f64::EPSILON {
    // All events at the same instant: a one-year window around it.
    t_min -= 182.625 * 86400.0;
    t_max += 182.625 * 86400.0;
  }
  let pad = (t_max - t_min) * 0.06 / 0.88;
  let (x_min, x_max) = (t_min - pad, t_max + pad);
  let x_to_pt = |t: f64| -> f64 {
    axis_x0 + (t - x_min) / (x_max - x_min) * (axis_x1 - axis_x0)
  };

  // Levels: in order of start, each event takes the lowest level whose
  // events all end (with room for the dots) before it begins.
  let mut order: Vec<usize> = (0..events.len()).collect();
  order.sort_by(|&a, &b| events[a].start.total_cmp(&events[b].start));
  let mut level_end: Vec<f64> = Vec::new();
  let mut levels = vec![0usize; events.len()];
  for &i in &order {
    let ev = &events[i];
    let left = x_to_pt(ev.start) - 6.0;
    let right = x_to_pt(ev.end.unwrap_or(ev.start)) + 6.0;
    let level = if let Some(l) = level_end.iter().position(|&end| end <= left) {
      l
    } else {
      level_end.push(f64::NEG_INFINITY);
      level_end.len() - 1
    };
    level_end[level] = right;
    levels[i] = level;
  }
  let n_levels = level_end.len().max(1);

  // The plot area (`AspectRatio -> 1/10` without labels) holds the levels
  // at heights 1, 2, … over a range padded to `levels + 1/2` and by 5%
  // above; labels add a band of callouts above it.
  let labeled = events.iter().any(|e| e.label.is_some());
  let range = (n_levels as f64 + 0.5) * 1.05;
  let (unit, plot_top) = if labeled {
    (16.41, 60.57)
  } else {
    (w / 10.0 / range, 0.0)
  };
  let axis_y = plot_top + unit * range;
  let label_room = 15.171875;
  let svg_height = (axis_y + label_room).ceil() as u32;
  let render_height = svg_height * RESOLUTION_SCALE;
  let level_y = |l: usize| axis_y - unit * (l + 1) as f64;

  let mut body = String::new();
  let px = |v: f64| v * sf;
  body.push_str(&format!(
    "<rect width=\"{render_width}\" height=\"{render_height}\" fill=\"{bg_color}\"/>\n"
  ));
  let circle = |body: &mut String, x: f64, y: f64, r: f64, fill: &str| {
    body.push_str(&format!(
      "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{:.1}\" fill=\"{fill}\"/>\n",
      px(x),
      px(y),
      px(r)
    ));
  };
  let line = |body: &mut String,
              (x1, y1): (f64, f64),
              (x2, y2): (f64, f64),
              width: f64,
              stroke: &str| {
    body.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{stroke}\" stroke-width=\"{:.1}\"/>\n",
        px(x1),
        px(y1),
        px(x2),
        px(y2),
        px(width)
      ));
  };

  for (i, ev) in events.iter().enumerate() {
    let y = level_y(levels[i]);
    let x = x_to_pt(ev.start);
    if let Some(end) = ev.end {
      let x_end = x_to_pt(end);
      line(&mut body, (x, y), (x_end, y), 4.0, halo);
      circle(&mut body, x, y, 5.0, halo);
      circle(&mut body, x_end, y, 5.0, halo);
      line(&mut body, (x, y), (x_end, y), 1.6, &event_color);
      for xe in [x, x_end] {
        circle(&mut body, xe, y, 3.5, &event_color);
        circle(&mut body, xe, y, 2.0, halo);
      }
    } else {
      circle(&mut body, x, y, 5.0, halo);
      circle(&mut body, x, y, 3.5, &event_color);
    }
    // The callout: a rounded box above the plot, a stem down to the event
    // and an arrowhead under the box.
    if let Some(label) = &ev.label {
      let font = 10.0;
      let half = est_text_width(label, font) / 2.0 + 5.0;
      let (box_top, box_bottom) = (14.43, 33.0);
      line(&mut body, (x, box_bottom + 5.0), (x, y - 4.0), 4.0, halo);
      line(&mut body, (x, box_bottom), (x, y - 3.0), 1.6, &event_color);
      body.push_str(&format!(
        "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{event_color}\"/>\n",
        px(x),
        px(box_bottom + 10.33),
        px(x + 5.33),
        px(box_bottom + 3.5),
        px(x - 5.33),
        px(box_bottom + 3.5)
      ));
      body.push_str(&format!(
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"{:.1}\" fill=\"{halo}\" stroke=\"{event_color}\" stroke-width=\"{:.1}\"/>\n",
        px(x - half),
        px(box_top),
        px(2.0 * half),
        px(box_bottom - box_top),
        px(5.0),
        px(1.0)
      ));
      body.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{label_fill}\">{}</text>\n",
        px(x),
        px(box_top + 14.46),
        px(font),
        html_escape(label),
      ));
    }
  }

  // The date axis: a hairline across the picture with ticks pointing up,
  // labeled below.
  line(
    &mut body,
    (axis_x0, axis_y),
    (axis_x1, axis_y),
    0.5,
    axis_color,
  );
  let (majors, minors) = timeline_ticks(x_min, x_max);
  for t in minors.iter().copied().chain(majors.iter().map(|(t, _)| *t)) {
    let x = x_to_pt(t);
    line(&mut body, (x, axis_y), (x, axis_y - 3.99), 0.5, axis_color);
  }
  for (t, label) in &majors {
    body.push_str(&format!(
      "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"{:.0}\" fill=\"{label_fill}\">{}</text>\n",
      px(x_to_pt(*t)),
      px(axis_y + 12.05),
      px(10.0),
      html_escape(label),
    ));
  }

  let mut buf = format!(
    "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {render_width} {render_height}\" \
     xmlns=\"http://www.w3.org/2000/svg\">\n{body}</svg>"
  );

  if full_width {
    let old = format!("width=\"{svg_width}\" height=\"{svg_height}\"");
    buf = buf.replacen(&old, "width=\"100%\"", 1);
  }

  buf
}

/// Theme colors: `(background, axis, label, event)`.
fn theme_colors() -> (&'static str, &'static str, String, String) {
  let (r, g, b) = PLOT_COLORS[0];
  let event = format!("rgb({r},{g},{b})");
  if crate::is_dark_mode() {
    ("#1a1a1a", "#999999", "#999".to_string(), event)
  } else {
    ("#ffffff", "#666666", "#666".to_string(), event)
  }
}
