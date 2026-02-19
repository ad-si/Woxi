use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, PLOT_COLORS, generate_bar_svg,
  generate_histogram_svg, parse_image_size,
};
use crate::syntax::Expr;

/// Extract a flat list of f64 values from the first argument.
fn extract_values(arg: &Expr) -> Result<Vec<f64>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  match &data {
    Expr::List(items) => {
      let mut vals = Vec::with_capacity(items.len());
      for item in items {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        if let Some(f) = try_eval_to_f64(&v) {
          vals.push(f);
        }
      }
      Ok(vals)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Chart: first argument must be a list".into(),
    )),
  }
}

/// Parsed chart options.
pub(crate) struct ChartOptions {
  pub svg_width: u32,
  pub svg_height: u32,
  pub full_width: bool,
  pub chart_labels: Vec<String>,
  pub plot_label: Option<String>,
  pub axes_label: Option<(String, String)>,
}

/// Extract a string from an Expr (Identifier or String).
fn expr_to_label(e: &Expr) -> Option<String> {
  match e {
    Expr::String(s) => Some(s.clone()),
    Expr::Identifier(s) => Some(s.clone()),
    _ => None,
  }
}

/// Parse options from chart arguments.
fn parse_chart_options(args: &[Expr]) -> ChartOptions {
  let mut opts = ChartOptions {
    svg_width: DEFAULT_WIDTH,
    svg_height: DEFAULT_HEIGHT,
    full_width: false,
    chart_labels: Vec::new(),
    plot_label: None,
    axes_label: None,
  };
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) = parse_image_size(replacement) {
            opts.svg_width = w;
            opts.svg_height = h;
            opts.full_width = fw;
          }
        }
        "ChartLabels" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Expr::List(items) = &val {
            for item in items {
              if let Some(s) = expr_to_label(item) {
                opts.chart_labels.push(s);
              }
            }
          }
        }
        "PlotLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Some(s) = expr_to_label(&val) {
            opts.plot_label = Some(s);
          }
        }
        "AxesLabel" => {
          let val =
            evaluate_expr_to_expr(replacement).unwrap_or(*replacement.clone());
          if let Expr::List(items) = &val
            && items.len() >= 2
          {
            let x = expr_to_label(&items[0]).unwrap_or_default();
            let y = expr_to_label(&items[1]).unwrap_or_default();
            opts.axes_label = Some((x, y));
          }
        }
        _ => {}
      }
    }
  }
  opts
}

fn svg_header(w: u32, h: u32, full_width: bool) -> String {
  if full_width {
    format!(
      "<svg width=\"100%\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"
    )
  } else {
    format!(
      "<svg width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n\
       <rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"
    )
  }
}

/// BarChart[{v1, v2, ...}]
pub fn bar_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let values = extract_values(&args[0])?;
  if values.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }
  let opts = parse_chart_options(args);

  let svg = generate_bar_svg(
    &values,
    opts.svg_width,
    opts.svg_height,
    opts.full_width,
    &opts.chart_labels,
    opts.plot_label.as_deref(),
    opts
      .axes_label
      .as_ref()
      .map(|(x, y)| (x.as_str(), y.as_str())),
  )?;
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// PieChart[{v1, v2, ...}]
pub fn pie_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let values = extract_values(&args[0])?;
  if values.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;
  let radius = (w.min(h) / 2.0) * 0.85;
  let total: f64 = values.iter().sum();
  if total <= 0.0 {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }

  let mut svg = svg_header(svg_width, svg_height, full_width);

  let mut start_angle = -std::f64::consts::FRAC_PI_2; // Start at top
  for (i, &val) in values.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
    let sweep = 2.0 * std::f64::consts::PI * val / total;
    let end_angle = start_angle + sweep;

    let x1 = cx + radius * start_angle.cos();
    let y1 = cy + radius * start_angle.sin();
    let x2 = cx + radius * end_angle.cos();
    let y2 = cy + radius * end_angle.sin();
    let large_arc = if sweep > std::f64::consts::PI { 1 } else { 0 };

    svg.push_str(&format!(
      "<path d=\"M{cx:.2},{cy:.2} L{x1:.2},{y1:.2} A{radius:.2},{radius:.2} 0 {large_arc},1 {x2:.2},{y2:.2} Z\" \
       fill=\"rgb({r},{g},{b})\" stroke=\"white\" stroke-width=\"1\"/>\n"
    ));

    start_angle = end_angle;
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// Histogram[{d1, d2, ...}]
pub fn histogram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let values = extract_values(&args[0])?;
  if values.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  let svg = generate_histogram_svg(&values, svg_width, svg_height, full_width)?;
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// Compute box-whisker statistics for a sorted dataset.
/// Uses Mathematica's quartile interpolation:
///   Q1 at position (n+2)/4, Q2 at (n+1)/2, Q3 at (3n+2)/4 (1-indexed).
fn box_stats(sorted: &[f64]) -> (f64, f64, f64, f64, f64) {
  let n = sorted.len() as f64;
  let interp = |pos: f64| -> f64 {
    let idx = pos - 1.0; // convert to 0-indexed
    let lo = idx.floor() as usize;
    let hi = (idx.ceil() as usize).min(sorted.len() - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
  };
  let q1 = interp((n + 2.0) / 4.0);
  let median = interp((n + 1.0) / 2.0);
  let q3 = interp((3.0 * n + 2.0) / 4.0);
  (sorted[0], q1, median, q3, sorted[sorted.len() - 1])
}

/// Parse the first argument into one or more datasets (Vec<Vec<f64>>).
/// Supports:
/// - `{d1, d2, ...}` → single dataset
/// - `{{d1, d2, ...}, {d3, d4, ...}}` → multiple datasets
fn parse_datasets(arg: &Expr) -> Result<Vec<Vec<f64>>, InterpreterError> {
  let data = evaluate_expr_to_expr(arg)?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BoxWhiskerChart: first argument must be a list".into(),
      ));
    }
  };

  if items.is_empty() {
    return Ok(vec![]);
  }

  // Check if the first element is a list (multiple datasets)
  if let Expr::List(_) = &items[0] {
    let mut datasets = Vec::new();
    for item in items {
      if let Expr::List(inner) = item {
        let mut vals = Vec::new();
        for v in inner {
          let ev = evaluate_expr_to_expr(v).unwrap_or(v.clone());
          if let Some(f) = try_eval_to_f64(&ev) {
            vals.push(f);
          }
        }
        if !vals.is_empty() {
          datasets.push(vals);
        }
      }
    }
    Ok(datasets)
  } else {
    // Flat list: single dataset
    let mut vals = Vec::new();
    for item in items {
      let ev = evaluate_expr_to_expr(item).unwrap_or(item.clone());
      if let Some(f) = try_eval_to_f64(&ev) {
        vals.push(f);
      }
    }
    if vals.is_empty() {
      Ok(vec![])
    } else {
      Ok(vec![vals])
    }
  }
}

/// BoxWhiskerChart[{d1, d2, ...}] or BoxWhiskerChart[{{d1,...}, {d2,...}, ...}]
pub fn box_whisker_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut datasets = parse_datasets(&args[0])?;
  if datasets.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  // Sort each dataset and compute stats
  let mut all_stats = Vec::new();
  let mut global_min = f64::INFINITY;
  let mut global_max = f64::NEG_INFINITY;
  for ds in &mut datasets {
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, q1, med, q3, hi) = box_stats(ds);
    global_min = global_min.min(lo);
    global_max = global_max.max(hi);
    all_stats.push((lo, q1, med, q3, hi));
  }

  let v_range = global_max - global_min;
  let v_pad = if v_range.abs() < f64::EPSILON {
    1.0
  } else {
    v_range * 0.08
  };
  let v_min = global_min - v_pad;
  let v_max = global_max + v_pad;

  // Use plotters for axes
  let n = datasets.len();
  // One tick mark per box (centered), no labels — matches Mathematica
  let x_tick_positions: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
  let area = crate::functions::plot::generate_axes_only_opts(
    (0.0, n as f64),
    (v_min, v_max),
    svg_width,
    svg_height,
    full_width,
    Some(&x_tick_positions),
  )?;

  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let map_y =
    |v: f64| -> f64 { plot_y0 + (v_max - v) / (v_max - v_min) * plot_h };
  let slot_w = plot_w / n as f64;
  let stroke_w = (area.render_width as f64 / 1000.0 * 1.5).max(1.0);

  for (i, &(lo, q1, med, q3, hi)) in all_stats.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[0];
    let cx = plot_x0 + (i as f64 + 0.5) * slot_w;
    let box_half_w = slot_w * 0.3;
    let cap_half_w = box_half_w * 0.5;

    // Whisker line (vertical)
    svg.push_str(&format!(
      "<line x1=\"{cx:.1}\" y1=\"{:.1}\" x2=\"{cx:.1}\" y2=\"{:.1}\" stroke=\"#666\" stroke-width=\"{stroke_w:.1}\"/>\n",
      map_y(hi), map_y(lo)
    ));
    // Top whisker cap
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#666\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - cap_half_w, map_y(hi), cx + cap_half_w, map_y(hi)
    ));
    // Bottom whisker cap
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#666\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - cap_half_w, map_y(lo), cx + cap_half_w, map_y(lo)
    ));
    // Box (Q1 to Q3)
    let box_top = map_y(q3);
    let box_bot = map_y(q1);
    let box_h = box_bot - box_top;
    svg.push_str(&format!(
      "<rect x=\"{:.1}\" y=\"{box_top:.1}\" width=\"{:.1}\" height=\"{box_h:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"#666\" stroke-width=\"{stroke_w:.1}\"/>\n",
      cx - box_half_w, box_half_w * 2.0
    ));
    // Median line
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"white\" stroke-width=\"{:.1}\"/>\n",
      cx - box_half_w, map_y(med), cx + box_half_w, map_y(med), stroke_w * 1.5
    ));
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// BubbleChart[{{x,y,z}, ...}]
pub fn bubble_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "BubbleChart: first argument must be a list of {x, y, z} triples"
          .into(),
      ));
    }
  };

  let mut triples = Vec::new();
  for item in items {
    if let Expr::List(inner) = item
      && inner.len() >= 3
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[0]).unwrap_or(inner[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[1]).unwrap_or(inner[1].clone()),
      );
      let z = try_eval_to_f64(
        &evaluate_expr_to_expr(&inner[2]).unwrap_or(inner[2].clone()),
      );
      if let (Some(x), Some(y), Some(z)) = (x, y, z) {
        triples.push((x, y, z));
      }
    }
  }

  if triples.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let margin = 50.0;
  let plot_w = w - 2.0 * margin;
  let plot_h = h - 2.0 * margin;

  let x_min = triples.iter().map(|t| t.0).fold(f64::INFINITY, f64::min);
  let x_max = triples
    .iter()
    .map(|t| t.0)
    .fold(f64::NEG_INFINITY, f64::max);
  let y_min = triples.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
  let y_max = triples
    .iter()
    .map(|t| t.1)
    .fold(f64::NEG_INFINITY, f64::max);
  let z_max = triples.iter().map(|t| t.2.abs()).fold(0.0_f64, f64::max);
  let max_radius = 20.0;

  let x_range = if (x_max - x_min).abs() < f64::EPSILON {
    1.0
  } else {
    x_max - x_min
  };
  let y_range = if (y_max - y_min).abs() < f64::EPSILON {
    1.0
  } else {
    y_max - y_min
  };

  let mut svg = svg_header(svg_width, svg_height, full_width);

  for (i, &(x, y, z)) in triples.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
    let sx = margin + ((x - x_min) / x_range) * plot_w;
    let sy = margin + ((y_max - y) / y_range) * plot_h;
    let radius = if z_max > 0.0 {
      (z.abs() / z_max) * max_radius
    } else {
      5.0
    };
    let radius = radius.max(2.0);
    svg.push_str(&format!(
      "<circle cx=\"{sx:.1}\" cy=\"{sy:.1}\" r=\"{radius:.1}\" fill=\"rgb({r},{g},{b})\" fill-opacity=\"0.7\"/>\n"
    ));
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// SectorChart[{{angle, radius}, ...}] - like PieChart but with variable radius
pub fn sector_chart_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "SectorChart: first argument must be a list".into(),
      ));
    }
  };

  let mut sectors: Vec<(f64, f64)> = Vec::new();
  for item in items {
    if let Expr::List(pair) = item {
      if pair.len() >= 2 {
        let a = try_eval_to_f64(
          &evaluate_expr_to_expr(&pair[0]).unwrap_or(pair[0].clone()),
        );
        let r = try_eval_to_f64(
          &evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone()),
        );
        if let (Some(a), Some(r)) = (a, r) {
          sectors.push((a, r));
        }
      }
    } else if let Some(v) =
      try_eval_to_f64(&evaluate_expr_to_expr(item).unwrap_or(item.clone()))
    {
      sectors.push((v, 1.0));
    }
  }

  if sectors.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;
  let max_radius = (w.min(h) / 2.0) * 0.85;
  let total_angle: f64 = sectors.iter().map(|s| s.0).sum();
  let r_max = sectors.iter().map(|s| s.1).fold(0.0_f64, f64::max);
  let r_max = if r_max <= 0.0 { 1.0 } else { r_max };

  let mut svg = svg_header(svg_width, svg_height, full_width);

  let mut start_angle = -std::f64::consts::FRAC_PI_2;
  for (i, &(angle_val, radius_val)) in sectors.iter().enumerate() {
    let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
    let sweep = 2.0 * std::f64::consts::PI * angle_val / total_angle;
    let sector_r = (radius_val / r_max) * max_radius;
    let end_angle = start_angle + sweep;

    let x1 = cx + sector_r * start_angle.cos();
    let y1 = cy + sector_r * start_angle.sin();
    let x2 = cx + sector_r * end_angle.cos();
    let y2 = cy + sector_r * end_angle.sin();
    let large_arc = if sweep > std::f64::consts::PI { 1 } else { 0 };

    svg.push_str(&format!(
      "<path d=\"M{cx:.2},{cy:.2} L{x1:.2},{y1:.2} A{sector_r:.2},{sector_r:.2} 0 {large_arc},1 {x2:.2},{y2:.2} Z\" \
       fill=\"rgb({r},{g},{b})\" stroke=\"white\" stroke-width=\"1\"/>\n"
    ));

    start_angle = end_angle;
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// DateListPlot[{{date, y}, ...}] - simplified: treats dates as numeric x values
pub fn date_list_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "DateListPlot: first argument must be a list".into(),
      ));
    }
  };

  // Try to extract {x, y} pairs (treating dates as numbers)
  let mut points = Vec::new();
  for (i, item) in items.iter().enumerate() {
    if let Expr::List(pair) = item
      && pair.len() >= 2
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&pair[0]).unwrap_or(pair[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&pair[1]).unwrap_or(pair[1].clone()),
      );
      if let (Some(x), Some(y)) = (x, y) {
        points.push((x, y));
        continue;
      }
    }
    // Fallback: treat as y value with sequential x
    if let Some(y) =
      try_eval_to_f64(&evaluate_expr_to_expr(item).unwrap_or(item.clone()))
    {
      points.push(((i + 1) as f64, y));
    }
  }

  if points.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }

  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);

  // Use plotters-based generate_svg for line plot
  let all_series = vec![points];
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  for &(x, y) in &all_series[0] {
    x_min = x_min.min(x);
    x_max = x_max.max(x);
    y_min = y_min.min(y);
    y_max = y_max.max(y);
  }
  let xp = (x_max - x_min) * 0.04;
  let yp = (y_max - y_min) * 0.04;
  let xp = if xp.abs() < f64::EPSILON { 1.0 } else { xp };
  let yp = if yp.abs() < f64::EPSILON { 1.0 } else { yp };

  let svg = crate::functions::plot::generate_svg(
    &all_series,
    (x_min - xp, x_max + xp),
    (y_min - yp, y_max + yp),
    svg_width,
    svg_height,
    full_width,
  )?;

  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// Escape special HTML characters in text content.
fn html_escape(s: &str) -> String {
  s.replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

/// WordCloud[{"word1", "word2", ...}]
pub fn word_cloud_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  use std::collections::HashMap;

  // Extract list of strings from first argument
  let data = evaluate_expr_to_expr(&args[0])?;
  let items = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "WordCloud: first argument must be a list of strings".into(),
      ));
    }
  };

  let mut words: Vec<String> = Vec::new();
  for item in items {
    let ev = evaluate_expr_to_expr(item).unwrap_or(item.clone());
    if let Expr::String(s) = &ev {
      words.push(s.clone());
    }
  }

  if words.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
  }

  // Count word frequencies
  let mut freq: HashMap<String, usize> = HashMap::new();
  for w in &words {
    *freq.entry(w.clone()).or_insert(0) += 1;
  }

  // Sort by frequency (descending), then alphabetically for stability
  let mut sorted_words: Vec<(String, usize)> = freq.into_iter().collect();
  sorted_words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

  // Parse options (ImageSize)
  let opts = parse_chart_options(args);
  let (svg_width, svg_height, full_width) =
    (opts.svg_width, opts.svg_height, opts.full_width);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cx = w / 2.0;
  let cy = h / 2.0;

  // Map frequencies to font sizes (linear scaling)
  let max_freq = sorted_words[0].1 as f64;
  let min_freq = sorted_words.last().unwrap().1 as f64;
  let min_font = 12.0_f64;
  let max_font = (h * 0.2).min(60.0);

  let font_size_for = |count: usize| -> f64 {
    if (max_freq - min_freq).abs() < f64::EPSILON {
      (min_font + max_font) / 2.0
    } else {
      min_font
        + (count as f64 - min_freq) / (max_freq - min_freq)
          * (max_font - min_font)
    }
  };

  // Placed word bounding boxes: (x_min, y_min, x_max, y_max)
  let mut placed: Vec<(f64, f64, f64, f64)> = Vec::new();

  struct PlacedWord {
    x: f64,
    y: f64,
    font_size: f64,
    text: String,
    color_idx: usize,
    rotated: bool,
  }
  let mut placed_words: Vec<PlacedWord> = Vec::new();

  let char_width_factor = 0.6;

  for (i, (word, count)) in sorted_words.iter().enumerate() {
    let font_size = font_size_for(*count);
    let rotated = i % 3 == 1; // every third word rotated

    // Estimate bounding box dimensions
    let text_w = word.len() as f64 * char_width_factor * font_size;
    let text_h = font_size;
    let (box_w, box_h) = if rotated {
      (text_h, text_w) // swap for rotation
    } else {
      (text_w, text_h)
    };

    // Try placing using Archimedean spiral from center
    let max_steps = 10000;
    let a = 1.5; // spiral spacing

    let mut placed_ok = false;
    for s in 0..max_steps {
      let t = s as f64 * 0.04;
      let x = cx + a * t * t.cos();
      let y = cy + a * t * t.sin() * 0.6; // compress vertically to fit aspect ratio

      // Bounding box centered at (x, y)
      let x_min = x - box_w / 2.0;
      let y_min = y - box_h / 2.0;
      let x_max = x + box_w / 2.0;
      let y_max = y + box_h / 2.0;

      // Check bounds
      if x_min < 2.0 || y_min < 2.0 || x_max > w - 2.0 || y_max > h - 2.0 {
        continue;
      }

      // Check collision with already placed words
      let collides = placed.iter().any(|&(px_min, py_min, px_max, py_max)| {
        x_min < px_max && x_max > px_min && y_min < py_max && y_max > py_min
      });

      if !collides {
        placed.push((x_min, y_min, x_max, y_max));
        placed_words.push(PlacedWord {
          x,
          y,
          font_size,
          text: word.clone(),
          color_idx: i,
          rotated,
        });
        placed_ok = true;
        break;
      }
    }

    if !placed_ok {
      continue; // skip words that can't be placed
    }
  }

  // Generate SVG
  let mut svg = svg_header(svg_width, svg_height, full_width);

  for pw in &placed_words {
    let (r, g, b) = PLOT_COLORS[pw.color_idx % PLOT_COLORS.len()];
    let transform = if pw.rotated {
      format!(
        " transform=\"translate({:.1},{:.1}) rotate(-90)\"",
        pw.x, pw.y
      )
    } else {
      format!(" transform=\"translate({:.1},{:.1})\"", pw.x, pw.y)
    };
    svg.push_str(&format!(
      "<text{transform} font-size=\"{:.1}\" fill=\"rgb({r},{g},{b})\" \
       font-family=\"sans-serif\" text-anchor=\"middle\" \
       dominant-baseline=\"central\">{}</text>\n",
      pw.font_size,
      html_escape(&pw.text)
    ));
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}
