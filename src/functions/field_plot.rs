use plotters::prelude::*;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::graphics::{Color as GfxColor, parse_color};
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::math_ast::{
  build_complex_float_expr, try_extract_complex_f64,
};
use crate::functions::plot::{
  DEFAULT_HEIGHT, DEFAULT_WIDTH, RESOLUTION_SCALE, generate_axes_only,
  parse_image_size, plot_theme, rewrite_svg_header, substitute_var,
};
use crate::syntax::Expr;

const FIELD_GRID: usize = 100;
const VECTOR_GRID: usize = 15;

/// Parse iterator: {var, min, max}
fn parse_iterator(
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

/// Evaluate f(x, y)
fn evaluate_at_xy(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<f64> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  try_eval_to_f64(&result)
}

/// Evaluate a boolean condition at (x, y) - returns true/false
fn evaluate_condition(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> bool {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  if let Ok(result) = evaluate_expr_to_expr(&sub2) {
    matches!(result, Expr::Identifier(ref s) if s == "True")
  } else {
    false
  }
}

/// Evaluate {vx, vy} at (x, y)
fn evaluate_vector(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  xval: f64,
  yval: f64,
) -> Option<(f64, f64)> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  if let Expr::List(items) = &result
    && items.len() == 2
  {
    let vx = try_eval_to_f64(&items[0])?;
    let vy = try_eval_to_f64(&items[1])?;
    return Some((vx, vy));
  }
  None
}

/// Parse ImageSize from options
fn parse_field_options(args: &[Expr], start: usize) -> (u32, u32, bool) {
  let mut svg_width = DEFAULT_WIDTH;
  let mut svg_height = DEFAULT_WIDTH; // Square default for field plots
  let mut full_width = false;
  for opt in &args[start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(name) if name == "ImageSize")
      && let Some((w, h, fw)) =
        parse_image_size(replacement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
    {
      svg_width = w;
      svg_height = h;
      full_width = fw;
    }
  }
  (svg_width, svg_height, full_width)
}

/// Simple SVG header for plots without plotters axes (ArrayPlot, MatrixPlot).
fn svg_header(w: u32, h: u32, full_width: bool) -> String {
  let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
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

/// Map value to a blue-white-red color
fn value_to_color(v: f64, v_min: f64, v_max: f64) -> (u8, u8, u8) {
  let range = v_max - v_min;
  let t = if range.abs() < f64::EPSILON {
    0.5
  } else {
    ((v - v_min) / range).clamp(0.0, 1.0)
  };

  if t < 0.5 {
    // Blue to white
    let s = t * 2.0;
    let r = (s * 255.0) as u8;
    let g = (s * 255.0) as u8;
    let b = 255;
    (r, g, b)
  } else {
    // White to red
    let s = (t - 0.5) * 2.0;
    let r = 255;
    let g = ((1.0 - s) * 255.0) as u8;
    let b = ((1.0 - s) * 255.0) as u8;
    (r, g, b)
  }
}

/// DensityPlot[f, {x, xmin, xmax}, {y, ymin, ymax}]
pub fn density_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "DensityPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "DensityPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  // Sample grid
  let mut grid = vec![vec![f64::NAN; FIELD_GRID]; FIELD_GRID];
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for i in 0..FIELD_GRID {
    let x = x_min + (i as f64 + 0.5) / FIELD_GRID as f64 * (x_max - x_min);
    for j in 0..FIELD_GRID {
      let y = y_min + (j as f64 + 0.5) / FIELD_GRID as f64 * (y_max - y_min);
      if let Some(v) = evaluate_at_xy(body, &xvar, &yvar, x, y)
        && v.is_finite()
      {
        grid[i][j] = v;
        v_min = v_min.min(v);
        v_max = v_max.max(v);
      }
    }
  }

  // Use plotters for axes, then overlay density cells
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  // Remove closing </svg> to append custom elements
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;

  for i in 0..FIELD_GRID {
    for j in 0..FIELD_GRID {
      let v = grid[i][j];
      if v.is_finite() {
        let (r, g, b) = value_to_color(v, v_min, v_max);
        let sx = area.plot_x0 + i as f64 * cell_w;
        let sy = area.plot_y0 + (FIELD_GRID - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// ContourPlot[f, {x, xmin, xmax}, {y, ymin, ymax}]
/// Uses marching squares to draw contour lines.
pub fn contour_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "ContourPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "ContourPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  let n = FIELD_GRID + 1;

  // Sample grid
  let mut grid = vec![vec![f64::NAN; n]; n];
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for i in 0..n {
    let x = x_min + i as f64 / FIELD_GRID as f64 * (x_max - x_min);
    for j in 0..n {
      let y = y_min + j as f64 / FIELD_GRID as f64 * (y_max - y_min);
      if let Some(v) = evaluate_at_xy(body, &xvar, &yvar, x, y)
        && v.is_finite()
      {
        grid[i][j] = v;
        v_min = v_min.min(v);
        v_max = v_max.max(v);
      }
    }
  }

  if !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  // Generate contour levels
  let num_levels = 12;
  let level_step = (v_max - v_min) / (num_levels + 1) as f64;
  let levels: Vec<f64> = (1..=num_levels)
    .map(|k| v_min + k as f64 * level_step)
    .collect();

  // Use plotters for axes
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;

  // Fill background with density colors
  for i in 0..FIELD_GRID {
    for j in 0..FIELD_GRID {
      let v =
        (grid[i][j] + grid[i + 1][j] + grid[i][j + 1] + grid[i + 1][j + 1])
          / 4.0;
      if v.is_finite() {
        let (r, g, b) = value_to_color(v, v_min, v_max);
        let sx = area.plot_x0 + i as f64 * cell_w;
        let sy = area.plot_y0 + (FIELD_GRID - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  // Marching squares for contour lines
  for &level in &levels {
    let mut segments = Vec::new();
    for i in 0..FIELD_GRID {
      for j in 0..FIELD_GRID {
        let v00 = grid[i][j];
        let v10 = grid[i + 1][j];
        let v01 = grid[i][j + 1];
        let v11 = grid[i + 1][j + 1];
        if !v00.is_finite()
          || !v10.is_finite()
          || !v01.is_finite()
          || !v11.is_finite()
        {
          continue;
        }

        let b00 = v00 >= level;
        let b10 = v10 >= level;
        let b01 = v01 >= level;
        let b11 = v11 >= level;
        let case = (b00 as u8)
          | ((b10 as u8) << 1)
          | ((b01 as u8) << 2)
          | ((b11 as u8) << 3);

        if case == 0 || case == 15 {
          continue;
        }

        let lerp = |va: f64, vb: f64| -> f64 {
          if (vb - va).abs() < f64::EPSILON {
            0.5
          } else {
            (level - va) / (vb - va)
          }
        };

        let sx = |fx: f64| -> f64 { area.plot_x0 + (i as f64 + fx) * cell_w };
        let sy = |fy: f64| -> f64 {
          area.plot_y0 + (FIELD_GRID as f64 - (j as f64 + fy)) * cell_h
        };

        // Edge midpoints (interpolated)
        let bottom = (sx(lerp(v00, v10)), sy(0.0));
        let top = (sx(lerp(v01, v11)), sy(1.0));
        let left = (sx(0.0), sy(lerp(v00, v01)));
        let right = (sx(1.0), sy(lerp(v10, v11)));

        let add_seg = |segs: &mut Vec<((f64, f64), (f64, f64))>,
                       a: (f64, f64),
                       b: (f64, f64)| {
          segs.push((a, b));
        };

        match case {
          1 | 14 => add_seg(&mut segments, bottom, left),
          2 | 13 => add_seg(&mut segments, bottom, right),
          3 | 12 => add_seg(&mut segments, left, right),
          4 | 11 => add_seg(&mut segments, left, top),
          5 => {
            add_seg(&mut segments, bottom, left);
            add_seg(&mut segments, top, right);
          }
          6 | 9 => add_seg(&mut segments, bottom, top),
          7 | 8 => add_seg(&mut segments, right, top),
          10 => {
            add_seg(&mut segments, bottom, right);
            add_seg(&mut segments, left, top);
          }
          _ => {}
        }
      }
    }

    // Draw contour line segments
    let contour_stroke = crate::functions::graphics::theme().text_primary;
    for (a, b) in &segments {
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{contour_stroke}\" stroke-width=\"{}\" stroke-opacity=\"0.7\"/>\n",
        a.0, a.1, b.0, b.1, area.render_width as f64 / 1000.0 * 3.0
      ));
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// RegionPlot[cond, {x, xmin, xmax}, {y, ymin, ymax}]
pub fn region_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "RegionPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "RegionPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  // Use plotters for axes
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / FIELD_GRID as f64;
  let cell_h = area.plot_h / FIELD_GRID as f64;

  let (r, g, b) = (0x5E, 0x81, 0xB5); // Default blue
  for i in 0..FIELD_GRID {
    let x = x_min + (i as f64 + 0.5) / FIELD_GRID as f64 * (x_max - x_min);
    for j in 0..FIELD_GRID {
      let y = y_min + (j as f64 + 0.5) / FIELD_GRID as f64 * (y_max - y_min);
      if evaluate_condition(body, &xvar, &yvar, x, y) {
        let sx = area.plot_x0 + i as f64 * cell_w;
        let sy = area.plot_y0 + (FIELD_GRID - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// VectorPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}]
pub fn vector_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "VectorPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "VectorPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  let x_step = (x_max - x_min) / VECTOR_GRID as f64;
  let y_step = (y_max - y_min) / VECTOR_GRID as f64;

  // Sample vectors and find max magnitude
  let mut vectors = Vec::new();
  let mut max_mag = 0.0_f64;
  for i in 0..=VECTOR_GRID {
    let x = x_min + i as f64 * x_step;
    for j in 0..=VECTOR_GRID {
      let y = y_min + j as f64 * y_step;
      if let Some((vx, vy)) = evaluate_vector(body, &xvar, &yvar, x, y) {
        let mag = (vx * vx + vy * vy).sqrt();
        max_mag = max_mag.max(mag);
        vectors.push((x, y, vx, vy, mag));
      }
    }
  }

  // Use plotters for axes
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  // Extract coordinate transform fields before moving svg
  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;
  let render_w = area.render_width;
  let ax_min = area.x_min;
  let ax_max = area.x_max;
  let ay_min = area.y_min;
  let ay_max = area.y_max;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let to_px = |x: f64, y: f64| -> (f64, f64) {
    let sx = plot_x0 + (x - ax_min) / (ax_max - ax_min) * plot_w;
    let sy = plot_y0 + (ay_max - y) / (ay_max - ay_min) * plot_h;
    (sx, sy)
  };

  let cell_size =
    (plot_w / VECTOR_GRID as f64).min(plot_h / VECTOR_GRID as f64);

  if max_mag > 0.0 {
    let arrow_scale = cell_size * 0.4 / max_mag;
    let stroke_w = render_w as f64 / 1000.0 * 1.5;
    for &(x, y, vx, vy, mag) in &vectors {
      if mag < 1e-15 {
        continue;
      }

      let (sx, sy) = to_px(x, y);
      let dx = vx * arrow_scale;
      let dy = -vy * arrow_scale; // SVG y is flipped

      let ex = sx + dx;
      let ey = sy + dy;

      // Color based on magnitude
      let t = (mag / max_mag).clamp(0.0, 1.0);
      let r = (t * 200.0) as u8 + 50;
      let g = ((1.0 - t) * 150.0) as u8 + 50;
      let b = 100_u8;

      // Arrow shaft
      svg.push_str(&format!(
        "<line x1=\"{sx:.1}\" y1=\"{sy:.1}\" x2=\"{ex:.1}\" y2=\"{ey:.1}\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"{stroke_w:.1}\"/>\n"
      ));

      // Arrowhead
      let len = (dx * dx + dy * dy).sqrt();
      if len > 2.0 {
        let ux = dx / len;
        let uy = dy / len;
        let head_len = len * 0.3;
        let head_w = head_len * 0.4;
        let px = -uy;
        let py = ux;
        let bx1 = ex - ux * head_len + px * head_w;
        let by1 = ey - uy * head_len + py * head_w;
        let bx2 = ex - ux * head_len - px * head_w;
        let by2 = ey - uy * head_len - py * head_w;
        svg.push_str(&format!(
          "<polygon points=\"{ex:.1},{ey:.1} {bx1:.1},{by1:.1} {bx2:.1},{by2:.1}\" fill=\"rgb({r},{g},{b})\"/>\n"
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// StreamPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}]
/// Uses RK4 integration from seed points.
pub fn stream_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "StreamPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "StreamPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  // Use plotters for axes
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;
  let render_w = area.render_width;
  let ax_min = area.x_min;
  let ax_max = area.x_max;
  let ay_min = area.y_min;
  let ay_max = area.y_max;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let to_px = |x: f64, y: f64| -> (f64, f64) {
    let sx = plot_x0 + (x - ax_min) / (ax_max - ax_min) * plot_w;
    let sy = plot_y0 + (ay_max - y) / (ay_max - ay_min) * plot_h;
    (sx, sy)
  };

  // Seed points on a grid
  let seed_n = 8;
  let x_step = (x_max - x_min) / seed_n as f64;
  let y_step = (y_max - y_min) / seed_n as f64;
  let dt = ((x_max - x_min) + (y_max - y_min)) / 2.0 / 200.0;
  let max_steps = 200;
  let stroke_w = render_w as f64 / 1000.0 * 1.5;

  for si in 0..seed_n {
    for sj in 0..seed_n {
      let mut x = x_min + (si as f64 + 0.5) * x_step;
      let mut y = y_min + (sj as f64 + 0.5) * y_step;

      let mut points = Vec::new();
      let (px, py) = to_px(x, y);
      points.push(format!("{:.1},{:.1}", px, py));

      for _ in 0..max_steps {
        // RK4 step
        let (k1x, k1y) =
          evaluate_vector(body, &xvar, &yvar, x, y).unwrap_or((0.0, 0.0));
        let (k2x, k2y) = evaluate_vector(
          body,
          &xvar,
          &yvar,
          x + dt * k1x / 2.0,
          y + dt * k1y / 2.0,
        )
        .unwrap_or((0.0, 0.0));
        let (k3x, k3y) = evaluate_vector(
          body,
          &xvar,
          &yvar,
          x + dt * k2x / 2.0,
          y + dt * k2y / 2.0,
        )
        .unwrap_or((0.0, 0.0));
        let (k4x, k4y) =
          evaluate_vector(body, &xvar, &yvar, x + dt * k3x, y + dt * k3y)
            .unwrap_or((0.0, 0.0));

        x += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
        y += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;

        if x < x_min || x > x_max || y < y_min || y > y_max {
          break;
        }
        let (px, py) = to_px(x, y);
        points.push(format!("{:.1},{:.1}", px, py));
      }

      if points.len() > 1 {
        let color_idx = (si * seed_n + sj) % 6;
        let (r, g, b) = crate::functions::plot::PLOT_COLORS[color_idx];
        svg.push_str(&format!(
          "<polyline points=\"{}\" fill=\"none\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"{stroke_w:.1}\" stroke-opacity=\"0.7\"/>\n",
          points.join(" ")
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// StreamDensityPlot: StreamPlot overlaid on DensityPlot background
pub fn stream_density_plot_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "StreamDensityPlot")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "StreamDensityPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 3);

  let grid_n = 60;

  // Compute magnitude field for density background
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;
  let mut mag_grid = vec![vec![f64::NAN; grid_n]; grid_n];

  for i in 0..grid_n {
    let x = x_min + (i as f64 + 0.5) / grid_n as f64 * (x_max - x_min);
    for j in 0..grid_n {
      let y = y_min + (j as f64 + 0.5) / grid_n as f64 * (y_max - y_min);
      if let Some((vx, vy)) = evaluate_vector(body, &xvar, &yvar, x, y) {
        let mag = (vx * vx + vy * vy).sqrt();
        if mag.is_finite() {
          mag_grid[i][j] = mag;
          v_min = v_min.min(mag);
          v_max = v_max.max(mag);
        }
      }
    }
  }

  // Use plotters for axes
  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let plot_x0 = area.plot_x0;
  let plot_y0 = area.plot_y0;
  let plot_w = area.plot_w;
  let plot_h = area.plot_h;
  let render_w = area.render_width;
  let ax_min = area.x_min;
  let ax_max = area.x_max;
  let ay_min = area.y_min;
  let ay_max = area.y_max;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let to_px = |x: f64, y: f64| -> (f64, f64) {
    let sx = plot_x0 + (x - ax_min) / (ax_max - ax_min) * plot_w;
    let sy = plot_y0 + (ay_max - y) / (ay_max - ay_min) * plot_h;
    (sx, sy)
  };

  let cell_w = plot_w / grid_n as f64;
  let cell_h = plot_h / grid_n as f64;

  // Density background
  for i in 0..grid_n {
    for j in 0..grid_n {
      let v = mag_grid[i][j];
      if v.is_finite() {
        let (r, g, b) = value_to_color(v, v_min, v_max);
        let sx = plot_x0 + i as f64 * cell_w;
        let sy = plot_y0 + (grid_n - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  // Streamlines
  let seed_n = 8;
  let x_step = (x_max - x_min) / seed_n as f64;
  let y_step_seed = (y_max - y_min) / seed_n as f64;
  let dt = ((x_max - x_min) + (y_max - y_min)) / 2.0 / 200.0;
  let max_steps = 200;
  let stroke_w = render_w as f64 / 1000.0 * 1.2;

  for si in 0..seed_n {
    for sj in 0..seed_n {
      let mut x = x_min + (si as f64 + 0.5) * x_step;
      let mut y = y_min + (sj as f64 + 0.5) * y_step_seed;
      let mut points = Vec::new();
      let (px, py) = to_px(x, y);
      points.push(format!("{:.1},{:.1}", px, py));

      for _ in 0..max_steps {
        let (k1x, k1y) =
          evaluate_vector(body, &xvar, &yvar, x, y).unwrap_or((0.0, 0.0));
        let (k2x, k2y) = evaluate_vector(
          body,
          &xvar,
          &yvar,
          x + dt * k1x / 2.0,
          y + dt * k1y / 2.0,
        )
        .unwrap_or((0.0, 0.0));
        let (k3x, k3y) = evaluate_vector(
          body,
          &xvar,
          &yvar,
          x + dt * k2x / 2.0,
          y + dt * k2y / 2.0,
        )
        .unwrap_or((0.0, 0.0));
        let (k4x, k4y) =
          evaluate_vector(body, &xvar, &yvar, x + dt * k3x, y + dt * k3y)
            .unwrap_or((0.0, 0.0));

        x += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0;
        y += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0;

        if x < x_min || x > x_max || y < y_min || y > y_max {
          break;
        }
        let (px, py) = to_px(x, y);
        points.push(format!("{:.1},{:.1}", px, py));
      }

      if points.len() > 1 {
        let stream_stroke = crate::functions::graphics::theme().text_primary;
        svg.push_str(&format!(
          "<polyline points=\"{}\" fill=\"none\" stroke=\"{stream_stroke}\" stroke-width=\"{stroke_w:.1}\" stroke-opacity=\"0.6\"/>\n",
          points.join(" ")
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// ListDensityPlot[data] - density plot from data
/// Accepts either:
///   - a matrix {{z11, z12, ...}, {z21, z22, ...}, ...}
///   - a list of {x, y, z} triples {{x1,y1,z1}, {x2,y2,z2}, ...}
pub fn list_density_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListDensityPlot: first argument must be a list of data".into(),
      ));
    }
  };

  // Determine if input is {x,y,z} triples or a matrix
  // It's triples if every element is a list of exactly 3 numeric items
  let is_triples = rows.iter().all(|r| {
    if let Expr::List(items) = r
      && items.len() == 3
    {
      return items.iter().all(|item| {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        try_eval_to_f64(&v).is_some()
      });
    }
    false
  }) && rows.len() > 1
    && {
      // Check that it's NOT a 3-column matrix by verifying the "x" values are not
      // sequential 1..n (ambiguous case: treat as triples if any x or y is non-integer
      // or doesn't match sequential indices)
      let first_row = if let Expr::List(items) = &rows[0] {
        items
          .iter()
          .filter_map(|item| {
            let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
            try_eval_to_f64(&v)
          })
          .collect::<Vec<_>>()
      } else {
        vec![]
      };
      // If the first inner list has exactly 3 elements, we need to distinguish
      // between a 3-column matrix and triples. Check if ALL inner lists have
      // exactly 3 elements. If the outer list has more than 3 rows, treat as
      // triples. If it has exactly 3 rows and each row has 3 elements, it's
      // ambiguous - Mathematica treats it as a matrix.
      // Simple heuristic: if we have more rows than columns (3), it's triples.
      rows.len() > 3 || {
        // For 3 rows of 3 elements, check if the values look like triples
        // (x values not all the same, y values not all the same)
        let mut xs = std::collections::HashSet::new();
        let mut ys = std::collections::HashSet::new();
        for r in rows {
          if let Expr::List(items) = r {
            if let Some(x) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
            ) {
              xs.insert(x.to_bits());
            }
            if let Some(y) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
            ) {
              ys.insert(y.to_bits());
            }
          }
        }
        // If x or y values are not simply 1..n, treat as triples
        xs.len() > 1 && ys.len() > 1 && first_row[0] != 1.0
      }
    };

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);

  if is_triples {
    list_density_plot_triples(rows, svg_width, svg_height, full_width)
  } else {
    list_density_plot_matrix(rows, svg_width, svg_height, full_width)
  }
}

/// ListDensityPlot from a matrix of z-values
fn list_density_plot_matrix(
  rows: &[Expr],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<Expr, InterpreterError> {
  let mut matrix: Vec<Vec<f64>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let vals: Vec<f64> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          try_eval_to_f64(&v).unwrap_or(f64::NAN)
        })
        .collect();
      for &v in &vals {
        if v.is_finite() {
          v_min = v_min.min(v);
          v_max = v_max.max(v);
        }
      }
      matrix.push(vals);
    }
  }

  if matrix.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(|r| r.len()).max().unwrap_or(1);

  // x ranges from 1 to n_cols, y ranges from 1 to n_rows
  let x_min = 1.0;
  let x_max = n_cols as f64;
  let y_min = 1.0;
  let y_max = n_rows as f64;

  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / n_cols as f64;
  let cell_h = area.plot_h / n_rows as f64;

  // Row 0 is the top (highest y), row n_rows-1 is the bottom (lowest y)
  for (i, row) in matrix.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      if val.is_finite() {
        let (r, g, b) = value_to_color(val, v_min, v_max);
        let sx = area.plot_x0 + j as f64 * cell_w;
        let sy = area.plot_y0 + i as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// ListDensityPlot from {x, y, z} triples using inverse distance weighting
fn list_density_plot_triples(
  rows: &[Expr],
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<Expr, InterpreterError> {
  let mut points: Vec<(f64, f64, f64)> = Vec::new();

  for row in rows {
    if let Expr::List(items) = row
      && items.len() == 3
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
      );
      let z = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[2]).unwrap_or(items[2].clone()),
      );
      if let (Some(x), Some(y), Some(z)) = (x, y, z)
        && x.is_finite()
        && y.is_finite()
        && z.is_finite()
      {
        points.push((x, y, z));
      }
    }
  }

  if points.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
  let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
  let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
  let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
  let z_min = points.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
  let z_max = points.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);

  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  // Interpolate using inverse distance weighting on a grid
  let grid_n = FIELD_GRID;
  let cell_w = area.plot_w / grid_n as f64;
  let cell_h = area.plot_h / grid_n as f64;
  let x_range = x_max - x_min;
  let y_range = y_max - y_min;

  for i in 0..grid_n {
    let gx = x_min + (i as f64 + 0.5) / grid_n as f64 * x_range;
    for j in 0..grid_n {
      let gy = y_min + (j as f64 + 0.5) / grid_n as f64 * y_range;

      // Inverse distance weighting
      let mut w_sum = 0.0;
      let mut z_sum = 0.0;
      let mut exact = None;

      for &(px, py, pz) in &points {
        let dx = gx - px;
        let dy = gy - py;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < 1e-20 {
          exact = Some(pz);
          break;
        }
        let w = 1.0 / dist_sq;
        w_sum += w;
        z_sum += w * pz;
      }

      let z = exact.unwrap_or_else(|| z_sum / w_sum);
      let (r, g, b) = value_to_color(z, z_min, z_max);
      let sx = area.plot_x0 + i as f64 * cell_w;
      let sy = area.plot_y0 + (grid_n - 1 - j) as f64 * cell_h;
      svg.push_str(&format!(
        "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
        cell_w + 0.5, cell_h + 0.5
      ));
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Parse list data and return a grid of z-values with axis ranges.
/// Returns (grid, x_min, x_max, y_min, y_max, v_min, v_max, n_rows, n_cols).
/// Grid is stored as grid[col][row] for marching squares compatibility.
fn parse_list_data_to_grid(
  rows: &[Expr],
  func_name: &str,
) -> Result<
  (Vec<Vec<f64>>, f64, f64, f64, f64, f64, f64, usize, usize),
  InterpreterError,
> {
  // Determine if input is {x,y,z} triples or a matrix
  let is_triples = is_triples_data(rows);

  if is_triples {
    parse_triples_to_grid(rows, func_name)
  } else {
    parse_matrix_to_grid(rows, func_name)
  }
}

/// Check if list data represents {x,y,z} triples
fn is_triples_data(rows: &[Expr]) -> bool {
  rows.iter().all(|r| {
    if let Expr::List(items) = r
      && items.len() == 3
    {
      return items.iter().all(|item| {
        let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
        try_eval_to_f64(&v).is_some()
      });
    }
    false
  }) && rows.len() > 1
    && {
      let first_row = if let Expr::List(items) = &rows[0] {
        items
          .iter()
          .filter_map(|item| {
            let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
            try_eval_to_f64(&v)
          })
          .collect::<Vec<_>>()
      } else {
        vec![]
      };
      rows.len() > 3 || {
        let mut xs = std::collections::HashSet::new();
        let mut ys = std::collections::HashSet::new();
        for r in rows {
          if let Expr::List(items) = r {
            if let Some(x) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
            ) {
              xs.insert(x.to_bits());
            }
            if let Some(y) = try_eval_to_f64(
              &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
            ) {
              ys.insert(y.to_bits());
            }
          }
        }
        xs.len() > 1 && ys.len() > 1 && first_row[0] != 1.0
      }
    }
}

/// Parse a matrix into a grid (grid[col][row] format)
fn parse_matrix_to_grid(
  rows: &[Expr],
  _func_name: &str,
) -> Result<
  (Vec<Vec<f64>>, f64, f64, f64, f64, f64, f64, usize, usize),
  InterpreterError,
> {
  let mut matrix: Vec<Vec<f64>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let vals: Vec<f64> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          try_eval_to_f64(&v).unwrap_or(f64::NAN)
        })
        .collect();
      for &v in &vals {
        if v.is_finite() {
          v_min = v_min.min(v);
          v_max = v_max.max(v);
        }
      }
      matrix.push(vals);
    }
  }

  if matrix.is_empty() {
    return Ok((vec![], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0));
  }

  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(|r| r.len()).max().unwrap_or(1);

  // Convert to grid[col][row] format for marching squares
  // Row 0 is top (highest y), row n_rows-1 is bottom (lowest y)
  let mut grid = vec![vec![f64::NAN; n_rows]; n_cols];
  for (i, row) in matrix.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      if j < n_cols {
        // grid[col][row_from_bottom]
        grid[j][n_rows - 1 - i] = val;
      }
    }
  }

  Ok((
    grid,
    1.0,
    n_cols as f64,
    1.0,
    n_rows as f64,
    v_min,
    v_max,
    n_rows,
    n_cols,
  ))
}

/// Parse {x,y,z} triples into a grid using inverse distance weighting
fn parse_triples_to_grid(
  rows: &[Expr],
  _func_name: &str,
) -> Result<
  (Vec<Vec<f64>>, f64, f64, f64, f64, f64, f64, usize, usize),
  InterpreterError,
> {
  let mut points: Vec<(f64, f64, f64)> = Vec::new();

  for row in rows {
    if let Expr::List(items) = row
      && items.len() == 3
    {
      let x = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[0]).unwrap_or(items[0].clone()),
      );
      let y = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[1]).unwrap_or(items[1].clone()),
      );
      let z = try_eval_to_f64(
        &evaluate_expr_to_expr(&items[2]).unwrap_or(items[2].clone()),
      );
      if let (Some(x), Some(y), Some(z)) = (x, y, z)
        && x.is_finite()
        && y.is_finite()
        && z.is_finite()
      {
        points.push((x, y, z));
      }
    }
  }

  if points.is_empty() {
    return Ok((vec![], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0));
  }

  let x_min = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
  let x_max = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
  let y_min = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
  let y_max = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
  let z_min = points.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
  let z_max = points.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);

  let grid_n = FIELD_GRID;
  let x_range = x_max - x_min;
  let y_range = y_max - y_min;

  // grid[col][row] with IDW interpolation
  let mut grid = vec![vec![f64::NAN; grid_n]; grid_n];
  for i in 0..grid_n {
    let gx = x_min + (i as f64 + 0.5) / grid_n as f64 * x_range;
    for j in 0..grid_n {
      let gy = y_min + (j as f64 + 0.5) / grid_n as f64 * y_range;

      let mut w_sum = 0.0;
      let mut z_sum = 0.0;
      let mut exact = None;

      for &(px, py, pz) in &points {
        let dx = gx - px;
        let dy = gy - py;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < 1e-20 {
          exact = Some(pz);
          break;
        }
        let w = 1.0 / dist_sq;
        w_sum += w;
        z_sum += w * pz;
      }

      grid[i][j] = exact.unwrap_or_else(|| z_sum / w_sum);
    }
  }

  Ok((
    grid, x_min, x_max, y_min, y_max, z_min, z_max, grid_n, grid_n,
  ))
}

/// ListContourPlot[data] - contour plot from data
/// Accepts same data formats as ListDensityPlot.
pub fn list_contour_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) if !items.is_empty() => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListContourPlot: first argument must be a list of data".into(),
      ));
    }
  };

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let (grid, x_min, x_max, y_min, y_max, v_min, v_max, _n_rows, _n_cols) =
    parse_list_data_to_grid(rows, "ListContourPlot")?;

  if grid.is_empty() || !v_min.is_finite() || !v_max.is_finite() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let area = generate_axes_only(
    (x_min, x_max),
    (y_min, y_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let grid_cols = grid.len();
  let grid_rows = grid[0].len();
  let cell_w = area.plot_w / grid_cols as f64;
  let cell_h = area.plot_h / grid_rows as f64;

  // Density background
  for i in 0..grid_cols {
    for j in 0..grid_rows {
      let v = grid[i][j];
      if v.is_finite() {
        let (r, g, b) = value_to_color(v, v_min, v_max);
        let sx = area.plot_x0 + i as f64 * cell_w;
        let sy = area.plot_y0 + (grid_rows - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
          "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
          cell_w + 0.5, cell_h + 0.5
        ));
      }
    }
  }

  // Contour lines using marching squares
  let num_levels = 12;
  let level_step = (v_max - v_min) / (num_levels + 1) as f64;
  let levels: Vec<f64> = (1..=num_levels)
    .map(|k| v_min + k as f64 * level_step)
    .collect();

  let ms_cols = grid_cols.saturating_sub(1);
  let ms_rows = grid_rows.saturating_sub(1);

  for &level in &levels {
    let mut segments = Vec::new();
    for i in 0..ms_cols {
      for j in 0..ms_rows {
        let v00 = grid[i][j];
        let v10 = grid[i + 1][j];
        let v01 = grid[i][j + 1];
        let v11 = grid[i + 1][j + 1];
        if !v00.is_finite()
          || !v10.is_finite()
          || !v01.is_finite()
          || !v11.is_finite()
        {
          continue;
        }

        let b00 = v00 >= level;
        let b10 = v10 >= level;
        let b01 = v01 >= level;
        let b11 = v11 >= level;
        let case = (b00 as u8)
          | ((b10 as u8) << 1)
          | ((b01 as u8) << 2)
          | ((b11 as u8) << 3);

        if case == 0 || case == 15 {
          continue;
        }

        let lerp = |va: f64, vb: f64| -> f64 {
          if (vb - va).abs() < f64::EPSILON {
            0.5
          } else {
            (level - va) / (vb - va)
          }
        };

        let sx = |fx: f64| -> f64 { area.plot_x0 + (i as f64 + fx) * cell_w };
        let sy = |fy: f64| -> f64 {
          area.plot_y0 + (grid_rows as f64 - (j as f64 + fy)) * cell_h
        };

        let bottom = (sx(lerp(v00, v10)), sy(0.0));
        let top = (sx(lerp(v01, v11)), sy(1.0));
        let left = (sx(0.0), sy(lerp(v00, v01)));
        let right = (sx(1.0), sy(lerp(v10, v11)));

        let add_seg = |segs: &mut Vec<((f64, f64), (f64, f64))>,
                       a: (f64, f64),
                       b: (f64, f64)| {
          segs.push((a, b));
        };

        match case {
          1 | 14 => add_seg(&mut segments, bottom, left),
          2 | 13 => add_seg(&mut segments, bottom, right),
          3 | 12 => add_seg(&mut segments, left, right),
          4 | 11 => add_seg(&mut segments, left, top),
          5 => {
            add_seg(&mut segments, bottom, left);
            add_seg(&mut segments, top, right);
          }
          6 | 9 => add_seg(&mut segments, bottom, top),
          7 | 8 => add_seg(&mut segments, right, top),
          10 => {
            add_seg(&mut segments, bottom, right);
            add_seg(&mut segments, left, top);
          }
          _ => {}
        }
      }
    }

    let contour_stroke = crate::functions::graphics::theme().text_primary;
    for (a, b) in &segments {
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{contour_stroke}\" stroke-width=\"{}\" stroke-opacity=\"0.7\"/>\n",
        a.0, a.1, b.0, b.1, area.render_width as f64 / 1000.0 * 3.0
      ));
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// A cell in an ArrayPlot matrix: either a numeric value or an explicit color.
#[derive(Clone)]
enum ArrayCell {
  Value(f64),
  Color(GfxColor),
}

/// Apply a named color function (gradient) to a normalized value t in [0,1].
fn apply_named_color_function(name: &str, t: f64) -> (u8, u8, u8) {
  let t = t.clamp(0.0, 1.0);
  match name {
    "Rainbow" => {
      // Hue-based rainbow: red -> yellow -> green -> cyan -> blue -> violet
      let c = GfxColor::from_hue(t * 0.83, 1.0, 1.0);
      (
        (c.r * 255.0).round() as u8,
        (c.g * 255.0).round() as u8,
        (c.b * 255.0).round() as u8,
      )
    }
    "TemperatureMap" => {
      // Blue -> white -> red
      value_to_color(t, 0.0, 1.0)
    }
    "SunsetColors" => {
      // Dark blue -> red -> yellow -> white
      let (r, g, b) = if t < 0.33 {
        let s = t / 0.33;
        (0.1 + s * 0.7, 0.05 + s * 0.0, 0.3 + s * (-0.2))
      } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (0.8 + s * 0.2, 0.05 + s * 0.55, 0.1 + s * 0.0)
      } else {
        let s = (t - 0.66) / 0.34;
        (1.0, 0.6 + s * 0.4, 0.1 + s * 0.9)
      };
      (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
      )
    }
    "ThermometerColors" => {
      // Blue -> cyan -> green -> yellow -> red
      let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
      } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
      } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
      } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
      };
      (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
      )
    }
    "GreenPinkTones" => {
      // Green -> white -> pink
      if t < 0.5 {
        let s = t * 2.0;
        (
          (s * 255.0).round() as u8,
          ((0.5 + 0.5 * s) * 255.0).round() as u8,
          (s * 255.0).round() as u8,
        )
      } else {
        let s = (t - 0.5) * 2.0;
        (
          255,
          ((1.0 - 0.5 * s) * 255.0).round() as u8,
          ((1.0 - 0.5 * s) * 255.0).round() as u8,
        )
      }
    }
    // Default: grayscale (same as "GrayTones")
    _ => {
      let gray = ((1.0 - t) * 255.0).round() as u8;
      (gray, gray, gray)
    }
  }
}

/// ArrayPlot[{{v11, ...}, ...}] - grayscale grid from matrix values
/// 0 is always white; the maximum value is black.
/// Supports options: ColorRules, Mesh, ColorFunction, ImageSize.
/// Cells can also be explicit color directives (e.g. Pink, Red).
pub fn array_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ArrayPlot: first argument must be a matrix (list of lists)".into(),
      ));
    }
  };

  // Parse options
  let mut color_rules: Vec<(f64, GfxColor)> = Vec::new();
  let mut mesh = false;
  let mut color_function: Option<String> = None;

  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ColorRules" => {
          if let Expr::List(rules) = replacement.as_ref() {
            for rule in rules {
              if let Expr::Rule {
                pattern: rp,
                replacement: rr,
              } = rule
              {
                let rp_eval =
                  evaluate_expr_to_expr(rp).unwrap_or_else(|_| *rp.clone());
                if let Some(val) = try_eval_to_f64(&rp_eval) {
                  let rr_eval =
                    evaluate_expr_to_expr(rr).unwrap_or_else(|_| *rr.clone());
                  if let Some(color) = parse_color(&rr_eval) {
                    color_rules.push((val, color));
                  }
                }
              }
            }
          }
        }
        "Mesh" => match replacement.as_ref() {
          Expr::Identifier(v) if v == "True" || v == "All" => mesh = true,
          _ => {}
        },
        "ColorFunction" => {
          if let Expr::String(s) = replacement.as_ref() {
            color_function = Some(s.clone());
          }
        }
        _ => {}
      }
    }
  }

  // Parse matrix: each cell is either a numeric value or a color directive
  let mut matrix: Vec<Vec<ArrayCell>> = Vec::new();
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let cells: Vec<ArrayCell> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          // Try as color first
          if let Some(color) = parse_color(&v) {
            ArrayCell::Color(color)
          } else {
            ArrayCell::Value(try_eval_to_f64(&v).unwrap_or(0.0))
          }
        })
        .collect();
      for cell in &cells {
        if let ArrayCell::Value(v) = cell
          && v.is_finite()
        {
          v_max = v_max.max(*v);
        }
      }
      matrix.push(cells);
    }
  }

  if matrix.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(|r| r.len()).max().unwrap_or(1);

  // Aspect ratio: match the matrix shape (wider if more cols, taller if more rows)
  let aspect = n_rows as f64 / n_cols as f64;
  let (svg_width, svg_height) = if aspect <= 1.0 {
    (svg_width, (svg_width as f64 * aspect).round() as u32)
  } else {
    ((svg_height as f64 / aspect).round() as u32, svg_height)
  };

  let s = RESOLUTION_SCALE;
  let render_width = svg_width * s;
  let render_height = svg_height * s;

  let (bg_color, _, _, _, _) = plot_theme();
  let border_color = RGBColor(0x66, 0x66, 0x66);

  let mut buf = String::new();
  {
    let root = SVGBackend::with_string(&mut buf, (render_width, render_height))
      .into_drawing_area();
    root.fill(&bg_color).map_err(|e| {
      InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
    })?;

    let cell_w = render_width as f64 / n_cols as f64;
    let cell_h = render_height as f64 / n_rows as f64;

    for (i, row) in matrix.iter().enumerate() {
      for (j, cell) in row.iter().enumerate() {
        let (r, g, b) = match cell {
          ArrayCell::Color(color) => (
            (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
          ),
          ArrayCell::Value(val) => {
            // Check ColorRules first
            let mut found = None;
            for (rule_val, rule_color) in &color_rules {
              if (*val - *rule_val).abs() < f64::EPSILON {
                found = Some((
                  (rule_color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (rule_color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (rule_color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                ));
                break;
              }
            }
            found.unwrap_or_else(|| {
              let t = if v_max.abs() < f64::EPSILON {
                0.0
              } else {
                (*val / v_max).clamp(0.0, 1.0)
              };
              if let Some(ref cf_name) = color_function {
                apply_named_color_function(cf_name, t)
              } else {
                let gray = ((1.0 - t) * 255.0).round() as u8;
                (gray, gray, gray)
              }
            })
          }
        };

        let x0 = (j as f64 * cell_w).round() as i32;
        let y0 = (i as f64 * cell_h).round() as i32;
        let x1 = ((j + 1) as f64 * cell_w).round() as i32;
        let y1 = ((i + 1) as f64 * cell_h).round() as i32;

        root
          .draw(&Rectangle::new(
            [(x0, y0), (x1, y1)],
            RGBColor(r, g, b).filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
    }

    // Draw mesh grid lines between cells
    if mesh {
      let mesh_color = RGBColor(0x66, 0x66, 0x66);
      let line_w = (s as i32).max(1);
      // Vertical lines
      for j in 1..n_cols {
        let x = (j as f64 * cell_w).round() as i32;
        root
          .draw(&Rectangle::new(
            [
              (x - line_w / 2, 0),
              (x + (line_w + 1) / 2, render_height as i32),
            ],
            mesh_color.filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
      // Horizontal lines
      for i in 1..n_rows {
        let y = (i as f64 * cell_h).round() as i32;
        root
          .draw(&Rectangle::new(
            [
              (0, y - line_w / 2),
              (render_width as i32, y + (line_w + 1) / 2),
            ],
            mesh_color.filled(),
          ))
          .map_err(|e| {
            InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
          })?;
      }
    }

    // Draw border around the entire plot
    let bw = s as i32;
    let rw = render_width as i32;
    let rh = render_height as i32;
    for rect in [
      [(0, 0), (rw, bw)],       // top
      [(0, rh - bw), (rw, rh)], // bottom
      [(0, 0), (bw, rh)],       // left
      [(rw - bw, 0), (rw, rh)], // right
    ] {
      root
        .draw(&Rectangle::new(rect, border_color.filled()))
        .map_err(|e| {
          InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
        })?;
    }

    root.present().map_err(|e| {
      InterpreterError::EvaluationError(format!("ArrayPlot: {e}"))
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

  Ok(crate::graphics_result(buf))
}

/// MatrixPlot[matrix] - like ArrayPlot with automatic color scaling
pub fn matrix_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let rows = match &data {
    Expr::List(items) => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "MatrixPlot: first argument must be a matrix".into(),
      ));
    }
  };

  let mut matrix: Vec<Vec<f64>> = Vec::new();
  let mut v_min = f64::INFINITY;
  let mut v_max = f64::NEG_INFINITY;

  for row in rows {
    if let Expr::List(items) = row {
      let vals: Vec<f64> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          try_eval_to_f64(&v).unwrap_or(0.0)
        })
        .collect();
      for &v in &vals {
        if v.is_finite() {
          v_min = v_min.min(v);
          v_max = v_max.max(v);
        }
      }
      matrix.push(vals);
    }
  }

  if matrix.is_empty() {
    return Ok(crate::graphics_result(
      "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>".to_string(),
    ));
  }

  let (svg_width, svg_height, full_width) = parse_field_options(args, 1);
  let n_rows = matrix.len();
  let n_cols = matrix.iter().map(|r| r.len()).max().unwrap_or(1);
  let w = svg_width as f64;
  let h = svg_height as f64;
  let cell_w = w / n_cols as f64;
  let cell_h = h / n_rows as f64;

  let mut svg = svg_header(svg_width, svg_height, full_width);

  for (i, row) in matrix.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
      let (r, g, b) = value_to_color(val, v_min, v_max);
      let sx = j as f64 * cell_w;
      let sy = i as f64 * cell_h;
      svg.push_str(&format!(
        "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
        cell_w + 0.5, cell_h + 0.5
      ));
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}

/// Parse a complex iterator: {z, zmin, zmax} where zmin/zmax are complex numbers.
/// Returns (var_name, re_min, re_max, im_min, im_max).
fn parse_complex_iterator(
  spec: &Expr,
  label: &str,
) -> Result<(String, f64, f64, f64, f64), InterpreterError> {
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
      let zmin_expr = evaluate_expr_to_expr(&items[1])?;
      let zmax_expr = evaluate_expr_to_expr(&items[2])?;
      let (re_min, im_min) =
        try_extract_complex_f64(&zmin_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{label}: cannot evaluate iterator min to a complex number"
          ))
        })?;
      let (re_max, im_max) =
        try_extract_complex_f64(&zmax_expr).ok_or_else(|| {
          InterpreterError::EvaluationError(format!(
            "{label}: cannot evaluate iterator max to a complex number"
          ))
        })?;
      Ok((var, re_min, re_max, im_min, im_max))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{label}: iterator must be {{var, zmin, zmax}}"
    ))),
  }
}

/// Evaluate a complex function f(z) at the point (re, im).
/// Substitutes z with the complex value and extracts (re, im) of the result.
fn evaluate_complex_at(
  body: &Expr,
  zvar: &str,
  re: f64,
  im: f64,
) -> Option<(f64, f64)> {
  let z_val = build_complex_float_expr(re, im);
  let sub = substitute_var(body, zvar, &z_val);
  let result = evaluate_expr_to_expr(&sub).ok()?;
  try_extract_complex_f64(&result)
}

/// Convert complex value to domain coloring HSB color.
/// Hue is determined by Arg(z), brightness by Abs(z).
fn complex_to_rgb(re: f64, im: f64) -> (u8, u8, u8) {
  let arg = im.atan2(re); // -PI to PI
  let abs = (re * re + im * im).sqrt();

  // Hue: map argument from [-PI, PI] to [0, 1]
  let hue = (arg + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);

  // Brightness: use a smooth function that maps [0, inf) to [0, 1)
  // Following Mathematica's approach: brightness varies with magnitude
  let brightness = 1.0 - 1.0 / (1.0 + abs.powf(0.3));

  // Saturation: high saturation, slightly reduced for very large/small magnitudes
  let saturation = 0.8 + 0.2 * (1.0 - (2.0 * brightness - 1.0).abs());

  hsb_to_rgb(hue, saturation, brightness)
}

/// Convert HSB (all in [0,1]) to RGB.
fn hsb_to_rgb(h: f64, s: f64, b: f64) -> (u8, u8, u8) {
  let h = h.fract();
  let h = if h < 0.0 { h + 1.0 } else { h };
  let hi = (h * 6.0).floor() as u32 % 6;
  let f = h * 6.0 - hi as f64;
  let p = b * (1.0 - s);
  let q = b * (1.0 - f * s);
  let t = b * (1.0 - (1.0 - f) * s);
  let (r, g, bl) = match hi {
    0 => (b, t, p),
    1 => (q, b, p),
    2 => (p, b, t),
    3 => (p, q, b),
    4 => (t, p, b),
    _ => (b, p, q),
  };
  (
    (r * 255.0).round() as u8,
    (g * 255.0).round() as u8,
    (bl * 255.0).round() as u8,
  )
}

/// ComplexPlot[f, {z, zmin, zmax}]
/// Domain coloring visualization of a complex function.
pub fn complex_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let body = &args[0];
  let (zvar, re_min, re_max, im_min, im_max) =
    parse_complex_iterator(&args[1], "ComplexPlot")?;
  let (svg_width, svg_height, full_width) = parse_field_options(args, 2);

  let grid_size = FIELD_GRID;

  // Use plotters for axes (Re on x-axis, Im on y-axis)
  let area = generate_axes_only(
    (re_min, re_max),
    (im_min, im_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  let mut svg = area.svg;
  // Remove closing </svg> to append custom elements
  if let Some(pos) = svg.rfind("</svg>") {
    svg.truncate(pos);
  }

  let cell_w = area.plot_w / grid_size as f64;
  let cell_h = area.plot_h / grid_size as f64;

  for i in 0..grid_size {
    let re = re_min + (i as f64 + 0.5) / grid_size as f64 * (re_max - re_min);
    for j in 0..grid_size {
      let im = im_min + (j as f64 + 0.5) / grid_size as f64 * (im_max - im_min);
      if let Some((fre, fim)) = evaluate_complex_at(body, &zvar, re, im)
        && fre.is_finite()
        && fim.is_finite()
      {
        let (r, g, b) = complex_to_rgb(fre, fim);
        let sx = area.plot_x0 + i as f64 * cell_w;
        // Flip y: higher im values at top
        let sy = area.plot_y0 + (grid_size - 1 - j) as f64 * cell_h;
        svg.push_str(&format!(
            "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"none\"/>\n",
            cell_w + 0.5, cell_h + 0.5
          ));
      }
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics_result(svg))
}
