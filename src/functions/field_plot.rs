use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  DEFAULT_WIDTH, generate_axes_only, parse_image_size, substitute_var,
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
    matches!(result, Expr::Identifier(s) if s == "True")
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
      && let Some((w, h, fw)) = parse_image_size(replacement)
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
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
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
    for (a, b) in &segments {
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#333\" stroke-width=\"{}\" stroke-opacity=\"0.7\"/>\n",
        a.0, a.1, b.0, b.1, area.render_width as f64 / 1000.0 * 3.0
      ));
    }
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
        svg.push_str(&format!(
          "<polyline points=\"{}\" fill=\"none\" stroke=\"#333\" stroke-width=\"{stroke_w:.1}\" stroke-opacity=\"0.6\"/>\n",
          points.join(" ")
        ));
      }
    }
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}

/// ArrayPlot[{{v11, ...}, ...}] - color grid from matrix, 0=white 1=black (grayscale)
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

  let mut matrix: Vec<Vec<f64>> = Vec::new();
  for row in rows {
    if let Expr::List(items) = row {
      let vals: Vec<f64> = items
        .iter()
        .map(|item| {
          let v = evaluate_expr_to_expr(item).unwrap_or(item.clone());
          try_eval_to_f64(&v).unwrap_or(0.0)
        })
        .collect();
      matrix.push(vals);
    }
  }

  if matrix.is_empty() {
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
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
      let gray = ((1.0 - val.clamp(0.0, 1.0)) * 255.0) as u8;
      let sx = j as f64 * cell_w;
      let sy = i as f64 * cell_h;
      svg.push_str(&format!(
        "<rect x=\"{sx:.1}\" y=\"{sy:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"rgb({gray},{gray},{gray})\" stroke=\"none\"/>\n",
        cell_w + 0.5, cell_h + 0.5
      ));
    }
  }

  svg.push_str("</svg>");
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
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
    crate::capture_graphics("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
    return Ok(Expr::Identifier("-Graphics-".to_string()));
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
  crate::capture_graphics(&svg);
  Ok(Expr::Identifier("-Graphics-".to_string()))
}
