use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  format_tick, nice_step, parse_image_size, substitute_var,
};
use crate::syntax::Expr;

const DEFAULT_SIZE: u32 = 360;
const GRID_N: usize = 50;
/// Matches Mathematica's default BoxRatios {1, 1, 0.4} for Plot3D.
const Z_SCALE: f64 = 0.4;

// --- 3D math types and helpers ---

#[derive(Clone, Copy)]
struct Point3D {
  x: f64,
  y: f64,
  z: f64,
}

struct Camera {
  azimuth: f64,
  elevation: f64,
}

impl Default for Camera {
  fn default() -> Self {
    // Matches Mathematica's default ViewPoint {1.3, -2.4, 2.0}
    Camera {
      azimuth: -1.07,  // ~-61 degrees
      elevation: 0.63, // ~36 degrees
    }
  }
}

struct Triangle {
  projected: [(f64, f64); 3],
  depth: f64,
  color: (u8, u8, u8),
}

/// Orthographic projection from a camera at spherical (azimuth, elevation).
/// Returns (screen_x, screen_y) in projected coordinates.
fn project(p: Point3D, cam: &Camera) -> (f64, f64) {
  let (sa, ca) = cam.azimuth.sin_cos();
  let (se, ce) = cam.elevation.sin_cos();

  // Right vector: (-sin(a), cos(a), 0)
  let screen_x = -p.x * sa + p.y * ca;
  // Up vector: (-cos(a)*sin(e), -sin(a)*sin(e), cos(e))
  let screen_y = -p.x * ca * se - p.y * sa * se + p.z * ce;

  (screen_x, screen_y)
}

/// Depth along the camera direction. Positive = further from viewer.
/// Used for painter's algorithm: draw largest depth first.
fn depth(p: Point3D, cam: &Camera) -> f64 {
  let (sa, ca) = cam.azimuth.sin_cos();
  let (se, ce) = cam.elevation.sin_cos();
  // Negate projection onto camera direction so positive = further
  -(p.x * ce * ca + p.y * ce * sa + p.z * se)
}

/// Cross product for triangle normal (used for lighting)
fn triangle_normal(v0: Point3D, v1: Point3D, v2: Point3D) -> [f64; 3] {
  let ux = v1.x - v0.x;
  let uy = v1.y - v0.y;
  let uz = v1.z - v0.z;
  let vx = v2.x - v0.x;
  let vy = v2.y - v0.y;
  let vz = v2.z - v0.z;
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  let len = (nx * nx + ny * ny + nz * nz).sqrt();
  if len < 1e-15 {
    [0.0, 0.0, 1.0]
  } else {
    [nx / len, ny / len, nz / len]
  }
}

/// Height-based color: blue at bottom to green in middle to orange at top
fn height_color(z_norm: f64) -> (u8, u8, u8) {
  let t = z_norm.clamp(0.0, 1.0);
  if t < 0.5 {
    let s = t * 2.0;
    let r = (0.37 * (1.0 - s) + 0.39 * s) * 255.0;
    let g = (0.51 * (1.0 - s) + 0.69 * s) * 255.0;
    let b = (0.71 * (1.0 - s) + 0.29 * s) * 255.0;
    (r as u8, g as u8, b as u8)
  } else {
    let s = (t - 0.5) * 2.0;
    let r = (0.39 * (1.0 - s) + 0.88 * s) * 255.0;
    let g = (0.69 * (1.0 - s) + 0.58 * s) * 255.0;
    let b = (0.29 * (1.0 - s) + 0.17 * s) * 255.0;
    (r as u8, g as u8, b as u8)
  }
}

/// Apply diffuse + ambient lighting
fn apply_lighting(color: (u8, u8, u8), normal: [f64; 3]) -> (u8, u8, u8) {
  // Light direction: upper-left-front (normalized)
  let lx = 0.4_f64;
  let ly = -0.5_f64;
  let lz = 0.76_f64;
  let len = (lx * lx + ly * ly + lz * lz).sqrt();
  let light = [lx / len, ly / len, lz / len];

  let dot = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
  let diffuse = dot.abs(); // use abs to light both sides

  let ambient = 0.35;
  let intensity = (ambient + (1.0 - ambient) * diffuse).clamp(0.0, 1.0);

  let r = (color.0 as f64 * intensity).round() as u8;
  let g = (color.1 as f64 * intensity).round() as u8;
  let b = (color.2 as f64 * intensity).round() as u8;
  (r, g, b)
}

/// Evaluate the function body at given (x, y) values
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
            "Plot3D: {label} iterator variable must be a symbol"
          )));
        }
      };
      let min_expr = evaluate_expr_to_expr(&items[1])?;
      let max_expr = evaluate_expr_to_expr(&items[2])?;
      let min_val = try_eval_to_f64(&min_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Plot3D: cannot evaluate {label} iterator min to a number"
        ))
      })?;
      let max_val = try_eval_to_f64(&max_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "Plot3D: cannot evaluate {label} iterator max to a number"
        ))
      })?;
      Ok((var, min_val, max_val))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Plot3D: {label} iterator must be {{var, min, max}}"
    ))),
  }
}

/// Implementation of Plot3D[f, {x, xmin, xmax}, {y, ymin, ymax}]
pub fn plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
            "Plot3D requires at least 3 arguments: Plot3D[f, {x, xmin, xmax}, {y, ymin, ymax}]"
                .into(),
        ));
  }

  let body = &args[0];

  // Parse iterators
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "first")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "second")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut show_mesh = true;
  let mut z_clip: Option<(f64, f64)> = None;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) = parse_image_size(replacement) {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          if matches!(replacement.as_ref(), Expr::Identifier(n) if n == "None")
          {
            show_mesh = false;
          }
        }
        Expr::Identifier(name) if name == "PlotRange" => {
          if let Expr::List(items) = replacement.as_ref()
            && items.len() == 2
          {
            let lo = try_eval_to_f64(&evaluate_expr_to_expr(&items[0])?);
            let hi = try_eval_to_f64(&evaluate_expr_to_expr(&items[1])?);
            if let (Some(lo), Some(hi)) = (lo, hi) {
              z_clip = Some((lo, hi));
            }
          }
        }
        _ => {}
      }
    }
  }

  // Collect function bodies: single function or list of functions
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

  let camera = Camera::default();
  let x_step = (x_max - x_min) / GRID_N as f64;
  let y_step = (y_max - y_min) / GRID_N as f64;

  // Phase 1: Sample all grids and compute global z range
  let mut grids: Vec<Vec<Vec<f64>>> = Vec::with_capacity(bodies.len());
  let mut global_z_min = f64::INFINITY;
  let mut global_z_max = f64::NEG_INFINITY;

  for func_body in &bodies {
    let mut grid = vec![vec![f64::NAN; GRID_N + 1]; GRID_N + 1];
    for i in 0..=GRID_N {
      let xval = x_min + i as f64 * x_step;
      for j in 0..=GRID_N {
        let yval = y_min + j as f64 * y_step;
        if let Some(z) = evaluate_at_xy(func_body, &xvar, &yvar, xval, yval)
          && z.is_finite()
        {
          grid[i][j] = z;
          global_z_min = global_z_min.min(z);
          global_z_max = global_z_max.max(z);
        }
      }
    }
    grids.push(grid);
  }

  if !global_z_min.is_finite() || !global_z_max.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "Plot3D: function produced no finite values in the given range".into(),
    ));
  }

  // Use PlotRange if specified, otherwise the global data range
  let (z_lo, z_hi) = z_clip.unwrap_or((global_z_min, global_z_max));
  let z_range = if (z_hi - z_lo).abs() < 1e-15 {
    1.0
  } else {
    z_hi - z_lo
  };

  // Phase 2: Build triangles using the shared z range
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for grid in &grids {
    for i in 0..GRID_N {
      for j in 0..GRID_N {
        let z00 = grid[i][j];
        let z10 = grid[i + 1][j];
        let z01 = grid[i][j + 1];
        let z11 = grid[i + 1][j + 1];

        let nx = |ii: usize| -> f64 { (ii as f64 / GRID_N as f64) * 2.0 - 1.0 };
        let ny = |jj: usize| -> f64 { (jj as f64 / GRID_N as f64) * 2.0 - 1.0 };
        let nz =
          |z: f64| -> f64 { ((z - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE };

        // Triangle 1: (i,j), (i+1,j), (i,j+1)
        if z00.is_finite() && z10.is_finite() && z01.is_finite() {
          let cz00 = z00.clamp(z_lo, z_hi);
          let cz10 = z10.clamp(z_lo, z_hi);
          let cz01 = z01.clamp(z_lo, z_hi);

          let v0 = Point3D {
            x: nx(i),
            y: ny(j),
            z: nz(cz00),
          };
          let v1 = Point3D {
            x: nx(i + 1),
            y: ny(j),
            z: nz(cz10),
          };
          let v2 = Point3D {
            x: nx(i),
            y: ny(j + 1),
            z: nz(cz01),
          };

          let avg_z_norm = ((cz00 - z_lo) / z_range
            + (cz10 - z_lo) / z_range
            + (cz01 - z_lo) / z_range)
            / 3.0;
          let base_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let color = apply_lighting(base_color, normal);

          let p0 = project(v0, &camera);
          let p1 = project(v1, &camera);
          let p2 = project(v2, &camera);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };

          all_triangles.push(Triangle {
            projected: [p0, p1, p2],
            depth: depth(center, &camera),
            color,
          });
        }

        // Triangle 2: (i+1,j+1), (i,j+1), (i+1,j)
        if z11.is_finite() && z01.is_finite() && z10.is_finite() {
          let cz11 = z11.clamp(z_lo, z_hi);
          let cz01 = z01.clamp(z_lo, z_hi);
          let cz10 = z10.clamp(z_lo, z_hi);

          let v0 = Point3D {
            x: nx(i + 1),
            y: ny(j + 1),
            z: nz(cz11),
          };
          let v1 = Point3D {
            x: nx(i),
            y: ny(j + 1),
            z: nz(cz01),
          };
          let v2 = Point3D {
            x: nx(i + 1),
            y: ny(j),
            z: nz(cz10),
          };

          let avg_z_norm = ((cz11 - z_lo) / z_range
            + (cz01 - z_lo) / z_range
            + (cz10 - z_lo) / z_range)
            / 3.0;
          let base_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let color = apply_lighting(base_color, normal);

          let p0 = project(v0, &camera);
          let p1 = project(v1, &camera);
          let p2 = project(v2, &camera);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };

          all_triangles.push(Triangle {
            projected: [p0, p1, p2],
            depth: depth(center, &camera),
            color,
          });
        }
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "Plot3D: function produced no finite values in the given range".into(),
    ));
  }

  // Finalize z range for axis labels
  // z_lo/z_hi already account for PlotRange and flat-range handling
  let (z_axis_min, z_axis_max) = if (z_lo - z_hi).abs() < 1e-15 {
    (z_lo - 0.5, z_hi + 0.5)
  } else {
    (z_lo, z_hi)
  };

  // Painter's algorithm: sort back-to-front (largest depth first)
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg = generate_svg(
    &all_triangles,
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    show_mesh,
  )?;

  crate::capture_graphics(&svg);

  Ok(Expr::Identifier("-Graphics3D-".to_string()))
}

#[allow(clippy::too_many_arguments)]
fn generate_svg(
  triangles: &[Triangle],
  camera: &Camera,
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  show_mesh: bool,
) -> Result<String, InterpreterError> {
  // Find bounding box of all projected points
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;

  for tri in triangles {
    for &(px, py) in &tri.projected {
      px_min = px_min.min(px);
      px_max = px_max.max(px);
      py_min = py_min.min(py);
      py_max = py_max.max(py);
    }
  }

  // Also include the bounding box corners for axes
  let bbox_corners = bounding_box_corners();
  for &corner in &bbox_corners {
    let (px, py) = project(corner, camera);
    px_min = px_min.min(px);
    px_max = px_max.max(px);
    py_min = py_min.min(py);
    py_max = py_max.max(py);
  }

  let p_width = px_max - px_min;
  let p_height = py_max - py_min;
  if p_width < 1e-15 || p_height < 1e-15 {
    return Err(InterpreterError::EvaluationError(
      "Plot3D: degenerate projection".into(),
    ));
  }

  // Compute scale and offset to map projected coords to SVG coords
  let margin = 50.0;
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = (px_min + px_max) / 2.0;
  let p_cy = (py_min + py_max) / 2.0;

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    let sx = cx + (px - p_cx) * scale;
    let sy = cy - (py - p_cy) * scale; // flip Y for SVG
    (sx, sy)
  };

  let mut svg = String::with_capacity(triangles.len() * 120 + 2000);

  if full_width {
    svg.push_str(&format!(
            "<svg width=\"100%\" viewBox=\"0 0 {} {}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            svg_width, svg_height
        ));
  } else {
    svg.push_str(&format!(
            "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            svg_width, svg_height, svg_width, svg_height
        ));
  }

  svg.push_str(&format!(
    "<rect width=\"{}\" height=\"{}\" fill=\"white\"/>\n",
    svg_width, svg_height
  ));

  // Render triangles
  let mesh_attrs = if show_mesh {
    " stroke=\"#00000018\" stroke-width=\"0.5\""
  } else {
    " stroke=\"none\""
  };

  for tri in triangles {
    let (x0, y0) = to_svg(tri.projected[0].0, tri.projected[0].1);
    let (x1, y1) = to_svg(tri.projected[1].0, tri.projected[1].1);
    let (x2, y2) = to_svg(tri.projected[2].0, tri.projected[2].1);
    let (r, g, b) = tri.color;
    svg.push_str(&format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({},{},{})\" {}/>\n",
            x0, y0, x1, y1, x2, y2, r, g, b, mesh_attrs
        ));
  }

  // Draw 3D axes
  draw_axes(&mut svg, camera, &to_svg, x_range, y_range, z_range);

  svg.push_str("</svg>");
  Ok(svg)
}

/// The 8 corners of the normalized [-1,1]^3 bounding box
fn bounding_box_corners() -> [Point3D; 8] {
  [
    Point3D {
      x: -1.0,
      y: -1.0,
      z: -Z_SCALE,
    },
    Point3D {
      x: 1.0,
      y: -1.0,
      z: -Z_SCALE,
    },
    Point3D {
      x: -1.0,
      y: 1.0,
      z: -Z_SCALE,
    },
    Point3D {
      x: 1.0,
      y: 1.0,
      z: -Z_SCALE,
    },
    Point3D {
      x: -1.0,
      y: -1.0,
      z: Z_SCALE,
    },
    Point3D {
      x: 1.0,
      y: -1.0,
      z: Z_SCALE,
    },
    Point3D {
      x: -1.0,
      y: 1.0,
      z: Z_SCALE,
    },
    Point3D {
      x: 1.0,
      y: 1.0,
      z: Z_SCALE,
    },
  ]
}

/// Draw 3D axis lines with ticks and labels
fn draw_axes(
  svg: &mut String,
  camera: &Camera,
  to_svg: &dyn Fn(f64, f64) -> (f64, f64),
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
) {
  let axis_color = "#666666";
  let font_size = 10;

  // Find the bottom corner (z=-1) closest to the viewer (smallest depth)
  let corners = bounding_box_corners();
  let mut min_depth_idx = 0;
  let mut min_depth = f64::INFINITY;
  for (idx, &corner) in corners.iter().enumerate() {
    if corner.z > -Z_SCALE + 0.01 {
      continue;
    }
    let d = depth(corner, camera);
    if d < min_depth {
      min_depth = d;
      min_depth_idx = idx;
    }
  }

  let origin = corners[min_depth_idx];

  // The three axis edges from the closest corner.
  // Each entry: (endpoint, value_range, axis_goes_negative).
  // When the axis direction goes from +1 to -1 in normalized space, the
  // normalized origin represents val_max and we must flip the mapping.
  let x_end = Point3D {
    x: -origin.x,
    y: origin.y,
    z: origin.z,
  };
  let y_end = Point3D {
    x: origin.x,
    y: -origin.y,
    z: origin.z,
  };
  let z_end = Point3D {
    x: origin.x,
    y: origin.y,
    z: Z_SCALE,
  };

  let axes: [(Point3D, (f64, f64), bool); 3] = [
    (x_end, x_range, origin.x > x_end.x),
    (y_end, y_range, origin.y > y_end.y),
    (z_end, z_range, false), // z always goes from -Z_SCALE to +Z_SCALE
  ];

  for &(end, (val_min, val_max), flipped) in &axes {
    let (sx0, sy0) =
      to_svg(project(origin, camera).0, project(origin, camera).1);
    let (sx1, sy1) = to_svg(project(end, camera).0, project(end, camera).1);

    // Axis line
    svg.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\"/>\n",
            sx0, sy0, sx1, sy1, axis_color
        ));

    // Ticks
    let step = nice_step(val_max - val_min, 4);
    if step <= 0.0 {
      continue;
    }
    let first_tick = (val_min / step).ceil() * step;
    let mut tick_val = first_tick;
    while tick_val <= val_max + step * 0.01 {
      // Map tick_val to parameter t along the axis [origin → end]
      let t_raw = if (val_max - val_min).abs() < 1e-15 {
        0.5
      } else {
        ((tick_val - val_min) / (val_max - val_min)).clamp(0.0, 1.0)
      };
      // If axis direction is reversed in normalized space, flip the parameter
      let t = if flipped { 1.0 - t_raw } else { t_raw };

      let pt = Point3D {
        x: origin.x + (end.x - origin.x) * t,
        y: origin.y + (end.y - origin.y) * t,
        z: origin.z + (end.z - origin.z) * t,
      };
      let (tx, ty) = to_svg(project(pt, camera).0, project(pt, camera).1);

      // Perpendicular tick mark
      let dx = sx1 - sx0;
      let dy = sy1 - sy0;
      let len = (dx * dx + dy * dy).sqrt();
      if len > 1.0 {
        let perpx = -dy / len * 4.0;
        let perpy = dx / len * 4.0;

        svg.push_str(&format!(
                    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\"/>\n",
                    tx, ty, tx + perpx, ty + perpy, axis_color
                ));

        let label = format_tick(tick_val);
        svg.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{}\" fill=\"{}\" text-anchor=\"middle\" dominant-baseline=\"middle\">{}</text>\n",
                    tx + perpx * 3.0, ty + perpy * 3.0, font_size, axis_color, label
                ));
      }

      tick_val += step;
    }
  }
}
