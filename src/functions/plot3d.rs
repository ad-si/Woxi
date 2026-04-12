use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  PLOT_COLORS, format_tick, nice_step, parse_image_size, substitute_var,
};
use crate::syntax::Expr;

const DEFAULT_SIZE: u32 = 360;
const GRID_N: usize = 50;
/// Matches Mathematica's default BoxRatios {1, 1, 0.4} for Plot3D.
const Z_SCALE: f64 = 0.4;

// --- 3D math types and helpers ---

#[derive(Clone, Copy)]
pub(crate) struct Point3D {
  pub x: f64,
  pub y: f64,
  pub z: f64,
}

pub(crate) struct Camera {
  pub azimuth: f64,
  pub elevation: f64,
}

impl Default for Camera {
  fn default() -> Self {
    // Matches Mathematica's default ViewPoint {1.3, -2.4, 2.0}
    Self {
      azimuth: -1.07,  // ~-61 degrees
      elevation: 0.63, // ~36 degrees
    }
  }
}

pub(crate) struct Triangle {
  pub projected: [(f64, f64); 3],
  pub depth: f64,
  pub color: (u8, u8, u8),
  pub opacity: f64,
}

struct MeshLine {
  projected: [(f64, f64); 2],
}

/// A bounding-box edge segment with a depth value so it can be
/// interleaved with surface triangles in the painter's algorithm.
struct BoxEdge {
  endpoints: [Point3D; 2],
  depth: f64,
}

/// Number of grid cells between default mesh lines.
/// GRID_N / MESH_STEP ≈ 16 lines per direction.
const MESH_STEP: usize = 3;

/// Orthographic projection from a camera at spherical (azimuth, elevation).
/// Returns (screen_x, screen_y) in projected coordinates.
pub(crate) fn project(p: Point3D, cam: &Camera) -> (f64, f64) {
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
pub(crate) fn depth(p: Point3D, cam: &Camera) -> f64 {
  let (sa, ca) = cam.azimuth.sin_cos();
  let (se, ce) = cam.elevation.sin_cos();
  // Negate projection onto camera direction so positive = further
  -(p.x * ce * ca + p.y * ce * sa + p.z * se)
}

/// Cross product for triangle normal (used for lighting)
pub(crate) fn triangle_normal(
  v0: Point3D,
  v1: Point3D,
  v2: Point3D,
) -> [f64; 3] {
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

/// Per-surface color based on height, tinted by the surface's palette color.
/// When there are multiple surfaces, each gets a distinct base hue from
/// PLOT_COLORS; the height variation is applied as a lightness shift on top.
fn surface_height_color(
  z_norm: f64,
  surface_idx: usize,
  num_surfaces: usize,
) -> (u8, u8, u8) {
  if num_surfaces <= 1 {
    return height_color(z_norm);
  }
  let base = PLOT_COLORS[surface_idx % PLOT_COLORS.len()];
  // Apply height-based brightness variation: darker at bottom, brighter at top
  let t = z_norm.clamp(0.0, 1.0);
  // Range from 0.6 (dark) to 1.1 (bright, clamped)
  let factor = 0.6 + t * 0.5;
  let r = (base.0 as f64 * factor).round().min(255.0) as u8;
  let g = (base.1 as f64 * factor).round().min(255.0) as u8;
  let b = (base.2 as f64 * factor).round().min(255.0) as u8;
  (r, g, b)
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
pub(crate) fn apply_lighting(
  color: (u8, u8, u8),
  normal: [f64; 3],
) -> (u8, u8, u8) {
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

/// Generate mesh lines from a sampled grid.
///
/// For `Default` mode, draws lines along the x- and y-parameter directions
/// at every `MESH_STEP` grid interval (~15 lines per direction).
/// For `All` mode, draws lines along every grid edge (both directions).
///
/// `grid_n` is the grid resolution, `z_lo`/`z_range` define the z mapping,
/// and `nz` converts a clamped z value to normalized coordinates.
fn generate_mesh_lines(
  grid: &[Vec<f64>],
  grid_n: usize,
  z_lo: f64,
  z_hi: f64,
  z_range: f64,
  mesh_mode: MeshMode,
  camera: &Camera,
) -> Vec<MeshLine> {
  let step = match mesh_mode {
    MeshMode::Default => MESH_STEP,
    // All mode uses triangle-edge strokes, not separate line elements
    MeshMode::All => return Vec::new(),
    MeshMode::None => return Vec::new(),
  };

  let nx = |ii: usize| -> f64 { (ii as f64 / grid_n as f64) * 2.0 - 1.0 };
  let ny = |jj: usize| -> f64 { (jj as f64 / grid_n as f64) * 2.0 - 1.0 };
  let nz = |z: f64| -> f64 { ((z - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE };

  let mut lines = Vec::new();

  // Lines along the x-direction (constant j)
  for j in (0..=grid_n).step_by(step) {
    for i in 0..grid_n {
      let z0 = grid[i][j];
      let z1 = grid[i + 1][j];
      if z0.is_finite() && z1.is_finite() {
        let cz0 = z0.clamp(z_lo, z_hi);
        let cz1 = z1.clamp(z_lo, z_hi);
        let p0 = Point3D {
          x: nx(i),
          y: ny(j),
          z: nz(cz0),
        };
        let p1 = Point3D {
          x: nx(i + 1),
          y: ny(j),
          z: nz(cz1),
        };
        lines.push(MeshLine {
          projected: [project(p0, camera), project(p1, camera)],
        });
      }
    }
  }

  // Lines along the y-direction (constant i)
  for i in (0..=grid_n).step_by(step) {
    for j in 0..grid_n {
      let z0 = grid[i][j];
      let z1 = grid[i][j + 1];
      if z0.is_finite() && z1.is_finite() {
        let cz0 = z0.clamp(z_lo, z_hi);
        let cz1 = z1.clamp(z_lo, z_hi);
        let p0 = Point3D {
          x: nx(i),
          y: ny(j),
          z: nz(cz0),
        };
        let p1 = Point3D {
          x: nx(i),
          y: ny(j + 1),
          z: nz(cz1),
        };
        lines.push(MeshLine {
          projected: [project(p0, camera), project(p1, camera)],
        });
      }
    }
  }

  lines
}

/// Mesh rendering mode for 3D plots.
#[derive(Clone, Copy, PartialEq)]
enum MeshMode {
  /// No mesh lines
  None,
  /// Default: semi-transparent mesh lines
  Default,
  /// Fully visible black mesh lines
  All,
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
  let mut mesh_mode = MeshMode::Default;
  let mut show_axes = true;
  let mut z_clip: Option<(f64, f64)> = None;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
            _ => {}
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
        Expr::Identifier(name) if name == "Boxed" => {
          match replacement.as_ref() {
            Expr::Identifier(s) if s == "False" => show_axes = false,
            Expr::Identifier(s) if s == "True" => show_axes = true,
            _ => {}
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
  let num_surfaces = grids.len();

  for (surface_idx, grid) in grids.iter().enumerate() {
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
          let base_color =
            surface_height_color(avg_z_norm, surface_idx, num_surfaces);
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
            opacity: 1.0,
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
          let base_color =
            surface_height_color(avg_z_norm, surface_idx, num_surfaces);
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
            opacity: 1.0,
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

  // Generate mesh lines from each surface grid
  let mut all_mesh_lines: Vec<MeshLine> = Vec::new();
  for grid in &grids {
    all_mesh_lines.extend(generate_mesh_lines(
      grid, GRID_N, z_lo, z_hi, z_range, mesh_mode, &camera,
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
    &all_mesh_lines,
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}

#[allow(clippy::too_many_arguments)]
fn generate_svg(
  triangles: &[Triangle],
  mesh_lines: &[MeshLine],
  camera: &Camera,
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
  mesh_mode: MeshMode,
  show_axes: bool,
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
  let margin = 25.0;
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

  {
    let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
    svg.push_str(&format!(
      "<rect width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>\n",
      svg_width, svg_height, bg.0, bg.1, bg.2
    ));
  }

  // Build depth-sorted box-edge segments to interleave with surface triangles.
  // Each of the 12 edges is subdivided into small segments so that per-segment
  // depth sorting produces correct occlusion against the surface.
  const EDGE_SUBDIVISIONS: usize = 20;
  let box_edges = if show_axes {
    let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
    let axis_color =
      format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
    let corners = bounding_box_corners();
    let edge_pairs: [(usize, usize); 12] = [
      (0, 1),
      (0, 2),
      (1, 3),
      (2, 3),
      (4, 5),
      (4, 6),
      (5, 7),
      (6, 7),
      (0, 4),
      (1, 5),
      (2, 6),
      (3, 7),
    ];
    let mut segments: Vec<BoxEdge> = Vec::with_capacity(12 * EDGE_SUBDIVISIONS);
    for &(i, j) in &edge_pairs {
      let a = corners[i];
      let b = corners[j];
      for s in 0..EDGE_SUBDIVISIONS {
        let t0 = s as f64 / EDGE_SUBDIVISIONS as f64;
        let t1 = (s + 1) as f64 / EDGE_SUBDIVISIONS as f64;
        let tm = (t0 + t1) * 0.5;
        let lerp = |t: f64| Point3D {
          x: a.x + (b.x - a.x) * t,
          y: a.y + (b.y - a.y) * t,
          z: a.z + (b.z - a.z) * t,
        };
        segments.push(BoxEdge {
          endpoints: [lerp(t0), lerp(t1)],
          depth: depth(lerp(tm), camera),
        });
      }
    }
    segments.sort_by(|a, b| {
      b.depth
        .partial_cmp(&a.depth)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    (segments, axis_color)
  } else {
    (Vec::new(), String::new())
  };
  let (sorted_edges, axis_color) = box_edges;

  // Merge-render triangles and box edges back-to-front (painter's algorithm)
  {
    let mut ei = 0; // index into sorted_edges
    for tri in triangles {
      // Emit any box edges that are further from the camera than this triangle
      while ei < sorted_edges.len() && sorted_edges[ei].depth >= tri.depth {
        let edge = &sorted_edges[ei];
        let (ex0, ey0) = to_svg(
          project(edge.endpoints[0], camera).0,
          project(edge.endpoints[0], camera).1,
        );
        let (ex1, ey1) = to_svg(
          project(edge.endpoints[1], camera).0,
          project(edge.endpoints[1], camera).1,
        );
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
          ex0, ey0, ex1, ey1, axis_color
        ));
        ei += 1;
      }
      // Emit triangle
      let (x0, y0) = to_svg(tri.projected[0].0, tri.projected[0].1);
      let (x1, y1) = to_svg(tri.projected[1].0, tri.projected[1].1);
      let (x2, y2) = to_svg(tri.projected[2].0, tri.projected[2].1);
      let (r, g, b) = tri.color;
      if mesh_mode == MeshMode::All {
        svg.push_str(&format!(
          "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({},{},{})\" stroke=\"#00000060\" stroke-width=\"0.5\"/>\n",
          x0, y0, x1, y1, x2, y2, r, g, b
        ));
      } else {
        svg.push_str(&format!(
          "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({},{},{})\" stroke=\"rgb({},{},{})\" stroke-width=\"0.5\"/>\n",
          x0, y0, x1, y1, x2, y2, r, g, b, r, g, b
        ));
      }
    }
    // Emit remaining box edges (closest to viewer)
    while ei < sorted_edges.len() {
      let edge = &sorted_edges[ei];
      let (ex0, ey0) = to_svg(
        project(edge.endpoints[0], camera).0,
        project(edge.endpoints[0], camera).1,
      );
      let (ex1, ey1) = to_svg(
        project(edge.endpoints[1], camera).0,
        project(edge.endpoints[1], camera).1,
      );
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
        ex0, ey0, ex1, ey1, axis_color
      ));
      ei += 1;
    }
  }

  // Draw mesh lines on top of all triangles so they're fully visible
  for line in mesh_lines {
    let (x0, y0) = to_svg(line.projected[0].0, line.projected[0].1);
    let (x1, y1) = to_svg(line.projected[1].0, line.projected[1].1);
    svg.push_str(&format!(
      "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"#000000a0\" stroke-width=\"0.5\"/>\n",
      x0, y0, x1, y1
    ));
  }

  // Draw axes (ticks, labels) on top of everything
  if show_axes {
    draw_axes(&mut svg, camera, &to_svg, x_range, y_range, z_range);
  }

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
  let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
  let axis_color = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
  let font_size = 13;

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
  // Place the z-axis on the vertical edge that is most to the left or right
  // in the viewport, so labels sit outside the bounding box and don't
  // overlap the plot surface.  Among the 4 bottom corners, pick the one
  // whose projection has the most extreme (leftmost or rightmost) x.
  let bottom_indices: [usize; 4] = [0, 1, 2, 3];
  let mut min_x = f64::INFINITY;
  let mut max_x = f64::NEG_INFINITY;
  let mut min_x_idx = 0usize;
  let mut max_x_idx = 0usize;
  let mut cx_bottom = 0.0;
  for &idx in &bottom_indices {
    let px = project(corners[idx], camera).0;
    cx_bottom += px;
    if px < min_x {
      min_x = px;
      min_x_idx = idx;
    }
    if px > max_x {
      max_x = px;
      max_x_idx = idx;
    }
  }
  cx_bottom /= 4.0;
  // Pick whichever extreme is farther from center
  let z_origin_idx = if (min_x - cx_bottom).abs() >= (max_x - cx_bottom).abs() {
    min_x_idx
  } else {
    max_x_idx
  };
  let z_origin = corners[z_origin_idx];
  let z_end = Point3D {
    x: z_origin.x,
    y: z_origin.y,
    z: Z_SCALE,
  };

  let axes: [(Point3D, Point3D, (f64, f64), bool); 3] = [
    (origin, x_end, x_range, origin.x > x_end.x),
    (origin, y_end, y_range, origin.y > y_end.y),
    (z_origin, z_end, z_range, false), // z always goes from -Z_SCALE to +Z_SCALE
  ];

  // Project the box center so we can orient tick labels outward
  let box_center = Point3D {
    x: 0.0,
    y: 0.0,
    z: 0.0,
  };
  let (cx, cy) =
    to_svg(project(box_center, camera).0, project(box_center, camera).1);

  for &(axis_origin, end, (val_min, val_max), flipped) in &axes {
    let (sx0, sy0) = to_svg(
      project(axis_origin, camera).0,
      project(axis_origin, camera).1,
    );
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
        x: axis_origin.x + (end.x - axis_origin.x) * t,
        y: axis_origin.y + (end.y - axis_origin.y) * t,
        z: axis_origin.z + (end.z - axis_origin.z) * t,
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

        // Place label on the outward side (away from box center)
        let mid_x = (sx0 + sx1) * 0.5;
        let mid_y = (sy0 + sy1) * 0.5;
        let to_center_x = cx - mid_x;
        let to_center_y = cy - mid_y;
        let sign = if perpx * to_center_x + perpy * to_center_y > 0.0 {
          -1.0
        } else {
          1.0
        };

        let label = format_tick(tick_val);
        svg.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{}\" fill=\"{}\" text-anchor=\"middle\" dominant-baseline=\"middle\">{}</text>\n",
                    tx + perpx * 3.0 * sign, ty + perpy * 3.0 * sign, font_size, axis_color, label
                ));
      }

      tick_val += step;
    }
  }
}

// ── VectorPlot3D implementation ─────────────────────────────────────

/// Grid resolution for VectorPlot3D (N x N x N sample points).
const VECTOR3D_GRID: usize = 7;

/// Evaluate a 3-component vector field {vx, vy, vz} at (x, y, z).
fn evaluate_vector3d(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  zvar: &str,
  xval: f64,
  yval: f64,
  zval: f64,
) -> Option<(f64, f64, f64)> {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let sub3 = substitute_var(&sub2, zvar, &Expr::Real(zval));
  let result = evaluate_expr_to_expr(&sub3).ok()?;
  if let Expr::List(items) = &result
    && items.len() == 3
  {
    let vx = try_eval_to_f64(&items[0])?;
    let vy = try_eval_to_f64(&items[1])?;
    let vz = try_eval_to_f64(&items[2])?;
    if vx.is_finite() && vy.is_finite() && vz.is_finite() {
      return Some((vx, vy, vz));
    }
  }
  None
}

/// VectorPlot3D[{vx, vy, vz}, {x, xmin, xmax}, {y, ymin, ymax}, {z, zmin, zmax}]
pub fn vector_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 4 {
    return Err(InterpreterError::EvaluationError(
      "VectorPlot3D requires at least 4 arguments: VectorPlot3D[{vx,vy,vz}, {x,xmin,xmax}, {y,ymin,ymax}, {z,zmin,zmax}]".into(),
    ));
  }

  // Collect one or more vector field bodies
  let bodies: Vec<&Expr> = if let Expr::List(items) = &args[0]
    && !items.is_empty()
    && items.iter().all(|e| matches!(e, Expr::List(_)))
  {
    items.iter().collect()
  } else {
    vec![&args[0]]
  };

  let (xvar, x_min, x_max) = parse_iterator(&args[1], "first")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "second")?;
  let (zvar, z_min, z_max) = parse_iterator(&args[3], "third")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut show_axes = true;
  let mut vector_markers = "Arrow"; // "Arrow" or "Tube"

  for opt in &args[4..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        "Axes" => {
          if let Expr::Identifier(v) = replacement.as_ref()
            && v == "False"
          {
            show_axes = false;
          }
        }
        "VectorMarkers" => {
          if let Expr::String(s) = replacement.as_ref()
            && s == "Tube"
          {
            vector_markers = "Tube";
          }
        }
        _ => {}
      }
    }
  }

  let grid_n = VECTOR3D_GRID;
  let x_step = (x_max - x_min) / grid_n as f64;
  let y_step = (y_max - y_min) / grid_n as f64;
  let z_step = (z_max - z_min) / grid_n as f64;

  // Sample vectors from all fields and find global max magnitude
  struct VecSample {
    /// Position in data space
    px: f64,
    py: f64,
    pz: f64,
    /// Vector components in data space
    vx: f64,
    vy: f64,
    vz: f64,
    mag: f64,
    field_idx: usize,
  }

  let mut samples: Vec<VecSample> = Vec::new();
  let mut max_mag = 0.0_f64;

  for (field_idx, body) in bodies.iter().enumerate() {
    for i in 0..=grid_n {
      let x = x_min + i as f64 * x_step;
      for j in 0..=grid_n {
        let y = y_min + j as f64 * y_step;
        for k in 0..=grid_n {
          let z = z_min + k as f64 * z_step;
          if let Some((vx, vy, vz)) =
            evaluate_vector3d(body, &xvar, &yvar, &zvar, x, y, z)
          {
            let mag = (vx * vx + vy * vy + vz * vz).sqrt();
            max_mag = max_mag.max(mag);
            samples.push(VecSample {
              px: x,
              py: y,
              pz: z,
              vx,
              vy,
              vz,
              mag,
              field_idx,
            });
          }
        }
      }
    }
  }

  if samples.is_empty() || max_mag < 1e-15 {
    return Err(InterpreterError::EvaluationError(
      "VectorPlot3D: vector field produced no finite nonzero vectors".into(),
    ));
  }

  // Map data coordinates to normalized [-1, 1] (x, y) and [-Z_SCALE, Z_SCALE] (z)
  let x_range_d = x_max - x_min;
  let y_range_d = y_max - y_min;
  let z_range_d = z_max - z_min;
  let x_range_d = if x_range_d.abs() < 1e-15 {
    1.0
  } else {
    x_range_d
  };
  let y_range_d = if y_range_d.abs() < 1e-15 {
    1.0
  } else {
    y_range_d
  };
  let z_range_d = if z_range_d.abs() < 1e-15 {
    1.0
  } else {
    z_range_d
  };

  let to_norm = |x: f64, y: f64, z: f64| -> Point3D {
    Point3D {
      x: (x - x_min) / x_range_d * 2.0 - 1.0,
      y: (y - y_min) / y_range_d * 2.0 - 1.0,
      z: ((z - z_min) / z_range_d * 2.0 - 1.0) * Z_SCALE,
    }
  };

  // Arrow scale: normalize so that the longest arrow fits roughly half a grid cell
  let cell_size = (2.0 / grid_n as f64)
    .min(2.0 / grid_n as f64)
    .min(2.0 * Z_SCALE / grid_n as f64);
  let arrow_scale = cell_size * 0.4 / max_mag;

  // Scale factors to convert data-space vector to normalized-space vector
  let sx = 2.0 / x_range_d;
  let sy = 2.0 / y_range_d;
  let sz = 2.0 * Z_SCALE / z_range_d;

  // Build projected arrow data for depth-sorted rendering
  struct ArrowData {
    start: Point3D,
    end: Point3D,
    depth: f64,
    color: (u8, u8, u8),
  }

  let camera = Camera::default();
  let mut arrows: Vec<ArrowData> = Vec::with_capacity(samples.len());

  let num_fields = bodies.len();
  for s in &samples {
    if s.mag < 1e-15 {
      continue;
    }

    let center = to_norm(s.px, s.py, s.pz);
    // Vector in normalized space
    let dvx = s.vx * sx * arrow_scale * 0.5;
    let dvy = s.vy * sy * arrow_scale * 0.5;
    let dvz = s.vz * sz * arrow_scale * 0.5;

    let start = Point3D {
      x: center.x - dvx,
      y: center.y - dvy,
      z: center.z - dvz,
    };
    let end = Point3D {
      x: center.x + dvx,
      y: center.y + dvy,
      z: center.z + dvz,
    };

    // Color: use PLOT_COLORS for multiple fields, magnitude gradient for single
    let color = if num_fields > 1 {
      PLOT_COLORS[s.field_idx % PLOT_COLORS.len()]
    } else {
      let t = (s.mag / max_mag).clamp(0.0, 1.0);
      (
        (t * 200.0) as u8 + 50,
        ((1.0 - t) * 150.0) as u8 + 50,
        100_u8,
      )
    };

    arrows.push(ArrowData {
      start,
      end,
      depth: depth(center, &camera),
      color,
    });
  }

  // Sort arrows back-to-front (painter's algorithm)
  arrows.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  // Compute projected bounding box (include the standard box corners)
  let bbox_corners = bounding_box_corners();
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;

  for &corner in &bbox_corners {
    let (px, py) = project(corner, &camera);
    px_min = px_min.min(px);
    px_max = px_max.max(px);
    py_min = py_min.min(py);
    py_max = py_max.max(py);
  }
  for arrow in &arrows {
    for pt in [&arrow.start, &arrow.end] {
      let (px, py) = project(*pt, &camera);
      px_min = px_min.min(px);
      px_max = px_max.max(px);
      py_min = py_min.min(py);
      py_max = py_max.max(py);
    }
  }

  let p_width = (px_max - px_min).max(1e-15);
  let p_height = (py_max - py_min).max(1e-15);

  let margin = 25.0;
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = (px_min + px_max) / 2.0;
  let p_cy = (py_min + py_max) / 2.0;

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    (cx + (px - p_cx) * scale, cy - (py - p_cy) * scale)
  };

  // Build depth-sorted box edges
  const EDGE_SUBDIVISIONS: usize = 20;
  let (sorted_edges, axis_color) = if show_axes {
    let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
    let ac = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
    let corners = bounding_box_corners();
    let edge_pairs: [(usize, usize); 12] = [
      (0, 1),
      (0, 2),
      (1, 3),
      (2, 3),
      (4, 5),
      (4, 6),
      (5, 7),
      (6, 7),
      (0, 4),
      (1, 5),
      (2, 6),
      (3, 7),
    ];
    let mut segs: Vec<BoxEdge> = Vec::with_capacity(12 * EDGE_SUBDIVISIONS);
    for &(i, j) in &edge_pairs {
      let a = corners[i];
      let b = corners[j];
      for s in 0..EDGE_SUBDIVISIONS {
        let t0 = s as f64 / EDGE_SUBDIVISIONS as f64;
        let t1 = (s + 1) as f64 / EDGE_SUBDIVISIONS as f64;
        let tm = (t0 + t1) * 0.5;
        let lerp = |t: f64| Point3D {
          x: a.x + (b.x - a.x) * t,
          y: a.y + (b.y - a.y) * t,
          z: a.z + (b.z - a.z) * t,
        };
        segs.push(BoxEdge {
          endpoints: [lerp(t0), lerp(t1)],
          depth: depth(lerp(tm), &camera),
        });
      }
    }
    segs.sort_by(|a, b| {
      b.depth
        .partial_cmp(&a.depth)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    (segs, ac)
  } else {
    (Vec::new(), String::new())
  };

  // Build SVG
  let mut svg = String::with_capacity(arrows.len() * 200 + 2000);

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
  {
    let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
    svg.push_str(&format!(
      "<rect width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>\n",
      svg_width, svg_height, bg.0, bg.1, bg.2
    ));
  }

  // Render box edges behind arrows, interleaved by depth
  {
    let mut ei = 0;
    for arrow in &arrows {
      // Emit box edges further from camera than this arrow
      while ei < sorted_edges.len() && sorted_edges[ei].depth >= arrow.depth {
        let edge = &sorted_edges[ei];
        let (ex0, ey0) = to_svg(
          project(edge.endpoints[0], &camera).0,
          project(edge.endpoints[0], &camera).1,
        );
        let (ex1, ey1) = to_svg(
          project(edge.endpoints[1], &camera).0,
          project(edge.endpoints[1], &camera).1,
        );
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
          ex0, ey0, ex1, ey1, axis_color
        ));
        ei += 1;
      }

      // Emit arrow
      let (sx0, sy0) = to_svg(
        project(arrow.start, &camera).0,
        project(arrow.start, &camera).1,
      );
      let (sx1, sy1) =
        to_svg(project(arrow.end, &camera).0, project(arrow.end, &camera).1);
      let (r, g, b) = arrow.color;
      let color_str = format!("rgb({r},{g},{b})");

      if vector_markers == "Tube" {
        // Tube: thicker stroke with rounded caps
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"3\" stroke-linecap=\"round\"/>\n",
          sx0, sy0, sx1, sy1, color_str
        ));
      } else {
        // Arrow: line + arrowhead
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1.2\"/>\n",
          sx0, sy0, sx1, sy1, color_str
        ));

        // Arrowhead
        let dx = sx1 - sx0;
        let dy = sy1 - sy0;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 2.0 {
          let ux = dx / len;
          let uy = dy / len;
          let hl = len * 0.3;
          let hw = hl * 0.4;
          let bx1 = sx1 - ux * hl + (-uy) * hw;
          let by1 = sy1 - uy * hl + ux * hw;
          let bx2 = sx1 - ux * hl - (-uy) * hw;
          let by2 = sy1 - uy * hl - ux * hw;
          svg.push_str(&format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"/>\n",
            sx1, sy1, bx1, by1, bx2, by2, color_str
          ));
        }
      }
    }

    // Emit remaining box edges
    while ei < sorted_edges.len() {
      let edge = &sorted_edges[ei];
      let (ex0, ey0) = to_svg(
        project(edge.endpoints[0], &camera).0,
        project(edge.endpoints[0], &camera).1,
      );
      let (ex1, ey1) = to_svg(
        project(edge.endpoints[1], &camera).0,
        project(edge.endpoints[1], &camera).1,
      );
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
        ex0, ey0, ex1, ey1, axis_color
      ));
      ei += 1;
    }
  }

  // Draw axes on top
  if show_axes {
    draw_axes(
      &mut svg,
      &camera,
      &to_svg,
      (x_min, x_max),
      (y_min, y_max),
      (z_min, z_max),
    );
  }

  svg.push_str("</svg>");
  Ok(crate::graphics3d_result(svg))
}

// ── Graphics3D implementation ────────────────────────────────────────

/// Parse a 3D point {x, y, z} from an expression.
fn parse_point3d(expr: &Expr) -> Option<Point3D> {
  if let Expr::List(items) = expr
    && items.len() == 3
  {
    let x = try_eval_to_f64(&evaluate_expr_to_expr(&items[0]).ok()?)?;
    let y = try_eval_to_f64(&evaluate_expr_to_expr(&items[1]).ok()?)?;
    let z = try_eval_to_f64(&evaluate_expr_to_expr(&items[2]).ok()?)?;
    return Some(Point3D { x, y, z });
  }
  None
}

/// Parse a list of 3D points.
fn parse_point3d_list(expr: &Expr) -> Option<Vec<Point3D>> {
  if let Expr::List(items) = expr {
    let pts: Vec<Point3D> = items.iter().filter_map(parse_point3d).collect();
    if !pts.is_empty() {
      return Some(pts);
    }
  }
  None
}

/// A 3D primitive for Graphics3D
#[derive(Clone)]
struct StyleState3D {
  color: Option<(u8, u8, u8)>,
  opacity: f64,
}

impl Default for StyleState3D {
  fn default() -> Self {
    Self {
      color: None, // None means use default blue
      opacity: 1.0,
    }
  }
}

enum Primitive3D {
  Sphere {
    center: Point3D,
    radius: f64,
    style: StyleState3D,
  },
  Cuboid {
    p_min: Point3D,
    p_max: Point3D,
    style: StyleState3D,
  },
  Polygon3D {
    points: Vec<Point3D>,
    style: StyleState3D,
  },
  Line3D {
    segments: Vec<Vec<Point3D>>,
    style: StyleState3D,
  },
  Point3DPrim {
    points: Vec<Point3D>,
    style: StyleState3D,
  },
  Arrow3D {
    points: Vec<Point3D>,
    style: StyleState3D,
  },
  Cylinder {
    p1: Point3D,
    p2: Point3D,
    radius: f64,
    style: StyleState3D,
  },
  Cone {
    p1: Point3D,
    p2: Point3D,
    radius: f64,
    style: StyleState3D,
  },
}

/// Try to apply a 3D style directive (color or Opacity). Returns true if consumed.
fn apply_3d_directive(expr: &Expr, style: &mut StyleState3D) -> bool {
  use crate::functions::graphics::parse_color;
  use crate::functions::math_ast::expr_to_f64;

  if let Some(color) = parse_color(expr) {
    let r = (color.r.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (color.g.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (color.b.clamp(0.0, 1.0) * 255.0).round() as u8;
    style.color = Some((r, g, b));
    if color.a < 1.0 {
      style.opacity = color.a;
    }
    return true;
  }

  if let Expr::FunctionCall { name, args } = expr
    && name == "Opacity"
    && !args.is_empty()
  {
    if let Some(o) = expr_to_f64(&args[0]) {
      style.opacity = o.clamp(0.0, 1.0);
      if args.len() >= 2
        && let Some(color) = parse_color(&args[1])
      {
        let r = (color.r.clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (color.g.clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (color.b.clamp(0.0, 1.0) * 255.0).round() as u8;
        style.color = Some((r, g, b));
      }
    }
    return true;
  }

  false
}

/// Collect 3D primitives from an expression.
fn collect_3d_primitives(
  expr: &Expr,
  style: &mut StyleState3D,
  prims: &mut Vec<Primitive3D>,
) {
  match expr {
    Expr::List(items) => {
      let saved = style.clone();
      for item in items {
        collect_3d_primitives(item, style, prims);
      }
      *style = saved;
    }
    Expr::Identifier(_) => {
      apply_3d_directive(expr, style);
    }
    Expr::FunctionCall { name, args } => {
      match name.as_str() {
        "Sphere" => {
          let center = if !args.is_empty() {
            parse_point3d(&args[0]).unwrap_or(Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            })
          } else {
            Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            }
          };
          let radius = if args.len() >= 2 {
            try_eval_to_f64(
              &evaluate_expr_to_expr(&args[1]).unwrap_or(args[1].clone()),
            )
            .unwrap_or(1.0)
          } else {
            1.0
          };
          prims.push(Primitive3D::Sphere {
            center,
            radius,
            style: style.clone(),
          });
        }
        "Cuboid" => {
          let p_min = if !args.is_empty() {
            parse_point3d(&args[0]).unwrap_or(Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            })
          } else {
            Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            }
          };
          let p_max = if args.len() >= 2 {
            parse_point3d(&args[1]).unwrap_or(Point3D {
              x: 1.0,
              y: 1.0,
              z: 1.0,
            })
          } else {
            Point3D {
              x: p_min.x + 1.0,
              y: p_min.y + 1.0,
              z: p_min.z + 1.0,
            }
          };
          prims.push(Primitive3D::Cuboid {
            p_min,
            p_max,
            style: style.clone(),
          });
        }
        "Polygon" if !args.is_empty() => {
          if let Some(pts) = parse_point3d_list(&args[0]) {
            prims.push(Primitive3D::Polygon3D {
              points: pts,
              style: style.clone(),
            });
          }
        }
        "Line" if !args.is_empty() => {
          if let Some(pts) = parse_point3d_list(&args[0]) {
            prims.push(Primitive3D::Line3D {
              segments: vec![pts],
              style: style.clone(),
            });
          }
        }
        "Point" if !args.is_empty() => {
          if let Some(pt) = parse_point3d(&args[0]) {
            prims.push(Primitive3D::Point3DPrim {
              points: vec![pt],
              style: style.clone(),
            });
          } else if let Some(pts) = parse_point3d_list(&args[0]) {
            prims.push(Primitive3D::Point3DPrim {
              points: pts,
              style: style.clone(),
            });
          }
        }
        "Arrow" if !args.is_empty() => {
          if let Some(pts) = parse_point3d_list(&args[0]) {
            prims.push(Primitive3D::Arrow3D {
              points: pts,
              style: style.clone(),
            });
          }
        }
        "Cylinder" => {
          let (p1, p2) = if !args.is_empty() {
            if let Expr::List(items) = &args[0] {
              if items.len() == 2 {
                let a = parse_point3d(&items[0]).unwrap_or(Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: -1.0,
                });
                let b = parse_point3d(&items[1]).unwrap_or(Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: 1.0,
                });
                (a, b)
              } else {
                (
                  Point3D {
                    x: 0.0,
                    y: 0.0,
                    z: -1.0,
                  },
                  Point3D {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                  },
                )
              }
            } else {
              (
                Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: -1.0,
                },
                Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: 1.0,
                },
              )
            }
          } else {
            (
              Point3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
              },
              Point3D {
                x: 0.0,
                y: 0.0,
                z: 1.0,
              },
            )
          };
          let radius = if args.len() >= 2 {
            try_eval_to_f64(
              &evaluate_expr_to_expr(&args[1]).unwrap_or(args[1].clone()),
            )
            .unwrap_or(1.0)
          } else {
            1.0
          };
          prims.push(Primitive3D::Cylinder {
            p1,
            p2,
            radius,
            style: style.clone(),
          });
        }
        "Cone" => {
          let (p1, p2) = if !args.is_empty() {
            if let Expr::List(items) = &args[0] {
              if items.len() == 2 {
                let a = parse_point3d(&items[0]).unwrap_or(Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: -1.0,
                });
                let b = parse_point3d(&items[1]).unwrap_or(Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: 1.0,
                });
                (a, b)
              } else {
                (
                  Point3D {
                    x: 0.0,
                    y: 0.0,
                    z: -1.0,
                  },
                  Point3D {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                  },
                )
              }
            } else {
              (
                Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: -1.0,
                },
                Point3D {
                  x: 0.0,
                  y: 0.0,
                  z: 1.0,
                },
              )
            }
          } else {
            (
              Point3D {
                x: 0.0,
                y: 0.0,
                z: -1.0,
              },
              Point3D {
                x: 0.0,
                y: 0.0,
                z: 1.0,
              },
            )
          };
          let radius = if args.len() >= 2 {
            try_eval_to_f64(
              &evaluate_expr_to_expr(&args[1]).unwrap_or(args[1].clone()),
            )
            .unwrap_or(1.0)
          } else {
            1.0
          };
          prims.push(Primitive3D::Cone {
            p1,
            p2,
            radius,
            style: style.clone(),
          });
        }
        _ => {
          // Try as directive first
          if !apply_3d_directive(expr, style) {
            // Recurse into unknown function calls
            for a in args {
              collect_3d_primitives(a, style, prims);
            }
          }
        }
      }
    }
    _ => {}
  }
}

/// Tessellate a sphere into triangles.
fn tessellate_sphere(
  center: &Point3D,
  radius: f64,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n_lat = 16;
  let n_lon = 24;
  let mut tris = Vec::new();
  let pi = std::f64::consts::PI;

  for i in 0..n_lat {
    let theta1 = pi * i as f64 / n_lat as f64;
    let theta2 = pi * (i + 1) as f64 / n_lat as f64;
    for j in 0..n_lon {
      let phi1 = 2.0 * pi * j as f64 / n_lon as f64;
      let phi2 = 2.0 * pi * (j + 1) as f64 / n_lon as f64;

      let p = |theta: f64, phi: f64| -> Point3D {
        Point3D {
          x: center.x + radius * theta.sin() * phi.cos(),
          y: center.y + radius * theta.sin() * phi.sin(),
          z: center.z + radius * theta.cos(),
        }
      };

      let a = p(theta1, phi1);
      let b = p(theta2, phi1);
      let c = p(theta2, phi2);
      let d = p(theta1, phi2);

      tris.push((a, b, c));
      tris.push((a, c, d));
    }
  }
  tris
}

/// Tessellate a cuboid into 12 triangles (2 per face).
pub(crate) fn tessellate_cuboid(
  p_min: &Point3D,
  p_max: &Point3D,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let (x0, y0, z0) = (p_min.x, p_min.y, p_min.z);
  let (x1, y1, z1) = (p_max.x, p_max.y, p_max.z);
  let v = [
    Point3D {
      x: x0,
      y: y0,
      z: z0,
    }, // 0
    Point3D {
      x: x1,
      y: y0,
      z: z0,
    }, // 1
    Point3D {
      x: x1,
      y: y1,
      z: z0,
    }, // 2
    Point3D {
      x: x0,
      y: y1,
      z: z0,
    }, // 3
    Point3D {
      x: x0,
      y: y0,
      z: z1,
    }, // 4
    Point3D {
      x: x1,
      y: y0,
      z: z1,
    }, // 5
    Point3D {
      x: x1,
      y: y1,
      z: z1,
    }, // 6
    Point3D {
      x: x0,
      y: y1,
      z: z1,
    }, // 7
  ];
  vec![
    // Bottom
    (v[0], v[1], v[2]),
    (v[0], v[2], v[3]),
    // Top
    (v[4], v[6], v[5]),
    (v[4], v[7], v[6]),
    // Front
    (v[0], v[5], v[1]),
    (v[0], v[4], v[5]),
    // Back
    (v[2], v[7], v[3]),
    (v[2], v[6], v[7]),
    // Left
    (v[0], v[3], v[7]),
    (v[0], v[7], v[4]),
    // Right
    (v[1], v[5], v[6]),
    (v[1], v[6], v[2]),
  ]
}

/// Tessellate a cylinder along its axis.
fn tessellate_cylinder(
  p1: &Point3D,
  p2: &Point3D,
  radius: f64,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n = 24;
  let pi = std::f64::consts::PI;
  // Axis vector
  let dx = p2.x - p1.x;
  let dy = p2.y - p1.y;
  let dz = p2.z - p1.z;
  let len = (dx * dx + dy * dy + dz * dz).sqrt();
  if len < 1e-15 {
    return vec![];
  }
  let ax = dx / len;
  let ay = dy / len;
  let az = dz / len;

  // Find a perpendicular vector via cross product of axis with (0,0,1) or (0,1,0)
  let (perpx, perpy, perpz) = if az.abs() < 0.9 {
    let cx = ay * 1.0 - az * 0.0;
    let cy = az * 0.0 - ax * 1.0;
    let cz = ax * 0.0 - ay * 0.0;
    let l = (cx * cx + cy * cy + cz * cz).sqrt();
    if l < 1e-15 {
      (1.0, 0.0, 0.0)
    } else {
      (cx / l, cy / l, cz / l)
    }
  } else {
    let cx = ay * 0.0 - az * 1.0;
    let cy = az * 0.0 - ax * 0.0;
    let cz = ax * 1.0 - ay * 0.0;
    let l = (cx * cx + cy * cy + cz * cz).sqrt();
    if l < 1e-15 {
      (0.0, 1.0, 0.0)
    } else {
      (cx / l, cy / l, cz / l)
    }
  };
  // Second perpendicular via cross product
  let binx = ay * perpz - az * perpy;
  let biny = az * perpx - ax * perpz;
  let binz = ax * perpy - ay * perpx;

  let mut tris = Vec::new();
  for i in 0..n {
    let a1 = 2.0 * pi * i as f64 / n as f64;
    let a2 = 2.0 * pi * (i + 1) as f64 / n as f64;
    let c1 = a1.cos();
    let s1 = a1.sin();
    let c2 = a2.cos();
    let s2 = a2.sin();

    let offset1 = (
      radius * (c1 * perpx + s1 * binx),
      radius * (c1 * perpy + s1 * biny),
      radius * (c1 * perpz + s1 * binz),
    );
    let offset2 = (
      radius * (c2 * perpx + s2 * binx),
      radius * (c2 * perpy + s2 * biny),
      radius * (c2 * perpz + s2 * binz),
    );

    let a = Point3D {
      x: p1.x + offset1.0,
      y: p1.y + offset1.1,
      z: p1.z + offset1.2,
    };
    let b = Point3D {
      x: p2.x + offset1.0,
      y: p2.y + offset1.1,
      z: p2.z + offset1.2,
    };
    let c = Point3D {
      x: p2.x + offset2.0,
      y: p2.y + offset2.1,
      z: p2.z + offset2.2,
    };
    let d = Point3D {
      x: p1.x + offset2.0,
      y: p1.y + offset2.1,
      z: p1.z + offset2.2,
    };

    tris.push((a, b, c));
    tris.push((a, c, d));
  }
  tris
}

/// Tessellate a cone.
fn tessellate_cone(
  base: &Point3D,
  tip: &Point3D,
  radius: f64,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n = 24;
  let pi = std::f64::consts::PI;
  let dx = tip.x - base.x;
  let dy = tip.y - base.y;
  let dz = tip.z - base.z;
  let len = (dx * dx + dy * dy + dz * dz).sqrt();
  if len < 1e-15 {
    return vec![];
  }
  let ax = dx / len;
  let ay = dy / len;
  let az = dz / len;

  let (perpx, perpy, perpz) = if az.abs() < 0.9 {
    let cx = ay * 1.0 - az * 0.0;
    let cy = az * 0.0 - ax * 1.0;
    let cz = ax * 0.0 - ay * 0.0;
    let l = (cx * cx + cy * cy + cz * cz).sqrt();
    if l < 1e-15 {
      (1.0, 0.0, 0.0)
    } else {
      (cx / l, cy / l, cz / l)
    }
  } else {
    let cx = ay * 0.0 - az * 1.0;
    let cy = az * 0.0 - ax * 0.0;
    let cz = ax * 1.0 - ay * 0.0;
    let l = (cx * cx + cy * cy + cz * cz).sqrt();
    if l < 1e-15 {
      (0.0, 1.0, 0.0)
    } else {
      (cx / l, cy / l, cz / l)
    }
  };
  let binx = ay * perpz - az * perpy;
  let biny = az * perpx - ax * perpz;
  let binz = ax * perpy - ay * perpx;

  let mut tris = Vec::new();
  for i in 0..n {
    let a1 = 2.0 * pi * i as f64 / n as f64;
    let a2 = 2.0 * pi * (i + 1) as f64 / n as f64;
    let c1 = a1.cos();
    let s1 = a1.sin();
    let c2 = a2.cos();
    let s2 = a2.sin();

    let b1 = Point3D {
      x: base.x + radius * (c1 * perpx + s1 * binx),
      y: base.y + radius * (c1 * perpy + s1 * biny),
      z: base.z + radius * (c1 * perpz + s1 * binz),
    };
    let b2 = Point3D {
      x: base.x + radius * (c2 * perpx + s2 * binx),
      y: base.y + radius * (c2 * perpy + s2 * biny),
      z: base.z + radius * (c2 * perpz + s2 * binz),
    };

    tris.push((*tip, b1, b2));
  }
  tris
}

/// Graphics3D[primitives, options...]
pub fn graphics3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let content = evaluate_expr_to_expr(&args[0])?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut show_box = true;
  let mut background: Option<(u8, u8, u8)> = None;
  for opt in &args[1..] {
    let opt_eval = evaluate_expr_to_expr(opt).unwrap_or(opt.clone());
    if let Expr::Rule {
      pattern,
      replacement,
    } = &opt_eval
      && let Expr::Identifier(name) = pattern.as_ref()
    {
      match name.as_str() {
        "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        "Boxed" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "False" => show_box = false,
          Expr::Identifier(s) if s == "True" => show_box = true,
          _ => {}
        },
        "Background" => {
          if let Some(color) =
            crate::functions::graphics::parse_color(replacement)
          {
            let r = (color.r.clamp(0.0, 1.0) * 255.0).round() as u8;
            let g = (color.g.clamp(0.0, 1.0) * 255.0).round() as u8;
            let b = (color.b.clamp(0.0, 1.0) * 255.0).round() as u8;
            background = Some((r, g, b));
          }
        }
        _ => {}
      }
    }
  }

  // Collect primitives
  let mut prims = Vec::new();
  let mut style3d = StyleState3D::default();
  collect_3d_primitives(&content, &mut style3d, &mut prims);

  if prims.is_empty() {
    // Even with no primitives, return the marker
    let empty_svg = format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"
    );
    return Ok(crate::graphics3d_result(empty_svg));
  }

  // Tessellate all primitives into triangles
  let camera = Camera::default();
  let mut all_triangles: Vec<Triangle> = Vec::new();
  let base_color = (0x5E_u8, 0x81_u8, 0xB5_u8); // Default blue

  for prim in &prims {
    let (tris, prim_style): (Vec<(Point3D, Point3D, Point3D)>, &StyleState3D) =
      match prim {
        Primitive3D::Sphere {
          center,
          radius,
          style,
        } => (tessellate_sphere(center, *radius), style),
        Primitive3D::Cuboid {
          p_min,
          p_max,
          style,
        } => (tessellate_cuboid(p_min, p_max), style),
        Primitive3D::Cylinder {
          p1,
          p2,
          radius,
          style,
        } => (tessellate_cylinder(p1, p2, *radius), style),
        Primitive3D::Cone {
          p1,
          p2,
          radius,
          style,
        } => (tessellate_cone(p1, p2, *radius), style),
        Primitive3D::Polygon3D { points, style } => {
          // Simple fan triangulation
          let t = if points.len() >= 3 {
            (1..points.len() - 1)
              .map(|i| (points[0], points[i], points[i + 1]))
              .collect()
          } else {
            vec![]
          };
          (t, style)
        }
        // Line and Point are handled separately below
        _ => (
          vec![],
          &StyleState3D {
            color: None,
            opacity: 1.0,
          },
        ),
      };
    let prim_color = prim_style.color.unwrap_or(base_color);
    let prim_opacity = prim_style.opacity;

    for (v0, v1, v2) in tris {
      let normal = triangle_normal(v0, v1, v2);
      let color = apply_lighting(prim_color, normal);
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
        opacity: prim_opacity,
      });
    }
  }

  // Painter's algorithm
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  // Compute 3D bounding box of all primitives for the wireframe box
  let mut x3_min = f64::INFINITY;
  let mut x3_max = f64::NEG_INFINITY;
  let mut y3_min = f64::INFINITY;
  let mut y3_max = f64::NEG_INFINITY;
  let mut z3_min = f64::INFINITY;
  let mut z3_max = f64::NEG_INFINITY;

  let extend_3d = |pt: &Point3D,
                   x3_min: &mut f64,
                   x3_max: &mut f64,
                   y3_min: &mut f64,
                   y3_max: &mut f64,
                   z3_min: &mut f64,
                   z3_max: &mut f64| {
    *x3_min = x3_min.min(pt.x);
    *x3_max = x3_max.max(pt.x);
    *y3_min = y3_min.min(pt.y);
    *y3_max = y3_max.max(pt.y);
    *z3_min = z3_min.min(pt.z);
    *z3_max = z3_max.max(pt.z);
  };

  for prim in &prims {
    match prim {
      Primitive3D::Sphere { center, radius, .. } => {
        let r = *radius;
        extend_3d(
          &Point3D {
            x: center.x - r,
            y: center.y - r,
            z: center.z - r,
          },
          &mut x3_min,
          &mut x3_max,
          &mut y3_min,
          &mut y3_max,
          &mut z3_min,
          &mut z3_max,
        );
        extend_3d(
          &Point3D {
            x: center.x + r,
            y: center.y + r,
            z: center.z + r,
          },
          &mut x3_min,
          &mut x3_max,
          &mut y3_min,
          &mut y3_max,
          &mut z3_min,
          &mut z3_max,
        );
      }
      Primitive3D::Cuboid { p_min, p_max, .. } => {
        extend_3d(
          p_min,
          &mut x3_min,
          &mut x3_max,
          &mut y3_min,
          &mut y3_max,
          &mut z3_min,
          &mut z3_max,
        );
        extend_3d(
          p_max,
          &mut x3_min,
          &mut x3_max,
          &mut y3_min,
          &mut y3_max,
          &mut z3_min,
          &mut z3_max,
        );
      }
      Primitive3D::Cylinder { p1, p2, radius, .. }
      | Primitive3D::Cone { p1, p2, radius, .. } => {
        let r = *radius;
        for p in [p1, p2] {
          extend_3d(
            &Point3D {
              x: p.x - r,
              y: p.y - r,
              z: p.z - r,
            },
            &mut x3_min,
            &mut x3_max,
            &mut y3_min,
            &mut y3_max,
            &mut z3_min,
            &mut z3_max,
          );
          extend_3d(
            &Point3D {
              x: p.x + r,
              y: p.y + r,
              z: p.z + r,
            },
            &mut x3_min,
            &mut x3_max,
            &mut y3_min,
            &mut y3_max,
            &mut z3_min,
            &mut z3_max,
          );
        }
      }
      Primitive3D::Polygon3D { points, .. } => {
        for pt in points {
          extend_3d(
            pt,
            &mut x3_min,
            &mut x3_max,
            &mut y3_min,
            &mut y3_max,
            &mut z3_min,
            &mut z3_max,
          );
        }
      }
      Primitive3D::Point3DPrim { points, .. }
      | Primitive3D::Arrow3D { points, .. } => {
        for pt in points {
          extend_3d(
            pt,
            &mut x3_min,
            &mut x3_max,
            &mut y3_min,
            &mut y3_max,
            &mut z3_min,
            &mut z3_max,
          );
        }
      }
      Primitive3D::Line3D { segments, .. } => {
        for seg in segments {
          for pt in seg {
            extend_3d(
              pt,
              &mut x3_min,
              &mut x3_max,
              &mut y3_min,
              &mut y3_max,
              &mut z3_min,
              &mut z3_max,
            );
          }
        }
      }
    }
  }

  // Add some padding to the 3D bounding box
  let pad_x = (x3_max - x3_min) * 0.05;
  let pad_y = (y3_max - y3_min) * 0.05;
  let pad_z = (z3_max - z3_min) * 0.05;
  x3_min -= pad_x;
  x3_max += pad_x;
  y3_min -= pad_y;
  y3_max += pad_y;
  z3_min -= pad_z;
  z3_max += pad_z;

  // Build box corners
  let box_corners = [
    Point3D {
      x: x3_min,
      y: y3_min,
      z: z3_min,
    },
    Point3D {
      x: x3_max,
      y: y3_min,
      z: z3_min,
    },
    Point3D {
      x: x3_min,
      y: y3_max,
      z: z3_min,
    },
    Point3D {
      x: x3_max,
      y: y3_max,
      z: z3_min,
    },
    Point3D {
      x: x3_min,
      y: y3_min,
      z: z3_max,
    },
    Point3D {
      x: x3_max,
      y: y3_min,
      z: z3_max,
    },
    Point3D {
      x: x3_min,
      y: y3_max,
      z: z3_max,
    },
    Point3D {
      x: x3_max,
      y: y3_max,
      z: z3_max,
    },
  ];

  // Compute projected bounding box (include box corners for sizing)
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;

  for tri in &all_triangles {
    for &(px, py) in &tri.projected {
      px_min = px_min.min(px);
      px_max = px_max.max(px);
      py_min = py_min.min(py);
      py_max = py_max.max(py);
    }
  }

  // Also check line/point primitives
  for prim in &prims {
    match prim {
      Primitive3D::Line3D { segments, .. } => {
        for seg in segments {
          for pt in seg {
            let (px, py) = project(*pt, &camera);
            px_min = px_min.min(px);
            px_max = px_max.max(px);
            py_min = py_min.min(py);
            py_max = py_max.max(py);
          }
        }
      }
      Primitive3D::Point3DPrim { points, .. }
      | Primitive3D::Arrow3D { points, .. } => {
        for pt in points {
          let (px, py) = project(*pt, &camera);
          px_min = px_min.min(px);
          px_max = px_max.max(px);
          py_min = py_min.min(py);
          py_max = py_max.max(py);
        }
      }
      _ => {}
    }
  }

  // Include box corners in projected bounding box
  if show_box {
    for &corner in &box_corners {
      let (px, py) = project(corner, &camera);
      px_min = px_min.min(px);
      px_max = px_max.max(px);
      py_min = py_min.min(py);
      py_max = py_max.max(py);
    }
  }

  if !px_min.is_finite() {
    px_min = -1.0;
    px_max = 1.0;
    py_min = -1.0;
    py_max = 1.0;
  }

  let p_width = px_max - px_min;
  let p_height = py_max - py_min;
  let p_width = if p_width < 1e-15 { 1.0 } else { p_width };
  let p_height = if p_height < 1e-15 { 1.0 } else { p_height };

  let margin = 10.0;
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = (px_min + px_max) / 2.0;
  let p_cy = (py_min + py_max) / 2.0;

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    (cx + (px - p_cx) * scale, cy - (py - p_cy) * scale)
  };

  // Build depth-sorted box-edge segments for interleaving with triangles
  const EDGE_SUBDIVISIONS: usize = 20;
  let sorted_edges = if show_box {
    let edge_pairs: [(usize, usize); 12] = [
      (0, 1),
      (0, 2),
      (1, 3),
      (2, 3),
      (4, 5),
      (4, 6),
      (5, 7),
      (6, 7),
      (0, 4),
      (1, 5),
      (2, 6),
      (3, 7),
    ];
    let mut segments: Vec<BoxEdge> = Vec::with_capacity(12 * EDGE_SUBDIVISIONS);
    for &(i, j) in &edge_pairs {
      let a = box_corners[i];
      let b = box_corners[j];
      for s in 0..EDGE_SUBDIVISIONS {
        let t0 = s as f64 / EDGE_SUBDIVISIONS as f64;
        let t1 = (s + 1) as f64 / EDGE_SUBDIVISIONS as f64;
        let tm = (t0 + t1) * 0.5;
        let lerp = |t: f64| Point3D {
          x: a.x + (b.x - a.x) * t,
          y: a.y + (b.y - a.y) * t,
          z: a.z + (b.z - a.z) * t,
        };
        segments.push(BoxEdge {
          endpoints: [lerp(t0), lerp(t1)],
          depth: depth(lerp(tm), &camera),
        });
      }
    }
    segments.sort_by(|a, b| {
      b.depth
        .partial_cmp(&a.depth)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    segments
  } else {
    Vec::new()
  };

  let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
  let axis_color = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);

  let mut svg = String::with_capacity(all_triangles.len() * 120 + 1000);
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
  {
    let (default_bg, _, _, _, _) = crate::functions::plot::plot_theme();
    let bg = background.unwrap_or((default_bg.0, default_bg.1, default_bg.2));
    svg.push_str(&format!(
      "<rect width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>\n",
      svg_width, svg_height, bg.0, bg.1, bg.2
    ));
  }

  // Render triangles interleaved with box edges (painter's algorithm)
  {
    let mut ei = 0;
    for tri in &all_triangles {
      // Emit any box edges further from camera than this triangle
      while ei < sorted_edges.len() && sorted_edges[ei].depth >= tri.depth {
        let edge = &sorted_edges[ei];
        let (ex0, ey0) = to_svg(
          project(edge.endpoints[0], &camera).0,
          project(edge.endpoints[0], &camera).1,
        );
        let (ex1, ey1) = to_svg(
          project(edge.endpoints[1], &camera).0,
          project(edge.endpoints[1], &camera).1,
        );
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
          ex0, ey0, ex1, ey1, axis_color
        ));
        ei += 1;
      }
      // Emit triangle
      let (x0, y0) = to_svg(tri.projected[0].0, tri.projected[0].1);
      let (x1, y1) = to_svg(tri.projected[1].0, tri.projected[1].1);
      let (x2, y2) = to_svg(tri.projected[2].0, tri.projected[2].1);
      let (r, g, b) = tri.color;
      let opacity_attr = if tri.opacity < 1.0 {
        format!(" opacity=\"{}\"", tri.opacity)
      } else {
        String::new()
      };
      svg.push_str(&format!(
        "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({},{},{})\" stroke=\"#00000018\" stroke-width=\"0.5\"{}/>\n",
        x0, y0, x1, y1, x2, y2, r, g, b, opacity_attr
      ));
    }
    // Emit remaining box edges (closest to viewer)
    while ei < sorted_edges.len() {
      let edge = &sorted_edges[ei];
      let (ex0, ey0) = to_svg(
        project(edge.endpoints[0], &camera).0,
        project(edge.endpoints[0], &camera).1,
      );
      let (ex1, ey1) = to_svg(
        project(edge.endpoints[1], &camera).0,
        project(edge.endpoints[1], &camera).1,
      );
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
        ex0, ey0, ex1, ey1, axis_color
      ));
      ei += 1;
    }
  }

  // Render lines and points
  let default_prim_color = crate::functions::graphics::theme().text_primary;
  for prim in &prims {
    match prim {
      Primitive3D::Line3D { segments, style } => {
        let stroke_color = if let Some((r, g, b)) = style.color {
          format!("rgb({r},{g},{b})")
        } else {
          default_prim_color.to_string()
        };
        let opacity_attr = if style.opacity < 1.0 {
          format!(" opacity=\"{}\"", style.opacity)
        } else {
          String::new()
        };
        for seg in segments {
          let pts: Vec<String> = seg
            .iter()
            .map(|p| {
              let (sx, sy) =
                to_svg(project(*p, &camera).0, project(*p, &camera).1);
              format!("{:.1},{:.1}", sx, sy)
            })
            .collect();
          svg.push_str(&format!(
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"{}/>\n",
            pts.join(" "), stroke_color, opacity_attr
          ));
        }
      }
      Primitive3D::Point3DPrim { points, style } => {
        let fill_color = if let Some((r, g, b)) = style.color {
          format!("rgb({r},{g},{b})")
        } else {
          default_prim_color.to_string()
        };
        let opacity_attr = if style.opacity < 1.0 {
          format!(" opacity=\"{}\"", style.opacity)
        } else {
          String::new()
        };
        for pt in points {
          let (sx, sy) =
            to_svg(project(*pt, &camera).0, project(*pt, &camera).1);
          svg.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\"{}/>\n",
            sx, sy, fill_color, opacity_attr
          ));
        }
      }
      Primitive3D::Arrow3D { points, style } if points.len() >= 2 => {
        let stroke_color = if let Some((r, g, b)) = style.color {
          format!("rgb({r},{g},{b})")
        } else {
          default_prim_color.to_string()
        };
        let opacity_attr = if style.opacity < 1.0 {
          format!(" opacity=\"{}\"", style.opacity)
        } else {
          String::new()
        };
        let pts: Vec<String> = points
          .iter()
          .map(|p| {
            let (sx, sy) =
              to_svg(project(*p, &camera).0, project(*p, &camera).1);
            format!("{:.1},{:.1}", sx, sy)
          })
          .collect();
        svg.push_str(&format!(
          "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"{}/>\n",
          pts.join(" "), stroke_color, opacity_attr
        ));
        // Arrowhead
        let last = points.len() - 1;
        let (sx1, sy1) = to_svg(
          project(points[last - 1], &camera).0,
          project(points[last - 1], &camera).1,
        );
        let (sx2, sy2) = to_svg(
          project(points[last], &camera).0,
          project(points[last], &camera).1,
        );
        let dx = sx2 - sx1;
        let dy = sy2 - sy1;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1.0 {
          let ux = dx / len;
          let uy = dy / len;
          let hl = 8.0;
          let hw = 3.0;
          let bx1 = sx2 - ux * hl + (-uy) * hw;
          let by1 = sy2 - uy * hl + ux * hw;
          let bx2 = sx2 - ux * hl - (-uy) * hw;
          let by2 = sy2 - uy * hl - ux * hw;
          svg.push_str(&format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"{}/>\n",
            sx2, sy2, bx1, by1, bx2, by2, stroke_color, opacity_attr
          ));
        }
      }
      _ => {}
    }
  }

  svg.push_str("</svg>");
  Ok(crate::graphics3d_result(svg))
}

/// Implementation of ListPlot3D[data, opts...].
/// Accepts two formats:
/// - `{{x1,y1,z1}, {x2,y2,z2}, ...}` — explicit 3D coordinates
/// - `{{z11,z12,...}, {z21,z22,...}, ...}` — 2D matrix where indices → x,y, values → z
pub fn list_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let data = &args[0];

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut _mesh_mode = MeshMode::Default;

  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => _mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => _mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        _ => {}
      }
    }
  }

  // Evaluate the data argument
  let evaled_data = evaluate_expr_to_expr(data)?;

  // Determine the data format and build a grid
  let (grid, rows, cols, x_min, x_max, y_min, y_max) = match &evaled_data {
    Expr::List(outer) if !outer.is_empty() => {
      // Check first element to determine format
      match &outer[0] {
        // 2D matrix format: {{z11, z12, ...}, {z21, z22, ...}, ...}
        Expr::List(first_row)
          if !first_row.is_empty()
            && first_row.iter().all(|e| !matches!(e, Expr::List(_))) =>
        {
          // Verify all rows are lists of the same length (or close)
          let num_rows = outer.len();
          let num_cols = first_row.len();

          let mut grid = vec![vec![f64::NAN; num_cols]; num_rows];
          for (i, row_expr) in outer.iter().enumerate() {
            if let Expr::List(row) = row_expr {
              for (j, val_expr) in row.iter().enumerate() {
                if j < num_cols
                  && let Some(v) = try_eval_to_f64(val_expr)
                  && v.is_finite()
                {
                  grid[i][j] = v;
                }
              }
            }
          }
          (
            grid,
            num_rows,
            num_cols,
            0.0,
            (num_cols as f64 - 1.0).max(1.0),
            0.0,
            (num_rows as f64 - 1.0).max(1.0),
          )
        }
        // Explicit 3D coordinates: {{x1,y1,z1}, {x2,y2,z2}, ...}
        _ => {
          // Parse as list of {x,y,z} points
          let mut points: Vec<(f64, f64, f64)> = Vec::new();
          for item in outer {
            if let Expr::List(coords) = item
              && coords.len() == 3
            {
              let x = try_eval_to_f64(&coords[0]);
              let y = try_eval_to_f64(&coords[1]);
              let z = try_eval_to_f64(&coords[2]);
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
            return Err(InterpreterError::EvaluationError(
              "ListPlot3D: no valid data points found".into(),
            ));
          }

          // Find x,y range
          let mut xmin = f64::INFINITY;
          let mut xmax = f64::NEG_INFINITY;
          let mut ymin = f64::INFINITY;
          let mut ymax = f64::NEG_INFINITY;
          for &(x, y, _) in &points {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
          }

          // Bin points onto a grid
          let grid_n = 50usize.min(points.len());
          let x_range = if (xmax - xmin).abs() < 1e-15 {
            1.0
          } else {
            xmax - xmin
          };
          let y_range = if (ymax - ymin).abs() < 1e-15 {
            1.0
          } else {
            ymax - ymin
          };

          let mut grid = vec![vec![f64::NAN; grid_n]; grid_n];
          let mut count = vec![vec![0u32; grid_n]; grid_n];

          for &(x, y, z) in &points {
            let i = (((x - xmin) / x_range * (grid_n - 1) as f64).round()
              as usize)
              .min(grid_n - 1);
            let j = (((y - ymin) / y_range * (grid_n - 1) as f64).round()
              as usize)
              .min(grid_n - 1);
            if count[i][j] == 0 {
              grid[i][j] = z;
            } else {
              grid[i][j] += z;
            }
            count[i][j] += 1;
          }

          // Average multiple points in same bin
          for i in 0..grid_n {
            for j in 0..grid_n {
              if count[i][j] > 1 {
                grid[i][j] /= count[i][j] as f64;
              }
            }
          }

          (grid, grid_n, grid_n, xmin, xmax, ymin, ymax)
        }
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ListPlot3D: first argument must be a list of data".into(),
      ));
    }
  };

  // Find z range
  let mut z_min = f64::INFINITY;
  let mut z_max = f64::NEG_INFINITY;
  for row in &grid {
    for &z in row {
      if z.is_finite() {
        z_min = z_min.min(z);
        z_max = z_max.max(z);
      }
    }
  }

  if !z_min.is_finite() || !z_max.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "ListPlot3D: data produced no finite values".into(),
    ));
  }

  let z_range = if (z_max - z_min).abs() < 1e-15 {
    1.0
  } else {
    z_max - z_min
  };

  let camera = Camera::default();
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for i in 0..rows.saturating_sub(1) {
    for j in 0..cols.saturating_sub(1) {
      let z00 = grid[i][j];
      let z10 = if i + 1 < rows {
        grid[i + 1][j]
      } else {
        f64::NAN
      };
      let z01 = if j + 1 < cols {
        grid[i][j + 1]
      } else {
        f64::NAN
      };
      let z11 = if i + 1 < rows && j + 1 < cols {
        grid[i + 1][j + 1]
      } else {
        f64::NAN
      };

      let nx = |ii: usize| -> f64 {
        (ii as f64 / (rows - 1).max(1) as f64) * 2.0 - 1.0
      };
      let ny = |jj: usize| -> f64 {
        (jj as f64 / (cols - 1).max(1) as f64) * 2.0 - 1.0
      };
      let nz =
        |z: f64| -> f64 { ((z - z_min) / z_range) * 2.0 * Z_SCALE - Z_SCALE };

      // Triangle 1: (i,j), (i+1,j), (i,j+1)
      if z00.is_finite() && z10.is_finite() && z01.is_finite() {
        let v0 = Point3D {
          x: nx(i),
          y: ny(j),
          z: nz(z00),
        };
        let v1 = Point3D {
          x: nx(i + 1),
          y: ny(j),
          z: nz(z10),
        };
        let v2 = Point3D {
          x: nx(i),
          y: ny(j + 1),
          z: nz(z01),
        };

        let avg = ((z00 - z_min) / z_range
          + (z10 - z_min) / z_range
          + (z01 - z_min) / z_range)
          / 3.0;
        let base_color = height_color(avg);
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
          opacity: 1.0,
        });
      }

      // Triangle 2: (i+1,j+1), (i,j+1), (i+1,j)
      if z11.is_finite() && z01.is_finite() && z10.is_finite() {
        let v0 = Point3D {
          x: nx(i + 1),
          y: ny(j + 1),
          z: nz(z11),
        };
        let v1 = Point3D {
          x: nx(i),
          y: ny(j + 1),
          z: nz(z01),
        };
        let v2 = Point3D {
          x: nx(i + 1),
          y: ny(j),
          z: nz(z10),
        };

        let avg = ((z11 - z_min) / z_range
          + (z01 - z_min) / z_range
          + (z10 - z_min) / z_range)
          / 3.0;
        let base_color = height_color(avg);
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
          opacity: 1.0,
        });
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ListPlot3D: data produced no renderable triangles".into(),
    ));
  }

  let (z_axis_min, z_axis_max) = if (z_min - z_max).abs() < 1e-15 {
    (z_min - 0.5, z_max + 0.5)
  } else {
    (z_min, z_max)
  };

  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    true, // show_axes: always show axes for list_plot3d
  )?;

  Ok(crate::graphics3d_result(svg))
}

// ── RevolutionPlot3D implementation ──────────────────────────────────

/// Evaluate a single-variable expression at a given value.
fn evaluate_at_t(body: &Expr, tvar: &str, tval: f64) -> Option<f64> {
  let sub = substitute_var(body, tvar, &Expr::Real(tval));
  let result = evaluate_expr_to_expr(&sub).ok()?;
  try_eval_to_f64(&result)
}

/// Evaluate a two-variable expression (t, theta) at given values.
fn evaluate_at_t_theta(
  body: &Expr,
  tvar: &str,
  tval: f64,
  theta_var: &str,
  theta_val: f64,
) -> Option<f64> {
  let sub1 = substitute_var(body, tvar, &Expr::Real(tval));
  let sub2 = substitute_var(&sub1, theta_var, &Expr::Real(theta_val));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  try_eval_to_f64(&result)
}

/// RevolutionPlot3D[f, {t, tmin, tmax}]
/// RevolutionPlot3D[{r, z}, {t, tmin, tmax}]
/// RevolutionPlot3D[f, {t, tmin, tmax}, {θ, θmin, θmax}]
pub fn revolution_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "RevolutionPlot3D requires at least 2 arguments".into(),
    ));
  }

  let body = &args[0];

  // Parse t iterator
  let (tvar, t_min, t_max) =
    parse_iterator_rev(&args[1], "RevolutionPlot3D", "first")?;

  // Check if we have an explicit theta range
  let mut opt_start = 2;
  let (theta_var, theta_min, theta_max) = if args.len() > 2
    && matches!(&args[2], Expr::List(items) if items.len() == 3
      && matches!(&items[0], Expr::Identifier(_)))
  {
    match parse_iterator_rev(&args[2], "RevolutionPlot3D", "second") {
      Ok((v, lo, hi)) => {
        opt_start = 3;
        (Some(v), lo, hi)
      }
      Err(_) => (None, 0.0, 2.0 * std::f64::consts::PI),
    }
  } else {
    (None, 0.0, 2.0 * std::f64::consts::PI)
  };

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut _mesh_mode = MeshMode::Default;
  let mut show_axes = true;
  let mut z_clip: Option<(f64, f64)> = None;

  for opt in &args[opt_start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => _mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => _mesh_mode = MeshMode::All,
            _ => {}
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
        Expr::Identifier(name) if name == "Boxed" => match replacement.as_ref()
        {
          Expr::Identifier(s) if s == "False" => show_axes = false,
          Expr::Identifier(s) if s == "True" => show_axes = true,
          _ => {}
        },
        _ => {}
      }
    }
  }

  // Determine if body is {r_expr, z_expr} (parametric) or scalar f(t)
  let is_parametric = matches!(body, Expr::List(items) if items.len() == 2);

  let has_theta = theta_var.is_some();

  let camera = Camera::default();
  let n_t = GRID_N;
  let n_theta = GRID_N;
  let t_step = (t_max - t_min) / n_t as f64;
  let theta_step = (theta_max - theta_min) / n_theta as f64;

  // Sample the surface grid: grid[i][j] = (x, y, z)
  let mut grid: Vec<Vec<Option<Point3D>>> =
    vec![vec![None; n_theta + 1]; n_t + 1];
  let mut global_z_min = f64::INFINITY;
  let mut global_z_max = f64::NEG_INFINITY;
  let mut global_r_max: f64 = 0.0;

  for i in 0..=n_t {
    let tval = t_min + i as f64 * t_step;
    for j in 0..=n_theta {
      let theta = theta_min + j as f64 * theta_step;

      let (r, z) = if has_theta {
        let theta_v = theta_var.as_ref().unwrap();
        if is_parametric {
          if let Expr::List(items) = body {
            let r_val =
              evaluate_at_t_theta(&items[0], &tvar, tval, theta_v, theta);
            let z_val =
              evaluate_at_t_theta(&items[1], &tvar, tval, theta_v, theta);
            match (r_val, z_val) {
              (Some(r), Some(z)) if r.is_finite() && z.is_finite() => (r, z),
              _ => continue,
            }
          } else {
            continue;
          }
        } else {
          // Scalar: r = f(t, θ), z = t
          match evaluate_at_t_theta(body, &tvar, tval, theta_v, theta) {
            Some(r) if r.is_finite() => (r, tval),
            _ => continue,
          }
        }
      } else if is_parametric {
        if let Expr::List(items) = body {
          let r_val = evaluate_at_t(&items[0], &tvar, tval);
          let z_val = evaluate_at_t(&items[1], &tvar, tval);
          match (r_val, z_val) {
            (Some(r), Some(z)) if r.is_finite() && z.is_finite() => (r, z),
            _ => continue,
          }
        } else {
          continue;
        }
      } else {
        // Scalar f(t): revolve (t, f(t)) → r = t, z = f(t)
        match evaluate_at_t(body, &tvar, tval) {
          Some(z) if z.is_finite() => (tval, z),
          _ => continue,
        }
      };

      let x = r * theta.cos();
      let y = r * theta.sin();

      grid[i][j] = Some(Point3D { x, y, z });

      global_z_min = global_z_min.min(z);
      global_z_max = global_z_max.max(z);
      global_r_max = global_r_max.max(r.abs());
    }
  }

  if !global_z_min.is_finite()
    || !global_z_max.is_finite()
    || global_r_max == 0.0
  {
    return Err(InterpreterError::EvaluationError(
      "RevolutionPlot3D: function produced no finite values in the given range"
        .into(),
    ));
  }

  let (z_lo, z_hi) = z_clip.unwrap_or((global_z_min, global_z_max));
  let z_range = if (z_hi - z_lo).abs() < 1e-15 {
    1.0
  } else {
    z_hi - z_lo
  };

  let r_scale = if global_r_max < 1e-15 {
    1.0
  } else {
    global_r_max
  };

  let nz = |z: f64| -> f64 {
    let cz = z.clamp(z_lo, z_hi);
    ((cz - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE
  };

  // Build triangles
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for i in 0..n_t {
    for j in 0..n_theta {
      let p00 = grid[i][j];
      let p10 = grid[i + 1][j];
      let p01 = grid[i][j + 1];
      let p11 = grid[i + 1][j + 1];

      let normalize = |p: Point3D| -> Point3D {
        Point3D {
          x: p.x / r_scale,
          y: p.y / r_scale,
          z: nz(p.z),
        }
      };

      let z_norm_of = |p: Point3D| -> f64 {
        ((p.z.clamp(z_lo, z_hi) - z_lo) / z_range).clamp(0.0, 1.0)
      };

      // Triangle 1: p00, p10, p01
      if let (Some(pp00), Some(pp10), Some(pp01)) = (p00, p10, p01) {
        let v0 = normalize(pp00);
        let v1 = normalize(pp10);
        let v2 = normalize(pp01);

        let avg_z_norm =
          (z_norm_of(pp00) + z_norm_of(pp10) + z_norm_of(pp01)) / 3.0;
        let base_color = height_color(avg_z_norm);
        let normal = triangle_normal(v0, v1, v2);
        let color = apply_lighting(base_color, normal);

        let proj0 = project(v0, &camera);
        let proj1 = project(v1, &camera);
        let proj2 = project(v2, &camera);
        let center = Point3D {
          x: (v0.x + v1.x + v2.x) / 3.0,
          y: (v0.y + v1.y + v2.y) / 3.0,
          z: (v0.z + v1.z + v2.z) / 3.0,
        };

        all_triangles.push(Triangle {
          projected: [proj0, proj1, proj2],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      }

      // Triangle 2: p11, p01, p10
      if let (Some(pp11), Some(pp01), Some(pp10)) = (p11, p01, p10) {
        let v0 = normalize(pp11);
        let v1 = normalize(pp01);
        let v2 = normalize(pp10);

        let avg_z_norm =
          (z_norm_of(pp11) + z_norm_of(pp01) + z_norm_of(pp10)) / 3.0;
        let base_color = height_color(avg_z_norm);
        let normal = triangle_normal(v0, v1, v2);
        let color = apply_lighting(base_color, normal);

        let proj0 = project(v0, &camera);
        let proj1 = project(v1, &camera);
        let proj2 = project(v2, &camera);
        let center = Point3D {
          x: (v0.x + v1.x + v2.x) / 3.0,
          y: (v0.y + v1.y + v2.y) / 3.0,
          z: (v0.z + v1.z + v2.z) / 3.0,
        };

        all_triangles.push(Triangle {
          projected: [proj0, proj1, proj2],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "RevolutionPlot3D: function produced no finite values in the given range"
        .into(),
    ));
  }

  // Sort for painter's algorithm
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let (z_axis_min, z_axis_max) = if (z_lo - z_hi).abs() < 1e-15 {
    (z_lo - 0.5, z_hi + 0.5)
  } else {
    (z_lo, z_hi)
  };

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (-global_r_max, global_r_max),
    (-global_r_max, global_r_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}

fn parse_iterator_generic(
  spec: &Expr,
  func_name: &str,
  label: &str,
) -> Result<(String, f64, f64), InterpreterError> {
  match spec {
    Expr::List(items) if items.len() == 3 => {
      let var = match &items[0] {
        Expr::Identifier(name) => name.clone(),
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "{func_name}: {label} iterator variable must be a symbol"
          )));
        }
      };
      let min_expr = evaluate_expr_to_expr(&items[1])?;
      let max_expr = evaluate_expr_to_expr(&items[2])?;
      let min_val = try_eval_to_f64(&min_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{func_name}: cannot evaluate {label} iterator min to a number"
        ))
      })?;
      let max_val = try_eval_to_f64(&max_expr).ok_or_else(|| {
        InterpreterError::EvaluationError(format!(
          "{func_name}: cannot evaluate {label} iterator max to a number"
        ))
      })?;
      Ok((var, min_val, max_val))
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "{func_name}: {label} iterator must be {{var, min, max}}"
    ))),
  }
}

// Keep old name for RevolutionPlot3D
fn parse_iterator_rev(
  spec: &Expr,
  func_name: &str,
  label: &str,
) -> Result<(String, f64, f64), InterpreterError> {
  parse_iterator_generic(spec, func_name, label)
}

// ── RegionPlot3D implementation ──────────────────────────────────────

const REGION3D_GRID: usize = 30;

/// Evaluate a 3D boolean condition at (x, y, z).
fn evaluate_condition_3d(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  zvar: &str,
  xval: f64,
  yval: f64,
  zval: f64,
) -> bool {
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let sub3 = substitute_var(&sub2, zvar, &Expr::Real(zval));
  if let Ok(result) = evaluate_expr_to_expr(&sub3) {
    matches!(result, Expr::Identifier(ref s) if s == "True")
  } else {
    false
  }
}

/// RegionPlot3D[cond, {x, xmin, xmax}, {y, ymin, ymax}, {z, zmin, zmax}]
pub fn region_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 4 {
    return Err(InterpreterError::EvaluationError(
      "RegionPlot3D requires at least 4 arguments: RegionPlot3D[cond, {x,xmin,xmax}, {y,ymin,ymax}, {z,zmin,zmax}]".into(),
    ));
  }

  let body = &args[0];
  let (xvar, x_min, x_max) =
    parse_iterator_generic(&args[1], "RegionPlot3D", "first")?;
  let (yvar, y_min, y_max) =
    parse_iterator_generic(&args[2], "RegionPlot3D", "second")?;
  let (zvar, z_min, z_max) =
    parse_iterator_generic(&args[3], "RegionPlot3D", "third")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut _mesh_mode = MeshMode::Default;
  let mut show_axes = true;

  for opt in &args[4..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => _mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => _mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "Boxed" => match replacement.as_ref()
        {
          Expr::Identifier(s) if s == "False" => show_axes = false,
          Expr::Identifier(s) if s == "True" => show_axes = true,
          _ => {}
        },
        _ => {}
      }
    }
  }

  let n = REGION3D_GRID;
  let x_step = (x_max - x_min) / n as f64;
  let y_step = (y_max - y_min) / n as f64;
  let z_step = (z_max - z_min) / n as f64;

  // Sample the boolean field on a 3D grid
  let mut field = vec![vec![vec![false; n + 1]; n + 1]; n + 1];
  for i in 0..=n {
    let xval = x_min + i as f64 * x_step;
    for j in 0..=n {
      let yval = y_min + j as f64 * y_step;
      for k in 0..=n {
        let zval = z_min + k as f64 * z_step;
        field[i][j][k] =
          evaluate_condition_3d(body, &xvar, &yvar, &zvar, xval, yval, zval);
      }
    }
  }

  // Normalize coordinates to [-1, 1] for x,y and [-Z_SCALE, Z_SCALE] for z
  let nx = |i: usize| -> f64 { (i as f64 / n as f64) * 2.0 - 1.0 };
  let ny = |j: usize| -> f64 { (j as f64 / n as f64) * 2.0 - 1.0 };
  let nz =
    |k: usize| -> f64 { (k as f64 / n as f64) * 2.0 * Z_SCALE - Z_SCALE };

  let camera = Camera::default();
  let mut all_triangles: Vec<Triangle> = Vec::new();

  // Default surface color (Mathematica-like blue with opacity)
  let base_r = 0x5E_u8;
  let base_g = 0x81_u8;
  let base_b = 0xB5_u8;

  // For each voxel, emit faces between true and false cells
  // Each face is a quad split into two triangles
  for i in 0..=n {
    for j in 0..=n {
      for k in 0..=n {
        if !field[i][j][k] {
          continue;
        }

        // Check each of 6 neighbor directions; if neighbor is false or out of bounds,
        // emit the face
        let neighbors: [(i32, i32, i32); 6] = [
          (1, 0, 0),
          (-1, 0, 0),
          (0, 1, 0),
          (0, -1, 0),
          (0, 0, 1),
          (0, 0, -1),
        ];

        for &(di, dj, dk) in &neighbors {
          let ni = i as i32 + di;
          let nj = j as i32 + dj;
          let nk = k as i32 + dk;

          let is_outside = ni < 0
            || nj < 0
            || nk < 0
            || ni > n as i32
            || nj > n as i32
            || nk > n as i32;

          let neighbor_true = if is_outside {
            false
          } else {
            field[ni as usize][nj as usize][nk as usize]
          };

          if neighbor_true {
            continue; // internal face, skip
          }

          // Emit a face quad at the boundary between cell (i,j,k) and neighbor
          // The face center is between (i,j,k) and (ni,nj,nk)
          let half = 0.5 / n as f64;
          let cx = nx(i) + di as f64 * half * 2.0;
          let cy = ny(j) + dj as f64 * half * 2.0;
          // For z, half step is Z_SCALE/n
          let z_half = Z_SCALE / n as f64;
          let cz = nz(k) + dk as f64 * z_half;

          // Build face vertices depending on which axis the face is perpendicular to
          let s = 1.0 / n as f64; // half-size of voxel in normalized xy coords
          let sz = Z_SCALE / n as f64; // half-size in z

          let (v0, v1, v2, v3) = if di != 0 {
            // Face perpendicular to x-axis
            (
              Point3D {
                x: cx,
                y: cy - s,
                z: cz - sz,
              },
              Point3D {
                x: cx,
                y: cy + s,
                z: cz - sz,
              },
              Point3D {
                x: cx,
                y: cy + s,
                z: cz + sz,
              },
              Point3D {
                x: cx,
                y: cy - s,
                z: cz + sz,
              },
            )
          } else if dj != 0 {
            // Face perpendicular to y-axis
            (
              Point3D {
                x: cx - s,
                y: cy,
                z: cz - sz,
              },
              Point3D {
                x: cx + s,
                y: cy,
                z: cz - sz,
              },
              Point3D {
                x: cx + s,
                y: cy,
                z: cz + sz,
              },
              Point3D {
                x: cx - s,
                y: cy,
                z: cz + sz,
              },
            )
          } else {
            // Face perpendicular to z-axis
            (
              Point3D {
                x: cx - s,
                y: cy - s,
                z: cz,
              },
              Point3D {
                x: cx + s,
                y: cy - s,
                z: cz,
              },
              Point3D {
                x: cx + s,
                y: cy + s,
                z: cz,
              },
              Point3D {
                x: cx - s,
                y: cy + s,
                z: cz,
              },
            )
          };

          // Triangle 1: v0, v1, v2
          {
            let normal = triangle_normal(v0, v1, v2);
            let color = apply_lighting((base_r, base_g, base_b), normal);
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
              opacity: 1.0,
            });
          }

          // Triangle 2: v0, v2, v3
          {
            let normal = triangle_normal(v0, v2, v3);
            let color = apply_lighting((base_r, base_g, base_b), normal);
            let p0 = project(v0, &camera);
            let p2 = project(v2, &camera);
            let p3 = project(v3, &camera);
            let center = Point3D {
              x: (v0.x + v2.x + v3.x) / 3.0,
              y: (v0.y + v2.y + v3.y) / 3.0,
              z: (v0.z + v2.z + v3.z) / 3.0,
            };
            all_triangles.push(Triangle {
              projected: [p0, p2, p3],
              depth: depth(center, &camera),
              color,
              opacity: 1.0,
            });
          }
        }
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "RegionPlot3D: no region satisfies the condition in the given range"
        .into(),
    ));
  }

  // Painter's algorithm
  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_min, z_max),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}

// ── ListPointPlot3D implementation ───────────────────────────────────

/// A projected point for scatter rendering.
struct ScatterPoint {
  sx: f64,
  sy: f64,
  depth: f64,
  color: (u8, u8, u8),
}

pub fn list_point_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;

  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ImageSize"
      && let Some((w, h, fw)) =
        parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
    {
      svg_width = w;
      svg_height = h;
      full_width = fw;
    }
  }

  // Evaluate the data argument
  let evaled_data = evaluate_expr_to_expr(&args[0])?;

  // Parse data: accept list of {x,y,z} or list of lists of {x,y,z}
  let mut datasets: Vec<Vec<(f64, f64, f64)>> = Vec::new();

  match &evaled_data {
    Expr::List(outer) if !outer.is_empty() => {
      // Check if first element is {x,y,z} or a list of {x,y,z}
      match &outer[0] {
        Expr::List(inner)
          if inner.len() == 3 && !matches!(&inner[0], Expr::List(_)) =>
        {
          // Single dataset: {{x,y,z}, {x,y,z}, ...}
          let pts = parse_xyz_points(outer);
          if !pts.is_empty() {
            datasets.push(pts);
          }
        }
        Expr::List(inner)
          if !inner.is_empty()
            && inner.iter().all(|e| !matches!(e, Expr::List(_))) =>
        {
          // 2D matrix format: {{z11, z12, ...}, {z21, z22, ...}, ...}
          // x = column index (1-based), y = row index (1-based)
          let num_rows = outer.len();
          let num_cols = inner.len();
          let mut pts = Vec::new();
          for (i, row_expr) in outer.iter().enumerate() {
            if let Expr::List(row) = row_expr {
              for (j, val_expr) in row.iter().enumerate() {
                if let Some(z) = try_eval_to_f64(val_expr)
                  && z.is_finite()
                {
                  let x = (j + 1) as f64;
                  let y = (i + 1) as f64;
                  pts.push((x, y, z));
                }
              }
            }
          }
          let _ = (num_rows, num_cols);
          if !pts.is_empty() {
            datasets.push(pts);
          }
        }
        Expr::List(_) => {
          // Multiple datasets: {{{x,y,z},...}, {{x,y,z},...}, ...}
          for item in outer {
            if let Expr::List(inner) = item {
              let pts = parse_xyz_points(inner);
              if !pts.is_empty() {
                datasets.push(pts);
              }
            }
          }
        }
        _ => {
          // Try as single dataset anyway
          let pts = parse_xyz_points(outer);
          if !pts.is_empty() {
            datasets.push(pts);
          }
        }
      }
    }
    _ => {}
  }

  if datasets.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ListPointPlot3D: no valid data points found".into(),
    ));
  }

  // Find global ranges
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  let mut z_min = f64::INFINITY;
  let mut z_max = f64::NEG_INFINITY;

  for ds in &datasets {
    for &(x, y, z) in ds {
      x_min = x_min.min(x);
      x_max = x_max.max(x);
      y_min = y_min.min(y);
      y_max = y_max.max(y);
      z_min = z_min.min(z);
      z_max = z_max.max(z);
    }
  }

  if !z_min.is_finite() || !z_max.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "ListPointPlot3D: data produced no finite values".into(),
    ));
  }

  let x_range_v = if (x_max - x_min).abs() < 1e-15 {
    1.0
  } else {
    x_max - x_min
  };
  let y_range_v = if (y_max - y_min).abs() < 1e-15 {
    1.0
  } else {
    y_max - y_min
  };
  let z_range_v = if (z_max - z_min).abs() < 1e-15 {
    1.0
  } else {
    z_max - z_min
  };

  let camera = Camera::default();

  // Dataset colors (Mathematica-like palette)
  let palette: [(u8, u8, u8); 6] = [
    (68, 114, 196),  // blue
    (237, 125, 49),  // orange
    (165, 165, 165), // gray
    (255, 192, 0),   // gold
    (91, 155, 213),  // light blue
    (112, 173, 71),  // green
  ];

  let mut scatter_points: Vec<ScatterPoint> = Vec::new();

  for (di, ds) in datasets.iter().enumerate() {
    let base_color = palette[di % palette.len()];
    for &(x, y, z) in ds {
      let nx = if x_range_v > 1e-15 {
        ((x - x_min) / x_range_v) * 2.0 - 1.0
      } else {
        0.0
      };
      let ny = if y_range_v > 1e-15 {
        ((y - y_min) / y_range_v) * 2.0 - 1.0
      } else {
        0.0
      };
      let nz = if z_range_v > 1e-15 {
        ((z - z_min) / z_range_v) * 2.0 * Z_SCALE - Z_SCALE
      } else {
        0.0
      };
      let p3 = Point3D {
        x: nx,
        y: ny,
        z: nz,
      };
      let (sx, sy) = project(p3, &camera);
      let d = depth(p3, &camera);
      scatter_points.push(ScatterPoint {
        sx,
        sy,
        depth: d,
        color: base_color,
      });
    }
  }

  // Sort far-to-near (painter's)
  scatter_points.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  // Generate SVG
  let (z_axis_min, z_axis_max) = if (z_min - z_max).abs() < 1e-15 {
    (z_min - 0.5, z_max + 0.5)
  } else {
    (z_min, z_max)
  };

  let svg = generate_scatter_svg(
    &scatter_points,
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
  )?;

  Ok(crate::graphics3d_result(svg))
}

fn parse_xyz_points(items: &[Expr]) -> Vec<(f64, f64, f64)> {
  let mut pts = Vec::new();
  for item in items {
    if let Expr::List(coords) = item
      && coords.len() == 3
    {
      let x = try_eval_to_f64(&coords[0]);
      let y = try_eval_to_f64(&coords[1]);
      let z = try_eval_to_f64(&coords[2]);
      if let (Some(x), Some(y), Some(z)) = (x, y, z)
        && x.is_finite()
        && y.is_finite()
        && z.is_finite()
      {
        pts.push((x, y, z));
      }
    }
  }
  pts
}

fn generate_scatter_svg(
  points: &[ScatterPoint],
  camera: &Camera,
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  // Find bounding box
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;

  for pt in points {
    px_min = px_min.min(pt.sx);
    px_max = px_max.max(pt.sx);
    py_min = py_min.min(pt.sy);
    py_max = py_max.max(pt.sy);
  }

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
      "ListPointPlot3D: degenerate projection".into(),
    ));
  }

  let margin = 25.0;
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = (px_min + px_max) / 2.0;
  let p_cy = (py_min + py_max) / 2.0;

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    let sx = cx + (px - p_cx) * scale;
    let sy = cy - (py - p_cy) * scale;
    (sx, sy)
  };

  let mut svg = String::with_capacity(points.len() * 80 + 2000);

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

  {
    let (bg, _, _, _, _) = crate::functions::plot::plot_theme();
    svg.push_str(&format!(
      "<rect width=\"{}\" height=\"{}\" fill=\"rgb({},{},{})\"/>\n",
      svg_width, svg_height, bg.0, bg.1, bg.2
    ));
  }

  // Draw axes first (behind points)
  draw_axes(&mut svg, camera, &to_svg, x_range, y_range, z_range);

  // Build bounding-box edge segments for depth-interleaving
  let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
  let axis_color = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
  let corners = bounding_box_corners();
  let edge_pairs: [(usize, usize); 12] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
  ];
  const EDGE_SUBDIVISIONS: usize = 20;
  let mut sorted_edges: Vec<BoxEdge> =
    Vec::with_capacity(12 * EDGE_SUBDIVISIONS);
  for &(i, j) in &edge_pairs {
    let a = corners[i];
    let b = corners[j];
    for s in 0..EDGE_SUBDIVISIONS {
      let t0 = s as f64 / EDGE_SUBDIVISIONS as f64;
      let t1 = (s + 1) as f64 / EDGE_SUBDIVISIONS as f64;
      let tm = (t0 + t1) * 0.5;
      let lerp = |t: f64| Point3D {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
      };
      sorted_edges.push(BoxEdge {
        endpoints: [lerp(t0), lerp(t1)],
        depth: depth(lerp(tm), camera),
      });
    }
  }
  sorted_edges.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  // Merge-render scatter points and box edges back-to-front (painter's algorithm)
  let radius = 3.0;
  {
    let mut ei = 0;
    for pt in points {
      // Emit any box edges further from camera than this point
      while ei < sorted_edges.len() && sorted_edges[ei].depth >= pt.depth {
        let edge = &sorted_edges[ei];
        let (ex0, ey0) = to_svg(
          project(edge.endpoints[0], camera).0,
          project(edge.endpoints[0], camera).1,
        );
        let (ex1, ey1) = to_svg(
          project(edge.endpoints[1], camera).0,
          project(edge.endpoints[1], camera).1,
        );
        svg.push_str(&format!(
          "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
          ex0, ey0, ex1, ey1, axis_color
        ));
        ei += 1;
      }
      // Emit point
      let (sx, sy) = to_svg(pt.sx, pt.sy);
      let (r, g, b) = pt.color;
      svg.push_str(&format!(
        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{}\" fill=\"rgb({},{},{})\" stroke=\"rgb({},{},{})\" stroke-width=\"0.5\" opacity=\"0.85\"/>\n",
        sx, sy, radius, r, g, b,
        (r as f64 * 0.7) as u8, (g as f64 * 0.7) as u8, (b as f64 * 0.7) as u8,
      ));
    }
    // Emit remaining box edges (closest to viewer)
    while ei < sorted_edges.len() {
      let edge = &sorted_edges[ei];
      let (ex0, ey0) = to_svg(
        project(edge.endpoints[0], camera).0,
        project(edge.endpoints[0], camera).1,
      );
      let (ex1, ey1) = to_svg(
        project(edge.endpoints[1], camera).0,
        project(edge.endpoints[1], camera).1,
      );
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n",
        ex0, ey0, ex1, ey1, axis_color
      ));
      ei += 1;
    }
  }

  svg.push_str("</svg>");
  Ok(svg)
}

// ── SphericalPlot3D implementation ───────────────────────────────────

const SPHERICAL_GRID: usize = 50;

/// SphericalPlot3D[r, {theta, t0, t1}, {phi, p0, p1}]
/// Plots r(theta, phi) in spherical coordinates.
pub fn spherical_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
      "SphericalPlot3D requires at least 3 arguments".into(),
    ));
  }

  let body = &args[0];

  // Parse theta iterator {theta, t0, t1}
  let (theta_var, theta_min, theta_max) =
    parse_iterator_generic(&args[1], "SphericalPlot3D", "theta")?;
  // Parse phi iterator {phi, p0, p1}
  let (phi_var, phi_min, phi_max) =
    parse_iterator_generic(&args[2], "SphericalPlot3D", "phi")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut _mesh_mode = MeshMode::Default;
  let mut show_axes = true;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => _mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => _mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "Axes" => {
          if matches!(replacement.as_ref(), Expr::Identifier(n) if n == "False")
          {
            show_axes = false;
          }
        }
        _ => {}
      }
    }
  }

  let n_theta = SPHERICAL_GRID;
  let n_phi = SPHERICAL_GRID;
  let theta_range = theta_max - theta_min;
  let phi_range = phi_max - phi_min;

  // Sample the function on a theta x phi grid
  let mut grid_pts: Vec<Vec<Option<Point3D>>> =
    vec![vec![None; n_phi + 1]; n_theta + 1];

  for i in 0..=n_theta {
    let theta = theta_min + (i as f64 / n_theta as f64) * theta_range;
    for j in 0..=n_phi {
      let phi = phi_min + (j as f64 / n_phi as f64) * phi_range;
      if let Some(r) =
        evaluate_at_t_theta(body, &theta_var, theta, &phi_var, phi)
        && r.is_finite()
      {
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        if x.is_finite() && y.is_finite() && z.is_finite() {
          grid_pts[i][j] = Some(Point3D { x, y, z });
        }
      }
    }
  }

  // Find coordinate ranges
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  let mut z_min = f64::INFINITY;
  let mut z_max = f64::NEG_INFINITY;

  for row in &grid_pts {
    for p in row.iter().flatten() {
      x_min = x_min.min(p.x);
      x_max = x_max.max(p.x);
      y_min = y_min.min(p.y);
      y_max = y_max.max(p.y);
      z_min = z_min.min(p.z);
      z_max = z_max.max(p.z);
    }
  }

  if !x_min.is_finite() || !z_min.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "SphericalPlot3D: no valid points computed".into(),
    ));
  }

  let x_range_v = (x_max - x_min).max(1e-15);
  let y_range_v = (y_max - y_min).max(1e-15);
  let z_range_v = (z_max - z_min).max(1e-15);

  let camera = Camera::default();
  let mut all_triangles: Vec<Triangle> = Vec::new();

  // Normalize a point to [-1,1] box
  let normalize = |p: Point3D| -> Point3D {
    Point3D {
      x: ((p.x - x_min) / x_range_v) * 2.0 - 1.0,
      y: ((p.y - y_min) / y_range_v) * 2.0 - 1.0,
      z: ((p.z - z_min) / z_range_v) * 2.0 * Z_SCALE - Z_SCALE,
    }
  };

  // Build triangles from grid
  for i in 0..n_theta {
    for j in 0..n_phi {
      let p00 = grid_pts[i][j];
      let p10 = grid_pts[i + 1][j];
      let p01 = grid_pts[i][j + 1];
      let p11 = grid_pts[i + 1][j + 1];

      // Triangle 1: (i,j), (i+1,j), (i,j+1)
      if let (Some(a), Some(b), Some(c)) = (p00, p10, p01) {
        let na = normalize(a);
        let nb = normalize(b);
        let nc = normalize(c);

        let avg_z = ((a.z - z_min) / z_range_v
          + (b.z - z_min) / z_range_v
          + (c.z - z_min) / z_range_v)
          / 3.0;
        let base_color = height_color(avg_z);
        let normal = triangle_normal(na, nb, nc);
        let color = apply_lighting(base_color, normal);

        let pa = project(na, &camera);
        let pb = project(nb, &camera);
        let pc = project(nc, &camera);
        let center = Point3D {
          x: (na.x + nb.x + nc.x) / 3.0,
          y: (na.y + nb.y + nc.y) / 3.0,
          z: (na.z + nb.z + nc.z) / 3.0,
        };

        all_triangles.push(Triangle {
          projected: [pa, pb, pc],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      }

      // Triangle 2: (i+1,j+1), (i,j+1), (i+1,j)
      if let (Some(a), Some(b), Some(c)) = (p11, p01, p10) {
        let na = normalize(a);
        let nb = normalize(b);
        let nc = normalize(c);

        let avg_z = ((a.z - z_min) / z_range_v
          + (b.z - z_min) / z_range_v
          + (c.z - z_min) / z_range_v)
          / 3.0;
        let base_color = height_color(avg_z);
        let normal = triangle_normal(na, nb, nc);
        let color = apply_lighting(base_color, normal);

        let pa = project(na, &camera);
        let pb = project(nb, &camera);
        let pc = project(nc, &camera);
        let center = Point3D {
          x: (na.x + nb.x + nc.x) / 3.0,
          y: (na.y + nb.y + nc.y) / 3.0,
          z: (na.z + nb.z + nc.z) / 3.0,
        };

        all_triangles.push(Triangle {
          projected: [pa, pb, pc],
          depth: depth(center, &camera),
          color,
          opacity: 1.0,
        });
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "SphericalPlot3D: no renderable triangles".into(),
    ));
  }

  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let (z_axis_min, z_axis_max) = if (z_min - z_max).abs() < 1e-15 {
    (z_min - 0.5, z_max + 0.5)
  } else {
    (z_min, z_max)
  };

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}

/// DiscretePlot3D[f, {x, xmin, xmax}, {y, ymin, ymax}]
/// Plots a function at discrete integer points in 3D
pub fn discrete_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D requires at least 3 arguments: DiscretePlot3D[f, {x, xmin, xmax}, {y, ymin, ymax}]".into(),
    ));
  }

  let body = &args[0];
  let (xvar, x_min, x_max) = parse_iterator(&args[1], "first")?;
  let (yvar, y_min, y_max) = parse_iterator(&args[2], "second")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let _mesh_mode = MeshMode::Default;
  let show_axes = true;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ImageSize"
      && let Some((w, h, fw)) =
        parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
    {
      svg_width = w;
      svg_height = h;
      full_width = fw;
    }
  }

  // Generate grid at integer points
  let x_start = x_min.ceil() as i64;
  let x_end = x_max.floor() as i64;
  let y_start = y_min.ceil() as i64;
  let y_end = y_max.floor() as i64;

  let nx = (x_end - x_start + 1) as usize;
  let ny = (y_end - y_start + 1) as usize;

  if nx < 2 || ny < 2 {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D: range must contain at least 2 integer points in each dimension".into(),
    ));
  }

  // Sample at integer points
  let mut grid = vec![vec![f64::NAN; ny]; nx];
  let mut z_lo = f64::INFINITY;
  let mut z_hi = f64::NEG_INFINITY;

  for (i, xi) in (x_start..=x_end).enumerate() {
    for (j, yj) in (y_start..=y_end).enumerate() {
      let sub1 = substitute_var(body, &xvar, &Expr::Integer(xi as i128));
      let sub2 = substitute_var(&sub1, &yvar, &Expr::Integer(yj as i128));
      if let Ok(result) = evaluate_expr_to_expr(&sub2)
        && let Some(z) = try_eval_to_f64(&result)
        && z.is_finite()
      {
        grid[i][j] = z;
        z_lo = z_lo.min(z);
        z_hi = z_hi.max(z);
      }
    }
  }

  if !z_lo.is_finite() || !z_hi.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D: could not compute any finite values".into(),
    ));
  }

  if (z_hi - z_lo).abs() < 1e-15 {
    z_hi = z_lo + 1.0;
  }

  let z_range_val = z_hi - z_lo;
  let camera = Camera::default();

  // Build triangles from the grid
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for i in 0..nx - 1 {
    for j in 0..ny - 1 {
      let z00 = grid[i][j];
      let z10 = grid[i + 1][j];
      let z01 = grid[i][j + 1];
      let z11 = grid[i + 1][j + 1];

      let nx_fn =
        |ii: usize| -> f64 { (ii as f64 / (nx - 1) as f64) * 2.0 - 1.0 };
      let ny_fn =
        |jj: usize| -> f64 { (jj as f64 / (ny - 1) as f64) * 2.0 - 1.0 };
      let nz = |z: f64| -> f64 {
        ((z - z_lo) / z_range_val) * 2.0 * Z_SCALE - Z_SCALE
      };

      // Triangle 1: (i,j), (i+1,j), (i,j+1)
      if z00.is_finite() && z10.is_finite() && z01.is_finite() {
        let cz00 = z00.clamp(z_lo, z_hi);
        let cz10 = z10.clamp(z_lo, z_hi);
        let cz01 = z01.clamp(z_lo, z_hi);

        let v0 = Point3D {
          x: nx_fn(i),
          y: ny_fn(j),
          z: nz(cz00),
        };
        let v1 = Point3D {
          x: nx_fn(i + 1),
          y: ny_fn(j),
          z: nz(cz10),
        };
        let v2 = Point3D {
          x: nx_fn(i),
          y: ny_fn(j + 1),
          z: nz(cz01),
        };

        let avg_z_norm = ((cz00 - z_lo) / z_range_val
          + (cz10 - z_lo) / z_range_val
          + (cz01 - z_lo) / z_range_val)
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
          color,
          depth: depth(center, &camera),
          opacity: 1.0,
        });
      }

      // Triangle 2: (i+1,j+1), (i,j+1), (i+1,j)
      if z11.is_finite() && z01.is_finite() && z10.is_finite() {
        let cz11 = z11.clamp(z_lo, z_hi);
        let cz01 = z01.clamp(z_lo, z_hi);
        let cz10 = z10.clamp(z_lo, z_hi);

        let v0 = Point3D {
          x: nx_fn(i + 1),
          y: ny_fn(j + 1),
          z: nz(cz11),
        };
        let v1 = Point3D {
          x: nx_fn(i),
          y: ny_fn(j + 1),
          z: nz(cz01),
        };
        let v2 = Point3D {
          x: nx_fn(i + 1),
          y: ny_fn(j),
          z: nz(cz10),
        };

        let avg_z_norm = ((cz11 - z_lo) / z_range_val
          + (cz01 - z_lo) / z_range_val
          + (cz10 - z_lo) / z_range_val)
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
          color,
          depth: depth(center, &camera),
          opacity: 1.0,
        });
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D: no renderable triangles".into(),
    ));
  }

  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_lo, z_hi),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}

/// Evaluate a parametric triple {fx(u,v), fy(u,v), fz(u,v)} at given u, v values.
fn evaluate_parametric_at_uv(
  fx: &Expr,
  fy: &Expr,
  fz: &Expr,
  uvar: &str,
  vvar: &str,
  uval: f64,
  vval: f64,
) -> Option<(f64, f64, f64)> {
  let eval_one = |body: &Expr| -> Option<f64> {
    let sub1 = substitute_var(body, uvar, &Expr::Real(uval));
    let sub2 = substitute_var(&sub1, vvar, &Expr::Real(vval));
    let result = evaluate_expr_to_expr(&sub2).ok()?;
    try_eval_to_f64(&result)
  };
  let x = eval_one(fx)?;
  let y = eval_one(fy)?;
  let z = eval_one(fz)?;
  if x.is_finite() && y.is_finite() && z.is_finite() {
    Some((x, y, z))
  } else {
    None
  }
}

/// Implementation of ParametricPlot3D[{fx, fy, fz}, {u, umin, umax}, {v, vmin, vmax}]
pub fn parametric_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot3D requires at least 3 arguments: ParametricPlot3D[{fx, fy, fz}, {u, umin, umax}, {v, vmin, vmax}]".into(),
    ));
  }

  let body = &args[0];

  // Parse iterators
  let (uvar, u_min, u_max) = parse_iterator(&args[1], "first")?;
  let (vvar, v_min, v_max) = parse_iterator(&args[2], "second")?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut _mesh_mode = MeshMode::Default;
  let mut show_axes = true;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "ImageSize" => {
          if let Some((w, h, fw)) =
            parse_image_size(replacement, DEFAULT_SIZE, DEFAULT_SIZE)
          {
            svg_width = w;
            svg_height = h;
            full_width = fw;
          }
        }
        Expr::Identifier(name) if name == "Mesh" => {
          match replacement.as_ref() {
            Expr::Identifier(n) if n == "None" => _mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => _mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "Boxed" => {
          match replacement.as_ref() {
            Expr::Identifier(s) if s == "False" => show_axes = false,
            Expr::Identifier(s) if s == "True" => show_axes = true,
            _ => {}
          }
        }
        _ => {}
      }
    }
  }

  // Parse parametric surfaces: body must be {fx, fy, fz} or {{fx1, fy1, fz1}, ...}
  struct ParametricSurface<'a> {
    fx: &'a Expr,
    fy: &'a Expr,
    fz: &'a Expr,
  }

  let surfaces: Vec<ParametricSurface> = match body {
    Expr::List(items) if !items.is_empty() => {
      if items.len() == 3 && !matches!(&items[0], Expr::List(_)) {
        vec![ParametricSurface {
          fx: &items[0],
          fy: &items[1],
          fz: &items[2],
        }]
      } else if items
        .iter()
        .all(|item| matches!(item, Expr::List(sub) if sub.len() == 3))
      {
        items
          .iter()
          .map(|item| {
            if let Expr::List(sub) = item {
              ParametricSurface {
                fx: &sub[0],
                fy: &sub[1],
                fz: &sub[2],
              }
            } else {
              unreachable!()
            }
          })
          .collect()
      } else {
        vec![ParametricSurface {
          fx: &items[0],
          fy: &items[1],
          fz: &items[2],
        }]
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ParametricPlot3D: first argument must be {fx, fy, fz}".into(),
      ));
    }
  };

  let camera = Camera::default();
  let u_step = (u_max - u_min) / GRID_N as f64;
  let v_step = (v_max - v_min) / GRID_N as f64;

  // Phase 1: Sample all parametric surfaces and compute global ranges
  let mut all_surface_points: Vec<Vec<Vec<Option<(f64, f64, f64)>>>> =
    Vec::new();
  let mut gx_min = f64::INFINITY;
  let mut gx_max = f64::NEG_INFINITY;
  let mut gy_min = f64::INFINITY;
  let mut gy_max = f64::NEG_INFINITY;
  let mut gz_min = f64::INFINITY;
  let mut gz_max = f64::NEG_INFINITY;

  for surface in &surfaces {
    let mut points = vec![vec![None; GRID_N + 1]; GRID_N + 1];
    for i in 0..=GRID_N {
      let uval = u_min + i as f64 * u_step;
      for j in 0..=GRID_N {
        let vval = v_min + j as f64 * v_step;
        if let Some((x, y, z)) = evaluate_parametric_at_uv(
          surface.fx, surface.fy, surface.fz, &uvar, &vvar, uval, vval,
        ) {
          points[i][j] = Some((x, y, z));
          gx_min = gx_min.min(x);
          gx_max = gx_max.max(x);
          gy_min = gy_min.min(y);
          gy_max = gy_max.max(y);
          gz_min = gz_min.min(z);
          gz_max = gz_max.max(z);
        }
      }
    }
    all_surface_points.push(points);
  }

  if !gx_min.is_finite() || !gy_min.is_finite() || !gz_min.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot3D: parametric function produced no finite values".into(),
    ));
  }

  let rx = if (gx_max - gx_min).abs() < 1e-15 {
    1.0
  } else {
    gx_max - gx_min
  };
  let ry = if (gy_max - gy_min).abs() < 1e-15 {
    1.0
  } else {
    gy_max - gy_min
  };
  let rz = if (gz_max - gz_min).abs() < 1e-15 {
    1.0
  } else {
    gz_max - gz_min
  };

  // Phase 2: Build triangles
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for sg in &all_surface_points {
    for i in 0..GRID_N {
      for j in 0..GRID_N {
        let p00 = sg[i][j];
        let p10 = sg[i + 1][j];
        let p01 = sg[i][j + 1];
        let p11 = sg[i + 1][j + 1];

        let normalize = |p: (f64, f64, f64)| -> Point3D {
          Point3D {
            x: ((p.0 - gx_min) / rx) * 2.0 - 1.0,
            y: ((p.1 - gy_min) / ry) * 2.0 - 1.0,
            z: ((p.2 - gz_min) / rz) * 2.0 * Z_SCALE - Z_SCALE,
          }
        };

        let z_norm = |z: f64| -> f64 { (z - gz_min) / rz };

        // Triangle 1
        if let (Some(a), Some(b), Some(c)) = (p00, p10, p01) {
          let v0 = normalize(a);
          let v1 = normalize(b);
          let v2 = normalize(c);
          let avg_z_norm = (z_norm(a.2) + z_norm(b.2) + z_norm(c.2)) / 3.0;
          let base_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let color = apply_lighting(base_color, normal);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };
          all_triangles.push(Triangle {
            projected: [
              project(v0, &camera),
              project(v1, &camera),
              project(v2, &camera),
            ],
            depth: depth(center, &camera),
            color,
            opacity: 1.0,
          });
        }

        // Triangle 2
        if let (Some(a), Some(b), Some(c)) = (p11, p01, p10) {
          let v0 = normalize(a);
          let v1 = normalize(b);
          let v2 = normalize(c);
          let avg_z_norm = (z_norm(a.2) + z_norm(b.2) + z_norm(c.2)) / 3.0;
          let base_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let color = apply_lighting(base_color, normal);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };
          all_triangles.push(Triangle {
            projected: [
              project(v0, &camera),
              project(v1, &camera),
              project(v2, &camera),
            ],
            depth: depth(center, &camera),
            color,
            opacity: 1.0,
          });
        }
      }
    }
  }

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot3D: parametric function produced no finite values".into(),
    ));
  }

  let (z_axis_min, z_axis_max) = if (gz_min - gz_max).abs() < 1e-15 {
    (gz_min - 0.5, gz_max + 0.5)
  } else {
    (gz_min, gz_max)
  };

  all_triangles.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let svg = generate_svg(
    &all_triangles,
    &[],
    &camera,
    (gx_min, gx_max),
    (gy_min, gy_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    _mesh_mode,
    show_axes,
  )?;

  Ok(crate::graphics3d_result(svg))
}
