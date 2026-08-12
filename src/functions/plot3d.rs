#[allow(unused_imports)]
use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{
  PLOT_COLORS, evaluate_at_xy, format_tick, nice_step, parse_image_size,
  substitute_var,
};

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

impl Camera {
  /// Wolfram's default `ViewPoint`, in units of the longest side of the
  /// displayed box.
  pub(crate) const DEFAULT_VIEW_POINT: [f64; 3] = [1.3, -2.4, 2.0];
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
  /// Whether edge *i* (from vertex `i` to vertex `i + 1`) is an outline of
  /// the primitive rather than an internal cut of its triangulation. Only
  /// outline edges are stroked, so a face with more than three corners
  /// does not show the diagonals it was split along.
  pub boundary: [bool; 3],
  /// The colour `EdgeForm[colour]` asked outline edges to be drawn in.
  /// `None` keeps the renderer's default dark grey. An explicit colour is
  /// also drawn at full opacity, so an outline stays visible around a
  /// transparent face.
  pub edge_color: Option<(u8, u8, u8)>,
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
  // Straight down (or up) the z axis the screen basis is degenerate — the
  // vertical `{0, 0, 1}` the camera is levelled against points at the
  // viewer — so Wolfram falls back to the plain top view: x to the right
  // and y up, mirrored when the view is from below. Without this the
  // picture came out turned a quarter turn from Wolfram's.
  if (cam.elevation.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9 {
    let up = if cam.elevation > 0.0 { 1.0 } else { -1.0 };
    return (p.x, p.y * up);
  }

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

/// The scene's single light, pointing from the surface towards the lamp
/// (upper-left-front), normalized.
fn light_direction() -> [f64; 3] {
  let lx = 0.4_f64;
  let ly = -0.5_f64;
  let lz = 0.76_f64;
  let len = (lx * lx + ly * ly + lz * lz).sqrt();
  [lx / len, ly / len, lz / len]
}

/// Apply diffuse + ambient lighting
pub(crate) fn apply_lighting(
  color: (u8, u8, u8),
  normal: [f64; 3],
) -> (u8, u8, u8) {
  let light = light_direction();

  let dot = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
  let diffuse = dot.abs(); // use abs to light both sides

  let ambient = 0.35;
  let intensity = (ambient + (1.0 - ambient) * diffuse).clamp(0.0, 1.0);

  let r = (color.0 as f64 * intensity).round() as u8;
  let g = (color.1 as f64 * intensity).round() as u8;
  let b = (color.2 as f64 * intensity).round() as u8;
  (r, g, b)
}

/// Diffuse + ambient lighting with the highlight a `Specularity` directive
/// asks for laid over it. `view_dir` points from the scene towards the
/// viewer, so the highlight sits where the light reflects into the camera —
/// which is what makes `Specularity[White, 10]` read as a glossy sphere
/// rather than just a brighter one.
pub(crate) fn apply_lighting_specular(
  color: (u8, u8, u8),
  normal: [f64; 3],
  specular: Option<((u8, u8, u8), f64)>,
  view_dir: [f64; 3],
) -> (u8, u8, u8) {
  let base = apply_lighting(color, normal);
  let Some((hi, exponent)) = specular else {
    return base;
  };
  let light = light_direction();
  // Blinn-Phong: the highlight peaks where the surface normal bisects the
  // directions to the lamp and to the viewer.
  let half = {
    let h = [
      light[0] + view_dir[0],
      light[1] + view_dir[1],
      light[2] + view_dir[2],
    ];
    let len = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    if len < 1e-12 {
      return base;
    }
    [h[0] / len, h[1] / len, h[2] / len]
  };
  // `abs`, matching the two-sided diffuse term above: a tessellated sphere
  // hands back outward and inward normals depending on winding.
  let cos = (normal[0] * half[0] + normal[1] * half[1] + normal[2] * half[2])
    .abs()
    .clamp(0.0, 1.0);
  let strength = cos.powf(exponent);
  let mix = |b: u8, h: u8| {
    (b as f64 + (h as f64 / 255.0) * strength * 255.0)
      .round()
      .min(255.0) as u8
  };
  (mix(base.0, hi.0), mix(base.1, hi.1), mix(base.2, hi.2))
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
  let (xvar, mut x_min, mut x_max) = parse_iterator(&args[1], "first")?;
  let (yvar, mut y_min, mut y_max) = parse_iterator(&args[2], "second")?;

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
          // `PlotRange -> {zmin, zmax}` bounds the vertical axis only. The
          // full form `{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}` bounds
          // all three; the horizontal bounds narrow the plotted domain, so
          // a surface asked for over a wider iterator range is cropped to
          // the box rather than drawn past it.
          if let Some([xr, yr, zr]) = parse_axis_ranges(replacement) {
            // An empty intersection would leave nothing to sample, so a
            // box that misses the iterator range leaves the domain alone.
            if x_min.max(xr.0) < x_max.min(xr.1)
              && y_min.max(yr.0) < y_max.min(yr.1)
            {
              x_min = x_min.max(xr.0);
              x_max = x_max.min(xr.1);
              y_min = y_min.max(yr.0);
              y_max = y_max.min(yr.1);
            }
            z_clip = Some(zr);
          } else if let Expr::List(items) = replacement.as_ref()
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

  // ── Symbolic structure: Graphics3D[GraphicsComplex[points, {…}]] ──
  // `Show` merges the *primitives* of the graphics it is given, so a
  // surface that exists only as a rendering is dropped when it is shown
  // together with a `Graphics3D`. Emitting the sampled surface in world
  // coordinates — coloured by height, the way the standalone render
  // colours it — lets `Show[Plot3D[…], Graphics3D[…]]` draw both.
  let structure = {
    let complexes: Vec<Expr> = grids
      .iter()
      .enumerate()
      .map(|(surface_idx, grid)| {
        let mut index_of: Vec<Vec<Option<usize>>> =
          vec![vec![None; GRID_N + 1]; GRID_N + 1];
        let mut point_exprs: Vec<Expr> = Vec::new();
        for (i, row) in grid.iter().enumerate() {
          for (j, &z) in row.iter().enumerate() {
            if !z.is_finite() {
              continue;
            }
            index_of[i][j] = Some(point_exprs.len());
            point_exprs.push(Expr::List(
              vec![
                Expr::Real(x_min + i as f64 * x_step),
                Expr::Real(y_min + j as f64 * y_step),
                Expr::Real(z.clamp(z_lo, z_hi)),
              ]
              .into(),
            ));
          }
        }
        let mut content: Vec<Expr> = Vec::new();
        // The sampling quads are far finer than the mesh a surface shows,
        // so their own outlines are suppressed and the mesh is drawn as
        // lines below — otherwise a shown surface turns into a wireframe.
        if !matches!(mesh_mode, MeshMode::All) {
          content.push(Expr::FunctionCall {
            name: "EdgeForm".to_string(),
            args: Vec::new().into(),
          });
        }
        for i in 0..GRID_N {
          for j in 0..GRID_N {
            let (Some(a), Some(b), Some(c), Some(d)) = (
              index_of[i][j],
              index_of[i + 1][j],
              index_of[i + 1][j + 1],
              index_of[i][j + 1],
            ) else {
              continue;
            };
            let avg_z_norm = [
              grid[i][j],
              grid[i + 1][j],
              grid[i + 1][j + 1],
              grid[i][j + 1],
            ]
            .iter()
            .map(|z| (z.clamp(z_lo, z_hi) - z_lo) / z_range)
            .sum::<f64>()
              / 4.0;
            let (cr, cg, cb) =
              surface_height_color(avg_z_norm, surface_idx, num_surfaces);
            // Colour and polygon travel in a sublist so the directive does
            // not leak onto the quads drawn after it.
            content.push(Expr::List(
              vec![
                Expr::FunctionCall {
                  name: "RGBColor".to_string(),
                  args: vec![
                    Expr::Real(cr as f64 / 255.0),
                    Expr::Real(cg as f64 / 255.0),
                    Expr::Real(cb as f64 / 255.0),
                  ]
                  .into(),
                },
                Expr::FunctionCall {
                  name: "Polygon".to_string(),
                  args: vec![Expr::List(
                    [a, b, c, d]
                      .iter()
                      .map(|&k| Expr::Integer(k as i128 + 1))
                      .collect::<Vec<_>>()
                      .into(),
                  )]
                  .into(),
                },
              ]
              .into(),
            ));
          }
        }
        // The mesh, at the spacing the standalone render rules it with.
        if matches!(mesh_mode, MeshMode::Default) {
          let mut segments: Vec<Expr> = Vec::new();
          let mut push_segment = |a: Option<usize>, b: Option<usize>| {
            if let (Some(a), Some(b)) = (a, b) {
              segments.push(Expr::List(
                vec![
                  Expr::Integer(a as i128 + 1),
                  Expr::Integer(b as i128 + 1),
                ]
                .into(),
              ));
            }
          };
          for j in (0..=GRID_N).step_by(MESH_STEP) {
            for i in 0..GRID_N {
              push_segment(index_of[i][j], index_of[i + 1][j]);
            }
          }
          for i in (0..=GRID_N).step_by(MESH_STEP) {
            for j in 0..GRID_N {
              push_segment(index_of[i][j], index_of[i][j + 1]);
            }
          }
          if !segments.is_empty() {
            content.push(Expr::List(
              vec![
                Expr::FunctionCall {
                  name: "Opacity".to_string(),
                  args: vec![Expr::Real(0.63)].into(),
                },
                Expr::FunctionCall {
                  name: "RGBColor".to_string(),
                  args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)]
                    .into(),
                },
                Expr::FunctionCall {
                  name: "AbsoluteThickness".to_string(),
                  args: vec![Expr::Real(0.5)].into(),
                },
                Expr::FunctionCall {
                  name: "Line".to_string(),
                  args: vec![Expr::List(segments.into())].into(),
                },
              ]
              .into(),
            ));
          }
        }
        Expr::FunctionCall {
          name: "GraphicsComplex".to_string(),
          args: vec![
            Expr::List(point_exprs.into()),
            Expr::List(content.into()),
          ]
          .into(),
        }
      })
      .collect();
    let content = if complexes.len() == 1 {
      complexes.into_iter().next().expect("one complex")
    } else {
      Expr::List(complexes.into())
    };
    // `Plot3D` draws axes and squats its box where a bare `Graphics3D`
    // does neither, so both defaults are spelled out: a surface shown
    // inside another graphic keeps the shape it was drawn with.
    let mut structure_args = vec![content];
    structure_args.extend(args[3..].iter().cloned());
    let names = |opt: &str| {
      structure_args.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == opt))
      })
    };
    let (names_axes, names_ratios) = (names("Axes"), names("BoxRatios"));
    if !names_axes {
      structure_args.push(Expr::Rule {
        pattern: Box::new(Expr::Identifier("Axes".to_string())),
        replacement: Box::new(Expr::Identifier("True".to_string())),
      });
    }
    if !names_ratios {
      structure_args.push(Expr::Rule {
        pattern: Box::new(Expr::Identifier("BoxRatios".to_string())),
        replacement: Box::new(Expr::List(
          vec![Expr::Integer(1), Expr::Integer(1), Expr::Real(Z_SCALE)].into(),
        )),
      });
    }
    Expr::FunctionCall {
      name: "Graphics3D".to_string(),
      args: structure_args.into(),
    }
  };

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
            boundary: [true; 3],
            edge_color: None,
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
            boundary: [true; 3],
            edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result_with_structure(svg, structure))
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

/// Draw 3D axis lines with ticks and labels over the normalized cube the
/// surface plots draw in. `AxesLabel` strings, when given, name the x, y
/// and z axis in that order.
fn draw_axes(
  svg: &mut String,
  camera: &Camera,
  to_svg: &dyn Fn(f64, f64) -> (f64, f64),
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
) {
  draw_axes_on_box(
    svg,
    camera,
    to_svg,
    &bounding_box_corners(),
    x_range,
    y_range,
    z_range,
    &[None, None, None],
  );
}

/// Draw 3D axis lines with ticks and labels along the edges of `corners`,
/// the box the scene is framed by. `corners` is ordered as
/// `bounding_box_corners()` builds it: the four `z_min` corners first,
/// each pair varying x fastest.
#[allow(clippy::too_many_arguments)]
fn draw_axes_on_box(
  svg: &mut String,
  camera: &Camera,
  to_svg: &dyn Fn(f64, f64) -> (f64, f64),
  corners: &[Point3D; 8],
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  axes_labels: &[Option<String>; 3],
) {
  let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
  let axis_color = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
  let font_size = 13;

  // The box in scene coordinates. `mirror_*` flips a coordinate to the
  // opposite face, which is how each axis edge finds its far end.
  let (box_x_lo, box_x_hi) = (corners[0].x, corners[7].x);
  let (box_y_lo, box_y_hi) = (corners[0].y, corners[7].y);
  let (box_z_lo, box_z_hi) = (corners[0].z, corners[7].z);
  let mirror_x = |x: f64| box_x_lo + box_x_hi - x;
  let mirror_y = |y: f64| box_y_lo + box_y_hi - y;

  // Find the bottom corner (z at the box floor) closest to the viewer
  // (smallest depth)
  let mut min_depth_idx = 0;
  let mut min_depth = f64::INFINITY;
  for (idx, &corner) in corners.iter().enumerate() {
    if corner.z > box_z_lo + (box_z_hi - box_z_lo) * 0.01 {
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
  // When the edge runs from the box's high face to its low one, the
  // origin corner stands for val_max and the mapping has to be flipped.
  let x_end = Point3D {
    x: mirror_x(origin.x),
    y: origin.y,
    z: origin.z,
  };
  let y_end = Point3D {
    x: origin.x,
    y: mirror_y(origin.y),
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
    z: box_z_hi,
  };

  let axes: [(Point3D, Point3D, (f64, f64), bool); 3] = [
    (origin, x_end, x_range, origin.x > x_end.x),
    (origin, y_end, y_range, origin.y > y_end.y),
    (z_origin, z_end, z_range, false), // z always runs from floor to ceiling
  ];

  // Project the box center so we can orient tick labels outward
  let box_center = Point3D {
    x: (box_x_lo + box_x_hi) / 2.0,
    y: (box_y_lo + box_y_hi) / 2.0,
    z: (box_z_lo + box_z_hi) / 2.0,
  };
  let (cx, cy) =
    to_svg(project(box_center, camera).0, project(box_center, camera).1);

  for (ai, &(axis_origin, end, (val_min, val_max), flipped)) in
    axes.iter().enumerate()
  {
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

    // The axis label sits centred on the axis, clear of the tick labels
    // already sitting there — where Wolfram puts it.
    if let Some(label) = axes_labels[ai].as_deref().filter(|l| !l.is_empty()) {
      let (dx, dy) = (sx1 - sx0, sy1 - sy0);
      let len = (dx * dx + dy * dy).sqrt();
      if len > 1.0 {
        let (mid_x, mid_y) = ((sx0 + sx1) * 0.5, (sy0 + sy1) * 0.5);
        let (perpx, perpy) = (-dy / len, dx / len);
        let sign = if perpx * (cx - mid_x) + perpy * (cy - mid_y) > 0.0 {
          -1.0
        } else {
          1.0
        };
        // How far the tick labels already reach, in the direction the
        // axis label is being pushed: their own offset plus half of the
        // widest one's box. Sideways of a vertical axis that is the text
        // width, below a horizontal one it is the line height.
        let tick_chars = tick_values(val_min, val_max)
          .iter()
          .map(|v| format_tick(*v).chars().count())
          .max()
          .unwrap_or(1) as f64;
        let half_w = tick_chars * font_size as f64 * 0.3;
        let half_h = font_size as f64 * 0.5;
        // …plus half of the label's own box, so `offset` can centre it
        // just past them.
        let own_w = label_half_width(label, font_size);
        let reach = TICK_LABEL_OFFSET
          + (perpx * sign).abs() * (half_w + own_w)
          + (perpy * sign).abs() * (half_h + half_h);
        let offset = reach + font_size as f64 * 0.5;
        svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{}\" fill=\"{}\" text-anchor=\"middle\" dominant-baseline=\"middle\">{}</text>\n",
          mid_x + perpx * offset * sign,
          mid_y + perpy * offset * sign,
          font_size,
          axis_color,
          label
        ));
      }
    }

    // Ticks
    for tick_val in tick_values(val_min, val_max) {
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
                    tx + perpx / 4.0 * TICK_LABEL_OFFSET * sign, ty + perpy / 4.0 * TICK_LABEL_OFFSET * sign, font_size, axis_color, label
                ));
      }
    }
  }
}

/// How far from the axis line the tick labels are centred.
const TICK_LABEL_OFFSET: f64 = 12.0;

/// Half the width a text label takes at `font_size`, estimated from its
/// character count — enough to keep labels from landing on each other.
fn label_half_width(label: &str, font_size: i32) -> f64 {
  markup_char_count(label) as f64 * font_size as f64 * 0.3
}

/// The margin a frame needs outside its box for `AxesLabel` text: the tick
/// labels' own room plus the widest axis label. Capped at a fifth of the
/// frame so a long label shrinks the plot rather than squeezing it away.
fn axes_label_margin(axes_labels: &[Option<String>; 3], size: u32) -> f64 {
  let font_size = 13;
  let widest = axes_labels
    .iter()
    .flatten()
    .map(|l| label_half_width(l, font_size) * 2.0)
    .fold(0.0_f64, f64::max);
  (25.0 + widest * 0.5).min(size as f64 / 5.0)
}

/// The tick positions along an axis spanning `val_min` to `val_max`, the
/// same ones the tick marks are drawn at.
fn tick_values(val_min: f64, val_max: f64) -> Vec<f64> {
  let step = nice_step(val_max - val_min, 4);
  if step <= 0.0 {
    return Vec::new();
  }
  let mut ticks = Vec::new();
  let mut tick_val = (val_min / step).ceil() * step;
  while tick_val <= val_max + step * 0.01 {
    ticks.push(tick_val);
    tick_val += step;
  }
  ticks
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);
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

/// Parse a list of 3D points, insisting that *every* element is one — so a
/// `{x, y, z}` point is not mistaken for a list of three malformed points.
fn parse_point3d_list_strict(expr: &Expr) -> Option<Vec<Point3D>> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
  items.iter().map(parse_point3d).collect()
}

/// The unbounded `Graphics3D` primitives. They have no extent of their own:
/// a picture shows the part of them that falls inside its box, so they are
/// expanded into an ordinary `Line`/`Polygon` once that box is known.
const UNBOUNDED_3D_HEADS: [&str; 4] =
  ["InfiniteLine", "HalfLine", "InfinitePlane", "HalfPlane"];

/// Does this scene contain an unbounded primitive anywhere?
fn has_unbounded_3d(expr: &Expr) -> bool {
  match expr {
    Expr::List(items) => items.iter().any(has_unbounded_3d),
    Expr::FunctionCall { name, args } => {
      UNBOUNDED_3D_HEADS.contains(&name.as_str())
        || args.iter().any(has_unbounded_3d)
    }
    _ => false,
  }
}

/// Replace every unbounded primitive in `expr` with the `Line`/`Polygon` it
/// shows inside `bounds`, leaving the rest of the scene untouched. One that
/// misses the box entirely disappears (an empty `Line[{}]`), which is what
/// Wolfram draws for it.
fn expand_unbounded_3d(expr: &Expr, bounds: &[(f64, f64); 3]) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|i| expand_unbounded_3d(i, bounds))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args }
      if UNBOUNDED_3D_HEADS.contains(&name.as_str()) =>
    {
      unbounded_3d_to_primitive(name, args, bounds).unwrap_or_else(empty_line3d)
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| expand_unbounded_3d(a, bounds))
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

/// A `Line` with no points: the drawn form of an unbounded primitive that
/// lies entirely outside the picture's box.
fn empty_line3d() -> Expr {
  Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(Vec::new().into())].into(),
  }
}

fn v_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn v_scale(a: [f64; 3], s: f64) -> [f64; 3] {
  [a[0] * s, a[1] * s, a[2] * s]
}

fn v_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v_cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

fn v_len(a: [f64; 3]) -> f64 {
  v_dot(a, a).sqrt()
}

fn point3d_expr(p: [f64; 3]) -> Expr {
  Expr::List(vec![Expr::Real(p[0]), Expr::Real(p[1]), Expr::Real(p[2])].into())
}

/// The two points of an unbounded primitive's first argument, as either
/// `{p1, p2}` (two points on it) or `p` with a direction in `args[1]`.
fn unbounded_3d_line(args: &[Expr]) -> Option<([f64; 3], [f64; 3])> {
  if args.len() >= 2
    && let Some(p) = eval_vec3(&args[0])
    && let Some(v) = eval_vec3(&args[1])
  {
    return Some((p, v));
  }
  let Expr::List(pts) = &args[0] else {
    return None;
  };
  if pts.len() != 2 {
    return None;
  }
  let p1 = eval_vec3(&pts[0])?;
  let p2 = eval_vec3(&pts[1])?;
  Some((p1, v_sub(p2, p1)))
}

/// Clip the line `p + t v` to the box, returning the parameter interval
/// inside it (the slab method). `t_min` starts the interval, so a ray
/// (`HalfLine`) passes `0.0` and a full line `-∞`.
fn clip_line_to_box(
  p: [f64; 3],
  v: [f64; 3],
  bounds: &[(f64, f64); 3],
  t_min: f64,
) -> Option<(f64, f64)> {
  let mut lo = t_min;
  let mut hi = f64::INFINITY;
  for axis in 0..3 {
    let (min, max) = bounds[axis];
    if v[axis].abs() < 1e-12 {
      // Parallel to this slab: inside it or nowhere at all.
      if p[axis] < min - 1e-9 || p[axis] > max + 1e-9 {
        return None;
      }
      continue;
    }
    let t1 = (min - p[axis]) / v[axis];
    let t2 = (max - p[axis]) / v[axis];
    lo = lo.max(t1.min(t2));
    hi = hi.min(t1.max(t2));
  }
  if !lo.is_finite() || !hi.is_finite() || lo > hi {
    return None;
  }
  Some((lo, hi))
}

/// Clip a convex polygon against the half space `n · x <= d`
/// (Sutherland–Hodgman).
fn clip_polygon_to_halfspace(
  poly: &[[f64; 3]],
  n: [f64; 3],
  d: f64,
) -> Vec<[f64; 3]> {
  let mut out: Vec<[f64; 3]> = Vec::new();
  for i in 0..poly.len() {
    let a = poly[i];
    let b = poly[(i + 1) % poly.len()];
    let da = v_dot(n, a) - d;
    let db = v_dot(n, b) - d;
    if da <= 0.0 {
      out.push(a);
    }
    if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
      let t = da / (da - db);
      out.push(v_add(a, v_scale(v_sub(b, a), t)));
    }
  }
  out
}

/// The convex polygon a plane through `p` spanned by `u` and `w` cuts out of
/// the box. `bound` optionally adds the half plane's own edge as one more
/// half space, so `HalfPlane` stops at its boundary line.
fn clip_plane_to_box(
  p: [f64; 3],
  u: [f64; 3],
  w: [f64; 3],
  bounds: &[(f64, f64); 3],
  bound: Option<([f64; 3], f64)>,
) -> Option<Vec<[f64; 3]>> {
  let normal = v_cross(u, w);
  if v_len(normal) < 1e-12 {
    return None;
  }
  // Start from a quad on the plane large enough to span the whole box, then
  // cut it down by the box's six faces.
  let diag = (0..3)
    .map(|i| bounds[i].1 - bounds[i].0)
    .fold(0.0f64, |acc, e| acc + e * e)
    .sqrt();
  let center = [
    (bounds[0].0 + bounds[0].1) / 2.0,
    (bounds[1].0 + bounds[1].1) / 2.0,
    (bounds[2].0 + bounds[2].1) / 2.0,
  ];
  // Project the box centre onto the plane so the quad is centred on the
  // part of it the box can actually see.
  let unit_n = v_scale(normal, 1.0 / v_len(normal));
  let base = v_add(center, v_scale(unit_n, -v_dot(unit_n, v_sub(center, p))));
  let e1 = v_scale(u, 1.0 / v_len(u));
  let e2 = v_cross(unit_n, e1);
  let r = diag.max(1e-6);
  let mut poly = vec![
    v_add(v_add(base, v_scale(e1, -r)), v_scale(e2, -r)),
    v_add(v_add(base, v_scale(e1, r)), v_scale(e2, -r)),
    v_add(v_add(base, v_scale(e1, r)), v_scale(e2, r)),
    v_add(v_add(base, v_scale(e1, -r)), v_scale(e2, r)),
  ];
  for axis in 0..3 {
    let mut n = [0.0; 3];
    n[axis] = 1.0;
    poly = clip_polygon_to_halfspace(&poly, n, bounds[axis].1);
    n[axis] = -1.0;
    poly = clip_polygon_to_halfspace(&poly, n, -bounds[axis].0);
    if poly.is_empty() {
      return None;
    }
  }
  if let Some((n, d)) = bound {
    poly = clip_polygon_to_halfspace(&poly, n, d);
  }
  if poly.len() < 3 {
    return None;
  }
  Some(poly)
}

/// A box the unbounded primitives can be clipped to. A scene made only of
/// them has no bounds of its own, and one lying in a plane has an axis of
/// no extent — neither can be cut against as it stands, so an empty axis is
/// given the width of the widest one (or the unit box, when there is no
/// extent at all).
fn clipping_bounds(bounds: [(f64, f64); 3]) -> [(f64, f64); 3] {
  let widest = (0..3)
    .filter(|&i| bounds[i].0.is_finite() && bounds[i].1.is_finite())
    .map(|i| bounds[i].1 - bounds[i].0)
    .fold(0.0f64, f64::max);
  let half = if widest > 0.0 { widest / 2.0 } else { 1.0 };
  std::array::from_fn(|i| {
    let (lo, hi) = bounds[i];
    if !lo.is_finite() || !hi.is_finite() {
      return (-half, half);
    }
    if hi > lo {
      return (lo, hi);
    }
    let mid = (lo + hi) / 2.0;
    (mid - half, mid + half)
  })
}

/// The `Line`/`Polygon` an unbounded primitive draws inside `bounds`.
fn unbounded_3d_to_primitive(
  head: &str,
  args: &[Expr],
  bounds: &[(f64, f64); 3],
) -> Option<Expr> {
  if args.is_empty()
    || bounds
      .iter()
      .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || hi <= lo)
  {
    return None;
  }
  match head {
    // A line or ray, clipped to the box.
    "InfiniteLine" | "HalfLine" => {
      let (p, v) = unbounded_3d_line(args)?;
      if v_len(v) < 1e-12 {
        return None;
      }
      let t_min = if head == "HalfLine" {
        0.0
      } else {
        f64::NEG_INFINITY
      };
      let (lo, hi) = clip_line_to_box(p, v, bounds, t_min)?;
      Some(Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(
          vec![
            point3d_expr(v_add(p, v_scale(v, lo))),
            point3d_expr(v_add(p, v_scale(v, hi))),
          ]
          .into(),
        )]
        .into(),
      })
    }
    // `InfinitePlane[{p1, p2, p3}]` — the plane through three points —
    // or `InfinitePlane[p, {v1, v2}]` — through `p`, spanned by `v1`, `v2`.
    "InfinitePlane" => {
      let (p, u, w) = if args.len() >= 2 {
        let p = eval_vec3(&args[0])?;
        let Expr::List(dirs) = &args[1] else {
          return None;
        };
        if dirs.len() != 2 {
          return None;
        }
        (p, eval_vec3(&dirs[0])?, eval_vec3(&dirs[1])?)
      } else {
        let Expr::List(pts) = &args[0] else {
          return None;
        };
        if pts.len() != 3 {
          return None;
        }
        let p1 = eval_vec3(&pts[0])?;
        let p2 = eval_vec3(&pts[1])?;
        let p3 = eval_vec3(&pts[2])?;
        (p1, v_sub(p2, p1), v_sub(p3, p1))
      };
      let poly = clip_plane_to_box(p, u, w, bounds, None)?;
      Some(polygon3d_expr(&poly))
    }
    // `HalfPlane[{p1, p2}, v]` — bounded by the line through `p1`, `p2`,
    // reaching out on the side `v` points to — or `HalfPlane[p, d, v]`,
    // where `d` is the boundary line's direction.
    "HalfPlane" => {
      let (p, d, v) = if args.len() >= 3 {
        (
          eval_vec3(&args[0])?,
          eval_vec3(&args[1])?,
          eval_vec3(&args[2])?,
        )
      } else {
        let (p, d) = unbounded_3d_line(&args[..1])?;
        (p, d, eval_vec3(args.get(1)?)?)
      };
      // The outward normal of the boundary: in the plane, across the edge,
      // pointing away from `v`.
      let normal = v_cross(v_cross(d, v), d);
      if v_len(normal) < 1e-12 {
        return None;
      }
      let outward = v_scale(normal, -1.0);
      let poly =
        clip_plane_to_box(p, d, v, bounds, Some((outward, v_dot(outward, p))))?;
      Some(polygon3d_expr(&poly))
    }
    _ => None,
  }
}

fn polygon3d_expr(poly: &[[f64; 3]]) -> Expr {
  Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(
      poly
        .iter()
        .map(|p| point3d_expr(*p))
        .collect::<Vec<_>>()
        .into(),
    )]
    .into(),
  }
}

/// A 3D primitive for Graphics3D
#[derive(Clone)]
struct StyleState3D {
  color: Option<(u8, u8, u8)>,
  /// `FaceForm[front, back]`: the colour of faces turned away from the
  /// viewer. None means both sides use `color`.
  back_color: Option<(u8, u8, u8)>,
  opacity: f64,
  /// Stroke width in pixels for Line3D primitives. None means default 1.5.
  thickness: Option<f64>,
  /// Whether an open surface's ends are closed off, as `CapForm` sets it.
  /// `CapForm[None]` leaves a `Tube` hollow; every other form caps it.
  capped: bool,
  /// Whether a face is outlined. `EdgeForm[]` asks for no edge at all,
  /// which is how a dissection shows its pieces as flat colour.
  edges: bool,
  /// The colour `EdgeForm[colour]` asks outlines to be drawn in. `None`
  /// leaves them the renderer's default dark grey. An explicit colour also
  /// makes the outline opaque and is what turns a curved primitive's
  /// silhouette on: `{Opacity[0], EdgeForm[Black], Cylinder[…]}` is the
  /// Demonstrations idiom for an unfilled circle in space.
  edge_color: Option<(u8, u8, u8)>,
  /// `Specularity[colour, exponent]`: the highlight a shiny surface throws
  /// back at the viewer, as `(colour, exponent)`. `None` is the default
  /// matte surface. A larger exponent tightens the highlight.
  specular: Option<((u8, u8, u8), f64)>,
}

impl Default for StyleState3D {
  fn default() -> Self {
    Self {
      color: None, // None means use default blue
      back_color: None,
      opacity: 1.0,
      thickness: None,
      capped: true,
      edges: true,
      edge_color: None,
      specular: None,
    }
  }
}

#[derive(Clone)]
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
    /// Boundaries cut out of the face (`Polygon[outer -> holes]`). Empty
    /// for an ordinary polygon.
    holes: Vec<Vec<Point3D>>,
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
  /// `Text[expr, {x, y, z}, offset]`: a label pinned to a point of the
  /// scene. Text does not turn with the camera — it is drawn flat, facing
  /// the viewer, at the projection of its point, which is what makes a
  /// labelled 3D schematic readable from any view.
  Text3D {
    /// The label as SVG `<text>` content (already typeset and escaped).
    label: String,
    pos: Point3D,
    /// Which point of the label's own box sits at `pos`, from -1
    /// (left/bottom) to 1 (right/top). `(0, 0)` centres it.
    offset: (f64, f64),
    font_size: f64,
    /// Character count of the visible text, for placing the box an
    /// `offset` displaces.
    width_chars: usize,
    style: StyleState3D,
  },
  /// A pre-tessellated triangle surface (Torus, FilledTorus, BSplineSurface,
  /// Raster3D voxels, …). `smooth` marks the ones that approximate a curved
  /// surface, whose triangle edges are internal cuts rather than outlines.
  Surface3D {
    tris: Vec<(Point3D, Point3D, Point3D)>,
    style: StyleState3D,
    smooth: bool,
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
    style.back_color = None;
    if color.a < 1.0 {
      style.opacity = color.a;
    }
    return true;
  }

  // FaceForm[front] colours both sides; FaceForm[front, back] gives the
  // side turned away from the viewer its own colour.
  if let Expr::FunctionCall { name, args } = expr
    && name == "FaceForm"
    && !args.is_empty()
  {
    let mut front = StyleState3D::default();
    if !apply_3d_directive(&args[0], &mut front) {
      return true;
    }
    style.color = front.color;
    style.opacity = front.opacity;
    style.back_color = match args.get(1) {
      Some(back) => {
        let mut b = StyleState3D::default();
        apply_3d_directive(back, &mut b)
          .then_some(b.color)
          .flatten()
      }
      None => None,
    };
    return true;
  }

  match expr {
    Expr::Identifier(s) => match s.as_str() {
      "Thick" => {
        style.thickness = Some(2.5);
        return true;
      }
      "Thin" => {
        style.thickness = Some(0.5);
        return true;
      }
      _ => {}
    },
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Opacity" if !args.is_empty() => {
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
      "Directive" => {
        for a in args {
          apply_3d_directive(a, style);
        }
        return true;
      }
      // `Specularity[colour]` / `Specularity[colour, exponent]` makes the
      // surfaces that follow shiny. It never repaints them: the colour it
      // carries is the *highlight's*, so it must be consumed here rather
      // than fall through to the generic recursion, which would pick the
      // colour up and use it as the face colour (a Demonstration's
      // `{GrayLevel[.25], Specularity[White, 10], Sphere[…]}` would render
      // white instead of dark grey).
      "Specularity" if !args.is_empty() => {
        style.specular = parse_specularity(args);
        return true;
      }
      // CapForm[None] leaves an open surface's ends open; every named form
      // ("Butt", "Square", "Round") closes them, as does the default.
      "CapForm" if args.len() == 1 => {
        style.capped = !matches!(&args[0], Expr::Identifier(s) if s == "None");
        return true;
      }
      // `EdgeForm[]` (and `EdgeForm[None]`) asks for faces with no outline;
      // any other form keeps the default edge. A colour directive among the
      // arguments — `EdgeForm[Black]`, `EdgeForm[{Thick, Red}]` — also sets
      // the outline's colour.
      "EdgeForm" => {
        style.edges = !(args.is_empty()
          || matches!(args.first(), Some(Expr::Identifier(s)) if s == "None"));
        style.edge_color = None;
        if style.edges {
          let flat = match args.first() {
            Some(Expr::List(items)) => items.to_vec(),
            _ => args.to_vec(),
          };
          for a in &flat {
            if let Some(c) = parse_color(a) {
              style.edge_color = Some((
                (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
              ));
            }
          }
        }
        return true;
      }
      "Thickness" if args.len() == 1 => {
        if let Expr::Identifier(s) = &args[0] {
          match s.as_str() {
            "Large" => style.thickness = Some(3.0),
            "Tiny" => style.thickness = Some(0.5),
            _ => {
              if let Some(t) = expr_to_f64(&args[0]) {
                // Relative thickness: fraction of plot width (~360 px)
                style.thickness = Some(t * 360.0);
              }
            }
          }
        } else if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = Some(t * 360.0);
        }
        return true;
      }
      "AbsoluteThickness" if args.len() == 1 => {
        if let Some(t) = expr_to_f64(&args[0]) {
          style.thickness = Some(t);
        }
        return true;
      }
      _ => {}
    },
    Expr::List(items) => {
      // {Red, Thick, …} – apply each element as a sub-directive
      for it in items {
        apply_3d_directive(it, style);
      }
      return true;
    }
    _ => {}
  }

  false
}

/// Read a `Specularity[…]` argument list into `(highlight colour, exponent)`.
/// The reflectance may be a colour (`Specularity[White, 10]`) or a plain
/// number standing for that grey level (`Specularity[.5]`); the exponent
/// defaults to Wolfram's `1`. A reflectance of zero (black) means a matte
/// surface, i.e. no highlight at all.
fn parse_specularity(args: &[Expr]) -> Option<((u8, u8, u8), f64)> {
  use crate::functions::graphics::parse_color;
  use crate::functions::math_ast::expr_to_f64;

  let color = match parse_color(&args[0]) {
    Some(c) => c,
    None => {
      let g = expr_to_f64(&args[0])?;
      crate::functions::graphics::Color::new(g, g, g)
    }
  };
  let rgb = (
    (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
    (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
    (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
  );
  if rgb == (0, 0, 0) {
    return None;
  }
  let exponent = args
    .get(1)
    .and_then(expr_to_f64)
    .filter(|n| *n > 0.0)
    .unwrap_or(1.0);
  Some((rgb, exponent))
}

/// Collect 3D primitives from an expression.
/// An affine 3D transform (`p ↦ m·p + t`) built from `Translate`, `Rotate`,
/// or `Scale` wrappers and applied to already-collected primitives.
struct Affine3 {
  m: [[f64; 3]; 3],
  t: [f64; 3],
}

impl Affine3 {
  fn apply(&self, p: Point3D) -> Point3D {
    Point3D {
      x: self.m[0][0] * p.x
        + self.m[0][1] * p.y
        + self.m[0][2] * p.z
        + self.t[0],
      y: self.m[1][0] * p.x
        + self.m[1][1] * p.y
        + self.m[1][2] * p.z
        + self.t[1],
      z: self.m[2][0] * p.x
        + self.m[2][1] * p.y
        + self.m[2][2] * p.z
        + self.t[2],
    }
  }

  /// Uniform length-scale factor of the linear part (cube root of |det|),
  /// used to scale the radii of Sphere/Cylinder/Cone primitives.
  fn length_scale(&self) -> f64 {
    let m = &self.m;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    det.abs().cbrt()
  }

  /// Does the linear part stretch every direction by the same factor?
  ///
  /// A similarity (rotation and/or uniform scaling) maps a sphere to a
  /// sphere, so `Sphere`/`Cylinder`/`Cone` primitives can keep their
  /// analytic form and just take a scaled radius. An anisotropic
  /// `Scale[g, {sx, sy, sz}]` does not — it turns a sphere into an
  /// ellipsoid — so those primitives have to be tessellated first and the
  /// transform applied to their vertices (see `transform_primitive3d`).
  fn is_similarity(&self) -> bool {
    // MᵀM is a multiple of the identity exactly when the columns are
    // mutually orthogonal and share one length.
    let col = |j: usize| [self.m[0][j], self.m[1][j], self.m[2][j]];
    let dot =
      |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (c0, c1, c2) = (col(0), col(1), col(2));
    let n0 = dot(c0, c0);
    let n1 = dot(c1, c1);
    let n2 = dot(c2, c2);
    let tol = 1e-9 * n0.max(n1).max(n2).max(1.0);
    (n0 - n1).abs() <= tol
      && (n1 - n2).abs() <= tol
      && dot(c0, c1).abs() <= tol
      && dot(c1, c2).abs() <= tol
      && dot(c0, c2).abs() <= tol
  }

  fn translation(v: [f64; 3]) -> Self {
    Affine3 {
      m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      t: v,
    }
  }

  /// Rotation by `angle` around the axis direction `w` through `anchor`
  /// (Rodrigues' formula).
  fn rotation(angle: f64, w: [f64; 3], anchor: [f64; 3]) -> Option<Self> {
    let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if !len.is_finite() || len < 1e-12 {
      return None;
    }
    let (ux, uy, uz) = (w[0] / len, w[1] / len, w[2] / len);
    let (s, c) = angle.sin_cos();
    let ic = 1.0 - c;
    let m = [
      [
        c + ux * ux * ic,
        ux * uy * ic - uz * s,
        ux * uz * ic + uy * s,
      ],
      [
        uy * ux * ic + uz * s,
        c + uy * uy * ic,
        uy * uz * ic - ux * s,
      ],
      [
        uz * ux * ic - uy * s,
        uz * uy * ic + ux * s,
        c + uz * uz * ic,
      ],
    ];
    // Conjugate with the anchor so the axis passes through it:
    // p ↦ m·(p - anchor) + anchor.
    let t = [
      anchor[0]
        - (m[0][0] * anchor[0] + m[0][1] * anchor[1] + m[0][2] * anchor[2]),
      anchor[1]
        - (m[1][0] * anchor[0] + m[1][1] * anchor[1] + m[1][2] * anchor[2]),
      anchor[2]
        - (m[2][0] * anchor[0] + m[2][1] * anchor[1] + m[2][2] * anchor[2]),
    ];
    Some(Affine3 { m, t })
  }

  fn scaling(factors: [f64; 3], center: [f64; 3]) -> Self {
    Affine3 {
      m: [
        [factors[0], 0.0, 0.0],
        [0.0, factors[1], 0.0],
        [0.0, 0.0, factors[2]],
      ],
      t: [
        center[0] * (1.0 - factors[0]),
        center[1] * (1.0 - factors[1]),
        center[2] * (1.0 - factors[2]),
      ],
    }
  }
}

/// Numeric `{x, y, z}` from an expression, evaluating each component.
fn eval_vec3(expr: &Expr) -> Option<[f64; 3]> {
  let Expr::List(items) = &evaluate_expr_to_expr(expr).ok()? else {
    return None;
  };
  if items.len() != 3 {
    return None;
  }
  let mut v = [0.0; 3];
  for (slot, item) in v.iter_mut().zip(items.iter()) {
    *slot = try_eval_to_f64(item)?;
  }
  Some(v)
}

/// Build the transform list for a `Translate`/`Rotate`/`Scale` wrapper from
/// its arguments (past the wrapped graphics). `Translate` with a list of
/// offset vectors produces one transform per copy; the others produce one.
///
/// `content_center` is the centre of the wrapped graphics' bounding box:
/// `Scale[g, s]` without an explicit centre scales *about that point*, not
/// about the origin, so `Scale[Sphere[{1.4, 0, 0}], 0.07]` leaves the
/// sphere where it is and only shrinks it.
fn parse_3d_transforms(
  name: &str,
  args: &[Expr],
  content_center: [f64; 3],
) -> Option<Vec<Affine3>> {
  match name {
    "Translate" if args.len() == 1 => {
      // Either a single {dx, dy, dz} or a list of offset vectors.
      if let Some(v) = eval_vec3(&args[0]) {
        return Some(vec![Affine3::translation(v)]);
      }
      if let Ok(Expr::List(ref rows)) = evaluate_expr_to_expr(&args[0]) {
        let offsets: Option<Vec<[f64; 3]>> =
          rows.iter().map(eval_vec3).collect();
        return offsets
          .map(|os| os.into_iter().map(Affine3::translation).collect());
      }
      None
    }
    "Rotate" if !args.is_empty() => {
      let angle = try_eval_to_f64(&evaluate_expr_to_expr(&args[0]).ok()?)?;
      match args.len() {
        // Rotate[g, θ, w]: axis direction w through the origin.
        // Rotate[g, θ, {p1, p2}]: axis through the points p1 and p2.
        2 => {
          if let Some(w) = eval_vec3(&args[1]) {
            return Affine3::rotation(angle, w, [0.0; 3]).map(|a| vec![a]);
          }
          if let Ok(Expr::List(ref pts)) = evaluate_expr_to_expr(&args[1])
            && pts.len() == 2
            && let (Some(p1), Some(p2)) =
              (eval_vec3(&pts[0]), eval_vec3(&pts[1]))
          {
            let w = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
            return Affine3::rotation(angle, w, p1).map(|a| vec![a]);
          }
          None
        }
        // Rotate[g, θ, w, p]: axis direction w through the point p.
        3 => {
          let w = eval_vec3(&args[1])?;
          let p = eval_vec3(&args[2])?;
          Affine3::rotation(angle, w, p).map(|a| vec![a])
        }
        _ => None,
      }
    }
    "Scale" if !args.is_empty() => {
      let factors = if let Some(f) = eval_vec3(&args[0]) {
        f
      } else {
        let s = try_eval_to_f64(&evaluate_expr_to_expr(&args[0]).ok()?)?;
        [s, s, s]
      };
      let center = args.get(1).and_then(eval_vec3).unwrap_or(content_center);
      Some(vec![Affine3::scaling(factors, center)])
    }
    _ => None,
  }
}

/// Centre of the bounding box of already-collected primitives, the point
/// `Scale[g, s]` scales about when no centre is given. An empty collection
/// has no box, so it falls back to the origin.
fn content_center(prims: &[Primitive3D]) -> [f64; 3] {
  if prims.is_empty() {
    return [0.0; 3];
  }
  let [(x0, x1), (y0, y1), (z0, z1)] = primitives_bounds(prims);
  let mid = |lo: f64, hi: f64| {
    if lo.is_finite() && hi.is_finite() {
      (lo + hi) / 2.0
    } else {
      0.0
    }
  };
  [mid(x0, x1), mid(y0, y1), mid(z0, z1)]
}

/// The style attached to a collected primitive.
fn primitive_style(prim: &Primitive3D) -> &StyleState3D {
  match prim {
    Primitive3D::Sphere { style, .. }
    | Primitive3D::Cuboid { style, .. }
    | Primitive3D::Polygon3D { style, .. }
    | Primitive3D::Line3D { style, .. }
    | Primitive3D::Point3DPrim { style, .. }
    | Primitive3D::Arrow3D { style, .. }
    | Primitive3D::Cylinder { style, .. }
    | Primitive3D::Cone { style, .. }
    | Primitive3D::Text3D { style, .. }
    | Primitive3D::Surface3D { style, .. } => style,
  }
}

/// Triangles for the curved primitives that an anisotropic transform turns
/// into a shape they cannot represent. Returns `None` for primitives whose
/// vertices already carry their whole geometry (polygons, lines, …) — those
/// transform exactly, point by point.
fn tessellate_for_transform(
  prim: &Primitive3D,
) -> Option<Vec<(Point3D, Point3D, Point3D)>> {
  match prim {
    Primitive3D::Sphere { center, radius, .. } => {
      Some(tessellate_sphere(center, *radius, (16, 24)))
    }
    Primitive3D::Cylinder { p1, p2, radius, .. } => {
      Some(tessellate_cylinder(p1, p2, *radius))
    }
    Primitive3D::Cone { p1, p2, radius, .. } => {
      Some(tessellate_cone(p1, p2, *radius))
    }
    _ => None,
  }
}

/// Apply an affine transform to a collected primitive in place.
fn transform_primitive3d(prim: &mut Primitive3D, xf: &Affine3) {
  let scale = xf.length_scale();
  // An anisotropic transform bends a sphere into an ellipsoid and a
  // cylinder/cone into an elliptic one — shapes the analytic primitives
  // cannot express. Tessellate first, then transform the vertices, and
  // keep the result marked `smooth` so it still shades as a curved
  // surface rather than growing facet outlines.
  if !xf.is_similarity()
    && let Some(tris) = tessellate_for_transform(prim)
  {
    let style = primitive_style(prim).clone();
    *prim = Primitive3D::Surface3D {
      tris: tris
        .into_iter()
        .map(|(a, b, c)| (xf.apply(a), xf.apply(b), xf.apply(c)))
        .collect(),
      style,
      smooth: true,
    };
    return;
  }
  match prim {
    Primitive3D::Sphere { center, radius, .. } => {
      *center = xf.apply(*center);
      *radius *= scale;
    }
    Primitive3D::Cuboid { p_min, p_max, .. } => {
      // Transform both corners and re-normalize; the box stays
      // axis-aligned, so rotations are only approximated.
      let a = xf.apply(*p_min);
      let b = xf.apply(*p_max);
      *p_min = Point3D {
        x: a.x.min(b.x),
        y: a.y.min(b.y),
        z: a.z.min(b.z),
      };
      *p_max = Point3D {
        x: a.x.max(b.x),
        y: a.y.max(b.y),
        z: a.z.max(b.z),
      };
    }
    Primitive3D::Polygon3D { points, holes, .. } => {
      for p in points.iter_mut().chain(holes.iter_mut().flatten()) {
        *p = xf.apply(*p);
      }
    }
    Primitive3D::Point3DPrim { points, .. }
    | Primitive3D::Arrow3D { points, .. } => {
      for p in points {
        *p = xf.apply(*p);
      }
    }
    Primitive3D::Text3D { pos, .. } => {
      *pos = xf.apply(*pos);
    }
    Primitive3D::Line3D { segments, .. } => {
      for seg in segments {
        for p in seg {
          *p = xf.apply(*p);
        }
      }
    }
    Primitive3D::Cylinder { p1, p2, radius, .. }
    | Primitive3D::Cone { p1, p2, radius, .. } => {
      *p1 = xf.apply(*p1);
      *p2 = xf.apply(*p2);
      *radius *= scale;
    }
    Primitive3D::Surface3D { tris, .. } => {
      for (a, b, c) in tris {
        *a = xf.apply(*a);
        *b = xf.apply(*b);
        *c = xf.apply(*c);
      }
    }
  }
}

/// Replace the 1-based vertex indices inside a `GraphicsComplex` primitive
/// (`Polygon`/`Line`/`Point`/`Arrow` arguments) with the coordinates they
/// refer to, leaving everything else untouched.
fn resolve_complex_indices(expr: &Expr, points: &[Expr]) -> Expr {
  fn substitute(expr: &Expr, points: &[Expr]) -> Expr {
    match expr {
      Expr::Integer(i) if *i >= 1 && (*i as usize) <= points.len() => {
        points[*i as usize - 1].clone()
      }
      Expr::List(items) => Expr::List(
        items
          .iter()
          .map(|e| substitute(e, points))
          .collect::<Vec<_>>()
          .into(),
      ),
      other => other.clone(),
    }
  }
  match expr {
    Expr::FunctionCall { name, args }
      if matches!(name.as_str(), "Polygon" | "Line" | "Point" | "Arrow")
        && !args.is_empty() =>
    {
      let mut new_args = args.to_vec();
      new_args[0] = substitute(&args[0], points);
      Expr::FunctionCall {
        name: name.clone(),
        args: new_args.into(),
      }
    }
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| resolve_complex_indices(e, points))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| resolve_complex_indices(e, points))
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

/// The alignment offset of `Text[expr, pos, offset]`: a pair running from
/// -1 (left/bottom) to 1 (right/top), written either as numbers or with the
/// alignment symbols.
fn parse_text3d_offset(spec: &Expr) -> Option<(f64, f64)> {
  fn component(e: &Expr, horizontal: bool) -> Option<f64> {
    if let Expr::Identifier(s) = e {
      return match (s.as_str(), horizontal) {
        ("Left", true) | ("Bottom", false) => Some(-1.0),
        ("Center", _) | ("Automatic", _) | ("Axis", _) | ("Baseline", _) => {
          Some(0.0)
        }
        ("Right", true) | ("Top", false) => Some(1.0),
        _ => None,
      };
    }
    crate::functions::math_ast::expr_to_f64(e)
  }
  match spec {
    Expr::List(items) if items.len() == 2 => {
      Some((component(&items[0], true)?, component(&items[1], false)?))
    }
    _ => None,
  }
}

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
        "Sphere" | "Ball" => {
          // `Sphere[{p1, p2, …}, r]` is a whole set of spheres of the same
          // radius, one per centre — how a scene marks several points at
          // once. A single `{x, y, z}` is the one-centre case of that.
          let origin = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
          };
          let centers = match args.first() {
            None => vec![origin],
            Some(arg) => match parse_point3d(arg) {
              Some(p) => vec![p],
              None => parse_point3d_list_strict(arg).unwrap_or(vec![origin]),
            },
          };
          let radius = if args.len() >= 2 {
            try_eval_to_f64(
              &evaluate_expr_to_expr(&args[1]).unwrap_or(args[1].clone()),
            )
            .unwrap_or(1.0)
          } else {
            1.0
          };
          for center in centers {
            prims.push(Primitive3D::Sphere {
              center,
              radius,
              style: style.clone(),
            });
          }
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
              holes: Vec::new(),
              style: style.clone(),
            });
          } else if let Some((outer, holes)) =
            crate::functions::polygon_holes::split_holes(
              &args[0],
              &parse_point3d_list,
            )
          {
            // Polygon[outer -> holes]: a face with the hole boundaries
            // cut out of it.
            prims.push(Primitive3D::Polygon3D {
              points: outer,
              holes,
              style: style.clone(),
            });
          } else if let Expr::List(poly_exprs) = &args[0] {
            // Polygon[{{p1, p2, …}, {q1, q2, …}, …}]: a collection of
            // polygons in one primitive.
            for poly in poly_exprs.iter() {
              if let Some(pts) = parse_point3d_list(poly) {
                prims.push(Primitive3D::Polygon3D {
                  points: pts,
                  holes: Vec::new(),
                  style: style.clone(),
                });
              } else if let Some((outer, holes)) =
                crate::functions::polygon_holes::split_holes(
                  poly,
                  &parse_point3d_list,
                )
              {
                prims.push(Primitive3D::Polygon3D {
                  points: outer,
                  holes,
                  style: style.clone(),
                });
              }
            }
          }
        }
        "Line" if !args.is_empty() => {
          if let Some(pts) = parse_point3d_list(&args[0]) {
            prims.push(Primitive3D::Line3D {
              segments: vec![pts],
              style: style.clone(),
            });
          } else if let Expr::List(seg_exprs) = &args[0] {
            // Line[{{p1, p2, …}, {q1, q2, …}, …}]: a collection of
            // polylines in one primitive.
            let segments: Option<Vec<Vec<Point3D>>> =
              seg_exprs.iter().map(parse_point3d_list).collect();
            if let Some(segments) = segments
              && !segments.is_empty()
            {
              prims.push(Primitive3D::Line3D {
                segments,
                style: style.clone(),
              });
            }
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
        // `Text[expr, {x, y, z}]` (optionally with an alignment offset)
        // labels a point of the scene. The label is typeset by the same
        // helper the 2D pictures and plot labels use, so `Subscript[N, B]`
        // reads the same wherever it is written.
        "Text" if args.len() >= 2 => {
          if let Some(pos) = parse_point3d(&args[1]) {
            let styled = crate::functions::chart::parse_styled_label(&args[0]);
            let (label, width_chars, font_size, color) = match styled {
              Some(s) => (
                s.svg(),
                s.text.chars().count(),
                s.font_size.unwrap_or(12.0),
                s.color,
              ),
              None => (String::new(), 0, 12.0, None),
            };
            if !label.is_empty() {
              let mut text_style = style.clone();
              if let Some(c) = color {
                text_style.color = Some((
                  (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                ));
              }
              prims.push(Primitive3D::Text3D {
                label,
                pos,
                offset: args
                  .get(2)
                  .and_then(parse_text3d_offset)
                  .unwrap_or((0.0, 0.0)),
                font_size,
                width_chars,
                style: text_style,
              });
            }
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
        // Tube[curve], Tube[curve, r] and Tube[curve, {r1, …, rn}] — a
        // surface of revolution about a polyline, one radius per vertex.
        // `curve` is a point list, a `Line[…]`, or a list of either.
        "Tube" if !args.is_empty() => {
          // `Tube[BSplineCurve[…], r]` runs the tube along the spline, so
          // the curve is sampled first — a tube around the bare control
          // points would cut every corner the spline rounds.
          let spline = match &args[0] {
            Expr::FunctionCall { name, args: inner }
              if name == "BSplineCurve" && !inner.is_empty() =>
            {
              bspline_curve_points(inner)
            }
            _ => None,
          };
          let curve = match &args[0] {
            Expr::FunctionCall { name, args: inner }
              if name == "Line" && !inner.is_empty() =>
            {
              inner[0].clone()
            }
            other => other.clone(),
          };
          let curve = evaluate_expr_to_expr(&curve).unwrap_or(curve);
          let curves: Vec<Vec<Point3D>> = match spline {
            Some(pts) => vec![pts],
            None => match parse_point3d_list(&curve) {
              Some(pts) => vec![pts],
              None => match &curve {
                Expr::List(items) => {
                  items.iter().filter_map(parse_point3d_list).collect()
                }
                _ => vec![],
              },
            },
          };
          for pts in curves {
            let radii: Vec<f64> = match args.get(1) {
              Some(spec) => {
                let spec = evaluate_expr_to_expr(spec).unwrap_or(spec.clone());
                match &spec {
                  Expr::List(items) if items.len() == pts.len() => items
                    .iter()
                    .map(|e| {
                      try_eval_to_f64(
                        &evaluate_expr_to_expr(e).unwrap_or(e.clone()),
                      )
                      .unwrap_or(0.0)
                    })
                    .collect(),
                  _ => {
                    let r = try_eval_to_f64(&spec).unwrap_or(0.0);
                    vec![r; pts.len()]
                  }
                }
              }
              // Wolfram sizes an unspecified tube radius from the scene
              // rather than fixing it; a hundredth of the curve's own
              // extent is the same kind of thin, and is all the notebooks
              // that omit it need (they nearly always give one).
              None => {
                let extent = pts
                  .iter()
                  .flat_map(|a| {
                    pts.iter().map(move |b| {
                      ((a.x - b.x).powi(2)
                        + (a.y - b.y).powi(2)
                        + (a.z - b.z).powi(2))
                      .sqrt()
                    })
                  })
                  .fold(0.0f64, f64::max);
                vec![extent / 100.0; pts.len()]
              }
            };
            let tris = tessellate_tube(&pts, &radii, style.capped);
            if !tris.is_empty() {
              prims.push(Primitive3D::Surface3D {
                tris,
                style: style.clone(),
                smooth: true,
              });
            }
          }
        }
        "Torus" | "FilledTorus" => {
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
          let (r1, r2) = match args.get(1) {
            Some(Expr::List(radii)) if radii.len() == 2 => {
              let num = |e: &Expr| {
                try_eval_to_f64(&evaluate_expr_to_expr(e).unwrap_or(e.clone()))
              };
              (num(&radii[0]).unwrap_or(0.5), num(&radii[1]).unwrap_or(1.0))
            }
            _ => (0.5, 1.0),
          };
          prims.push(Primitive3D::Surface3D {
            tris: tessellate_torus(&center, r1, r2),
            style: style.clone(),
            smooth: true,
          });
        }
        // `BSplineCurve[{p1, …}, opts…]` draws the spline its control
        // points define. (`Tube[BSplineCurve[…], r]` thickens the same
        // curve — see the `Tube` arm above.)
        "BSplineCurve" if !args.is_empty() => {
          if let Some(pts) = bspline_curve_points(args)
            && pts.len() >= 2
          {
            prims.push(Primitive3D::Line3D {
              segments: vec![pts],
              style: style.clone(),
            });
          }
        }
        "BSplineSurface" if !args.is_empty() => {
          if let Some(grid) = parse_point3d_grid(&args[0]) {
            prims.push(Primitive3D::Surface3D {
              tris: tessellate_bspline_surface(&grid),
              style: style.clone(),
              smooth: true,
            });
          }
        }
        "Raster3D" if !args.is_empty() => {
          collect_raster3d(&args[0], style, prims);
        }
        // GraphicsComplex[points, prims]: primitives reference the shared
        // coordinate list by 1-based index.
        "GraphicsComplex" if args.len() >= 2 => {
          if let Ok(Expr::List(ref points)) = evaluate_expr_to_expr(&args[0]) {
            let resolved = resolve_complex_indices(&args[1], points);
            let saved = style.clone();
            collect_3d_primitives(&resolved, style, prims);
            *style = saved;
          }
        }
        // Geometric transforms: collect the wrapped graphics, then map the
        // primitives through the affine transform (per copy for the
        // multi-offset Translate form).
        "Translate" | "Rotate" | "Scale" if args.len() >= 2 => {
          let mut sub = Vec::new();
          let saved = style.clone();
          collect_3d_primitives(&args[0], style, &mut sub);
          *style = saved;
          match parse_3d_transforms(name, &args[1..], content_center(&sub)) {
            Some(transforms) => {
              for xf in &transforms {
                for prim in &sub {
                  let mut copy = prim.clone();
                  transform_primitive3d(&mut copy, xf);
                  prims.push(copy);
                }
              }
            }
            // Unrecognized transform spec: keep the primitives untransformed
            // rather than dropping them.
            None => prims.extend(sub),
          }
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
/// Pick a sphere's tessellation detail `(n_lat, n_lon)`. Small scenes
/// always get full detail; in a scene with many spheres (e.g. a
/// Demonstrations circle packing with thousands of them) each sphere's
/// detail scales with its size relative to the scene so the SVG stays a
/// tractable size — a sphere spanning a few pixels doesn't need 768
/// triangles.
fn sphere_detail(
  radius: f64,
  scene_extent: f64,
  sphere_count: usize,
) -> (usize, usize) {
  if sphere_count <= 32 || scene_extent <= 0.0 {
    return (16, 24);
  }
  let rel = radius / scene_extent;
  if rel > 0.08 {
    (12, 18)
  } else if rel > 0.04 {
    (8, 12)
  } else if rel > 0.02 {
    (6, 9)
  } else {
    (4, 6)
  }
}

fn tessellate_sphere(
  center: &Point3D,
  radius: f64,
  detail: (usize, usize),
) -> Vec<(Point3D, Point3D, Point3D)> {
  let (n_lat, n_lon) = detail;
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

/// Tessellate a torus with inner radius r1 and outer radius r2 (so the tube
/// of radius (r2 - r1)/2 follows a circle of radius (r1 + r2)/2 in the
/// z = center.z plane).
fn tessellate_torus(
  center: &Point3D,
  r1: f64,
  r2: f64,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let ring = (r1 + r2) / 2.0;
  let tube = (r2 - r1) / 2.0;
  let n_u = 32;
  let n_v = 16;
  let pi = std::f64::consts::PI;
  let p = |u: f64, v: f64| -> Point3D {
    Point3D {
      x: center.x + (ring + tube * v.cos()) * u.cos(),
      y: center.y + (ring + tube * v.cos()) * u.sin(),
      z: center.z + tube * v.sin(),
    }
  };
  let mut tris = Vec::new();
  for i in 0..n_u {
    let u1 = 2.0 * pi * i as f64 / n_u as f64;
    let u2 = 2.0 * pi * (i + 1) as f64 / n_u as f64;
    for j in 0..n_v {
      let v1 = 2.0 * pi * j as f64 / n_v as f64;
      let v2 = 2.0 * pi * (j + 1) as f64 / n_v as f64;
      let a = p(u1, v1);
      let b = p(u2, v1);
      let c = p(u2, v2);
      let d = p(u1, v2);
      tris.push((a, b, c));
      tris.push((a, c, d));
    }
  }
  tris
}

/// Parse a rectangular grid of 3D control points (a list of equal-length
/// rows with at least 2 rows and 2 columns).
fn parse_point3d_grid(expr: &Expr) -> Option<Vec<Vec<Point3D>>> {
  let Expr::List(rows) = expr else {
    return None;
  };
  let mut grid = Vec::with_capacity(rows.len());
  for row in rows {
    grid.push(parse_point3d_list(row)?);
  }
  if grid.len() >= 2
    && grid[0].len() >= 2
    && grid.iter().all(|r| r.len() == grid[0].len())
  {
    Some(grid)
  } else {
    None
  }
}

/// B-spline basis weights for `n` control points sampled at `num_samples`
/// evenly spaced parameter values (degree min(3, n-1), clamped uniform
/// knots): one weight row per sample.
fn bspline_sample_weights(n: usize, num_samples: usize) -> Vec<Vec<f64>> {
  let degree = 3usize.min(n - 1);
  let num_knots = n + degree + 1;
  let mut knots = Vec::with_capacity(num_knots);
  knots.extend(std::iter::repeat_n(0.0, degree + 1));
  let num_internal = num_knots - 2 * (degree + 1);
  for i in 1..=num_internal {
    knots.push(i as f64);
  }
  let max_knot = (num_internal + 1) as f64;
  knots.extend(std::iter::repeat_n(max_knot, degree + 1));
  let t_min = knots[degree];
  let t_max = knots[n];
  (0..num_samples)
    .map(|s| {
      let t = t_min + (t_max - t_min) * s as f64 / (num_samples - 1) as f64;
      (0..n)
        .map(|j| {
          crate::functions::graphics::bspline_basis(j, degree, t, &knots)
        })
        .collect()
    })
    .collect()
}

/// The polyline a `BSplineCurve[{p1, …}, opts…]` stands for: the uniform
/// B-spline of degree `min(3, n - 1)` over its control points, sampled
/// finely enough to read as a curve. `SplineClosed -> True` wraps the
/// leading control points onto the end so the curve closes on itself.
/// Arguments after the control points that are not options are ignored,
/// the way Wolfram ignores them. `None` when the first argument is not a
/// list of 3-D points.
fn bspline_curve_points(args: &[Expr]) -> Option<Vec<Point3D>> {
  let pts_expr =
    evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
  let control = parse_point3d_list(&pts_expr)?;
  if control.len() < 2 {
    return Some(control);
  }
  let closed = args.iter().skip(1).any(|arg| {
    matches!(arg,
      Expr::Rule { pattern, replacement }
        if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "SplineClosed")
        && matches!(replacement.as_ref(), Expr::Identifier(s) if s == "True"))
  });
  let control = if closed {
    let degree = 3usize.min(control.len() - 1);
    let mut cp = control.clone();
    cp.extend_from_slice(&control[..degree]);
    cp
  } else {
    control
  };
  let n = control.len();
  // Enough samples that the curve is smooth without the tube it may feed
  // exploding into triangles: a handful per control point, bounded.
  let samples = (n * 6).clamp(64, 600);
  Some(
    bspline_sample_weights(n, samples)
      .iter()
      .map(|weights| {
        let mut acc = Point3D {
          x: 0.0,
          y: 0.0,
          z: 0.0,
        };
        for (j, &b) in weights.iter().enumerate() {
          if b == 0.0 {
            continue;
          }
          acc.x += b * control[j].x;
          acc.y += b * control[j].y;
          acc.z += b * control[j].z;
        }
        acc
      })
      .collect(),
  )
}

/// Tessellate a B-spline surface from its control-point grid by sampling
/// the tensor-product spline on a fixed grid.
fn tessellate_bspline_surface(
  grid: &[Vec<Point3D>],
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n_rows = grid.len();
  let n_cols = grid[0].len();
  let samples = 24;
  let wu = bspline_sample_weights(n_rows, samples);
  let wv = bspline_sample_weights(n_cols, samples);
  let surface: Vec<Vec<Point3D>> = wu
    .iter()
    .map(|row_w| {
      wv.iter()
        .map(|col_w| {
          let mut acc = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
          };
          for (r, &rw) in row_w.iter().enumerate() {
            if rw == 0.0 {
              continue;
            }
            for (c, &cw) in col_w.iter().enumerate() {
              let w = rw * cw;
              if w == 0.0 {
                continue;
              }
              acc.x += w * grid[r][c].x;
              acc.y += w * grid[r][c].y;
              acc.z += w * grid[r][c].z;
            }
          }
          acc
        })
        .collect()
    })
    .collect();
  let mut tris = Vec::new();
  for i in 0..samples - 1 {
    for j in 0..samples - 1 {
      let a = surface[i][j];
      let b = surface[i + 1][j];
      let c = surface[i + 1][j + 1];
      let d = surface[i][j + 1];
      tris.push((a, b, c));
      tris.push((a, c, d));
    }
  }
  tris
}

/// Raster3D[data] — data is a nested list of layers (z), rows (y), and
/// cells (x); each cell is a grayscale value in [0, 1] or an {r, g, b} /
/// {r, g, b, a} list. Every cell becomes a unit voxel cuboid.
fn collect_raster3d(
  data: &Expr,
  style: &StyleState3D,
  prims: &mut Vec<Primitive3D>,
) {
  let num =
    |e: &Expr| try_eval_to_f64(&evaluate_expr_to_expr(e).unwrap_or(e.clone()));
  let Expr::List(layers) = data else {
    return;
  };
  for (k, layer) in layers.iter().enumerate() {
    let Expr::List(rows) = layer else {
      return;
    };
    for (j, row) in rows.iter().enumerate() {
      let Expr::List(cells) = row else {
        return;
      };
      for (i, cell) in cells.iter().enumerate() {
        let (r, g, b, a) = match cell {
          Expr::List(channels) => {
            let vals: Vec<f64> =
              match channels.iter().map(num).collect::<Option<Vec<_>>>() {
                Some(v) => v,
                None => return,
              };
            match vals.len() {
              3 => (vals[0], vals[1], vals[2], 1.0),
              4 => (vals[0], vals[1], vals[2], vals[3]),
              _ => return,
            }
          }
          other => match num(other) {
            Some(v) => (v, v, v, 1.0),
            None => return,
          },
        };
        let mut voxel_style = style.clone();
        voxel_style.color = Some((
          (r.clamp(0.0, 1.0) * 255.0).round() as u8,
          (g.clamp(0.0, 1.0) * 255.0).round() as u8,
          (b.clamp(0.0, 1.0) * 255.0).round() as u8,
        ));
        voxel_style.opacity = a.clamp(0.0, 1.0) * style.opacity;
        let p_min = Point3D {
          x: i as f64,
          y: j as f64,
          z: k as f64,
        };
        let p_max = Point3D {
          x: i as f64 + 1.0,
          y: j as f64 + 1.0,
          z: k as f64 + 1.0,
        };
        prims.push(Primitive3D::Surface3D {
          tris: tessellate_cuboid(&p_min, &p_max),
          style: voxel_style,
          smooth: false,
        });
      }
    }
  }
}

/// Whether a polygon's corners all turn the same way, so a fan from the
/// first one covers it. Measured about the polygon's own plane (Newell's
/// normal, which is robust for slightly non-planar input); a straight
/// corner turns neither way and says nothing either way.
fn is_convex_polygon3d(points: &[Point3D]) -> bool {
  let n = points.len();
  if n < 4 {
    return true;
  }
  let mut normal = [0.0f64; 3];
  for i in 0..n {
    let a = points[i];
    let b = points[(i + 1) % n];
    normal[0] += (a.y - b.y) * (a.z + b.z);
    normal[1] += (a.z - b.z) * (a.x + b.x);
    normal[2] += (a.x - b.x) * (a.y + b.y);
  }
  let scale =
    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
      .sqrt();
  if !scale.is_finite() || scale == 0.0 {
    return true;
  }
  let mut sign = 0i8;
  for i in 0..n {
    let a = points[i];
    let b = points[(i + 1) % n];
    let c = points[(i + 2) % n];
    let u = [b.x - a.x, b.y - a.y, b.z - a.z];
    let v = [c.x - b.x, c.y - b.y, c.z - b.z];
    let turn = (u[1] * v[2] - u[2] * v[1]) * normal[0]
      + (u[2] * v[0] - u[0] * v[2]) * normal[1]
      + (u[0] * v[1] - u[1] * v[0]) * normal[2];
    // Scaled by the polygon's own size so the tolerance means the same
    // thing whatever the coordinates are.
    let this = if turn > 1e-9 * scale {
      1
    } else if turn < -1e-9 * scale {
      -1
    } else {
      0
    };
    if this != 0 {
      if sign != 0 && this != sign {
        return false;
      }
      sign = this;
    }
  }
  true
}

/// Tessellate a planar polygon with holes (`Polygon[outer -> holes]`).
///
/// The face is flattened onto its own plane, triangulated there, and the
/// triangles are lifted back to 3D. The second result marks, per triangle,
/// which of its three edges lie on the outer boundary or on a hole
/// boundary — the rest are internal cuts that must not be stroked.
fn tessellate_polygon_with_holes(
  outer: &[Point3D],
  holes: &[Vec<Point3D>],
) -> (Vec<(Point3D, Point3D, Point3D)>, Vec<[bool; 3]>) {
  use crate::functions::polygon_holes::triangulate_with_holes;

  if outer.len() < 3 {
    return (Vec::new(), Vec::new());
  }
  // Newell's method: a plane normal that is robust for non-planar input.
  let mut n = Point3D {
    x: 0.0,
    y: 0.0,
    z: 0.0,
  };
  for i in 0..outer.len() {
    let a = outer[i];
    let b = outer[(i + 1) % outer.len()];
    n.x += (a.y - b.y) * (a.z + b.z);
    n.y += (a.z - b.z) * (a.x + b.x);
    n.z += (a.x - b.x) * (a.y + b.y);
  }
  let len = (n.x * n.x + n.y * n.y + n.z * n.z).sqrt();
  if !len.is_finite() || len == 0.0 {
    return (Vec::new(), Vec::new());
  }
  let n = Point3D {
    x: n.x / len,
    y: n.y / len,
    z: n.z / len,
  };
  // Any vector not parallel to the normal yields an in-plane basis.
  let helper = if n.x.abs() < 0.9 {
    Point3D {
      x: 1.0,
      y: 0.0,
      z: 0.0,
    }
  } else {
    Point3D {
      x: 0.0,
      y: 1.0,
      z: 0.0,
    }
  };
  let cross = |a: Point3D, b: Point3D| Point3D {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
  let u = cross(n, helper);
  let ulen = (u.x * u.x + u.y * u.y + u.z * u.z).sqrt();
  let u = Point3D {
    x: u.x / ulen,
    y: u.y / ulen,
    z: u.z / ulen,
  };
  let v = cross(n, u);
  let origin = outer[0];
  let flatten = |p: &Point3D| {
    let d = Point3D {
      x: p.x - origin.x,
      y: p.y - origin.y,
      z: p.z - origin.z,
    };
    (
      d.x * u.x + d.y * u.y + d.z * u.z,
      d.x * v.x + d.y * v.y + d.z * v.z,
    )
  };

  let mut verts3: Vec<Point3D> = outer.to_vec();
  let outer_idx: Vec<usize> = (0..outer.len()).collect();
  let mut hole_idx: Vec<Vec<usize>> = Vec::with_capacity(holes.len());
  for hole in holes {
    let start = verts3.len();
    verts3.extend(hole.iter().copied());
    hole_idx.push((start..verts3.len()).collect());
  }
  let verts2: Vec<(f64, f64)> = verts3.iter().map(flatten).collect();

  let result = triangulate_with_holes(&verts2, &outer_idx, &hole_idx);
  let tris = result
    .triangles
    .iter()
    .map(|t| (verts3[t[0]], verts3[t[1]], verts3[t[2]]))
    .collect();
  let flags = result
    .triangles
    .iter()
    .map(|t| {
      [
        result.is_boundary(t[0], t[1]),
        result.is_boundary(t[1], t[2]),
        result.is_boundary(t[2], t[0]),
      ]
    })
    .collect();
  (tris, flags)
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

/// Which edges of a box triangle are edges of the box itself. Two corners of
/// an axis-aligned box are joined by an edge exactly when they differ in one
/// coordinate; the rest are diagonals across a face, internal cuts of the
/// two triangles it is split into. Only the box's own edges are outlined,
/// the way Wolfram draws a `Cuboid`.
fn box_edge_flags(tri: &(Point3D, Point3D, Point3D)) -> [bool; 3] {
  let corners = [tri.0, tri.1, tri.2];
  let is_edge = |a: Point3D, b: Point3D| {
    let differs = [(a.x, b.x), (a.y, b.y), (a.z, b.z)]
      .into_iter()
      .filter(|(p, q)| (p - q).abs() > 1e-12)
      .count();
    differs == 1
  };
  [
    is_edge(corners[0], corners[1]),
    is_edge(corners[1], corners[2]),
    is_edge(corners[2], corners[0]),
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

/// Tessellate a tube of the given per-point radii around a polyline.
///
/// Each vertex gets a ring of `TUBE_SIDES` points lying in the plane normal
/// to the curve there (the bisector of the two adjacent segments at an
/// interior vertex), so consecutive segments share a ring and the surface
/// stays closed through a bend.  The rings are kept aligned by transporting
/// one reference normal along the curve rather than recomputing it per
/// segment, which would twist the tube.  With `capped`, flat disks close
/// both ends — Wolfram's default; `CapForm[None]` drops them.
fn tessellate_tube(
  points: &[Point3D],
  radii: &[f64],
  capped: bool,
) -> Vec<(Point3D, Point3D, Point3D)> {
  const TUBE_SIDES: usize = 24;
  if points.len() < 2 {
    return vec![];
  }
  let sub = |a: &Point3D, b: &Point3D| (a.x - b.x, a.y - b.y, a.z - b.z);
  let norm = |v: (f64, f64, f64)| (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
  let unit = |v: (f64, f64, f64)| {
    let l = norm(v);
    if l < 1e-15 {
      None
    } else {
      Some((v.0 / l, v.1 / l, v.2 / l))
    }
  };
  let cross = |a: (f64, f64, f64), b: (f64, f64, f64)| {
    (
      a.1 * b.2 - a.2 * b.1,
      a.2 * b.0 - a.0 * b.2,
      a.0 * b.1 - a.1 * b.0,
    )
  };

  // Segment directions, skipping repeated points.
  let mut dirs: Vec<(f64, f64, f64)> = Vec::new();
  let mut keep: Vec<usize> = vec![0];
  for i in 1..points.len() {
    if let Some(d) = unit(sub(&points[i], &points[keep[keep.len() - 1]])) {
      dirs.push(d);
      keep.push(i);
    }
  }
  if dirs.is_empty() {
    return vec![];
  }

  // Vertex tangents: the segment direction at the ends, the normalized sum
  // of the two adjacent directions in between.
  let tangents: Vec<(f64, f64, f64)> = (0..keep.len())
    .map(|i| {
      if i == 0 {
        dirs[0]
      } else if i == keep.len() - 1 {
        dirs[dirs.len() - 1]
      } else {
        let (a, b) = (dirs[i - 1], dirs[i]);
        unit((a.0 + b.0, a.1 + b.1, a.2 + b.2)).unwrap_or(b)
      }
    })
    .collect();

  // Seed a normal perpendicular to the first tangent, then transport it.
  let t0 = tangents[0];
  let seed = if t0.2.abs() < 0.9 {
    (0.0, 0.0, 1.0)
  } else {
    (1.0, 0.0, 0.0)
  };
  let mut normal = unit(cross(t0, seed)).unwrap_or((1.0, 0.0, 0.0));

  let mut rings: Vec<Vec<Point3D>> = Vec::with_capacity(keep.len());
  for (i, &idx) in keep.iter().enumerate() {
    let t = tangents[i];
    // Re-project the carried normal into this vertex's normal plane.
    let dot = normal.0 * t.0 + normal.1 * t.1 + normal.2 * t.2;
    let projected = (
      normal.0 - dot * t.0,
      normal.1 - dot * t.1,
      normal.2 - dot * t.2,
    );
    normal = unit(projected).unwrap_or(normal);
    let binormal = cross(t, normal);
    let r = radii.get(idx).copied().unwrap_or(0.0);
    // A bend shortens the ring's projection onto the segments; widening it
    // by 1/cos(half-angle) keeps the tube's radius constant along the curve.
    let widen = if i == 0 || i == keep.len() - 1 {
      1.0
    } else {
      let d = dirs[i - 1].0 * t.0 + dirs[i - 1].1 * t.1 + dirs[i - 1].2 * t.2;
      if d.abs() < 1e-6 { 1.0 } else { 1.0 / d }
    };
    rings.push(
      (0..TUBE_SIDES)
        .map(|k| {
          let a = 2.0 * std::f64::consts::PI * k as f64 / TUBE_SIDES as f64;
          let (c, s) = (a.cos() * r * widen, a.sin() * r * widen);
          Point3D {
            x: points[idx].x + c * normal.0 + s * binormal.0,
            y: points[idx].y + c * normal.1 + s * binormal.1,
            z: points[idx].z + c * normal.2 + s * binormal.2,
          }
        })
        .collect(),
    );
  }

  let mut tris = Vec::new();
  for i in 0..rings.len() - 1 {
    for k in 0..TUBE_SIDES {
      let k2 = (k + 1) % TUBE_SIDES;
      let (a, b) = (rings[i][k], rings[i][k2]);
      let (c, d) = (rings[i + 1][k2], rings[i + 1][k]);
      tris.push((a, b, c));
      tris.push((a, c, d));
    }
  }
  if capped {
    for (ring, centre) in [
      (&rings[0], points[keep[0]]),
      (&rings[rings.len() - 1], points[keep[keep.len() - 1]]),
    ] {
      for k in 0..TUBE_SIDES {
        tris.push((centre, ring[k], ring[(k + 1) % TUBE_SIDES]));
      }
    }
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

/// Which triangle edges of a tessellated cylinder lie on one of its two end
/// circles. `tessellate_cylinder` emits a quad per segment as the triangles
/// `(a, b, c)` and `(a, c, d)`, with `a`/`d` on the first circle and `b`/`c`
/// on the second — so the rim edges are `b→c` and `d→a`.
fn cylinder_edge_flags(tri_count: usize) -> Vec<[bool; 3]> {
  (0..tri_count)
    .map(|i| {
      if i % 2 == 0 {
        [false, true, false]
      } else {
        [false, false, true]
      }
    })
    .collect()
}

/// Which triangle edges of a tessellated cone lie on its base circle.
/// `tessellate_cone` emits one triangle `(tip, b1, b2)` per segment, so the
/// rim edge is always `b1→b2`.
fn cone_edge_flags(tri_count: usize) -> Vec<[bool; 3]> {
  vec![[false, true, false]; tri_count]
}

/// The world-coordinate bounding box of a set of 3D primitives, as
/// `[(xlo, xhi), (ylo, yhi), (zlo, zhi)]`. Infinite when there is nothing
/// to bound.
fn primitives_bounds(prims: &[Primitive3D]) -> [(f64, f64); 3] {
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

  for prim in prims {
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
      Primitive3D::Text3D { pos, .. } => {
        extend_3d(
          pos,
          &mut x3_min,
          &mut x3_max,
          &mut y3_min,
          &mut y3_max,
          &mut z3_min,
          &mut z3_max,
        );
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
      Primitive3D::Surface3D { tris, .. } => {
        for (a, b, c) in tris {
          for pt in [a, b, c] {
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
  [(x3_min, x3_max), (y3_min, y3_max), (z3_min, z3_max)]
}

/// Radius of the sphere centred on `center` that encloses every primitive.
///
/// Wolfram's `SphericalRegion -> True` scales the picture so this sphere
/// fits the display area. It is the sphere around the *contents*, not
/// around their bounding box: a lone `Sphere[]` keeps its size (radius 1)
/// instead of shrinking to the box's half-diagonal.
fn enclosing_sphere_radius(prims: &[Primitive3D], center: Point3D) -> f64 {
  let dist = |p: &Point3D| {
    ((p.x - center.x).powi(2)
      + (p.y - center.y).powi(2)
      + (p.z - center.z).powi(2))
    .sqrt()
  };
  let mut radius: f64 = 0.0;
  for prim in prims {
    match prim {
      Primitive3D::Sphere {
        center: c,
        radius: r,
        ..
      } => radius = radius.max(dist(c) + r),
      Primitive3D::Cylinder {
        p1, p2, radius: r, ..
      }
      | Primitive3D::Cone {
        p1, p2, radius: r, ..
      } => {
        for p in [p1, p2] {
          radius = radius.max(dist(p) + r);
        }
      }
      Primitive3D::Cuboid { p_min, p_max, .. } => {
        for x in [p_min.x, p_max.x] {
          for y in [p_min.y, p_max.y] {
            for z in [p_min.z, p_max.z] {
              radius = radius.max(dist(&Point3D { x, y, z }));
            }
          }
        }
      }
      Primitive3D::Text3D { pos, .. } => {
        radius = radius.max(dist(pos));
      }
      Primitive3D::Polygon3D { points, .. }
      | Primitive3D::Point3DPrim { points, .. }
      | Primitive3D::Arrow3D { points, .. } => {
        for pt in points {
          radius = radius.max(dist(pt));
        }
      }
      Primitive3D::Line3D { segments, .. } => {
        for seg in segments {
          for pt in seg {
            radius = radius.max(dist(pt));
          }
        }
      }
      Primitive3D::Surface3D { tris, .. } => {
        for (a, b, c) in tris {
          for pt in [a, b, c] {
            radius = radius.max(dist(pt));
          }
        }
      }
    }
  }
  radius
}

/// Graphics3D[primitives, options...]
pub fn graphics3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let args = &crate::functions::graphics::splice_option_lists(args)[..];
  let content = evaluate_expr_to_expr(&args[0])?;

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut show_box = true;
  let mut background: Option<(u8, u8, u8)> = None;
  let mut camera = Camera::default();
  // `PlotRange -> r` (a single number): the displayed region is the fixed
  // cube [-r, r]³, so the framing stays put while contents move (as in a
  // Manipulate re-render). `PlotRange -> {{x0, x1}, {y0, y1}, {z0, z1}}`
  // pins each axis separately.
  let mut plot_range: Option<[(f64, f64); 3]> = None;
  // `BoxRatios -> {rx, ry, rz}`; `None` is Wolfram's `Automatic`, where the
  // box simply has the proportions of the data.
  let mut box_ratios: Option<[f64; 3]> = None;
  // `SphericalRegion -> True`: frame the picture by the sphere that
  // encloses the contents instead of by their projected outline, so the
  // scale stays put as the view turns or the contents move.
  let mut spherical_region = false;
  // `ViewAngle -> θ`: the field of view of the camera, in radians. With one
  // given the picture is no longer scaled to fit its contents — the view
  // volume sets the scale, so a small object stays small. `view_distance`
  // is `|ViewPoint|`, which Wolfram measures in units of the longest side
  // of the displayed box.
  let mut view_angle: Option<f64> = None;
  let mut view_distance = {
    let d = Camera::DEFAULT_VIEW_POINT;
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
  };
  let mut show_axes = false;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
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
        "SphericalRegion" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "False" => spherical_region = false,
          Expr::Identifier(s) if s == "True" => spherical_region = true,
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
        "ViewPoint" => {
          // Symbolic viewpoints stand for axis-aligned directions
          // (Wolfram's `Above` is `{0, 0, ∞}`, etc.).
          let symbolic: Option<[f64; 3]> = match replacement.as_ref() {
            Expr::Identifier(s) => match s.as_str() {
              "Above" => Some([0.0, 0.0, 2.0]),
              "Below" => Some([0.0, 0.0, -2.0]),
              "Front" => Some([0.0, -2.0, 0.0]),
              "Back" => Some([0.0, 2.0, 0.0]),
              "Left" => Some([-2.0, 0.0, 0.0]),
              "Right" => Some([2.0, 0.0, 0.0]),
              _ => None,
            },
            _ => None,
          };
          if let Some(vp) = symbolic.or_else(|| eval_vec3(replacement)) {
            camera = Camera {
              azimuth: vp[1].atan2(vp[0]),
              elevation: vp[2].atan2(vp[0].hypot(vp[1])),
            };
            view_distance =
              (vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2]).sqrt();
          }
        }
        // `ViewAngle -> θ` (radians, or a `Quantity`/`Degree` product).
        "ViewAngle" => {
          if let Some(a) = try_eval_to_f64(
            &evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.as_ref().clone()),
          ) && a > 0.0
            && a < std::f64::consts::PI
          {
            view_angle = Some(a);
          }
        }
        "PlotRange" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          if let Some(r) = try_eval_to_f64(&value) {
            if r > 0.0 {
              plot_range = Some([(-r, r); 3]);
            }
          } else if let Some(ranges) = parse_axis_ranges(&value) {
            plot_range = Some(ranges);
          }
        }
        // `BoxRatios -> {rx, ry, rz}` fixes the *shape* of the bounding
        // box regardless of how large the data is along each axis, which
        // is what keeps a plot whose values run far wider than its domain
        // (or the other way round) from drawing as a sliver.
        "BoxRatios" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          if let Some(r) = eval_vec3(&value)
            && r.iter().all(|v| *v > 0.0)
          {
            box_ratios = Some(r);
          }
        }
        "Axes" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "False" => show_axes = false,
          Expr::Identifier(s) if s == "True" => show_axes = true,
          // `Axes -> {True, True, False}`: drawn as long as any is on.
          Expr::List(items) => {
            show_axes = items
              .iter()
              .any(|i| matches!(i, Expr::Identifier(s) if s == "True"));
          }
          _ => {}
        },
        "AxesLabel" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          let items: Vec<Expr> = match &value {
            Expr::List(items) => items.to_vec(),
            other => vec![other.clone()],
          };
          for (i, item) in items.iter().take(3).enumerate() {
            axes_labels[i] = axis_label_markup(item);
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

  // Unbounded primitives (`InfiniteLine`, `InfinitePlane`, …) show only the
  // part of themselves that falls inside the picture's box, so they are
  // expanded in a second pass: the finite contents — or an explicit
  // `PlotRange` — fix the box, and the whole scene is collected again with
  // each of them replaced by the `Line`/`Polygon` it draws there. Clipping
  // to exactly those bounds is what keeps an unbounded object from
  // widening the range it was measured against.
  if has_unbounded_3d(&content) {
    let bounds =
      clipping_bounds(plot_range.unwrap_or_else(|| primitives_bounds(&prims)));
    let expanded = expand_unbounded_3d(&content, &bounds);
    prims.clear();
    style3d = StyleState3D::default();
    collect_3d_primitives(&expanded, &mut style3d, &mut prims);
  }

  // The symbolic form carried on the rendered result so that Part can
  // index it (`Graphics3D[…][[1]]` → the content).
  let structure = Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: std::iter::once(content.clone())
      .chain(args[1..].iter().cloned())
      .collect::<Vec<_>>()
      .into(),
  };

  if prims.is_empty() {
    // Even with no primitives, return the marker — with its title, since
    // a `PlotLabel` is drawn whether or not the picture has contents.
    let empty_svg = format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"
    );
    let empty_svg =
      with_plot_label(empty_svg, &args[1..], svg_width, svg_height);
    return Ok(crate::graphics3d_result_with_structure(
      empty_svg, structure,
    ));
  }

  // `BoxRatios` asks for a box of a given shape, so each axis is scaled to
  // bring the world coordinates into those proportions before anything is
  // tessellated. The axes keep reporting the values before scaling —
  // `axis_scale` divides them back out where they are drawn.
  let mut axis_scale = [1.0_f64; 3];
  if let Some(ratios) = box_ratios {
    let bounds = plot_range.unwrap_or_else(|| primitives_bounds(&prims));
    let extents = [
      bounds[0].1 - bounds[0].0,
      bounds[1].1 - bounds[1].0,
      bounds[2].1 - bounds[2].0,
    ];
    // Every axis is brought down to the tightest of the requested
    // proportions, so honouring them never inflates the picture.
    let unit = (0..3)
      .filter(|&i| extents[i].is_finite() && extents[i] > 0.0)
      .map(|i| extents[i] / ratios[i])
      .fold(f64::INFINITY, f64::min);
    if unit.is_finite() && unit > 0.0 {
      for i in 0..3 {
        if extents[i].is_finite() && extents[i] > 0.0 {
          axis_scale[i] = ratios[i] * unit / extents[i];
        }
      }
      let xf = Affine3 {
        m: [
          [axis_scale[0], 0.0, 0.0],
          [0.0, axis_scale[1], 0.0],
          [0.0, 0.0, axis_scale[2]],
        ],
        t: [0.0; 3],
      };
      for prim in &mut prims {
        transform_primitive3d(prim, &xf);
      }
      if let Some(range) = plot_range.as_mut() {
        for i in 0..3 {
          range[i] = (range[i].0 * axis_scale[i], range[i].1 * axis_scale[i]);
        }
      }
    }
  }

  // Tessellate all primitives into triangles
  let mut all_triangles: Vec<Triangle> = Vec::new();
  let base_color = (0x5E_u8, 0x81_u8, 0xB5_u8); // Default blue

  // Sphere-scene statistics for adaptive tessellation (see
  // `sphere_detail`): how many spheres there are and how large the
  // sphere-covered region is.
  let mut sphere_count = 0usize;
  let mut sph_min = [f64::INFINITY; 3];
  let mut sph_max = [f64::NEG_INFINITY; 3];
  for prim in &prims {
    if let Primitive3D::Sphere { center, radius, .. } = prim {
      sphere_count += 1;
      for (i, c) in [center.x, center.y, center.z].into_iter().enumerate() {
        sph_min[i] = sph_min[i].min(c - radius);
        sph_max[i] = sph_max[i].max(c + radius);
      }
    }
  }
  let sphere_extent = if sphere_count > 0 {
    (0..3).fold(0.0f64, |m, i| m.max(sph_max[i] - sph_min[i]))
  } else {
    0.0
  };

  for prim in &prims {
    // Per-triangle edge flags for a polygon with holes; the ordinary
    // hole-free cases derive theirs from the fan below.
    let mut holed_boundaries: Vec<[bool; 3]> = Vec::new();
    let (tris, prim_style): (Vec<(Point3D, Point3D, Point3D)>, &StyleState3D) =
      match prim {
        Primitive3D::Sphere {
          center,
          radius,
          style,
        } => (
          tessellate_sphere(
            center,
            *radius,
            sphere_detail(*radius, sphere_extent, sphere_count),
          ),
          style,
        ),
        Primitive3D::Cuboid {
          p_min,
          p_max,
          style,
        } => {
          let tris = tessellate_cuboid(p_min, p_max);
          holed_boundaries = tris.iter().map(box_edge_flags).collect();
          (tris, style)
        }
        // A cylinder/cone is tessellated as a ring of quads split into two
        // triangles each; the quad's ends lie on the two end circles. With
        // an explicit `EdgeForm[colour]` those end circles are drawn — a
        // flat cylinder is how a Demonstration draws a circle in space —
        // while the longitudinal cuts stay internal, as on any smooth
        // surface.
        Primitive3D::Cylinder {
          p1,
          p2,
          radius,
          style,
        } => {
          let tris = tessellate_cylinder(p1, p2, *radius);
          if style.edge_color.is_some() {
            holed_boundaries = cylinder_edge_flags(tris.len());
          }
          (tris, style)
        }
        Primitive3D::Cone {
          p1,
          p2,
          radius,
          style,
        } => {
          let tris = tessellate_cone(p1, p2, *radius);
          if style.edge_color.is_some() {
            holed_boundaries = cone_edge_flags(tris.len());
          }
          (tris, style)
        }
        Primitive3D::Polygon3D {
          points,
          holes,
          style,
        } if holes.is_empty() => {
          // A fan from the first corner only covers a convex polygon; on a
          // concave one its triangles spill outside the outline, which
          // shows as spikes off the shape. A concave face goes through the
          // same triangulator the holed case uses.
          let t = if points.len() < 3 {
            vec![]
          } else if is_convex_polygon3d(points) {
            (1..points.len() - 1)
              .map(|i| (points[0], points[i], points[i + 1]))
              .collect()
          } else {
            let (t, flags) = tessellate_polygon_with_holes(points, &[]);
            holed_boundaries = flags;
            t
          };
          (t, style)
        }
        Primitive3D::Polygon3D {
          points,
          holes,
          style,
        } => {
          let (t, flags) = tessellate_polygon_with_holes(points, holes);
          holed_boundaries = flags;
          (t, style)
        }
        Primitive3D::Surface3D {
          tris,
          style,
          smooth,
        } => {
          // A Raster3D voxel is a box, so only its own edges are outlined.
          if !*smooth {
            holed_boundaries = tris.iter().map(box_edge_flags).collect();
          }
          (tris.clone(), style)
        }
        // Line and Point are handled separately below
        _ => (
          vec![],
          &StyleState3D {
            color: None,
            back_color: None,
            opacity: 1.0,
            thickness: None,
            capped: true,
            edges: true,
            edge_color: None,
            specular: None,
          },
        ),
      };
    let prim_color = prim_style.color.unwrap_or(base_color);
    let prim_back_color = prim_style.back_color;
    let prim_opacity = prim_style.opacity;
    // Direction from the scene towards the viewer; a face whose normal
    // points along it shows its front side.
    let view_dir = {
      let (sa, ca) = camera.azimuth.sin_cos();
      let (se, ce) = camera.elevation.sin_cos();
      [ce * ca, ce * sa, se]
    };

    // A polygon with more than three corners was fan-triangulated above:
    // every triangle keeps the polygon edge it sits on, but the cuts back
    // to the fan's first corner are internal and must not be stroked.
    let fan_corners = match prim {
      Primitive3D::Polygon3D { points, holes, .. } if holes.is_empty() => {
        points.len()
      }
      _ => 0,
    };
    // Triangles that approximate a curved surface have no outline edges at
    // all: Wolfram draws no facet edges on a sphere, cylinder, cone, tube
    // or torus. Every edge is an internal cut, so the hairline that closes
    // the anti-aliasing seam between neighbours is drawn in the surface's
    // own colour rather than the usual dark one — otherwise a small sphere
    // reads as speckled instead of smooth.
    let smooth_surface = matches!(
      prim,
      Primitive3D::Sphere { .. }
        | Primitive3D::Cylinder { .. }
        | Primitive3D::Cone { .. }
        | Primitive3D::Surface3D { smooth: true, .. }
    );
    let tri_count = tris.len();
    for (i, (v0, v1, v2)) in tris.into_iter().enumerate() {
      // `EdgeForm[]` asks for faces with no outline, so every edge counts
      // as an internal cut and is stroked in the face's own colour. A
      // curved primitive that was given an explicit edge colour keeps the
      // silhouette flags computed above — its facet cuts stay internal,
      // but its end circles are drawn.
      let outlined_smooth = smooth_surface
        && prim_style.edge_color.is_some()
        && !holed_boundaries.is_empty();
      let boundary =
        if !prim_style.edges || (smooth_surface && !outlined_smooth) {
          [false; 3]
        } else if let Some(flags) = holed_boundaries.get(i) {
          *flags
        } else if fan_corners > 3 {
          [i == 0, true, i + 1 == tri_count]
        } else {
          [true; 3]
        };
      let normal = triangle_normal(v0, v1, v2);
      let facing = normal[0] * view_dir[0]
        + normal[1] * view_dir[1]
        + normal[2] * view_dir[2];
      let side_color = match prim_back_color {
        Some(back) if facing < 0.0 => back,
        _ => prim_color,
      };
      let color = apply_lighting_specular(
        side_color,
        normal,
        prim_style.specular,
        view_dir,
      );
      let p0 = project(v0, &camera);
      let p1 = project(v1, &camera);
      let p2 = project(v2, &camera);
      let center = Point3D {
        x: (v0.x + v1.x + v2.x) / 3.0,
        y: (v0.y + v1.y + v2.y) / 3.0,
        z: (v0.z + v1.z + v2.z) / 3.0,
      };
      all_triangles.push(Triangle {
        boundary,
        edge_color: prim_style.edge_color,
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
  let [
    (mut x3_min, mut x3_max),
    (mut y3_min, mut y3_max),
    (mut z3_min, mut z3_max),
  ] = primitives_bounds(&prims);

  // The length Wolfram measures `ViewPoint` (and so `ViewAngle`) in: the
  // longest side of the *displayed* box, before the framing padding below,
  // which Wolfram does not apply to an automatic 3D range.
  let mut view_box_side =
    (x3_max - x3_min).max(y3_max - y3_min).max(z3_max - z3_min);

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

  // An explicit PlotRange pins the displayed region regardless of the
  // content's extent, keeping the framing stable across re-renders.
  if let Some([(xl, xh), (yl, yh), (zl, zh)]) = plot_range {
    x3_min = xl;
    x3_max = xh;
    y3_min = yl;
    y3_max = yh;
    z3_min = zl;
    z3_max = zh;
    view_box_side = (xh - xl).max(yh - yl).max(zh - zl);
  }

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
      Primitive3D::Text3D { pos, .. } => {
        let (px, py) = project(*pos, &camera);
        px_min = px_min.min(px);
        px_max = px_max.max(px);
        py_min = py_min.min(py);
        py_max = py_max.max(py);
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

  // Include box corners in the projected bounding box — always when the
  // box is drawn, and also for a fixed PlotRange so the zoom level does
  // not follow the contents.
  if show_box || plot_range.is_some() {
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

  // `SphericalRegion -> True` fits the enclosing sphere of the contents
  // (of the box, when `PlotRange` pins one) instead of their projected
  // outline. The sphere projects to the same circle from every direction,
  // so the scale no longer follows the view angle or a moving shape.
  if spherical_region {
    let center = Point3D {
      x: (x3_min + x3_max) / 2.0,
      y: (y3_min + y3_max) / 2.0,
      z: (z3_min + z3_max) / 2.0,
    };
    let radius = if plot_range.is_some() {
      ((x3_max - x3_min).powi(2)
        + (y3_max - y3_min).powi(2)
        + (z3_max - z3_min).powi(2))
      .sqrt()
        / 2.0
    } else {
      enclosing_sphere_radius(&prims, center)
    };
    if radius > 0.0 {
      let (pcx, pcy) = project(center, &camera);
      px_min = pcx - radius;
      px_max = pcx + radius;
      py_min = pcy - radius;
      py_max = pcy + radius;
    }
  }

  let p_width = px_max - px_min;
  let p_height = py_max - py_min;
  let p_width = if p_width < 1e-15 { 1.0 } else { p_width };
  let p_height = if p_height < 1e-15 { 1.0 } else { p_height };

  // Axes need room outside the box for their ticks, and more again for
  // the axis labels sitting beyond those.
  let margin = match (show_axes, axes_labels.iter().any(Option::is_some)) {
    (true, true) => axes_label_margin(&axes_labels, svg_width.min(svg_height)),
    (true, false) => 25.0,
    (false, _) => 10.0,
  };
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let mut scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let mut p_cx = (px_min + px_max) / 2.0;
  let mut p_cy = (py_min + py_max) / 2.0;

  // An explicit `ViewAngle` fixes the camera's field of view, so the scale
  // comes from the view volume rather than from fitting the contents: the
  // frame spans `2 · d · tan(θ/2)` in the units Wolfram measures `ViewPoint`
  // in, which are those of the longest side of the displayed box. A small
  // object then stays small instead of being blown up to fill the picture.
  // Measured against wolframscript over a sweep of angles, distances and
  // plot ranges: a world length `l` covers `(l / L) / (2 d tan(θ/2))` of
  // the frame.
  if let Some(theta) = view_angle {
    let span = 2.0 * view_distance * (theta / 2.0).tan() * view_box_side;
    if span > 0.0 && span.is_finite() {
      scale = svg_width.min(svg_height) as f64 / span;
      // The view centres on the middle of the displayed box, not on
      // whatever part of it the contents happen to occupy.
      let (bcx, bcy) = project(
        Point3D {
          x: (x3_min + x3_max) / 2.0,
          y: (y3_min + y3_max) / 2.0,
          z: (z3_min + z3_max) / 2.0,
        },
        &camera,
      );
      p_cx = bcx;
      p_cy = bcy;
    }
  }

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    (cx + (px - p_cx) * scale, cy - (py - p_cy) * scale)
  };

  let (_, axis_rgb, _, _, _) = crate::functions::plot::plot_theme();
  let axis_color = format!("rgb({},{},{})", axis_rgb.0, axis_rgb.1, axis_rgb.2);
  let default_prim_color = crate::functions::graphics::theme().text_primary;

  // A depth-sorted line segment interleaved with the surface triangles
  // (painter's algorithm) so lines behind a surface are hidden by it.
  struct SceneEdge {
    endpoints: [Point3D; 2],
    depth: f64,
    color: String,
    width: f64,
    opacity: f64,
  }

  // Build depth-sorted segments: the bounding-box edges plus all Line
  // primitives. Each is subdivided so per-segment depth sorting produces
  // correct occlusion against the surface.
  const EDGE_SUBDIVISIONS: usize = 20;
  let mut sorted_edges: Vec<SceneEdge> = Vec::new();
  if show_box {
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
        sorted_edges.push(SceneEdge {
          endpoints: [lerp(t0), lerp(t1)],
          depth: depth(lerp(tm), &camera),
          color: axis_color.clone(),
          width: 0.5,
          opacity: 0.4,
        });
      }
    }
  }
  {
    // Lines sit exactly on the surfaces they outline, so nudge them
    // toward the viewer by a fraction of the scene size: a face's own
    // outline stays visible while lines behind other faces are hidden.
    let diag = ((x3_max - x3_min).powi(2)
      + (y3_max - y3_min).powi(2)
      + (z3_max - z3_min).powi(2))
    .sqrt();
    let bias = diag * 1e-3;
    const LINE_SUBDIVISIONS: usize = 8;
    for prim in &prims {
      let Primitive3D::Line3D { segments, style } = prim else {
        continue;
      };
      let color = match style.color {
        Some((r, g, b)) => format!("rgb({r},{g},{b})"),
        None => default_prim_color.to_string(),
      };
      let width = style.thickness.unwrap_or(1.5);
      for seg in segments {
        for pair in seg.windows(2) {
          let (a, b) = (pair[0], pair[1]);
          for s in 0..LINE_SUBDIVISIONS {
            let t0 = s as f64 / LINE_SUBDIVISIONS as f64;
            let t1 = (s + 1) as f64 / LINE_SUBDIVISIONS as f64;
            let tm = (t0 + t1) * 0.5;
            let lerp = |t: f64| Point3D {
              x: a.x + (b.x - a.x) * t,
              y: a.y + (b.y - a.y) * t,
              z: a.z + (b.z - a.z) * t,
            };
            sorted_edges.push(SceneEdge {
              endpoints: [lerp(t0), lerp(t1)],
              depth: depth(lerp(tm), &camera) - bias,
              color: color.clone(),
              width,
              opacity: style.opacity,
            });
          }
        }
      }
    }
  }
  sorted_edges.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

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
    let emit_edge = |svg: &mut String, edge: &SceneEdge| {
      let (ex0, ey0) = to_svg(
        project(edge.endpoints[0], &camera).0,
        project(edge.endpoints[0], &camera).1,
      );
      let (ex1, ey1) = to_svg(
        project(edge.endpoints[1], &camera).0,
        project(edge.endpoints[1], &camera).1,
      );
      let opacity_attr = if edge.opacity < 1.0 {
        format!(" opacity=\"{}\"", edge.opacity)
      } else {
        String::new()
      };
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"{:.1}\" stroke-linecap=\"round\"{}/>\n",
        ex0, ey0, ex1, ey1, edge.color, edge.width, opacity_attr
      ));
    };
    let mut ei = 0;
    for tri in &all_triangles {
      // Emit any edge segments further from the camera than this triangle
      while ei < sorted_edges.len() && sorted_edges[ei].depth >= tri.depth {
        emit_edge(&mut svg, &sorted_edges[ei]);
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
      // The outline colour: the default dark grey unless `EdgeForm[colour]`
      // named one. A named colour is also drawn opaque, so the outline of a
      // transparent face still shows — that is what makes
      // `{Opacity[0], EdgeForm[Black], Cylinder[…]}` an unfilled circle.
      let (edge_stroke, edge_opacity_attr) = match tri.edge_color {
        Some((er, eg, eb)) => (format!("rgb({er},{eg},{eb})"), String::new()),
        None => ("rgb(64,64,64)".to_string(), opacity_attr.clone()),
      };
      // The hairline stroke closes the anti-aliasing seam between
      // neighbouring triangles. Where the seam is an internal cut of a
      // fan-triangulated polygon it is stroked in the triangle's own
      // colour instead, so the face reads as one flat surface.
      if tri.boundary == [true; 3] {
        svg.push_str(&format!(
          "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({},{},{})\" stroke=\"{edge_stroke}\" stroke-width=\"1\"{}/>\n",
          x0, y0, x1, y1, x2, y2, r, g, b, opacity_attr
        ));
      } else {
        svg.push_str(&format!(
          "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"0.5\"{}/>\n",
          x0, y0, x1, y1, x2, y2, opacity_attr
        ));
        let corners = [(x0, y0), (x1, y1), (x2, y2)];
        for (e, on_outline) in tri.boundary.iter().enumerate() {
          if !on_outline {
            continue;
          }
          let (ax, ay) = corners[e];
          let (bx, by) = corners[(e + 1) % 3];
          svg.push_str(&format!(
            "<line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"{edge_stroke}\" stroke-width=\"1\"{edge_opacity_attr}/>\n",
          ));
        }
      }
    }
    // Emit remaining edge segments (closest to viewer)
    while ei < sorted_edges.len() {
      emit_edge(&mut svg, &sorted_edges[ei]);
      ei += 1;
    }
  }

  // Render points and arrows on top (lines are depth-sorted with the
  // triangles above so surfaces occlude them correctly)
  for prim in &prims {
    match prim {
      Primitive3D::Text3D {
        label,
        pos,
        offset,
        font_size,
        width_chars,
        style,
      } => {
        let fill_color = if let Some((r, g, b)) = style.color {
          format!("rgb({r},{g},{b})")
        } else {
          default_prim_color.to_string()
        };
        let (px, py) = project(*pos, &camera);
        let (sx, sy) = to_svg(px, py);
        // The offset names which point of the label's box sits at the
        // projected point, so the box moves the other way — half its width
        // per unit across, half its height per unit up (and the vertical
        // sign flips, SVG counting y downwards).
        let text_w = *width_chars as f64 * font_size * 0.6;
        let x = sx - offset.0 * text_w / 2.0;
        let y = sy + offset.1 * font_size / 2.0;
        svg.push_str(&format!(
          "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" \
           dominant-baseline=\"middle\" font-family=\"sans-serif\" \
           font-size=\"{font_size:.1}\" fill=\"{fill_color}\">{label}</text>\n",
        ));
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

  // Axes (with their ticks and labels) go on top of the scene.
  if show_axes {
    // The ticks report coordinates as they were given, so a box reshaped
    // by `BoxRatios` still reads off the data's own values.
    draw_axes_on_box(
      &mut svg,
      &camera,
      &to_svg,
      &box_corners,
      (x3_min / axis_scale[0], x3_max / axis_scale[0]),
      (y3_min / axis_scale[1], y3_max / axis_scale[1]),
      (z3_min / axis_scale[2], z3_max / axis_scale[2]),
      &axes_labels,
    );
  }

  svg.push_str("</svg>");
  // A `PlotLabel` sets a title above the finished picture. `args[1..]` are
  // the options; `args[0]` is the content, which may itself be a `Rule`
  // (`Graphics3D[a -> b]` draws nothing but is legal) and must not be read
  // as one.
  let svg = with_plot_label(svg, &args[1..], svg_width, svg_height);
  Ok(crate::graphics3d_result_with_structure(svg, structure))
}

/// `PlotRange -> {{x0, x1}, {y0, y1}, {z0, z1}}`: one explicit interval per
/// axis. Anything else (a single interval, `Automatic`, …) is not this form.
fn parse_axis_ranges(expr: &Expr) -> Option<[(f64, f64); 3]> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 3 {
    return None;
  }
  let mut ranges = [(0.0, 0.0); 3];
  for (i, item) in items.iter().enumerate() {
    let Expr::List(pair) = item else {
      return None;
    };
    if pair.len() != 2 {
      return None;
    }
    let lo = try_eval_to_f64(&evaluate_expr_to_expr(&pair[0]).ok()?)?;
    let hi = try_eval_to_f64(&evaluate_expr_to_expr(&pair[1]).ok()?)?;
    // A reversed or degenerate interval (or a NaN bound) is not a frame.
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
      return None;
    }
    ranges[i] = (lo, hi);
  }
  Some(ranges)
}

/// The SVG markup of one `AxesLabel` entry. `None` for `None` — the way an
/// axis asks to stay unlabelled. Everything else is typeset by the shared
/// label renderer, so a label written in the FrontEnd as linear syntax
/// (`"\!\(\*FractionBox[\(d\[Theta]\), \(dt\)]\) (rad/s)"`) draws as the
/// fraction it stands for.
fn axis_label_markup(expr: &Expr) -> Option<String> {
  match expr {
    Expr::Identifier(s) if s == "None" => None,
    _ => {
      let markup = crate::functions::graphics::expr_to_svg_markup(expr);
      (!markup.is_empty()).then_some(markup)
    }
  }
}

/// The height of the strip a `PlotLabel` reserves above a 3-D picture.
/// Matches the strip the 2-D renderer reserves, so a labelled `Graphics`
/// and a labelled `Graphics3D` set their titles at the same height.
const PLOT_LABEL_STRIP: u32 = 26;

/// The typeset markup of the `PlotLabel` an option list carries, or `None`
/// when it carries none (`PlotLabel -> None` included). Any expression can
/// label a plot — a string, a `Style[…]`, a `Row[…]` of computed values —
/// so it is typeset the way every other label is rather than printed.
fn plot_label_markup(opts: &[Expr]) -> Option<String> {
  for opt in opts {
    let (pattern, replacement): (&Expr, &Expr) = match opt {
      Expr::Rule {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      _ => continue,
    };
    if !matches!(pattern, Expr::Identifier(n) if n == "PlotLabel") {
      continue;
    }
    let value = evaluate_expr_to_expr(replacement)
      .unwrap_or_else(|_| replacement.clone());
    if matches!(&value, Expr::Identifier(s) if s == "None" || s == "Null") {
      return None;
    }
    let markup = crate::functions::graphics::expr_to_svg_markup(&value);
    if !markup.is_empty() {
      return Some(markup);
    }
  }
  None
}

/// Set a 3-D picture's `PlotLabel` above it. The finished SVG is nested,
/// untouched, into a canvas one label strip taller, with the label centred
/// in the strip — the projection inside was scaled to the picture's own
/// size, so growing the canvas rather than the drawing area leaves it
/// exactly as it was. Returns `svg` unchanged when there is no label.
fn with_plot_label(
  svg: String,
  opts: &[Expr],
  width: u32,
  height: u32,
) -> String {
  let Some(label) = plot_label_markup(opts) else {
    return svg;
  };
  let total = height + PLOT_LABEL_STRIP;
  // The strip is painted in the picture's own background so the two read
  // as one canvas — an explicit `Background` reaches it too.
  let bg = opts
    .iter()
    .find_map(|opt| match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Background") => {
        crate::functions::graphics::parse_color(replacement)
      }
      _ => None,
    })
    .map(|c| {
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
    })
    .unwrap_or_else(|| {
      let (theme_bg, _, _, _, _) = crate::functions::plot::plot_theme();
      (theme_bg.0, theme_bg.1, theme_bg.2)
    });
  // `width="100%"` (an `ImageSize -> Full` picture) has to stay responsive,
  // so the outer canvas repeats whichever sizing the inner one chose.
  let sizing = if svg.starts_with("<svg width=\"100%\"") {
    format!(
      "width=\"100%\" viewBox=\"0 0 {width} {total}\" \
       preserveAspectRatio=\"xMidYMid meet\""
    )
  } else {
    format!(
      "width=\"{width}\" height=\"{total}\" viewBox=\"0 0 {width} {total}\""
    )
  };
  let cx = width as f64 / 2.0;
  format!(
    "<svg {sizing} xmlns=\"http://www.w3.org/2000/svg\">\n\
     <rect width=\"{width}\" height=\"{total}\" fill=\"rgb({},{},{})\"/>\n\
     <text x=\"{cx:.1}\" y=\"17\" text-anchor=\"middle\" \
     font-family=\"sans-serif\" font-size=\"16\" fill=\"#333333\">{label}</text>\n\
     <g transform=\"translate(0,{PLOT_LABEL_STRIP})\">{svg}</g>\n\
     </svg>",
    bg.0, bg.1, bg.2,
  )
}

/// The characters an SVG markup fragment actually shows, for width
/// estimation: everything outside its tags.
fn markup_char_count(markup: &str) -> usize {
  let mut count = 0;
  let mut in_tag = false;
  for c in markup.chars() {
    match c {
      '<' => in_tag = true,
      '>' => in_tag = false,
      _ if !in_tag => count += 1,
      _ => {}
    }
  }
  count.max(1)
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
          boundary: [true; 3],
          edge_color: None,
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
          boundary: [true; 3],
          edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

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
  let mut plot_style: Option<Expr> = None;

  for opt in &args[opt_start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_style = Some(replacement.as_ref().clone());
        }
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

  // ── Symbolic structure: GraphicsComplex[points, {style…, Polygon[quads]}] ──
  // `First[RevolutionPlot3D[…]]` is the surface itself, in world
  // coordinates, so it can be re-drawn inside another `Graphics3D` (the
  // way the notebooks that build solids out of plot slices do).  The
  // rendering below works in normalized coordinates and is unaffected.
  let structure = {
    let mut index_of: Vec<Vec<Option<usize>>> =
      vec![vec![None; n_theta + 1]; n_t + 1];
    let mut point_exprs: Vec<Expr> = Vec::new();
    for (i, row) in grid.iter().enumerate() {
      for (j, cell) in row.iter().enumerate() {
        if let Some(p) = cell {
          index_of[i][j] = Some(point_exprs.len());
          point_exprs.push(Expr::List(
            vec![Expr::Real(p.x), Expr::Real(p.y), Expr::Real(p.z)].into(),
          ));
        }
      }
    }
    let mut quads: Vec<Expr> = Vec::new();
    for i in 0..n_t {
      for j in 0..n_theta {
        if let (Some(a), Some(b), Some(c), Some(d)) = (
          index_of[i][j],
          index_of[i + 1][j],
          index_of[i + 1][j + 1],
          index_of[i][j + 1],
        ) {
          quads.push(Expr::List(
            [a, b, c, d]
              .iter()
              .map(|&k| Expr::Integer(k as i128 + 1))
              .collect::<Vec<_>>()
              .into(),
          ));
        }
      }
    }
    let mut content: Vec<Expr> = Vec::new();
    // `PlotStyle -> Opacity[…]` belongs to the surface, so a caller that
    // lifts it out of the plot keeps the translucency it asked for.
    match &plot_style {
      Some(Expr::List(items)) => content.extend(items.iter().cloned()),
      Some(other) => content.push(other.clone()),
      None => {}
    }
    content.push(Expr::FunctionCall {
      name: "Polygon".to_string(),
      args: vec![Expr::List(quads.into())].into(),
    });
    Expr::FunctionCall {
      name: "Graphics3D".to_string(),
      args: vec![Expr::FunctionCall {
        name: "GraphicsComplex".to_string(),
        args: vec![Expr::List(point_exprs.into()), Expr::List(content.into())]
          .into(),
      }]
      .into(),
    }
  };

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
          boundary: [true; 3],
          edge_color: None,
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
          boundary: [true; 3],
          edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result_with_structure(svg, structure))
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
              boundary: [true; 3],
              edge_color: None,
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
              boundary: [true; 3],
              edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

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

// ── ListLinePlot3D implementation ────────────────────────────────────

/// A projected 3D polyline segment awaiting depth-sorted emission.
struct LineSeg3D {
  x0: f64,
  y0: f64,
  x1: f64,
  y1: f64,
  depth: f64,
  color: (u8, u8, u8),
}

/// Implementation of ListLinePlot3D[data, opts...].
/// Accepts three formats:
/// - `{{z11, z12, ...}, {z21, z22, ...}, ...}` — matrix of heights; each row
///   becomes a curve through {x = column, y = row, z = value}
/// - `{{x1,y1,z1}, {x2,y2,z2}, ...}` — a single curve through explicit points
/// - `{{{x,y,z}, ...}, {{x,y,z}, ...}, ...}` — multiple curves
pub fn list_line_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
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

  // Each dataset is one polyline through its points (in data order).
  let mut datasets: Vec<Vec<(f64, f64, f64)>> = Vec::new();

  match &evaled_data {
    Expr::List(outer) if !outer.is_empty() => {
      match &outer[0] {
        Expr::List(inner)
          if inner.len() == 3 && !matches!(&inner[0], Expr::List(_)) =>
        {
          // Single curve: {{x,y,z}, {x,y,z}, ...}
          let pts = parse_xyz_points(outer);
          if !pts.is_empty() {
            datasets.push(pts);
          }
        }
        Expr::List(inner)
          if !inner.is_empty()
            && inner.iter().all(|e| !matches!(e, Expr::List(_))) =>
        {
          // Matrix of heights: {{z11, z12, ...}, ...}
          // Each row i becomes a curve through {j+1, i+1, z[i][j]}.
          for (i, row_expr) in outer.iter().enumerate() {
            if let Expr::List(row) = row_expr {
              let mut pts = Vec::new();
              for (j, val_expr) in row.iter().enumerate() {
                if let Some(z) = try_eval_to_f64(val_expr)
                  && z.is_finite()
                {
                  pts.push(((j + 1) as f64, (i + 1) as f64, z));
                }
              }
              if !pts.is_empty() {
                datasets.push(pts);
              }
            }
          }
        }
        Expr::List(_) => {
          // Multiple curves: {{{x,y,z},...}, {{x,y,z},...}, ...}
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
      "ListLinePlot3D: no valid data points found".into(),
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
      "ListLinePlot3D: data produced no finite values".into(),
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

  let normalize = |x: f64, y: f64, z: f64| -> Point3D {
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
    Point3D {
      x: nx,
      y: ny,
      z: nz,
    }
  };

  let mut segments: Vec<LineSeg3D> = Vec::new();

  for (di, ds) in datasets.iter().enumerate() {
    let color = crate::functions::plot::PLOT_COLORS
      [di % crate::functions::plot::PLOT_COLORS.len()];
    for pair in ds.windows(2) {
      let a = normalize(pair[0].0, pair[0].1, pair[0].2);
      let b = normalize(pair[1].0, pair[1].1, pair[1].2);
      let (ax, ay) = project(a, &camera);
      let (bx, by) = project(b, &camera);
      let mid = Point3D {
        x: (a.x + b.x) * 0.5,
        y: (a.y + b.y) * 0.5,
        z: (a.z + b.z) * 0.5,
      };
      segments.push(LineSeg3D {
        x0: ax,
        y0: ay,
        x1: bx,
        y1: by,
        depth: depth(mid, &camera),
        color,
      });
    }
    // A single-point dataset still contributes to the ranges but has no
    // segments; that matches drawing a zero-length line.
  }

  // Sort far-to-near (painter's)
  segments.sort_by(|a, b| {
    b.depth
      .partial_cmp(&a.depth)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let (z_axis_min, z_axis_max) = if (z_min - z_max).abs() < 1e-15 {
    (z_min - 0.5, z_max + 0.5)
  } else {
    (z_min, z_max)
  };

  let svg = generate_line3d_svg(
    &segments,
    &camera,
    (x_min, x_max),
    (y_min, y_max),
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result(svg))
}

#[allow(clippy::too_many_arguments)]
fn generate_line3d_svg(
  segments: &[LineSeg3D],
  camera: &Camera,
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  svg_width: u32,
  svg_height: u32,
  full_width: bool,
) -> Result<String, InterpreterError> {
  // Find bounding box of the projected segments and the axes box
  let mut px_min = f64::INFINITY;
  let mut px_max = f64::NEG_INFINITY;
  let mut py_min = f64::INFINITY;
  let mut py_max = f64::NEG_INFINITY;

  for seg in segments {
    px_min = px_min.min(seg.x0).min(seg.x1);
    px_max = px_max.max(seg.x0).max(seg.x1);
    py_min = py_min.min(seg.y0).min(seg.y1);
    py_max = py_max.max(seg.y0).max(seg.y1);
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
      "ListLinePlot3D: degenerate projection".into(),
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

  let mut svg = String::with_capacity(segments.len() * 100 + 2000);

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

  // Draw axes first (behind lines)
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

  // Merge-render data segments and box edges back-to-front (painter's)
  let emit_edge = |svg: &mut String, edge: &BoxEdge| {
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
  };

  let mut ei = 0;
  for seg in segments {
    while ei < sorted_edges.len() && sorted_edges[ei].depth >= seg.depth {
      emit_edge(&mut svg, &sorted_edges[ei]);
      ei += 1;
    }
    let (sx0, sy0) = to_svg(seg.x0, seg.y0);
    let (sx1, sy1) = to_svg(seg.x1, seg.y1);
    let (r, g, b) = seg.color;
    svg.push_str(&format!(
      "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"rgb({},{},{})\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>\n",
      sx0, sy0, sx1, sy1, r, g, b
    ));
  }
  while ei < sorted_edges.len() {
    emit_edge(&mut svg, &sorted_edges[ei]);
    ei += 1;
  }

  svg.push_str("</svg>");
  Ok(svg)
}

// ── SphericalPlot3D implementation ───────────────────────────────────

const SPHERICAL_GRID: usize = 50;

/// SphericalPlot3D[r, {theta, t0, t1}, {phi, p0, p1}]
/// Plots r(theta, phi) in spherical coordinates.
///
/// The rendered result carries its symbolic form —
/// `Graphics3D[GraphicsComplex[points, {Polygon[…], …}], opts]` — so that
/// `plot[[1]]` yields the surface mesh for reuse inside other graphics
/// (as Wolfram Demonstrations do).
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
  // `PlotPoints -> n`: number of samples per direction (n - 1 grid cells),
  // as in Wolfram. Low values matter: Demonstrations use PlotPoints -> 2
  // with MaxRecursion -> 0 to get intentionally flat facets.
  let mut plot_points: Option<usize> = None;
  // `RegionFunction -> f`: keep only surface parts where
  // f[x, y, z, theta, phi, r] is True; cells crossing the boundary are
  // clipped to it.
  let mut region_fn: Option<Expr> = None;
  // `BoundaryStyle -> style`: draw the boundary edges of the (possibly
  // region-clipped) surface with this color.
  let mut boundary_color: Option<(u8, u8, u8)> = None;

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
        Expr::Identifier(name) if name == "PlotPoints" => {
          let n = match evaluate_expr_to_expr(replacement) {
            Ok(Expr::Integer(n)) => Some(n),
            Ok(Expr::List(ref items)) => match items.first() {
              Some(Expr::Integer(n)) => Some(*n),
              _ => None,
            },
            _ => None,
          };
          if let Some(n) = n
            && n >= 2
          {
            plot_points = Some(n as usize);
          }
        }
        Expr::Identifier(name) if name == "RegionFunction" => {
          region_fn = Some(
            evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.as_ref().clone()),
          );
        }
        Expr::Identifier(name) if name == "BoundaryStyle" => {
          if !matches!(replacement.as_ref(), Expr::Identifier(n) if n == "None")
            && let Some(color) =
              crate::functions::graphics::parse_color(replacement)
          {
            boundary_color = Some((
              (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
              (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
              (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            ));
          }
        }
        _ => {}
      }
    }
  }

  let n_theta = plot_points.map(|p| p - 1).unwrap_or(SPHERICAL_GRID);
  let n_phi = plot_points.map(|p| p - 1).unwrap_or(SPHERICAL_GRID);
  let theta_range = theta_max - theta_min;
  let phi_range = phi_max - phi_min;

  // Evaluate the surface point at arbitrary (theta, phi) parameters.
  let surface_at = |theta: f64, phi: f64| -> Option<(Point3D, f64)> {
    let r = evaluate_at_t_theta(body, &theta_var, theta, &phi_var, phi)?;
    if !r.is_finite() {
      return None;
    }
    let p = Point3D {
      x: r * theta.sin() * phi.cos(),
      y: r * theta.sin() * phi.sin(),
      z: r * theta.cos(),
    };
    (p.x.is_finite() && p.y.is_finite() && p.z.is_finite()).then_some((p, r))
  };

  // Whether the surface point at (theta, phi) exists and satisfies the
  // region function (which receives x, y, z, theta, phi, r as in Wolfram).
  let inside_at = |theta: f64, phi: f64| -> bool {
    let Some((p, r)) = surface_at(theta, phi) else {
      return false;
    };
    let Some(region) = &region_fn else {
      return true;
    };
    let call = Expr::CurriedCall {
      func: Box::new(region.clone()),
      args: vec![
        Expr::Real(p.x),
        Expr::Real(p.y),
        Expr::Real(p.z),
        Expr::Real(theta),
        Expr::Real(phi),
        Expr::Real(r),
      ],
    };
    matches!(
      evaluate_expr_to_expr(&call),
      Ok(Expr::Identifier(ref s)) if s == "True"
    )
  };

  // Sample the function on a theta x phi grid
  let mut grid_pts: Vec<Vec<Option<Point3D>>> =
    vec![vec![None; n_phi + 1]; n_theta + 1];
  let mut grid_inside: Vec<Vec<bool>> =
    vec![vec![false; n_phi + 1]; n_theta + 1];

  let param_at = |i: usize, j: usize| -> (f64, f64) {
    (
      theta_min + (i as f64 / n_theta as f64) * theta_range,
      phi_min + (j as f64 / n_phi as f64) * phi_range,
    )
  };

  for i in 0..=n_theta {
    for j in 0..=n_phi {
      let (theta, phi) = param_at(i, j);
      if let Some((p, _)) = surface_at(theta, phi) {
        grid_pts[i][j] = Some(p);
        grid_inside[i][j] = if region_fn.is_some() {
          inside_at(theta, phi)
        } else {
          true
        };
      }
    }
  }

  // Clip a triangle (given in parameter space with per-vertex inside
  // flags) against the region boundary, bisecting crossing edges so cut
  // points land on the boundary. Returns the surviving polygon.
  let clip_triangle = |verts: [((f64, f64), bool); 3]| -> Vec<(f64, f64)> {
    if verts.iter().all(|(_, inside)| *inside) {
      return verts.iter().map(|(p, _)| *p).collect();
    }
    if verts.iter().all(|(_, inside)| !inside) {
      return Vec::new();
    }
    let bisect = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) {
      // `a` inside, `b` outside; converge onto the boundary.
      let (mut lo, mut hi) = (a, b);
      for _ in 0..24 {
        let mid = ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0);
        if inside_at(mid.0, mid.1) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      ((lo.0 + hi.0) / 2.0, (lo.1 + hi.1) / 2.0)
    };
    let mut out = Vec::new();
    for k in 0..3 {
      let (a, a_in) = verts[k];
      let (b, b_in) = verts[(k + 1) % 3];
      if a_in {
        out.push(a);
      }
      if a_in != b_in {
        out.push(if a_in { bisect(a, b) } else { bisect(b, a) });
      }
    }
    out
  };

  // Build the world-space triangles, clipping cells that cross the
  // region boundary.
  let mut world_tris: Vec<[Point3D; 3]> = Vec::new();
  for i in 0..n_theta {
    for j in 0..n_phi {
      // The cell's two triangles in grid corners:
      // (i,j), (i+1,j), (i,j+1) and (i+1,j+1), (i,j+1), (i+1,j).
      for corner_set in [
        [(i, j), (i + 1, j), (i, j + 1)],
        [(i + 1, j + 1), (i, j + 1), (i + 1, j)],
      ] {
        if corner_set
          .iter()
          .any(|&(ci, cj)| grid_pts[ci][cj].is_none())
        {
          continue;
        }
        let verts =
          corner_set.map(|(ci, cj)| (param_at(ci, cj), grid_inside[ci][cj]));
        let polygon = if region_fn.is_some() {
          clip_triangle(verts)
        } else {
          verts.iter().map(|(p, _)| *p).collect()
        };
        if polygon.len() < 3 {
          continue;
        }
        // Fan-triangulate the clipped polygon back into world space.
        let points: Vec<Point3D> = polygon
          .iter()
          .filter_map(|&(th, ph)| surface_at(th, ph).map(|(p, _)| p))
          .collect();
        if points.len() < 3 {
          continue;
        }
        for k in 1..points.len() - 1 {
          let (a, b, c) = (points[0], points[k], points[k + 1]);
          // Skip degenerate slivers (e.g. the collapsed pole edge).
          let n = triangle_normal(a, b, c);
          let ab =
            ((b.x - a.x).powi(2) + (b.y - a.y).powi(2) + (b.z - a.z).powi(2))
              .sqrt();
          let ac =
            ((c.x - a.x).powi(2) + (c.y - a.y).powi(2) + (c.z - a.z).powi(2))
              .sqrt();
          let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
          if area2 < 1e-12 * (ab * ac).max(1e-300) {
            continue;
          }
          world_tris.push([a, b, c]);
        }
      }
    }
  }

  if world_tris.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "SphericalPlot3D: no renderable triangles".into(),
    ));
  }

  // ── Symbolic structure: GraphicsComplex[points, {Polygon[…], …}] ──
  // Deduplicate shared vertices into an indexed coordinate list.
  let mut point_index: std::collections::HashMap<(i64, i64, i64), usize> =
    std::collections::HashMap::new();
  let mut points: Vec<Point3D> = Vec::new();
  let mut index_of = |p: Point3D| -> usize {
    let key = (
      (p.x * 1e9).round() as i64,
      (p.y * 1e9).round() as i64,
      (p.z * 1e9).round() as i64,
    );
    *point_index.entry(key).or_insert_with(|| {
      points.push(p);
      points.len() - 1
    })
  };
  let mut tri_indices: Vec<[usize; 3]> = Vec::new();
  for tri in &world_tris {
    tri_indices.push([index_of(tri[0]), index_of(tri[1]), index_of(tri[2])]);
  }

  // Boundary edges (used by BoundaryStyle): edges belonging to exactly
  // one triangle of the mesh.
  let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
    std::collections::HashMap::new();
  for tri in &tri_indices {
    for k in 0..3 {
      let (a, b) = (tri[k], tri[(k + 1) % 3]);
      if a != b {
        *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
      }
    }
  }
  let mut boundary_edges: Vec<(usize, usize)> = edge_count
    .iter()
    .filter(|&(_, &count)| count == 1)
    .map(|(&edge, _)| edge)
    .collect();
  boundary_edges.sort_unstable();

  let point_exprs: Vec<Expr> = points
    .iter()
    .map(|p| {
      Expr::List(vec![Expr::Real(p.x), Expr::Real(p.y), Expr::Real(p.z)].into())
    })
    .collect();
  let polygon_expr = Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(
      tri_indices
        .iter()
        .map(|tri| {
          Expr::List(
            tri
              .iter()
              .map(|&idx| Expr::Integer(idx as i128 + 1))
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )]
    .into(),
  };
  let mut gc_content = vec![polygon_expr];
  if let Some((r, g, b)) = boundary_color
    && !boundary_edges.is_empty()
  {
    let line_expr = Expr::FunctionCall {
      name: "Line".to_string(),
      args: vec![Expr::List(
        boundary_edges
          .iter()
          .map(|&(a, b)| {
            Expr::List(
              vec![Expr::Integer(a as i128 + 1), Expr::Integer(b as i128 + 1)]
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      )]
      .into(),
    };
    let color_expr = Expr::FunctionCall {
      name: "RGBColor".to_string(),
      args: vec![
        Expr::Real(r as f64 / 255.0),
        Expr::Real(g as f64 / 255.0),
        Expr::Real(b as f64 / 255.0),
      ]
      .into(),
    };
    gc_content.push(Expr::List(vec![color_expr, line_expr].into()));
  }
  let graphics_complex = Expr::FunctionCall {
    name: "GraphicsComplex".to_string(),
    args: vec![
      Expr::List(point_exprs.into()),
      Expr::List(gc_content.into()),
    ]
    .into(),
  };
  let structure = Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: std::iter::once(graphics_complex)
      .chain(args[3..].iter().cloned())
      .collect::<Vec<_>>()
      .into(),
  };

  // ── Standalone rendering (the plot's own SVG) ──
  // Find coordinate ranges
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  let mut z_min = f64::INFINITY;
  let mut z_max = f64::NEG_INFINITY;

  for tri in &world_tris {
    for p in tri {
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

  for tri in &world_tris {
    let [a, b, c] = *tri;
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
      boundary: [true; 3],
      edge_color: None,
      projected: [pa, pb, pc],
      depth: depth(center, &camera),
      color,
      opacity: 1.0,
    });
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result_with_structure(svg, structure))
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
          boundary: [true; 3],
          edge_color: None,
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
          boundary: [true; 3],
          edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result(svg))
}

/// Split a `PlotStyle` value into the `Tube[…]` directive it carries (as
/// that `Tube`'s arguments after the curve — its radius, when given) and
/// what remains of the style. `Tube` asks for a shape rather than a
/// colour, so it has to be lifted out of the directive list and applied to
/// the curve itself. Returns `(None, style)` when there is no `Tube`.
fn take_tube_directive(style: &Expr) -> (Option<Vec<Expr>>, Expr) {
  match style {
    Expr::FunctionCall { name, args } if name == "Tube" => {
      (Some(args.to_vec()), Expr::List(vec![].into()))
    }
    // `{colour, Tube[r]}` and `Directive[colour, Tube[r]]` both group
    // directives, so look inside either for the `Tube`.
    Expr::List(items) => {
      let (tube, kept) = partition_tube(items);
      (tube, Expr::List(kept.into()))
    }
    Expr::FunctionCall { name, args } if name == "Directive" => {
      let (tube, kept) = partition_tube(args);
      (
        tube,
        Expr::FunctionCall {
          name: name.clone(),
          args: kept.into(),
        },
      )
    }
    other => (None, other.clone()),
  }
}

/// The `Tube[…]` among a list of directives (its arguments) and the other
/// directives, searched one level down as well so that a nested grouping
/// (`{colour, Directive[Tube[r]]}`) is found too.
fn partition_tube(items: &[Expr]) -> (Option<Vec<Expr>>, Vec<Expr>) {
  let mut tube: Option<Vec<Expr>> = None;
  let mut kept: Vec<Expr> = Vec::with_capacity(items.len());
  for item in items {
    match item {
      Expr::FunctionCall { name, args } if name == "Tube" && tube.is_none() => {
        tube = Some(args.to_vec());
      }
      Expr::List(_) | Expr::FunctionCall { .. }
        if tube.is_none() && contains_tube(item) =>
      {
        let (inner, rest) = take_tube_directive(item);
        tube = inner;
        kept.push(rest);
      }
      other => kept.push(other.clone()),
    }
  }
  (tube, kept)
}

/// Whether a directive group holds a `Tube[…]` one level down.
fn contains_tube(expr: &Expr) -> bool {
  let items: &[Expr] = match expr {
    Expr::List(items) => items,
    Expr::FunctionCall { name, args } if name == "Directive" => args,
    _ => return false,
  };
  items
    .iter()
    .any(|i| matches!(i, Expr::FunctionCall { name, .. } if name == "Tube"))
}

/// 1-iterator (curve) form of ParametricPlot3D.
/// Returns an unevaluated `Graphics3D[{<directives>, Line[points]}, opts]`
/// so that downstream consumers like `Show[]` can merge it with other
/// 3D primitives. Only `PlotStyle` is interpreted from `opts`; remaining
/// options are forwarded to `Graphics3D` verbatim.
fn parametric_plot3d_curve_ast(
  body: &Expr,
  tvar: &str,
  t_min: f64,
  t_max: f64,
  opts: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Body may be a single triple {fx, fy, fz} or a list of triples.
  struct Curve<'a> {
    fx: &'a Expr,
    fy: &'a Expr,
    fz: &'a Expr,
  }
  let curves: Vec<Curve> = match body {
    Expr::List(items)
      if items.len() == 3 && !matches!(&items[0], Expr::List(_)) =>
    {
      vec![Curve {
        fx: &items[0],
        fy: &items[1],
        fz: &items[2],
      }]
    }
    Expr::List(items)
      if !items.is_empty()
        && items
          .iter()
          .all(|it| matches!(it, Expr::List(sub) if sub.len() == 3)) =>
    {
      items
        .iter()
        .map(|item| {
          if let Expr::List(sub) = item {
            Curve {
              fx: &sub[0],
              fy: &sub[1],
              fz: &sub[2],
            }
          } else {
            unreachable!()
          }
        })
        .collect()
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "ParametricPlot3D: first argument must be {fx, fy, fz}".into(),
      ));
    }
  };

  // Sample each curve at SAMPLE_N+1 points uniformly in [t_min, t_max].
  const SAMPLE_N: usize = 200;
  let dt = (t_max - t_min) / SAMPLE_N as f64;

  // Extract PlotStyle option (if any). Other options are kept verbatim.
  let mut plot_style: Option<&Expr> = None;
  let mut forwarded_opts: Vec<Expr> = Vec::new();
  for opt in opts {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "PlotStyle"
    {
      plot_style = Some(replacement.as_ref());
    } else {
      forwarded_opts.push(opt.clone());
    }
  }

  // `PlotStyle -> Tube[r]` asks for the curve to be drawn as a tube of
  // radius `r` instead of as a line — how a Demonstration gives a knot its
  // thickness. It is not a colour or a thickness, so it comes out of the
  // style list and turns each sampled polyline into a `Tube` instead.
  let (tube_args, plot_style) = match plot_style {
    Some(ps) => {
      let (tube, rest) = take_tube_directive(ps);
      (tube, Some(rest))
    }
    None => (None, None),
  };

  // Build a primitives list: [<style directives>, Line[pts1], Line[pts2], …]
  let mut prim_items: Vec<Expr> = Vec::new();
  if let Some(ps) = &plot_style {
    // Wrap whatever PlotStyle holds into a Directive[…] so that
    // collect_3d_primitives picks up nested colors/Thickness/etc.
    prim_items.push(Expr::FunctionCall {
      name: "Directive".to_string(),
      args: vec![ps.clone()].into(),
    });
  }

  let mut produced_any = false;
  for curve in &curves {
    let mut current_segment: Vec<Expr> = Vec::new();
    let flush = |seg: &mut Vec<Expr>, sink: &mut Vec<Expr>| {
      if seg.len() >= 2 {
        let points = Expr::List(std::mem::take(seg).into());
        sink.push(match &tube_args {
          Some(extra) => Expr::FunctionCall {
            name: "Tube".to_string(),
            args: std::iter::once(points)
              .chain(extra.iter().cloned())
              .collect::<Vec<_>>()
              .into(),
          },
          None => Expr::FunctionCall {
            name: "Line".to_string(),
            args: vec![points].into(),
          },
        });
      } else {
        seg.clear();
      }
    };
    for i in 0..=SAMPLE_N {
      let t = t_min + i as f64 * dt;
      if let Some((x, y, z)) =
        evaluate_parametric_at_t(curve.fx, curve.fy, curve.fz, tvar, t)
      {
        current_segment.push(Expr::List(
          vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)].into(),
        ));
      } else {
        flush(&mut current_segment, &mut prim_items);
      }
    }
    if current_segment.len() >= 2 {
      produced_any = true;
    }
    flush(&mut current_segment, &mut prim_items);
  }

  if !produced_any
    && prim_items.iter().all(
      |e| !matches!(e, Expr::FunctionCall { name, .. } if name == "Line" || name == "Tube"),
    )
  {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot3D: parametric function produced no finite values".into(),
    ));
  }

  // Build Graphics3D[{...}, opts...] as an unevaluated FunctionCall so
  // that `Show[]` can merge it with other 3D primitives. When this
  // expression is the top-level result, `render_graphics_fc_if_needed`
  // dispatches it to `graphics3d_ast`.
  let content = Expr::List(prim_items.into());
  let mut g3d_args = vec![content];
  g3d_args.extend(forwarded_opts);

  Ok(Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: g3d_args.into(),
  })
}

/// Evaluate a parametric triple {fx(t), fy(t), fz(t)} at a given t value.
fn evaluate_parametric_at_t(
  fx: &Expr,
  fy: &Expr,
  fz: &Expr,
  tvar: &str,
  tval: f64,
) -> Option<(f64, f64, f64)> {
  let eval_one = |body: &Expr| -> Option<f64> {
    let sub = substitute_var(body, tvar, &Expr::Real(tval));
    let result = evaluate_expr_to_expr(&sub).ok()?;
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

/// Implementation of ParametricPlot3D.
///
/// Two forms are supported:
/// - Curve: `ParametricPlot3D[{fx, fy, fz}, {t, tmin, tmax}, opts...]`
///   Samples the curve and returns an unevaluated
///   `Graphics3D[{<directives>, Line[{...}]}, opts]` so that `Show[]` can
///   merge it with other 3D primitives.
/// - Surface: `ParametricPlot3D[{fx, fy, fz}, {u, umin, umax}, {v, vmin, vmax}, opts...]`
///   Triangulates the surface and returns a fully rendered Graphics3D SVG.
pub fn parametric_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Err(InterpreterError::EvaluationError(
      "ParametricPlot3D requires at least 2 arguments: ParametricPlot3D[{fx, fy, fz}, {t, tmin, tmax}]".into(),
    ));
  }

  let body = &args[0];

  // Parse the first iterator. If the second argument also parses as an
  // iterator, fall through to the surface (2-iterator) implementation.
  // Otherwise treat as a curve and return early.
  let (uvar, u_min, u_max) = parse_iterator(&args[1], "first")?;
  let surface_iter = args.get(2).and_then(|a| parse_iterator(a, "second").ok());

  if surface_iter.is_none() {
    return parametric_plot3d_curve_ast(body, &uvar, u_min, u_max, &args[2..]);
  }

  let (vvar, v_min, v_max) = surface_iter.unwrap();

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
            boundary: [true; 3],
            edge_color: None,
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
            boundary: [true; 3],
            edge_color: None,
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
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::graphics3d_result(svg))
}
