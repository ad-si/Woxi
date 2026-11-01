use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::{try_eval_to_f64, try_eval_to_f64_lenient};
use crate::functions::plot::{
  PLOT_COLORS, format_tick, nice_step, parse_image_size, substitute_var,
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
  /// `ViewVertical`: the direction that points up on screen. The default
  /// `{0, 0, 1}` is the one the azimuth/elevation basis already assumes.
  pub vertical: [f64; 3],
}

impl Camera {
  /// Wolfram's default `ViewPoint`, in units of the longest side of the
  /// displayed box.
  pub(crate) const DEFAULT_VIEW_POINT: [f64; 3] = [1.3, -2.4, 2.0];
  pub(crate) const DEFAULT_VERTICAL: [f64; 3] = [0.0, 0.0, 1.0];

  /// Orthonormal screen basis `(right, up)` for a `ViewVertical` other than
  /// the default: `up` is the vertical with its component along the view
  /// direction removed, as Wolfram levels the camera against it. `None` for
  /// a degenerate vertical (parallel to the view direction).
  fn tilted_basis(&self) -> Option<([f64; 3], [f64; 3])> {
    let (sa, ca) = self.azimuth.sin_cos();
    let (se, ce) = self.elevation.sin_cos();
    let d = [ce * ca, ce * sa, se];
    let v = self.vertical;
    let cross = |a: [f64; 3], b: [f64; 3]| {
      [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ]
    };
    let right = cross(v, d);
    let len =
      (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    if len < 1e-9 {
      return None;
    }
    let right = [right[0] / len, right[1] / len, right[2] / len];
    Some((right, cross(d, right)))
  }
}

impl Default for Camera {
  fn default() -> Self {
    // Matches Mathematica's default ViewPoint {1.3, -2.4, 2.0}
    Self {
      azimuth: -1.07,  // ~-61 degrees
      elevation: 0.63, // ~36 degrees
      vertical: Camera::DEFAULT_VERTICAL,
    }
  }
}

#[derive(Clone)]
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
  /// The stroke width `EdgeForm[{…, Thick}]` asked outline edges to be
  /// drawn with. `None` keeps the default hairline.
  pub edge_width: Option<f64>,
  /// The image painted over a polygon's triangle, when a `Texture` is in
  /// force and the polygon carries `VertexTextureCoordinates`.
  pub texture: Option<TriangleTexture>,
}

/// A `Texture` image placed on one triangle by the texture coordinate of
/// each of its vertices (`(0, 0)` bottom-left, `(1, 1)` top-right).
#[derive(Clone)]
pub(crate) struct TriangleTexture {
  pub image: std::sync::Arc<image::RgbImage>,
  pub uv: [(f64, f64); 3],
}

/// Largest scene `painter_order` refines: candidate pairs come from a sweep
/// over the projected bounding boxes, but a dense scene can still be close
/// to quadratic, so bigger scenes keep the plain centroid sort.
const PAINTER_ORDER_MAX_TRIANGLES: usize = 200000;

/// Largest scene `split_interpenetrating` cuts up; splitting multiplies the
/// triangle count, which must stay within `PAINTER_ORDER_MAX_TRIANGLES`.
const SPLIT_MAX_TRIANGLES: usize = 60000;

/// Whether two projected triangles share interior area (touching along an
/// edge or at a corner does not count).
fn projected_triangles_overlap(
  a: &[(f64, f64); 3],
  b: &[(f64, f64); 3],
) -> bool {
  let eps = 1e-9;
  for tri in [a, b] {
    for i in 0..3 {
      let (x0, y0) = tri[i];
      let (x1, y1) = tri[(i + 1) % 3];
      let (nx, ny) = (y1 - y0, x0 - x1);
      let range = |t: &[(f64, f64); 3]| {
        t.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &(x, y)| {
          let v = x * nx + y * ny;
          (lo.min(v), hi.max(v))
        })
      };
      let (alo, ahi) = range(a);
      let (blo, bhi) = range(b);
      let tol = eps * (nx.abs() + ny.abs()).max(1e-300);
      if ahi <= blo + tol || bhi <= alo + tol {
        return false;
      }
    }
  }
  true
}

/// Where triangle `t` lies relative to the plane of `plane`, seen from the
/// viewer: `Some(true)` when wholly behind (or on) it, `Some(false)` when
/// wholly in front, `None` when it straddles the plane.
fn side_of_plane(
  t: &[Point3D; 3],
  plane: &[Point3D; 3],
  gradient: [f64; 3],
) -> Option<bool> {
  let mut n = triangle_normal(plane[0], plane[1], plane[2]);
  if n[0] * gradient[0] + n[1] * gradient[1] + n[2] * gradient[2] > 0.0 {
    n = [-n[0], -n[1], -n[2]];
  }
  // `n` now points toward the viewer.
  let eps = 1e-9;
  let (mut behind, mut front) = (true, true);
  for v in t {
    let d = n[0] * (v.x - plane[0].x)
      + n[1] * (v.y - plane[0].y)
      + n[2] * (v.z - plane[0].z);
    if d > eps {
      behind = false;
    }
    if d < -eps {
      front = false;
    }
  }
  match (behind, front) {
    (true, _) => Some(true),
    (_, true) => Some(false),
    _ => None,
  }
}

/// Calls `f(i, j)` for every pair of triangles whose projections share
/// interior area. Candidate pairs come from a sweep along x over the
/// projected bounding boxes.
fn for_each_overlapping_pair(
  projected: &[[(f64, f64); 3]],
  mut f: impl FnMut(usize, usize),
) {
  let n = projected.len();
  let bbox: Vec<[f64; 4]> = projected
    .iter()
    .map(|t| {
      let xs = t.iter().map(|p| p.0);
      let ys = t.iter().map(|p| p.1);
      [
        xs.clone().fold(f64::MAX, f64::min),
        xs.fold(f64::MIN, f64::max),
        ys.clone().fold(f64::MAX, f64::min),
        ys.fold(f64::MIN, f64::max),
      ]
    })
    .collect();
  let mut by_x: Vec<usize> = (0..n).collect();
  by_x.sort_by(|&a, &b| {
    bbox[a][0]
      .partial_cmp(&bbox[b][0])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  for (pos, &i) in by_x.iter().enumerate() {
    for &j in &by_x[pos + 1..] {
      let (a, b) = (&bbox[i], &bbox[j]);
      if b[0] >= a[1] {
        break;
      }
      if a[3] <= b[2] || b[3] <= a[2] {
        continue;
      }
      if projected_triangles_overlap(&projected[i], &projected[j]) {
        f(i, j);
      }
    }
  }
}

/// The gradient of `depth` in scene coordinates: depth(p) = gradient . p.
fn depth_gradient(camera: &Camera) -> [f64; 3] {
  let unit = |x, y, z| Point3D { x, y, z };
  let d0 = depth(unit(0.0, 0.0, 0.0), camera);
  [
    depth(unit(1.0, 0.0, 0.0), camera) - d0,
    depth(unit(0.0, 1.0, 0.0), camera) - d0,
    depth(unit(0.0, 0.0, 1.0), camera) - d0,
  ]
}

/// Most cutting planes applied to a single triangle, bounding the growth of
/// a scene whose surfaces cross each other heavily.
const MAX_SPLIT_PLANES: usize = 256;

/// Most pieces one triangle is cut into.
const MAX_PIECES_PER_TRIANGLE: usize = 64;

/// Cuts a convex polygon (corner, flag of the edge to the next corner) by
/// the plane through `plane`, returning the parts on its near and far side.
/// Edges created by the cut carry a `false` flag.
fn clip_polygon_by_plane(
  poly: &[(Point3D, bool)],
  plane: &[Point3D; 3],
) -> (Vec<(Point3D, bool)>, Vec<(Point3D, bool)>) {
  let n = triangle_normal(plane[0], plane[1], plane[2]);
  let eps = 1e-9;
  let dist = |p: Point3D| {
    n[0] * (p.x - plane[0].x)
      + n[1] * (p.y - plane[0].y)
      + n[2] * (p.z - plane[0].z)
  };
  let mut pos: Vec<(Point3D, bool)> = Vec::new();
  let mut neg: Vec<(Point3D, bool)> = Vec::new();
  for k in 0..poly.len() {
    let (a, flag) = poly[k];
    let b = poly[(k + 1) % poly.len()].0;
    let (da, db) = (dist(a), dist(b));
    if da >= -eps {
      pos.push((a, flag));
    }
    if da <= eps {
      neg.push((a, flag));
    }
    if (da > eps && db < -eps) || (da < -eps && db > eps) {
      let t = da / (da - db);
      let m = Point3D {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y),
        z: a.z + t * (b.z - a.z),
      };
      // The first half of the split edge keeps its flag; the second half
      // starts at the cut point.
      if da > 0.0 {
        pos.push((m, false));
        neg.push((m, flag));
      } else {
        neg.push((m, false));
        pos.push((m, flag));
      }
    }
  }
  (pos, neg)
}

/// Splits triangles that pass through one another where they overlap on
/// screen, so each piece lies wholly on one side of every triangle it
/// crosses and a back-to-front sort can order them (the cones of a
/// Voronoi construction, for example, meet along curves).
fn split_interpenetrating(
  verts: &mut Vec<[Point3D; 3]>,
  triangles: &mut Vec<Triangle>,
  camera: &Camera,
) {
  let n = verts.len();
  if !(2..=SPLIT_MAX_TRIANGLES).contains(&n) {
    return;
  }
  let gradient = depth_gradient(camera);
  let projected: Vec<_> = triangles.iter().map(|t| t.projected).collect();
  let mut planes: Vec<Vec<usize>> = vec![Vec::new(); n];
  for_each_overlapping_pair(&projected, |i, j| {
    if side_of_plane(&verts[i], &verts[j], gradient).is_none()
      && side_of_plane(&verts[j], &verts[i], gradient).is_none()
    {
      if planes[i].len() < MAX_SPLIT_PLANES {
        planes[i].push(j);
      }
      if planes[j].len() < MAX_SPLIT_PLANES {
        planes[j].push(i);
      }
    }
  });
  if planes.iter().all(std::vec::Vec::is_empty) {
    return;
  }
  let original = verts.clone();
  let mut new_verts = Vec::with_capacity(n);
  let mut new_tris = Vec::with_capacity(n);
  for (k, tri) in triangles.drain(..).enumerate() {
    // The split scene has to stay within what `painter_order` sorts: once
    // the triangles still to come would only just fit, they stay whole.
    let remaining = n - k;
    if planes[k].is_empty()
      || tri.texture.is_some()
      || new_tris.len() + remaining * MAX_PIECES_PER_TRIANGLE
        > PAINTER_ORDER_MAX_TRIANGLES
    {
      new_verts.push(original[k]);
      new_tris.push(tri);
      continue;
    }
    let mut pieces = vec![
      (0..3)
        .map(|c| (original[k][c], tri.boundary[c]))
        .collect::<Vec<_>>(),
    ];
    for &other in &planes[k] {
      // A triangle that many surfaces cross (a thick tube wound tighter
      // than its radius) would split into thousands of slivers.
      if pieces.len() >= MAX_PIECES_PER_TRIANGLE {
        break;
      }
      let mut next = Vec::with_capacity(pieces.len() + 1);
      for piece in pieces {
        let (a, b) = clip_polygon_by_plane(&piece, &original[other]);
        if a.len() >= 3 && b.len() >= 3 {
          next.push(a);
          next.push(b);
        } else {
          next.push(piece);
        }
      }
      pieces = next;
    }
    for poly in pieces {
      let m = poly.len();
      for c in 1..m - 1 {
        let (v0, v1, v2) = (poly[0].0, poly[c].0, poly[c + 1].0);
        let center = Point3D {
          x: (v0.x + v1.x + v2.x) / 3.0,
          y: (v0.y + v1.y + v2.y) / 3.0,
          z: (v0.z + v1.z + v2.z) / 3.0,
        };
        let mut piece = tri.clone();
        piece.projected = [
          project(v0, camera),
          project(v1, camera),
          project(v2, camera),
        ];
        piece.depth = depth(center, camera);
        piece.boundary =
          [c == 1 && poly[0].1, poly[c].1, c + 2 == m && poly[m - 1].1];
        new_verts.push([v0, v1, v2]);
        new_tris.push(piece);
      }
    }
  }
  *verts = new_verts;
  *triangles = new_tris;
}

/// Back-to-front draw order for the triangles. A centroid sort alone draws a
/// large face that lies behind a small one on top of it (a cube inside a
/// polyhedron pokes through its faces), so every overlapping pair is first
/// ordered by which one lies wholly behind the other's plane; the centroid
/// depth only breaks the remaining ties and cycles.
fn painter_order(
  verts: &[[Point3D; 3]],
  projected: &[[(f64, f64); 3]],
  depths: &[f64],
  camera: &Camera,
) -> Vec<usize> {
  let n = verts.len();
  let mut by_depth: Vec<usize> = (0..n).collect();
  by_depth.sort_by(|&a, &b| {
    depths[b]
      .partial_cmp(&depths[a])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  if n > PAINTER_ORDER_MAX_TRIANGLES {
    return by_depth;
  }
  let gradient = depth_gradient(camera);
  // `after[i]` lists the triangles that must be drawn after `i`.
  let mut after: Vec<Vec<usize>> = vec![Vec::new(); n];
  let mut indegree = vec![0usize; n];
  for_each_overlapping_pair(projected, |i, j| {
    let i_behind = side_of_plane(&verts[i], &verts[j], gradient)
      .or_else(|| side_of_plane(&verts[j], &verts[i], gradient).map(|s| !s));
    match i_behind {
      Some(true) => {
        after[i].push(j);
        indegree[j] += 1;
      }
      Some(false) => {
        after[j].push(i);
        indegree[i] += 1;
      }
      None => {}
    }
  });
  let mut order = Vec::with_capacity(n);
  let mut done = vec![false; n];
  // Ready triangles, keyed by their far-to-near rank in `by_depth` so the
  // farthest one is drawn first (`Reverse` turns the max-heap into a min-heap).
  let mut rank = vec![0usize; n];
  for (r, &k) in by_depth.iter().enumerate() {
    rank[k] = r;
  }
  let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0
    ..n)
    .filter(|&k| indegree[k] == 0)
    .map(|k| std::cmp::Reverse(rank[k]))
    .collect();
  // First possibly-undrawn entry of `by_depth`, for breaking cycles.
  let mut cursor = 0;
  while order.len() < n {
    let pick = if let Some(std::cmp::Reverse(r)) = ready.pop() {
      by_depth[r]
    } else {
      while done[by_depth[cursor]] {
        cursor += 1;
      }
      by_depth[cursor]
    };
    if done[pick] {
      continue;
    }
    done[pick] = true;
    order.push(pick);
    for &k in &after[pick] {
      indegree[k] = indegree[k].saturating_sub(1);
      if indegree[k] == 0 && !done[k] {
        ready.push(std::cmp::Reverse(rank[k]));
      }
    }
  }
  order
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
  if cam.vertical != Camera::DEFAULT_VERTICAL
    && let Some((r, u)) = cam.tilted_basis()
  {
    return (
      p.x * r[0] + p.y * r[1] + p.z * r[2],
      p.x * u[0] + p.y * u[1] + p.z * u[2],
    );
  }
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

/// Evaluates a `MeshFunctions`/`RegionFunction`-style function at a
/// facet's `(x, y, z)`, called positionally exactly like
/// `Plot3D`/`ContourPlot3D`'s region function (see
/// `field_plot::region_allows`), so it does not need the plot's own
/// variable names substituted in first.
fn eval_mesh_function(f: &Expr, x: f64, y: f64, z: f64) -> Option<f64> {
  let call = Expr::CurriedCall {
    func: Box::new(f.clone()),
    args: vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)],
  };
  try_eval_to_f64(&evaluate_expr_to_expr(&call).ok()?)
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

/// A `{var, lo, hi}` iterator whose bounds are expressions in the outer
/// iterator variable `outer` (`{v, 0, h - Sin[u]}`); they are evaluated
/// per sample later, so they are returned unevaluated.
fn parse_dependent_iterator(
  spec: &Expr,
  outer: &str,
) -> Option<(String, Expr, Expr)> {
  match spec {
    Expr::List(items) if items.len() == 3 => match &items[0] {
      Expr::Identifier(name) if name != outer => {
        Some((name.clone(), items[1].clone(), items[2].clone()))
      }
      _ => None,
    },
    _ => None,
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
/// `Plot3D[f, {x, -Infinity, Infinity}, …]`: every iterator with an
/// infinite limit is folded onto a finite display range, and the matching
/// map back onto the real line is substituted into `f` — the same
/// compactification `Plot` uses. `None` when every limit is finite.
fn compactify_infinite_iterators(args: &[Expr]) -> Option<Vec<Expr>> {
  let mut new_args = args.to_vec();
  let mut changed = false;
  for idx in [1, 2] {
    let Some(Expr::List(items)) = args.get(idx) else {
      continue;
    };
    let [Expr::Identifier(var), lo, hi] = items.as_slice() else {
      continue;
    };
    let limit = |e: &Expr| {
      let v = evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone());
      crate::functions::math_ast::try_eval_to_f64_with_infinity(&v)
    };
    let (Some(lo), Some(hi)) = (limit(lo), limit(hi)) else {
      continue;
    };
    if lo.is_finite() && hi.is_finite() {
      continue;
    }
    let (d_lo, d_hi, transform) =
      crate::functions::plot::compactify_plot_range(var, lo, hi).ok()?;
    if let Some(t) = transform {
      new_args[0] = crate::syntax::substitute_variable(&new_args[0], var, &t);
    }
    new_args[idx] = Expr::List(
      vec![
        Expr::Identifier(var.clone()),
        Expr::Real(d_lo),
        Expr::Real(d_hi),
      ]
      .into(),
    );
    changed = true;
  }
  changed.then_some(new_args)
}

pub fn plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 {
    return Err(InterpreterError::EvaluationError(
            "Plot3D requires at least 3 arguments: Plot3D[f, {x, xmin, xmax}, {y, ymin, ymax}]"
                .into(),
        ));
  }

  if let Some(compact) = compactify_infinite_iterators(args) {
    return plot3d_ast(&compact);
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
  let mut plot_style_expr: Option<&Expr> = None;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
  let mut color_fn = FacetColorFunction::default();

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      color_fn.read_option(pattern, replacement);
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_style_expr = Some(replacement.as_ref());
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
          } else {
            z_clip = parse_z_only_range(replacement);
          }
        }
        Expr::Identifier(name) if name == "Boxed" => {
          match replacement.as_ref() {
            Expr::Identifier(s) if s == "False" => show_axes = false,
            Expr::Identifier(s) if s == "True" => show_axes = true,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "AxesLabel" => {
          axes_labels =
            parse_axes_labels(replacement, Some([&xvar, &yvar, "z"]));
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

  // `PlotStyle`, one style per surface (cycling); empty means it was not
  // given, so the height-based rainbow default is used unchanged.
  let plot_styles: Vec<StyleState3D> = plot_style_expr
    .map(|e| parse_plot_style_3d(e, bodies.len()))
    .unwrap_or_default();

  let camera = Camera::default();
  // Direction from the scene towards the viewer, used to place a
  // `Specularity` highlight from `PlotStyle`.
  let view_dir = {
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };
  let x_step = (x_max - x_min) / GRID_N as f64;
  let y_step = (y_max - y_min) / GRID_N as f64;

  // Phase 1: Sample all grids and compute global z range
  let mut grids: Vec<Vec<Vec<f64>>> = Vec::with_capacity(bodies.len());
  let mut global_z_min = f64::INFINITY;
  let mut global_z_max = f64::NEG_INFINITY;

  for func_body in &bodies {
    let mut grid = vec![vec![f64::NAN; GRID_N + 1]; GRID_N + 1];
    let compiled =
      crate::functions::plot::PlotBody2::new(func_body, &xvar, &yvar);
    for i in 0..=GRID_N {
      let xval = x_min + i as f64 * x_step;
      for j in 0..=GRID_N {
        let yval = y_min + j as f64 * y_step;
        if let Some(z) = compiled.at(xval, yval)
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

  // The style `ColorFunction` gives the facet over grid cell `(i, j)`,
  // evaluated at the cell's centre.
  let cell_color_style = |grid: &[Vec<f64>], i: usize, j: usize| {
    color_fn.function.as_ref()?;
    let corners = [
      grid[i][j],
      grid[i + 1][j],
      grid[i + 1][j + 1],
      grid[i][j + 1],
    ];
    if corners.iter().any(|z| !z.is_finite()) {
      return None;
    }
    let z = corners.iter().map(|z| z.clamp(z_lo, z_hi)).sum::<f64>() / 4.0;
    color_fn.style_at(
      [
        x_min + (i as f64 + 0.5) * x_step,
        y_min + (j as f64 + 0.5) * y_step,
        z,
      ],
      [(x_min, x_max), (y_min, y_max), (z_lo, z_hi)],
    )
  };

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
          content.push(call0("EdgeForm"));
        }
        content.extend(surface_opacity_directive(&plot_styles, surface_idx));
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
            let height_color =
              surface_height_color(avg_z_norm, surface_idx, num_surfaces);
            let facet_style = cell_color_style(grid, i, j);
            let (cr, cg, cb) = match facet_style
              .as_ref()
              .or(plot_style_for_surface(&plot_styles, surface_idx))
            {
              Some(style) => style.color.unwrap_or(height_color),
              None => height_color,
            };
            // Colour and polygon travel in a sublist so the directive does
            // not leak onto the quads drawn after it.
            let mut quad: Vec<Expr> = Vec::new();
            if let Some(style) = &facet_style
              && style.opacity < 1.0
            {
              quad.push(call1("Opacity", Expr::Real(style.opacity)));
            }
            quad.push(call(
              "RGBColor",
              vec![
                Expr::Real(cr as f64 / 255.0),
                Expr::Real(cg as f64 / 255.0),
                Expr::Real(cb as f64 / 255.0),
              ],
            ));
            quad.push(call1(
              "Polygon",
              Expr::List(
                [a, b, c, d]
                  .iter()
                  .map(|&k| Expr::Integer(k as i128 + 1))
                  .collect::<Vec<_>>()
                  .into(),
              ),
            ));
            content.push(Expr::List(quad.into()));
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
                call1("Opacity", Expr::Real(0.63)),
                call(
                  "RGBColor",
                  vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
                ),
                call1("AbsoluteThickness", Expr::Real(0.5)),
                call1("Line", Expr::List(segments.into())),
              ]
              .into(),
            ));
          }
        }
        call(
          "GraphicsComplex",
          vec![Expr::List(point_exprs.into()), Expr::List(content.into())],
        )
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
    // `AxesLabel -> Automatic` names the axes after the plot's variables;
    // a `Graphics3D` has no variables to name them by, so spell them out.
    for opt in &mut structure_args {
      if let Expr::Rule {
        pattern,
        replacement,
      } = opt
        && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "AxesLabel")
      {
        let name = |s: &str| Expr::Identifier(s.to_string());
        let resolve = |e: &Expr, default: &str| match e {
          Expr::Identifier(a) if a == "Automatic" => name(default),
          other => other.clone(),
        };
        let defaults = [xvar.as_str(), yvar.as_str(), "z"];
        match replacement.as_ref() {
          Expr::Identifier(a) if a == "Automatic" => {
            **replacement = Expr::List(
              defaults.iter().map(|d| name(d)).collect::<Vec<_>>().into(),
            );
          }
          Expr::List(items) => {
            let resolved: Vec<Expr> = items
              .iter()
              .enumerate()
              .map(|(i, e)| resolve(e, defaults.get(i).unwrap_or(&"")))
              .collect();
            **replacement = Expr::List(resolved.into());
          }
          _ => {}
        }
      }
    }
    let names = |opt: &str| {
      structure_args.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == opt))
      })
    };
    let (names_axes, names_ratios) = (names("Axes"), names("BoxRatios"));
    if !names_axes {
      structure_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
    }
    if !names_ratios {
      structure_args.push(rule_expr(
        id_expr("BoxRatios"),
        Expr::List(
          vec![Expr::Integer(1), Expr::Integer(1), Expr::Real(Z_SCALE)].into(),
        ),
      ));
    }
    call("Graphics3D", structure_args)
  };

  for (surface_idx, grid) in grids.iter().enumerate() {
    for i in 0..GRID_N {
      for j in 0..GRID_N {
        let z00 = grid[i][j];
        let z10 = grid[i + 1][j];
        let z01 = grid[i][j + 1];
        let z11 = grid[i + 1][j + 1];
        let cell_style = cell_color_style(grid, i, j);

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
          let default_color =
            surface_height_color(avg_z_norm, surface_idx, num_surfaces);
          let normal = triangle_normal(v0, v1, v2);
          let style = cell_style
            .as_ref()
            .or(plot_style_for_surface(&plot_styles, surface_idx));
          let (color, opacity) =
            shade_facet(default_color, style, normal, view_dir);

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
            edge_width: None,
            projected: [p0, p1, p2],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
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
          let default_color =
            surface_height_color(avg_z_norm, surface_idx, num_surfaces);
          let normal = triangle_normal(v0, v1, v2);
          let style = cell_style
            .as_ref()
            .or(plot_style_for_surface(&plot_styles, surface_idx));
          let (color, opacity) =
            shade_facet(default_color, style, normal, view_dir);

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
            edge_width: None,
            projected: [p0, p1, p2],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
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
    [1.0, 1.0, Z_SCALE],
    &axes_labels,
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::plot3d_result_with_structure(svg, structure))
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
  box_ratios: [f64; 3],
  axes_labels: &[Option<String>; 3],
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
  let bbox_corners = bounding_box_corners_with_ratios(box_ratios);
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

  // Compute scale and offset to map projected coords to SVG coords. An
  // `AxesLabel` given while axes are shown needs extra room outside the
  // box beyond what the tick labels alone reserve (mirrors
  // `graphics3d_ast`'s margin); every other case keeps the fixed margin
  // this renderer always used.
  let margin = if show_axes && axes_labels.iter().any(Option::is_some) {
    axes_label_margin(axes_labels, svg_width.min(svg_height))
  } else {
    25.0
  };
  let draw_w = svg_width as f64 - 2.0 * margin;
  let draw_h = svg_height as f64 - 2.0 * margin;
  let scale = (draw_w / p_width).min(draw_h / p_height);
  let cx = margin + draw_w / 2.0;
  let cy = margin + draw_h / 2.0;
  let p_cx = f64::midpoint(px_min, px_max);
  let p_cy = f64::midpoint(py_min, py_max);

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    let sx = cx + (px - p_cx) * scale;
    let sy = cy - (py - p_cy) * scale; // flip Y for SVG
    (sx, sy)
  };

  let mut svg = String::with_capacity(triangles.len() * 120 + 2000);

  if full_width {
    svg.push_str(&format!(
            "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
        ));
  } else {
    svg.push_str(&format!(
            "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
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
    let corners = bounding_box_corners_with_ratios(box_ratios);
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
        let tm = f64::midpoint(t0, t1);
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
          "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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
      if mesh_mode == MeshMode::All {
        svg.push_str(&format!(
          "<polygon points=\"{x0:.1},{y0:.1} {x1:.1},{y1:.1} {x2:.1},{y2:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"#00000060\" stroke-width=\"0.5\"{opacity_attr}/>\n"
        ));
      } else {
        // A translucent triangle's own-colour hairline overlaps its
        // neighbours' and darkens every shared seam, tracing the
        // tessellation across what should be a smooth surface. Only an
        // opaque triangle needs it to close the anti-aliasing gap.
        let seam_stroke = if tri.opacity < 1.0 && tri.boundary == [false; 3] {
          "none".to_string()
        } else {
          format!("rgb({r},{g},{b})")
        };
        svg.push_str(&format!(
          "<polygon points=\"{x0:.1},{y0:.1} {x1:.1},{y1:.1} {x2:.1},{y2:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"{seam_stroke}\" stroke-width=\"0.5\"{opacity_attr}/>\n"
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
        "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
      ));
      ei += 1;
    }
  }

  // Draw mesh lines on top of all triangles so they're fully visible
  for line in mesh_lines {
    let (x0, y0) = to_svg(line.projected[0].0, line.projected[0].1);
    let (x1, y1) = to_svg(line.projected[1].0, line.projected[1].1);
    svg.push_str(&format!(
      "<line x1=\"{x0:.1}\" y1=\"{y0:.1}\" x2=\"{x1:.1}\" y2=\"{y1:.1}\" stroke=\"#000000a0\" stroke-width=\"0.5\"/>\n"
    ));
  }

  // Draw axes (ticks, labels) on top of everything
  if show_axes {
    draw_axes_on_box(
      &mut svg,
      camera,
      &to_svg,
      &bounding_box_corners_with_ratios(box_ratios),
      x_range,
      y_range,
      z_range,
      axes_labels,
    );
  }

  svg.push_str("</svg>");
  Ok(svg)
}

/// The 8 corners of the normalized [-1,1]x[-1,1]x[-Z_SCALE,Z_SCALE] bounding
/// box — the default box shape shared by the Plot3D-family renderers.
fn bounding_box_corners() -> [Point3D; 8] {
  bounding_box_corners_with_ratios([1.0, 1.0, Z_SCALE])
}

/// The 8 corners of a box shaped `[-rx,rx]x[-ry,ry]x[-rz,rz]`. Only the
/// ratios *between* `rx`, `ry` and `rz` matter — `ParametricPlot3D`'s
/// `BoxRatios` option passes its three values straight through, and its
/// `Automatic` (the box shaped like the data's own extents, rather than a
/// fixed default shape) passes the data's actual per-axis spread instead.
fn bounding_box_corners_with_ratios(ratios: [f64; 3]) -> [Point3D; 8] {
  let [rx, ry, rz] = ratios;
  [
    Point3D {
      x: -rx,
      y: -ry,
      z: -rz,
    },
    Point3D {
      x: rx,
      y: -ry,
      z: -rz,
    },
    Point3D {
      x: -rx,
      y: ry,
      z: -rz,
    },
    Point3D {
      x: rx,
      y: ry,
      z: -rz,
    },
    Point3D {
      x: -rx,
      y: -ry,
      z: rz,
    },
    Point3D {
      x: rx,
      y: -ry,
      z: rz,
    },
    Point3D {
      x: -rx,
      y: ry,
      z: rz,
    },
    Point3D {
      x: rx,
      y: ry,
      z: rz,
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
  draw_axes_on_box_with_ticks(
    svg,
    camera,
    to_svg,
    corners,
    x_range,
    y_range,
    z_range,
    axes_labels,
    &[None, None, None],
    None,
  );
}

/// Parse a 3D `Ticks` option into one entry per axis: `None` keeps the
/// automatic ticks, `Some(positions)` pins them (`None` → no ticks at all,
/// a list of positions or `{position, label}` pairs → just those).
fn parse_ticks_3d(value: &Expr) -> [Option<Vec<f64>>; 3] {
  let mut out: [Option<Vec<f64>>; 3] = [None, None, None];
  if matches!(value, Expr::Identifier(s) if s == "None") {
    return [Some(Vec::new()), Some(Vec::new()), Some(Vec::new())];
  }
  let Expr::List(axes) = value else {
    return out;
  };
  for (slot, axis) in out.iter_mut().zip(axes.iter()) {
    *slot = match axis {
      Expr::Identifier(s) if s == "None" => Some(Vec::new()),
      Expr::List(items) => Some(
        items
          .iter()
          .filter_map(|item| match item {
            Expr::List(pair) => pair.first().and_then(try_eval_to_f64),
            other => try_eval_to_f64(other),
          })
          .collect(),
      ),
      _ => None,
    };
  }
  out
}

/// `draw_axes_on_box` with per-axis explicit tick positions (`Ticks`).
#[allow(clippy::too_many_arguments)]
fn draw_axes_on_box_with_ticks(
  svg: &mut String,
  camera: &Camera,
  to_svg: &dyn Fn(f64, f64) -> (f64, f64),
  corners: &[Point3D; 8],
  x_range: (f64, f64),
  y_range: (f64, f64),
  z_range: (f64, f64),
  axes_labels: &[Option<String>; 3],
  ticks: &[Option<Vec<f64>>; 3],
  crossing: Option<&[Option<f64>; 3]>,
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

  // An unboxed scene with an explicit `AxesOrigin` draws its three axes
  // crossing at that point, each running the full extent of its range,
  // instead of along the edges of the box.
  let axes: [(Point3D, Point3D, (f64, f64), bool); 3] = match crossing {
    Some(at) => {
      // Data value -> normalized box coordinate along one axis.
      let norm =
        |v: Option<f64>, (lo, hi): (f64, f64), (b0, b1): (f64, f64)| match v {
          Some(v) if (hi - lo).abs() > 1e-15 => {
            b0 + ((v - lo) / (hi - lo)).clamp(0.0, 1.0) * (b1 - b0)
          }
          _ => b0,
        };
      let cross = Point3D {
        x: norm(at[0], x_range, (box_x_lo, box_x_hi)),
        y: norm(at[1], y_range, (box_y_lo, box_y_hi)),
        z: norm(at[2], z_range, (box_z_lo, box_z_hi)),
      };
      [
        (
          Point3D {
            x: box_x_lo,
            ..cross
          },
          Point3D {
            x: box_x_hi,
            ..cross
          },
          x_range,
          false,
        ),
        (
          Point3D {
            y: box_y_lo,
            ..cross
          },
          Point3D {
            y: box_y_hi,
            ..cross
          },
          y_range,
          false,
        ),
        (
          Point3D {
            z: box_z_lo,
            ..cross
          },
          Point3D {
            z: box_z_hi,
            ..cross
          },
          z_range,
          false,
        ),
      ]
    }
    None => axes,
  };

  // Project the box center so we can orient tick labels outward
  let box_center = Point3D {
    x: f64::midpoint(box_x_lo, box_x_hi),
    y: f64::midpoint(box_y_lo, box_y_hi),
    z: f64::midpoint(box_z_lo, box_z_hi),
  };
  let (cx, cy) =
    to_svg(project(box_center, camera).0, project(box_center, camera).1);

  let axis_ticks = |ai: usize, lo: f64, hi: f64| -> Vec<f64> {
    match &ticks[ai] {
      Some(positions) => positions
        .iter()
        .copied()
        .filter(|v| *v >= lo.min(hi) && *v <= lo.max(hi))
        .collect(),
      None => tick_values(lo, hi),
    }
  };

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
            "<line x1=\"{sx0:.1}\" y1=\"{sy0:.1}\" x2=\"{sx1:.1}\" y2=\"{sy1:.1}\" stroke=\"{axis_color}\" stroke-width=\"1\"/>\n"
        ));

    // The axis label sits centred on the axis, clear of the tick labels
    // already sitting there — where Wolfram puts it.
    if let Some(label) = axes_labels[ai].as_deref().filter(|l| !l.is_empty()) {
      let (dx, dy) = (sx1 - sx0, sy1 - sy0);
      let len = (dx * dx + dy * dy).sqrt();
      if len > 1.0 {
        let (mid_x, mid_y) = (f64::midpoint(sx0, sx1), f64::midpoint(sy0, sy1));
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
        let tick_chars = axis_ticks(ai, val_min, val_max)
          .iter()
          .map(|v| format_tick(*v).chars().count())
          .max()
          .unwrap_or(1) as f64;
        let half_w = tick_chars * font_size as f64 * 0.3;
        let half_h = font_size as f64 * 0.5;
        // …plus half of the label's own box, so `offset` can centre it
        // just past them.
        // A near-vertical axis (the z axis) gets its label turned to run
        // alongside it, as Wolfram does, so it only needs a line height
        // of room sideways instead of its whole width.
        let vertical = dy.abs() > 2.0 * dx.abs();
        let own_w = if vertical {
          half_h
        } else {
          label_half_width(label, font_size)
        };
        let reach = TICK_LABEL_OFFSET
          + (perpx * sign).abs() * (half_w + own_w)
          + (perpy * sign).abs() * (half_h + half_h);
        let offset = reach + font_size as f64 * 0.5;
        let (lx, ly) =
          (mid_x + perpx * offset * sign, mid_y + perpy * offset * sign);
        let rotate = if vertical {
          format!(" transform=\"rotate(-90,{lx:.1},{ly:.1})\"")
        } else {
          String::new()
        };
        svg.push_str(&format!(
          "<text x=\"{lx:.1}\" y=\"{ly:.1}\" font-size=\"{font_size}\" fill=\"{axis_color}\" text-anchor=\"middle\" dominant-baseline=\"middle\"{rotate}>{label}</text>\n",
        ));
      }
    }

    // Ticks
    for tick_val in axis_ticks(ai, val_min, val_max) {
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
        let mid_x = f64::midpoint(sx0, sx1);
        let mid_y = f64::midpoint(sy0, sy1);
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
    // A span far below the values' precision (a float-noise extent) has a
    // step that no longer advances the tick; stop instead of looping.
    if tick_val + step == tick_val {
      break;
    }
    tick_val += step;
  }
  ticks
}

// ── VectorPlot3D implementation ─────────────────────────────────────

/// Grid resolution for VectorPlot3D (N x N x N sample points).
const VECTOR3D_GRID: usize = 7;

/// `VectorStyle`'s per-field directives: the color to draw that field's
/// arrows in, and the `Thickness`/`Arrowheads` sizes (fractions of the
/// image width, same convention 2D `Graphics` directives use), each `None`
/// when the directive list doesn't set it (falling back to the automatic
/// default).
#[derive(Clone, Copy, Default)]
struct VecStyleSpec {
  color: Option<(u8, u8, u8)>,
  thickness_frac: Option<f64>,
  arrowhead_frac: Option<f64>,
}

fn parse_vector_style_spec3d(expr: &Expr) -> VecStyleSpec {
  let mut spec = VecStyleSpec::default();
  let directives: Vec<Expr> = match expr {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  for d in &directives {
    if let Some(c) = crate::functions::graphics::parse_color(d) {
      spec.color = Some((
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      ));
      continue;
    }
    if let Expr::FunctionCall { name, args } = d {
      if name == "Thickness" && args.len() == 1 {
        if let Some(t) = try_eval_to_f64(&args[0]) {
          spec.thickness_frac = Some(t);
        }
      } else if name == "Arrowheads"
        && args.len() == 1
        && let Some(a) = try_eval_to_f64(&args[0])
      {
        spec.arrowhead_frac = Some(a.abs());
      }
    }
  }
  spec
}

/// `VectorStyle -> style` (one style shared by every field) or `VectorStyle
/// -> {style1, style2, …}` (one per field, matching a
/// `VectorPlot3D[{field1, field2, …}, …]` overlay's field order).
fn parse_vector_style3d(
  expr: &Expr,
  num_fields: usize,
) -> Vec<Option<VecStyleSpec>> {
  if num_fields > 1
    && let Expr::List(items) = expr
    && items.len() == num_fields
    && items.iter().all(|i| matches!(i, Expr::List(_)))
  {
    return items
      .iter()
      .map(|i| Some(parse_vector_style_spec3d(i)))
      .collect();
  }
  vec![Some(parse_vector_style_spec3d(expr)); num_fields]
}

/// `VectorPoints -> {{x1,y1,z1}, …}`: explicit sample points in place of the
/// automatic grid.
fn parse_vector_points3d(expr: &Expr) -> Option<Vec<(f64, f64, f64)>> {
  let evaluated = evaluate_expr_to_expr(expr).ok()?;
  let Expr::List(items) = &evaluated else {
    return None;
  };
  let mut pts = Vec::with_capacity(items.len());
  for item in items {
    let Expr::List(coords) = item else {
      return None;
    };
    if coords.len() != 3 {
      return None;
    }
    let x = try_eval_to_f64(&coords[0])?;
    let y = try_eval_to_f64(&coords[1])?;
    let z = try_eval_to_f64(&coords[2])?;
    pts.push((x, y, z));
  }
  Some(pts)
}

/// `VectorPoints -> n`: `n` sample points across every axis; returned as the
/// number of grid intervals (`n - 1`).
fn parse_vector_points_count3d(expr: &Expr) -> Option<usize> {
  let n = try_eval_to_f64(&evaluate_expr_to_expr(expr).ok()?)?;
  (n >= 2.0).then(|| n.round() as usize - 1)
}

/// A `ColorFunction` option: a named gradient or an arbitrary function.
enum ColorFn {
  Named(String),
  Expr(Expr),
}

impl ColorFn {
  /// The color for a magnitude `t` scaled to `[0, 1]`.
  fn color(&self, t: f64) -> (u8, u8, u8) {
    match self {
      ColorFn::Named(name) => {
        crate::functions::field_plot::apply_named_color_function(name, t)
      }
      ColorFn::Expr(f) => {
        let call = Expr::CurriedCall {
          func: Box::new(f.clone()),
          args: vec![Expr::Real(t)],
        };
        evaluate_expr_to_expr(&call)
          .ok()
          .and_then(|r| parse_color(&r))
          .map_or((0, 0, 0), |c| {
            (
              (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
              (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
              (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            )
          })
      }
    }
  }
}

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
  let mut vector_points: Option<Vec<(f64, f64, f64)>> = None;
  let mut vector_scale: Option<f64> = None;
  let mut vector_grid: Option<usize> = None;
  let mut color_function: Option<ColorFn> = None;
  let mut vector_styles: Vec<Option<VecStyleSpec>> = vec![None; bodies.len()];

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
        // `VectorPoints -> {{x1,y1,z1}, …}`: sample the field at exactly
        // these points instead of an automatic grid — a Demonstration
        // commonly builds a custom (often non-uniform) sample set with
        // `Table`/`Flatten` to concentrate arrows where the field is
        // interesting.
        "VectorPoints" => {
          if let Some(pts) = parse_vector_points3d(replacement) {
            vector_points = Some(pts);
          } else if let Some(n) = parse_vector_points_count3d(replacement) {
            vector_grid = Some(n);
          }
        }
        // `ColorFunction -> f`: colors a single field's arrows by their
        // magnitude, scaled to 0–1 (a named gradient or a pure function).
        "ColorFunction" => {
          if let Some(s) =
            crate::functions::field_plot::color_function_scheme_name(
              replacement,
            )
          {
            color_function = Some(ColorFn::Named(s));
          } else if !matches!(replacement.as_ref(), Expr::Identifier(v) if v == "Automatic" || v == "None")
          {
            color_function = Some(ColorFn::Expr((**replacement).clone()));
          }
        }
        // `VectorScale -> s` (or `{smax, …}`, of which only the leading
        // `smax` is used here): the longest vector's length as a fraction
        // of the plot's overall width, overriding the automatic
        // one-grid-cell-relative sizing below.
        "VectorScale" => {
          let first = match replacement.as_ref() {
            Expr::List(items) => items.first().cloned(),
            other => Some(other.clone()),
          };
          if let Some(v) = first.and_then(|e| {
            evaluate_expr_to_expr(&e)
              .ok()
              .and_then(|r| try_eval_to_f64(&r))
          }) {
            vector_scale = Some(v);
          }
        }
        // `VectorStyle -> {color, Thickness[t], Arrowheads[a]}` (one style
        // for every field) or `{style1, style2, …}` (one per field, for a
        // `VectorPlot3D[{field1, field2, …}, …]` overlay) — a Demonstration
        // color-codes several overlaid fields (e.g. E, H and Poynting
        // vectors) this way.
        "VectorStyle" => {
          vector_styles = parse_vector_style3d(replacement, bodies.len());
        }
        _ => {}
      }
    }
  }

  let grid_n = vector_grid.unwrap_or(VECTOR3D_GRID);
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

  let mut sample_at =
    |field_idx: usize, body: &Expr, x: f64, y: f64, z: f64| {
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
    };

  if let Some(pts) = &vector_points {
    for (field_idx, body) in bodies.iter().enumerate() {
      for &(x, y, z) in pts {
        sample_at(field_idx, body, x, y, z);
      }
    }
  } else {
    for (field_idx, body) in bodies.iter().enumerate() {
      for i in 0..=grid_n {
        let x = x_min + i as f64 * x_step;
        for j in 0..=grid_n {
          let y = y_min + j as f64 * y_step;
          for k in 0..=grid_n {
            let z = z_min + k as f64 * z_step;
            sample_at(field_idx, body, x, y, z);
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

  // Arrow scale: normalize so that the longest arrow fits roughly half a
  // grid cell, or — when `VectorScale -> s` was given — so the longest
  // vector's full length is `s` times the plot's overall (normalized)
  // width, matching Wolfram's documented meaning of the option.
  let cell_size = (2.0 / grid_n as f64)
    .min(2.0 / grid_n as f64)
    .min(2.0 * Z_SCALE / grid_n as f64);
  let arrow_scale = match vector_scale {
    Some(s) => s * 2.0 / max_mag,
    None => cell_size * 0.4 / max_mag,
  };

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
    /// Line stroke width in px; `None` keeps the built-in default.
    thickness_px: Option<f64>,
    /// Arrowhead length in px (absolute, not relative to the arrow's own
    /// length); `None` keeps the built-in relative sizing.
    arrowhead_px: Option<f64>,
  }

  let camera = Camera::default();
  let mut arrows: Vec<ArrowData> = Vec::with_capacity(samples.len());
  // Each arrow as a data-space `{color, Thickness[…]?, Arrowheads[…]?,
  // Arrow[{p0, p1}]}` primitive group, so a `Show[VectorPlot3D[…],
  // VectorPlot3D[…], Graphics3D[…]]` (a Demonstration overlaying several
  // field arrows on a shared scene) merges them as real Graphics3D
  // primitives sharing one camera/lighting, instead of stacking each
  // call's own independently-projected picture.
  let mut structure_items: Vec<Expr> = Vec::with_capacity(samples.len());

  let num_fields = bodies.len();
  let image_extent = svg_width.max(svg_height) as f64;
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

    let style = vector_styles.get(s.field_idx).copied().flatten();

    // Color: an explicit `VectorStyle` color wins; otherwise PLOT_COLORS
    // for multiple fields, a magnitude gradient for a single one.
    let color = style.and_then(|st| st.color).unwrap_or_else(|| {
      if num_fields > 1 {
        PLOT_COLORS[s.field_idx % PLOT_COLORS.len()]
      } else if let Some(cf) = &color_function {
        cf.color((s.mag / max_mag).clamp(0.0, 1.0))
      } else {
        let t = (s.mag / max_mag).clamp(0.0, 1.0);
        (
          (t * 200.0) as u8 + 50,
          ((1.0 - t) * 150.0) as u8 + 50,
          100_u8,
        )
      }
    });
    let thickness_px = style
      .and_then(|st| st.thickness_frac)
      .map(|t| t * image_extent);
    let arrowhead_px = style
      .and_then(|st| st.arrowhead_frac)
      .map(|a| a * image_extent);

    arrows.push(ArrowData {
      start,
      end,
      depth: depth(center, &camera),
      color,
      thickness_px,
      arrowhead_px,
    });

    // Data-space half-delta: `arrow_scale` was chosen so that, once scaled
    // into normalized space by `sx`/`sy`/`sz`, the arrow reads at the
    // intended fraction of the plot's width — dividing back out those
    // per-axis factors here (`dvx / sx == s.vx * arrow_scale * 0.5`, since
    // `sx = 2 / x_range_d`) gives the same arrow length in data
    // coordinates, for `Graphics3D` to project with its own camera.
    let ddx = s.vx * arrow_scale * 0.5;
    let ddy = s.vy * arrow_scale * 0.5;
    let ddz = s.vz * arrow_scale * 0.5;
    let p0 = Expr::List(
      vec![
        Expr::Real(s.px - ddx),
        Expr::Real(s.py - ddy),
        Expr::Real(s.pz - ddz),
      ]
      .into(),
    );
    let p1 = Expr::List(
      vec![
        Expr::Real(s.px + ddx),
        Expr::Real(s.py + ddy),
        Expr::Real(s.pz + ddz),
      ]
      .into(),
    );
    let mut group: Vec<Expr> = vec![call(
      "RGBColor",
      vec![
        Expr::Real(color.0 as f64 / 255.0),
        Expr::Real(color.1 as f64 / 255.0),
        Expr::Real(color.2 as f64 / 255.0),
      ],
    )];
    if let Some(t) = style.and_then(|st| st.thickness_frac) {
      group.push(call1("Thickness", Expr::Real(t)));
    }
    if let Some(a) = style.and_then(|st| st.arrowhead_frac) {
      group.push(call1("Arrowheads", Expr::Real(a)));
    }
    group.push(call1("Arrow", Expr::List(vec![p0, p1].into())));
    structure_items.push(Expr::List(group.into()));
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
  let p_cx = f64::midpoint(px_min, px_max);
  let p_cy = f64::midpoint(py_min, py_max);

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
        let tm = f64::midpoint(t0, t1);
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
      "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
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
          "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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
        let width = arrow.thickness_px.unwrap_or(3.0);
        svg.push_str(&format!(
          "<line x1=\"{sx0:.1}\" y1=\"{sy0:.1}\" x2=\"{sx1:.1}\" y2=\"{sy1:.1}\" stroke=\"{color_str}\" stroke-width=\"{width}\" stroke-linecap=\"round\"/>\n"
        ));
      } else {
        // Arrow: line + arrowhead
        let width = arrow.thickness_px.unwrap_or(1.2);
        svg.push_str(&format!(
          "<line x1=\"{sx0:.1}\" y1=\"{sy0:.1}\" x2=\"{sx1:.1}\" y2=\"{sy1:.1}\" stroke=\"{color_str}\" stroke-width=\"{width}\"/>\n"
        ));

        // Arrowhead
        let dx = sx1 - sx0;
        let dy = sy1 - sy0;
        let len = (dx * dx + dy * dy).sqrt();
        if len > 2.0 {
          let ux = dx / len;
          let uy = dy / len;
          // An explicit `Arrowheads[a]` size is absolute (a fraction of the
          // image width), not relative to this one arrow's own length.
          let hl = arrow.arrowhead_px.unwrap_or(len * 0.3);
          let hw = hl * 0.4;
          let bx1 = sx1 - ux * hl + (-uy) * hw;
          let by1 = sy1 - uy * hl + ux * hw;
          let bx2 = sx1 - ux * hl - (-uy) * hw;
          let by2 = sy1 - uy * hl - ux * hw;
          svg.push_str(&format!(
            "<polygon points=\"{sx1:.1},{sy1:.1} {bx1:.1},{by1:.1} {bx2:.1},{by2:.1}\" fill=\"{color_str}\"/>\n"
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
        "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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
  let structure = call1("Graphics3D", Expr::List(structure_items.into()));
  Ok(crate::plot3d_result_with_structure(svg, structure))
}

// ── Graphics3D implementation ────────────────────────────────────────

/// Parse a 3D point {x, y, z} from an expression.
fn parse_point3d(expr: &Expr) -> Option<Point3D> {
  // A point is usually already a `{x, y, z}` literal, but it can also be an
  // arithmetic expression that only reduces to one at evaluation time (e.g.
  // an arrow tip computed as `base + vector`) — evaluate first so those
  // aren't silently dropped as "not a point".
  let evaluated = match expr {
    Expr::List(_) => expr.clone(),
    _ => evaluate_expr_to_expr(expr).ok()?,
  };
  if let Expr::List(items) = &evaluated
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
  call1("Line", Expr::List(Vec::new().into()))
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

fn point_in_box(p: &Point3D, bounds: &[(f64, f64); 3]) -> bool {
  let eps = 1e-9;
  [p.x, p.y, p.z]
    .iter()
    .zip(bounds)
    .all(|(v, (lo, hi))| *v >= lo - eps && *v <= hi + eps)
}

/// Clip a polyline to the box an explicit `PlotRange` pins, splitting it
/// where it leaves and re-enters — Wolfram draws nothing outside the range.
fn clip_polyline_to_box(
  seg: &[Point3D],
  bounds: &[(f64, f64); 3],
) -> Vec<Vec<Point3D>> {
  let mut out: Vec<Vec<Point3D>> = Vec::new();
  let mut cur: Vec<Point3D> = Vec::new();
  for w in seg.windows(2) {
    let (a, b) = (w[0], w[1]);
    let p = [a.x, a.y, a.z];
    let v = [b.x - a.x, b.y - a.y, b.z - a.z];
    match clip_line_to_box(p, v, bounds, 0.0) {
      Some((lo, hi)) if lo <= 1.0 => {
        let hi = hi.min(1.0);
        let at = |t: f64| Point3D {
          x: a.x + t * v[0],
          y: a.y + t * v[1],
          z: a.z + t * v[2],
        };
        if lo > 0.0 && !cur.is_empty() {
          out.push(std::mem::take(&mut cur));
        }
        if cur.is_empty() {
          cur.push(at(lo));
        }
        cur.push(at(hi));
        if hi < 1.0 {
          out.push(std::mem::take(&mut cur));
        }
      }
      _ => {
        if !cur.is_empty() {
          out.push(std::mem::take(&mut cur));
        }
      }
    }
  }
  if !cur.is_empty() {
    out.push(cur);
  }
  out.retain(|s| s.len() >= 2);
  out
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
    f64::midpoint(bounds[0].0, bounds[0].1),
    f64::midpoint(bounds[1].0, bounds[1].1),
    f64::midpoint(bounds[2].0, bounds[2].1),
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

/// The convex polygon a plane through `point` with the given `normal` cuts
/// out of the axis-aligned box `bounds`. Used by `Area[RegionIntersection[
/// Cube[…]/Cuboid[…], ImplicitRegion[…]]]`, where the `ImplicitRegion` is a
/// single linear equation (a plane) — Wolfram leaves that intersection as an
/// unevaluated `BooleanRegion`, so its `Area` is computed geometrically
/// instead of symbolically. Builds an arbitrary in-plane basis (`u`, `w`)
/// from `normal` and delegates to [`clip_plane_to_box`], the same clipping
/// [`InfinitePlane`](unbounded_3d_to_primitive) rendering uses.
pub(crate) fn plane_polygon_in_box(
  normal: [f64; 3],
  point: [f64; 3],
  bounds: [(f64, f64); 3],
) -> Option<Vec<[f64; 3]>> {
  if v_len(normal) < 1e-12 {
    return None;
  }
  // Cross with whichever axis is least aligned with the normal, for a
  // numerically stable in-plane basis.
  let axis = if normal[0].abs() <= normal[1].abs()
    && normal[0].abs() <= normal[2].abs()
  {
    [1.0, 0.0, 0.0]
  } else if normal[1].abs() <= normal[2].abs() {
    [0.0, 1.0, 0.0]
  } else {
    [0.0, 0.0, 1.0]
  };
  let u = v_cross(normal, axis);
  let w = v_cross(normal, u);
  // `clip_plane_to_box` returns `None` only because the plane misses the
  // box (or clips down to a sliver) — a legitimate empty intersection, not
  // a computation failure, now that `normal` is known non-degenerate.
  Some(clip_plane_to_box(point, u, w, &bounds, None).unwrap_or_default())
}

/// The area of a simple planar polygon given as an ordered list of 3-D
/// vertices, via Newell's method: summing `Vi × Vi+1` gives twice the area
/// along the polygon's own normal direction, so its length halved is the
/// area regardless of which way the polygon winds or which plane it lies in.
pub(crate) fn polygon3d_area(poly: &[[f64; 3]]) -> f64 {
  if poly.len() < 3 {
    return 0.0;
  }
  let mut normal = [0.0; 3];
  for i in 0..poly.len() {
    let a = poly[i];
    let b = poly[(i + 1) % poly.len()];
    normal = v_add(normal, v_cross(a, b));
  }
  0.5 * v_len(normal)
}

/// The area of the intersection between a disk (`center`, `radius`) and a
/// convex polygon, both given in the same 2-D coordinate system. Used by
/// `Area[RegionIntersection[Ball[…], planar-region]]`: the planar region's
/// polygon and the ball's cross-sectional disk are first projected into the
/// plane's own 2-D basis (see callers), then this computes the disk clipped
/// to the polygon by summing, over each polygon edge, the signed area the
/// disk contributes to the triangle `(center, edge start, edge end)`.
pub(crate) fn disk_convex_polygon_area(
  center: [f64; 2],
  radius: f64,
  poly: &[[f64; 2]],
) -> f64 {
  if poly.len() < 3 || radius <= 0.0 {
    return 0.0;
  }
  let mut total = 0.0;
  for i in 0..poly.len() {
    let a = [poly[i][0] - center[0], poly[i][1] - center[1]];
    let b = [
      poly[(i + 1) % poly.len()][0] - center[0],
      poly[(i + 1) % poly.len()][1] - center[1],
    ];
    total += disk_triangle_signed_area(a, b, radius);
  }
  total.abs()
}

/// The signed area shared by the disk of `radius` centered at the origin and
/// the triangle `(origin, a, b)`. Standard circle/polygon-clipping building
/// block: handles both points inside the disk (plain triangle), both outside
/// (either a miss, contributing nothing, or a chord, contributing the two
/// circular-sector wedges either side of the chord plus the chord's own
/// triangle), and one of each (the segment crosses the disk boundary once).
fn disk_triangle_signed_area(a: [f64; 2], b: [f64; 2], radius: f64) -> f64 {
  let cross = a[0] * b[1] - a[1] * b[0];
  if cross.abs() < 1e-15 {
    return 0.0;
  }
  let len = |p: [f64; 2]| (p[0] * p[0] + p[1] * p[1]).sqrt();
  // The angle between `p` and `q` by direction alone — dividing by their own
  // lengths rather than `radius^2` — so this also works for the far corners
  // `a`/`b`, not just points already on the circle.
  let sector_area = |p: [f64; 2], q: [f64; 2]| -> f64 {
    let denom = len(p) * len(q);
    if denom < 1e-15 {
      return 0.0;
    }
    let angle = (p[0] * q[0] + p[1] * q[1]) / denom;
    let angle = angle.clamp(-1.0, 1.0).acos();
    let signed = if p[0] * q[1] - p[1] * q[0] < 0.0 {
      -angle
    } else {
      angle
    };
    0.5 * radius * radius * signed
  };
  // The (up to two) points where the infinite line through a, b crosses the
  // circle of `radius`, as parameters t along a + t*(b - a).
  let circle_line_params = |a: [f64; 2], b: [f64; 2]| -> Vec<f64> {
    let d = [b[0] - a[0], b[1] - a[1]];
    let aa = d[0] * d[0] + d[1] * d[1];
    let bb = 2.0 * (a[0] * d[0] + a[1] * d[1]);
    let cc = a[0] * a[0] + a[1] * a[1] - radius * radius;
    let disc = bb * bb - 4.0 * aa * cc;
    if aa < 1e-15 || disc < 0.0 {
      return Vec::new();
    }
    let sq = disc.sqrt();
    vec![(-bb - sq) / (2.0 * aa), (-bb + sq) / (2.0 * aa)]
  };
  let da = len(a);
  let db = len(b);
  if da <= radius && db <= radius {
    return 0.5 * cross;
  }
  if da >= radius && db >= radius {
    let params: Vec<f64> = circle_line_params(a, b)
      .into_iter()
      .filter(|&t| t > 1e-12 && t < 1.0 - 1e-12)
      .collect();
    if params.len() < 2 {
      // The chord (if any) falls outside the segment: the whole wedge is
      // either entirely inside or entirely outside the disk.
      return sector_area(a, b);
    }
    let (t1, t2) = (params[0].min(params[1]), params[0].max(params[1]));
    let p1 = [a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1])];
    let p2 = [a[0] + t2 * (b[0] - a[0]), a[1] + t2 * (b[1] - a[1])];
    return sector_area(a, p1)
      + 0.5 * (p1[0] * p2[1] - p1[1] * p2[0])
      + sector_area(p2, b);
  }
  // Exactly one endpoint inside: the segment crosses the boundary once.
  let t = circle_line_params(a, b)
    .into_iter()
    .find(|&t| (0.0..=1.0).contains(&t))
    .unwrap_or(0.0);
  let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
  if da < db {
    0.5 * (a[0] * p[1] - a[1] * p[0]) + sector_area(p, b)
  } else {
    sector_area(a, p) + 0.5 * (p[0] * b[1] - p[1] * b[0])
  }
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
    let mid = f64::midpoint(lo, hi);
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
      Some(call1(
        "Line",
        Expr::List(
          vec![
            point3d_expr(v_add(p, v_scale(v, lo))),
            point3d_expr(v_add(p, v_scale(v, hi))),
          ]
          .into(),
        ),
      ))
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
  call1(
    "Polygon",
    Expr::List(
      poly
        .iter()
        .map(|p| point3d_expr(*p))
        .collect::<Vec<_>>()
        .into(),
    ),
  )
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
  /// The outline width a thickness inside `EdgeForm[…]` asks for.
  edge_width: Option<f64>,
  /// `Specularity[colour, exponent]`: the highlight a shiny surface throws
  /// back at the viewer, as `(colour, exponent)`. `None` is the default
  /// matte surface. A larger exponent tightens the highlight.
  specular: Option<((u8, u8, u8), f64)>,
  /// `Glow[colour]`: light the surface emits on its own, added on top of
  /// the lit colour (`Glow[White]` renders a face plain white whatever its
  /// angle to the lamp). `None` is a non-emissive surface.
  glow: Option<(u8, u8, u8)>,
  /// `Dashing[{d1, d2, ...}]`/`Dashed`/`Dotted`/`DotDashed` for `Line3D`
  /// primitives, in the same units `dash_size_to_f64` uses for 2D: negative
  /// is literal pixels (a named size), positive is a fraction of the image
  /// width. `None` draws a solid stroke.
  dashing: Option<Vec<f64>>,
  /// `PointSize[…]`/`AbsolutePointSize[…]` for `Point3D` primitives, in the
  /// same units the 2D renderer's `point_size` uses: positive is a fraction
  /// of the image width, negative (stored) is an absolute size in printer's
  /// points. `None` draws the default 3px dot.
  point_size: Option<f64>,
  /// `Texture[image]`: surfaces coloured by sampling this image at each
  /// facet's texture coordinates (see [`texture_color`]).
  texture: Option<std::sync::Arc<image::RgbImage>>,
  /// `VertexTextureCoordinates` of the polygon being added: one `(u, v)`
  /// per corner, placing `texture` on it.
  texture_coords: Option<std::sync::Arc<Vec<(f64, f64)>>>,
  /// The font size `Style[…, size]` gives the `Text` labels inside it.
  /// `None` leaves the default 12.
  font_size: Option<f64>,
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
      edge_width: None,
      specular: None,
      glow: None,
      dashing: None,
      point_size: None,
      texture: None,
      texture_coords: None,
      font_size: None,
    }
  }
}

#[derive(Clone)]
enum Primitive3D {
  Sphere {
    center: Point3D,
    radius: f64,
    /// `Sphere[c, Scaled[r]]`: `radius` holds the fraction `r` of the plot
    /// range until `resolve_scaled_sphere_radii` turns it into a length.
    scaled: bool,
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
    /// A label laid out as a table (a `Column` holding a `Grid`) is a
    /// picture rather than a text run: its SVG and natural size, drawn
    /// centred on the label's anchor.
    picture: Option<(String, f64, f64)>,
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

  // {Red, Thick, Opacity[.4], …} – apply each element as a sub-directive.
  // This precedes the colour parse, which would otherwise read a list of
  // directives as its one colour and drop the rest (the opacity).
  if let Expr::List(items) = expr {
    for it in items {
      apply_3d_directive(it, style);
    }
    return true;
  }

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
      "Dashed" => {
        style.dashing = Some(vec![
          -crate::functions::graphics::SMALL_DASH_PX,
          -crate::functions::graphics::SMALL_DASH_PX,
        ]);
        return true;
      }
      "Dotted" => {
        style.dashing =
          Some(vec![0.0, -crate::functions::graphics::SMALL_DASH_PX]);
        return true;
      }
      "DotDashed" => {
        style.dashing = Some(vec![
          0.0,
          -crate::functions::graphics::SMALL_DASH_PX,
          -crate::functions::graphics::SMALL_DASH_PX,
          -crate::functions::graphics::SMALL_DASH_PX,
        ]);
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
      "PointSize" if args.len() == 1 => {
        if let Some(s) =
          crate::functions::graphics::symbolic_point_size(&args[0])
        {
          style.point_size = Some(s);
        } else if let Some(s) = expr_to_f64(&args[0]) {
          style.point_size = Some(s);
        }
        return true;
      }
      "AbsolutePointSize" if args.len() == 1 => {
        if let Some(s) =
          crate::functions::graphics::symbolic_point_size(&args[0])
        {
          style.point_size = Some(s);
        } else if let Some(s) = expr_to_f64(&args[0]) {
          style.point_size = Some(-s);
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
      // `Glow[colour]` makes the surfaces that follow emit light of their
      // own; like `Specularity` its colour never repaints the face.
      "Glow" if args.len() == 1 => {
        style.glow = parse_glow(&args[0]);
        return true;
      }
      // `Texture[image]` paints the surfaces that follow with the image.
      "Texture" if args.len() == 1 => {
        let img = crate::functions::image_ast::texture_source_image(&args[0]);
        if let Some(Expr::Image {
          width,
          height,
          channels,
          ref data,
          ..
        }) = img
        {
          style.texture = Some(std::sync::Arc::new(
            crate::functions::image_ast::expr_to_dynamic_image(
              width, height, channels, data,
            )
            .to_rgb8(),
          ));
        }
        return true;
      }
      // CapForm[None] leaves an open surface's ends open; every named form
      // ("Butt", "Square", "Round") closes them, as does the default.
      "CapForm" if args.len() == 1 => {
        style.capped = !matches!(&args[0], Expr::Identifier(s) if s == "None");
        return true;
      }
      // `EdgeForm[]` (and `EdgeForm[None]`) asks for faces with no outline;
      // any other form keeps the default edge. A colour or thickness among
      // the arguments — `EdgeForm[Black]`, `EdgeForm[{Thick, Red}]`,
      // `EdgeForm[Directive[Red, AbsoluteThickness[2]]]` — also styles the
      // outline.
      "EdgeForm" => {
        style.edges = !(args.is_empty()
          || matches!(args.first(), Some(Expr::Identifier(s)) if s == "None"));
        let mut edge = StyleState3D::default();
        if style.edges {
          for a in args {
            apply_3d_directive(a, &mut edge);
          }
        }
        style.edge_color = edge.color;
        style.edge_width = edge.thickness;
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
      "Dashing" if !args.is_empty() => {
        if let Some(d) = parse_3d_dashing(&args[0], false) {
          style.dashing = d;
        }
        return true;
      }
      "AbsoluteDashing" if !args.is_empty() => {
        if let Some(d) = parse_3d_dashing(&args[0], true) {
          style.dashing = d;
        }
        return true;
      }
      _ => {}
    },
    _ => {}
  }

  false
}

/// Parse a `Dashing[…]`/`AbsoluteDashing[…]` first argument into the dash
/// pattern to install, mirroring 2D's `apply_directive` handling of the
/// same directives. `absolute` marks a plain number as literal pixels
/// (`AbsoluteDashing`) rather than a fraction of the image width
/// (`Dashing`), matching `AbsoluteThickness`'s relationship to
/// `Thickness`. Returns `None` when the argument names no update at all
/// (so the caller leaves the current dashing alone), `Some(None)` for
/// `None`/`{}` (back to a solid stroke), and `Some(Some(pattern))`
/// otherwise.
fn parse_3d_dashing(arg: &Expr, absolute: bool) -> Option<Option<Vec<f64>>> {
  use crate::functions::graphics::dash_size_to_f64;
  use crate::functions::math_ast::expr_to_f64;
  match arg {
    Expr::Identifier(s) if s == "None" => Some(None),
    Expr::List(items) if items.is_empty() => Some(None),
    Expr::List(items) => {
      let dashes: Vec<f64> = items
        .iter()
        .filter_map(|e| {
          dash_size_to_f64(e).or_else(|| {
            expr_to_f64(e).map(|d| if absolute { -d.abs() } else { d })
          })
        })
        .collect();
      if dashes.is_empty() {
        None
      } else {
        Some(Some(dashes))
      }
    }
    _ => dash_size_to_f64(arg)
      .or_else(|| expr_to_f64(arg).map(|d| if absolute { -d.abs() } else { d }))
      .map(|d| Some(vec![d, d])),
  }
}

/// `stroke-dasharray`/`stroke-dashoffset` for a `Line3D` edge. Mirrors 2D
/// graphics' dash-length convention (negative = literal pixels, positive =
/// a fraction of the image width, zero = a one-pixel dot) and adds a
/// `stroke-dashoffset` so the pattern stays continuous across a line's
/// depth-sorted subdivisions (see `LINE_SUBDIVISIONS` above) instead of
/// restarting at each one.
fn dash3d_attr(
  dashing: Option<&Vec<f64>>,
  svg_w: f64,
  offset_px: f64,
) -> String {
  let Some(dashes) = dashing else {
    return String::new();
  };
  let px: Vec<String> = dashes
    .iter()
    .map(|d| {
      let px = if *d < 0.0 { -*d } else { *d * svg_w };
      format!("{:.1}", if px <= 0.0 { 1.0 } else { px })
    })
    .collect();
  format!(
    " stroke-dasharray=\"{}\" stroke-dashoffset=\"{:.1}\"",
    px.join(","),
    offset_px
  )
}

/// Read a `Glow[…]` colour (or grey level) into the emitted RGB; black emits
/// nothing and so is `None`.
fn parse_glow(arg: &Expr) -> Option<(u8, u8, u8)> {
  use crate::functions::graphics::parse_color;
  use crate::functions::math_ast::expr_to_f64;

  let color = if let Some(c) = parse_color(arg) {
    c
  } else {
    let g = expr_to_f64(arg)?;
    crate::functions::graphics::Color::new(g, g, g)
  };
  let rgb = (
    (color.r.clamp(0.0, 1.0) * 255.0).round() as u8,
    (color.g.clamp(0.0, 1.0) * 255.0).round() as u8,
    (color.b.clamp(0.0, 1.0) * 255.0).round() as u8,
  );
  (rgb != (0, 0, 0)).then_some(rgb)
}

/// Add a `Glow` emission to an already lit colour.
fn add_glow(color: (u8, u8, u8), glow: Option<(u8, u8, u8)>) -> (u8, u8, u8) {
  match glow {
    Some(g) => (
      color.0.saturating_add(g.0),
      color.1.saturating_add(g.1),
      color.2.saturating_add(g.2),
    ),
    None => color,
  }
}

/// Read a `Specularity[…]` argument list into `(highlight colour, exponent)`.
/// The reflectance may be a colour (`Specularity[White, 10]`) or a plain
/// number standing for that grey level (`Specularity[.5]`); the exponent
/// defaults to Wolfram's `1`. A reflectance of zero (black) means a matte
/// surface, i.e. no highlight at all.
fn parse_specularity(args: &[Expr]) -> Option<((u8, u8, u8), f64)> {
  use crate::functions::graphics::parse_color;
  use crate::functions::math_ast::expr_to_f64;

  let color = if let Some(c) = parse_color(&args[0]) {
    c
  } else {
    let g = expr_to_f64(&args[0])?;
    crate::functions::graphics::Color::new(g, g, g)
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

/// `ColorFunction -> Function[{x, y, z}, …]` of a `Plot3D`/`RegionPlot3D`:
/// the style (colour and, for `Directive[Opacity[o], colour]`, opacity) it
/// gives the facet at a point. Named schemes are not handled here.
#[derive(Default)]
struct FacetColorFunction {
  function: Option<Expr>,
  /// `ColorFunctionScaling -> False` hands the function raw coordinates
  /// instead of ones rescaled to 0..1 over the plotted range.
  unscaled: bool,
}

impl FacetColorFunction {
  fn read_option(&mut self, pattern: &Expr, replacement: &Expr) {
    match pattern {
      Expr::Identifier(name) if name == "ColorFunction" => {
        self.function = matches!(replacement,
          Expr::FunctionCall { name, .. } if name == "Function")
        .then(|| replacement.clone());
      }
      Expr::Identifier(name) if name == "ColorFunctionScaling" => {
        self.unscaled =
          matches!(replacement, Expr::Identifier(n) if n == "False");
      }
      _ => {}
    }
  }

  /// The style at `point`, given the `(lo, hi)` range of each axis.
  fn style_at(
    &self,
    point: [f64; 3],
    ranges: [(f64, f64); 3],
  ) -> Option<StyleState3D> {
    let function = self.function.as_ref()?;
    let args = (0..3)
      .map(|k| {
        let (lo, hi) = ranges[k];
        let t = if self.unscaled || (hi - lo).abs() < 1e-15 {
          point[k]
        } else {
          (point[k] - lo) / (hi - lo)
        };
        Expr::Real(t)
      })
      .collect();
    let value = evaluate_expr_to_expr(&Expr::CurriedCall {
      func: Box::new(function.clone()),
      args,
    })
    .ok()?;
    let mut style = StyleState3D::default();
    apply_3d_directive(&value, &mut style);
    (style.color.is_some() || style.opacity < 1.0).then_some(style)
  }
}

/// Resolve which entry of a `PlotStyle` style list a surface uses, cycling
/// through fewer styles than surfaces (`PlotStyle -> {Red, Blue}` on three
/// surfaces gives Red, Blue, Red). `None` — an absent or empty list — means
/// `PlotStyle` was not given, so callers keep their default colouring.
fn plot_style_for_surface(
  styles: &[StyleState3D],
  surface_idx: usize,
) -> Option<&StyleState3D> {
  if styles.is_empty() {
    None
  } else {
    Some(&styles[surface_idx % styles.len()])
  }
}

/// The `Opacity[…]` a translucent `PlotStyle` gives a surface, placed at the
/// head of its symbolic `GraphicsComplex` so the surface stays translucent
/// when `Show` re-renders it together with other graphics.
fn surface_opacity_directive(
  styles: &[StyleState3D],
  surface_idx: usize,
) -> Option<Expr> {
  plot_style_for_surface(styles, surface_idx)
    .filter(|style| style.opacity < 1.0)
    .map(|style| call1("Opacity", Expr::Real(style.opacity)))
}

/// Split a `PlotStyle` option value into one style item per plotted
/// object (a surface or a curve), mirroring
/// [`crate::functions::plot::parse_plot_style`]'s disambiguation for 2D
/// `Plot`: with more than one object, a flat list of colours/directives is
/// read as one style per object (`PlotStyle -> {Red, Blue}`); a single
/// style — one bare directive, or a list that mixes directives for a
/// single style, like `{RGBColor[...], Opacity[.5]}` — applies to every
/// object alike, so the whole value comes back as the sole item.
fn plot_style_items(replacement: &Expr, num_objects: usize) -> Vec<Expr> {
  use crate::functions::graphics::parse_color;

  let val =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  match &val {
    Expr::List(list_items) if num_objects > 1 => {
      let per_object = list_items.iter().any(|item| {
        matches!(item, Expr::FunctionCall { name, .. } if name == "Directive")
          || matches!(item, Expr::List(_))
          || parse_color(item).is_some()
      });
      if per_object {
        list_items.to_vec()
      } else {
        vec![val]
      }
    }
    _ => vec![val],
  }
}

/// Parse a `PlotStyle` option value into one style per surface, cycling
/// through fewer styles than surfaces. An empty result means `PlotStyle`
/// was not given (or produced nothing applicable), so callers keep drawing
/// the height-based rainbow default.
fn parse_plot_style_3d(
  replacement: &Expr,
  num_surfaces: usize,
) -> Vec<StyleState3D> {
  plot_style_items(replacement, num_surfaces)
    .into_iter()
    .map(|item| {
      let mut style = StyleState3D::default();
      apply_3d_directive(&item, &mut style);
      style
    })
    .collect()
}

/// Shade one triangle facet, folding in a `PlotStyle` override:
/// `default_color` is the plot's ordinary colour for this facet (height-
/// based, as today) and is used unchanged — through the same diffuse +
/// ambient lighting as before — when `style` is `None`, so a plot without
/// `PlotStyle` renders byte-identically to before this existed. When a
/// style is given, its own colour (falling back to `default_color` if the
/// style set no colour of its own, e.g. `PlotStyle -> Opacity[.5]`) is
/// shaded instead, with any `Specularity` the style carries layered on top,
/// and the style's opacity is returned alongside the colour for the
/// triangle to draw translucently. `PlotStyle -> FaceForm[front, back]`
/// swaps in the style's `back_color` for a facet turned away from the
/// viewer, the same test `Graphics3D`'s own polygon shading uses.
fn shade_facet(
  default_color: (u8, u8, u8),
  style: Option<&StyleState3D>,
  normal: [f64; 3],
  view_dir: [f64; 3],
) -> ((u8, u8, u8), f64) {
  match style {
    Some(s) => {
      let facing = normal[0] * view_dir[0]
        + normal[1] * view_dir[1]
        + normal[2] * view_dir[2];
      let base = match s.back_color {
        Some(back) if facing < 0.0 => back,
        _ => s.color.unwrap_or(default_color),
      };
      (
        add_glow(
          apply_lighting_specular(base, normal, s.specular, view_dir),
          s.glow,
        ),
        s.opacity,
      )
    }
    None => (apply_lighting(default_color, normal), 1.0),
  }
}

/// The first `Texture[image]` inside a style expression.
fn find_texture_expr(expr: &Expr) -> Option<Expr> {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Texture" && args.len() == 1 =>
    {
      Some(expr.clone())
    }
    Expr::FunctionCall { args, .. } => args.iter().find_map(find_texture_expr),
    Expr::List(items) => items.iter().find_map(find_texture_expr),
    _ => None,
  }
}

/// `VertexTextureCoordinates -> {{u, v}, …}` among a `Polygon`'s options.
fn parse_texture_coords3d(opts: &[Expr]) -> Option<Vec<(f64, f64)>> {
  opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(k) if k == "VertexTextureCoordinates") =>
    {
      let Expr::List(items) = replacement.as_ref() else {
        return None;
      };
      items
        .iter()
        .map(|p| match p {
          Expr::List(xy) if xy.len() == 2 => {
            Some((
              crate::functions::math_ast::expr_to_f64(&xy[0])?,
              crate::functions::math_ast::expr_to_f64(&xy[1])?,
            ))
          }
          _ => None,
        })
        .collect()
    }
    _ => None,
  })
}

/// The texture of triangle `i` in the fan triangulation of a textured,
/// convex polygon: corners `0`, `i + 1`, `i + 2` of the polygon.
fn fan_texture(
  prim: &Primitive3D,
  style: &StyleState3D,
  i: usize,
  tri_count: usize,
) -> Option<TriangleTexture> {
  let Primitive3D::Polygon3D { points, holes, .. } = prim else {
    return None;
  };
  let (image, coords) =
    (style.texture.as_ref()?, style.texture_coords.as_ref()?);
  if !holes.is_empty()
    || coords.len() != points.len()
    || tri_count + 2 != points.len()
    || !is_convex_polygon3d(points)
  {
    return None;
  }
  Some(TriangleTexture {
    image: image.clone(),
    uv: [coords[0], coords[i + 1], coords[i + 2]],
  })
}

/// Emit a triangle filled with its texture: the image clipped to the
/// triangle and placed by the affine map from texture space to the screen.
fn emit_textured_triangle(
  svg: &mut String,
  tex: &TriangleTexture,
  corners: [(f64, f64); 3],
  ids: &mut Vec<(*const image::RgbImage, String)>,
) -> bool {
  use base64::Engine;
  let (w, h) = (tex.image.width() as f64, tex.image.height() as f64);
  let s: Vec<(f64, f64)> = tex
    .uv
    .iter()
    .map(|&(u, v)| (u * w, (1.0 - v) * h))
    .collect();
  let det = (s[1].0 - s[0].0) * (s[2].1 - s[0].1)
    - (s[2].0 - s[0].0) * (s[1].1 - s[0].1);
  if det.abs() < 1e-9 {
    return false;
  }
  let solve = |d0: f64, d1: f64, d2: f64| {
    let p =
      ((d1 - d0) * (s[2].1 - s[0].1) - (d2 - d0) * (s[1].1 - s[0].1)) / det;
    let q =
      ((d2 - d0) * (s[1].0 - s[0].0) - (d1 - d0) * (s[2].0 - s[0].0)) / det;
    (p, q, d0 - p * s[0].0 - q * s[0].1)
  };
  let (a, c, e) = solve(corners[0].0, corners[1].0, corners[2].0);
  let (b, d, f) = solve(corners[0].1, corners[1].1, corners[2].1);
  let key = std::sync::Arc::as_ptr(&tex.image);
  let id = if let Some((_, id)) = ids.iter().find(|(k, _)| *k == key) {
    id.clone()
  } else {
    let mut png = Vec::new();
    if image::DynamicImage::ImageRgb8((*tex.image).clone())
      .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
      .is_err()
    {
      return false;
    }
    let id = format!("tex{}", ids.len());
    svg.push_str(&format!(
      "<defs><image id=\"{id}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"none\" href=\"data:image/png;base64,{}\"/></defs>\n",
      tex.image.width(),
      tex.image.height(),
      base64::engine::general_purpose::STANDARD.encode(&png)
    ));
    ids.push((key, id.clone()));
    id
  };
  let clip = format!("{id}c{}", svg.len());
  svg.push_str(&format!(
    "<clipPath id=\"{clip}\"><polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\"/></clipPath>\n<g clip-path=\"url(#{clip})\"><use href=\"#{id}\" transform=\"matrix({a:.6} {b:.6} {c:.6} {d:.6} {e:.4} {f:.4})\"/></g>\n",
    corners[0].0, corners[0].1, corners[1].0, corners[1].1, corners[2].0, corners[2].1
  ));
  true
}

/// The colour of `texture` at texture coordinates `(u, v)`: `(0, 0)` is the
/// bottom-left corner and the image repeats outside the unit square.
fn texture_color(
  texture: &image::RgbImage,
  u: f64,
  v: f64,
) -> Option<(u8, u8, u8)> {
  if !u.is_finite() || !v.is_finite() {
    return None;
  }
  let (w, h) = (texture.width(), texture.height());
  if w == 0 || h == 0 {
    return None;
  }
  let x = ((u.rem_euclid(1.0) * w as f64) as u32).min(w - 1);
  let y = (((1.0 - v.rem_euclid(1.0)) * h as f64) as u32).min(h - 1);
  let px = texture.get_pixel(x, y);
  Some((px[0], px[1], px[2]))
}

/// The linear part of a rotation by `angle` around the axis direction `w`
/// (Rodrigues' formula), `None` for a degenerate (zero-length) axis. Shared
/// with [`crate::functions::polyhedron_operations`], which rotates raw
/// points the same way `Rotate` rotates rendered primitives.
pub(crate) fn rotation_matrix(
  angle: f64,
  w: [f64; 3],
) -> Option<[[f64; 3]; 3]> {
  let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
  if !len.is_finite() || len < 1e-12 {
    return None;
  }
  let (ux, uy, uz) = (w[0] / len, w[1] / len, w[2] / len);
  let (s, c) = angle.sin_cos();
  let ic = 1.0 - c;
  Some([
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
  ])
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

  /// Is the linear part diagonal (no rotation/shear component)? Such a
  /// transform — translation composed with a per-axis scale, including
  /// axis flips — maps an axis-aligned box to another axis-aligned box, so
  /// `Cuboid` can stay a `Cuboid` under it. Anything else (any rotation)
  /// tilts the box's faces off the coordinate planes, which `Cuboid`'s
  /// two-corner representation cannot express — see `transform_primitive3d`.
  fn is_axis_aligned(&self) -> bool {
    let tol = 1e-9
      * self
        .m
        .iter()
        .flatten()
        .fold(1.0_f64, |acc, v| acc.max(v.abs()));
    self.m[0][1].abs() <= tol
      && self.m[0][2].abs() <= tol
      && self.m[1][0].abs() <= tol
      && self.m[1][2].abs() <= tol
      && self.m[2][0].abs() <= tol
      && self.m[2][1].abs() <= tol
  }

  fn translation(v: [f64; 3]) -> Self {
    Self {
      m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      t: v,
    }
  }

  /// Rotation by `angle` around the axis direction `w` through `anchor`
  /// (Rodrigues' formula).
  fn rotation(angle: f64, w: [f64; 3], anchor: [f64; 3]) -> Option<Self> {
    let m = rotation_matrix(angle, w)?;
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
    Some(Self { m, t })
  }

  fn scaling(factors: [f64; 3], center: [f64; 3]) -> Self {
    Self {
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
/// `Dynamic[expr, …]` in an option position (e.g. `ViewPoint ->
/// Dynamic[var]` inside a `Manipulate`) stands for the current value of
/// `expr` — Dynamic is HoldFirst, so the generic evaluator leaves it
/// symbolic. Release the hold before evaluating so the option reads the
/// control's value instead of silently falling back to the default.
fn strip_dynamic(expr: &Expr) -> &Expr {
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Dynamic" && !args.is_empty() =>
    {
      &args[0]
    }
    _ => expr,
  }
}

/// A `ViewPoint` containing infinite components, e.g. `{0, 0, Infinity}`
/// (the same as `Above`). The infinite components decide the direction;
/// finite ones vanish in the limit. Returns a vector of length 2 along that
/// direction, or `None` when no component is infinite.
fn eval_view_direction(expr: &Expr) -> Option<[f64; 3]> {
  let Expr::List(items) = &evaluate_expr_to_expr(strip_dynamic(expr)).ok()?
  else {
    return None;
  };
  if items.len() != 3 {
    return None;
  }
  let mut dir = [0.0; 3];
  let mut any_infinite = false;
  for (slot, item) in dir.iter_mut().zip(items.iter()) {
    match crate::functions::graphics::directed_infinity_glyph(item) {
      Some(glyph) => {
        *slot = if glyph.starts_with('-') { -1.0 } else { 1.0 };
        any_infinite = true;
      }
      None => {
        try_eval_to_f64(item)?;
      }
    }
  }
  if !any_infinite {
    return None;
  }
  let len = dir.iter().map(|c| c * c).sum::<f64>().sqrt();
  Some(dir.map(|c| 2.0 * c / len))
}

fn eval_vec3(expr: &Expr) -> Option<[f64; 3]> {
  let Expr::List(items) = &evaluate_expr_to_expr(strip_dynamic(expr)).ok()?
  else {
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

/// The affine transforms a 3D `GeometricTransformation` second argument
/// names: the `{matrix, vector}` pair a `TransformationFunction` normalizes
/// to, a bare 3×3 matrix, the homogeneous 4×4 `TransformationFunction[…]`,
/// or a list of any of those (one transformed copy per entry).
fn parse_3d_affine_transforms(expr: &Expr) -> Vec<Affine3> {
  fn matrix3(expr: &Expr) -> Option<[[f64; 3]; 3]> {
    let Expr::List(rows) = expr else { return None };
    if rows.len() != 3 {
      return None;
    }
    let mut m = [[0.0; 3]; 3];
    for (row, slot) in rows.iter().zip(m.iter_mut()) {
      *slot = eval_vec3(row)?;
    }
    Some(m)
  }
  fn homogeneous(expr: &Expr) -> Option<Affine3> {
    let Expr::List(rows) = expr else { return None };
    if rows.len() != 4 {
      return None;
    }
    let mut m = [[0.0; 3]; 3];
    let mut t = [0.0; 3];
    for (i, row) in rows.iter().take(3).enumerate() {
      let Expr::List(cells) = row else { return None };
      if cells.len() != 4 {
        return None;
      }
      for j in 0..3 {
        m[i][j] = try_eval_to_f64(&cells[j])?;
      }
      t[i] = try_eval_to_f64(&cells[3])?;
    }
    Some(Affine3 { m, t })
  }
  fn single(expr: &Expr) -> Option<Affine3> {
    if let Expr::FunctionCall { name, args } = expr
      && name == "TransformationFunction"
      && args.len() == 1
    {
      return homogeneous(&args[0]);
    }
    if let Expr::List(items) = expr
      && items.len() == 2
      && let Some(m) = matrix3(&items[0])
      && let Some(t) = eval_vec3(&items[1])
    {
      return Some(Affine3 { m, t });
    }
    matrix3(expr).map(|m| Affine3 { m, t: [0.0; 3] })
  }
  let expr = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  if let Some(one) = single(&expr) {
    return vec![one];
  }
  match &expr {
    Expr::List(items) => items.iter().filter_map(single).collect(),
    _ => Vec::new(),
  }
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
      f64::midpoint(lo, hi)
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
    Primitive3D::Cylinder {
      p1,
      p2,
      radius,
      style,
    } => Some(tessellate_cylinder(
      p1,
      p2,
      *radius,
      style.capped,
      MAX_SIDE_SUBDIVISIONS,
    )),
    Primitive3D::Cone {
      p1,
      p2,
      radius,
      style,
    } => Some(tessellate_cone(p1, p2, *radius, style.capped)),
    Primitive3D::Cuboid { p_min, p_max, .. } => {
      Some(tessellate_cuboid(p_min, p_max))
    }
    _ => None,
  }
}

/// Apply an affine transform to a collected primitive in place.
fn transform_primitive3d(prim: &mut Primitive3D, xf: &Affine3) {
  let scale = xf.length_scale();
  // An anisotropic transform bends a sphere into an ellipsoid and a
  // cylinder/cone into an elliptic one — shapes the analytic primitives
  // cannot express; a rotation does the same to a `Cuboid`, tilting its
  // faces off the coordinate planes, which its two-corner representation
  // cannot express either (a pure rotation is a similarity, so it needs
  // its own, stricter check). Tessellate first, then transform the
  // vertices, and keep the result marked `smooth` so a curved surface still
  // shades as one rather than growing facet outlines (a tessellated
  // `Cuboid` is flat-faced regardless, so `smooth` is moot for it).
  let needs_tessellation = match prim {
    Primitive3D::Cuboid { .. } => !xf.is_axis_aligned(),
    _ => !xf.is_similarity(),
  };
  if needs_tessellation && let Some(tris) = tessellate_for_transform(prim) {
    let style = primitive_style(prim).clone();
    *prim = Primitive3D::Surface3D {
      tris: tris
        .into_iter()
        .map(|(a, b, c)| (xf.apply(a), xf.apply(b), xf.apply(c)))
        .collect(),
      style,
      // A tilted box keeps its flat faces and their outline, unlike the
      // curved solids, whose facet cuts must stay invisible.
      smooth: !matches!(prim, Primitive3D::Cuboid { .. }),
    };
    return;
  }
  match prim {
    Primitive3D::Sphere { center, radius, .. } => {
      *center = xf.apply(*center);
      *radius *= scale;
    }
    Primitive3D::Cuboid { p_min, p_max, .. } => {
      // Reached only when `xf` is axis-aligned (translation/per-axis
      // scale): transforming both corners keeps the box axis-aligned, so
      // it can stay a cheap `Cuboid` instead of a tessellated mesh.
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
/// (`Polygon`/`Line`/`Point`/`Arrow`/`Tube`/`Sphere` arguments) with the coordinates they
/// refer to, leaving everything else untouched.
///
/// With `colors` (the complex's `VertexColors`, one colour per point) each
/// `Polygon` facet is drawn in the mean colour of its vertices. With
/// `tex` (its `VertexTextureCoordinates`, one `(u, v)` per point) each
/// facet carries the coordinates of its own corners.
fn resolve_complex_indices(
  expr: &Expr,
  points: &[Expr],
  colors: Option<&[(f64, f64, f64)]>,
  tex: Option<&[(f64, f64)]>,
) -> Expr {
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
  // The facet's mean vertex colour as an `RGBColor`, for the 1-based index
  // list `indices`.
  fn facet_color(indices: &Expr, colors: &[(f64, f64, f64)]) -> Option<Expr> {
    let Expr::List(items) = indices else {
      return None;
    };
    let picked: Vec<(f64, f64, f64)> = items
      .iter()
      .filter_map(|i| match i {
        Expr::Integer(i) if *i >= 1 => colors.get(*i as usize - 1).copied(),
        _ => None,
      })
      .collect();
    if picked.is_empty() {
      return None;
    }
    let n = picked.len() as f64;
    let mean = |f: fn(&(f64, f64, f64)) -> f64| {
      Expr::Real(picked.iter().map(f).sum::<f64>() / n)
    };
    Some(call(
      "RGBColor",
      vec![mean(|c| c.0), mean(|c| c.1), mean(|c| c.2)],
    ))
  }
  match expr {
    Expr::FunctionCall { name, args }
      if name == "Polygon"
        && !args.is_empty()
        && (colors.is_some() || tex.is_some()) =>
    {
      let facet = |indices: &Expr| -> Expr {
        // Not `args.to_vec()`: that copied every facet's indices for each
        // facet.
        let mut new_args = Vec::with_capacity(args.len() + 1);
        new_args.push(substitute(indices, points));
        new_args.extend(args[1..].iter().cloned());
        if let Some(tex) = tex
          && let Expr::List(items) = indices
        {
          let uvs: Option<Vec<Expr>> = items
            .iter()
            .map(|i| match i {
              Expr::Integer(i) if *i >= 1 => {
                tex.get(*i as usize - 1).map(|&(u, v)| {
                  Expr::List(vec![Expr::Real(u), Expr::Real(v)].into())
                })
              }
              _ => None,
            })
            .collect();
          if let Some(uvs) = uvs {
            new_args.push(Expr::Rule {
              pattern: Box::new(Expr::Identifier(
                "VertexTextureCoordinates".to_string(),
              )),
              replacement: Box::new(Expr::List(uvs.into())),
            });
          }
        }
        let polygon = Expr::FunctionCall {
          name: name.clone(),
          args: new_args.into(),
        };
        match colors.and_then(|c| facet_color(indices, c)) {
          Some(color) => Expr::List(vec![color, polygon].into()),
          None => polygon,
        }
      };
      match &args[0] {
        Expr::List(items)
          if !items.is_empty()
            && items.iter().all(|e| matches!(e, Expr::List(_))) =>
        {
          Expr::List(items.iter().map(facet).collect::<Vec<_>>().into())
        }
        single => facet(single),
      }
    }
    Expr::FunctionCall { name, args }
      if (matches!(
        name.as_str(),
        "Polygon" | "Line" | "Point" | "Arrow" | "Tube"
      ) || name == "Sphere")
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
        .map(|e| resolve_complex_indices(e, points, colors, tex))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| resolve_complex_indices(e, points, colors, tex))
        .collect::<Vec<_>>()
        .into(),
    },
    other => other.clone(),
  }
}

/// The per-point colours of a `GraphicsComplex`'s `VertexColors -> {…}`
/// option, as unit-range RGB triples.
fn complex_vertex_colors(options: &[Expr]) -> Option<Vec<(f64, f64, f64)>> {
  options.iter().find_map(|opt| {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return None;
    };
    if !matches!(pattern.as_ref(), Expr::Identifier(k) if k == "VertexColors") {
      return None;
    }
    let value = evaluate_expr_to_expr(replacement).ok()?;
    let Expr::List(items) = &value else {
      return None;
    };
    items
      .iter()
      .map(|c| {
        // A bare `{r, g, b}` triple (what a stored complex holds) is an
        // RGB colour too.
        if let Expr::List(rgb) = c
          && rgb.len() == 3
          && let Some(v) = rgb
            .iter()
            .map(crate::functions::math_ast::expr_to_f64)
            .collect::<Option<Vec<f64>>>()
        {
          return Some((v[0], v[1], v[2]));
        }
        crate::functions::graphics::parse_color(c).map(|c| (c.r, c.g, c.b))
      })
      .map(|c| {
        c.map(|(r, g, b)| {
          (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
        })
      })
      .collect()
  })
}

/// The alignment offset of `Text[expr, pos, offset]`: a pair running from
/// -1 (left/bottom) to 1 (right/top), written either as numbers or with the
/// alignment symbols.
fn parse_text3d_offset(spec: &Expr) -> Option<(f64, f64)> {
  fn component(e: &Expr, horizontal: bool) -> Option<f64> {
    if let Expr::Identifier(s) = e {
      return match (s.as_str(), horizontal) {
        ("Left", true) | ("Bottom", false) => Some(-1.0),
        ("Center" | "Automatic" | "Axis" | "Baseline", _) => Some(0.0),
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

/// Typesets `label_expr` (a `Text`/`Inset` label, e.g. a plain string or
/// `Style[…]`) and pushes a `Text3D` primitive for it at `pos`, if it
/// renders to non-empty text.
fn push_text3d_label(
  label_expr: &Expr,
  pos: Point3D,
  offset: (f64, f64),
  style: &StyleState3D,
  prims: &mut Vec<Primitive3D>,
) {
  if let Some(picture) =
    crate::functions::graphics::layout_grid_picture(label_expr)
  {
    prims.push(Primitive3D::Text3D {
      label: String::new(),
      pos,
      offset,
      font_size: 12.0,
      width_chars: 0,
      picture: Some(picture),
      style: style.clone(),
    });
    return;
  }
  let styled = crate::functions::chart::parse_styled_label(label_expr);
  let (label, width_chars, font_size, color) = match styled {
    Some(s) => (
      // A stacked label keeps its lines apart by newlines; the renderer
      // lays them out.
      std::iter::once(s.svg())
        .chain(s.extra_lines.iter().cloned())
        .collect::<Vec<_>>()
        .join("\n"),
      s.max_line_chars(),
      s.font_size.or(style.font_size).unwrap_or(12.0),
      s.color,
    ),
    None => (String::new(), 0, 12.0, None),
  };
  if label.is_empty() {
    return;
  }
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
    offset,
    font_size,
    width_chars,
    picture: None,
    style: text_style,
  });
}

/// The primitives of a `Graphics3D` used as the content of a `Text`/`Inset`
/// in another 3D scene, whether it is still the held call or the evaluated
/// picture remembering its source.
fn inset_graphics3d_content(expr: &Expr) -> Option<&Expr> {
  let call = match expr {
    Expr::Graphics {
      is_3d: true,
      structure: Some(structure),
      ..
    } => structure.as_ref(),
    other => other,
  };
  match call {
    Expr::FunctionCall { name, args } if name == "Graphics3D" => args.first(),
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
          // `Sphere[{p1, p2, …}, {r1, r2, …}]` instead gives each centre
          // its own radius (a common way to plot a particle system whose
          // particles vary in size).
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
          let radius_arg = args
            .get(1)
            .map(|a| evaluate_expr_to_expr(a).unwrap_or_else(|_| a.clone()));
          let scaled_fraction = match &radius_arg {
            Some(Expr::FunctionCall { name, args })
              if name == "Scaled" && args.len() == 1 =>
            {
              try_eval_to_f64(&args[0])
            }
            _ => None,
          };
          let radii: Vec<f64> = match &radius_arg {
            Some(_) if scaled_fraction.is_some() => {
              vec![scaled_fraction.unwrap_or(1.0); centers.len()]
            }
            Some(Expr::List(items)) if items.len() == centers.len() => items
              .iter()
              .map(|i| try_eval_to_f64(i).unwrap_or(1.0))
              .collect(),
            Some(other) => {
              vec![try_eval_to_f64(other).unwrap_or(1.0); centers.len()]
            }
            None => vec![1.0; centers.len()],
          };
          for (center, radius) in centers.into_iter().zip(radii) {
            prims.push(Primitive3D::Sphere {
              center,
              radius,
              scaled: scaled_fraction.is_some(),
              style: style.clone(),
            });
          }
        }
        "Cuboid" => {
          // `Cuboid[{pmin, pmax}]` names the two opposite corners in one
          // list, the same box as `Cuboid[pmin, pmax]`.
          let corner_pair = match &args[..] {
            [Expr::List(pair)] if pair.len() == 2 => {
              parse_point3d(&pair[0]).zip(parse_point3d(&pair[1]))
            }
            _ => None,
          };
          let p_min = if let Some((lo, _)) = corner_pair {
            lo
          } else if args.is_empty() {
            Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            }
          } else {
            parse_point3d(&args[0]).unwrap_or(Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            })
          };
          let p_max = if let Some((_, hi)) = corner_pair {
            hi
          } else if args.len() >= 2 {
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
        // Solids given by their corners (`Hexahedron`, `Tetrahedron`,
        // `Prism`, `Pyramid`), corners plus faces (`Polyhedron`), or a
        // placement of a Platonic solid (`Cube[]`, `Icosahedron[{θ, ϕ}]`, …)
        // are drawn as one filled polygon per face.
        "Hexahedron" | "Tetrahedron" | "Prism" | "Pyramid" | "Polyhedron"
        | "Cube" | "Octahedron" | "Dodecahedron" | "Icosahedron"
          if crate::functions::geometric_test_ast::polyhedron_faces(expr)
            .is_some() =>
        {
          if let Some((points, faces)) =
            crate::functions::geometric_test_ast::polyhedron_faces(expr)
          {
            for face in faces {
              let points = face
                .iter()
                .map(|&i| Point3D {
                  x: points[i][0],
                  y: points[i][1],
                  z: points[i][2],
                })
                .collect();
              prims.push(Primitive3D::Polygon3D {
                points,
                holes: Vec::new(),
                style: style.clone(),
              });
            }
          }
        }
        "Polygon" | "Triangle" if !args.is_empty() => {
          if let Some(pts) = parse_point3d_list(&args[0]) {
            let mut style = style.clone();
            style.texture_coords = parse_texture_coords3d(&args[1..])
              .filter(|c| c.len() == pts.len())
              .map(std::sync::Arc::new);
            let (points, holes) = if style.texture_coords.is_some() {
              (pts, Vec::new())
            } else {
              split_nested_loops(pts)
            };
            prims.push(Primitive3D::Polygon3D {
              points,
              holes,
              style,
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
            for poly in poly_exprs {
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
        // `Text[Graphics3D[…], Scaled[{½, ½, ½}]]` is the front end's way of
        // overlaying a second scene (with its own camera options) on the
        // first. The inner scene's primitives share the outer coordinates,
        // so they are drawn as part of this scene.
        "Text" | "Inset"
          if args.len() >= 2
            && inset_graphics3d_content(&args[0]).is_some() =>
        {
          if let Some(content) = inset_graphics3d_content(&args[0]) {
            collect_3d_primitives(content, style, prims);
          }
        }
        "Text" if args.len() >= 2 => {
          if let Some(pos) = parse_point3d(&args[1]) {
            let offset = args
              .get(2)
              .and_then(parse_text3d_offset)
              .unwrap_or((0.0, 0.0));
            push_text3d_label(&args[0], pos, offset, style, prims);
          }
        }
        // `Inset[obj, {x, y, z}]` places `obj` at a point of the scene.
        // The Demonstrations gallery uses it almost exclusively as
        // `Inset[Text[Style[…]], pos]`, so unwrap a `Text[…]` wrapper and
        // fall through to the same label rendering `Text` uses above.
        "Inset" if args.len() >= 2 => {
          if let Some(pos) = parse_point3d(&args[1]) {
            let label_expr = match &args[0] {
              Expr::FunctionCall { name, args: inner }
                if name == "Text" && !inner.is_empty() =>
              {
                &inner[0]
              }
              other => other,
            };
            push_text3d_label(label_expr, pos, (0.0, 0.0), style, prims);
          }
        }
        // `Cylinder[{p1, p2}, r]` and `Cone[{p1, p2}, r]`; a list of
        // `{p1, p2}` pairs draws one solid per axis, all with radius `r`
        // (a `Dynamic[r]` radius is drawn at its current value).
        "Cylinder" | "Cone" => {
          let default_axis = (
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
          );
          let axes: Vec<(Point3D, Point3D)> = match args.first() {
            None => vec![default_axis],
            Some(first) => {
              let first = evaluate_expr_to_expr(first).unwrap_or(first.clone());
              match &first {
                Expr::List(items)
                  if !items.is_empty()
                    && items.iter().all(|it| {
                      matches!(it, Expr::List(pair) if pair.len() == 2
                      && parse_point3d(&pair[0]).is_some()
                      && parse_point3d(&pair[1]).is_some())
                    }) =>
                {
                  items
                    .iter()
                    .filter_map(|it| match it {
                      Expr::List(pair) => Some((
                        parse_point3d(&pair[0])?,
                        parse_point3d(&pair[1])?,
                      )),
                      _ => None,
                    })
                    .collect()
                }
                Expr::List(items) if items.len() == 2 => vec![(
                  parse_point3d(&items[0]).unwrap_or(default_axis.0),
                  parse_point3d(&items[1]).unwrap_or(default_axis.1),
                )],
                _ => vec![default_axis],
              }
            }
          };
          let radius = match args.get(1) {
            Some(r) => {
              let r = match r {
                Expr::FunctionCall { name, args: inner }
                  if name == "Dynamic" && !inner.is_empty() =>
                {
                  &inner[0]
                }
                other => other,
              };
              try_eval_to_f64(&evaluate_expr_to_expr(r).unwrap_or(r.clone()))
                .unwrap_or(1.0)
            }
            None => 1.0,
          };
          for (p1, p2) in axes {
            prims.push(if name == "Cylinder" {
              Primitive3D::Cylinder {
                p1,
                p2,
                radius,
                style: style.clone(),
              }
            } else {
              Primitive3D::Cone {
                p1,
                p2,
                radius,
                style: style.clone(),
              }
            });
          }
        }
        // Tube[curve], Tube[curve, r] and Tube[curve, {r1, …, rn}] — a
        // surface of revolution about a polyline, one radius per vertex.
        // `curve` is a point list, a `Line[…]`, or a list of either.
        // `TubeBox` is the box form `Tube` is typeset as and draws
        // identically, so notebooks that spell it out render the same.
        "Tube" | "TubeBox" if !args.is_empty() => {
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
            let radii: Vec<f64> = if let Some(spec) = args.get(1) {
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
                // A list of another length (a radius per spline control
                // point, say) is spread along the curve and interpolated.
                Expr::List(items) if items.len() >= 2 => {
                  let rs: Vec<f64> = items
                    .iter()
                    .map(|e| {
                      try_eval_to_f64(
                        &evaluate_expr_to_expr(e).unwrap_or(e.clone()),
                      )
                      .unwrap_or(0.0)
                    })
                    .collect();
                  resample_tube_radii(&pts, &rs)
                }
                _ => {
                  let r = try_eval_to_f64(&spec).unwrap_or(0.0);
                  vec![r; pts.len()]
                }
              }
            } else {
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
            };
            let budget = tube_subdivision_budget(SCENE_TUBE_COUNT.get());
            let tris = tessellate_tube(&pts, &radii, style.capped, budget);
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
          let center = if args.is_empty() {
            Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            }
          } else {
            parse_point3d(&args[0]).unwrap_or(Point3D {
              x: 0.0,
              y: 0.0,
              z: 0.0,
            })
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
            let colors = complex_vertex_colors(&args[2..]);
            let tex = parse_texture_coords3d(&args[2..]);
            let resolved = resolve_complex_indices(
              &args[1],
              points,
              colors.as_deref(),
              tex.as_deref(),
            );
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
        // `GeometricTransformation[g, t]` maps the wrapped primitives
        // through the affine map `t` (a list of maps draws one copy each).
        "GeometricTransformation" if args.len() == 2 => {
          let mut sub = Vec::new();
          let saved = style.clone();
          collect_3d_primitives(&args[0], style, &mut sub);
          *style = saved;
          let transforms = parse_3d_affine_transforms(&args[1]);
          if transforms.is_empty() {
            // Unrecognized transform spec: keep the primitives untransformed.
            prims.extend(sub);
          } else {
            for xf in &transforms {
              for prim in &sub {
                let mut copy = prim.clone();
                transform_primitive3d(&mut copy, xf);
                prims.push(copy);
              }
            }
          }
        }
        // `Style[prims, directives…]` applies its directives (colours, a
        // numeric font size, …) to its first argument only, so they must
        // not leak into the primitives that follow it in the enclosing list.
        "Style" if args.len() >= 2 => {
          let saved = style.clone();
          for directive in &args[1..] {
            let evaluated =
              evaluate_expr_to_expr(directive).unwrap_or(directive.clone());
            match try_eval_to_f64(&evaluated) {
              Some(size) if size > 0.0 => style.font_size = Some(size),
              _ => {
                apply_3d_directive(&evaluated, style);
              }
            }
          }
          collect_3d_primitives(&args[0], style, prims);
          *style = saved;
        }
        // `Dynamic[content]` holds its content; the scene shows what it
        // evaluates to now. Walking the held `Table[…]` instead would read
        // its body outside the iterator (`sides[[i, t]]` with `i` unbound).
        "Dynamic" if !args.is_empty() => {
          let content =
            evaluate_expr_to_expr(&args[0]).unwrap_or_else(|_| args[0].clone());
          collect_3d_primitives(&content, style, prims);
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
  let ring = f64::midpoint(r1, r2);
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

/// One radius per point of `pts` from a shorter or longer list `radii`: the
/// list is laid at equal steps along the curve's arc length and linearly
/// interpolated at each point.
fn resample_tube_radii(pts: &[Point3D], radii: &[f64]) -> Vec<f64> {
  let mut arc = vec![0.0f64; pts.len()];
  for i in 1..pts.len() {
    let (a, b) = (&pts[i - 1], &pts[i]);
    arc[i] = arc[i - 1]
      + ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2))
        .sqrt();
  }
  let total = arc.last().copied().unwrap_or(0.0);
  let last = (radii.len() - 1) as f64;
  arc
    .iter()
    .enumerate()
    .map(|(i, &d)| {
      let t = if total > 0.0 {
        d / total
      } else {
        i as f64 / (pts.len().max(2) - 1) as f64
      };
      let u = (t * last).clamp(0.0, last);
      let k = (u.floor() as usize).min(radii.len() - 2);
      let f = u - k as f64;
      radii[k] * (1.0 - f) + radii[k + 1] * f
    })
    .collect()
}

/// The polyline a `BSplineCurve[{p1, …}, opts…]` stands for: the uniform
/// B-spline of degree `min(3, n - 1)` over its control points, sampled
/// finely enough to read as a curve. `SplineClosed -> True` wraps the
/// leading control points onto the end so the curve closes on itself.
/// `SplineDegree -> k` overrides the default degree `min(3, n - 1)`.
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
  let degree = crate::functions::graphics::spline_degree_from_args(
    &args[1..],
    3usize.min(control.len() - 1),
  );
  let control = if closed {
    let mut cp = control.clone();
    cp.extend_from_slice(&control[..degree]);
    cp
  } else {
    control
  };
  let clamped =
    !closed && !crate::functions::graphics::spline_knots_unclamped(&args[1..]);
  let n = control.len();
  // Enough samples that the curve is smooth without the tube it may feed
  // exploding into triangles: a handful per control point, bounded.
  let samples = (n * 6).clamp(64, 600);
  Some(
    crate::functions::graphics::bspline_sample_weights(
      n, degree, samples, clamped,
    )
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
  let wu = crate::functions::graphics::bspline_sample_weights(
    n_rows,
    3usize.min(n_rows - 1),
    samples,
    true,
  );
  let wv = crate::functions::graphics::bspline_sample_weights(
    n_cols,
    3usize.min(n_cols - 1),
    samples,
    true,
  );
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

/// Split a polygon whose point list revisits a vertex into its closed
/// loops (`{a, b, c, a, d, e, f, d}` is the loops `a b c` and `d e f`).
/// Wolfram fills such a polygon with the even-odd rule, so a loop lying
/// inside the first one is a hole; any loop that does not is kept as part
/// of the outline, leaving the polygon untouched.
fn split_nested_loops(
  points: Vec<Point3D>,
) -> (Vec<Point3D>, Vec<Vec<Point3D>>) {
  let same = |a: &Point3D, b: &Point3D| {
    (a.x - b.x).abs() < 1e-12
      && (a.y - b.y).abs() < 1e-12
      && (a.z - b.z).abs() < 1e-12
  };
  let mut loops: Vec<Vec<Point3D>> = Vec::new();
  let mut current: Vec<Point3D> = Vec::new();
  for p in &points {
    if let Some(start) = current.iter().position(|q| same(q, p)) {
      // Back at an earlier vertex: the points since then form a loop.
      let ring = current.split_off(start);
      if ring.len() >= 3 {
        loops.push(ring);
      }
      current.push(*p);
    } else {
      current.push(*p);
    }
  }
  if current.len() >= 3 {
    loops.push(current);
  }
  if loops.len() < 2 {
    return (points, Vec::new());
  }
  // Drop the axis the outer loop's normal points along; containment is
  // then a 2D ray cast.
  let outer = &loops[0];
  let mut n = [0.0f64; 3];
  for i in 0..outer.len() {
    let a = outer[i];
    let b = outer[(i + 1) % outer.len()];
    n[0] += (a.y - b.y) * (a.z + b.z);
    n[1] += (a.z - b.z) * (a.x + b.x);
    n[2] += (a.x - b.x) * (a.y + b.y);
  }
  let drop = (0..3)
    .max_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
    .unwrap_or(2);
  let flat = |p: &Point3D| match drop {
    0 => (p.y, p.z),
    1 => (p.x, p.z),
    _ => (p.x, p.y),
  };
  let inside = |ring: &[Point3D], p: &Point3D| {
    let (px, py) = flat(p);
    let mut hit = false;
    for i in 0..ring.len() {
      let (x1, y1) = flat(&ring[i]);
      let (x2, y2) = flat(&ring[(i + 1) % ring.len()]);
      if (y1 > py) != (y2 > py) && px < (x2 - x1) * (py - y1) / (y2 - y1) + x1 {
        hit = !hit;
      }
    }
    hit
  };
  if !loops[1..]
    .iter()
    .all(|l| l.iter().all(|p| inside(&loops[0], p)))
  {
    return (points, Vec::new());
  }
  let mut it = loops.into_iter();
  let outer = it.next().unwrap_or_default();
  (outer, it.collect())
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

/// The six faces of an axis-aligned box in the order `tessellate_cuboid`
/// emits them (bottom, top, front, back, left, right), each as a hashable
/// key: the axis it is perpendicular to, which side of the box it is on,
/// and its plane coordinate and rectangle snapped to a 1e-9 grid.
fn cuboid_face_keys(
  p_min: &Point3D,
  p_max: &Point3D,
) -> [(u8, bool, [i64; 5]); 6] {
  let q = |v: f64| (v * 1e9).round() as i64;
  let (x0, y0, z0) = (q(p_min.x), q(p_min.y), q(p_min.z));
  let (x1, y1, z1) = (q(p_max.x), q(p_max.y), q(p_max.z));
  [
    (2, false, [z0, x0, x1, y0, y1]),
    (2, true, [z1, x0, x1, y0, y1]),
    (1, false, [y0, x0, x1, z0, z1]),
    (1, true, [y1, x0, x1, z0, z1]),
    (0, false, [x0, y0, y1, z0, z1]),
    (0, true, [x1, y0, y1, z0, z1]),
  ]
}

/// Faces of opaque boxes that sit exactly against an opposite-facing face
/// of another opaque box (same plane, same rectangle). They are interior
/// to the union and never visible, but a grid of many boxes (e.g. a
/// fractal built from unit cubes) would otherwise emit them all.
fn hidden_cuboid_faces(
  prims: &[Primitive3D],
) -> std::collections::HashSet<(u8, [i64; 5])> {
  let mut seen: std::collections::HashSet<(u8, bool, [i64; 5])> =
    std::collections::HashSet::new();
  let mut hidden = std::collections::HashSet::new();
  for prim in prims {
    if let Primitive3D::Cuboid {
      p_min,
      p_max,
      style,
    } = prim
      && style.opacity >= 1.0
    {
      for (axis, hi, key) in cuboid_face_keys(p_min, p_max) {
        if seen.contains(&(axis, !hi, key)) {
          hidden.insert((axis, key));
        }
        seen.insert((axis, hi, key));
      }
    }
  }
  hidden
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

/// Which edges of a flat-faced mesh's triangles outline a face: an edge is an
/// internal cut, not an outline, when the triangle on its other side lies in
/// the same plane. Unlike [`box_edge_flags`] this holds for a box turned in
/// any direction.
fn coplanar_cut_flags(tris: &[(Point3D, Point3D, Point3D)]) -> Vec<[bool; 3]> {
  use std::collections::HashMap;
  let key = |p: Point3D| {
    let q = |v: f64| (v * 1e8).round() as i64;
    (q(p.x), q(p.y), q(p.z))
  };
  let edge_key = |a: Point3D, b: Point3D| {
    let (ka, kb) = (key(a), key(b));
    if ka <= kb { (ka, kb) } else { (kb, ka) }
  };
  let normals: Vec<[f64; 3]> = tris
    .iter()
    .map(|t| triangle_normal(t.0, t.1, t.2))
    .collect();
  let mut seen: HashMap<_, Vec<usize>> = HashMap::new();
  for (i, t) in tris.iter().enumerate() {
    for (a, b) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
      seen.entry(edge_key(a, b)).or_default().push(i);
    }
  }
  tris
    .iter()
    .enumerate()
    .map(|(i, t)| {
      let outline = |a: Point3D, b: Point3D| {
        !seen[&edge_key(a, b)].iter().any(|&j| {
          let (n, m) = (normals[i], normals[j]);
          j != i && (n[0] * m[0] + n[1] * m[1] + n[2] * m[2]).abs() > 1.0 - 1e-9
        })
      };
      [outline(t.0, t.1), outline(t.1, t.2), outline(t.2, t.0)]
    })
    .collect()
}

/// Bounds how elongated a single tessellated side face can be. Without
/// this, a long cylinder or tube tessellates its side into quads that span
/// its whole length in one piece, and the painter's algorithm sorts each
/// quad by its centroid depth — averaging a face's near and far ends
/// together lets a smaller object poking through only the far half of the
/// face sort as if it were in front of the whole thing. Subdividing the
/// length so each face stays roughly as long as it is wide keeps every
/// face's centroid a good local depth estimate.
const MAX_SIDE_FACE_ASPECT: f64 = 1.0;
const MAX_SIDE_SUBDIVISIONS: usize = 200;
/// Radial subdivision count for `Cylinder`/`Cone`, shared between the
/// tessellators and the call site that rebuilds their rim-edge flags.
const CYLINDER_SIDES: usize = 24;

/// How many rings to place along a side of the given length and radius so
/// that no single tessellated face is more elongated than
/// `MAX_SIDE_FACE_ASPECT` times as long as it is wide, given `sides`
/// facets around the circumference. `max_subdivisions` caps the result —
/// pass `MAX_SIDE_SUBDIVISIONS` for the unconstrained default, or a
/// smaller scene-aware budget from `cylinder_max_subdivisions`.
fn side_subdivision_steps(
  len: f64,
  radius: f64,
  sides: usize,
  max_subdivisions: usize,
) -> usize {
  let facet_width =
    2.0 * std::f64::consts::PI * radius.abs().max(1e-9) / sides as f64;
  ((len / facet_width) / MAX_SIDE_FACE_ASPECT)
    .ceil()
    .clamp(1.0, max_subdivisions as f64) as usize
}

/// Scales the longitudinal subdivision budget for `Cylinder` down once a
/// scene has many of them and a given one's radius is a small fraction of
/// the whole scene's span — mirroring `sphere_detail`'s reasoning for
/// `Sphere`. Regression: a branching-tree Demonstration built its picture
/// from ~100 hair-thin `Cylinder`s (radius 0.01 against a scene spanning
/// ~1.6 units); `side_subdivision_steps`'s raw length/radius aspect target
/// drove most of them to the unconditional 200-ring cap, tessellating the
/// whole scene into over half a million triangles for a picture where
/// each cylinder is only ever a few screen pixels wide — the
/// depth-sorting seams that budget guards against are imperceptible on an
/// object that thin. A lone or modestly sized Cylinder keeps the full
/// budget; only a crowd of relatively tiny ones gets throttled.
fn cylinder_max_subdivisions(
  radius: f64,
  scene_extent: f64,
  cylinder_count: usize,
) -> usize {
  if cylinder_count <= 8 || scene_extent <= 0.0 {
    return MAX_SIDE_SUBDIVISIONS;
  }
  let rel = radius.abs() / scene_extent;
  let by_size = if rel > 0.08 {
    MAX_SIDE_SUBDIVISIONS
  } else if rel > 0.04 {
    100
  } else if rel > 0.02 {
    50
  } else {
    16
  };
  // A crowd of hundreds of cylinders (a ring-stacked molecule model) shares
  // a total ring budget, so the SVG stays tractable even when each one is
  // individually too fat to be throttled by its relative size.
  let by_count = (TOTAL_CYLINDER_RING_BUDGET / cylinder_count).max(2);
  by_size.min(by_count)
}

/// Total longitudinal rings shared by all `Cylinder`s of a crowded scene.
const TOTAL_CYLINDER_RING_BUDGET: usize = 1500;

thread_local! {
  /// Number of `Tube`s in the `Graphics3D` currently being collected, so
  /// each can size its tessellation to the crowd it is drawn in.
  static SCENE_TUBE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Count the tubes in a 3D scene: one per curve, so `Tube[{curve1, curve2,
/// …}]` counts each of its curves (343 of them in one Demonstration, which
/// the lone-tube budget tessellated into gigabytes of polygons).
fn count_tubes(expr: &Expr) -> usize {
  match expr {
    Expr::List(items) => items.iter().map(count_tubes).sum(),
    Expr::FunctionCall { name, args } if name == "Tube" => {
      match args.first() {
        // A list of curves, each a list of points.
        Some(Expr::List(curves))
          if curves.iter().all(|c| {
            matches!(c, Expr::List(pts)
              if pts.iter().all(|p| matches!(p, Expr::List(_))))
          }) =>
        {
          curves.len().max(1)
        }
        _ => 1,
      }
    }
    Expr::FunctionCall { args, .. } => args.iter().map(count_tubes).sum(),
    _ => 0,
  }
}

/// Longitudinal subdivision budget for one `Tube` in a scene with
/// `tube_count` of them. Regression: a maze Demonstration drew every wall
/// as a hair-thin `Tube`; the lengthwise ring budget meant for a lone tube
/// gave each one ~190 rings, tessellating the scene into nearly a million
/// polygons. In a crowd each tube is only a few pixels wide, so the
/// depth-sorting seams the budget guards against are imperceptible.
fn tube_subdivision_budget(tube_count: usize) -> usize {
  if tube_count <= 8 {
    MAX_SIDE_SUBDIVISIONS
  } else {
    16
  }
}

/// Tessellate a cylinder along its axis. With `capped`, flat disks close
/// both ends — Wolfram's default; `CapForm[None]` leaves it a hollow tube.
/// A flat cylinder (its two ends nearly coincident) is how a Demonstration
/// draws a filled disk in space, so the caps matter even when the side wall
/// is too thin to see.
fn tessellate_cylinder(
  p1: &Point3D,
  p2: &Point3D,
  radius: f64,
  capped: bool,
  max_subdivisions: usize,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n = CYLINDER_SIDES;
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

  let steps = side_subdivision_steps(len, radius, n, max_subdivisions);
  let ring_at = |t: f64| -> Vec<Point3D> {
    let center = Point3D {
      x: p1.x + dx * t,
      y: p1.y + dy * t,
      z: p1.z + dz * t,
    };
    (0..n)
      .map(|i| {
        let a = 2.0 * pi * i as f64 / n as f64;
        let (c, s) = (a.cos(), a.sin());
        Point3D {
          x: center.x + radius * (c * perpx + s * binx),
          y: center.y + radius * (c * perpy + s * biny),
          z: center.z + radius * (c * perpz + s * binz),
        }
      })
      .collect()
  };
  let rings: Vec<Vec<Point3D>> = (0..=steps)
    .map(|s| ring_at(s as f64 / steps as f64))
    .collect();

  let mut tris = Vec::new();
  for ring_pair in rings.windows(2) {
    let (ring1, ring2) = (&ring_pair[0], &ring_pair[1]);
    for i in 0..n {
      let i2 = (i + 1) % n;
      // Same vertex order as the un-subdivided quad (a, b, c) / (a, c, d)
      // with a = ring1[i], b = ring2[i], c = ring2[i2], d = ring1[i2], so
      // the winding — and therefore the face normal direction — matches.
      tris.push((ring1[i], ring2[i], ring2[i2]));
      tris.push((ring1[i], ring2[i2], ring1[i2]));
    }
  }
  if capped {
    // The `p1` disk's outward normal points against the axis, so its fan
    // needs the opposite winding from `p2`'s (see `tessellate_cone` for the
    // same reasoning): `(centre, ring[i+1], ring[i])` there vs.
    // `(centre, ring[i], ring[i+1])` here.
    for i in 0..n {
      tris.push((*p1, rings[0][(i + 1) % n], rings[0][i]));
    }
    let last = &rings[rings.len() - 1];
    for i in 0..n {
      tris.push((*p2, last[i], last[(i + 1) % n]));
    }
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
  max_subdivisions: usize,
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

  let ring_at = |center: Point3D,
                 radius: f64,
                 normal: (f64, f64, f64),
                 binormal: (f64, f64, f64)|
   -> Vec<Point3D> {
    (0..TUBE_SIDES)
      .map(|k| {
        let a = 2.0 * std::f64::consts::PI * k as f64 / TUBE_SIDES as f64;
        let (c, s) = (a.cos() * radius, a.sin() * radius);
        Point3D {
          x: center.x + c * normal.0 + s * binormal.0,
          y: center.y + c * normal.1 + s * binormal.1,
          z: center.z + c * normal.2 + s * binormal.2,
        }
      })
      .collect()
  };

  // The subdivision budget is shared by the whole polyline: a curve already
  // sampled into many short segments (a `ParametricPlot3D` helix with
  // `PlotStyle -> Tube[r]`) needs no further rings, and splitting each of
  // its segments to the radius' aspect target would multiply the mesh into
  // tens of thousands of faces per redraw.
  let max_steps = (max_subdivisions / dirs.len()).max(1);
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
    rings.push(ring_at(points[idx], r * widen, normal, binormal));

    // Subdivide the straight run to the next vertex so no single
    // tessellated face spans an outsized fraction of the tube's length
    // (see `side_subdivision_steps`). The normal/binormal frame doesn't
    // need to change along a straight run, so the interpolated rings just
    // reuse this vertex's frame; only the centre and radius move.
    if i + 1 < keep.len() {
      let idx_next = keep[i + 1];
      let seg_len = norm(sub(&points[idx_next], &points[idx]));
      let r_next = radii.get(idx_next).copied().unwrap_or(0.0);
      let steps =
        side_subdivision_steps(seg_len, r.max(r_next), TUBE_SIDES, max_steps);
      for s in 1..steps {
        let frac = s as f64 / steps as f64;
        let center = Point3D {
          x: points[idx].x + (points[idx_next].x - points[idx].x) * frac,
          y: points[idx].y + (points[idx_next].y - points[idx].y) * frac,
          z: points[idx].z + (points[idx_next].z - points[idx].z) * frac,
        };
        let rr = r + (r_next - r) * frac;
        rings.push(ring_at(center, rr, normal, binormal));
      }
    }
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

/// Tessellate a cone. With `capped`, a flat disk closes the base — Wolfram's
/// default; `CapForm[None]` leaves it open.
fn tessellate_cone(
  base: &Point3D,
  tip: &Point3D,
  radius: f64,
  capped: bool,
) -> Vec<(Point3D, Point3D, Point3D)> {
  let n = CYLINDER_SIDES;
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
    if capped {
      // Reversed winding vs. the lateral triangle above: its outward
      // normal points along the radius and the axis (away from the base
      // plane), so the base disk's normal — pointing the other way along
      // the axis, away from the cone — needs `b2, b1` instead of `b1, b2`.
      tris.push((*base, b2, b1));
    }
  }
  tris
}

/// Which triangle edges of a tessellated cylinder lie on one of its two end
/// circles. `tessellate_cylinder` emits a quad per segment as the triangles
/// `(a, b, c)` and `(a, c, d)`, with `a`/`d` on the first circle and `b`/`c`
/// on the second — so the rim edges are `b→c` and `d→a`.
/// `steps` is the number of longitudinal ring-pair segments and `n` the
/// radial subdivision count `tessellate_cylinder` was built with; a
/// subdivided cylinder's interior ring seams are not rims, only the first
/// and last are. When `capped`, the `2 * n` fan triangles appended after
/// the side triangles close both rims, each contributing its own rim edge.
fn cylinder_edge_flags(steps: usize, n: usize, capped: bool) -> Vec<[bool; 3]> {
  let mut flags =
    Vec::with_capacity(steps * 2 * n + if capped { 2 * n } else { 0 });
  for seg in 0..steps {
    let is_first = seg == 0;
    let is_last = seg == steps - 1;
    for _ in 0..n {
      flags.push([false, is_last, false]);
      flags.push([false, false, is_first]);
    }
  }
  if capped {
    flags.extend(std::iter::repeat_n([false, true, false], 2 * n));
  }
  flags
}

/// Which triangle edges of a tessellated cone lie on its base circle.
/// `tessellate_cone` emits one lateral triangle `(tip, b1, b2)` per segment,
/// so the rim edge is always `b1→b2`; when `capped`, each is followed by a
/// base-disk fan triangle whose own rim edge is `b2→b1`.
fn cone_edge_flags(n: usize, capped: bool) -> Vec<[bool; 3]> {
  if capped {
    std::iter::repeat_n([[false, true, false], [false, true, false]], n)
      .flatten()
      .collect()
  } else {
    vec![[false, true, false]; n]
  }
}

/// Moves the contents into the screen space of a perspective camera at
/// `eye` looking at `target`: a point at camera-space `(x, y, z)` (`z`
/// the distance along the view axis) becomes `(x/z, -1/z, y/z)`. What lies
/// behind the camera is dropped; a sphere keeps its round outline with the
/// radius it has at its centre's distance.
fn perspective_warp(
  prims: &mut Vec<Primitive3D>,
  eye: [f64; 3],
  target: [f64; 3],
  vertical: [f64; 3],
) {
  const NEAR: f64 = 1e-6;
  let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  let unit = |v: [f64; 3]| {
    let n = norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
  };
  let cross = |a: [f64; 3], b: [f64; 3]| {
    [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]
  };
  let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let dir = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
  if norm(dir) < 1e-12 {
    return;
  }
  let fwd = unit(dir);
  let mut right = cross(fwd, vertical);
  if norm(right) < 1e-9 {
    right = cross(fwd, [0.0, 1.0, 0.0]);
  }
  if norm(right) < 1e-9 {
    right = cross(fwd, [1.0, 0.0, 0.0]);
  }
  let right = unit(right);
  let up = cross(right, fwd);

  // The warped point, or `None` behind the camera.
  let warp = |p: &Point3D| -> Option<Point3D> {
    let d = [p.x - eye[0], p.y - eye[1], p.z - eye[2]];
    let z = dot(d, fwd);
    (z > NEAR).then(|| Point3D {
      x: dot(d, right) / z,
      y: -1.0 / z,
      z: dot(d, up) / z,
    })
  };
  let warp_all = |pts: &[Point3D]| -> Option<Vec<Point3D>> {
    pts.iter().map(&warp).collect()
  };
  let dist = |p: &Point3D| dot([p.x - eye[0], p.y - eye[1], p.z - eye[2]], fwd);

  let old = std::mem::take(prims);
  for prim in old {
    match prim {
      Primitive3D::Sphere {
        center,
        radius,
        scaled,
        style,
      } => {
        if let Some(c) = warp(&center) {
          prims.push(Primitive3D::Sphere {
            center: c,
            radius: radius / dist(&center),
            scaled,
            style,
          });
        }
      }
      Primitive3D::Cuboid {
        p_min,
        p_max,
        style,
      } => {
        let corner = |i: usize| Point3D {
          x: if i & 1 == 0 { p_min.x } else { p_max.x },
          y: if i & 2 == 0 { p_min.y } else { p_max.y },
          z: if i & 4 == 0 { p_min.z } else { p_max.z },
        };
        for face in [
          [0, 1, 3, 2],
          [4, 5, 7, 6],
          [0, 1, 5, 4],
          [2, 3, 7, 6],
          [0, 2, 6, 4],
          [1, 3, 7, 5],
        ] {
          let pts: Vec<Point3D> = face.iter().map(|&i| corner(i)).collect();
          if let Some(points) = warp_all(&pts) {
            prims.push(Primitive3D::Polygon3D {
              points,
              holes: Vec::new(),
              style: style.clone(),
            });
          }
        }
      }
      Primitive3D::Polygon3D {
        points,
        holes,
        style,
      } => {
        let warped_holes: Option<Vec<Vec<Point3D>>> =
          holes.iter().map(|h| warp_all(h)).collect();
        if let (Some(points), Some(holes)) = (warp_all(&points), warped_holes) {
          prims.push(Primitive3D::Polygon3D {
            points,
            holes,
            style,
          });
        }
      }
      Primitive3D::Line3D { segments, style } => {
        let segments: Vec<Vec<Point3D>> =
          segments.iter().filter_map(|s| warp_all(s)).collect();
        if !segments.is_empty() {
          prims.push(Primitive3D::Line3D { segments, style });
        }
      }
      Primitive3D::Point3DPrim { points, style } => {
        let points: Vec<Point3D> = points.iter().filter_map(&warp).collect();
        if !points.is_empty() {
          prims.push(Primitive3D::Point3DPrim { points, style });
        }
      }
      Primitive3D::Arrow3D { points, style } => {
        if let Some(points) = warp_all(&points) {
          prims.push(Primitive3D::Arrow3D { points, style });
        }
      }
      Primitive3D::Cylinder {
        p1,
        p2,
        radius,
        style,
      } => {
        if let (Some(a), Some(b)) = (warp(&p1), warp(&p2)) {
          let r = radius / dist(&p1).min(dist(&p2));
          prims.push(Primitive3D::Cylinder {
            p1: a,
            p2: b,
            radius: r,
            style,
          });
        }
      }
      Primitive3D::Cone {
        p1,
        p2,
        radius,
        style,
      } => {
        if let (Some(a), Some(b)) = (warp(&p1), warp(&p2)) {
          let r = radius / dist(&p1).min(dist(&p2));
          prims.push(Primitive3D::Cone {
            p1: a,
            p2: b,
            radius: r,
            style,
          });
        }
      }
      Primitive3D::Text3D {
        label,
        pos,
        offset,
        font_size,
        width_chars,
        picture,
        style,
      } => {
        if let Some(pos) = warp(&pos) {
          prims.push(Primitive3D::Text3D {
            label,
            pos,
            offset,
            font_size,
            width_chars,
            picture,
            style,
          });
        }
      }
      Primitive3D::Surface3D {
        tris,
        style,
        smooth,
      } => {
        let tris: Vec<(Point3D, Point3D, Point3D)> = tris
          .iter()
          .filter_map(|(a, b, c)| Some((warp(a)?, warp(b)?, warp(c)?)))
          .collect();
        if !tris.is_empty() {
          prims.push(Primitive3D::Surface3D {
            tris,
            style,
            smooth,
          });
        }
      }
    }
  }
}

/// Turn each `Sphere[c, Scaled[r]]` radius from the fraction `r` into a
/// length: that fraction of the picture's widest extent — the explicit
/// `PlotRange`, or else the box around everything but the scaled spheres.
fn resolve_scaled_sphere_radii(
  prims: &mut [Primitive3D],
  plot_range: Option<[(f64, f64); 3]>,
) {
  let is_scaled =
    |p: &Primitive3D| matches!(p, Primitive3D::Sphere { scaled: true, .. });
  if !prims.iter().any(is_scaled) {
    return;
  }
  let bounds = plot_range.unwrap_or_else(|| {
    // Scaled spheres contribute their centres only.
    let probe: Vec<Primitive3D> = prims
      .iter()
      .map(|p| match p {
        Primitive3D::Sphere {
          center,
          scaled: true,
          style,
          ..
        } => Primitive3D::Sphere {
          center: *center,
          radius: 0.0,
          scaled: false,
          style: style.clone(),
        },
        other => other.clone(),
      })
      .collect();
    primitives_bounds(&probe)
  });
  let extent = bounds
    .iter()
    .map(|(lo, hi)| hi - lo)
    .filter(|e| e.is_finite())
    .fold(0.0_f64, f64::max);
  for prim in prims {
    if let Primitive3D::Sphere { radius, scaled, .. } = prim
      && *scaled
    {
      *radius *= extent;
      *scaled = false;
    }
  }
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
  // `PlotRange -> {zmin, zmax}`: pins the vertical axis only, the other two
  // keep the extent of the contents (what a `Plot3D` carries into `Show`).
  let mut z_only_range: Option<(f64, f64)> = None;
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
  // `ViewCenter -> {a, b, c}`: the point the camera looks at, as fractions
  // of the displayed box along each axis (`{1/2, 1/2, 1/2}` is its middle).
  let mut view_center: Option<[f64; 3]> = None;
  // `ViewVector -> {x, y, z}` or `{eye, target}`: a perspective camera
  // placed at an absolute position of the scene. The target is the middle
  // of the contents when only the position is given.
  let mut view_vector: Option<([f64; 3], Option<[f64; 3]>)> = None;
  let mut view_distance = {
    let d = Camera::DEFAULT_VIEW_POINT;
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
  };
  let mut show_axes = false;
  let mut axes_origin: Option<[Option<f64>; 3]> = None;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
  let mut tick_spec: [Option<Vec<f64>>; 3] = [None, None, None];
  // `AspectRatio -> r`: the picture is `r` times as tall as it is wide.
  let mut aspect_ratio: Option<f64> = None;
  // `Lighting -> "Neutral"`: colourless lights, so a surface without a
  // colour directive shows its own white rather than the default blue.
  let mut neutral_lighting = false;
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
        "Lighting" => {
          neutral_lighting =
            matches!(replacement.as_ref(), Expr::String(s) if s == "Neutral");
        }
        "SphericalRegion" => match replacement.as_ref() {
          Expr::Identifier(s) if s == "False" => spherical_region = false,
          Expr::Identifier(s) if s == "True" => spherical_region = true,
          _ => {}
        },
        "Background" => {
          if let Some(color) = parse_background_color(replacement) {
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
          if let Some(vp) = symbolic
            .or_else(|| eval_view_direction(replacement))
            .or_else(|| eval_vec3(replacement))
          {
            camera.azimuth = vp[1].atan2(vp[0]);
            camera.elevation = vp[2].atan2(vp[0].hypot(vp[1]));
            view_distance =
              (vp[0] * vp[0] + vp[1] * vp[1] + vp[2] * vp[2]).sqrt();
          }
        }
        "ViewVector" => {
          if let Some(eye) = eval_vec3(replacement) {
            view_vector = Some((eye, None));
          } else if let Ok(Expr::List(pair)) =
            &evaluate_expr_to_expr(strip_dynamic(replacement))
            && pair.len() == 2
            && let (Some(eye), Some(target)) =
              (eval_vec3(&pair[0]), eval_vec3(&pair[1]))
          {
            view_vector = Some((eye, Some(target)));
          }
        }
        // `ViewVertical -> {x, y, z}`: which way is up on screen.
        "ViewVertical" => {
          if let Some(v) = eval_vec3(replacement)
            && v.iter().any(|c| *c != 0.0)
          {
            camera.vertical = v;
          }
        }
        // `ViewAngle -> θ` (radians, or a `Quantity`/`Degree` product).
        "ViewAngle" => {
          let replacement = strip_dynamic(replacement);
          if let Some(a) = try_eval_to_f64(
            &evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.clone()),
          ) && a > 0.0
            && a < std::f64::consts::PI
          {
            view_angle = Some(a);
          }
        }
        "ViewCenter" => {
          if let Some(c) = eval_vec3(replacement) {
            view_center = Some(c);
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
          } else if let Some((lo, hi)) = parse_z_only_range(&value)
            && hi > lo
          {
            z_only_range = Some((lo, hi));
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
        "Ticks" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          tick_spec = parse_ticks_3d(&value);
        }
        "AspectRatio" => {
          if let Some(r) = try_eval_to_f64(
            &evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.as_ref().clone()),
          ) && r > 0.0
            && r.is_finite()
          {
            aspect_ratio = Some(r);
          }
        }
        "AxesOrigin" => {
          // `{x, y, z}` where the axes cross; `Automatic` (or anything
          // non-numeric) leaves that coordinate at the box's low end.
          if let Expr::List(items) = replacement.as_ref()
            && items.len() == 3
          {
            let mut at = [None; 3];
            for (slot, item) in at.iter_mut().zip(items.iter()) {
              *slot = try_eval_to_f64(
                &evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone()),
              );
            }
            axes_origin = Some(at);
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
          axes_labels = parse_axes_labels(replacement, None);
        }
        _ => {}
      }
    }
  }

  if let Some(r) = aspect_ratio {
    svg_height = ((svg_width as f64 * r).round() as u32).max(1);
  }

  // Collect primitives
  let mut prims = Vec::new();
  let mut style3d = StyleState3D::default();
  SCENE_TUBE_COUNT.set(count_tubes(&content));
  collect_3d_primitives(&content, &mut style3d, &mut prims);

  if plot_range.is_none()
    && let Some(z) = z_only_range
  {
    let [x, y, _] = primitives_bounds(&prims);
    if x.0.is_finite() && x.1.is_finite() && y.0.is_finite() && y.1.is_finite()
    {
      plot_range = Some([x, y, z]);
    }
  }

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

  resolve_scaled_sphere_radii(&mut prims, plot_range);

  // An explicit `PlotRange` cuts off whatever of a line lies outside it.
  if let Some(bounds) = plot_range {
    for prim in &mut prims {
      if let Primitive3D::Line3D { segments, .. } = prim {
        *segments = segments
          .iter()
          .flat_map(|seg| clip_polyline_to_box(seg, &bounds))
          .collect();
      }
    }
    prims.retain(|p| {
      !matches!(p, Primitive3D::Line3D { segments, .. } if segments.is_empty())
    });

    // Likewise a polygon is cut down to the part inside the box, and
    // dropped when nothing of it is left (a large translucent plane shown
    // in a small range).
    for prim in &mut prims {
      if let Primitive3D::Polygon3D { points, holes, .. } = prim
        && holes.is_empty()
        && points.iter().any(|p| !point_in_box(p, &bounds))
      {
        let mut poly: Vec<[f64; 3]> =
          points.iter().map(|p| [p.x, p.y, p.z]).collect();
        for axis in 0..3 {
          let mut n = [0.0; 3];
          n[axis] = 1.0;
          poly = clip_polygon_to_halfspace(&poly, n, bounds[axis].1);
          n[axis] = -1.0;
          poly = clip_polygon_to_halfspace(&poly, n, -bounds[axis].0);
        }
        *points = poly
          .into_iter()
          .map(|[x, y, z]| Point3D { x, y, z })
          .collect();
      }
    }
    prims.retain(|p| {
      !matches!(p, Primitive3D::Polygon3D { points, .. } if points.len() < 3)
    });
  }

  // The symbolic form carried on the rendered result so that Part can
  // index it (`Graphics3D[…][[1]]` → the content).
  let structure = call(
    "Graphics3D",
    std::iter::once(content.clone())
      .chain(args[1..].iter().cloned())
      .collect::<Vec<_>>(),
  );

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

  // A `ViewVector` camera sits inside the scene, so the picture is a true
  // perspective one. A perspective map takes lines to lines and planes to
  // planes, so warping the contents into the camera's screen space lets the
  // orthographic pipeline below draw them: x and z carry the screen
  // position and y the depth.
  let perspective = view_vector.is_some();
  if let Some((eye, target)) = view_vector {
    let target = target.unwrap_or_else(|| {
      let b = primitives_bounds(&prims);
      [
        f64::midpoint(b[0].0, b[0].1),
        f64::midpoint(b[1].0, b[1].1),
        f64::midpoint(b[2].0, b[2].1),
      ]
    });
    perspective_warp(&mut prims, eye, target, camera.vertical);
    camera.azimuth = -std::f64::consts::FRAC_PI_2;
    camera.elevation = 0.0;
    camera.vertical = Camera::DEFAULT_VERTICAL;
    show_axes = false;
    if prims.is_empty() {
      let empty_svg = format!(
        "<svg width=\"{svg_width}\" height=\"{svg_height}\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"
      );
      return Ok(crate::graphics3d_result_with_structure(
        empty_svg, structure,
      ));
    }
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
  let mut triangle_vertices: Vec<[Point3D; 3]> = Vec::new();
  let base_color = if neutral_lighting {
    (0xFF_u8, 0xFF_u8, 0xFF_u8)
  } else {
    (0x5E_u8, 0x81_u8, 0xB5_u8) // Default blue
  };

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

  // Cylinder-scene statistics for adaptive tessellation (see
  // `cylinder_max_subdivisions`): how many Cylinders there are and how
  // large the region they occupy is.
  let mut cyl_count = 0usize;
  let mut cyl_min = [f64::INFINITY; 3];
  let mut cyl_max = [f64::NEG_INFINITY; 3];
  for prim in &prims {
    if let Primitive3D::Cylinder { p1, p2, radius, .. } = prim {
      cyl_count += 1;
      for p in [p1, p2] {
        for (i, c) in [p.x, p.y, p.z].into_iter().enumerate() {
          cyl_min[i] = cyl_min[i].min(c - radius);
          cyl_max[i] = cyl_max[i].max(c + radius);
        }
      }
    }
  }
  let cyl_extent = if cyl_count > 0 {
    (0..3).fold(0.0f64, |m, i| m.max(cyl_max[i] - cyl_min[i]))
  } else {
    0.0
  };

  let hidden_faces = hidden_cuboid_faces(&prims);

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
          ..
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
          let mut tris = tessellate_cuboid(p_min, p_max);
          if style.opacity >= 1.0 && !hidden_faces.is_empty() {
            let keys = cuboid_face_keys(p_min, p_max);
            let mut face = 0;
            tris.retain(|_| {
              let (axis, _, key) = keys[face / 2];
              face += 1;
              !hidden_faces.contains(&(axis, key))
            });
          }
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
          let max_subdiv =
            cylinder_max_subdivisions(*radius, cyl_extent, cyl_count);
          let tris =
            tessellate_cylinder(p1, p2, *radius, style.capped, max_subdiv);
          if style.edge_color.is_some() {
            let len = ((p2.x - p1.x).powi(2)
              + (p2.y - p1.y).powi(2)
              + (p2.z - p1.z).powi(2))
            .sqrt();
            let steps =
              side_subdivision_steps(len, *radius, CYLINDER_SIDES, max_subdiv);
            holed_boundaries =
              cylinder_edge_flags(steps, CYLINDER_SIDES, style.capped);
          }
          (tris, style)
        }
        Primitive3D::Cone {
          p1,
          p2,
          radius,
          style,
        } => {
          let tris = tessellate_cone(p1, p2, *radius, style.capped);
          if style.edge_color.is_some() {
            holed_boundaries = cone_edge_flags(CYLINDER_SIDES, style.capped);
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
            holed_boundaries = coplanar_cut_flags(tris);
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
            edge_width: None,
            specular: None,
            glow: None,
            dashing: None,
            point_size: None,
            texture: None,
            texture_coords: None,
            font_size: None,
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
      let color = add_glow(
        apply_lighting_specular(
          side_color,
          normal,
          prim_style.specular,
          view_dir,
        ),
        prim_style.glow,
      );
      let p0 = project(v0, &camera);
      let p1 = project(v1, &camera);
      let p2 = project(v2, &camera);
      let center = Point3D {
        x: (v0.x + v1.x + v2.x) / 3.0,
        y: (v0.y + v1.y + v2.y) / 3.0,
        z: (v0.z + v1.z + v2.z) / 3.0,
      };
      triangle_vertices.push([v0, v1, v2]);
      all_triangles.push(Triangle {
        boundary,
        edge_color: prim_style.edge_color,
        edge_width: prim_style.edge_width,
        projected: [p0, p1, p2],
        depth: depth(center, &camera),
        color,
        opacity: prim_opacity,
        texture: fan_texture(prim, prim_style, i, tri_count),
      });
    }
  }

  // Painter's algorithm
  split_interpenetrating(&mut triangle_vertices, &mut all_triangles, &camera);
  {
    let projected: Vec<_> = all_triangles.iter().map(|t| t.projected).collect();
    let depths: Vec<_> = all_triangles.iter().map(|t| t.depth).collect();
    let order = painter_order(&triangle_vertices, &projected, &depths, &camera);
    let mut slots: Vec<Option<Triangle>> =
      all_triangles.drain(..).map(Some).collect();
    all_triangles = order.into_iter().filter_map(|i| slots[i].take()).collect();
  }

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
  // not follow the contents. A pinned region is framed by its box alone:
  // content reaching beyond it (a long axis cylinder, a big sphere) is
  // cut off by the picture rather than shrinking it.
  if plot_range.is_some() {
    px_min = f64::INFINITY;
    px_max = f64::NEG_INFINITY;
    py_min = f64::INFINITY;
    py_max = f64::NEG_INFINITY;
  }
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
  if spherical_region && !perspective {
    let center = Point3D {
      x: f64::midpoint(x3_min, x3_max),
      y: f64::midpoint(y3_min, y3_max),
      z: f64::midpoint(z3_min, z3_max),
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
  let mut p_cx = f64::midpoint(px_min, px_max);
  let mut p_cy = f64::midpoint(py_min, py_max);

  // An explicit `ViewAngle` fixes the camera's field of view, so the scale
  // comes from the view volume rather than from fitting the contents: the
  // frame spans `2 · d · tan(θ/2)` in the units Wolfram measures `ViewPoint`
  // in, which are those of the longest side of the displayed box. A small
  // object then stays small instead of being blown up to fill the picture.
  // Measured against wolframscript over a sweep of angles, distances and
  // plot ranges: a world length `l` covers `(l / L) / (2 d tan(θ/2))` of
  // the frame.
  // `ViewCenter` moves the point the camera looks at off the box middle.
  let look_at = |c: [f64; 3]| {
    project(
      Point3D {
        x: x3_min + c[0] * (x3_max - x3_min),
        y: y3_min + c[1] * (y3_max - y3_min),
        z: z3_min + c[2] * (z3_max - z3_min),
      },
      &camera,
    )
  };
  if let Some(c) = view_center
    && view_angle.is_none()
  {
    (p_cx, p_cy) = look_at(c);
  }
  if let Some(theta) = view_angle {
    let span = 2.0 * view_distance * (theta / 2.0).tan() * view_box_side;
    if span > 0.0 && span.is_finite() {
      scale = svg_width.min(svg_height) as f64 / span;
      // The view centres on the middle of the displayed box, not on
      // whatever part of it the contents happen to occupy.
      let (bcx, bcy) = project(
        Point3D {
          x: f64::midpoint(x3_min, x3_max),
          y: f64::midpoint(y3_min, y3_max),
          z: f64::midpoint(z3_min, z3_max),
        },
        &camera,
      );
      p_cx = bcx;
      p_cy = bcy;
      if let Some(c) = view_center {
        (p_cx, p_cy) = look_at(c);
      }
    }
  }

  // The warped scene is in units of the camera's tangent plane, so the
  // field of view alone sets the scale, with the view axis at the centre.
  if perspective {
    let theta = view_angle.unwrap_or(35.0_f64.to_radians());
    scale = svg_width.min(svg_height) as f64 / (2.0 * (theta / 2.0).tan());
    p_cx = 0.0;
    p_cy = 0.0;
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
    dashing: Option<Vec<f64>>,
    dash_offset: f64,
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
        let tm = f64::midpoint(t0, t1);
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
          dashing: None,
          dash_offset: 0.0,
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
          // Total on-screen length of the *undivided* segment, so a dash
          // pattern's offset can be measured continuously across the
          // depth-sorting subdivisions below instead of restarting at
          // each one.
          let (apx, apy) = to_svg(project(a, &camera).0, project(a, &camera).1);
          let (bpx, bpy) = to_svg(project(b, &camera).0, project(b, &camera).1);
          let total_px = ((bpx - apx).powi(2) + (bpy - apy).powi(2)).sqrt();
          for s in 0..LINE_SUBDIVISIONS {
            let t0 = s as f64 / LINE_SUBDIVISIONS as f64;
            let t1 = (s + 1) as f64 / LINE_SUBDIVISIONS as f64;
            let tm = f64::midpoint(t0, t1);
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
              dashing: style.dashing.clone(),
              dash_offset: t0 * total_px,
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
      "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
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
      let dash_attr =
        dash3d_attr(edge.dashing.as_ref(), svg_width as f64, edge.dash_offset);
      svg.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"{:.1}\" stroke-linecap=\"round\"{}{}/>\n",
        ex0, ey0, ex1, ey1, edge.color, edge.width, opacity_attr, dash_attr
      ));
    };
    let mut ei = 0;
    let mut texture_ids = Vec::new();
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
      if let Some(tex) = &tri.texture {
        let corners = [(x0, y0), (x1, y1), (x2, y2)];
        if emit_textured_triangle(&mut svg, tex, corners, &mut texture_ids) {
          let edge_stroke = match tri.edge_color {
            Some((er, eg, eb)) => format!("rgb({er},{eg},{eb})"),
            None => "rgb(64,64,64)".to_string(),
          };
          let edge_width = tri.edge_width.unwrap_or(1.0);
          for (e, on_outline) in tri.boundary.iter().enumerate() {
            if !on_outline {
              continue;
            }
            let (ax, ay) = corners[e];
            let (bx, by) = corners[(e + 1) % 3];
            svg.push_str(&format!(
              "<line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"{edge_stroke}\" stroke-width=\"{edge_width}\"/>\n",
            ));
          }
          continue;
        }
      }
      let (r, g, b) = tri.color;
      let opacity_attr = if tri.opacity < 1.0 {
        format!(" opacity=\"{}\"", tri.opacity)
      } else {
        String::new()
      };
      // The outline colour: the default dark grey unless `EdgeForm[colour]`
      // named one. `Opacity` tints the face only — the outline is drawn by
      // the default `EdgeForm`, which stays opaque — so the outline of a
      // transparent face still shows. That is what makes
      // `{Opacity[0], Cuboid[…]}` a wireframe box.
      let edge_stroke = match tri.edge_color {
        Some((er, eg, eb)) => format!("rgb({er},{eg},{eb})"),
        None => "rgb(64,64,64)".to_string(),
      };
      let edge_width = tri.edge_width.unwrap_or(1.0);
      let fill_opacity_attr = if tri.opacity < 1.0 {
        format!(" fill-opacity=\"{}\"", tri.opacity)
      } else {
        String::new()
      };
      // The hairline stroke closes the anti-aliasing seam between
      // neighbouring triangles. Where the seam is an internal cut of a
      // fan-triangulated polygon it is stroked in the triangle's own
      // colour instead, so the face reads as one flat surface.
      if tri.boundary == [true; 3] {
        svg.push_str(&format!(
          "<polygon points=\"{x0:.1},{y0:.1} {x1:.1},{y1:.1} {x2:.1},{y2:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"{edge_stroke}\" stroke-width=\"{edge_width}\"{fill_opacity_attr}/>\n"
        ));
      } else {
        // A translucent triangle's own-colour hairline overlaps its
        // neighbours' and darkens every shared seam, tracing the
        // tessellation across what should be a smooth surface. Only an
        // opaque triangle needs it to close the anti-aliasing gap.
        let seam_stroke = if tri.opacity < 1.0 && tri.boundary == [false; 3] {
          "none".to_string()
        } else {
          format!("rgb({r},{g},{b})")
        };
        svg.push_str(&format!(
          "<polygon points=\"{x0:.1},{y0:.1} {x1:.1},{y1:.1} {x2:.1},{y2:.1}\" fill=\"rgb({r},{g},{b})\" stroke=\"{seam_stroke}\" stroke-width=\"0.5\"{opacity_attr}/>\n"
        ));
        let corners = [(x0, y0), (x1, y1), (x2, y2)];
        for (e, on_outline) in tri.boundary.iter().enumerate() {
          if !on_outline {
            continue;
          }
          let (ax, ay) = corners[e];
          let (bx, by) = corners[(e + 1) % 3];
          svg.push_str(&format!(
            "<line x1=\"{ax:.1}\" y1=\"{ay:.1}\" x2=\"{bx:.1}\" y2=\"{by:.1}\" stroke=\"{edge_stroke}\" stroke-width=\"{edge_width}\"/>\n",
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
        picture,
        style,
      } => {
        let fill_color = if let Some((r, g, b)) = style.color {
          format!("rgb({r},{g},{b})")
        } else {
          default_prim_color.to_string()
        };
        let (px, py) = project(*pos, &camera);
        let (sx, sy) = to_svg(px, py);
        if let Some((pic, w, h)) = picture {
          let x = sx - w / 2.0 - offset.0 * w / 2.0;
          let y = sy - h / 2.0 + offset.1 * h / 2.0;
          svg.push_str(&pic.replacen(
            "<svg ",
            &format!("<svg x=\"{x:.1}\" y=\"{y:.1}\" "),
            1,
          ));
          svg.push('\n');
          continue;
        }
        // The offset names which point of the label's box sits at the
        // projected point, so the box moves the other way — half its width
        // per unit across, half its height per unit up (and the vertical
        // sign flips, SVG counting y downwards).
        let text_w = *width_chars as f64 * font_size * 0.6;
        let x = sx - offset.0 * text_w / 2.0;
        let y = sy + offset.1 * font_size / 2.0;
        let lines = crate::functions::graphics::split_markup_lines(label);
        let line_height = font_size * 1.2;
        let mut body = String::new();
        if lines.len() == 1 {
          body.push_str(&lines[0]);
        } else {
          // The block of lines is centred on the anchor point.
          let first_dy = -(lines.len() as f64 - 1.0) * line_height / 2.0;
          for (i, line) in lines.iter().enumerate() {
            let dy = if i == 0 { first_dy } else { line_height };
            body.push_str(&format!(
              "<tspan x=\"{x:.1}\" dy=\"{dy:.1}\">{line}</tspan>"
            ));
          }
        }
        svg.push_str(&format!(
          "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"middle\" \
           dominant-baseline=\"middle\" font-family=\"sans-serif\" \
           font-size=\"{font_size:.1}\" fill=\"{fill_color}\">{body}</text>\n",
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
        let radius = style.point_size.map_or(3.0, |ps| {
          crate::functions::graphics::point_radius(ps, svg_width as f64)
        });
        for pt in points {
          let (sx, sy) =
            to_svg(project(*pt, &camera).0, project(*pt, &camera).1);
          svg.push_str(&format!(
            "<circle cx=\"{sx:.1}\" cy=\"{sy:.1}\" r=\"{radius:.2}\" fill=\"{fill_color}\"{opacity_attr}/>\n"
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
            format!("{sx:.1},{sy:.1}")
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
            "<polygon points=\"{sx2:.1},{sy2:.1} {bx1:.1},{by1:.1} {bx2:.1},{by2:.1}\" fill=\"{stroke_color}\"{opacity_attr}/>\n"
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
    draw_axes_on_box_with_ticks(
      &mut svg,
      &camera,
      &to_svg,
      &box_corners,
      (x3_min / axis_scale[0], x3_max / axis_scale[0]),
      (y3_min / axis_scale[1], y3_max / axis_scale[1]),
      (z3_min / axis_scale[2], z3_max / axis_scale[2]),
      &axes_labels,
      &tick_spec,
      axes_origin.as_ref().filter(|_| !show_box),
    );
  }

  svg.push_str("</svg>");
  let svg = with_prolog_epilog(svg, &args[1..], svg_width, svg_height);
  // A `PlotLabel` sets a title above the finished picture. `args[1..]` are
  // the options; `args[0]` is the content, which may itself be a `Rule`
  // (`Graphics3D[a -> b]` draws nothing but is legal) and must not be read
  // as one.
  let svg = with_plot_label(svg, &args[1..], svg_width, svg_height);
  Ok(crate::graphics3d_result_with_structure(svg, structure))
}

/// `Prolog -> g` / `Epilog -> g` on a `Graphics3D`: 2-D primitives drawn
/// under / over the finished 3-D picture. They live in the picture's scaled
/// coordinates, `{0, 0}` the lower-left corner and `{1, 1}` the upper-right,
/// so each is rendered as a 2-D `Graphics` over that unit square and nested
/// into the picture as an `<svg>` of the same size.
fn with_prolog_epilog(
  svg: String,
  opts: &[Expr],
  width: u32,
  height: u32,
) -> String {
  let layer = |name: &str| -> Option<String> {
    let content = opts.iter().find_map(|opt| match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name) => {
        Some(replacement.as_ref().clone())
      }
      _ => None,
    })?;
    let unit = Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into());
    let rule = |k: &str, v: Expr| Expr::Rule {
      pattern: Box::new(Expr::Identifier(k.to_string())),
      replacement: Box::new(v),
    };
    let overlay = crate::functions::graphics::graphics_ast(&[
      content,
      rule("PlotRange", Expr::List(vec![unit.clone(), unit].into())),
      rule("PlotRangePadding", Expr::Identifier("None".to_string())),
      rule(
        "ImageSize",
        Expr::List(
          vec![
            Expr::Integer(i128::from(width)),
            Expr::Integer(i128::from(height)),
          ]
          .into(),
        ),
      ),
      rule("Background", Expr::Identifier("None".to_string())),
    ])
    .ok()?;
    match &overlay {
      Expr::Graphics { svg, .. } => Some(svg.clone()),
      _ => None,
    }
  };
  let mut svg = svg;
  if let Some(under) = layer("Prolog")
    && let Some(start) = svg.find("<rect ")
    && let Some(len) = svg[start..].find("/>")
  {
    svg.insert_str(start + len + 2, &format!("\n{under}"));
  }
  if let Some(over) = layer("Epilog")
    && let Some(end) = svg.rfind("</svg>")
  {
    svg.insert_str(end, &format!("{over}\n"));
  }
  svg
}

/// `PlotRange -> {zmin, zmax}`: the vertical-axis-only clipping form shared
/// by every 3D height-field plot (`Plot3D`, `ListPlot3D`) alongside the
/// full per-axis form `parse_axis_ranges` handles.
fn parse_z_only_range(expr: &Expr) -> Option<(f64, f64)> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 2 {
    return None;
  }
  let lo = try_eval_to_f64(&evaluate_expr_to_expr(&items[0]).ok()?)?;
  let hi = try_eval_to_f64(&evaluate_expr_to_expr(&items[1]).ok()?)?;
  Some((lo, hi))
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

/// The three axis labels an `AxesLabel` value names. A lone label (or a
/// shorter list) labels the leading axes; `defaults` are the names
/// `Automatic` stands for.
fn parse_axes_labels(
  replacement: &Expr,
  defaults: Option<[&str; 3]>,
) -> [Option<String>; 3] {
  let value =
    evaluate_expr_to_expr(replacement).unwrap_or_else(|_| replacement.clone());
  let items: Vec<Expr> = match &value {
    Expr::List(items) => items.to_vec(),
    // A bare `Automatic` labels every axis, not just the first.
    Expr::Identifier(s) if s == "Automatic" => vec![value.clone(); 3],
    other => vec![other.clone()],
  };
  let mut labels: [Option<String>; 3] = [None, None, None];
  for (i, item) in items.iter().take(3).enumerate() {
    labels[i] = axis_label_markup(item, defaults.map(|d| d[i]));
  }
  labels
}

/// The SVG markup of one `AxesLabel` entry. `None` for `None` — the way an
/// axis asks to stay unlabelled. Everything else is typeset by the shared
/// label renderer, so a label written in the FrontEnd as linear syntax
/// (`"\!\(\*FractionBox[\(d\[Theta]\), \(dt\)]\) (rad/s)"`) draws as the
/// fraction it stands for.
fn axis_label_markup(expr: &Expr, default: Option<&str>) -> Option<String> {
  match expr {
    Expr::Identifier(s) if s == "None" => None,
    // `Automatic` asks for the plot's default label: the variable's name
    // for a function plot, nothing for a bare `Graphics3D`.
    Expr::Identifier(s) if s == "Automatic" => default.map(|name| {
      crate::functions::graphics::expr_to_svg_markup(&Expr::Identifier(
        name.to_string(),
      ))
    }),
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

/// The typeset label of the `PlotLabel` an option list carries, or `None`
/// when it carries none (`PlotLabel -> None` included). Any expression can
/// label a plot — a string, a `Style[…]`, a `Grid`/`Column` that stacks
/// several lines — so it is typeset the way every other label is rather
/// than printed.
fn plot_label_styled(
  opts: &[Expr],
) -> Option<crate::functions::chart::StyledLabel> {
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
    if let Some(label) = crate::functions::chart::parse_styled_label(&value) {
      return Some(label);
    }
  }
  None
}

/// Parse a `Background -> colour` value. The 3D plotting heads hold their
/// arguments, so the value may still be an unevaluated expression such as
/// `RGBColor[r, g, b]` over control variables; evaluate it first.
fn parse_background_color(
  expr: &Expr,
) -> Option<crate::functions::graphics::Color> {
  let val = evaluate_expr_to_expr(expr).unwrap_or_else(|_| expr.clone());
  crate::functions::graphics::parse_color(&val)
}

/// Repaint the picture's backdrop (the first `<rect>` the 3-D renderers
/// emit) with an explicit `Background -> color` option.
fn with_background_option(svg: String, opts: &[Expr]) -> String {
  let color = opts.iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Background") => {
      parse_background_color(replacement)
    }
    _ => None,
  });
  let Some(c) = color else {
    return svg;
  };
  let channel = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
  let fill = format!("rgb({},{},{})", channel(c.r), channel(c.g), channel(c.b));
  let Some(start) = svg.find("<rect ") else {
    return svg;
  };
  let Some(len) = svg[start..].find("/>") else {
    return svg;
  };
  let tag = &svg[start..start + len];
  let Some(f) = tag.find("fill=\"") else {
    return svg;
  };
  let value_start = start + f + 6;
  let Some(value_len) = svg[value_start..].find('"') else {
    return svg;
  };
  format!(
    "{}{}{}",
    &svg[..value_start],
    fill,
    &svg[value_start + value_len..]
  )
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
  let svg = with_background_option(svg, opts);
  let Some(styled) = plot_label_styled(opts) else {
    return svg;
  };
  // A `FontSize` directive on the label resizes it, same as a 2-D title;
  // otherwise it keeps the default 16pt.
  let font_size = styled.font_size.unwrap_or(16.0);
  // A stacked title (`Grid`/`Column`) needs one more line height per extra
  // row, on top of the strip a single-line title reserves.
  let line_height = font_size * 1.2;
  let extra_lines = styled.extra_line_count() as f64;
  let strip = PLOT_LABEL_STRIP + (extra_lines * line_height).round() as u32;
  let total = height + strip;
  // The strip is painted in the picture's own background so the two read
  // as one canvas — an explicit `Background` reaches it too.
  let bg = opts
    .iter()
    .find_map(|opt| match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Background") => {
        parse_background_color(replacement)
      }
      _ => None,
    }).map_or_else(|| {
      let (theme_bg, _, _, _, _) = crate::functions::plot::plot_theme();
      (theme_bg.0, theme_bg.1, theme_bg.2)
    }, |c| {
      (
        (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
      )
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
  let fill = styled
    .color
    .as_ref()
    .map_or_else(|| "#333333".to_string(), |c| c.to_svg_rgb());
  let mut style_attrs = String::new();
  if styled.bold {
    style_attrs.push_str(" font-weight=\"bold\"");
  }
  if styled.italic {
    style_attrs.push_str(" font-style=\"italic\"");
  }
  let label = styled.svg_scaled_stacked(1.0, cx, line_height);
  format!(
    "<svg {sizing} xmlns=\"http://www.w3.org/2000/svg\">\n\
     <rect width=\"{width}\" height=\"{total}\" fill=\"rgb({},{},{})\"/>\n\
     <text x=\"{cx:.1}\" y=\"17\" text-anchor=\"middle\" \
     font-family=\"sans-serif\" font-size=\"{font_size:.0}\" fill=\"{fill}\"{style_attrs}>{label}</text>\n\
     <g transform=\"translate(0,{strip})\">{svg}</g>\n\
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

/// True when `data` is a list of `{x, y, z}` triples of numbers that fill a
/// complete tensor grid. Such a list is gridded point data, not a z-matrix
/// that merely happens to have three columns.
fn is_xyz_point_list(data: &[Expr]) -> bool {
  let mut points = Vec::with_capacity(data.len());
  for row in data {
    match row {
      Expr::List(r) if r.len() == 3 => {
        match (
          try_eval_to_f64(&r[0]),
          try_eval_to_f64(&r[1]),
          try_eval_to_f64(&r[2]),
        ) {
          (Some(x), Some(y), Some(z))
            if x.is_finite() && y.is_finite() && z.is_finite() =>
          {
            points.push((x, y, z));
          }
          _ => return false,
        }
      }
      _ => return false,
    }
  }
  resample_tensor_grid(&points).is_some()
}

/// If `points` fill a complete tensor grid (every combination of the unique
/// x and y values occurs), bilinearly resample it onto a uniform square grid.
/// Returns `grid[i][j]` with `i` along x and `j` along y.
fn resample_tensor_grid(points: &[(f64, f64, f64)]) -> Option<Vec<Vec<f64>>> {
  fn uniq(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    v
  }
  let xs = uniq(points.iter().map(|p| p.0).collect());
  let ys = uniq(points.iter().map(|p| p.1).collect());
  if xs.len() < 2 || ys.len() < 2 || xs.len() * ys.len() != points.len() {
    return None;
  }
  let find = |v: &[f64], t: f64| {
    v.binary_search_by(|a| {
      if (*a - t).abs() < 1e-12 {
        std::cmp::Ordering::Equal
      } else {
        a.partial_cmp(&t).unwrap()
      }
    })
    .ok()
  };
  let mut z = vec![vec![f64::NAN; ys.len()]; xs.len()];
  for &(x, y, v) in points {
    let (i, j) = (find(&xs, x)?, find(&ys, y)?);
    if z[i][j].is_finite() {
      return None;
    }
    z[i][j] = v;
  }
  let n = xs.len().max(ys.len()).clamp(2, 60);
  // Index of the cell containing `t` and the fraction within it.
  let locate = |v: &[f64], t: f64| {
    let k = v.partition_point(|a| *a <= t).clamp(1, v.len() - 1) - 1;
    (k, ((t - v[k]) / (v[k + 1] - v[k])).clamp(0.0, 1.0))
  };
  let (x0, x1) = (xs[0], xs[xs.len() - 1]);
  let (y0, y1) = (ys[0], ys[ys.len() - 1]);
  let mut grid = vec![vec![0.0; n]; n];
  for (i, row) in grid.iter_mut().enumerate() {
    let (a, fa) = locate(&xs, x0 + (x1 - x0) * i as f64 / (n - 1) as f64);
    for (j, cell) in row.iter_mut().enumerate() {
      let (b, fb) = locate(&ys, y0 + (y1 - y0) * j as f64 / (n - 1) as f64);
      *cell = z[a][b] * (1.0 - fa) * (1.0 - fb)
        + z[a + 1][b] * fa * (1.0 - fb)
        + z[a][b + 1] * (1.0 - fa) * fb
        + z[a + 1][b + 1] * fa * fb;
    }
  }
  Some(grid)
}

/// The surfaces of `ListPlot3D[{m1, m2, …}]`: a list of at least two
/// height matrices (rectangular arrays of scalars). Rows of three numbers
/// stay a single set of `{x, y, z}` points, as before.
fn list_plot3d_datasets(data: &Expr) -> Option<Vec<Expr>> {
  let Expr::List(outer) = data else {
    return None;
  };
  let is_matrix = |e: &Expr| match e {
    Expr::List(rows) if !rows.is_empty() => {
      let mut width = None;
      rows.iter().all(|r| match r {
        Expr::List(row)
          if !row.is_empty()
            && row.iter().all(|v| !matches!(v, Expr::List(_))) =>
        {
          *width.get_or_insert(row.len()) == row.len() && row.len() != 3
        }
        _ => false,
      })
    }
    _ => false,
  };
  (outer.len() >= 2 && outer.iter().all(is_matrix)).then(|| outer.to_vec())
}

/// `ListPlot3D[{m1, m2, …}, opts]`: every dataset is plotted on its own and
/// the surfaces are combined with `Show`, each in the colour its
/// `PlotStyle` entry gives (cycling), or the default plot colours.
fn list_plot3d_multi(
  datasets: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let style_items: Vec<Expr> = args[1..]
    .iter()
    .find_map(|opt| match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle") => {
        Some(plot_style_items(replacement, datasets.len()))
      }
      _ => None,
    })
    .unwrap_or_default();
  let other_opts: Vec<Expr> = args[1..]
    .iter()
    .filter(|opt| {
      !matches!(opt, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle"))
    })
    .cloned()
    .collect();
  let mut surfaces = Vec::with_capacity(datasets.len());
  for (i, data) in datasets.iter().enumerate() {
    let style = if style_items.is_empty() {
      let (r, g, b) = PLOT_COLORS[i % PLOT_COLORS.len()];
      call(
        "RGBColor",
        vec![
          Expr::Real(r as f64 / 255.0),
          Expr::Real(g as f64 / 255.0),
          Expr::Real(b as f64 / 255.0),
        ],
      )
    } else {
      style_items[i % style_items.len()].clone()
    };
    let mut sub_args = vec![
      data.clone(),
      crate::helpers::rule_expr(Expr::Identifier("PlotStyle".into()), style),
    ];
    sub_args.extend(other_opts.iter().cloned());
    surfaces.push(list_plot3d_ast(&sub_args)?);
  }
  // The combined picture keeps a single surface's box and axes unless the
  // call sets its own.
  let given = |name: &str| {
    other_opts.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name))
    })
  };
  if !given("Axes") {
    surfaces.push(crate::helpers::rule_expr(
      Expr::Identifier("Axes".into()),
      Expr::Identifier("True".into()),
    ));
  }
  if !given("BoxRatios") {
    surfaces.push(crate::helpers::rule_expr(
      Expr::Identifier("BoxRatios".into()),
      Expr::List(
        vec![Expr::Integer(1), Expr::Integer(1), Expr::Real(0.4)].into(),
      ),
    ));
  }
  evaluate_expr_to_expr(&call("Show", surfaces))
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
  let mut mesh_mode = MeshMode::Default;
  // `ColorFunction -> "SouthwestColors"` (or `ColorData["…"]`): colors the
  // surface by height through the same named gradient `ColorFunction` uses
  // for `Plot3D`-family density/contour plots, instead of the fixed
  // blue-green-orange default.
  let mut color_function: Option<String> = None;
  // `PlotRange -> {zmin, zmax}`: clips (and re-normalizes the height color
  // of) the surface to that vertical range instead of the raw data extent,
  // same as `Plot3D`.
  let mut z_clip: Option<(f64, f64)> = None;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
  // `PlotStyle`: the surface's colour (and opacity, specularity, …); empty
  // keeps the height gradient.
  let mut plot_styles: Vec<StyleState3D> = Vec::new();

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
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "ColorFunction" => {
          color_function =
            crate::functions::field_plot::color_function_scheme_name(
              replacement,
            );
        }
        Expr::Identifier(name) if name == "PlotRange" => {
          z_clip = parse_z_only_range(replacement);
        }
        Expr::Identifier(name) if name == "AxesLabel" => {
          axes_labels = parse_axes_labels(replacement, None);
        }
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_styles = parse_plot_style_3d(replacement, 1);
        }
        _ => {}
      }
    }
  }

  // The color a normalized height `t` (0 at the low end, 1 at the high end)
  // is drawn in: a recognized `ColorFunction` scheme, or the surface's
  // default height gradient when none was given (or it named a scheme Woxi
  // does not have swatches for).
  let color_at = |t: f64| -> (u8, u8, u8) {
    match &color_function {
      Some(name) => {
        crate::functions::field_plot::apply_named_color_function(name, t)
      }
      None => height_color(t),
    }
  };

  // Evaluate the data argument
  let evaled_data = evaluate_expr_to_expr(data)?;

  // A list of height matrices is several surfaces, drawn together.
  if let Some(datasets) = list_plot3d_datasets(&evaled_data) {
    return list_plot3d_multi(&datasets, args);
  }
  let surface_style = plot_style_for_surface(&plot_styles, 0);
  let view_dir = {
    let camera = Camera::default();
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };

  // Determine the data format and build a grid
  let (grid, rows, cols, x_min, x_max, y_min, y_max) = match &evaled_data {
    Expr::List(outer) if !outer.is_empty() => {
      // Check first element to determine format
      match &outer[0] {
        // 2D matrix format: {{z11, z12, ...}, {z21, z22, ...}, ...}
        Expr::List(first_row)
          if !first_row.is_empty()
            && first_row.iter().all(|e| !matches!(e, Expr::List(_)))
            && !is_xyz_point_list(outer) =>
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
          // An array of rows of points (`Table[{x, y, z}, {y, …}, {x, …}]`)
          // is the same point set, one row at a time.
          let flattened: Vec<&Expr> = outer
            .iter()
            .flat_map(|item| match item {
              Expr::List(row)
                if row
                  .iter()
                  .all(|p| matches!(p, Expr::List(c) if c.len() == 3)) =>
              {
                row.iter().collect::<Vec<_>>()
              }
              _ => vec![item],
            })
            .collect();
          for item in flattened {
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

          // Points on a complete (possibly non-uniform) tensor grid, e.g.
          // `Flatten[Table[{x, y, f}, {x, xs}, {y, ys}], 1]`, are resampled
          // bilinearly onto a uniform grid instead of being binned.
          let (grid, grid_n) = if let Some(grid) = resample_tensor_grid(&points)
          {
            let n = grid.len();
            (grid, n)
          } else {
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

            (grid, grid_n)
          };

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

  // An explicit `PlotRange` overrides the data extent; values outside it
  // are clamped (matching `Plot3D`) rather than stretching the axis.
  let (z_lo, z_hi) = z_clip.unwrap_or((z_min, z_max));
  let z_range = if (z_hi - z_lo).abs() < 1e-15 {
    1.0
  } else {
    z_hi - z_lo
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
      let nz = |z: f64| -> f64 {
        ((z.clamp(z_lo, z_hi) - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE
      };

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

        let avg = ((z00.clamp(z_lo, z_hi) - z_lo) / z_range
          + (z10.clamp(z_lo, z_hi) - z_lo) / z_range
          + (z01.clamp(z_lo, z_hi) - z_lo) / z_range)
          / 3.0;
        let base_color = color_at(avg);
        let normal = triangle_normal(v0, v1, v2);
        let (color, opacity) =
          shade_facet(base_color, surface_style, normal, view_dir);

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
          edge_width: None,
          projected: [p0, p1, p2],
          depth: depth(center, &camera),
          color,
          opacity,
          texture: None,
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

        let avg = ((z11.clamp(z_lo, z_hi) - z_lo) / z_range
          + (z01.clamp(z_lo, z_hi) - z_lo) / z_range
          + (z10.clamp(z_lo, z_hi) - z_lo) / z_range)
          / 3.0;
        let base_color = color_at(avg);
        let normal = triangle_normal(v0, v1, v2);
        let (color, opacity) =
          shade_facet(base_color, surface_style, normal, view_dir);

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
          edge_width: None,
          projected: [p0, p1, p2],
          depth: depth(center, &camera),
          color,
          opacity,
          texture: None,
        });
      }
    }
  }

  // Data spanning no area (every point on one line, as in two copies of
  // the same collinear row) draws the empty box, as in wolframscript.

  let (z_axis_min, z_axis_max) = if (z_lo - z_hi).abs() < 1e-15 {
    (z_lo - 0.5, z_hi + 0.5)
  } else {
    (z_lo, z_hi)
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
    mesh_mode,
    true, // show_axes: always show axes for list_plot3d
    [1.0, 1.0, Z_SCALE],
    &axes_labels,
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  // ── Symbolic structure: Graphics3D[GraphicsComplex[points, {…}]] ──
  // Mirrors `plot3d_ast`'s own structure above: without one, `Show` cannot
  // merge this surface with another graphic's primitives (a `Graphics3D`
  // marker drawn alongside it, the Demonstrations "surface plus pointer"
  // idiom) — the two would render as separately-projected flat pictures,
  // so a `ViewPoint`/`BoxRatios` given to *this* call never turned the
  // other layer with it, and one given to `Show` itself (or to that other
  // layer) never turned this surface, whose own picture was already baked
  // above at a fixed default camera. Emitting the grid in its own index
  // coordinates — colored the way the standalone render colors it, with
  // this call's options carried along so `Show` still sees them — lets it
  // re-render everything together under one shared camera instead.
  let mut index_of: Vec<Vec<Option<usize>>> = vec![vec![None; cols]; rows];
  let mut point_exprs: Vec<Expr> = Vec::new();
  for (i, row) in grid.iter().enumerate() {
    for (j, &z) in row.iter().enumerate() {
      if !z.is_finite() {
        continue;
      }
      index_of[i][j] = Some(point_exprs.len());
      point_exprs.push(Expr::List(
        vec![
          Expr::Real(i as f64),
          Expr::Real(j as f64),
          Expr::Real(z.clamp(z_lo, z_hi)),
        ]
        .into(),
      ));
    }
  }
  let mut complex_content: Vec<Expr> = Vec::new();
  // The sampling quads are far finer than the mesh a surface shows, so
  // their own outlines are suppressed and the mesh is drawn as lines below
  // — otherwise a shown surface turns into a wireframe.
  if !matches!(mesh_mode, MeshMode::All) {
    complex_content.push(call0("EdgeForm"));
  }
  complex_content.extend(surface_opacity_directive(&plot_styles, 0));
  for i in 0..rows.saturating_sub(1) {
    for j in 0..cols.saturating_sub(1) {
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
      let (cr, cg, cb) = surface_style
        .and_then(|st| st.color)
        .unwrap_or_else(|| color_at(avg_z_norm));
      complex_content.push(Expr::List(
        vec![
          call(
            "RGBColor",
            vec![
              Expr::Real(cr as f64 / 255.0),
              Expr::Real(cg as f64 / 255.0),
              Expr::Real(cb as f64 / 255.0),
            ],
          ),
          call1(
            "Polygon",
            Expr::List(
              [a, b, c, d]
                .iter()
                .map(|&k| Expr::Integer(k as i128 + 1))
                .collect::<Vec<_>>()
                .into(),
            ),
          ),
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
          vec![Expr::Integer(a as i128 + 1), Expr::Integer(b as i128 + 1)]
            .into(),
        ));
      }
    };
    for j in (0..cols).step_by(MESH_STEP) {
      for i in 0..rows.saturating_sub(1) {
        push_segment(index_of[i][j], index_of[i + 1][j]);
      }
    }
    for i in (0..rows).step_by(MESH_STEP) {
      for j in 0..cols.saturating_sub(1) {
        push_segment(index_of[i][j], index_of[i][j + 1]);
      }
    }
    if !segments.is_empty() {
      complex_content.push(Expr::List(
        vec![
          call1("Opacity", Expr::Real(0.63)),
          call(
            "RGBColor",
            vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
          ),
          call1("AbsoluteThickness", Expr::Real(0.5)),
          call1("Line", Expr::List(segments.into())),
        ]
        .into(),
      ));
    }
  }
  let complex = call(
    "GraphicsComplex",
    vec![
      Expr::List(point_exprs.into()),
      Expr::List(complex_content.into()),
    ],
  );
  let mut structure_args = vec![complex];
  structure_args.extend(args[1..].iter().cloned());
  let structure = call("Graphics3D", structure_args);

  Ok(crate::plot3d_result_with_structure(svg, structure))
}

// ── RevolutionPlot3D implementation ──────────────────────────────────

/// Evaluate a single-variable expression at a given value.
fn evaluate_at_t(body: &Expr, tvar: &str, tval: f64) -> Option<f64> {
  let sub = substitute_var(body, tvar, &Expr::Real(tval));
  let result = evaluate_expr_to_expr(&sub).ok()?;
  if let Some(v) = try_eval_to_f64_lenient(&result) {
    return Some(v);
  }
  // See evaluate_at_t_theta: body may be a bound variable that only
  // resolved to a function of tvar during evaluation.
  let sub = substitute_var(&result, tvar, &Expr::Real(tval));
  let result = evaluate_expr_to_expr(&sub).ok()?;
  try_eval_to_f64_lenient(&result)
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
  if let Some(v) = try_eval_to_f64_lenient(&result) {
    return Some(v);
  }
  // The body may reference a variable (e.g. `r` holding a `SphericalHarmonicY`
  // combination) that only resolved to a function of tvar/theta_var during
  // evaluation — after the substitution above already ran. Substitute into
  // the evaluated form and evaluate once more.
  let sub1 = substitute_var(&result, tvar, &Expr::Real(tval));
  let sub2 = substitute_var(&sub1, theta_var, &Expr::Real(theta_val));
  let result = evaluate_expr_to_expr(&sub2).ok()?;
  try_eval_to_f64_lenient(&result)
}

/// `RevolutionAxis -> {x, y, z}` (or `"X"`, `"Y"`, `"Z"`) as a unit vector.
fn parse_revolution_axis(spec: &Expr) -> Option<[f64; 3]> {
  let v: [f64; 3] = match spec {
    Expr::String(s) | Expr::Identifier(s) => match s.as_str() {
      "X" => [1.0, 0.0, 0.0],
      "Y" => [0.0, 1.0, 0.0],
      "Z" => [0.0, 0.0, 1.0],
      _ => return None,
    },
    Expr::List(items) if items.len() == 3 => {
      let mut v = [0.0; 3];
      for (slot, item) in v.iter_mut().zip(items.iter()) {
        *slot = try_eval_to_f64(&evaluate_expr_to_expr(item).ok()?)?;
      }
      v
    }
    _ => return None,
  };
  let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  if len < 1e-12 || !len.is_finite() {
    return None;
  }
  Some([v[0] / len, v[1] / len, v[2] / len])
}

/// Rotate `p` by `theta` about the unit vector `k` (Rodrigues' formula).
fn rotate_about_axis(p: [f64; 3], k: [f64; 3], theta: f64) -> [f64; 3] {
  let (s, c) = theta.sin_cos();
  let dot = k[0] * p[0] + k[1] * p[1] + k[2] * p[2];
  let cross = [
    k[1] * p[2] - k[2] * p[1],
    k[2] * p[0] - k[0] * p[2],
    k[0] * p[1] - k[1] * p[0],
  ];
  [
    p[0] * c + cross[0] * s + k[0] * dot * (1.0 - c),
    p[1] * c + cross[1] * s + k[1] * dot * (1.0 - c),
    p[2] * c + cross[2] * s + k[2] * dot * (1.0 - c),
  ]
}

/// Map `v` from `[lo, hi]` onto `[-1, 1]` (0 for a degenerate extent).
fn norm_extent(v: f64, lo: f64, hi: f64) -> f64 {
  if (hi - lo).abs() < 1e-15 {
    0.0
  } else {
    (v - lo) / (hi - lo) * 2.0 - 1.0
  }
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

  // A list of profile curves `{{r1, z1}, {r2, z2}, …}` revolves each curve
  // into its own surface, all drawn in one scene.
  let bodies: Vec<&Expr> = match &args[0] {
    Expr::List(items)
      if !items.is_empty()
        && items.iter().all(
          |it| matches!(it, Expr::List(c) if c.len() == 2 || c.len() == 3),
        ) =>
    {
      items.iter().collect()
    }
    other => vec![other],
  };

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
  let mut mesh_mode = MeshMode::Default;
  let mut show_axes = true;
  let mut z_clip: Option<(f64, f64)> = None;
  let mut plot_style: Option<Expr> = None;
  let mut axis: [f64; 3] = [0.0, 0.0, 1.0];
  let mut color_fn = FacetColorFunction::default();

  for opt in &args[opt_start..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      color_fn.read_option(pattern, replacement);
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_style = Some(replacement.as_ref().clone());
        }
        Expr::Identifier(name) if name == "RevolutionAxis" => {
          if let Some(a) = parse_revolution_axis(replacement) {
            axis = a;
          }
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
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "PlotRange" => {
          // `{zmin, zmax}` or the full `{{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}`
          let z_items = match replacement.as_ref() {
            Expr::List(items) if items.len() == 2 => Some(items.clone()),
            Expr::List(items) if items.len() == 3 => match &items[2] {
              Expr::List(z) if z.len() == 2 => Some(z.clone()),
              _ => None,
            },
            _ => None,
          };
          if let Some(items) = z_items {
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

  // Determine if body is {r_expr, z_expr} (parametric), {fx_expr, fy_expr,
  // fz_expr} (a 3D curve swept around the z axis), or scalar f(t). The
  // 3-coordinate form is the Demonstrations idiom for a curve built with
  // `Sqrt`/trig components that already sits off the xz-plane; sweeping it
  // rotates the (fx, fy) vector by theta instead of assuming fy = 0.
  let has_theta = theta_var.is_some();

  let camera = Camera::default();
  let n_t = GRID_N;
  let n_theta = GRID_N;
  let t_step = (t_max - t_min) / n_t as f64;
  let theta_step = (theta_max - theta_min) / n_theta as f64;

  // Sample each surface grid: grid[i][j] = (x, y, z)
  let mut grids: Vec<Vec<Vec<Option<Point3D>>>> = Vec::new();
  let mut global_z_min = f64::INFINITY;
  let mut global_z_max = f64::NEG_INFINITY;
  let mut global_r_max: f64 = 0.0;
  let (mut global_x_min, mut global_x_max) = (f64::INFINITY, f64::NEG_INFINITY);
  let (mut global_y_min, mut global_y_max) = (f64::INFINITY, f64::NEG_INFINITY);
  // The sweep is about the z axis unless `RevolutionAxis` says otherwise;
  // only then do x and y need their own extents when normalizing.
  let axis_is_z = axis[0] == 0.0 && axis[1] == 0.0;

  for body in &bodies {
    let parametric_len: Option<usize> = match body {
      Expr::List(items) if items.len() == 2 || items.len() == 3 => {
        Some(items.len())
      }
      _ => None,
    };
    // A scalar body that ignores θ is the plain profile `z = f(t)` (the θ
    // iterator only sets the sweep angle); one that varies with θ gives the
    // radius as a function of both variables.
    let scalar_is_profile = match (&theta_var, parametric_len) {
      (Some(theta_v), None) => {
        let tmid = f64::midpoint(t_min, t_max);
        evaluate_at_t_theta(body, &tvar, tmid, theta_v, theta_min)
          == evaluate_at_t_theta(body, &tvar, tmid, theta_v, theta_max)
      }
      _ => false,
    };
    let mut grid: Vec<Vec<Option<Point3D>>> =
      vec![vec![None; n_theta + 1]; n_t + 1];
    for i in 0..=n_t {
      let tval = t_min + i as f64 * t_step;
      for j in 0..=n_theta {
        let theta = theta_min + j as f64 * theta_step;

        let (fx, fy, z) = if has_theta {
          let theta_v = theta_var.as_ref().unwrap();
          if let Some(n) = parametric_len {
            if let Expr::List(items) = body {
              let fx_val =
                evaluate_at_t_theta(&items[0], &tvar, tval, theta_v, theta);
              let fy_val = if n == 3 {
                evaluate_at_t_theta(&items[1], &tvar, tval, theta_v, theta)
              } else {
                Some(0.0)
              };
              let fz_val =
                evaluate_at_t_theta(&items[n - 1], &tvar, tval, theta_v, theta);
              match (fx_val, fy_val, fz_val) {
                (Some(fx), Some(fy), Some(fz))
                  if fx.is_finite() && fy.is_finite() && fz.is_finite() =>
                {
                  (fx, fy, fz)
                }
                _ => continue,
              }
            } else {
              continue;
            }
          } else {
            match evaluate_at_t_theta(body, &tvar, tval, theta_v, theta) {
              Some(z) if z.is_finite() && scalar_is_profile => (tval, 0.0, z),
              // Scalar: r = f(t, θ), z = t
              Some(r) if r.is_finite() => (r, 0.0, tval),
              _ => continue,
            }
          }
        } else if let Some(n) = parametric_len {
          if let Expr::List(items) = body {
            let fx_val = evaluate_at_t(&items[0], &tvar, tval);
            let fy_val = if n == 3 {
              evaluate_at_t(&items[1], &tvar, tval)
            } else {
              Some(0.0)
            };
            let fz_val = evaluate_at_t(&items[n - 1], &tvar, tval);
            match (fx_val, fy_val, fz_val) {
              (Some(fx), Some(fy), Some(fz))
                if fx.is_finite() && fy.is_finite() && fz.is_finite() =>
              {
                (fx, fy, fz)
              }
              _ => continue,
            }
          } else {
            continue;
          }
        } else {
          // Scalar f(t): revolve (t, f(t)) → r = t, z = f(t)
          match evaluate_at_t(body, &tvar, tval) {
            Some(z) if z.is_finite() => (tval, 0.0, z),
            _ => continue,
          }
        };

        // Rotate the (fx, fy) vector by theta around the z axis; fy = 0
        // reduces to the plain r·cos/r·sin sweep the 2-coordinate and scalar
        // forms already used.
        let [x, y, z] = rotate_about_axis([fx, fy, z], axis, theta);

        grid[i][j] = Some(Point3D { x, y, z });

        global_x_min = global_x_min.min(x);
        global_x_max = global_x_max.max(x);
        global_y_min = global_y_min.min(y);
        global_y_max = global_y_max.max(y);
        global_z_min = global_z_min.min(z);
        global_z_max = global_z_max.max(z);
        global_r_max = global_r_max.max((x * x + y * y).sqrt());
      }
    }
    grids.push(grid);
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
  let surfaces: Vec<Expr> = grids
    .iter()
    .map(|grid| {
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
      // The plot's own picture draws no outline on the sampling quads, so
      // the extracted surface must not pick up the default polygon edge
      // when re-rendered inside a `Graphics3D` (e.g. `plot[[1]]`).
      let mut content: Vec<Expr> = vec![call0("EdgeForm")];
      // `PlotStyle -> Opacity[…]` belongs to the surface, so a caller that
      // lifts it out of the plot keeps the translucency it asked for.
      match &plot_style {
        Some(Expr::List(items)) => content.extend(items.iter().cloned()),
        Some(other) => content.push(other.clone()),
        None => {}
      }
      content.push(call1("Polygon", Expr::List(quads.into())));
      call(
        "GraphicsComplex",
        vec![Expr::List(point_exprs.into()), Expr::List(content.into())],
      )
    })
    .collect();
  let structure = call1(
    "Graphics3D",
    if surfaces.len() == 1 {
      surfaces.into_iter().next().unwrap()
    } else {
      Expr::List(surfaces.into())
    },
  );

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

  // `RevolutionPlot3D` draws a single surface, so `PlotStyle` is one style
  // rather than a per-surface list.
  let render_style: Option<StyleState3D> = plot_style.as_ref().map(|ps| {
    let mut style = StyleState3D::default();
    apply_3d_directive(ps, &mut style);
    style
  });
  // Direction from the scene towards the viewer, used to place a
  // `Specularity` highlight from `PlotStyle`.
  let view_dir = {
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };

  let nz = |z: f64| -> f64 {
    let cz = z.clamp(z_lo, z_hi);
    ((cz - z_lo) / z_range) * 2.0 * Z_SCALE - Z_SCALE
  };

  // The style `ColorFunction` gives the facet with corners `ps`, evaluated
  // at the facet's centre in world coordinates; it takes precedence over
  // `PlotStyle`.
  let x_range = if axis_is_z {
    (-global_r_max, global_r_max)
  } else {
    (global_x_min, global_x_max)
  };
  let y_range = if axis_is_z {
    (-global_r_max, global_r_max)
  } else {
    (global_y_min, global_y_max)
  };
  let facet_style = |ps: [Point3D; 3]| -> Option<StyleState3D> {
    color_fn.function.as_ref()?;
    let mid = |f: &dyn Fn(&Point3D) -> f64| ps.iter().map(f).sum::<f64>() / 3.0;
    color_fn.style_at(
      [
        mid(&|p| p.x),
        mid(&|p| p.y),
        mid(&|p| p.z.clamp(z_lo, z_hi)),
      ],
      [x_range, y_range, (z_lo, z_hi)],
    )
  };

  // Build triangles
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for grid in &grids {
    for i in 0..n_t {
      for j in 0..n_theta {
        let p00 = grid[i][j];
        let p10 = grid[i + 1][j];
        let p01 = grid[i][j + 1];
        let p11 = grid[i + 1][j + 1];

        let normalize = |p: Point3D| -> Point3D {
          if axis_is_z {
            Point3D {
              x: p.x / r_scale,
              y: p.y / r_scale,
              z: nz(p.z),
            }
          } else {
            Point3D {
              x: norm_extent(p.x, global_x_min, global_x_max),
              y: norm_extent(p.y, global_y_min, global_y_max),
              z: nz(p.z),
            }
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
          let default_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let cf_style = facet_style([pp00, pp10, pp01]);
          let (color, opacity) = shade_facet(
            default_color,
            cf_style.as_ref().or(render_style.as_ref()),
            normal,
            view_dir,
          );

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
            edge_width: None,
            projected: [proj0, proj1, proj2],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
          });
        }

        // Triangle 2: p11, p01, p10
        if let (Some(pp11), Some(pp01), Some(pp10)) = (p11, p01, p10) {
          let v0 = normalize(pp11);
          let v1 = normalize(pp01);
          let v2 = normalize(pp10);

          let avg_z_norm =
            (z_norm_of(pp11) + z_norm_of(pp01) + z_norm_of(pp10)) / 3.0;
          let default_color = height_color(avg_z_norm);
          let normal = triangle_normal(v0, v1, v2);
          let cf_style = facet_style([pp11, pp01, pp10]);
          let (color, opacity) = shade_facet(
            default_color,
            cf_style.as_ref().or(render_style.as_ref()),
            normal,
            view_dir,
          );

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
            edge_width: None,
            projected: [proj0, proj1, proj2],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
          });
        }
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
    x_range,
    y_range,
    (z_axis_min, z_axis_max),
    svg_width,
    svg_height,
    full_width,
    mesh_mode,
    show_axes,
    [1.0, 1.0, Z_SCALE],
    &[None, None, None],
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::plot3d_result_with_structure(svg, structure))
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

  let (xvar, x_min, x_max) =
    parse_iterator_generic(&args[1], "RegionPlot3D", "first")?;
  let (yvar, y_min, y_max) =
    parse_iterator_generic(&args[2], "RegionPlot3D", "second")?;
  let (zvar, z_min, z_max) =
    parse_iterator_generic(&args[3], "RegionPlot3D", "third")?;
  // `cond[n] = x < 2 y + z; RegionPlot3D[cond[3], …]` — the head holds its
  // arguments, so a condition named indirectly is read through its value.
  let resolved = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[xvar.as_str(), yvar.as_str(), zvar.as_str()],
  );
  let body = resolved.as_ref().unwrap_or(&args[0]);

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut mesh_mode = MeshMode::Default;
  let mut show_axes = true;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
  let mut color_fn = FacetColorFunction::default();

  for opt in &args[4..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      color_fn.read_option(pattern, replacement);
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "AxesLabel" => {
          axes_labels = parse_axes_labels(replacement, None);
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
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
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
  // Data-space counterparts of `nx`/`ny`/`nz`, kept alongside the rendering
  // scale so the surface's symbolic `structure` (see below) carries real
  // `{x, y, z}` coordinates rather than the normalized render-space ones —
  // the same convention `RevolutionPlot3D`'s structure follows, so
  // `First[RegionPlot3D[…]]` can be `Translate`d/`Rotate`d and recombined
  // with other data-space primitives.
  let wx = |i: usize| -> f64 { x_min + i as f64 * x_step };
  let wy = |j: usize| -> f64 { y_min + j as f64 * y_step };
  let wz = |k: usize| -> f64 { z_min + k as f64 * z_step };

  let camera = Camera::default();
  let mut all_triangles: Vec<Triangle> = Vec::new();
  // One world-space quad per face emitted below, mirrored 1:1 against the
  // (normalized, screen-projected) triangles pushed into `all_triangles`.
  let mut world_quads: Vec<[Point3D; 4]> = Vec::new();
  // The `ColorFunction` style of each of those quads.
  let mut quad_styles: Vec<Option<StyleState3D>> = Vec::new();

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

          // The same face quad in data-space coordinates, for `structure`
          // (see `world_quads` above) — built from `wx`/`wy`/`wz` the same
          // way `v0..v3` above are built from `nx`/`ny`/`nz`.
          let wcx = wx(i) + di as f64 * x_step * 0.5;
          let wcy = wy(j) + dj as f64 * y_step * 0.5;
          let wcz = wz(k) + dk as f64 * z_step * 0.5;
          let (wsx, wsy, wsz) = (x_step * 0.5, y_step * 0.5, z_step * 0.5);
          world_quads.push(if di != 0 {
            [
              Point3D {
                x: wcx,
                y: wcy - wsy,
                z: wcz - wsz,
              },
              Point3D {
                x: wcx,
                y: wcy + wsy,
                z: wcz - wsz,
              },
              Point3D {
                x: wcx,
                y: wcy + wsy,
                z: wcz + wsz,
              },
              Point3D {
                x: wcx,
                y: wcy - wsy,
                z: wcz + wsz,
              },
            ]
          } else if dj != 0 {
            [
              Point3D {
                x: wcx - wsx,
                y: wcy,
                z: wcz - wsz,
              },
              Point3D {
                x: wcx + wsx,
                y: wcy,
                z: wcz - wsz,
              },
              Point3D {
                x: wcx + wsx,
                y: wcy,
                z: wcz + wsz,
              },
              Point3D {
                x: wcx - wsx,
                y: wcy,
                z: wcz + wsz,
              },
            ]
          } else {
            [
              Point3D {
                x: wcx - wsx,
                y: wcy - wsy,
                z: wcz,
              },
              Point3D {
                x: wcx + wsx,
                y: wcy - wsy,
                z: wcz,
              },
              Point3D {
                x: wcx + wsx,
                y: wcy + wsy,
                z: wcz,
              },
              Point3D {
                x: wcx - wsx,
                y: wcy + wsy,
                z: wcz,
              },
            ]
          });

          let face_style = color_fn.style_at(
            [wcx, wcy, wcz],
            [(x_min, x_max), (y_min, y_max), (z_min, z_max)],
          );
          let face_base = face_style
            .as_ref()
            .and_then(|s| s.color)
            .unwrap_or((base_r, base_g, base_b));
          let face_opacity = face_style.as_ref().map_or(1.0, |s| s.opacity);
          quad_styles.push(face_style);

          // Triangle 1: v0, v1, v2
          {
            let normal = triangle_normal(v0, v1, v2);
            let color = apply_lighting(face_base, normal);
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
              edge_width: None,
              projected: [p0, p1, p2],
              depth: depth(center, &camera),
              color,
              opacity: face_opacity,
              texture: None,
            });
          }

          // Triangle 2: v0, v2, v3
          {
            let normal = triangle_normal(v0, v2, v3);
            let color = apply_lighting(face_base, normal);
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
              edge_width: None,
              projected: [p0, p2, p3],
              depth: depth(center, &camera),
              color,
              opacity: face_opacity,
              texture: None,
            });
          }
        }
      }
    }
  }

  // A condition true nowhere in the box draws the empty box, as in
  // wolframscript, rather than failing the whole display.

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
    mesh_mode,
    show_axes,
    [1.0, 1.0, Z_SCALE],
    &axes_labels,
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  // ── Symbolic structure: GraphicsComplex[points, {Polygon[quads]}] ──
  // `First[RegionPlot3D[…]]` is the voxel surface itself, in data-space
  // coordinates, mirroring `RevolutionPlot3D`'s structure — a Demonstration
  // that builds a solid out of a `RegionPlot3D` face (e.g. `Translate`d and
  // `Rotate`d to close off another surface) needs the same access.
  let structure = {
    let mut point_exprs: Vec<Expr> = Vec::with_capacity(world_quads.len() * 4);
    let mut quads: Vec<Expr> = Vec::with_capacity(world_quads.len());
    // The voxel faces are an implementation detail, not the plot's mesh, so
    // their outlines are suppressed unless `Mesh -> All` asks for them —
    // otherwise a recombined solid reads as a wireframe of tiny cubes.
    let mut complex_content: Vec<Expr> = Vec::new();
    if !matches!(mesh_mode, MeshMode::All) {
      complex_content.push(call0("EdgeForm"));
    }
    for quad in &world_quads {
      let base = point_exprs.len() as i128;
      point_exprs.extend(quad.iter().map(|p| {
        Expr::List(
          vec![Expr::Real(p.x), Expr::Real(p.y), Expr::Real(p.z)].into(),
        )
      }));
      quads.push(Expr::List(
        (1..=4)
          .map(|off| Expr::Integer(base + off))
          .collect::<Vec<_>>()
          .into(),
      ));
    }
    // A `ColorFunction` colours each face on its own; otherwise one polygon
    // holds every face.
    if color_fn.function.is_some() {
      for (quad, style) in quads.into_iter().zip(&quad_styles) {
        let mut face: Vec<Expr> = Vec::new();
        if let Some(style) = style {
          if style.opacity < 1.0 {
            face.push(call1("Opacity", Expr::Real(style.opacity)));
          }
          if let Some((r, g, b)) = style.color {
            face.push(call(
              "RGBColor",
              vec![
                Expr::Real(r as f64 / 255.0),
                Expr::Real(g as f64 / 255.0),
                Expr::Real(b as f64 / 255.0),
              ],
            ));
          }
        }
        face.push(call1("Polygon", Expr::List(vec![quad].into())));
        complex_content.push(Expr::List(face.into()));
      }
    } else {
      complex_content.push(call1("Polygon", Expr::List(quads.into())));
    }
    // `RegionPlot3D` draws its axes where a bare `Graphics3D` does not, so
    // the default is spelled out for a plot shown inside another graphic.
    // An empty region holds no primitives at all: `{{}, {}}` in WL.
    let primitives = if world_quads.is_empty() {
      Expr::List(
        vec![Expr::List(vec![].into()), Expr::List(vec![].into())].into(),
      )
    } else {
      call(
        "GraphicsComplex",
        vec![
          Expr::List(point_exprs.into()),
          Expr::List(complex_content.into()),
        ],
      )
    };
    let mut structure_args = vec![primitives];
    structure_args.extend(args[4..].iter().cloned());
    let names_axes = structure_args.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Axes"))
    });
    if show_axes && !names_axes {
      structure_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
    }
    call("Graphics3D", structure_args)
  };

  Ok(crate::plot3d_result_with_structure(svg, structure))
}

// ── ContourPlot3D implementation ─────────────────────────────────────
//
// Extracts the isosurface `f == c` with marching tetrahedra: each grid
// cube is split into 6 tetrahedra sharing the corner-to-corner diagonal
// (a fixed split, so adjacent cubes' cut faces always agree and no cracks
// appear), and each tetrahedron contributes 0, 1 or 2 triangles depending
// on how many of its corners are above the level.

/// Grid resolution per axis. An isosurface's triangle count grows with the
/// *surface area* it crosses, not the full N^3 volume, so this stays much
/// cheaper in practice than the worst-case cube count suggests.
const CONTOUR3D_GRID: usize = 24;

/// How many times a grid cell that may hide a sheet thinner than itself is
/// halved, and the total number of cells that may be halved per plot. Only
/// used when the plain grid finds no surface at all.
const CONTOUR3D_REFINE_DEPTH: usize = 2;
const CONTOUR3D_REFINE_BUDGET: usize = 20_000;

/// The vertices of a unit cube, sharing corner-to-corner diagonal 0–6, in
/// the cyclic order that fans around that diagonal. Splitting a cube into
/// the 6 tetrahedra `{0, 6, ring[w], ring[w+1]}` is the standard
/// diagonal-preserving decomposition used by marching tetrahedra.
const TETRA_RING: [usize; 6] = [1, 2, 3, 7, 4, 5];

thread_local! {
  /// [`evaluate_scalar_3d`]'s compiled body: (address of the body, its
  /// compiled form if it compiles without fallbacks).
  static CONTOUR3D_COMPILED: std::cell::RefCell<
    Option<(usize, Option<crate::functions::polynomial_ast::solve::NumNode>)>,
  > = const { std::cell::RefCell::new(None) };
}

fn evaluate_scalar_3d(
  body: &Expr,
  xvar: &str,
  yvar: &str,
  zvar: &str,
  xval: f64,
  yval: f64,
  zval: f64,
) -> Option<f64> {
  // The body compiled once per plot (keyed by its address, the cache being
  // cleared when a plot starts) — a contour surface is sampled at tens of
  // thousands of points.
  let key = std::ptr::from_ref::<Expr>(body) as usize;
  let compiled = CONTOUR3D_COMPILED.with(|c| {
    let mut c = c.borrow_mut();
    if c.as_ref().is_none_or(|(k, _)| *k != key) {
      let vars = [xvar.to_string(), yvar.to_string(), zvar.to_string()];
      let node =
        crate::functions::polynomial_ast::solve::compile_numeric(body, &vars)
          .filter(|n| !n.has_fallback());
      *c = Some((key, node));
    }
    c.as_ref()
      .and_then(|(_, n)| n.as_ref().map(|n| n.eval(&[xval, yval, zval])))
  });
  if let Some(v) = compiled {
    return v.is_finite().then_some(v);
  }
  let sub1 = substitute_var(body, xvar, &Expr::Real(xval));
  let sub2 = substitute_var(&sub1, yvar, &Expr::Real(yval));
  let sub3 = substitute_var(&sub2, zvar, &Expr::Real(zval));
  let result = evaluate_expr_to_expr(&sub3).ok()?;
  try_eval_to_f64(&result)
}

/// The point where the level `level` crosses the segment from `verts[i]`
/// (value `vals[i]`) to `verts[j]` (value `vals[j]`), by linear
/// interpolation.
fn lerp_crossing(
  verts: &[Point3D; 4],
  vals: &[f64; 4],
  level: f64,
  i: usize,
  j: usize,
) -> Point3D {
  let denom = vals[j] - vals[i];
  let t = if denom.abs() < 1e-12 {
    0.5
  } else {
    ((level - vals[i]) / denom).clamp(0.0, 1.0)
  };
  Point3D {
    x: verts[i].x + t * (verts[j].x - verts[i].x),
    y: verts[i].y + t * (verts[j].y - verts[i].y),
    z: verts[i].z + t * (verts[j].z - verts[i].z),
  }
}

/// Triangulate the part of the level surface `level` that cuts through one
/// tetrahedron, calling `emit` with each triangle it produces. Corners at
/// or above `level` are "inside"; the surface separates them from the
/// corners below it.
fn march_tetra(
  verts: [Point3D; 4],
  vals: [f64; 4],
  level: f64,
  emit: &mut impl FnMut([Point3D; 3]),
) {
  let inside: Vec<usize> = (0..4).filter(|&i| vals[i] >= level).collect();
  let outside: Vec<usize> = (0..4).filter(|&i| vals[i] < level).collect();
  // The six tetrahedra of a cube wind alternately, so a triangle's vertex
  // order says nothing about which way it faces. Orient every emitted
  // triangle so its normal points toward the side where the field is above
  // `level`; `FaceForm[front, back]` then colours a connected surface
  // consistently instead of speckling it.
  let centroid = |idx: &[usize]| -> [f64; 3] {
    let n = idx.len().max(1) as f64;
    let mut c = [0.0; 3];
    for &i in idx {
      c[0] += verts[i].x / n;
      c[1] += verts[i].y / n;
      c[2] += verts[i].z / n;
    }
    c
  };
  let (ci, co) = (centroid(&inside), centroid(&outside));
  let toward_high = [ci[0] - co[0], ci[1] - co[1], ci[2] - co[2]];
  let mut emit = |t: [Point3D; 3]| {
    let n = triangle_normal(t[0], t[1], t[2]);
    let dot =
      n[0] * toward_high[0] + n[1] * toward_high[1] + n[2] * toward_high[2];
    if dot < 0.0 {
      emit([t[0], t[2], t[1]]);
    } else {
      emit(t);
    }
  };

  match (inside.len(), outside.len()) {
    (1, 3) => {
      let a = inside[0];
      let p0 = lerp_crossing(&verts, &vals, level, a, outside[0]);
      let p1 = lerp_crossing(&verts, &vals, level, a, outside[1]);
      let p2 = lerp_crossing(&verts, &vals, level, a, outside[2]);
      emit([p0, p1, p2]);
    }
    (3, 1) => {
      let a = outside[0];
      let p0 = lerp_crossing(&verts, &vals, level, a, inside[0]);
      let p1 = lerp_crossing(&verts, &vals, level, a, inside[1]);
      let p2 = lerp_crossing(&verts, &vals, level, a, inside[2]);
      emit([p0, p2, p1]);
    }
    (2, 2) => {
      let (i0, i1) = (inside[0], inside[1]);
      let (o0, o1) = (outside[0], outside[1]);
      let p00 = lerp_crossing(&verts, &vals, level, i0, o0);
      let p01 = lerp_crossing(&verts, &vals, level, i0, o1);
      let p10 = lerp_crossing(&verts, &vals, level, i1, o0);
      let p11 = lerp_crossing(&verts, &vals, level, i1, o1);
      emit([p00, p01, p11]);
      emit([p00, p11, p10]);
    }
    _ => {}
  }
}

/// Triangulate the part of the level surface that cuts through one grid
/// cube, via its 6-tetrahedra decomposition. `corners` and `vals` follow
/// the standard cube corner order: `{x,y,z}` in `{0,1}` with `x` varying
/// fastest, i.e. corner `i` has bit 0 = x, bit 1 = y, bit 2 = z.
fn march_cube(
  corners: &[Point3D; 8],
  vals: &[f64; 8],
  level: f64,
  emit: &mut impl FnMut([Point3D; 3]),
) {
  for w in 0..6 {
    let c = TETRA_RING[w];
    let d = TETRA_RING[(w + 1) % 6];
    march_tetra(
      [corners[0], corners[6], corners[c], corners[d]],
      [vals[0], vals[6], vals[c], vals[d]],
      level,
      emit,
    );
  }
}

/// Repaint the full-canvas background `<rect>` that `generate_svg` emits
/// right after the opening `<svg>` tag.
fn replace_canvas_background(svg: &str, (r, g, b): (u8, u8, u8)) -> String {
  let Some(start) = svg.find("<rect width=") else {
    return svg.to_string();
  };
  let Some(rel) = svg[start..].find("fill=\"rgb(") else {
    return svg.to_string();
  };
  let fill_start = start + rel;
  let Some(end_rel) = svg[fill_start..].find(")\"") else {
    return svg.to_string();
  };
  let fill_end = fill_start + end_rel + 2;
  format!(
    "{}fill=\"rgb({r},{g},{b})\"{}",
    &svg[..fill_start],
    &svg[fill_end..]
  )
}

/// ContourPlot3D[f == c, {x, xmin, xmax}, {y, ymin, ymax}, {z, zmin, zmax}]
/// draws the implicit surface `f == c` (a bare expression, or one lacking
/// `== c`, is its own zero contour). A list of equations draws each
/// surface together, as `ContourStyle` colours them.
pub fn contour_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  CONTOUR3D_COMPILED.with(|c| *c.borrow_mut() = None);
  if args.len() < 4 {
    return Err(InterpreterError::EvaluationError(
      "ContourPlot3D requires at least 4 arguments: ContourPlot3D[f == c, {x,xmin,xmax}, {y,ymin,ymax}, {z,zmin,zmax}]".into(),
    ));
  }

  let (xvar, x_min, x_max) =
    parse_iterator_generic(&args[1], "ContourPlot3D", "first")?;
  let (yvar, y_min, y_max) =
    parse_iterator_generic(&args[2], "ContourPlot3D", "second")?;
  let (zvar, z_min, z_max) =
    parse_iterator_generic(&args[3], "ContourPlot3D", "third")?;
  if x_max <= x_min || y_max <= y_min || z_max <= z_min {
    return Err(InterpreterError::EvaluationError(
      "ContourPlot3D: iterator ranges must be increasing".into(),
    ));
  }

  // A body naming a symbol whose value depends on the plot variables is
  // read through that value first (see `resolve_indirect_plot_body`), so
  // `b = f == c; ContourPlot3D[b, …]` is seen as the equation it names.
  let plot_vars = [xvar.as_str(), yvar.as_str(), zvar.as_str()];
  let resolved =
    crate::functions::plot::resolve_indirect_plot_body(&args[0], &plot_vars);
  let body = resolved.as_ref().unwrap_or(&args[0]);
  // `f == c` (or a list of such equations) plots the zero contour of
  // `f - c`; a bare expression is its own zero contour.
  let bodies: Vec<Expr> = match body {
    Expr::List(items) if !items.is_empty() => items
      .iter()
      .map(|i| {
        crate::functions::field_plot::equation_zero_body(i)
          .unwrap_or_else(|| i.clone())
      })
      .collect(),
    other => vec![
      crate::functions::field_plot::equation_zero_body(other)
        .unwrap_or_else(|| other.clone()),
    ],
  };

  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut show_axes = true;
  let mut styles: Vec<StyleState3D> = Vec::new();
  // `PlotPoints -> n`: number of samples per direction (n - 1 grid cells),
  // matching the convention used elsewhere in this module (e.g.
  // SphericalPlot3D). Demonstrations commonly toggle this between a coarse
  // preview and a refined render.
  let mut plot_points: Option<usize> = None;
  // `Contours -> n` draws n isosurfaces evenly spaced across the function's
  // actual sampled range; `Contours -> {v1, v2, ...}` draws explicit levels.
  // With no `Contours` option, ContourPlot3D draws the single surface where
  // the (equation-normalized) body is zero.
  let mut contour_count: Option<usize> = None;
  let mut contour_levels: Option<Vec<f64>> = None;
  let mut background: Option<(u8, u8, u8)> = None;

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
        Expr::Identifier(name) if name == "Boxed" => match replacement.as_ref()
        {
          Expr::Identifier(s) if s == "False" => show_axes = false,
          Expr::Identifier(s) if s == "True" => show_axes = true,
          _ => {}
        },
        Expr::Identifier(name) if name == "ContourStyle" => {
          styles = parse_plot_style_3d(replacement, bodies.len());
        }
        Expr::Identifier(name) if name == "Background" => {
          background = parse_background_color(replacement).map(|c| {
            let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            (q(c.r), q(c.g), q(c.b))
          });
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
        Expr::Identifier(name) if name == "Contours" => {
          let resolved = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| (**replacement).clone());
          match &resolved {
            Expr::Integer(cnt) if *cnt > 0 => {
              contour_count = Some(*cnt as usize);
            }
            Expr::List(items) => {
              let levels: Vec<f64> = items
                .iter()
                .filter_map(|item| {
                  let v = evaluate_expr_to_expr(item)
                    .unwrap_or_else(|_| item.clone());
                  try_eval_to_f64(&v)
                })
                .collect();
              if !levels.is_empty() {
                contour_levels = Some(levels);
              }
            }
            _ => {}
          }
        }
        _ => {}
      }
    }
  }

  let n = plot_points.map_or(CONTOUR3D_GRID, |p| p - 1);
  let x_step = (x_max - x_min) / n as f64;
  let y_step = (y_max - y_min) / n as f64;
  let z_step = (z_max - z_min) / n as f64;

  let camera = Camera::default();
  let view_dir = {
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };

  // Normalize coordinates to [-1, 1] for x,y and [-Z_SCALE, Z_SCALE] for z,
  // matching every other 3D plot in this module.
  let nx = |v: f64| -> f64 { (v - x_min) / (x_max - x_min) * 2.0 - 1.0 };
  let ny = |v: f64| -> f64 { (v - y_min) / (y_max - y_min) * 2.0 - 1.0 };
  let nz =
    |v: f64| -> f64 { (v - z_min) / (z_max - z_min) * 2.0 * Z_SCALE - Z_SCALE };

  let default_color = (0x5E_u8, 0x81_u8, 0xB5_u8);
  let mut all_triangles: Vec<Triangle> = Vec::new();
  // The surfaces' vertices, shared between the triangles that meet at them
  // (keyed by their exact coordinates), with accumulated face normals.
  let mut points: Vec<[f64; 3]> = Vec::new();
  let mut normals: Vec<[f64; 3]> = Vec::new();
  let mut point_index: std::collections::HashMap<[u64; 3], usize> =
    std::collections::HashMap::new();
  let mut content: Vec<Expr> = Vec::new();
  let mut refinement_budget = CONTOUR3D_REFINE_BUDGET;

  for (surf_idx, body) in bodies.iter().enumerate() {
    let mut field = vec![vec![vec![f64::NAN; n + 1]; n + 1]; n + 1];
    for (i, xrow) in field.iter_mut().enumerate() {
      let xval = x_min + i as f64 * x_step;
      for (j, yrow) in xrow.iter_mut().enumerate() {
        let yval = y_min + j as f64 * y_step;
        for (k, cell) in yrow.iter_mut().enumerate() {
          let zval = z_min + k as f64 * z_step;
          if let Some(v) =
            evaluate_scalar_3d(body, &xvar, &yvar, &zvar, xval, yval, zval)
          {
            *cell = v;
          }
        }
      }
    }

    let style = if styles.is_empty() {
      None
    } else {
      Some(&styles[surf_idx % styles.len()])
    };
    let (cr, cg, cb) = style.and_then(|s| s.color).unwrap_or(default_color);
    let color_directive = call(
      "RGBColor",
      vec![
        Expr::Real(cr as f64 / 255.0),
        Expr::Real(cg as f64 / 255.0),
        Expr::Real(cb as f64 / 255.0),
      ],
    );

    let levels: Vec<f64> = if let Some(explicit) = &contour_levels {
      explicit.clone()
    } else if let Some(cnt) = contour_count {
      let mut v_min = f64::INFINITY;
      let mut v_max = f64::NEG_INFINITY;
      for xrow in &field {
        for yrow in xrow {
          for &v in yrow {
            if v.is_finite() {
              v_min = v_min.min(v);
              v_max = v_max.max(v);
            }
          }
        }
      }
      if v_max > v_min {
        let step = (v_max - v_min) / (cnt as f64 + 1.0);
        (1..=cnt).map(|k| v_min + k as f64 * step).collect()
      } else {
        vec![0.0]
      }
    } else {
      vec![0.0]
    };

    let triangles_before = all_triangles.len();
    for refine in [false, true] {
      if refine && all_triangles.len() > triangles_before {
        break;
      }
      for &level in &levels {
        // This level's triangles, as one Polygon of vertex-index triples.
        let mut level_triangles: Vec<[usize; 3]> = Vec::new();
        for i in 0..n {
          for j in 0..n {
            for k in 0..n {
              let cvals = [
                field[i][j][k],
                field[i + 1][j][k],
                field[i + 1][j + 1][k],
                field[i][j + 1][k],
                field[i][j][k + 1],
                field[i + 1][j][k + 1],
                field[i + 1][j + 1][k + 1],
                field[i][j + 1][k + 1],
              ];
              let xs =
                [x_min + i as f64 * x_step, x_min + (i + 1) as f64 * x_step];
              let ys =
                [y_min + j as f64 * y_step, y_min + (j + 1) as f64 * y_step];
              let zs =
                [z_min + k as f64 * z_step, z_min + (k + 1) as f64 * z_step];
              // A sheet thinner than a cell leaves every corner on one side
              // of the level; such cells are subdivided (below) so the
              // surface is still found.
              let mut stack = vec![(xs, ys, zs, cvals, 0_usize)];
              while let Some((xs, ys, zs, cvals, depth_level)) = stack.pop() {
                if cvals.iter().any(|v| !v.is_finite()) {
                  continue;
                }
                let corners_world: [Point3D; 8] = [
                  Point3D {
                    x: xs[0],
                    y: ys[0],
                    z: zs[0],
                  },
                  Point3D {
                    x: xs[1],
                    y: ys[0],
                    z: zs[0],
                  },
                  Point3D {
                    x: xs[1],
                    y: ys[1],
                    z: zs[0],
                  },
                  Point3D {
                    x: xs[0],
                    y: ys[1],
                    z: zs[0],
                  },
                  Point3D {
                    x: xs[0],
                    y: ys[0],
                    z: zs[1],
                  },
                  Point3D {
                    x: xs[1],
                    y: ys[0],
                    z: zs[1],
                  },
                  Point3D {
                    x: xs[1],
                    y: ys[1],
                    z: zs[1],
                  },
                  Point3D {
                    x: xs[0],
                    y: ys[1],
                    z: zs[1],
                  },
                ];

                let mut emitted = false;
                march_cube(&corners_world, &cvals, level, &mut |tri_world| {
                  emitted = true;
                  let tri_norm = tri_world.map(|p| Point3D {
                    x: nx(p.x),
                    y: ny(p.y),
                    z: nz(p.z),
                  });
                  let normal =
                    triangle_normal(tri_norm[0], tri_norm[1], tri_norm[2]);
                  let (color, opacity) =
                    shade_facet(default_color, style, normal, view_dir);
                  let p0 = project(tri_norm[0], &camera);
                  let p1 = project(tri_norm[1], &camera);
                  let p2 = project(tri_norm[2], &camera);
                  let center = Point3D {
                    x: (tri_norm[0].x + tri_norm[1].x + tri_norm[2].x) / 3.0,
                    y: (tri_norm[0].y + tri_norm[1].y + tri_norm[2].y) / 3.0,
                    z: (tri_norm[0].z + tri_norm[1].z + tri_norm[2].z) / 3.0,
                  };
                  all_triangles.push(Triangle {
                    boundary: [true; 3],
                    edge_color: style.and_then(|s| s.edge_color),
                    edge_width: style.and_then(|s| s.edge_width),
                    projected: [p0, p1, p2],
                    depth: depth(center, &camera),
                    color,
                    opacity,
                    texture: None,
                  });

                  // World-coordinate copy for `Show[]` to merge via GraphicsComplex.
                  let wn =
                    triangle_normal(tri_world[0], tri_world[1], tri_world[2]);
                  let mut tri = [0usize; 3];
                  for (slot, p) in tri.iter_mut().zip(&tri_world) {
                    let key = [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
                    let idx = *point_index.entry(key).or_insert_with(|| {
                      points.push([p.x, p.y, p.z]);
                      normals.push([0.0; 3]);
                      points.len() - 1
                    });
                    for d in 0..3 {
                      normals[idx][d] += wn[d];
                    }
                    *slot = idx;
                  }
                  level_triangles.push(tri);
                });
                if refine
                  && !emitted
                  && depth_level < CONTOUR3D_REFINE_DEPTH
                  && refinement_budget > 0
                {
                  let (lo, hi) = cvals.iter().fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(lo, hi), &v| (lo.min(v), hi.max(v)),
                  );
                  // The field dips no further below its corner values than it
                  // varies across the cell, so a cell whose nearest corner is
                  // within that range of the level may hide a thin sheet.
                  let gap = (lo - level).abs().min((hi - level).abs());
                  if gap <= hi - lo {
                    refinement_budget -= 1;
                    let mid = |a: [f64; 2]| f64::midpoint(a[0], a[1]);
                    let pts = |a: [f64; 2]| [a[0], mid(a), a[1]];
                    let (px, py, pz) = (pts(xs), pts(ys), pts(zs));
                    let mut sub = [[[f64::NAN; 3]; 3]; 3];
                    for (a, plane) in sub.iter_mut().enumerate() {
                      for (b, row) in plane.iter_mut().enumerate() {
                        for (c, cell) in row.iter_mut().enumerate() {
                          *cell = evaluate_scalar_3d(
                            body, &xvar, &yvar, &zvar, px[a], py[b], pz[c],
                          )
                          .unwrap_or(f64::NAN);
                        }
                      }
                    }
                    for a in 0..2 {
                      for b in 0..2 {
                        for c in 0..2 {
                          stack.push((
                            [px[a], px[a + 1]],
                            [py[b], py[b + 1]],
                            [pz[c], pz[c + 1]],
                            [
                              sub[a][b][c],
                              sub[a + 1][b][c],
                              sub[a + 1][b + 1][c],
                              sub[a][b + 1][c],
                              sub[a][b][c + 1],
                              sub[a + 1][b][c + 1],
                              sub[a + 1][b + 1][c + 1],
                              sub[a][b + 1][c + 1],
                            ],
                            depth_level + 1,
                          ));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if !level_triangles.is_empty() {
          let mut item = vec![color_directive.clone()];
          // A translucent surface stays translucent when its primitives
          // are merged into another graphic.
          if let Some(st) = style
            && st.opacity < 1.0
          {
            item.push(call1("Opacity", Expr::Real(st.opacity)));
          }
          item.push(call1(
            "Polygon",
            Expr::List(
              level_triangles
                .iter()
                .map(|t| {
                  Expr::List(
                    t.iter().map(|&v| Expr::Integer(v as i128 + 1)).collect(),
                  )
                })
                .collect(),
            ),
          ));
          content.push(Expr::List(item.into()));
        }
      }
    }
  }
  let point_exprs: Vec<Expr> = points
    .iter()
    .map(|p| Expr::List(p.iter().map(|&c| Expr::Real(c)).collect()))
    .collect();
  let normal_exprs: Vec<Expr> = normals
    .iter()
    .map(|nv| {
      let len = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
      let len = if len > 0.0 { len } else { 1.0 };
      Expr::List(nv.iter().map(|&c| Expr::Real(c / len)).collect())
    })
    .collect();

  if all_triangles.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "ContourPlot3D: the surface has no points in the given range".into(),
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
    (z_min, z_max),
    svg_width,
    svg_height,
    full_width,
    MeshMode::None,
    show_axes,
    [1.0, 1.0, Z_SCALE],
    &[None, None, None],
  )?;
  // `Background -> colour` repaints the canvas rectangle drawn first.
  let svg = match background {
    Some(bg) => replace_canvas_background(&svg, bg),
    None => svg,
  };
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  let structure = {
    // The plot's own picture draws no outline on the surface triangles, so
    // the extracted primitives must not pick up the default polygon edge
    // when re-rendered inside a `Graphics3D` (e.g. `plot[[1]]`).
    content.insert(0, call0("EdgeForm"));
    let complex = call(
      "GraphicsComplex",
      vec![
        Expr::List(point_exprs.into()),
        Expr::List(content.into()),
        rule_expr(id_expr("VertexNormals"), Expr::List(normal_exprs.into())),
      ],
    );
    let mut structure_args = vec![complex];
    structure_args.extend(args[4..].iter().cloned());
    let names = |opt: &str| {
      structure_args.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == opt))
      })
    };
    let (names_axes, names_ratios) = (names("Axes"), names("BoxRatios"));
    if !names_axes {
      structure_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
    }
    if !names_ratios {
      structure_args.push(rule_expr(
        id_expr("BoxRatios"),
        Expr::List(
          vec![Expr::Integer(1), Expr::Integer(1), Expr::Real(Z_SCALE)].into(),
        ),
      ));
    }
    call("Graphics3D", structure_args)
  };

  Ok(crate::plot3d_result_with_structure(svg, structure))
}

// ── ListPointPlot3D implementation ───────────────────────────────────

/// A projected point for scatter rendering.
struct ScatterPoint {
  sx: f64,
  sy: f64,
  depth: f64,
  color: (u8, u8, u8),
  /// Text drawn beside the point (`data -> labels`).
  label: Option<String>,
}

/// One `ListPointPlot3D` dataset: a list of `{x, y, z}` triples, or a
/// matrix of heights `{{z11, z12, …}, …}` placed at x = column and y = row
/// (both 1-based). `None` when `items` is neither, e.g. a list of datasets.
fn point_plot3d_dataset(items: &[Expr]) -> Option<Vec<(f64, f64, f64)>> {
  let is_scalar = |e: &Expr| !matches!(e, Expr::List(_));
  let rows: Vec<&[Expr]> = items
    .iter()
    .map(|e| match e {
      Expr::List(row) if !row.is_empty() && row.iter().all(is_scalar) => {
        Some(row.as_slice())
      }
      _ => None,
    })
    .collect::<Option<_>>()?;
  if rows.is_empty() {
    return None;
  }
  // Rows of three numbers are points; any other rectangular array of
  // scalars is a height matrix.
  if rows.iter().all(|r| r.len() == 3) {
    return Some(parse_xyz_points(items));
  }
  let mut pts = Vec::new();
  for (i, row) in rows.iter().enumerate() {
    for (j, val) in row.iter().enumerate() {
      if let Some(z) = try_eval_to_f64(val)
        && z.is_finite()
      {
        pts.push(((j + 1) as f64, (i + 1) as f64, z));
      }
    }
  }
  Some(pts)
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
  // `data -> {label1, label2, …}` labels the points of the data in order;
  // `LabelingFunction -> None` (or `Tooltip`) leaves them unlabelled.
  let hide_labels = args[1..].iter().any(|opt| {
    matches!(opt, Expr::Rule { pattern, replacement }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "LabelingFunction")
        && matches!(replacement.as_ref(),
          Expr::Identifier(v) if v == "None" || v == "Tooltip"))
  });
  let (evaled_data, point_labels): (Expr, Vec<String>) = match &evaled_data {
    Expr::Rule {
      pattern,
      replacement,
    } => {
      let labels = match &evaluate_expr_to_expr(replacement)
        .unwrap_or_else(|_| *replacement.clone())
      {
        Expr::List(items) if !hide_labels => items
          .iter()
          .map(|it| match it {
            Expr::String(s) => s.clone(),
            other => {
              crate::syntax::format_expr(other, crate::syntax::ExprForm::Output)
            }
          })
          .collect(),
        _ => Vec::new(),
      };
      (evaluate_expr_to_expr(pattern)?, labels)
    }
    other => (other.clone(), Vec::new()),
  };

  // Parse data: a list of {x, y, z} triples, a matrix of heights, or a list
  // of such datasets (each one drawn in its own colour).
  let mut datasets: Vec<Vec<(f64, f64, f64)>> = Vec::new();
  if let Expr::List(outer) = &evaled_data {
    match point_plot3d_dataset(outer) {
      Some(pts) => datasets.push(pts),
      None => {
        for item in outer {
          if let Expr::List(inner) = item
            && let Some(pts) = point_plot3d_dataset(inner)
          {
            datasets.push(pts);
          }
        }
      }
    }
  }
  datasets.retain(|ds| !ds.is_empty());

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
    for (pi, &(x, y, z)) in ds.iter().enumerate() {
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
        label: if di == 0 {
          point_labels.get(pi).cloned()
        } else {
          None
        },
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

  // A symbolic `Graphics3D[…]` structure recorded alongside the pre-rendered
  // scatter SVG above, so `Show[ListPointPlot3D[…], opts]` can recompose the
  // scene through the ordinary Graphics3D pipeline — honoring
  // Axes/Boxed/BoxRatios/PlotRange/SphericalRegion/ViewAngle and merging
  // properly with other Graphics3D content, none of which the standalone
  // scatter camera above understands.
  let structure = {
    // `PlotStyle -> style` (or a list of per-dataset styles) prepends its
    // directives — e.g. `PointSize[…]` or a colour — to each dataset's
    // points, same as the surface plots' `plot_style_items` already does.
    let plot_style_per_dataset: Option<Vec<Expr>> =
      args[1..].iter().find_map(|opt| match opt {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle") =>
        {
          Some(plot_style_items(replacement, datasets.len()))
        }
        _ => None,
      });

    let mut content: Vec<Expr> = Vec::new();
    for (di, ds) in datasets.iter().enumerate() {
      let base_color = palette[di % palette.len()];
      let mut group = vec![call(
        "RGBColor",
        vec![
          Expr::Real(base_color.0 as f64 / 255.0),
          Expr::Real(base_color.1 as f64 / 255.0),
          Expr::Real(base_color.2 as f64 / 255.0),
        ],
      )];
      if let Some(style_items) = &plot_style_per_dataset
        && !style_items.is_empty()
      {
        group.push(style_items[di % style_items.len()].clone());
      }
      let point_exprs: Vec<Expr> = ds
        .iter()
        .map(|&(x, y, z)| {
          Expr::List(vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)].into())
        })
        .collect();
      group.push(call1("Point", Expr::List(point_exprs.into())));
      content.push(Expr::List(group.into()));
    }

    let mut structure_args = vec![Expr::List(content.into())];
    structure_args.extend(args[1..].iter().filter(|opt| {
      !matches!(opt, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotStyle"))
    }).cloned());
    let names = |args: &[Expr], opt: &str| {
      args.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == opt))
      })
    };
    let (names_axes, names_ratios) = (
      names(&structure_args, "Axes"),
      names(&structure_args, "BoxRatios"),
    );
    if !names_axes {
      structure_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
    }
    if !names_ratios {
      structure_args.push(rule_expr(
        id_expr("BoxRatios"),
        Expr::List(
          vec![Expr::Integer(1), Expr::Integer(1), Expr::Real(Z_SCALE)].into(),
        ),
      ));
    }
    call("Graphics3D", structure_args)
  };

  Ok(crate::plot3d_result_with_structure(svg, structure))
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
  let p_cx = f64::midpoint(px_min, px_max);
  let p_cy = f64::midpoint(py_min, py_max);

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    let sx = cx + (px - p_cx) * scale;
    let sy = cy - (py - p_cy) * scale;
    (sx, sy)
  };

  let mut svg = String::with_capacity(points.len() * 80 + 2000);

  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
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
      let tm = f64::midpoint(t0, t1);
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
          "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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
      if let Some(label) = &pt.label {
        svg.push_str(&format!(
          "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" \
           font-size=\"11\" fill=\"{axis_color}\">{}</text>\n",
          sx + radius + 2.0,
          sy - radius - 2.0,
          crate::functions::graphics::box_string_to_svg(label)
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
        "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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

  // Evaluate the data argument; `Quantity` data plots as its magnitudes.
  let evaled_data = crate::functions::chart::strip_quantities(
    &evaluate_expr_to_expr(&args[0])?,
  );

  // Each curve is one polyline through its points (in data order), drawn
  // in the colour of the dataset it belongs to.
  let mut datasets: Vec<Vec<(f64, f64, f64)>> = Vec::new();
  let mut curve_colors: Vec<usize> = Vec::new();
  if let Expr::List(outer) = &evaled_data {
    // `{m1, m2, …}` of height matrices: each matrix is one dataset, all of
    // its rows sharing a colour.
    let is_matrix_of_heights = |e: &Expr| {
      matches!(e, Expr::List(rows) if !rows.is_empty()
        && rows.iter().all(|r| matches!(r, Expr::List(vs)
          if vs.len() != 3 && vs.iter().all(|v| !matches!(v, Expr::List(_))))))
    };
    if outer.len() > 1 && outer.iter().all(is_matrix_of_heights) {
      for (di, m) in outer.iter().enumerate() {
        for curve in line3d_curves(m) {
          datasets.push(curve);
          curve_colors.push(di);
        }
      }
    } else {
      for (ci, curve) in line3d_curves(&evaled_data).into_iter().enumerate() {
        datasets.push(curve);
        curve_colors.push(ci);
      }
    }
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

  let palette = crate::functions::plot::PLOT_COLORS;
  for (ds, &di) in datasets.iter().zip(&curve_colors) {
    let color = palette[di % palette.len()];
    for pair in ds.windows(2) {
      let a = normalize(pair[0].0, pair[0].1, pair[0].2);
      let b = normalize(pair[1].0, pair[1].1, pair[1].2);
      let (ax, ay) = project(a, &camera);
      let (bx, by) = project(b, &camera);
      let mid = Point3D {
        x: f64::midpoint(a.x, b.x),
        y: f64::midpoint(a.y, b.y),
        z: f64::midpoint(a.z, b.z),
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
  let graphics = crate::graphics3d_result(svg);

  // `PlotLegends -> {l1, l2, …}` names the datasets with a line legend.
  let legends = args[1..].iter().find_map(|opt| match opt {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotLegends") => {
      match evaluate_expr_to_expr(replacement) {
        Ok(Expr::List(ref labels)) if !labels.is_empty() => Some(labels.clone()),
        _ => None,
      }
    }
    _ => None,
  });
  let Some(labels) = legends else {
    return Ok(graphics);
  };
  let colors: Vec<Expr> = (0..labels.len())
    .map(|i| {
      let (r, g, b) = palette[i % palette.len()];
      Expr::FunctionCall {
        name: "RGBColor".into(),
        args: [r, g, b]
          .iter()
          .map(|&c| Expr::Real(f64::from(c) / 255.0))
          .collect::<Vec<_>>()
          .into(),
      }
    })
    .collect();
  Ok(Expr::FunctionCall {
    name: "Legended".into(),
    args: vec![
      graphics,
      Expr::FunctionCall {
        name: "LineLegend".into(),
        args: vec![Expr::List(colors.into()), Expr::List(labels)].into(),
      },
    ]
    .into(),
  })
}

/// The curves of one `ListLinePlot3D` dataset: a list of `{x, y, z}`
/// points is one curve, a matrix of heights one curve per row (through
/// `{j, i, z_ij}`), and a list of point lists one curve per list.
fn line3d_curves(data: &Expr) -> Vec<Vec<(f64, f64, f64)>> {
  let mut curves = Vec::new();
  let Expr::List(outer) = data else {
    return curves;
  };
  match outer.first() {
    Some(Expr::List(inner))
      if inner.len() == 3 && !matches!(&inner[0], Expr::List(_)) =>
    {
      curves.push(parse_xyz_points(outer));
    }
    Some(Expr::List(inner))
      if !inner.is_empty()
        && inner.iter().all(|e| !matches!(e, Expr::List(_))) =>
    {
      for (i, row_expr) in outer.iter().enumerate() {
        if let Expr::List(row) = row_expr {
          curves.push(
            row
              .iter()
              .enumerate()
              .filter_map(|(j, v)| {
                try_eval_to_f64(v)
                  .filter(|z| z.is_finite())
                  .map(|z| ((j + 1) as f64, (i + 1) as f64, z))
              })
              .collect(),
          );
        }
      }
    }
    Some(Expr::List(_)) => {
      for item in outer {
        if let Expr::List(inner) = item {
          curves.push(parse_xyz_points(inner));
        }
      }
    }
    _ => curves.push(parse_xyz_points(outer)),
  }
  curves.retain(|c| !c.is_empty());
  curves
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
  let p_cx = f64::midpoint(px_min, px_max);
  let p_cy = f64::midpoint(py_min, py_max);

  let to_svg = |px: f64, py: f64| -> (f64, f64) {
    let sx = cx + (px - p_cx) * scale;
    let sy = cy - (py - p_cy) * scale;
    (sx, sy)
  };

  let mut svg = String::with_capacity(segments.len() * 100 + 2000);

  if full_width {
    svg.push_str(&format!(
      "<svg width=\"100%\" viewBox=\"0 0 {svg_width} {svg_height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
  } else {
    svg.push_str(&format!(
      "<svg width=\"{svg_width}\" height=\"{svg_height}\" viewBox=\"0 0 {svg_width} {svg_height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
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
      let tm = f64::midpoint(t0, t1);
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
      "<line x1=\"{ex0:.1}\" y1=\"{ey0:.1}\" x2=\"{ex1:.1}\" y2=\"{ey1:.1}\" stroke=\"{axis_color}\" stroke-width=\"0.5\" opacity=\"0.4\"/>\n"
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
      "<line x1=\"{sx0:.2}\" y1=\"{sy0:.2}\" x2=\"{sx1:.2}\" y2=\"{sy1:.2}\" stroke=\"rgb({r},{g},{b})\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>\n"
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
/// Grid density of a textured `SphericalPlot3D`, whose facets are the
/// texture's pixels.
const SPHERICAL_TEXTURE_GRID: usize = 80;

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
  // One or more radial functions: `SphericalPlot3D[{r1, r2, …}, …]` draws
  // each as its own surface, styled in turn by `PlotStyle`.
  let bodies: Vec<&Expr> = match body {
    Expr::List(items) => items.iter().collect(),
    _ => vec![body],
  };

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
  let mut mesh_mode = MeshMode::Default;
  let mut show_axes = true;
  let mut axes_labels: [Option<String>; 3] = [None, None, None];
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
  let mut plot_style_expr: Option<&Expr> = None;
  // `TextureCoordinateFunction -> f`: maps a facet to the point of a
  // `Texture` image painted there.
  let mut texture_fn: Option<Expr> = None;
  // `ColorFunction -> f`: colours each facet by
  // `f[x, y, z, theta, phi, r]`. With `ColorFunctionScaling` (the default)
  // x, y, z and the angles arrive rescaled to 0..1 over their ranges.
  let mut color_fn: Option<Expr> = None;
  let mut color_scaling = true;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_style_expr = Some(replacement.as_ref());
        }
        Expr::Identifier(name) if name == "TextureCoordinateFunction" => {
          texture_fn = Some(replacement.as_ref().clone());
        }
        Expr::Identifier(name) if name == "ColorFunction" => {
          color_fn = Some(replacement.as_ref().clone());
        }
        Expr::Identifier(name) if name == "ColorFunctionScaling" => {
          color_scaling = !matches!(
            replacement.as_ref(),
            Expr::Identifier(n) if n == "False"
          );
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
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "Axes" => {
          if matches!(replacement.as_ref(), Expr::Identifier(n) if n == "False")
          {
            show_axes = false;
          }
        }
        Expr::Identifier(name) if name == "AxesLabel" => {
          axes_labels = parse_axes_labels(replacement, None);
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
            && let Some(color) = parse_background_color(replacement)
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

  // `PlotStyle`, one style per surface (cycling); empty means it was not
  // given, so the height-based default colouring is used unchanged.
  let plot_styles: Vec<StyleState3D> = plot_style_expr
    .map(|e| parse_plot_style_3d(e, bodies.len()))
    .unwrap_or_default();

  // A texture is sampled once per facet, so it needs finer facets than a
  // plain surface to show any detail.
  let default_grid = if plot_styles.iter().any(|s| s.texture.is_some()) {
    SPHERICAL_TEXTURE_GRID
  } else {
    SPHERICAL_GRID
  };
  let n_theta = plot_points.map_or(default_grid, |p| p - 1);
  let n_phi = plot_points.map_or(default_grid, |p| p - 1);
  let theta_range = theta_max - theta_min;
  let phi_range = phi_max - phi_min;

  let param_at = |i: usize, j: usize| -> (f64, f64) {
    (
      theta_min + (i as f64 / n_theta as f64) * theta_range,
      phi_min + (j as f64 / n_phi as f64) * phi_range,
    )
  };

  // The body is held, so resolve user definitions (e.g. `r[1, 2]` standing
  // for a large trigonometric expression) once up front instead of looking
  // them up and re-evaluating them at every sample point.
  let evaluated_bodies: Vec<Expr> = bodies
    .iter()
    .map(|b| evaluate_expr_to_expr(b).unwrap_or_else(|_| (*b).clone()))
    .collect();

  // Build each surface's world-space triangles independently, clipping
  // cells that cross the region boundary.
  let mut surface_tris: Vec<Vec<[Point3D; 3]>> =
    Vec::with_capacity(bodies.len());
  // The (theta, phi) centroid of each triangle, for texture coordinates.
  let mut surface_tri_params: Vec<Vec<(f64, f64)>> =
    Vec::with_capacity(bodies.len());
  // The (theta, phi) of each triangle's three corners, for the per-vertex
  // `VertexTextureCoordinates` of the symbolic structure.
  let mut surface_tri_vparams: Vec<Vec<[(f64, f64); 3]>> =
    Vec::with_capacity(bodies.len());
  for surface_body in &evaluated_bodies {
    // Evaluate the surface point at arbitrary (theta, phi) parameters.
    let surface_at = |theta: f64, phi: f64| -> Option<(Point3D, f64)> {
      let r =
        evaluate_at_t_theta(surface_body, &theta_var, theta, &phi_var, phi)?;
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
          let mid = (f64::midpoint(lo.0, hi.0), f64::midpoint(lo.1, hi.1));
          if inside_at(mid.0, mid.1) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        (f64::midpoint(lo.0, hi.0), f64::midpoint(lo.1, hi.1))
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

    let mut tris: Vec<[Point3D; 3]> = Vec::new();
    let mut tri_params: Vec<(f64, f64)> = Vec::new();
    let mut tri_vparams: Vec<[(f64, f64); 3]> = Vec::new();
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
          let (points, params): (Vec<Point3D>, Vec<(f64, f64)>) = polygon
            .iter()
            .filter_map(|&(th, ph)| {
              surface_at(th, ph).map(|(p, _)| (p, (th, ph)))
            })
            .unzip();
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
            tris.push([a, b, c]);
            tri_vparams.push([params[0], params[k], params[k + 1]]);
            tri_params.push((
              (params[0].0 + params[k].0 + params[k + 1].0) / 3.0,
              (params[0].1 + params[k].1 + params[k + 1].1) / 3.0,
            ));
          }
        }
      }
    }
    surface_tris.push(tris);
    surface_tri_params.push(tri_params);
    surface_tri_vparams.push(tri_vparams);
  }

  if surface_tris.iter().all(std::vec::Vec::is_empty) {
    return Err(InterpreterError::EvaluationError(
      "SphericalPlot3D: no renderable triangles".into(),
    ));
  }

  // Find coordinate ranges across every surface
  let mut x_min = f64::INFINITY;
  let mut x_max = f64::NEG_INFINITY;
  let mut y_min = f64::INFINITY;
  let mut y_max = f64::NEG_INFINITY;
  let mut z_min = f64::INFINITY;
  let mut z_max = f64::NEG_INFINITY;

  for tris in &surface_tris {
    for tri in tris {
      for p in tri {
        x_min = x_min.min(p.x);
        x_max = x_max.max(p.x);
        y_min = y_min.min(p.y);
        y_max = y_max.max(p.y);
        z_min = z_min.min(p.z);
        z_max = z_max.max(p.z);
      }
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

  // Texture coordinates of a surface point `p` at parameters (theta, phi):
  // the `TextureCoordinateFunction` applied to the point scaled to the
  // unit cube over the plot (as `TextureCoordinateScaling -> True` does),
  // or the scaled (theta, phi) by default.
  let texture_uv = |p: Point3D, theta: f64, phi: f64| -> Option<(f64, f64)> {
    let scaled = |v: f64, lo: f64, range: f64| {
      if range.abs() < 1e-15 {
        0.0
      } else {
        (v - lo) / range
      }
    };
    let u_theta = scaled(theta, theta_min, theta_range);
    let u_phi = scaled(phi, phi_min, phi_range);
    match &texture_fn {
      None => Some((u_theta, u_phi)),
      Some(f) => {
        let call = Expr::CurriedCall {
          func: Box::new(f.clone()),
          args: vec![
            Expr::Real(scaled(p.x, x_min, x_range_v)),
            Expr::Real(scaled(p.y, y_min, y_range_v)),
            Expr::Real(scaled(p.z, z_min, z_range_v)),
            Expr::Real(u_theta),
            Expr::Real(u_phi),
            Expr::Real((p.x * p.x + p.y * p.y + p.z * p.z).sqrt()),
          ],
        };
        match evaluate_expr_to_expr(&call).ok()? {
          Expr::List(ref uv) if uv.len() == 2 => {
            Some((try_eval_to_f64(&uv[0])?, try_eval_to_f64(&uv[1])?))
          }
          _ => None,
        }
      }
    }
  };
  // `ColorFunction`: the colour of every facet, from `f` at the facet's
  // centroid and its centre in parameter space.
  let facet_colors: Vec<Vec<Option<(u8, u8, u8)>>> = surface_tris
    .iter()
    .enumerate()
    .map(|(surface_idx, tris)| {
      let Some(cf) = &color_fn else {
        return vec![None; tris.len()];
      };
      let unit = |v: f64, lo: f64, range: f64| {
        if range.abs() < 1e-15 {
          0.0
        } else {
          (v - lo) / range
        }
      };
      tris
        .iter()
        .enumerate()
        .map(|(tri_idx, tri)| {
          let (theta, phi) = surface_tri_params[surface_idx][tri_idx];
          let c = [
            (tri[0].x + tri[1].x + tri[2].x) / 3.0,
            (tri[0].y + tri[1].y + tri[2].y) / 3.0,
            (tri[0].z + tri[1].z + tri[2].z) / 3.0,
          ];
          let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
          let call_args = if color_scaling {
            [
              unit(c[0], x_min, x_range_v),
              unit(c[1], y_min, y_range_v),
              unit(c[2], z_min, z_range_v),
              unit(theta, theta_min, theta_range),
              unit(phi, phi_min, phi_range),
              r,
            ]
          } else {
            [c[0], c[1], c[2], theta, phi, r]
          };
          let mut value = evaluate_expr_to_expr(&Expr::CurriedCall {
            func: Box::new(cf.clone()),
            args: call_args.iter().map(|a| Expr::Real(*a)).collect(),
          })
          .ok()?;
          if let Expr::List(channels) = &value
            && channels.len() == 3
            && channels.iter().all(|ch| try_eval_to_f64(ch).is_some())
          {
            value = Expr::FunctionCall {
              name: "RGBColor".to_string(),
              args: channels.clone(),
            };
          }
          crate::functions::graphics::parse_color(&value).map(|col| {
            (
              (col.r.clamp(0.0, 1.0) * 255.0).round() as u8,
              (col.g.clamp(0.0, 1.0) * 255.0).round() as u8,
              (col.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            )
          })
        })
        .collect()
    })
    .collect();
  // The `Texture[…]` of the `PlotStyle`, carried into the symbolic structure.
  let texture_expr: Option<Expr> = plot_style_expr.and_then(find_texture_expr);

  // ── Symbolic structure: one GraphicsComplex[points, {Polygon[…], …}]
  // per surface ──
  let complexes: Vec<Expr> = surface_tris
    .iter()
    .enumerate()
    .filter(|(_, tris)| !tris.is_empty())
    .map(|(surface_idx, tris)| {
      // Deduplicate shared vertices into an indexed coordinate list. A
      // textured surface keeps vertices apart that differ in their texture
      // coordinates (the seam of a wrapped texture, the poles).
      let textured = texture_expr.is_some()
        && plot_style_for_surface(&plot_styles, surface_idx)
          .is_some_and(|s| s.texture.is_some());
      let mut point_index: std::collections::HashMap<
        (i64, i64, i64, i64, i64),
        usize,
      > = std::collections::HashMap::new();
      let mut points: Vec<Point3D> = Vec::new();
      let mut point_uvs: Vec<(f64, f64)> = Vec::new();
      let mut index_of = |p: Point3D, param: (f64, f64)| -> usize {
        let uv = if textured {
          texture_uv(p, param.0, param.1).unwrap_or((0.0, 0.0))
        } else {
          (0.0, 0.0)
        };
        let key = (
          (p.x * 1e9).round() as i64,
          (p.y * 1e9).round() as i64,
          (p.z * 1e9).round() as i64,
          (uv.0 * 1e9).round() as i64,
          (uv.1 * 1e9).round() as i64,
        );
        *point_index.entry(key).or_insert_with(|| {
          points.push(p);
          point_uvs.push(uv);
          points.len() - 1
        })
      };
      let mut tri_indices: Vec<[usize; 3]> = Vec::new();
      for (tri_idx, tri) in tris.iter().enumerate() {
        let vparams = surface_tri_vparams[surface_idx][tri_idx];
        tri_indices.push([
          index_of(tri[0], vparams[0]),
          index_of(tri[1], vparams[1]),
          index_of(tri[2], vparams[2]),
        ]);
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
          Expr::List(
            vec![Expr::Real(p.x), Expr::Real(p.y), Expr::Real(p.z)].into(),
          )
        })
        .collect();
      let polygon_of = |tris: &[[usize; 3]]| {
        call1(
          "Polygon",
          Expr::List(
            tris
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
          ),
        )
      };
      // A `ColorFunction` colours facets individually: consecutive facets
      // of one colour share a coloured `Polygon`.
      let colors = &facet_colors[surface_idx];
      let polygon_expr = if colors.iter().any(Option::is_some) {
        let mut runs: Vec<Expr> = Vec::new();
        let mut start = 0;
        while start < tri_indices.len() {
          let mut end = start + 1;
          while end < tri_indices.len() && colors[end] == colors[start] {
            end += 1;
          }
          let polygon = polygon_of(&tri_indices[start..end]);
          runs.push(match colors[start] {
            Some((r, g, b)) => Expr::List(
              vec![
                call(
                  "RGBColor",
                  vec![
                    Expr::Real(r as f64 / 255.0),
                    Expr::Real(g as f64 / 255.0),
                    Expr::Real(b as f64 / 255.0),
                  ],
                ),
                polygon,
              ]
              .into(),
            ),
            None => polygon,
          });
          start = end;
        }
        Expr::List(runs.into())
      } else {
        polygon_of(&tri_indices)
      };
      // A `PlotStyle` colour wraps the whole surface's polygons; with no
      // style given this is the bare polygon list, unchanged from before
      // multi-surface support existed.
      let mut gc_content =
        match plot_style_for_surface(&plot_styles, surface_idx)
          .and_then(|s| s.color)
        {
          Some((r, g, b)) => vec![Expr::List(
            vec![
              call(
                "RGBColor",
                vec![
                  Expr::Real(r as f64 / 255.0),
                  Expr::Real(g as f64 / 255.0),
                  Expr::Real(b as f64 / 255.0),
                ],
              ),
              polygon_expr,
            ]
            .into(),
          )],
          None => vec![polygon_expr],
        };
      if let Some(opacity) =
        surface_opacity_directive(&plot_styles, surface_idx)
      {
        gc_content.insert(0, opacity);
      }
      // The sampling triangles are far finer than any mesh the plot
      // shows, so their outlines are suppressed — otherwise extracting
      // the primitives (`plot[[1]]`) into a `Graphics3D` would turn the
      // surface into a wireframe.
      if textured && let Some(texture) = &texture_expr {
        gc_content.insert(0, texture.clone());
      }
      if !matches!(mesh_mode, MeshMode::All) {
        gc_content.insert(0, call0("EdgeForm"));
      }
      if let Some((r, g, b)) = boundary_color
        && !boundary_edges.is_empty()
      {
        let line_expr = call1(
          "Line",
          Expr::List(
            boundary_edges
              .iter()
              .map(|&(a, b)| {
                Expr::List(
                  vec![
                    Expr::Integer(a as i128 + 1),
                    Expr::Integer(b as i128 + 1),
                  ]
                  .into(),
                )
              })
              .collect::<Vec<_>>()
              .into(),
          ),
        );
        let color_expr = call(
          "RGBColor",
          vec![
            Expr::Real(r as f64 / 255.0),
            Expr::Real(g as f64 / 255.0),
            Expr::Real(b as f64 / 255.0),
          ],
        );
        gc_content.push(Expr::List(vec![color_expr, line_expr].into()));
      }
      let mut complex_args = vec![
        Expr::List(point_exprs.into()),
        Expr::List(gc_content.into()),
      ];
      if textured {
        complex_args.push(Expr::Rule {
          pattern: Box::new(Expr::Identifier(
            "VertexTextureCoordinates".to_string(),
          )),
          replacement: Box::new(Expr::List(
            point_uvs
              .iter()
              .map(|&(u, v)| {
                Expr::List(vec![Expr::Real(u), Expr::Real(v)].into())
              })
              .collect::<Vec<_>>()
              .into(),
          )),
        });
      }
      call("GraphicsComplex", complex_args)
    })
    .collect();

  let content = if complexes.len() == 1 {
    complexes.into_iter().next().expect("one complex")
  } else {
    Expr::List(complexes.into())
  };
  let structure = call(
    "Graphics3D",
    std::iter::once(content)
      .chain(args[3..].iter().cloned())
      .collect::<Vec<_>>(),
  );

  // ── Standalone rendering (the plot's own SVG) ──
  let camera = Camera::default();
  // Direction from the scene towards the viewer, used to place a
  // `Specularity` highlight from `PlotStyle`.
  let view_dir = {
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };
  let mut all_triangles: Vec<Triangle> = Vec::new();

  // Normalize a point to [-1,1] box
  let normalize = |p: Point3D| -> Point3D {
    Point3D {
      x: ((p.x - x_min) / x_range_v) * 2.0 - 1.0,
      y: ((p.y - y_min) / y_range_v) * 2.0 - 1.0,
      z: ((p.z - z_min) / z_range_v) * 2.0 * Z_SCALE - Z_SCALE,
    }
  };

  for (surface_idx, tris) in surface_tris.iter().enumerate() {
    let style = plot_style_for_surface(&plot_styles, surface_idx);
    for (tri_idx, tri) in tris.iter().enumerate() {
      let [a, b, c] = *tri;
      let na = normalize(a);
      let nb = normalize(b);
      let nc = normalize(c);

      let avg_z = ((a.z - z_min) / z_range_v
        + (b.z - z_min) / z_range_v
        + (c.z - z_min) / z_range_v)
        / 3.0;
      let default_color = height_color(avg_z);
      let normal = triangle_normal(na, nb, nc);
      // A textured surface takes its colour from the image at the facet's
      // texture coordinates instead of from its height.
      let textured = style.and_then(|s| s.texture.as_ref()).and_then(|tex| {
        let (theta, phi) = surface_tri_params[surface_idx][tri_idx];
        let centroid = Point3D {
          x: (a.x + b.x + c.x) / 3.0,
          y: (a.y + b.y + c.y) / 3.0,
          z: (a.z + b.z + c.z) / 3.0,
        };
        let (u, v) = texture_uv(centroid, theta, phi)?;
        texture_color(tex, u, v)
      });
      let base_color = facet_colors[surface_idx][tri_idx].or(textured);
      let (color, opacity) = match (style, base_color) {
        (Some(s), Some(base)) => {
          let mut painted = s.clone();
          painted.color = Some(base);
          shade_facet(default_color, Some(&painted), normal, view_dir)
        }
        (None, Some(base)) => {
          let painted = StyleState3D {
            color: Some(base),
            ..StyleState3D::default()
          };
          shade_facet(default_color, Some(&painted), normal, view_dir)
        }
        _ => shade_facet(default_color, style, normal, view_dir),
      };

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
        edge_width: None,
        projected: [pa, pb, pc],
        depth: depth(center, &camera),
        color,
        opacity,
        texture: None,
      });
    }
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
    mesh_mode,
    show_axes,
    [1.0, 1.0, Z_SCALE],
    &axes_labels,
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::plot3d_result_with_structure(svg, structure))
}

/// How far a `DiscretePlot3D` bar reaches either side of its point
/// (`ExtentSize`), as `[x_left, x_right, y_left, y_right]` plus whether
/// those are fractions of the sampling step (`Full`, `Left`, `Right`,
/// `Scaled[s]`) rather than plot units (`{{sxl, sxr}, {syl, syr}}`). `None`
/// draws points instead of bars.
fn discrete_extent(spec: &Expr) -> Option<([f64; 4], bool)> {
  let num = |e: &Expr| {
    try_eval_to_f64(&evaluate_expr_to_expr(e).unwrap_or_else(|_| e.clone()))
      .filter(|v| v.is_finite() && *v >= 0.0)
  };
  match spec {
    Expr::Identifier(n) if n == "Full" => Some(([0.5; 4], true)),
    Expr::Identifier(n) if n == "Right" => Some(([0.0, 1.0, 0.0, 1.0], true)),
    Expr::Identifier(n) if n == "Left" => Some(([1.0, 0.0, 1.0, 0.0], true)),
    Expr::FunctionCall { name, args }
      if name == "Scaled" && args.len() == 1 =>
    {
      num(&args[0]).map(|s| ([s / 2.0; 4], true))
    }
    Expr::List(axes) if axes.len() == 2 => {
      let side = |e: &Expr| match e {
        Expr::List(lr) if lr.len() == 2 => Some((num(&lr[0])?, num(&lr[1])?)),
        other => num(other).map(|w| (w, w)),
      };
      let (xl, xr) = side(&axes[0])?;
      let (yl, yr) = side(&axes[1])?;
      Some(([xl, xr, yl, yr], false))
    }
    _ => None,
  }
}

/// The spacing of the sample coordinates along one axis: the smallest gap
/// between distinct values (1 for a single value).
fn sampling_step(mut values: Vec<f64>) -> f64 {
  values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
  let step = values
    .windows(2)
    .map(|w| w[1] - w[0])
    .fold(f64::INFINITY, f64::min);
  if step.is_finite() && step > 0.0 {
    step
  } else {
    1.0
  }
}

/// A `DiscretePlot3D` function with its own style: `Style[f, sty]` gives
/// `(f, Some(sty))`; display wrappers (`Tooltip`, `Callout`, `Labeled`,
/// `PopupWindow`, …) are looked through.
fn discrete_body_style(body: &Expr) -> (Expr, Option<Expr>) {
  match body {
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      match name.as_str() {
        "Style" if args.len() >= 2 => {
          let (inner, own) = discrete_body_style(&args[0]);
          let style = if args.len() == 2 {
            args[1].clone()
          } else {
            Expr::List(args[1..].to_vec().into())
          };
          (inner, own.or(Some(style)))
        }
        "Tooltip" | "Callout" | "Labeled" | "Legended" | "PopupWindow"
        | "Annotation" | "Hyperlink" | "Mouseover" | "StatusArea" => {
          discrete_body_style(&args[0])
        }
        _ => (body.clone(), None),
      }
    }
    _ => (body.clone(), None),
  }
}

/// DiscretePlot3D[f, {i, …}, {j, …}] (and a list of functions): the values
/// of `f` at the iterator points, drawn as points with stems down to the
/// bottom of the plot (`Filling -> Bottom`), or as cuboid bars with
/// `ExtentSize`. The iterators take every `Table` form — `{i, n}`,
/// `{i, min, max}`, `{i, min, max, step}`, `{i, {values…}}` — and the second
/// may depend on the first. The plot is built as a `Graphics3D` of those
/// primitives, so the ordinary 3D renderer draws it.
pub fn discrete_plot3d_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // The functions, each with the style its own `Style` wrapper gives it; a
  // `Style` around the whole list styles every function without one.
  let bodies: Vec<(Expr, Option<Expr>)> = {
    let body = match &args[0] {
      Expr::FunctionCall { name, args: inner }
        if name == "Evaluate" && inner.len() == 1 =>
      {
        evaluate_expr_to_expr(&inner[0])?
      }
      other => other.clone(),
    };
    let (body, outer_style) = discrete_body_style(&body);
    match &body {
      Expr::List(items) => items
        .iter()
        .map(|item| {
          let (f, own) = discrete_body_style(item);
          (f, own.or_else(|| outer_style.clone()))
        })
        .collect(),
      _ => vec![(body.clone(), outer_style)],
    }
  };
  let iter_var = |spec: &Expr| match spec {
    Expr::List(items) => match items.first() {
      Some(Expr::Identifier(v)) => Some(v.clone()),
      _ => None,
    },
    _ => None,
  };
  let (Some(ivar), Some(jvar)) = (iter_var(&args[1]), iter_var(&args[2]))
  else {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D: iterators must be {var, …}".into(),
    ));
  };

  let mut extent: Option<([f64; 4], bool)> = None;
  let mut filling: Option<Expr> = Some(Expr::Identifier("Bottom".into()));
  let mut style_items: Vec<Expr> = Vec::new();
  let mut graphics_opts: Vec<Expr> = Vec::new();
  for opt in &args[3..] {
    let (pattern, replacement) = match opt {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => (pattern.as_ref(), replacement.as_ref()),
      _ => continue,
    };
    match pattern {
      Expr::Identifier(n) if n == "ExtentSize" => {
        extent = discrete_extent(replacement);
      }
      Expr::Identifier(n) if n == "Filling" => {
        filling = match replacement {
          Expr::Identifier(v) if v == "None" || v == "False" => None,
          other => Some(other.clone()),
        };
      }
      Expr::Identifier(n) if n == "PlotStyle" => {
        style_items = plot_style_items(replacement, bodies.len());
      }
      Expr::Identifier(n)
        if matches!(
          n.as_str(),
          "ColorFunction"
            | "ColorFunctionScaling"
            | "FillingStyle"
            | "PlotLegends"
            | "PlotTheme"
        ) => {}
      _ => graphics_opts.push(opt.clone()),
    }
  }

  // The {i, j, f} points of every function, from `Table`.
  let mut datasets: Vec<Vec<(f64, f64, f64)>> =
    Vec::with_capacity(bodies.len());
  for (body, _) in &bodies {
    let table = call(
      "Table",
      vec![
        Expr::List(
          vec![
            Expr::Identifier(ivar.clone()),
            Expr::Identifier(jvar.clone()),
            body.clone(),
          ]
          .into(),
        ),
        args[1].clone(),
        args[2].clone(),
      ],
    );
    let rows = evaluate_expr_to_expr(&table)?;
    let mut pts = Vec::new();
    if let Expr::List(rows) = &rows {
      for row in rows {
        if let Expr::List(cells) = row {
          pts.extend(parse_xyz_points(cells));
        }
      }
    }
    datasets.push(pts);
  }
  let (z_min, z_max) = datasets.iter().flatten().fold(
    (f64::INFINITY, f64::NEG_INFINITY),
    |(lo, hi), &(_, _, z)| (lo.min(z), hi.max(z)),
  );
  if !z_min.is_finite() {
    return Err(InterpreterError::EvaluationError(
      "DiscretePlot3D: could not compute any finite values".into(),
    ));
  }
  // Where stems and bars start: the bottom of the data (`Bottom`), the
  // xy-plane (`Axis`), or a given height.
  let base = filling.as_ref().map(|f| match f {
    Expr::Identifier(v) if v == "Axis" => 0.0,
    Expr::Identifier(v) if v == "Top" => z_max,
    other => try_eval_to_f64(other).unwrap_or(z_min),
  });

  // Bar extents in plot units, scaled by the sampling step when relative.
  let extent = extent.map(|(e, relative)| {
    if relative {
      let all = || datasets.iter().flatten();
      let dx = sampling_step(all().map(|p| p.0).collect());
      let dy = sampling_step(all().map(|p| p.1).collect());
      [e[0] * dx, e[1] * dx, e[2] * dy, e[3] * dy]
    } else {
      e
    }
  });
  let num = |v: f64| Expr::Real(v);
  let triple =
    |x: f64, y: f64, z: f64| Expr::List(vec![num(x), num(y), num(z)].into());
  let mut content: Vec<Expr> = Vec::new();
  for (k, pts) in datasets.iter().enumerate() {
    let color = if let Some(own) = &bodies[k].1 {
      own.clone()
    } else if style_items.is_empty() {
      let (r, g, b) = PLOT_COLORS[k % PLOT_COLORS.len()];
      call(
        "RGBColor",
        vec![
          num(r as f64 / 255.0),
          num(g as f64 / 255.0),
          num(b as f64 / 255.0),
        ],
      )
    } else {
      style_items[k % style_items.len()].clone()
    };
    let mut group = vec![color];
    if let Some([xl, xr, yl, yr]) = extent {
      group.push(call1("EdgeForm", call1("GrayLevel", num(0.5))));
      for &(x, y, z) in pts {
        let bottom = base.unwrap_or(z_min);
        let (lo, hi) = if z >= bottom {
          (bottom, z)
        } else {
          (z, bottom)
        };
        group.push(call(
          "Cuboid",
          vec![triple(x - xl, y - yl, lo), triple(x + xr, y + yr, hi)],
        ));
      }
    } else {
      group.push(call1("PointSize", num(0.012)));
      group.push(call1(
        "Point",
        Expr::List(pts.iter().map(|&(x, y, z)| triple(x, y, z)).collect()),
      ));
      if let Some(bottom) = base {
        let stems: Vec<Expr> = pts
          .iter()
          .map(|&(x, y, z)| {
            Expr::List(vec![triple(x, y, bottom), triple(x, y, z)].into())
          })
          .collect();
        group.push(Expr::List(
          vec![
            call1("Opacity", num(0.5)),
            call1("Line", Expr::List(stems.into())),
          ]
          .into(),
        ));
      }
    }
    content.push(Expr::List(group.into()));
  }

  let has_opt = |name: &str| {
    graphics_opts.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == name))
    })
  };
  let mut g_args = vec![Expr::List(content.into())];
  g_args.extend(graphics_opts.iter().cloned());
  if !has_opt("Axes") {
    g_args.push(crate::helpers::rule_expr(
      Expr::Identifier("Axes".into()),
      Expr::Identifier("True".into()),
    ));
  }
  if !has_opt("BoxRatios") {
    g_args.push(crate::helpers::rule_expr(
      Expr::Identifier("BoxRatios".into()),
      Expr::List(vec![Expr::Integer(1), Expr::Integer(1), num(0.4)].into()),
    ));
  }
  evaluate_expr_to_expr(&call("Graphics3D", g_args))
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
  struct Curve {
    fx: Expr,
    fy: Expr,
    fz: Expr,
  }
  let curves: Vec<Curve> = resolve_parametric_triples(body, &[tvar])?
    .into_iter()
    .map(|(fx, fy, fz)| Curve { fx, fy, fz })
    .collect();

  // Sample each curve at SAMPLE_N+1 points uniformly in [t_min, t_max].
  const SAMPLE_N: usize = 200;
  let dt = (t_max - t_min) / SAMPLE_N as f64;

  // Extract PlotStyle option (if any). Other options are kept verbatim.
  let mut plot_style: Option<&Expr> = None;
  let mut color_function: Option<Expr> = None;
  let mut color_scaling = true;
  let mut plot_range: Option<[(f64, f64); 3]> = None;
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
    } else if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ColorFunction"
    {
      // Drawn per segment below instead of being forwarded to `Graphics3D`.
      color_function = Some(replacement.as_ref().clone());
    } else if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(name) = pattern.as_ref()
      && name == "ColorFunctionScaling"
    {
      color_scaling =
        !matches!(replacement.as_ref(), Expr::Identifier(n) if n == "False");
    } else {
      if let Expr::Rule {
        pattern,
        replacement,
      } = opt
        && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotRange")
      {
        plot_range = parse_axis_ranges(replacement);
      }
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

  // One style per curve (cycling through fewer styles than curves), so
  // `PlotStyle -> {Blue, Purple}` on two curves colors them individually
  // instead of applying the whole list as a single blended directive.
  let per_curve_styles: Vec<Expr> = plot_style
    .as_ref()
    .map(|ps| plot_style_items(ps, curves.len()))
    .unwrap_or_default();

  // `ColorFunction -> f` colors the curve along its length by
  // `f[x, y, z, t]`. With `ColorFunctionScaling` (the default) x, y and z
  // arrive rescaled to 0..1 over the plot range (the explicit `PlotRange`,
  // else the sampled extent) and t over the parameter interval.
  let color_bounds: [(f64, f64); 3] = match (&color_function, plot_range) {
    (Some(_), Some(r)) => r,
    (Some(_), None) => {
      let mut b = [(f64::INFINITY, f64::NEG_INFINITY); 3];
      for c in &curves {
        for i in 0..=SAMPLE_N {
          let t = t_min + i as f64 * dt;
          if let Some((x, y, z)) =
            evaluate_parametric_at_t(&c.fx, &c.fy, &c.fz, tvar, t)
          {
            for (k, v) in [x, y, z].into_iter().enumerate() {
              b[k].0 = b[k].0.min(v);
              b[k].1 = b[k].1.max(v);
            }
          }
        }
      }
      b
    }
    _ => [(0.0, 1.0); 3],
  };
  let scale = |v: f64, (lo, hi): (f64, f64)| -> f64 {
    if !color_scaling {
      v
    } else if hi > lo {
      (v - lo) / (hi - lo)
    } else {
      0.5
    }
  };
  let segment_color = |p: [f64; 3], t: f64| -> Option<Expr> {
    let cf = color_function.as_ref()?;
    let t_scaled = if color_scaling && t_max > t_min {
      (t - t_min) / (t_max - t_min)
    } else {
      t
    };
    let (x, y, z) = (
      scale(p[0], color_bounds[0]),
      scale(p[1], color_bounds[1]),
      scale(p[2], color_bounds[2]),
    );
    if let Some(name) =
      crate::functions::field_plot::color_function_scheme_name(cf)
    {
      let (r, g, b) =
        crate::functions::field_plot::apply_named_color_function(&name, z);
      return Some(call(
        "RGBColor",
        vec![
          Expr::Real(r as f64 / 255.0),
          Expr::Real(g as f64 / 255.0),
          Expr::Real(b as f64 / 255.0),
        ],
      ));
    }
    let args = [
      Expr::Real(x),
      Expr::Real(y),
      Expr::Real(z),
      Expr::Real(t_scaled),
    ];
    crate::evaluator::function_application::apply_curried_call(cf, &args).ok()
  };

  // Build a primitives list: [<style>, Line[pts1], <style>, Line[pts2], …]
  let mut prim_items: Vec<Expr> = Vec::new();

  let mut produced_any = false;
  for (curve_idx, curve) in curves.iter().enumerate() {
    if !per_curve_styles.is_empty() {
      // Wrap whatever PlotStyle holds into a Directive[…] so that
      // collect_3d_primitives picks up nested colors/Thickness/etc.
      let style = &per_curve_styles[curve_idx % per_curve_styles.len()];
      prim_items.push(call1("Directive", style.clone()));
    } else if color_function.is_none() {
      // Without `PlotStyle`, each curve takes the next color of the
      // default plot palette (blue first) rather than plain gray.
      let (r, g, b) = crate::functions::plot::PLOT_COLORS
        [curve_idx % crate::functions::plot::PLOT_COLORS.len()];
      prim_items.push(call1(
        "Directive",
        call(
          "RGBColor",
          vec![
            Expr::Real(f64::from(r) / 255.0),
            Expr::Real(f64::from(g) / 255.0),
            Expr::Real(f64::from(b) / 255.0),
          ],
        ),
      ));
    }
    let mut current_segment: Vec<Expr> = Vec::new();
    let flush = |seg: &mut Vec<Expr>, sink: &mut Vec<Expr>| {
      if seg.len() >= 2 {
        let points = Expr::List(std::mem::take(seg).into());
        sink.push(match &tube_args {
          Some(extra) => call(
            "Tube",
            std::iter::once(points)
              .chain(extra.iter().cloned())
              .collect::<Vec<_>>(),
          ),
          None => call1("Line", points),
        });
      } else {
        seg.clear();
      }
    };
    let mut prev: Option<(f64, f64, f64, f64)> = None;
    let samples = sample_parametric_curve_adaptive(
      &curve.fx, &curve.fy, &curve.fz, tvar, t_min, t_max, SAMPLE_N,
    );
    for (t, point) in samples {
      if let Some((x, y, z)) = point {
        if color_function.is_some() {
          // One colored segment per step, joined to the previous vertex.
          if let Some((px, py, pz, pt)) = prev
            && let Some(col) = segment_color(
              [
                f64::midpoint(px, x),
                f64::midpoint(py, y),
                f64::midpoint(pz, z),
              ],
              f64::midpoint(pt, t),
            )
          {
            prim_items.push(call1("Directive", col));
            prim_items.push(call1(
              "Line",
              Expr::List(
                vec![
                  Expr::List(
                    vec![Expr::Real(px), Expr::Real(py), Expr::Real(pz)].into(),
                  ),
                  Expr::List(
                    vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)].into(),
                  ),
                ]
                .into(),
              ),
            ));
            produced_any = true;
          }
          prev = Some((x, y, z, t));
          continue;
        }
        current_segment.push(Expr::List(
          vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)].into(),
        ));
      } else {
        prev = None;
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
  // `ParametricPlot3D` draws its axes by default, unlike `Graphics3D`.
  let has_axes = forwarded_opts.iter().any(|o| {
    matches!(o, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Axes"))
  });
  g3d_args.extend(forwarded_opts);
  if !has_axes {
    g3d_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
  }

  Ok(call("Graphics3D", g3d_args))
}

/// Sample a parametric space curve on `initial_n + 1` uniform parameter
/// values, then bisect the intervals around every vertex where the polyline
/// turns sharply, so a curve that winds many times over a long parameter
/// range (an ODE trajectory, say) is drawn smoothly instead of as a coarse
/// polygon. Mathematica refines the same way. Non-finite points are kept as
/// `None` so the caller can break the line there.
fn sample_parametric_curve_adaptive(
  fx: &Expr,
  fy: &Expr,
  fz: &Expr,
  tvar: &str,
  t_min: f64,
  t_max: f64,
  initial_n: usize,
) -> Vec<(f64, Option<(f64, f64, f64)>)> {
  // Total sample cap and the cosine of the largest turning angle that
  // still counts as smooth (about 14 degrees).
  const MAX_TOTAL: usize = 6000;
  const MAX_PASSES: usize = 8;
  const SMOOTH_COS: f64 = 0.97;

  let dt = (t_max - t_min) / initial_n as f64;
  let eval = |t: f64| evaluate_parametric_at_t(fx, fy, fz, tvar, t);
  let mut samples: Vec<(f64, Option<(f64, f64, f64)>)> = (0..=initial_n)
    .map(|i| {
      let t = t_min + i as f64 * dt;
      (t, eval(t))
    })
    .collect();

  for _ in 0..MAX_PASSES {
    if samples.len() >= MAX_TOTAL {
      break;
    }
    let n = samples.len();
    let mut split = vec![false; n - 1];
    for i in 1..n - 1 {
      let (Some(a), Some(b), Some(c)) =
        (samples[i - 1].1, samples[i].1, samples[i + 1].1)
      else {
        continue;
      };
      let u = [b.0 - a.0, b.1 - a.1, b.2 - a.2];
      let v = [c.0 - b.0, c.1 - b.1, c.2 - b.2];
      let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
      let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
      if nu == 0.0 || nv == 0.0 {
        continue;
      }
      let cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv);
      if cos < SMOOTH_COS {
        split[i - 1] = true;
        split[i] = true;
      }
    }
    let mut refined = Vec::with_capacity(n * 2);
    let mut added = 0;
    for i in 0..n - 1 {
      refined.push(samples[i]);
      let (t0, t1) = (samples[i].0, samples[i + 1].0);
      if split[i]
        && (t1 - t0).abs() > (t_max - t_min).abs() * 1e-9
        && samples.len() + added < MAX_TOTAL
      {
        let tm = f64::midpoint(t0, t1);
        refined.push((tm, eval(tm)));
        added += 1;
      }
    }
    refined.push(samples[n - 1]);
    if added == 0 {
      break;
    }
    samples = refined;
  }
  samples
}

/// Evaluate a parametric triple {fx(t), fy(t), fz(t)} at a given t value.
fn evaluate_parametric_at_t(
  fx: &Expr,
  fy: &Expr,
  fz: &Expr,
  tvar: &str,
  tval: f64,
) -> Option<(f64, f64, f64)> {
  // The plot sampler's own evaluator: it also resolves a component that
  // depends on `t` only through a variable's value (`c = ArcTan[t/5]`).
  let eval_one = |body: &Expr| -> Option<f64> {
    crate::functions::plot::evaluate_at_point(body, tvar, tval)
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

/// Peel off wrapping single-element lists, e.g. `{{fx, fy, fz}}`. A
/// `soln = NDSolve[...]` result is a list of solution branches even when
/// there is only one, so `{fx, fy, fz} /. soln` threads over that outer
/// list and comes back as a 1-element list around the triple rather than
/// the triple itself.
fn unwrap_singleton_list(mut e: &Expr) -> &Expr {
  while let Expr::List(sub) = e {
    if sub.len() == 1 {
      e = &sub[0];
    } else {
      break;
    }
  }
  e
}

/// Resolve one curve/surface item into a `{fx, fy, fz}` triple: either it is
/// already a literal 3-list, or it is a held expression like
/// `{x[t], z[t], y[t]} /. soln` (an unevaluated `ReplaceAll` around a
/// literal triple) that must be evaluated once, with `shadow_vars` cleared,
/// to discover its shape.
fn resolve_one_parametric_triple(
  item: &Expr,
  shadow_vars: &[&str],
) -> Option<(Expr, Expr, Expr)> {
  if let Expr::List(sub) = unwrap_singleton_list(item) {
    return (sub.len() == 3)
      .then(|| (sub[0].clone(), sub[1].clone(), sub[2].clone()));
  }
  let resolved =
    crate::functions::plot::eval_body_vars_symbolic(item, shadow_vars);
  match unwrap_singleton_list(&resolved) {
    Expr::List(sub) if sub.len() == 3 => {
      Some((sub[0].clone(), sub[1].clone(), sub[2].clone()))
    }
    // Stays symbolic until the parameters are numeric (e.g. an `If` on a
    // sum of squares of `u`, `v`): probe at a sample point and, if it is
    // a numeric 3-vector, read its components off with `Part`.
    _ => {
      let mut probe = item.clone();
      for (i, var) in shadow_vars.iter().enumerate() {
        probe =
          substitute_var(&probe, var, &Expr::Real(0.37 + 0.24 * i as f64));
      }
      let value = evaluate_expr_to_expr(&probe).ok()?;
      match &value {
        Expr::List(sub)
          if sub.len() == 3
            && sub.iter().all(|c| try_eval_to_f64(c).is_some()) =>
        {
          Some((part_of(item, 1), part_of(item, 2), part_of(item, 3)))
        }
        _ => None,
      }
    }
  }
}

/// Read an already-evaluated shape as zero, one, or several `{fx, fy, fz}`
/// triples: `{}` contributes no curve (a Demonstration idiom, `If[cond,
/// Through[{f, g}][##]&][...], {}]`, picks between a fixed set of curves and
/// "no curve" depending on a control's value), a literal 3-list is one
/// curve, and a list whose every element is itself a 3-list is that many
/// curves (e.g. `Through[{f, g}][t, ...]` producing two curves at once).
fn parametric_curves_from_list(e: &Expr) -> Option<Vec<(Expr, Expr, Expr)>> {
  let Expr::List(sub) = e else {
    return None;
  };
  if sub.is_empty() {
    return Some(vec![]);
  }
  let as_multi: Option<Vec<(Expr, Expr, Expr)>> = sub
    .iter()
    .map(|s| match unwrap_singleton_list(s) {
      Expr::List(inner) if inner.len() == 3 => {
        Some((inner[0].clone(), inner[1].clone(), inner[2].clone()))
      }
      _ => None,
    })
    .collect();
  if let Some(curves) = as_multi {
    return Some(curves);
  }
  (sub.len() == 3)
    .then(|| vec![(sub[0].clone(), sub[1].clone(), sub[2].clone())])
}

/// Resolve one item of a `ParametricPlot3D` curve list into zero, one, or
/// several `{fx, fy, fz}` triples (see `parametric_curves_from_list`),
/// evaluating a held expression (with `shadow_vars` cleared) when it is not
/// already a literal list, exactly as `resolve_one_parametric_triple` does
/// for the single-curve case.
fn resolve_one_parametric_item_to_curves(
  item: &Expr,
  shadow_vars: &[&str],
) -> Option<Vec<(Expr, Expr, Expr)>> {
  if let Expr::List(_) = unwrap_singleton_list(item)
    && let Some(curves) =
      parametric_curves_from_list(unwrap_singleton_list(item))
  {
    return Some(curves);
  }
  let resolved =
    crate::functions::plot::eval_body_vars_symbolic(item, shadow_vars);
  if let Some(curves) =
    parametric_curves_from_list(unwrap_singleton_list(&resolved))
  {
    return Some(curves);
  }
  // A body that only takes shape for numbers — a compiled function, which
  // stays unevaluated on free plot variables — is probed at a sample point:
  // one numeric 3-vector is one curve, a list of them that many, read off
  // with `Part` so each still varies with the plot variables.
  let mut probe = item.clone();
  for (i, var) in shadow_vars.iter().enumerate() {
    probe = substitute_var(&probe, var, &Expr::Real(0.37 + 0.24 * i as f64));
  }
  let value = evaluate_expr_to_expr(&probe).ok()?;
  let is_point = |e: &Expr| {
    matches!(e, Expr::List(c) if c.len() == 3
      && c.iter().all(|x| try_eval_to_f64(x).is_some()))
  };
  match &value {
    Expr::List(sub) if sub.is_empty() => Some(vec![]),
    v if is_point(v) => {
      Some(vec![(part_of(item, 1), part_of(item, 2), part_of(item, 3))])
    }
    Expr::List(sub) if sub.iter().all(is_point) => Some(
      (1..=sub.len() as i128)
        .map(|i| {
          let curve = part_of(item, i);
          (part_of(&curve, 1), part_of(&curve, 2), part_of(&curve, 3))
        })
        .collect(),
    ),
    _ => None,
  }
}

/// Resolve `body` into one or more `{fx, fy, fz}` triples for
/// `ParametricPlot3D`. `body` normally is already a literal `{fx, fy, fz}`
/// (or a list of such triples), but a Demonstration idiom passes a helper
/// call instead — e.g. `ParametricPlot3D[projection[u, v, ...], {u, ...},
/// {v, ...}]`, or wraps each curve of a multi-curve plot in a `ReplaceAll`
/// like `{{fx, fy, fz} /. soln, {gx, gy, gz} /. soln}` — which `HoldAll`
/// leaves unevaluated and which must be evaluated once to discover its
/// shape. `shadow_vars` (the iterator variables) are cleared for that one
/// evaluation so a stray global left over from elsewhere in the notebook
/// (e.g. an `Initialization :> (u = 0.; ...)`) doesn't freeze the surface at
/// a single point instead of leaving `u`/`v` symbolic for the later
/// per-sample substitution.
fn resolve_parametric_triples(
  body: &Expr,
  shadow_vars: &[&str],
) -> Result<Vec<(Expr, Expr, Expr)>, InterpreterError> {
  // A component written as a symbol whose value depends on the plot
  // variables (`Module[{x, u, v}, x = Sin[v]; ParametricPlot3D[{x, u, v},
  // …]]`) would never see the per-sample substitution; read its value first.
  // Only a bare symbol: a `Part[item, k]` read-off must stay as it is, or
  // evaluating it symbolically takes a part of the unevaluated `If` itself.
  let resolve = |c: Expr| match c {
    Expr::Identifier(_) => {
      crate::functions::plot::resolve_indirect_plot_body(&c, shadow_vars)
        .unwrap_or(c)
    }
    other => other,
  };
  Ok(
    resolve_parametric_triples_raw(body, shadow_vars)?
      .into_iter()
      .map(|(x, y, z)| (resolve(x), resolve(y), resolve(z)))
      .collect(),
  )
}

fn resolve_parametric_triples_raw(
  body: &Expr,
  shadow_vars: &[&str],
) -> Result<Vec<(Expr, Expr, Expr)>, InterpreterError> {
  let err = || {
    InterpreterError::EvaluationError(
      "ParametricPlot3D: first argument must be {fx, fy, fz}".into(),
    )
  };
  // A single `{fx, fy, fz}` triple, or a list of such triples (each item
  // possibly still needing its own resolution, e.g. a held `helper[u, v]`
  // call or a `{fx, fy, fz} /. rules` substitution).
  let resolve_items =
    |items: &[Expr]| -> Result<Vec<(Expr, Expr, Expr)>, InterpreterError> {
      if items.is_empty() {
        return Err(err());
      }
      // Exactly three items whose first is not already a literal list are
      // ambiguous: either the three components of one curve/surface, or
      // three whole curves that only take triple shape once each item is
      // evaluated — the `{{0, x[t], y[t]} /. sol, {t, x[t], 0} /. sol,
      // {t, c, y[t]} /. sol}` shape a Demonstration uses to draw an
      // `NDSolve` trajectory beside its two coordinate projections. Only
      // the latter has *every* item resolve to a triple of its own
      // (a scalar component like `Sin[t]` or `0` never does), so try that
      // reading first and fall back to the single-triple one. Wolfram
      // disambiguates the same way, by the depth of the evaluated body.
      if items.len() == 3 && !matches!(&items[0], Expr::List(_)) {
        let as_three_curves: Option<Vec<(Expr, Expr, Expr)>> = items
          .iter()
          .map(|item| resolve_one_parametric_triple(item, shadow_vars))
          .collect();
        return Ok(as_three_curves.unwrap_or_else(|| {
          vec![(items[0].clone(), items[1].clone(), items[2].clone())]
        }));
      }
      let mut curves = Vec::new();
      for item in items {
        match resolve_one_parametric_item_to_curves(item, shadow_vars) {
          Some(item_curves) => curves.extend(item_curves),
          None => return Err(err()),
        }
      }
      Ok(curves)
    };
  match body {
    Expr::List(items) => resolve_items(items),
    other => {
      let resolved =
        crate::functions::plot::eval_body_vars_symbolic(other, shadow_vars);
      match unwrap_singleton_list(&resolved) {
        // A symbolic list that is no curve can still be one once the
        // variables are numbers (a `Piecewise` of points, whose `Part`s and
        // `Flatten` only make sense for a chosen piece): read the body off
        // at each sample, as the probe in
        // `resolve_one_parametric_item_to_curves` decides.
        Expr::List(items) => resolve_items(items).or_else(|e| {
          resolve_one_parametric_item_to_curves(other, shadow_vars).ok_or(e)
        }),
        // The body only yields a point once `u`/`v` are numeric (e.g. a
        // `BSplineFunction` surface `f[u, v]`), so read the components off
        // the evaluated body at each sample with `Part`.
        _ => Ok(vec![(
          part_of(other, 1),
          part_of(other, 2),
          part_of(other, 3),
        )]),
      }
    }
  }
}

fn part_of(body: &Expr, index: i128) -> Expr {
  call("Part", vec![body.clone(), Expr::Integer(index)])
}

/// Evaluate a parametric triple {fx(u,v), fy(u,v), fz(u,v)} at given u, v values.
fn evaluate_parametric_at_uv(
  fx: &crate::functions::plot::PlotBody2,
  fy: &crate::functions::plot::PlotBody2,
  fz: &crate::functions::plot::PlotBody2,
  uval: f64,
  vval: f64,
) -> Option<(f64, f64, f64)> {
  let x = fx.at(uval, vval)?;
  let y = fy.at(uval, vval)?;
  let z = fz.at(uval, vval)?;
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

  // `Legended[curve, label]` curves are plotted as the curves themselves.
  let unlegend = |e: &Expr| match e {
    Expr::FunctionCall { name, args }
      if name == "Legended" && !args.is_empty() =>
    {
      args[0].clone()
    }
    other => other.clone(),
  };
  let body = &match &unlegend(&args[0]) {
    Expr::List(items) => Expr::List(items.iter().map(unlegend).collect()),
    other => other.clone(),
  };

  // Parse the first iterator. If the second argument also parses as an
  // iterator, fall through to the surface (2-iterator) implementation.
  // Otherwise treat as a curve and return early.
  let (uvar, u_min, u_max) = parse_iterator(&args[1], "first")?;
  let surface_iter = args.get(2).and_then(|a| {
    parse_iterator(a, "second")
      .ok()
      .map(|(v, lo, hi)| (v, Expr::Real(lo), Expr::Real(hi)))
      .or_else(|| parse_dependent_iterator(a, &uvar))
  });

  let Some((vvar, v_lo_expr, v_hi_expr)) = surface_iter else {
    return parametric_plot3d_curve_ast(body, &uvar, u_min, u_max, &args[2..]);
  };

  // Parse options
  let mut svg_width = DEFAULT_SIZE;
  let mut svg_height = DEFAULT_SIZE;
  let mut full_width = false;
  let mut mesh_mode = MeshMode::Default;
  // `Mesh -> n`: `n` evenly spaced mesh lines in each parameter direction.
  let mut mesh_count: Option<usize> = None;
  let mut show_axes = true;
  let mut plot_style_expr: Option<&Expr> = None;
  // `MeshFunctions -> {f}` / `Mesh -> {{v1, v2, ...}}` / `MeshShading ->
  // {c1, c2, ...}`: colour each facet by which region of `f`'s range
  // (partitioned at the sorted break values) its average point falls in,
  // cycling through `MeshShading`'s colours in that same order. Only a
  // single mesh function is supported; a nested `Mesh` value for more than
  // one keeps just the first function's break list.
  let mut mesh_function: Option<Expr> = None;
  let mut mesh_breaks: Vec<f64> = Vec::new();
  let mut mesh_shading: Vec<Option<(u8, u8, u8)>> = Vec::new();
  // `MeshStyle -> None`: the facet grid drawn by `mesh_mode`'s default
  // `EdgeForm[]` is suppressed, leaving flat colour with no outline —
  // the usual way a mesh-shaded dissection hides its own seams.
  let mut mesh_style_none = false;
  // `BoxRatios -> {rx, ry, rz}` reshapes the bounding box to those
  // proportions regardless of the surface's actual extent along each axis.
  // Otherwise (Wolfram's default `Automatic`, unlike `Plot3D`'s
  // `{1, 1, 0.4}`) the box is shaped like the data itself.
  let mut box_ratios_explicit: Option<[f64; 3]> = None;
  // `PlotRange -> {{x0,x1},{y0,y1},{z0,z1}}` pins the displayed region
  // regardless of the sampled surface's own extent, the same way it does
  // for `Graphics3D` — needed so a `BoxRatios` given alongside it (as
  // `Manipulate`-driven demonstrations commonly do, to keep the framing
  // stable as the surface changes) reshapes the region that was asked
  // for, not just whatever the current frame happens to sample.
  let mut plot_range: Option<[(f64, f64); 3]> = None;
  // `PlotPoints -> n` (or `{nu, nv}`, of which the first is used): `n`
  // samples per direction, i.e. `n - 1` grid cells. Demonstrations lower it
  // (`PlotPoints -> 15`) to keep a `Table` of many patches interactive.
  let mut plot_points: Option<usize> = None;
  // `RegionFunction -> f`: keep only the parts of the surface where
  // `f[x, y, z, u, v]` is `True`.
  let mut region_fn: Option<Expr> = None;
  // `ColorFunction -> f`: colour each facet by `f[x, y, z, u, v]`. With
  // `ColorFunctionScaling` (the default) x, y and z arrive rescaled to 0..1
  // over the plotted region and u, v over their parameter intervals.
  let mut surface_color_fn: Option<Expr> = None;
  let mut color_scaling = true;

  for opt in &args[3..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
    {
      match pattern.as_ref() {
        Expr::Identifier(name) if name == "PlotStyle" => {
          plot_style_expr = Some(replacement.as_ref());
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
        Expr::Identifier(name) if name == "ColorFunction" => {
          surface_color_fn = Some(replacement.as_ref().clone());
        }
        Expr::Identifier(name) if name == "ColorFunctionScaling" => {
          color_scaling = !matches!(replacement.as_ref(), Expr::Identifier(n) if n == "False");
        }
        Expr::Identifier(name) if name == "RegionFunction" => {
          region_fn = Some(
            evaluate_expr_to_expr(replacement)
              .unwrap_or_else(|_| replacement.as_ref().clone()),
          );
        }
        Expr::Identifier(name) if name == "BoxRatios" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          if let Some(r) = eval_vec3(&value)
            && r.iter().all(|v| *v > 0.0)
          {
            box_ratios_explicit = Some(r);
          }
        }
        Expr::Identifier(name) if name == "PlotRange" => {
          let value = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          if let Some(ranges) = parse_axis_ranges(&value) {
            plot_range = Some(ranges);
          }
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
            Expr::Identifier(n) if n == "None" => mesh_mode = MeshMode::None,
            Expr::Identifier(n) if n == "All" => mesh_mode = MeshMode::All,
            Expr::Integer(n) if *n >= 0 => mesh_count = Some(*n as usize),
            Expr::List(items) => {
              let level_items: &[Expr] =
                if items.iter().all(|it| matches!(it, Expr::List(_))) {
                  match items.first() {
                    Some(Expr::List(first)) => first,
                    _ => &[],
                  }
                } else {
                  items
                };
              let mut breaks: Vec<f64> = level_items
                .iter()
                .filter_map(|item| {
                  let v = evaluate_expr_to_expr(item)
                    .unwrap_or_else(|_| item.clone());
                  try_eval_to_f64(&v)
                })
                .collect();
              breaks.sort_by(|a, b| a.partial_cmp(b).unwrap());
              mesh_breaks = breaks;
            }
            _ => {}
          }
        }
        Expr::Identifier(name) if name == "MeshFunctions" => {
          mesh_function = match replacement.as_ref() {
            Expr::List(items) => items.first().cloned(),
            other => Some(other.clone()),
          };
        }
        Expr::Identifier(name) if name == "MeshShading" => {
          // The value is often computed rather than written literally
          // (`MeshShading -> Which[cond1, {c1, c2}, cond2, {c3, c4}, ...]`
          // picking a colour list from the current mesh breaks), so it must
          // be evaluated before checking its shape.
          let resolved = evaluate_expr_to_expr(replacement)
            .unwrap_or_else(|_| replacement.as_ref().clone());
          if let Expr::List(items) = &resolved {
            mesh_shading = items
              .iter()
              .map(|item| {
                let v =
                  evaluate_expr_to_expr(item).unwrap_or_else(|_| item.clone());
                crate::functions::graphics::parse_color(&v).map(|c| {
                  (
                    (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                    (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                  )
                })
              })
              .collect();
          }
        }
        Expr::Identifier(name) if name == "MeshStyle" => {
          if matches!(replacement.as_ref(), Expr::Identifier(s) if s == "None")
          {
            mesh_style_none = true;
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
  let mesh_shading_active = mesh_function.is_some()
    && !mesh_breaks.is_empty()
    && !mesh_shading.is_empty();

  // Parse parametric surfaces: body must be {fx, fy, fz} or {{fx1, fy1, fz1}, ...}
  struct ParametricSurface {
    fx: Expr,
    fy: Expr,
    fz: Expr,
  }

  let surfaces: Vec<ParametricSurface> =
    resolve_parametric_triples(body, &[&uvar, &vvar])?
      .into_iter()
      .map(|(fx, fy, fz)| ParametricSurface { fx, fy, fz })
      .collect();

  // `PlotStyle`, one style per surface (cycling); empty means it was not
  // given, so the height-based rainbow default is used unchanged.
  let plot_styles: Vec<StyleState3D> = plot_style_expr
    .map(|e| parse_plot_style_3d(e, surfaces.len()))
    .unwrap_or_default();

  let camera = Camera::default();
  // Direction from the scene towards the viewer, used to place a
  // `Specularity` highlight from `PlotStyle`.
  let view_dir = {
    let (sa, ca) = camera.azimuth.sin_cos();
    let (se, ce) = camera.elevation.sin_cos();
    [ce * ca, ce * sa, se]
  };
  // Without an explicit `PlotPoints` the grid is 50 samples per direction,
  // refined to about 20 per full turn (2π) of a long parameter span, so a
  // surface winding many times around (a Klein bottle over `{u, 0, 30 π}`)
  // is not reduced to a handful of facets per turn.
  let auto_grid = |span: f64| -> usize {
    let turns = (span.abs() / std::f64::consts::TAU).ceil();
    if turns.is_finite() {
      ((turns as usize).saturating_mul(20)).clamp(GRID_N, 300)
    } else {
      GRID_N
    }
  };
  let grid_u = plot_points.map_or_else(|| auto_grid(u_max - u_min), |p| p - 1);
  let u_step = (u_max - u_min) / grid_u as f64;
  // The `v` range may depend on `u` (`{v, 0, h - Sin[u]}`), so each grid
  // column `i` gets its own `(lo, hi)`; a column whose bounds do not
  // evaluate to numbers stays empty.
  let v_ranges: Vec<(f64, f64)> = (0..=grid_u)
    .map(|i| {
      let uval = u_min + i as f64 * u_step;
      let bound = |e: &Expr| {
        let sub = substitute_var(e, &uvar, &Expr::Real(uval));
        evaluate_expr_to_expr(&sub)
          .ok()
          .and_then(|r| try_eval_to_f64(&r))
          .unwrap_or(f64::NAN)
      };
      (bound(&v_lo_expr), bound(&v_hi_expr))
    })
    .collect();
  let grid_v = plot_points.map_or_else(
    || {
      let span = v_ranges
        .iter()
        .filter(|(lo, hi)| lo.is_finite() && hi.is_finite())
        .map(|(lo, hi)| hi - lo)
        .fold(0.0_f64, f64::max);
      auto_grid(span)
    },
    |p| p - 1,
  );
  let v_step_at = |i: usize| (v_ranges[i].1 - v_ranges[i].0) / grid_v as f64;
  let (v_min, v_max) = v_ranges
    .iter()
    .filter(|(lo, hi)| lo.is_finite() && hi.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), (lo, hi)| {
      (a.min(*lo), b.max(*hi))
    });

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
    let mut points = vec![vec![None; grid_v + 1]; grid_u + 1];
    let (fx, fy, fz) = (
      crate::functions::plot::PlotBody2::new(&surface.fx, &uvar, &vvar),
      crate::functions::plot::PlotBody2::new(&surface.fy, &uvar, &vvar),
      crate::functions::plot::PlotBody2::new(&surface.fz, &uvar, &vvar),
    );
    for i in 0..=grid_u {
      let uval = u_min + i as f64 * u_step;
      for j in 0..=grid_v {
        let vval = v_ranges[i].0 + j as f64 * v_step_at(i);
        if let Some((x, y, z)) =
          evaluate_parametric_at_uv(&fx, &fy, &fz, uval, vval)
        {
          if let Some(region) = &region_fn {
            let call = Expr::CurriedCall {
              func: Box::new(region.clone()),
              args: vec![
                Expr::Real(x),
                Expr::Real(y),
                Expr::Real(z),
                Expr::Real(uval),
                Expr::Real(vval),
              ],
            };
            let inside = matches!(
              evaluate_expr_to_expr(&call),
              Ok(Expr::Identifier(ref s)) if s == "True"
            );
            if !inside {
              continue;
            }
          }
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

  // An explicit `PlotRange` pins the displayed region regardless of the
  // sampled surface's own extent, matching `Graphics3D`.
  if let Some([(xl, xh), (yl, yh), (zl, zh)]) = plot_range {
    gx_min = xl;
    gx_max = xh;
    gy_min = yl;
    gy_max = yh;
    gz_min = zl;
    gz_max = zh;
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

  let box_ratios: [f64; 3] = box_ratios_explicit.unwrap_or([rx, ry, rz]);

  // `ColorFunction`: the colour of facet `(i, j)`, from `f` at the facet's
  // mean point and its centre in parameter space.
  let facet_colors: Vec<Vec<Vec<Option<(u8, u8, u8)>>>> = all_surface_points
    .iter()
    .map(|sg| {
      (0..grid_u)
        .map(|i| {
          (0..grid_v)
            .map(|j| {
              let cf = surface_color_fn.as_ref()?;
              let pts: Vec<(f64, f64, f64)> =
                [sg[i][j], sg[i + 1][j], sg[i + 1][j + 1], sg[i][j + 1]]
                  .iter()
                  .filter_map(|p| *p)
                  .collect();
              if pts.is_empty() {
                return None;
              }
              let n = pts.len() as f64;
              let x = pts.iter().map(|p| p.0).sum::<f64>() / n;
              let y = pts.iter().map(|p| p.1).sum::<f64>() / n;
              let z = pts.iter().map(|p| p.2).sum::<f64>() / n;
              let u = u_min + (i as f64 + 0.5) * u_step;
              let v = v_ranges[i].0 + (j as f64 + 0.5) * v_step_at(i);
              let args = if color_scaling {
                let unit = |t: f64, lo: f64, hi: f64| {
                  if (hi - lo).abs() < 1e-15 {
                    0.0
                  } else {
                    (t - lo) / (hi - lo)
                  }
                };
                [
                  unit(x, gx_min, gx_max),
                  unit(y, gy_min, gy_max),
                  unit(z, gz_min, gz_max),
                  unit(u, u_min, u_max),
                  unit(v, v_min, v_max),
                ]
              } else {
                [x, y, z, u, v]
              };
              let call = Expr::CurriedCall {
                func: Box::new(cf.clone()),
                args: args.iter().map(|a| Expr::Real(*a)).collect(),
              };
              let mut value = evaluate_expr_to_expr(&call).ok()?;
              // A bare `{r, g, b}` list (a sampled texture) stands for the
              // colour `RGBColor[r, g, b]`.
              if let Expr::List(channels) = &value
                && channels.len() == 3
                && channels.iter().all(|c| try_eval_to_f64(c).is_some())
              {
                value = Expr::FunctionCall {
                  name: "RGBColor".to_string(),
                  args: channels.clone(),
                };
              }
              crate::functions::graphics::parse_color(&value).map(|c| {
                (
                  (c.r.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (c.g.clamp(0.0, 1.0) * 255.0).round() as u8,
                  (c.b.clamp(0.0, 1.0) * 255.0).round() as u8,
                )
              })
            })
            .collect()
        })
        .collect()
    })
    .collect();

  // ── Symbolic structure: Graphics3D[GraphicsComplex[points, {…}]] ──
  // `Show` merges the *primitives* of the graphics it is given, so a
  // surface that exists only as a rendering is dropped when it is shown
  // together with a `Graphics3D`. Emitting the sampled surface in world
  // coordinates — coloured by height, the way the standalone render
  // colours it — lets `Show[ParametricPlot3D[…], Graphics3D[…]]` draw both.
  let structure = {
    let complexes: Vec<Expr> = all_surface_points
      .iter()
      .enumerate()
      .map(|(surface_idx, sg)| {
        let mut index_of: Vec<Vec<Option<usize>>> =
          vec![vec![None; grid_v + 1]; grid_u + 1];
        let mut point_exprs: Vec<Expr> = Vec::new();
        for (i, row) in sg.iter().enumerate() {
          for (j, p) in row.iter().enumerate() {
            if let Some((x, y, z)) = p {
              index_of[i][j] = Some(point_exprs.len());
              point_exprs.push(Expr::List(
                vec![Expr::Real(*x), Expr::Real(*y), Expr::Real(*z)].into(),
              ));
            }
          }
        }
        let mut content: Vec<Expr> = Vec::new();
        // Bare `EdgeForm[]` (no styling directive) asks for no facet
        // outline at all (see `apply_3d_directive`'s "EdgeForm" arm), which
        // is the plot's own default (`MeshMode::Default`/`::None`) as well
        // as what an explicit `MeshStyle -> None` asks for even when
        // `Mesh -> All` would otherwise leave edges at the struct's
        // default-on.
        if mesh_style_none || !matches!(mesh_mode, MeshMode::All) {
          content.push(call0("EdgeForm"));
        }
        content.extend(surface_opacity_directive(&plot_styles, surface_idx));
        for i in 0..grid_u {
          for j in 0..grid_v {
            let (Some(a), Some(b), Some(c), Some(d)) = (
              index_of[i][j],
              index_of[i + 1][j],
              index_of[i + 1][j + 1],
              index_of[i][j + 1],
            ) else {
              continue;
            };
            let corners =
              [sg[i][j], sg[i + 1][j], sg[i + 1][j + 1], sg[i][j + 1]];
            let avg_z = corners
              .iter()
              .filter_map(|p| p.map(|(_, _, z)| z))
              .sum::<f64>()
              / 4.0;
            let mesh_color = mesh_shading_active.then(|| {
              let avg_x = corners
                .iter()
                .filter_map(|p| p.map(|(x, _, _)| x))
                .sum::<f64>()
                / 4.0;
              let avg_y = corners
                .iter()
                .filter_map(|p| p.map(|(_, y, _)| y))
                .sum::<f64>()
                / 4.0;
              eval_mesh_function(
                mesh_function.as_ref().expect("mesh_shading_active"),
                avg_x,
                avg_y,
                avg_z,
              )
              .and_then(|val| {
                let region = mesh_breaks.iter().filter(|&&b| b <= val).count();
                // More regions than colours: cycle, the same convention
                // `PlotStyle`'s per-surface colours already use.
                mesh_shading
                  .get(region % mesh_shading.len())
                  .copied()
                  .flatten()
              })
            });
            // Only an explicit colour (`PlotStyle`, `MeshShading`) is baked
            // into a facet. Facets of the default look carry none, like
            // wolframscript's `ParametricPlot3D[…][[1]]`, so a colour
            // directive placed before the extracted complex still applies.
            let explicit = facet_colors[surface_idx][i][j].or_else(|| {
              match plot_style_for_surface(&plot_styles, surface_idx) {
                Some(style) => style.color,
                None => mesh_color.flatten(),
              }
            });
            let polygon = call1(
              "Polygon",
              Expr::List(
                [a, b, c, d]
                  .iter()
                  .map(|&k| Expr::Integer(k as i128 + 1))
                  .collect::<Vec<_>>()
                  .into(),
              ),
            );
            let Some((cr, cg, cb)) = explicit else {
              content.push(polygon);
              continue;
            };
            content.push(Expr::List(
              vec![
                call(
                  "RGBColor",
                  vec![
                    Expr::Real(cr as f64 / 255.0),
                    Expr::Real(cg as f64 / 255.0),
                    Expr::Real(cb as f64 / 255.0),
                  ],
                ),
                polygon,
              ]
              .into(),
            ));
          }
        }
        if let Some(n) = mesh_count.filter(|&n| n > 0) {
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
          for m in 1..=n {
            let ku = (grid_u * m + n.div_ceil(2)) / (n + 1);
            let kv = (grid_v * m + n.div_ceil(2)) / (n + 1);
            for i in 0..grid_u {
              push_segment(index_of[i][kv], index_of[i + 1][kv]);
            }
            for j in 0..grid_v {
              push_segment(index_of[ku][j], index_of[ku][j + 1]);
            }
          }
          if !segments.is_empty() {
            content.push(Expr::List(
              vec![
                call1("Opacity", Expr::Real(0.63)),
                call(
                  "RGBColor",
                  vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
                ),
                call1("AbsoluteThickness", Expr::Real(0.5)),
                call1("Line", Expr::List(segments.into())),
              ]
              .into(),
            ));
          }
        }
        call(
          "GraphicsComplex",
          vec![Expr::List(point_exprs.into()), Expr::List(content.into())],
        )
      })
      .collect();
    let content = if complexes.len() == 1 {
      complexes.into_iter().next().expect("one complex")
    } else {
      Expr::List(complexes.into())
    };
    // `ParametricPlot3D` draws axes where a bare `Graphics3D` does not, so
    // that default is spelled out: a surface shown inside another graphic
    // — with `ParametricPlot3D` itself first in the `Show` — keeps them.
    // Its box is shaped like the data, which is `Graphics3D`'s own default.
    let mut structure_args = vec![content];
    structure_args.extend(args[3..].iter().cloned());
    let names_axes = structure_args.iter().any(|o| {
      matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "Axes"))
    });
    if !names_axes {
      structure_args.push(rule_expr(id_expr("Axes"), bool_expr(true)));
    }
    call("Graphics3D", structure_args)
  };

  // Phase 2: Build triangles
  let mut all_triangles: Vec<Triangle> = Vec::new();

  for (surface_idx, sg) in all_surface_points.iter().enumerate() {
    let style = plot_style_for_surface(&plot_styles, surface_idx);
    for i in 0..grid_u {
      for j in 0..grid_v {
        let p00 = sg[i][j];
        let p10 = sg[i + 1][j];
        let p01 = sg[i][j + 1];
        let p11 = sg[i + 1][j + 1];

        let normalize = |p: (f64, f64, f64)| -> Point3D {
          Point3D {
            x: ((p.0 - gx_min) / rx) * 2.0 * box_ratios[0] - box_ratios[0],
            y: ((p.1 - gy_min) / ry) * 2.0 * box_ratios[1] - box_ratios[1],
            z: ((p.2 - gz_min) / rz) * 2.0 * box_ratios[2] - box_ratios[2],
          }
        };

        let z_norm = |z: f64| -> f64 { (z - gz_min) / rz };

        // `MeshFunctions`/`Mesh`/`MeshShading`: override a facet's default
        // height-based colour with the region colour its average `(x, y,
        // z)` falls in — only when there is no `PlotStyle` for this surface,
        // matching Phase 1's `content` colouring above.
        let mesh_facet_color = |a: (f64, f64, f64),
                                b: (f64, f64, f64),
                                c: (f64, f64, f64)|
         -> Option<(u8, u8, u8)> {
          if style.is_some() || !mesh_shading_active {
            return None;
          }
          let avg_x = (a.0 + b.0 + c.0) / 3.0;
          let avg_y = (a.1 + b.1 + c.1) / 3.0;
          let avg_z = (a.2 + b.2 + c.2) / 3.0;
          let val = eval_mesh_function(
            mesh_function.as_ref().expect("mesh_shading_active"),
            avg_x,
            avg_y,
            avg_z,
          )?;
          let region = mesh_breaks.iter().filter(|&&br| br <= val).count();
          // More regions than colours: cycle, the same convention
          // `PlotStyle`'s per-surface colours already use.
          mesh_shading
            .get(region % mesh_shading.len())
            .copied()
            .flatten()
        };

        // Triangle 1
        if let (Some(a), Some(b), Some(c)) = (p00, p10, p01) {
          let v0 = normalize(a);
          let v1 = normalize(b);
          let v2 = normalize(c);
          let avg_z_norm = (z_norm(a.2) + z_norm(b.2) + z_norm(c.2)) / 3.0;
          let default_color = facet_colors[surface_idx][i][j]
            .or_else(|| mesh_facet_color(a, b, c))
            .unwrap_or_else(|| height_color(avg_z_norm));
          let normal = triangle_normal(v0, v1, v2);
          let (color, opacity) =
            shade_facet(default_color, style, normal, view_dir);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };
          all_triangles.push(Triangle {
            boundary: [true; 3],
            edge_color: None,
            edge_width: None,
            projected: [
              project(v0, &camera),
              project(v1, &camera),
              project(v2, &camera),
            ],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
          });
        }

        // Triangle 2
        if let (Some(a), Some(b), Some(c)) = (p11, p01, p10) {
          let v0 = normalize(a);
          let v1 = normalize(b);
          let v2 = normalize(c);
          let avg_z_norm = (z_norm(a.2) + z_norm(b.2) + z_norm(c.2)) / 3.0;
          let default_color = facet_colors[surface_idx][i][j]
            .or_else(|| mesh_facet_color(a, b, c))
            .unwrap_or_else(|| height_color(avg_z_norm));
          let normal = triangle_normal(v0, v1, v2);
          let (color, opacity) =
            shade_facet(default_color, style, normal, view_dir);
          let center = Point3D {
            x: (v0.x + v1.x + v2.x) / 3.0,
            y: (v0.y + v1.y + v2.y) / 3.0,
            z: (v0.z + v1.z + v2.z) / 3.0,
          };
          all_triangles.push(Triangle {
            boundary: [true; 3],
            edge_color: None,
            edge_width: None,
            projected: [
              project(v0, &camera),
              project(v1, &camera),
              project(v2, &camera),
            ],
            depth: depth(center, &camera),
            color,
            opacity,
            texture: None,
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
    mesh_mode,
    show_axes,
    box_ratios,
    &[None, None, None],
  )?;
  // A `PlotLabel` sets a title above the finished picture.
  let svg = with_plot_label(svg, args, svg_width, svg_height);

  Ok(crate::plot3d_result_with_structure(svg, structure))
}

#[cfg(test)]
mod area_geometry_debug_tests {
  use super::*;

  #[test]
  fn disk_fully_inside_square() {
    let square = vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let area = disk_convex_polygon_area([0.0, 0.0], 0.5, &square);
    assert!(
      (area - std::f64::consts::PI * 0.25).abs() < 1e-9,
      "got {area}"
    );
  }

  #[test]
  fn disk_tangent_to_square_edges() {
    // half-height 0.5 == radius: circle touches top/bottom, full circle
    // still counts.
    let rect = vec![[-2.0, -0.5], [2.0, -0.5], [2.0, 0.5], [-2.0, 0.5]];
    let area = disk_convex_polygon_area([0.0, 0.0], 0.5, &rect);
    assert!(
      (area - std::f64::consts::PI * 0.25).abs() < 1e-9,
      "got {area}"
    );
  }

  #[test]
  fn polygon3d_area_unit_square_in_xy() {
    let poly = vec![
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [1.0, 1.0, 0.0],
      [0.0, 1.0, 0.0],
    ];
    assert!((polygon3d_area(&poly) - 1.0).abs() < 1e-9);
  }

  #[test]
  fn plane_polygon_in_diagonal_cube_cross_section() {
    let normal = [1.0, 1.0, 0.0];
    let point = [1.0, 0.0, 0.0];
    let bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let poly = plane_polygon_in_box(normal, point, bounds).expect("poly");
    let area = polygon3d_area(&poly);
    assert!(
      (area - 2.0f64.sqrt()).abs() < 1e-9,
      "got {area}, poly={poly:?}"
    );
  }
}

#[cfg(test)]
mod cuboid_rotation_tests {
  use super::*;

  fn corner(p: Point3D) -> (f64, f64, f64) {
    (
      (p.x * 1e9).round() / 1e9,
      (p.y * 1e9).round() / 1e9,
      (p.z * 1e9).round() / 1e9,
    )
  }

  #[test]
  fn axis_aligned_transform_detects_translation_and_scaling_only() {
    assert!(Affine3::translation([1.0, -2.0, 3.0]).is_axis_aligned());
    assert!(
      Affine3::scaling([2.0, 0.5, -1.0], [0.0, 0.0, 0.0]).is_axis_aligned()
    );
    // A 90-degree rotation about a coordinate axis happens to keep the
    // linear part diagonal-free of off-axis coupling along that axis, but
    // still mixes the other two — off-axis entries appear, so it is
    // correctly *not* axis-aligned.
    assert!(
      !Affine3::rotation(
        std::f64::consts::FRAC_PI_2,
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
      )
      .unwrap()
      .is_axis_aligned()
    );
    assert!(
      !Affine3::rotation(0.3, [0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        .unwrap()
        .is_axis_aligned()
    );
  }

  /// A `Cuboid` under a genuine rotation cannot stay a `Cuboid` — its
  /// two-corner form only ever describes an axis-aligned box. Regression:
  /// `transform_primitive3d` used to rotate just the two corners and
  /// re-derive a new axis-aligned box from them ("only approximated" per
  /// its old comment), which for a thin box at a generic angle erases the
  /// tilt entirely rather than producing a tilted box. It must instead
  /// tessellate into a `Surface3D`, the same way `Sphere`/`Cylinder`/`Cone`
  /// already do for a transform their analytic form cannot express.
  #[test]
  fn rotated_cuboid_tessellates_instead_of_approximating() {
    let mut prim = Primitive3D::Cuboid {
      p_min: Point3D {
        x: -0.05,
        y: -0.05,
        z: 0.0,
      },
      p_max: Point3D {
        x: 0.05,
        y: 0.05,
        z: 2.0,
      },
      style: StyleState3D::default(),
    };
    let xf = Affine3::rotation(0.7, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    transform_primitive3d(&mut prim, &xf);
    assert!(
      matches!(prim, Primitive3D::Surface3D { .. }),
      "a rotated Cuboid must tessellate, not stay an (approximated) Cuboid"
    );
    // The tessellated box's own vertices, rotated back by the inverse
    // angle, must land exactly on the original axis-aligned corners —
    // exact because both rotations act on the same fixed set of vertices,
    // unlike the old corner-remap-then-re-normalize approach, whose second
    // application starts from an already wrong, previously re-normalized
    // box and does not undo the first.
    let inverse =
      Affine3::rotation(-0.7, [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]).unwrap();
    let Primitive3D::Surface3D { tris, .. } = &prim else {
      unreachable!()
    };
    let mut restored: Vec<(f64, f64, f64)> = tris
      .iter()
      .flat_map(|(a, b, c)| [a, b, c])
      .map(|p| corner(inverse.apply(*p)))
      .collect();
    restored.sort_by(|a, b| a.partial_cmp(b).unwrap());
    restored.dedup();
    let mut expected = vec![
      (-0.05, -0.05, 0.0),
      (0.05, -0.05, 0.0),
      (0.05, 0.05, 0.0),
      (-0.05, 0.05, 0.0),
      (-0.05, -0.05, 2.0),
      (0.05, -0.05, 2.0),
      (0.05, 0.05, 2.0),
      (-0.05, 0.05, 2.0),
    ];
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(restored, expected);
  }

  /// The same rotation, but through an off-origin anchor point — exercising
  /// the `Rotate[g, angle, axis, point]` 4-argument form (the one the
  /// ladder-climber–style Demonstrations use to swing a rung assembly about
  /// a pivot partway up a wall) rather than the 3-argument origin-anchored
  /// one above.
  #[test]
  fn rotated_cuboid_about_an_off_origin_axis_tessellates() {
    let mut prim = Primitive3D::Cuboid {
      p_min: Point3D {
        x: -0.1,
        y: 0.0,
        z: -1.0,
      },
      p_max: Point3D {
        x: 0.0,
        y: 0.03,
        z: 1.0,
      },
      style: StyleState3D::default(),
    };
    let xf = Affine3::rotation(1.1, [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]).unwrap();
    transform_primitive3d(&mut prim, &xf);
    assert!(matches!(prim, Primitive3D::Surface3D { .. }));
    let inverse =
      Affine3::rotation(-1.1, [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]).unwrap();
    let Primitive3D::Surface3D { tris, .. } = &prim else {
      unreachable!()
    };
    let mut restored: Vec<(f64, f64, f64)> = tris
      .iter()
      .flat_map(|(a, b, c)| [a, b, c])
      .map(|p| corner(inverse.apply(*p)))
      .collect();
    restored.sort_by(|a, b| a.partial_cmp(b).unwrap());
    restored.dedup();
    let mut expected = vec![
      (-0.1, 0.0, -1.0),
      (0.0, 0.0, -1.0),
      (0.0, 0.03, -1.0),
      (-0.1, 0.03, -1.0),
      (-0.1, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (0.0, 0.03, 1.0),
      (-0.1, 0.03, 1.0),
    ];
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(restored, expected);
  }

  /// A `Cuboid` under a transform that keeps it axis-aligned (translation
  /// composed with a per-axis scale) stays the cheap `Cuboid` form rather
  /// than tessellating unnecessarily.
  #[test]
  fn axis_aligned_transform_keeps_cuboid_representation() {
    let mut prim = Primitive3D::Cuboid {
      p_min: Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
      },
      p_max: Point3D {
        x: 1.0,
        y: 1.0,
        z: 1.0,
      },
      style: StyleState3D::default(),
    };
    let xf = Affine3::translation([2.0, -1.0, 0.5]);
    transform_primitive3d(&mut prim, &xf);
    match prim {
      Primitive3D::Cuboid { p_min, p_max, .. } => {
        assert_eq!(corner(p_min), (2.0, -1.0, 0.5));
        assert_eq!(corner(p_max), (3.0, 0.0, 1.5));
      }
      _ => panic!("an axis-aligned transform must not tessellate a Cuboid"),
    }
  }
}

#[cfg(test)]
mod march_cube_orientation_tests {
  use super::*;

  /// Every triangle of a level surface faces toward the side where the
  /// field exceeds the level — whichever of the cube's six tetrahedra
  /// produced it — so `FaceForm[front, back]` colours it consistently.
  #[test]
  fn triangles_face_toward_higher_field_values() {
    let n = 6;
    let h = 2.0 / n as f64;
    // A tilted plane: its gradient (1, 2, 3) is the same everywhere.
    let f = |x: f64, y: f64, z: f64| x + 2.0 * y + 3.0 * z - 0.1;
    let mut checked = 0;
    for i in 0..n {
      for j in 0..n {
        for k in 0..n {
          let at = |di: usize, dj: usize, dk: usize| Point3D {
            x: -1.0 + (i + di) as f64 * h,
            y: -1.0 + (j + dj) as f64 * h,
            z: -1.0 + (k + dk) as f64 * h,
          };
          let corners: [Point3D; 8] =
            std::array::from_fn(|c| at(c & 1, (c >> 1) & 1, (c >> 2) & 1));
          let vals: [f64; 8] = std::array::from_fn(|c| {
            f(corners[c].x, corners[c].y, corners[c].z)
          });
          march_cube(&corners, &vals, 0.0, &mut |t| {
            let nrm = triangle_normal(t[0], t[1], t[2]);
            let dot = nrm[0] + 2.0 * nrm[1] + 3.0 * nrm[2];
            assert!(dot > 0.0, "triangle faces the low side");
            checked += 1;
          });
        }
      }
    }
    assert!(checked > 0);
  }
}

#[cfg(test)]
mod painter_order_tests {
  use super::*;

  fn flat_triangle(verts: [Point3D; 3], camera: &Camera) -> Triangle {
    Triangle {
      projected: std::array::from_fn(|i| project(verts[i], camera)),
      depth: 0.0,
      color: (0, 0, 0),
      opacity: 1.0,
      boundary: [true; 3],
      edge_color: None,
      edge_width: None,
      texture: None,
    }
  }

  /// Two triangles that pierce each other cannot be ordered as wholes, so
  /// they are cut along the crossing; afterwards every overlapping pair of
  /// pieces lies wholly on one side of the other.
  #[test]
  fn crossing_triangles_are_split_into_orderable_pieces() {
    let camera = Camera::default();
    let p = |x: f64, y: f64, z: f64| Point3D { x, y, z };
    let mut verts = vec![
      [p(-1.0, -1.0, 0.0), p(1.0, -1.0, 0.0), p(0.0, 1.0, 0.0)],
      [p(-1.0, 0.0, -1.0), p(1.0, 0.0, -1.0), p(0.0, 0.0, 1.0)],
    ];
    let mut tris: Vec<Triangle> =
      verts.iter().map(|v| flat_triangle(*v, &camera)).collect();
    split_interpenetrating(&mut verts, &mut tris, &camera);
    assert!(verts.len() > 2);
    assert_eq!(verts.len(), tris.len());
    let gradient = depth_gradient(&camera);
    let projected: Vec<_> = tris.iter().map(|t| t.projected).collect();
    for_each_overlapping_pair(&projected, |i, j| {
      assert!(
        side_of_plane(&verts[i], &verts[j], gradient).is_some()
          || side_of_plane(&verts[j], &verts[i], gradient).is_some()
      );
    });
  }

  /// Triangles that never cross are left untouched.
  #[test]
  fn disjoint_triangles_are_not_split() {
    let camera = Camera::default();
    let p = |x: f64, y: f64, z: f64| Point3D { x, y, z };
    let mut verts = vec![
      [p(-1.0, -1.0, 0.0), p(1.0, -1.0, 0.0), p(0.0, 1.0, 0.0)],
      [p(-1.0, -1.0, 1.0), p(1.0, -1.0, 1.0), p(0.0, 1.0, 1.0)],
    ];
    let mut tris: Vec<Triangle> =
      verts.iter().map(|v| flat_triangle(*v, &camera)).collect();
    split_interpenetrating(&mut verts, &mut tris, &camera);
    assert_eq!(verts.len(), 2);
  }

  /// A large face below a dense field of small faces (more than the old
  /// 2500-triangle cap) must still be drawn first, even though the small
  /// faces' centroids lie farther from the viewer than its own.
  #[test]
  fn large_face_stays_behind_many_small_faces() {
    let camera = Camera::default();
    let p = |x: f64, y: f64, z: f64| Point3D { x, y, z };
    let mut verts =
      vec![[p(-10.0, -10.0, 0.0), p(10.0, -10.0, 0.0), p(0.0, 10.0, 0.0)]];
    let cells = 40;
    let step = 6.0 / cells as f64;
    for i in 0..cells {
      for j in 0..cells {
        let (x, y) = (-3.0 + i as f64 * step, -3.0 + j as f64 * step);
        verts.push([p(x, y, 0.1), p(x + step, y, 0.1), p(x, y + step, 0.1)]);
        verts.push([
          p(x + step, y, 0.1),
          p(x + step, y + step, 0.1),
          p(x, y + step, 0.1),
        ]);
      }
    }
    assert!(verts.len() > 2500);
    let projected: Vec<[(f64, f64); 3]> = verts
      .iter()
      .map(|t| std::array::from_fn(|i| project(t[i], &camera)))
      .collect();
    let depths: Vec<f64> = verts
      .iter()
      .map(|t| {
        depth(
          p(
            (t[0].x + t[1].x + t[2].x) / 3.0,
            (t[0].y + t[1].y + t[2].y) / 3.0,
            (t[0].z + t[1].z + t[2].z) / 3.0,
          ),
          &camera,
        )
      })
      .collect();
    assert!(
      depths[1..].iter().any(|d| *d > depths[0]),
      "scenario must defeat a plain centroid sort"
    );
    let order = painter_order(&verts, &projected, &depths, &camera);
    assert_eq!(order.len(), verts.len());
    assert_eq!(order[0], 0, "the large back face is drawn first");
  }
}

#[cfg(test)]
mod tick_values_tests {
  use super::*;

  #[test]
  fn float_noise_extent_terminates() {
    let ticks = tick_values(1.9999999999999998, 2.0);
    assert!(ticks.len() <= 2);
  }

  #[test]
  fn regular_extent_has_ticks() {
    assert_eq!(tick_values(0.0, 10.0), vec![0.0, 2.5, 5.0, 7.5, 10.0]);
  }
}

/// `AbsoluteOptions[g, PlotRange]` of a 3D graphic: an explicit
/// `PlotRange -> {{x0, x1}, {y0, y1}, {z0, z1}}` as given, otherwise the
/// bounding box of its primitives.
pub fn absolute_plot_range_3d(g: &Expr) -> Option<[(f64, f64); 3]> {
  let (content, options): (Expr, Vec<Expr>) = match g {
    Expr::Graphics {
      is_3d: true,
      structure: Some(structure),
      ..
    } => return absolute_plot_range_3d(structure),
    Expr::FunctionCall { name, args }
      if name == "Graphics3D" && !args.is_empty() =>
    {
      (args[0].clone(), args[1..].to_vec())
    }
    _ => return None,
  };
  for opt in &options {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && matches!(pattern.as_ref(), Expr::Identifier(n) if n == "PlotRange")
      && let Some(ranges) = parse_axis_ranges(replacement)
    {
      return Some(ranges);
    }
  }
  let content = evaluate_expr_to_expr(&content).ok()?;
  let mut prims = Vec::new();
  let mut style3d = StyleState3D::default();
  collect_3d_primitives(&content, &mut style3d, &mut prims);
  let bounds = primitives_bounds(&prims);
  bounds
    .iter()
    .all(|(lo, hi)| lo.is_finite() && hi.is_finite())
    .then_some(bounds)
}

/// `ListSurfacePlot3D[points, opts]`: a surface through 3D points, drawn as
/// a `Graphics3D` of triangles.
///
/// An array of `{x, y, z}` rows (`Table[{x, y, z}, {u, …}, {v, …}]`) is
/// joined cell by cell; a plain list of points is wrapped in its convex
/// hull. `ColorFunction -> Function[{x, y, z}, …]` colours every triangle
/// at its centroid, with the coordinates rescaled to 0..1 over the point
/// set unless `ColorFunctionScaling -> False`. The remaining options go to
/// the `Graphics3D`.
pub fn list_surface_plot3d_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let data = evaluate_expr_to_expr(&args[0])?;
  let Expr::List(outer) = &data else {
    return Ok(Expr::FunctionCall {
      name: "ListSurfacePlot3D".to_string(),
      args: args.to_vec().into(),
    });
  };
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "ListSurfacePlot3D".to_string(),
      args: args.to_vec().into(),
    })
  };

  // Rows of points when every entry is a list of {x, y, z} points.
  let mut rows: Vec<Vec<[f64; 3]>> = Vec::new();
  let mut flat: Vec<[f64; 3]> = Vec::new();
  let is_array = !outer.is_empty()
    && outer.iter().all(|r| matches!(r, Expr::List(items)
      if !items.is_empty() && items.iter().all(|p| matches!(p, Expr::List(c) if c.len() == 3))));
  if is_array {
    for r in outer {
      let Expr::List(items) = r else { continue };
      let mut row = Vec::new();
      for p in items {
        match eval_vec3(p) {
          Some(v) if v.iter().all(|c| c.is_finite()) => row.push(v),
          _ => return unevaluated(),
        }
      }
      rows.push(row);
    }
    if rows.iter().any(|r| r.len() != rows[0].len()) {
      return unevaluated();
    }
    flat = rows.iter().flatten().copied().collect();
  } else {
    for p in outer {
      match eval_vec3(p) {
        Some(v) if v.iter().all(|c| c.is_finite()) => flat.push(v),
        _ => return unevaluated(),
      }
    }
  }
  if flat.len() < 3 {
    return unevaluated();
  }

  let mut triangles: Vec<Vec<[f64; 3]>> = Vec::new();
  if is_array {
    for i in 0..rows.len().saturating_sub(1) {
      for j in 0..rows[i].len().saturating_sub(1) {
        let (a, b, c, d) = (
          rows[i][j],
          rows[i][j + 1],
          rows[i + 1][j + 1],
          rows[i + 1][j],
        );
        triangles.push(vec![a, b, c]);
        triangles.push(vec![a, c, d]);
      }
    }
  } else {
    let tuples: Vec<(f64, f64, f64)> =
      flat.iter().map(|p| (p[0], p[1], p[2])).collect();
    let Some(faces) =
      crate::functions::convex_hull::convex_hull_polygons(&tuples)
    else {
      return unevaluated();
    };
    for face in faces {
      for k in 1..face.len().saturating_sub(1) {
        triangles.push(vec![flat[face[0]], flat[face[k]], flat[face[k + 1]]]);
      }
    }
  }
  if triangles.is_empty() {
    return unevaluated();
  }

  let mut facet_color = FacetColorFunction::default();
  const CONSUMED: &[&str] = &[
    "ColorFunction",
    "ColorFunctionScaling",
    "BoundaryStyle",
    "Mesh",
    "MaxPlotPoints",
    "PlotPoints",
  ];
  let mut graphics_opts: Vec<Expr> = Vec::new();
  for opt in &args[1..] {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(n) = pattern.as_ref()
      && CONSUMED.contains(&n.as_str())
    {
      facet_color.read_option(pattern, replacement);
    } else {
      graphics_opts.push(opt.clone());
    }
  }

  let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 3];
  for p in &flat {
    for k in 0..3 {
      ranges[k].0 = ranges[k].0.min(p[k]);
      ranges[k].1 = ranges[k].1.max(p[k]);
    }
  }

  let polygon = |tri: &[[f64; 3]]| Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(
      tri
        .iter()
        .map(|p| {
          Expr::List(
            p.iter().map(|c| Expr::Real(*c)).collect::<Vec<_>>().into(),
          )
        })
        .collect::<Vec<_>>()
        .into(),
    )]
    .into(),
  };
  let mut primitives = vec![Expr::FunctionCall {
    name: "EdgeForm".to_string(),
    args: vec![Expr::Identifier("None".to_string())].into(),
  }];
  for tri in &triangles {
    let n = tri.len() as f64;
    let centroid = [0, 1, 2].map(|k| tri.iter().map(|p| p[k]).sum::<f64>() / n);
    let poly = polygon(tri);
    match facet_color.function.as_ref().and_then(|f| {
      let args = (0..3)
        .map(|k| {
          let (lo, hi) = ranges[k];
          Expr::Real(if facet_color.unscaled || (hi - lo).abs() < 1e-15 {
            centroid[k]
          } else {
            (centroid[k] - lo) / (hi - lo)
          })
        })
        .collect();
      evaluate_expr_to_expr(&Expr::CurriedCall {
        func: Box::new(f.clone()),
        args,
      })
      .ok()
    }) {
      Some(color) => primitives.push(Expr::List(vec![color, poly].into())),
      None => primitives.push(poly),
    }
  }

  let mut g_args = vec![Expr::List(primitives.into())];
  g_args.extend(graphics_opts);
  evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: g_args.into(),
  })
}
