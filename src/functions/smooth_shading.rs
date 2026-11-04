//! Smooth (Gouraud) shading for the SVG 3D renderers.
//!
//! A curved surface is lit at its vertices, each with a normal averaged
//! from the faces around it (or the exact one of an analytic primitive),
//! and every triangle is filled with an SVG `linearGradient` that
//! reproduces the colours interpolated between its corners.
//!
//! A linear gradient varies along one screen direction only, so it is
//! exact when the colour over the triangle changes along one direction —
//! all three channels together. Under coloured lights the red, green and
//! blue gradients point different ways; the fill then uses the direction
//! and slopes that minimise the squared colour error over the triangle's
//! area (the best rank-1 approximation of the colour's screen gradient,
//! weighted by the triangle's second moment).

use crate::functions::plot3d::Point3D;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

type Vec3 = [f64; 3];

fn sub(a: Point3D, b: Point3D) -> Vec3 {
  [a.x - b.x, a.y - b.y, a.z - b.z]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: Vec3) -> Option<Vec3> {
  let l = dot(a, a).sqrt();
  (l > 1e-300).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

/// Cosine of the largest angle between two faces meeting at a vertex that
/// still counts as one smooth surface; a sharper fold (a cylinder's cap
/// against its side) keeps a crease.
pub(crate) const CREASE_COS: f64 = 0.5;

/// The normal at each corner of each triangle: the area-weighted average
/// of the normals of the triangles sharing that corner's position whose
/// planes are within the crease angle of this one. Triangles sharing a
/// corner may be wound either way; their normals are aligned before
/// averaging. Each result points to the same side as its own face.
pub(crate) fn vertex_normals(tris: &[[Point3D; 3]]) -> Vec<[Vec3; 3]> {
  let face: Vec<Vec3> = tris
    .iter()
    .map(|t| cross(sub(t[1], t[0]), sub(t[2], t[0])))
    .collect();
  let unit: Vec<Option<Vec3>> = face.iter().map(|&n| normalize(n)).collect();
  // Weld corners by position, to a tolerance relative to the extent.
  let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
  for t in tris {
    for p in t {
      for (k, c) in [p.x, p.y, p.z].into_iter().enumerate() {
        lo[k] = lo[k].min(c);
        hi[k] = hi[k].max(c);
      }
    }
  }
  let extent = (0..3)
    .map(|k| hi[k] - lo[k])
    .fold(0.0f64, f64::max)
    .max(1e-300);
  let q = 1e-7 * extent;
  let key = |p: Point3D| {
    (
      (p.x / q).round() as i64,
      (p.y / q).round() as i64,
      (p.z / q).round() as i64,
    )
  };
  let mut at: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
  for (i, t) in tris.iter().enumerate() {
    if unit[i].is_none() {
      continue;
    }
    for &p in t {
      let list = at.entry(key(p)).or_default();
      if list.last() != Some(&i) {
        list.push(i);
      }
    }
  }
  tris
    .iter()
    .enumerate()
    .map(|(i, t)| {
      let Some(own) = unit[i] else {
        return [[0.0, 0.0, 1.0]; 3];
      };
      t.map(|p| {
        let mut sum = [0.0; 3];
        for &j in at.get(&key(p)).map_or(&[][..], Vec::as_slice) {
          let Some(other) = unit[j] else { continue };
          let c = dot(own, other);
          if c.abs() < CREASE_COS {
            continue;
          }
          let s = if c < 0.0 { -1.0 } else { 1.0 };
          for k in 0..3 {
            sum[k] += s * face[j][k];
          }
        }
        normalize(sum).unwrap_or(own)
      })
    })
    .collect()
}

/// Largest mesh `closed_convex_interior` examines (its test is quadratic).
const CONVEX_TEST_MAX_TRIANGLES: usize = 4000;

/// A point inside the mesh when it is a closed convex surface (every edge
/// shared by exactly two triangles, every vertex on the inner side of
/// every face plane) — an ellipsoid, a capsule — whose far side is then
/// hidden. `None` for an open or non-convex surface.
pub(crate) fn closed_convex_interior(tris: &[[Point3D; 3]]) -> Option<Point3D> {
  if tris.len() < 4 || tris.len() > CONVEX_TEST_MAX_TRIANGLES {
    return None;
  }
  let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
  let mut sum = [0.0; 3];
  for t in tris {
    for p in t {
      for (k, c) in [p.x, p.y, p.z].into_iter().enumerate() {
        lo[k] = lo[k].min(c);
        hi[k] = hi[k].max(c);
        sum[k] += c;
      }
    }
  }
  let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f64, f64::max);
  if extent <= 0.0 {
    return None;
  }
  let q = 1e-7 * extent;
  let key = |p: Point3D| {
    (
      (p.x / q).round() as i64,
      (p.y / q).round() as i64,
      (p.z / q).round() as i64,
    )
  };
  let mut edges: HashMap<((i64, i64, i64), (i64, i64, i64)), u32> =
    HashMap::new();
  for t in tris {
    for e in 0..3 {
      let (a, b) = (key(t[e]), key(t[(e + 1) % 3]));
      if a == b {
        continue;
      }
      *edges
        .entry(if a < b { (a, b) } else { (b, a) })
        .or_default() += 1;
    }
  }
  if edges.values().any(|&n| n != 2) {
    return None;
  }
  let n = (tris.len() * 3) as f64;
  let c = Point3D {
    x: sum[0] / n,
    y: sum[1] / n,
    z: sum[2] / n,
  };
  let eps = 1e-6 * extent;
  for t in tris {
    let Some(normal) = normalize(cross(sub(t[1], t[0]), sub(t[2], t[0])))
    else {
      continue;
    };
    let inner = dot(normal, sub(c, t[0]));
    if inner.abs() <= eps {
      return None;
    }
    let s = inner.signum();
    for u in tris {
      for &p in u {
        if dot(normal, sub(p, t[0])) * s < -eps {
          return None;
        }
      }
    }
  }
  Some(c)
}

/// Bends a vertex normal so the vertex faces the viewer (along `toward`)
/// the same way as its triangle, whose normal is `face`. Near a silhouette
/// an averaged normal can tip past the edge of view; lit as-is it would
/// be flipped to the other side and light the rim from behind.
pub(crate) fn orient_vertex_normal(
  normal: Vec3,
  face: Vec3,
  toward: Vec3,
) -> Vec3 {
  let mut n = normal;
  if dot(n, face) < 0.0 {
    n = [-n[0], -n[1], -n[2]];
  }
  let side = if dot(face, toward) < 0.0 { -1.0 } else { 1.0 };
  let along = dot(n, toward) * side;
  let min = 1e-3;
  if along < min {
    let fix = (min - along) * side;
    n = [
      n[0] + fix * toward[0],
      n[1] + fix * toward[1],
      n[2] + fix * toward[2],
    ];
  }
  normalize(n).unwrap_or(face)
}

/// The exact normal of an analytic surface at a corner: called with the
/// triangle's index, the corner, the triangle's unit normal and its
/// centroid (which settles a point where the normal is not unique — a
/// cone's apex, a cylinder's rim).
pub(crate) type ExactNormalFn = dyn Fn(usize, Point3D, Vec3, Point3D) -> Vec3;
pub(crate) type ExactNormal<'a> =
  &'a (dyn Fn(usize, Point3D, Vec3, Point3D) -> Vec3 + 'a);

/// The lit colour at each corner of the triangles of one smooth surface.
/// `exact` gives the surface's true normals (an analytic primitive);
/// otherwise normals are averaged from the faces. `shade(i, normal,
/// point)` lights corner `point` of triangle `i`; `toward` points at the
/// viewer.
pub(crate) fn corner_colors(
  tris: &[[Point3D; 3]],
  exact: Option<ExactNormal>,
  toward: Vec3,
  mut shade: impl FnMut(usize, Vec3, Point3D) -> (u8, u8, u8),
) -> Vec<[(u8, u8, u8); 3]> {
  let faces: Vec<Vec3> = tris
    .iter()
    .map(|t| {
      normalize(cross(sub(t[1], t[0]), sub(t[2], t[0]))).unwrap_or(toward)
    })
    .collect();
  let normals: Vec<[Vec3; 3]> = match exact {
    Some(f) => tris
      .iter()
      .zip(&faces)
      .enumerate()
      .map(|(i, (t, &face))| {
        let centroid = Point3D {
          x: (t[0].x + t[1].x + t[2].x) / 3.0,
          y: (t[0].y + t[1].y + t[2].y) / 3.0,
          z: (t[0].z + t[1].z + t[2].z) / 3.0,
        };
        t.map(|p| normalize(f(i, p, face, centroid)).unwrap_or(face))
      })
      .collect(),
    None => vertex_normals(tris),
  };
  tris
    .iter()
    .zip(normals)
    .zip(faces)
    .enumerate()
    .map(|(i, ((t, n), face))| {
      [0, 1, 2]
        .map(|c| shade(i, orient_vertex_normal(n[c], face, toward), t[c]))
    })
    .collect()
}

/// Exact normals of a cylinder (`cone == false`) or cone from `base` to
/// `tip` with base radius `radius`: radial on the side (tilted by the
/// cone's slope), the face's own on the flat end caps.
pub(crate) fn round_solid_normal(
  base: Point3D,
  tip: Point3D,
  radius: f64,
  cone: bool,
) -> impl Fn(usize, Point3D, Vec3, Point3D) -> Vec3 {
  let axis = sub(tip, base);
  let len = dot(axis, axis).sqrt();
  let a = normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
  move |_, p, face, centroid| {
    // An end cap is flat.
    if dot(face, a).abs() > 0.999 {
      return face;
    }
    let radial = |q: Point3D| {
      let d = sub(q, base);
      let along = dot(d, a);
      normalize([
        d[0] - along * a[0],
        d[1] - along * a[1],
        d[2] - along * a[2],
      ])
    };
    // The apex lies on the axis: it takes the direction of its face.
    let r = if cone && dot(sub(p, tip), sub(p, tip)) < 1e-18 * len * len {
      radial(centroid)
    } else {
      radial(p).or_else(|| radial(centroid))
    };
    let Some(r) = r else {
      return face;
    };
    if cone && len > 0.0 {
      // The side leans in by atan(radius / height).
      normalize([
        len * r[0] + radius * a[0],
        len * r[1] + radius * a[1],
        len * r[2] + radius * a[2],
      ])
      .unwrap_or(face)
    } else {
      r
    }
  }
}

/// A linear gradient across a triangle: the colours at `from` and `to`,
/// and the rms error of the fit (in 0–255 units, over the triangle's
/// area, averaged over the channels).
pub(crate) struct GradientFit {
  pub from: (f64, f64),
  pub to: (f64, f64),
  pub from_color: [f64; 3],
  pub to_color: [f64; 3],
  // Read by the error measurements in the tests below.
  #[cfg_attr(not(test), allow(dead_code))]
  pub rms_error: f64,
}

/// The best linear gradient for a triangle with screen corners `pts` and
/// corner colours `colors`; `None` when the colour hardly changes over the
/// triangle (a flat fill does) or the triangle is degenerate.
pub(crate) fn fit_gradient(
  pts: [(f64, f64); 3],
  colors: [[f64; 3]; 3],
) -> Option<GradientFit> {
  let (e1, e2) = (
    (pts[1].0 - pts[0].0, pts[1].1 - pts[0].1),
    (pts[2].0 - pts[0].0, pts[2].1 - pts[0].1),
  );
  let det = e1.0 * e2.1 - e2.0 * e1.1;
  let span = (e1.0.abs() + e1.1.abs() + e2.0.abs() + e2.1.abs()).max(1e-300);
  if det.abs() < 1e-9 * span * span {
    return None;
  }
  // The colour's screen gradient A (one row per channel): A·e = Δc.
  // E⁻¹ for E = [e1 e2] (columns).
  let inv = [[e2.1 / det, -e2.0 / det], [-e1.1 / det, e1.0 / det]];
  let mut a = [[0.0; 2]; 3];
  for k in 0..3 {
    let d1 = colors[1][k] - colors[0][k];
    let d2 = colors[2][k] - colors[0][k];
    a[k] = [
      d1 * inv[0][0] + d2 * inv[1][0],
      d1 * inv[0][1] + d2 * inv[1][1],
    ];
  }
  let centre = (
    (pts[0].0 + pts[1].0 + pts[2].0) / 3.0,
    (pts[0].1 + pts[1].1 + pts[2].1) / 3.0,
  );
  // Second moment of the triangle about its centroid, M = L·Lᵀ.
  let mut m = [0.0; 3]; // m11, m12, m22
  for p in pts {
    let (dx, dy) = (p.0 - centre.0, p.1 - centre.1);
    m[0] += dx * dx / 12.0;
    m[1] += dx * dy / 12.0;
    m[2] += dy * dy / 12.0;
  }
  let l11 = m[0].sqrt();
  if l11 <= 0.0 {
    return None;
  }
  let l21 = m[1] / l11;
  let l22 = (m[2] - l21 * l21).max(0.0).sqrt();
  if l22 <= 0.0 {
    return None;
  }
  // B = A·L; its best rank-1 part is B·w·wᵀ for the top right singular
  // vector w (top eigenvector of BᵀB).
  let b: Vec<[f64; 2]> = a
    .iter()
    .map(|r| [r[0] * l11 + r[1] * l21, r[1] * l22])
    .collect();
  let (mut s11, mut s12, mut s22) = (0.0, 0.0, 0.0);
  for r in &b {
    s11 += r[0] * r[0];
    s12 += r[0] * r[1];
    s22 += r[1] * r[1];
  }
  let total = s11 + s22;
  // The colour change over the triangle (rms, 0–255 units).
  if total.sqrt() * 255.0 < 0.35 {
    return None;
  }
  let half_tr = 0.5 * total;
  let disc = (0.25 * (s11 - s22).powi(2) + s12 * s12).sqrt();
  let lambda = half_tr + disc;
  let w = if s12.abs() > 1e-300 {
    let v = [lambda - s22, s12];
    let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
    [v[0] / l, v[1] / l]
  } else if s11 >= s22 {
    [1.0, 0.0]
  } else {
    [0.0, 1.0]
  };
  // Residual energy: the smaller eigenvalue, per channel.
  let rms_error = ((half_tr - disc).max(0.0) / 3.0).sqrt() * 255.0;
  // Slopes along w: u = B·w; the direction in screen space v = L⁻ᵀ·w.
  let u: Vec<f64> = b.iter().map(|r| r[0] * w[0] + r[1] * w[1]).collect();
  let v2 = w[1] / l22;
  let v1 = (w[0] - l21 * v2) / l11;
  let s: Vec<f64> = pts
    .iter()
    .map(|p| v1 * (p.0 - centre.0) + v2 * (p.1 - centre.1))
    .collect();
  let smin = s.iter().copied().fold(f64::INFINITY, f64::min);
  let smax = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  let vv = v1 * v1 + v2 * v2;
  if vv <= 0.0 || smax - smin <= 0.0 {
    return None;
  }
  let mean: Vec<f64> = (0..3)
    .map(|k| (colors[0][k] + colors[1][k] + colors[2][k]) / 3.0)
    .collect();
  let at = |t: f64| (centre.0 + v1 * t / vv, centre.1 + v2 * t / vv);
  let color_at =
    |t: f64| [0, 1, 2].map(|k| (mean[k] + u[k] * t).clamp(0.0, 1.0));
  Some(GradientFit {
    from: at(smin),
    to: at(smax),
    from_color: color_at(smin),
    to_color: color_at(smax),
    rms_error,
  })
}

/// The largest per-channel difference between corner colours that is still
/// filled flat.
const FLAT_SPREAD: u8 = 2;

/// The gradients one SVG document defines, so a gradient used twice is
/// written once.
#[derive(Default)]
pub(crate) struct GradientDefs {
  seen: HashSet<u64>,
}

fn hex(c: [f64; 3]) -> String {
  let q = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u8;
  format!("#{:02x}{:02x}{:02x}", q(c[0]), q(c[1]), q(c[2]))
}

impl GradientDefs {
  /// The SVG paint for a triangle with screen corners `pts` and corner
  /// colours `colors`: `url(#…)` of a linear gradient (written to `out`
  /// the first time it is needed), or `None` when a flat fill in the
  /// average colour is as good.
  pub(crate) fn paint(
    &mut self,
    out: &mut String,
    pts: [(f64, f64); 3],
    colors: [(u8, u8, u8); 3],
  ) -> Option<String> {
    // Corners within a couple of levels of each other: a flat fill in
    // their mean is off by at most one level, which no one sees, and
    // saves the gradient.
    let spread = |k: fn(&(u8, u8, u8)) -> u8| {
      let v = colors.map(|c| k(&c));
      v.iter().max().unwrap() - v.iter().min().unwrap()
    };
    if spread(|c| c.0).max(spread(|c| c.1)).max(spread(|c| c.2)) <= FLAT_SPREAD
    {
      return None;
    }
    let c = colors
      .map(|(r, g, b)| [r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0]);
    let fit = fit_gradient(pts, c)?;
    let (c1, c2) = (hex(fit.from_color), hex(fit.to_color));
    if c1 == c2 {
      return None;
    }
    let body = format!(
      "x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"><stop stop-color=\"{c1}\"/><stop offset=\"1\" stop-color=\"{c2}\"/>",
      fit.from.0, fit.from.1, fit.to.0, fit.to.1
    );
    // Ids derive from the content, so they cannot clash with another
    // picture's gradients on the same page (a clash would be identical).
    let hash = {
      use std::hash::{Hash, Hasher};
      let mut h = std::collections::hash_map::DefaultHasher::new();
      body.hash(&mut h);
      h.finish() & 0xff_ffff_ffff
    };
    let id = format!("s{hash:010x}");
    if self.seen.insert(hash) {
      let _ = writeln!(
        out,
        "<linearGradient id=\"{id}\" gradientUnits=\"userSpaceOnUse\" {body}</linearGradient>"
      );
    }
    Some(format!("url(#{id})"))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn affine_single_direction_is_exact() {
    // Colour changes along x only, every channel the same way.
    let pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    let colors = [[0.0, 0.2, 0.4], [1.0, 0.7, 0.4], [0.0, 0.2, 0.4]];
    let fit = fit_gradient(pts, colors).unwrap();
    assert!(fit.rms_error < 1e-9);
    assert!((fit.from.1 - fit.to.1).abs() < 1e-9);
  }

  #[test]
  fn crossed_channels_have_bounded_error() {
    let pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
    let colors = [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.6, 0.5]];
    let fit = fit_gradient(pts, colors).unwrap();
    assert!(fit.rms_error > 0.0 && fit.rms_error < 0.1 * 255.0);
  }
}
