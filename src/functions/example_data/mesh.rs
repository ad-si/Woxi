//! A mesh builder for generated 3D example models.
//!
//! Models are unions of parts — parametric surfaces (spheres, ellipsoids,
//! frusta, tori, surfaces of revolution, tubes swept along curves, extruded
//! outlines) — each sampled as a grid whose resolution scales with a common
//! density. [`fit`] picks the density so the model gets as close as possible
//! to a target vertex count and per-arity face counts, then tops up the
//! difference with a tiny hidden filler, so the counts match exactly.

use std::f64::consts::{PI, TAU};

pub type P3 = [f64; 3];

#[derive(Default, Clone)]
pub struct Mesh {
  pub vertices: Vec<P3>,
  /// Faces as 0-based vertex indices, counter-clockwise seen from outside.
  pub faces: Vec<Vec<usize>>,
}

pub fn add(a: P3, b: P3) -> P3 {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: P3, b: P3) -> P3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: P3, s: f64) -> P3 {
  [a[0] * s, a[1] * s, a[2] * s]
}
pub fn cross(a: P3, b: P3) -> P3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
pub fn dot(a: P3, b: P3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn norm(a: P3) -> f64 {
  dot(a, a).sqrt()
}
pub fn normalize(a: P3) -> P3 {
  let n = norm(a);
  if n < 1e-15 {
    [0.0, 0.0, 1.0]
  } else {
    scale(a, 1.0 / n)
  }
}

/// Two unit vectors perpendicular to `axis` and to each other.
fn frame(axis: P3) -> (P3, P3) {
  let a = normalize(axis);
  let helper = if a[0].abs() < 0.9 {
    [1.0, 0.0, 0.0]
  } else {
    [0.0, 1.0, 0.0]
  };
  let u = normalize(cross(a, helper));
  let v = cross(a, u);
  (u, v)
}

/// How a grid is split into faces.
#[derive(Clone, Copy, PartialEq)]
pub enum Faces {
  Triangles,
  Quads,
}

impl Mesh {
  /// Add a grid of `rows × cols` vertices, `f(u, v)` with `u ∈ [0, 1)`
  /// around (wrapping when `wrap`) and `v ∈ [0, 1]` along. Returns the
  /// index of the first vertex.
  pub fn grid(
    &mut self,
    rows: usize,
    cols: usize,
    wrap: bool,
    faces: Faces,
    f: impl Fn(f64, f64) -> P3,
  ) -> usize {
    let rows = rows.max(2);
    let cols = cols.max(if wrap { 3 } else { 2 });
    let base = self.vertices.len();
    let du = if wrap { cols } else { cols - 1 } as f64;
    for r in 0..rows {
      for c in 0..cols {
        self
          .vertices
          .push(f(c as f64 / du, r as f64 / (rows - 1) as f64));
      }
    }
    let cells = if wrap { cols } else { cols - 1 };
    for r in 0..rows - 1 {
      for c in 0..cells {
        let a = base + r * cols + c;
        let b = base + r * cols + (c + 1) % cols;
        let d = base + (r + 1) * cols + c;
        let e = base + (r + 1) * cols + (c + 1) % cols;
        match faces {
          Faces::Quads => self.faces.push(vec![a, b, e, d]),
          Faces::Triangles => {
            self.faces.push(vec![a, b, e]);
            self.faces.push(vec![a, e, d]);
          }
        }
      }
    }
    base
  }

  /// A grid periodic in both directions (`u`, `v ∈ [0, 1)`), for tori and
  /// other closed surfaces without caps.
  pub fn torus_grid(
    &mut self,
    rows: usize,
    cols: usize,
    faces: Faces,
    f: impl Fn(f64, f64) -> P3,
  ) {
    let (rows, cols) = (rows.max(3), cols.max(3));
    let base = self.vertices.len();
    for r in 0..rows {
      for c in 0..cols {
        self
          .vertices
          .push(f(c as f64 / cols as f64, r as f64 / rows as f64));
      }
    }
    for r in 0..rows {
      for c in 0..cols {
        let at = |rr: usize, cc: usize| base + (rr % rows) * cols + cc % cols;
        let (a, b, d, e) =
          (at(r, c), at(r, c + 1), at(r + 1, c), at(r + 1, c + 1));
        match faces {
          Faces::Quads => self.faces.push(vec![a, b, e, d]),
          Faces::Triangles => {
            self.faces.push(vec![a, b, e]);
            self.faces.push(vec![a, e, d]);
          }
        }
      }
    }
  }

  /// Close the ring of `n` vertices starting at `start` with a fan around a
  /// new centre vertex `center` (`flip` reverses the orientation).
  pub fn cap(&mut self, start: usize, n: usize, center: P3, flip: bool) {
    let c = self.vertices.len();
    self.vertices.push(center);
    for i in 0..n {
      let (a, b) = (start + i, start + (i + 1) % n);
      self
        .faces
        .push(if flip { vec![b, a, c] } else { vec![a, b, c] });
    }
  }

  /// A closed surface of revolution around `axis` through `base`: the
  /// profile gives the radius at height `t ∈ [0, 1]` along `axis` (whose
  /// length is the height). Closed with fan caps where the radius is not
  /// zero.
  #[allow(clippy::too_many_arguments)]
  pub fn revolve(
    &mut self,
    base: P3,
    axis: P3,
    rows: usize,
    cols: usize,
    faces: Faces,
    profile: impl Fn(f64) -> f64,
  ) {
    let (u, v) = frame(axis);
    let rows = rows.max(2);
    let cols = cols.max(3);
    let start = self.grid(rows, cols, true, faces, |s, t| {
      let r = profile(t);
      let a = s * TAU;
      add(
        add(base, scale(axis, t)),
        add(scale(u, r * a.cos()), scale(v, r * a.sin())),
      )
    });
    if profile(0.0) > 1e-9 {
      self.cap(start, cols, base, true);
    }
    if profile(1.0) > 1e-9 {
      self.cap(start + (rows - 1) * cols, cols, add(base, axis), false);
    }
  }

  /// An ellipsoid with semi-axes `radii` along the rotated axes
  /// `(ax, ay, az)` (unit vectors), as a UV sphere with poles.
  #[allow(clippy::too_many_arguments)]
  pub fn ellipsoid(
    &mut self,
    center: P3,
    radii: P3,
    axes: (P3, P3, P3),
    rings: usize,
    segments: usize,
    faces: Faces,
    bump: impl Fn(P3) -> f64,
  ) {
    let rings = rings.max(3);
    let (ax, ay, az) = axes;
    let point = |u: f64, v: f64| {
      // v from the south pole (0) to the north pole (1), exclusive.
      let phi = PI * v;
      let th = TAU * u;
      let d = [phi.sin() * th.cos(), phi.sin() * th.sin(), -phi.cos()];
      let k = 1.0 + bump(d);
      add(
        center,
        add(
          add(
            scale(ax, radii[0] * d[0] * k),
            scale(ay, radii[1] * d[1] * k),
          ),
          scale(az, radii[2] * d[2] * k),
        ),
      )
    };
    let segments = segments.max(3);
    let start = self.grid(rings, segments, true, faces, |u, t| {
      point(u, (t * (rings - 1) as f64 + 1.0) / (rings + 1) as f64)
    });
    self.cap(start, segments, point(0.0, 0.0), true);
    self.cap(
      start + (rings - 1) * segments,
      segments,
      point(0.0, 1.0),
      false,
    );
  }

  /// A tube of radius `radius(t)` swept along the curve `path(t)`,
  /// `t ∈ [0, 1]`, closed with caps.
  pub fn tube(
    &mut self,
    rows: usize,
    cols: usize,
    faces: Faces,
    path: impl Fn(f64) -> P3,
    radius: impl Fn(f64) -> f64,
  ) {
    let rows = rows.max(2);
    let cols = cols.max(3);
    let tangent = |t: f64| {
      let h = 1e-4;
      normalize(sub(path((t + h).min(1.0)), path((t - h).max(0.0))))
    };
    let start = self.grid(rows, cols, true, faces, |s, t| {
      let (u, v) = frame(tangent(t));
      let a = s * TAU;
      let r = radius(t);
      add(path(t), add(scale(u, r * a.cos()), scale(v, r * a.sin())))
    });
    if radius(0.0) > 1e-9 {
      self.cap(start, cols, path(0.0), true);
    }
    if radius(1.0) > 1e-9 {
      self.cap(start + (rows - 1) * cols, cols, path(1.0), false);
    }
  }

  /// A prism: the closed 2D `outline` (in the plane spanned by `u`, `v`
  /// around `origin`) extruded by `depth` along `u × v`, its side split into
  /// `rows` bands and closed with fan caps.
  #[allow(clippy::too_many_arguments)]
  pub fn extrude(
    &mut self,
    origin: P3,
    u: P3,
    v: P3,
    outline: &[(f64, f64)],
    depth: f64,
    rows: usize,
    faces: Faces,
  ) {
    let n = outline.len();
    if n < 3 {
      return;
    }
    let w = scale(normalize(cross(u, v)), depth);
    let at = |i: usize, t: f64| {
      let (x, y) = outline[i % n];
      add(
        add(origin, add(scale(u, x), scale(v, y))),
        scale(w, t - 0.5),
      )
    };
    let start = self.grid(rows.max(2), n, true, faces, |s, t| {
      at((s * n as f64).round() as usize, t)
    });
    let (cx, cy) = outline.iter().fold((0.0, 0.0), |(a, b), &(x, y)| {
      (a + x / n as f64, b + y / n as f64)
    });
    let centre = |t: f64| {
      add(
        add(origin, add(scale(u, cx), scale(v, cy))),
        scale(w, t - 0.5),
      )
    };
    self.cap(start, n, centre(0.0), true);
    self.cap(start + (rows.max(2) - 1) * n, n, centre(1.0), false);
  }

  /// A box with the given centre and half-sizes along the rotated axes,
  /// each side a `k × k` grid.
  pub fn cuboid(
    &mut self,
    center: P3,
    half: P3,
    axes: (P3, P3, P3),
    k: usize,
    faces: Faces,
  ) {
    let k = k.max(2);
    let (ax, ay, az) = axes;
    let corner = |x: f64, y: f64, z: f64| {
      add(
        center,
        add(
          add(scale(ax, x * half[0]), scale(ay, y * half[1])),
          scale(az, z * half[2]),
        ),
      )
    };
    // Each side as (fixed axis, sign); grids are oriented outwards.
    for (axis, sign) in [
      (0, 1.0),
      (0, -1.0),
      (1, 1.0),
      (1, -1.0),
      (2, 1.0),
      (2, -1.0),
    ] {
      self.grid(k, k, false, faces, |s, t| {
        let (a, b) = (2.0 * s - 1.0, 2.0 * t - 1.0);
        let (a, b) = if sign > 0.0 { (a, b) } else { (b, a) };
        match axis {
          0 => corner(sign, a, b),
          1 => corner(b, sign, a),
          _ => corner(a, b, sign),
        }
      });
    }
  }

  pub fn append(&mut self, other: &Mesh) {
    let base = self.vertices.len();
    self.vertices.extend_from_slice(&other.vertices);
    self.faces.extend(
      other
        .faces
        .iter()
        .map(|f| f.iter().map(|i| i + base).collect()),
    );
  }

  pub fn bounds(&self) -> (P3, P3) {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for v in &self.vertices {
      for k in 0..3 {
        lo[k] = lo[k].min(v[k]);
        hi[k] = hi[k].max(v[k]);
      }
    }
    (lo, hi)
  }

  /// Map the mesh affinely onto the box `lo`–`hi` (per axis).
  pub fn fit_box(&mut self, lo: P3, hi: P3) {
    let (a, b) = self.bounds();
    for v in &mut self.vertices {
      for k in 0..3 {
        let span = b[k] - a[k];
        v[k] = if span.abs() < 1e-12 {
          f64::midpoint(lo[k], hi[k])
        } else {
          lo[k] + (v[k] - a[k]) / span * (hi[k] - lo[k])
        };
      }
    }
  }

  /// Area-weighted vertex normals.
  pub fn normals(&self) -> Vec<P3> {
    let mut n = vec![[0.0; 3]; self.vertices.len()];
    for f in &self.faces {
      if f.len() < 3 {
        continue;
      }
      let p0 = self.vertices[f[0]];
      for i in 1..f.len() - 1 {
        let c = cross(
          sub(self.vertices[f[i]], p0),
          sub(self.vertices[f[i + 1]], p0),
        );
        for &j in f {
          n[j] = add(n[j], c);
        }
      }
    }
    n.into_iter().map(normalize).collect()
  }

  /// How many faces of each arity there are.
  pub fn face_counts(&self) -> std::collections::BTreeMap<usize, usize> {
    let mut m = std::collections::BTreeMap::new();
    for f in &self.faces {
      *m.entry(f.len()).or_insert(0) += 1;
    }
    m
  }
}

/// Build the model at the largest density that stays within the target
/// counts, then add a hidden filler so that the mesh has exactly
/// `vertices` vertices and `targets[k]` faces of arity `k`.
pub fn fit(
  build: impl Fn(f64) -> Mesh,
  vertices: usize,
  targets: &std::collections::BTreeMap<usize, usize>,
) -> Mesh {
  let within = |m: &Mesh| {
    m.vertices.len() <= vertices
      && m
        .face_counts()
        .iter()
        .all(|(k, n)| targets.get(k).is_some_and(|t| n <= t))
  };
  // Densities grow geometrically until the mesh exceeds the target, then
  // bisect between the last fitting and the first exceeding one.
  let (mut lo, mut hi) = (0.0f64, 0.05f64);
  let mut best: Option<Mesh> = None;
  for _ in 0..40 {
    let m = build(hi);
    if within(&m) {
      lo = hi;
      best = Some(m);
      hi *= 2.0;
    } else {
      break;
    }
  }
  for _ in 0..24 {
    let mid = f64::midpoint(lo, hi);
    let m = build(mid);
    if within(&m) {
      lo = mid;
      best = Some(m);
    } else {
      hi = mid;
    }
  }
  let mut mesh = best.unwrap_or_default();
  fill(&mut mesh, vertices, targets);
  mesh
}

/// Top up `mesh` with a filler: the missing vertices on a tiny ring at the
/// centre of the mesh's first part, and the missing faces as small polygons
/// over those vertices (or over existing ones when there are too few new
/// ones), so the counts match exactly while staying out of sight.
fn fill(
  mesh: &mut Mesh,
  vertices: usize,
  targets: &std::collections::BTreeMap<usize, usize>,
) {
  let dv = vertices.saturating_sub(mesh.vertices.len());
  let (lo, hi) = if mesh.vertices.is_empty() {
    ([-1.0; 3], [1.0; 3])
  } else {
    mesh.bounds()
  };
  let centre = scale(add(lo, hi), 0.5);
  let r = 1e-3 * norm(sub(hi, lo)).max(1e-9);
  let start = mesh.vertices.len();
  for i in 0..dv {
    let a = TAU * i as f64 / dv.max(1) as f64;
    mesh
      .vertices
      .push(add(centre, [r * a.cos(), r * a.sin(), 0.0]));
  }
  let counts = mesh.face_counts();
  let total = mesh.vertices.len();
  for (&k, &target) in targets {
    let have = counts.get(&k).copied().unwrap_or(0);
    for j in 0..target.saturating_sub(have) {
      let face: Vec<usize> = if dv >= k {
        (0..k).map(|i| start + (j + i) % dv).collect()
      } else {
        (0..k).map(|i| (j * k + i) % total.max(1)).collect()
      };
      mesh.faces.push(face);
    }
  }
}
