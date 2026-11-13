//! Support for `Polygon[outer -> holes]` — a filled polygon with one or
//! more holes cut out of it.
//!
//! Wolfram writes such a polygon as a rule whose left side is the outer
//! boundary and whose right side lists the hole boundaries:
//! `Polygon[{p1, …, pn} -> {{q1, …, qm}, …}]`. A single hole may also be
//! given unwrapped (`{p1, …, pn} -> {q1, …, qm}`), which is the shape most
//! Demonstrations use.
//!
//! Two-dimensional output renders holes directly (an SVG path with the
//! even-odd fill rule), but three-dimensional output has to tessellate the
//! face into triangles, so this module also provides a general
//! triangulation for polygons with holes: each hole is spliced into the
//! outer boundary through a bridge edge, and the resulting simple polygon
//! is ear-clipped.

use super::*;

/// Split a `Polygon`/`Triangle` first argument into its outer boundary and
/// hole boundaries. Returns `None` when the argument is not a rule, i.e.
/// for an ordinary hole-free polygon.
///
/// `points_of` extracts a coordinate list of the caller's dimensionality
/// (2D for `Graphics`, 3D for `Graphics3D`), so the same shape analysis
/// serves both renderers.
pub fn split_holes<P, T>(
  arg: &Expr,
  points_of: &P,
) -> Option<(Vec<T>, Vec<Vec<T>>)>
where
  P: Fn(&Expr) -> Option<Vec<T>>,
{
  let Expr::Rule {
    pattern,
    replacement,
  } = arg
  else {
    return None;
  };
  let outer = points_of(pattern)?;
  // `outer -> {{q…}, …}`: every element of the right side is a hole.
  // `outer -> {q…}`: the right side is one hole. The two are told apart by
  // whether the right side parses as a coordinate list itself.
  let holes = if let Some(single) = points_of(replacement) {
    vec![single]
  } else {
    let Expr::List(items) = replacement.as_ref() else {
      return None;
    };
    items.iter().filter_map(points_of).collect()
  };
  Some((outer, holes))
}

/// Signed area of a closed 2D ring (positive when counterclockwise).
fn signed_area(pts: &[(f64, f64)], ring: &[usize]) -> f64 {
  let mut acc = 0.0;
  for i in 0..ring.len() {
    let (x1, y1) = pts[ring[i]];
    let (x2, y2) = pts[ring[(i + 1) % ring.len()]];
    acc += x1 * y2 - x2 * y1;
  }
  acc / 2.0
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
  (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Is `p` inside (or on the edge of) triangle `a b c`?
fn point_in_triangle(
  p: (f64, f64),
  a: (f64, f64),
  b: (f64, f64),
  c: (f64, f64),
) -> bool {
  let d1 = cross(a, b, p);
  let d2 = cross(b, c, p);
  let d3 = cross(c, a, p);
  let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
  let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
  !(has_neg && has_pos)
}

/// Splice `hole` (a clockwise ring) into `outer` (a counterclockwise ring)
/// through a bridge edge, yielding a single ring that traces both.
///
/// The bridge starts at the hole's rightmost vertex `m` and runs to a
/// vertex of `outer` that `m` can see: cast a ray from `m` toward +x, take
/// the first outer edge it meets and that edge's right-hand endpoint `p`,
/// then — if any reflex outer vertex blocks the segment `m…p` — replace `p`
/// by the blocking vertex closest in angle to the ray.
fn bridge_hole(
  pts: &[(f64, f64)],
  outer: &[usize],
  hole: &[usize],
) -> Vec<usize> {
  let Some(m) = (0..hole.len()).max_by(|&i, &j| {
    pts[hole[i]]
      .partial_cmp(&pts[hole[j]])
      .unwrap_or(std::cmp::Ordering::Equal)
  }) else {
    return outer.to_vec();
  };
  let mp = pts[hole[m]];

  // Nearest intersection of the +x ray from `mp` with an outer edge.
  let mut best: Option<(f64, usize)> = None;
  for i in 0..outer.len() {
    let a = pts[outer[i]];
    let b = pts[outer[(i + 1) % outer.len()]];
    if (a.1 > mp.1) == (b.1 > mp.1) {
      continue;
    }
    let t = (mp.1 - a.1) / (b.1 - a.1);
    let x = a.0 + t * (b.0 - a.0);
    if x < mp.0 {
      continue;
    }
    if best.is_none_or(|(bx, _)| x < bx) {
      best = Some((x, i));
    }
  }
  // A hole that is not enclosed by the outer ring cannot be bridged;
  // leave the outer ring alone.
  let Some((ix, edge)) = best else {
    return outer.to_vec();
  };
  let intersection = (ix, mp.1);

  // The candidate bridge endpoint is the edge endpoint with the larger x.
  let e0 = outer[edge];
  let e1 = outer[(edge + 1) % outer.len()];
  let mut p_at = if pts[e0].0 >= pts[e1].0 {
    edge
  } else {
    (edge + 1) % outer.len()
  };

  // Any reflex outer vertex inside the triangle (mp, intersection, p)
  // blocks the direct bridge; the visible one is the one whose direction
  // from `mp` makes the smallest angle with the +x ray.
  let mut best_angle = f64::INFINITY;
  let mut best_dist = f64::INFINITY;
  for i in 0..outer.len() {
    let v = pts[outer[i]];
    let prev = pts[outer[(i + outer.len() - 1) % outer.len()]];
    let next = pts[outer[(i + 1) % outer.len()]];
    // Reflex in a counterclockwise ring means a clockwise turn.
    if cross(prev, v, next) > 0.0 {
      continue;
    }
    if !point_in_triangle(v, mp, intersection, pts[outer[p_at]]) {
      continue;
    }
    let dx = v.0 - mp.0;
    let dy = v.1 - mp.1;
    let dist = (dx * dx + dy * dy).sqrt();
    if dist == 0.0 {
      continue;
    }
    let angle = (dy / dist).abs();
    if angle < best_angle || (angle == best_angle && dist < best_dist) {
      best_angle = angle;
      best_dist = dist;
      p_at = i;
    }
  }

  // outer[..=p] + hole[m..] + hole[..=m] + outer[p..]
  let mut out = Vec::with_capacity(outer.len() + hole.len() + 2);
  out.extend_from_slice(&outer[..=p_at]);
  out.extend(hole[m..].iter().copied());
  out.extend(hole[..=m].iter().copied());
  out.extend_from_slice(&outer[p_at..]);
  out
}

/// Drop repeated vertices from a ring, including a final vertex that
/// repeats the first. Wolfram code routinely writes rings closed
/// (`{a, b, c, a}`); a duplicated corner has no ear and would stall the
/// clipper.
fn close_ring(pts: &[(f64, f64)], ring: &[usize]) -> Vec<usize> {
  let mut out: Vec<usize> = Vec::with_capacity(ring.len());
  for &i in ring {
    if out.last().is_some_and(|&j| pts[j] == pts[i]) {
      continue;
    }
    out.push(i);
  }
  while out.len() > 1 && pts[out[0]] == pts[out[out.len() - 1]] {
    out.pop();
  }
  out
}

/// Ear-clip a simple counterclockwise ring into triangles.
fn ear_clip(pts: &[(f64, f64)], ring: &[usize]) -> Vec<[usize; 3]> {
  let mut idx: Vec<usize> = ring.to_vec();
  let mut tris = Vec::new();
  let mut guard = 0;
  while idx.len() > 3 {
    let n = idx.len();
    guard += 1;
    // Give up on a self-intersecting ring rather than spin forever; the
    // fallback below still emits a fan so the face is not lost.
    if guard > 4 * ring.len() + 8 {
      break;
    }
    let mut clipped = false;
    for i in 0..n {
      let a = idx[(i + n - 1) % n];
      let b = idx[i];
      let c = idx[(i + 1) % n];
      if cross(pts[a], pts[b], pts[c]) <= 0.0 {
        continue;
      }
      let contains = idx.iter().any(|&v| {
        v != a
          && v != b
          && v != c
          && point_in_triangle(pts[v], pts[a], pts[b], pts[c])
      });
      if contains {
        continue;
      }
      tris.push([a, b, c]);
      idx.remove(i);
      clipped = true;
      break;
    }
    if !clipped {
      break;
    }
  }
  if idx.len() == 3 {
    tris.push([idx[0], idx[1], idx[2]]);
  } else if idx.len() > 3 {
    for i in 1..idx.len() - 1 {
      tris.push([idx[0], idx[i], idx[i + 1]]);
    }
  }
  tris
}

/// The result of triangulating a polygon with holes.
pub struct HoledTriangulation {
  /// Triangles, as indices into the caller's vertex list.
  pub triangles: Vec<[usize; 3]>,
  /// Unordered index pairs that lie on the outer boundary or on a hole
  /// boundary — the edges a renderer should stroke. Everything else is an
  /// internal cut introduced by the triangulation.
  pub boundary_edges: std::collections::HashSet<(usize, usize)>,
}

/// Is the edge between `a` and `b` part of an original contour?
impl HoledTriangulation {
  pub fn is_boundary(&self, a: usize, b: usize) -> bool {
    self.boundary_edges.contains(&(a.min(b), a.max(b)))
  }
}

/// Triangulate a planar polygon with holes.
///
/// `outer` and each hole are given in a shared index space: the caller
/// supplies `pts` (all vertices, 2D), `outer` (indices of the outer ring)
/// and `holes` (indices of each hole ring). The returned triangles index
/// into `pts`.
pub fn triangulate_with_holes(
  pts: &[(f64, f64)],
  outer: &[usize],
  holes: &[Vec<usize>],
) -> HoledTriangulation {
  let mut boundary_edges = std::collections::HashSet::new();
  let mut record = |ring: &[usize]| {
    for i in 0..ring.len() {
      let a = ring[i];
      let b = ring[(i + 1) % ring.len()];
      boundary_edges.insert((a.min(b), a.max(b)));
    }
  };
  let mut ring = close_ring(pts, outer);
  if ring.len() < 3 {
    return HoledTriangulation {
      triangles: Vec::new(),
      boundary_edges,
    };
  }
  record(&ring);
  if signed_area(pts, &ring) < 0.0 {
    ring.reverse();
  }
  // Bridge the rightmost hole first: a hole spliced earlier can sit
  // between a later hole and the original boundary, and the bridge search
  // walks the ring as it stands.
  let mut ordered: Vec<Vec<usize>> = holes
    .iter()
    .map(|h| close_ring(pts, h))
    .filter(|h| h.len() >= 3)
    .map(|mut h| {
      if signed_area(pts, &h) > 0.0 {
        h.reverse();
      }
      h
    })
    .collect();
  ordered.sort_by(|a, b| {
    let max_x = |h: &Vec<usize>| {
      h.iter()
        .fold(f64::NEG_INFINITY, |acc, &i| acc.max(pts[i].0))
    };
    max_x(b)
      .partial_cmp(&max_x(a))
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  for hole in &ordered {
    record(hole);
    ring = bridge_hole(pts, &ring, hole);
  }
  HoledTriangulation {
    triangles: ear_clip(pts, &ring),
    boundary_edges,
  }
}

/// A vertex of a self-intersecting polygon's arrangement: it lies on the
/// polygon's edge `edge` (from corner `edge` to the next corner), a
/// fraction `t` of the way along it, so a caller can place it in its own
/// space (in 3D, between the two corners it was flattened from).
#[derive(Clone, Copy, Debug)]
pub struct EdgePoint {
  pub edge: usize,
  pub t: f64,
}

/// The pieces of a self-intersecting polygon that the even-odd rule fills.
///
/// The polygon's edges are cut wherever they cross one another, the plane
/// regions bounded by those pieces are traced, and the regions an odd
/// number of edges away from the outside are kept. A pentagram comes back
/// as its five tips, without the pentagon in the middle — exactly what
/// `Graphics` draws for it with SVG's `evenodd` fill.
///
/// Returns the arrangement's vertices (first the polygon's own corners,
/// then the crossing points) and the kept regions as counterclockwise
/// rings of indices into them, or `None` when no two edges cross, so the
/// polygon is simple and needs none of this.
pub fn even_odd_regions(
  pts: &[(f64, f64)],
) -> Option<(Vec<((f64, f64), EdgePoint)>, Vec<Vec<usize>>)> {
  let n = pts.len();
  if n < 4 {
    return None;
  }
  let scale = pts
    .iter()
    .fold(0.0f64, |m, p| m.max(p.0.abs()).max(p.1.abs()))
    .max(f64::MIN_POSITIVE);
  let eps = scale * 1e-10;
  let mut verts: Vec<((f64, f64), EdgePoint)> = pts
    .iter()
    .enumerate()
    .map(|(i, &p)| (p, EdgePoint { edge: i, t: 0.0 }))
    .collect();
  // The points along each edge, as (fraction, vertex index).
  let mut along: Vec<Vec<(f64, usize)>> =
    (0..n).map(|i| vec![(0.0, i), (1.0, (i + 1) % n)]).collect();
  let find_or_add =
    |verts: &mut Vec<((f64, f64), EdgePoint)>, p: (f64, f64), at: EdgePoint| {
      if let Some(k) = verts
        .iter()
        .position(|(q, _)| (q.0 - p.0).abs() <= eps && (q.1 - p.1).abs() <= eps)
      {
        k
      } else {
        verts.push((p, at));
        verts.len() - 1
      }
    };
  let mut crossed = false;
  for i in 0..n {
    let (a, b) = (pts[i], pts[(i + 1) % n]);
    for j in i + 1..n {
      // Neighboring edges share a corner and cannot cross properly.
      if j == i + 1 || (i == 0 && j == n - 1) {
        continue;
      }
      let (c, d) = (pts[j], pts[(j + 1) % n]);
      let r = (b.0 - a.0, b.1 - a.1);
      let s = (d.0 - c.0, d.1 - c.1);
      let denom = r.0 * s.1 - r.1 * s.0;
      if denom.abs() <= eps * eps {
        continue;
      }
      let t = ((c.0 - a.0) * s.1 - (c.1 - a.1) * s.0) / denom;
      let u = ((c.0 - a.0) * r.1 - (c.1 - a.1) * r.0) / denom;
      let inside = |x: f64| x > 1e-9 && x < 1.0 - 1e-9;
      if !inside(t) || !inside(u) {
        continue;
      }
      crossed = true;
      let p = (a.0 + t * r.0, a.1 + t * r.1);
      let k = find_or_add(&mut verts, p, EdgePoint { edge: i, t });
      along[i].push((t, k));
      along[j].push((u, k));
    }
  }
  if !crossed {
    return None;
  }
  // The undirected pieces between consecutive points along every edge.
  let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); verts.len()];
  for list in &mut along {
    list.sort_by(|x, y| x.0.total_cmp(&y.0));
    for w in list.windows(2) {
      let (p, q) = (w[0].1, w[1].1);
      if p != q && !neighbors[p].contains(&q) {
        neighbors[p].push(q);
        neighbors[q].push(p);
      }
    }
  }
  let angle = |from: usize, to: usize| {
    let (p, q) = (verts[from].0, verts[to].0);
    (q.1 - p.1).atan2(q.0 - p.0)
  };
  for (v, list) in neighbors.iter_mut().enumerate() {
    list.sort_by(|&x, &y| angle(v, x).total_cmp(&angle(v, y)));
  }
  // Trace every region with it on the left: arriving at `v` from `u`, the
  // region continues along the first piece clockwise from `v -> u`.
  let mut used = std::collections::HashSet::new();
  let mut regions = Vec::new();
  for start in 0..verts.len() {
    for &first in &neighbors[start] {
      if used.contains(&(start, first)) {
        continue;
      }
      let mut ring = Vec::new();
      let (mut u, mut v) = (start, first);
      while used.insert((u, v)) {
        ring.push(u);
        let list = &neighbors[v];
        let back = list.iter().position(|&w| w == u).unwrap_or(0);
        let next = list[(back + list.len() - 1) % list.len()];
        (u, v) = (v, next);
      }
      let flat: Vec<(f64, f64)> = ring.iter().map(|&k| verts[k].0).collect();
      // Clockwise rings trace the outside of the whole arrangement.
      if ring.len() < 3
        || signed_area(&flat, &(0..flat.len()).collect::<Vec<_>>()) <= 0.0
      {
        continue;
      }
      // Probe just inside the region, off the middle of its first side.
      let (a, b) = (flat[0], flat[1]);
      let len = (b.0 - a.0).hypot(b.1 - a.1).max(f64::MIN_POSITIVE);
      let step = len * 1e-6;
      let probe = (
        f64::midpoint(a.0, b.0) - (b.1 - a.1) / len * step,
        f64::midpoint(a.1, b.1) + (b.0 - a.0) / len * step,
      );
      let mut odd = false;
      for i in 0..n {
        let (p, q) = (pts[i], pts[(i + 1) % n]);
        if (p.1 > probe.1) != (q.1 > probe.1)
          && probe.0 < p.0 + (probe.1 - p.1) / (q.1 - p.1) * (q.0 - p.0)
        {
          odd = !odd;
        }
      }
      if odd {
        regions.push(ring);
      }
    }
  }
  Some((verts, regions))
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Five corners of a regular pentagon, visited in pentagram order.
  fn pentagram() -> Vec<(f64, f64)> {
    (0..5)
      .map(|k| {
        let a = std::f64::consts::FRAC_PI_2
          + 4.0 * std::f64::consts::PI * k as f64 / 5.0;
        (a.cos(), a.sin())
      })
      .collect()
  }

  #[test]
  fn even_odd_regions_of_a_pentagram_are_its_tips() {
    let pts = pentagram();
    let (verts, regions) = even_odd_regions(&pts).unwrap();
    // Five corners plus the five points where the chords cross.
    assert_eq!(verts.len(), 10);
    // The five tips are kept; the pentagon in the middle, covered twice,
    // is not.
    assert_eq!(regions.len(), 5);
    assert!(regions.iter().all(|r| r.len() == 3));
    let center_hit = regions.iter().any(|r| {
      let flat: Vec<(f64, f64)> = r.iter().map(|&k| verts[k].0).collect();
      point_in_triangle((0.0, 0.0), flat[0], flat[1], flat[2])
    });
    assert!(!center_hit);
    // Every crossing point lies on the edge it reports, where it says.
    for (p, at) in &verts {
      let a = pts[at.edge];
      let b = pts[(at.edge + 1) % pts.len()];
      let q = (a.0 + at.t * (b.0 - a.0), a.1 + at.t * (b.1 - a.1));
      assert!((p.0 - q.0).abs() < 1e-12 && (p.1 - q.1).abs() < 1e-12);
    }
  }

  #[test]
  fn even_odd_regions_of_a_bow_tie_are_its_two_triangles() {
    let pts = vec![(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)];
    let (_, regions) = even_odd_regions(&pts).unwrap();
    assert_eq!(regions.len(), 2);
  }

  #[test]
  fn even_odd_regions_skip_simple_polygons() {
    let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    assert!(even_odd_regions(&square).is_none());
    let notch = vec![
      (0.0, 0.0),
      (3.0, 0.0),
      (3.0, 1.0),
      (1.0, 1.0),
      (1.0, 2.0),
      (3.0, 2.0),
      (3.0, 3.0),
      (0.0, 3.0),
    ];
    assert!(even_odd_regions(&notch).is_none());
  }

  fn area(pts: &[(f64, f64)], tris: &HoledTriangulation) -> f64 {
    tris
      .triangles
      .iter()
      .map(|t| cross(pts[t[0]], pts[t[1]], pts[t[2]]).abs() / 2.0)
      .sum()
  }

  #[test]
  fn triangulates_square_with_square_hole() {
    let pts = vec![
      (0.0, 0.0),
      (4.0, 0.0),
      (4.0, 4.0),
      (0.0, 4.0),
      (1.0, 1.0),
      (3.0, 1.0),
      (3.0, 3.0),
      (1.0, 3.0),
    ];
    let tris = triangulate_with_holes(&pts, &[0, 1, 2, 3], &[vec![4, 5, 6, 7]]);
    // 16 (outer) - 4 (hole) = 12
    assert!((area(&pts, &tris) - 12.0).abs() < 1e-9);
  }

  #[test]
  fn triangulates_polygon_without_holes() {
    let pts = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
    let tris = triangulate_with_holes(&pts, &[0, 1, 2, 3], &[]);
    assert_eq!(tris.triangles.len(), 2);
    assert!((area(&pts, &tris) - 4.0).abs() < 1e-9);
    assert!(tris.is_boundary(0, 1));
    // The fan cut from corner 0 to corner 2 is internal.
    assert!(!tris.is_boundary(0, 2));
  }

  #[test]
  fn triangulates_two_holes() {
    let pts = vec![
      (0.0, 0.0),
      (10.0, 0.0),
      (10.0, 10.0),
      (0.0, 10.0),
      (1.0, 1.0),
      (3.0, 1.0),
      (3.0, 3.0),
      (1.0, 3.0),
      (6.0, 6.0),
      (8.0, 6.0),
      (8.0, 8.0),
      (6.0, 8.0),
    ];
    let tris = triangulate_with_holes(
      &pts,
      &[0, 1, 2, 3],
      &[vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
    );
    assert!((area(&pts, &tris) - (100.0 - 4.0 - 4.0)).abs() < 1e-9);
  }

  #[test]
  fn accepts_clockwise_input() {
    let pts = vec![
      (0.0, 0.0),
      (0.0, 4.0),
      (4.0, 4.0),
      (4.0, 0.0),
      (1.0, 1.0),
      (1.0, 3.0),
      (3.0, 3.0),
      (3.0, 1.0),
    ];
    let tris = triangulate_with_holes(&pts, &[0, 1, 2, 3], &[vec![4, 5, 6, 7]]);
    assert!((area(&pts, &tris) - 12.0).abs() < 1e-9);
  }
}
