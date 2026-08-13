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

use crate::syntax::Expr;

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
/// vertex of `outer` that `m` can see: cast a ray from `m` towards +x, take
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

#[cfg(test)]
mod tests {
  use super::*;

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
