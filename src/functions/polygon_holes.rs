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

/// Split a `Polygon`/`Triangle` first argument into faces, each an outer
/// boundary with its hole boundaries. Returns `None` when the argument is
/// not a rule, i.e. for an ordinary hole-free polygon.
///
/// Both rule forms are accepted:
/// - `outer -> holes`: one face, where `holes` is a single ring `{q…}` or
///   a list of rings `{{q…}, …}`;
/// - `{outer1, outer2, …} -> {holes1, holes2, …}`: one face per outer
///   boundary, paired by position with its holes spec (again a single ring
///   or a list of rings).
///
/// `points_of` extracts a coordinate list of the caller's dimensionality
/// (2D for `Graphics`, 3D for `Graphics3D`), so the same shape analysis
/// serves both renderers.
pub fn split_holes<P, T>(
  arg: &Expr,
  points_of: &P,
) -> Option<Vec<(Vec<T>, Vec<Vec<T>>)>>
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
  if let Some(outer) = points_of(pattern) {
    return Some(vec![(outer, holes_of(replacement, points_of)?)]);
  }
  // `{outer…} -> {holes…}`: several faces, each with its own holes.
  let (Expr::List(outers), Expr::List(hole_specs)) =
    (pattern.as_ref(), replacement.as_ref())
  else {
    return None;
  };
  if outers.is_empty() || outers.len() != hole_specs.len() {
    return None;
  }
  outers
    .iter()
    .zip(hole_specs.iter())
    .map(|(outer, spec)| Some((points_of(outer)?, holes_of(spec, points_of)?)))
    .collect()
}

/// The hole rings of one face: `{q…}` is one hole, `{{q…}, …}` several.
/// The two are told apart by whether the spec parses as a coordinate list
/// itself.
fn holes_of<P, T>(spec: &Expr, points_of: &P) -> Option<Vec<Vec<T>>>
where
  P: Fn(&Expr) -> Option<Vec<T>>,
{
  if let Some(single) = points_of(spec) {
    return Some(vec![single]);
  }
  let Expr::List(items) = spec else {
    return None;
  };
  Some(items.iter().filter_map(points_of).collect())
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
/// edge from corner `from` to corner `to` (indices into the caller's
/// vertex list), a fraction `t` of the way along it, so a caller can place
/// it in its own space (in 3D, between the two corners it was flattened
/// from).
#[derive(Clone, Copy, Debug)]
pub struct EdgePoint {
  pub from: usize,
  pub to: usize,
  pub t: f64,
}

/// The pieces of a set of crossing rings that the even-odd rule fills.
///
/// `rings` lists the boundaries (an outline, or an outline followed by its
/// holes) as indices into `pts`. Their edges are cut wherever they cross
/// one another, the plane regions bounded by those pieces are traced, and
/// the regions an odd number of edges away from the outside are kept. A
/// pentagram comes back as its five tips, without the pentagon in the
/// middle, and a hole that reaches past its outline (`Polygon[outer ->
/// hole]`) leaves the outline's remainder plus the hole's protruding
/// parts — exactly what `Graphics` draws for them with SVG's `evenodd`
/// fill.
///
/// Returns the arrangement's vertices (first the corners in `pts`, then
/// the crossing points) and the kept regions, as rings of indices into
/// them, or `None` when no two edges cross and no ring revisits a corner,
/// so the rings are simple and disjoint and need none of this.
/// A revisited corner — an outline that runs around one loop and then a
/// second loop from the same point — counts as a touch: a loop nested
/// inside the other is wound twice and left empty, as with a crossing.
pub fn even_odd_regions(
  pts: &[(f64, f64)],
  rings: &[Vec<usize>],
) -> Option<(Vec<((f64, f64), EdgePoint)>, Vec<EvenOddRegion>)> {
  // Every ring's edges, with repeated (closing) corners dropped. Each edge
  // also records whether it neighbors the next one in its ring.
  let mut edges: Vec<(usize, usize)> = Vec::new();
  let mut ring_of: Vec<(usize, usize)> = Vec::new();
  for ring in rings {
    let ring = close_ring(pts, ring);
    if ring.len() < 3 {
      continue;
    }
    let first = edges.len();
    for i in 0..ring.len() {
      edges.push((ring[i], ring[(i + 1) % ring.len()]));
      ring_of.push((first, ring.len()));
    }
  }
  if edges.len() < 4 {
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
    .map(|(i, &p)| {
      (
        p,
        EdgePoint {
          from: i,
          to: i,
          t: 0.0,
        },
      )
    })
    .collect();
  // A corner the outline passes through again later (two loops touching
  // at a point) is one junction of the arrangement: every visit maps to
  // its first occurrence, so the regions on either side are traced apart.
  let same = |p: (f64, f64), q: (f64, f64)| {
    (p.0 - q.0).abs() <= eps && (p.1 - q.1).abs() <= eps
  };
  let canon: Vec<usize> = (0..pts.len())
    .map(|i| (0..i).find(|&j| same(pts[j], pts[i])).unwrap_or(i))
    .collect();
  // A ring (already rid of doubled neighbors and its closing repeat) that
  // comes back to a corner it visited makes two loops touch there.
  let mut crossed = rings.iter().any(|ring| {
    let ring = close_ring(pts, ring);
    let mut seen = std::collections::HashSet::new();
    ring.iter().any(|&k| !seen.insert(canon[k]))
  });
  // The points along each edge, as (fraction, vertex index).
  let mut along: Vec<Vec<(f64, usize)>> = edges
    .iter()
    .map(|&(a, b)| vec![(0.0, canon[a]), (1.0, canon[b])])
    .collect();
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
  for i in 0..edges.len() {
    let (a, b) = (pts[edges[i].0], pts[edges[i].1]);
    for j in i + 1..edges.len() {
      // Neighboring edges of one ring share a corner and cannot cross
      // properly.
      let (first, len) = ring_of[i];
      if ring_of[j].0 == first
        && (j == i + 1 || (i == first && j == first + len - 1))
      {
        continue;
      }
      let (c, d) = (pts[edges[j].0], pts[edges[j].1]);
      let r = (b.0 - a.0, b.1 - a.1);
      let s = (d.0 - c.0, d.1 - c.1);
      let denom = r.0 * s.1 - r.1 * s.0;
      // (Nearly) parallel edges meet, if at all, only where one's corner
      // lies on the other, which is handled below.
      if denom.abs() <= 1e-9 * r.0.hypot(r.1) * s.0.hypot(s.1) {
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
      let at = EdgePoint {
        from: edges[i].0,
        to: edges[i].1,
        t,
      };
      let k = find_or_add(&mut verts, p, at);
      along[i].push((t, k));
      along[j].push((u, k));
    }
  }
  // A corner lying on another edge (two loops sharing a stretch of
  // boundary, as when one is the other scaled about a common corner)
  // splits that edge there, so overlapping edges are cut into the same
  // pieces and traced as one.
  let corners: std::collections::BTreeSet<usize> =
    edges.iter().flat_map(|&(a, b)| [canon[a], canon[b]]).collect();
  for (i, &(ea, eb)) in edges.iter().enumerate() {
    let (a, b) = (pts[ea], pts[eb]);
    let r = (b.0 - a.0, b.1 - a.1);
    let len2 = r.0 * r.0 + r.1 * r.1;
    if len2 <= eps * eps {
      continue;
    }
    for &c in &corners {
      if c == canon[ea] || c == canon[eb] {
        continue;
      }
      let p = pts[c];
      let t = ((p.0 - a.0) * r.0 + (p.1 - a.1) * r.1) / len2;
      if !(t > 1e-9 && t < 1.0 - 1e-9) {
        continue;
      }
      let off = ((p.0 - a.0) * r.1 - (p.1 - a.1) * r.0).abs() / len2.sqrt();
      if off <= scale * 1e-9 && !along[i].iter().any(|&(_, v)| v == c) {
        crossed = true;
        along[i].push((t, c));
      }
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
  let point_in_ring = |probe: (f64, f64), flat: &[(f64, f64)]| {
    let mut odd = false;
    for i in 0..flat.len() {
      let (p, q) = (flat[i], flat[(i + 1) % flat.len()]);
      if (p.1 > probe.1) != (q.1 > probe.1)
        && probe.0 < p.0 + (probe.1 - p.1) / (q.1 - p.1) * (q.0 - p.0)
      {
        odd = !odd;
      }
    }
    odd
  };
  let mut used = std::collections::HashSet::new();
  // Counterclockwise faces as (ring, area, kept by the even-odd rule),
  // and the clockwise outsides of the arrangement's separate pieces.
  let mut faces: Vec<(Vec<usize>, f64, bool)> = Vec::new();
  let mut outsides: Vec<Vec<usize>> = Vec::new();
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
      if ring.len() < 3 {
        continue;
      }
      let flat: Vec<(f64, f64)> = ring.iter().map(|&k| verts[k].0).collect();
      let area = signed_area(&flat, &(0..flat.len()).collect::<Vec<_>>());
      // Clockwise rings trace the outside of a piece of the arrangement.
      if area <= 0.0 {
        outsides.push(ring);
        continue;
      }
      // Probe just inside the region, off the middle of its longest side:
      // a short side may end where the region pinches to nothing (at a
      // corner where two loops touch), leaving no room beside it.
      let side = |i: usize| {
        let (a, b) = (flat[i], flat[(i + 1) % flat.len()]);
        (b.0 - a.0).hypot(b.1 - a.1)
      };
      let longest = (0..flat.len())
        .max_by(|&i, &j| side(i).total_cmp(&side(j)))
        .unwrap_or(0);
      let (a, b) = (flat[longest], flat[(longest + 1) % flat.len()]);
      let len = (b.0 - a.0).hypot(b.1 - a.1).max(f64::MIN_POSITIVE);
      let step = len * 1e-6;
      let probe = (
        f64::midpoint(a.0, b.0) - (b.1 - a.1) / len * step,
        f64::midpoint(a.1, b.1) + (b.0 - a.0) / len * step,
      );
      let mut odd = false;
      for &(p, q) in &edges {
        let (p, q) = (pts[p], pts[q]);
        if (p.1 > probe.1) != (q.1 > probe.1)
          && probe.0 < p.0 + (probe.1 - p.1) / (q.1 - p.1) * (q.0 - p.0)
        {
          odd = !odd;
        }
      }
      faces.push((ring, area, odd));
    }
  }
  // A piece lying wholly inside a face (a hole that crosses nothing, next
  // to one that does) is cut out of the smallest face around it.
  let mut regions: Vec<EvenOddRegion> = faces
    .iter()
    .map(|(ring, _, _)| EvenOddRegion {
      outer: ring.clone(),
      holes: Vec::new(),
    })
    .collect();
  for outside in outsides {
    let probe = verts[outside[0]].0;
    let around = faces
      .iter()
      .enumerate()
      .filter(|(_, (ring, _, _))| !ring.contains(&outside[0]))
      .filter(|(_, (ring, _, _))| {
        let flat: Vec<(f64, f64)> = ring.iter().map(|&k| verts[k].0).collect();
        point_in_ring(probe, &flat)
      })
      .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
      .map(|(i, _)| i);
    if let Some(i) = around {
      regions[i].holes.push(outside);
    }
  }
  let regions = regions
    .into_iter()
    .zip(&faces)
    .filter(|(_, (_, _, kept))| *kept)
    .map(|(region, _)| region)
    .collect();
  Some((verts, regions))
}

/// One region the even-odd rule fills: a counterclockwise outline and the
/// boundaries of the separate pieces of the arrangement lying inside it,
/// all as indices into the arrangement's vertices.
#[derive(Clone, Debug)]
pub struct EvenOddRegion {
  pub outer: Vec<usize>,
  pub holes: Vec<Vec<usize>>,
}

#[cfg(test)]
mod tests {
  use super::*;

  /// One ring through all of `pts`, in order.
  fn whole(pts: &[(f64, f64)]) -> Vec<usize> {
    (0..pts.len()).collect()
  }

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
  fn split_holes_pairs_listed_outers_with_their_holes() {
    use crate::functions::graphics::expr_to_point_list;
    let arg = crate::interpret_to_expr(
      "{{{0, 0}, {4, 0}, {4, 4}}, {{5, 0}, {9, 0}, {9, 4}}} -> \
       {{{1, 1}, {2, 1}, {2, 2}}, {{{6, 1}, {7, 1}, {7, 2}}, {{8, 1}, {8.5, 1}, {8.5, 2}}}}",
    )
    .unwrap();
    let faces = split_holes(&arg, &expr_to_point_list).unwrap();
    assert_eq!(faces.len(), 2);
    assert_eq!(faces[0].0[1], (4.0, 0.0));
    assert_eq!(faces[0].1.len(), 1);
    assert_eq!(faces[1].0[0], (5.0, 0.0));
    assert_eq!(faces[1].1.len(), 2);
    // Mismatched outer/holes counts are not a valid spec.
    let bad = crate::interpret_to_expr(
      "{{{0, 0}, {4, 0}, {4, 4}}} -> {{{1, 1}, {2, 1}, {2, 2}}, {{1, 1}, {2, 1}, {2, 2}}}",
    )
    .unwrap();
    assert!(split_holes(&bad, &expr_to_point_list).is_none());
  }

  #[test]
  fn even_odd_regions_of_a_pentagram_are_its_tips() {
    let pts = pentagram();
    let (verts, regions) = even_odd_regions(&pts, &[whole(&pts)]).unwrap();
    // Five corners plus the five points where the chords cross.
    assert_eq!(verts.len(), 10);
    // The five tips are kept; the pentagon in the middle, covered twice,
    // is not.
    assert_eq!(regions.len(), 5);
    assert!(
      regions
        .iter()
        .all(|r| r.outer.len() == 3 && r.holes.is_empty())
    );
    let center_hit = regions.iter().any(|r| {
      let flat: Vec<(f64, f64)> = r.outer.iter().map(|&k| verts[k].0).collect();
      point_in_triangle((0.0, 0.0), flat[0], flat[1], flat[2])
    });
    assert!(!center_hit);
    // Every crossing point lies on the edge it reports, where it says.
    for (p, at) in &verts {
      let (a, b) = (pts[at.from], pts[at.to]);
      let q = (a.0 + at.t * (b.0 - a.0), a.1 + at.t * (b.1 - a.1));
      assert!((p.0 - q.0).abs() < 1e-12 && (p.1 - q.1).abs() < 1e-12);
    }
  }

  #[test]
  fn even_odd_regions_of_a_bow_tie_are_its_two_triangles() {
    let pts = vec![(0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)];
    let (_, regions) = even_odd_regions(&pts, &[whole(&pts)]).unwrap();
    assert_eq!(regions.len(), 2);
  }

  /// Twice around from the same corner: a square, then a smaller square
  /// inside it that starts where the first one did.
  fn nested_loops_from_one_corner() -> Vec<(f64, f64)> {
    vec![
      (0.0, 0.0),
      (4.0, 0.0),
      (4.0, 4.0),
      (0.0, 4.0),
      (0.0, 0.0),
      (3.0, 1.0),
      (3.0, 3.0),
      (1.0, 3.0),
    ]
  }

  #[test]
  fn even_odd_regions_leave_a_loop_nested_at_a_shared_corner_empty() {
    let pts = nested_loops_from_one_corner();
    let (verts, regions) = even_odd_regions(&pts, &[whole(&pts)]).unwrap();
    // Nothing crosses; the revisited corner is the only junction.
    assert_eq!(verts.len(), pts.len());
    assert_eq!(regions.len(), 1);
    let flat: Vec<(f64, f64)> =
      regions[0].outer.iter().map(|&k| verts[k].0).collect();
    let all: Vec<usize> = (0..flat.len()).collect();
    // The big square less the inner loop (area 16 - 6), wound once.
    assert!((signed_area(&flat, &all) - 10.0).abs() < 1e-9);
    // Triangulated, the region covers that area and not the inner loop.
    let pts2: Vec<(f64, f64)> = verts.iter().map(|(p, _)| *p).collect();
    let tris = triangulate_with_holes(&pts2, &regions[0].outer, &[]);
    assert!((area(&pts2, &tris) - 10.0).abs() < 1e-9);
    assert!(!tris.triangles.iter().any(|t| point_in_triangle(
      (2.2, 2.0),
      pts2[t[0]],
      pts2[t[1]],
      pts2[t[2]]
    )));
  }

  #[test]
  fn even_odd_regions_keep_a_crescent_pinched_at_the_shared_corner() {
    // A circle, then a smaller circle tangent inside it at the corner
    // both start from: the band between them narrows to nothing there,
    // so its sides next to that corner leave no room for a probe.
    for (n, r) in [(48, 5.0 / 6.0), (30, 0.5), (64, 0.9)] {
      let ring = |r: f64| {
        (0..n).map(move |k| {
          let a = std::f64::consts::TAU * k as f64 / n as f64;
          (r * a.sin(), 1.0 - r + r * a.cos())
        })
      };
      let pts: Vec<(f64, f64)> = ring(1.0).chain(ring(r)).collect();
      let (verts, regions) = even_odd_regions(&pts, &[whole(&pts)]).unwrap();
      let pts2: Vec<(f64, f64)> = verts.iter().map(|(p, _)| *p).collect();
      let filled: f64 = regions
        .iter()
        .map(|reg| {
          area(&pts2, &triangulate_with_holes(&pts2, &reg.outer, &reg.holes))
        })
        .sum();
      let all = |k: usize| (0..k).collect::<Vec<_>>();
      let outer = signed_area(&pts[..n], &all(n)).abs();
      let inner = signed_area(&pts[n..], &all(n)).abs();
      assert!(
        (filled - (outer - inner)).abs() < 1e-3 * outer,
        "n={n} r={r}: filled {filled}, band {}",
        outer - inner
      );
    }
  }

  #[test]
  fn even_odd_regions_ignore_doubled_and_closing_corners() {
    let doubled = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
    assert!(even_odd_regions(&doubled, &[whole(&doubled)]).is_none());
    let closed = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
    assert!(even_odd_regions(&closed, &[whole(&closed)]).is_none());
  }

  #[test]
  fn even_odd_regions_skip_simple_polygons() {
    let square = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    assert!(even_odd_regions(&square, &[whole(&square)]).is_none());
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
    assert!(even_odd_regions(&notch, &[whole(&notch)]).is_none());
  }

  /// Total area the even-odd regions fill, with each region triangulated
  /// around the pieces lying inside it.
  fn even_odd_area(pts: &[(f64, f64)], rings: &[Vec<usize>]) -> Option<f64> {
    let (verts, regions) = even_odd_regions(pts, rings)?;
    let flat: Vec<(f64, f64)> = verts.iter().map(|(p, _)| *p).collect();
    Some(
      regions
        .iter()
        .map(|r| {
          area(&flat, &triangulate_with_holes(&flat, &r.outer, &r.holes))
        })
        .sum(),
    )
  }

  #[test]
  fn even_odd_regions_of_a_hole_reaching_past_its_outline() {
    // A 4x4 square whose 2x2 hole sticks 1 unit out of its right side:
    // the square keeps 16 - 2 = 14, and the protruding half of the hole
    // (covered once) is filled as well.
    let pts = vec![
      (0.0, 0.0),
      (4.0, 0.0),
      (4.0, 4.0),
      (0.0, 4.0),
      (3.0, 1.0),
      (5.0, 1.0),
      (5.0, 3.0),
      (3.0, 3.0),
    ];
    let total =
      even_odd_area(&pts, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]]).unwrap();
    assert!((total - 16.0).abs() < 1e-9, "{total}");
    // A closed ring (repeating its first corner) gives the same result.
    let total =
      even_odd_area(&pts, &[vec![0, 1, 2, 3, 0], vec![4, 5, 6, 7, 4]]).unwrap();
    assert!((total - 16.0).abs() < 1e-9, "{total}");
  }

  #[test]
  fn even_odd_regions_keep_a_separate_hole_cut_out() {
    // One hole crosses the square's right side; a second one lies wholly
    // inside the square and must stay cut out of the region around it.
    let pts = vec![
      (0.0, 0.0),
      (10.0, 0.0),
      (10.0, 10.0),
      (0.0, 10.0),
      (9.0, 4.0),
      (11.0, 4.0),
      (11.0, 6.0),
      (9.0, 6.0),
      (2.0, 2.0),
      (4.0, 2.0),
      (4.0, 4.0),
      (2.0, 4.0),
    ];
    let rings = [vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
    let total = even_odd_area(&pts, &rings).unwrap();
    // 100 - 2 (inner half of the crossing hole) + 2 (its outer half)
    // - 4 (the separate hole).
    assert!((total - 96.0).abs() < 1e-9, "{total}");
  }

  #[test]
  fn even_odd_regions_skip_holes_inside_their_outline() {
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
    assert!(
      even_odd_regions(&pts, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]]).is_none()
    );
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
