//! Boolean operations on planar regions bounded by polygonal rings — the
//! geometry behind `RegionIntersection`, `RegionUnion`, `RegionDifference`
//! and `RegionSymmetricDifference` of 2-D mesh regions.
//!
//! A region is a set of closed rings filled by the even-odd rule. Each ring
//! is first oriented so the region lies on its left (outer boundaries
//! counter-clockwise, holes clockwise). The edges of both operands are split
//! at every point where they meet, each piece is classified as inside,
//! outside or on the boundary of the other operand, the pieces the operation
//! keeps are chosen, and those are linked back into rings.

pub type P = (f64, f64);
pub type Ring = Vec<P>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
  Intersection,
  Union,
  Difference,
}

pub fn signed_area(ring: &[P]) -> f64 {
  let n = ring.len();
  (0..n)
    .map(|i| {
      let (a, b) = (ring[i], ring[(i + 1) % n]);
      a.0 * b.1 - b.0 * a.1
    })
    .sum::<f64>()
    / 2.0
}

fn cross(o: P, a: P, b: P) -> f64 {
  (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Whether `p` lies inside the even-odd fill of `rings`.
pub fn inside(rings: &[Ring], p: P) -> bool {
  let mut crossings = 0usize;
  for ring in rings {
    let n = ring.len();
    for i in 0..n {
      let (a, b) = (ring[i], ring[(i + 1) % n]);
      if (a.1 > p.1) != (b.1 > p.1)
        && p.0 < a.0 + (p.1 - a.1) / (b.1 - a.1) * (b.0 - a.0)
      {
        crossings += 1;
      }
    }
  }
  crossings % 2 == 1
}

fn dist_to_segment(p: P, a: P, b: P) -> f64 {
  let (dx, dy) = (b.0 - a.0, b.1 - a.1);
  let len2 = dx * dx + dy * dy;
  let t = if len2 == 0.0 {
    0.0
  } else {
    (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
  };
  (p.0 - a.0 - t * dx).hypot(p.1 - a.1 - t * dy)
}

/// A point strictly inside the polygon `ring` near its first edge, used to
/// decide how deeply the ring is nested among the others.
fn probe_point(ring: &[P]) -> P {
  let (a, b) = (ring[0], ring[1 % ring.len()]);
  let mid = (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1));
  let len = (b.0 - a.0).hypot(b.1 - a.1).max(f64::MIN_POSITIVE);
  // The left normal points into a counter-clockwise ring.
  let sign = if signed_area(ring) >= 0.0 { 1.0 } else { -1.0 };
  let step = len * 1e-6;
  (
    mid.0 - sign * (b.1 - a.1) / len * step,
    mid.1 + sign * (b.0 - a.0) / len * step,
  )
}

/// The rings oriented so the region lies on their left: a ring nested in an
/// even number of the others bounds the region from outside
/// (counter-clockwise), one nested in an odd number is a hole (clockwise).
pub fn orient(rings: &[Ring]) -> Vec<Ring> {
  let rings: Vec<Ring> = rings
    .iter()
    .map(|r| dedupe(r))
    .filter(|r| r.len() >= 3 && signed_area(r).abs() > 0.0)
    .collect();
  (0..rings.len())
    .map(|i| {
      let probe = probe_point(&rings[i]);
      let depth = (0..rings.len())
        .filter(|&j| j != i && inside(std::slice::from_ref(&rings[j]), probe))
        .count();
      let ccw = signed_area(&rings[i]) > 0.0;
      let mut ring = rings[i].clone();
      if ccw != depth.is_multiple_of(2) {
        ring.reverse();
      }
      ring
    })
    .collect()
}

/// Drop consecutive repeated points (and a closing repeat of the first).
fn dedupe(ring: &[P]) -> Ring {
  let mut out: Ring = Vec::with_capacity(ring.len());
  for &p in ring {
    if out.last() != Some(&p) {
      out.push(p);
    }
  }
  while out.len() > 1 && out.first() == out.last() {
    out.pop();
  }
  out
}

/// Vertices identified up to a tolerance, so the point where two edges
/// cross is one vertex of both.
struct Vertices {
  points: Vec<P>,
  grid: std::collections::HashMap<(i64, i64), Vec<usize>>,
  eps: f64,
}

impl Vertices {
  fn new(eps: f64) -> Self {
    Vertices {
      points: Vec::new(),
      grid: std::collections::HashMap::new(),
      eps,
    }
  }

  fn cell(&self, p: P) -> (i64, i64) {
    (
      (p.0 / (4.0 * self.eps)).floor() as i64,
      (p.1 / (4.0 * self.eps)).floor() as i64,
    )
  }

  fn id(&mut self, p: P) -> usize {
    let (cx, cy) = self.cell(p);
    for dx in -1..=1 {
      for dy in -1..=1 {
        if let Some(ids) = self.grid.get(&(cx + dx, cy + dy)) {
          for &i in ids {
            let q = self.points[i];
            if (q.0 - p.0).abs() <= self.eps && (q.1 - p.1).abs() <= self.eps {
              return i;
            }
          }
        }
      }
    }
    let i = self.points.len();
    self.points.push(p);
    self.grid.entry((cx, cy)).or_default().push(i);
    i
  }
}

/// The edges of both operands, split where they meet, as vertex-id pairs.
fn split_edges(
  a: &[Ring],
  b: &[Ring],
  verts: &mut Vertices,
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
  let edges = |rings: &[Ring]| -> Vec<(P, P)> {
    rings
      .iter()
      .flat_map(|r| (0..r.len()).map(move |i| (r[i], r[(i + 1) % r.len()])))
      .collect()
  };
  let (ea, eb) = (edges(a), edges(b));
  // Split points of each edge: (parameter along the edge, vertex id).
  let mut splits_a: Vec<Vec<(f64, usize)>> = ea
    .iter()
    .map(|&(p, q)| vec![(0.0, verts.id(p)), (1.0, verts.id(q))])
    .collect();
  let mut splits_b: Vec<Vec<(f64, usize)>> = eb
    .iter()
    .map(|&(p, q)| vec![(0.0, verts.id(p)), (1.0, verts.id(q))])
    .collect();
  let eps = verts.eps;
  let param = |p: P, q: P, x: P| -> f64 {
    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
    ((x.0 - p.0) * dx + (x.1 - p.1) * dy) / (dx * dx + dy * dy)
  };
  for (i, &(p1, q1)) in ea.iter().enumerate() {
    let (min1x, max1x) = (p1.0.min(q1.0) - eps, p1.0.max(q1.0) + eps);
    let (min1y, max1y) = (p1.1.min(q1.1) - eps, p1.1.max(q1.1) + eps);
    for (j, &(p2, q2)) in eb.iter().enumerate() {
      if p2.0.max(q2.0) < min1x
        || p2.0.min(q2.0) > max1x
        || p2.1.max(q2.1) < min1y
        || p2.1.min(q2.1) > max1y
      {
        continue;
      }
      // An endpoint of one edge on the other splits it there.
      let mut touch = |x: P, onto_a: bool| {
        let (p, q) = if onto_a { (p1, q1) } else { (p2, q2) };
        if dist_to_segment(x, p, q) <= eps {
          let t = param(p, q, x);
          if t > 0.0 && t < 1.0 {
            let id = verts.id(x);
            if onto_a {
              splits_a[i].push((t, id));
            } else {
              splits_b[j].push((t, id));
            }
          }
        }
      };
      touch(p2, true);
      touch(q2, true);
      touch(p1, false);
      touch(q1, false);
      // A proper crossing.
      let d1 = (q1.0 - p1.0, q1.1 - p1.1);
      let d2 = (q2.0 - p2.0, q2.1 - p2.1);
      let denom = d1.0 * d2.1 - d1.1 * d2.0;
      let scale = d1.0.hypot(d1.1) * d2.0.hypot(d2.1);
      if denom.abs() <= 1e-14 * scale {
        continue;
      }
      let w = (p2.0 - p1.0, p2.1 - p1.1);
      let t = (w.0 * d2.1 - w.1 * d2.0) / denom;
      let u = (w.0 * d1.1 - w.1 * d1.0) / denom;
      if t > 0.0 && t < 1.0 && u > 0.0 && u < 1.0 {
        let x = (p1.0 + t * d1.0, p1.1 + t * d1.1);
        let id = verts.id(x);
        splits_a[i].push((t, id));
        splits_b[j].push((u, id));
      }
    }
  }
  let pieces = |splits: Vec<Vec<(f64, usize)>>| -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for mut s in splits {
      s.sort_by(|x, y| x.0.total_cmp(&y.0));
      let mut ids: Vec<usize> = Vec::with_capacity(s.len());
      for (_, id) in s {
        if ids.last() != Some(&id) {
          ids.push(id);
        }
      }
      for w in ids.windows(2) {
        if w[0] != w[1] {
          out.push((w[0], w[1]));
        }
      }
    }
    out
  };
  (pieces(splits_a), pieces(splits_b))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
  Inside,
  Outside,
  /// On the other operand's boundary, running the same way as its edge.
  Shared,
  /// On the other operand's boundary, running against its edge.
  Opposed,
}

fn classify(
  edges: &[(usize, usize)],
  other_edges: &std::collections::HashSet<(usize, usize)>,
  other: &[Ring],
  verts: &Vertices,
) -> Vec<Side> {
  edges
    .iter()
    .map(|&(a, b)| {
      if other_edges.contains(&(a, b)) {
        return Side::Shared;
      }
      if other_edges.contains(&(b, a)) {
        return Side::Opposed;
      }
      let (p, q) = (verts.points[a], verts.points[b]);
      let mid = (f64::midpoint(p.0, q.0), f64::midpoint(p.1, q.1));
      if inside(other, mid) {
        Side::Inside
      } else {
        Side::Outside
      }
    })
    .collect()
}

/// Link directed edges into closed rings. At a vertex with several ways
/// on, the edge turning furthest left is taken, so regions that only touch
/// at a point come out as separate simple rings.
fn link(edges: &[(usize, usize)], verts: &Vertices) -> Vec<Ring> {
  let mut outgoing: std::collections::HashMap<usize, Vec<usize>> =
    std::collections::HashMap::new();
  for (k, &(a, _)) in edges.iter().enumerate() {
    outgoing.entry(a).or_default().push(k);
  }
  let mut used = vec![false; edges.len()];
  let mut rings = Vec::new();
  for start in 0..edges.len() {
    if used[start] {
      continue;
    }
    let mut ring_ids = vec![edges[start].0];
    let mut current = start;
    used[start] = true;
    let closed = loop {
      let (a, b) = edges[current];
      if b == ring_ids[0] {
        break true;
      }
      ring_ids.push(b);
      let pa = verts.points[a];
      let pb = verts.points[b];
      let back = (pa.0 - pb.0, pa.1 - pb.1);
      let next = outgoing.get(&b).and_then(|cands| {
        cands
          .iter()
          .copied()
          .filter(|&k| !used[k])
          .min_by(|&x, &y| {
            let angle = |k: usize| {
              let q = verts.points[edges[k].1];
              let d = (q.0 - pb.0, q.1 - pb.1);
              // Clockwise angle from the way back to the candidate.
              let ccw = (back.0 * d.1 - back.1 * d.0)
                .atan2(back.0 * d.0 + back.1 * d.1);
              let cw = -ccw;
              if cw <= 0.0 {
                cw + std::f64::consts::TAU
              } else {
                cw
              }
            };
            angle(x).total_cmp(&angle(y))
          })
      });
      match next {
        Some(k) => {
          used[k] = true;
          current = k;
        }
        None => break false,
      }
    };
    if closed && ring_ids.len() >= 3 {
      let ring: Ring = ring_ids.iter().map(|&i| verts.points[i]).collect();
      if signed_area(&ring).abs() > verts.eps * verts.eps {
        rings.push(ring);
      }
    }
  }
  rings
}

/// `op` applied to the regions bounded by `a` and `b`: rings with the
/// region on their left (outer boundaries counter-clockwise, holes
/// clockwise).
pub fn boolean(a: &[Ring], b: &[Ring], op: Op) -> Vec<Ring> {
  let a = orient(a);
  let b = orient(b);
  let scale = a
    .iter()
    .chain(&b)
    .flatten()
    .fold(0.0f64, |m, p| m.max(p.0.abs()).max(p.1.abs()))
    .max(1e-300);
  let mut verts = Vertices::new(scale * 1e-10);
  let (ea, eb) = split_edges(&a, &b, &mut verts);
  let set_a: std::collections::HashSet<(usize, usize)> =
    ea.iter().copied().collect();
  let set_b: std::collections::HashSet<(usize, usize)> =
    eb.iter().copied().collect();
  let side_a = classify(&ea, &set_b, &b, &verts);
  let side_b = classify(&eb, &set_a, &a, &verts);
  let mut kept: Vec<(usize, usize)> = Vec::new();
  for (&e, &s) in ea.iter().zip(&side_a) {
    let keep = match op {
      Op::Intersection => matches!(s, Side::Inside | Side::Shared),
      Op::Union => matches!(s, Side::Outside | Side::Shared),
      Op::Difference => matches!(s, Side::Outside | Side::Opposed),
    };
    if keep {
      kept.push(e);
    }
  }
  for (&(p, q), &s) in eb.iter().zip(&side_b) {
    match (op, s) {
      (Op::Intersection, Side::Inside) | (Op::Union, Side::Outside) => {
        kept.push((p, q));
      }
      (Op::Difference, Side::Inside) => kept.push((q, p)),
      _ => {}
    }
  }
  link(&kept, &verts)
}

/// The rings grouped into polygons: each outer (counter-clockwise) ring with
/// the holes it immediately contains.
pub fn group_holes(rings: &[Ring]) -> Vec<(Ring, Vec<Ring>)> {
  let (outers, holes): (Vec<&Ring>, Vec<&Ring>) =
    rings.iter().partition(|r| signed_area(r) > 0.0);
  let mut groups: Vec<(Ring, Vec<Ring>)> =
    outers.iter().map(|r| ((*r).clone(), Vec::new())).collect();
  for hole in holes {
    let probe = probe_point(hole);
    let owner = groups
      .iter()
      .enumerate()
      .filter(|(_, (outer, _))| inside(std::slice::from_ref(outer), probe))
      .min_by(|x, y| signed_area(&x.1.0).total_cmp(&signed_area(&y.1.0)))
      .map(|(i, _)| i);
    if let Some(i) = owner {
      groups[i].1.push(hole.clone());
    }
  }
  groups
}

/// Whether the edge `c d` gets in the way of a bridge `a b`: it crosses
/// it, or touches it anywhere but at the bridge's own ends.
fn bridge_blocked(a: P, b: P, c: P, d: P) -> bool {
  if segments_cross(a, b, c, d) {
    return true;
  }
  let at_end = |p: P| p == a || p == b;
  let on_bridge = |p: P| {
    !at_end(p) && cross(a, b, p) == 0.0 && {
      let t = (p.0 - a.0) * (b.0 - a.0) + (p.1 - a.1) * (b.1 - a.1);
      let len2 = (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2);
      t > 0.0 && t < len2
    }
  };
  on_bridge(c) || on_bridge(d)
}

fn segments_cross(a: P, b: P, c: P, d: P) -> bool {
  let (d1, d2) = (cross(c, d, a), cross(c, d, b));
  let (d3, d4) = (cross(a, b, c), cross(a, b, d));
  ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
    && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// One ring tracing an outer boundary and its holes (index rings into
/// `pts`), each hole joined to the outside by a doubled bridge, so a
/// simple-polygon triangulation can fill a region with holes.
pub fn bridge_holes(
  pts: &[P],
  outer: &[usize],
  holes: &[Vec<usize>],
) -> Vec<usize> {
  let mut ring: Vec<usize> = outer.to_vec();
  let mut holes: Vec<Vec<usize>> = holes.to_vec();
  let max_x = |r: &Vec<usize>| {
    r.iter()
      .map(|&i| pts[i].0)
      .fold(f64::NEG_INFINITY, f64::max)
  };
  // Rightmost holes first, so each bridge stays clear of the others.
  holes.sort_by(|x, y| max_x(y).total_cmp(&max_x(x)));
  for (h_index, hole) in holes.iter().enumerate() {
    let m = (0..hole.len())
      .max_by(|&i, &j| pts[hole[i]].0.total_cmp(&pts[hole[j]].0))
      .unwrap_or(0);
    let pm = pts[hole[m]];
    let edges_of = |r: &[usize]| -> Vec<(P, P)> {
      (0..r.len())
        .map(|i| (pts[r[i]], pts[r[(i + 1) % r.len()]]))
        .collect()
    };
    let mut obstacles = edges_of(&ring);
    for h in &holes[h_index..] {
      obstacles.extend(edges_of(h));
    }
    let target = (0..ring.len())
      .filter(|&i| {
        let v = pts[ring[i]];
        !obstacles.iter().any(|&(c, d)| bridge_blocked(pm, v, c, d))
      })
      .min_by(|&i, &j| {
        let d = |k: usize| {
          let q = pts[ring[k]];
          (q.0 - pm.0).hypot(q.1 - pm.1)
        };
        d(i).total_cmp(&d(j))
      });
    let Some(v) = target else { continue };
    let mut spliced: Vec<usize> = ring[..=v].to_vec();
    spliced.extend((0..=hole.len()).map(|k| hole[(m + k) % hole.len()]));
    spliced.push(ring[v]);
    spliced.extend_from_slice(&ring[v + 1..]);
    ring = spliced;
  }
  ring
}
