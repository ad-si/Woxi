//! Planar finite elements: `DiscretizeRegion` for 2D regions, the
//! two-dimensional `NDEigensystem` / `NDEigenvalues`, the piecewise-linear
//! `InterpolatingFunction` they return, and the `Element[{x, y}, region]`
//! domain of `Plot3D`, `ContourPlot` and `DensityPlot`.
//!
//! A region is first turned into closed boundary rings (the outer boundary
//! and the holes). A triangle mesh is then built from boundary points spaced
//! evenly along the rings and a hexagonal lattice of interior points, joined
//! by a Delaunay triangulation of which only the triangles inside the region
//! are kept. The Laplace-type operator `a u_xx + b u_yy + c u` is discretized
//! with linear elements and a lumped mass matrix, and the symmetric
//! eigenproblem is solved densely (Householder tridiagonalization followed
//! by implicit QL).

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

type P = (f64, f64);
type Ring = Vec<P>;

/// Tag in the data of a mesh-based `InterpolatingFunction`.
const TAG: &str = "Woxi`FiniteElement2D";

fn ev(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn id(name: &str) -> Expr {
  Expr::Identifier(name.into())
}

fn real_pair(a: f64, b: f64) -> Expr {
  Expr::List(vec![Expr::Real(a), Expr::Real(b)].into())
}

fn point(e: &Expr) -> Option<P> {
  match e {
    Expr::List(v) if v.len() == 2 => {
      Some((try_eval_to_f64(&v[0])?, try_eval_to_f64(&v[1])?))
    }
    _ => None,
  }
}

/// A radius: one number, or the pair of semi-axes of an ellipse.
fn radii(e: &Expr) -> Option<P> {
  match try_eval_to_f64(e) {
    Some(r) => Some((r, r)),
    None => point(e),
  }
}

/// The points of an arc of the (elliptical) circle, `from` to `to`, as
/// `segments + 1` points.
fn arc(c: P, r: P, from: f64, to: f64, segments: usize) -> Ring {
  (0..=segments)
    .map(|i| {
      let t = from + (to - from) * i as f64 / segments as f64;
      (c.0 + r.0 * t.cos(), c.1 + r.1 * t.sin())
    })
    .collect()
}

/// A full circle, without repeating its first point.
fn circle(c: P, r: P) -> Ring {
  const N: usize = 256;
  let mut ring = arc(c, r, 0.0, std::f64::consts::TAU, N);
  ring.pop();
  ring
}

fn arc_segments(span: f64) -> usize {
  ((256.0 * span.abs() / std::f64::consts::TAU).ceil() as usize).max(8)
}

/// The boundary rings (outer boundary first, then holes) of a planar region
/// with numeric data, or `None` for anything else.
pub fn region_rings(region: &Expr) -> Option<Vec<Ring>> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  match (name.as_str(), args.as_slice()) {
    ("Rectangle", []) => Some(vec![rect((0.0, 0.0), (1.0, 1.0))]),
    ("Rectangle", [p]) => {
      let p = point(p)?;
      Some(vec![rect(p, (p.0 + 1.0, p.1 + 1.0))])
    }
    ("Rectangle", [p, q]) => Some(vec![rect(point(p)?, point(q)?)]),
    ("Triangle", []) => Some(vec![vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]]),
    ("Triangle" | "Polygon", [Expr::List(pts)]) if pts.len() >= 3 => {
      let ring: Option<Ring> = pts.iter().map(point).collect();
      let ring = ring?;
      (name != "Triangle" || ring.len() == 3).then(|| vec![ring])
    }
    ("RegularPolygon", [n]) => regular_polygon((0.0, 0.0), 1.0, n),
    ("RegularPolygon", [r, n]) => {
      regular_polygon((0.0, 0.0), try_eval_to_f64(r)?, n)
    }
    ("RegularPolygon", [c, r, n]) => {
      regular_polygon(point(c)?, try_eval_to_f64(r)?, n)
    }
    ("Disk", []) => Some(vec![circle((0.0, 0.0), (1.0, 1.0))]),
    ("Disk", [c]) => Some(vec![circle(point(c)?, (1.0, 1.0))]),
    ("Disk", [c, r]) => Some(vec![circle(point(c)?, radii(r)?)]),
    ("Disk", [c, r, Expr::List(angles)]) if angles.len() == 2 => {
      let (a, b) = (try_eval_to_f64(&angles[0])?, try_eval_to_f64(&angles[1])?);
      let c = point(c)?;
      let mut ring = arc(c, radii(r)?, a, b, arc_segments(b - a));
      ring.push(c);
      Some(vec![ring])
    }
    ("Annulus", []) => Some(vec![
      circle((0.0, 0.0), (1.0, 1.0)),
      circle((0.0, 0.0), (0.5, 0.5)),
    ]),
    ("Annulus", [c]) => {
      let c = point(c)?;
      Some(vec![circle(c, (1.0, 1.0)), circle(c, (0.5, 0.5))])
    }
    ("Annulus", [c, Expr::List(r)]) if r.len() == 2 => {
      let c = point(c)?;
      let (r1, r2) = (try_eval_to_f64(&r[0])?, try_eval_to_f64(&r[1])?);
      let (inner, outer) = (r1.min(r2), r1.max(r2));
      Some(vec![circle(c, (outer, outer)), circle(c, (inner, inner))])
    }
    ("DiskSegment", [c, r, Expr::List(angles)]) if angles.len() == 2 => {
      let (a, b) = (try_eval_to_f64(&angles[0])?, try_eval_to_f64(&angles[1])?);
      Some(vec![arc(point(c)?, radii(r)?, a, b, arc_segments(b - a))])
    }
    ("Parallelepiped", [o, Expr::List(v)]) if v.len() == 2 => {
      let (o, v1, v2) = (point(o)?, point(&v[0])?, point(&v[1])?);
      Some(vec![vec![
        o,
        (o.0 + v1.0, o.1 + v1.1),
        (o.0 + v1.0 + v2.0, o.1 + v1.1 + v2.1),
        (o.0 + v2.0, o.1 + v2.1),
      ]])
    }
    ("ImplicitRegion", [condition, Expr::List(vars)]) if vars.len() == 2 => {
      linear_implicit_polygon(condition, &vars[0], &vars[1]).map(|r| vec![r])
    }
    _ => None,
  }
}

/// The bounded polygon an `ImplicitRegion` of linear inequalities in two
/// variables describes (`0 <= -y <= 1 && 0 <= -x + y <= 1`): the half-planes
/// clip a large square one by one. None for any other condition, or an
/// unbounded or empty region.
fn linear_implicit_polygon(
  condition: &Expr,
  x: &Expr,
  y: &Expr,
) -> Option<Ring> {
  // Each constraint as `a x + b y + c <= 0`.
  fn collect(e: &Expr, out: &mut Vec<(Expr, Expr, bool)>) -> Option<()> {
    match e {
      Expr::FunctionCall { name, args } if name == "And" => {
        for a in args {
          collect(a, out)?;
        }
        Some(())
      }
      Expr::BinaryOp {
        op: crate::syntax::BinaryOperator::And,
        left,
        right,
      } => {
        collect(left, out)?;
        collect(right, out)
      }
      Expr::Comparison {
        operands,
        operators,
      } => {
        for (i, op) in operators.iter().enumerate() {
          let (l, r) = (operands[i].clone(), operands[i + 1].clone());
          match op {
            crate::syntax::ComparisonOp::LessEqual
            | crate::syntax::ComparisonOp::Less => out.push((l, r, true)),
            crate::syntax::ComparisonOp::GreaterEqual
            | crate::syntax::ComparisonOp::Greater => {
              out.push((r, l, true));
            }
            _ => return None,
          }
        }
        Some(())
      }
      _ => None,
    }
  }
  let mut constraints = Vec::new();
  collect(condition, &mut constraints)?;
  let (Expr::Identifier(xn), Expr::Identifier(yn)) = (x, y) else {
    return None;
  };
  let value_at = |e: &Expr, px: f64, py: f64| -> Option<f64> {
    let s = crate::syntax::substitute_variable(e, xn, &Expr::Real(px));
    let s = crate::syntax::substitute_variable(&s, yn, &Expr::Real(py));
    try_eval_to_f64(&crate::evaluator::evaluate_expr_to_expr(&s).ok()?)
  };
  let mut planes: Vec<(f64, f64, f64)> = Vec::new();
  for (l, r, _) in &constraints {
    let f =
      |px: f64, py: f64| Some(value_at(l, px, py)? - value_at(r, px, py)?);
    let c = f(0.0, 0.0)?;
    let a = f(1.0, 0.0)? - c;
    let b = f(0.0, 1.0)? - c;
    // Linear: the plane through three points predicts a fourth.
    let check = f(2.0, -3.0)?;
    if (check - (2.0 * a - 3.0 * b + c)).abs() > 1e-9 * (1.0 + check.abs()) {
      return None;
    }
    planes.push((a, b, c));
  }
  const BIG: f64 = 1e6;
  let mut ring: Ring = vec![(-BIG, -BIG), (BIG, -BIG), (BIG, BIG), (-BIG, BIG)];
  for (a, b, c) in planes {
    let side = |p: P| a * p.0 + b * p.1 + c;
    let mut clipped: Ring = Vec::new();
    for i in 0..ring.len() {
      let (p, q) = (ring[i], ring[(i + 1) % ring.len()]);
      let (sp, sq) = (side(p), side(q));
      if sp <= 0.0 {
        clipped.push(p);
      }
      if (sp < 0.0 && sq > 0.0) || (sp > 0.0 && sq < 0.0) {
        let t = sp / (sp - sq);
        clipped.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
      }
    }
    ring = clipped;
    if ring.len() < 3 {
      return None;
    }
  }
  // A vertex left on the big square means the region is unbounded.
  if ring
    .iter()
    .any(|p| p.0.abs() >= BIG * 0.5 || p.1.abs() >= BIG * 0.5)
  {
    return None;
  }
  Some(ring)
}

fn rect(p: P, q: P) -> Ring {
  let (x0, x1) = (p.0.min(q.0), p.0.max(q.0));
  let (y0, y1) = (p.1.min(q.1), p.1.max(q.1));
  vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

fn regular_polygon(c: P, r: f64, n: &Expr) -> Option<Vec<Ring>> {
  let Expr::Integer(n) = n else { return None };
  if *n < 3 {
    return None;
  }
  let half_pi = std::f64::consts::FRAC_PI_2;
  Some(vec![
    (0..*n)
      .map(|k| {
        let t = half_pi + std::f64::consts::TAU * k as f64 / *n as f64;
        (c.0 + r * t.cos(), c.1 + r * t.sin())
      })
      .collect(),
  ])
}

/// Even-odd point-in-region test over all rings.
fn inside(rings: &[Ring], p: P) -> bool {
  let mut crossings = 0;
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

fn ring_area(ring: &Ring) -> f64 {
  let n = ring.len();
  (0..n)
    .map(|i| {
      let (a, b) = (ring[i], ring[(i + 1) % n]);
      a.0 * b.1 - b.0 * a.1
    })
    .sum::<f64>()
    .abs()
    / 2.0
}

fn dist_to_segment(p: P, a: P, b: P) -> f64 {
  let (dx, dy) = (b.0 - a.0, b.1 - a.1);
  let len2 = dx * dx + dy * dy;
  let t = if len2 == 0.0 {
    0.0
  } else {
    (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
  };
  ((p.0 - a.0 - t * dx).powi(2) + (p.1 - a.1 - t * dy).powi(2)).sqrt()
}

/// Points spaced about `h` apart along a closed ring, keeping its corners.
fn resample_ring(ring: &Ring, h: f64) -> Ring {
  let n = ring.len();
  let direction = |i: usize| {
    let (a, b) = (ring[i], ring[(i + 1) % n]);
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = dx.hypot(dy);
    (
      dx / len.max(f64::MIN_POSITIVE),
      dy / len.max(f64::MIN_POSITIVE),
    )
  };
  let mut corners: Vec<usize> = (0..n)
    .filter(|&i| {
      let (u, v) = (direction((i + n - 1) % n), direction(i));
      u.0 * v.0 + u.1 * v.1 < 0.9
    })
    .collect();
  if corners.is_empty() {
    corners.push(0);
  }
  let mut out = Ring::new();
  for (k, &start) in corners.iter().enumerate() {
    let end = corners[(k + 1) % corners.len()];
    let mut chain = vec![ring[start]];
    let mut i = start;
    loop {
      i = (i + 1) % n;
      chain.push(ring[i]);
      if i == end {
        break;
      }
    }
    let lengths: Vec<f64> = chain
      .windows(2)
      .map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1))
      .collect();
    let total: f64 = lengths.iter().sum();
    let pieces = ((total / h).round() as usize).max(1);
    for s in 0..pieces {
      let target = total * s as f64 / pieces as f64;
      let mut walked = 0.0;
      for (j, len) in lengths.iter().enumerate() {
        if walked + len >= target || j == lengths.len() - 1 {
          let t = if *len > 0.0 {
            ((target - walked) / len).clamp(0.0, 1.0)
          } else {
            0.0
          };
          out.push((
            chain[j].0 + t * (chain[j + 1].0 - chain[j].0),
            chain[j].1 + t * (chain[j + 1].1 - chain[j].1),
          ));
          break;
        }
        walked += len;
      }
    }
  }
  out
}

/// A triangle mesh.
pub struct TriMesh {
  pub nodes: Vec<P>,
  pub triangles: Vec<[usize; 3]>,
}

/// A mesh of about `target` nodes for the region bounded by `rings`.
pub fn mesh_from_rings(rings: &[Ring], target: usize) -> Option<TriMesh> {
  let area =
    ring_area(rings.first()?) - rings[1..].iter().map(ring_area).sum::<f64>();
  if area <= 0.0 {
    return None;
  }
  let h = (area / (0.866 * target as f64)).sqrt();
  let boundary: Vec<Ring> = rings.iter().map(|r| resample_ring(r, h)).collect();
  let mut nodes: Vec<P> = boundary.iter().flatten().copied().collect();
  let (mut x0, mut y0, mut x1, mut y1) = (
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
  );
  for &(x, y) in &nodes {
    x0 = x0.min(x);
    y0 = y0.min(y);
    x1 = x1.max(x);
    y1 = y1.max(y);
  }
  let segments: Vec<(P, P)> = boundary
    .iter()
    .flat_map(|r| (0..r.len()).map(move |i| (r[i], r[(i + 1) % r.len()])))
    .collect();
  let row_height = h * 3f64.sqrt() / 2.0;
  let mut row = 0usize;
  let mut y = y0 + row_height;
  while y < y1 {
    let mut x = x0 + if row.is_multiple_of(2) { h / 2.0 } else { h };
    while x < x1 {
      let p = (x, y);
      if inside(rings, p)
        && segments
          .iter()
          .all(|(a, b)| dist_to_segment(p, *a, *b) >= 0.75 * h)
      {
        nodes.push(p);
      }
      x += h;
    }
    y += row_height;
    row += 1;
  }
  let points: Vec<[f64; 2]> = nodes.iter().map(|p| [p.0, p.1]).collect();
  let triangles = crate::functions::delaunay::triangulate(&points)?;
  let kept: Vec<[usize; 3]> = triangles
    .into_iter()
    .filter(|t| {
      let (a, b, c) = (nodes[t[0]], nodes[t[1]], nodes[t[2]]);
      let twice_area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
      twice_area.abs() > 1e-12 * h * h
        && inside(rings, ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0))
    })
    .collect();
  if kept.is_empty() {
    return None;
  }
  // Drop nodes no triangle uses.
  let mut index = vec![usize::MAX; nodes.len()];
  let mut used = Vec::new();
  let triangles = kept
    .iter()
    .map(|t| {
      t.map(|v| {
        if index[v] == usize::MAX {
          index[v] = used.len();
          used.push(nodes[v]);
        }
        index[v]
      })
    })
    .collect();
  Some(TriMesh {
    nodes: used,
    triangles,
  })
}

/// The coordinates and triangles of a `MeshRegion` made of triangles.
fn mesh_from_mesh_region(region: &Expr) -> Option<TriMesh> {
  let mesh = crate::functions::mesh_region::parse_mesh(region)?;
  if mesh.boundary || mesh.cells[2].is_empty() {
    return None;
  }
  let nodes: Option<Vec<P>> = mesh.coordinates.iter().map(point).collect();
  let triangles: Option<Vec<[usize; 3]>> = mesh.cells[2]
    .iter()
    .map(|c| match c.as_slice() {
      [a, b, c] => Some([a - 1, b - 1, c - 1]),
      _ => None,
    })
    .collect();
  let (nodes, triangles) = (nodes?, triangles?);
  triangles
    .iter()
    .all(|t| t.iter().all(|&v| v < nodes.len()))
    .then_some(TriMesh { nodes, triangles })
}

fn mesh_expr(mesh: &TriMesh) -> Expr {
  let coordinates = Expr::List(
    mesh
      .nodes
      .iter()
      .map(|p| real_pair(p.0, p.1))
      .collect::<Vec<_>>()
      .into(),
  );
  let cells = Expr::List(
    mesh
      .triangles
      .iter()
      .map(|t| {
        Expr::List(
          t.iter()
            .map(|v| Expr::Integer(*v as i128 + 1))
            .collect::<Vec<_>>()
            .into(),
        )
      })
      .collect::<Vec<_>>()
      .into(),
  );
  call(
    "MeshRegion",
    vec![
      coordinates,
      Expr::List(vec![call("Triangle", vec![cells])].into()),
    ],
  )
}

/// Nodes in a mesh region meant for display.
const DISPLAY_NODES: usize = 400;
/// Nodes in a mesh the eigenproblem is solved on.
const SOLVE_NODES: usize = 800;

/// `DiscretizeRegion[region]`: a triangle `MeshRegion` for a planar region.
pub fn discretize_region_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || crate::helpers::unevaluated("DiscretizeRegion", args);
  if args.is_empty() {
    return Ok(original());
  }
  if crate::functions::mesh_region::is_mesh_region(&args[0]) {
    return Ok(args[0].clone());
  }
  match region_rings(&args[0])
    .and_then(|rings| mesh_from_rings(&rings, DISPLAY_NODES))
  {
    Some(mesh) => Ok(mesh_expr(&mesh)),
    None => Ok(original()),
  }
}

/// The mesh for a region argument: a triangle `MeshRegion` as it is, any
/// other planar region discretized finely.
fn solver_mesh(region: &Expr) -> Option<TriMesh> {
  mesh_from_mesh_region(region).or_else(|| {
    region_rings(region).and_then(|rings| mesh_from_rings(&rings, SOLVE_NODES))
  })
}

// ─── Eigenproblem ───────────────────────────────────────────────────────

/// Householder reduction of the symmetric matrix `v` (row-major, `n × n`)
/// to tridiagonal form: diagonal `d`, subdiagonal `e`, and the orthogonal
/// transform accumulated in `v`.
fn tridiagonalize(v: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
  let at = |i: usize, j: usize| i * n + j;
  for j in 0..n {
    d[j] = v[at(n - 1, j)];
  }
  for i in (1..n).rev() {
    let mut scale = 0.0;
    let mut h = 0.0;
    for dk in d.iter().take(i) {
      scale += dk.abs();
    }
    if scale == 0.0 {
      e[i] = d[i - 1];
      for j in 0..i {
        d[j] = v[at(i - 1, j)];
        v[at(i, j)] = 0.0;
        v[at(j, i)] = 0.0;
      }
    } else {
      for dk in d.iter_mut().take(i) {
        *dk /= scale;
        h += *dk * *dk;
      }
      let mut f = d[i - 1];
      let mut g = h.sqrt();
      if f > 0.0 {
        g = -g;
      }
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for ej in e.iter_mut().take(i) {
        *ej = 0.0;
      }
      for j in 0..i {
        f = d[j];
        v[at(j, i)] = f;
        g = e[j] + v[at(j, j)] * f;
        for k in j + 1..i {
          g += v[at(k, j)] * d[k];
          e[k] += v[at(k, j)] * f;
        }
        e[j] = g;
      }
      f = 0.0;
      for j in 0..i {
        e[j] /= h;
        f += e[j] * d[j];
      }
      let hh = f / (h + h);
      for j in 0..i {
        e[j] -= hh * d[j];
      }
      for j in 0..i {
        f = d[j];
        g = e[j];
        for k in j..i {
          v[at(k, j)] -= f * e[k] + g * d[k];
        }
        d[j] = v[at(i - 1, j)];
        v[at(i, j)] = 0.0;
      }
    }
    d[i] = h;
  }
  for i in 0..n - 1 {
    v[at(n - 1, i)] = v[at(i, i)];
    v[at(i, i)] = 1.0;
    let h = d[i + 1];
    if h != 0.0 {
      for k in 0..=i {
        d[k] = v[at(k, i + 1)] / h;
      }
      for j in 0..=i {
        let mut g = 0.0;
        for k in 0..=i {
          g += v[at(k, i + 1)] * v[at(k, j)];
        }
        for k in 0..=i {
          v[at(k, j)] -= g * d[k];
        }
      }
    }
    for k in 0..=i {
      v[at(k, i + 1)] = 0.0;
    }
  }
  for j in 0..n {
    d[j] = v[at(n - 1, j)];
    v[at(n - 1, j)] = 0.0;
  }
  v[at(n - 1, n - 1)] = 1.0;
  e[0] = 0.0;
}

/// Implicit QL iteration on the tridiagonal `(d, e)`, rotating the columns
/// of `v` into the eigenvectors; `d` ends up holding the eigenvalues.
fn ql_implicit(v: &mut [f64], n: usize, d: &mut [f64], e: &mut [f64]) {
  for i in 1..n {
    e[i - 1] = e[i];
  }
  e[n - 1] = 0.0;
  let mut f = 0.0;
  let mut tst1: f64 = 0.0;
  let eps = f64::EPSILON;
  for l in 0..n {
    tst1 = tst1.max(d[l].abs() + e[l].abs());
    let mut m = l;
    while m < n - 1 && e[m].abs() > eps * tst1 {
      m += 1;
    }
    if m > l {
      for _ in 0..200 {
        let mut g = d[l];
        let mut p = (d[l + 1] - g) / (2.0 * e[l]);
        let mut r = p.hypot(1.0);
        if p < 0.0 {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        let dl1 = d[l + 1];
        let mut h = g - d[l];
        for di in d.iter_mut().take(n).skip(l + 2) {
          *di -= h;
        }
        f += h;
        p = d[m];
        let (mut c, mut c2, mut c3) = (1.0, 1.0, 1.0);
        let el1 = e[l + 1];
        let (mut s, mut s2) = (0.0, 0.0);
        for i in (l..m).rev() {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = p.hypot(e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          for k in 0..n {
            h = v[k * n + i + 1];
            v[k * n + i + 1] = s * v[k * n + i] + c * h;
            v[k * n + i] = c * v[k * n + i] - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        if e[l].abs() <= eps * tst1 {
          break;
        }
      }
    }
    d[l] += f;
    e[l] = 0.0;
  }
}

/// A planar problem `a u_xx + b u_yy + c(x, y) u = λ u` on a mesh.
struct Problem {
  a: f64,
  b: f64,
  c: Expr,
  x: String,
  y: String,
  mesh: TriMesh,
  /// Whether `u = 0` holds on the whole boundary (otherwise it is natural).
  dirichlet: bool,
}

/// The eigenvalues smallest in magnitude and their nodal eigenfunctions
/// (`∫ u² = 1`, largest entry positive).
fn solve(
  problem: &Problem,
  count: usize,
) -> Result<Option<(Vec<f64>, Vec<Vec<f64>>)>, InterpreterError> {
  let mesh = &problem.mesh;
  let total = mesh.nodes.len();
  // Boundary nodes: those on an edge only one triangle has.
  let mut edge_count = std::collections::HashMap::new();
  for t in &mesh.triangles {
    for k in 0..3 {
      let (i, j) = (t[k], t[(k + 1) % 3]);
      *edge_count.entry((i.min(j), i.max(j))).or_insert(0usize) += 1;
    }
  }
  let mut on_boundary = vec![false; total];
  for ((i, j), c) in &edge_count {
    if *c == 1 {
      on_boundary[*i] = true;
      on_boundary[*j] = true;
    }
  }
  let mut unknown = vec![usize::MAX; total];
  let mut free = Vec::new();
  for i in 0..total {
    if !(problem.dirichlet && on_boundary[i]) {
      unknown[i] = free.len();
      free.push(i);
    }
  }
  let n = free.len();
  if n == 0 {
    return Ok(None);
  }
  let mut stiffness = vec![0.0; n * n];
  let mut mass = vec![0.0; n];
  for t in &mesh.triangles {
    let p = [mesh.nodes[t[0]], mesh.nodes[t[1]], mesh.nodes[t[2]]];
    let signed = (p[1].0 - p[0].0) * (p[2].1 - p[0].1)
      - (p[2].0 - p[0].0) * (p[1].1 - p[0].1);
    let area = signed.abs() / 2.0;
    if area == 0.0 {
      continue;
    }
    let gradient: Vec<P> = (0..3)
      .map(|i| {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        ((p[j].1 - p[k].1) / signed, (p[k].0 - p[j].0) / signed)
      })
      .collect();
    let centroid = (
      (p[0].0 + p[1].0 + p[2].0) / 3.0,
      (p[0].1 + p[1].1 + p[2].1) / 3.0,
    );
    let c = ev(&crate::syntax::substitute_variable(
      &crate::syntax::substitute_variable(
        &problem.c,
        &problem.x,
        &Expr::Real(centroid.0),
      ),
      &problem.y,
      &Expr::Real(centroid.1),
    ))?;
    let Some(c) = try_eval_to_f64(&c) else {
      return Ok(None);
    };
    for i in 0..3 {
      let ni = unknown[t[i]];
      if ni == usize::MAX {
        continue;
      }
      mass[ni] += area / 3.0;
      stiffness[ni * n + ni] += c * area / 3.0;
      for j in 0..3 {
        let nj = unknown[t[j]];
        if nj == usize::MAX {
          continue;
        }
        stiffness[ni * n + nj] -= area
          * (problem.a * gradient[i].0 * gradient[j].0
            + problem.b * gradient[i].1 * gradient[j].1);
      }
    }
  }
  if mass.iter().any(|m| *m <= 0.0) {
    return Ok(None);
  }
  let scale: Vec<f64> = mass.iter().map(|m| m.sqrt()).collect();
  for i in 0..n {
    for j in 0..n {
      stiffness[i * n + j] /= scale[i] * scale[j];
    }
  }
  let mut d = vec![0.0; n];
  let mut e = vec![0.0; n];
  if n == 1 {
    d[0] = stiffness[0];
    stiffness[0] = 1.0;
  } else {
    tridiagonalize(&mut stiffness, n, &mut d, &mut e);
    ql_implicit(&mut stiffness, n, &mut d, &mut e);
  }
  let mut order: Vec<usize> = (0..n).collect();
  order.sort_by(|&i, &j| {
    d[i]
      .abs()
      .total_cmp(&d[j].abs())
      .then(d[i].total_cmp(&d[j]))
  });
  order.truncate(count);
  let mut values = Vec::new();
  let mut functions = Vec::new();
  for &k in &order {
    let mut full = vec![0.0; total];
    for (idx, &node) in free.iter().enumerate() {
      full[node] = stiffness[idx * n + k] / scale[idx];
    }
    let peak = full
      .iter()
      .copied()
      .fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
    if peak < 0.0 {
      for v in &mut full {
        *v = -*v;
      }
    }
    values.push(d[k]);
    functions.push(full);
  }
  Ok(Some((values, functions)))
}

/// `Element[{x, y}, region]`, the variables and the region.
fn element_domain(e: &Expr) -> Option<(String, String, &Expr)> {
  let Expr::FunctionCall { name, args } = e else {
    return None;
  };
  if name != "Element" || args.len() != 2 {
    return None;
  }
  let Expr::List(vars) = &args[0] else {
    return None;
  };
  match vars.as_slice() {
    [Expr::Identifier(x), Expr::Identifier(y)] => {
      Some((x.clone(), y.clone(), &args[1]))
    }
    _ => None,
  }
}

/// Placeholders for `u_xx`, `u_yy` and `u`, and the operator's coefficients
/// on them: `Some((a, b, c))` for `a u_xx + b u_yy + c u`.
fn linear_coefficients(
  op: &Expr,
  u: &str,
) -> Result<Option<(Expr, Expr, Expr)>, InterpreterError> {
  let placeholders = ["Woxi`EigenUxx", "Woxi`EigenUyy", "Woxi`EigenU0"];
  let params = || Expr::List(vec![id("Woxi`EigenX"), id("Woxi`EigenY")].into());
  let to = |p: &str| call("Function", vec![params(), id(p)]);
  let derivative = |i: i128, j: i128| Expr::CurriedCall {
    func: Box::new(call(
      "Derivative",
      vec![Expr::Integer(i), Expr::Integer(j)],
    )),
    args: vec![id(u)],
  };
  let rule = |pattern: Expr, p: &str| Expr::Rule {
    pattern: Box::new(pattern),
    replacement: Box::new(to(p)),
  };
  let replaced = ev(&call(
    "ReplaceAll",
    vec![
      op.clone(),
      Expr::List(
        vec![
          rule(derivative(2, 0), placeholders[0]),
          rule(derivative(0, 2), placeholders[1]),
          rule(id(u), placeholders[2]),
        ]
        .into(),
      ),
    ],
  ))?;
  let coefficient = |p: &str| ev(&call("D", vec![replaced.clone(), id(p)]));
  let (a, b, c) = (
    coefficient(placeholders[0])?,
    coefficient(placeholders[1])?,
    coefficient(placeholders[2])?,
  );
  let times = |k: &Expr, p: &str| call("Times", vec![k.clone(), id(p)]);
  let rest = ev(&call(
    "Expand",
    vec![call(
      "Plus",
      vec![
        replaced.clone(),
        call(
          "Times",
          vec![
            Expr::Integer(-1),
            call(
              "Plus",
              vec![
                times(&a, placeholders[0]),
                times(&b, placeholders[1]),
                times(&c, placeholders[2]),
              ],
            ),
          ],
        ),
      ],
    )],
  ))?;
  let free = |e: &Expr| {
    placeholders
      .iter()
      .all(|p| crate::functions::calculus_ast::is_constant_wrt(e, p))
      && crate::functions::calculus_ast::is_constant_wrt(e, u)
  };
  let zero = try_eval_to_f64(&rest).is_some_and(|v| v == 0.0);
  if !zero || !free(&a) || !free(&b) || !free(&c) {
    return Ok(None);
  }
  Ok(Some((a, b, c)))
}

/// Whether the condition is the homogeneous `DirichletCondition[u[x, y] ==
/// 0, True]` on the whole boundary.
fn is_full_dirichlet(cond: &Expr, u: &str, x: &str, y: &str) -> bool {
  let Expr::FunctionCall { name, args } = cond else {
    return false;
  };
  if name != "DirichletCondition"
    || args.len() != 2
    || !matches!(&args[1], Expr::Identifier(t) if t == "True")
  {
    return false;
  }
  let Expr::Comparison {
    operands,
    operators,
  } = &args[0]
  else {
    return false;
  };
  if operands.len() != 2
    || !matches!(operators[0], crate::syntax::ComparisonOp::Equal)
  {
    return false;
  }
  let is_u = |e: &Expr| {
    matches!(e, Expr::FunctionCall { name, args }
      if name == u
        && matches!(args.as_slice(),
          [Expr::Identifier(a), Expr::Identifier(b)] if a == x && b == y))
  };
  let zero = |e: &Expr| try_eval_to_f64(e).is_some_and(|v| v == 0.0);
  (is_u(&operands[0]) && zero(&operands[1]))
    || (is_u(&operands[1]) && zero(&operands[0]))
}

/// The planar problem in `NDEigensystem[op or {op, conds…}, u[x, y] or u,
/// Element[{x, y}, region], n, …]`, whether `u` was given as `u[x, y]`, and
/// the number of eigenpairs. `None` when the arguments are not of that form.
fn read_problem(
  args: &[Expr],
) -> Result<Option<(Problem, bool, usize)>, InterpreterError> {
  if args.len() < 4 {
    return Ok(None);
  }
  let Some((x, y, region)) = element_domain(&args[2]) else {
    return Ok(None);
  };
  let (u, applied) = match &args[1] {
    Expr::Identifier(u) => (u.clone(), false),
    Expr::FunctionCall { name, args: ua } if ua.len() == 2 => {
      (name.clone(), true)
    }
    _ => return Ok(None),
  };
  let Expr::Integer(count) = &args[3] else {
    return Ok(None);
  };
  if *count < 1 {
    return Ok(None);
  }
  let Some(mesh) = solver_mesh(region) else {
    return Ok(None);
  };
  let (op, conditions): (Expr, Vec<Expr>) = match &args[0] {
    Expr::List(items) if !items.is_empty() => {
      (items[0].clone(), items[1..].to_vec())
    }
    single => (single.clone(), Vec::new()),
  };
  if !conditions.iter().all(|c| is_full_dirichlet(c, &u, &x, &y)) {
    return Ok(None);
  }
  let op = ev(&call("Activate", vec![op]))?;
  let Some((a, b, c)) = linear_coefficients(&op, &u)? else {
    return Ok(None);
  };
  let (Some(a), Some(b)) = (
    try_eval_to_f64(&a).filter(|a| *a != 0.0),
    try_eval_to_f64(&b).filter(|b| *b != 0.0),
  ) else {
    return Ok(None);
  };
  Ok(Some((
    Problem {
      a,
      b,
      c,
      x,
      y,
      mesh,
      dirichlet: !conditions.is_empty(),
    },
    applied,
    *count as usize,
  )))
}

/// The piecewise-linear `InterpolatingFunction` through nodal `values`.
fn interpolating_function(mesh: &TriMesh, values: &[f64]) -> Expr {
  let (mut x0, mut y0, mut x1, mut y1) = (
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
  );
  for &(x, y) in &mesh.nodes {
    x0 = x0.min(x);
    y0 = y0.min(y);
    x1 = x1.max(x);
    y1 = y1.max(y);
  }
  let data = Expr::List(
    vec![
      Expr::String(TAG.to_string()),
      Expr::List(
        mesh
          .nodes
          .iter()
          .zip(values)
          .map(|(p, v)| {
            Expr::List(
              vec![Expr::Real(p.0), Expr::Real(p.1), Expr::Real(*v)].into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::List(
        mesh
          .triangles
          .iter()
          .map(|t| {
            Expr::List(
              t.iter()
                .map(|v| Expr::Integer(*v as i128 + 1))
                .collect::<Vec<_>>()
                .into(),
            )
          })
          .collect::<Vec<_>>()
          .into(),
      ),
    ]
    .into(),
  );
  call(
    "InterpolatingFunction",
    vec![
      Expr::List(vec![real_pair(x0, x1), real_pair(y0, y1)].into()),
      data,
    ],
  )
}

/// `NDEigensystem[…]` on a planar region: `Some({eigenvalues,
/// eigenfunctions})`, or `None` when the arguments are not such a problem.
pub fn nd_eigensystem_2d(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((problem, applied, count)) = read_problem(args)? else {
    return Ok(None);
  };
  let Some((values, functions)) = solve(&problem, count)? else {
    return Ok(None);
  };
  let functions = functions
    .iter()
    .map(|f| {
      let interpolation = interpolating_function(&problem.mesh, f);
      if applied {
        Expr::CurriedCall {
          func: Box::new(interpolation),
          args: vec![id(&problem.x), id(&problem.y)],
        }
      } else {
        interpolation
      }
    })
    .collect::<Vec<_>>();
  Ok(Some(Expr::List(
    vec![
      Expr::List(
        values
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ),
      Expr::List(functions.into()),
    ]
    .into(),
  )))
}

/// `NDEigenvalues[…]` on a planar region.
pub fn nd_eigenvalues_2d(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((problem, _, count)) = read_problem(args)? else {
    return Ok(None);
  };
  let Some((values, _)) = solve(&problem, count)? else {
    return Ok(None);
  };
  Ok(Some(Expr::List(
    values
      .into_iter()
      .map(Expr::Real)
      .collect::<Vec<_>>()
      .into(),
  )))
}

// ─── Evaluating the interpolant ─────────────────────────────────────────

/// Whether the data of an `InterpolatingFunction` is a planar mesh.
pub fn is_mesh_interpolation(func_args: &[Expr]) -> bool {
  matches!(func_args.get(1),
    Some(Expr::List(data))
      if matches!(data.first(), Some(Expr::String(t)) if t == TAG))
}

/// `InterpolatingFunction[domain, mesh data][x, y]`: the linear
/// interpolant in the triangle holding the point, or the call unevaluated
/// for symbolic arguments or points outside the mesh.
pub fn evaluate_mesh_interpolation(func_args: &[Expr], args: &[Expr]) -> Expr {
  let held = || Expr::CurriedCall {
    func: Box::new(call("InterpolatingFunction", func_args.to_vec())),
    args: args.to_vec(),
  };
  let (Some(Expr::List(data)), [x, y]) = (func_args.get(1), args) else {
    return held();
  };
  let (Some(x), Some(y)) = (try_eval_to_f64(x), try_eval_to_f64(y)) else {
    return held();
  };
  let [_, Expr::List(nodes), Expr::List(triangles)] = data.as_slice() else {
    return held();
  };
  let node = |i: usize| -> Option<(f64, f64, f64)> {
    match nodes.get(i)? {
      Expr::List(v) if v.len() == 3 => Some((
        try_eval_to_f64(&v[0])?,
        try_eval_to_f64(&v[1])?,
        try_eval_to_f64(&v[2])?,
      )),
      _ => None,
    }
  };
  const SLACK: f64 = 1e-9;
  for t in triangles {
    let Expr::List(idx) = t else { continue };
    let corner = |k: usize| match idx.get(k) {
      Some(Expr::Integer(i)) if *i >= 1 => node(*i as usize - 1),
      _ => None,
    };
    let (Some(a), Some(b), Some(c)) = (corner(0), corner(1), corner(2)) else {
      continue;
    };
    let det = (b.1 - c.1) * (a.0 - c.0) + (c.0 - b.0) * (a.1 - c.1);
    if det == 0.0 {
      continue;
    }
    let l1 = ((b.1 - c.1) * (x - c.0) + (c.0 - b.0) * (y - c.1)) / det;
    let l2 = ((c.1 - a.1) * (x - c.0) + (a.0 - c.0) * (y - c.1)) / det;
    let l3 = 1.0 - l1 - l2;
    if l1 >= -SLACK && l2 >= -SLACK && l3 >= -SLACK {
      return Expr::Real(l1 * a.2 + l2 * b.2 + l3 * c.2);
    }
  }
  held()
}

// ─── Plotting over a region ─────────────────────────────────────────────

/// `Plot3D[f, Element[{x, y}, region], opts…]` (and `ContourPlot`,
/// `DensityPlot`) in terms of the rectangular ranges the plotters take: the
/// bounding box of the region, with a `RegionFunction` that keeps the region.
pub fn region_domain_args(args: &[Expr]) -> Option<Vec<Expr>> {
  let (x, y, region) = element_domain(args.get(1)?)?;
  // The plotters hold their arguments, so a symbol standing for the region
  // is evaluated here.
  let region = &ev(region).ok()?;
  let has_region_function = args[2..].iter().any(|a| {
    matches!(a, Expr::Rule { pattern, .. }
      if matches!(&**pattern, Expr::Identifier(n) if n == "RegionFunction"))
  });
  if has_region_function {
    return None;
  }
  let points: Vec<P> = match region_rings(region) {
    Some(rings) => rings.into_iter().flatten().collect(),
    None => mesh_from_mesh_region(region)?.nodes,
  };
  let (mut x0, mut y0, mut x1, mut y1) = (
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
  );
  for &(px, py) in &points {
    x0 = x0.min(px);
    y0 = y0.min(py);
    x1 = x1.max(px);
    y1 = y1.max(py);
  }
  let range = |v: &str, lo: f64, hi: f64| {
    Expr::List(vec![id(v), Expr::Real(lo), Expr::Real(hi)].into())
  };
  let member = call(
    "RegionMember",
    vec![
      region.clone(),
      Expr::List(vec![id("Woxi`PlotX"), id("Woxi`PlotY")].into()),
    ],
  );
  let region_function = call(
    "Function",
    vec![
      Expr::List(
        vec![id("Woxi`PlotX"), id("Woxi`PlotY"), id("Woxi`PlotZ")].into(),
      ),
      member,
    ],
  );
  // A body that reaches the variables through a held symbol or a `Part` of
  // a list of eigenfunctions is resolved once instead of at every sample.
  let body = crate::functions::plot::resolve_indirect_plot_body(
    &args[0],
    &[x.as_str(), y.as_str()],
  )
  .unwrap_or_else(|| args[0].clone());
  let mut out = vec![body, range(&x, x0, x1), range(&y, y0, y1)];
  out.extend(args[2..].iter().cloned());
  out.push(Expr::Rule {
    pattern: Box::new(id("RegionFunction")),
    replacement: Box::new(region_function),
  });
  Some(out)
}

/// `RegionMember[region, {x, y}]` for the planar regions above, decided by
/// their boundary rings.
pub fn ring_region_member(region: &Expr, p: P) -> Option<bool> {
  let rings = region_rings(region)?;
  let on_boundary = rings.iter().any(|ring| {
    let n = ring.len();
    (0..n).any(|i| dist_to_segment(p, ring[i], ring[(i + 1) % n]) < 1e-10)
  });
  Some(on_boundary || inside(&rings, p))
}
