//! `DelaunayMesh[{{x, y}, …}]` — the Delaunay triangulation of a 2D point
//! set, as the `MeshRegion[coords, {Polygon[{…}]}]` object the region
//! machinery reads.

use super::*;
use crate::functions::math_ast::try_eval_to_f64;

/// The circumcircle test: whether `d` lies strictly inside the circle through
/// `a`, `b`, `c` (given counter-clockwise).
fn in_circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
  let ax = a[0] - d[0];
  let ay = a[1] - d[1];
  let bx = b[0] - d[0];
  let by = b[1] - d[1];
  let cx = c[0] - d[0];
  let cy = c[1] - d[1];
  let det = (ax * ax + ay * ay) * (bx * cy - by * cx)
    - (bx * bx + by * by) * (ax * cy - ay * cx)
    + (cx * cx + cy * cy) * (ax * by - ay * bx);
  det > 1e-12
}

fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
  (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Bowyer–Watson: insert the points one by one into a super-triangle, then
/// drop every triangle that still touches it. Triangles come out with their
/// vertices counter-clockwise.
pub(crate) fn triangulate(points: &[[f64; 2]]) -> Option<Vec<[usize; 3]>> {
  let n = points.len();
  if n < 3 {
    return None;
  }
  let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
  let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
  for p in points {
    min_x = min_x.min(p[0]);
    min_y = min_y.min(p[1]);
    max_x = max_x.max(p[0]);
    max_y = max_y.max(p[1]);
  }
  let span = (max_x - min_x).max(max_y - min_y).max(1.0) * 1000.0;
  let cx = f64::midpoint(min_x, max_x);
  let cy = f64::midpoint(min_y, max_y);
  // The super-triangle's vertices sit past the end of the point list.
  let mut all: Vec<[f64; 2]> = points.to_vec();
  all.push([cx - span, cy - span]);
  all.push([cx + span, cy - span]);
  all.push([cx, cy + span]);
  let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

  for (i, point) in points.iter().enumerate() {
    // Every triangle whose circumcircle holds the point loses its place.
    let mut cavity: Vec<[usize; 2]> = Vec::new();
    let mut kept: Vec<[usize; 3]> = Vec::new();
    for tri in &triangles {
      if in_circumcircle(all[tri[0]], all[tri[1]], all[tri[2]], *point) {
        for k in 0..3 {
          let edge = [tri[k], tri[(k + 1) % 3]];
          // An edge shared by two removed triangles is interior to the cavity.
          match cavity
            .iter()
            .position(|e| (e[0] == edge[1] && e[1] == edge[0]) || *e == edge)
          {
            Some(pos) => {
              cavity.remove(pos);
            }
            None => cavity.push(edge),
          }
        }
      } else {
        kept.push(*tri);
      }
    }
    triangles = kept;
    for edge in cavity {
      let tri = if orientation(all[edge[0]], all[edge[1]], *point) >= 0.0 {
        [edge[0], edge[1], i]
      } else {
        [edge[1], edge[0], i]
      };
      triangles.push(tri);
    }
  }

  // Drop the triangles that still lean on the super-triangle, and any that
  // came out degenerate.
  let mut out: Vec<[usize; 3]> = triangles
    .into_iter()
    .filter(|t| t.iter().all(|v| *v < n))
    .filter(|t| {
      orientation(points[t[0]], points[t[1]], points[t[2]]).abs() > 1e-12
    })
    .collect();
  if out.is_empty() {
    return None;
  }
  // A stable order: by the vertices each triangle names.
  for tri in &mut out {
    // Rotate so the smallest index leads, keeping the orientation.
    let smallest = (0..3).min_by_key(|k| tri[*k]).unwrap();
    *tri = [
      tri[smallest],
      tri[(smallest + 1) % 3],
      tri[(smallest + 2) % 3],
    ];
  }
  out.sort_unstable();
  Some(out)
}

/// `DelaunayMesh[{{x, y}, …}]`.
pub fn delaunay_mesh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("DelaunayMesh", args);
  if args.len() != 1 {
    return Ok(original());
  }
  let Expr::List(items) = &args[0] else {
    return Ok(original());
  };
  if items.is_empty() {
    return Ok(original());
  }
  let mut points: Vec<[f64; 2]> = Vec::with_capacity(items.len());
  let mut exact = true;
  for item in items {
    let Expr::List(coords) = item else {
      return Ok(original());
    };
    if coords.len() != 2 {
      return Ok(original());
    }
    let mut values = [0.0; 2];
    for (k, c) in coords.iter().enumerate() {
      if matches!(c, Expr::Real(_) | Expr::BigFloat(..)) {
        exact = false;
      }
      match try_eval_to_f64(c) {
        Some(v) => values[k] = v,
        None => return Ok(original()),
      }
    }
    points.push(values);
  }

  let coordinates = Expr::List(items.iter().cloned().collect());
  let mut mesh_args = vec![coordinates];
  // With only three points qhull cannot build its lifted simplex, and
  // wolframscript answers with the points themselves.
  if points.len() <= 3 {
    mesh_args.push(Expr::List(
      vec![call1(
        "Point",
        Expr::List(
          (1..=points.len())
            .map(|i| Expr::List(vec![Expr::Integer(i as i128)].into()))
            .collect(),
        ),
      )]
      .into(),
    ));
  } else {
    let Some(triangles) = triangulate(&points) else {
      return Ok(original());
    };
    mesh_args.push(Expr::List(
      vec![call1(
        "Polygon",
        Expr::List(
          triangles
            .into_iter()
            .map(|t| {
              Expr::List(
                t.iter().map(|v| Expr::Integer(*v as i128 + 1)).collect(),
              )
            })
            .collect(),
        ),
      )]
      .into(),
    ));
  }
  if exact {
    mesh_args.push(rule_expr(id_expr("WorkingPrecision"), id_expr("Infinity")));
  }
  Ok(call("MeshRegion", mesh_args))
}

/// The real 2D points of a list argument, or `None` for anything else.
fn read_points(arg: &Expr) -> Option<Vec<[f64; 2]>> {
  let Expr::List(items) = arg else {
    return None;
  };
  items
    .iter()
    .map(|item| match item {
      Expr::List(c) if c.len() == 2 => {
        Some([try_eval_to_f64(&c[0])?, try_eval_to_f64(&c[1])?])
      }
      _ => None,
    })
    .collect()
}

/// `ComputationalGeometry`DelaunayTriangulation[{{x, y}, …}]` — the legacy
/// package's adjacency-list form: `{i, {neighbours of point i}}` per point.
pub fn delaunay_triangulation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original =
    || unevaluated("ComputationalGeometry`DelaunayTriangulation", args);
  let Some(points) = args.first().and_then(read_points) else {
    return Ok(original());
  };
  let Some(triangles) = triangulate(&points) else {
    return Ok(original());
  };
  let mut neighbours: Vec<std::collections::BTreeSet<usize>> =
    vec![std::collections::BTreeSet::new(); points.len()];
  for t in &triangles {
    for k in 0..3 {
      let (a, b) = (t[k], t[(k + 1) % 3]);
      neighbours[a].insert(b);
      neighbours[b].insert(a);
    }
  }
  Ok(Expr::List(
    neighbours
      .into_iter()
      .enumerate()
      .map(|(i, set)| {
        Expr::List(
          vec![
            Expr::Integer(i as i128 + 1),
            Expr::List(
              set
                .into_iter()
                .map(|j| Expr::Integer(j as i128 + 1))
                .collect(),
            ),
          ]
          .into(),
        )
      })
      .collect(),
  ))
}

fn circumcenter_of(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
  let d =
    2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
  let (a2, b2, c2) = (
    a[0] * a[0] + a[1] * a[1],
    b[0] * b[0] + b[1] * b[1],
    c[0] * c[0] + c[1] * c[1],
  );
  [
    (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
    (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d,
  ]
}

/// Liang–Barsky: the part of the segment `p → q` inside `[x0, x1] × [y0, y1]`.
fn clip_segment(
  p: [f64; 2],
  q: [f64; 2],
  bounds: [f64; 4],
) -> Option<([f64; 2], [f64; 2])> {
  let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
  let d = [q[0] - p[0], q[1] - p[1]];
  let checks = [
    (-d[0], p[0] - bounds[0]),
    (d[0], bounds[1] - p[0]),
    (-d[1], p[1] - bounds[2]),
    (d[1], bounds[3] - p[1]),
  ];
  for (dir, dist) in checks {
    if dir == 0.0 {
      if dist < 0.0 {
        return None;
      }
    } else {
      let t = dist / dir;
      if dir < 0.0 {
        t0 = t0.max(t);
      } else {
        t1 = t1.min(t);
      }
    }
  }
  if t0 >= t1 {
    return None;
  }
  Some((
    [p[0] + t0 * d[0], p[1] + t0 * d[1]],
    [p[0] + t1 * d[0], p[1] + t1 * d[1]],
  ))
}

/// `ComputationalGeometry`DiagramPlot[{{x, y}, …}]` — the Voronoi diagram as
/// graphics: the cell edges as `Line`s (unbounded ones cut at a frame around
/// the sites), the sites as `Point`s, and with `LabelPoints -> True` their
/// 1-based indices as `Text`.
pub fn diagram_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`DiagramPlot", args);
  let Some(points) = args.first().and_then(read_points) else {
    return Ok(original());
  };
  let label_points = args[1..].iter().any(|opt| {
    matches!(opt, Expr::Rule { pattern, replacement }
      if matches!(&**pattern, Expr::Identifier(n) if n == "ComputationalGeometry`LabelPoints" || n == "LabelPoints")
        && matches!(&**replacement, Expr::Identifier(v) if v == "True"))
  });
  let Some(triangles) = triangulate(&points) else {
    return Ok(original());
  };

  let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
  for p in &points {
    for k in 0..2 {
      lo[k] = lo[k].min(p[k]);
      hi[k] = hi[k].max(p[k]);
    }
  }
  let pad = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0) * 0.5;
  let bounds = [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad];
  let far = pad * 1000.0;

  let centers: Vec<[f64; 2]> = triangles
    .iter()
    .map(|t| circumcenter_of(points[t[0]], points[t[1]], points[t[2]]))
    .collect();
  // Each undirected Delaunay edge with the triangles (and opposite vertices)
  // on its sides.
  let mut edges: std::collections::BTreeMap<[usize; 2], Vec<(usize, usize)>> =
    std::collections::BTreeMap::new();
  for (ti, t) in triangles.iter().enumerate() {
    for k in 0..3 {
      let (a, b, opposite) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
      edges
        .entry([a.min(b), a.max(b)])
        .or_default()
        .push((ti, opposite));
    }
  }

  let real = |v: f64| Expr::Real(v);
  let point_expr =
    |p: [f64; 2]| Expr::List(vec![real(p[0]), real(p[1])].into());
  let mut lines = Vec::new();
  for ([a, b], sides) in &edges {
    let segment = match sides.as_slice() {
      [(t1, _), (t2, _)] => Some((centers[*t1], centers[*t2])),
      [(t, opposite)] => {
        // A hull edge: its cell edge is a ray from the circumcentre, away
        // from the triangle's third vertex.
        let (pa, pb, po) = (points[*a], points[*b], points[*opposite]);
        let mut dir = [-(pb[1] - pa[1]), pb[0] - pa[0]];
        let mid = [f64::midpoint(pa[0], pb[0]), f64::midpoint(pa[1], pb[1])];
        if dir[0] * (mid[0] - po[0]) + dir[1] * (mid[1] - po[1]) < 0.0 {
          dir = [-dir[0], -dir[1]];
        }
        let norm = dir[0].hypot(dir[1]);
        let c = centers[*t];
        Some((c, [c[0] + dir[0] / norm * far, c[1] + dir[1] / norm * far]))
      }
      _ => None,
    };
    if let Some((p, q)) = segment
      && let Some((p, q)) = clip_segment(p, q, bounds)
    {
      lines.push(call1(
        "Line",
        Expr::List(vec![point_expr(p), point_expr(q)].into()),
      ));
    }
  }
  let mut primitives = vec![Expr::List(lines.into())];
  primitives.push(Expr::List(
    points
      .iter()
      .map(|p| call1("Point", point_expr(*p)))
      .collect(),
  ));
  if label_points {
    primitives.push(Expr::List(
      points
        .iter()
        .enumerate()
        .map(|(i, p)| {
          call("Text", vec![Expr::Integer(i as i128 + 1), point_expr(*p)])
        })
        .collect(),
    ));
  }
  Ok(call("Graphics", vec![Expr::List(primitives.into())]))
}

/// `ComputationalGeometry`VoronoiDiagram[{{x, y}, …}]` — the legacy package's
/// `{vertices, cells}`: the finite Voronoi vertices followed by one
/// `Ray[origin, direction]` per unbounded cell edge, and per site
/// `{i, {vertex numbers}}` listing its cell counter-clockwise (an unbounded
/// cell starts and ends with a ray).
pub fn voronoi_diagram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`VoronoiDiagram", args);
  let Some(points) = args.first().and_then(read_points) else {
    return Ok(original());
  };
  let Some(triangles) = triangulate(&points) else {
    return Ok(original());
  };

  // Circumcentres, merging those of cocircular sites into one vertex.
  let mut vertices: Vec<[f64; 2]> = Vec::new();
  let mut triangle_vertex = Vec::with_capacity(triangles.len());
  for t in &triangles {
    let c = circumcenter_of(points[t[0]], points[t[1]], points[t[2]]);
    let tol = 1e-9 * (1.0 + c[0].abs().max(c[1].abs()));
    let found = vertices
      .iter()
      .position(|v| (v[0] - c[0]).abs() <= tol && (v[1] - c[1]).abs() <= tol);
    triangle_vertex.push(found.unwrap_or_else(|| {
      vertices.push(c);
      vertices.len() - 1
    }));
  }

  // Hull edges (seen by one triangle only) give the rays.
  let mut edges: std::collections::BTreeMap<[usize; 2], Vec<(usize, usize)>> =
    std::collections::BTreeMap::new();
  for (ti, t) in triangles.iter().enumerate() {
    for k in 0..3 {
      let (a, b, opposite) = (t[k], t[(k + 1) % 3], t[(k + 2) % 3]);
      edges
        .entry([a.min(b), a.max(b)])
        .or_default()
        .push((ti, opposite));
    }
  }
  // (site a, site b, origin vertex, unit direction)
  let mut rays: Vec<(usize, usize, usize, [f64; 2])> = Vec::new();
  for ([a, b], sides) in &edges {
    if let [(t, opposite)] = sides.as_slice() {
      let (pa, pb, po) = (points[*a], points[*b], points[*opposite]);
      let mut dir = [-(pb[1] - pa[1]), pb[0] - pa[0]];
      let mid = [f64::midpoint(pa[0], pb[0]), f64::midpoint(pa[1], pb[1])];
      if dir[0] * (mid[0] - po[0]) + dir[1] * (mid[1] - po[1]) < 0.0 {
        dir = [-dir[0], -dir[1]];
      }
      let norm = dir[0].hypot(dir[1]);
      rays.push((*a, *b, triangle_vertex[*t], [dir[0] / norm, dir[1] / norm]));
    }
  }

  let cross = |u: [f64; 2], v: [f64; 2]| u[0] * v[1] - u[1] * v[0];
  let mut cells = Vec::with_capacity(points.len());
  for (i, site) in points.iter().enumerate() {
    let mut finite: Vec<usize> = triangles
      .iter()
      .enumerate()
      .filter(|(_, t)| t.contains(&i))
      .map(|(ti, _)| triangle_vertex[ti])
      .collect();
    finite.sort_unstable();
    finite.dedup();
    let own: Vec<usize> = rays
      .iter()
      .enumerate()
      .filter(|(_, r)| r.0 == i || r.1 == i)
      .map(|(k, _)| k)
      .collect();

    // Directions from the site, with the unbounded cell's entering ray (the
    // one with the cell on its left) as the sweep's start.
    let start = own.iter().copied().find(|&k| {
      let (_, _, o, d) = rays[k];
      let to_site = [site[0] - vertices[o][0], site[1] - vertices[o][1]];
      cross([-d[0], -d[1]], to_site) > 0.0
    });
    let start_dir = start.map(|k| rays[k].3);
    let sweep = |d: [f64; 2]| -> f64 {
      let a = match start_dir {
        Some(s) => cross(s, d).atan2(s[0] * d[0] + s[1] * d[1]),
        None => d[1].atan2(d[0]),
      };
      if a < 0.0 {
        a + std::f64::consts::TAU
      } else {
        a
      }
    };
    // (sweep angle, vertex number in the final list)
    let mut items: Vec<(f64, usize)> = finite
      .iter()
      .map(|&v| {
        (
          sweep([vertices[v][0] - site[0], vertices[v][1] - site[1]]),
          v,
        )
      })
      .collect();
    for &k in &own {
      let angle = if Some(k) == start { 0.0 } else { f64::INFINITY };
      items.push((angle, vertices.len() + k));
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    cells.push(Expr::List(
      vec![
        Expr::Integer(i as i128 + 1),
        Expr::List(
          items
            .iter()
            .map(|(_, v)| Expr::Integer(*v as i128 + 1))
            .collect(),
        ),
      ]
      .into(),
    ));
  }

  let real = |v: f64| Expr::Real(v);
  let pair = |p: [f64; 2]| Expr::List(vec![real(p[0]), real(p[1])].into());
  let mut vertex_exprs: Vec<Expr> = vertices.iter().map(|v| pair(*v)).collect();
  for (_, _, origin, dir) in &rays {
    vertex_exprs.push(call("Ray", vec![pair(vertices[*origin]), pair(*dir)]));
  }
  Ok(Expr::List(
    vec![Expr::List(vertex_exprs.into()), Expr::List(cells.into())].into(),
  ))
}
