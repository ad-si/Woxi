#[allow(unused_imports)]
use super::*;
use crate::functions::math_ast::try_eval_to_f64;

/// `ConvexHullMesh[{{x, y}, ...}]` → `BoundaryMeshRegion` for a 2D point set,
/// or `ConvexHullMesh[{{x, y, z}, ...}]` for a 3D one.
///
/// The 2D result mirrors wolframscript exactly: the vertex list holds only
/// the strict hull corners (collinear-on-edge and interior points are
/// dropped), kept in the order they appear in the input, and the boundary
/// `Line` walks those vertices counter-clockwise starting at vertex 1. The 3D
/// result triangulates the hull surface (a standard incremental hull, not a
/// qhull-identical one, so it does not merge coplanar facets into larger
/// polygons the way wolframscript's may) with the same input-order vertex
/// convention. Exact inputs (integers/rationals) stay exact and gain
/// `WorkingPrecision -> Infinity`; any machine real switches every
/// coordinate to a real and drops that option.
///
/// Higher dimensions are left unevaluated.
pub fn convex_hull_mesh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("ConvexHullMesh", args));
  }
  let Expr::List(pts) = &args[0] else {
    return Ok(unevaluated("ConvexHullMesh", args));
  };
  if pts.is_empty() {
    return Ok(unevaluated("ConvexHullMesh", args));
  }

  // Embedding dimension is taken from the first point; every point must match.
  let dim = match &pts[0] {
    Expr::List(c) => c.len(),
    _ => return Ok(unevaluated("ConvexHullMesh", args)),
  };
  match dim {
    2 => Ok(convex_hull_mesh_2d(pts, args)),
    3 => Ok(convex_hull_mesh_3d(pts, args)),
    _ => Ok(unevaluated("ConvexHullMesh", args)),
  }
}

fn convex_hull_mesh_2d(pts: &[Expr], args: &[Expr]) -> Expr {
  // Parse points, keeping the original coordinate expressions (to preserve
  // exact display) and tracking whether every coordinate is exact.
  let mut coords: Vec<(f64, f64)> = Vec::new();
  let mut orig: Vec<[Expr; 2]> = Vec::new();
  let mut all_exact = true;
  for p in pts {
    let Expr::List(c) = p else {
      return unevaluated("ConvexHullMesh", args);
    };
    if c.len() != 2 {
      return unevaluated("ConvexHullMesh", args);
    }
    let (Some(x), Some(y)) = (try_eval_to_f64(&c[0]), try_eval_to_f64(&c[1]))
    else {
      return unevaluated("ConvexHullMesh", args);
    };
    if !is_exact_number(&c[0]) || !is_exact_number(&c[1]) {
      all_exact = false;
    }
    coords.push((x, y));
    orig.push([c[0].clone(), c[1].clone()]);
  }

  // Deduplicate coincident points, keeping the first occurrence's index so the
  // output vertex ordering follows the input order.
  let mut unique: Vec<(f64, f64, usize)> = Vec::new();
  for (i, &(x, y)) in coords.iter().enumerate() {
    if !unique
      .iter()
      .any(|&(ux, uy, _)| (ux - x).abs() <= EPS && (uy - y).abs() <= EPS)
    {
      unique.push((x, y, i));
    }
  }

  // Counter-clockwise hull as a list of input indices (strict corners only).
  let hull = convex_hull_ccw(&unique);
  if hull.len() < 3 {
    // Fewer than three affinely independent points: wolframscript issues a
    // message and leaves the call unevaluated.
    return unevaluated("ConvexHullMesh", args);
  }

  // Output vertices are the hull corners in input order; assign 1-based
  // indices to that ordering.
  let mut ordered_inputs = hull.clone();
  ordered_inputs.sort_unstable();
  let out_index = |input_idx: usize| -> usize {
    ordered_inputs.iter().position(|&i| i == input_idx).unwrap() + 1
  };

  let verts_expr: Vec<Expr> = ordered_inputs
    .iter()
    .map(|&i| {
      if all_exact {
        Expr::List(orig[i].to_vec().into())
      } else {
        Expr::List(
          vec![Expr::Real(coords[i].0), Expr::Real(coords[i].1)].into(),
        )
      }
    })
    .collect();

  // Boundary walk: rotate the CCW cycle so it starts at output vertex 1.
  let cycle: Vec<usize> = hull.iter().map(|&i| out_index(i)).collect();
  let start = cycle.iter().position(|&v| v == 1).unwrap();
  let rotated: Vec<usize> = cycle[start..]
    .iter()
    .chain(&cycle[..start])
    .copied()
    .collect();

  let k = rotated.len();
  let edges: Vec<Expr> = (0..k)
    .map(|i| {
      let a = rotated[i] as i128;
      let b = rotated[(i + 1) % k] as i128;
      Expr::List(vec![Expr::Integer(a), Expr::Integer(b)].into())
    })
    .collect();

  let line = call1("Line", Expr::List(edges.into()));

  // Options. Method -> {"SeparateBoundaries" -> False} always; exact inputs also
  // carry WorkingPrecision -> Infinity.
  let method = Expr::Rule {
    pattern: Box::new(id_expr("Method")),
    replacement: Box::new(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::String("SeparateBoundaries".to_string())),
        replacement: Box::new(bool_expr(false)),
      }]
      .into(),
    )),
  };

  let mut mesh_args = vec![
    Expr::List(verts_expr.into()),
    Expr::List(vec![line].into()),
    method,
  ];
  if all_exact {
    mesh_args.push(Expr::Rule {
      pattern: Box::new(id_expr("WorkingPrecision")),
      replacement: Box::new(id_expr("Infinity")),
    });
  }

  call("BoundaryMeshRegion", mesh_args)
}

const EPS: f64 = 1e-10;

/// Andrew's monotone chain. Returns the input indices of the strict convex-hull
/// corners in counter-clockwise order (collinear points are dropped).
fn convex_hull_ccw(pts: &[(f64, f64, usize)]) -> Vec<usize> {
  let mut p = pts.to_vec();
  p.sort_by(|a, b| {
    a.0
      .partial_cmp(&b.0)
      .unwrap()
      .then(a.1.partial_cmp(&b.1).unwrap())
  });
  let n = p.len();
  if n < 3 {
    return p.iter().map(|t| t.2).collect();
  }

  let cross =
    |o: &(f64, f64, usize),
     a: &(f64, f64, usize),
     b: &(f64, f64, usize)|
     -> f64 { (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0) };

  let mut lower: Vec<(f64, f64, usize)> = Vec::new();
  for &pt in &p {
    while lower.len() >= 2
      && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], &pt) <= EPS
    {
      lower.pop();
    }
    lower.push(pt);
  }

  let mut upper: Vec<(f64, f64, usize)> = Vec::new();
  for &pt in p.iter().rev() {
    while upper.len() >= 2
      && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], &pt) <= EPS
    {
      upper.pop();
    }
    upper.push(pt);
  }

  lower.pop();
  upper.pop();
  lower.into_iter().chain(upper).map(|t| t.2).collect()
}

fn convex_hull_mesh_3d(pts: &[Expr], args: &[Expr]) -> Expr {
  // Parse points, keeping the original coordinate expressions (to preserve
  // exact display) and tracking whether every coordinate is exact.
  let mut coords: Vec<(f64, f64, f64)> = Vec::new();
  let mut orig: Vec<[Expr; 3]> = Vec::new();
  let mut all_exact = true;
  for p in pts {
    let Expr::List(c) = p else {
      return unevaluated("ConvexHullMesh", args);
    };
    if c.len() != 3 {
      return unevaluated("ConvexHullMesh", args);
    }
    let (Some(x), Some(y), Some(z)) = (
      try_eval_to_f64(&c[0]),
      try_eval_to_f64(&c[1]),
      try_eval_to_f64(&c[2]),
    ) else {
      return unevaluated("ConvexHullMesh", args);
    };
    if !is_exact_number(&c[0])
      || !is_exact_number(&c[1])
      || !is_exact_number(&c[2])
    {
      all_exact = false;
    }
    coords.push((x, y, z));
    orig.push([c[0].clone(), c[1].clone(), c[2].clone()]);
  }

  // Deduplicate coincident points, keeping the first occurrence's index so
  // the output vertex ordering follows the input order.
  let mut unique: Vec<(f64, f64, f64, usize)> = Vec::new();
  for (i, &(x, y, z)) in coords.iter().enumerate() {
    if !unique.iter().any(|&(ux, uy, uz, _)| {
      (ux - x).abs() <= EPS && (uy - y).abs() <= EPS && (uz - z).abs() <= EPS
    }) {
      unique.push((x, y, z, i));
    }
  }

  let unique_pts: Vec<(f64, f64, f64)> =
    unique.iter().map(|&(x, y, z, _)| (x, y, z)).collect();

  let Some(faces) = convex_hull_3d(&unique_pts) else {
    // Fewer than 4 affinely independent points (coplanar, collinear, too
    // few, or coincident): wolframscript issues a message and leaves the
    // call unevaluated, matched here as the 2D case already does for its
    // own degenerate inputs.
    return unevaluated("ConvexHullMesh", args);
  };

  // Output vertices are the hull corners in input order: the set of unique
  // indices any face uses, sorted by the input index they first appeared
  // at (mirrors the 2D case's `ordered_inputs`).
  let mut used: Vec<usize> = faces
    .iter()
    .flat_map(|f| f.iter().copied())
    .collect::<std::collections::BTreeSet<_>>()
    .into_iter()
    .collect();
  used.sort_unstable_by_key(|&ui| unique[ui].3);

  let out_index = |unique_idx: usize| -> usize {
    used.iter().position(|&u| u == unique_idx).unwrap() + 1
  };

  let verts_expr: Vec<Expr> = used
    .iter()
    .map(|&ui| {
      let orig_i = unique[ui].3;
      if all_exact {
        Expr::List(orig[orig_i].to_vec().into())
      } else {
        let (x, y, z) = unique_pts[ui];
        Expr::List(vec![Expr::Real(x), Expr::Real(y), Expr::Real(z)].into())
      }
    })
    .collect();

  let face_exprs: Vec<Expr> = faces
    .iter()
    .map(|&[a, b, c]| {
      Expr::List(
        vec![
          Expr::Integer(out_index(a) as i128),
          Expr::Integer(out_index(b) as i128),
          Expr::Integer(out_index(c) as i128),
        ]
        .into(),
      )
    })
    .collect();

  let polygon = call1("Polygon", Expr::List(face_exprs.into()));

  let method = Expr::Rule {
    pattern: Box::new(id_expr("Method")),
    replacement: Box::new(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::String("SeparateBoundaries".to_string())),
        replacement: Box::new(bool_expr(false)),
      }]
      .into(),
    )),
  };

  let mut mesh_args = vec![
    Expr::List(verts_expr.into()),
    Expr::List(vec![polygon].into()),
    method,
  ];
  if all_exact {
    mesh_args.push(Expr::Rule {
      pattern: Box::new(id_expr("WorkingPrecision")),
      replacement: Box::new(id_expr("Infinity")),
    });
  }

  call("BoundaryMeshRegion", mesh_args)
}

/// A standard incremental ("beneath-beyond") 3D convex hull. Returns
/// triangular faces as index triples into `points`, each wound so its
/// outward normal follows the right-hand rule, or `None` when fewer than 4
/// points are affinely independent (too few points, or all collinear/
/// coplanar) — the 3D analogue of the 2D hull needing 3 non-collinear
/// points.
fn convex_hull_3d(points: &[(f64, f64, f64)]) -> Option<Vec<[usize; 3]>> {
  let n = points.len();
  if n < 4 {
    return None;
  }

  let sub =
    |a: (f64, f64, f64), b: (f64, f64, f64)| (a.0 - b.0, a.1 - b.1, a.2 - b.2);
  let cross = |a: (f64, f64, f64), b: (f64, f64, f64)| {
    (
      a.1 * b.2 - a.2 * b.1,
      a.2 * b.0 - a.0 * b.2,
      a.0 * b.1 - a.1 * b.0,
    )
  };
  let dot =
    |a: (f64, f64, f64), b: (f64, f64, f64)| a.0 * b.0 + a.1 * b.1 + a.2 * b.2;
  let norm = |a: (f64, f64, f64)| dot(a, a).sqrt();

  // A relative epsilon, so the hull works on point clouds of any scale, not
  // just ones near the origin.
  let scale = points
    .iter()
    .flat_map(|&(x, y, z)| [x.abs(), y.abs(), z.abs()])
    .fold(1.0_f64, f64::max);
  let eps = 1e-9 * scale;

  // Seed a tetrahedron: p0 is a coordinate-minimal point (an arbitrary but
  // deterministic start), p1 is the farthest point from it, p2 the farthest
  // from line p0-p1, and p3 the farthest from plane p0-p1-p2. Each step
  // bailing out to `None` when the remaining points are all within `eps` of
  // being degenerate with what came before.
  let mut p0 = 0usize;
  for i in 1..n {
    if points[i].0 < points[p0].0
      || (points[i].0 == points[p0].0 && points[i].1 < points[p0].1)
    {
      p0 = i;
    }
  }
  let mut p1 = usize::MAX;
  let mut best = eps;
  for i in 0..n {
    if i == p0 {
      continue;
    }
    let d = norm(sub(points[i], points[p0]));
    if d > best {
      best = d;
      p1 = i;
    }
  }
  if p1 == usize::MAX {
    return None; // every point coincides with p0
  }

  let dir = sub(points[p1], points[p0]);
  let mut p2 = usize::MAX;
  let mut best = eps;
  for i in 0..n {
    if i == p0 || i == p1 {
      continue;
    }
    let d = norm(cross(dir, sub(points[i], points[p0])));
    if d > best {
      best = d;
      p2 = i;
    }
  }
  if p2 == usize::MAX {
    return None; // collinear
  }

  let normal012 =
    cross(sub(points[p1], points[p0]), sub(points[p2], points[p0]));
  let mut p3 = usize::MAX;
  let mut best = eps * norm(normal012).max(1.0);
  for i in 0..n {
    if i == p0 || i == p1 || i == p2 {
      continue;
    }
    let d = dot(normal012, sub(points[i], points[p0])).abs();
    if d > best {
      best = d;
      p3 = i;
    }
  }
  if p3 == usize::MAX {
    return None; // coplanar
  }

  let centroid = (
    (points[p0].0 + points[p1].0 + points[p2].0 + points[p3].0) / 4.0,
    (points[p0].1 + points[p1].1 + points[p2].1 + points[p3].1) / 4.0,
    (points[p0].2 + points[p1].2 + points[p2].2 + points[p3].2) / 4.0,
  );
  let face_normal = |pts: &[(f64, f64, f64)], a: usize, b: usize, c: usize| {
    cross(sub(pts[b], pts[a]), sub(pts[c], pts[a]))
  };
  // Winds (a, b, c) so its outward normal points away from `centroid`.
  let orient =
    |pts: &[(f64, f64, f64)], a: usize, b: usize, c: usize| -> [usize; 3] {
      let n = face_normal(pts, a, b, c);
      if dot(n, sub(centroid, pts[a])) > 0.0 {
        [a, c, b]
      } else {
        [a, b, c]
      }
    };

  let mut faces: Vec<[usize; 3]> = vec![
    orient(points, p0, p1, p2),
    orient(points, p0, p1, p3),
    orient(points, p0, p2, p3),
    orient(points, p1, p2, p3),
  ];

  let mut used = std::collections::HashSet::new();
  used.insert(p0);
  used.insert(p1);
  used.insert(p2);
  used.insert(p3);

  // Add every remaining point: drop the faces it can see, and re-triangulate
  // the hole with new faces from the point to each horizon edge.
  for q in 0..n {
    if used.contains(&q) {
      continue;
    }

    let visible: Vec<bool> = faces
      .iter()
      .map(|&[a, b, c]| {
        let n = face_normal(points, a, b, c);
        dot(n, sub(points[q], points[a])) > eps
      })
      .collect();

    if !visible.iter().any(|&v| v) {
      // q lies inside (or on) the current hull: not a vertex.
      continue;
    }

    // A directed edge (a, b) of a visible face is a horizon edge exactly
    // when its opposite-directed edge (b, a) — the same undirected edge,
    // seen from the other face that shares it — belongs to a face that is
    // not visible. A closed, consistently wound triangulation has each
    // directed edge on exactly one face, so this lookup is unambiguous.
    let mut edge_owner: std::collections::HashMap<(usize, usize), usize> =
      std::collections::HashMap::new();
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
      edge_owner.insert((a, b), fi);
      edge_owner.insert((b, c), fi);
      edge_owner.insert((c, a), fi);
    }

    let mut horizon: Vec<(usize, usize)> = Vec::new();
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
      if !visible[fi] {
        continue;
      }
      for &(u, v) in &[(a, b), (b, c), (c, a)] {
        let opposite_visible =
          edge_owner.get(&(v, u)).is_some_and(|&ofi| visible[ofi]);
        if !opposite_visible {
          horizon.push((u, v));
        }
      }
    }

    let mut new_faces: Vec<[usize; 3]> = faces
      .iter()
      .enumerate()
      .filter(|&(fi, _)| !visible[fi])
      .map(|(_, &f)| f)
      .collect();
    for (u, v) in horizon {
      // Reusing the visible face's own edge direction keeps the new face's
      // winding (and so its outward-pointing normal) consistent with the
      // rest of the hull.
      new_faces.push([u, v, q]);
    }
    faces = new_faces;
    used.insert(q);
  }

  Some(faces)
}

/// Whether an expression is an exact number (integer or rational), so the hull
/// vertices should keep their exact form rather than being converted to reals.
fn is_exact_number(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) => true,
    Expr::FunctionCall { name, .. } if name == "Rational" => true,
    Expr::UnaryOp { operand, .. } => is_exact_number(operand),
    Expr::BinaryOp { left, right, .. } => {
      is_exact_number(left) && is_exact_number(right)
    }
    _ => false,
  }
}
