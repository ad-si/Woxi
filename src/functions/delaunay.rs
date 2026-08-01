//! `DelaunayMesh[{{x, y}, …}]` — the Delaunay triangulation of a 2D point
//! set, as the `MeshRegion[coords, {Polygon[{…}]}]` object the region
//! machinery reads.

#[allow(unused_imports)]
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
fn triangulate(points: &[[f64; 2]]) -> Option<Vec<[usize; 3]>> {
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
  let cx = (min_x + max_x) / 2.0;
  let cy = (min_y + max_y) / 2.0;
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
  for tri in out.iter_mut() {
    // Rotate so the smallest index leads, keeping the orientation.
    let smallest = (0..3).min_by_key(|k| tri[*k]).unwrap();
    *tri = [
      tri[smallest],
      tri[(smallest + 1) % 3],
      tri[(smallest + 2) % 3],
    ];
  }
  out.sort();
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
  for item in items.iter() {
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
      vec![Expr::FunctionCall {
        name: "Point".to_string(),
        args: vec![Expr::List(
          (1..=points.len())
            .map(|i| Expr::List(vec![Expr::Integer(i as i128)].into()))
            .collect(),
        )]
        .into(),
      }]
      .into(),
    ));
  } else {
    let Some(triangles) = triangulate(&points) else {
      return Ok(original());
    };
    mesh_args.push(Expr::List(
      vec![Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(
          triangles
            .into_iter()
            .map(|t| {
              Expr::List(
                t.iter().map(|v| Expr::Integer(*v as i128 + 1)).collect(),
              )
            })
            .collect(),
        )]
        .into(),
      }]
      .into(),
    ));
  }
  if exact {
    mesh_args.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("WorkingPrecision".to_string())),
      replacement: Box::new(Expr::Identifier("Infinity".to_string())),
    });
  }
  Ok(Expr::FunctionCall {
    name: "MeshRegion".to_string(),
    args: mesh_args.into(),
  })
}
