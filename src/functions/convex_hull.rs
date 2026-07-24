use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::{Expr, unevaluated};

/// `ConvexHullMesh[{{x, y}, ...}]` → `BoundaryMeshRegion` for a 2D point set.
///
/// The result mirrors wolframscript exactly: the vertex list holds only the
/// strict hull corners (collinear-on-edge and interior points are dropped),
/// kept in the order they appear in the input, and the boundary `Line` walks
/// those vertices counter-clockwise starting at vertex 1. Exact inputs
/// (integers/rationals) stay exact and gain `WorkingPrecision -> Infinity`;
/// any machine real switches every coordinate to a real and drops that option.
///
/// 3D (and higher) hulls need qhull-specific face triangulation/ordering, so
/// they are left unevaluated for now.
pub fn convex_hull_mesh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(unevaluated("ConvexHullMesh", args));
  }
  let pts = match &args[0] {
    Expr::List(v) => v,
    _ => return Ok(unevaluated("ConvexHullMesh", args)),
  };
  if pts.is_empty() {
    return Ok(unevaluated("ConvexHullMesh", args));
  }

  // Embedding dimension is taken from the first point; every point must match.
  let dim = match &pts[0] {
    Expr::List(c) => c.len(),
    _ => return Ok(unevaluated("ConvexHullMesh", args)),
  };
  if dim != 2 {
    return Ok(unevaluated("ConvexHullMesh", args));
  }

  convex_hull_mesh_2d(pts, args)
}

fn convex_hull_mesh_2d(
  pts: &[Expr],
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  // Parse points, keeping the original coordinate expressions (to preserve
  // exact display) and tracking whether every coordinate is exact.
  let mut coords: Vec<(f64, f64)> = Vec::new();
  let mut orig: Vec<[Expr; 2]> = Vec::new();
  let mut all_exact = true;
  for p in pts {
    let Expr::List(c) = p else {
      return Ok(unevaluated("ConvexHullMesh", args));
    };
    if c.len() != 2 {
      return Ok(unevaluated("ConvexHullMesh", args));
    }
    let (Some(x), Some(y)) = (try_eval_to_f64(&c[0]), try_eval_to_f64(&c[1]))
    else {
      return Ok(unevaluated("ConvexHullMesh", args));
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
    return Ok(unevaluated("ConvexHullMesh", args));
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

  let line = Expr::FunctionCall {
    name: "Line".to_string(),
    args: vec![Expr::List(edges.into())].into(),
  };

  // Options. Method -> {SeparateBoundaries -> False} always; exact inputs also
  // carry WorkingPrecision -> Infinity.
  let method = Expr::Rule {
    pattern: Box::new(Expr::Identifier("Method".to_string())),
    replacement: Box::new(Expr::List(
      vec![Expr::Rule {
        pattern: Box::new(Expr::Identifier("SeparateBoundaries".to_string())),
        replacement: Box::new(Expr::Identifier("False".to_string())),
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
      pattern: Box::new(Expr::Identifier("WorkingPrecision".to_string())),
      replacement: Box::new(Expr::Identifier("Infinity".to_string())),
    });
  }

  Ok(Expr::FunctionCall {
    name: "BoundaryMeshRegion".to_string(),
    args: mesh_args.into(),
  })
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
