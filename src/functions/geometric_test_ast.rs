//! `GeometricTest[...]` — synthetic-geometry predicate tests.
//!
//! `GeometricTest[objs, prop]` determines whether one or more geometric
//! objects satisfy a named property or relation, returning `True` or `False`.
//! (Wolfram also returns algebraic *conditions* when the input contains
//! symbolic variables; those cases are intentionally left unevaluated here.)
//!
//! Only fully numeric coordinates are handled. Anything that cannot be
//! reduced to concrete points — symbolic coordinates, unsupported
//! properties, malformed input — makes the whole call return `None` so the
//! expression is left unevaluated (matching how the interpreter treats
//! unsupported cases elsewhere).

use crate::InterpreterError;
use crate::syntax::Expr;

/// Absolute tolerance for orientation / sign tests (exact-zero comparisons on
/// well-conditioned inputs).
const EPS: f64 = 1e-9;
/// Relative tolerance for comparing magnitudes (lengths, cosines, ratios).
const REL: f64 = 1e-8;

type Pt = (f64, f64);

fn sub(a: Pt, b: Pt) -> Pt {
  (a.0 - b.0, a.1 - b.1)
}
fn cross(a: Pt, b: Pt) -> f64 {
  a.0 * b.1 - a.1 * b.0
}
fn dot(a: Pt, b: Pt) -> f64 {
  a.0 * b.0 + a.1 * b.1
}
fn mag(a: Pt) -> f64 {
  (a.0 * a.0 + a.1 * a.1).sqrt()
}
fn dist(a: Pt, b: Pt) -> f64 {
  mag(sub(a, b))
}

/// Scale-independent parallelism test: `|sin(angle)| <= REL`.
fn parallel_vec(a: Pt, b: Pt) -> bool {
  let m = mag(a) * mag(b);
  m > EPS && cross(a, b).abs() <= REL * m
}
/// Scale-independent perpendicularity test: `|cos(angle)| <= REL`.
fn perp_vec(a: Pt, b: Pt) -> bool {
  let m = mag(a) * mag(b);
  m > EPS && dot(a, b).abs() <= REL * m
}
/// Relative equality of two non-negative magnitudes.
fn approx_eq(a: f64, b: f64) -> bool {
  (a - b).abs() <= REL * (1.0 + a.abs().max(b.abs()))
}

fn bool_expr(b: bool) -> Expr {
  Expr::Identifier(if b { "True" } else { "False" }.to_string())
}

/// Parse a single 2D point `{x, y}` with numeric coordinates.
fn extract_point(expr: &Expr) -> Option<Pt> {
  if let Expr::List(items) = expr
    && items.len() == 2
  {
    let x = super::math_ast::try_eval_to_f64(&items[0])?;
    let y = super::math_ast::try_eval_to_f64(&items[1])?;
    return Some((x, y));
  }
  None
}

/// Extract an ordered list of points from a bare point list `{{x,y},...}` or
/// from a geometric wrapper (`Point`, `Polygon`, `Triangle`, `Line`,
/// `InfiniteLine`) whose sole argument is such a list.
fn extract_points(expr: &Expr) -> Option<Vec<Pt>> {
  match expr {
    Expr::List(items) if !items.is_empty() => {
      items.iter().map(extract_point).collect()
    }
    Expr::FunctionCall { name, args } if args.len() == 1 => match name.as_str()
    {
      "Point" | "Polygon" | "Triangle" | "Line" | "InfiniteLine" => {
        extract_points(&args[0])
      }
      _ => None,
    },
    _ => None,
  }
}

/// A line represented as a point on it plus a direction vector.
fn extract_line(expr: &Expr) -> Option<(Pt, Pt)> {
  if let Expr::FunctionCall { name, args } = expr {
    match name.as_str() {
      "Line" | "InfiniteLine" if args.len() == 1 => {
        if let Expr::List(pts) = &args[0]
          && pts.len() == 2
        {
          let p1 = extract_point(&pts[0])?;
          let p2 = extract_point(&pts[1])?;
          return Some((p1, sub(p2, p1)));
        }
      }
      // InfiniteLine[point, direction]
      "InfiniteLine" if args.len() == 2 => {
        let p = extract_point(&args[0])?;
        let d = extract_point(&args[1])?;
        return Some((p, d));
      }
      _ => {}
    }
  }
  None
}

/// Extract one or more lines: either a single line object or a list of them.
fn extract_lines(expr: &Expr) -> Option<Vec<(Pt, Pt)>> {
  match expr {
    Expr::List(items) => items.iter().map(extract_line).collect(),
    _ => extract_line(expr).map(|l| vec![l]),
  }
}

/// Extract a list of objects, each a point set (used for `Congruent` /
/// `Similar`, whose first argument is a list of geometric objects).
fn extract_object_list(expr: &Expr) -> Option<Vec<Vec<Pt>>> {
  if let Expr::List(items) = expr {
    items.iter().map(extract_points).collect()
  } else {
    None
  }
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

fn collinear(pts: &[Pt]) -> bool {
  if pts.len() <= 2 {
    return true;
  }
  let p0 = pts[0];
  let dir = pts[1..].iter().map(|&p| sub(p, p0)).find(|d| mag(*d) > EPS);
  match dir {
    Some(d) => pts.iter().all(|&p| {
      let v = sub(p, p0);
      mag(v) <= EPS || parallel_vec(d, v)
    }),
    None => true, // all points coincide
  }
}

fn all_distinct(pts: &[Pt]) -> bool {
  for i in 0..pts.len() {
    for j in (i + 1)..pts.len() {
      if dist(pts[i], pts[j]) <= EPS {
        return false;
      }
    }
  }
  true
}

fn all_parallel(lines: &[(Pt, Pt)]) -> bool {
  lines.len() >= 2 && lines[1..].iter().all(|l| parallel_vec(lines[0].1, l.1))
}

fn all_perpendicular(lines: &[(Pt, Pt)]) -> bool {
  // Every distinct pair of lines is mutually perpendicular. Only meaningful
  // for two lines in the plane, but generalises to the pairwise check.
  if lines.len() < 2 {
    return false;
  }
  for i in 0..lines.len() {
    for j in (i + 1)..lines.len() {
      if !perp_vec(lines[i].1, lines[j].1) {
        return false;
      }
    }
  }
  true
}

/// Intersection point of two lines, or `None` when parallel.
fn line_intersection(l1: (Pt, Pt), l2: (Pt, Pt)) -> Option<Pt> {
  let denom = cross(l1.1, l2.1);
  if denom.abs() <= EPS * mag(l1.1) * mag(l2.1) {
    return None;
  }
  let t = cross(sub(l2.0, l1.0), l2.1) / denom;
  Some((l1.0.0 + t * l1.1.0, l1.0.1 + t * l1.1.1))
}

fn point_on_line(q: Pt, l: (Pt, Pt)) -> bool {
  let v = sub(q, l.0);
  mag(v) <= EPS || parallel_vec(l.1, v)
}

fn concurrent(lines: &[(Pt, Pt)]) -> bool {
  if lines.len() < 2 {
    return false;
  }
  // Find any concrete intersection to use as the common-point candidate.
  let mut pivot = None;
  'outer: for i in 0..lines.len() {
    for j in (i + 1)..lines.len() {
      if let Some(p) = line_intersection(lines[i], lines[j]) {
        pivot = Some(p);
        break 'outer;
      }
    }
  }
  match pivot {
    Some(p) => lines.iter().all(|&l| point_on_line(p, l)),
    None => false, // all lines mutually parallel → no common point
  }
}

fn all_horizontal(lines: &[(Pt, Pt)]) -> bool {
  !lines.is_empty()
    && lines
      .iter()
      .all(|l| mag(l.1) > EPS && l.1.1.abs() <= REL * mag(l.1))
}
fn all_vertical(lines: &[(Pt, Pt)]) -> bool {
  !lines.is_empty()
    && lines
      .iter()
      .all(|l| mag(l.1) > EPS && l.1.0.abs() <= REL * mag(l.1))
}

fn convex(pts: &[Pt]) -> bool {
  let n = pts.len();
  if n < 3 {
    return false;
  }
  let mut sign = 0i32;
  for i in 0..n {
    let a = pts[i];
    let b = pts[(i + 1) % n];
    let c = pts[(i + 2) % n];
    let cr = cross(sub(b, a), sub(c, b));
    if cr.abs() > EPS {
      let s = if cr > 0.0 { 1 } else { -1 };
      if sign == 0 {
        sign = s;
      } else if sign != s {
        return false;
      }
    }
  }
  true
}

fn equilateral(pts: &[Pt]) -> bool {
  let n = pts.len();
  if n < 3 {
    return false;
  }
  let d0 = dist(pts[0], pts[1]);
  (0..n).all(|i| approx_eq(dist(pts[i], pts[(i + 1) % n]), d0))
}

/// Cosine of the interior angle at vertex `i`.
fn interior_cos(pts: &[Pt], i: usize) -> f64 {
  let n = pts.len();
  let prev = pts[(i + n - 1) % n];
  let cur = pts[i];
  let next = pts[(i + 1) % n];
  let v1 = sub(prev, cur);
  let v2 = sub(next, cur);
  let m = mag(v1) * mag(v2);
  if m <= EPS {
    1.0
  } else {
    (dot(v1, v2) / m).clamp(-1.0, 1.0)
  }
}

fn equiangular(pts: &[Pt]) -> bool {
  let n = pts.len();
  if n < 3 {
    return false;
  }
  let c0 = interior_cos(pts, 0);
  (0..n).all(|i| approx_eq(interior_cos(pts, i), c0))
}

fn rectangle(pts: &[Pt]) -> bool {
  pts.len() == 4 && (0..4).all(|i| interior_cos(pts, i).abs() <= REL)
}

fn parallelogram(pts: &[Pt]) -> bool {
  if pts.len() != 4 {
    return false;
  }
  // Diagonals of a parallelogram bisect each other.
  let mid_ac = ((pts[0].0 + pts[2].0) / 2.0, (pts[0].1 + pts[2].1) / 2.0);
  let mid_bd = ((pts[1].0 + pts[3].0) / 2.0, (pts[1].1 + pts[3].1) / 2.0);
  dist(mid_ac, mid_bd) <= REL * (1.0 + mag(mid_ac))
}

/// Twice the signed area (positive = counterclockwise) via the shoelace sum.
fn signed_area2(pts: &[Pt]) -> f64 {
  let n = pts.len();
  let mut s = 0.0;
  for i in 0..n {
    let a = pts[i];
    let b = pts[(i + 1) % n];
    s += a.0 * b.1 - b.0 * a.1;
  }
  s
}

fn orient(a: Pt, b: Pt, c: Pt) -> f64 {
  cross(sub(b, a), sub(c, a))
}

/// Whether point `p` (already known collinear with `a`-`b`) lies within the
/// segment's bounding box.
fn on_seg_bbox(a: Pt, b: Pt, p: Pt) -> bool {
  p.0 <= a.0.max(b.0) + EPS
    && p.0 >= a.0.min(b.0) - EPS
    && p.1 <= a.1.max(b.1) + EPS
    && p.1 >= a.1.min(b.1) - EPS
}

fn segments_intersect(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
  let d1 = orient(c, d, a);
  let d2 = orient(c, d, b);
  let d3 = orient(a, b, c);
  let d4 = orient(a, b, d);
  if ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS))
    && ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS))
  {
    return true;
  }
  (d1.abs() <= EPS && on_seg_bbox(c, d, a))
    || (d2.abs() <= EPS && on_seg_bbox(c, d, b))
    || (d3.abs() <= EPS && on_seg_bbox(a, b, c))
    || (d4.abs() <= EPS && on_seg_bbox(a, b, d))
}

/// A polygon is simple when no two non-adjacent edges intersect.
fn simple(pts: &[Pt]) -> bool {
  let n = pts.len();
  if n < 3 {
    return false;
  }
  for i in 0..n {
    let a1 = pts[i];
    let a2 = pts[(i + 1) % n];
    for j in (i + 1)..n {
      // Skip edges that share a vertex with edge i.
      if (i + 1) % n == j || (j + 1) % n == i {
        continue;
      }
      let b1 = pts[j];
      let b2 = pts[(j + 1) % n];
      if segments_intersect(a1, a2, b1, b2) {
        return false;
      }
    }
  }
  true
}

/// Sorted edge lengths of a polygon (used for triangle congruence/similarity).
fn sorted_sides(pts: &[Pt]) -> Vec<f64> {
  let n = pts.len();
  let mut s: Vec<f64> =
    (0..n).map(|i| dist(pts[i], pts[(i + 1) % n])).collect();
  s.sort_by(|a, b| a.partial_cmp(b).unwrap());
  s
}

fn congruent(objs: &[Vec<Pt>]) -> Option<bool> {
  if objs.len() < 2 || !objs.iter().all(|o| o.len() == 3) {
    return None; // only triangles are handled exactly
  }
  let base = sorted_sides(&objs[0]);
  Some(objs.iter().all(|o| {
    sorted_sides(o)
      .iter()
      .zip(&base)
      .all(|(a, b)| approx_eq(*a, *b))
  }))
}

fn similar(objs: &[Vec<Pt>]) -> Option<bool> {
  if objs.len() < 2 || !objs.iter().all(|o| o.len() == 3) {
    return None;
  }
  let base = sorted_sides(&objs[0]);
  if base[0] <= EPS {
    return Some(false);
  }
  Some(objs.iter().all(|o| {
    let s = sorted_sides(o);
    if s[0] <= EPS {
      return false;
    }
    let ratio = s[0] / base[0];
    s.iter().zip(&base).all(|(a, b)| approx_eq(a / b, ratio))
  }))
}

/// Evaluate a single named property against the object(s). Returns `None`
/// (leaving the call unevaluated) for symbolic input or unsupported cases.
fn test_property(obj: &Expr, prop: &str) -> Option<bool> {
  match prop {
    "Collinear" => extract_points(obj).map(|p| collinear(&p)),
    "Distinct" => extract_points(obj).map(|p| all_distinct(&p)),
    "Parallel" => extract_lines(obj)
      .filter(|l| l.len() >= 2)
      .map(|l| all_parallel(&l)),
    "Perpendicular" => extract_lines(obj)
      .filter(|l| l.len() >= 2)
      .map(|l| all_perpendicular(&l)),
    "Concurrent" => extract_lines(obj)
      .filter(|l| l.len() >= 2)
      .map(|l| concurrent(&l)),
    "Horizontal" => extract_lines(obj).map(|l| all_horizontal(&l)),
    "Vertical" => extract_lines(obj).map(|l| all_vertical(&l)),
    "Convex" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| convex(&p)),
    "Equilateral" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| equilateral(&p)),
    "Equiangular" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| equiangular(&p)),
    "Regular" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| equilateral(&p) && equiangular(&p)),
    "Rectangle" => extract_points(obj)
      .filter(|p| p.len() == 4)
      .map(|p| rectangle(&p)),
    "Parallelogram" => extract_points(obj)
      .filter(|p| p.len() == 4)
      .map(|p| parallelogram(&p)),
    "Simple" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| simple(&p)),
    "Clockwise" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| signed_area2(&p) < -EPS),
    "Counterclockwise" => extract_points(obj)
      .filter(|p| p.len() >= 3)
      .map(|p| signed_area2(&p) > EPS),
    "Congruent" => extract_object_list(obj).and_then(|o| congruent(&o)),
    "Similar" => extract_object_list(obj).and_then(|o| similar(&o)),
    _ => None,
  }
}

/// `GeometricTest[objs, prop1, prop2, ...]`.
pub fn geometric_test(args: &[Expr]) -> Option<Result<Expr, InterpreterError>> {
  if args.len() < 2 {
    return None;
  }
  let obj = &args[0];
  // All remaining arguments must be string property names.
  let mut props = Vec::with_capacity(args.len() - 1);
  for a in &args[1..] {
    match a {
      Expr::String(s) => props.push(s.as_str()),
      _ => return None,
    }
  }
  // Every requested property must hold (`True` only if all are satisfied).
  let mut all = true;
  for p in props {
    match test_property(obj, p) {
      Some(b) => all = all && b,
      None => return None, // symbolic / unsupported → leave unevaluated
    }
  }
  Some(Ok(bool_expr(all)))
}

// ---------------------------------------------------------------------------
// CollinearPoints — exact-arithmetic test (arbitrary dimension)
// ---------------------------------------------------------------------------

/// Extract a list of n-dimensional points with a common dimension. Points may
/// carry symbolic coordinates; those are handled by the caller.
fn extract_nd_points(expr: &Expr) -> Option<Vec<Vec<Expr>>> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
  let mut pts = Vec::with_capacity(items.len());
  let mut dim = None;
  for item in items.iter() {
    let Expr::List(coords) = item else {
      return None;
    };
    if coords.is_empty() {
      return None;
    }
    match dim {
      None => dim = Some(coords.len()),
      Some(d) if d != coords.len() => return None,
      _ => {}
    }
    pts.push(coords.to_vec());
  }
  Some(pts)
}

/// Whether `e` is a numeric quantity in the Wolfram sense (`NumericQ`), i.e.
/// reduces to a definite number (integers, rationals, reals, Pi, Sqrt[2], …).
fn coord_is_numeric(e: &Expr) -> bool {
  matches!(
    crate::evaluator::evaluate_function_call_ast("NumericQ", &[e.clone()]),
    Ok(Expr::Identifier(ref s)) if s == "True"
  )
}

/// Evaluate `e` and decide whether it is exactly zero. Exact rationals reduce
/// to `Integer(0)`; reals and other numeric constants are compared strictly
/// against 0 (matching how wolframscript treats machine numbers). Returns
/// `None` when the value cannot be reduced to a definite number.
fn expr_is_zero(e: &Expr) -> Option<bool> {
  let v = crate::evaluator::evaluate_expr_to_expr(e).ok()?;
  match &v {
    Expr::Integer(n) => Some(*n == 0),
    Expr::BigInteger(n) => Some(*n == num_bigint::BigInt::from(0)),
    Expr::Real(r) => Some(*r == 0.0),
    _ => super::math_ast::try_eval_to_f64(&v).map(|f| f == 0.0),
  }
}

fn sub_expr(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Subtract".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  }
}

fn mul_expr(a: &Expr, b: &Expr) -> Expr {
  Expr::FunctionCall {
    name: "Times".to_string(),
    args: vec![a.clone(), b.clone()].into(),
  }
}

/// CollinearPoints[{p1, p2, …}] — `True`/`False` when every coordinate is
/// numeric. Points are collinear iff the matrix of difference vectors
/// `pi - p1` has rank at most one, tested exactly via vanishing 2×2 minors.
/// Symbolic coordinates (where Wolfram returns an algebraic condition) are
/// left unevaluated.
pub fn collinear_points_ast(
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  if args.len() != 1 {
    return None;
  }
  let pts = extract_nd_points(&args[0])?;
  let n = pts.len();
  // Zero, one, or two points are always collinear.
  if n <= 2 {
    return Some(Ok(bool_expr(true)));
  }
  // Only fully numeric inputs are handled here.
  if !pts.iter().flatten().all(coord_is_numeric) {
    return None;
  }

  let dim = pts[0].len();
  let p0 = &pts[0];
  // Difference vectors pi - p1.
  let diffs: Vec<Vec<Expr>> = pts[1..]
    .iter()
    .map(|p| {
      (0..dim)
        .map(|j| sub_expr(&p[j], &p0[j]))
        .collect::<Vec<_>>()
    })
    .collect();

  // Reference direction: the first non-zero difference vector.
  let mut reference: Option<&Vec<Expr>> = None;
  for d in &diffs {
    let is_zero_vec = d.iter().all(|c| expr_is_zero(c).unwrap_or(false));
    if !is_zero_vec {
      reference = Some(d);
      break;
    }
  }
  let Some(r) = reference else {
    // Every point coincides with the first.
    return Some(Ok(bool_expr(true)));
  };

  // Each difference vector must be parallel to the reference: all 2×2 minors
  // r[j]*v[k] - r[k]*v[j] vanish.
  for v in &diffs {
    for j in 0..dim {
      for k in (j + 1)..dim {
        let minor = sub_expr(&mul_expr(&r[j], &v[k]), &mul_expr(&r[k], &v[j]));
        match expr_is_zero(&minor) {
          Some(true) => {}
          Some(false) => return Some(Ok(bool_expr(false))),
          None => return None,
        }
      }
    }
  }
  Some(Ok(bool_expr(true)))
}
