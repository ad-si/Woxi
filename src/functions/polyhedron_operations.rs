//! The legacy `PolyhedronOperations` package: ``Truncate`` and ``Stellate``.
//!
//! Woxi has no package system (see [`crate::evaluator::contexts`]), so these
//! two functions are implemented directly under their fully-qualified names
//! — the way code that has run `Needs["PolyhedronOperations`"]` calls them.
//!
//! Both operate on a "graphics expression": an arbitrarily nested `List` of
//! `Polygon[…]` primitives, possibly wrapped in `Rotate`/`Translate` — a
//! `Rotate[g, θ, axis]` outside an actual graphic stays a symbolic wrapper
//! around `g` rather than eagerly recomputing `g`'s coordinates (see the
//! `Rotate` reference page's "outside a graphic" details; `Normal[expr]`
//! is the documented way to resolve such a wrapper to concrete
//! coordinates). [`resolve_transforms`] does the equivalent resolution so
//! the truncation/stellation itself can run per `Polygon` face.

use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot3d::rotation_matrix;
use crate::helpers::call1;
use crate::syntax::Expr;

/// `Truncate[expr]` truncates to this fraction of each edge's length.
pub(crate) const DEFAULT_TRUNCATE_RATIO: f64 = 0.3;
/// `Stellate[expr]` raises each face's pyramid to this ratio.
pub(crate) const DEFAULT_STELLATE_RATIO: f64 = 2.0;

type Vec3 = [f64; 3];

fn add(a: Vec3, b: Vec3) -> Vec3 {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vec3, s: f64) -> Vec3 {
  [a[0] * s, a[1] * s, a[2] * s]
}

fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
  add(a, scale(sub(b, a), t))
}

fn apply_matrix(m: &[[f64; 3]; 3], p: Vec3) -> Vec3 {
  [
    m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
    m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
    m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
  ]
}

fn vec3_to_expr(v: Vec3) -> Expr {
  Expr::List(vec![Expr::Real(v[0]), Expr::Real(v[1]), Expr::Real(v[2])].into())
}

/// `expr` as a point — a `List` of exactly three numeric entries.
fn as_point(expr: &Expr) -> Option<Vec3> {
  let Expr::List(items) = expr else {
    return None;
  };
  if items.len() != 3 {
    return None;
  }
  Some([
    try_eval_to_f64(&items[0])?,
    try_eval_to_f64(&items[1])?,
    try_eval_to_f64(&items[2])?,
  ])
}

/// Apply `f` to every point (three-element numeric `List`) found anywhere
/// inside `expr`, leaving everything else — heads, non-point lists, style
/// directives — structurally unchanged.
fn map_points(expr: &Expr, f: &dyn Fn(Vec3) -> Vec3) -> Expr {
  if let Some(p) = as_point(expr) {
    return vec3_to_expr(f(p));
  }
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| map_points(e, f))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| map_points(e, f))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(map_points(pattern, f)),
      replacement: Box::new(map_points(replacement, f)),
    },
    other => other.clone(),
  }
}

/// The transform a `Rotate[g, θ, …]` or `Translate[g, offset]` wrapper
/// applies to its content's points, built from the wrapper's arguments
/// (everything after `g`). `None` when the arguments don't describe a
/// transform Woxi can resolve (e.g. `Translate` with a list of several
/// offset vectors, which produces multiple copies rather than one).
fn wrapper_transform(
  name: &str,
  rest: &[Expr],
) -> Option<Box<dyn Fn(Vec3) -> Vec3>> {
  match name {
    "Translate" if rest.len() == 1 => {
      let offset = as_point(&rest[0])?;
      Some(Box::new(move |p| add(p, offset)))
    }
    "Rotate" if rest.len() == 2 => {
      let angle = try_eval_to_f64(&rest[0])?;
      // `Rotate[g, θ, w]`: axis direction `w` through the origin.
      if let Some(w) = as_point(&rest[1]) {
        let m = rotation_matrix(angle, w)?;
        return Some(Box::new(move |p| apply_matrix(&m, p)));
      }
      // `Rotate[g, θ, {p1, p2}]`: axis through the points `p1` and `p2`.
      if let Expr::List(pts) = &rest[1]
        && pts.len() == 2
        && let (Some(p1), Some(p2)) = (as_point(&pts[0]), as_point(&pts[1]))
      {
        let w = sub(p2, p1);
        let m = rotation_matrix(angle, w)?;
        return Some(Box::new(move |p| add(apply_matrix(&m, sub(p, p1)), p1)));
      }
      None
    }
    "Rotate" if rest.len() == 3 => {
      let angle = try_eval_to_f64(&rest[0])?;
      let w = as_point(&rest[1])?;
      let anchor = as_point(&rest[2])?;
      let m = rotation_matrix(angle, w)?;
      Some(Box::new(move |p| {
        add(apply_matrix(&m, sub(p, anchor)), anchor)
      }))
    }
    _ => None,
  }
}

/// Resolve every `Rotate`/`Translate` wrapper in `expr` to concrete
/// coordinates — needed because those wrappers stay symbolic (see the
/// `Rotate` reference page) until something asks for the transformed
/// points explicitly, the way `Normal` would for a rendered graphic.
pub(crate) fn resolve_transforms(expr: &Expr) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(resolve_transforms)
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args }
      if (name == "Rotate" || name == "Translate") && !args.is_empty() =>
    {
      let inner = resolve_transforms(&args[0]);
      match wrapper_transform(name, &args[1..]) {
        Some(f) => map_points(&inner, f.as_ref()),
        None => Expr::FunctionCall {
          name: name.clone(),
          args: std::iter::once(inner)
            .chain(args[1..].iter().cloned())
            .collect::<Vec<_>>()
            .into(),
        },
      }
    }
    other => other.clone(),
  }
}

/// The points of a `Polygon[…]` face, `None` if `expr` isn't a `List` of
/// points (e.g. the `Polygon[points -> holes]` form, left untouched).
fn face_points(expr: &Expr) -> Option<Vec<Vec3>> {
  let Expr::List(items) = expr else {
    return None;
  };
  items.iter().map(as_point).collect()
}

fn polygon_expr(points: &[Vec3]) -> Expr {
  call1(
    "Polygon",
    Expr::List(
      points
        .iter()
        .copied()
        .map(vec3_to_expr)
        .collect::<Vec<_>>()
        .into(),
    ),
  )
}

/// Replace every `Polygon[…]` face found anywhere inside `expr` — already
/// resolved to concrete points via [`resolve_transforms`] — with `f`
/// applied to its points, leaving everything else unchanged.
fn map_faces(expr: &Expr, f: &dyn Fn(&[Vec3]) -> Expr) -> Expr {
  match expr {
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|e| map_faces(e, f))
        .collect::<Vec<_>>()
        .into(),
    ),
    Expr::FunctionCall { name, args }
      if name == "Polygon" && args.len() == 1 =>
    {
      match face_points(&args[0]) {
        Some(pts) if pts.len() >= 3 => f(&pts),
        _ => expr.clone(),
      }
    }
    other => other.clone(),
  }
}

/// Corner-cut each face's polygon to `ratio` of its edge length: every
/// vertex is replaced by the two points that `ratio` of the way along its
/// adjoining edges, turning an *n*-gon face into a 2*n*-gon face.
fn truncate_face(pts: &[Vec3], ratio: f64) -> Expr {
  let n = pts.len();
  let mut cut = Vec::with_capacity(2 * n);
  for i in 0..n {
    let prev = pts[(i + n - 1) % n];
    let cur = pts[i];
    let next = pts[(i + 1) % n];
    cut.push(lerp(cur, prev, ratio));
    cut.push(lerp(cur, next, ratio));
  }
  polygon_expr(&cut)
}

/// Replace each face's polygon by a pyramid with the polygon as its base:
/// the apex sits at `ratio` times the face's centroid (so `ratio = 1`
/// degenerates to the face itself, and `ratio < 1` pulls the apex toward —
/// or past — the solid's center, the "concave" stellation the reference
/// documents).
fn stellate_face(pts: &[Vec3], ratio: f64) -> Expr {
  let n = pts.len();
  let sum = pts.iter().fold([0.0; 3], |acc, &p| add(acc, p));
  let centroid = scale(sum, 1.0 / n as f64);
  let apex = scale(centroid, ratio);
  let triangles: Vec<Expr> = (0..n)
    .map(|i| polygon_expr(&[pts[i], pts[(i + 1) % n], apex]))
    .collect();
  Expr::List(triangles.into())
}

/// `` PolyhedronOperations`Truncate[expr, ratio] ``.
pub(crate) fn truncate(expr: &Expr, ratio: f64) -> Expr {
  map_faces(&resolve_transforms(expr), &|pts| truncate_face(pts, ratio))
}

/// `` PolyhedronOperations`Stellate[expr, ratio] ``.
pub(crate) fn stellate(expr: &Expr, ratio: f64) -> Expr {
  map_faces(&resolve_transforms(expr), &|pts| stellate_face(pts, ratio))
}
