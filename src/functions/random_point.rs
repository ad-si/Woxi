//! `RandomPoint[reg]`, `RandomPoint[reg, n]`, `RandomPoint[reg, {n1, n2, …}]`:
//! points distributed uniformly in a region with respect to its own measure
//! (length for curves, area for surfaces, volume for solids).
//!
//! Primitive regions are sampled directly. Boolean combinations are sampled by
//! rejection: draw from the bounding box of their parts and keep the points
//! `RegionMember` accepts.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::unevaluated;
use crate::syntax::{Expr, expr_to_string};
use rand::RngExt;

type Point = Vec<f64>;

/// How to draw one point from a region.
enum Sampler {
  /// Pick one of the points.
  Points(Vec<Point>),
  /// Pick a part with probability proportional to its weight, then sample it.
  Weighted(Vec<(f64, Sampler)>),
  /// Uniform on a segment.
  Segment(Point, Point),
  /// Uniform in the simplex spanned by the vertices.
  Simplex(Vec<Point>),
  /// Uniform in the axis-aligned box.
  Box(Point, Point),
  /// `origin + s*u + t*v` for uniform s, t.
  Parallelogram(Point, Point, Point),
  /// Elliptical disk sector: centre, radii, angle range.
  Disk(Point, [f64; 2], [f64; 2]),
  /// Elliptical arc: centre, radii, angle range.
  Arc(Point, [f64; 2], [f64; 2]),
  /// Solid ball or sphere surface in any dimension.
  Ball(Point, f64, bool),
  /// Points of `bounds` the region expression contains.
  Rejection(Expr, Point, Point),
  /// `ParametricRegion`: parameters drawn from their box and accepted with
  /// probability proportional to the area element `sqrt(det(JᵀJ))`, so
  /// the image is uniform in its own measure.
  Parametric(Parametrization),
}

struct Parametrization {
  map: Expr,
  vars: Vec<String>,
  ranges: Vec<(f64, f64)>,
  /// Conditions beyond the parameter box, checked for each sample.
  residual: Vec<Expr>,
  /// An upper bound on the area element over the parameter box.
  max_element: f64,
}

impl Parametrization {
  fn substitute(&self, e: &Expr, params: &[f64]) -> Expr {
    let mut e = e.clone();
    for (v, x) in self.vars.iter().zip(params) {
      e = crate::functions::plot::substitute_var(&e, v, &Expr::Real(*x));
    }
    e
  }

  fn admits(&self, params: &[f64]) -> bool {
    self.residual.iter().all(|c| {
      matches!(
        evaluate_expr_to_expr(&self.substitute(c, params)),
        Ok(Expr::Identifier(ref s)) if s == "True"
      )
    })
  }

  fn eval(&self, params: &[f64]) -> Option<Point> {
    to_point(&evaluate_expr_to_expr(&self.substitute(&self.map, params)).ok()?)
  }

  /// `sqrt(det(JᵀJ))` by central differences.
  fn element(&self, params: &[f64]) -> Option<f64> {
    let k = params.len();
    let mut jac: Vec<Point> = Vec::with_capacity(k);
    for i in 0..k {
      let (a, b) = self.ranges[i];
      let h = 1e-6 * (b - a).abs().max(1e-12);
      let mut lo = params.to_vec();
      let mut hi = params.to_vec();
      lo[i] -= h;
      hi[i] += h;
      let (pl, ph) = (self.eval(&lo)?, self.eval(&hi)?);
      jac.push(
        ph.iter()
          .zip(&pl)
          .map(|(x, y)| (x - y) / (2.0 * h))
          .collect(),
      );
    }
    // The Gram determinant of the columns is the squared k-volume of the
    // simplex they span times (k!)².
    let mut pts = vec![vec![0.0; jac[0].len()]];
    pts.extend(jac);
    let factorial: f64 = (1..=k).map(|i| i as f64).product();
    Some(simplex_measure(&pts) * factorial)
  }
}

/// Bounds `(lower, upper, pure)` a comparison chain such as `0 <= v <= 1`,
/// `Inequality[0, LessEqual, v, Less, 1]`, `v >= 0` or `0 <= u <= v <= 1`
/// places on parameter `var`. `pure` is false when the chain also involves
/// other parameters, so it still has to be checked per sample.
fn condition_bounds(cond: &Expr, var: &str) -> Option<(f64, f64, bool)> {
  use crate::syntax::ComparisonOp;
  // Normalise to an operand chain with `<`/`<=` (or `>`/`>=`) between them.
  let (operands, ascending): (Vec<&Expr>, bool) = match cond {
    Expr::Comparison {
      operands,
      operators,
    } => {
      let asc = operators
        .iter()
        .all(|o| matches!(o, ComparisonOp::Less | ComparisonOp::LessEqual));
      let desc = operators.iter().all(|o| {
        matches!(o, ComparisonOp::Greater | ComparisonOp::GreaterEqual)
      });
      if !(asc || desc) {
        return None;
      }
      (operands.iter().collect(), asc)
    }
    Expr::FunctionCall { name, args } => match name.as_str() {
      "Less" | "LessEqual" => (args.iter().collect(), true),
      "Greater" | "GreaterEqual" => (args.iter().collect(), false),
      "Inequality" if args.len() % 2 == 1 => {
        let ops: Vec<&str> = args
          .iter()
          .skip(1)
          .step_by(2)
          .map(|o| match o {
            Expr::Identifier(s) => s.as_str(),
            _ => "",
          })
          .collect();
        let asc = ops.iter().all(|o| matches!(*o, "Less" | "LessEqual"));
        let desc = ops.iter().all(|o| matches!(*o, "Greater" | "GreaterEqual"));
        if !(asc || desc) {
          return None;
        }
        (args.iter().step_by(2).collect(), asc)
      }
      _ => return None,
    },
    _ => return None,
  };
  let pos = operands
    .iter()
    .position(|e| matches!(e, Expr::Identifier(s) if s == var))?;
  // By transitivity the tightest numeric operands on either side bound
  // `var`, even when the chain also relates it to other parameters.
  let nums: Vec<Option<f64>> =
    operands.iter().map(|e| try_eval_to_f64(e)).collect();
  let left = nums[..pos].iter().flatten().copied();
  let right = nums[pos + 1..].iter().flatten().copied();
  // Ascending: left operands are lower bounds, right ones upper bounds;
  // a descending chain swaps the roles.
  let (lo, hi) = if ascending {
    (left.reduce(f64::max), right.reduce(f64::min))
  } else {
    (right.reduce(f64::max), left.reduce(f64::min))
  };
  let pure = nums
    .iter()
    .enumerate()
    .all(|(i, n)| i == pos || n.is_some());
  Some((
    lo.unwrap_or(f64::NEG_INFINITY),
    hi.unwrap_or(f64::INFINITY),
    pure,
  ))
}

fn conjuncts(cond: &Expr) -> Vec<&Expr> {
  match cond {
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().flat_map(conjuncts).collect()
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::And,
      left,
      right,
    } => {
      let mut v = conjuncts(left);
      v.extend(conjuncts(right));
      v
    }
    other => vec![other],
  }
}

/// `ParametricRegion[{f, cond}, {u, v, …}]` (the canonical form): the
/// parameter box comes from the comparison chains in `cond`; any other
/// conjuncts are tested per sample.
fn parametric_sampler(region_args: &[Expr]) -> Option<Sampler> {
  let [Expr::List(spec), Expr::List(var_list)] = region_args else {
    return None;
  };
  let [map, cond] = spec.as_slice() else {
    return None;
  };
  let vars: Vec<String> = var_list
    .iter()
    .map(|v| match v {
      Expr::Identifier(s) => Some(s.clone()),
      _ => None,
    })
    .collect::<Option<_>>()?;
  let mut ranges = vec![(f64::NEG_INFINITY, f64::INFINITY); vars.len()];
  let mut residual = Vec::new();
  for c in conjuncts(cond) {
    let mut settled = false;
    for (v, range) in vars.iter().zip(ranges.iter_mut()) {
      if let Some((lo, hi, pure)) = condition_bounds(c, v) {
        range.0 = range.0.max(lo);
        range.1 = range.1.min(hi);
        settled |= pure;
      }
    }
    if !settled {
      residual.push(c.clone());
    }
  }
  if ranges
    .iter()
    .any(|(a, b)| !(a.is_finite() && b.is_finite()) || a > b)
  {
    return None;
  }
  if vars.is_empty() || !matches!(map, Expr::List(_)) {
    return None;
  }
  let mut param = Parametrization {
    map: map.clone(),
    vars,
    ranges,
    residual,
    max_element: 0.0,
  };
  // Bound the area element on a grid (with a safety margin).
  let k = param.vars.len();
  let steps = match k {
    1 => 200,
    2 => 30,
    _ => 8,
  };
  let mut idx = vec![0usize; k];
  let mut max = 0.0_f64;
  loop {
    let params: Vec<f64> = (0..k)
      .map(|i| {
        let (a, b) = param.ranges[i];
        a + (b - a) * (idx[i] as f64 + 0.5) / steps as f64
      })
      .collect();
    max = max.max(param.element(&params)?);
    let mut d = 0;
    while d < k {
      idx[d] += 1;
      if idx[d] < steps {
        break;
      }
      idx[d] = 0;
      d += 1;
    }
    if d == k {
      break;
    }
  }
  if !(max.is_finite() && max > 0.0) {
    return None;
  }
  param.max_element = 1.25 * max;
  Some(Sampler::Parametric(param))
}

pub fn random_point_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let call_str = || expr_to_string(&unevaluated("RandomPoint", args));
  let dims: Option<Vec<usize>> = match args.get(1) {
    None => None,
    Some(spec) => {
      if let Some(d) = parse_dims(spec) {
        Some(d)
      } else {
        crate::emit_message(&format!(
          "RandomPoint::array: The array dimensions {} given in position 2 of {} should be a list of non-negative machine-sized integers giving the dimensions for the result.",
          expr_to_string(spec),
          call_str()
        ));
        return Ok(unevaluated("RandomPoint", args));
      }
    }
  };
  let Some(sampler) = region_sampler(&args[0]) else {
    crate::emit_message(&format!(
      "RandomPoint::creg: The first argument {} is expected to be a parameter-free region.",
      expr_to_string(&args[0])
    ));
    return Ok(unevaluated("RandomPoint", args));
  };
  let mut failed = false;
  let mut draw = || {
    if let Some(p) = sample(&sampler) {
      Expr::List(p.into_iter().map(Expr::Real).collect())
    } else {
      failed = true;
      Expr::List(Vec::new().into())
    }
  };
  let result = match dims {
    None => draw(),
    Some(d) => build_array(&d, &mut draw),
  };
  if failed {
    return Ok(unevaluated("RandomPoint", args));
  }
  Ok(result)
}

fn parse_dims(spec: &Expr) -> Option<Vec<usize>> {
  let dim = |e: &Expr| match e {
    Expr::Integer(k) if *k >= 0 && *k <= i64::MAX as i128 => Some(*k as usize),
    _ => None,
  };
  match spec {
    Expr::List(items) => items.iter().map(dim).collect(),
    other => dim(other).map(|k| vec![k]),
  }
}

fn build_array(dims: &[usize], draw: &mut dyn FnMut() -> Expr) -> Expr {
  match dims.split_first() {
    None => draw(),
    Some((&k, rest)) => {
      Expr::List((0..k).map(|_| build_array(rest, draw)).collect())
    }
  }
}

fn to_point(e: &Expr) -> Option<Point> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      items.iter().map(try_eval_to_f64).collect()
    }
    _ => None,
  }
}

fn to_points(e: &Expr) -> Option<Vec<Point>> {
  match e {
    Expr::List(items) if !items.is_empty() => {
      let pts: Option<Vec<Point>> = items.iter().map(to_point).collect();
      let pts = pts?;
      let d = pts[0].len();
      pts.iter().all(|p| p.len() == d).then_some(pts)
    }
    _ => None,
  }
}

fn sub(a: &[f64], b: &[f64]) -> Point {
  a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn norm(v: &[f64]) -> f64 {
  v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// k-dimensional volume of the simplex spanned by `pts` (k + 1 points), via
/// the Gram determinant of its edge vectors.
fn simplex_measure(pts: &[Point]) -> f64 {
  let k = pts.len() - 1;
  let edges: Vec<Point> = pts[1..].iter().map(|p| sub(p, &pts[0])).collect();
  let mut gram: Vec<Vec<f64>> = (0..k)
    .map(|i| {
      (0..k)
        .map(|j| edges[i].iter().zip(&edges[j]).map(|(a, b)| a * b).sum())
        .collect()
    })
    .collect();
  // Gaussian elimination for the determinant.
  let mut det = 1.0;
  for c in 0..k {
    let Some(p) = (c..k).max_by(|&a, &b| {
      gram[a][c]
        .abs()
        .partial_cmp(&gram[b][c].abs())
        .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
      return 0.0;
    };
    if gram[p][c] == 0.0 {
      return 0.0;
    }
    gram.swap(p, c);
    det *= gram[c][c];
    for r in c + 1..k {
      let f = gram[r][c] / gram[c][c];
      for j in c..k {
        gram[r][j] -= f * gram[c][j];
      }
    }
  }
  let factorial: f64 = (1..=k).map(|i| i as f64).product();
  det.abs().sqrt() / factorial
}

fn polyline_sampler(pts: &[Point]) -> Option<Sampler> {
  let parts: Vec<(f64, Sampler)> = pts
    .windows(2)
    .map(|w| {
      (
        norm(&sub(&w[1], &w[0])),
        Sampler::Segment(w[0].clone(), w[1].clone()),
      )
    })
    .collect();
  weighted(parts)
}

fn weighted(parts: Vec<(f64, Sampler)>) -> Option<Sampler> {
  if parts.is_empty() || parts.iter().map(|p| p.0).sum::<f64>() <= 0.0 {
    return None;
  }
  Some(Sampler::Weighted(parts))
}

fn polygon_sampler(pts: &[Point]) -> Option<Sampler> {
  if pts.len() < 3 {
    return None;
  }
  if pts[0].len() != 2 {
    // A planar polygon in 3D: fan triangulation from the first vertex.
    let parts = (1..pts.len() - 1)
      .map(|i| {
        let tri = vec![pts[0].clone(), pts[i].clone(), pts[i + 1].clone()];
        (simplex_measure(&tri), Sampler::Simplex(tri))
      })
      .collect();
    return weighted(parts);
  }
  let flat: Vec<(f64, f64)> = pts.iter().map(|p| (p[0], p[1])).collect();
  let ring: Vec<usize> = (0..flat.len()).collect();
  let tri =
    crate::functions::polygon_holes::triangulate_with_holes(&flat, &ring, &[]);
  let parts = tri
    .triangles
    .iter()
    .map(|t| {
      let v: Vec<Point> = t.iter().map(|&i| pts[i].clone()).collect();
      (simplex_measure(&v), Sampler::Simplex(v))
    })
    .collect();
  weighted(parts)
}

fn radii(e: Option<&Expr>) -> Option<[f64; 2]> {
  match e {
    None => Some([1.0, 1.0]),
    Some(Expr::List(rs)) if rs.len() == 2 => {
      Some([try_eval_to_f64(&rs[0])?, try_eval_to_f64(&rs[1])?])
    }
    Some(r) => {
      let r = try_eval_to_f64(r)?;
      Some([r, r])
    }
  }
}

fn angles(e: Option<&Expr>) -> Option<[f64; 2]> {
  match e {
    None => Some([0.0, std::f64::consts::TAU]),
    Some(Expr::List(a)) if a.len() == 2 => {
      Some([try_eval_to_f64(&a[0])?, try_eval_to_f64(&a[1])?])
    }
    _ => None,
  }
}

/// Bounding box of what `sampler` can produce.
fn bounds(sampler: &Sampler) -> Option<(Point, Point)> {
  let of_points = |pts: &[Point]| {
    let d = pts.first()?.len();
    let lo = (0..d)
      .map(|i| pts.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min))
      .collect();
    let hi = (0..d)
      .map(|i| pts.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max))
      .collect();
    Some((lo, hi))
  };
  match sampler {
    Sampler::Points(p) | Sampler::Simplex(p) => of_points(p),
    Sampler::Segment(a, b) => of_points(&[a.clone(), b.clone()]),
    Sampler::Parametric(_) => None,
    Sampler::Box(lo, hi) | Sampler::Rejection(_, lo, hi) => {
      Some((lo.clone(), hi.clone()))
    }
    Sampler::Parallelogram(o, u, v) => {
      let add = |a: &[f64], b: &[f64]| -> Point {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
      };
      let ou = add(o, u);
      let ov = add(o, v);
      let ouv = add(&ou, v);
      of_points(&[o.clone(), ou, ov, ouv])
    }
    Sampler::Disk(c, r, _) | Sampler::Arc(c, r, _) => Some((
      vec![c[0] - r[0], c[1] - r[1]],
      vec![c[0] + r[0], c[1] + r[1]],
    )),
    Sampler::Ball(c, r, _) => Some((
      c.iter().map(|x| x - r).collect(),
      c.iter().map(|x| x + r).collect(),
    )),
    Sampler::Weighted(parts) => {
      let boxes: Option<Vec<(Point, Point)>> =
        parts.iter().map(|(_, s)| bounds(s)).collect();
      let boxes = boxes?;
      let corners: Vec<Point> =
        boxes.into_iter().flat_map(|(lo, hi)| [lo, hi]).collect();
      of_points(&corners)
    }
  }
}

fn region_sampler(region: &Expr) -> Option<Sampler> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  let arg = |i: usize| args.get(i);
  match name.as_str() {
    "Point" => match to_points(arg(0)?) {
      Some(pts) => Some(Sampler::Points(pts)),
      None => Some(Sampler::Points(vec![to_point(arg(0)?)?])),
    },
    "Line" => {
      if let Some(pts) = to_points(arg(0)?) {
        polyline_sampler(&pts)
      } else {
        // A list of polylines.
        let Expr::List(lines) = arg(0)? else {
          return None;
        };
        let parts: Option<Vec<(f64, Sampler)>> = lines
          .iter()
          .map(|l| {
            let pts = to_points(l)?;
            let len: f64 =
              pts.windows(2).map(|w| norm(&sub(&w[1], &w[0]))).sum();
            Some((len, polyline_sampler(&pts)?))
          })
          .collect();
        weighted(parts?)
      }
    }
    // A union of 1D pieces, each a segment of the line.
    "Interval" => {
      let parts: Option<Vec<(f64, Sampler)>> = args
        .iter()
        .map(|piece| {
          let ends = to_point(piece)?;
          let [a, b] = ends.as_slice() else {
            return None;
          };
          Some(((b - a).abs(), Sampler::Segment(vec![*a], vec![*b])))
        })
        .collect();
      weighted(parts?)
    }
    "Triangle" | "Simplex" => {
      let pts = match arg(0) {
        None if name == "Triangle" => {
          vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]
        }
        Some(Expr::Integer(k)) if name == "Simplex" && *k >= 0 => {
          // Simplex[n]: the standard n-simplex.
          let n = *k as usize;
          (0..=n)
            .map(|i| {
              (0..n).map(|j| if j + 1 == i { 1.0 } else { 0.0 }).collect()
            })
            .collect()
        }
        Some(e) => to_points(e)?,
        None => return None,
      };
      Some(Sampler::Simplex(pts))
    }
    "Polygon" => {
      if let Some(pts) = to_points(arg(0)?) {
        polygon_sampler(&pts)
      } else {
        let Expr::List(polys) = arg(0)? else {
          return None;
        };
        let parts: Option<Vec<(f64, Sampler)>> = polys
          .iter()
          .map(|p| {
            let s = polygon_sampler(&to_points(p)?)?;
            let Sampler::Weighted(ref tris) = s else {
              return None;
            };
            let area = tris.iter().map(|t| t.0).sum();
            Some((area, s))
          })
          .collect();
        weighted(parts?)
      }
    }
    "Rectangle" | "Cuboid" => {
      let dim = if name == "Rectangle" { 2 } else { 3 };
      let lo = match arg(0) {
        None => vec![0.0; dim],
        Some(e) => to_point(e)?,
      };
      let hi = match arg(1) {
        None => lo.iter().map(|x| x + 1.0).collect(),
        Some(e) => to_point(e)?,
      };
      (lo.len() == hi.len()).then_some(Sampler::Box(lo, hi))
    }
    "Parallelogram" => {
      let o = match arg(0) {
        None => vec![0.0, 0.0],
        Some(e) => to_point(e)?,
      };
      let vs = match arg(1) {
        None => vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        Some(e) => to_points(e)?,
      };
      (vs.len() == 2)
        .then(|| Sampler::Parallelogram(o, vs[0].clone(), vs[1].clone()))
    }
    "Disk" | "Circle" => {
      let c = match arg(0) {
        None => vec![0.0, 0.0],
        Some(e) => to_point(e)?,
      };
      if c.len() != 2 {
        return None;
      }
      let r = radii(arg(1))?;
      let a = angles(arg(2))?;
      Some(if name == "Disk" {
        Sampler::Disk(c, r, a)
      } else {
        Sampler::Arc(c, r, a)
      })
    }
    "Ball" | "Sphere" => {
      let c = match arg(0) {
        None => vec![0.0, 0.0, 0.0],
        Some(e) => to_point(e)?,
      };
      let r = match arg(1) {
        None => 1.0,
        Some(e) => try_eval_to_f64(e)?,
      };
      Some(Sampler::Ball(c, r, name == "Sphere"))
    }
    "ParametricRegion" => {
      // Canonicalise `ParametricRegion[f, {{u, a, b}, …}]` first.
      match &evaluate_expr_to_expr(region).ok()? {
        Expr::FunctionCall { name: h, args: a } if h == "ParametricRegion" => {
          parametric_sampler(a)
        }
        _ => None,
      }
    }
    "BooleanRegion" => {
      let Expr::List(parts) = arg(1)? else {
        return None;
      };
      let boxes: Option<Vec<(Point, Point)>> = parts
        .iter()
        .map(|p| region_sampler(p).and_then(|s| bounds(&s)))
        .collect();
      let corners: Vec<Point> =
        boxes?.into_iter().flat_map(|(lo, hi)| [lo, hi]).collect();
      let (lo, hi) = bounds(&Sampler::Points(corners))?;
      Some(Sampler::Rejection(region.clone(), lo, hi))
    }
    "RegionUnion"
    | "RegionIntersection"
    | "RegionDifference"
    | "RegionSymmetricDifference" => {
      // These evaluate to a BooleanRegion when their parts are regions.
      let evaluated = evaluate_expr_to_expr(region).ok()?;
      match &evaluated {
        Expr::FunctionCall { name: h, .. } if h == "BooleanRegion" => {
          region_sampler(&evaluated)
        }
        _ => None,
      }
    }
    _ => None,
  }
}

fn uniform(rng: &mut dyn rand::Rng) -> f64 {
  rng.random::<f64>()
}

fn gaussian(rng: &mut dyn rand::Rng) -> f64 {
  // Box–Muller.
  let u1 = 1.0 - uniform(rng);
  let u2 = uniform(rng);
  (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn sample(sampler: &Sampler) -> Option<Point> {
  if let Sampler::Rejection(region, lo, hi) = sampler {
    for _ in 0..100_000 {
      let p: Point = crate::with_rng(|rng| {
        lo.iter()
          .zip(hi)
          .map(|(a, b)| a + (b - a) * uniform(rng))
          .collect()
      });
      let member = evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "RegionMember".into(),
        args: vec![
          region.clone(),
          Expr::List(p.iter().copied().map(Expr::Real).collect()),
        ]
        .into(),
      })
      .ok()?;
      match member {
        Expr::Identifier(ref s) if s == "True" => return Some(p),
        Expr::Identifier(ref s) if s == "False" => {}
        _ => return None,
      }
    }
    return None;
  }
  if let Sampler::Parametric(param) = sampler {
    for _ in 0..100_000 {
      let (params, u): (Vec<f64>, f64) = crate::with_rng(|rng| {
        let params = param
          .ranges
          .iter()
          .map(|(a, b)| a + (b - a) * uniform(rng))
          .collect();
        (params, uniform(rng))
      });
      if u * param.max_element <= param.element(&params)?
        && param.admits(&params)
      {
        return param.eval(&params);
      }
    }
    return None;
  }
  crate::with_rng(|rng| Some(sample_direct(sampler, rng)))
}

fn sample_direct(sampler: &Sampler, rng: &mut dyn rand::Rng) -> Point {
  match sampler {
    Sampler::Points(pts) => pts[rng.random_range(0..pts.len())].clone(),
    Sampler::Weighted(parts) => {
      let total: f64 = parts.iter().map(|p| p.0).sum();
      let mut x = uniform(rng) * total;
      for (w, s) in parts {
        if x < *w {
          return sample_direct(s, rng);
        }
        x -= w;
      }
      sample_direct(&parts[parts.len() - 1].1, rng)
    }
    Sampler::Segment(a, b) => {
      let t = uniform(rng);
      a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
    }
    Sampler::Simplex(pts) => {
      // Dirichlet(1, …, 1) barycentric weights.
      let w: Vec<f64> =
        pts.iter().map(|_| -(1.0 - uniform(rng)).ln()).collect();
      let s: f64 = w.iter().sum();
      (0..pts[0].len())
        .map(|i| pts.iter().zip(&w).map(|(p, wi)| p[i] * wi).sum::<f64>() / s)
        .collect()
    }
    Sampler::Box(lo, hi) => lo
      .iter()
      .zip(hi)
      .map(|(a, b)| a + (b - a) * uniform(rng))
      .collect(),
    Sampler::Parallelogram(o, u, v) => {
      let (s, t) = (uniform(rng), uniform(rng));
      (0..o.len()).map(|i| o[i] + s * u[i] + t * v[i]).collect()
    }
    Sampler::Disk(c, r, a) => {
      let rho = uniform(rng).sqrt();
      let th = a[0] + (a[1] - a[0]) * uniform(rng);
      vec![c[0] + r[0] * rho * th.cos(), c[1] + r[1] * rho * th.sin()]
    }
    Sampler::Arc(c, r, a) => {
      // Uniform in arc length: accept an angle with probability
      // proportional to the speed of the parametrisation there.
      let max_speed = r[0].abs().max(r[1].abs());
      loop {
        let th = a[0] + (a[1] - a[0]) * uniform(rng);
        let speed = (r[0] * th.sin()).hypot(r[1] * th.cos());
        if max_speed == 0.0 || uniform(rng) * max_speed <= speed {
          return vec![c[0] + r[0] * th.cos(), c[1] + r[1] * th.sin()];
        }
      }
    }
    Sampler::Ball(c, r, surface) => {
      let d = c.len();
      let mut g: Vec<f64> = (0..d).map(|_| gaussian(rng)).collect();
      let n = norm(&g).max(f64::MIN_POSITIVE);
      let scale = if *surface {
        r / n
      } else {
        r * uniform(rng).powf(1.0 / d as f64) / n
      };
      for (gi, ci) in g.iter_mut().zip(c) {
        *gi = ci + *gi * scale;
      }
      g
    }
    Sampler::Rejection(..) | Sampler::Parametric(_) => {
      unreachable!("handled in sample")
    }
  }
}

/// `ParametricRegion[f, {{u, a, b}, v, …}]` → the canonical
/// `ParametricRegion[{f, a <= u <= b && …}, {u, v, …}]`. Parameters that `f`
/// does not mention are dropped, a reversed numeric range gives
/// `EmptyRegion[dim]`, and an infinite end leaves a one-sided bound.
pub fn parametric_region_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let keep = || Ok(unevaluated("ParametricRegion", args));
  let [first, Expr::List(specs)] = args else {
    return keep();
  };
  // `{f, cond}` (already conditioned) or a bare `f`.
  let (map, mut conds) = match first {
    Expr::List(items)
      if items.len() == 2
        && matches!(items[0], Expr::List(_))
        && !matches!(items[1], Expr::List(_)) =>
    {
      (items[0].clone(), vec![items[1].clone()])
    }
    Expr::List(items) if !items.is_empty() => (first.clone(), Vec::new()),
    _ => return keep(),
  };
  if specs.iter().all(|s| matches!(s, Expr::Identifier(_))) {
    return keep();
  }
  let dim = match &map {
    Expr::List(items) => items.len(),
    _ => 0,
  };
  let mentions = |v: &Expr| {
    matches!(
      crate::functions::predicate_ast::free_q_ast(&[map.clone(), v.clone()]),
      Ok(Expr::Identifier(ref s)) if s == "False"
    )
  };
  let is = |e: &Expr, s: &str| matches!(e, Expr::Identifier(x) if x == s);
  let is_neg_infinity = |e: &Expr| {
    matches!(
      evaluate_expr_to_expr(&Expr::FunctionCall {
        name: "SameQ".into(),
        args: vec![e.clone(), Expr::FunctionCall {
          name: "DirectedInfinity".into(),
          args: vec![Expr::Integer(-1)].into(),
        }]
        .into(),
      }),
      Ok(Expr::Identifier(ref s)) if s == "True"
    )
  };
  let call = |name: &str, a: Vec<Expr>| Expr::FunctionCall {
    name: name.into(),
    args: a.into(),
  };
  let mut vars = Vec::new();
  for spec in specs {
    let (v, range) = match spec {
      Expr::Identifier(_) => (spec.clone(), None),
      Expr::List(parts) if parts.len() == 3 => {
        (parts[0].clone(), Some((parts[1].clone(), parts[2].clone())))
      }
      _ => return keep(),
    };
    if !mentions(&v) {
      continue;
    }
    if let Some((a, b)) = range {
      if let (Some(x), Some(y)) = (try_eval_to_f64(&a), try_eval_to_f64(&b))
        && x > y
      {
        return Ok(call("EmptyRegion", vec![Expr::Integer(dim as i128)]));
      }
      let lower_open = is_neg_infinity(&a);
      let upper_open = is(&b, "Infinity")
        || matches!(&b, Expr::FunctionCall { name, args } if name == "DirectedInfinity" && matches!(args.as_slice(), [Expr::Integer(1)]));
      match (lower_open, upper_open) {
        (true, true) => {}
        // Evaluated so they take the operator form (`s <= 1`).
        (true, false) => conds.push(evaluate_expr_to_expr(&call(
          "LessEqual",
          vec![v.clone(), b],
        ))?),
        (false, true) => conds.push(evaluate_expr_to_expr(&call(
          "GreaterEqual",
          vec![v.clone(), a],
        ))?),
        (false, false) => conds.push(call(
          "Inequality",
          vec![
            a,
            Expr::Identifier("LessEqual".into()),
            v.clone(),
            Expr::Identifier("LessEqual".into()),
            b,
          ],
        )),
      }
    }
    vars.push(v);
  }
  let region_spec = match conds.len() {
    0 => map,
    1 => Expr::List(vec![map, conds.pop().unwrap()].into()),
    _ => Expr::List(vec![map, call("And", conds)].into()),
  };
  Ok(call(
    "ParametricRegion",
    vec![region_spec, Expr::List(vars.into())],
  ))
}
