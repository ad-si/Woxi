//! The legacy `ComputationalGeometry`` package's planar algorithms, following
//! the package's own procedures step by step so that every index list comes
//! out in the order wolframscript gives (the order is part of the result:
//! a hull starts at a particular corner, an adjacency list at a particular
//! neighbour).

use super::*;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;

/// `Chop`'s tolerance in the package's machine-precision code:
/// `10.^(1. - MachinePrecision)`.
fn machine_chop_size() -> f64 {
  10f64.powf(1.0 - 15.954589770191003)
}

fn chop(x: f64, delta: f64) -> f64 {
  if x.abs() < delta { 0.0 } else { x }
}

fn sign(x: f64) -> i32 {
  if x > 0.0 {
    1
  } else if x < 0.0 {
    -1
  } else {
    0
  }
}

/// The points of a `{{x, y}, …}` list as machine numbers.
pub(crate) fn read_points(arg: &Expr) -> Option<Vec<[f64; 2]>> {
  let Expr::List(items) = arg else {
    return None;
  };
  if items.is_empty() {
    return None;
  }
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

/// The package's `polarSort`: point indices (1-based) ordered by the angle
/// and squared distance about the centroid, coincident points dropped.
fn polar_sort(coords: &[[f64; 2]]) -> Vec<usize> {
  let n = coords.len();
  let mut sum = [0.0, 0.0];
  for p in coords {
    sum[0] += p[0];
    sum[1] += p[1];
  }
  let centre = [sum[0] / n as f64, sum[1] / n as f64];
  let mut keyed: Vec<(f64, f64, usize)> = coords
    .iter()
    .enumerate()
    .map(|(i, p)| {
      let d = [p[0] - centre[0], p[1] - centre[1]];
      let angle = if d[0] == 0.0 && d[1] == 0.0 {
        0.0
      } else {
        d[1].atan2(d[0])
      };
      (angle, d[0] * d[0] + d[1] * d[1], i + 1)
    })
    .collect();
  keyed.sort_by(|a, b| {
    a.0
      .total_cmp(&b.0)
      .then(a.1.total_cmp(&b.1))
      .then(a.2.cmp(&b.2))
  });
  let mut out = vec![keyed[0].2];
  let mut previous = (keyed[0].0, keyed[0].1);
  for k in &keyed[1..] {
    if (k.0, k.1) != previous {
      previous = (k.0, k.1);
      out.push(k.2);
    }
  }
  out
}

/// The package's `collinearQ`: every point on the line through the first
/// two (to `Chop`'s default tolerance).
fn collinear_q(points: &[[f64; 2]]) -> bool {
  let ([x1, y1], [x2, y2]) = (points[0], points[1]);
  let (a, b, c) = (y1 - y2, x2 - x1, x1 * y2 - x2 * y1);
  points[2..]
    .iter()
    .all(|p| chop(a * p[0] + b * p[1] + c, 1e-10) == 0.0)
}

/// `ComputationalGeometry`ConvexHull[points]`: Graham's scan as the package
/// runs it. The hull starts at the first rightmost point of the polar order
/// (which is the lowest of the rightmost ones) and runs counter-clockwise;
/// with `AllPoints -> True` (the default) points on a hull edge are kept.
/// Returns 1-based indices, or None when the points are not numeric pairs.
pub(crate) fn convex_hull(coords: &[[f64; 2]], all_points: bool) -> Vec<usize> {
  let polar = polar_sort(coords);
  if polar.len() < coords.len() {
    crate::emit_message(&format!(
      "ConvexHull::duplicated: {} duplicated point(s) ignored.",
      coords.len() - polar.len()
    ));
  }
  if polar.len() <= 2 {
    return polar;
  }
  let ordered: Vec<[f64; 2]> = polar.iter().map(|&i| coords[i - 1]).collect();
  if collinear_q(&ordered) {
    let mut by_position: Vec<([f64; 2], usize)> =
      ordered.iter().copied().zip(polar.iter().copied()).collect();
    by_position.sort_by(|a, b| {
      a.0[0]
        .total_cmp(&b.0[0])
        .then(a.0[1].total_cmp(&b.0[1]))
        .then(a.1.cmp(&b.1))
    });
    return by_position.into_iter().map(|(_, i)| i).collect();
  }
  graham_scan(&polar, coords, all_points)
}

fn graham_scan(
  polar: &[usize],
  coords: &[[f64; 2]],
  all_points: bool,
) -> Vec<usize> {
  let n = coords.len();
  // Circular doubly linked list over the polar order (1-based; 0 unused).
  let mut next = vec![0usize; n + 1];
  let mut prev = vec![0usize; n + 1];
  let m = polar.len();
  for k in 0..m {
    next[polar[k]] = polar[(k + 1) % m];
    prev[polar[k]] = polar[(k + m - 1) % m];
  }
  let mut start = polar[0];
  let mut max_x = f64::NEG_INFINITY;
  for &i in polar {
    if coords[i - 1][0] > max_x {
      max_x = coords[i - 1][0];
      start = i;
    }
  }
  let chop_size = machine_chop_size();
  let at = |i: usize| coords[i - 1];
  let mut scan_complete = false;
  let mut v = start;
  let last = prev[start];
  while start != next[v] || !scan_complete {
    if last == next[v] {
      scan_complete = true;
    }
    let next_of_v = next[v];
    let next_next_of_v = next[next_of_v];
    let ([x1, y1], [x2, y2], [x3, y3]) =
      (at(v), at(next_of_v), at(next_next_of_v));
    let area_sign = sign(chop(
      (x2 - x3) * (y2 - y1) + (x2 - x1) * (y3 - y2),
      chop_size,
    ));
    let left = match area_sign {
      1 => true,
      -1 => false,
      _ if (x1 - x2) * (x2 - x3) + (y1 - y2) * (y2 - y3) < 0.0 => {
        next_of_v == start
      }
      _ => all_points,
    };
    if left {
      v = next_of_v;
    } else {
      next[v] = next_next_of_v;
      prev[next_next_of_v] = v;
      v = prev[v];
    }
  }
  let mut out = vec![start];
  let mut v = next[start];
  while v != start {
    out.push(v);
    v = next[v];
  }
  out
}

/// The `AllPoints` option of `ConvexHull` (default True).
fn all_points_option(opts: &[Expr]) -> Option<bool> {
  let mut value = true;
  for opt in opts {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return None;
    };
    if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ComputationalGeometry`AllPoints" || n == "AllPoints")
    {
      value =
        matches!(replacement.as_ref(), Expr::Identifier(t) if t == "True");
    }
  }
  Some(value)
}

/// `ComputationalGeometry`ConvexHull[{{x, y}, …}, opts]`.
pub fn convex_hull_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`ConvexHull", args);
  let (Some(coords), Some(all_points)) = (
    args.first().and_then(read_points),
    all_points_option(&args[1.min(args.len())..]),
  ) else {
    return Ok(original());
  };
  Ok(index_list(&convex_hull(&coords, all_points)))
}

pub(crate) fn index_list(indices: &[usize]) -> Expr {
  Expr::List(
    indices
      .iter()
      .map(|&i| Expr::Integer(i as i128))
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `ComputationalGeometry`ConvexHullArea[points]`: the package sums, over
/// the hull's edges, the triangles the edge makes with the centroid of all
/// the points (`Abs[Det[...]]/2`), exact for exact points.
pub fn convex_hull_area_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`ConvexHullArea", args);
  let [points @ Expr::List(items)] = args else {
    return Ok(original());
  };
  let Some(coords) = read_points(points) else {
    return Ok(original());
  };
  let hull = convex_hull(&coords, true);
  let mean = evaluate_expr_to_expr(&call1("Mean", points.clone()))?;
  let mut cycle = hull.clone();
  cycle.push(hull[0]);
  let mut terms = Vec::with_capacity(hull.len());
  for pair in cycle.windows(2) {
    let triangle = Expr::List(
      vec![
        items[pair[0] - 1].clone(),
        items[pair[1] - 1].clone(),
        mean.clone(),
      ]
      .into(),
    );
    // signedvolume[m] = Det[Prepend[Transpose[m], {1, 1, 1}]]
    let matrix = call(
      "Prepend",
      vec![
        call1("Transpose", triangle),
        Expr::List(vec![Expr::Integer(1); 3].into()),
      ],
    );
    terms.push(call1("Abs", call1("Det", matrix)));
  }
  evaluate_expr_to_expr(&call(
    "Times",
    vec![
      call("Plus", terms),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    ],
  ))
}

// ---------------------------------------------------------------------------
// DelaunayTriangulation: the package's divide-and-conquer triangulation.
// Point labels are 1-based throughout, as in the package; `val[i - 1]` is
// the adjacency list of point `i`.
// ---------------------------------------------------------------------------

/// The package's `SignOfArea`: the turn `a → b → c` makes (1 left, -1
/// right, 0 straight). Points that share their abscissa or their ordinate
/// are straight without arithmetic.
fn sign_of_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i32 {
  let same = |k: usize| a[k] - b[k] == 0.0 && a[k] - c[k] == 0.0;
  if same(0) || same(1) {
    return 0;
  }
  let area = (b[0] - c[0]) * (b[1] - a[1]) + (b[0] - a[0]) * (c[1] - b[1]);
  sign(chop(area, machine_chop_size()))
}

fn succ(i: usize, m: usize) -> usize {
  i % m + 1
}

fn pred(i: usize, m: usize) -> usize {
  (i + 2 * m - 2) % m + 1
}

/// The package's `CollinearQ`: every cyclic triple of the set is straight.
fn collinear_set(set: &[[f64; 2]]) -> bool {
  let n = set.len();
  (1..=n).all(|i| {
    let next = succ(i, n);
    sign_of_area(set[i - 1], set[next - 1], set[succ(next, n) - 1]) == 0
  })
}

/// The package's `PolarSort` of labelled points about their centroid (or a
/// given centre): by the angle in [0, 2 Pi), then by distance.
fn polar_sort_labelled(
  l: &[(usize, [f64; 2])],
  centre: Option<[f64; 2]>,
) -> Vec<(usize, [f64; 2])> {
  let origin = centre.unwrap_or_else(|| {
    let mut sum = [0.0, 0.0];
    for (_, p) in l {
      sum[0] += p[0];
      sum[1] += p[1];
    }
    [sum[0] / l.len() as f64, sum[1] / l.len() as f64]
  });
  let mut keyed: Vec<(f64, f64, (usize, [f64; 2]))> = l
    .iter()
    .map(|&(label, p)| {
      let c = [p[0] - origin[0], p[1] - origin[1]];
      let angle = if c[0] == 0.0 && c[1] == 0.0 {
        0.0
      } else if c[1] >= 0.0 {
        c[1].atan2(c[0])
      } else {
        c[1].atan2(c[0]) + std::f64::consts::TAU
      };
      (angle, c[0] * c[0] + c[1] * c[1], (label, p))
    })
    .collect();
  keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
  keyed.into_iter().map(|k| k.2).collect()
}

/// The package's `CircleMemberQ[{a, b, c, d}]`: whether `d` lies strictly
/// inside the circle through `a`, `b`, `c`, from the expanded determinant.
fn circle_member_q(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
  let ([xa, ya], [xb, yb], [xc, yc], [xd, yd]) = (a, b, c, d);
  let terms = [
    -(xb * xb * xc * ya),
    xb * xc * xc * ya,
    xb * xb * xd * ya,
    -(xc * xc * xd * ya),
    -(xb * xd * xd * ya),
    xc * xd * xd * ya,
    xa * xa * xc * yb,
    -(xa * xc * xc * yb),
    -(xa * xa * xd * yb),
    xc * xc * xd * yb,
    xa * xd * xd * yb,
    -(xc * xd * xd * yb),
    xc * ya * ya * yb,
    -(xd * ya * ya * yb),
    -(xc * ya * yb * yb),
    xd * ya * yb * yb,
    -(xa * xa * xb * yc),
    xa * xb * xb * yc,
    xa * xa * xd * yc,
    -(xb * xb * xd * yc),
    -(xa * xd * xd * yc),
    xb * xd * xd * yc,
    -(xb * ya * ya * yc),
    xd * ya * ya * yc,
    xa * yb * yb * yc,
    -(xd * yb * yb * yc),
    xb * ya * yc * yc,
    -(xd * ya * yc * yc),
    -(xa * yb * yc * yc),
    xd * yb * yc * yc,
    xa * xa * xb * yd,
    -(xa * xb * xb * yd),
    -(xa * xa * xc * yd),
    xb * xb * xc * yd,
    xa * xc * xc * yd,
    -(xb * xc * xc * yd),
    xb * ya * ya * yd,
    -(xc * ya * ya * yd),
    -(xa * yb * yb * yd),
    xc * yb * yb * yd,
    xa * yc * yc * yd,
    -(xb * yc * yc * yd),
    -(xb * ya * yd * yd),
    xc * ya * yd * yd,
    xa * yb * yd * yd,
    -(xc * yb * yd * yd),
    -(xa * yc * yd * yd),
    xb * yc * yd * yd,
  ];
  let det: f64 = terms.iter().sum();
  det < 0.0
}

struct Triangulation {
  val: Vec<Vec<usize>>,
  hull: Vec<usize>,
  leftmost: usize,
  rightmost: usize,
}

/// A position-of helper: the 1-based position of `x` in `list`.
fn position_of(list: &[usize], x: usize) -> Option<usize> {
  list.iter().position(|&y| y == x).map(|p| p + 1)
}

fn rotate_until(
  hull: &mut Vec<usize>,
  ok: impl Fn(&[usize]) -> bool,
) -> Option<()> {
  for _ in 0..=hull.len() {
    if ok(hull) {
      return Some(());
    }
    hull.rotate_left(1);
  }
  None
}

/// `Del[s]`: the triangulation of points already sorted by abscissa.
fn del(s: &[[f64; 2]]) -> Option<Triangulation> {
  match s.len() {
    0 | 1 => None,
    2 => {
      let ([x1, y1], [x2, y2]) = (s[0], s[1]);
      let hull = if x1 > x2 || (x1 - x2 == 0.0 && y1 < y2) {
        vec![1, 2]
      } else {
        vec![2, 1]
      };
      Some(Triangulation {
        val: vec![vec![2], vec![1]],
        hull,
        leftmost: 2,
        rightmost: 1,
      })
    }
    3 => {
      let sorted =
        polar_sort_labelled(&[(1, s[0]), (2, s[1]), (3, s[2])], None);
      let mut hull: Vec<usize> = sorted.iter().map(|p| p.0).collect();
      let max_x = sorted.iter().map(|p| p.1[0]).fold(f64::MIN, f64::max);
      let min_y = sorted
        .iter()
        .filter(|p| p.1[0] - max_x == 0.0)
        .map(|p| p.1[1])
        .fold(f64::MAX, f64::min);
      rotate_until(&mut hull, |h| s[h[0] - 1] == [max_x, min_y])?;
      let val = (1..=3)
        .map(|v| {
          let p = position_of(&hull, v).unwrap();
          let mut rotated = hull.clone();
          rotated.rotate_left(p % 3);
          rotated.pop();
          rotated
        })
        .collect();
      let (h2, h3) = (s[hull[1] - 1], s[hull[2] - 1]);
      let leftmost = if h2[0] < h3[0] || (h2[0] - h3[0] == 0.0 && h2[1] > h3[1])
      {
        2
      } else {
        3
      };
      Some(Triangulation {
        val,
        hull,
        leftmost,
        rightmost: 1,
      })
    }
    n => {
      if collinear_set(s) {
        return None;
      }
      let leng1 = n.div_ceil(2);
      let leng2 = n - leng1;
      let (s1, s2) = s.split_at(leng1);
      let collinear1 = leng1 > 2 && collinear_set(s1);
      let collinear2 = leng2 > 2 && collinear_set(s2);
      let (val1, val2, lowertan, uppertan, hull, leftptr, rightptr);
      if collinear1 || collinear2 {
        let mut h = convex_hull(s, true);
        let len = h.len();
        rotate_until(&mut h, |h| h[0] < leng1 + 1 && h[len - 1] > leng1)?;
        let lpart1 = h.iter().filter(|&&i| i < leng1 + 1).count();
        uppertan = (h[0], h[len - 1]);
        rotate_until(&mut h, |h| h[0] > leng1 && h[len - 1] < leng1 + 1)?;
        let lpart2 = h.iter().filter(|&&i| i > leng1).count();
        lowertan = (h[len - 1], h[0]);
        let xs: Vec<f64> = h.iter().map(|&i| s[i - 1][0]).collect();
        let min_x = xs.iter().copied().fold(f64::MAX, f64::min);
        let max_x = xs.iter().copied().fold(f64::MIN, f64::max);
        leftptr = xs.iter().position(|&x| x == min_x)? + 1;
        rightptr = xs.iter().position(|&x| x == max_x)? + 1;
        // A collinear part is a chain along its (sorted) points.
        let chain = |first: usize, last: usize, forward: bool| {
          let mut v = vec![vec![first + 1]];
          for k in first + 1..last {
            v.push(if forward {
              vec![k + 1, k - 1]
            } else {
              vec![k - 1, k + 1]
            });
          }
          v.push(vec![last - 1]);
          v
        };
        val1 = if collinear1 {
          chain(1, leng1, lpart1 == 1 || (lpart1 > 1 && lowertan.0 != 1))
        } else {
          del(s1)?.val
        };
        val2 = if collinear2 {
          chain(
            leng1 + 1,
            leng1 + leng2,
            lpart2 == 1 || (lpart2 > 1 && lowertan.1 == leng1 + 1),
          )
        } else {
          offset(del(s2)?.val, leng1)
        };
        hull = h;
      } else {
        let t1 = del(s1)?;
        let t2 = del(s2)?;
        let hull1 = t1.hull;
        let hull2: Vec<usize> = t2.hull.iter().map(|i| i + leng1).collect();
        let (len1, len2) = (hull1.len(), hull2.len());
        let ((ltl, ltr), (utl, utr)) =
          common_tangents(s, &hull1, &hull2, t1.rightmost, t2.leftmost)?;
        lowertan = (hull1[ltl - 1], hull2[ltr - 1]);
        uppertan = (hull1[utl - 1], hull2[utr - 1]);
        let modp = |a: isize, m: usize| a.rem_euclid(m as isize) as usize;
        let num1 = modp(ltl as isize - utl as isize, len1) + 1;
        let mut part1 = hull1.clone();
        part1.rotate_left((utl - 1) % len1);
        part1.truncate(num1);
        leftptr = modp(t1.leftmost as isize - utl as isize, len1) + 1;
        let num2 = modp(utr as isize - ltr as isize, len2) + 1;
        let mut part2 = hull2.clone();
        part2.rotate_left((ltr - 1) % len2);
        part2.truncate(num2);
        rightptr = modp(t2.rightmost as isize - ltr as isize, len2) + 1 + num1;
        hull = part1.into_iter().chain(part2).collect();
        val1 = t1.val;
        val2 = offset(t2.val, leng1);
      }
      let mut joined = val1;
      joined.extend(val2);
      let val = merge_del(s, joined, lowertan, uppertan)?;
      Some(Triangulation {
        val,
        hull,
        leftmost: leftptr,
        rightmost: rightptr,
      })
    }
  }
}

fn offset(val: Vec<Vec<usize>>, by: usize) -> Vec<Vec<usize>> {
  val
    .into_iter()
    .map(|l| l.into_iter().map(|i| i + by).collect())
    .collect()
}

/// The package's `CommonTangents`: positions in the two hulls of the lower
/// and the upper common tangent.
#[allow(clippy::type_complexity)]
fn common_tangents(
  set: &[[f64; 2]],
  hull_left: &[usize],
  hull_right: &[usize],
  rightmost_left: usize,
  leftmost_right: usize,
) -> Option<((usize, usize), (usize, usize))> {
  let (len_l, len_r) = (hull_left.len(), hull_right.len());
  let pt = |h: &[usize], k: usize| set[h[k - 1] - 1];
  let limit = 4 * (len_l + len_r) + 8;

  let (mut x, mut y) = (rightmost_left, leftmost_right);
  let (mut predx, mut succy) = (pred(x, len_l), succ(y, len_r));
  let mut lower = None;
  for _ in 0..limit {
    let soa_r =
      sign_of_area(pt(hull_left, x), pt(hull_right, y), pt(hull_right, succy));
    if soa_r <= 0 {
      if soa_r == 0 {
        return comtan(set, hull_left, hull_right);
      }
      (succy, y) = (succ(succy, len_r), succy);
    } else {
      let soa_l =
        sign_of_area(pt(hull_left, predx), pt(hull_left, x), pt(hull_right, y));
      if soa_l <= 0 {
        if soa_l == 0 {
          return comtan(set, hull_left, hull_right);
        }
        (predx, x) = (pred(predx, len_l), predx);
      } else {
        lower = Some((x, y));
        break;
      }
    }
  }
  let lower = lower?;

  let (mut x, mut y) = (rightmost_left, leftmost_right);
  let (mut succx, mut predy) = (succ(x, len_l), pred(y, len_r));
  for _ in 0..limit {
    let soa_r =
      sign_of_area(pt(hull_left, x), pt(hull_right, y), pt(hull_right, predy));
    if soa_r >= 0 {
      if soa_r == 0 {
        return comtan(set, hull_left, hull_right);
      }
      (predy, y) = (pred(predy, len_r), predy);
    } else {
      let soa_l =
        sign_of_area(pt(hull_left, succx), pt(hull_left, x), pt(hull_right, y));
      if soa_l >= 0 {
        if soa_l == 0 {
          return comtan(set, hull_left, hull_right);
        }
        (succx, x) = (succ(succx, len_l), succx);
      } else {
        return Some((lower, (x, y)));
      }
    }
  }
  None
}

/// The package's `comtan`: the tangents read off the hull of both hulls.
#[allow(clippy::type_complexity)]
fn comtan(
  set: &[[f64; 2]],
  hull_left: &[usize],
  hull_right: &[usize],
) -> Option<((usize, usize), (usize, usize))> {
  let joined: Vec<[f64; 2]> = hull_left
    .iter()
    .chain(hull_right)
    .map(|&i| set[i - 1])
    .collect();
  let mut hull = convex_hull(&joined, true);
  let leng1 = hull_left.len();
  let len = hull.len();
  rotate_until(&mut hull, |h| h[0] < leng1 + 1 && h[len - 1] > leng1)?;
  let upper = (hull[0], hull[len - 1] - leng1);
  rotate_until(&mut hull, |h| h[0] > leng1 && h[len - 1] < leng1 + 1)?;
  let lower = (hull[len - 1], hull[0] - leng1);
  Some((lower, upper))
}

/// The package's `MergeDel`: stitch the two halves together from the lower
/// to the upper tangent, deleting the edges that stop being Delaunay.
fn merge_del(
  s: &[[f64; 2]],
  mut merge: Vec<Vec<usize>>,
  lowertan: (usize, usize),
  uppertan: (usize, usize),
) -> Option<Vec<Vec<usize>>> {
  let pt = |i: usize| s[i - 1];
  let (mut l0, mut r0) = lowertan;
  let mut lengr0 = merge[r0 - 1].len() + 1;
  let mut lengl0 = merge[l0 - 1].len() + 1;
  let mut r0ptr = 1;
  let mut l0ptr = lengr0;
  merge[l0 - 1].insert(r0ptr - 1, r0);
  merge[r0 - 1].insert(l0ptr - 1, l0);
  let get =
    |m: &Vec<Vec<usize>>, v: usize, k: usize| m[v - 1].get(k - 1).copied();
  let mut guard = 0usize;
  let guard_limit = 64 * s.len() * s.len() + 64;
  while (l0, r0) != uppertan {
    guard += 1;
    if guard > guard_limit {
      return None;
    }
    let mut left_turn = true;
    let mut right_turn = true;

    let mut r1ptr = pred(l0ptr, lengr0);
    let mut r1 = get(&merge, r0, r1ptr)?;
    if sign_of_area(pt(l0), pt(r0), pt(r1)) == 1 {
      let mut r2ptr = pred(r1ptr, lengr0);
      let mut r2 = get(&merge, r0, r2ptr)?;
      while merge[r1 - 1].contains(&r2)
        && circle_member_q(pt(r1), pt(r0), pt(r2), pt(l0))
      {
        let pos = position_of(&merge[r1 - 1], r0)?;
        merge[r0 - 1].remove(r1ptr - 1);
        merge[r1 - 1].remove(pos - 1);
        if l0ptr > r1ptr {
          l0ptr -= 1;
        }
        if r2ptr > r1ptr {
          r2ptr -= 1;
        }
        lengr0 -= 1;
        r1ptr = r2ptr;
        r2ptr = pred(r1ptr, lengr0);
        r1 = get(&merge, r0, r1ptr)?;
        r2 = get(&merge, r0, r2ptr)?;
      }
    } else {
      left_turn = false;
    }

    let mut l1ptr = succ(r0ptr, lengl0);
    let mut l1 = get(&merge, l0, l1ptr)?;
    if sign_of_area(pt(r0), pt(l0), pt(l1)) == -1 {
      let mut l2ptr = succ(l1ptr, lengl0);
      let mut l2 = get(&merge, l0, l2ptr)?;
      while merge[l1 - 1].contains(&l2)
        && circle_member_q(pt(l0), pt(l1), pt(l2), pt(r0))
      {
        let pos = position_of(&merge[l1 - 1], l0)?;
        merge[l0 - 1].remove(l1ptr - 1);
        merge[l1 - 1].remove(pos - 1);
        if r0ptr > l1ptr {
          r0ptr -= 1;
        }
        if l2ptr > l1ptr {
          l2ptr -= 1;
        }
        lengl0 -= 1;
        l1ptr = l2ptr;
        l2ptr = succ(l1ptr, lengl0);
        l1 = get(&merge, l0, l1ptr)?;
        l2 = get(&merge, l0, l2ptr)?;
      }
    } else {
      right_turn = false;
    }

    let connect_r0_to_l1 = if !left_turn {
      true
    } else if !right_turn {
      false
    } else {
      circle_member_q(pt(r1), pt(l0), pt(r0), pt(l1))
    };
    if connect_r0_to_l1 {
      let l1val = merge[l1 - 1].clone();
      lengr0 += 1;
      lengl0 = l1val.len() + 1;
      let new_l0 = get(&merge, l0, l1ptr)?;
      r0ptr = succ(position_of(&l1val, l0)?, lengl0);
      l0 = new_l0;
    } else {
      let r1val = merge[r1 - 1].clone();
      lengl0 += 1;
      lengr0 = r1val.len() + 1;
      let new_r0 = get(&merge, r0, r1ptr)?;
      l0ptr = position_of(&r1val, r0)?;
      r0ptr = succ(r0ptr, lengl0);
      r0 = new_r0;
    }

    if r0ptr > merge[l0 - 1].len() + 1 || l0ptr > merge[r0 - 1].len() + 1 {
      return None;
    }
    merge[l0 - 1].insert(r0ptr - 1, r0);
    merge[r0 - 1].insert(l0ptr - 1, l0);
  }
  Some(merge)
}

/// The package's `Distinct` and abscissa sort: the distinct points, each
/// with the position of its first occurrence, by abscissa — ties by
/// decreasing ordinate.
fn distinct_sorted(points: &[[f64; 2]]) -> Vec<(usize, [f64; 2])> {
  let mut unique: Vec<(usize, [f64; 2])> = Vec::new();
  for (i, p) in points.iter().enumerate() {
    if !unique.iter().any(|(_, q)| q == p) {
      unique.push((i + 1, *p));
    }
  }
  // `Union` orders them by abscissa; the package's sort then puts points of
  // one abscissa in decreasing order of the ordinate.
  unique
    .sort_by(|a, b| a.1[0].total_cmp(&b.1[0]).then(b.1[1].total_cmp(&a.1[1])));
  unique
}

/// The triangulation as the package returns it: `{label, neighbours}` per
/// distinct point (by label) and the hull, counter-clockwise from the
/// lowest rightmost point. None for collinear or too few points.
#[allow(clippy::type_complexity)]
pub(crate) fn delaunay_triangulation(
  points: &[[f64; 2]],
) -> Result<Option<(Vec<(usize, Vec<usize>)>, Vec<usize>)>, ()> {
  let distinct = distinct_sorted(points);
  if distinct.len() < 3 {
    return Ok(None);
  }
  let labels: Vec<usize> = distinct.iter().map(|d| d.0).collect();
  let coords: Vec<[f64; 2]> = distinct.iter().map(|d| d.1).collect();
  let Some(t) = del(&coords) else {
    return Err(());
  };
  let mut delval: Vec<(usize, Vec<usize>)> = t
    .val
    .iter()
    .enumerate()
    .map(|(v, adj)| (labels[v], adj.iter().map(|&a| labels[a - 1]).collect()))
    .collect();
  delval.sort_by_key(|d| d.0);
  let hull = t.hull.iter().map(|&h| labels[h - 1]).collect();
  Ok(Some((delval, hull)))
}

fn delaunay_value(delval: &[(usize, Vec<usize>)]) -> Expr {
  Expr::List(
    delval
      .iter()
      .map(|(v, adj)| {
        Expr::List(vec![Expr::Integer(*v as i128), index_list(adj)].into())
      })
      .collect::<Vec<_>>()
      .into(),
  )
}

/// `ComputationalGeometry`DelaunayTriangulation[points, Hull -> h]`.
pub fn delaunay_triangulation_ast(
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original =
    || unevaluated("ComputationalGeometry`DelaunayTriangulation", args);
  let Some(points) = args.first().and_then(read_points) else {
    return Ok(original());
  };
  let mut want_hull = false;
  for opt in &args[1..] {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "ComputationalGeometry`Hull" || n == "Hull") =>
      {
        want_hull =
          matches!(replacement.as_ref(), Expr::Identifier(t) if t == "True");
      }
      Expr::Rule { .. } => {}
      _ => return Ok(original()),
    }
  }
  match delaunay_triangulation(&points) {
    Ok(Some((delval, hull))) => {
      let value = delaunay_value(&delval);
      Ok(if want_hull {
        Expr::List(vec![value, index_list(&hull)].into())
      } else {
        value
      })
    }
    Ok(None) => Ok(original()),
    Err(()) => {
      crate::emit_message(&format!(
        "DelaunayTriangulation::collin: Points {} are collinear.",
        expr_to_string(&args[0])
      ));
      Ok(id_expr("$Failed"))
    }
  }
}

// ---------------------------------------------------------------------------
// VoronoiDiagram and DiagramPlot. The package computes these from the points
// as given — exact points give exact vertices — so the arithmetic is done on
// expressions; the Delaunay adjacency underneath comes from the machine
// triangulation above.
// ---------------------------------------------------------------------------

fn ev(e: Expr) -> Result<Expr, InterpreterError> {
  evaluate_expr_to_expr(&e)
}

fn pair_expr(x: Expr, y: Expr) -> Expr {
  Expr::List(vec![x, y].into())
}

fn coords_of(p: &Expr) -> Option<(Expr, Expr)> {
  match p {
    Expr::List(c) if c.len() == 2 => Some((c[0].clone(), c[1].clone())),
    _ => None,
  }
}

/// `SignOfArea` on expressions: the sign of the (exact or machine) turn.
fn sign_of_area_expr(
  a: &Expr,
  b: &Expr,
  c: &Expr,
) -> Result<i32, InterpreterError> {
  let (Some((x1, y1)), Some((x2, y2)), Some((x3, y3))) =
    (coords_of(a), coords_of(b), coords_of(c))
  else {
    return Ok(0);
  };
  let f = |e: &Expr| try_eval_to_f64(e);
  let all_equal = |a: &Expr, b: &Expr, c: &Expr| matches!((f(a), f(b), f(c)), (Some(a), Some(b), Some(c)) if a == b && b == c);
  if all_equal(&x1, &x2, &x3) || all_equal(&y1, &y2, &y3) {
    return Ok(0);
  }
  let area = ev(plus(vec![
    times(vec![
      minus2(x2.clone(), x3.clone()),
      minus2(y2.clone(), y1.clone()),
    ]),
    times(vec![minus2(x2, x1), minus2(y3, y2)]),
  ]))?;
  let value = try_eval_to_f64(&area).unwrap_or(0.0);
  let inexact = crate::syntax::contains_inexact(&area);
  Ok(sign(if inexact {
    chop(value, machine_chop_size())
  } else {
    value
  }))
}

/// The package's `circlecenter`: the circumcentre from the dot products.
fn circle_center(
  p1: &Expr,
  p2: &Expr,
  p3: &Expr,
) -> Result<Expr, InterpreterError> {
  let dot = |a: &Expr, b: &Expr, c: &Expr| {
    call(
      "Dot",
      vec![minus2(a.clone(), b.clone()), minus2(c.clone(), b.clone())],
    )
  };
  let d1 = ev(dot(p3, p1, p2))?;
  let d2 = ev(dot(p3, p2, p1))?;
  let d3 = ev(dot(p1, p3, p2))?;
  let c1 = ev(times(vec![d2.clone(), d3.clone()]))?;
  let c2 = ev(times(vec![d3, d1.clone()]))?;
  let c3 = ev(times(vec![d1, d2]))?;
  let ca = ev(plus(vec![c1.clone(), c2.clone(), c3.clone()]))?;
  ev(times2(
    plus(vec![
      times(vec![plus2(c2.clone(), c3.clone()), p1.clone()]),
      times(vec![plus2(c3, c1.clone()), p2.clone()]),
      times(vec![plus2(c1, c2), p3.clone()]),
    ]),
    pow2(times(vec![Expr::Integer(2), ca]), Expr::Integer(-1)),
  ))
}

/// The package's `raytail`: the far end of the ray from `head` that
/// bisects the hull edge `{e1, e2}`, on the right of `e1 → head`.
fn ray_tail(
  head: &Expr,
  e1: &Expr,
  e2: &Expr,
) -> Result<Expr, InterpreterError> {
  let (Some((xh, yh)), Some((x1, y1)), Some((x2, y2))) =
    (coords_of(head), coords_of(e1), coords_of(e2))
  else {
    return Ok(head.clone());
  };
  let sqrt_sum_sq = |dx: Expr, dy: Expr| {
    call1(
      "Sqrt",
      plus2(pow2(dx, Expr::Integer(2)), pow2(dy, Expr::Integer(2))),
    )
  };
  let mut d0 = ev(sqrt_sum_sq(
    minus2(x1.clone(), x2.clone()),
    minus2(y1.clone(), y2.clone()),
  ))?;
  let half = |a: Expr, b: Expr| {
    times(vec![
      plus2(a, b),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    ])
  };
  let xm = ev(half(x1.clone(), x2.clone()))?;
  let ym = ev(half(y1.clone(), y2.clone()))?;
  let mid = pair_expr(xm.clone(), ym.clone());
  let from_head = sign_of_area_expr(e1, &mid, head)? == -1;
  if !from_head {
    d0 = ev(plus2(
      d0,
      sqrt_sum_sq(
        minus2(xh.clone(), xm.clone()),
        minus2(yh.clone(), ym.clone()),
      ),
    ))?;
  }
  let dy = ev(minus2(y1.clone(), y2.clone()))?;
  let vertical = match try_eval_to_f64(&dy) {
    Some(v) if crate::syntax::contains_inexact(&dy) => {
      chop(v, machine_chop_size()) == 0.0
    }
    Some(v) => v == 0.0,
    None => false,
  };
  let out: Vec<Expr> = if vertical {
    vec![
      pair_expr(xh.clone(), ev(minus2(yh.clone(), d0.clone()))?),
      pair_expr(xh.clone(), ev(plus2(yh.clone(), d0.clone()))?),
    ]
  } else {
    let m = ev(neg(div2(minus2(x1.clone(), x2.clone()), dy)))?;
    // c = ym - m xm - yh
    let c = ev(plus(vec![
      ym.clone(),
      neg(times(vec![m.clone(), xm.clone()])),
      neg(yh.clone()),
    ]))?;
    let one_m2 = plus2(Expr::Integer(1), pow2(m.clone(), Expr::Integer(2)));
    let b = ev(div2(
      minus2(times(vec![m.clone(), c.clone()]), xh.clone()),
      one_m2.clone(),
    ))?;
    let root = ev(call1(
      "Sqrt",
      minus2(
        pow2(b.clone(), Expr::Integer(2)),
        div2(
          plus(vec![
            pow2(xh.clone(), Expr::Integer(2)),
            pow2(c, Expr::Integer(2)),
            neg(pow2(d0, Expr::Integer(2))),
          ]),
          one_m2,
        ),
      ),
    ))?;
    let mut out = Vec::with_capacity(2);
    for x in [
      ev(plus2(neg(b.clone()), root.clone()))?,
      ev(minus2(neg(b), root))?,
    ] {
      let y = ev(plus2(
        times(vec![m.clone(), minus2(x.clone(), xm.clone())]),
        ym.clone(),
      ))?;
      out.push(pair_expr(x, y));
    }
    out
  };
  let turn = if from_head {
    sign_of_area_expr(e1, head, &out[0])?
  } else {
    sign_of_area_expr(e1, &mid, &out[0])?
  };
  Ok(if turn == -1 {
    out[0].clone()
  } else {
    out[1].clone()
  })
}

/// An element of a Voronoi polygon before relabelling: a Delaunay triangle
/// (a true vertex) or a hull edge with the triangle it borders (a quasi
/// vertex, the start of a ray).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum VorItem {
  // Ordered first, like the package's shorter lists.
  Quasi([usize; 2], [usize; 3]),
  True([usize; 3]),
}

fn sorted3(mut t: [usize; 3]) -> [usize; 3] {
  t.sort_unstable();
  t
}

fn sorted2(a: usize, b: usize) -> [usize; 2] {
  [a.min(b), a.max(b)]
}

/// `VoronoiDiagram[set]` for a set that is not collinear: the vertices
/// (true ones, then the rays) and the polygon of each Delaunay vertex.
fn voronoi_general(
  items: &[Expr],
  delval: &[(usize, Vec<usize>)],
  hull: &[usize],
) -> Result<Option<Expr>, InterpreterError> {
  let lhull = hull.len();
  let mut vorval: Vec<Vec<VorItem>> = Vec::with_capacity(delval.len());
  for (delvert1, list) in delval {
    let k = list.len();
    let mut vorpoly: Vec<VorItem> = (0..k)
      .map(|i| VorItem::True(sorted3([*delvert1, list[i], list[(i + 1) % k]])))
      .collect();
    if let Some(hp) = position_of(hull, *delvert1) {
      let a = hull[(hp + lhull - 2) % lhull];
      let b = hull[hp % lhull];
      // The pair {a, b} of neighbours spans the hull gap.
      let Some(pos) = (0..k).find(|&i| list[i] == a && list[(i + 1) % k] == b)
      else {
        return Ok(None);
      };
      let triangle = |i: usize| match &vorpoly[i] {
        VorItem::True(t) => Some(*t),
        VorItem::Quasi(_, t) => Some(*t),
      };
      let before = triangle((pos + k - 1) % k).unwrap();
      let after = triangle((pos + 1) % k).unwrap();
      vorpoly[pos] = VorItem::Quasi(sorted2(*delvert1, a), before);
      vorpoly.insert(pos + 1, VorItem::Quasi(sorted2(*delvert1, b), after));
    }
    vorval.push(vorpoly);
  }
  let mut vorvertices: Vec<VorItem> =
    vorval.iter().flatten().cloned().collect();
  vorvertices.sort();
  vorvertices.dedup();
  if vorvertices.len() < lhull {
    return Ok(None);
  }
  let (quasi, truev) = vorvertices.split_at(lhull);
  let mut coordinates = Vec::with_capacity(truev.len());
  for v in truev {
    let VorItem::True(t) = v else {
      return Ok(None);
    };
    coordinates.push(circle_center(
      &items[t[0] - 1],
      &items[t[1] - 1],
      &items[t[2] - 1],
    )?);
  }
  // Distinct vertex coordinates, in canonical order.
  let union = ev(call1("Union", Expr::List(coordinates.clone().into())))?;
  let Expr::List(union) = &union else {
    return Ok(None);
  };
  let same = |a: &Expr, b: &Expr| -> Result<bool, InterpreterError> {
    let d = ev(minus2(a.clone(), b.clone()))?;
    Ok(match coords_of(&d) {
      Some((x, y)) => {
        try_eval_to_f64(&x) == Some(0.0) && try_eval_to_f64(&y) == Some(0.0)
      }
      None => false,
    })
  };
  let mut true_coordinates: Vec<Expr> = Vec::new();
  let mut rest: Vec<Expr> = union.to_vec();
  while !rest.is_empty() {
    let first = rest.remove(0);
    let mut kept = Vec::with_capacity(rest.len());
    for r in rest {
      if !same(&r, &first)? {
        kept.push(r);
      }
    }
    rest = kept;
    true_coordinates.push(first);
  }
  let ltrue = true_coordinates.len();
  let mut label_of_true: Vec<usize> = vec![0; truev.len()];
  for (k, c) in true_coordinates.iter().enumerate() {
    for (i, coord) in coordinates.iter().enumerate() {
      if label_of_true[i] == 0 && same(coord, c)? {
        label_of_true[i] = k + 1;
      }
    }
  }
  let label = |item: &VorItem| -> usize {
    match item {
      VorItem::Quasi(..) => quasi
        .iter()
        .position(|q| q == item)
        .map_or(0, |p| p + 1 + ltrue),
      VorItem::True(_) => truev
        .iter()
        .position(|t| t == item)
        .map_or(0, |p| label_of_true[p]),
    }
  };
  let mut cells = Vec::with_capacity(vorval.len());
  for (i, poly) in vorval.iter().enumerate() {
    let mut unique: Vec<usize> = Vec::new();
    for item in poly {
      let l = label(item);
      if unique.last() != Some(&l) {
        unique.push(l);
      }
    }
    cells.push(Expr::List(
      vec![Expr::Integer(delval[i].0 as i128), index_list(&unique)].into(),
    ));
  }
  let mut vertices = true_coordinates.clone();
  for q in quasi {
    let VorItem::Quasi(edge, triangle) = q else {
      return Ok(None);
    };
    let true_label = label(&VorItem::True(*triangle));
    let Some(ptr1) = position_of(hull, edge[0]) else {
      return Ok(None);
    };
    let edge = if edge[1] == hull[succ(ptr1, lhull) - 1] {
      *edge
    } else {
      [edge[1], edge[0]]
    };
    let Some(head) = true_label
      .checked_sub(1)
      .and_then(|l| true_coordinates.get(l))
    else {
      return Ok(None);
    };
    let tail = ray_tail(head, &items[edge[0] - 1], &items[edge[1] - 1])?;
    vertices.push(call("ComputationalGeometry`Ray", vec![head.clone(), tail]));
  }
  Ok(Some(Expr::List(
    vec![Expr::List(vertices.into()), Expr::List(cells.into())].into(),
  )))
}

/// `VoronoiDiagram` of collinear points: a pair of rays between each two
/// neighbouring distinct points.
fn voronoi_collinear(items: &[Expr]) -> Result<Expr, InterpreterError> {
  // Distinct points in sorted order, each with its first position.
  let sorted = ev(call1("Sort", Expr::List(items.to_vec().into())))?;
  let Expr::List(sorted) = &sorted else {
    return Ok(unevaluated("ComputationalGeometry`VoronoiDiagram", items));
  };
  let numeric = |p: &Expr| {
    coords_of(p)
      .and_then(|(x, y)| Some([try_eval_to_f64(&x)?, try_eval_to_f64(&y)?]))
  };
  let mut minset: Vec<Expr> = Vec::new();
  for p in sorted {
    if !minset.iter().any(|q| numeric(q) == numeric(p)) {
      minset.push(p.clone());
    }
  }
  if minset.len() == 1 {
    return Ok(Expr::List(
      vec![
        Expr::List(minset.into()),
        Expr::List(
          vec![Expr::List(vec![Expr::Integer(1), index_list(&[1])].into())]
            .into(),
        ),
      ]
      .into(),
    ));
  }
  let posns: Vec<usize> = minset
    .iter()
    .map(|m| {
      items
        .iter()
        .position(|p| numeric(p) == numeric(m))
        .map_or(1, |i| i + 1)
    })
    .collect();
  let mut rays = Vec::new();
  for w in posns.windows(2) {
    let (pt1, pt2) = (&items[w[0] - 1], &items[w[1] - 1]);
    let mid = ev(times(vec![
      plus2(pt1.clone(), pt2.clone()),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    ]))?;
    let dir = ev(call1("Reverse", minus2(pt2.clone(), pt1.clone())))?;
    for signs in [[1, -1], [-1, 1]] {
      let s = Expr::List(
        vec![Expr::Integer(signs[0]), Expr::Integer(signs[1])].into(),
      );
      let tail = ev(plus2(
        mid.clone(),
        times(vec![Expr::Real(1.5), s, dir.clone()]),
      ))?;
      rays.push(call("ComputationalGeometry`Ray", vec![mid.clone(), tail]));
    }
  }
  let n = minset.len();
  let mut cells = vec![Expr::List(
    vec![Expr::Integer(posns[0] as i128), index_list(&[1, 2])].into(),
  )];
  for (k, p) in posns[1..n - 1].iter().enumerate() {
    let b = 2 * (k + 1);
    cells.push(Expr::List(
      vec![
        Expr::Integer(*p as i128),
        index_list(&[b - 1, b, b + 1, b + 2]),
      ]
      .into(),
    ));
  }
  let b = 2 * (n - 1);
  cells.push(Expr::List(
    vec![Expr::Integer(posns[n - 1] as i128), index_list(&[b - 1, b])].into(),
  ));
  Ok(Expr::List(
    vec![Expr::List(rays.into()), Expr::List(cells.into())].into(),
  ))
}

/// The diagram `{vertices, cells}` of a numeric point list, or None when the
/// package gives up (too few distinct points).
fn voronoi_diagram(arg: &Expr) -> Result<Option<Expr>, InterpreterError> {
  let (Expr::List(items), Some(points)) = (arg, read_points(arg)) else {
    return Ok(None);
  };
  if items.len() == 1 {
    return Ok(Some(Expr::List(
      vec![
        arg.clone(),
        Expr::List(
          vec![Expr::List(vec![Expr::Integer(1), index_list(&[1])].into())]
            .into(),
        ),
      ]
      .into(),
    )));
  }
  if collinear_set(&points) {
    return voronoi_collinear(items).map(Some);
  }
  match delaunay_triangulation(&points) {
    Ok(Some((delval, hull))) => voronoi_general(items, &delval, &hull),
    _ => Ok(None),
  }
}

/// `ComputationalGeometry`VoronoiDiagram[points]`.
pub fn voronoi_diagram_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`VoronoiDiagram", args);
  let [arg] = args else {
    return Ok(original());
  };
  Ok(voronoi_diagram(arg)?.unwrap_or_else(original))
}

/// `ComputationalGeometry`DiagramPlot[points, opts]`: the site labels (or
/// points), the Voronoi vertices, the rays and the cell outlines, over a
/// square range around everything with a 5% margin.
pub fn diagram_plot_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = || unevaluated("ComputationalGeometry`DiagramPlot", args);
  let Some(points) = args.first() else {
    return Ok(original());
  };
  let Expr::List(items) = points else {
    return Ok(original());
  };
  let mut label_points = true;
  let mut graphics_opts = Vec::new();
  for opt in &args[1..] {
    let Expr::Rule {
      pattern,
      replacement,
    } = opt
    else {
      return Ok(original());
    };
    match pattern.as_ref() {
      Expr::Identifier(n)
        if n == "ComputationalGeometry`LabelPoints" || n == "LabelPoints" =>
      {
        label_points =
          matches!(replacement.as_ref(), Expr::Identifier(t) if t == "True");
      }
      Expr::Identifier(n)
        if n == "ComputationalGeometry`TrimPoints" || n == "TrimPoints" => {}
      _ => graphics_opts.push(opt.clone()),
    }
  }
  let Some(diagram) = voronoi_diagram(points)? else {
    return Ok(original());
  };
  let Expr::List(parts) = &diagram else {
    return Ok(original());
  };
  let (Expr::List(vorvert), Expr::List(val)) = (&parts[0], &parts[1]) else {
    return Ok(original());
  };
  let is_ray = |e: &Expr| matches!(e, Expr::FunctionCall { name, .. } if name == "Ray" || name == "ComputationalGeometry`Ray");
  let ray_parts = |e: &Expr| match e {
    Expr::FunctionCall { args, .. } => args.to_vec(),
    _ => vec![],
  };
  let site_of = |cell: &Expr| -> Option<(usize, Vec<usize>)> {
    let Expr::List(c) = cell else { return None };
    let label = expr_to_i128_opt(&c[0])? as usize;
    let Expr::List(vs) = &c[1] else { return None };
    let vs = vs
      .iter()
      .map(|v| expr_to_i128_opt(v).map(|i| i as usize))
      .collect::<Option<Vec<_>>>()?;
    Some((label, vs))
  };
  let cells: Vec<(usize, Vec<usize>)> =
    match val.iter().map(site_of).collect::<Option<Vec<_>>>() {
      Some(c) => c,
      None => return Ok(original()),
    };
  let point_size = call1("PointSize", Expr::Real(0.012));
  let centers: Vec<Expr> = if label_points {
    cells
      .iter()
      .map(|(l, _)| {
        call(
          "Text",
          vec![Expr::Integer(*l as i128), items[l - 1].clone()],
        )
      })
      .collect()
  } else {
    std::iter::once(point_size.clone())
      .chain(
        cells
          .iter()
          .map(|(l, _)| call1("Point", items[l - 1].clone())),
      )
      .collect()
  };
  let mut vertices = vec![point_size];
  let mut lines = vec![call1("Thickness", Expr::Real(0.003))];
  for v in vorvert {
    if is_ray(v) {
      lines.push(call1("Line", Expr::List(ray_parts(v).into())));
    } else {
      vertices.push(call1("Point", v.clone()));
    }
  }
  for (_, vs) in &cells {
    let mut list: Vec<Expr> =
      vs.iter().map(|&v| vorvert[v - 1].clone()).collect();
    if let Some(first_ray) = list.iter().position(is_ray) {
      let len = list.len().max(1);
      list.rotate_left((first_ray + 1) % len);
      list.retain(|e| !is_ray(e));
    } else if let Some(first) = list.first().cloned() {
      list.push(first);
    }
    lines.push(call1("Line", Expr::List(list.into())));
  }
  // absolutePlotRange over the sites, the vertices and the ray tails, in
  // the points' own arithmetic; only the 5% margin is a machine number.
  let mut xs = Vec::new();
  let mut ys = Vec::new();
  for p in items.iter().chain(vorvert.iter().map(|v| {
    if is_ray(v) {
      match v {
        Expr::FunctionCall { args, .. } if args.len() == 2 => &args[1],
        _ => v,
      }
    } else {
      v
    }
  })) {
    let Some((x, y)) = coords_of(p) else {
      return Ok(original());
    };
    xs.push(x);
    ys.push(y);
  }
  let extreme = |head: &str, v: &[Expr]| ev(call(head, v.to_vec()));
  let (min_x, max_x) = (extreme("Min", &xs)?, extreme("Max", &xs)?);
  let (min_y, max_y) = (extreme("Min", &ys)?, extreme("Max", &ys)?);
  let half = |hi: &Expr, lo: &Expr| {
    ev(times(vec![
      minus2(hi.clone(), lo.clone()),
      call("Rational", vec![Expr::Integer(1), Expr::Integer(2)]),
    ]))
  };
  let (xhalf, yhalf) = (half(&max_x, &min_x)?, half(&max_y, &min_y)?);
  let wider = try_eval_to_f64(&xhalf).zip(try_eval_to_f64(&yhalf));
  let Some((xh, yh)) = wider else {
    return Ok(original());
  };
  let ((x0, x1), (y0, y1)) = if xh > yh {
    let ymid = ev(plus2(min_y.clone(), yhalf.clone()))?;
    (
      (min_x, max_x),
      (
        ev(minus2(ymid.clone(), xhalf.clone()))?,
        ev(plus2(ymid, xhalf.clone()))?,
      ),
    )
  } else {
    let xmid = ev(plus2(min_x.clone(), xhalf.clone()))?;
    (
      (
        ev(minus2(xmid.clone(), yhalf.clone()))?,
        ev(plus2(xmid, yhalf.clone()))?,
      ),
      (min_y, max_y),
    )
  };
  let margin = |lo: &Expr, hi: &Expr| {
    ev(times(vec![
      Expr::Real(0.05 / 1.9),
      minus2(hi.clone(), lo.clone()),
    ]))
  };
  let (dx, dy) = (margin(&x0, &x1)?, margin(&y0, &y1)?);
  let range = Expr::List(
    vec![
      Expr::List(vec![ev(minus2(x0, dx.clone()))?, ev(plus2(x1, dx))?].into()),
      Expr::List(vec![ev(minus2(y0, dy.clone()))?, ev(plus2(y1, dy))?].into()),
    ]
    .into(),
  );
  let graphics = call(
    "Graphics",
    vec![
      Expr::List(
        vec![
          Expr::List(centers.into()),
          Expr::List(vertices.into()),
          Expr::List(lines.into()),
        ]
        .into(),
      ),
      rule_expr(id_expr("PlotRange"), range),
      rule_expr(id_expr("AspectRatio"), Expr::Integer(1)),
    ],
  );
  ev(call(
    "Show",
    vec![graphics, Expr::List(graphics_opts.into())],
  ))
}

fn expr_to_i128_opt(e: &Expr) -> Option<i128> {
  match e {
    Expr::Integer(i) => Some(*i),
    _ => None,
  }
}
