//! `Mesh`, `MeshFunctions`, `MeshStyle` and `MeshShading` along the curves
//! of `Plot`, joined `ListPlot` and `ListLinePlot`.
//!
//! A curve's mesh points are where a mesh function `f[x, y]` crosses one of
//! its mesh values, found by linear interpolation of `f` between the curve's
//! consecutive points. `Mesh -> n` takes `n` values equally spaced over the
//! range of `f` along the drawn curve (its ends excluded); `Mesh -> {{v1, …}}`
//! gives them explicitly, optionally as `{v, style}` pairs. `MeshShading`
//! styles the stretches between mesh values: with `k` mesh functions it is a
//! rank-`k` array indexed, per function, by the number of that function's
//! mesh values below `f` there (cyclically), as in Wolfram.

use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot::{SeriesStyle, apply_style_directive};
use crate::syntax::Expr;

/// The mesh values of one mesh function.
#[derive(Clone, Debug)]
pub(crate) enum MeshLevels {
  /// `n` values equally spaced over the function's range along the curve.
  Count(usize),
  /// Explicit values, each with an optional style of its own.
  Values(Vec<(f64, Option<SeriesStyle>)>),
}

/// `Mesh -> …` for a curve. `All` marks every point of the drawn curve,
/// `Full` the regular points (`Plot`'s initial sampling grid, a list plot's
/// data points — the two differ from `All` when the curve is refined or
/// interpolated).
#[derive(Clone, Debug, Default)]
pub(crate) enum Mesh {
  #[default]
  None,
  All,
  Full,
  /// One entry per mesh function; a single entry (`Mesh -> n`) applies to
  /// every mesh function.
  Levels(Vec<MeshLevels>),
}

impl Mesh {
  pub(crate) fn is_none(&self) -> bool {
    matches!(self, Mesh::None)
  }
}

/// One `MeshShading` entry.
#[derive(Clone, Debug)]
pub(crate) enum MeshShade {
  /// `Automatic`: the curve's own style (PlotStyle / ColorFunction).
  Automatic,
  /// `None`: the stretch is not drawn.
  Hidden,
  Style(Box<SeriesStyle>),
  /// A further array level, for the next mesh function.
  List(Vec<MeshShade>),
}

/// A style from a directive or a list of directives, keeping every field it
/// sets (`PointSize[…]` or `Opacity[…]` alone are styles too).
pub(crate) fn parse_mesh_style_entry(value: &Expr) -> Option<SeriesStyle> {
  if matches!(value, Expr::Identifier(s) if s == "Automatic" || s == "None") {
    return None;
  }
  let mut style = SeriesStyle::default();
  match value {
    Expr::List(items) => {
      for item in items {
        apply_style_directive(item, &mut style);
      }
    }
    other => apply_style_directive(other, &mut style),
  }
  Some(style)
}

/// `MeshStyle -> s` (one style for every mesh function) or
/// `MeshStyle -> {s1, s2, …}` (cycled over the mesh functions). WL 15 still
/// draws the points of `MeshStyle -> None` in the curve's style.
pub(crate) fn parse_mesh_styles(value: &Expr) -> Vec<Option<SeriesStyle>> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items) if !items.is_empty() => {
      items.iter().map(parse_mesh_style_entry).collect()
    }
    other => vec![parse_mesh_style_entry(other)],
  }
}

fn parse_levels(spec: &Expr) -> Option<MeshLevels> {
  match spec {
    Expr::Integer(n) if *n >= 0 => Some(MeshLevels::Count(*n as usize)),
    Expr::Identifier(s) if s == "None" => Some(MeshLevels::Count(0)),
    Expr::List(values) => {
      let mut out = Vec::with_capacity(values.len());
      for v in values {
        match v {
          Expr::List(pair) if pair.len() == 2 => {
            out.push((
              try_eval_to_f64(&pair[0])?,
              parse_mesh_style_entry(&pair[1]),
            ));
          }
          _ => out.push((try_eval_to_f64(v)?, None)),
        }
      }
      Some(MeshLevels::Values(out))
    }
    _ => None,
  }
}

/// Parse the value of `Mesh -> …`. Specifications Wolfram rejects (with
/// `Mesh::ilevels`) leave the curve without a mesh.
pub(crate) fn parse_mesh(value: &Expr) -> Mesh {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::Identifier(v) if v == "All" => Mesh::All,
    Expr::Identifier(v) if v == "Full" => Mesh::Full,
    Expr::Integer(n) if *n > 0 => {
      Mesh::Levels(vec![MeshLevels::Count(*n as usize)])
    }
    Expr::List(per_function) if !per_function.is_empty() => per_function
      .iter()
      .map(parse_levels)
      .collect::<Option<Vec<_>>>()
      .map_or(Mesh::None, Mesh::Levels),
    _ => Mesh::None,
  }
}

/// `MeshFunctions -> {f1, f2, …}` (a bare function counts as a list of one).
pub(crate) fn parse_mesh_functions(value: &Expr) -> Vec<Expr> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items) => items.iter().cloned().collect(),
    Expr::Identifier(s) if s == "Automatic" => Vec::new(),
    other => vec![other.clone()],
  }
}

fn parse_shade(value: &Expr) -> MeshShade {
  match value {
    Expr::Identifier(s) if s == "Automatic" => MeshShade::Automatic,
    Expr::Identifier(s) if s == "None" => MeshShade::Hidden,
    Expr::List(items) => {
      MeshShade::List(items.iter().map(parse_shade).collect())
    }
    other => parse_mesh_style_entry(other)
      .map_or(MeshShade::Automatic, |st| MeshShade::Style(Box::new(st))),
  }
}

/// `MeshShading -> {s1, s2, …}` (or a nested array for several mesh
/// functions). `None`/`Automatic` at the top level shade nothing.
pub(crate) fn parse_mesh_shading(value: &Expr) -> Option<MeshShade> {
  let val = evaluate_expr_to_expr(value).unwrap_or_else(|_| value.clone());
  match &val {
    Expr::List(items) if !items.is_empty() => Some(parse_shade(&val)),
    _ => None,
  }
}

/// Whether `shade` is an array of rank exactly `rank` with no empty level
/// (Wolfram's `MeshShading::colshd` otherwise, and the curve stays plain).
fn shading_has_rank(shade: &MeshShade, rank: usize) -> bool {
  match shade {
    MeshShade::List(items) => {
      rank > 0
        && !items.is_empty()
        && items.iter().all(|s| shading_has_rank(s, rank - 1))
    }
    _ => rank == 0,
  }
}

fn shading_pick<'a>(shade: &'a MeshShade, regions: &[usize]) -> &'a MeshShade {
  let mut node = shade;
  for &r in regions {
    if let MeshShade::List(items) = node {
      node = &items[r % items.len()];
    }
  }
  node
}

/// A mesh point of a curve: where mesh function `func` takes one of its
/// values, with that value's own style (`Mesh -> {{{v, style}, …}}`).
#[derive(Clone, Debug)]
pub(crate) struct MeshDot {
  pub pos: (f64, f64),
  pub func: usize,
  pub style: Option<SeriesStyle>,
}

/// The mesh of one curve: its mesh points and, under `MeshShading`, the
/// curve cut into stretches with the shading entry of each (`Hidden`
/// stretches omitted).
pub(crate) struct CurveMesh {
  pub dots: Vec<MeshDot>,
  pub pieces: Option<Vec<(MeshShade, Vec<(f64, f64)>)>>,
}

/// `#1 &` and `#2 &` (and `Function[{x, y}, x]`-style projections) are read
/// straight off the point instead of going through the evaluator.
fn projection(f: &Expr) -> Option<usize> {
  match f {
    Expr::Function { body } => match body.as_ref() {
      Expr::Slot(1) => Some(0),
      Expr::Slot(2) => Some(1),
      _ => None,
    },
    _ => None,
  }
}

fn mesh_function_value(f: &Expr, (x, y): (f64, f64)) -> Option<f64> {
  match projection(f) {
    Some(0) => return Some(x),
    Some(_) => return Some(y),
    None => {}
  }
  let r = crate::functions::list_helpers_ast::apply_func_to_two_args(
    f,
    &Expr::Real(x),
    &Expr::Real(y),
  )
  .ok()?;
  try_eval_to_f64(&r).filter(|v| v.is_finite())
}

/// Interpolate the parameter on the closest original sample edge, including
/// points introduced when clipping a curve to its plot range.
fn parameter_at(
  p: (f64, f64),
  samples: &[(f64, f64, f64)],
  start: &mut usize,
) -> Option<f64> {
  let mut best = (f64::INFINITY, None);
  for (index, edge) in samples.windows(2).enumerate().skip(*start) {
    let (x, y, t) = edge[0];
    let (u, v, s) = edge[1];
    if ![x, y, t, u, v, s].iter().all(|v| v.is_finite()) {
      continue;
    }
    let d2 = (u - x).powi(2) + (v - y).powi(2);
    let f = if d2 == 0.0 {
      0.0
    } else {
      (((p.0 - x) * (u - x) + (p.1 - y) * (v - y)) / d2).clamp(0.0, 1.0)
    };
    let distance =
      (p.0 - x - f * (u - x)).powi(2) + (p.1 - y - f * (v - y)).powi(2);
    if distance < best.0 {
      best = (distance, Some(t + f * (s - t)));
      *start = index;
    }
  }
  best.1
}

/// The values of each mesh function at each point of a curve, segment by
/// segment. `functions` are the mesh functions (empty: `{#1 &}`);
/// `by_parameter` makes the default mesh function the position along the
/// point list, the curve parameter of a `ParametricPlot`.
fn function_values(
  segments: &[Vec<(f64, f64)>],
  functions: &[Expr],
  by_parameter: bool,
  parameters: &[(f64, f64, f64)],
) -> Vec<Vec<Vec<Option<f64>>>> {
  (0..functions.len().max(1))
    .map(|k| {
      let mut counter = 0.0;
      let mut parameter_index = 0;
      segments
        .iter()
        .map(|seg| {
          seg
            .iter()
            .map(|&p| {
              let t = parameter_at(p, parameters, &mut parameter_index)
                .unwrap_or_else(|| {
                  counter += 1.0;
                  counter
                });
              match functions.get(k) {
                Some(f) if by_parameter => {
                  let result = evaluate_expr_to_expr(&Expr::CurriedCall {
                    func: Box::new(f.clone()),
                    args: vec![Expr::Real(p.0), Expr::Real(p.1), Expr::Real(t)],
                  })
                  .ok()?;
                  try_eval_to_f64(&result).filter(|v| v.is_finite())
                }
                Some(f) => mesh_function_value(f, p),
                None if by_parameter => Some(t),
                None => Some(p.0),
              }
            })
            .collect()
        })
        .collect()
    })
    .collect()
}

/// The sorted mesh values of each mesh function. `Mesh -> n` spaces its
/// values over the range the function takes on all the curves together
/// (`ListLinePlot[{Range[8], Range[300]}, Mesh -> 1]` marks only the long
/// curve, at x = 150.5). `None` unless `mesh` asks for mesh values.
pub(crate) fn mesh_levels(
  curves: &[Vec<Vec<(f64, f64)>>],
  mesh: &Mesh,
  functions: &[Expr],
  by_parameter: bool,
  parameters: &[Vec<(f64, f64, f64)>],
) -> Option<Vec<Vec<(f64, Option<SeriesStyle>)>>> {
  let Mesh::Levels(specs) = mesh else {
    return None;
  };
  let needs_range = specs.iter().any(|s| matches!(s, MeshLevels::Count(_)));
  let values: Vec<Vec<Vec<Vec<Option<f64>>>>> = if needs_range {
    curves
      .iter()
      .enumerate()
      .map(|(i, c)| {
        function_values(
          c,
          functions,
          by_parameter,
          parameters.get(i).map_or(&[], |p| p.as_slice()),
        )
      })
      .collect()
  } else {
    Vec::new()
  };
  Some(
    (0..functions.len().max(1))
      .map(|k| {
        let spec = if specs.len() == 1 {
          specs.first()
        } else {
          specs.get(k)
        };
        let mut lv = match spec {
          Some(MeshLevels::Count(n)) if *n > 0 => {
            let (lo, hi) = values
              .iter()
              .flat_map(|curve| curve[k].iter().flatten().flatten())
              .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
              });
            if hi > lo {
              (1..=*n)
                .map(|i| (lo + (hi - lo) * i as f64 / (*n + 1) as f64, None))
                .collect()
            } else {
              Vec::new()
            }
          }
          Some(MeshLevels::Values(vs)) => vs.clone(),
          _ => Vec::new(),
        };
        lv.sort_by(|a, b| a.0.total_cmp(&b.0));
        lv
      })
      .collect(),
  )
}

/// The mesh of a curve drawn as `segments` (its finite, in-view pieces),
/// for the mesh values `levels` of [`mesh_levels`]. With `open_ends` a
/// value met exactly at an end of a segment is not marked: `Plot` samples
/// just inside its domain, so `Mesh -> {{0}}` on `Sin` has no point at
/// x = 0, while a list plot's first data point can be a mesh point.
pub(crate) fn curve_mesh(
  segments: &[Vec<(f64, f64)>],
  levels: &[Vec<(f64, Option<SeriesStyle>)>],
  functions: &[Expr],
  shading: Option<&MeshShade>,
  by_parameter: bool,
  open_ends: bool,
  parameters: &[(f64, f64, f64)],
) -> CurveMesh {
  let n_funcs = functions.len().max(1);
  let values = function_values(segments, functions, by_parameter, parameters);

  // Mesh points: a value is met on the half-open stretch [va, vb) of each
  // pair of points, and at the last point of a segment, so a value taken
  // exactly at a vertex is marked once.
  let mut dots = Vec::new();
  for (k, lv) in levels.iter().enumerate() {
    for (s, seg) in segments.iter().enumerate() {
      let vals = &values[k][s];
      for i in 0..seg.len().saturating_sub(1) {
        let (Some(va), Some(vb)) = (vals[i], vals[i + 1]) else {
          continue;
        };
        let last = i + 2 == seg.len();
        for (l, style) in lv {
          let hit = (va <= *l && *l < vb)
            || (vb < *l && *l <= va)
            || (last && *l == vb && va != vb);
          let at_end = (i == 0 && *l == va) || (last && *l == vb);
          if open_ends && at_end {
            continue;
          }
          if !hit {
            continue;
          }
          let t = if vb == va { 0.0 } else { (l - va) / (vb - va) };
          let (a, b) = (seg[i], seg[i + 1]);
          dots.push(MeshDot {
            pos: (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)),
            func: k,
            style: style.clone(),
          });
        }
      }
    }
  }

  let pieces = shading
    .filter(|sh| shading_has_rank(sh, n_funcs))
    .map(|sh| shade_pieces(segments, &values, levels, sh));
  CurveMesh { dots, pieces }
}

/// Cut the curve where any mesh function crosses one of its values and give
/// each stretch its `MeshShading` entry, from the mesh regions at its
/// middle. Consecutive stretches in the same region form one polyline.
fn shade_pieces(
  segments: &[Vec<(f64, f64)>],
  values: &[Vec<Vec<Option<f64>>>],
  levels: &[Vec<(f64, Option<SeriesStyle>)>],
  shading: &MeshShade,
) -> Vec<(MeshShade, Vec<(f64, f64)>)> {
  let region = |k: usize, v: f64| levels[k].iter().filter(|l| v > l.0).count();
  let mut out: Vec<(Vec<usize>, Vec<(f64, f64)>)> = Vec::new();
  for (s, seg) in segments.iter().enumerate() {
    let mut current: Option<(Vec<usize>, Vec<(f64, f64)>)> = None;
    for i in 0..seg.len().saturating_sub(1) {
      let vals: Option<Vec<(f64, f64)>> = values
        .iter()
        .map(|fv| Some((fv[s][i]?, fv[s][i + 1]?)))
        .collect();
      let Some(vals) = vals else {
        // A point where a mesh function is undefined breaks the curve.
        out.extend(current.take());
        continue;
      };
      // Crossing parameters inside this pair, in travel order.
      let mut ts: Vec<f64> = vec![0.0, 1.0];
      for (k, &(va, vb)) in vals.iter().enumerate() {
        if va == vb {
          continue;
        }
        for (l, _) in &levels[k] {
          let t = (l - va) / (vb - va);
          if t > 0.0 && t < 1.0 {
            ts.push(t);
          }
        }
      }
      ts.sort_by(f64::total_cmp);
      ts.dedup();
      let (a, b) = (seg[i], seg[i + 1]);
      let at = |t: f64| (a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1));
      for w in ts.windows(2) {
        let mid = f64::midpoint(w[0], w[1]);
        let r: Vec<usize> = vals
          .iter()
          .enumerate()
          .map(|(k, &(va, vb))| region(k, va + mid * (vb - va)))
          .collect();
        let (p0, p1) = (at(w[0]), at(w[1]));
        match &mut current {
          Some((cr, pts)) if *cr == r => pts.push(p1),
          _ => {
            out.extend(current.take());
            current = Some((r, vec![p0, p1]));
          }
        }
      }
    }
    out.extend(current);
  }
  out
    .into_iter()
    .filter_map(|(r, pts)| match shading_pick(shading, &r) {
      MeshShade::Hidden => None,
      shade => Some((shade.clone(), pts)),
    })
    .collect()
}
