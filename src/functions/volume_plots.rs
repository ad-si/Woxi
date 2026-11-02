//! Plots of a function of three variables over a box:
//!
//! - `DensityPlot3D[f, {x, …}, {y, …}, {z, …}]` — a `Raster3D` of the
//!   sampled values, coloured by a `ColorFunction` that hides the values
//!   below the midpoint of its scale, as in Wolfram.
//! - `SliceDensityPlot3D[f, surf, …]` — `f` coloured continuously on the
//!   slice surfaces `surf`.
//! - `SliceContourPlot3D[f, surf, …]` — `f`'s contour bands on them.
//!
//! All three return symbolic `Graphics3D` (so `Part`, `Cases` and `Show`
//! see the primitives), which the ordinary `Graphics3D` renderer draws.

use std::collections::HashMap;

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::functions::field_plot::{
  ContourSpec, apply_named_color_function, color_function_scheme_name,
  resolve_contour_levels,
};
use crate::functions::math_ast::try_eval_to_f64;
use crate::functions::plot3d::{
  Point3D, evaluate_scalar_3d, march_cube, parse_iterator_generic,
  reset_scalar_3d_cache,
};
use crate::syntax::{ComparisonOp, Expr};

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn rule(key: &str, value: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(Expr::Identifier(key.to_string())),
    replacement: Box::new(value),
  }
}

fn reals(values: &[f64]) -> Expr {
  Expr::List(values.iter().map(|&v| Expr::Real(v)).collect())
}

fn rgb((r, g, b): (u8, u8, u8)) -> Expr {
  call(
    "RGBColor",
    vec![
      Expr::Real(r as f64 / 255.0),
      Expr::Real(g as f64 / 255.0),
      Expr::Real(b as f64 / 255.0),
    ],
  )
}

/// The box `{x, x0, x1} × {y, y0, y1} × {z, z0, z1}` and its variables.
struct Volume {
  vars: [String; 3],
  lo: [f64; 3],
  hi: [f64; 3],
}

impl Volume {
  fn parse(iterators: &[Expr], head: &str) -> Result<Volume, InterpreterError> {
    let mut vars: [String; 3] = Default::default();
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for (k, label) in ["first", "second", "third"].iter().enumerate() {
      let (v, a, b) = parse_iterator_generic(&iterators[k], head, label)?;
      if a.partial_cmp(&b) != Some(std::cmp::Ordering::Less) {
        return Err(InterpreterError::EvaluationError(format!(
          "{head}: the {label} iterator range must be increasing"
        )));
      }
      vars[k] = v;
      lo[k] = a;
      hi[k] = b;
    }
    Ok(Volume { vars, lo, hi })
  }

  fn var_names(&self) -> [&str; 3] {
    [&self.vars[0], &self.vars[1], &self.vars[2]]
  }

  /// `body` at the point `p`, when it is a finite number.
  fn value(&self, body: &Expr, p: [f64; 3]) -> Option<f64> {
    evaluate_scalar_3d(
      body,
      &self.vars[0],
      &self.vars[1],
      &self.vars[2],
      p[0],
      p[1],
      p[2],
    )
    .filter(|v| v.is_finite())
  }

  /// The `k`-th of `n` evenly spaced sample coordinates along `axis`.
  fn sample(&self, axis: usize, k: usize, n: usize) -> f64 {
    self.lo[axis] + (self.hi[axis] - self.lo[axis]) * k as f64 / (n - 1) as f64
  }

  fn plot_range(&self) -> Expr {
    Expr::List(
      (0..3)
        .map(|k| reals(&[self.lo[k], self.hi[k]]))
        .collect::<Vec<_>>()
        .into(),
    )
  }
}

/// How sampled values are turned into colours.
enum Colors {
  /// The default density gradient.
  Default,
  /// A named gradient (`"Rainbow"`, `ColorData["Rainbow"]`).
  Named(String),
  /// Any other function, applied to the (scaled) value.
  Function(Expr),
}

impl Colors {
  fn parse(spec: Option<&Expr>) -> Colors {
    match spec {
      None => Colors::Default,
      Some(Expr::Identifier(s)) if s == "Automatic" => Colors::Default,
      Some(e) => match color_function_scheme_name(e) {
        Some(name) => Colors::Named(name),
        None => Colors::Function(e.clone()),
      },
    }
  }

  /// The colour directive for the value `t` (scaled to `[0, 1]`, or the
  /// raw value with `ColorFunctionScaling -> False`).
  fn at(&self, t: f64) -> Expr {
    match self {
      Colors::Default => {
        rgb(apply_named_color_function("WL14DefaultDensityGradient", t))
      }
      Colors::Named(name) => rgb(apply_named_color_function(name, t)),
      Colors::Function(f) => {
        let applied = Expr::CurriedCall {
          func: Box::new(f.clone()),
          args: vec![Expr::Real(t)],
        };
        evaluate_expr_to_expr(&applied)
          .unwrap_or_else(|_| call("GrayLevel", vec![Expr::Real(t)]))
      }
    }
  }

  /// The colour directive for the value `t` given as an expression (an
  /// exact band position).
  fn at_expr(&self, t: Expr) -> Expr {
    match self {
      Colors::Function(f) => {
        let applied = Expr::CurriedCall {
          func: Box::new(f.clone()),
          args: vec![t.clone()],
        };
        evaluate_expr_to_expr(&applied)
          .unwrap_or_else(|_| call("GrayLevel", vec![t]))
      }
      _ => self.at(try_eval_to_f64(&t).unwrap_or(0.5)),
    }
  }

  /// The colour function itself, as a legend shows it.
  fn as_expr(&self) -> Expr {
    match self {
      Colors::Default => Expr::String("WL14DefaultDensityGradient".into()),
      Colors::Named(name) => Expr::String(name.clone()),
      Colors::Function(f) => f.clone(),
    }
  }
}

/// The options shared by the three plots, with the ones that only steer
/// the sampling taken out; the rest are passed on to `Graphics3D`.
struct PlotOptions {
  plot_points: Option<usize>,
  colors: Colors,
  scaling: bool,
  contours: ContourSpec,
  legend: bool,
  graphics: Vec<Expr>,
}

impl PlotOptions {
  fn parse(options: &[Expr]) -> PlotOptions {
    let mut opts = PlotOptions {
      plot_points: None,
      colors: Colors::Default,
      scaling: true,
      contours: ContourSpec::Automatic,
      legend: false,
      graphics: Vec::new(),
    };
    let mut color_function = None;
    for opt in options {
      let (key, value) = match opt {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => match pattern.as_ref() {
          Expr::Identifier(k) => (k.as_str(), replacement.as_ref()),
          _ => continue,
        },
        _ => continue,
      };
      match key {
        "PlotPoints" => {
          let n = match &evaluate_expr_to_expr(value) {
            Ok(Expr::Integer(n)) => Some(*n),
            Ok(Expr::List(items)) => match items.first() {
              Some(Expr::Integer(n)) => Some(*n),
              _ => None,
            },
            _ => None,
          };
          if let Some(n) = n
            && n >= 2
          {
            opts.plot_points = Some(n.min(400) as usize);
          }
        }
        "ColorFunction" => color_function = Some(value.clone()),
        "ColorFunctionScaling" => {
          opts.scaling = !matches!(value, Expr::Identifier(v) if v == "False");
        }
        "Contours" => match &evaluate_expr_to_expr(value) {
          Ok(Expr::Integer(n)) if *n > 0 => {
            opts.contours = ContourSpec::Count(*n as usize);
          }
          Ok(Expr::List(items)) => {
            let levels: Vec<f64> = items
              .iter()
              .filter_map(try_eval_to_f64)
              .filter(|v| v.is_finite())
              .collect();
            if !levels.is_empty() {
              opts.contours = ContourSpec::Levels(levels);
            }
          }
          _ => {}
        },
        "PlotLegends" => {
          opts.legend = !matches!(value, Expr::Identifier(v) if v == "None");
        }
        // Sampling and styling controls with no counterpart in the
        // Graphics3D the plot returns.
        "PerformanceGoal"
        | "MaxRecursion"
        | "BoundaryStyle"
        | "ContourStyle"
        | "ContourShading"
        | "Mesh"
        | "MeshStyle"
        | "MeshFunctions"
        | "RegionFunction"
        | "OpacityFunction"
        | "OpacityFunctionScaling"
        | "EvaluationMonitor"
        | "Exclusions"
        | "ExclusionsStyle"
        | "ClippingStyle"
        | "PlotTheme"
        | "WorkingPrecision"
        | "Method"
        | "PlotRange" => {}
        _ => opts.graphics.push(opt.clone()),
      }
    }
    opts.colors = Colors::parse(
      color_function
        .map(|e| evaluate_expr_to_expr(&e).unwrap_or(e))
        .as_ref(),
    );
    opts
  }

  /// The `Graphics3D` options: the user's, then the plot's defaults for
  /// any they do not set.
  fn graphics_options(&self, vol: &Volume) -> Vec<Expr> {
    let mut out = self.graphics.clone();
    let has = |k: &str| {
      out.iter().any(|o| {
        matches!(o, Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. }
          if matches!(pattern.as_ref(), Expr::Identifier(n) if n == k))
      })
    };
    let mut defaults = Vec::new();
    if !has("Axes") {
      defaults.push(rule("Axes", Expr::Identifier("True".into())));
    }
    if !has("BoxRatios") {
      defaults.push(rule(
        "BoxRatios",
        Expr::List(vec![Expr::Integer(1); 3].into()),
      ));
    }
    if !has("Lighting") {
      defaults.push(rule("Lighting", Expr::String("Neutral".into())));
    }
    defaults.push(rule("PlotRange", vol.plot_range()));
    out.extend(defaults);
    out
  }
}

/// A piece of a slice surface: its corners and `f` at each of them.
struct Facet {
  points: Vec<[f64; 3]>,
  values: Vec<f64>,
}

/// The axis-aligned plane `axis == c`, as a grid of `n × n` samples.
fn plane_facets(
  vol: &Volume,
  body: &Expr,
  axis: usize,
  c: f64,
  n: usize,
  out: &mut Vec<Facet>,
) {
  if c < vol.lo[axis] || c > vol.hi[axis] {
    return;
  }
  let (a, b) = match axis {
    0 => (1, 2),
    1 => (0, 2),
    _ => (0, 1),
  };
  let point = |i: usize, j: usize| {
    let mut p = [0.0; 3];
    p[axis] = c;
    p[a] = vol.sample(a, i, n);
    p[b] = vol.sample(b, j, n);
    p
  };
  let values: Vec<Vec<Option<f64>>> = (0..n)
    .map(|i| (0..n).map(|j| vol.value(body, point(i, j))).collect())
    .collect();
  for i in 0..n - 1 {
    for j in 0..n - 1 {
      let corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)];
      let vals: Option<Vec<f64>> =
        corners.iter().map(|&(p, q)| values[p][q]).collect();
      if let Some(vals) = vals {
        out.push(Facet {
          points: corners.iter().map(|&(p, q)| point(p, q)).collect(),
          values: vals,
        });
      }
    }
  }
}

/// The implicit surface `g == 0` through the box, triangulated by marching
/// tetrahedra over an `n × n × n` grid.
fn implicit_facets(
  vol: &Volume,
  body: &Expr,
  g: &Expr,
  n: usize,
  out: &mut Vec<Facet>,
) {
  let n = n.clamp(3, 40);
  let at = |i: usize, j: usize, k: usize| {
    [
      vol.sample(0, i, n),
      vol.sample(1, j, n),
      vol.sample(2, k, n),
    ]
  };
  // The whole surface field first: `evaluate_scalar_3d` keeps one compiled
  // body at a time, so `g` and `f` are not interleaved.
  let field: Vec<f64> = (0..n * n * n)
    .map(|idx| {
      let (i, j, k) = (idx / (n * n), (idx / n) % n, idx % n);
      vol.value(g, at(i, j, k)).unwrap_or(f64::NAN)
    })
    .collect();
  let f_at = |i: usize, j: usize, k: usize| field[(i * n + j) * n + k];
  let mut triangles: Vec<[Point3D; 3]> = Vec::new();
  for i in 0..n - 1 {
    for j in 0..n - 1 {
      for k in 0..n - 1 {
        let idx = [
          (i, j, k),
          (i + 1, j, k),
          (i + 1, j + 1, k),
          (i, j + 1, k),
          (i, j, k + 1),
          (i + 1, j, k + 1),
          (i + 1, j + 1, k + 1),
          (i, j + 1, k + 1),
        ];
        let vals = idx.map(|(a, b, c)| f_at(a, b, c));
        if vals.iter().any(|v| !v.is_finite()) {
          continue;
        }
        let corners = idx.map(|(a, b, c)| {
          let p = at(a, b, c);
          Point3D {
            x: p[0],
            y: p[1],
            z: p[2],
          }
        });
        march_cube(&corners, &vals, 0.0, &mut |tri| triangles.push(tri));
      }
    }
  }
  let mut cache: HashMap<[u64; 3], Option<f64>> = HashMap::new();
  for tri in triangles {
    let points: Vec<[f64; 3]> = tri.iter().map(|p| [p.x, p.y, p.z]).collect();
    let values: Option<Vec<f64>> = points
      .iter()
      .map(|p| {
        *cache
          .entry(p.map(f64::to_bits))
          .or_insert_with(|| vol.value(body, *p))
      })
      .collect();
    if let Some(values) = values {
      out.push(Facet { points, values });
    }
  }
}

/// The plane positions of `"XStackedPlanes"` and its kin: `count` planes
/// evenly spaced strictly inside the range, or the given coordinates.
fn stacked_positions(
  vol: &Volume,
  axis: usize,
  spec: Option<&Expr>,
) -> Vec<f64> {
  let evenly = |count: usize| {
    (1..=count)
      .map(|i| {
        vol.lo[axis]
          + (vol.hi[axis] - vol.lo[axis]) * i as f64 / (count + 1) as f64
      })
      .collect()
  };
  match spec {
    None => evenly(3),
    Some(Expr::Integer(n)) if *n > 0 => evenly(*n as usize),
    Some(Expr::List(items)) => {
      items.iter().filter_map(try_eval_to_f64).collect()
    }
    Some(_) => Vec::new(),
  }
}

fn stacked_axis(name: &str) -> Option<usize> {
  match name {
    "XStackedPlanes" => Some(0),
    "YStackedPlanes" => Some(1),
    "ZStackedPlanes" => Some(2),
    _ => None,
  }
}

/// Collect the facets of the slice specification `spec`. Returns `false`
/// for a specification this module does not know.
fn slice_facets(
  vol: &Volume,
  body: &Expr,
  spec: &Expr,
  n: usize,
  out: &mut Vec<Facet>,
) -> bool {
  let mid = |k: usize| f64::midpoint(vol.lo[k], vol.hi[k]);
  match spec {
    Expr::String(s) if s == "CenterPlanes" => {
      for k in 0..3 {
        plane_facets(vol, body, k, mid(k), n, out);
      }
      true
    }
    // The three faces furthest from the default viewpoint.
    Expr::String(s) if s == "BackPlanes" => {
      plane_facets(vol, body, 0, vol.lo[0], n, out);
      plane_facets(vol, body, 1, vol.hi[1], n, out);
      plane_facets(vol, body, 2, vol.lo[2], n, out);
      true
    }
    Expr::String(s) if stacked_axis(s).is_some() => {
      let axis = stacked_axis(s).unwrap();
      for c in stacked_positions(vol, axis, None) {
        plane_facets(vol, body, axis, c, n, out);
      }
      true
    }
    Expr::List(items)
      if items.len() == 2
        && matches!(&items[0], Expr::String(s) if stacked_axis(s).is_some())
        && matches!(&items[1], Expr::Integer(_) | Expr::List(_)) =>
    {
      let Expr::String(s) = &items[0] else {
        return false;
      };
      let axis = stacked_axis(s).unwrap();
      for c in stacked_positions(vol, axis, Some(&items[1])) {
        plane_facets(vol, body, axis, c, n, out);
      }
      true
    }
    Expr::List(items) => items
      .iter()
      .all(|item| slice_facets(vol, body, item, n, out)),
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators[0] == ComparisonOp::Equal => {
      let axis_of = |e: &Expr| match e {
        Expr::Identifier(v) => vol.vars.iter().position(|w| w == v),
        _ => None,
      };
      // `z == c` is the plane itself, sampled on a grid rather than
      // marched (a plane through grid nodes leaves marching degenerate).
      for (a, b) in [(&operands[0], &operands[1]), (&operands[1], &operands[0])]
      {
        if let (Some(axis), Some(c)) = (axis_of(a), try_eval_to_f64(b)) {
          plane_facets(vol, body, axis, c, n, out);
          return true;
        }
      }
      let g = call(
        "Plus",
        vec![
          operands[0].clone(),
          call("Times", vec![Expr::Integer(-1), operands[1].clone()]),
        ],
      );
      reset_scalar_3d_cache();
      implicit_facets(vol, body, &g, n, out);
      true
    }
    _ => false,
  }
}

/// The smallest and largest of the facets' values.
fn value_range(facets: &[Facet]) -> Option<(f64, f64)> {
  let mut lo = f64::INFINITY;
  let mut hi = f64::NEG_INFINITY;
  for v in facets.iter().flat_map(|f| f.values.iter()) {
    lo = lo.min(*v);
    hi = hi.max(*v);
  }
  (lo <= hi).then_some((lo, hi))
}

fn scale(v: f64, lo: f64, hi: f64) -> f64 {
  if hi > lo {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
  } else {
    0.5
  }
}

/// Shared vertex list of a `GraphicsComplex`, keyed by exact coordinates.
#[derive(Default)]
struct Vertices {
  points: Vec<[f64; 3]>,
  index: HashMap<[u64; 3], usize>,
}

impl Vertices {
  fn polygon(&mut self, points: &[[f64; 3]]) -> Expr {
    Expr::List(
      points
        .iter()
        .map(|p| {
          let next = self.points.len();
          let i = *self.index.entry(p.map(f64::to_bits)).or_insert(next);
          if i == next {
            self.points.push(*p);
          }
          Expr::Integer(i as i128 + 1)
        })
        .collect(),
    )
  }

  fn complex(self, primitives: Vec<Expr>) -> Expr {
    let mut content = vec![call("EdgeForm", vec![])];
    content.extend(primitives);
    call(
      "GraphicsComplex",
      vec![
        Expr::List(self.points.iter().map(|p| reals(p)).collect()),
        Expr::List(content.into()),
      ],
    )
  }
}

/// Parse `head[f, surf, {x, …}, {y, …}, {z, …}, opts…]` into the slice
/// facets, or `None` when the call is not of that shape.
fn slice_setup(
  head: &str,
  args: &[Expr],
) -> Result<Option<(Volume, PlotOptions, Vec<Facet>)>, InterpreterError> {
  if args.len() < 5 {
    return Ok(None);
  }
  reset_scalar_3d_cache();
  let vol = Volume::parse(&args[2..5], head)?;
  let opts = PlotOptions::parse(&args[5..]);
  let names = vol.var_names();
  let body =
    crate::functions::plot::resolve_indirect_plot_body(&args[0], &names)
      .unwrap_or_else(|| args[0].clone());
  let spec = crate::functions::plot::eval_body_vars_symbolic(&args[1], &names);
  let mut facets = Vec::new();
  let n = opts.plot_points.unwrap_or(33);
  if !slice_facets(&vol, &body, &spec, n, &mut facets) {
    return Ok(None);
  }
  Ok(Some((vol, opts, facets)))
}

/// Wrap a plot in `Legended[…, Placed[BarLegend[{colors, {lo, hi}}], After]]`
/// when `PlotLegends` asks for a legend.
fn with_legend(
  plot: Expr,
  opts: &PlotOptions,
  (lo, hi): (f64, f64),
  levels: Option<&[f64]>,
) -> Expr {
  if !opts.legend {
    return plot;
  }
  let mut bar_args = vec![Expr::List(
    vec![opts.colors.as_expr(), reals(&[lo, hi])].into(),
  )];
  if let Some(levels) = levels {
    bar_args.push(reals(levels));
  }
  call(
    "Legended",
    vec![
      plot,
      call(
        "Placed",
        vec![
          call("BarLegend", bar_args),
          Expr::Identifier("After".into()),
        ],
      ),
    ],
  )
}

fn graphics3d(content: Vec<Expr>, options: Vec<Expr>) -> Expr {
  let mut args = vec![Expr::List(content.into())];
  args.extend(options);
  call("Graphics3D", args)
}

/// `SliceDensityPlot3D[f, surf, {x, …}, {y, …}, {z, …}]`.
pub fn slice_density_plot3d_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((vol, opts, facets)) = slice_setup("SliceDensityPlot3D", args)?
  else {
    return Ok(None);
  };
  let (lo, hi) = value_range(&facets).unwrap_or((0.0, 1.0));
  let mut vertices = Vertices::default();
  let mut primitives = Vec::with_capacity(facets.len());
  for facet in &facets {
    let mean = facet.values.iter().sum::<f64>() / facet.values.len() as f64;
    let t = if opts.scaling {
      scale(mean, lo, hi)
    } else {
      mean
    };
    let polygon = vertices.polygon(&facet.points);
    primitives.push(Expr::List(
      vec![opts.colors.at(t), call("Polygon", vec![polygon])].into(),
    ));
  }
  let plot = graphics3d(
    vec![vertices.complex(primitives)],
    opts.graphics_options(&vol),
  );
  let plot = evaluate_expr_to_expr(&plot)?;
  Ok(Some(with_legend(plot, &opts, (lo, hi), None)))
}

/// The part of the convex polygon `poly` (with values) where the value is
/// at least (`keep_above`) or at most `level`, by Sutherland–Hodgman.
fn clip(
  poly: &[([f64; 3], f64)],
  level: f64,
  keep_above: bool,
) -> Vec<([f64; 3], f64)> {
  let inside = |v: f64| if keep_above { v >= level } else { v <= level };
  let mut out = Vec::with_capacity(poly.len() + 2);
  for k in 0..poly.len() {
    let (pa, va) = poly[k];
    let (pb, vb) = poly[(k + 1) % poly.len()];
    if inside(va) {
      out.push((pa, va));
    }
    if inside(va) != inside(vb) {
      let t = (level - va) / (vb - va);
      let p = [0, 1, 2].map(|d| pa[d] + (pb[d] - pa[d]) * t);
      out.push((p, level));
    }
  }
  out
}

/// `SliceContourPlot3D[f, surf, {x, …}, {y, …}, {z, …}]`.
pub fn slice_contour_plot3d_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  let Some((vol, opts, facets)) = slice_setup("SliceContourPlot3D", args)?
  else {
    return Ok(None);
  };
  let (lo, hi) = value_range(&facets).unwrap_or((0.0, 1.0));
  let levels = resolve_contour_levels(&opts.contours, lo, hi);
  let mut bounds = vec![lo];
  bounds.extend(levels.iter().copied());
  bounds.push(hi);
  let band_of =
    |v: f64| bounds[1..bounds.len() - 1].partition_point(|&l| l <= v);
  let mut bands: Vec<Vec<Vec<[f64; 3]>>> = vec![Vec::new(); bounds.len() - 1];
  for facet in &facets {
    let first = band_of(facet.values[0]);
    if facet.values.iter().all(|&v| band_of(v) == first) {
      bands[first].push(facet.points.clone());
      continue;
    }
    let poly: Vec<([f64; 3], f64)> = facet
      .points
      .iter()
      .copied()
      .zip(facet.values.iter().copied())
      .collect();
    for (b, band) in bands.iter_mut().enumerate() {
      let mut piece = poly.clone();
      if b > 0 {
        piece = clip(&piece, bounds[b], true);
      }
      if b + 1 < bounds.len() - 1 {
        piece = clip(&piece, bounds[b + 1], false);
      }
      if piece.len() >= 3 {
        band.push(piece.into_iter().map(|(p, _)| p).collect());
      }
    }
  }
  let mut vertices = Vertices::default();
  let mut primitives = Vec::new();
  // Band k of the n + 1 bands takes the colour at k/(n + 2) (exactly, as
  // a rational) of the colour function's scale.
  let band_count = bands.len() as i128;
  for (b, polygons) in bands.iter().enumerate() {
    if polygons.is_empty() {
      continue;
    }
    let t = evaluate_expr_to_expr(&call(
      "Rational",
      vec![Expr::Integer(b as i128 + 1), Expr::Integer(band_count + 1)],
    ))?;
    let t = if opts.scaling {
      t
    } else {
      Expr::Real(lo + (hi - lo) * (b as f64 + 1.0) / (band_count as f64 + 1.0))
    };
    let polys: Vec<Expr> =
      polygons.iter().map(|p| vertices.polygon(p)).collect();
    primitives.push(Expr::List(
      vec![
        opts.colors.at_expr(t),
        call("Polygon", vec![Expr::List(polys.into())]),
      ]
      .into(),
    ));
  }
  let plot = graphics3d(
    vec![vertices.complex(primitives)],
    opts.graphics_options(&vol),
  );
  let plot = evaluate_expr_to_expr(&plot)?;
  Ok(Some(with_legend(plot, &opts, (lo, hi), Some(&levels))))
}

/// `DensityPlot3D[f, {x, …}, {y, …}, {z, …}]`: the values sampled on an
/// `n × n × n` grid (51 per side by default), mapped to `[1/2, 1]` and
/// stored, `z`-layer by `y`-row, in a `Raster3D` spanning the box. Its
/// `ColorFunction` shows the upper half of that scale through the plot's
/// colour function and leaves the lower half transparent.
pub fn density_plot3d_ast(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  if args.len() < 4 {
    return Ok(None);
  }
  reset_scalar_3d_cache();
  let vol = Volume::parse(&args[1..4], "DensityPlot3D")?;
  let opts = PlotOptions::parse(&args[4..]);
  let names = vol.var_names();
  let body =
    crate::functions::plot::resolve_indirect_plot_body(&args[0], &names)
      .unwrap_or_else(|| args[0].clone());
  let n = opts.plot_points.unwrap_or(51).min(101);
  let mut values = vec![f64::NAN; n * n * n];
  for k in 0..n {
    for j in 0..n {
      for i in 0..n {
        let p = [
          vol.sample(0, i, n),
          vol.sample(1, j, n),
          vol.sample(2, k, n),
        ];
        if let Some(v) = vol.value(&body, p) {
          values[(k * n + j) * n + i] = v;
        }
      }
    }
  }
  let (lo, hi) = values
    .iter()
    .filter(|v| v.is_finite())
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
      (lo.min(v), hi.max(v))
    });
  let mapped = |v: f64| {
    if !v.is_finite() {
      0.0
    } else if hi > lo {
      0.5 + 0.5 * (v - lo) / (hi - lo)
    } else {
      0.75
    }
  };
  let data = Expr::List(
    (0..n)
      .map(|k| {
        Expr::List(
          (0..n)
            .map(|j| {
              Expr::List(
                (0..n)
                  .map(|i| Expr::Real(mapped(values[(k * n + j) * n + i])))
                  .collect(),
              )
            })
            .collect(),
        )
      })
      .collect(),
  );
  // `If[#1 >= 0.5, Opacity[o, colors[-1. + 2. #1]], RGBColor[0, 0, 0, 0]] &`
  let rescaled = call(
    "Plus",
    vec![
      Expr::Real(-1.0),
      call("Times", vec![Expr::Real(2.0), Expr::Slot(1)]),
    ],
  );
  let colors = match &opts.colors {
    Colors::Function(f) => f.clone(),
    other => call("ColorData", vec![other.as_expr()]),
  };
  let color_function = Expr::Function {
    body: Box::new(call(
      "If",
      vec![
        Expr::Comparison {
          operands: vec![Expr::Slot(1), Expr::Real(0.5)],
          operators: vec![ComparisonOp::GreaterEqual],
        },
        call(
          "Opacity",
          vec![
            Expr::Real(0.3),
            Expr::CurriedCall {
              func: Box::new(colors),
              args: vec![rescaled],
            },
          ],
        ),
        call("RGBColor", vec![Expr::Integer(0); 4]),
      ],
    )),
  };
  let raster = call(
    "Raster3D",
    vec![
      data,
      Expr::List(
        vec![
          reals(&[vol.lo[0], vol.lo[1], vol.lo[2]]),
          reals(&[vol.hi[0], vol.hi[1], vol.hi[2]]),
        ]
        .into(),
      ),
      rule("ColorFunction", color_function),
    ],
  );
  let mut graphics_opts = opts.graphics_options(&vol);
  graphics_opts.retain(|o| {
    !matches!(o, Expr::Rule { pattern, .. }
      if matches!(pattern.as_ref(), Expr::Identifier(k) if k == "Lighting"))
  });
  let plot = graphics3d(vec![raster, Expr::List(vec![].into())], graphics_opts);
  let plot = evaluate_expr_to_expr(&plot)?;
  Ok(Some(with_legend(plot, &opts, (lo, hi), None)))
}
