//! `Region[reg]` — visual rendering of geometric regions.
//!
//! `Region[reg]` (optionally with Graphics options) displays a geometric
//! region as a plot, matching Wolfram's notebook behavior of visualizing
//! regions with embedding dimension 1–3. The region is converted to the
//! graphics primitives Woxi can already draw and rendered through
//! `Graphics`/`Graphics3D` with the default region styling (the same light
//! blue Wolfram uses for `MeshRegion`).
//!
//! Regions with symbolic coordinates or heads the renderer cannot draw
//! leave the `Region[…]` expression unevaluated, like other unsupported
//! cases elsewhere in the interpreter.

use crate::functions::graphics::Color;
use crate::functions::math_ast::try_eval_to_f64;
use crate::syntax::Expr;

/// Default face color for 2-dimensional regions (the `MeshRegion` blue).
fn area_color() -> Expr {
  Color::new(0.626, 0.836, 0.919).to_expr()
}

/// Default color for curves (1D) and point sets (0D): a darker shade of the
/// area blue so thin strokes and points stay visible.
fn curve_color() -> Expr {
  Color::new(0.24, 0.6, 0.8).to_expr()
}

fn is_num(e: &Expr) -> bool {
  try_eval_to_f64(e).is_some()
}

/// `{x, y}` (dim = 2) or `{x, y, z}` (dim = 3) with numeric entries.
fn is_point(e: &Expr, dim: usize) -> bool {
  matches!(e, Expr::List(items) if items.len() == dim && items.iter().all(is_num))
}

/// Non-empty `{{x1, y1}, …}` list of numeric points of dimension `dim`.
fn is_point_list(e: &Expr, dim: usize) -> bool {
  matches!(e, Expr::List(items) if !items.is_empty() && items.iter().all(|p| is_point(p, dim)))
}

/// Embedding dimension (2 or 3) of a point, a point list, or a list of
/// point lists (multi-segment `Line`).
fn coords_dim(e: &Expr) -> Option<usize> {
  for dim in [2usize, 3] {
    if is_point(e, dim)
      || is_point_list(e, dim)
      || matches!(e, Expr::List(items) if !items.is_empty()
          && items.iter().all(|seg| is_point_list(seg, dim)))
    {
      return Some(dim);
    }
  }
  None
}

/// A numeric radius: a number or (2D only) an `{rx, ry}` pair.
fn is_radius(e: &Expr, dim: usize) -> bool {
  is_num(e) || (dim == 2 && is_point(e, 2))
}

/// Region primitives plus a 3D flag, or `None` when the region cannot be
/// drawn (symbolic coordinates or an unsupported head).
fn region_primitives(reg: &Expr) -> Option<(Vec<Expr>, bool)> {
  let Expr::FunctionCall { name, args } = reg else {
    return None;
  };
  let area_2d = |prim: Expr| Some((vec![area_color(), prim], false));
  let curve_2d = |prim: Expr| Some((vec![curve_color(), prim], false));
  let solid_3d = |prim: Expr| Some((vec![area_color(), prim], true));
  let curve_3d = |prim: Expr| Some((vec![curve_color(), prim], true));

  match name.as_str() {
    // ── Point sets (0D) and curves (1D) ────────────────────────────────
    "Point" if args.len() == 1 => match coords_dim(&args[0])? {
      2 => curve_2d(reg.clone()),
      _ => curve_3d(reg.clone()),
    },
    "Line" if args.len() == 1 => match coords_dim(&args[0])? {
      2 => curve_2d(reg.clone()),
      // The 3D renderer draws a single point list per Line.
      _ if is_point_list(&args[0], 3) => curve_3d(reg.clone()),
      _ => None,
    },
    // Circle[], Circle[c], Circle[c, r] — full circles/ellipses only; the
    // renderer ignores an arc spec, so Circle[c, r, {θ1, θ2}] stays
    // unevaluated rather than drawing a wrong (full) circle.
    "Circle" => match args.len() {
      0 => curve_2d(reg.clone()),
      1 if is_point(&args[0], 2) => curve_2d(reg.clone()),
      2 if is_point(&args[0], 2) && is_radius(&args[1], 2) => {
        curve_2d(reg.clone())
      }
      _ => None,
    },
    // ── Areas (2D) ─────────────────────────────────────────────────────
    "Disk" => match args.len() {
      0 => area_2d(reg.clone()),
      1 if is_point(&args[0], 2) => area_2d(reg.clone()),
      2 if is_point(&args[0], 2) && is_radius(&args[1], 2) => {
        area_2d(reg.clone())
      }
      // Disk[c, r, {θ1, θ2}] — circular sector.
      3 if is_point(&args[0], 2)
        && is_radius(&args[1], 2)
        && is_point(&args[2], 2) =>
      {
        area_2d(reg.clone())
      }
      _ => None,
    },
    "Rectangle" if args.len() <= 2 => {
      if args.iter().all(|a| is_point(a, 2)) {
        area_2d(reg.clone())
      } else {
        None
      }
    }
    "Triangle" => match args.len() {
      // Triangle[] is the standard 2-simplex.
      0 => area_2d(Expr::FunctionCall {
        name: "Triangle".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::List(vec![Expr::Integer(0), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(1), Expr::Integer(0)].into()),
            Expr::List(vec![Expr::Integer(0), Expr::Integer(1)].into()),
          ]
          .into(),
        )]
        .into(),
      }),
      1 if is_point_list(&args[0], 2) => area_2d(reg.clone()),
      // A triangle in 3-space renders as a 3D polygon face.
      1 if is_point_list(&args[0], 3) => solid_3d(Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: args.clone(),
      }),
      _ => None,
    },
    "Polygon" if args.len() == 1 => {
      if is_point_list(&args[0], 2) {
        area_2d(reg.clone())
      } else if is_point_list(&args[0], 3) {
        solid_3d(reg.clone())
      } else {
        None
      }
    }
    // RegularPolygon[n] and RegularPolygon[{x, y}, r, n] (the forms the
    // renderer draws).
    "RegularPolygon" => match args.len() {
      1 if matches!(&args[0], Expr::Integer(n) if *n >= 3) => {
        area_2d(reg.clone())
      }
      3 if is_point(&args[0], 2)
        && is_num(&args[1])
        && matches!(&args[2], Expr::Integer(n) if *n >= 3) =>
      {
        area_2d(reg.clone())
      }
      _ => None,
    },
    // MeshRegion[verts, {Polygon[…], …}] (e.g. from VoronoiMesh) reuses the
    // existing conversion, which brings its own edge/face styling.
    "MeshRegion" if args.len() == 2 => {
      crate::functions::graphics::mesh_region_to_graphics_prims(
        &args[0], &args[1],
      )
      .map(|prims| (prims, false))
    }
    // ── Solids (3D) ────────────────────────────────────────────────────
    // Ball[c, r]: a disk in the plane, a solid ball (drawn as a sphere
    // surface) in space.
    "Ball" => match args.len() {
      0 => solid_3d(Expr::FunctionCall {
        name: "Sphere".to_string(),
        args: args.clone(),
      }),
      1 | 2 if args.len() < 2 || is_num(&args[1]) => {
        if is_point(&args[0], 2) {
          area_2d(Expr::FunctionCall {
            name: "Disk".to_string(),
            args: args.clone(),
          })
        } else if is_point(&args[0], 3) {
          solid_3d(Expr::FunctionCall {
            name: "Sphere".to_string(),
            args: args.clone(),
          })
        } else {
          None
        }
      }
      _ => None,
    },
    // Sphere[c, r]: a circle in the plane, a spherical surface in space.
    "Sphere" => match args.len() {
      0 => solid_3d(reg.clone()),
      1 | 2 if args.len() < 2 || is_num(&args[1]) => {
        if is_point(&args[0], 2) {
          curve_2d(Expr::FunctionCall {
            name: "Circle".to_string(),
            args: args.clone(),
          })
        } else if is_point(&args[0], 3) {
          solid_3d(reg.clone())
        } else {
          None
        }
      }
      _ => None,
    },
    "Cuboid" if args.len() <= 2 => {
      if args.iter().all(|a| is_point(a, 3)) {
        solid_3d(reg.clone())
      } else {
        None
      }
    }
    // Cylinder[] / Cylinder[{p1, p2}] / Cylinder[{p1, p2}, r]; same for Cone.
    "Cylinder" | "Cone" => match args.len() {
      0 => solid_3d(reg.clone()),
      1 | 2
        if is_point_list(&args[0], 3)
          && matches!(&args[0], Expr::List(pts) if pts.len() == 2)
          && (args.len() < 2 || is_num(&args[1])) =>
      {
        solid_3d(reg.clone())
      }
      _ => None,
    },
    _ => None,
  }
}

/// Render `Region[reg, opts…]` to an `Expr::Graphics` (capturing the SVG
/// for notebook front ends), or `None` to leave the call unevaluated.
pub fn region_to_graphics(args: &[Expr]) -> Option<Expr> {
  let reg = args.first()?;
  // Trailing arguments must be options; they pass through to the
  // underlying Graphics/Graphics3D call (Region accepts Graphics options).
  let opts = &args[1..];
  if !opts
    .iter()
    .all(|o| matches!(o, Expr::Rule { .. } | Expr::RuleDelayed { .. }))
  {
    return None;
  }
  let (prims, is_3d) = region_primitives(reg)?;

  let mut g_args: Vec<Expr> = vec![Expr::List(prims.into())];
  g_args.extend(opts.iter().cloned());
  let rendered = if is_3d {
    crate::functions::plot3d::graphics3d_ast(&g_args)
  } else {
    crate::functions::graphics::graphics_ast(&g_args)
  }
  .ok()?;

  // Report `Region` as the head (like GeoGraphics does) while rendering
  // identically to the underlying graphic.
  if let Expr::Graphics {
    svg, is_3d, source, ..
  } = &rendered
  {
    Some(Expr::Graphics {
      svg: svg.clone(),
      is_3d: *is_3d,
      source: source.clone(),
      head: Some("Region".to_string()),
    })
  } else {
    None
  }
}
