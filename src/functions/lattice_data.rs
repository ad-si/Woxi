//! `LatticeData` — the three cubic Bravais lattices (simple, body-centered
//! and face-centered cubic) with their conventional unit cell as a
//! `Graphics3D` image plus a few scalar properties.

use super::*;

/// Canonical lattice name for a user-given name or alias.
fn canonical_name(name: &str) -> Option<&'static str> {
  match name {
    "SimpleCubic" | "SC" | "Cubic" | "CubicPrimitive" => Some("SimpleCubic"),
    "BodyCenteredCubic" | "BCC" | "CubicBodyCentered" => {
      Some("BodyCenteredCubic")
    }
    "FaceCenteredCubic" | "FCC" | "CubicFaceCentered" => {
      Some("FaceCenteredCubic")
    }
    _ => None,
  }
}

/// Lattice points of the conventional cubic cell, in unit-cell coordinates.
fn cell_points(lattice: &str) -> Vec<[f64; 3]> {
  let mut pts = Vec::new();
  for &x in &[0.0, 1.0] {
    for &y in &[0.0, 1.0] {
      for &z in &[0.0, 1.0] {
        pts.push([x, y, z]);
      }
    }
  }
  match lattice {
    "BodyCenteredCubic" => pts.push([0.5, 0.5, 0.5]),
    "FaceCenteredCubic" => {
      for axis in 0..3 {
        for &v in &[0.0, 1.0] {
          let mut p = [0.5; 3];
          p[axis] = v;
          pts.push(p);
        }
      }
    }
    _ => {}
  }
  pts
}

/// `Graphics3D` of the conventional cell: a sphere per lattice point and
/// the twelve cube edges as lines.
fn unit_cell_image(lattice: &str) -> Result<Expr, InterpreterError> {
  let fmt = |p: &[f64; 3]| format!("{{{:?},{:?},{:?}}}", p[0], p[1], p[2]);
  let spheres: Vec<String> = cell_points(lattice).iter().map(fmt).collect();
  let mut edges = Vec::new();
  for a in 0..8u8 {
    for bit in [1u8, 2, 4] {
      let b = a | bit;
      if b != a {
        let corner = |m: u8| {
          [(m & 1) as f64, ((m >> 1) & 1) as f64, ((m >> 2) & 1) as f64]
        };
        edges.push(format!(
          "Line[{{{},{}}}]",
          fmt(&corner(a)),
          fmt(&corner(b))
        ));
      }
    }
  }
  let src = format!(
    "Graphics3D[{{{{Gray, {}}}, {{Sphere[{{{}}}, 0.12]}}}}]",
    edges.join(", "),
    spheres.join(", ")
  );
  let parsed = crate::functions::string_ast::parse_program_to_expr(&src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

pub fn lattice_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("LatticeData", args));
  let Some(Expr::String(name)) = args.first() else {
    return unevaluated();
  };
  let Some(lattice) = canonical_name(name) else {
    return unevaluated();
  };
  let prop = match args.get(1) {
    None => "Image",
    Some(Expr::String(p)) => p.as_str(),
    _ => return unevaluated(),
  };
  if args.len() > 2 {
    return unevaluated();
  }
  match prop {
    "Image" => unit_cell_image(lattice),
    "CoordinationNumber" => Ok(Expr::Integer(match lattice {
      "SimpleCubic" => 6,
      "BodyCenteredCubic" => 8,
      _ => 12,
    })),
    "Name" => Ok(Expr::String(lattice.to_string())),
    // Generating vectors, as wolframscript lists them (the simple cubic
    // identity as a SparseArray).
    "Basis" => {
      let src = match lattice {
        "SimpleCubic" => "SparseArray[IdentityMatrix[3]]",
        "BodyCenteredCubic" => "{{2, 0, 0}, {0, 2, 0}, {1, 1, 1}}",
        _ => "{{-1, -1, 0}, {1, -1, 0}, {0, 1, -1}}",
      };
      crate::evaluator::evaluate_expr_to_expr(
        &crate::functions::string_ast::parse_program_to_expr(src)?,
      )
    }
    "LatticePoints" | "UnitCellPoints" => {
      let src = format!(
        "{{{}}}",
        cell_points(lattice)
          .iter()
          .map(|p| format!("{{{:?},{:?},{:?}}}", p[0], p[1], p[2]))
          .collect::<Vec<_>>()
          .join(", ")
      );
      crate::evaluator::evaluate_expr_to_expr(
        &crate::functions::string_ast::parse_program_to_expr(&src)?,
      )
    }
    _ => unevaluated(),
  }
}
