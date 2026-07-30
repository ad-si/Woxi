//! PolyhedronData[name] and PolyhedronData[name, property] for the Platonic
//! solids. All metric properties refer to unit edge length and are stored as
//! exact Wolfram Language expressions so results stay symbolic.

use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

struct PolyhedronInfo {
  name: &'static str,
  vertex_count: i128,
  edge_count: i128,
  face_count: i128,
  /// Exact metric properties (unit edge length) as WL source.
  volume: &'static str,
  surface_area: &'static str,
  circumradius: &'static str,
  inradius: &'static str,
  /// Exact vertex coordinates for unit edge length, as WL source, in
  /// Wolfram's canonical orientation and vertex order (the z axis is the
  /// polar symmetry axis where there is one). Used both for the
  /// "VertexCoordinates" property and (numerically) for rendering.
  vertices_src: &'static str,
  /// Faces as 1-based indices into `vertices_src`, in Wolfram's order and
  /// winding, as WL source. Used both for the "FaceIndices" property and
  /// for rendering the solid.
  faces_src: &'static str,
}

static POLYHEDRA: &[PolyhedronInfo] = &[
  PolyhedronInfo {
    name: "Tetrahedron",
    vertex_count: 4,
    edge_count: 6,
    face_count: 4,
    volume: "1/(6*Sqrt[2])",
    surface_area: "Sqrt[3]",
    circumradius: "Sqrt[3/8]",
    inradius: "1/(2*Sqrt[6])",
    // Apex on the +z axis, then the base triangle in the
    // z = -Inradius plane.
    vertices_src: "{\
      {0, 0, Sqrt[2/3] - 1/(2*Sqrt[6])}, \
      {-1/(2*Sqrt[3]), -1/2, -1/(2*Sqrt[6])}, \
      {-1/(2*Sqrt[3]), 1/2, -1/(2*Sqrt[6])}, \
      {1/Sqrt[3], 0, -1/(2*Sqrt[6])}}",
    faces_src: "{{2, 3, 4}, {3, 2, 1}, {4, 1, 2}, {1, 4, 3}}",
  },
  PolyhedronInfo {
    name: "Cube",
    vertex_count: 8,
    edge_count: 12,
    face_count: 6,
    volume: "1",
    surface_area: "6",
    circumradius: "Sqrt[3]/2",
    inradius: "1/2",
    vertices_src: "{\
      {-1/2, -1/2, -1/2}, {-1/2, -1/2, 1/2}, \
      {-1/2, 1/2, -1/2}, {-1/2, 1/2, 1/2}, \
      {1/2, -1/2, -1/2}, {1/2, -1/2, 1/2}, \
      {1/2, 1/2, -1/2}, {1/2, 1/2, 1/2}}",
    faces_src: "{{8, 4, 2, 6}, {8, 6, 5, 7}, {8, 7, 3, 4}, \
      {4, 3, 1, 2}, {1, 3, 7, 5}, {2, 1, 5, 6}}",
  },
  PolyhedronInfo {
    name: "Octahedron",
    vertex_count: 6,
    edge_count: 12,
    face_count: 8,
    volume: "Sqrt[2]/3",
    surface_area: "2*Sqrt[3]",
    circumradius: "1/Sqrt[2]",
    inradius: "1/Sqrt[6]",
    vertices_src: "{\
      {-1/Sqrt[2], 0, 0}, {0, 1/Sqrt[2], 0}, \
      {0, 0, -1/Sqrt[2]}, {0, 0, 1/Sqrt[2]}, \
      {0, -1/Sqrt[2], 0}, {1/Sqrt[2], 0, 0}}",
    faces_src: "{{4, 5, 6}, {4, 6, 2}, {4, 2, 1}, {4, 1, 5}, \
      {5, 1, 3}, {5, 3, 6}, {3, 1, 2}, {6, 3, 2}}",
  },
  PolyhedronInfo {
    name: "Dodecahedron",
    vertex_count: 20,
    edge_count: 30,
    face_count: 12,
    volume: "(15 + 7*Sqrt[5])/4",
    surface_area: "3*Sqrt[5*(5 + 2*Sqrt[5])]",
    circumradius: "(Sqrt[15] + Sqrt[3])/4",
    inradius: "Sqrt[250 + 110*Sqrt[5]]/20",
    // Two wide vertex rings around the equator (antipodal pairs first),
    // then the two rings of the top and bottom faces; z is the C5 axis.
    vertices_src: "{\
      {-Sqrt[1 + 2/Sqrt[5]], 0, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1 + 2/Sqrt[5]], 0, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], -1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], 1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], -1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], 1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/2 + Sqrt[5]/10], 0, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/2 + Sqrt[5]/10], 0, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}}",
    faces_src: "{{15, 10, 9, 14, 1}, {2, 6, 12, 11, 5}, {5, 11, 7, 3, 19}, \
      {11, 12, 8, 16, 7}, {12, 6, 20, 4, 8}, {6, 2, 13, 18, 20}, \
      {2, 5, 19, 17, 13}, {4, 20, 18, 10, 15}, {18, 13, 17, 9, 10}, \
      {17, 19, 3, 14, 9}, {3, 7, 16, 1, 14}, {16, 8, 4, 15, 1}}",
  },
  PolyhedronInfo {
    name: "Icosahedron",
    vertex_count: 12,
    edge_count: 30,
    face_count: 20,
    volume: "(5*(3 + Sqrt[5]))/12",
    surface_area: "5*Sqrt[3]",
    circumradius: "Sqrt[10 + 2*Sqrt[5]]/4",
    inradius: "(3*Sqrt[3] + Sqrt[15])/12",
    // The two poles, then the two staggered vertex rings (antipodal
    // pairs adjacent); z is the C5 axis through the poles.
    vertices_src: "{\
      {0, 0, -Sqrt[5/8 + Sqrt[5]/8]}, \
      {0, 0, Sqrt[5/8 + Sqrt[5]/8]}, \
      {-Sqrt[1/2 + Sqrt[5]/10], 0, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/2 + Sqrt[5]/10], 0, Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], -1/2, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], 1/2, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], -1/2, Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], 1/2, Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[1/8 + Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[1/8 + Sqrt[5]/40]}}",
    faces_src: "{{2, 12, 8}, {2, 8, 7}, {2, 7, 11}, {2, 11, 4}, {2, 4, 12}, \
      {5, 9, 1}, {6, 5, 1}, {10, 6, 1}, {3, 10, 1}, {9, 3, 1}, \
      {12, 10, 8}, {8, 3, 7}, {7, 9, 11}, {11, 5, 4}, {4, 6, 12}, \
      {5, 11, 9}, {6, 4, 5}, {10, 12, 6}, {3, 8, 10}, {9, 7, 3}}",
  },
  // The rhombic dodecahedron is the one Catalan solid here: its faces are
  // rhombi, not regular polygons, so it has no circumradius (its vertices
  // are not all the same distance from the center).
  PolyhedronInfo {
    name: "RhombicDodecahedron",
    vertex_count: 14,
    edge_count: 24,
    face_count: 12,
    volume: "16/(3*Sqrt[3])",
    surface_area: "8*Sqrt[2]",
    circumradius: "Missing[\"NotApplicable\"]",
    inradius: "Sqrt[2/3]",
    vertices_src: "{\
      {-Sqrt[2/3], -Sqrt[2/3], 0}, {-Sqrt[2/3], 0, -1/Sqrt[3]}, \
      {-Sqrt[2/3], 0, 1/Sqrt[3]}, {-Sqrt[2/3], Sqrt[2/3], 0}, \
      {0, -Sqrt[2/3], -1/Sqrt[3]}, {0, -Sqrt[2/3], 1/Sqrt[3]}, \
      {0, 0, -2/Sqrt[3]}, {0, 0, 2/Sqrt[3]}, \
      {0, Sqrt[2/3], -1/Sqrt[3]}, {0, Sqrt[2/3], 1/Sqrt[3]}, \
      {Sqrt[2/3], -Sqrt[2/3], 0}, {Sqrt[2/3], 0, -1/Sqrt[3]}, \
      {Sqrt[2/3], 0, 1/Sqrt[3]}, {Sqrt[2/3], Sqrt[2/3], 0}}",
    faces_src: "{{2, 1, 3, 4}, {1, 2, 7, 5}, {6, 8, 3, 1}, {2, 4, 9, 7}, \
      {8, 10, 4, 3}, {11, 6, 1, 5}, {9, 4, 10, 14}, {5, 7, 12, 11}, \
      {11, 13, 8, 6}, {7, 9, 14, 12}, {13, 14, 10, 8}, {14, 13, 11, 12}}",
  },
];

fn find_polyhedron(name: &str) -> Option<&'static PolyhedronInfo> {
  // "Hexahedron" is the standard alternative name for the cube.
  let name = if name == "Hexahedron" { "Cube" } else { name };
  POLYHEDRA.iter().find(|p| p.name == name)
}

/// The exact unit-edge volume of a Platonic solid, as WL source.
pub fn unit_volume_src(name: &str) -> Option<&'static str> {
  find_polyhedron(name).map(|p| p.volume)
}

/// The exact unit-edge surface area of a Platonic solid, as WL source.
pub fn unit_surface_area_src(name: &str) -> Option<&'static str> {
  find_polyhedron(name).map(|p| p.surface_area)
}

/// Evaluate a polyhedron's exact vertex list to numeric `[x, y, z]` rows
/// (for rendering and for deriving the edge list).
fn numeric_vertices(
  info: &PolyhedronInfo,
) -> Result<Vec<[f64; 3]>, InterpreterError> {
  let evaluated = eval_wl(&format!("N[{}]", info.vertices_src))?;
  let Expr::List(rows) = &evaluated else {
    return Err(InterpreterError::EvaluationError(format!(
      "PolyhedronData: vertex data for {} did not evaluate to a list",
      info.name
    )));
  };
  let mut vertices = Vec::with_capacity(rows.len());
  for row in rows.iter() {
    let Expr::List(coords) = row else {
      return Err(InterpreterError::EvaluationError(format!(
        "PolyhedronData: vertex row for {} is not a coordinate triple",
        info.name
      )));
    };
    let mut point = [0.0; 3];
    for (slot, coord) in point.iter_mut().zip(coords.iter()) {
      *slot = match coord {
        Expr::Real(r) => *r,
        Expr::Integer(i) => *i as f64,
        _ => {
          return Err(InterpreterError::EvaluationError(format!(
            "PolyhedronData: vertex coordinate for {} is not numeric",
            info.name
          )));
        }
      };
    }
    vertices.push(point);
  }
  Ok(vertices)
}

/// The edges of a polyhedron as 1-based vertex index pairs, in canonical
/// (lexicographic) order: every vertex pair at minimal (= edge) distance.
fn edge_indices(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  let vertices = numeric_vertices(info)?;
  let dist = |a: [f64; 3], b: [f64; 3]| -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
  };
  let mut min_dist = f64::INFINITY;
  for i in 0..vertices.len() {
    for j in i + 1..vertices.len() {
      min_dist = min_dist.min(dist(vertices[i], vertices[j]));
    }
  }
  let mut pairs = Vec::new();
  for i in 0..vertices.len() {
    for j in i + 1..vertices.len() {
      if dist(vertices[i], vertices[j]) < min_dist * (1.0 + 1e-9) {
        pairs.push(Expr::List(
          vec![Expr::Integer(i as i128 + 1), Expr::Integer(j as i128 + 1)]
            .into(),
        ));
      }
    }
  }
  Ok(Expr::List(pairs.into()))
}

/// `Sphere[{0, 0, 0}, r]` with the polyhedron's exact inradius: the sphere
/// inscribed in the (origin-centered) solid.
fn insphere(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  let center = Expr::List(
    vec![Expr::Integer(0), Expr::Integer(0), Expr::Integer(0)].into(),
  );
  let radius = eval_wl(info.inradius)?;
  Ok(Expr::FunctionCall {
    name: "Sphere".to_string(),
    args: vec![center, radius].into(),
  })
}

/// The faces of a polyhedron as 1-based vertex index lists, in Wolfram's
/// order and winding.
fn face_indices(info: &PolyhedronInfo) -> Result<Expr, InterpreterError> {
  eval_wl(info.faces_src)
}

/// The face index lists as plain `usize` rows, for rendering.
fn numeric_faces(
  info: &PolyhedronInfo,
) -> Result<Vec<Vec<usize>>, InterpreterError> {
  let evaluated = face_indices(info)?;
  let Expr::List(rows) = &evaluated else {
    return Err(InterpreterError::EvaluationError(format!(
      "PolyhedronData: face data for {} did not evaluate to a list",
      info.name
    )));
  };
  let mut faces = Vec::with_capacity(rows.len());
  for row in rows.iter() {
    let Expr::List(items) = row else {
      return Err(InterpreterError::EvaluationError(format!(
        "PolyhedronData: face row for {} is not a list",
        info.name
      )));
    };
    let mut face = Vec::with_capacity(items.len());
    for item in items.iter() {
      let Expr::Integer(idx) = item else {
        return Err(InterpreterError::EvaluationError(format!(
          "PolyhedronData: face index for {} is not an integer",
          info.name
        )));
      };
      face.push(*idx as usize - 1);
    }
    faces.push(face);
  }
  Ok(faces)
}

/// Build the Graphics3D expression for a polyhedron and evaluate it into
/// the rendered graphics object.
fn polyhedron_graphics(
  info: &PolyhedronInfo,
) -> Result<Expr, InterpreterError> {
  let vertices = numeric_vertices(info)?;
  let faces = numeric_faces(info)?;
  let polygons: Vec<Expr> = faces
    .iter()
    .map(|face| {
      let pts: Vec<Expr> = face
        .iter()
        .map(|&idx| {
          Expr::List(
            vertices[idx]
              .iter()
              .map(|&c| Expr::Real(c))
              .collect::<Vec<_>>()
              .into(),
          )
        })
        .collect();
      Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(pts.into())].into(),
      }
    })
    .collect();
  let graphics = Expr::FunctionCall {
    name: "Graphics3D".to_string(),
    args: vec![Expr::List(polygons.into())].into(),
  };
  crate::evaluator::evaluate_expr_to_expr(&graphics)
}

/// Evaluate a stored exact WL value.
fn eval_wl(src: &str) -> Result<Expr, InterpreterError> {
  let parsed = crate::functions::string_ast::parse_program_to_expr(src)?;
  crate::evaluator::evaluate_expr_to_expr(&parsed)
}

/// Metric/count properties exposed by `PolyhedronData[name, property]`,
/// returned (sorted) by `PolyhedronData["Properties"]`.
static PROPERTIES: &[&str] = &[
  "Circumradius",
  "EdgeCount",
  "EdgeIndices",
  "FaceCount",
  "FaceIndices",
  "Inradius",
  "Insphere",
  "SurfaceArea",
  "VertexCoordinates",
  "VertexCount",
  "Volume",
];

/// Classes the built-in solids belong to, returned by
/// `PolyhedronData["Classes"]`. Disjoint from `PROPERTIES`.
static CLASSES: &[&str] = &["Convex", "Platonic", "Regular"];

/// Build a `List` of string entries.
fn string_list(items: &[&str]) -> Expr {
  Expr::List(
    items
      .iter()
      .map(|s| Expr::String(s.to_string()))
      .collect::<Vec<_>>()
      .into(),
  )
}

pub fn polyhedron_data_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || Ok(unevaluated("PolyhedronData", args));

  // `PolyhedronData[All]` — the list of known entities (by name).
  if let Some(Expr::Identifier(sym)) = args.first()
    && sym == "All"
    && args.len() == 1
  {
    let mut names: Vec<&str> = POLYHEDRA.iter().map(|p| p.name).collect();
    names.sort_unstable();
    return Ok(string_list(&names));
  }

  // `PolyhedronData["Properties"]` / `PolyhedronData["Classes"]` — the
  // available property and class names. Handled before `find_polyhedron`
  // so these reserved strings aren't reported as unknown entities.
  if let Some(Expr::String(kind)) = args.first()
    && args.len() == 1
  {
    match kind.as_str() {
      "Properties" => return Ok(string_list(PROPERTIES)),
      "Classes" => return Ok(string_list(CLASSES)),
      _ => {}
    }
  }

  let Some(Expr::String(name)) = args.first() else {
    return unevaluated();
  };
  let Some(info) = find_polyhedron(name) else {
    crate::emit_message(&format!(
      "PolyhedronData::notent: {name} is not a known entity, class, or tag for PolyhedronData. Use PolyhedronData[] for a list of entities."
    ));
    // Wolfram emits the message but leaves the call unevaluated.
    return unevaluated();
  };
  match args.len() {
    1 => polyhedron_graphics(info),
    2 => {
      let Expr::String(property) = &args[1] else {
        return unevaluated();
      };
      match property.as_str() {
        "VertexCount" => Ok(Expr::Integer(info.vertex_count)),
        "EdgeCount" => Ok(Expr::Integer(info.edge_count)),
        "FaceCount" => Ok(Expr::Integer(info.face_count)),
        "Volume" => eval_wl(info.volume),
        "SurfaceArea" => eval_wl(info.surface_area),
        "Circumradius" => eval_wl(info.circumradius),
        "Inradius" => eval_wl(info.inradius),
        "VertexCoordinates" => eval_wl(info.vertices_src),
        "EdgeIndices" => edge_indices(info),
        "FaceIndices" => face_indices(info),
        "Insphere" => insphere(info),
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
