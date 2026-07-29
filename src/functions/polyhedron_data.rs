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
    // Base triangle in the z = -Inradius plane, apex on the +z axis.
    vertices_src: "{\
      {-1/(2*Sqrt[3]), -1/2, -1/(2*Sqrt[6])}, \
      {-1/(2*Sqrt[3]), 1/2, -1/(2*Sqrt[6])}, \
      {1/Sqrt[3], 0, -1/(2*Sqrt[6])}, \
      {0, 0, Sqrt[3/8]}}",
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
      {-1/Sqrt[2], 0, 0}, {1/Sqrt[2], 0, 0}, \
      {0, -1/Sqrt[2], 0}, {0, 1/Sqrt[2], 0}, \
      {0, 0, -1/Sqrt[2]}, {0, 0, 1/Sqrt[2]}}",
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
      {-Sqrt[1 + 2/Sqrt[5]], 0, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1 + 2/Sqrt[5]], 0, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[5/8 + 11*Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], -(3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {Sqrt[1/8 + Sqrt[5]/40], (3 + Sqrt[5])/4, Sqrt[1/8 - Sqrt[5]/40]}, \
      {-Sqrt[1/2 + Sqrt[5]/10], 0, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/2 + Sqrt[5]/10], 0, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], -1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/4 + Sqrt[5]/10], 1/2, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], -1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/4 + Sqrt[5]/10], 1/2, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {-Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, -Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], -(1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}, \
      {Sqrt[1/8 - Sqrt[5]/40], (1 + Sqrt[5])/4, Sqrt[5/8 + 11*Sqrt[5]/40]}}",
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

/// Compute the faces of a convex polyhedron from its vertices: every plane
/// through three vertices that has all remaining vertices strictly on one
/// side is a supporting plane, and the vertices lying on it (ordered by
/// angle around the face centroid) form a face.
fn convex_faces(vertices: &[[f64; 3]]) -> Vec<Vec<usize>> {
  const EPS: f64 = 1e-9;
  let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  let cross = |a: [f64; 3], b: [f64; 3]| {
    [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]
  };
  let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let norm = |a: [f64; 3]| dot(a, a).sqrt();

  let n = vertices.len();
  let mut seen: std::collections::HashSet<Vec<usize>> =
    std::collections::HashSet::new();
  let mut faces: Vec<Vec<usize>> = Vec::new();
  for i in 0..n {
    for j in i + 1..n {
      for k in j + 1..n {
        let mut normal =
          cross(sub(vertices[j], vertices[i]), sub(vertices[k], vertices[i]));
        let len = norm(normal);
        if len < EPS {
          continue;
        }
        normal = [normal[0] / len, normal[1] / len, normal[2] / len];
        let d = dot(normal, vertices[i]);
        let mut above = false;
        let mut below = false;
        let mut on_plane = Vec::new();
        for (idx, v) in vertices.iter().enumerate() {
          let s = dot(normal, *v) - d;
          if s > EPS {
            above = true;
          } else if s < -EPS {
            below = true;
          } else {
            on_plane.push(idx);
          }
        }
        if above && below {
          continue;
        }
        // Orient the normal outward so face winding is consistent.
        let outward = if above {
          [-normal[0], -normal[1], -normal[2]]
        } else {
          normal
        };
        // Order face vertices by angle around the face centroid.
        let centroid = on_plane.iter().fold([0.0; 3], |acc, &idx| {
          [
            acc[0] + vertices[idx][0] / on_plane.len() as f64,
            acc[1] + vertices[idx][1] / on_plane.len() as f64,
            acc[2] + vertices[idx][2] / on_plane.len() as f64,
          ]
        });
        let x_axis = {
          let v = sub(vertices[on_plane[0]], centroid);
          let len = norm(v);
          [v[0] / len, v[1] / len, v[2] / len]
        };
        let y_axis = cross(outward, x_axis);
        let mut ordered = on_plane.clone();
        ordered.sort_by(|&a, &b| {
          let angle = |idx: usize| {
            let v = sub(vertices[idx], centroid);
            dot(v, y_axis).atan2(dot(v, x_axis))
          };
          angle(a)
            .partial_cmp(&angle(b))
            .unwrap_or(std::cmp::Ordering::Equal)
        });
        // Many vertex triples describe the same plane; deduplicate faces
        // by their sorted index set.
        let mut key = ordered.clone();
        key.sort_unstable();
        if seen.insert(key) {
          faces.push(ordered);
        }
      }
    }
  }
  faces
}

/// Build the Graphics3D expression for a polyhedron and evaluate it into
/// the rendered graphics object.
fn polyhedron_graphics(
  info: &PolyhedronInfo,
) -> Result<Expr, InterpreterError> {
  let vertices = numeric_vertices(info)?;
  let faces = convex_faces(&vertices);
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
        "Insphere" => insphere(info),
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
