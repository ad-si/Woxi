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
  /// Vertex coordinates for unit edge length (used for rendering).
  vertices: fn() -> Vec<[f64; 3]>,
}

const PHI: f64 = 1.618_033_988_749_895; // golden ratio (1 + Sqrt[5])/2

fn tetrahedron_vertices() -> Vec<[f64; 3]> {
  // Edge length of {±1, ±1, ±1} alternated corners is 2 Sqrt[2].
  let s = 1.0 / (2.0 * std::f64::consts::SQRT_2);
  vec![[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]]
}

fn cube_vertices() -> Vec<[f64; 3]> {
  let mut v = Vec::with_capacity(8);
  for &x in &[-0.5, 0.5] {
    for &y in &[-0.5, 0.5] {
      for &z in &[-0.5, 0.5] {
        v.push([x, y, z]);
      }
    }
  }
  v
}

fn octahedron_vertices() -> Vec<[f64; 3]> {
  let s = 1.0 / std::f64::consts::SQRT_2;
  vec![
    [s, 0.0, 0.0],
    [-s, 0.0, 0.0],
    [0.0, s, 0.0],
    [0.0, -s, 0.0],
    [0.0, 0.0, s],
    [0.0, 0.0, -s],
  ]
}

fn dodecahedron_vertices() -> Vec<[f64; 3]> {
  // The classic (±1, ±1, ±1) / (0, ±1/φ, ±φ) / … coordinates have edge
  // length 2/φ; scale by φ/2 for a unit edge.
  let s = PHI / 2.0;
  let a = 1.0 / PHI;
  let mut v = Vec::with_capacity(20);
  for &x in &[-1.0, 1.0] {
    for &y in &[-1.0, 1.0] {
      for &z in &[-1.0, 1.0] {
        v.push([s * x, s * y, s * z]);
      }
    }
  }
  for &p in &[-1.0f64, 1.0] {
    for &q in &[-1.0f64, 1.0] {
      v.push([0.0, s * p * a, s * q * PHI]);
      v.push([s * p * a, s * q * PHI, 0.0]);
      v.push([s * p * PHI, 0.0, s * q * a]);
    }
  }
  v
}

fn icosahedron_vertices() -> Vec<[f64; 3]> {
  // Cyclic permutations of (0, ±1, ±φ) have edge length 2; halve for unit.
  let mut v = Vec::with_capacity(12);
  for &p in &[-0.5f64, 0.5] {
    for &q in &[-0.5f64, 0.5] {
      v.push([0.0, p, q * PHI]);
      v.push([p, q * PHI, 0.0]);
      v.push([q * PHI, 0.0, p]);
    }
  }
  v
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
    vertices: tetrahedron_vertices,
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
    vertices: cube_vertices,
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
    vertices: octahedron_vertices,
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
    vertices: dodecahedron_vertices,
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
    vertices: icosahedron_vertices,
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
  let vertices = (info.vertices)();
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
  "FaceCount",
  "Inradius",
  "SurfaceArea",
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
        _ => unevaluated(),
      }
    }
    _ => unevaluated(),
  }
}
