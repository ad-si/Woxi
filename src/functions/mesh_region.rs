//! Mesh regions: the `MeshRegion[coords, cells]` and
//! `BoundaryMeshRegion[coords, cells]` objects `ConvexHullMesh`,
//! `DelaunayMesh` and `VoronoiMesh` build, and the accessors and measures
//! that read them.
//!
//! The measures delegate to the ordinary `Polygon` machinery — one polygon
//! per 2-cell, or the boundary loop of a boundary mesh — so exact
//! coordinates keep exact answers.

use crate::InterpreterError;
use crate::syntax::{Expr, unevaluated};

/// A parsed mesh: its coordinates and its cells by dimension.
pub struct Mesh {
  pub coordinates: Vec<Expr>,
  /// Index lists per dimension: `cells[0]` are points, `cells[1]` lines,
  /// `cells[2]` polygons. Indices are 1-based, as they are written.
  pub cells: Vec<Vec<Vec<usize>>>,
  /// True for `BoundaryMeshRegion`, whose cells describe the boundary only.
  pub boundary: bool,
}

/// Whether the expression is a mesh region object.
pub fn is_mesh_region(expr: &Expr) -> bool {
  matches!(expr, Expr::FunctionCall { name, .. }
    if name == "MeshRegion" || name == "BoundaryMeshRegion")
}

fn index_list(expr: &Expr) -> Option<Vec<usize>> {
  let Expr::List(items) = expr else {
    return None;
  };
  let mut out = Vec::with_capacity(items.len());
  for item in items.iter() {
    match item {
      Expr::Integer(i) if *i >= 1 => out.push(*i as usize),
      _ => return None,
    }
  }
  Some(out)
}

/// Read a `MeshRegion` / `BoundaryMeshRegion` object. Cell primitives may
/// carry one index list or a list of them (`Polygon[{{1, 2, 3}, {3, 2, 4}}]`).
pub fn parse_mesh(expr: &Expr) -> Option<Mesh> {
  let Expr::FunctionCall { name, args } = expr else {
    return None;
  };
  let boundary = match name.as_str() {
    "MeshRegion" => false,
    "BoundaryMeshRegion" => true,
    _ => return None,
  };
  if args.len() < 2 {
    return None;
  }
  let Expr::List(coords) = &args[0] else {
    return None;
  };
  let Expr::List(cell_specs) = &args[1] else {
    return None;
  };
  let mut cells: Vec<Vec<Vec<usize>>> =
    vec![Vec::new(), Vec::new(), Vec::new()];
  for spec in cell_specs.iter() {
    let Expr::FunctionCall {
      name: head,
      args: spec_args,
    } = spec
    else {
      return None;
    };
    let dimension = match head.as_str() {
      "Point" => 0,
      "Line" => 1,
      "Polygon" => 2,
      _ => return None,
    };
    if spec_args.len() != 1 {
      return None;
    }
    // Either one index list, or a list of them.
    if let Some(single) = index_list(&spec_args[0]) {
      cells[dimension].push(single);
    } else if let Expr::List(groups) = &spec_args[0] {
      for group in groups.iter() {
        cells[dimension].push(index_list(group)?);
      }
    } else {
      return None;
    }
  }
  Some(Mesh {
    coordinates: coords.iter().cloned().collect(),
    cells,
    boundary,
  })
}

impl Mesh {
  /// The coordinates an index list names.
  fn points(&self, indices: &[usize]) -> Option<Vec<Expr>> {
    indices
      .iter()
      .map(|i| self.coordinates.get(*i - 1).cloned())
      .collect()
  }

  /// The dimension of the region: the highest cell dimension present, with a
  /// boundary mesh standing for the region its boundary encloses.
  pub fn dimension(&self) -> usize {
    let highest = (0..3)
      .rev()
      .find(|d| !self.cells[*d].is_empty())
      .unwrap_or(0);
    if self.boundary { highest + 1 } else { highest }
  }

  /// The polygons the region covers: its 2-cells, or the loop a boundary mesh
  /// of lines encloses.
  pub fn polygons(&self) -> Vec<Expr> {
    if !self.cells[2].is_empty() {
      return self.cells[2]
        .iter()
        .filter_map(|face| self.points(face))
        .map(|pts| Expr::FunctionCall {
          name: "Polygon".to_string(),
          args: vec![Expr::List(pts.into())].into(),
        })
        .collect();
    }
    if self.boundary
      && let Some(loop_indices) = self.boundary_loop()
      && let Some(pts) = self.points(&loop_indices)
    {
      return vec![Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(pts.into())].into(),
      }];
    }
    Vec::new()
  }

  /// Walk the boundary lines into a single closed loop of vertex indices.
  fn boundary_loop(&self) -> Option<Vec<usize>> {
    let edges = &self.cells[1];
    if edges.is_empty() || edges.iter().any(|e| e.len() != 2) {
      return None;
    }
    let mut remaining: Vec<[usize; 2]> =
      edges.iter().map(|e| [e[0], e[1]]).collect();
    let start = remaining[0][0];
    let mut loop_indices = vec![start];
    let mut current = remaining.remove(0)[1];
    while current != start {
      loop_indices.push(current);
      let next = remaining
        .iter()
        .position(|e| e[0] == current || e[1] == current)?;
      let edge = remaining.remove(next);
      current = if edge[0] == current { edge[1] } else { edge[0] };
    }
    Some(loop_indices)
  }

  /// The edges of the region that belong to exactly one 2-cell — its
  /// boundary. A boundary mesh already lists them.
  fn boundary_edges(&self) -> Vec<[usize; 2]> {
    if self.cells[2].is_empty() {
      return self.cells[1]
        .iter()
        .filter(|e| e.len() == 2)
        .map(|e| [e[0], e[1]])
        .collect();
    }
    let mut counts: Vec<([usize; 2], usize)> = Vec::new();
    for face in &self.cells[2] {
      for i in 0..face.len() {
        let a = face[i];
        let b = face[(i + 1) % face.len()];
        let key = if a <= b { [a, b] } else { [b, a] };
        match counts.iter_mut().find(|(e, _)| *e == key) {
          Some((_, n)) => *n += 1,
          None => counts.push((key, 1)),
        }
      }
    }
    counts
      .into_iter()
      .filter(|(_, n)| *n == 1)
      .map(|(e, _)| e)
      .collect()
  }

  /// The cell primitives of dimension `d`, written with indices.
  pub fn cell_expressions(&self, d: usize) -> Option<Vec<Expr>> {
    let head = match d {
      0 => "Point",
      1 => "Line",
      2 => "Polygon",
      _ => return None,
    };
    // Every coordinate is a 0-cell, whether or not the object lists them.
    if d == 0 && self.cells[0].is_empty() {
      return Some(
        (1..=self.coordinates.len())
          .map(|i| Expr::FunctionCall {
            name: "Point".to_string(),
            args: vec![Expr::Integer(i as i128)].into(),
          })
          .collect(),
      );
    }
    // A boundary mesh fills its interior: the loop is its single 2-cell.
    if d == 2 && self.cells[2].is_empty() && self.boundary {
      let loop_indices = self.boundary_loop()?;
      return Some(vec![Expr::FunctionCall {
        name: "Polygon".to_string(),
        args: vec![Expr::List(
          loop_indices
            .into_iter()
            .map(|i| Expr::Integer(i as i128))
            .collect(),
        )]
        .into(),
      }]);
    }
    Some(
      self.cells[d]
        .iter()
        .map(|indices| {
          let list = Expr::List(
            indices.iter().map(|i| Expr::Integer(*i as i128)).collect(),
          );
          Expr::FunctionCall {
            name: head.to_string(),
            args: vec![if d == 0 {
              Expr::Integer(indices[0] as i128)
            } else {
              list
            }]
            .into(),
          }
        })
        .collect(),
    )
  }

  /// The same cells with their coordinates written out.
  pub fn cell_primitives(&self, d: usize) -> Option<Vec<Expr>> {
    let head = match d {
      0 => "Point",
      1 => "Line",
      2 => "Polygon",
      _ => return None,
    };
    let cells = self.cell_expressions(d)?;
    let mut out = Vec::with_capacity(cells.len());
    for cell in cells {
      let Expr::FunctionCall { args, .. } = &cell else {
        return None;
      };
      let indices = match &args[0] {
        Expr::Integer(i) => vec![*i as usize],
        other => index_list(other)?,
      };
      let pts = self.points(&indices)?;
      out.push(Expr::FunctionCall {
        name: head.to_string(),
        args: vec![if d == 0 {
          pts[0].clone()
        } else {
          Expr::List(pts.into())
        }]
        .into(),
      });
    }
    Some(out)
  }

  /// `{points, lines, faces…}` — the number of cells of each dimension. The
  /// edges of a mesh of faces are counted even though the object lists only
  /// the faces; a boundary mesh counts only the boundary it names.
  pub fn cell_counts(&self) -> Vec<usize> {
    let mut counts = vec![self.coordinates.len().max(self.cells[0].len())];
    if !self.cells[1].is_empty() {
      counts.push(self.cells[1].len());
    } else if !self.cells[2].is_empty() {
      let mut edges: Vec<[usize; 2]> = Vec::new();
      for face in &self.cells[2] {
        for i in 0..face.len() {
          let a = face[i];
          let b = face[(i + 1) % face.len()];
          let key = if a <= b { [a, b] } else { [b, a] };
          if !edges.contains(&key) {
            edges.push(key);
          }
        }
      }
      counts.push(edges.len());
    }
    if !self.cells[2].is_empty() {
      counts.push(self.cells[2].len());
    }
    counts
  }
}

/// Sum the measures the polygons of a mesh contribute, through the ordinary
/// region machinery so exact coordinates stay exact.
fn sum_over_polygons(polygons: &[Expr], head: &str) -> Option<Expr> {
  let mut terms = Vec::with_capacity(polygons.len());
  for polygon in polygons {
    let value = crate::evaluator::evaluate_function_call_ast(
      head,
      std::slice::from_ref(polygon),
    )
    .ok()?;
    terms.push(value);
  }
  crate::evaluator::evaluate_function_call_ast("Plus", &terms).ok()
}

/// `RegionMeasure` / `Area` of a mesh region.
pub fn mesh_measure(mesh: &Mesh) -> Option<Expr> {
  let polygons = mesh.polygons();
  if polygons.is_empty() {
    return None;
  }
  sum_over_polygons(&polygons, "Area")
}

/// `Perimeter` of a mesh region: the length of the edges that bound it.
pub fn mesh_perimeter(mesh: &Mesh) -> Option<Expr> {
  let edges = mesh.boundary_edges();
  if edges.is_empty() {
    return None;
  }
  let mut lengths = Vec::with_capacity(edges.len());
  for [a, b] in edges {
    let pts = mesh.points(&[a, b])?;
    let line = Expr::FunctionCall {
      name: "Line".to_string(),
      args: vec![Expr::List(pts.into())].into(),
    };
    lengths.push(
      crate::evaluator::evaluate_function_call_ast(
        "ArcLength",
        std::slice::from_ref(&line),
      )
      .ok()?,
    );
  }
  crate::evaluator::evaluate_function_call_ast("Plus", &lengths).ok()
}

/// `RegionCentroid` of a mesh region: its cells' centroids weighted by area.
pub fn mesh_centroid(mesh: &Mesh) -> Option<Expr> {
  let polygons = mesh.polygons();
  if polygons.is_empty() {
    return None;
  }
  if polygons.len() == 1 {
    return crate::evaluator::evaluate_function_call_ast(
      "RegionCentroid",
      std::slice::from_ref(&polygons[0]),
    )
    .ok();
  }
  let mut weighted: Vec<Expr> = Vec::new();
  let mut areas: Vec<Expr> = Vec::new();
  for polygon in &polygons {
    let area = crate::evaluator::evaluate_function_call_ast(
      "Area",
      std::slice::from_ref(polygon),
    )
    .ok()?;
    let centroid = crate::evaluator::evaluate_function_call_ast(
      "RegionCentroid",
      std::slice::from_ref(polygon),
    )
    .ok()?;
    weighted.push(
      crate::evaluator::evaluate_function_call_ast(
        "Times",
        &[area.clone(), centroid],
      )
      .ok()?,
    );
    areas.push(area);
  }
  let total =
    crate::evaluator::evaluate_function_call_ast("Plus", &areas).ok()?;
  let sum =
    crate::evaluator::evaluate_function_call_ast("Plus", &weighted).ok()?;
  crate::evaluator::evaluate_function_call_ast("Divide", &[sum, total]).ok()
}

/// `RegionBounds` of a mesh region: the range each coordinate covers.
pub fn mesh_bounds(mesh: &Mesh) -> Option<Expr> {
  let first = mesh.coordinates.first()?;
  let Expr::List(pt) = first else {
    return None;
  };
  let dimensions = pt.len();
  let mut ranges = Vec::with_capacity(dimensions);
  for axis in 0..dimensions {
    let mut values = Vec::with_capacity(mesh.coordinates.len());
    for point in &mesh.coordinates {
      let Expr::List(coords) = point else {
        return None;
      };
      values.push(coords.get(axis)?.clone());
    }
    let min =
      crate::evaluator::evaluate_function_call_ast("Min", &values).ok()?;
    let max =
      crate::evaluator::evaluate_function_call_ast("Max", &values).ok()?;
    ranges.push(Expr::List(vec![min, max].into()));
  }
  Some(Expr::List(ranges.into()))
}

/// `RegionMember` for a mesh region: inside when any of its cells holds the
/// point.
pub fn mesh_member(mesh: &Mesh, point: &Expr) -> Option<Expr> {
  let polygons = mesh.polygons();
  if polygons.is_empty() {
    return None;
  }
  for polygon in polygons {
    let inside = crate::evaluator::evaluate_function_call_ast(
      "RegionMember",
      &[polygon, point.clone()],
    )
    .ok()?;
    if matches!(&inside, Expr::Identifier(s) if s == "True") {
      return Some(crate::syntax::bool_expr(true));
    }
  }
  Some(crate::syntax::bool_expr(false))
}

/// `MeshCoordinates`, `MeshCells`, `MeshPrimitives` and `MeshCellCount`.
pub fn mesh_accessor_ast(
  name: &str,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let original = || unevaluated(name, args);
  if args.is_empty() {
    return Ok(original());
  }
  let Some(mesh) = parse_mesh(&args[0]) else {
    return Ok(original());
  };
  let dimension = |arg: Option<&Expr>| -> Option<usize> {
    match arg {
      Some(Expr::Integer(d)) if *d >= 0 => Some(*d as usize),
      _ => None,
    }
  };
  match name {
    "MeshCoordinates" if args.len() == 1 => {
      Ok(Expr::List(mesh.coordinates.clone().into()))
    }
    "MeshCells" if args.len() == 2 => {
      match dimension(args.get(1)).and_then(|d| mesh.cell_expressions(d)) {
        Some(cells) => Ok(Expr::List(cells.into())),
        None => Ok(original()),
      }
    }
    "MeshPrimitives" if args.len() == 2 => {
      match dimension(args.get(1)).and_then(|d| mesh.cell_primitives(d)) {
        Some(cells) => Ok(Expr::List(cells.into())),
        None => Ok(original()),
      }
    }
    "MeshCellCount" if args.len() == 1 => Ok(Expr::List(
      mesh
        .cell_counts()
        .into_iter()
        .map(|n| Expr::Integer(n as i128))
        .collect(),
    )),
    "MeshCellCount" if args.len() == 2 => match dimension(args.get(1)) {
      Some(d) => match mesh.cell_counts().get(d) {
        Some(n) => Ok(Expr::Integer(*n as i128)),
        None => Ok(Expr::Integer(0)),
      },
      None => Ok(original()),
    },
    _ => Ok(original()),
  }
}
