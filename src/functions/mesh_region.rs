//! Mesh regions: the `MeshRegion[coords, cells]` and
//! `BoundaryMeshRegion[coords, cells]` objects `ConvexHullMesh`,
//! `DelaunayMesh` and `VoronoiMesh` build, and the accessors and measures
//! that read them.
//!
//! The measures delegate to the ordinary `Polygon` machinery — one polygon
//! per 2-cell, or the boundary loop of a boundary mesh — so exact
//! coordinates keep exact answers.

use super::*;

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
  for item in items {
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
  // A single cell primitive may stand without the enclosing list.
  // `BoundaryMeshRegion[coords, {Line[…]}, {Line[…]}, opts]` lists each
  // boundary component separately; the cell lists run until the options.
  let mut cell_specs: Vec<Expr> = Vec::new();
  for (k, arg) in args[1..].iter().enumerate() {
    match arg {
      Expr::List(specs) => cell_specs.extend(specs.iter().cloned()),
      Expr::FunctionCall { .. } => cell_specs.push(arg.clone()),
      _ if k == 0 => return None,
      _ => break,
    }
  }
  let cell_specs: &[Expr] = &cell_specs;
  let mut cells: Vec<Vec<Vec<usize>>> =
    vec![Vec::new(), Vec::new(), Vec::new()];
  for spec in cell_specs {
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
      "Polygon" | "Triangle" => 2,
      _ => return None,
    };
    if spec_args.len() != 1 {
      return None;
    }
    // Either one index list, or a list of them.
    if let Some(single) = index_list(&spec_args[0]) {
      if dimension == 1 && single.len() > 2 {
        // A polyline `Line[{1, 2, 3}]` stands for its consecutive edges.
        cells[1].extend(single.windows(2).map(<[usize]>::to_vec));
      } else {
        cells[dimension].push(single);
      }
    } else if let Expr::List(groups) = &spec_args[0] {
      for group in groups {
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
        .map(|pts| call1("Polygon", Expr::List(pts.into())))
        .collect();
    }
    if self.boundary
      && let Some(loop_indices) = self.boundary_loop()
      && let Some(pts) = self.points(&loop_indices)
    {
      return vec![call1("Polygon", Expr::List(pts.into()))];
    }
    // A boundary mesh of several loops (holes, separate pieces) is filled
    // by triangles.
    if self.boundary
      && let Some(triangles) = self.triangle_indices()
    {
      return triangles
        .iter()
        .filter_map(|t| self.points(t))
        .map(|pts| call1("Polygon", Expr::List(pts.into())))
        .collect();
    }
    Vec::new()
  }

  /// The boundary loop of a boundary mesh with a single component.
  fn boundary_loop(&self) -> Option<Vec<usize>> {
    let mut loops = self.boundary_loops()?;
    (loops.len() == 1).then(|| loops.remove(0))
  }

  /// Walk the boundary lines into closed loops of vertex indices.
  fn boundary_loops(&self) -> Option<Vec<Vec<usize>>> {
    let edges = &self.cells[1];
    if edges.is_empty() || edges.iter().any(|e| e.len() != 2) {
      return None;
    }
    let mut remaining: Vec<[usize; 2]> =
      edges.iter().map(|e| [e[0], e[1]]).collect();
    let mut loops = Vec::new();
    while !remaining.is_empty() {
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
      loops.push(loop_indices);
    }
    Some(loops)
  }

  /// The boundary of a planar region as rings of points: the loops of a
  /// boundary mesh, or the sides that belong to a single 2-cell, each
  /// running the way its cell is wound counter-clockwise.
  pub fn planar_rings(&self) -> Option<Vec<Vec<(f64, f64)>>> {
    let pts = self.planar_points()?;
    let at = |i: usize| pts.get(i.checked_sub(1)?).copied();
    if self.cells[2].is_empty() {
      if !self.boundary {
        return None;
      }
      return self
        .boundary_loops()?
        .iter()
        .map(|l| l.iter().map(|&i| at(i)).collect::<Option<Vec<_>>>())
        .collect();
    }
    let mut directed: Vec<(usize, usize)> = Vec::new();
    let mut count: std::collections::HashMap<(usize, usize), usize> =
      std::collections::HashMap::new();
    for face in &self.cells[2] {
      let ring: Vec<(f64, f64)> =
        face.iter().map(|&i| at(i)).collect::<Option<_>>()?;
      let ccw = crate::functions::polygon_boolean::signed_area(&ring) >= 0.0;
      let n = face.len();
      for k in 0..n {
        let (a, b) = (face[k], face[(k + 1) % n]);
        let (a, b) = if ccw { (a, b) } else { (b, a) };
        directed.push((a, b));
        *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
      }
    }
    let boundary: Vec<(usize, usize)> = directed
      .into_iter()
      .filter(|&(a, b)| count[&(a.min(b), a.max(b))] == 1)
      .collect();
    let mut outgoing: std::collections::HashMap<usize, Vec<usize>> =
      std::collections::HashMap::new();
    for (k, &(a, _)) in boundary.iter().enumerate() {
      outgoing.entry(a).or_default().push(k);
    }
    let mut used = vec![false; boundary.len()];
    let mut rings = Vec::new();
    for start in 0..boundary.len() {
      if used[start] {
        continue;
      }
      used[start] = true;
      let first = boundary[start].0;
      let mut ring = vec![at(first)?];
      let mut current = boundary[start].1;
      while current != first {
        ring.push(at(current)?);
        let next = outgoing
          .get(&current)?
          .iter()
          .copied()
          .find(|&k| !used[k])?;
        used[next] = true;
        current = boundary[next].1;
      }
      rings.push(ring);
    }
    Some(rings)
  }

  /// The distinct sides of the 2-cells, in the order they first appear.
  fn face_edges(&self) -> Vec<[usize; 2]> {
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();
    for face in &self.cells[2] {
      for i in 0..face.len() {
        let (a, b) = (face[i], face[(i + 1) % face.len()]);
        if seen.insert(if a <= b { (a, b) } else { (b, a) }) {
          edges.push([a, b]);
        }
      }
    }
    edges
  }

  /// The cells of dimension `d` that lie on the boundary ("Boundary") or
  /// do not touch it at all ("Interior"), as index expressions. A cell
  /// touches the boundary through any of its vertices.
  pub fn cells_by_location(
    &self,
    d: usize,
    interior: bool,
  ) -> Option<Vec<Expr>> {
    let boundary_edges = self.boundary_edges();
    let on_boundary: std::collections::HashSet<usize> =
      boundary_edges.iter().flatten().copied().collect();
    let edge_set: std::collections::HashSet<(usize, usize)> = boundary_edges
      .iter()
      .map(|&[a, b]| if a <= b { (a, b) } else { (b, a) })
      .collect();
    let cells = self.cell_expressions(d)?;
    let indices_of = |cell: &Expr| -> Option<Vec<usize>> {
      let Expr::FunctionCall { args, .. } = cell else {
        return None;
      };
      match &args[0] {
        Expr::Integer(i) => Some(vec![*i as usize]),
        other => index_list(other),
      }
    };
    let mut out = Vec::new();
    for cell in cells {
      let idx = indices_of(&cell)?;
      let keep = if interior {
        idx.iter().all(|i| !on_boundary.contains(i))
      } else {
        match d {
          0 => on_boundary.contains(&idx[0]),
          1 => {
            let (a, b) = (idx[0], idx[1]);
            edge_set.contains(&if a <= b { (a, b) } else { (b, a) })
          }
          // The boundary of a region of faces has no 2-cells.
          _ => false,
        }
      };
      if keep {
        out.push(cell);
      }
    }
    Some(out)
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
          .map(|i| call1("Point", Expr::Integer(i as i128)))
          .collect(),
      );
    }
    // A mesh of faces has the faces' sides as its 1-cells, each once.
    if d == 1 && self.cells[1].is_empty() && !self.cells[2].is_empty() {
      return Some(
        self
          .face_edges()
          .into_iter()
          .map(|[a, b]| {
            call1(
              "Line",
              Expr::List(
                vec![Expr::Integer(a as i128), Expr::Integer(b as i128)].into(),
              ),
            )
          })
          .collect(),
      );
    }
    // A boundary mesh fills its interior: the loop is its single 2-cell.
    if d == 2 && self.cells[2].is_empty() && self.boundary {
      let loop_indices = self.boundary_loop()?;
      return Some(vec![call1(
        "Polygon",
        Expr::List(
          loop_indices
            .into_iter()
            .map(|i| Expr::Integer(i as i128))
            .collect(),
        ),
      )]);
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

/// Evaluate `head[args…]`.
fn eval_call(head: &str, args: &[Expr]) -> Option<Expr> {
  crate::evaluator::evaluate_function_call_ast(head, args).ok()
}

/// The tetrahedra a solid's boundary fans into from the origin, as
/// `(signed volume × 6, apex-free centroid × 4)` pairs: the determinant
/// `a . (b × c)` of each fan triangle `(a, b, c)` of an outward-wound face,
/// and `a + b + c`. Summed, the signed volumes give the solid's volume
/// whatever the origin, since the faces close up.
fn solid_fan(mesh: &Mesh) -> Option<Vec<(Expr, Expr)>> {
  let mut out = Vec::new();
  for face in &mesh.cells[2] {
    let pts = mesh.points(face)?;
    for i in 1..pts.len().saturating_sub(1) {
      let (a, b, c) = (&pts[0], &pts[i], &pts[i + 1]);
      let det = eval_call(
        "Dot",
        &[a.clone(), eval_call("Cross", &[b.clone(), c.clone()])?],
      )?;
      let sum = eval_call("Plus", &[a.clone(), b.clone(), c.clone()])?;
      out.push((det, sum));
    }
  }
  (!out.is_empty()).then_some(out)
}

/// Whether the mesh is a solid: a boundary mesh of 3-D polygons.
fn is_solid(mesh: &Mesh) -> bool {
  mesh.boundary && mesh.dimension() == 3
}

/// `Volume` of a solid mesh region, exact for exact coordinates.
fn mesh_volume(mesh: &Mesh) -> Option<Expr> {
  let dets: Vec<Expr> = solid_fan(mesh)?.into_iter().map(|(d, _)| d).collect();
  let total = eval_call("Plus", &dets)?;
  eval_call("Abs", &[eval_call("Divide", &[total, Expr::Integer(6)])?])
}

/// `SurfaceArea` of a solid mesh region: its faces' total area, which
/// wolframscript reports as a machine number even for exact coordinates.
pub fn mesh_surface_area(mesh: &Mesh) -> Option<Expr> {
  if !is_solid(mesh) {
    return Some(id_expr("Undefined"));
  }
  let total = sum_over_polygons(&mesh.polygons(), "Area")?;
  eval_call("N", &[total])
}

/// The measure `head` (`ArcLength`, `Area`, `Volume`) of a mesh region:
/// defined only when it is the measure of the region's own dimension
/// (`Area` of a solid is `Undefined`).
pub fn mesh_named_measure(mesh: &Mesh, head: &str) -> Option<Expr> {
  let wanted = match head {
    "ArcLength" => 1,
    "Area" => 2,
    "Volume" => 3,
    _ => return None,
  };
  let dimension = mesh.dimension();
  if dimension != wanted {
    // Only the solid and planar cases are known to be well formed here.
    return (dimension >= 2 && wanted >= 2).then(|| id_expr("Undefined"));
  }
  mesh_measure(mesh)
}

/// `RegionMeasure` of a mesh region: the volume of a solid, the area of a
/// planar region.
pub fn mesh_measure(mesh: &Mesh) -> Option<Expr> {
  if is_solid(mesh) {
    return mesh_volume(mesh);
  }
  let polygons = mesh.polygons();
  if polygons.is_empty() {
    return None;
  }
  sum_over_polygons(&polygons, "Area")
}

/// `Perimeter` of a mesh region: the length of the edges that bound it
/// (`Undefined` for a solid).
pub fn mesh_perimeter(mesh: &Mesh) -> Option<Expr> {
  if is_solid(mesh) {
    return Some(id_expr("Undefined"));
  }
  let edges = mesh.boundary_edges();
  if edges.is_empty() {
    return None;
  }
  let mut lengths = Vec::with_capacity(edges.len());
  for [a, b] in edges {
    let pts = mesh.points(&[a, b])?;
    let line = call1("Line", Expr::List(pts.into()));
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

/// `RegionCentroid` of a mesh region: its cells' centroids weighted by area,
/// or for a solid its fan tetrahedra's centroids weighted by volume.
pub fn mesh_centroid(mesh: &Mesh) -> Option<Expr> {
  if is_solid(mesh) {
    // Each fan tetrahedron (origin, a, b, c) has centroid (a + b + c)/4.
    let fan = solid_fan(mesh)?;
    let mut weighted = Vec::with_capacity(fan.len());
    let mut dets = Vec::with_capacity(fan.len());
    for (det, sum) in fan {
      weighted.push(eval_call("Times", &[det.clone(), sum])?);
      dets.push(det);
    }
    let total = eval_call("Plus", &dets)?;
    let moment = eval_call("Plus", &weighted)?;
    return eval_call(
      "Divide",
      &[moment, eval_call("Times", &[Expr::Integer(4), total])?],
    );
  }
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
  // A numeric point in a mesh of triangles needs no evaluator round trip
  // per cell.
  if let Expr::List(xy) = point
    && let [x, y] = xy.as_slice()
    && let (Some(x), Some(y)) = (
      crate::functions::math_ast::try_eval_to_f64(x),
      crate::functions::math_ast::try_eval_to_f64(y),
    )
    && !mesh.boundary
    && !mesh.cells[2].is_empty()
    && mesh.cells[2].iter().all(|c| c.len() == 3)
    && let Some(triangles) = mesh.cells[2]
      .iter()
      .map(|c| {
        let p = |i: usize| match mesh.coordinates.get(c[i] - 1)? {
          Expr::List(v) if v.len() == 2 => Some((
            crate::functions::math_ast::try_eval_to_f64(&v[0])?,
            crate::functions::math_ast::try_eval_to_f64(&v[1])?,
          )),
          _ => None,
        };
        Some([p(0)?, p(1)?, p(2)?])
      })
      .collect::<Option<Vec<_>>>()
  {
    const SLACK: f64 = 1e-10;
    let found = triangles.iter().any(|[a, b, c]| {
      let det = (b.1 - c.1) * (a.0 - c.0) + (c.0 - b.0) * (a.1 - c.1);
      if det == 0.0 {
        return false;
      }
      let l1 = ((b.1 - c.1) * (x - c.0) + (c.0 - b.0) * (y - c.1)) / det;
      let l2 = ((c.1 - a.1) * (x - c.0) + (a.0 - c.0) * (y - c.1)) / det;
      l1 >= -SLACK && l2 >= -SLACK && 1.0 - l1 - l2 >= -SLACK
    });
    return Some(bool_expr(found));
  }
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
      return Some(bool_expr(true));
    }
  }
  Some(bool_expr(false))
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
    "MeshCells" | "MeshPrimitives"
      if args.len() == 2
        && let Expr::List(spec) = &args[1]
        && let [d, Expr::String(location)] = spec.as_slice()
        && (location == "Interior" || location == "Boundary") =>
    {
      let Some(d) = dimension(Some(d)) else {
        return Ok(original());
      };
      let Some(cells) = mesh.cells_by_location(d, location == "Interior")
      else {
        return Ok(original());
      };
      if name == "MeshCells" {
        return Ok(Expr::List(cells.into()));
      }
      let mut out = Vec::with_capacity(cells.len());
      for cell in cells {
        let Expr::FunctionCall {
          name: head,
          args: cargs,
        } = &cell
        else {
          return Ok(original());
        };
        let indices = match &cargs[0] {
          Expr::Integer(i) => vec![*i as usize],
          other => match index_list(other) {
            Some(i) => i,
            None => return Ok(original()),
          },
        };
        let Some(pts) = mesh.points(&indices) else {
          return Ok(original());
        };
        out.push(call1(
          head,
          if d == 0 {
            pts[0].clone()
          } else {
            Expr::List(pts.into())
          },
        ));
      }
      Ok(Expr::List(out.into()))
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
    "ConnectedMeshComponents" if args.len() == 1 => {
      Ok(match connected_components(&mesh) {
        Some(parts) => Expr::List(parts.into()),
        None => original(),
      })
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

/// `ConnectedMeshComponents[mesh]`: the pieces of a mesh whose top-
/// dimensional cells are joined through shared vertices, each a mesh of its
/// own kind with the coordinates it uses, in order of first use.
fn connected_components(mesh: &Mesh) -> Option<Vec<Expr>> {
  let top = (0..3).rev().find(|&d| !mesh.cells[d].is_empty())?;
  let cells = &mesh.cells[top];
  let n = mesh.coordinates.len();
  let mut parent: Vec<usize> = (0..=n).collect();
  fn find(parent: &mut [usize], i: usize) -> usize {
    let mut r = i;
    while parent[r] != r {
      r = parent[r];
    }
    let mut i = i;
    while parent[i] != r {
      let next = parent[i];
      parent[i] = r;
      i = next;
    }
    r
  }
  for cell in cells {
    if cell.iter().any(|&i| i == 0 || i > n) {
      return None;
    }
    for w in cell.windows(2) {
      let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
      parent[a] = b;
    }
  }
  // Components in the order their first cell appears.
  let mut order: Vec<usize> = Vec::new();
  let mut members: std::collections::HashMap<usize, Vec<&Vec<usize>>> =
    std::collections::HashMap::new();
  for cell in cells {
    let root = find(&mut parent, cell[0]);
    if !members.contains_key(&root) {
      order.push(root);
    }
    members.entry(root).or_default().push(cell);
  }
  let (head, cell_head) = match (mesh.boundary, top) {
    (true, _) => ("BoundaryMeshRegion", "Line"),
    (false, 0) => ("MeshRegion", "Point"),
    (false, 1) => ("MeshRegion", "Line"),
    (false, _) => ("MeshRegion", "Polygon"),
  };
  // Machine coordinates, as Wolfram returns them.
  let numeric = crate::evaluator::evaluate_expr_to_expr(&call1(
    "N",
    Expr::List(mesh.coordinates.clone().into()),
  ))
  .ok()?;
  let Expr::List(numeric) = &numeric else {
    return None;
  };
  let mut out = Vec::new();
  for root in order {
    let mut renumber: std::collections::HashMap<usize, usize> =
      std::collections::HashMap::new();
    let mut coords = Vec::new();
    let mut new_cells = Vec::new();
    for cell in &members[&root] {
      let ids: Vec<Expr> = cell
        .iter()
        .map(|&i| {
          let next = renumber.len() + 1;
          let id = *renumber.entry(i).or_insert_with(|| {
            coords.push(numeric[i - 1].clone());
            next
          });
          Expr::Integer(id as i128)
        })
        .collect();
      new_cells.push(if top == 0 {
        ids.into_iter().next()?
      } else {
        Expr::List(ids.into())
      });
    }
    out.push(call(
      head,
      vec![
        Expr::List(coords.into()),
        Expr::List(vec![call1(cell_head, Expr::List(new_cells.into()))].into()),
      ],
    ));
  }
  Some(out)
}

/// Twice the signed area of the triangle `a b c` (positive when it winds
/// counter-clockwise).
fn cross(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
  (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Ear-clip a simple polygon, given as indices into `pts` in either winding,
/// into counter-clockwise triangles. Vertices that sit on a straight edge
/// are dropped rather than emitted as zero-area triangles.
fn ear_clip(pts: &[(f64, f64)], indices: &[usize]) -> Option<Vec<[usize; 3]>> {
  let mut ring: Vec<usize> = indices.to_vec();
  let at = |i: usize| pts.get(i - 1).copied();
  let mut area2 = 0.0;
  for k in 0..ring.len() {
    let a = at(ring[k])?;
    let b = at(ring[(k + 1) % ring.len()])?;
    area2 += a.0 * b.1 - b.0 * a.1;
  }
  if area2 < 0.0 {
    ring.reverse();
  }
  let mut triangles = Vec::new();
  while ring.len() > 3 {
    let n = ring.len();
    let mut clipped = false;
    let mut straight = None;
    for k in 0..n {
      let (ia, ib, ic) = (ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]);
      let (a, b, c) = (at(ia)?, at(ib)?, at(ic)?);
      let turn = cross(a, b, c);
      // A vertex on a straight stretch is no ear; it is clipped as the
      // corner of a neighboring ear instead, so it stays a vertex of the
      // triangulation (dropping it would leave it in the middle of a
      // triangle's edge — a hole's corner, say).
      if turn == 0.0 {
        straight.get_or_insert(k);
        continue;
      }
      if turn < 0.0 {
        continue;
      }
      let blocked = ring.iter().any(|&other| {
        if other == ia || other == ib || other == ic {
          return false;
        }
        let Some(p) = at(other) else { return false };
        if p == a || p == b || p == c {
          return false;
        }
        cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
      });
      if !blocked {
        triangles.push([ia, ib, ic]);
        ring.remove(k);
        clipped = true;
        break;
      }
    }
    if !clipped {
      // Only straight vertices left to take: drop one.
      ring.remove(straight?);
    }
  }
  if ring.len() == 3 && cross(at(ring[0])?, at(ring[1])?, at(ring[2])?) != 0.0 {
    triangles.push([ring[0], ring[1], ring[2]]);
  }
  Some(triangles)
}

impl Mesh {
  fn planar_points(&self) -> Option<Vec<(f64, f64)>> {
    self
      .coordinates
      .iter()
      .map(|c| match c {
        Expr::List(xy) if xy.len() == 2 => Some((
          crate::functions::math_ast::try_eval_to_f64(&xy[0])?,
          crate::functions::math_ast::try_eval_to_f64(&xy[1])?,
        )),
        _ => None,
      })
      .collect()
  }

  /// Triangles (1-based vertex indices) that tile a planar region: each
  /// 2-cell, or the loop a boundary mesh of lines encloses, ear-clipped.
  fn triangle_indices(&self) -> Option<Vec<[usize; 3]>> {
    let pts = self.planar_points()?;
    let loops: Vec<Vec<usize>> = if !self.cells[2].is_empty() {
      self.cells[2].clone()
    } else if self.boundary {
      let loops = self.boundary_loops()?;
      if loops.len() == 1 {
        loops
      } else {
        filled_loops(&pts, &loops)
      }
    } else {
      return None;
    };
    let mut triangles = Vec::new();
    for ring in loops {
      if ring.len() < 3 {
        return None;
      }
      triangles.extend(ear_clip(&pts, &ring)?);
    }
    Some(triangles)
  }
}

/// Several boundary loops (1-based index rings into `pts`) as simple
/// polygons: each outer loop with the holes nested directly inside it
/// bridged into one ring.
fn filled_loops(pts: &[(f64, f64)], loops: &[Vec<usize>]) -> Vec<Vec<usize>> {
  use crate::functions::polygon_boolean::{inside, signed_area};
  let ring = |l: &Vec<usize>| -> Vec<(f64, f64)> {
    l.iter().map(|&i| pts[i - 1]).collect()
  };
  // Nesting depth decides: even depth bounds the region, odd is a hole.
  let depth = |k: usize| {
    let probe = ring(&loops[k])[0];
    (0..loops.len())
      .filter(|&j| j != k && inside(&[ring(&loops[j])], probe))
      .count()
  };
  let depths: Vec<usize> = (0..loops.len()).map(depth).collect();
  let zero_based: Vec<Vec<usize>> = loops
    .iter()
    .map(|l| l.iter().map(|&i| i - 1).collect())
    .collect();
  let mut out = Vec::new();
  for (k, outer) in zero_based.iter().enumerate() {
    if depths[k] % 2 == 1 {
      continue;
    }
    let outer_ring = ring(&loops[k]);
    let mut outer = outer.clone();
    if signed_area(&outer_ring) < 0.0 {
      outer.reverse();
    }
    let holes: Vec<Vec<usize>> = (0..loops.len())
      .filter(|&j| {
        depths[j] == depths[k] + 1
          && inside(std::slice::from_ref(&outer_ring), ring(&loops[j])[0])
      })
      .map(|j| {
        let mut h = zero_based[j].clone();
        if signed_area(&ring(&loops[j])) > 0.0 {
          h.reverse();
        }
        h
      })
      .collect();
    let bridged =
      crate::functions::polygon_boolean::bridge_holes(pts, &outer, &holes);
    out.push(bridged.into_iter().map(|i| i + 1).collect());
  }
  out
}

/// The `MeshRegion` (or, with `boundary`, `BoundaryMeshRegion`) bounded by
/// `rings`, which have the region on their left. A boundary mesh lists each
/// ring as its own counter-clockwise loop of lines; a mesh of polygons has
/// one polygon per ring, or triangles when the region has holes.
pub fn mesh_from_rings_expr(rings: &[Vec<(f64, f64)>], boundary: bool) -> Expr {
  let mut coords: Vec<(f64, f64)> = Vec::new();
  let mut index: std::collections::HashMap<(u64, u64), usize> =
    std::collections::HashMap::new();
  let loops: Vec<Vec<usize>> = rings
    .iter()
    .map(|r| {
      r.iter()
        .map(|&p| {
          *index
            .entry(((p.0 + 0.0).to_bits(), (p.1 + 0.0).to_bits()))
            .or_insert_with(|| {
              coords.push(p);
              coords.len()
            })
        })
        .collect()
    })
    .collect();
  let coordinate_list = Expr::List(
    coords
      .iter()
      .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)].into()))
      .collect::<Vec<_>>()
      .into(),
  );
  let int_list = |l: &[usize]| {
    Expr::List(l.iter().map(|&i| Expr::Integer(i as i128)).collect())
  };
  if boundary {
    let mut args = vec![coordinate_list];
    for (ring, l) in rings.iter().zip(&loops) {
      let mut l = l.clone();
      if crate::functions::polygon_boolean::signed_area(ring) < 0.0 {
        l.reverse();
      }
      let n = l.len();
      let segments: Vec<Expr> =
        (0..n).map(|k| int_list(&[l[k], l[(k + 1) % n]])).collect();
      args.push(Expr::List(
        vec![call1("Line", Expr::List(segments.into()))].into(),
      ));
    }
    return call("BoundaryMeshRegion", args);
  }
  let has_holes = rings
    .iter()
    .any(|r| crate::functions::polygon_boolean::signed_area(r) < 0.0);
  let cells: Vec<Expr> = if has_holes {
    filled_loops(&coords, &loops)
      .iter()
      .filter_map(|l| ear_clip(&coords, l))
      .flatten()
      .map(|t| int_list(&t))
      .collect()
  } else {
    loops.iter().map(|l| int_list(l)).collect()
  };
  call(
    "MeshRegion",
    vec![
      coordinate_list,
      Expr::List(vec![call1("Polygon", Expr::List(cells.into()))].into()),
    ],
  )
}

/// `RegionIntersection`, `RegionUnion`, `RegionDifference` and
/// `RegionSymmetricDifference` of two planar mesh regions, as a mesh region
/// (a boundary mesh when either operand is one), or `EmptyRegion[2]`.
pub fn mesh_boolean(name: &str, a: &Expr, b: &Expr) -> Option<Expr> {
  use crate::functions::polygon_boolean::{Op, boolean};
  let (ma, mb) = (parse_mesh(a)?, parse_mesh(b)?);
  if ma.dimension() != 2 || mb.dimension() != 2 {
    return None;
  }
  let (ra, rb) = (ma.planar_rings()?, mb.planar_rings()?);
  let rings = match name {
    "RegionIntersection" => boolean(&ra, &rb, Op::Intersection),
    "RegionUnion" => boolean(&ra, &rb, Op::Union),
    "RegionDifference" => boolean(&ra, &rb, Op::Difference),
    "RegionSymmetricDifference" => {
      let mut rings = boolean(&ra, &rb, Op::Difference);
      rings.extend(boolean(&rb, &ra, Op::Difference));
      rings
    }
    _ => return None,
  };
  if rings.is_empty() {
    return Some(call1("EmptyRegion", Expr::Integer(2)));
  }
  Some(mesh_from_rings_expr(&rings, ma.boundary || mb.boundary))
}

/// The value of a `MaxCellMeasure -> spec` option among `opts`.
fn max_cell_measure(opts: &[Expr]) -> Option<&Expr> {
  opts.iter().find_map(|o| match o {
    Expr::Rule {
      pattern,
      replacement,
    } if matches!(pattern.as_ref(), Expr::Identifier(n) if n == "MaxCellMeasure") => {
      Some(replacement.as_ref())
    }
    _ => None,
  })
}

/// A planar region re-meshed by triangles that meet a `MaxCellMeasure`
/// specification.
fn refined_mesh(mesh: &Mesh, spec: &Expr) -> Option<Expr> {
  let h = crate::functions::fem2d::edge_length_for_max_cell_measure(spec)?;
  let rings = crate::functions::polygon_boolean::orient(&mesh.planar_rings()?);
  let tri = crate::functions::fem2d::mesh_from_rings_h(&rings, h)?;
  Some(crate::functions::fem2d::mesh_expr(&tri))
}

/// `TriangulateMesh[mesh]`: the planar region as a `MeshRegion` of triangles.
pub fn triangulate_mesh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(mesh) = parse_mesh(&args[0]) else {
    return Ok(unevaluated("TriangulateMesh", args));
  };
  if let Some(spec) = max_cell_measure(&args[1..]) {
    return Ok(
      refined_mesh(&mesh, spec)
        .unwrap_or_else(|| unevaluated("TriangulateMesh", args)),
    );
  }
  let Some(triangles) = mesh.triangle_indices() else {
    return Ok(unevaluated("TriangulateMesh", args));
  };
  let cells: Vec<Expr> = triangles
    .iter()
    .map(|t| Expr::List(t.iter().map(|&i| Expr::Integer(i as i128)).collect()))
    .collect();
  Ok(call(
    "MeshRegion",
    vec![
      Expr::List(mesh.coordinates.clone().into()),
      call1("Triangle", Expr::List(cells.into())),
    ],
  ))
}

/// Gauss–Legendre nodes and weights on `[0, 1]`.
pub(crate) fn gauss_legendre_unit(n: usize) -> Vec<(f64, f64)> {
  (0..n)
    .map(|i| {
      let mut x =
        (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
      let mut dp = 1.0;
      for _ in 0..100 {
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
          let kf = k as f64;
          let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
          p0 = p1;
          p1 = p2;
        }
        dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        let dx = p1 / dp;
        x -= dx;
        if dx.abs() < 1e-15 {
          break;
        }
      }
      (f64::midpoint(x, 1.0), 1.0 / ((1.0 - x * x) * dp * dp))
    })
    .collect()
}

/// `NIntegrate[f, Element[{x, y}, mesh]]`: Gauss–Legendre quadrature over
/// the triangles tiling a planar mesh region. `None` when the region is not a
/// planar mesh or the integrand does not evaluate to a number.
pub fn nintegrate_over_mesh(
  integrand: &Expr,
  vars: &[String],
  region: &Expr,
) -> Option<Expr> {
  let mesh = parse_mesh(region)?;
  if vars.len() != 2 {
    return None;
  }
  let pts = mesh.planar_points()?;
  let nodes = gauss_legendre_unit(8);
  let mut total = 0.0;
  for [ia, ib, ic] in mesh.triangle_indices()? {
    let (a, b, c) = (pts[ia - 1], pts[ib - 1], pts[ic - 1]);
    let area2 = cross(a, b, c).abs();
    for &(s, ws) in &nodes {
      for &(t, wt) in &nodes {
        let x = a.0 + s * ((b.0 - a.0) + t * (c.0 - b.0));
        let y = a.1 + s * ((b.1 - a.1) + t * (c.1 - b.1));
        let value = crate::syntax::substitute_variable(
          &crate::syntax::substitute_variable(
            integrand,
            &vars[0],
            &Expr::Real(x),
          ),
          &vars[1],
          &Expr::Real(y),
        );
        let value = crate::evaluator::evaluate_expr_to_expr(&value).ok()?;
        let f = crate::functions::math_ast::try_eval_to_f64(&value)?;
        total += area2 * ws * wt * s * f;
      }
    }
  }
  // Quadrature round-off (`6.000000000000004` for an exact `6`) is below
  // the precision the integral is good to; keep twelve significant digits.
  if total != 0.0 && total.is_finite() {
    let scale = 10f64.powi(11 - total.abs().log10().floor() as i32);
    let rounded = (total * scale).round() / scale;
    if (rounded - total).abs() <= 1e-12 * total.abs() {
      total = rounded;
    }
  }
  Some(Expr::Real(total))
}

/// `DiscretizeGraphics[Graphics[prims]]`: the points, lines and polygons
/// of a graphic as one `MeshRegion` — the coordinates (as machine reals,
/// shared corners merged) and one cell primitive per kind listing every
/// cell of that kind. `None` for a graphic with other primitives.
/// `RegionBoundary[region]` for the primitive regions: a filled polygon
/// (`Polygon`, `Rectangle`, `Triangle`, plane only) is bounded by its
/// closed outline `Line`, a `Disk`/`Ball` by its `Circle`/`Sphere`, a
/// `Cuboid` by its six faces (in Wolfram's order), and a `Point` or full
/// `Circle` is its own boundary. `None` for anything else.
pub fn region_boundary(region: &Expr) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  let list = |v: Vec<Expr>| Expr::List(v.into());
  let point_dim = |p: &Expr| match p {
    Expr::List(c) if c.iter().all(|v| !matches!(v, Expr::List(_))) => {
      Some(c.len())
    }
    _ => None,
  };
  // A closed loop through `ring`'s points.
  let closed = |ring: &[Expr]| -> Option<Expr> {
    if ring.len() < 3 || !ring.iter().all(|p| point_dim(p) == Some(2)) {
      return None;
    }
    let mut pts = ring.to_vec();
    pts.push(ring[0].clone());
    Some(list(pts))
  };
  let origin = |n: usize| list(vec![Expr::Integer(0); n]);
  let corner = |c: &Expr| match c {
    Expr::List(v) => Some(v.to_vec()),
    _ => None,
  };
  match (name.as_str(), args.as_slice()) {
    ("Polygon", [Expr::List(items)]) => {
      if items.first().and_then(point_dim).is_some() {
        Some(call("Line", vec![closed(items)?]))
      } else {
        let rings: Option<Vec<Expr>> = items
          .iter()
          .map(|r| match r {
            Expr::List(ring) => closed(ring),
            _ => None,
          })
          .collect();
        Some(call("Line", vec![list(rings?)]))
      }
    }
    ("Triangle", [Expr::List(pts)]) if pts.len() == 3 => {
      Some(call("Line", vec![closed(pts)?]))
    }
    ("Rectangle", [] | [_] | [_, _]) => {
      let lo = args.first().cloned().unwrap_or_else(|| origin(2));
      let lo = corner(&lo).filter(|c| c.len() == 2)?;
      let hi = match args.get(1) {
        Some(h) => corner(h).filter(|c| c.len() == 2)?,
        None => lo
          .iter()
          .map(|v| crate::helpers::plus(vec![v.clone(), Expr::Integer(1)]))
          .collect(),
      };
      let p = |x: &Expr, y: &Expr| list(vec![x.clone(), y.clone()]);
      let hi: Vec<Expr> = hi
        .iter()
        .map(|v| {
          crate::evaluator::evaluate_expr_to_expr(v).unwrap_or(v.clone())
        })
        .collect();
      Some(call(
        "Line",
        vec![list(vec![
          p(&lo[0], &lo[1]),
          p(&hi[0], &lo[1]),
          p(&hi[0], &hi[1]),
          p(&lo[0], &hi[1]),
          p(&lo[0], &lo[1]),
        ])],
      ))
    }
    ("Disk" | "Ball" | "Circle", [] | [_] | [_, _]) => {
      let n = if name == "Ball" { 3 } else { 2 };
      let center = args.first().cloned().unwrap_or_else(|| origin(n));
      if point_dim(&center) != Some(n) {
        return None;
      }
      // Only a full circle bounds itself.
      let radius = args.get(1).cloned().unwrap_or(Expr::Integer(1));
      let head = if name == "Ball" { "Sphere" } else { "Circle" };
      Some(call(head, vec![center, radius]))
    }
    ("Point", [_]) => Some(region.clone()),
    ("Cuboid", [] | [_] | [_, _]) => {
      let lo = args.first().cloned().unwrap_or_else(|| origin(3));
      let lo = corner(&lo).filter(|c| c.len() == 3)?;
      let hi: Vec<Expr> = match args.get(1) {
        Some(h) => corner(h).filter(|c| c.len() == 3)?,
        None => lo
          .iter()
          .map(|v| crate::helpers::plus(vec![v.clone(), Expr::Integer(1)]))
          .collect(),
      };
      let hi: Vec<Expr> = hi
        .iter()
        .map(|v| {
          crate::evaluator::evaluate_expr_to_expr(v).unwrap_or(v.clone())
        })
        .collect();
      let (x0, y0, z0, x1, y1, z1) =
        (&lo[0], &lo[1], &lo[2], &hi[0], &hi[1], &hi[2]);
      let p = |x: &Expr, y: &Expr, z: &Expr| {
        list(vec![x.clone(), y.clone(), z.clone()])
      };
      let faces = vec![
        list(vec![
          p(x1, y0, z0),
          p(x1, y1, z0),
          p(x0, y1, z0),
          p(x0, y0, z0),
        ]),
        list(vec![
          p(x0, y0, z0),
          p(x0, y1, z0),
          p(x0, y1, z1),
          p(x0, y0, z1),
        ]),
        list(vec![
          p(x0, y1, z0),
          p(x1, y1, z0),
          p(x1, y1, z1),
          p(x0, y1, z1),
        ]),
        list(vec![
          p(x1, y1, z0),
          p(x1, y0, z0),
          p(x1, y0, z1),
          p(x1, y1, z1),
        ]),
        list(vec![
          p(x1, y0, z0),
          p(x0, y0, z0),
          p(x0, y0, z1),
          p(x1, y0, z1),
        ]),
        list(vec![
          p(x0, y0, z1),
          p(x0, y1, z1),
          p(x1, y1, z1),
          p(x1, y0, z1),
        ]),
      ];
      Some(call("Polygon", vec![list(faces)]))
    }
    _ => None,
  }
}

pub fn discretize_graphics_ast(args: &[Expr]) -> Option<Expr> {
  let graphic = match &args[0] {
    g @ Expr::Graphics { .. } => {
      crate::evaluator::part_extraction::graphics_symbolic_form(g)?
    }
    other => other.clone(),
  };
  let Expr::FunctionCall { name, args: gargs } = &graphic else {
    return None;
  };
  if !matches!(name.as_str(), "Graphics" | "Graphics3D") || gargs.is_empty() {
    return None;
  }
  let mut coords: Vec<Vec<f64>> = Vec::new();
  let mut index: std::collections::HashMap<Vec<u64>, usize> =
    std::collections::HashMap::new();
  // Cells by kind, in the order of first appearance.
  let mut cells: Vec<(&'static str, Vec<Expr>)> = Vec::new();
  fn point(e: &Expr) -> Option<Vec<f64>> {
    let Expr::List(c) = e else { return None };
    c.iter()
      .map(crate::functions::math_ast::try_eval_to_f64)
      .collect::<Option<Vec<f64>>>()
      .filter(|v| v.len() == 2 || v.len() == 3)
  }
  // With `MaxCellMeasure`, a curved boundary is sampled finely enough for
  // cells of that size.
  let edge_length = max_cell_measure(&args[1..])
    .and_then(crate::functions::fem2d::edge_length_for_max_cell_measure);
  // `complex` holds a `GraphicsComplex`'s points, which its primitives
  // refer to by (1-based) index; they are numbered, like any vertex, in
  // the order the cells first use them.
  fn walk(
    e: &Expr,
    coords: &mut Vec<Vec<f64>>,
    index: &mut std::collections::HashMap<Vec<u64>, usize>,
    cells: &mut Vec<(&'static str, Vec<Expr>)>,
    edge_length: Option<f64>,
    complex: Option<&[Vec<f64>]>,
  ) -> Option<()> {
    let mut id_of = |p: Vec<f64>| {
      let key: Vec<u64> = p.iter().map(|v| (v + 0.0).to_bits()).collect();
      let next = coords.len();
      let i = *index.entry(key).or_insert(next);
      if i == next {
        coords.push(p);
      }
      Expr::Integer(i as i128 + 1)
    };
    // A vertex: coordinates, or an index into the enclosing complex.
    let vertex =
      |p: &Expr, id_of: &mut dyn FnMut(Vec<f64>) -> Expr| match (p, complex) {
        (Expr::Integer(k), Some(pts)) => {
          let k = usize::try_from(*k).ok()?.checked_sub(1)?;
          pts.get(k).cloned().map(id_of)
        }
        _ => point(p).map(id_of),
      };
    match e {
      Expr::List(items) => {
        for item in items {
          walk(item, coords, index, cells, edge_length, complex)?;
        }
        Some(())
      }
      Expr::FunctionCall { name, args }
        if name == "GraphicsComplex" && args.len() >= 2 =>
      {
        let Expr::List(pts) = &args[0] else {
          return None;
        };
        let pts: Vec<Vec<f64>> =
          pts.iter().map(point).collect::<Option<Vec<_>>>()?;
        walk(&args[1], coords, index, cells, edge_length, Some(&pts))
      }
      Expr::FunctionCall { name, args }
        if matches!(name.as_str(), "Polygon" | "Line" | "Point")
          && args.len() == 1 =>
      {
        let Expr::List(items) = &args[0] else {
          return None;
        };
        // One cell (a list of points), or several.
        let is_vertex = |p: &Expr| {
          point(p).is_some()
            || (complex.is_some() && matches!(p, Expr::Integer(_)))
        };
        let groups: Vec<&Expr> =
          if items.first().is_some_and(is_vertex) || name == "Point" {
            vec![&args[0]]
          } else {
            items.iter().collect()
          };
        let kind: &'static str = match name.as_str() {
          "Polygon" => "Polygon",
          "Line" => "Line",
          _ => "Point",
        };
        let mut new_cells = Vec::new();
        for g in groups {
          let Expr::List(pts) = g else { return None };
          if kind == "Point" {
            for p in pts {
              new_cells.push(vertex(p, &mut id_of)?);
            }
          } else {
            let ids: Option<Vec<Expr>> =
              pts.iter().map(|p| vertex(p, &mut id_of)).collect();
            new_cells.push(Expr::List(ids?.into()));
          }
        }
        match cells.iter_mut().find(|(k, _)| *k == kind) {
          Some((_, list)) => list.extend(new_cells),
          None => cells.push((kind, new_cells)),
        }
        Some(())
      }
      // A full disk or ellipse is a 43-gon, as Wolfram discretizes it, or
      // finer under `MaxCellMeasure`.
      Expr::FunctionCall { name, args }
        if name == "Disk" && args.len() <= 2 =>
      {
        let center = match args.first() {
          Some(c) => point(c).filter(|c| c.len() == 2)?,
          None => vec![0.0, 0.0],
        };
        let radii = match args.get(1) {
          None => (1.0, 1.0),
          Some(r) => {
            if let Some(r) = crate::functions::math_ast::try_eval_to_f64(r) {
              (r, r)
            } else {
              let r = point(r).filter(|r| r.len() == 2)?;
              (r[0], r[1])
            }
          }
        };
        // Ramanujan's approximation of the ellipse's perimeter.
        let (a, b) = (radii.0.abs(), radii.1.abs());
        let perimeter = std::f64::consts::PI
          * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt());
        let sides =
          edge_length.map_or(43, |h| ((perimeter / h).ceil() as usize).max(43));
        let ids: Vec<Expr> = (0..sides)
          .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / sides as f64;
            id_of(vec![
              center[0] + radii.0 * t.cos(),
              center[1] + radii.1 * t.sin(),
            ])
          })
          .collect();
        let cell = Expr::List(ids.into());
        match cells.iter_mut().find(|(k, _)| *k == "Polygon") {
          Some((_, list)) => list.push(cell),
          None => cells.push(("Polygon", vec![cell])),
        }
        Some(())
      }
      // Other filled planar regions by their boundary.
      Expr::FunctionCall { name, .. }
        if matches!(
          name.as_str(),
          "Rectangle" | "Triangle" | "RegularPolygon" | "Parallelogram"
        ) =>
      {
        let rings = crate::functions::fem2d::region_rings(e)?;
        if rings.len() != 1 {
          return None;
        }
        let ids: Vec<Expr> =
          rings[0].iter().map(|&(x, y)| id_of(vec![x, y])).collect();
        let cell = Expr::List(ids.into());
        match cells.iter_mut().find(|(k, _)| *k == "Polygon") {
          Some((_, list)) => list.push(cell),
          None => cells.push(("Polygon", vec![cell])),
        }
        Some(())
      }
      // Styling directives carry no geometry.
      Expr::FunctionCall { name, .. }
        if matches!(
          name.as_str(),
          "RGBColor"
            | "GrayLevel"
            | "Hue"
            | "Opacity"
            | "EdgeForm"
            | "FaceForm"
            | "Thickness"
            | "AbsoluteThickness"
            | "PointSize"
            | "AbsolutePointSize"
            | "Directive"
            | "Dashing"
            | "Specularity"
            | "Glow"
        ) =>
      {
        Some(())
      }
      Expr::Identifier(_) => Some(()),
      _ => None,
    }
  }
  walk(
    &gargs[0],
    &mut coords,
    &mut index,
    &mut cells,
    edge_length,
    None,
  )?;
  if coords.is_empty() || cells.is_empty() {
    return None;
  }
  let primitives: Vec<Expr> = cells
    .into_iter()
    .map(|(kind, list)| Expr::FunctionCall {
      name: kind.to_string(),
      args: vec![Expr::List(list.into())].into(),
    })
    .collect();
  let region = Expr::FunctionCall {
    name: "MeshRegion".to_string(),
    args: vec![
      Expr::List(
        coords
          .iter()
          .map(|p| Expr::List(p.iter().map(|&v| Expr::Real(v)).collect()))
          .collect(),
      ),
      Expr::List(primitives.into()),
    ]
    .into(),
  };
  // `MaxCellMeasure` re-meshes a filled planar result into small triangles.
  if let Some(spec) = max_cell_measure(&args[1..])
    && let Some(mesh) = parse_mesh(&region)
    && mesh.dimension() == 2
  {
    return refined_mesh(&mesh, spec);
  }
  Some(region)
}

/// The cells a mesh property item names: `d` (the dimension as a whole),
/// `{d, All}`, `{d, i}` or `{d, {i1, i2, …}}` (1-based cell indices).
enum MeshItem {
  Dimension(usize),
  All(usize),
  One(usize, usize),
  Many(usize, Vec<usize>),
}

fn mesh_item(item: &Expr) -> Option<MeshItem> {
  let dim = |e: &Expr| match e {
    Expr::Integer(d) if (0..=3).contains(d) => Some(*d as usize),
    _ => None,
  };
  match item {
    Expr::Integer(_) => Some(MeshItem::Dimension(dim(item)?)),
    Expr::List(parts) if parts.len() == 2 => {
      let d = dim(&parts[0])?;
      match &parts[1] {
        Expr::Identifier(s) if s == "All" => Some(MeshItem::All(d)),
        Expr::Integer(i) if *i >= 1 => Some(MeshItem::One(d, *i as usize)),
        other => Some(MeshItem::Many(d, index_list(other)?)),
      }
    }
    _ => None,
  }
}

/// The `Properties -> {{d, i} -> name -> value, …}` entries of a mesh.
fn mesh_properties(region: &Expr) -> Vec<(Expr, String, Expr)> {
  let Expr::FunctionCall { args, .. } = region else {
    return Vec::new();
  };
  let mut out = Vec::new();
  for arg in args.iter().skip(2) {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Properties")
      && let Expr::List(entries) = replacement.as_ref()
    {
      for entry in entries {
        if let Expr::Rule {
          pattern: key,
          replacement: inner,
        } = entry
          && let Expr::Rule {
            pattern: prop,
            replacement: value,
          } = inner.as_ref()
          && let Expr::Identifier(prop) = prop.as_ref()
        {
          out.push((
            key.as_ref().clone(),
            prop.clone(),
            value.as_ref().clone(),
          ));
        }
      }
    }
  }
  out
}

/// The value of a user-set property for the cell key `{d, i}` (or
/// `{d, Default}`), falling back to the dimension's default and then to
/// `Automatic`.
fn stored_mesh_property(
  stored: &[(Expr, String, Expr)],
  d: usize,
  index: Option<usize>,
  property: &str,
) -> Expr {
  let key = |i: Option<usize>| {
    Expr::List(
      vec![
        Expr::Integer(d as i128),
        match i {
          Some(i) => Expr::Integer(i as i128),
          None => Expr::Identifier("Default".into()),
        },
      ]
      .into(),
    )
  };
  let find = |k: &Expr| {
    stored
      .iter()
      .find(|(sk, sp, _)| {
        sp == property && expr_to_string(sk) == expr_to_string(k)
      })
      .map(|(_, _, v)| v.clone())
  };
  index
    .and_then(|i| find(&key(Some(i))))
    .or_else(|| find(&key(None)))
    .unwrap_or_else(|| Expr::Identifier("Automatic".into()))
}

/// `PropertyValue[{mesh, item}, property]`: the computed `MeshCellMeasure`
/// and `MeshCellCentroid` of the named cells, or a property set with
/// `SetProperty` (`Automatic` when unset). `None` when `region` is not a
/// mesh or the item is malformed; `$Failed` for an unknown property.
pub fn mesh_property_value(
  region: &Expr,
  item: &Expr,
  property: &str,
) -> Option<Expr> {
  let mesh = parse_mesh(region)?;
  let item = mesh_item(item)?;
  let (d, indices) = match &item {
    MeshItem::Dimension(d) | MeshItem::All(d) => (*d, None),
    MeshItem::One(d, i) => (*d, Some(vec![*i])),
    MeshItem::Many(d, is) => (*d, Some(is.clone())),
  };
  let count = mesh.cell_counts().get(d).copied().unwrap_or(0);
  let indices: Vec<usize> = indices.unwrap_or_else(|| (1..=count).collect());
  if indices.iter().any(|&i| i > count) {
    return None;
  }
  let values: Vec<Expr> = match property {
    "MeshCellMeasure" | "MeshCellCentroid" => {
      let primitives = if count == 0 {
        Vec::new()
      } else {
        mesh.cell_primitives(d)?
      };
      let mut out = Vec::with_capacity(indices.len());
      for &i in &indices {
        let primitive = primitives.get(i - 1)?;
        let value = match (property, d) {
          ("MeshCellMeasure", 0) => Expr::Real(1.0),
          ("MeshCellMeasure", _) => eval_call(
            "N",
            &[eval_call("RegionMeasure", std::slice::from_ref(primitive))?],
          )?,
          _ => eval_call(
            "N",
            &[eval_call(
              "RegionCentroid",
              std::slice::from_ref(primitive),
            )?],
          )?,
        };
        out.push(value);
      }
      out
    }
    "MeshCellLabel"
    | "MeshCellStyle"
    | "MeshCellHighlight"
    | "MeshCellMarker"
    | "MeshCellShapeFunction" => {
      let stored = mesh_properties(region);
      if let MeshItem::Dimension(d) = item {
        return Some(stored_mesh_property(&stored, d, None, property));
      }
      indices
        .iter()
        .map(|&i| stored_mesh_property(&stored, d, Some(i), property))
        .collect()
    }
    _ => return Some(Expr::Identifier("$Failed".into())),
  };
  Some(match item {
    MeshItem::One(..) => values.into_iter().next()?,
    _ => Expr::List(values.into()),
  })
}

/// `SetProperty[{mesh, item}, property -> value]`: the mesh with `value`
/// recorded under `Properties`, per cell for `{d, All}` / `{d, i}` /
/// `{d, {…}}` items (a list value assigns one entry per cell) and as the
/// dimension's default for a bare `d`.
pub fn mesh_set_property(
  region: &Expr,
  item: &Expr,
  property: &str,
  value: &Expr,
) -> Option<Expr> {
  let Expr::FunctionCall { name, args } = region else {
    return None;
  };
  let mesh = parse_mesh(region)?;
  let item = mesh_item(item)?;
  let count = |d: usize| mesh.cell_counts().get(d).copied().unwrap_or(0);
  let key = |d: usize, i: Option<usize>| {
    Expr::List(
      vec![
        Expr::Integer(d as i128),
        match i {
          Some(i) => Expr::Integer(i as i128),
          None => Expr::Identifier("Default".into()),
        },
      ]
      .into(),
    )
  };
  let (d, assignments): (usize, Vec<(Option<usize>, Expr)>) = match item {
    MeshItem::Dimension(d) => (d, vec![(None, value.clone())]),
    MeshItem::One(d, i) => (d, vec![(Some(i), value.clone())]),
    MeshItem::All(d) | MeshItem::Many(d, _) => {
      let indices: Vec<usize> = match &item {
        MeshItem::Many(_, is) => is.clone(),
        _ => (1..=count(d)).collect(),
      };
      let values: Vec<Expr> = match value {
        Expr::List(vs) if vs.len() == indices.len() => vs.to_vec(),
        Expr::List(_) => return None,
        other => vec![other.clone(); indices.len()],
      };
      (d, indices.into_iter().map(Some).zip(values).collect())
    }
  };
  let mut stored = mesh_properties(region);
  for (i, v) in assignments {
    let k = key(d, i);
    let k_text = expr_to_string(&k);
    stored
      .retain(|(sk, sp, _)| !(sp == property && expr_to_string(sk) == k_text));
    stored.push((k, property.to_string(), v));
  }
  // A per-cell assignment leaves the dimension's default explicitly
  // `Automatic`, as wolframscript writes it.
  let default_key = key(d, None);
  if !stored.iter().any(|(sk, sp, _)| {
    sp == property && expr_to_string(sk) == expr_to_string(&default_key)
  }) {
    stored.push((
      default_key,
      property.to_string(),
      Expr::Identifier("Automatic".into()),
    ));
  }
  // Entries in key order, the default of each dimension last.
  let rank = |k: &Expr| match k {
    Expr::List(p) => match (&p[0], &p[1]) {
      (Expr::Integer(d), Expr::Integer(i)) => (*d, *i),
      (Expr::Integer(d), _) => (*d, i128::MAX),
      _ => (i128::MAX, i128::MAX),
    },
    _ => (i128::MAX, i128::MAX),
  };
  stored.sort_by_key(|(k, p, _)| (rank(k), p.clone()));
  let entries: Vec<Expr> = stored
    .into_iter()
    .map(|(k, p, v)| Expr::Rule {
      pattern: Box::new(k),
      replacement: Box::new(Expr::Rule {
        pattern: Box::new(Expr::Identifier(p)),
        replacement: Box::new(v),
      }),
    })
    .collect();
  let mut new_args: Vec<Expr> = args
    .iter()
    .filter(|a| {
      !matches!(a, Expr::Rule { pattern, .. }
        if matches!(pattern.as_ref(), Expr::Identifier(s) if s == "Properties"))
    })
    .cloned()
    .collect();
  new_args.push(Expr::Rule {
    pattern: Box::new(Expr::Identifier("Properties".into())),
    replacement: Box::new(Expr::List(entries.into())),
  });
  Some(Expr::FunctionCall {
    name: name.clone(),
    args: new_args.into(),
  })
}
