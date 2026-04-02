use crate::InterpreterError;
use crate::functions::graphics::Color;
use crate::syntax::Expr;

pub fn voronoi_mesh_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Ok(Expr::FunctionCall {
      name: "VoronoiMesh".to_string(),
      args: args.to_vec(),
    });
  }

  let pts_expr = match &args[0] {
    Expr::List(v) => v,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "VoronoiMesh".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Empty list: return unevaluated (Wolfram issues VoronoiMesh::pts message)
  if pts_expr.is_empty() {
    return Ok(Expr::FunctionCall {
      name: "VoronoiMesh".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse input points
  let mut sites: Vec<(f64, f64)> = Vec::new();
  for pt in pts_expr {
    if let Expr::List(coords) = pt
      && coords.len() == 2
      && let (Some(x), Some(y)) = (
        crate::functions::math_ast::try_eval_to_f64(&coords[0]),
        crate::functions::math_ast::try_eval_to_f64(&coords[1]),
      )
    {
      sites.push((x, y));
      continue;
    }
    // Non-numeric or malformed points: return EmptyRegion[2]
    return Ok(Expr::FunctionCall {
      name: "EmptyRegion".to_string(),
      args: vec![Expr::Integer(2)],
    });
  }

  let n = sites.len();
  if n < 2 {
    return Ok(Expr::FunctionCall {
      name: "EmptyRegion".to_string(),
      args: vec![Expr::Integer(2)],
    });
  }

  // Special case: 2 points — split bounding box by the perpendicular bisector
  if n == 2 {
    return voronoi_2_sites(&sites);
  }

  // Check for collinear points
  if are_collinear(&sites) {
    return voronoi_collinear(&sites);
  }

  // Compute bounding box with 25% padding
  let mut xmin = f64::INFINITY;
  let mut xmax = f64::NEG_INFINITY;
  let mut ymin = f64::INFINITY;
  let mut ymax = f64::NEG_INFINITY;
  for &(x, y) in &sites {
    if x < xmin {
      xmin = x;
    }
    if x > xmax {
      xmax = x;
    }
    if y < ymin {
      ymin = y;
    }
    if y > ymax {
      ymax = y;
    }
  }
  let dx = xmax - xmin;
  let dy = ymax - ymin;
  let pad = dx.max(dy) * 0.25;
  let bb_xmin = xmin - pad;
  let bb_xmax = xmax + pad;
  let bb_ymin = ymin - pad;
  let bb_ymax = ymax + pad;

  // --- Bowyer-Watson Delaunay triangulation ---
  let margin = dx.max(dy).max(1.0) * 10.0;
  let cx = (xmin + xmax) / 2.0;
  let cy = (ymin + ymax) / 2.0;

  let mut all_pts: Vec<(f64, f64)> = sites.clone();
  all_pts.push((cx - 2.0 * margin, cy - margin));
  all_pts.push((cx + 2.0 * margin, cy - margin));
  all_pts.push((cx, cy + 2.0 * margin));

  let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

  for i in 0..n {
    let pt = all_pts[i];
    let mut bad: Vec<usize> = Vec::new();
    for (idx, tri) in triangles.iter().enumerate() {
      if in_circumcircle(&all_pts, tri, pt) {
        bad.push(idx);
      }
    }

    // Find boundary edges of the polygonal hole
    let mut boundary_edges: Vec<(usize, usize)> = Vec::new();
    for &bi in &bad {
      let t = triangles[bi];
      let edges = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])];
      for &(ea, eb) in &edges {
        let shared = bad.iter().any(|&bj| {
          if bj == bi {
            return false;
          }
          let t2 = triangles[bj];
          let e2 = [(t2[0], t2[1]), (t2[1], t2[2]), (t2[2], t2[0])];
          e2.iter()
            .any(|&(a, b)| (a == eb && b == ea) || (a == ea && b == eb))
        });
        if !shared {
          boundary_edges.push((ea, eb));
        }
      }
    }

    bad.sort_unstable();
    for &bi in bad.iter().rev() {
      triangles.swap_remove(bi);
    }

    for &(ea, eb) in &boundary_edges {
      triangles.push([ea, eb, i]);
    }
  }

  // Remove super-triangle
  triangles.retain(|t| t[0] < n && t[1] < n && t[2] < n);

  // --- Build Voronoi diagram ---
  // Map: site → adjacent triangle indices
  let mut site_tris: Vec<Vec<usize>> = vec![Vec::new(); n];
  for (ti, tri) in triangles.iter().enumerate() {
    site_tris[tri[0]].push(ti);
    site_tris[tri[1]].push(ti);
    site_tris[tri[2]].push(ti);
  }

  // Circumcenters
  let circumcenters: Vec<(f64, f64)> = triangles
    .iter()
    .map(|t| circumcenter(&all_pts, t))
    .collect();

  // For each site, build the Voronoi cell
  let mut all_voronoi_verts: Vec<(f64, f64)> = Vec::new();
  let mut vert_map: std::collections::HashMap<u128, usize> =
    std::collections::HashMap::new();
  let mut cells: Vec<Vec<usize>> = Vec::new();

  for site_idx in 0..n {
    let tris = &site_tris[site_idx];
    if tris.is_empty() {
      cells.push(Vec::new());
      continue;
    }

    // Order triangles around the site by walking adjacency
    let (ordered, is_closed) =
      order_triangles_around_site(site_idx, tris, &triangles);

    // Collect circumcenter coordinates
    let cc: Vec<(f64, f64)> =
      ordered.iter().map(|&ti| circumcenters[ti]).collect();

    let cell_polygon = if is_closed {
      // Interior cell - already a closed polygon
      cc
    } else {
      // Hull cell - extend the two end edges to far away, then clip
      let far = dx.max(dy).max(1.0) * 100.0;
      let mut poly = Vec::new();

      // Extend first ray
      if let Some(&first_ti) = ordered.first() {
        let dir = ray_direction(
          &sites,
          site_idx,
          &triangles[first_ti],
          true,
          &ordered,
          &triangles,
        );
        let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
        if len > 1e-14 {
          poly.push((cc[0].0 + dir.0 / len * far, cc[0].1 + dir.1 / len * far));
        }
      }

      poly.extend_from_slice(&cc);

      // Extend last ray
      if let Some(&last_ti) = ordered.last() {
        let dir = ray_direction(
          &sites,
          site_idx,
          &triangles[last_ti],
          false,
          &ordered,
          &triangles,
        );
        let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
        if len > 1e-14 {
          poly.push((
            cc.last().unwrap().0 + dir.0 / len * far,
            cc.last().unwrap().1 + dir.1 / len * far,
          ));
        }
      }

      let clipped =
        clip_polygon_to_bbox(&poly, bb_xmin, bb_xmax, bb_ymin, bb_ymax);
      // Remove duplicate consecutive vertices (clipping artifact)
      dedup_polygon(clipped)
    };

    if cell_polygon.len() < 3 {
      cells.push(Vec::new());
      continue;
    }

    // Ensure CCW winding
    let cell_polygon = ensure_ccw(cell_polygon);

    // Register vertices
    let cell_indices: Vec<usize> = cell_polygon
      .iter()
      .map(|&(x, y)| {
        let key = point_key(x, y);
        if let Some(&idx) = vert_map.get(&key) {
          idx
        } else {
          let idx = all_voronoi_verts.len() + 1; // 1-indexed
          all_voronoi_verts.push((x, y));
          vert_map.insert(key, idx);
          idx
        }
      })
      .collect();

    cells.push(cell_indices);
  }

  build_mesh_region(all_voronoi_verts, cells)
}

fn are_collinear(sites: &[(f64, f64)]) -> bool {
  if sites.len() <= 2 {
    return true;
  }
  let (x0, y0) = sites[0];
  let (x1, y1) = sites[1];
  for &(x, y) in &sites[2..] {
    let cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
    if cross.abs() > 1e-10 {
      return false;
    }
  }
  true
}

fn voronoi_collinear(sites: &[(f64, f64)]) -> Result<Expr, InterpreterError> {
  // Sort sites along their collinear direction
  let (x0, y0) = sites[0];
  let (x1, y1) = sites[1];
  let dir = (x1 - x0, y1 - y0);
  let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
  let dir_norm = if len > 1e-14 {
    (dir.0 / len, dir.1 / len)
  } else {
    (1.0, 0.0)
  };

  // Project each site onto the line and sort by projection
  let mut indexed: Vec<(f64, usize)> = sites
    .iter()
    .enumerate()
    .map(|(i, &(x, y))| {
      let proj = (x - x0) * dir_norm.0 + (y - y0) * dir_norm.1;
      (proj, i)
    })
    .collect();
  indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

  let sorted_sites: Vec<(f64, f64)> =
    indexed.iter().map(|&(_, i)| sites[i]).collect();

  // Bounding box
  let mut xmin = f64::INFINITY;
  let mut xmax = f64::NEG_INFINITY;
  let mut ymin = f64::INFINITY;
  let mut ymax = f64::NEG_INFINITY;
  for &(x, y) in &sorted_sites {
    if x < xmin {
      xmin = x;
    }
    if x > xmax {
      xmax = x;
    }
    if y < ymin {
      ymin = y;
    }
    if y > ymax {
      ymax = y;
    }
  }
  let dx = xmax - xmin;
  let dy = ymax - ymin;
  let pad = dx.max(dy) * 0.25;
  let bb_xmin = xmin - pad;
  let bb_xmax = xmax + pad;
  let bb_ymin = ymin - pad;
  let bb_ymax = ymax + pad;

  let far = dx.max(dy).max(1.0) * 100.0;
  let perp = (-dir_norm.1, dir_norm.0);

  // For each pair of consecutive sorted sites, compute perpendicular bisector
  // This creates n strips
  let n = sorted_sites.len();
  let mut bisector_points: Vec<((f64, f64), (f64, f64))> = Vec::new(); // pairs of far-away points on each bisector

  for i in 0..n - 1 {
    let (ax, ay) = sorted_sites[i];
    let (bx, by) = sorted_sites[i + 1];
    let mid = ((ax + bx) / 2.0, (ay + by) / 2.0);
    let p1 = (mid.0 + perp.0 * far, mid.1 + perp.1 * far);
    let p2 = (mid.0 - perp.0 * far, mid.1 - perp.1 * far);
    bisector_points.push((p1, p2));
  }

  // Build strip polygons by clipping to bbox
  let mut all_verts: Vec<(f64, f64)> = Vec::new();
  let mut vert_map: std::collections::HashMap<u128, usize> =
    std::collections::HashMap::new();
  let mut cells: Vec<Vec<usize>> = Vec::new();

  for i in 0..n {
    // Strip i is bounded by bisector i-1 (left) and bisector i (right)
    // Create a large polygon for this strip
    let mut poly = Vec::new();

    if i == 0 {
      // First strip: from -infinity to bisector 0
      let (p1, p2) = bisector_points[0];
      poly.push(p1);
      poly.push(p2);
      // Extend to -infinity along dir
      poly.push((p2.0 - dir_norm.0 * far, p2.1 - dir_norm.1 * far));
      poly.push((p1.0 - dir_norm.0 * far, p1.1 - dir_norm.1 * far));
    } else if i == n - 1 {
      // Last strip: from bisector n-2 to +infinity
      let (p1, p2) = bisector_points[n - 2];
      poly.push(p2);
      poly.push(p1);
      poly.push((p1.0 + dir_norm.0 * far, p1.1 + dir_norm.1 * far));
      poly.push((p2.0 + dir_norm.0 * far, p2.1 + dir_norm.1 * far));
    } else {
      // Middle strip: between bisector i-1 and bisector i
      let (l1, l2) = bisector_points[i - 1];
      let (r1, r2) = bisector_points[i];
      poly.push(l2);
      poly.push(l1);
      poly.push(r1);
      poly.push(r2);
    }

    let clipped =
      clip_polygon_to_bbox(&poly, bb_xmin, bb_xmax, bb_ymin, bb_ymax);
    let clipped = dedup_polygon(ensure_ccw(clipped));

    if clipped.len() >= 3 {
      let indices: Vec<usize> = clipped
        .iter()
        .map(|&(x, y)| {
          let key = point_key(x, y);
          if let Some(&idx) = vert_map.get(&key) {
            idx
          } else {
            let idx = all_verts.len() + 1;
            all_verts.push((x, y));
            vert_map.insert(key, idx);
            idx
          }
        })
        .collect();
      cells.push(indices);
    }
  }

  build_mesh_region(all_verts, cells)
}

fn voronoi_2_sites(sites: &[(f64, f64)]) -> Result<Expr, InterpreterError> {
  let (x0, y0) = sites[0];
  let (x1, y1) = sites[1];

  // Bounding box with 25% padding
  let xmin = x0.min(x1);
  let xmax = x0.max(x1);
  let ymin = y0.min(y1);
  let ymax = y0.max(y1);
  let dx = xmax - xmin;
  let dy = ymax - ymin;
  let pad = dx.max(dy) * 0.25;
  let bb_xmin = xmin - pad;
  let bb_xmax = xmax + pad;
  let bb_ymin = ymin - pad;
  let bb_ymax = ymax + pad;

  // Perpendicular bisector splits the bbox into two half-planes
  let mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
  let dir = (-(y1 - y0), x1 - x0); // perpendicular direction
  let far = dx.max(dy).max(1.0) * 100.0;
  let len = (dir.0 * dir.0 + dir.1 * dir.1).sqrt();
  let (dx_norm, dy_norm) = (dir.0 / len, dir.1 / len);

  // Two endpoints of the bisector line far away
  let p1 = (mid.0 + dx_norm * far, mid.1 + dy_norm * far);
  let p2 = (mid.0 - dx_norm * far, mid.1 - dy_norm * far);

  // Create two half-plane polygons and clip each to bbox
  // Cell for site 0: the half containing site 0
  // Cell for site 1: the half containing site 1
  let corners = [
    (bb_xmin, bb_ymin),
    (bb_xmax, bb_ymin),
    (bb_xmax, bb_ymax),
    (bb_xmin, bb_ymax),
  ];

  // Build polygon for site 0: corners on site 0's side + bisector endpoints
  let mut poly0 = Vec::new();
  let mut poly1 = Vec::new();

  // Signed distance from bisector line
  let normal = (x1 - x0, y1 - y0);
  let side = |px: f64, py: f64| -> f64 {
    normal.0 * (px - mid.0) + normal.1 * (py - mid.1)
  };

  // Classify corners
  for &c in &corners {
    if side(c.0, c.1) <= 0.0 {
      poly0.push(c);
    } else {
      poly1.push(c);
    }
  }

  // Use Sutherland-Hodgman clipping approach:
  // Clip the full bbox polygon by each half-plane
  let half_plane_0: Vec<(f64, f64)> = {
    let mut big_poly = vec![p1, p2];
    // Add far away points on site 0's side to make a huge polygon
    let offset = (-(normal.0) * far, -(normal.1) * far);
    big_poly.push((p2.0 + offset.0, p2.1 + offset.1));
    big_poly.push((p1.0 + offset.0, p1.1 + offset.1));
    clip_polygon_to_bbox(&big_poly, bb_xmin, bb_xmax, bb_ymin, bb_ymax)
  };

  let half_plane_1: Vec<(f64, f64)> = {
    let mut big_poly = vec![p2, p1];
    let offset = (normal.0 * far, normal.1 * far);
    big_poly.push((p1.0 + offset.0, p1.1 + offset.1));
    big_poly.push((p2.0 + offset.0, p2.1 + offset.1));
    clip_polygon_to_bbox(&big_poly, bb_xmin, bb_xmax, bb_ymin, bb_ymax)
  };

  let cell0 = dedup_polygon(ensure_ccw(half_plane_0));
  let cell1 = dedup_polygon(ensure_ccw(half_plane_1));

  // Build MeshRegion
  let mut all_verts: Vec<(f64, f64)> = Vec::new();
  let mut vert_map: std::collections::HashMap<u128, usize> =
    std::collections::HashMap::new();

  let register = |x: f64,
                  y: f64,
                  verts: &mut Vec<(f64, f64)>,
                  map: &mut std::collections::HashMap<u128, usize>|
   -> usize {
    let key = point_key(x, y);
    if let Some(&idx) = map.get(&key) {
      idx
    } else {
      let idx = verts.len() + 1;
      verts.push((x, y));
      map.insert(key, idx);
      idx
    }
  };

  let mut cells = Vec::new();
  for cell in [&cell0, &cell1] {
    if cell.len() >= 3 {
      let indices: Vec<usize> = cell
        .iter()
        .map(|&(x, y)| register(x, y, &mut all_verts, &mut vert_map))
        .collect();
      cells.push(indices);
    }
  }

  build_mesh_region(all_verts, cells)
}

fn build_mesh_region(
  all_verts: Vec<(f64, f64)>,
  cells: Vec<Vec<usize>>,
) -> Result<Expr, InterpreterError> {
  // Build MeshRegion[{{x1,y1},...}, {Polygon[{{i1,i2,...},{j1,j2,...}}]}]
  // Vertices list (1-indexed coordinates)
  let verts_expr: Vec<Expr> = all_verts
    .iter()
    .map(|&(x, y)| Expr::List(vec![Expr::Real(x), Expr::Real(y)]))
    .collect();

  // Group polygon cells into a single Polygon with multiple faces
  let faces: Vec<Expr> = cells
    .iter()
    .filter(|cell| cell.len() >= 3)
    .map(|cell| {
      Expr::List(cell.iter().map(|&idx| Expr::Integer(idx as i128)).collect())
    })
    .collect();

  let polygon = Expr::FunctionCall {
    name: "Polygon".to_string(),
    args: vec![Expr::List(faces)],
  };

  Ok(Expr::FunctionCall {
    name: "MeshRegion".to_string(),
    args: vec![Expr::List(verts_expr), Expr::List(vec![polygon])],
  })
}

fn point_key(x: f64, y: f64) -> u128 {
  ((x.to_bits() as u128) << 64) | (y.to_bits() as u128)
}

fn circumcenter(pts: &[(f64, f64)], t: &[usize; 3]) -> (f64, f64) {
  let (ax, ay) = pts[t[0]];
  let (bx, by) = pts[t[1]];
  let (cx, cy) = pts[t[2]];
  let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if d.abs() < 1e-14 {
    return ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);
  }
  let ux = ((ax * ax + ay * ay) * (by - cy)
    + (bx * bx + by * by) * (cy - ay)
    + (cx * cx + cy * cy) * (ay - by))
    / d;
  let uy = ((ax * ax + ay * ay) * (cx - bx)
    + (bx * bx + by * by) * (ax - cx)
    + (cx * cx + cy * cy) * (bx - ax))
    / d;
  (ux, uy)
}

fn in_circumcircle(pts: &[(f64, f64)], t: &[usize; 3], p: (f64, f64)) -> bool {
  let (ax, ay) = pts[t[0]];
  let (bx, by) = pts[t[1]];
  let (cx, cy) = pts[t[2]];
  let (px, py) = p;
  let dax = ax - px;
  let day = ay - py;
  let dbx = bx - px;
  let dby = by - py;
  let dcx = cx - px;
  let dcy = cy - py;
  let det = dax
    * (dby * (dcx * dcx + dcy * dcy) - dcy * (dbx * dbx + dby * dby))
    - day * (dbx * (dcx * dcx + dcy * dcy) - dcx * (dbx * dbx + dby * dby))
    + (dax * dax + day * day) * (dbx * dcy - dby * dcx);
  let orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  if orient > 0.0 { det > 0.0 } else { det < 0.0 }
}

fn other_verts(t: &[usize; 3], site: usize) -> (usize, usize) {
  if t[0] == site {
    (t[1], t[2])
  } else if t[1] == site {
    (t[0], t[2])
  } else {
    (t[0], t[1])
  }
}

/// Order triangles around a site. Returns (ordered_tri_indices, is_closed).
fn order_triangles_around_site(
  site: usize,
  tris: &[usize],
  all_triangles: &[[usize; 3]],
) -> (Vec<usize>, bool) {
  if tris.len() <= 1 {
    return (tris.to_vec(), false);
  }

  let first = tris[0];
  let (va, vb) = other_verts(&all_triangles[first], site);

  // Walk forward from vb
  let mut ordered = vec![first];
  let mut visited = std::collections::HashSet::new();
  visited.insert(first);
  let mut cur_shared = vb;

  loop {
    let next = tris.iter().find(|&&ti| {
      if visited.contains(&ti) {
        return false;
      }
      let (oa, ob) = other_verts(&all_triangles[ti], site);
      oa == cur_shared || ob == cur_shared
    });
    match next {
      Some(&ti) => {
        visited.insert(ti);
        ordered.push(ti);
        let (oa, ob) = other_verts(&all_triangles[ti], site);
        cur_shared = if oa == cur_shared { ob } else { oa };
      }
      None => break,
    }
  }

  let closed_end = cur_shared;

  // Walk backward from va
  if ordered.len() < tris.len() {
    let mut prefix = Vec::new();
    let mut cur_shared = va;
    loop {
      let next = tris.iter().find(|&&ti| {
        if visited.contains(&ti) {
          return false;
        }
        let (oa, ob) = other_verts(&all_triangles[ti], site);
        oa == cur_shared || ob == cur_shared
      });
      match next {
        Some(&ti) => {
          visited.insert(ti);
          prefix.push(ti);
          let (oa, ob) = other_verts(&all_triangles[ti], site);
          cur_shared = if oa == cur_shared { ob } else { oa };
        }
        None => break,
      }
    }
    prefix.reverse();
    prefix.extend(ordered);
    ordered = prefix;
    // Can't be closed if we needed to walk both directions
    return (ordered, false);
  }

  // Check if closed: the ring is closed if walking forward reached back to va
  let is_closed = closed_end == va && ordered.len() == tris.len();
  (ordered, is_closed)
}

/// Compute the direction of an unbounded Voronoi ray at the start or end of an open chain.
/// `is_start` means this is the first triangle in the ordered chain.
fn ray_direction(
  sites: &[(f64, f64)],
  site_idx: usize,
  tri: &[usize; 3],
  is_start: bool,
  ordered: &[usize],
  all_triangles: &[[usize; 3]],
) -> (f64, f64) {
  let (oa, ob) = other_verts(tri, site_idx);

  // For the start triangle: the unshared edge is the one whose vertex
  // doesn't connect to the previous triangle. Since there's no previous,
  // find which of (oa, ob) is NOT shared with the second triangle.
  // For the end triangle: similarly, find which is NOT shared with the previous.
  let unshared_neighbor = if is_start {
    if ordered.len() > 1 {
      let next_tri = &all_triangles[ordered[1]];
      let (na, nb) = other_verts(next_tri, site_idx);
      // The shared vertex between this tri and next is either oa or ob
      if oa == na || oa == nb { ob } else { oa }
    } else {
      ob // Only one triangle, pick either
    }
  } else if ordered.len() > 1 {
    let prev_tri = &all_triangles[ordered[ordered.len() - 2]];
    let (pa, pb) = other_verts(prev_tri, site_idx);
    if oa == pa || oa == pb { ob } else { oa }
  } else {
    oa
  };

  // The ray is perpendicular to the edge (site_idx, unshared_neighbor),
  // pointing away from the other vertex of the triangle
  let (sx, sy) = sites[site_idx];
  let (nx, ny) = sites[unshared_neighbor];
  let (dx, dy) = (nx - sx, ny - sy);
  let mid_x = (sx + nx) / 2.0;
  let mid_y = (sy + ny) / 2.0;
  let perp = (-dy, dx);

  // The ray should point away from the third vertex (the other neighbor of site_idx)
  let other_neighbor = if unshared_neighbor == oa { ob } else { oa };
  let (ox, oy) = sites[other_neighbor];
  let dot = perp.0 * (ox - mid_x) + perp.1 * (oy - mid_y);
  if dot > 0.0 {
    (dy, -dx) // point opposite to the other neighbor
  } else {
    perp
  }
}

fn dedup_polygon(poly: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
  if poly.len() <= 1 {
    return poly;
  }
  let eps = 1e-10;
  let mut result = vec![poly[0]];
  for i in 1..poly.len() {
    let prev = result.last().unwrap();
    if (poly[i].0 - prev.0).abs() > eps || (poly[i].1 - prev.1).abs() > eps {
      result.push(poly[i]);
    }
  }
  // Check last vs first
  if result.len() > 1 {
    let first = result[0];
    let last = *result.last().unwrap();
    if (first.0 - last.0).abs() <= eps && (first.1 - last.1).abs() <= eps {
      result.pop();
    }
  }
  result
}

fn ensure_ccw(mut poly: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
  let area: f64 = poly
    .iter()
    .enumerate()
    .map(|(i, &(x1, y1))| {
      let (x2, y2) = poly[(i + 1) % poly.len()];
      x1 * y2 - x2 * y1
    })
    .sum();
  if area < 0.0 {
    poly.reverse();
  }
  poly
}

fn clip_polygon_to_bbox(
  poly: &[(f64, f64)],
  xmin: f64,
  xmax: f64,
  ymin: f64,
  ymax: f64,
) -> Vec<(f64, f64)> {
  fn clip_edge(
    poly: &[(f64, f64)],
    inside: impl Fn(f64, f64) -> bool,
    intersect: impl Fn((f64, f64), (f64, f64)) -> (f64, f64),
  ) -> Vec<(f64, f64)> {
    if poly.is_empty() {
      return Vec::new();
    }
    let mut output = Vec::new();
    let n = poly.len();
    for i in 0..n {
      let cur = poly[i];
      let prev = poly[(i + n - 1) % n];
      let cur_in = inside(cur.0, cur.1);
      let prev_in = inside(prev.0, prev.1);
      if cur_in {
        if !prev_in {
          output.push(intersect(prev, cur));
        }
        output.push(cur);
      } else if prev_in {
        output.push(intersect(prev, cur));
      }
    }
    output
  }

  let p = clip_edge(
    poly,
    |x, _| x >= xmin,
    |(ax, ay), (bx, by)| {
      let t = (xmin - ax) / (bx - ax);
      (xmin, ay + t * (by - ay))
    },
  );
  let p = clip_edge(
    &p,
    |x, _| x <= xmax,
    |(ax, ay), (bx, by)| {
      let t = (xmax - ax) / (bx - ax);
      (xmax, ay + t * (by - ay))
    },
  );
  let p = clip_edge(
    &p,
    |_, y| y >= ymin,
    |(ax, ay), (bx, by)| {
      let t = (ymin - ay) / (by - ay);
      (ax + t * (bx - ax), ymin)
    },
  );
  clip_edge(
    &p,
    |_, y| y <= ymax,
    |(ax, ay), (bx, by)| {
      let t = (ymax - ay) / (by - ay);
      (ax + t * (bx - ax), ymax)
    },
  )
}

/// Render a MeshRegion[vertices, {Polygon[...], ...}] as SVG.
pub fn mesh_region_to_svg(
  vertices_expr: &Expr,
  primitives_expr: &Expr,
) -> Option<String> {
  // Parse vertices
  let vertices_list = match vertices_expr {
    Expr::List(v) => v,
    _ => return None,
  };
  let mut vertices: Vec<(f64, f64)> = Vec::new();
  for v in vertices_list {
    if let Expr::List(coords) = v
      && coords.len() == 2
      && let (Some(x), Some(y)) = (
        crate::functions::math_ast::try_eval_to_f64(&coords[0]),
        crate::functions::math_ast::try_eval_to_f64(&coords[1]),
      )
    {
      vertices.push((x, y));
      continue;
    }
    return None;
  }
  if vertices.is_empty() {
    return None;
  }

  // Parse polygons
  let prims = match primitives_expr {
    Expr::List(v) => v,
    _ => return None,
  };
  let mut polygons: Vec<Vec<usize>> = Vec::new(); // 0-indexed vertex indices
  for prim in prims {
    if let Expr::FunctionCall { name, args } = prim {
      if name == "Polygon" && args.len() == 1 {
        if let Expr::List(index_lists) = &args[0] {
          for idx_list in index_lists {
            if let Expr::List(indices) = idx_list {
              let mut poly = Vec::new();
              for idx in indices {
                if let Some(i) =
                  crate::functions::math_ast::try_eval_to_f64(idx)
                {
                  let i = i as usize;
                  if i >= 1 && i <= vertices.len() {
                    poly.push(i - 1); // Convert to 0-indexed
                  }
                }
              }
              if poly.len() >= 3 {
                polygons.push(poly);
              }
            }
          }
        }
      } else if name == "Line" && args.len() == 1 {
        // Line primitives in MeshRegion (e.g., CantorMesh) — skip SVG for now
      }
    }
  }

  if polygons.is_empty() {
    return None;
  }

  // Compute bounding box
  let mut xmin = f64::INFINITY;
  let mut xmax = f64::NEG_INFINITY;
  let mut ymin = f64::INFINITY;
  let mut ymax = f64::NEG_INFINITY;
  for &(x, y) in &vertices {
    if x < xmin {
      xmin = x;
    }
    if x > xmax {
      xmax = x;
    }
    if y < ymin {
      ymin = y;
    }
    if y > ymax {
      ymax = y;
    }
  }

  let data_w = xmax - xmin;
  let data_h = ymax - ymin;
  let pad = data_w.max(data_h) * 0.02;
  let xmin = xmin - pad;
  let xmax = xmax + pad;
  let ymin = ymin - pad;
  let ymax = ymax + pad;
  let data_w = xmax - xmin;
  let data_h = ymax - ymin;

  let svg_size = 360.0;
  let (svg_w, svg_h) = if data_w > data_h {
    (svg_size, svg_size * data_h / data_w)
  } else {
    (svg_size * data_w / data_h, svg_size)
  };

  let tx = |x: f64| (x - xmin) / data_w * svg_w;
  let ty = |y: f64| (1.0 - (y - ymin) / data_h) * svg_h; // Flip y

  // Default MeshRegion fill color (matches Mathematica)
  let fill_color = Color::new(0.626, 0.836, 0.919);

  let mut svg = format!(
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
    svg_w.round() as i32,
    svg_h.round() as i32,
    svg_w.round() as i32,
    svg_h.round() as i32,
  );

  // White background
  svg.push_str(&format!(
    "<rect width=\"{}\" height=\"{}\" fill=\"white\"/>",
    svg_w.round() as i32,
    svg_h.round() as i32,
  ));

  // Draw polygons
  for poly in &polygons {
    let points: Vec<String> = poly
      .iter()
      .map(|&vi| format!("{:.2},{:.2}", tx(vertices[vi].0), ty(vertices[vi].1)))
      .collect();
    svg.push_str(&format!(
      "<polygon points=\"{}\" fill=\"{}\" stroke=\"rgb(100,100,100)\" stroke-width=\"1\"/>",
      points.join(" "),
      fill_color.to_svg_rgb(),
    ));
  }

  svg.push_str("</svg>");
  Some(svg)
}
