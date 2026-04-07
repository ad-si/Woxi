use crate::InterpreterError;
use crate::functions::graphics::{Color, graphics_ast, parse_color};
use crate::syntax::{Expr, expr_to_output, expr_to_string};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Build a petgraph DiGraph from Wolfram Graph arguments (vertices list, edges list).
///
/// Returns the graph and a map from vertex string keys to NodeIndex.
/// Undirected edges are marked via the `directed` flag on EdgeData.
pub(crate) fn build_digraph(
  vertices: &[Expr],
  raw_edges: &[Expr],
) -> (DiGraph<usize, bool>, HashMap<String, NodeIndex>) {
  let mut graph = DiGraph::new();
  let mut index_map: HashMap<String, NodeIndex> = HashMap::new();

  for (i, v) in vertices.iter().enumerate() {
    let key = expr_to_string(v);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for e in raw_edges {
    let (edge_expr, _label) = unwrap_labeled(e);
    match edge_expr {
      Expr::FunctionCall {
        name: ename,
        args: eargs,
      } if eargs.len() == 2 => {
        let directed = ename == "DirectedEdge";
        let src_key = expr_to_string(&eargs[0]);
        let dst_key = expr_to_string(&eargs[1]);
        if let (Some(&si), Some(&di)) =
          (index_map.get(&src_key), index_map.get(&dst_key))
        {
          graph.add_edge(si, di, directed);
        }
      }
      Expr::Rule {
        pattern,
        replacement,
      } => {
        let src_key = expr_to_string(pattern);
        let dst_key = expr_to_string(replacement);
        if let (Some(&si), Some(&di)) =
          (index_map.get(&src_key), index_map.get(&dst_key))
        {
          graph.add_edge(si, di, true); // Rules are directed
        }
      }
      _ => {}
    }
  }

  (graph, index_map)
}

/// Build a petgraph UnGraph (undirected) from Wolfram Graph arguments.
/// All edges (both DirectedEdge and UndirectedEdge) are treated as undirected.
/// Edge weight is () (unit).
pub(crate) fn build_ungraph(
  vertices: &[Expr],
  edges: &[Expr],
) -> (UnGraph<usize, ()>, HashMap<String, NodeIndex>) {
  let mut graph = UnGraph::new_undirected();
  let mut index_map: HashMap<String, NodeIndex> = HashMap::new();

  for (i, v) in vertices.iter().enumerate() {
    let key = expr_to_string(v);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for edge in edges {
    if let Expr::FunctionCall { args: eargs, .. } = edge
      && eargs.len() == 2
    {
      let from_str = expr_to_string(&eargs[0]);
      let to_str = expr_to_string(&eargs[1]);
      if let (Some(&fi), Some(&ti)) =
        (index_map.get(&from_str), index_map.get(&to_str))
      {
        graph.add_edge(fi, ti, ());
      }
    }
  }

  (graph, index_map)
}

/// Build a petgraph DiGraph with u32 edge weights (capacity 1 per edge).
/// For max-flow: directed edges get capacity in one direction,
/// undirected edges get capacity in both.
pub(crate) fn build_flow_graph(
  vertices: &[Expr],
  edges: &[Expr],
) -> (DiGraph<usize, u32>, HashMap<String, NodeIndex>) {
  let mut graph = DiGraph::new();
  let mut index_map: HashMap<String, NodeIndex> = HashMap::new();

  for (i, v) in vertices.iter().enumerate() {
    let key = expr_to_string(v);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for edge in edges {
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = edge
      && eargs.len() == 2
    {
      let from_str = expr_to_string(&eargs[0]);
      let to_str = expr_to_string(&eargs[1]);
      if let (Some(&fi), Some(&ti)) =
        (index_map.get(&from_str), index_map.get(&to_str))
      {
        graph.add_edge(fi, ti, 1u32);
        if ename == "UndirectedEdge" {
          graph.add_edge(ti, fi, 1u32);
        }
      }
    }
  }

  (graph, index_map)
}

/// Data associated with each edge in the rendering petgraph.
#[derive(Clone, Debug)]
struct RenderEdgeData {
  directed: bool,
  label: Option<String>,
}

/// Build a petgraph DiGraph for rendering (preserves edge labels and direction info).
fn build_render_graph(
  vertices: &[Expr],
  raw_edges: &[Expr],
) -> (DiGraph<usize, RenderEdgeData>, HashMap<String, NodeIndex>) {
  let mut graph = DiGraph::new();
  let mut index_map: HashMap<String, NodeIndex> = HashMap::new();

  for (i, v) in vertices.iter().enumerate() {
    let key = expr_to_output(v);
    let idx = graph.add_node(i);
    index_map.insert(key, idx);
  }

  for e in raw_edges {
    let (edge_expr, label) = unwrap_labeled(e);
    if let Expr::FunctionCall {
      name: ename,
      args: eargs,
    } = edge_expr
      && eargs.len() == 2
    {
      let directed = ename == "DirectedEdge";
      let src_key = expr_to_output(&eargs[0]);
      let dst_key = expr_to_output(&eargs[1]);
      if let (Some(&si), Some(&di)) =
        (index_map.get(&src_key), index_map.get(&dst_key))
      {
        graph.add_edge(
          si,
          di,
          RenderEdgeData {
            directed,
            label: label.clone(),
          },
        );
      }
    }
  }

  (graph, index_map)
}

/// Render a Graph[{vertices}, {edges}, options...] expression to SVG
/// via the Graphics pipeline, using petgraph as the underlying data structure.
pub fn graph_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 {
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: args.to_vec(),
    });
  }

  let vertices = match &args[0] {
    Expr::List(v) => v.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let raw_edges = match &args[1] {
    Expr::List(e) => e.clone(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Graph".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let n = vertices.len();
  if n == 0 {
    return Ok(Expr::FunctionCall {
      name: "Graph".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse options from remaining args
  let options = &args[2..];
  let mut vertex_style: Option<Vec<Expr>> = None;
  let mut edge_style: Option<Color> = None;
  let mut vertex_labels = false;
  let mut vertex_shape: Option<String> = None;
  let mut vertex_size_scale: f64 = 1.0;

  for opt in options {
    if let Expr::Rule {
      pattern,
      replacement,
    } = opt
      && let Expr::Identifier(oname) = pattern.as_ref()
    {
      match oname.as_str() {
        "VertexStyle" => {
          vertex_style = Some(collect_directives(replacement));
        }
        "EdgeStyle" => {
          edge_style = parse_color(replacement);
        }
        "VertexLabels" => {
          if let Expr::String(s) = replacement.as_ref()
            && s == "Name"
          {
            vertex_labels = true;
          }
        }
        "VertexShapeFunction" => {
          if let Expr::String(s) = replacement.as_ref() {
            vertex_shape = Some(s.clone());
          }
        }
        "VertexSize" => {
          vertex_size_scale = match replacement.as_ref() {
            Expr::Identifier(s) => match s.as_str() {
              "Tiny" => 0.5,
              "Small" => 0.75,
              "Medium" => 1.25,
              "Large" => 1.5,
              _ => 1.0,
            },
            _ => crate::functions::graphics::expr_to_f64(replacement)
              .unwrap_or(1.0),
          };
        }
        _ => {}
      }
    }
  }

  // Build petgraph for rendering
  let (graph, _index_map) = build_render_graph(&vertices, &raw_edges);

  // Compute vertex positions using circular embedding
  let positions: Vec<(f64, f64)> = compute_layout(n);

  // Compute base radius for vertices
  let base_radius = if n <= 2 {
    0.15
  } else {
    let min_dist = positions
      .iter()
      .enumerate()
      .flat_map(|(i, &(x1, y1))| {
        positions[i + 1..]
          .iter()
          .map(move |&(x2, y2)| ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt())
      })
      .fold(f64::INFINITY, f64::min);
    (min_dist * 0.2).min(0.15).max(0.04)
  };
  let vertex_radius = base_radius * vertex_size_scale;

  // Build Graphics primitives
  let mut primitives: Vec<Expr> = Vec::new();

  // --- Draw edges ---
  let default_edge_color = Color::new(0.6, 0.6, 0.6);
  let edge_color = edge_style.as_ref().unwrap_or(&default_edge_color);
  primitives.push(edge_color.to_expr());
  primitives.push(Expr::FunctionCall {
    name: "AbsoluteThickness".to_string(),
    args: vec![Expr::Real(1.5)],
  });

  for edge_ref in graph.edge_references() {
    let si = edge_ref.source().index();
    let di = edge_ref.target().index();
    let edge_data = edge_ref.weight();
    let (x1, y1) = positions[si];
    let (x2, y2) = positions[di];

    if si == di {
      // Self-loop: circular arc that starts and ends exactly on the
      // vertex boundary.  The loop bulges upward (+y in data coords).
      let loop_r = vertex_radius * 1.4;
      let gap_half = 0.45_f64;
      let start_angle = -PI / 2.0 + gap_half;
      let end_angle = -PI / 2.0 - gap_half + 2.0 * PI;
      let cx = x1;
      let cy = y1 + vertex_radius - loop_r * start_angle.sin();

      // Compute exact attachment points on the vertex circle.
      // The arc endpoint lies on the loop circle; find the
      // corresponding point on the vertex circle at the same
      // angular direction from the vertex center.
      let attach = |angle: f64| -> (f64, f64) {
        let lx = cx + loop_r * angle.cos();
        let ly = cy + loop_r * angle.sin();
        let dx = lx - x1;
        let dy = ly - y1;
        let d = (dx * dx + dy * dy).sqrt();
        if d > 0.0 {
          (x1 + dx / d * vertex_radius, y1 + dy / d * vertex_radius)
        } else {
          (lx, ly)
        }
      };

      let segments = 24;
      let mut pts = Vec::with_capacity(segments + 1);
      for k in 0..=segments {
        let t = k as f64 / segments as f64;
        let angle = start_angle + t * (end_angle - start_angle);
        if k == 0 || k == segments {
          // Snap first/last point exactly onto vertex boundary
          let (px, py) = attach(angle);
          pts.push(Expr::List(vec![Expr::Real(px), Expr::Real(py)]));
        } else {
          let px = cx + loop_r * angle.cos();
          let py = cy + loop_r * angle.sin();
          pts.push(Expr::List(vec![Expr::Real(px), Expr::Real(py)]));
        }
      }
      if edge_data.directed {
        primitives.push(Expr::FunctionCall {
          name: "Arrow".to_string(),
          args: vec![Expr::List(pts)],
        });
      } else {
        primitives.push(Expr::FunctionCall {
          name: "Line".to_string(),
          args: vec![Expr::List(pts)],
        });
      }
    } else if edge_data.directed {
      // Draw arrow for directed edges — shorten to vertex boundary
      let dx = x2 - x1;
      let dy = y2 - y1;
      let len = (dx * dx + dy * dy).sqrt();
      let ux = dx / len;
      let uy = dy / len;
      let sx = x1 + ux * vertex_radius;
      let sy = y1 + uy * vertex_radius;
      let ex = x2 - ux * vertex_radius;
      let ey = y2 - uy * vertex_radius;
      primitives.push(Expr::FunctionCall {
        name: "Arrow".to_string(),
        args: vec![Expr::List(vec![
          Expr::List(vec![Expr::Real(sx), Expr::Real(sy)]),
          Expr::List(vec![Expr::Real(ex), Expr::Real(ey)]),
        ])],
      });
    } else {
      primitives.push(Expr::FunctionCall {
        name: "Line".to_string(),
        args: vec![Expr::List(vec![
          Expr::List(vec![Expr::Real(x1), Expr::Real(y1)]),
          Expr::List(vec![Expr::Real(x2), Expr::Real(y2)]),
        ])],
      });
    }

    // Edge label
    if let Some(lbl) = &edge_data.label {
      let mx = if si == di { x1 } else { (x1 + x2) / 2.0 };
      let my = if si == di {
        y1 + vertex_radius * 2.0 + vertex_radius * 1.8 * 1.5
      } else {
        (y1 + y2) / 2.0
      };
      primitives.push(Expr::FunctionCall {
        name: "Text".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Style".to_string(),
            args: vec![Expr::String(lbl.clone()), Expr::Integer(10)],
          },
          Expr::List(vec![Expr::Real(mx), Expr::Real(my)]),
        ],
      });
    }
  }

  // --- Draw vertices ---
  let (v_fill, v_edge_form) = parse_vertex_style(&vertex_style);

  if let Some(ef) = &v_edge_form {
    primitives.push(ef.clone());
  } else {
    primitives.push(Expr::FunctionCall {
      name: "EdgeForm".to_string(),
      args: vec![],
    });
  }

  let default_fill =
    Color::new(0.2588235294117647, 0.4549019607843137, 0.7176470588235294); // Wolfram default blue
  if let Some(fc) = &v_fill {
    primitives.push(fc.to_expr());
  } else {
    primitives.push(default_fill.to_expr());
  }

  for node_idx in graph.node_indices() {
    let i = node_idx.index();
    let (x, y) = positions[i];

    match vertex_shape.as_deref() {
      Some("Diamond") => {
        let r = vertex_radius * 1.3;
        primitives.push(Expr::FunctionCall {
          name: "Polygon".to_string(),
          args: vec![Expr::List(vec![
            Expr::List(vec![Expr::Real(x), Expr::Real(y + r)]),
            Expr::List(vec![Expr::Real(x + r), Expr::Real(y)]),
            Expr::List(vec![Expr::Real(x), Expr::Real(y - r)]),
            Expr::List(vec![Expr::Real(x - r), Expr::Real(y)]),
          ])],
        });
      }
      Some("Square") => {
        let r = vertex_radius * 0.9;
        primitives.push(Expr::FunctionCall {
          name: "Rectangle".to_string(),
          args: vec![
            Expr::List(vec![Expr::Real(x - r), Expr::Real(y - r)]),
            Expr::List(vec![Expr::Real(x + r), Expr::Real(y + r)]),
          ],
        });
      }
      _ => {
        primitives.push(Expr::FunctionCall {
          name: "Disk".to_string(),
          args: vec![
            Expr::List(vec![Expr::Real(x), Expr::Real(y)]),
            Expr::Real(vertex_radius),
          ],
        });
      }
    }

    if vertex_labels {
      let label_text = expr_to_output(&vertices[i]);
      primitives.push(Expr::FunctionCall {
        name: "RGBColor".to_string(),
        args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(0.0)],
      });
      primitives.push(Expr::FunctionCall {
        name: "Text".to_string(),
        args: vec![
          Expr::FunctionCall {
            name: "Style".to_string(),
            args: vec![Expr::String(label_text), Expr::Integer(10)],
          },
          Expr::List(vec![Expr::Real(x), Expr::Real(y + vertex_radius + 0.08)]),
        ],
      });
      if let Some(fc) = &v_fill {
        primitives.push(fc.to_expr());
      } else {
        primitives.push(default_fill.to_expr());
      }
    }
  }

  let content = Expr::List(primitives);
  let image_size_opt = Expr::Rule {
    pattern: Box::new(Expr::Identifier("ImageSize".to_string())),
    replacement: Box::new(Expr::Integer(360)),
  };

  graphics_ast(&[content, image_size_opt])
}

/// Circular layout: place n vertices equally spaced on a circle
fn compute_layout(n: usize) -> Vec<(f64, f64)> {
  if n == 1 {
    return vec![(0.0, 0.0)];
  }
  if n == 2 {
    return vec![(-0.5, 0.0), (0.5, 0.0)];
  }
  (0..n)
    .map(|k| {
      let angle = PI / 2.0 + (k as f64) * 2.0 * PI / (n as f64);
      (snap_coord(angle.cos()), snap_coord(angle.sin()))
    })
    .collect()
}

fn snap_coord(v: f64) -> f64 {
  for &target in &[0.0, 0.5, -0.5, 1.0, -1.0] {
    if (v - target).abs() < 1e-14 {
      return target;
    }
  }
  v
}

/// Extract the inner expression and optional label from a Labeled wrapper
fn unwrap_labeled(expr: &Expr) -> (&Expr, Option<String>) {
  if let Expr::FunctionCall { name, args } = expr
    && name == "Labeled"
    && args.len() == 2
  {
    let label = match &args[1] {
      Expr::String(s) => s.clone(),
      other => expr_to_output(other),
    };
    (&args[0], Some(label))
  } else {
    (expr, None)
  }
}

fn collect_directives(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::FunctionCall { name, args } if name == "Directive" => args.clone(),
    _ => vec![expr.clone()],
  }
}

fn parse_vertex_style(
  directives: &Option<Vec<Expr>>,
) -> (Option<Color>, Option<Expr>) {
  let dirs = match directives {
    Some(d) => d,
    None => return (None, None),
  };

  let mut fill_color: Option<Color> = None;
  let mut edge_form: Option<Expr> = None;

  for d in dirs {
    if let Some(c) = parse_color(d) {
      fill_color = Some(c);
    } else if let Expr::FunctionCall { name, .. } = d
      && name == "EdgeForm"
    {
      edge_form = Some(d.clone());
    }
  }

  (fill_color, edge_form)
}
